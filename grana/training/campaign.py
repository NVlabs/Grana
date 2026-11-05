# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreground campaign supervision using the canonical Ray launcher.

Only processes descended from this coordinator are observed or signalled;
Linux PID start times prevent PID reuse from targeting another process.
The CLI builds portable plans in :mod:`grana.training.campaign_cli`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


WANDB_SERVICE_VARIABLES = ("WANDB_SERVICE", "_WANDB_SERVICE")


def isolated_environment(environment: dict | None = None) -> dict:
    """A job must own its W&B service, rather than inherit the launcher's service."""
    result = dict(os.environ if environment is None else environment)
    for name in WANDB_SERVICE_VARIABLES:
        result.pop(name, None)
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def process_identity(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        end = raw.rfind(")")
        fields = raw[end + 2:].split()
        identity = {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "state": fields[0],
                "start_ticks": int(fields[19]), "comm": raw[raw.find("(") + 1:end]}
        identity["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return identity
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def identity_is_live(identity: dict | None) -> bool:
    if not identity or not identity.get("pid"):
        return False
    current = process_identity(identity["pid"])
    return bool(current and current["state"] != "Z"
                and current["start_ticks"] == identity.get("start_ticks")
                and (not identity.get("boot_id") or current.get("boot_id") == identity["boot_id"]))


def same_process(left: dict | None, right: dict | None) -> bool:
    return bool(left and right and left.get("pid") == right.get("pid")
                and left.get("start_ticks") == right.get("start_ticks")
                and (not left.get("boot_id") or not right.get("boot_id")
                     or left["boot_id"] == right["boot_id"]))


def process_snapshot() -> dict[int, dict]:
    result = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            record = process_identity(int(path.name))
            if record is not None:
                result[record["pid"]] = record
    return result


def last_json_line(path: Path) -> dict | None:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16384))
            lines = stream.read().splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def verify_frozen_execution(plan: dict) -> dict:
    """Check a prepared campaign before every launch and after every run.

    Historical manifests without fingerprint fields retain their original
    behavior. New prepared campaigns require both source and runtime receipts.
    """
    result = {"verified_utc": utc_now()}
    source = plan.get("source_fingerprint")
    if source:
        files = json.loads(Path(source).read_text())["files_sha256"]
        changed = []
        for path, expected in files.items():
            candidate = Path(path)
            if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                changed.append(path)
        if changed:
            raise ValueError("Source/config files changed after preparation: " + ", ".join(changed))
        result["source_files_verified"] = len(files)
    runtime = plan.get("runtime_fingerprint")
    if runtime:
        from .runtime_fingerprint import validate_runtime_fingerprint
        payload = Path(runtime).read_bytes()
        validate_runtime_fingerprint(json.loads(payload))
        result["runtime_fingerprint_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def verify_job_provenance(plan: dict, job: dict) -> dict:
    """Compare the executed driver/worker receipts with this campaign baseline."""
    if not plan.get("source_fingerprint") and not plan.get("runtime_fingerprint"):
        return {}
    import yaml
    output, repo = Path(job["output_dir"]), Path(plan["repo"])
    result = {}
    if plan.get("runtime_fingerprint"):
        from .runtime_fingerprint import validate_runtime_fingerprint
        expected = json.loads(Path(plan["runtime_fingerprint"]).read_text())
        worker = json.loads((output / "environment.json").read_text())["runtime_fingerprint"]
        validate_runtime_fingerprint(expected, worker)
        actual_config = json.loads((output / "resolved_config.json").read_text())
        expected_config = yaml.safe_load(Path(job["config"]).read_text())
        if actual_config != expected_config:
            raise ValueError("Executed configuration differs from the prepared recipe")
        result.update(runtime_fingerprint_sha256=worker["sha256"], resolved_config_equal=True)
    if plan.get("source_fingerprint"):
        files = json.loads(Path(plan["source_fingerprint"]).read_text())["files_sha256"]
        expected_code = {str(Path(path).relative_to(repo)): digest for path, digest in files.items()
                         if Path(path).suffix == ".py" and
                         (Path(path).is_relative_to(repo / "grana") or Path(path).parent == repo / "scripts/ray_migration")}
        actual_code = json.loads((output / "ray_run.json").read_text()).get("code_sha256")
        if not expected_code or actual_code != expected_code:
            raise ValueError("Executed Ray source differs from the prepared source")
        result["source_files_verified"] = len(expected_code)
    return result


class Coordinator:
    def __init__(self, plan: dict, *, recover: bool = False):
        self.plan = plan
        self.control = Path(plan["control_dir"])
        self.deadline = datetime.fromisoformat(plan["deadline_utc"]).timestamp()
        self.active = {}
        self.known = {}
        self.cancel_reason = None
        self.forced_failures = {}
        self.recover = recover
        self.retry_policy = {"max_attempts": 1, "backoff_seconds": 30,
                             **plan.get("retry_policy", {})}
        if self.retry_policy["max_attempts"] < 1 or self.retry_policy["backoff_seconds"] < 0:
            raise ValueError("Retry attempts must be positive and backoff nonnegative")
        self.state = {"campaign": plan["campaign"], "status": "running", "started_utc": utc_now(),
                      "started_unix": time.time(), "deadline_utc": plan["deadline_utc"],
                      "coordinator_generation": 1,
                      "coordinator": process_identity(os.getpid()),
                      "jobs": {job["name"]: {**job, "status": "pending", "pid": None,
                                             "started_utc": None, "exitcode": None, "attempts": []}
                               for job in plan["jobs"]}}
        self.state["started_at_utc"] = self.state["started_utc"]
        self.state["deadline_at_utc"] = self.state["deadline_utc"]
        self.state["source_snapshot"] = plan.get("source_snapshot")

    def recover_state(self):
        """Reconcile a dead owner before starting another attempt; never adopt live jobs."""
        source = self.control / "status.json"
        if not source.exists():
            return
        if not self.recover:
            raise FileExistsError("Campaign already started; use --recover after observing the owner")
        previous = json.loads(source.read_text())
        if previous.get("campaign") != self.plan["campaign"]:
            raise ValueError("Recovery status belongs to another campaign")
        identities = [previous.get("coordinator"), *previous.get("observed_owned_processes", []),
                      *previous.get("owned_live_processes", [])]
        identities.extend(job.get("process_identity") for job in previous.get("jobs", {}).values())
        if any(identity_is_live(owner) for owner in identities):
            raise RuntimeError("Recovery refused: the previous coordinator or an owned process is still live")
        history = self.control / "history"
        history.mkdir(exist_ok=True)
        archive = history / f"status-{time.time_ns()}.json"
        atomic_json(archive, previous)
        self.state.update(started_utc=previous.get("started_utc", self.state["started_utc"]),
                          started_unix=previous.get("started_unix", self.state["started_unix"]),
                          recovered_utc=utc_now(), previous_status=str(archive),
                          coordinator_generation=previous.get("coordinator_generation", 1) + 1)
        for job in self.plan["jobs"]:
            name = job["name"]
            old = copy.deepcopy(previous["jobs"][name])
            attempts = old.setdefault("attempts", [])
            if not attempts and (old.get("pid") or Path(job["output_dir"]).exists()):
                attempts.append({key: old[key] for key in
                                 ("pid", "started_utc", "started_unix", "finished_utc", "exitcode",
                                  "command", "process_identity", "error", "status") if key in old})
                attempts[-1].update(number=1, mode="fresh_start", log=job["log"])
            if attempts and old["status"] not in ("succeeded", "failed", "skipped"):
                attempts[-1].update(status="interrupted", finished_utc=utc_now(),
                                    error="Previous coordinator disappeared without recording completion")
            state = self.state["jobs"][name]
            state["attempts"] = attempts
            if old["status"] == "succeeded":
                # A stale success label is not sufficient evidence for a dependency.
                state["frozen_execution_verification"] = verify_frozen_execution(self.plan)
                state.update(status="succeeded", checkpoint_verification=self.verify_job(job),
                             finished_utc=old.get("finished_utc"))
            elif old["status"] in ("failed", "skipped") and not old.get("failure_transient"):
                state.update(status=old["status"], error=old.get("error"), reason=old.get("reason"))
            elif len(attempts) >= self.retry_policy["max_attempts"]:
                state.update(status="failed", error="Recovery attempt limit exhausted")
            else:
                state.update(status="pending", recovery_reason="previous coordinator interrupted",
                             retry_at_unix=time.time())

    def prepare_attempt(self, job: dict) -> tuple[list[str], Path, dict]:
        state = self.state["jobs"][job["name"]]
        number = len(state["attempts"]) + 1
        output = Path(job["output_dir"])
        command = ["bash", "scripts/train.sh", "--config", job["config"]]
        attempt = {"number": number, "status": "starting", "mode": "fresh_start",
                   "started_utc": utc_now(), "source_step": 0}
        if output.exists():
            if number == 1:
                raise FileExistsError(output)
            archive = self.control / "attempts" / job["name"] / f"attempt-{number - 1:03d}" / "output"
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
            attempt.update(mode="fresh_restart", archived_output=str(archive),
                           reason="Restart from the original seed and backbone; preserve the failed attempt")
        log_path = (Path(job["log"]) if number == 1 else
                    Path(job["log"]).with_name(f"{Path(job['log']).stem}.attempt-{number:03d}.log"))
        attempt.update(log=str(log_path), command=command)
        return command, log_path, attempt

    def failure_is_transient(self, code: int, log_path: Path) -> bool:
        try:
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 131072))
                message = stream.read().decode(errors="replace").lower()
        except OSError:
            message = ""
        deterministic = ("cuda out of memory", "torch.outofmemoryerror", "device-side assert",
                         "valueerror:", "filenotfounderror:", "modulenotfounderror:", "syntaxerror:")
        if any(token in message for token in deterministic):
            return False
        transient = ("workerdiederror", "workercrashederror", "actordiederror", "nodediederror",
                     "worker process has died", "connection reset by peer", "broken pipe",
                     "node has been marked dead", "gcs has been unavailable", "connectionerror:",
                     "communicate with the wandb service", "wandb service has shut down")
        return code in (-1, -9, -15, 129, 137, 143) or any(token in message for token in transient)

    def handle_signal(self, number, _frame):
        self.cancel_reason = f"received {signal.Signals(number).name}"

    def refresh_owned(self) -> dict[int, dict]:
        snapshot = process_snapshot()
        live = {pid: value for pid, value in self.known.items()
                if pid in snapshot and snapshot[pid]["start_ticks"] == value["start_ticks"]}
        changed = True
        while changed:
            changed = False
            for pid, record in snapshot.items():
                prior = self.known.get(pid)
                # A new owned descendant may reuse a previously observed PID.
                # It is adoptable only through a currently verified parent.
                if (prior is None or prior["start_ticks"] != record["start_ticks"]) and record["ppid"] in live:
                    value = {**record, "job": live[record["ppid"]]["job"]}
                    self.known[pid] = live[pid] = value
                    changed = True
        return {pid: {**snapshot[pid], "job": value["job"]}
                for pid, value in live.items() if snapshot[pid]["state"] != "Z"}

    def write_status(self):
        live = self.refresh_owned()
        self.state["observed_utc"] = utc_now()
        self.state["seconds_until_deadline"] = max(0, self.deadline - time.time())
        self.state["owned_live_processes"] = list(live.values())
        self.state["observed_owned_processes"] = list(self.known.values())
        for name, state in self.state["jobs"].items():
            state["started_at_utc"] = state.get("started_utc")
            state["finished_at_utc"] = state.get("finished_utc")
            state["exit_code"] = state.get("exitcode")
            metric = last_json_line(Path(state["output_dir"]) / "training_metrics.jsonl")
            if metric is not None:
                state["latest_training"] = {key: metric[key] for key in
                                            ("optimizer_steps", "loss", "elapsed_seconds", "update_seconds",
                                             "time/training_seconds", "gpu_peak_allocated_gib")
                                            if key in metric}
            if state["status"] == "running":
                step = metric.get("optimizer_steps", 0) if metric else 0
                if state.get("progress_step") != step:
                    state.update(progress_step=step, last_progress_unix=time.time(), last_progress_utc=utc_now())
                evaluated = last_json_line(Path(state["output_dir"]) / "evaluation_metrics.jsonl") or {}
                evaluating = (step in state.get("expected_eval_steps", [])
                              and evaluated.get("optimizer_steps", -1) < step)
                phase = "evaluating" if evaluating else "training"
                if state.get("phase") != phase:
                    state.update(last_progress_unix=time.time(), last_progress_utc=utc_now())
                state["phase"] = phase
                state["seconds_without_progress"] = time.time() - state.get("last_progress_unix", time.time())
            atomic_json(Path(state["receipt_path"]), state)
        atomic_json(self.control / "status.json", self.state)

    def launch(self, job: dict):
        if self.cancel_reason or time.time() >= self.deadline - self.plan["cleanup_reserve_seconds"]:
            raise RuntimeError("Refusing a new launch during campaign cancellation/cleanup")
        frozen = verify_frozen_execution(self.plan)
        command, log_path, attempt = self.prepare_attempt(job)
        attempt["frozen_execution_before"] = frozen
        environment = isolated_environment()
        environment.update(self.plan["environment"])
        environment = isolated_environment(environment)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job["gpus"]))
        log = log_path.open("xb")
        state = self.state["jobs"][job["name"]]
        state["attempts"].append(attempt)
        try:
            process = subprocess.Popen(command, cwd=self.plan["repo"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except BaseException:
            log.close()
            raise
        identity = process_identity(process.pid)
        if identity is not None:
            self.known[process.pid] = {**identity, "job": job["name"]}
        attempt.update(status="running", pid=process.pid, process_identity=identity)
        state.update(status="running", pid=process.pid, started_utc=utc_now(),
                     started_unix=time.time(), command=command, process_identity=identity,
                     active_log=str(log_path), attempt_number=attempt["number"],
                     restart_mode=attempt["mode"])
        state.update(progress_step=None, last_progress_unix=time.time(), last_progress_utc=utc_now())
        for key in ("error", "failure_transient", "finished_utc", "finished_unix", "retry_at_unix"):
            state.pop(key, None)
        self.active[job["name"]] = (process, log)
        print(json.dumps({"launched": job["name"], "pid": process.pid, "gpus": job["gpus"],
                          "log": str(log_path), "attempt": attempt["number"],
                          "mode": attempt["mode"], "source_step": attempt["source_step"]}), flush=True)
        self.write_status()

    def signal_owned(self, number: int, job_name: str | None = None):
        live = self.refresh_owned()
        # Individual PID signals are checked against Linux start-time identity;
        # an unrelated process that reused a recorded PID is never signalled.
        for pid, record in sorted(live.items(), reverse=True):
            if job_name is not None and record["job"] != job_name:
                continue
            current = process_identity(pid)
            if current and current["start_ticks"] == record["start_ticks"]:
                try:
                    os.kill(pid, number)
                except ProcessLookupError:
                    pass

    def cleanup(self, job_name: str | None = None):
        self.refresh_owned()
        # First let canonical drivers run their own ray.shutdown() finally block.
        for name, (process, _log) in list(self.active.items()):
            if job_name is not None and name != job_name:
                continue
            identity = self.known.get(process.pid)
            current = process_identity(process.pid)
            if identity and current and current["start_ticks"] == identity["start_ticks"] and current["pgid"] == process.pid:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        def remaining():
            live = self.refresh_owned()
            return [r for r in live.values() if job_name is None or r["job"] == job_name]
        for signum, grace in ((None, 4), (signal.SIGTERM, 8), (signal.SIGKILL, 2)):
            if signum is not None:
                self.signal_owned(signum, job_name)
            until = min(time.monotonic() + grace, time.monotonic() + max(0, self.deadline - time.time()))
            while remaining() and time.monotonic() < until:
                time.sleep(0.2)
            if not remaining():
                break
        survivors = remaining()
        if survivors:
            self.state.setdefault("cleanup_survivors", []).extend(survivors)
        for name, (process, log) in list(self.active.items()):
            if job_name is not None and name != job_name:
                continue
            try:
                process.wait(timeout=min(0.25, max(0, self.deadline - time.time())))
            except subprocess.TimeoutExpired:
                pass
            self.state["jobs"][name]["exitcode"] = process.poll()
            log.close()

    def verify_job(self, job: dict) -> dict:
        import torch
        from safetensors.torch import load_file
        from grana.modeling.checkpoint import FORMAT, tensor_digest

        torch.set_num_threads(2)

        output = Path(job["output_dir"])
        ray = json.loads((output / "ray_run.json").read_text())
        result = json.loads((output / "result.json").read_text())
        if ray.get("status") != "succeeded" or result.get("status") != "completed":
            raise ValueError("Canonical Ray driver and worker must both report success")
        if result.get("optimizer_steps") != job["expected_steps"]:
            raise ValueError("Final optimizer step count is incomplete")
        if result.get("checkpoint_reload_equal") is not True or result.get("checkpoint_prediction_parity") is not True:
            raise ValueError("Final checkpoint reload/parity check did not pass")
        provenance = verify_job_provenance(self.plan, job)
        checkpoint = Path(job["expected_checkpoint"])
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        if metadata.get("format") != FORMAT or metadata.get("source_step") != job["expected_steps"]:
            raise ValueError("Final checkpoint format/step mismatch")
        if metadata.get("source_stage") != job["stage"] or metadata["model"]["name_or_path"] != job["model"]:
            raise ValueError("Final checkpoint stage/model mismatch")
        weights = load_file(str(checkpoint / "weights.safetensors"), device="cpu")
        digest = tensor_digest(weights)
        if digest != metadata.get("weights_sha256"):
            raise ValueError("Final checkpoint tensor digest mismatch")
        if not all(bool(torch.isfinite(value).all()) for value in weights.values()):
            raise ValueError("Final checkpoint contains non-finite weights")
        return {"verified_utc": utc_now(), "checkpoint": str(checkpoint), "tensor_digest": digest,
                "tensor_count": len(weights), "finite": True, "reload_equal": True,
                "prediction_parity": True, "optimizer_steps": result["optimizer_steps"],
                "executed_provenance": provenance}

    def poll_finished(self):
        self.refresh_owned()
        for name, (process, log) in list(self.active.items()):
            code = process.poll()
            if code is None:
                continue
            state = self.state["jobs"][name]
            state.update(exitcode=code, finished_utc=utc_now(), finished_unix=time.time())
            attempt = state["attempts"][-1]
            attempt.update(exitcode=code, finished_utc=state["finished_utc"])
            log.close()
            del self.active[name]
            job = next(job for job in self.plan["jobs"] if job["name"] == name)
            try:
                forced_failure = self.forced_failures.pop(name, None)
                if forced_failure:
                    raise RuntimeError(forced_failure)
                if code != 0:
                    raise RuntimeError(f"Canonical launcher exited with status {code}")
                state["frozen_execution_verification"] = verify_frozen_execution(self.plan)
                state["checkpoint_verification"] = self.verify_job(job)
                live = [r for r in self.refresh_owned().values() if r["job"] == name]
                if live:
                    self.cleanup(name)
                if any(r["job"] == name for r in self.refresh_owned().values()):
                    raise RuntimeError("Owned child processes remained after canonical completion")
                state["status"] = "succeeded"
                attempt["status"] = "succeeded"
            except Exception as error:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")
                attempt.update(status="failed", error=state["error"])
                self.cleanup(name)
                transient = bool(forced_failure) or code != 0 and self.failure_is_transient(code, Path(attempt["log"]))
                state["failure_transient"] = attempt["failure_transient"] = transient
                if (transient and len(state["attempts"]) < self.retry_policy["max_attempts"]
                        and not self.cancel_reason
                        and not any(r["job"] == name for r in self.refresh_owned().values())):
                    state.update(status="retrying", retry_at_unix=time.time() + self.retry_policy["backoff_seconds"])
            print(json.dumps({"finished": name, "status": state["status"], "exitcode": code}), flush=True)

    def check_stalled_jobs(self):
        limits = self.plan.get("progress_timeouts", {})
        if not limits:
            return
        for name in list(self.active):
            state = self.state["jobs"][name]
            phase = state.get("phase", "training")
            limit = limits.get(f"{phase}_seconds")
            if limit and state.get("seconds_without_progress", 0) > limit and name not in self.forced_failures:
                reason = f"{phase} made no committed progress for {limit} seconds"
                self.forced_failures[name] = reason
                state["attempts"][-1]["stall_reason"] = reason
                self.cleanup(name)

    def dependencies_ready(self, job: dict) -> bool:
        """Require the checkpoint source, then wait for the device queue slot."""
        # Explicit source_pretrain separates checkpoint transfer from GPU reuse.
        # Older plans only have dependency, whose success remains mandatory.
        source = job.get("source_pretrain", job["dependency"])
        if source is not None:
            source_status = self.state["jobs"][source]["status"]
            if source_status in ("failed", "cancelled", "skipped"):
                self.state["jobs"][job["name"]].update(
                    status="skipped", reason="checkpoint dependency did not succeed")
                return False
            if source_status != "succeeded":
                return False
        dependency = job["dependency"]
        if dependency is None:
            return True
        if self.state["jobs"][dependency]["status"] not in ("succeeded", "failed", "cancelled", "skipped"):
            return False
        # A failed launcher can leave descendants alive even after cleanup.
        # Its queue slot is not available until those owned processes exit.
        return not any(record["job"] == dependency for record in self.refresh_owned().values())

    def run(self) -> int:
        previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        with (self.control / "coordinator.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("A coordinator already owns this campaign lock") from error
            try:
                self.recover_state()
                return self._run()
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)

    def _run(self) -> int:
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        try:
            for job in self.plan["jobs"]:
                if (self.state["jobs"][job["name"]]["status"] == "pending"
                        and self.dependencies_ready(job)):
                    self.launch(job)
            while True:
                self.check_stalled_jobs()
                self.poll_finished()
                if self.cancel_reason:
                    break
                if time.time() >= self.deadline - self.plan["cleanup_reserve_seconds"]:
                    self.cancel_reason = "hard campaign deadline cleanup window reached"
                    break
                for job in self.plan["jobs"]:
                    state = self.state["jobs"][job["name"]]
                    if state["status"] == "retrying" and time.time() >= state["retry_at_unix"]:
                        self.launch(job)
                        continue
                    if state["status"] != "pending":
                        continue
                    if self.dependencies_ready(job):
                        self.launch(job)
                self.write_status()
                if not self.active and all(state["status"] not in ("pending", "retrying") for state in self.state["jobs"].values()):
                    break
                time.sleep(1)
        except BaseException as error:
            self.cancel_reason = f"{type(error).__name__}: {error}"
            self.state["exception_traceback"] = traceback.format_exc()
        finally:
            if self.cancel_reason:
                self.state["cancel_reason"] = self.cancel_reason
                self.cleanup()
                for state in self.state["jobs"].values():
                    if state["status"] in ("running", "pending", "retrying"):
                        state.update(status="cancelled", reason=self.cancel_reason, finished_utc=utc_now())
                        if state["attempts"] and state["attempts"][-1]["status"] == "running":
                            state["attempts"][-1].update(status="cancelled", reason=self.cancel_reason,
                                                        finished_utc=state["finished_utc"])
            successes = sum(state["status"] == "succeeded" for state in self.state["jobs"].values())
            self.state.update(status="succeeded" if successes == len(self.plan["jobs"]) else
                              ("cancelled" if self.cancel_reason else "failed"),
                              finished_utc=utc_now(), finished_unix=time.time(), successful_jobs=successes,
                              verification_scope="Canonical completion and finite, digest-verified, reload-tested local checkpoints. W&B/render visibility requires independent final audit.")
            self.state["finished_at_utc"] = self.state["finished_utc"]
            self.write_status()
            print(json.dumps({"campaign_status": self.state["status"], "successful_jobs": successes,
                              "status_file": str(self.control / "status.json")}), flush=True)
        return 0 if self.state["status"] == "succeeded" else (130 if self.cancel_reason else 1)


class CampaignWatchdog:
    """A detached owner monitors coordinator exits and bounded heartbeat failures.

    It never initializes W&B. Separate process sessions and closed stdin allow the
    owner to outlive the terminal or tool call which requested the launch.
    """

    def __init__(self, plan: dict, command: list[str], *, recover: bool = False,
                 max_restarts: int = 3, poll_seconds: float = 5,
                 heartbeat_timeout_seconds: float = 120):
        self.plan, self.command, self.recover = plan, command, recover
        self.control = Path(plan["control_dir"])
        self.max_restarts = max_restarts
        self.poll_seconds = poll_seconds
        self.heartbeat_timeout = heartbeat_timeout_seconds
        self.stop_reason = None
        self.child = None
        self.child_identity = None
        self.state = {"campaign": plan["campaign"], "status": "starting", "started_utc": utc_now(),
                      "owner": process_identity(os.getpid()), "generations": [],
                      "max_coordinator_restarts": max_restarts,
                      "heartbeat_timeout_seconds": heartbeat_timeout_seconds}

    def write_status(self, **values):
        self.state.update(observed_utc=utc_now(), **values)
        atomic_json(self.control / "watchdog.json", self.state)

    def handle_signal(self, number, _frame):
        self.stop_reason = f"received {signal.Signals(number).name}"

    def campaign_status(self) -> dict:
        try:
            return json.loads((self.control / "status.json").read_text())
        except (OSError, ValueError):
            return {}

    def cleanup_recorded_children(self, status: dict):
        # The coordinator's last snapshot is an ownership receipt, not a PID list
        # to kill blindly. Reused and unrelated processes are filtered below.
        coordinator = status.get("coordinator")
        if identity_is_live(coordinator) and not same_process(coordinator, self.child_identity):
            raise RuntimeError("Another live coordinator owns the recorded children")
        cleanup = Coordinator(self.plan)
        for owner in [*status.get("observed_owned_processes", []), *status.get("owned_live_processes", [])]:
            if owner.get("job") and identity_is_live(owner):
                cleanup.known[owner["pid"]] = owner
        cleanup.cleanup()
        return list(cleanup.refresh_owned().values())

    def stop_child(self):
        if self.child is None:
            return []
        if identity_is_live(self.child_identity):
            os.kill(self.child_identity["pid"], signal.SIGTERM)
        if self.child is not None:
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if identity_is_live(self.child_identity):
                    os.kill(self.child_identity["pid"], signal.SIGKILL)
                self.child.wait(timeout=5)
        return self.cleanup_recorded_children(self.campaign_status())

    def run(self) -> int:
        if self.max_restarts < 0 or self.poll_seconds <= 0 or self.heartbeat_timeout <= 0:
            raise ValueError("Watchdog bounds must be positive (restart count may be zero)")
        with (self.control / "watchdog.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("A watchdog already owns this campaign") from error
            previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
            for number in previous:
                signal.signal(number, self.handle_signal)
            try:
                return self._run()
            except BaseException as error:
                survivors = self.stop_child()
                self.write_status(status="failed", error=f"{type(error).__name__}: {error}",
                                  cleanup_survivors=survivors, finished_utc=utc_now())
                raise
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)

    def _run(self) -> int:
        existing = self.campaign_status()
        if identity_is_live(existing.get("coordinator")):
            raise RuntimeError("A live coordinator already owns this campaign")
        if existing and not self.recover:
            raise RuntimeError("Campaign already started; detach with --recover to reconcile the interruption")
        if existing and self.cleanup_recorded_children(existing):
            raise RuntimeError("Owned processes survived cleanup; refusing duplicate launch")
        prior_watchdog = self.control / "watchdog.json"
        if prior_watchdog.exists():
            history = self.control / "history"
            history.mkdir(exist_ok=True)
            atomic_json(history / f"watchdog-{time.time_ns()}.json", json.loads(prior_watchdog.read_text()))
        for generation in range(self.max_restarts + 1):
            if self.stop_reason:
                break
            command = self.command + (["--recover"] if self.campaign_status() else [])
            log_path = self.control / "logs" / f"coordinator-{time.time_ns()}.log"
            with log_path.open("xb") as log:
                self.child = subprocess.Popen(command, cwd=self.plan["repo"], env=isolated_environment(),
                                              stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                self.child_identity = process_identity(self.child.pid)
                receipt = {"number": generation + 1, "started_utc": utc_now(), "log": str(log_path),
                           "command": command, "owner": self.child_identity}
                self.state["generations"].append(receipt)
                self.write_status(status="running", coordinator=self.child_identity,
                                  coordinator_log=str(log_path), restart_count=generation)
                started = time.monotonic()
                heartbeat_restart = False
                while self.child.poll() is None and not self.stop_reason:
                    status = self.campaign_status()
                    heartbeat = status.get("observed_utc")
                    age = ((datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat)).total_seconds()
                           if heartbeat else time.monotonic() - started)
                    # The previous generation's heartbeat may predate this launch.
                    age = min(age, time.monotonic() - started)
                    self.write_status(status="running", coordinator_heartbeat_age_seconds=age,
                                      campaign_recorded_status=status.get("status"))
                    if age > self.heartbeat_timeout:
                        receipt["restart_reason"] = "coordinator heartbeat expired"
                        heartbeat_restart = True
                        self.stop_child()
                        break
                    time.sleep(self.poll_seconds)
                if self.stop_reason:
                    self.stop_child()
            code = self.child.poll()
            status = self.campaign_status()
            receipt.update(exitcode=code, finished_utc=utc_now(), campaign_status=status.get("status"))
            if self.stop_reason:
                break
            if code == 0 and status.get("status") == "succeeded":
                return self.complete_campaign()
            # Deterministic failures and exhausted worker attempts stay failed;
            # a watchdog must not reset the worker attempt budget by looping.
            if status.get("status") in ("failed", "cancelled", "succeeded") and not heartbeat_restart:
                self.write_status(status="failed", error="Coordinator completed unsuccessfully; see preserved job attempts",
                                  finished_utc=utc_now())
                return 1
            survivors = self.cleanup_recorded_children(status)
            if survivors:
                self.write_status(status="failed", error="Owned processes survived cleanup; refusing duplicate launch",
                                  cleanup_survivors=survivors, finished_utc=utc_now())
                return 1
            receipt.setdefault("restart_reason", "coordinator disappeared without terminal campaign status")
            self.write_status(status="recovering")
        if self.stop_reason:
            self.write_status(status="cancelled", reason=self.stop_reason, finished_utc=utc_now())
            return 130
        self.write_status(status="failed", error="Coordinator restart limit exhausted", finished_utc=utc_now())
        return 1

    def complete_campaign(self) -> int:
        """Final acceptance is part of the durable lifecycle, without retraining on audit failure."""
        command = self.plan.get("completion_command")
        if command:
            if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
                raise ValueError("completion_command must be a nonempty argv list")
            log_path = self.control / "completion.log"
            if log_path.exists():
                log_path = self.control / f"completion-{time.time_ns()}.log"
            with log_path.open("xb") as log:
                self.child = subprocess.Popen(command, cwd=self.plan["repo"], env=isolated_environment(),
                                              stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                self.child_identity = process_identity(self.child.pid)
                self.write_status(status="auditing", completion_command=command, completion_log=str(log_path),
                                  completion_owner=self.child_identity, completion_started_utc=utc_now())
                while self.child.poll() is None and not self.stop_reason:
                    self.write_status(status="auditing")
                    time.sleep(self.poll_seconds)
                if self.stop_reason:
                    self.stop_child()
            code = self.child.poll()
            self.write_status(completion_rc=code, completion_finished_utc=utc_now())
            if self.stop_reason:
                self.write_status(status="cancelled", reason=self.stop_reason, finished_utc=utc_now())
                return 130
            if code != 0:
                self.write_status(status="failed", error="Final acceptance command failed; training will not be repeated",
                                  finished_utc=utc_now())
                return 1
        self.write_status(status="succeeded", finished_utc=utc_now())
        return 0


def detach_watchdog(plan: dict, command: list[str], *, recover: bool = False) -> dict:
    """Launch the watchdog without retaining terminal pipes or a caller session."""
    control = Path(plan["control_dir"])
    for path, key in ((control / "watchdog.json", "owner"), (control / "status.json", "coordinator")):
        if path.exists() and identity_is_live(json.loads(path.read_text()).get(key)):
            raise RuntimeError("A live owner already supervises this campaign")
    if (control / "status.json").exists() and not recover:
        raise RuntimeError("Campaign already started; use --recover")
    log_path = control / "logs" / f"watchdog-{time.time_ns()}.log"
    with log_path.open("xb") as log:
        process = subprocess.Popen(command + (["--recover"] if recover else []), cwd=plan["repo"],
                                   env=isolated_environment(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"owner": process_identity(process.pid), "log": str(log_path),
               "command": command + (["--recover"] if recover else []), "requested_utc": utc_now()}
    atomic_json(control / "watchdog_launch.json", receipt)
    return receipt
