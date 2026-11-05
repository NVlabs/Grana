# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable plans, observer behavior, child ownership, and upload verification."""
from argparse import Namespace
import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from grana.training import campaign, campaign_cli as cli, campaign_report as reports


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    pretrain = tmp_path / "pretraining.yaml"
    train_manifest = tmp_path / "train.yaml"
    train_manifest.write_text("caches: [train-cache]\n")
    pretrain.write_text("data: {manifest: train.yaml}\ntraining: {stage: pretrain}\n")
    posttrains = []
    for name in ("task-a", "task-b"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("data: {manifest: train.yaml}\ntraining: {stage: posttrain}\n"
                        "evaluation: {enabled: true, caches: [/unused/heldout], deploy_dense: {enabled: true}}\n")
        posttrains.append((name, path))
    return Namespace(lane=[cli.lane("2B=1")], model=[], output_root=tmp_path / "new-campaign",
                     pretrain_config=pretrain, posttrain_config=posttrains,
                     name=None, timeout_seconds=28800, smoke=False, check_data=False,
                     logging_mode=None, entity=None, project=None, group=None)


def test_two_gpu_lanes_parallelize_posttrains_and_keep_same_size_sources(arguments):
    arguments.lane = [cli.lane("0.8B=0,1"), cli.lane("4B=2,3")]
    plan, configs = cli.build_plan(arguments)
    assert not arguments.output_root.exists()
    assert len(plan["jobs"]) == 6
    for size in ("0.8B", "4B"):
        pre, task_a, task_b = [job for job in plan["jobs"] if job["size"] == size]
        assert pre["expected_steps"] == 5000 and len(pre["gpus"]) == 2
        assert pre["supervision_budget"]["role_rows_per_update_global"] == {"A": 40, "B": 40, "C": 48, "D": 0}
        assert task_a["expected_steps"] == task_b["expected_steps"] == 200
        assert task_a["gpus"] != task_b["gpus"]
        assert task_a["dependency"] == task_b["dependency"] == pre["name"]
        for post in (task_a, task_b):
            cfg = configs[post["name"]]
            assert post["source_pretrain"] == pre["name"]
            assert cfg["training"]["init_checkpoint"] == pre["expected_checkpoint"]
            assert cfg["model"]["name_or_path"] == f"Qwen/Qwen3.5-{size}"
            assert cfg["evaluation"]["renders"]["points_per_trajectory"] == 24
            assert cfg["evaluation"]["renders"]["enabled"] is False
            assert cfg["evaluation"]["deploy_dense"]["enabled"] is True
            assert cfg["evaluation"]["deploy_dense"]["frame_stride"] == 10
            assert post["supervision_budget"]["role_rows_per_update_global"] == {"A": 0, "B": 80, "C": 24, "D": 24}


def test_one_gpu_smokes_sequence_jobs_but_both_tasks_load_pretrain(arguments):
    arguments.lane = [cli.lane("2B=1"), cli.lane("9B=2"), cli.lane("27B=3")]
    arguments.smoke = True
    plan, configs = cli.build_plan(arguments)
    assert len(plan["jobs"]) == 9
    for size in ("2B", "9B", "27B"):
        pre, task_a, task_b = [job for job in plan["jobs"] if job["size"] == size]
        assert task_b["dependency"] == task_a["name"]
        assert task_b["source_pretrain"] == pre["name"]
        for job in (pre, task_a, task_b):
            cfg = configs[job["name"]]
            assert job["expected_steps"] == cfg["training"]["max_steps"] == 2
            assert cfg["model"]["delta_rule_backend"] == ("torch" if size == "27B" else "auto")
            assert cfg["logging"]["mode"] == "disabled"
            assert cfg["data"]["max_pixels"] is None
        assert configs[task_b["name"]]["training"]["init_checkpoint"] == pre["expected_checkpoint"]
        assert configs[task_b["name"]]["evaluation"]["deploy_dense"]["enabled"] is True
        assert configs[task_b["name"]]["evaluation"]["deploy_dense"]["max_frames"] == 2
        assert configs[pre["name"]]["evaluation"]["deploy_dense"]["enabled"] is False
        assert configs[pre["name"]]["roles"]["D"] == 0
        assert configs[task_b["name"]]["roles"]["A"] == 0
        assert pre["supervision_budget"]["global_batch_size"] == 6


def test_campaign_dependencies_follow_overridden_recipe_budgets(arguments):
    arguments.pretrain_set = ["training.max_steps=750", "training.save_steps=250"]
    arguments.posttrain_set = ["training.max_steps=300", "training.global_batch_size=256"]
    plan, configs = cli.build_plan(arguments)
    pre, task_a, task_b = plan["jobs"]
    assert pre["expected_steps"] == 750 and pre["expected_checkpoint"].endswith("checkpoint-000750")
    for post in (task_a, task_b):
        assert post["init_checkpoint"] == pre["expected_checkpoint"]
        assert post["expected_steps"] == 300
        assert post["supervision_budget"]["independent_C_windows_per_update"] == 48


@pytest.mark.parametrize("lanes", [["2B=1", "9B=1"], ["2B=1", "2B=2"]])
def test_shared_gpu_or_duplicate_size_is_rejected_before_writes(arguments, lanes):
    arguments.lane = [cli.lane(value) for value in lanes]
    with pytest.raises(ValueError, match="only one lane"):
        cli.build_plan(arguments)
    assert not arguments.output_root.exists()


def test_dry_run_resolves_quoted_model_paths_without_launch_or_writes(arguments, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not create a child process")
    monkeypatch.setattr(campaign.subprocess, "Popen", forbidden)
    assert cli.main(["run", "--lane", "2B=1", "--model", "2B=/model path/`literal`",
                     "--pretrain-config", str(arguments.pretrain_config),
                     "--posttrain-config", f"task-a={arguments.posttrain_config[0][1]}",
                     "--output-root", str(arguments.output_root), "--dry-run"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["training_launched"] is observed["files_written"] is False
    assert observed["plan"]["jobs"][0]["model"] == "/model path/`literal`"
    assert not arguments.output_root.exists()


def test_campaign_uses_only_explicit_recipes_and_preserves_offline_defaults(arguments):
    arguments.posttrain_config = []
    plan, configs = cli.build_plan(arguments)
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    assert job["portable_config"] == str(arguments.pretrain_config)
    assert job["task"] == "pretrain"
    assert configs[job["name"]]["logging"]["mode"] == "disabled"
    assert configs[job["name"]]["logging"]["entity"] is None
    assert configs[job["name"]]["data"]["caches"] == [str(arguments.pretrain_config.parent / "train-cache")]


def test_campaign_round_robin_queues_multiple_tasks_per_explicit_device(arguments):
    arguments.lane = [cli.lane("2B=0,1,2,3")]
    arguments.pretrain_set = ["training.global_batch_size=256"]
    base = arguments.posttrain_config[0][1]
    arguments.posttrain_config = [(f"adapt-{index}", base) for index in range(6)]
    plan, _ = cli.build_plan(arguments)
    pre, *post = plan["jobs"]
    assert pre["gpus"] == ["0", "1", "2", "3"]
    assert [job["gpus"] for job in post] == [["0"], ["1"], ["2"], ["3"], ["0"], ["1"]]
    assert all(job["dependency"] == pre["name"] for job in post[:4])
    assert post[4]["dependency"] == post[0]["name"]
    assert post[5]["dependency"] == post[1]["name"]
    assert all(job["source_pretrain"] == pre["name"] for job in post)


def test_campaign_rejects_duplicate_tasks_and_wrong_phase(arguments):
    arguments.posttrain_config *= 2
    with pytest.raises(ValueError, match="unique"):
        cli.build_plan(arguments)
    arguments.posttrain_config = [("wrong", arguments.pretrain_config)]
    with pytest.raises(ValueError, match="training.stage"):
        cli.build_plan(arguments)
    assert not arguments.output_root.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "has space", "bad\nname"])
def test_campaign_name_cannot_escape_output_root(arguments, name):
    arguments.name = name
    with pytest.raises(ValueError, match="Campaign name"):
        cli.build_plan(arguments)


@pytest.mark.parametrize("value", ["pretrain=config.yaml", "../task=config.yaml", "task=", "task"])
def test_adaptation_recipe_names_are_validated(value):
    from argparse import ArgumentTypeError
    with pytest.raises(ArgumentTypeError, match="NAME=CONFIG_PATH"):
        cli.posttrain_recipe(value)


def test_campaign_smoke_does_not_require_unconfigured_evaluation(arguments):
    arguments.smoke = True
    arguments.posttrain_config = []
    plan, configs = cli.build_plan(arguments)
    cfg = configs[plan["jobs"][0]["name"]]
    assert cfg["evaluation"]["enabled"] is False
    assert cfg["evaluation"]["caches"] == []


@pytest.mark.parametrize("phase", ["pretrain", "posttrain"])
@pytest.mark.parametrize("origin", ["recipe", "override"])
@pytest.mark.parametrize("field,value", [("lora_rank", 16), ("lora_alpha", 32), ("revision", "other-revision")])
def test_campaign_rejects_incompatible_resolved_checkpoint_transfer(arguments, phase, origin, field, value):
    import yaml
    if origin == "override":
        setattr(arguments, f"{phase}_set", [f"model.{field}={json.dumps(value)}"])
    else:
        path = arguments.pretrain_config if phase == "pretrain" else arguments.posttrain_config[0][1]
        recipe = yaml.safe_load(path.read_text())
        recipe["model"] = {field: value}
        path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=rf"checkpoint transfer.*model\.{field}"):
        cli.build_plan(arguments)
    assert not arguments.output_root.exists()


def test_campaign_checks_effective_transfer_identity_without_constraining_objectives(arguments):
    import yaml
    # Recipe rank differs, but the explicit phase override resolves the mismatch.
    path = arguments.posttrain_config[0][1]
    recipe = yaml.safe_load(path.read_text())
    recipe["model"] = {"lora_rank": 64}
    path.write_text(yaml.safe_dump(recipe))
    identity = ["model.lora_rank=16", "model.lora_alpha=32", "model.revision=shared-revision"]
    arguments.pretrain_set = identity
    arguments.posttrain_set = identity + [
        "model.lora_dropout=0.1", "model.delta_rule_backend=torch",
        "data.max_pixels=1024", "data.views=[wrist]", "loss.stage.cdf_weight=0.5",
        "roles={A: 0, B: 2, C: 1, D: 1}",
        "training.allow_fresh_heads=[progress_head,temporal_head]",
    ]
    plan, configs = cli.build_plan(arguments)
    pretrain, *adaptations = plan["jobs"]
    for job in adaptations:
        target = configs[job["name"]]
        assert target["training"]["init_checkpoint"] == pretrain["expected_checkpoint"]
        assert target["model"]["lora_rank"] == 16
        assert target["model"]["lora_dropout"] == 0.1
        assert target["model"]["delta_rule_backend"] == "torch"
        assert target["data"]["max_pixels"] == 1024
        assert target["data"]["views"] == ["wrist"]
        assert target["loss"]["stage"]["cdf_weight"] == 0.5
        assert target["roles"] == {"A": 0, "B": 2, "C": 1, "D": 1}
        assert target["training"]["allow_fresh_heads"] == ["progress_head", "temporal_head"]
    assert not arguments.output_root.exists()


def test_incompatible_transfer_fails_before_runtime_preparation_or_launch(arguments, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible checkpoint transfer must fail before runtime or launch")
    monkeypatch.setattr("grana.runtime.require_owned_environment", forbidden)
    monkeypatch.setattr(cli, "prepare_run", forbidden)
    monkeypatch.setattr(cli, "Coordinator", forbidden)
    assert cli.main(["run", "--lane", "2B=1",
                     "--pretrain-config", str(arguments.pretrain_config),
                     "--posttrain-config", f"task-a={arguments.posttrain_config[0][1]}",
                     "--posttrain-set", "model.lora_alpha=128",
                     "--output-root", str(arguments.output_root)]) == 1
    assert "checkpoint transfer" in capsys.readouterr().err
    assert not arguments.output_root.exists()


@pytest.mark.parametrize("origin", ["omitted", "recipe", "inherited", "override"])
def test_campaign_model_overlay_preserves_explicit_backend_and_27b_default(arguments, origin):
    import yaml
    arguments.lane = [cli.lane("27B=0")]
    for path in [arguments.pretrain_config, *(path for _, path in arguments.posttrain_config)]:
        recipe = yaml.safe_load(path.read_text())
        if origin == "recipe":
            recipe["model"] = {"delta_rule_backend": "auto"}
        elif origin == "inherited":
            parent = path.parent / "model-policy.yaml"
            parent.write_text("model: {delta_rule_backend: auto}\n")
            recipe["extends"] = parent.name
        path.write_text(yaml.safe_dump(recipe))
    if origin == "override":
        arguments.pretrain_set = arguments.posttrain_set = ["model.delta_rule_backend=auto"]
    _, configs = cli.build_plan(arguments)
    assert {cfg["model"]["name_or_path"] for cfg in configs.values()} == {"Qwen/Qwen3.5-27B"}
    assert {cfg["model"]["delta_rule_backend"] for cfg in configs.values()} == {
        "torch" if origin == "omitted" else "auto"}


@pytest.mark.parametrize("model", [None, [], "invalid"])
def test_campaign_model_overlay_does_not_hide_invalid_recipe_structure(arguments, model):
    import yaml
    recipe = yaml.safe_load(arguments.pretrain_config.read_text())
    recipe["model"] = model
    arguments.pretrain_config.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="model must be a mapping"):
        cli.build_plan(arguments)
    assert not arguments.output_root.exists()


@pytest.mark.parametrize("model_location", ["./models/local", "../local-model", "existing-model",
                                            "Qwen/Qwen3.5-2B"])
def test_campaign_freezes_paths_before_training_changes_working_directory(arguments, monkeypatch, model_location):
    import yaml
    from grana.config import load_config
    project = arguments.pretrain_config.parent
    (project / "existing-model").mkdir()
    arguments.model = [f"2B={model_location}"]
    # Exercise string/mapping inline caches, while retaining a manifest-backed
    # sibling adaptation. All selectors must survive freezing unmodified.
    recipe = yaml.safe_load(arguments.pretrain_config.read_text())
    recipe["data"] = {"caches": ["cache/train", {"path": "../shared-cache", "roles": ["A"],
                                               "include_ids": ["selected"]}]}
    recipe["evaluation"] = {"enabled": True, "caches": [{"path": "cache/eval", "views": ["wrist"]}]}
    arguments.pretrain_config.write_text(yaml.safe_dump(recipe))
    monkeypatch.chdir(project)
    plan, configs = cli.build_plan(arguments)
    # The real coordinator runs train.sh from PROJECT_ROOT. Reload the frozen
    # runtime recipe there without touching a cache payload or loading a model.
    monkeypatch.chdir(cli.PROJECT_ROOT)
    control = cli.prepare_run(plan, configs)
    assert control.is_dir()
    source = load_config(plan["jobs"][0]["config"])
    assert source["data"]["caches"] == [str(project / "cache/train"),
        {"path": str(project.parent / "shared-cache"), "roles": ["A"], "include_ids": ["selected"]}]
    assert source["evaluation"]["caches"] == [{"path": str(project / "cache/eval"), "views": ["wrist"]}]
    for job in plan["jobs"]:
        cfg = load_config(job["config"])
        expected_model = ("Qwen/Qwen3.5-2B" if model_location.startswith("Qwen/")
                          else str((project / model_location).resolve()))
        assert cfg["model"]["name_or_path"] == job["model"] == expected_model
        if job["stage"] == "posttrain":
            assert cfg["data"]["caches"] == [str(project / "train-cache")]


def test_runtime_configs_are_frozen_and_existing_outputs_are_rejected(arguments):
    from grana.config import load_config
    plan, configs = cli.build_plan(arguments)
    control = cli.prepare_run(plan, configs)
    for job in plan["jobs"]:
        cfg = load_config(job["config"])
        assert cfg["data"]["manifest"] is None
        assert cfg["data"]["caches"] == configs[job["name"]]["data"]["caches"]
    assert (control / "runtime_source_fingerprint.json").is_file()
    assert Path(plan["runtime_fingerprint"]).is_file()
    assert campaign.verify_frozen_execution(plan)["source_files_verified"] > 0
    with pytest.raises(FileExistsError):
        cli.prepare_run(plan, configs)
    with pytest.raises(FileExistsError):
        cli.build_plan(arguments)


def test_wait_timeout_is_observation_only(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(cli, "status_snapshot", lambda _: {
        "status": "running", "coordinator_live": True, "jobs": [{"status": "running", "stage": "pretrain"}]})
    monkeypatch.setattr(campaign.os, "kill", lambda *args: pytest.fail("observer must not signal training"))
    assert cli.wait_for_campaign("/unused", timeout_seconds=3, poll_seconds=2) == 124
    assert clock[0] == 3


def test_wait_rejects_dead_coordinator_and_accepts_pretrain_before_posttrain_finishes(monkeypatch, tmp_path):
    snapshot = {"status": "running", "coordinator_live": False,
                "jobs": [{"status": "running", "stage": "pretrain"}]}
    monkeypatch.setattr(cli, "status_snapshot", lambda _: snapshot)
    assert cli.wait_for_campaign("/unused") == 1
    output = tmp_path / "pretrain"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({"status": "completed", "optimizer_steps": 2,
                                                   "checkpoint_reload_equal": True, "checkpoint_prediction_parity": True}))
    snapshot.update(coordinator_live=True, jobs=[{"status": "succeeded", "stage": "pretrain"},
                                               {"status": "running", "stage": "posttrain"}])
    monkeypatch.setattr(cli, "load_campaign", lambda _: (tmp_path, {"jobs": [
        {"stage": "pretrain", "output_dir": str(output), "expected_steps": 2}]}, {}))
    assert cli.wait_for_campaign("/unused", until="pretrain") == 0


def fake_plan(tmp_path, *, fail=False, sleeper=False, tasks=("task_a", "task_b"), devices=("1",)):
    repo, control = tmp_path / "fake-repo", tmp_path / "control"
    (repo / "scripts").mkdir(parents=True)
    for folder in ("logs", "receipts", "configs"):
        (control / folder).mkdir(parents=True)
    script = repo / "scripts/train.sh"
    script.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "${{BASH_SOURCE[0]}}.py" "$@"\n')
    script.with_suffix(".sh.py").write_text('''import json,sys,time,subprocess
from pathlib import Path
cfg=json.loads(Path(sys.argv[2]).read_text())
output=Path(cfg['output_dir']);output.mkdir(parents=True)
if cfg['sleep']:
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
 (output/'descendant.json').write_text(json.dumps({'pid':child.pid}))
 time.sleep(60)
if cfg['fail']:sys.exit(3)
''')
    jobs, previous_by_device = [], {}
    for index, task in enumerate(("pretrain", *tasks)):
        assigned = list(devices) if index == 0 else [devices[(index - 1) % len(devices)]]
        dependency = None if index == 0 else previous_by_device.get(assigned[0], "pretrain")
        config = control / "configs" / f"{task}.json"
        output = tmp_path / "runs" / task
        config.write_text(json.dumps({"output_dir": str(output), "sleep": sleeper and index == 0,
                                      "fail": fail and index == 0}))
        jobs.append({"name": task, "size": "fixture", "task": task,
                     "stage": "pretrain" if index == 0 else "posttrain", "gpus": assigned,
                     "config": str(config), "output_dir": str(output),
                     "dependency": dependency,
                     "source_pretrain": None if index == 0 else "pretrain",
                     "log": str(control / "logs" / f"{task}.log"),
                     "receipt_path": str(control / "receipts" / f"{task}.json")})
        if index:
            previous_by_device[assigned[0]] = task
    return {"campaign": "fixture", "repo": str(repo), "control_dir": str(control),
            "deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=1.2 if sleeper else 20)).isoformat(),
            "cleanup_reserve_seconds": .3, "jobs": jobs, "environment": {}}


class FakeCoordinator(campaign.Coordinator):
    def verify_job(self, job):
        return {"fixture_process_exit_verified": True}


def freeze_fixture_source(plan, source):
    fingerprint = Path(plan["control_dir"]) / "frozen-source.json"
    campaign.atomic_json(fingerprint, {"files_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()}})
    plan["source_fingerprint"] = str(fingerprint)


def test_queued_job_refuses_source_drift_after_a_previous_job_finished(tmp_path):
    plan = fake_plan(tmp_path)
    script = Path(plan["repo"]) / "scripts/train.sh.py"
    freeze_fixture_source(plan, script)
    class EditingCoordinator(FakeCoordinator):
        def verify_job(self, job):
            result = super().verify_job(job)
            script.write_text(script.read_text() + "\n# edit between queued jobs\n")
            return result
    runner = EditingCoordinator(plan)
    assert runner.run() != 0
    assert runner.state["jobs"]["pretrain"]["status"] == "succeeded"
    assert runner.state["jobs"]["task_a"]["attempts"] == []
    assert not Path(plan["jobs"][1]["output_dir"]).exists()
    assert "changed after preparation" in runner.state["cancel_reason"]


def test_successful_process_with_source_drift_fails_without_training_retry(tmp_path):
    plan = fake_plan(tmp_path)
    plan["retry_policy"] = {"max_attempts": 3, "backoff_seconds": 0}
    script = Path(plan["repo"]) / "scripts/train.sh.py"
    script.write_text(script.read_text() + "\nPath(__file__).write_text(Path(__file__).read_text() + '\\n# changed\\n')\n")
    freeze_fixture_source(plan, script)
    runner = FakeCoordinator(plan)
    assert runner.run() == 1
    state = runner.state["jobs"]["pretrain"]
    assert state["exitcode"] == 0 and state["status"] == "failed"
    assert state["failure_transient"] is False and len(state["attempts"]) == 1
    assert Path(state["output_dir"]).is_dir()


def test_retry_checks_source_before_archiving_failed_output(tmp_path):
    plan = fake_plan(tmp_path)
    job = plan["jobs"][0]
    script = Path(plan["repo"]) / "scripts/train.sh.py"
    freeze_fixture_source(plan, script)
    runner = FakeCoordinator(plan)
    runner.state["jobs"][job["name"]]["attempts"] = [{"number": 1, "status": "failed"}]
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    marker = output / "partial-checkpoint"
    marker.write_text("preserve this failed attempt")
    script.write_text(script.read_text() + "\n# incompatible edit\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        runner.launch(job)
    assert marker.read_text() == "preserve this failed attempt"
    assert not (Path(plan["control_dir"]) / "attempts").exists()


def test_runtime_drift_prevents_job_launch(tmp_path, monkeypatch):
    from grana.training import runtime_fingerprint
    plan = fake_plan(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text("{}")
    plan["runtime_fingerprint"] = str(path)
    def changed(_):
        raise runtime_fingerprint.RuntimeFingerprintMismatch("causal-conv1d changed")
    monkeypatch.setattr(runtime_fingerprint, "validate_runtime_fingerprint", changed)
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("drift must prevent launch"))
    with pytest.raises(ValueError, match="causal-conv1d"):
        FakeCoordinator(plan).launch(plan["jobs"][0])


@pytest.mark.parametrize("fail", [False, True])
def test_foreground_supervisor_waits_and_skips_failed_dependencies(tmp_path, fail):
    plan = fake_plan(tmp_path, fail=fail)
    before = signal.getsignal(signal.SIGINT)
    runner = FakeCoordinator(plan)
    assert runner.run() == (1 if fail else 0)
    assert signal.getsignal(signal.SIGINT) == before
    status = json.loads((Path(plan["control_dir"]) / "status.json").read_text())
    assert status["owned_live_processes"] == []
    if fail:
        assert status["jobs"]["pretrain"]["status"] == "failed"
        assert status["jobs"]["task_a"]["status"] == status["jobs"]["task_b"]["status"] == "skipped"
    else:
        assert status["successful_jobs"] == 3
        assert status["jobs"]["task_b"]["started_at_utc"] >= status["jobs"]["task_a"]["finished_at_utc"]


@pytest.mark.parametrize("tasks", [(), ("align",), ("align", "grasp", "place", "inspect", "release")])
def test_arbitrary_adaptation_counts_preserve_device_queues_and_observer_coverage(tmp_path, monkeypatch, tasks):
    plan = fake_plan(tmp_path, tasks=tasks, devices=("2", "5"))
    for job in plan["jobs"]:
        job["expected_steps"] = 1
    control = Path(plan["control_dir"])
    campaign.atomic_json(control / "campaign.json", plan)

    class ObservedCoordinator(FakeCoordinator):
        def launch(self, job):
            for active in self.active:
                assert not set(job["gpus"]) & set(self.state["jobs"][active]["gpus"])
            if job["source_pretrain"]:
                assert self.state["jobs"][job["source_pretrain"]]["status"] == "succeeded"
            super().launch(job)

        def verify_job(self, job):
            campaign.atomic_json(Path(job["output_dir"]) / "result.json", {
                "status": "completed", "optimizer_steps": 1,
                "checkpoint_reload_equal": True, "checkpoint_prediction_parity": True})
            return super().verify_job(job)

    runner = ObservedCoordinator(plan)
    assert runner.run() == 0
    assert runner.state["successful_jobs"] == len(tasks) + 1
    for job in plan["jobs"][1:]:
        state = runner.state["jobs"][job["name"]]
        assert state["source_pretrain"] == "pretrain"
        assert state["started_at_utc"] >= runner.state["jobs"][job["dependency"]]["finished_at_utc"]
    monkeypatch.setattr(cli, "identity_is_live", lambda identity: False)
    snapshot = cli.status_snapshot(control)
    assert [job["name"] for job in snapshot["jobs"]] == ["pretrain", *tasks]
    assert snapshot["status"] == "succeeded"
    assert cli.wait_for_campaign(control, until="pretrain", timeout_seconds=1) == 0
    assert cli.wait_for_campaign(control, until="all", timeout_seconds=1) == 0


@pytest.mark.parametrize("explicit_source", [False, True])
@pytest.mark.parametrize("failed_job", ["pretrain", "align"])
def test_queue_failure_does_not_replace_explicit_checkpoint_dependency(tmp_path, explicit_source, failed_job):
    tasks = ("align", "grasp", "place", "inspect", "release")
    plan = fake_plan(tmp_path, tasks=tasks, devices=("2", "5"))
    for job in plan["jobs"]:
        if not explicit_source:
            del job["source_pretrain"]
        config = json.loads(Path(job["config"]).read_text())
        config["fail"] = job["name"] == failed_job
        Path(job["config"]).write_text(json.dumps(config))
    runner = FakeCoordinator(plan)
    assert runner.run() == 1
    states = runner.state["jobs"]
    assert states[failed_job]["status"] == "failed"
    for name in tasks:
        if name == failed_job:
            continue
        skipped = failed_job == "pretrain" or (not explicit_source and name in ("place", "release"))
        assert states[name]["status"] == ("skipped" if skipped else "succeeded")
        if not skipped:
            predecessor = states[name]["dependency"]
            assert states[name]["started_at_utc"] >= states[predecessor]["finished_at_utc"]
    assert runner.state["owned_live_processes"] == []


def test_device_queue_waits_for_failed_predecessor_descendants_to_exit(tmp_path, monkeypatch):
    runner = FakeCoordinator(fake_plan(tmp_path))
    runner.state["jobs"]["pretrain"]["status"] = "succeeded"
    runner.state["jobs"]["task_a"]["status"] = "failed"
    waiting = runner.plan["jobs"][2]
    monkeypatch.setattr(runner, "refresh_owned", lambda: {123: {"job": "task_a"}})
    assert runner.dependencies_ready(waiting) is False
    assert runner.state["jobs"]["task_b"]["status"] == "pending"
    monkeypatch.setattr(runner, "refresh_owned", lambda: {})
    assert runner.dependencies_ready(waiting) is True


@pytest.mark.parametrize("source_status", ["pending", "running", "retrying", "failed", "cancelled", "skipped"])
def test_free_device_never_bypasses_unfinished_checkpoint_source(tmp_path, source_status):
    runner = FakeCoordinator(fake_plan(tmp_path))
    runner.state["jobs"]["pretrain"]["status"] = source_status
    runner.state["jobs"]["task_a"]["status"] = "failed"
    assert runner.dependencies_ready(runner.plan["jobs"][2]) is False
    expected = "skipped" if source_status in ("failed", "cancelled", "skipped") else "pending"
    assert runner.state["jobs"]["task_b"]["status"] == expected


def test_timeout_cleans_owned_descendants_and_preserves_unrelated_process(tmp_path):
    outsider = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        plan = fake_plan(tmp_path, sleeper=True)
        runner = FakeCoordinator(plan)
        assert runner.run() == 130
        assert runner.state["owned_live_processes"] == []
        assert not runner.state.get("cleanup_survivors")
        descendant = json.loads((Path(plan["jobs"][0]["output_dir"]) / "descendant.json").read_text())["pid"]
        record = campaign.process_identity(descendant)
        assert record is None or record["state"] == "Z"
        assert outsider.poll() is None
    finally:
        outsider.terminate()
        outsider.wait(timeout=5)


def test_pid_reuse_does_not_signal_the_new_process(tmp_path, monkeypatch):
    runner = FakeCoordinator(fake_plan(tmp_path))
    monkeypatch.setattr(runner, "refresh_owned", lambda: {12345: {"job": "pretrain", "start_ticks": 100}})
    monkeypatch.setattr(campaign, "process_identity", lambda _: {"start_ticks": 101})
    monkeypatch.setattr(campaign.os, "kill", lambda *args: pytest.fail("Reused PID must not be signalled"))
    runner.signal_owned(signal.SIGTERM)


def test_reused_pid_is_adopted_only_when_its_new_parent_is_owned(tmp_path, monkeypatch):
    runner = FakeCoordinator(fake_plan(tmp_path))
    runner.known = {10: {"start_ticks": 100, "job": "pretrain"},
                    20: {"start_ticks": 200, "job": "task_a"},
                    30: {"start_ticks": 300, "job": "task_a"}}
    snapshot = {10: {"pid": 10, "ppid": 1, "start_ticks": 100, "state": "S"},
                20: {"pid": 20, "ppid": 10, "start_ticks": 201, "state": "S"},
                30: {"pid": 30, "ppid": 999, "start_ticks": 301, "state": "S"}}
    monkeypatch.setattr(campaign, "process_snapshot", lambda: snapshot)
    observed = runner.refresh_owned()
    assert observed[20]["start_ticks"] == 201 and observed[20]["job"] == "pretrain"
    assert 30 not in observed


def test_worker_death_retries_with_preserved_fresh_restart_history(tmp_path):
    plan = fake_plan(tmp_path, fail=True)
    # Shared-host /proc traversal can be slow; this case exercises successful
    # retries, while deadline expiry is covered by separate tests.
    plan["deadline_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    plan["retry_policy"] = {"max_attempts": 2, "backoff_seconds": 0}
    script = Path(plan["repo"]) / "scripts/train.sh.py"
    script.write_text(script.read_text().replace("if cfg['fail']:sys.exit(3)", """if cfg['fail']:
 marker=Path(__file__).parent/'first-failure'
 if not marker.exists():
  marker.write_text('failed first attempt')
  print('WorkerDiedError: worker process has died',flush=True)
  sys.exit(1)
"""))
    runner = FakeCoordinator(plan)
    assert runner.run() == 0
    attempts = runner.state["jobs"]["pretrain"]["attempts"]
    assert [(row["status"], row["mode"]) for row in attempts] == [
        ("failed", "fresh_start"), ("succeeded", "fresh_restart")]
    assert attempts[0]["failure_transient"] is True
    assert Path(attempts[1]["archived_output"]).is_dir()
    assert "WorkerDiedError" in Path(attempts[0]["log"]).read_text()
    assert Path(attempts[1]["log"]) != Path(attempts[0]["log"])
    assert runner.state["owned_live_processes"] == []


def test_deterministic_oom_is_not_retried_even_with_signal_exit_code(tmp_path):
    runner = FakeCoordinator(fake_plan(tmp_path))
    log = tmp_path / "oom.log"
    log.write_text("torch.OutOfMemoryError: CUDA out of memory")
    assert runner.failure_is_transient(137, log) is False


def test_recovery_archives_interrupted_run_instead_of_claiming_resume(tmp_path):
    plan = fake_plan(tmp_path)
    plan["retry_policy"] = {"max_attempts": 3, "backoff_seconds": 0}
    previous = FakeCoordinator(plan).state
    previous["coordinator"] = {"pid": 99999999, "start_ticks": 1}
    interrupted = previous["jobs"]["pretrain"]
    interrupted.update(status="running", pid=99999998, process_identity={"pid": 99999998, "start_ticks": 1})
    output = Path(interrupted["output_dir"])
    output.mkdir(parents=True)
    (output / "training_metrics.jsonl").write_text('{"optimizer_steps": 3}\n')
    Path(interrupted["log"]).write_text("interrupted first launch\n")
    campaign.atomic_json(Path(plan["control_dir"]) / "status.json", previous)
    runner = FakeCoordinator(plan, recover=True)
    assert runner.run() == 0
    attempts = runner.state["jobs"]["pretrain"]["attempts"]
    assert attempts[0]["status"] == "interrupted"
    assert attempts[1]["mode"] == "fresh_restart" and attempts[1]["source_step"] == 0
    assert (Path(attempts[1]["archived_output"]) / "training_metrics.jsonl").is_file()
    assert json.loads(Path(runner.state["previous_status"]).read_text())["jobs"]["pretrain"]["pid"] == 99999998


def test_recovery_and_watchdog_refuse_live_owner_without_signalling_it(tmp_path, monkeypatch):
    plan = fake_plan(tmp_path)
    previous = FakeCoordinator(plan).state
    status = Path(plan["control_dir"]) / "status.json"
    campaign.atomic_json(status, previous)
    original = status.read_bytes()
    monkeypatch.setattr(campaign.os, "kill", lambda *args: pytest.fail("An existing owner's processes must not be signalled"))
    with pytest.raises(RuntimeError, match="still live"):
        FakeCoordinator(plan, recover=True).run()
    watchdog = campaign.CampaignWatchdog(plan, [sys.executable, "unused"], recover=True)
    with pytest.raises(RuntimeError, match="live coordinator"):
        watchdog.run()
    assert status.read_bytes() == original


def test_status_detects_stale_running_and_boot_identity_without_mutation(tmp_path, monkeypatch):
    plan = fake_plan(tmp_path)
    for job in plan["jobs"]:
        job["expected_steps"] = 500
    control = Path(plan["control_dir"])
    previous = FakeCoordinator(plan).state
    previous["coordinator"] = {**campaign.process_identity(campaign.os.getpid()), "boot_id": "previous-boot"}
    previous["jobs"]["pretrain"]["status"] = "running"
    campaign.atomic_json(control / "campaign.json", plan)
    campaign.atomic_json(control / "status.json", previous)
    original = (control / "status.json").read_bytes()
    snapshot = cli.status_snapshot(control)
    assert snapshot["status"] == snapshot["jobs"][0]["status"] == "interrupted"
    assert snapshot["recorded_status"] == "running" and snapshot["coordinator_live"] is False
    assert (control / "status.json").read_bytes() == original


def test_long_eval_is_allowed_and_completion_resets_training_stall_clock(tmp_path, monkeypatch):
    plan = fake_plan(tmp_path)
    plan["progress_timeouts"] = {"training_seconds": 1800, "evaluating_seconds": 21600}
    runner = FakeCoordinator(plan)
    state = runner.state["jobs"]["pretrain"]
    output = Path(state["output_dir"])
    output.mkdir(parents=True)
    (output / "training_metrics.jsonl").write_text('{"optimizer_steps":250}\n')
    state.update(status="running", expected_eval_steps=[250, 500], progress_step=250,
                 phase="evaluating", last_progress_unix=time.time()-7200)
    runner.active["pretrain"] = (None, None)
    runner.write_status()
    monkeypatch.setattr(runner, "cleanup", lambda _: pytest.fail("Healthy dense evaluation must not be killed"))
    runner.check_stalled_jobs()
    assert state["phase"] == "evaluating" and state["seconds_without_progress"] > 7000
    (output / "evaluation_metrics.jsonl").write_text('{"optimizer_steps":250}\n')
    runner.write_status()
    assert state["phase"] == "training" and state["seconds_without_progress"] < 1
    runner.check_stalled_jobs()


@pytest.mark.parametrize("failure", ["exit", "heartbeat"])
def test_watchdog_recovers_actual_coordinator_process_failure(tmp_path, failure):
    plan = fake_plan(tmp_path)
    plan["deadline_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    script = tmp_path / "coordinator.py"
    control = Path(plan["control_dir"])
    script.write_text(f'''import json,os,sys,time
sys.path.insert(0,{str(Path(campaign.__file__).resolve().parents[2])!r})
from pathlib import Path
from grana.training.campaign import atomic_json,process_identity,utc_now
control=Path({str(control)!r})
marker=control/'launch-count'
count=int(marker.read_text())+1 if marker.exists() else 1
marker.write_text(str(count))
atomic_json(control/'child-env.json',{{'service_inherited':any(name in os.environ for name in ('WANDB_SERVICE','_WANDB_SERVICE'))}})
atomic_json(control/'status.json',{{'status':'running' if count==1 else 'succeeded','coordinator':process_identity(os.getpid()),'observed_utc':utc_now()}})
if count==1:
 if {failure!r}=='heartbeat':time.sleep(30)
 sys.exit(1)
''')
    # Real interpreter startup must finish before a missing heartbeat is called
    # a failure; the injected 30-second stall still exercises heartbeat recovery.
    watchdog = campaign.CampaignWatchdog(plan, [sys.executable, str(script)], max_restarts=1,
                                         poll_seconds=.02, heartbeat_timeout_seconds=5)
    before = dict(campaign.os.environ)
    campaign.os.environ["WANDB_SERVICE"] = "test-parent-service-token"
    try:
        assert watchdog.run() == 0
    finally:
        campaign.os.environ.clear()
        campaign.os.environ.update(before)
    assert (control / "launch-count").read_text() == "2"
    assert len(watchdog.state["generations"]) == 2
    assert json.loads((control / "child-env.json").read_text())["service_inherited"] is False
    assert "--recover" in watchdog.state["generations"][1]["command"]
    assert all(not campaign.identity_is_live(row["owner"]) for row in watchdog.state["generations"])


@pytest.mark.parametrize("audit_code", [0, 7])
def test_watchdog_final_acceptance_failure_does_not_repeat_training(tmp_path, audit_code):
    plan = fake_plan(tmp_path)
    marker = tmp_path / "audited.txt"
    plan["completion_command"] = [sys.executable, "-c",
        f"from pathlib import Path;Path({str(marker)!r}).write_text('audit ran');raise SystemExit({audit_code})"]
    watchdog = campaign.CampaignWatchdog(plan, ["unused"], poll_seconds=.02)
    assert watchdog.complete_campaign() == (1 if audit_code else 0)
    assert watchdog.state["completion_rc"] == audit_code
    assert marker.read_text() == "audit ran"
    assert watchdog.state["generations"] == []
    assert watchdog.state["status"] == ("failed" if audit_code else "succeeded")


def test_detached_owner_outlives_the_invoking_process_and_closes_terminal_pipes(tmp_path):
    plan = fake_plan(tmp_path)
    worker = tmp_path / "detached-owner.py"
    marker = tmp_path / "detached-owner.json"
    worker.write_text(f'''import json,os,time
from pathlib import Path
time.sleep(.1)
Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'ppid':os.getppid(),'sid':os.getsid(0),
 'stdin':os.readlink('/proc/self/fd/0'),'service_inherited':'WANDB_SERVICE' in os.environ}}))
time.sleep(30)
''')
    invoker = tmp_path / "invoker.py"
    invoker.write_text(f'''import json,os,sys
sys.path.insert(0,{str(Path(campaign.__file__).resolve().parents[2])!r})
from grana.training.campaign import detach_watchdog
os.environ['WANDB_SERVICE']='parent-owned-token'
receipt=detach_watchdog({plan!r},[sys.executable,{str(worker)!r}])
print(json.dumps({{'invoker':os.getpid(),'receipt':receipt}}),flush=True)
''')
    completed = subprocess.run([sys.executable, str(invoker)], stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=5, check=True)
    receipt = json.loads(completed.stdout)
    identity = receipt["receipt"]["owner"]
    try:
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(.02)
        observed = json.loads(marker.read_text())
        assert campaign.identity_is_live(identity)
        assert observed["sid"] == observed["pid"] == identity["pid"]
        assert observed["ppid"] != receipt["invoker"]
        assert observed["stdin"] == "/dev/null" and observed["service_inherited"] is False
    finally:
        if campaign.identity_is_live(identity):
            campaign.os.kill(identity["pid"], signal.SIGTERM)


def test_rmse_is_recomputed_from_poi_predictions(tmp_path):
    path = tmp_path / "evaluation.json"
    value = {"optimizer_steps": 2, "metrics": {"role_A_poi_rmse": .5}, "records": [
        {"role": "A", "predictions": [0., 0., 0., 0., 0., .5, 0., 0.], "targets": [0.] * 8}]}
    path.write_text(json.dumps(value))
    assert reports.verify_evaluation(path, 2) == value
    value["metrics"]["role_A_poi_rmse"] = .1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not match predictions"):
        reports.verify_evaluation(path, 2)


@pytest.mark.parametrize("tampered", [False, True])
@pytest.mark.parametrize("provenance", [None, "valid", "flat", "runtime", "config"])
def test_wandb_audit_checks_uploaded_bytes_without_creating_runs(tmp_path, monkeypatch, tampered, provenance):
    media = tmp_path / "render.png"
    media.write_bytes(b"local-render-bytes")
    checksum = base64.b64encode(hashlib.md5(media.read_bytes()).digest()).decode()
    (tmp_path / "wandb_run.json").write_text(json.dumps({"mode": "online", "status": "finished",
                                                       "entity": "team", "project": "grana", "id": "run1",
                                                       "evaluation_logging_schema": "grana-compact-eval-v2",
                                                       "history_step_schema": "grana-optimizer-step-history-v1",
                                                       "evaluation_artifacts": []}))
    (tmp_path / "resolved_config.json").write_text(json.dumps({"logging": {"group": "group1"}}))
    local = {"name": "fixture", "optimizer_steps": 2, "training_compute_seconds": 3.,
             "engine_elapsed_seconds": 4., "final_heldout_metrics": {}, "evaluation_events": [],
             "media": [{"path": str(media)}]}
    monkeypatch.setattr(reports, "report_campaign", lambda _: {"passed": True, "failures": [], "jobs": [copy.deepcopy(local)]})
    plan = {"jobs": [{"name": "fixture", "output_dir": str(tmp_path)}]}
    monkeypatch.setattr(cli, "load_campaign", lambda _: (tmp_path, plan, {}))
    run = SimpleNamespace(id="run1", entity="team", project="grana", name="fixture", group="group1", state="finished", url="https://fixture/run1",
                          summary={"optimizer_steps": 2, "logged_optimizer_steps": 2,
                                   "checkpoint_reload_equal": True, "checkpoint_prediction_parity": True,
                                   "time/training_seconds": 3., "total_elapsed_seconds": 4.},
                          scan_history=lambda **kwargs: [{"_step": i, "optimizer_steps": i, "loss": 1., "update_seconds": i} for i in (1, 2)],
                          files=lambda **kwargs: [SimpleNamespace(name="media/images/render.png", size=18,
                                                                  md5="wrong-bytes" if tampered else checksum)])
    if provenance:
        from grana.training.runtime_fingerprint import capture_runtime_fingerprint
        runtime = capture_runtime_fingerprint(package_names=[])
        runtime_path = tmp_path / "runtime.json"
        runtime_path.write_text(json.dumps(runtime))
        plan["runtime_fingerprint"] = str(runtime_path)
        actual_runtime = (capture_runtime_fingerprint(package_names=["missing-test-kernel"])
                          if provenance == "runtime" else runtime)
        run.summary["environment"] = {"runtime_fingerprint": actual_runtime}
        if provenance == "flat":
            def flatten(value, prefix):
                if isinstance(value, dict) and value:
                    for key, child in value.items():
                        flatten(child, prefix + "." + key)
                else:
                    run.summary[prefix] = value
            flatten(run.summary.pop("environment"), "environment")
        run.config = {"logging": {"group": "different" if provenance == "config" else "group1"},
                      "_wandb": {"sdk_private": True}}
    looked_up = []
    def lookup(path):
        looked_up.append(path)
        return run
    authenticated = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        login=lambda **kwargs: authenticated.append(kwargs),
        Api=lambda **kwargs: SimpleNamespace(run=lookup)))
    result = reports.audit_wandb(tmp_path)
    assert authenticated == [{"verify": True, "timeout": 30}]
    assert looked_up == ["team/grana/run1"]
    assert result["passed"] is (not tampered and provenance not in ("runtime", "config"))
    if tampered and provenance not in ("runtime", "config"):
        assert "digests" in result["failures"][0]
    if result["passed"] and provenance:
        assert result["remote"][0]["executed_provenance"]["runtime_fingerprint_sha256"] == runtime["sha256"]
