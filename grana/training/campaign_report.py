# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only local results and optional W&B upload checks; never starts training."""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)



def record_poi_slot(record):
    """Validate the saved query position without importing the model/evaluator."""
    slot = record.get("poi_slot", 5)
    require(type(slot) is int and 0 <= slot < 8, "Record poi_slot must be an integer in 0..7")
    return slot

def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def close(left, right):
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9))


def independent_quality_audit(evaluation):
    """Recompute requested quality from raw records using only stdlib math.

    This deliberately does not call the training evaluator's metric helpers.
    Confusion/support counts and every progress stratum are recomputed from
    the saved predictions, including unassessed strata and fixed baselines.
    """
    records, metrics, diagnostics = evaluation["records"], evaluation["metrics"], evaluation["diagnostics"]
    checked = 0

    def check(actual, expected, label):
        nonlocal checked
        checked += 1
        equal = close(actual, expected) if isinstance(expected, (int, float)) else actual == expected
        require(equal, f"Independent quality audit differs: {label}")

    def scalar(actual, expected, prefix):
        for key, value in expected.items():
            check(actual.get(key), value, f"{prefix}/{key}")
            if isinstance(value, (int, float)):
                check(metrics.get(f"diagnostics/{prefix}/{key}"), value, f"diagnostics/{prefix}/{key}")

    def progress(rows, actual, prefix):
        count = len(rows)
        expected = {"support": count, "status": "assessed" if count else "unassessed"}
        for name in ("rmse", "mae", "accuracy_within_0.1", "constant_0.5_rmse", "constant_0.5_mae", "constant_0.5_accuracy_within_0.1"):
            expected[name] = None
        if count:
            differences = [row["predictions"][record_poi_slot(row)] - row["targets"][record_poi_slot(row)] for row in rows]
            constant_differences = [.5 - row["targets"][record_poi_slot(row)] for row in rows]
            for prefix_name, values in (("", differences), ("constant_0.5_", constant_differences)):
                expected[prefix_name + "rmse"] = math.sqrt(math.fsum(value * value for value in values) / count)
                expected[prefix_name + "mae"] = math.fsum(map(abs, values)) / count
                expected[prefix_name + "accuracy_within_0.1"] = len([value for value in values if abs(value) <= .10000001]) / count
        scalar(actual, expected, prefix)

    tasks = sorted({row["task"] for row in records})
    groups = sorted({row["group"] for row in records if "group" in row})
    require(set(diagnostics["per_task"]) == set(tasks) and set(diagnostics["per_group"]) == set(groups),
            "Independent quality audit task/group coverage differs")
    scopes = [("overall", records, diagnostics["overall"])]
    scopes += [(f"per_task/{task}", [row for row in records if row["task"] == task], diagnostics["per_task"][task]) for task in tasks]
    scopes += [(f"per_group/{group}", [row for row in records if row.get("group") == group], diagnostics["per_group"][group]) for group in groups]
    identity = lambda row: (row["cache"], row["task"], row["trajectory_id"], row["poi"])
    for prefix, rows, actual in scopes:
        stages = [row for row in rows if row["role"] == "C"]
        number = max((row["num_stages"] for row in stages), default=0)
        counts = Counter((int(row["targets"][record_poi_slot(row)]), int(row["predictions"][record_poi_slot(row)])) for row in stages)
        matrix = [[counts[true, predicted] for predicted in range(number)] for true in range(number)]
        support = [sum(counts[true, predicted] for predicted in range(number)) for true in range(number)]
        accuracies = [counts[index, index] / count if count else None for index, count in enumerate(support)]
        observed = [accuracy for accuracy in accuracies if accuracy is not None]
        stage_expected = {"support": len(stages), "status": "assessed" if stages else "unassessed",
                          "poi_accuracy": sum(counts[index, index] for index in range(number)) / len(stages) if stages else None,
                          "stage_macro_accuracy": math.fsum(observed) / len(observed) if observed else None,
                          "stage_macro_supported_classes": len(observed),
                          "stage_mae": math.fsum(abs(true - predicted) * count for (true, predicted), count in counts.items()) / len(stages) if stages else None,
                          "abs_error_ge_2_rate": sum(count for (true, predicted), count in counts.items() if abs(true - predicted) >= 2) / len(stages) if stages else None}
        scalar(actual["C"], stage_expected, f"{prefix}/C")
        check(actual["C"]["confusion_matrix"], matrix, f"{prefix}/C/confusion_matrix")
        check(actual["C"]["stage_support"], support, f"{prefix}/C/stage_support")
        for index, value in enumerate(accuracies):
            check(actual["C"]["stage_accuracy"][index], value, f"{prefix}/C/stage_{index}/accuracy")
            check(metrics.get(f"diagnostics/{prefix}/C/stage_{index}/support"), support[index], f"{prefix}/C/stage_{index}/support")
            if value is not None:
                check(metrics.get(f"diagnostics/{prefix}/C/stage_{index}/accuracy"), value, f"{prefix}/C/stage_{index}/accuracy")
        correct = {identity(row): row["targets"][record_poi_slot(row)] == row["predictions"][record_poi_slot(row)] for row in stages}
        for role in ("A", "B", "D"):
            selected = [row for row in rows if row["role"] == role]
            role_prefix = f"{prefix}/{role}"
            progress(selected, actual[role], role_prefix)
            progress([row for row in selected if row["targets"][record_poi_slot(row)] < .05], actual[role]["gt_lt_0.05"], role_prefix + "/gt_lt_0.05")
            if role == "D":
                for stratum, correctness in (("correct_C", True), ("incorrect_C", False)):
                    subset = [row for row in selected if correct[identity(row)] == correctness]
                    progress(subset, actual[role][stratum], role_prefix + "/" + stratum)
                    progress([row for row in subset if row["targets"][record_poi_slot(row)] < .05], actual[role][stratum]["gt_lt_0.05"], role_prefix + "/" + stratum + "/gt_lt_0.05")

    if any(key.startswith("accumulated/") for key in metrics):
        by_window = {}
        for row in records:
            by_window.setdefault(identity(row), {})[row["role"]] = row
        points = []
        for rows in by_window.values():
            if "C" in rows:
                c, d = rows["C"], rows["D"]
                within = c.get("true_within_stage_progress")
                if within is None:
                    require("B" in rows, "Independent accumulated audit requires true within-stage progress or oracle B")
                    within = rows["B"]["targets"][record_poi_slot(rows["B"])]
                value = (c["predictions"][record_poi_slot(c)] + d["predictions"][record_poi_slot(d)], c["targets"][record_poi_slot(c)] + within, c["num_stages"])
                source = c
            else:
                source = rows["A"]
                value = source["predictions"][record_poi_slot(source)], source["targets"][record_poi_slot(source)], 1
            points.append((source, value))
        accum_scopes = [("overall", points)]
        accum_scopes += [(f"task/{task}", [point for point in points if point[0]["task"] == task]) for task in tasks]
        accum_scopes += [(f"group/{group}", [point for point in points if point[0].get("group", point[0]["task"]) == group])
                         for group in sorted({point[0].get("group", point[0]["task"]) for point in points})]
        for scope, subset in accum_scopes:
            count = len(subset)
            raw = [prediction - target for _, (prediction, target, _) in subset]
            normalized = [(prediction - target) / scale for _, (prediction, target, scale) in subset]
            constant = [.5 - target / scale for _, (_, target, scale) in subset]
            values = {"support": count, "trajectory_support": len({identity(source)[:3] for source, _ in subset})}
            for kind, errors in (("raw", raw), ("normalized", normalized), ("constant_0.5_normalized", constant)):
                values[kind + "_rmse"] = math.sqrt(math.fsum(error * error for error in errors) / count)
                values[kind + "_mae"] = math.fsum(map(abs, errors)) / count
                if kind != "raw":
                    values[kind + "_accuracy_within_0.1"] = sum(abs(error) <= .10000001 for error in errors) / count
            low = [(prediction - target) / scale for _, (prediction, target, scale) in subset if target / scale < .05]
            values["normalized_gt_lt_0.05_support"] = len(low)
            if low:
                values["normalized_gt_lt_0.05_rmse"] = math.sqrt(math.fsum(error * error for error in low) / len(low))
                values["normalized_gt_lt_0.05_accuracy_within_0.1"] = sum(abs(error) <= .10000001 for error in low) / len(low)
            for key, expected in values.items():
                check(metrics.get(f"accumulated/{scope}/{key}"), expected, f"accumulated/{scope}/{key}")
    return {"status": "passed", "implementation": "independent_stdlib_raw_record_arithmetic",
            "checked_values": checked, "tasks": len(tasks), "groups": len(groups)}


def verify_evaluation(path, step):
    evaluation = read(path)
    require(evaluation["optimizer_steps"] == step, f"Evaluation step mismatch: {path}")
    records, metrics = evaluation["records"], evaluation["metrics"]
    for role in "ABCD":
        rows = [row for row in records if row["role"] == role]
        if not rows:
            continue
        require(all(len(row["targets"]) == len(row["predictions"]) == 8 for row in rows), "Invalid eight-frame records")
        if role == "C":
            value = sum(row["predictions"][record_poi_slot(row)] == row["targets"][record_poi_slot(row)] for row in rows) / len(rows)
            key = "role_C_poi_accuracy"
        else:
            value = math.sqrt(sum((row["predictions"][record_poi_slot(row)] - row["targets"][record_poi_slot(row)]) ** 2 for row in rows) / len(rows))
            key = f"role_{role}_poi_rmse"
        require(close(value, metrics[key]), f"Evaluation metric does not match predictions: {key}")
    if "diagnostics" in evaluation:
        independent_quality_audit(evaluation)
    return evaluation


def verify_dense_bundle(path, evaluation):
    from .deploy_dense import SCHEMA, dense_metrics

    path = Path(path)
    bundle = read(path)
    require(bundle["schema"] == SCHEMA and bundle["optimizer_steps"] == evaluation["optimizer_steps"]
            and bundle["plan_sha256"] == evaluation["plan_sha256"], "Dense bundle step/plan differs from evaluated predictions")
    require(close(bundle["training_elapsed_seconds"], evaluation["training_elapsed_seconds"]), "Dense training time differs")
    if "records_sha256" in bundle:
        records_sha256 = hashlib.sha256(json.dumps(evaluation["records"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(bundle["records_sha256"] == records_sha256, "Dense media did not use the exact persisted evaluation records")
    records = {(r["cache"], r["task"], r["trajectory_id"], r["poi"], r["role"]): r for r in evaluation["records"]}
    identities, total = set(), 0
    for series in bundle["series"]:
        identity = (series["cache"], series["task"], series["trajectory_id"])
        require(identity not in identities, "Duplicate dense trajectory")
        identities.add(identity)
        if bundle.get("group_by_cache"):
            require(series.get("group") == bundle["group_by_cache"].get(series["cache"]), "Dense group differs from exact cache mapping")
        pois = [p["poi"] for p in series["points"]]
        require(pois == sorted(set(range(0, series["num_frames"], bundle["frame_stride"])) | {series["num_frames"] - 1}),
                "Dense series omits planned source frames")
        squared_error = sum((p["prediction"] - p["target"]) ** 2 for p in series["points"])
        require(close(series["metrics"]["raw_rmse"], math.sqrt(squared_error / len(pois)))
                and close(series["metrics"]["normalized_rmse"], math.sqrt(squared_error / len(pois)) / series["scale"]),
                "Dense trajectory RMSE differs from measured points")
        for point in series["points"]:
            key = (*identity, point["poi"])
            if series["route"] == "predicted_stage_route":
                c, d = records[(*key, "C")], records[(*key, "D")]
                require(point["predicted_stage"] == c["predictions"][record_poi_slot(c)] == d["focus_stage"]
                        and point["target_stage"] == c["targets"][record_poi_slot(c)], "Dense stage route differs from native C")
                require(c.get("group", c["task"]) == d.get("group", d["task"]) == series.get("group", series["task"]),
                        "Dense group differs from evaluated records")
                require(close(point["prediction"], c["predictions"][record_poi_slot(c)] + d["predictions"][record_poi_slot(d)]), "Dense accumulated prediction differs")
                if (*key, "B") in records:
                    b = records[(*key, "B")]
                    require(close(point["target"], c["targets"][record_poi_slot(c)] + b["targets"][record_poi_slot(b)])
                            and close(point["oracle"], c["targets"][record_poi_slot(c)] + b["predictions"][record_poi_slot(b)]), "Dense oracle/target differs")
            else:
                a = records[(*key, "A")]
                require(close(point["prediction"], a["predictions"][record_poi_slot(a)]) and close(point["target"], a["targets"][record_poi_slot(a)]),
                        "Dense whole-task series differs")
            total += 1
    require(total == bundle["evaluated_pois"] and len(identities) == bundle["available_trajectories"], "Dense POI/trajectory count differs")
    require(total == sum(row["role"] in ("C", "A") for row in evaluation["records"]), "Dense media cohort omits evaluated trajectories")
    metrics = dense_metrics(bundle["series"])
    require(metrics.keys() == bundle["metrics"].keys()
            and all(close(value, bundle["metrics"][key]) for key, value in metrics.items()), "Dense scalar metrics differ from measured series")
    media = []
    expected_choices = set()
    groups = {}
    for series in bundle["series"]:
        groups.setdefault(series.get("group", series["task"]), []).append(series)
    for group, rows in groups.items():
        expected_choices.add((group, "best_raw_rmse"))
        if len(rows) > 1:
            expected_choices.add((group, "worst_raw_rmse"))
    require({(item.get("group", item["task"]), item["selection"]) for item in bundle["items"]} == expected_choices
            and len(bundle["items"]) == len(expected_choices), "Dense media does not contain every group extremum")
    for item in bundle["items"]:
        rows = groups[item.get("group", item["task"])]
        ranked = sorted(rows, key=lambda r: (r["metrics"]["raw_rmse"], r["cache"], r["trajectory_id"]))
        chosen = ranked[0] if item["selection"] == "best_raw_rmse" else ranked[-1]
        require(chosen["trajectory_id"] == item["trajectory_id"] and chosen["task"] == item["task"]
                and chosen["cache"] == item.get("cache", chosen["cache"]), "Dense best/worst selection differs")
        require(set(item["video_pois"]) <= {p["poi"] for p in chosen["points"]}, "Dense video includes an unevaluated frame")
        for kind, suffix in (("image", ".png"), ("video", ".mp4")):
            payload = Path(item[kind]).resolve()
            require(payload.is_relative_to(path.parent.resolve()) and payload.suffix == suffix
                    and payload.stat().st_size > 0 and digest(payload) == item[kind + "_sha256"], "Dense media file/digest differs")
            media.append({"optimizer_steps": bundle["optimizer_steps"], "kind": kind, "path": str(payload),
                          "sha256": item[kind + "_sha256"], "manifest": str(path)})
    return media


def verify_expected_job_contract(job, config, result, output):
    """Enforce explicit campaign acceptance fields without changing old plans."""
    if "expected_eval_steps" in job:
        require([event["optimizer_steps"] for event in result["evaluation_events"]] == job["expected_eval_steps"],
                "Evaluation events differ from the required optimizer steps")
    if "expected_pois_per_event" in job:
        plan = read(output / "evaluation_plan.json")
        expected = job["expected_pois_per_event"]
        require(config["evaluation"]["mode"] == "dense" and plan["settings"]["metric_scope"] == "full_heldout_dense",
                "Campaign requires full held-out dense primary evaluation")
        require(plan["window_count"] == expected and plan["model_calls"] == 3 * expected
                and plan["role_window_counts"] == {"A": 0, "B": expected, "C": expected, "D": expected},
                "Full held-out B/C/D prediction coverage differs from the required POIs")
        expected_keys = {(w["cache"], w["task"], w["trajectory_id"], w["poi"], role)
                         for w in plan["windows"] for role in w["roles"]}
        for event in result["evaluation_events"]:
            step = event["optimizer_steps"]
            require(event.get("deploy_dense_reused_primary_predictions") is True
                    and event["model_calls"] == event["deploy_dense_model_calls"] == 3 * expected,
                    "Dense media did not reuse exactly one full primary prediction pass")
            primary = read(output / f"evaluation-{step:06d}.json")
            dense = read(output / f"deploy-dense-evaluation-{step:06d}.json")
            require(primary == dense and primary["provenance"]["mode"] == "during_training",
                    "Primary and dense predictions differ or were not recorded during training")
            actual_keys = [(r["cache"], r["task"], r["trajectory_id"], r["poi"], r["role"]) for r in primary["records"]]
            require(set(actual_keys) == expected_keys and len(actual_keys) == len(expected_keys)
                    and primary["plan_sha256"] == plan["plan_sha256"], "Evaluation records do not cover the exact full held-out plan")
    checkpoints = []
    for checkpoint in job.get("expected_checkpoints", []):
        checkpoint = Path(checkpoint)
        metadata = read(checkpoint / "metadata.json")
        require(checkpoint.name == f"checkpoint-{metadata['source_step']:06d}" and metadata["format"] == "grana-weights-v1",
                "Required checkpoint metadata has the wrong optimizer step/format")
        require((checkpoint / "weights.safetensors").stat().st_size > 0, "Required checkpoint weights are empty")
        from safetensors.torch import load_file
        from grana.modeling.checkpoint import tensor_digest
        require(tensor_digest(load_file(str(checkpoint / "weights.safetensors"))) == metadata["weights_sha256"],
                "Required checkpoint tensor digest differs")
        checkpoints.append({"optimizer_steps": metadata["source_step"], "path": str(checkpoint),
                            "weights_sha256": metadata["weights_sha256"]})
    return checkpoints


def summarize_job(job, jobs):
    from .evaluation_reporting import evaluation_report

    output = Path(job["output_dir"])
    result, ray = read(output / "result.json"), read(output / "ray_run.json")
    config = read(output / "resolved_config.json")
    steps = job["expected_steps"]
    checkpoint = Path(job["expected_checkpoint"])
    metadata = read(checkpoint / "metadata.json")
    require(result["status"] == "completed" and ray["status"] == "succeeded", "Canonical Ray run has not succeeded")
    require(result["optimizer_steps"] == metadata["source_step"] == steps, "Incomplete optimizer updates")
    require(result["checkpoint_reload_equal"] is True and result["checkpoint_prediction_parity"] is True,
            "Checkpoint reload/prediction verification did not pass")
    require(metadata["weights_sha256"] == result["checkpoint_load"]["weights_sha256"], "Checkpoint digest reports disagree")
    history = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines() if line.strip()]
    require([row["optimizer_steps"] for row in history] == list(range(1, steps + 1)), "Incomplete optimizer history")
    compute = sum(row["update_seconds"] for row in history)
    require(close(compute, result["training_compute_seconds"]), "Training compute time does not match actual updates")
    require(math.isfinite(result["total_elapsed_seconds"]) and result["total_elapsed_seconds"] >= compute > 0,
            "Invalid training/elapsed time")
    verified_checkpoints = verify_expected_job_contract(job, config, result, output)
    # A queue dependency only orders GPU reuse. Explicit null means this run
    # initializes independently; only older manifests infer a weight source.
    source = (job["source_pretrain"] if "source_pretrain" in job else
              job.get("dependency") if job["stage"] == "posttrain" else None)
    if source:
        predecessor = jobs[source]
        expected = read(Path(predecessor["expected_checkpoint"]) / "metadata.json")
        initialization = read(output / "initialization.json")["report"]
        require(predecessor["size"] == job["size"] and predecessor["stage"] == "pretrain", "Wrong pretrain model size")
        require(initialization["weights_sha256"] == expected["weights_sha256"], "Posttrain did not load its own pretrain")
        require(all(initialization["heads"][head]["outcome"] == "loaded" for head in ("progress_head", "temporal_head")),
                "Posttrain did not restore both auxiliary heads")
    events, media, final_metrics = [], [], {}
    for event in result["evaluation_events"]:
        step = event["optimizer_steps"]
        evaluation = verify_evaluation(output / f"evaluation-{step:06d}.json", step)
        require(close(evaluation["training_elapsed_seconds"], event["training_elapsed_seconds"]), "Evaluation time differs")
        final_metrics = evaluation["metrics"]
        events.append({"optimizer_steps": step, "training_elapsed_seconds": event["training_elapsed_seconds"],
                       "evaluation_seconds": evaluation["elapsed_seconds"],
                       "evaluation_report": evaluation_report(final_metrics, scope=config["evaluation"].get("mode")),
                       "metrics": final_metrics})
        if config["evaluation"]["renders"]["enabled"]:
            rendered = verify_evaluation(output / f"render-evaluation-{step:06d}.json", step)
            manifest_path = output / f"eval_renders/step-{step:06d}/manifest.json"
            bundle = read(manifest_path)
            require(bundle["optimizer_steps"] == step and bundle["evaluation_metrics"] == rendered["metrics"],
                    "Render step/metrics differ from render evaluation")
            require(close(bundle["training_elapsed_seconds"], event["training_elapsed_seconds"]), "Render training time differs")
            for item in bundle["items"]:
                require(item["sampled_pois"] == item["replay_pois"], "Replay omits evaluated POIs")
                for kind, suffix in (("image", ".png"), ("video", ".gif")):
                    path = Path(item[kind]).resolve()
                    require(path.is_relative_to(manifest_path.parent.resolve()) and path.suffix == suffix
                            and path.stat().st_size > 0 and digest(path) == item[kind + "_sha256"], "Render file/digest mismatch")
                    media.append({"optimizer_steps": step, "kind": kind, "path": str(path),
                                  "sha256": item[kind + "_sha256"], "manifest": str(manifest_path)})
        if config["evaluation"].get("deploy_dense", {}).get("enabled"):
            dense = verify_evaluation(output / f"deploy-dense-evaluation-{step:06d}.json", step)
            media.extend(verify_dense_bundle(output / f"deploy_dense/step-{step:06d}/manifest.json", dense))
    if config["evaluation"]["enabled"]:
        require(events and events[-1]["optimizer_steps"] == steps, "Final held-out evaluation is missing")
    logging = read(output / "wandb_run.json")
    return {"name": job["name"], "size": job["size"], "task": job["task"], "stage": job["stage"],
            "optimizer_steps": steps, "training_compute_seconds": compute,
            "engine_elapsed_seconds": result["total_elapsed_seconds"],
            "peak_allocated_gib": max(row["gpu_peak_allocated_gib"] for row in history),
            "checkpoint": str(checkpoint), "weights_sha256": metadata["weights_sha256"],
            "verified_checkpoints": verified_checkpoints,
            "source_pretrain": source,
            "final_evaluation_report": evaluation_report(final_metrics, scope=config["evaluation"].get("mode")),
            "final_heldout_metrics": final_metrics,
            "evaluation_events": events, "media": media, "wandb_url": logging.get("url"),
            "logging_mode": logging["mode"]}


def report_campaign(control):
    from .campaign_cli import load_campaign

    control, plan, state = load_campaign(control)
    by_name = {job["name"]: job for job in plan["jobs"]}
    report = {"schema": "grana-campaign-report-v1", "campaign": plan["campaign"], "control_dir": str(control),
              "scope": "Local completion, update times, checkpoint metadata, recomputed metrics, dense route composition and PNG/GIF/MP4 byte digests; no cloud requests or GPU model loading.",
              "time_semantics": "training_compute_seconds sums synchronized optimizer updates; engine_elapsed_seconds also includes evaluation/persistence, and overlapping jobs are not summed as wall time.",
              "failures": [], "jobs": []}
    if state["status"] != "succeeded" or state.get("owned_live_processes") or state.get("cleanup_survivors"):
        report["failures"].append("Campaign is incomplete or still has owned processes")
    for job in plan["jobs"]:
        try:
            require(state["jobs"][job["name"]]["status"] == "succeeded", "Job has not succeeded")
            from .campaign import verify_job_provenance
            provenance = verify_job_provenance(plan, job)
            report["jobs"].append({**summarize_job(job, by_name), "executed_provenance": provenance})
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["failures"].append(f"{job['name']}: {error}")
    report["passed"] = not report["failures"] and len(report["jobs"]) == len(plan["jobs"])
    return report


def runtime_fingerprint_from_summary(summary, expected):
    """Read nested or dotted public-API summaries without splitting package names.

    Some W&B backends flatten summary objects but preserve lists as leaf values.
    The frozen schema supplies the exact dictionary boundaries, including any
    literal dots in dependency names; missing, extra, or ambiguous leaves fail.
    """
    environment = summary.get("environment")
    if isinstance(environment, dict) and "runtime_fingerprint" in environment:
        require(isinstance(environment["runtime_fingerprint"], dict), "Remote runtime fingerprint must be an object")
        return environment["runtime_fingerprint"]
    prefix = "environment.runtime_fingerprint"
    if prefix in summary:
        require(isinstance(summary[prefix], dict), "Remote runtime fingerprint must be an object")
        return summary[prefix]

    leaves = {}
    def collect(value, path=()):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                collect(child, (*path, key))
        else:
            dotted = ".".join(path)
            require(dotted not in leaves, "Frozen runtime has ambiguous dotted summary keys")
            leaves[dotted] = path
    collect(expected)
    observed = {key[len(prefix) + 1:]: value for key, value in summary.items()
                if key.startswith(prefix + ".")}
    require(observed, "Remote runtime fingerprint is missing")
    require(observed.keys() == leaves.keys(),
            "Remote runtime fingerprint leaves differ: "
            f"missing={sorted(leaves.keys() - observed.keys())}, "
            f"unexpected={sorted(observed.keys() - leaves.keys())}")
    def rebuild(value, path=()):
        if isinstance(value, dict) and value:
            return {key: rebuild(child, (*path, key)) for key, child in value.items()}
        return observed[".".join(path)]
    return rebuild(expected)


def verify_remote_config(remote, expected):
    """Check values while accounting for W&B's omitted empty mapping fields.

    The API can omit ``group_by_cache: {}`` from an otherwise intact config.
    Only absent empty dictionaries are accepted; populated mappings, lists,
    scalar values, extra keys, and array order still have to match.
    """
    omitted = []

    def compare(actual, wanted, path):
        message = f"Remote resolved configuration differs: {path or '<root>'}"
        if isinstance(wanted, dict):
            require(isinstance(actual, dict) and not actual.keys() - wanted.keys(), message)
            for key, value in wanted.items():
                child = f"{path}.{key}" if path else key
                if key not in actual and isinstance(value, dict) and not value:
                    omitted.append(child)
                else:
                    require(key in actual, f"Remote resolved configuration missing: {child}")
                    compare(actual[key], value, child)
        elif isinstance(wanted, list):
            require(isinstance(actual, list) and len(actual) == len(wanted), message)
            for index, (left, right) in enumerate(zip(actual, wanted, strict=True)):
                compare(left, right, f"{path}[{index}]")
        else:
            require(isinstance(actual, bool) == isinstance(wanted, bool) and actual == wanted, message)

    compare(remote, expected, "")
    return {"resolved_config_equal": True, "remote_omitted_empty_config_mappings": omitted}


def audit_wandb(control, *, local_report=None):
    """Compare compact history, full cloud artifacts and uploaded media digests."""
    from copy import deepcopy
    report = report_campaign(control) if local_report is None else deepcopy(local_report)
    report["schema"] = "grana-campaign-wandb-check-v2"
    report["scope"] = "Read-only W&B finished state, complete optimizer history, compact evaluation curves, full evaluation artifact downloads and PNG/GIF/MP4 upload digests."
    report["remote"] = []
    report["remote_skipped"] = []
    if not report["passed"]:
        return report
    online = []
    for job in report["jobs"]:
        if job.get("logging_mode", "online") == "online":
            online.append(job)
        else:
            report["remote_skipped"].append({"name": job["name"], "logging_mode": job["logging_mode"]})
    if not online:
        return report
    import wandb

    try:
        # Initialize the SDK from existing credentials before constructing Public
        # API; a fresh process otherwise can report a spurious relogin error.
        wandb.login(verify=True, timeout=30)
        api = wandb.Api(timeout=60)
    except Exception as error:
        report["failures"].append(f"W&B API unavailable: {type(error).__name__}")
        report["passed"] = False
        return report
    from .campaign_cli import load_campaign
    _, plan, _ = load_campaign(control)
    by_name = {job["name"]: job for job in plan["jobs"]}
    for local in online:
        name = local["name"]
        try:
            receipt = read(Path(by_name[name]["output_dir"]) / "wandb_run.json")
            require(receipt["mode"] == "online" and receipt["status"] == "finished", "Requires an online, finished W&B receipt")
            run = api.run(f"{receipt['entity']}/{receipt['project']}/{receipt['id']}")
            summary = dict(run.summary)
            require(run.state == "finished", "Remote run is not finished")
            require(run.id == receipt["id"] and run.entity == receipt["entity"] and run.project == receipt["project"]
                    and run.name == name, "Remote run identity differs")
            config = read(Path(by_name[name]["output_dir"]) / "resolved_config.json")
            require(run.group == config["logging"]["group"], "Remote campaign group differs")
            provenance = {}
            if plan.get("runtime_fingerprint"):
                from .runtime_fingerprint import validate_runtime_fingerprint
                expected_runtime = read(plan["runtime_fingerprint"])
                remote_runtime = runtime_fingerprint_from_summary(summary, expected_runtime)
                validate_runtime_fingerprint(expected_runtime, remote_runtime)
                remote_config = {key: value for key, value in run.config.items() if not key.startswith("_")}
                provenance = {"runtime_fingerprint_sha256": remote_runtime["sha256"],
                              **verify_remote_config(remote_config, config)}
            require(summary["optimizer_steps"] == summary["logged_optimizer_steps"] == local["optimizer_steps"], "Remote optimizer count differs")
            require(summary["checkpoint_reload_equal"] is True and summary["checkpoint_prediction_parity"] is True,
                    "Remote checkpoint verification is missing")
            rows = [row for row in run.scan_history(page_size=1000) if row.get("loss") is not None and row.get("update_seconds") is not None]
            require([row["optimizer_steps"] for row in rows] == list(range(1, local["optimizer_steps"] + 1)), "Remote optimizer history is incomplete")
            require(close(summary["time/training_seconds"], local["training_compute_seconds"])
                    and close(summary["total_elapsed_seconds"], local["engine_elapsed_seconds"]), "Remote training time differs")
            from .logging_audit import audit_evaluation_artifacts
            artifacts = audit_evaluation_artifacts(
                api, run, receipt, by_name[name]["output_dir"],
                expected_steps=[event["optimizer_steps"] for event in local["evaluation_events"]])
            files = [file for file in run.files(per_page=500) if file.name.endswith((".png", ".gif", ".mp4")) and file.size > 0]
            available = Counter(file.md5 for file in files if file.md5)
            expected = Counter(base64.b64encode(bytes.fromhex(digest(item["path"], "md5"))).decode() for item in local["media"])
            require(all(available[checksum] >= count for checksum, count in expected.items()), "Uploaded PNG/GIF digests are absent or differ")
            report["remote"].append({"name": name, "state": run.state, "url": run.url,
                                      "optimizer_updates": len(rows), "verified_media": sum(expected.values()),
                                      "executed_provenance": provenance,
                                      "verified_evaluation_artifacts": artifacts})
        except Exception as error:
            # SDK/network exception strings may contain request details.
            detail = str(error) if isinstance(error, (ValueError, KeyError)) else type(error).__name__
            report["failures"].append(f"{name}: {detail}")
    report["passed"] = not report["failures"] and len(report["remote"]) == len(online)
    return report


def accept_campaign(control, *, max_attempts=3, retry_seconds=30):
    """Persist local and cloud acceptance independently of completed training.

    Every invocation keeps its reports. Retry only cloud visibility checks: a
    failed audit never invokes the coordinator or starts a new model attempt.
    """
    import time
    from .campaign import atomic_json, utc_now, verify_frozen_execution
    from .campaign_cli import load_campaign

    if max_attempts < 1 or not math.isfinite(retry_seconds) or retry_seconds < 0:
        raise ValueError("Acceptance attempts must be positive and retry seconds finite/nonnegative")
    control, plan, _ = load_campaign(control)
    directory = control / "acceptance" / str(time.time_ns())
    directory.mkdir(parents=True)
    receipt = {"schema": "grana-campaign-acceptance-v1", "campaign": plan["campaign"],
               "started_utc": utc_now(), "passed": False, "cloud_attempts": [],
               "local_report": str(directory / "local-report.json"),
               "training_restarted": False, "failures": []}
    atomic_json(control / "acceptance.json", receipt)
    try:
        receipt["frozen_execution"] = verify_frozen_execution(plan)
        local = report_campaign(control)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        local = {"passed": False, "failures": [f"{type(error).__name__}: {error}"], "jobs": []}
    atomic_json(Path(receipt["local_report"]), local)
    if not local["passed"]:
        receipt["failures"] = local["failures"]
    else:
        for attempt in range(1, max_attempts + 1):
            cloud = audit_wandb(control, local_report=local)
            path = directory / f"wandb-audit-{attempt:03d}.json"
            atomic_json(path, cloud)
            receipt["cloud_attempts"].append({"number": attempt, "report": str(path),
                                               "passed": cloud["passed"], "finished_utc": utc_now()})
            receipt["failures"] = cloud["failures"]
            atomic_json(control / "acceptance.json", receipt)
            if cloud["passed"]:
                try:
                    receipt["frozen_execution"] = verify_frozen_execution(plan)
                    receipt["passed"] = True
                    receipt["wandb_urls"] = [run["url"] for run in cloud["remote"]]
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    receipt["failures"] = [f"{type(error).__name__}: {error}"]
                break
            if attempt < max_attempts:
                time.sleep(retry_seconds)
    receipt["finished_utc"] = utc_now()
    atomic_json(directory / "receipt.json", receipt)
    atomic_json(control / "acceptance.json", receipt)
    return receipt
