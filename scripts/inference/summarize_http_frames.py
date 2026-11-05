#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and summarize paired POI/LAST HTTP evaluation journals.

The primary metric is the arithmetic mean of trajectory RMSEs. Independent
temporal sampling rounds are combined within each trajectory before its RMSE
is calculated. Fixed repeatability anchors never enter the quality metrics.
Separate macro MSE and sample-pooled micro MSE/RMSE make the weighting and
placement of the square root explicit. This script loads recorded predictions
and labels; it never runs a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grana.data.frame_sampling import DEFAULT_EVAL_STRIDE_PAIRS
from grana.inference.window import compose_frame_indices

MODES = ("poi", "last")
SLOTS = {"poi": 5, "last": 7}
METRICS = ("raw_rmse", "normalized_rmse", "C_stage_accuracy", "D_focus_relative_rmse")
MSE_METRICS = ("raw_mse", "normalized_mse")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prediction_digest(record):
    values = {name: record[name] for name in ("C_ids", "D_values", "D_focus_stage")}
    return digest(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(value, *, low=0.0, high=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= low
            and (high is None or value <= high))


def source_frames(values, count, label):
    require(isinstance(values, list) and values, f"{label} must be a nonempty list")
    require(all(type(value) is int and 0 <= value < count for value in values),
            f"{label} must contain in-range integer source frames")
    require(len(values) == len(set(values)), f"{label} must not repeat source frames")


def resolve_path(value, experiment):
    require(isinstance(value, str) and value, "Input paths must be nonempty strings")
    path = Path(value).expanduser()
    return path if path.is_absolute() else experiment / path


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == "grana-http-frame-comparison-plan-v1",
            "Expected grana-http-frame-comparison-plan-v1 plan")
    seeds = plan.get("seeds")
    require(isinstance(seeds, list) and seeds and all(type(seed) is int for seed in seeds),
            "Plan seeds must be a nonempty list of integers")
    require(len(seeds) == len(set(seeds)), "Plan seeds must be distinct")
    require(plan.get("stride_pair") == list(DEFAULT_EVAL_STRIDE_PAIRS[0]),
            "Plan stride pair must match shared evaluation CTSP(16,4)")
    samples = plan.get("samples_per_trajectory")
    require(type(samples) is int and samples > 0, "samples_per_trajectory must be positive")
    checkpoint = plan.get("checkpoint")
    require(isinstance(checkpoint, dict) and sha256(checkpoint.get("weights_sha256")),
            "Plan needs a checkpoint weights SHA256")
    trajectories = plan.get("trajectories")
    require(isinstance(trajectories, list) and trajectories, "Plan needs trajectories")
    keys = set()
    identities = set()
    for row in trajectories:
        require(isinstance(row, dict), "Each trajectory must be an object")
        key = row.get("key")
        require(isinstance(key, str) and key and key not in keys, "Trajectory keys must be unique strings")
        keys.add(key)
        require(row.get("group") in ("L1L2", "L3L4"), f"Unknown cache group for {key}")
        identity = row.get("trajectory_id")
        require(isinstance(identity, str) and identity, f"Missing trajectory identity for {key}")
        exact_identity = (row["group"], identity)
        require(exact_identity not in identities, f"Duplicate trajectory identity: {exact_identity}")
        identities.add(exact_identity)
        count, stages = row.get("num_frames"), row.get("stage_count")
        require(type(count) is int and count > 0, f"Invalid source frame count for {key}")
        require(type(stages) is int and 1 <= stages <= 26, f"Invalid stage count for {key}")
        require(finite(row.get("source_fps")) and row["source_fps"] > 0, f"Invalid FPS for {key}")
        targets = row.get("targets")
        require(isinstance(targets, dict) and set(targets) == {str(seed) for seed in seeds},
                f"Target seeds differ from the plan for {key}")
        for seed in seeds:
            source_frames(targets[str(seed)], count, f"{key} seed {seed} targets")
            require(len(targets[str(seed)]) == min(samples, count),
                    f"Wrong target count for {key} seed {seed}")
        source_frames(row.get("anchors"), count, f"{key} anchors")
        if "anchors_per_trajectory" in plan:
            maximum = plan["anchors_per_trajectory"]
            require(type(maximum) is int and maximum > 0, "anchors_per_trajectory must be positive")
            require(len(row["anchors"]) == min(maximum, count), f"Wrong anchor count for {key}")
    return plan


def validate_metadata(row, metadata):
    require(isinstance(metadata, dict), f"Invalid metadata for {row['key']}")
    for name in ("stage_ids", "within_stage_progress"):
        require(isinstance(metadata.get(name), list) and len(metadata[name]) == row["num_frames"],
                f"Metadata {name} must cover every source frame for {row['key']}")
    require(all(type(value) is int and 0 <= value < row["stage_count"] for value in metadata["stage_ids"]),
            f"Invalid metadata stage IDs for {row['key']}")
    require(all(finite(value, high=1.0) for value in metadata["within_stage_progress"]),
            f"Invalid metadata progress for {row['key']}")


def validate_record(record, row, metadata, weights):
    require(isinstance(record, dict), "Journal records must be objects")
    mode, frame = record.get("mode"), record.get("source_frame")
    require(mode in MODES, "Prediction mode must be poi or last")
    require(type(frame) is int and 0 <= frame < row["num_frames"], "Invalid source frame")
    slot = record.get("slot")
    require(type(slot) is int and slot == SLOTS[mode], "Prediction slot differs from mode")
    indices = record.get("frame_indices")
    require(isinstance(indices, list) and len(indices) == 8 and all(type(index) is int for index in indices),
            "Expected eight integer frame indices")
    expected = compose_frame_indices(frame, row["num_frames"], prediction_frame=mode)
    require(indices == list(expected), "Recorded window differs from shared CTSP composition")
    require(indices[slot] == frame, "Selected slot differs from the physical target frame")
    require(mode != "last" or max(indices) <= frame, "LAST window contains future observations")
    stages, progress = record.get("C_ids"), record.get("D_values")
    require(isinstance(stages, list) and len(stages) == 8
            and all(type(value) is int and 0 <= value < row["stage_count"] for value in stages),
            "Expected eight valid task-local C stage IDs")
    require(isinstance(progress, list) and len(progress) == 8
            and all(finite(value, high=1.0) for value in progress), "Expected eight finite D scores in [0,1]")
    require(type(record.get("D_focus_stage")) is int and record["D_focus_stage"] == stages[slot],
            "D focus differs from C at the selected slot")
    gt = record.get("gt")
    require(isinstance(gt, dict) and type(gt.get("stage_id")) is int
            and 0 <= gt["stage_id"] < row["stage_count"]
            and finite(gt.get("within_stage_progress"), high=1.0), "Invalid ground-truth values")
    require(gt["stage_id"] == metadata["stage_ids"][frame]
            and gt["within_stage_progress"] == metadata["within_stage_progress"][frame],
            "Journal ground truth differs from source metadata")
    require(record.get("checkpoint_sha256") == weights, "Journal checkpoint differs from the plan")
    if record.get("execution") == "local_batched":
        require("client_total_ms" not in record and "server_handler_ms" not in record,
                "Local batch measurements must not pretend to be HTTP timings")
        require(record.get("transport") == "local" and record.get("input_identity_only") is True,
                "Local batch records must identify their transport and input hash purpose")
        require(all(finite(record.get(name)) for name in ("inference_amortized_ms", "inference_batch_seconds")),
                "Local inference timing must be finite and nonnegative")
        require(type(record.get("batch_size")) is int and record["batch_size"] > 0
                and isinstance(record.get("batch_id"), str) and record["batch_id"],
                "Local inference must identify its physical batch")
        require(math.isclose(record["inference_amortized_ms"],
                             record["inference_batch_seconds"] * 1000 / record["batch_size"],
                             rel_tol=1e-9, abs_tol=1e-6), "Amortized timing differs from its physical batch")
    else:
        require(record.get("execution", "http") == "http", "Unknown execution route")
        require(all(finite(record.get(name)) for name in ("client_total_ms", "server_handler_ms")),
                "Request timing must be finite and nonnegative")
    require(all(sha256(record.get(name)) for name in ("input_sha256", "prediction_sha256")),
            "Input and prediction digests must be SHA256 strings")
    require(record["prediction_sha256"] == prediction_digest(record),
            "Prediction digest differs from the eight C IDs and D values")
    require(isinstance(record.get("request_id"), str) and record["request_id"], "Missing request ID")


def load_experiment(experiment):
    experiment = Path(experiment).expanduser().resolve()
    plan_path = experiment / "plan.json"
    plan_bytes = plan_path.read_bytes()
    plan = validate_plan(strict_json(plan_bytes))
    weights = plan["checkpoint"]["weights_sha256"]
    plan_sha = digest(plan_bytes)
    trajectories = {row["key"]: row for row in plan["trajectories"]}
    metadata, source_hashes = {}, []
    for key, row in trajectories.items():
        path = resolve_path(row.get("metadata_path"), experiment)
        data = path.read_bytes()
        metadata[key] = strict_json(data)
        validate_metadata(row, metadata[key])
        if "metadata_sha256" in row:
            require(row["metadata_sha256"] == digest(data), f"Metadata hash differs from the plan for {key}")
        source_hashes.append({"key": key, "path": str(path), "sha256": digest(data)})
    expected = {(seed, row["key"], kind, frame, mode)
                for seed in plan["seeds"] for row in plan["trajectories"]
                for kind, frames in (("quality", row["targets"][str(seed)]), ("anchor", row["anchors"]))
                for frame in frames for mode in MODES}
    records, request_ids, journals, input_hashes = {}, set(), [], {}
    for seed in plan["seeds"]:
        run_path = experiment / "runs" / f"seed_{seed}"
        journal_path, completion_path = run_path / "predictions.jsonl", run_path / "completion.json"
        data, completion_bytes = journal_path.read_bytes(), completion_path.read_bytes()
        completion = strict_json(completion_bytes)
        require(isinstance(completion, dict) and type(completion.get("seed")) is int
                and completion["seed"] == seed, "Completion seed differs from the plan")
        require(completion.get("plan_sha256") == plan_sha, "Completion refers to a different plan")
        require(completion.get("checkpoint_sha256") == weights, "Completion checkpoint differs from the plan")
        lines = data.splitlines()
        require(type(completion.get("requests_written")) is int and completion["requests_written"] == len(lines),
                "Completion request count differs from the journal")
        for line_number, line in enumerate(lines, 1):
            require(bool(line.strip()), f"Empty journal line at {journal_path}:{line_number}")
            record = strict_json(line)
            require(isinstance(record, dict) and type(record.get("seed")) is int and record["seed"] == seed,
                    "Journal seed differs from its run directory")
            require(isinstance(record.get("key"), str) and record["key"] in trajectories,
                    "Journal contains an undeclared trajectory")
            require(record.get("kind") in ("quality", "anchor"), "Journal contains an unknown record kind")
            require(type(record.get("source_frame")) is int and isinstance(record.get("mode"), str),
                    "Journal pair must identify an integer source frame and mode")
            identity = (seed, record["key"], record["kind"], record["source_frame"], record["mode"])
            require(identity in expected, f"Unexpected evaluation pair: {identity}")
            require(identity not in records, f"Duplicate evaluation pair: {identity}")
            validate_record(record, trajectories[record["key"]], metadata[record["key"]], weights)
            require(record["request_id"] not in request_ids, "Request IDs must be globally unique")
            input_identity = record["key"], record["source_frame"], record["mode"]
            require(input_identity not in input_hashes or input_hashes[input_identity] == record["input_sha256"],
                    f"Repeated target input hashes differ: {input_identity}")
            input_hashes[input_identity] = record["input_sha256"]
            request_ids.add(record["request_id"])
            records[identity] = record
        journals.append({"seed": seed, "path": str(journal_path), "sha256": digest(data),
                         "completion_path": str(completion_path), "completion_sha256": digest(completion_bytes),
                         "requests_written": len(lines)})
    missing = expected - records.keys()
    require(not missing, f"Missing {len(missing)} declared evaluation pairs: {sorted(missing)[:3]}")
    routes = sorted({record.get("execution", "http") for record in records.values()})
    require(len(routes) == 1, "Do not combine HTTP and local-batch measurements in one comparison")
    sources = {"experiment": str(experiment), "plan_path": str(plan_path), "plan_sha256": plan_sha,
               "journals": journals, "metadata": source_hashes, "execution_routes": routes}
    return plan, records, sources


def metrics(records, stage_count):
    require(bool(records), "Cannot summarize an empty trajectory")
    raw_errors, focus_errors, correct = [], [], 0
    for record in records:
        c, d = record["C_ids"][record["slot"]], record["D_values"][record["slot"]]
        gt = record["gt"]
        raw_errors.append(c + d - gt["stage_id"] - gt["within_stage_progress"])
        focus_target = (1.0 if c < gt["stage_id"] else 0.0 if c > gt["stage_id"]
                        else gt["within_stage_progress"])
        focus_errors.append(d - focus_target)
        correct += c == gt["stage_id"]
    raw_mse = math.fsum(error * error for error in raw_errors) / len(records)
    raw_rmse = math.sqrt(raw_mse)
    return {"support": len(records), "raw_rmse": raw_rmse, "normalized_rmse": raw_rmse / stage_count,
            "raw_mse": raw_mse, "normalized_mse": raw_mse / stage_count ** 2,
            "C_stage_accuracy": correct / len(records), "C_stage_correct": correct,
            "D_focus_relative_rmse": math.sqrt(math.fsum(error * error for error in focus_errors) / len(records))}


def winner(poi, last):
    delta = last - poi
    return "tie" if abs(delta) <= 1e-12 else "last" if delta < 0 else "poi"


def comparison(values):
    return {"delta_last_minus_poi": {name: values["last"][name] - values["poi"][name] for name in METRICS},
            "winner": winner(values["poi"]["normalized_rmse"], values["last"]["normalized_rmse"]),
            "last_relative_rmse_reduction": (None if values["poi"]["normalized_rmse"] == 0
                                              else 1 - values["last"]["normalized_rmse"] / values["poi"]["normalized_rmse"])}


def aggregate(rows, *, pooled=False):
    require(bool(rows), "Cannot aggregate an empty cohort")
    result = {"trajectory_support": len(rows),
              "support_per_mode": sum(row["metrics"]["poi"]["support"] for row in rows)}
    for mode in MODES:
        values = [row["metrics"][mode] for row in rows]
        if pooled:
            count = sum(value["support"] for value in values)
            result[mode] = {name: (math.fsum(value[name] * value["support"] for value in values) / count
                                  if name == "C_stage_accuracy" else
                                  math.sqrt(math.fsum(value[name] ** 2 * value["support"] for value in values) / count))
                            for name in METRICS}
        else:
            result[mode] = {name: math.fsum(value[name] for value in values) / len(values) for name in METRICS}
    result.update(comparison(result))
    counts = Counter(row["winner"] for row in rows)
    result["trajectory_winners"] = {name: counts[name] for name in ("poi", "last", "tie")}
    return result


def mse_aggregate(rows, *, micro=False):
    """Average squared errors before taking a root, with explicit weights.

    Macro weights each trajectory equally. Micro weights each evaluated
    observation equally, including repeated quality targets in different
    rounds. Neither operation invents unevaluated source-frame predictions.
    """
    require(bool(rows), "Cannot aggregate an empty cohort")
    counts = [row["metrics"]["poi"]["support"] for row in rows]
    require(all(row["metrics"]["last"]["support"] == count for row, count in zip(rows, counts, strict=True)),
            "POI and LAST must have identical observation counts")
    weights = counts if micro else [1] * len(rows)
    denominator = sum(weights)
    root_keys = ("raw_rmse", "normalized_rmse") if micro else ("sqrt_raw_mse", "sqrt_normalized_mse")
    result = {"trajectory_support": len(rows), "support_per_mode": sum(counts),
              "weighting": "each sampled quality observation equally" if micro else "each trajectory equally"}
    for mode in MODES:
        value = {name: math.fsum(row["metrics"][mode][name] * weight
                                for row, weight in zip(rows, weights, strict=True)) / denominator
                 for name in MSE_METRICS}
        value.update({root: math.sqrt(value[name]) for name, root in zip(MSE_METRICS, root_keys, strict=True)})
        result[mode] = value
    result["delta_last_minus_poi"] = {name: result["last"][name] - result["poi"][name]
                                      for name in (*MSE_METRICS, *root_keys)}
    result["winner"] = winner(result["poi"][root_keys[1]], result["last"][root_keys[1]])
    result["last_relative_mse_reduction"] = (None if result["poi"]["normalized_mse"] == 0 else
                                            1 - result["last"]["normalized_mse"] / result["poi"]["normalized_mse"])
    return result


def quantile(values, fraction):
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap(rows, *, repetitions=20000, seed=1729):
    require(type(repetitions) is int and repetitions > 0, "Bootstrap repetitions must be positive")
    require(type(seed) is int, "Bootstrap seed must be an integer")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    rng = random.Random(seed)
    draws = {name: [] for name in ("raw_rmse", "normalized_rmse")}
    for _ in range(repetitions):
        sampled = [rng.choice(members) for _, members in sorted(groups.items()) for _ in members]
        for name in draws:
            draws[name].append(math.fsum(row["metrics"]["last"][name] - row["metrics"]["poi"][name]
                                         for row in sampled) / len(sampled))
    result = {"method": "paired trajectory bootstrap, stratified by exact cache group",
              "resampling_unit": "trajectory, retaining both modes and all sampling rounds",
              "aggregation": "arithmetic mean of trajectory RMSE",
              "repetitions": repetitions, "seed": seed, "confidence_level": 0.95,
              "strata": {group: len(members) for group, members in sorted(groups.items())},
              "delta_convention": "LAST minus POI; negative favors LAST", "delta_last_minus_poi": {}}
    for name, values in draws.items():
        values.sort()
        result["delta_last_minus_poi"][name] = {
            "observed": math.fsum(row["metrics"]["last"][name] - row["metrics"]["poi"][name] for row in rows) / len(rows),
            "percentile_95_interval": [quantile(values, .025), quantile(values, .975)],
            "fraction_resamples_below_zero": sum(value < 0 for value in values) / len(values)}
    return result


def anchors_summary(plan, records):
    groups = []
    for row in plan["trajectories"]:
        for frame in row["anchors"]:
            for mode in MODES:
                repeated = [records[seed, row["key"], "anchor", frame, mode] for seed in plan["seeds"]]
                require(len({record["input_sha256"] for record in repeated}) == 1,
                        f"Repeated anchor input hashes differ: {row['key']} frame {frame} {mode}")
                slot = SLOTS[mode]
                group = {"key": row["key"], "source_frame": frame, "mode": mode,
                         "identical_prediction_hashes": len({record["prediction_sha256"] for record in repeated}) == 1,
                         "identical_C_ids": len({tuple(record["C_ids"]) for record in repeated}) == 1,
                         "identical_selected_C": len({record["C_ids"][slot] for record in repeated}) == 1,
                         "max_abs_D_difference": max(max(record["D_values"][index] for record in repeated)
                                                     - min(record["D_values"][index] for record in repeated) for index in range(8)),
                         "max_abs_selected_D_difference": max(record["D_values"][slot] for record in repeated)
                                                          - min(record["D_values"][slot] for record in repeated)}
                groups.append(group)
    return {"anchor_groups": len(groups), "requests": len(groups) * len(plan["seeds"]),
            "comparison": "All eight C stage IDs, all eight D scores, and selected C-conditioned D focus; no probability/logit matrices",
            "excluded_from_quality_metrics": True, "identical_input_hashes": True,
            "identical_prediction_hash_groups": sum(group["identical_prediction_hashes"] for group in groups),
            "identical_C_id_groups": sum(group["identical_C_ids"] for group in groups),
            "identical_selected_C_groups": sum(group["identical_selected_C"] for group in groups),
            "max_abs_D_difference": max(group["max_abs_D_difference"] for group in groups),
            "max_abs_selected_D_difference": max(group["max_abs_selected_D_difference"] for group in groups),
            "nonidentical_groups": [group for group in groups if not group["identical_prediction_hashes"]]}


def summarize(experiment, *, bootstrap_repetitions=20000, bootstrap_seed=1729):
    plan, records, sources = load_experiment(experiment)
    rows, unique_rows, run_rows = [], [], {seed: [] for seed in plan["seeds"]}
    overlap_groups, overlap_changed = 0, 0
    for trajectory in plan["trajectories"]:
        key, scale = trajectory["key"], trajectory["stage_count"]
        row = {name: trajectory[name] for name in ("key", "trajectory_id", "group", "num_frames", "stage_count", "source_fps")}
        values, unique_values, seed_values = {}, {}, {}
        for mode in MODES:
            measured = [records[seed, key, "quality", frame, mode] for seed in plan["seeds"]
                        for frame in trajectory["targets"][str(seed)]]
            values[mode] = metrics(measured, scale)
            by_frame = defaultdict(list)
            for record in measured:
                by_frame[record["source_frame"]].append(record)
            unique_values[mode] = metrics([members[0] for _, members in sorted(by_frame.items())], scale)
            for members in by_frame.values():
                if len(members) > 1:
                    overlap_groups += 1
                    overlap_changed += len({record["prediction_sha256"] for record in members}) > 1
        for seed in plan["seeds"]:
            seed_metrics = {mode: metrics([records[seed, key, "quality", frame, mode]
                                          for frame in trajectory["targets"][str(seed)]], scale) for mode in MODES}
            seed_values[str(seed)] = {"metrics": seed_metrics, **comparison(seed_metrics)}
            run_rows[seed].append({"key": key, "group": row["group"], **seed_values[str(seed)]})
        row.update(metrics=values, **comparison(values), seeds=seed_values,
                   unique_target_metrics=unique_values,
                   unique_target_count=unique_values["poi"]["support"],
                   unique_target_coverage=unique_values["poi"]["support"] / row["num_frames"])
        seed_winners = [seed_values[str(seed)]["winner"] for seed in plan["seeds"]]
        row["rank_consistent_across_seeds"] = len(set(seed_winners)) == 1
        row["all_seeds_favor_last"] = all(value == "last" for value in seed_winners)
        rows.append(row)
        unique_rows.append({"key": key, "group": row["group"], "metrics": unique_values, **comparison(unique_values)})

    def cohort(members, *, pooled=False, squared_error=None):
        def reduce(selected):
            return (aggregate(selected, pooled=pooled) if squared_error is None else
                    mse_aggregate(selected, micro=squared_error == "micro"))

        groups = sorted({row["group"] for row in members})
        return {"all": reduce(members),
                "groups": {group: reduce([row for row in members if row["group"] == group])
                           for group in groups}}

    quality_count = sum(record["kind"] == "quality" for record in records.values())
    anchor_count = len(records) - quality_count
    total_source_frames = sum(row["num_frames"] for row in rows)
    unique_targets = sum(row["unique_target_count"] for row in rows)
    return {"schema": ("grana-http-frame-comparison-summary-v1" if sources["execution_routes"] == ["http"]
                        else "grana-local-frame-comparison-summary-v1"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": plan["checkpoint"], "sources": sources,
            "sampling": {"seeds": plan["seeds"], "samples_per_trajectory": plan["samples_per_trajectory"],
                         "stride_pair": plan["stride_pair"], "trajectories": len(rows),
                         "source_frames": total_source_frames, "unique_quality_targets": unique_targets,
                         "unique_source_frame_coverage": unique_targets / total_source_frames,
                         "quality_requests": quality_count, "anchor_requests": anchor_count,
                         "sampling_limit_note": "All declared trajectories are evaluated at sampled target frames; this is not an exhaustive evaluation of every source frame.",
                         "comparison": "Same physical target frame per pair; separate canonical POI slot-5 and no-future LAST slot-7 windows."},
            "validation": {"all_declared_pairs_present": True, "requests": len(records),
                           "canonical_CTSP_windows": True, "selected_source_alignment": True,
                           "D_focus_matches_selected_C": True, "ground_truth_matches_metadata": True,
                           "checkpoint_constant": True, "prediction_hashes_recomputed": True,
                           "repeated_target_input_hashes_constant": True},
            "metric_definitions": {
                "prediction": "zero-based predicted C stage + D progress conditioned on that predicted stage",
                "target": "zero-based GT stage + GT within-stage progress",
                "trajectory_raw_rmse": "sqrt(mean(squared accumulated error over all quality records from the declared sampling rounds))",
                "trajectory_normalized_rmse": "trajectory_raw_rmse / that trajectory's stage_count",
                "primary_macro_rmse": "arithmetic mean of the individual trajectory RMSEs, giving each trajectory equal weight",
                "macro_mse": "arithmetic mean of individual trajectory MSEs; each trajectory has equal weight",
                "sqrt_macro_mse": "square root after averaging trajectory MSEs; generally differs from primary_macro_rmse",
                "micro_mse": "sum squared errors / count of sampled quality observations; longer sampled lists have more weight",
                "micro_rmse": "sqrt(micro_mse); this is the existing pooled_supplementary RMSE",
                "normalization_before_pooling": "divide each observation's accumulated error by its trajectory's stage_count before squaring",
                "micro_sampling_scope": "Repeated-sample micro counts each round's quality observation, including source frames selected in more than one round. Unique-target micro counts each sampled physical target once. Neither is an exhaustive all-source-frame metric.",
                "C_stage_accuracy": "mean of individual trajectory selected-stage accuracies",
                "D_focus_relative_rmse": "mean of trajectory RMSEs versus target 1/within-stage/0 when predicted focus is earlier/equal/later than the true stage",
                "tie_tolerance": 1e-12},
            "primary_macro": cohort(rows),
            "per_run_macro": [{"seed": seed, **cohort(run_rows[seed])} for seed in plan["seeds"]],
            "macro_mse": {"aggregation": "mean of trajectory MSEs; roots are explicitly named sqrt_*_mse and are not mean trajectory RMSE",
                          **cohort(rows, squared_error="macro")},
            "micro": {"aggregation": "pooled sampled quality observations across the declared rounds; anchors excluded",
                      **cohort(rows, squared_error="micro")},
            "per_run_macro_mse": [{"seed": seed, **cohort(run_rows[seed], squared_error="macro")} for seed in plan["seeds"]],
            "per_run_micro": [{"seed": seed, **cohort(run_rows[seed], squared_error="micro")} for seed in plan["seeds"]],
            "pooled_supplementary": {"aggregation": "sqrt(sum squared errors / number of sampled targets); not the primary macro metric",
                                     **cohort(rows, pooled=True)},
            "unique_targets_supplementary": {
                "selection": "One observation per (trajectory, physical target, mode), using the first declared seed containing it; anchors excluded",
                "repeated_target_mode_groups": overlap_groups, "nonidentical_prediction_groups": overlap_changed,
                **cohort(unique_rows),
                "macro_mse": cohort(unique_rows, squared_error="macro"),
                "micro": cohort(unique_rows, squared_error="micro")},
            "paired_trajectory_bootstrap": bootstrap(rows, repetitions=bootstrap_repetitions, seed=bootstrap_seed),
            "anchor_determinism": anchors_summary(plan, records),
            "rank_consistency": {"same_winner_in_every_seed": sum(row["rank_consistent_across_seeds"] for row in rows),
                                 "last_wins_every_seed": sum(row["all_seeds_favor_last"] for row in rows),
                                 "trajectory_support": len(rows)},
            "trajectories": rows}


def markdown_report(report):
    sample, primary = report["sampling"], report["primary_macro"]
    transport = "HTTP" if report["sources"].get("execution_routes", ["http"]) == ["http"] else "local batched"
    lines = [f"# Paired {transport} POI/LAST evaluation", "",
             f"Checkpoint weights: `{report['checkpoint']['weights_sha256']}`.", "",
             f"{sample['trajectories']} trajectories, {len(sample['seeds'])} sampling rounds "
             f"(seeds {', '.join(map(str, sample['seeds']))}), up to {sample['samples_per_trajectory']} paired targets per trajectory and round.",
             "", "The primary macro metric is the arithmetic mean of trajectory RMSEs. "
             "Each trajectory's RMSE combines its quality samples across all rounds; repeatability anchors are excluded.", "",
             "| Cohort | POI macro raw RMSE | LAST macro raw RMSE | POI macro normalized RMSE | LAST macro normalized RMSE | LAST-winning trajectories |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, value in [("All", primary["all"]), *primary["groups"].items()]:
        lines.append(f"| {name} | {value['poi']['raw_rmse']:.6f} | {value['last']['raw_rmse']:.6f} | "
                     f"{value['poi']['normalized_rmse']:.6f} | {value['last']['normalized_rmse']:.6f} | "
                     f"{value['trajectory_winners']['last']}/{value['trajectory_support']} |")
    lines += ["", "| Sampling seed | POI macro normalized RMSE | LAST macro normalized RMSE | LAST-winning trajectories |",
              "|---|---:|---:|---:|"]
    for run in report["per_run_macro"]:
        value = run["all"]
        lines.append(f"| {run['seed']} | {value['poi']['normalized_rmse']:.6f} | {value['last']['normalized_rmse']:.6f} | "
                     f"{value['trajectory_winners']['last']}/{value['trajectory_support']} |")
    if "micro" in report:
        macro_mse, micro = report["macro_mse"]["all"], report["micro"]["all"]
        lines += ["", "The existing headline is **mean trajectory RMSE**, not macro MSE. "
                  "Macro MSE averages squared error within each trajectory and then averages those MSEs equally. "
                  "Micro MSE pools the evaluated quality observations; micro RMSE takes its square root. "
                  "Averaging RMSEs and taking the square root after averaging MSEs are different operations.",
                  "", "| All-trajectory aggregation, normalized | POI | LAST |",
                  "|---|---:|---:|",
                  f"| Mean trajectory RMSE (existing headline) | {primary['all']['poi']['normalized_rmse']:.9f} | {primary['all']['last']['normalized_rmse']:.9f} |",
                  f"| Macro MSE | {macro_mse['poi']['normalized_mse']:.9f} | {macro_mse['last']['normalized_mse']:.9f} |",
                  f"| Square root of macro MSE | {macro_mse['poi']['sqrt_normalized_mse']:.9f} | {macro_mse['last']['sqrt_normalized_mse']:.9f} |",
                  f"| Micro MSE | {micro['poi']['normalized_mse']:.9f} | {micro['last']['normalized_mse']:.9f} |",
                  f"| Micro RMSE | {micro['poi']['normalized_rmse']:.9f} | {micro['last']['normalized_rmse']:.9f} |",
                  "", "| Cohort | Quality observations per mode | POI micro MSE | LAST micro MSE | POI micro RMSE | LAST micro RMSE |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, value in [("All", micro), *report["micro"]["groups"].items()]:
            lines.append(f"| {name} | {value['support_per_mode']} | {value['poi']['normalized_mse']:.9f} | "
                         f"{value['last']['normalized_mse']:.9f} | {value['poi']['normalized_rmse']:.6f} | "
                         f"{value['last']['normalized_rmse']:.6f} |")
        lines += ["", "| Sampling seed | POI micro MSE | LAST micro MSE | POI micro RMSE | LAST micro RMSE |",
                  "|---|---:|---:|---:|---:|"]
        for run in report["per_run_micro"]:
            value = run["all"]
            lines.append(f"| {run['seed']} | {value['poi']['normalized_mse']:.9f} | {value['last']['normalized_mse']:.9f} | "
                         f"{value['poi']['normalized_rmse']:.6f} | {value['last']['normalized_rmse']:.6f} |")
        unique = report["unique_targets_supplementary"]["micro"]["all"]
        lines += ["", f"These micro results pool {micro['support_per_mode']:,} sampled quality observations per mode, "
                  "counting each sampling round. Frames selected in multiple rounds therefore count more than once. "
                  f"The deduplicated micro result uses {unique['support_per_mode']:,} unique target frames: "
                  f"POI MSE {unique['poi']['normalized_mse']:.9f}, RMSE {unique['poi']['normalized_rmse']:.6f}; "
                  f"LAST MSE {unique['last']['normalized_mse']:.9f}, RMSE {unique['last']['normalized_rmse']:.6f}. "
                  "Neither result claims exhaustive micro error over all source frames. "
                  "Since sampling caps most trajectories at the same number of targets, these sample-pooled weights "
                  "do not reproduce full-video-duration weighting. All MSE/RMSE values above use per-trajectory stage-count normalization."]
    interval = report["paired_trajectory_bootstrap"]["delta_last_minus_poi"]["normalized_rmse"]
    low, high = interval["percentile_95_interval"]
    lines += ["", f"Macro normalized RMSE difference (LAST minus POI): {interval['observed']:+.6f}; "
              f"paired trajectory bootstrap 95% interval [{low:+.6f}, {high:+.6f}]. "
              "The bootstrap resamples whole trajectories within each cache group, retaining both modes and all rounds.",
              "", "Accumulated prediction is C stage index + D progress. Its target is GT stage index + GT within-stage progress; "
              "normalization divides by each trajectory's stage count. C accuracy and focus-relative D error are separate diagnostics in the JSON/CSV.",
              "", "| Group | Trajectory | POI normalized RMSE | LAST normalized RMSE | Winner | Same winner in every seed |",
              "|---|---|---:|---:|---|---|"]
    for row in report["trajectories"]:
        lines.append(f"| {row['group']} | `{row['trajectory_id']}` | {row['metrics']['poi']['normalized_rmse']:.6f} | "
                     f"{row['metrics']['last']['normalized_rmse']:.6f} | {row['winner']} | "
                     f"{'yes' if row['rank_consistent_across_seeds'] else 'no'} |")
    anchors = report["anchor_determinism"]
    lines += ["", f"Repeatability: {anchors['identical_prediction_hash_groups']}/{anchors['anchor_groups']} anchor groups "
              "had identical eight-slot C IDs, D scores and D focus across runs. "
              f"Maximum absolute D-score difference was {anchors['max_abs_D_difference']:.9g}.",
              "", f"Quality samples cover {sample['unique_quality_targets']:,}/{sample['source_frames']:,} unique source frames "
              f"({sample['unique_source_frame_coverage']:.1%}). {sample['sampling_limit_note']} "
              "A cohort-average advantage does not imply a win on every trajectory or frame. "
              "POI uses up to eight later source frames; LAST uses no future frames.", ""]
    return "\n".join(lines)


def write_outputs(report, output):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(markdown_report(report))
    fields = ["key", "trajectory_id", "group", "num_frames", "stage_count", "samples_per_mode", "unique_target_count",
              *[f"{mode}_{metric}" for mode in MODES for metric in (*METRICS, *MSE_METRICS)],
              "delta_raw_rmse", "delta_normalized_rmse", "winner", "rank_consistent_across_seeds",
              *[f"seed_{seed}_{mode}_{metric}" for seed in report["sampling"]["seeds"]
                for mode in MODES for metric in ("raw_rmse", "normalized_rmse", *MSE_METRICS)],
              *[f"seed_{seed}_winner" for seed in report["sampling"]["seeds"]]]
    with (output / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in report["trajectories"]:
            value = {key: row[key] for key in ("key", "trajectory_id", "group", "num_frames", "stage_count", "unique_target_count",
                                              "winner", "rank_consistent_across_seeds")}
            value.update(samples_per_mode=row["metrics"]["poi"]["support"],
                         delta_raw_rmse=row["delta_last_minus_poi"]["raw_rmse"],
                         delta_normalized_rmse=row["delta_last_minus_poi"]["normalized_rmse"])
            value.update({f"{mode}_{metric}": row["metrics"][mode][metric] for mode in MODES for metric in (*METRICS, *MSE_METRICS)})
            value.update({f"seed_{seed}_{mode}_{metric}": row["seeds"][str(seed)]["metrics"][mode][metric]
                          for seed in report["sampling"]["seeds"] for mode in MODES
                          for metric in ("raw_rmse", "normalized_rmse", *MSE_METRICS)})
            value.update({f"seed_{seed}_winner": row["seeds"][str(seed)]["winner"] for seed in report["sampling"]["seeds"]})
            writer.writerow(value)
    aggregate_fields = ["scope", "sampling_round", "cohort", "trajectory_count", "quality_observations_per_mode", "mode",
                        *[f"{scale}_{name}" for scale in ("raw", "normalized")
                          for name in ("mean_trajectory_rmse", "macro_mse", "sqrt_macro_mse", "micro_mse", "micro_rmse")]]
    aggregate_sets = [("repeated_sampling_rounds", "combined", report["primary_macro"], report["macro_mse"], report["micro"])]
    for mean_rmse, macro_mse, micro in zip(report["per_run_macro"], report["per_run_macro_mse"], report["per_run_micro"], strict=True):
        require(mean_rmse["seed"] == macro_mse["seed"] == micro["seed"], "Aggregation seeds differ")
        aggregate_sets.append(("single_sampling_round", mean_rmse["seed"], mean_rmse, macro_mse, micro))
    unique = report["unique_targets_supplementary"]
    aggregate_sets.append(("unique_sampled_targets", "deduplicated", unique, unique["macro_mse"], unique["micro"]))
    with (output / "aggregates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields, lineterminator="\n")
        writer.writeheader()
        for scope, seed, means, macros, micros in aggregate_sets:
            for group in ("all", *sorted(means["groups"])):
                mean = means["all"] if group == "all" else means["groups"][group]
                macro = macros["all"] if group == "all" else macros["groups"][group]
                micro = micros["all"] if group == "all" else micros["groups"][group]
                for mode in MODES:
                    value = {"scope": scope, "sampling_round": seed, "cohort": group, "mode": mode,
                             "trajectory_count": mean["trajectory_support"], "quality_observations_per_mode": mean["support_per_mode"]}
                    for scale in ("raw", "normalized"):
                        value.update({f"{scale}_mean_trajectory_rmse": mean[mode][f"{scale}_rmse"],
                                      f"{scale}_macro_mse": macro[mode][f"{scale}_mse"],
                                      f"{scale}_sqrt_macro_mse": macro[mode][f"sqrt_{scale}_mse"],
                                      f"{scale}_micro_mse": micro[mode][f"{scale}_mse"],
                                      f"{scale}_micro_rmse": micro[mode][f"{scale}_rmse"]})
                    writer.writerow(value)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="Root containing plan.json and runs/seed_*/ journals")
    parser.add_argument("--output", type=Path, required=True, help="New directory for JSON, trajectory/aggregation CSVs and Markdown reports")
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.output.expanduser().exists():
        raise FileExistsError(f"Use a new summary output directory: {args.output}")
    report = summarize(args.experiment, bootstrap_repetitions=args.bootstrap_repetitions, bootstrap_seed=args.bootstrap_seed)
    destination = write_outputs(report, args.output)
    print(json.dumps({"output": str(destination), "trajectories": report["sampling"]["trajectories"],
                      "primary_macro": report["primary_macro"]["all"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
