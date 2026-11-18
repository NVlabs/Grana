# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import csv
import json
import math
from pathlib import Path

import pytest

from grana.inference.window import compose_frame_indices
from scripts.inference import summarize_http_frames as summary


def write_run(experiment, seed, records, *, completion_updates=None):
    directory = experiment / "runs" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    plan_bytes = (experiment / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    complete = {"seed": seed, "plan_sha256": summary.digest(plan_bytes),
                "checkpoint_sha256": plan["checkpoint"]["weights_sha256"], "requests_written": len(records)}
    complete.update(completion_updates or {})
    (directory / "completion.json").write_text(json.dumps(complete))


@pytest.fixture
def campaign(tmp_path):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    plan = {"schema": "grana-http-frame-comparison-plan-v1", "seeds": [11, 29, 47],
            "samples_per_trajectory": 3, "anchors_per_trajectory": 2, "stride_pair": [16, 4],
            "checkpoint": {"path": "/checkpoint", "weights_sha256": "a" * 64}, "trajectories": []}
    for key, group in (("one", "L1L2"), ("two", "L3L4")):
        count = 120
        metadata_path = experiment / f"{key}_metadata.json"
        metadata_path.write_text(json.dumps({"stage_ids": [0] * count, "within_stage_progress": [0.] * count}))
        plan["trajectories"].append({"key": key, "trajectory_id": f"trajectory-{key}", "group": group,
                                     "num_frames": count, "stage_count": 2, "source_fps": 10,
                                     "targets": {str(seed): [0, 60, 119] for seed in plan["seeds"]},
                                     "anchors": [0, 119], "request_path": f"{key}_request.json",
                                     "metadata_path": str(metadata_path)})
    (experiment / "plan.json").write_text(json.dumps(plan))
    journals = {}
    for seed in plan["seeds"]:
        rows = []
        for trajectory in plan["trajectories"]:
            for kind, targets in (("quality", trajectory["targets"][str(seed)]), ("anchor", trajectory["anchors"])):
                for frame in targets:
                    for mode in summary.MODES:
                        value = (0. if trajectory["key"] == "one" else 1.) if mode == "poi" else .25
                        # Deliberately poor anchors expose accidental inclusion in quality metrics.
                        if kind == "anchor":
                            value = 1.
                        record = {"seed": seed, "key": trajectory["key"], "kind": kind, "source_frame": frame,
                                  "mode": mode, "slot": summary.SLOTS[mode],
                                  "frame_indices": list(compose_frame_indices(frame, trajectory["num_frames"], prediction_frame=mode)),
                                  "request_id": f"{seed}-{trajectory['key']}-{kind}-{frame}-{mode}",
                                  "input_sha256": summary.digest(f"{trajectory['key']}-{frame}-{mode}".encode()),
                                  "C_ids": [0] * 8, "D_values": [value] * 8, "D_focus_stage": 0,
                                  "gt": {"stage_id": 0, "within_stage_progress": 0.},
                                  "client_total_ms": 150., "server_handler_ms": 145., "checkpoint_sha256": "a" * 64}
                        record["prediction_sha256"] = summary.prediction_digest(record)
                        rows.append(record)
        journals[seed] = rows
        write_run(experiment, seed, rows)
    return experiment, plan, journals


@pytest.mark.parametrize("group", [None, "", " \t ", 1, [], {}])
def test_plan_requires_nonempty_string_cache_groups(campaign, group):
    _, plan, _ = campaign
    plan["trajectories"][0]["group"] = group
    with pytest.raises(ValueError, match="Cache group.*nonempty string"):
        summary.validate_plan(plan)


def test_arbitrary_cache_groups_keep_trajectory_identity_uniqueness(campaign):
    _, plan, _ = campaign
    first, second = plan["trajectories"]
    first.update(group="assembly-heldout", trajectory_id="same-episode")
    second.update(group="/datasets/site-b/eval", trajectory_id="same-episode")
    summary.validate_plan(plan)
    second["group"] = first["group"]
    with pytest.raises(ValueError, match="Duplicate trajectory identity"):
        summary.validate_plan(plan)


def test_cache_named_all_exports_its_metrics_separately_from_overall(campaign, tmp_path):
    experiment, plan, journals = campaign
    plan["trajectories"][0]["group"] = "all"
    plan["trajectories"][1]["group"] = "assembly-heldout"
    (experiment / "plan.json").write_text(json.dumps(plan))
    for seed, records in journals.items():
        write_run(experiment, seed, records)
    report = summary.summarize(experiment, bootstrap_repetitions=10)
    output = summary.write_outputs(report, tmp_path / "summary")
    with (output / "aggregates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    identities = {(row["scope"], row["sampling_round"], row["cohort_type"], row["cohort"], row["mode"]) for row in rows}
    assert len(identities) == len(rows) == 30
    for row in rows:
        if row["mode"] != "poi":
            continue
        expected = .5 if row["cohort_type"] == "overall" else 0. if row["cohort"] == "all" else 1.
        assert float(row["raw_mean_trajectory_rmse"]) == expected
        assert int(row["trajectory_count"]) == (2 if row["cohort_type"] == "overall" else 1)


def test_primary_is_arithmetic_mean_of_trajectory_rmse_and_anchors_are_excluded(campaign):
    experiment, _, _ = campaign
    report = summary.summarize(experiment, bootstrap_repetitions=100)
    macro = report["primary_macro"]["all"]
    pooled = report["pooled_supplementary"]["all"]
    assert macro["poi"]["raw_rmse"] == .5
    assert macro["last"]["raw_rmse"] == .25
    assert macro["poi"]["normalized_rmse"] == .25
    assert pooled["poi"]["raw_rmse"] == pytest.approx(math.sqrt(.5))
    assert macro["trajectory_winners"] == {"poi": 1, "last": 1, "tie": 0}
    assert macro["support_per_mode"] == 18
    assert report["sampling"]["quality_requests"] == 36
    assert report["sampling"]["anchor_requests"] == 24
    assert report["anchor_determinism"]["excluded_from_quality_metrics"] is True
    assert report["anchor_determinism"]["identical_prediction_hash_groups"] == 8
    assert report["anchor_determinism"]["max_abs_D_difference"] == 0
    assert all(run["all"]["poi"]["raw_rmse"] == .5 for run in report["per_run_macro"])
    assert report["primary_macro"]["groups"]["L1L2"]["poi"]["raw_rmse"] == 0
    assert report["rank_consistency"]["last_wins_every_seed"] == 1
    assert report["unique_targets_supplementary"]["all"]["support_per_mode"] == 6
    assert report["unique_targets_supplementary"]["repeated_target_mode_groups"] == 12
    assert report["sampling"]["unique_quality_targets"] == 6


def test_macro_mse_mean_rmse_and_micro_rmse_keep_square_root_placement_explicit(campaign):
    experiment, _, _ = campaign
    report = summary.summarize(experiment, bootstrap_repetitions=10)
    mean_rmse = report["primary_macro"]["all"]["poi"]
    macro_mse = report["macro_mse"]["all"]["poi"]
    micro = report["micro"]["all"]["poi"]
    # The two trajectories have errors 0 and 1. Their equally weighted mean
    # RMSE is .5, whereas taking the root after mean MSE gives sqrt(.5).
    assert mean_rmse["raw_rmse"] == .5
    assert macro_mse["raw_mse"] == .5
    assert macro_mse["normalized_mse"] == .125
    assert macro_mse["sqrt_raw_mse"] == pytest.approx(math.sqrt(.5))
    assert macro_mse["sqrt_raw_mse"] != mean_rmse["raw_rmse"]
    assert micro["raw_mse"] == .5
    assert micro["normalized_mse"] == .125
    assert micro["raw_rmse"] == pytest.approx(math.sqrt(.5))
    assert micro["normalized_rmse"] == pytest.approx(math.sqrt(.125))
    assert report["micro"]["all"]["support_per_mode"] == 18
    for run in report["per_run_micro"]:
        assert run["all"]["poi"]["normalized_mse"] == .125
        assert run["all"]["support_per_mode"] == 6
    for cohort, value in [("all", report["micro"]["all"]), *report["micro"]["groups"].items()]:
        old = report["pooled_supplementary"]["all"] if cohort == "all" else report["pooled_supplementary"]["groups"][cohort]
        for mode in summary.MODES:
            assert value[mode]["normalized_rmse"] == pytest.approx(old[mode]["normalized_rmse"])
    assert "Neither is an exhaustive all-source-frame metric" in report["metric_definitions"]["micro_sampling_scope"]


def test_local_batch_timings_are_separate_from_http_and_do_not_change_quality(campaign):
    experiment, _, journals = campaign
    previous = summary.summarize(experiment, bootstrap_repetitions=10)
    for seed, records in journals.items():
        for record in records:
            record.pop("client_total_ms")
            record.pop("server_handler_ms")
            record.update(execution="local_batched", transport="local", input_identity_only=True,
                          inference_amortized_ms=100., inference_batch_seconds=.2,
                          batch_id=f"batch-{record['request_id']}", batch_size=2, worker_device=0)
        write_run(experiment, seed, records)
    report = summary.summarize(experiment, bootstrap_repetitions=10)
    assert report["schema"] == "grana-local-frame-comparison-summary-v1"
    assert report["sources"]["execution_routes"] == ["local_batched"]
    assert report["primary_macro"] == previous["primary_macro"]
    assert report["micro"] == previous["micro"]
    assert summary.markdown_report(report).startswith("# Paired local batched")
    row = journals[11][0]
    row["client_total_ms"] = 100.
    write_run(experiment, 11, journals[11])
    with pytest.raises(ValueError, match="pretend to be HTTP"):
        summary.summarize(experiment, bootstrap_repetitions=10)
    row.pop("client_total_ms")
    row["inference_amortized_ms"] = 1.
    write_run(experiment, 11, journals[11])
    with pytest.raises(ValueError, match="Amortized timing"):
        summary.summarize(experiment, bootstrap_repetitions=10)


def test_unique_micro_counts_each_sampled_target_once_and_retains_round_results(campaign):
    experiment, _, journals = campaign
    for row in journals[29]:
        if row["key"] == "two" and row["kind"] == "quality" and row["mode"] == "last":
            row["D_values"] = [.5] * 8
            row["prediction_sha256"] = summary.prediction_digest(row)
    write_run(experiment, 29, journals[29])
    report = summary.summarize(experiment, bootstrap_repetitions=10)
    micro = report["micro"]["all"]
    unique = report["unique_targets_supplementary"]["micro"]["all"]
    assert micro["support_per_mode"] == 18
    assert unique["support_per_mode"] == 6
    assert micro["last"]["raw_mse"] == pytest.approx(.09375)
    assert unique["last"]["raw_mse"] == pytest.approx(.0625)
    assert report["per_run_micro"][1]["all"]["last"]["raw_mse"] == pytest.approx((.25 ** 2 + .5 ** 2) / 2)
    assert report["per_run_micro"][0]["all"]["last"]["raw_mse"] == pytest.approx(.25 ** 2)
    assert report["unique_targets_supplementary"]["nonidentical_prediction_groups"] == 3


def test_accumulated_rmse_is_not_focus_relative_d_rmse(campaign):
    experiment, _, journals = campaign
    for seed, records in journals.items():
        for record in records:
            if record["key"] == "one" and record["kind"] == "quality" and record["mode"] == "poi":
                record.update(C_ids=[1] * 8, D_values=[.2] * 8, D_focus_stage=1)
                record["prediction_sha256"] = summary.prediction_digest(record)
        write_run(experiment, seed, records)
    report = summary.summarize(experiment, bootstrap_repetitions=100)
    metrics = report["trajectories"][0]["metrics"]["poi"]
    assert metrics["raw_rmse"] == pytest.approx(1.2)
    assert metrics["normalized_rmse"] == pytest.approx(.6)
    assert metrics["D_focus_relative_rmse"] == pytest.approx(.2)
    assert metrics["C_stage_accuracy"] == 0


@pytest.mark.parametrize("damage,match", [
    ("missing", "Missing.*evaluation pairs"),
    ("duplicate", "Duplicate evaluation pair"),
    ("unexpected_frame", "Unexpected evaluation pair"),
    ("wrong_seed", "Journal seed differs"),
    ("unknown_key", "undeclared trajectory"),
    ("bad_focus", "D focus differs"),
    ("wrong_geometry", "CTSP composition"),
    ("wrong_slot", "Prediction slot differs"),
    ("bad_gt", "ground truth differs"),
    ("wrong_checkpoint", "Journal checkpoint differs"),
    ("out_of_range_D", "finite D scores"),
    ("float_C", "valid task-local C"),
    ("bad_prediction_hash", "Prediction digest differs"),
    ("bad_request_id", "globally unique"),
])
def test_incomplete_or_inconsistent_data_fails_without_silently_dropping_pairs(campaign, damage, match):
    experiment, _, journals = campaign
    records = deepcopy(journals[11])
    record = records[0]
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records.append(deepcopy(record))
    elif damage == "unexpected_frame":
        record["source_frame"] = 1
    elif damage == "wrong_seed":
        record["seed"] = 29
    elif damage == "unknown_key":
        record["key"] = "undeclared"
    elif damage == "bad_focus":
        record["D_focus_stage"] = 1
    elif damage == "wrong_geometry":
        record["frame_indices"][0] = 1
    elif damage == "wrong_slot":
        record["slot"] = 7
    elif damage == "bad_gt":
        record["gt"]["within_stage_progress"] = .1
    elif damage == "wrong_checkpoint":
        record["checkpoint_sha256"] = "b" * 64
    elif damage == "out_of_range_D":
        record["D_values"][0] = 1.1
    elif damage == "float_C":
        record["C_ids"][0] = 0.
    elif damage == "bad_prediction_hash":
        record["prediction_sha256"] = "b" * 64
    elif damage == "bad_request_id":
        records[1]["request_id"] = record["request_id"]
    write_run(experiment, 11, records)
    with pytest.raises(ValueError, match=match):
        summary.summarize(experiment, bootstrap_repetitions=10)


@pytest.mark.parametrize("field,value,match", [
    ("seed", 99, "Completion seed differs"),
    ("plan_sha256", "b" * 64, "different plan"),
    ("checkpoint_sha256", "b" * 64, "Completion checkpoint differs"),
    ("requests_written", 0, "request count differs"),
])
def test_completion_receipt_is_required_and_bound_to_plan(campaign, field, value, match):
    experiment, _, journals = campaign
    write_run(experiment, 11, journals[11], completion_updates={field: value})
    with pytest.raises(ValueError, match=match):
        summary.load_experiment(experiment)


def test_short_trajectory_caps_samples_and_normalizes_by_its_own_stage_count(campaign):
    experiment, plan, journals = campaign
    short = plan["trajectories"][1]
    short.update(num_frames=2, stage_count=4, targets={str(seed): [0, 1] for seed in plan["seeds"]}, anchors=[0, 1])
    Path(short["metadata_path"]).write_text(json.dumps({"stage_ids": [0, 0], "within_stage_progress": [0., 0.]}))
    (experiment / "plan.json").write_text(json.dumps(plan))
    for seed, rows in journals.items():
        records = []
        for record in rows:
            record = deepcopy(record)
            if record["key"] == "two":
                if record["source_frame"] == 60:
                    continue
                record["source_frame"] = int(record["source_frame"] == 119)
                record["frame_indices"] = list(compose_frame_indices(record["source_frame"], 2, prediction_frame=record["mode"]))
            records.append(record)
        write_run(experiment, seed, records)
    report = summary.summarize(experiment, bootstrap_repetitions=50)
    short_result = report["trajectories"][1]
    assert short_result["metrics"]["poi"]["support"] == 6
    assert short_result["metrics"]["poi"]["raw_rmse"] == 1
    assert short_result["metrics"]["poi"]["normalized_rmse"] == .25
    assert short_result["unique_target_count"] == 2
    assert short_result["unique_target_coverage"] == 1
    assert report["primary_macro"]["all"]["poi"]["normalized_rmse"] == .125
    assert report["primary_macro"]["all"]["support_per_mode"] == 15
    # Equal-trajectory macro MSE weights the 9- and 6-observation trajectories
    # equally. Micro uses their actual observation counts. Normalization must
    # happen before pooling because these two trajectories have 2 and 4 stages.
    assert report["macro_mse"]["all"]["poi"]["raw_mse"] == .5
    assert report["macro_mse"]["all"]["poi"]["normalized_mse"] == .03125
    assert report["macro_mse"]["all"]["poi"]["sqrt_normalized_mse"] == pytest.approx(math.sqrt(.03125))
    micro = report["micro"]["all"]["poi"]
    assert micro["raw_mse"] == pytest.approx(6 / 15)
    assert micro["normalized_mse"] == pytest.approx(6 * .25 ** 2 / 15)
    assert micro["raw_rmse"] == pytest.approx(math.sqrt(.4))
    assert micro["normalized_rmse"] == pytest.approx(math.sqrt(.025))


def test_anchor_differences_are_reported_and_input_changes_are_rejected(campaign):
    experiment, _, journals = campaign
    records = deepcopy(journals[29])
    changed = next(row for row in records if row["kind"] == "anchor" and row["mode"] == "last")
    changed["D_values"][7] -= .01
    changed["prediction_sha256"] = summary.prediction_digest(changed)
    write_run(experiment, 29, records)
    report = summary.summarize(experiment, bootstrap_repetitions=50)
    assert report["anchor_determinism"]["identical_prediction_hash_groups"] == 7
    assert report["anchor_determinism"]["identical_C_id_groups"] == 8
    assert report["anchor_determinism"]["max_abs_D_difference"] == pytest.approx(.01)
    assert report["anchor_determinism"]["max_abs_selected_D_difference"] == pytest.approx(.01)
    assert report["primary_macro"]["all"]["last"]["raw_rmse"] == .25
    changed["input_sha256"] = "c" * 64
    write_run(experiment, 29, records)
    with pytest.raises(ValueError, match="Repeated target input hashes differ"):
        summary.summarize(experiment, bootstrap_repetitions=10)


def test_bootstrap_uses_paired_trajectory_rmse_and_preserves_groups(campaign):
    experiment, _, _ = campaign
    report = summary.summarize(experiment, bootstrap_repetitions=200, bootstrap_seed=123)
    bootstrap = report["paired_trajectory_bootstrap"]
    assert bootstrap["repetitions"] == 200 and bootstrap["strata"] == {"L1L2": 1, "L3L4": 1}
    # Each group has one trajectory, so stratified resampling must keep their exact mean difference.
    assert bootstrap["delta_last_minus_poi"]["raw_rmse"]["percentile_95_interval"] == [-.25, -.25]
    assert bootstrap["delta_last_minus_poi"]["normalized_rmse"]["percentile_95_interval"] == [-.125, -.125]
    assert bootstrap == summary.summarize(experiment, bootstrap_repetitions=200, bootstrap_seed=123)["paired_trajectory_bootstrap"]


def test_known_windows_are_validated_and_cli_writes_explicit_aggregation_outputs(campaign, tmp_path):
    experiment, plan, journals = campaign
    poi = next(row for row in journals[11] if row["source_frame"] == 60 and row["mode"] == "poi")
    last = next(row for row in journals[11] if row["source_frame"] == 60 and row["mode"] == "last")
    assert poi["frame_indices"] == [12, 28, 44, 52, 56, 60, 64, 68]
    assert last["frame_indices"] == [4, 20, 36, 44, 48, 52, 56, 60]
    # Distinct seed-29 quality results ensure per-seed CSV columns do not repeat
    # the combined metric or another sampling round's result.
    for row in journals[29]:
        if row["key"] == "two" and row["kind"] == "quality" and row["mode"] == "last":
            row["D_values"] = [.5] * 8
            row["prediction_sha256"] = summary.prediction_digest(row)
    write_run(experiment, 29, journals[29])
    output = tmp_path / "summary"
    summary.main(["--experiment", str(experiment), "--output", str(output), "--bootstrap-repetitions", "50"])
    report = json.loads((output / "summary.json").read_text())
    assert report["sampling"]["trajectories"] == len(plan["trajectories"])
    with (output / "trajectories.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[0]["key"] == "one"
    assert float(rows[1]["poi_raw_rmse"]) == 1
    assert float(rows[1]["poi_raw_mse"]) == 1
    assert float(rows[1]["poi_normalized_mse"]) == .25
    for seed, last_raw in ((11, .25), (29, .5), (47, .25)):
        assert float(rows[1][f"seed_{seed}_poi_raw_rmse"]) == 1
        assert float(rows[1][f"seed_{seed}_poi_normalized_rmse"]) == .5
        assert float(rows[1][f"seed_{seed}_last_raw_rmse"]) == last_raw
        assert float(rows[1][f"seed_{seed}_last_normalized_rmse"]) == last_raw / 2
        assert float(rows[1][f"seed_{seed}_last_raw_mse"]) == last_raw ** 2
        assert float(rows[1][f"seed_{seed}_last_normalized_mse"]) == (last_raw / 2) ** 2
    with (output / "aggregates.csv").open() as stream:
        aggregates = list(csv.DictReader(stream))
    assert len(aggregates) == 30  # all / two groups, two modes, combined / 3 seeds / unique
    by_identity = {(row["scope"], row["sampling_round"], row["cohort"], row["mode"]): row for row in aggregates}
    seed29 = by_identity["single_sampling_round", "29", "all", "last"]
    assert float(seed29["raw_micro_mse"]) == (.25 ** 2 + .5 ** 2) / 2
    combined = by_identity["repeated_sampling_rounds", "combined", "all", "poi"]
    assert float(combined["raw_mean_trajectory_rmse"]) == .5
    assert float(combined["raw_macro_mse"]) == .5
    assert float(combined["raw_sqrt_macro_mse"]) == pytest.approx(math.sqrt(.5))
    assert b"\r\n" not in (output / "aggregates.csv").read_bytes()
    assert b"\r\n" not in (output / "trajectories.csv").read_bytes()
    assert "arithmetic mean of trajectory RMSEs" in (output / "report.md").read_text()
    assert "not macro MSE" in (output / "report.md").read_text()
    assert "Neither result claims exhaustive micro error" in (output / "report.md").read_text()
    with pytest.raises(FileExistsError, match="new summary output"):
        summary.main(["--experiment", str(experiment), "--output", str(output)])


def test_nonfinite_json_and_duplicate_object_keys_are_rejected():
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        summary.strict_json('{"value": NaN}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        summary.strict_json('{"key": 1, "key": 2}')
