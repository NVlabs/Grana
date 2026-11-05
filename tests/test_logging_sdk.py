# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the installed W&B SDK's persisted history without network or GPUs."""
import json

from PIL import Image

from grana.training.logging import RunLogger


def test_offline_sdk_persists_one_explicit_step_for_training_eval_and_media(tmp_path, monkeypatch):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr("grana.training.logging.environment_summary", lambda: {"test": "CPU offline SDK"})
    cfg = {"training": {"output_dir": str(tmp_path / "run"), "stage": "pretrain"},
           "roles": {"B": 2, "C": 3, "D": 3}, "loss": {"role_weights": {"B": 1, "C": 10, "D": 1}},
           "logging": {"mode": "offline", "project": "grana-step-test", "entity": None,
                       "group": None, "name": "offline-step-test", "job_type": "test",
                       "tags": [], "upload_checkpoint_weights": False}}
    image_path, video_path = tmp_path / "sample.png", tmp_path / "sample.gif"
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    frames[0].save(image_path)
    frames[0].save(video_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    with RunLogger(cfg) as logger:
        for step in range(1, 5):
            logger.log_step({"loss": 1 / step, "time/training_step": float(step)}, step=step)
            if step in (2, 4):
                # Native ordering: media is queued before the complete eval time.
                logger.log_evaluation_renders(
                    {"schema": "grana-training-eval-renders-v1", "optimizer_steps": step,
                     "training_elapsed_seconds": float(step), "rendering_seconds": .1,
                     "plan_sha256": "plan", "replay_type": "sampled", "available_trajectories": 1,
                     "rendered_trajectories": 1, "manifest": str(manifest),
                     "items": [{"key": "stable", "task": "task", "trajectory_id": "heldout",
                                "image": str(image_path), "video": str(video_path), "video_format": "gif",
                                "metrics": {}, "sampled_pois": [0, 1], "replay_pois": [0, 1]}]}, step=step)
                logger.log_evaluation({"role_C_poi_accuracy": .75, "evaluation_and_persistence_seconds": .5,
                                       "role_A_poi_rmse": .4, "role_A_task_macro_poi_rmse": .3,
                                       "accumulated/overall/raw_rmse": .4,
                                       "accumulated/overall/normalized_rmse": .4}, step=step)
            if step < 4:
                logger.commit_step(step=step)
        # The final pending step must be committed by successful finish as well.
    files = list((tmp_path / "run").rglob("*.wandb"))
    assert len(files) == 1
    store = DataStore()
    store.open_for_scan(str(files[0]))
    rows = []
    try:
        while (payload := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            if record.WhichOneof("record_type") == "history":
                row = {}
                for item in record.history.item:
                    keys = list(item.nested_key) or [item.key]
                    current = row
                    for key in keys[:-1]:
                        current = current.setdefault(key, {})
                    current[keys[-1]] = json.loads(item.value_json)
                rows.append(row)
    finally:
        store.close()
    assert len(rows) == 4
    assert [row["_step"] for row in rows] == [1, 2, 3, 4]
    assert [row["optimizer_steps"] for row in rows] == [1, 2, 3, 4]
    for step, row in enumerate(rows, start=1):
        assert row["loss"] == 1 / step and row["time/training_step"] == float(step)
        if step in (2, 4):
            assert row["eval/role_C_poi_accuracy"] == .75
            assert row["eval/evaluation_and_persistence_seconds"] == .5
            assert row["eval/overall/compressed_whole_task_progress_task_macro_raw_rmse"] == .3
            assert row["eval/overall/compressed_whole_task_progress_poi_micro_raw_rmse"] == .4
            assert row["eval/overall/compressed_whole_task_progress_poi_micro_normalized_rmse"] == .4
            assert "eval/overall/compressed_whole_task_progress_raw_rmse" not in row
            assert "eval/overall/compressed_whole_task_progress_task_macro_rmse" not in row
            assert row["eval/renders/stable"]["_type"] == "image-file"
            assert row["eval/replays/stable"]["_type"] == "video-file"
        else:
            assert not any(key.startswith("eval/") for key in row)
    receipt = json.loads((tmp_path / "run/wandb_run.json").read_text())
    assert receipt["evaluation_logging_schema"] == "grana-compact-eval-v3"
