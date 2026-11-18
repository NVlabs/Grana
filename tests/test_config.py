# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest
import yaml

from grana.config import load_config


RECIPE = Path(__file__).resolve().parents[1] / "configs/pretrain.yaml"


@pytest.mark.parametrize("command,check_data_available", [("train", False), ("validate", True)])
def test_cli_help_offers_data_check_only_for_validate(monkeypatch, capsys, command, check_data_available):
    from grana import cli
    monkeypatch.setattr(sys, "argv", ["grana", command, "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert ("--check-data" in capsys.readouterr().out) is check_data_available


def test_train_rejects_check_data_before_loading_or_launching(monkeypatch, capsys):
    from grana import cli
    monkeypatch.setattr(sys, "argv", ["grana", "train", "--config", "missing.yaml", "--check-data"])
    monkeypatch.setattr(cli, "load_config", lambda *_: pytest.fail("Invalid CLI options loaded the recipe"))
    monkeypatch.setattr(cli.subprocess, "run", lambda *_, **__: pytest.fail("Invalid CLI options launched training"))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "unrecognized arguments: --check-data" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["pretrain.yaml", "posttrain.yaml"])
def test_sample_prefetch_defaults_on_and_can_be_disabled(name):
    path = RECIPE.parent / name
    baseline = load_config(path)
    candidate = load_config(path, ["training.prefetch_samples=false"])
    assert baseline["training"]["prefetch_samples"] is True
    assert candidate["training"]["prefetch_samples"] is False
    for section in ("data", "roles", "loss", "evaluation", "model"):
        assert candidate[section] == baseline[section]


@pytest.mark.parametrize("raw", ["null", "1", "0", "'true'", "[]"])
def test_sample_prefetch_requires_boolean(raw):
    with pytest.raises(ValueError, match="training.prefetch_samples"):
        load_config(RECIPE, [f"training.prefetch_samples={raw}"])


def test_gpu_resize_policy_is_shared_by_training_and_evaluation():
    assert load_config(RECIPE)["data"]["gpu_image_resize"] is False
    assert load_config(RECIPE, ["data.gpu_image_resize=true"])["data"]["gpu_image_resize"] is True
    # A separate eval toggle would allow the two modes to silently diverge.
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(RECIPE, ["evaluation.gpu_image_resize=true"])


@pytest.mark.parametrize("raw", ["null", "1", "0", "'true'", "[]"])
def test_gpu_resize_policy_requires_boolean(raw):
    with pytest.raises(ValueError, match="data.gpu_image_resize"):
        load_config(RECIPE, [f"data.gpu_image_resize={raw}"])


@pytest.mark.parametrize("name", ["pretrain.yaml", "posttrain.yaml"])
def test_train_and_eval_have_independent_stride_defaults_and_overrides(name):
    path = RECIPE.parent / name
    training_pairs = [[10, 2], [16, 4], [24, 6], [32, 8]]
    cfg = load_config(path)
    assert cfg["data"]["stride_pairs"] == training_pairs
    assert cfg["evaluation"]["stride_pairs"] == [[16, 4]]
    train_override = load_config(path, ["data.stride_pairs=[[40,4],[20,2]]"])
    assert train_override["data"]["stride_pairs"] == [[40, 4], [20, 2]]
    assert train_override["evaluation"]["stride_pairs"] == [[16, 4]]
    eval_override = load_config(path, ["evaluation.stride_pairs=[[24,6]]"])
    assert eval_override["data"]["stride_pairs"] == training_pairs
    assert eval_override["evaluation"]["stride_pairs"] == [[24, 6]]


@pytest.mark.parametrize("section", ["data", "evaluation"])
@pytest.mark.parametrize("raw", ["[]", "null", "[[16]]", "[[16,4,2]]", "[[true,4]]", "[[16,0]]", "[[16,4],[16,4]]"])
def test_stride_support_requires_nonempty_unique_positive_integer_pairs(section, raw):
    with pytest.raises(ValueError, match=f"{section}.stride_pairs"):
        load_config(RECIPE, [f"{section}.stride_pairs={raw}"])


@pytest.mark.parametrize("name", ["pretrain.yaml", "posttrain.yaml"])
def test_two_layer_sampling_defaults_and_canonical_normalization(name):
    path = RECIPE.parent / name
    defaults = load_config(path)["data"]["train_frame_sampling"]
    assert defaults == {
        "frame_selection_profile_weights": {"regular_ctsp": .6, "irregular_stratified": .4},
        "query_order_profile_weights": {"forward": .5, "backward": .2, "tail_rewind": .3}}
    actual = load_config(path, [
        "data.train_frame_sampling.frame_selection_profile_weights={irregular_stratified: 8, regular_ctsp: 2}",
        "data.train_frame_sampling.query_order_profile_weights={tail_rewind: 3, backward: 1, forward: 0}",
    ])["data"]["train_frame_sampling"]
    assert list(actual["frame_selection_profile_weights"]) == ["regular_ctsp", "irregular_stratified"]
    assert tuple(actual["frame_selection_profile_weights"].values()) == (.2, .8)
    assert tuple(actual["query_order_profile_weights"].values()) == (0., .25, .75)


@pytest.mark.parametrize("key", ["frame_selection_profile_weights.regular_ctsp", "query_order_profile_weights.backward"])
@pytest.mark.parametrize("raw", ["true", "false", "-1", "null", "[]", "'1'", ".inf", ".nan"])
def test_invalid_two_layer_sampling_weights_fail_early(key, raw):
    with pytest.raises(ValueError, match="train_frame_sampling"):
        load_config(RECIPE, [f"data.train_frame_sampling.{key}={raw}"])


@pytest.mark.parametrize("override", [
    "profile=two_layer_v1", "enabled=true", "frame_selection_profile_weights.random=1",
    "frame_selection_profile_weights.uniform_window=0", "frame_selection_profile_weights.uniform_window=1",
    "query_order_profile_weights.reverse=1",
    "frame_selection_profile_weights={regular_ctsp: 0, irregular_stratified: 0}",
    "query_order_profile_weights={forward: 0, backward: 0, tail_rewind: 0}",
])
def test_two_layer_sampling_has_no_gate_or_unknown_or_zero_support(override):
    with pytest.raises(ValueError, match="configuration key|positive total"):
        load_config(RECIPE, [f"data.train_frame_sampling.{override}"])


def test_unified_loss_has_independent_cdf_and_ceiling_coefficients():
    cfg = load_config(RECIPE)
    assert cfg["loss"]["stage"] == {"cdf_weight": 1.0}
    assert cfg["loss"]["progress"] == {"cdf_weight": 1.0, "ceiling_weight": 0.0}
    cfg = load_config(RECIPE, ["loss.stage.cdf_weight=0.5", "loss.progress.cdf_weight=2.0",
                              "loss.progress.ceiling_weight=3.0"])
    assert cfg["loss"]["stage"]["cdf_weight"] == 0.5
    assert cfg["loss"]["progress"]["cdf_weight"] == 2.0
    assert cfg["loss"]["progress"]["ceiling_weight"] == 3.0


@pytest.mark.parametrize("head", ["stage", "progress"])
def test_loss_has_no_profile_selector(head):
    with pytest.raises(ValueError, match=f"Unknown configuration key: loss.{head}.profile"):
        load_config(RECIPE, [f"loss.{head}.profile=ce_cdf"])


@pytest.mark.parametrize("key", ["stage.cdf_weight", "progress.cdf_weight", "progress.ceiling_weight"])
@pytest.mark.parametrize("weight", ["-1", "true", "null", ".nan", ".inf", "[]", "'1'"])
def test_invalid_loss_coefficients_are_rejected(key, weight):
    with pytest.raises(ValueError, match=f"loss.{key}"):
        load_config(RECIPE, [f"loss.{key}={weight}"])


def test_mapping_override_keeps_other_role_lanes():
    cfg = load_config(RECIPE, ["loss.role_weights={C: 2.5}"])
    assert cfg["loss"]["role_weights"] == {"A": 1.0, "B": 1.0, "C": 2.5, "D": 1.0}


def test_delta_rule_backend_defaults_to_auto_and_allows_torch():
    assert load_config(RECIPE)["model"]["delta_rule_backend"] == "auto"
    assert load_config(RECIPE, ["model.delta_rule_backend=torch"])["model"]["delta_rule_backend"] == "torch"


def test_training_profiling_is_bounded_and_opt_in():
    assert load_config(RECIPE)["training"]["profile_steps"] == 0
    assert load_config(RECIPE, ["training.profile_steps=8"])["training"]["profile_steps"] == 8


def test_eval_batch_and_separate_vision_optimization_defaults():
    cfg = load_config(RECIPE)
    assert cfg["evaluation"]["batch_size"] == 1
    assert cfg["evaluation"]["batched_vision_sdpa"] is False
    assert cfg["training"]["batched_vision_sdpa"] is False
    cfg = load_config(RECIPE, ["evaluation.batch_size=64", "evaluation.batched_vision_sdpa=true"])
    assert cfg["evaluation"]["batch_size"] == 64
    assert cfg["evaluation"]["batched_vision_sdpa"] is True
    assert cfg["training"]["batched_vision_sdpa"] is False
    assert cfg["model"]["gradient_checkpointing"] is True


@pytest.mark.parametrize("raw", ["0", "-1", "true", "false", "1.5", "null", "'64'", "[]"])
def test_invalid_eval_batch_size_fails_before_launch(raw):
    with pytest.raises(ValueError, match="evaluation.batch_size"):
        load_config(RECIPE, [f"evaluation.batch_size={raw}"])


@pytest.mark.parametrize("section", ["training", "evaluation"])
@pytest.mark.parametrize("raw", ["1", "0", "null", "'true'", "[]"])
def test_vision_optimization_switch_requires_boolean(section, raw):
    with pytest.raises(ValueError, match=f"{section}.batched_vision_sdpa"):
        load_config(RECIPE, [f"{section}.batched_vision_sdpa={raw}"])


@pytest.mark.parametrize("raw", ["-1", "true", "1.5", "null"])
def test_invalid_training_profile_budget_fails_before_launch(raw):
    with pytest.raises(ValueError, match="training.profile_steps"):
        load_config(RECIPE, [f"training.profile_steps={raw}"])


@pytest.mark.parametrize("raw", ["null", "true", "1", "[]", "{}", "fla", "Torch"])
def test_invalid_delta_rule_backend_fails_before_model_loading(raw):
    with pytest.raises(ValueError, match="model.delta_rule_backend"):
        load_config(RECIPE, [f"model.delta_rule_backend={raw}"])


def test_rendering_is_opt_in_and_requires_heldout_evaluation():
    assert load_config(RECIPE)["evaluation"]["renders"]["enabled"] is False
    with pytest.raises(ValueError, match="require enabled held-out evaluation"):
        load_config(RECIPE, ["evaluation.renders.enabled=true"])
    cfg = load_config(RECIPE, ["evaluation.enabled=true", "evaluation.caches=[/unused/heldout]",
                              "evaluation.renders.enabled=true", "evaluation.renders.points_per_trajectory=48"])
    assert cfg["evaluation"]["renders"]["points_per_trajectory"] == 48


def test_dense_rendering_requires_a_heldout_evaluation():
    with pytest.raises(ValueError, match="requires enabled held-out"):
        load_config(RECIPE, ["evaluation.deploy_dense.enabled=true"])


@pytest.mark.parametrize("override", ["enabled=1", "frame_stride=0", "frame_stride=true",
                                       "max_trajectories_per_task=0", "include_oracle=1",
                                       "max_frames=0", "replay_fps=0", "replay_fps=true"])
def test_invalid_dense_config_fails_before_loading_a_model(override):
    with pytest.raises(ValueError, match="deploy_dense"):
        load_config(RECIPE, ["evaluation.deploy_dense." + override])


@pytest.mark.parametrize("override", [
    "enabled=1", "max_trajectories_per_task=0", "max_trajectories_per_task=true",
    "points_per_trajectory=1", "max_frames=0", "replay_fps=0", "replay_fps=21", "replay_fps=true",
])
def test_invalid_render_configuration_fails_before_model_loading(override):
    with pytest.raises(ValueError, match="evaluation.renders"):
        load_config(RECIPE, ["evaluation.renders." + override])


def test_mapping_override_cannot_hide_unknown_key():
    with pytest.raises(ValueError, match="Unknown configuration key: model.typo"):
        load_config(RECIPE, ["model={typo: 1}"])


@pytest.mark.parametrize("key", ["context_stride", "tail_stride"])
def test_retired_dead_stride_override_is_rejected(key):
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(RECIPE, [f"data.{key}=7"])


def test_model_and_training_locations_expand_env_and_user_without_resolving_hf_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("GRANA_TEST_LOCATIONS", str(tmp_path / "artifacts"))
    cfg = load_config(RECIPE, ["model.name_or_path=${GRANA_TEST_LOCATIONS}/model",
                              "training.output_dir=$GRANA_TEST_LOCATIONS/run",
                              "training.init_checkpoint=${GRANA_TEST_LOCATIONS}/pretrain/checkpoint-000400"])
    assert cfg["model"]["name_or_path"] == str(tmp_path / "artifacts/model")
    assert cfg["training"]["output_dir"] == str(tmp_path / "artifacts/run")
    assert cfg["training"]["init_checkpoint"] == str(tmp_path / "artifacts/pretrain/checkpoint-000400")
    assert load_config(RECIPE)["model"]["name_or_path"] == "Qwen/Qwen3.5-0.8B"
    tilde = load_config(RECIPE, ["model.name_or_path=~/models/qwen", "training.output_dir=~/runs/grana"])
    assert tilde["model"]["name_or_path"] == str(Path.home() / "models/qwen")
    assert tilde["training"]["output_dir"] == str(Path.home() / "runs/grana")
    assert load_config(RECIPE, ["training.output_dir=outputs/relative"])["training"]["output_dir"] == "outputs/relative"


@pytest.mark.parametrize("key", ["model.name_or_path", "training.output_dir", "training.init_checkpoint", "data.manifest"])
def test_unset_path_variables_fail_before_runtime(key, monkeypatch):
    monkeypatch.delenv("GRANA_TEST_UNSET_LOCATION", raising=False)
    with pytest.raises(ValueError, match="Unresolved environment variable"):
        load_config(RECIPE, [f"{key}=${{GRANA_TEST_UNSET_LOCATION}}/missing"])


def test_manifests_resolve_relative_to_their_own_location_and_preserve_selectors(tmp_path, monkeypatch):
    recipes, manifests = tmp_path / "recipes", tmp_path / "manifests"
    recipes.mkdir()
    manifests.mkdir()
    monkeypatch.setenv("GRANA_TEST_PAYLOAD_ROOT", str(tmp_path / "external"))
    train_manifest = {"caches": [{"path": "cache/train", "include_ids": ["a", "b"], "roles": ["A"], "views": ["wrist", "left"], "task_weights": {"A": 2.}}]}
    eval_manifest = {"caches": ["${GRANA_TEST_PAYLOAD_ROOT}/eval"]}
    (manifests / "train.yaml").write_text(yaml.safe_dump(train_manifest))
    (manifests / "eval.yaml").write_text(yaml.safe_dump(eval_manifest))
    recipe = recipes / "train.yaml"
    recipe.write_text(yaml.safe_dump({"data": {"manifest": "../manifests/train.yaml"},
                                     "evaluation": {"enabled": True, "manifest": "../manifests/eval.yaml"}}))
    cfg = load_config(recipe)
    entry = cfg["data"]["caches"][0]
    assert entry == {**train_manifest["caches"][0], "path": str(manifests / "cache/train")}
    assert cfg["evaluation"]["caches"] == [str(tmp_path / "external/eval")]
    assert cfg["data"]["manifest"] == str(manifests / "train.yaml")
    with pytest.raises(ValueError, match="Choose data.manifest or data.caches"):
        load_config(recipe, ["data.caches=[/different/cache]"])


@pytest.mark.parametrize("value", [7, {}, {"roles": ["A"]}])
def test_manifest_invalid_entry_has_explicit_error(tmp_path, value):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"caches": [value]}))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"data": {"manifest": "manifest.yaml"}}))
    with pytest.raises(ValueError, match="Cache manifest entry"):
        load_config(recipe)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_distributed_timeout_requires_positive_integer(value):
    with pytest.raises(ValueError, match="training.distributed_timeout_seconds"):
        load_config(RECIPE, [f"training.distributed_timeout_seconds={str(value).lower()}"])
    assert load_config(RECIPE)["training"]["distributed_timeout_seconds"] == 3600


@pytest.mark.parametrize("location", ["global", "data", "evaluation"])
@pytest.mark.parametrize("views", [[], "top", ["left", "left"], ["left", " left "], [None],
                                  ["<|prog_token|>"], ["<|image_pad|>"], ["left<|"], ["|>right"],
                                  ["x" * 129], [" left"], ["left "], ["\tleft\n"]])
def test_camera_names_are_checked_before_loading_data(tmp_path, location, views):
    config = {"data": {"caches": ["/unused/default"]}}
    if location == "global":
        config["data"]["views"] = views
    else:
        config[location] = {"caches": [{"path": "/unused/cache", "views": views}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="views"):
        load_config(path)


def test_valid_camera_names_are_not_normalized(tmp_path):
    names = ["x" * 128, "left camera"]
    config = {"data": {"caches": [{"path": "/unused/train", "views": names}], "views": names},
              "evaluation": {"caches": [{"path": "/unused/eval", "views": names}]}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_config(path)
    assert loaded["data"]["views"] == names
    for section in ("data", "evaluation"):
        assert loaded[section]["caches"][0]["views"] == names


def test_default_resize_is_explicit_processor_default_and_numeric_budget_is_preserved():
    assert load_config(RECIPE)["data"]["max_pixels"] is None
    assert load_config(RECIPE, ["data.max_pixels=null"])["data"]["max_pixels"] is None
    assert load_config(RECIPE, ["data.max_pixels=1024"])["data"]["max_pixels"] == 1024
    assert load_config(RECIPE, ["data.max_pixels=76800"])["data"]["max_pixels"] == 76800


@pytest.mark.parametrize("raw", ["true", "0", "1023", "76800.0", "'76800'", "[]", "{}"])
def test_invalid_explicit_pixel_budget_fails_configuration(raw):
    with pytest.raises(ValueError, match="max_pixels"):
        load_config(RECIPE, [f"data.max_pixels={raw}"])


def test_model_revision_and_checkpointing_choices_are_explicit():
    assert load_config(RECIPE)["model"]["revision"] is None
    cfg = load_config(RECIPE, ["model.revision=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                              "model.gradient_checkpointing=false"])
    assert cfg["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert cfg["model"]["gradient_checkpointing"] is False


@pytest.mark.parametrize("key,raw", [("revision", "''"), ("revision", "true"), ("revision", "42"),
                                     ("gradient_checkpointing", "1"), ("gradient_checkpointing", "null")])
def test_invalid_model_runtime_choices_fail_before_launch(key, raw):
    with pytest.raises(ValueError, match=f"model.{key}"):
        load_config(RECIPE, [f"model.{key}={raw}"])


def test_full_dense_config_supports_all_trajectories_and_exact_cache_groups(monkeypatch, tmp_path):
    monkeypatch.setenv("GRANA_TEST_DENSE_ROOT", str(tmp_path))
    cfg = load_config(RECIPE, ["evaluation.enabled=true", "evaluation.caches=[/unused/eval]",
                              "evaluation.mode=dense", "evaluation.max_trajectories_per_task=null",
                              "evaluation.deploy_dense.enabled=true",
                              "evaluation.deploy_dense.max_trajectories_per_task=null",
                              "evaluation.deploy_dense.group_by_cache={'${GRANA_TEST_DENSE_ROOT}/eval': L1L2}"])
    assert cfg["evaluation"]["mode"] == "dense"
    assert cfg["evaluation"]["max_trajectories_per_task"] is None
    assert cfg["evaluation"]["deploy_dense"]["max_trajectories_per_task"] is None
    assert cfg["evaluation"]["deploy_dense"]["group_by_cache"] == {str(tmp_path / "eval"): "L1L2"}


@pytest.mark.parametrize("overrides,match", [
    (["evaluation.mode=unknown"], "evaluation.mode"),
    (["evaluation.mode=dense"], "requires enabled held-out"),
    (["evaluation.deploy_dense.group_by_cache=[]"], "must be a mapping"),
    (["evaluation.deploy_dense.group_by_cache={'/cache': ''}"], "group_by_cache"),
    (["evaluation.deploy_dense.group_by_cache={'/cache': 1}"], "group_by_cache"),
])
def test_invalid_dense_evaluation_contracts_fail_before_launch(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_config(RECIPE, overrides)


@pytest.mark.parametrize("recipe", ["pretraining/multitask.yaml"])
def test_public_manifests_load_without_private_data_or_fixed_gpu_count(recipe, monkeypatch, tmp_path):
    monkeypatch.setenv("GRANA_DATA_ROOT", str(tmp_path / "user-caches"))
    cfg = load_config(RECIPE.parent / recipe)
    assert cfg["training"]["num_workers"] == 1
    assert cfg["model"]["gradient_checkpointing"] is True
    assert cfg["logging"]["mode"] == "disabled"
    assert cfg["logging"]["entity"] is None
    assert len(cfg["data"]["caches"]) == len(cfg["evaluation"]["caches"]) == 2
    for section in ("data", "evaluation"):
        assert all(entry["path"].startswith(str(tmp_path / "user-caches")) for entry in cfg[section]["caches"])
    assert not (tmp_path / "user-caches").exists()


@pytest.mark.parametrize("phase,workers,physical,expected_roles", [
    *(("pretrain", workers, 32, {"A": 40, "B": 40, "C": 48, "D": 0}) for workers in (1, 2)),
    *(("posttrain", workers, 8, {"A": 0, "B": 80, "C": 24, "D": 24}) for workers in (1, 2, 4, 8)),
])
def test_run_specs_dry_run_loads_portable_recipe_without_owned_runtime_or_payloads(
        tmp_path, phase, workers, physical, expected_roles):
    import json
    import subprocess
    import sys
    root = RECIPE.parents[1]
    recipe = RECIPE.parent / f"{phase}.yaml"
    result = subprocess.run([sys.executable, str(root / "scripts/ray_migration/run_specs_cli.py"),
                             "--config", str(recipe), "--set", f"training.num_workers={workers}",
                             "--dry-run"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["supervision_budget"]["global_batch_size"] == 128
    assert report["supervision_budget"]["role_rows_per_update_global"] == expected_roles
    assert report["physical_batch_size"] == physical
    assert report["config"]["training"]["gradient_accumulation_steps"] == 128 // (physical * workers)
    assert report["config"]["training"]["num_workers"] == workers
    cache = "train_cache" if phase == "pretrain" else "task_train_cache"
    assert report["config"]["data"]["caches"] == [f"path/to/{cache}"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("accumulation", [1, 3])
def test_pretrain_accepts_complete_single_forward_cycles(accumulation):
    from grana.config import supervision_budget, physical_batch_size
    cfg = load_config(RECIPE, [
        "training.num_workers=4", "training.gradient_accumulation_steps=null",
        f"training.global_batch_size={128 * accumulation}",
    ])
    assert cfg["training"]["gradient_accumulation_steps"] == accumulation
    assert physical_batch_size(cfg) == 32
    budget = supervision_budget(cfg)
    assert budget["role_rows_per_update_global"] == {
        "A": 40 * accumulation, "B": 40 * accumulation,
        "C": 48 * accumulation, "D": 0,
    }
    assert cfg["training"]["prefetch_samples"] is True


def test_paired_d_still_rejects_an_incomplete_forward_cycle():
    with pytest.raises(ValueError, match="must be even"):
        load_config(RECIPE, [
            "roles={A: 0, B: 8, C: 12, D: 12}",
            "training.global_batch_size=null", "training.gradient_accumulation_steps=1",
        ])
