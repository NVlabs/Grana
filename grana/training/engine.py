# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Ray Train worker: dense ABCD batches and fresh weights-only runs."""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from contextlib import ExitStack
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from grana.modeling.checkpoint import load_weights, save_weights, tensor_digest, trainable_state
from grana.modeling.inputs import collate_examples, preprocessing_spec
from grana.modeling.image_preprocessing import configure_image_preprocessing
from .losses import LOSS_COMPONENTS, ROLES, normalized_loss, role_loss_components, window_role_counts
from .profiling import UpdateProfiler
from grana.runtime import require_ray_worker


def detached_stage_choices(output: dict, examples: list) -> list[int] | None:
    """Read each compact C row at its designated query occurrence, before backward."""
    indices, logits = output["stage_indices"], output["stage_logits"]
    expected = {index for index, example in enumerate(examples) if example.role == "C"}
    if (any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices) or len(set(indices)) != len(indices)
            or set(indices) != expected):
        raise ValueError("Stage output indices must identify every C example exactly once")
    if logits is None:
        if indices:
            raise ValueError("C examples require stage logits")
        return None
    if logits.ndim != 3 or logits.shape[:2] != (len(indices), 8):
        raise ValueError("Stage logits must contain eight slots for each C example")
    slots = [getattr(examples[index], "poi_slot", 5) for index in indices]
    if any(isinstance(slot, bool) or not isinstance(slot, Integral) or not 0 <= slot < 8 for slot in slots):
        raise ValueError("Each C example needs an integer poi_slot in 0..7")
    return logits.detach()[torch.arange(len(indices), device=logits.device),
                           torch.tensor(slots, device=logits.device)].argmax(-1).tolist()


def _validate_c_d_pair(source, derived) -> None:
    fields = ("frame_indices", "view_names", "trajectory_id", "task", "num_stages", "stage_prompts",
              "stage_ids", "within_stage_progress", "stride_pair", "observation_digest", "poi_slot",
              "frame_weight_order", "frame_selection_profile", "query_order_profile")
    if (source.role != "C" or derived.role != "D" or derived.images is not source.images
            or any(getattr(derived, field) != getattr(source, field) for field in fields)):
        raise RuntimeError("C-to-D paired observations or sampling metadata changed")


def _sample_signature(model) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().flatten()[:64].float().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def initialization_receipt(model, config: dict, report: dict | None = None) -> dict:
    """Fingerprint every initial tensor plus the complete transferable state.

    Hash before the first update, without sampling RNG or changing weights.
    Added vocabulary rows are retained even when the backbone embeddings are
    frozen; they are part of the checkpoint's represented function.
    """
    started = time.monotonic()
    adapted = trainable_state(model)
    embeddings = (model.backbone.get_input_embeddings(), model.backbone.get_output_embeddings())
    added = {}
    for name, embedding in zip(("added_input_embeddings", "added_output_embeddings"), embeddings, strict=True):
        ids = torch.tensor(list(model.token_ids.values()), device=embedding.weight.device)
        added[name] = embedding.weight.index_select(0, ids).detach().cpu().contiguous().clone()
    full = model.state_dict()
    return {
        "initialization": "weights_only" if report is not None else "pretrained_backbone",
        "report": report, "training_stage": config["training"]["stage"],
        "init_checkpoint": config["training"]["init_checkpoint"], "model": model.model_spec,
        "seed": config["training"]["seed"], "sampler_seed": config["data"]["seed"],
        "all_weights_sha256": tensor_digest(full),
        "trainable_weights_sha256": tensor_digest(adapted),
        "added_embeddings_sha256": tensor_digest(added),
        "transferable_weights_sha256": tensor_digest({**adapted, **added}),
        "all_tensor_count": len(full), "trainable_tensor_count": len(adapted),
        "token_ids": model.token_ids,
        "input_embedding_requires_grad": embeddings[0].weight.requires_grad,
        "output_embedding_requires_grad": embeddings[1].weight.requires_grad,
        "fingerprint_seconds": time.monotonic() - started,
    }


def _sum_across_workers(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _sampling_profile_coverage(joint_counts, poi_counts, *, device, world_size):
    """Aggregate independent draws only; D has no selection/order RNG."""
    from grana.data.frame_sampling import FRAME_SELECTION_PROFILES, QUERY_ORDER_PROFILES

    pairs = [(family, order) for family in FRAME_SELECTION_PROFILES for order in QUERY_ORDER_PROFILES]
    if (any(set(joint_counts[role]) - set(pairs) for role in "ABC")
            or any(set(poi_counts[role]) - set(range(8)) for role in "ABC")):
        raise ValueError("Observed training sampling profile or POI slot is outside its contract")
    joint = torch.tensor([[joint_counts[role].get(pair, 0) for pair in pairs] for role in "ABC"],
                         device=device, dtype=torch.int64)
    slots = torch.tensor([[poi_counts[role].get(slot, 0) for slot in range(8)] for role in "ABC"],
                         device=device, dtype=torch.int64)
    joint = _sum_across_workers(joint, world_size).cpu().tolist()
    slots = _sum_across_workers(slots, world_size).cpu().tolist()
    if any(sum(a) != sum(b) for a, b in zip(joint, slots, strict=True)):
        raise RuntimeError("Sampling profile and POI slot accounting differ")
    return {
        "independent_profile_order_by_role": {
            role: {family: {order: row[pairs.index((family, order))] for order in QUERY_ORDER_PROFILES}
                   for family in FRAME_SELECTION_PROFILES}
            for role, row in zip("ABC", joint, strict=True)},
        "independent_poi_slots_by_role": {
            role: {str(slot): count for slot, count in enumerate(row)}
            for role, row in zip("ABC", slots, strict=True)},
    }


def _vision_runtime_metrics(stats: dict, phase: str) -> dict:
    """Keep optimization hits and whole-encoder spans visible in step history."""
    counts = stats[phase]
    result = {f"vision_sdpa/{key}": counts[key] for key in ("calls", "hits", "images")}
    result.update({f"vision_sdpa/fallback/{reason}": count for reason, count in counts["fallbacks"].items()})
    profile = stats["profile"]
    if profile["enabled"]:
        result.update({"time/profile/vision_host_seconds": profile["host_seconds"],
                       "time/profile/vision_cuda_span_seconds": profile["cuda_seconds"],
                       "vision_sdpa/profile_encoder_calls": profile["vision_calls"],
                       "vision_sdpa/pending_cuda_spans": profile["pending_cuda_spans"]})
    return result


def _evaluation_runtime_metrics(result: dict) -> dict:
    """Execution metrics are separate from logical role-row coverage."""
    values = {key: result[key] for key in ("execution_world_size", "forward_calls", "batch_size", "max_forward_batch_size",
                                          "windows_per_second", "role_rows_per_second") if key in result}
    for group in ("timing_seconds", "cuda_memory"):
        values.update({f"runtime/{group}/{key}": value for key, value in (result.get(group) or {}).items()})
    for worker, memory in (result.get("cuda_memory_by_worker") or {}).items():
        values.update({f"runtime/worker_{worker}/cuda_memory/{key}": value
                       for key, value in (memory or {}).items()})
    if "vision_sdpa" in result:
        values.update(_vision_runtime_metrics(result["vision_sdpa"], "evaluation"))
    return values


def learning_rate_for_step(training: dict, step: int) -> float:
    """Actual update LR: linear warmup, then constant or cosine decay."""
    warmup = training["warmup_steps"]
    if warmup and step <= warmup:
        factor = step / warmup
    elif training["lr_schedule"] == "cosine":
        progress = (step - warmup - 1) / (training["max_steps"] - warmup)
        factor = 0.5 * (1 + math.cos(math.pi * progress))
    else:
        factor = 1.0
    return training["learning_rate"] * factor


def train(config: dict) -> None:
    """Called exclusively by the checkout's Ray launcher, never as a direct CLI."""
    require_ray_worker()
    from ray import train as ray_train
    from .logging import RunLogger

    context = ray_train.get_context()
    with RunLogger(config, rank=context.get_world_rank(), world_size=context.get_world_size()) as logger:
        _train_worker(config, logger)


def _train_worker(config: dict, logger) -> None:
    with ExitStack() as resources:
        _train_worker_impl(config, logger, resources)


def _train_worker_impl(config: dict, logger, resources) -> None:
    require_ray_worker()
    from ray import train as ray_train
    from ray.train.torch import get_device, prepare_model
    from grana.data.cache import CachedTrajectories
    from grana.config import validate_config, supervision_budget
    from .evaluation import build_evaluation_plan, evaluate
    from grana.modeling.model import build_model
    from grana.data.sampling import RoleSampler, stage_answer_count
    from .stage_diagnostics import empty_diagnostics, record_stage_rows, merge_diagnostics

    validate_config(config)
    context = ray_train.get_context()
    rank, world_size = context.get_world_rank(), context.get_world_size()
    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("The release training recipe requires a CUDA worker")
    # NCCL object collectives use the current device, including plan setup
    # before prepare_model() has selected the worker's DDP device.
    torch.cuda.set_device(device)
    training = config["training"]
    if world_size != training["num_workers"]:
        raise RuntimeError("Ray worker world_size does not match training.num_workers")
    seed = training["seed"]
    # Identical initial heads across ranks; data RNG is independent per rank.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "training_metrics.jsonl").exists() or list(output_dir.glob("checkpoint-*")):
            raise ValueError("Use a new output directory for each weights-only run")
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (output_dir / "supervision_budget.json").write_text(json.dumps(supervision_budget(config), indent=2) + "\n")
    if world_size > 1:
        dist.barrier()
    lifecycle = {"rank": rank, "world_size": world_size, "pid": os.getpid(),
                 "ppid": os.getppid(), "pgid": os.getpgid(0), "sid": os.getsid(0),
                 "started_unix": time.time(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "device": str(device), "cgroup": Path("/proc/self/cgroup").read_text(),
                 "proc_stat": Path("/proc/self/stat").read_text()}
    (output_dir / f"worker-{rank:03d}.json").write_text(json.dumps(lifecycle, indent=2) + "\n")
    dataset = CachedTrajectories(config["data"]["caches"], views=config["data"]["views"])
    sampler = RoleSampler(dataset, config["data"], seed=config["data"]["seed"] + rank, worker_rank=rank)
    sampler.validate_role_budget(config["roles"])
    stage_policy = sampler.augmentation_summary()
    if rank == 0:
        (output_dir / "stage_augmentation.json").write_text(json.dumps(stage_policy, indent=2) + "\n")
    sampled_tasks = {role: {} for role in "ABC"}
    sampled_joint = {role: Counter() for role in "ABC"}
    sampled_poi_slots = {role: Counter() for role in "ABC"}

    def sample(request):
        with profiler.section("sample"):
            example = sample_source.sample(request)
        role = example.role
        counts = sampled_tasks[role]
        counts[example.task] = counts.get(example.task, 0) + 1
        sampled_joint[role][example.frame_selection_profile, example.query_order_profile] += 1
        sampled_poi_slots[role][example.poi_slot] += 1
        return example

    evaluation_plan = None
    render_plan = None
    dense_plan = None
    dense_mode = config["evaluation"]["mode"] == "dense"
    if config["evaluation"]["enabled"]:
        planning_error = None
        try:
            evaluation_dataset = CachedTrajectories(config["evaluation"]["caches"], views=config["data"]["views"])
            evaluation_plan = build_evaluation_plan(evaluation_dataset, config["evaluation"], config["data"],
                                                    training_dataset=dataset, roles=config["roles"])
            if rank == 0:
                (output_dir / "evaluation_selection_plan.json").write_text(json.dumps(evaluation_plan.manifest(), indent=2) + "\n")
            if config["evaluation"]["deploy_dense"]["enabled"]:
                from .deploy_dense import build_dense_plan, require_encoder
                if rank == 0:
                    require_encoder()
                dense_settings = config["evaluation"]["deploy_dense"]
                dense_plan = build_dense_plan(evaluation_plan,
                                             max_trajectories_per_task=dense_settings["max_trajectories_per_task"],
                                             frame_stride=dense_settings["frame_stride"],
                                             include_oracle=dense_settings["include_oracle"],
                                             group_by_cache=dense_settings["group_by_cache"])
                if rank == 0:
                    (output_dir / "deploy_dense_plan.json").write_text(json.dumps(dense_plan.manifest(), indent=2) + "\n")
                if dense_mode:
                    evaluation_plan = dense_plan
            if rank == 0:
                (output_dir / "evaluation_plan.json").write_text(json.dumps(evaluation_plan.manifest(), indent=2) + "\n")
            if config["evaluation"]["renders"]["enabled"] and not dense_mode:
                from .render_plan import build_render_plan
                render_settings = config["evaluation"]["renders"]
                render_plan = build_render_plan(
                    evaluation_plan,
                    max_trajectories_per_task=render_settings["max_trajectories_per_task"],
                    points_per_trajectory=render_settings["points_per_trajectory"],
                )
                if rank == 0:
                    (output_dir / "render_plan.json").write_text(json.dumps(render_plan.manifest(), indent=2) + "\n")
        except Exception as error:
            planning_error = f"rank {rank}: {type(error).__name__}: {error}"
        planning_errors = [planning_error]
        if world_size > 1:
            planning_errors = [None] * world_size
            dist.all_gather_object(planning_errors, planning_error)
        failures = [error for error in planning_errors if error is not None]
        if failures:
            raise RuntimeError("Evaluation plan preparation failed: " + "; ".join(failures))
    raw_model, processor = build_model(config)
    gpu_image_resize = config["data"].get("gpu_image_resize", False)
    processor = configure_image_preprocessing(processor, device, resize_on_device=gpu_image_resize)
    image_preprocessing = preprocessing_spec(config["data"]["max_pixels"], gpu_image_resize=gpu_image_resize)
    vision_flags = {"enabled": config["evaluation"].get("batched_vision_sdpa", False),
                    "training_enabled": training.get("batched_vision_sdpa", False)}
    observe_vision = any(vision_flags.values()) or training.get("profile_steps", 0) > 0
    if observe_vision:
        from grana.modeling.vision_sdpa import configure_batched_vision_sdpa, batched_vision_sdpa_stats
        configure_batched_vision_sdpa(raw_model, **vision_flags)

    def evaluate_local(plan):
        if observe_vision:
            batched_vision_sdpa_stats(raw_model, reset=True)
        result = evaluate(raw_model, processor, plan, device=device,
                          max_pixels=config["data"]["max_pixels"],
                          frame_weights=config["loss"]["frame_weights"],
                          role_weights=config["loss"]["role_weights"], loss_config=config["loss"])
        if observe_vision:
            result["vision_sdpa"] = batched_vision_sdpa_stats(raw_model)
        return result

    def run_evaluation(plan, step, purpose):
        if world_size == 1:
            return evaluate_local(plan)
        from .distributed_evaluation import evaluate_distributed
        return evaluate_distributed(
            plan, evaluate_local,
            output_dir=output_dir / "evaluation_shards" / f"step-{step:06d}" / purpose,
            frame_weights=config["loss"]["frame_weights"], role_weights=config["loss"]["role_weights"],
            loss_config=config["loss"])

    initialization = None
    if training["init_checkpoint"]:
        initialization = load_weights(raw_model, training["init_checkpoint"], training["allow_fresh_heads"])
    initialization = initialization_receipt(raw_model, config, initialization)
    fingerprint_keys = ("all_weights_sha256", "trainable_weights_sha256", "added_embeddings_sha256",
                        "transferable_weights_sha256")
    worker_hashes = [{key: initialization[key] for key in fingerprint_keys}]
    if world_size > 1:
        worker_hashes = [None] * world_size
        dist.all_gather_object(worker_hashes, {key: initialization[key] for key in fingerprint_keys})
    if any(hashes != worker_hashes[0] for hashes in worker_hashes):
        raise RuntimeError("Workers did not start from identical model weights")
    initialization.update(rank=rank, world_size=world_size, worker_hashes=worker_hashes,
                          all_workers_initial_weights_equal=True,
                          effective_sampler_seed=config["data"]["seed"] + rank)
    (output_dir / f"initialization-{rank:03d}.json").write_text(json.dumps(initialization, indent=2) + "\n")
    if rank == 0:
        (output_dir / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")
        (output_dir / "data_summary.json").write_text(json.dumps(dataset.summary(), indent=2) + "\n")
    model = prepare_model(raw_model, move_to_device=True, parallel_strategy="ddp", parallel_strategy_kwargs={"find_unused_parameters": True})
    logger.start(model_summary={**raw_model.model_spec,
                               "stage_augmentation": stage_policy,
                               "image_preprocessing": image_preprocessing,
                               "trainable_parameters": sum(p.numel() for p in raw_model.parameters() if p.requires_grad),
                               "total_parameters": sum(p.numel() for p in raw_model.parameters())},
                 data_summary=dataset.summary(), initialization=initialization)
    # Give dropout independent trajectories after DDP synchronized model weights.
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=training["learning_rate"], weight_decay=training["weight_decay"])
    initial_signature = _sample_signature(raw_model)
    counts = window_role_counts(config["roles"], training["gradient_accumulation_steps"])
    budget = supervision_budget(config)
    if {role: count * world_size for role, count in counts.items()} != budget["role_rows_per_update_global"]:
        raise RuntimeError("Resolved supervision budget does not match runtime worker count")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    runtime = {"rank": rank, "gpu_name": torch.cuda.get_device_name(device),
               "gpu_total_gib": total_bytes / 2**30, "gpu_free_before_training_gib": free_bytes / 2**30,
               "gpu_allocated_before_training_gib": torch.cuda.memory_allocated(device) / 2**30,
               "gpu_reserved_before_training_gib": torch.cuda.memory_reserved(device) / 2**30,
               "gradient_checkpointing": config["model"]["gradient_checkpointing"],
               "image_preprocessing": image_preprocessing,
               "gpu_image_resize": gpu_image_resize,
               "evaluation_batch_size": config["evaluation"].get("batch_size", 1),
               "evaluation_batched_vision_sdpa": vision_flags["enabled"],
               "training_batched_vision_sdpa": vision_flags["training_enabled"],
               "prefetch_samples": training.get("prefetch_samples", False),
               "stage_augmentation": stage_policy,
               "supervision_budget": budget}
    (output_dir / f"runtime-{rank:03d}.json").write_text(json.dumps(runtime, indent=2) + "\n")
    local_denominators = torch.tensor([counts[role] * sum(config["loss"]["frame_weights"]) for role in ROLES], device=device, dtype=torch.float64)
    global_masses = _sum_across_workers(local_denominators, world_size)
    denominators = dict(zip(ROLES, global_masses.tolist()))
    started = time.monotonic()
    checkpoint = None
    latest_metrics = {}
    evaluation_events = []
    training_compute_seconds = 0.0
    update_durations = []
    peak_allocated_gib = peak_reserved_gib = 0.0
    model.train()
    component_names = tuple(name for name in LOSS_COMPONENTS if name != "total")

    def run_phase(examples: list, requests: tuple) -> tuple[dict, torch.Tensor, torch.Tensor]:
        phase_counts = {role: sum(example.role == role for example in examples) for role in ROLES}
        for role in ROLES:
            actual_counts[role] += phase_counts[role]
        phase_layouts.append(phase_counts)
        with profiler.section("collate_and_transfer"):
            inputs = collate_examples(processor, examples, config["data"]["max_pixels"], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with profiler.section("forward", cuda=True):
                result = model(inputs, [example.role for example in examples], [stage_answer_count(example) for example in examples])
            with profiler.section("objective_and_route", cuda=True):
                components = role_loss_components(result, examples, config["loss"]["frame_weights"],
                                                  loss_config=config["loss"])
                numerators = {role: components[role]["total"] for role in ROLES}
                loss = normalized_loss(numerators, denominators, config["loss"]["role_weights"], world_size)
        with profiler.section("objective_and_route", cuda=True):
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            # Hard C choice is detached BEFORE the C graph is released by backward.
            choices = detached_stage_choices(result, examples)
            if diagnostics_enabled:
                record_stage_rows(stage_diagnostics, examples, requests, result["stage_indices"], choices, rank=rank)
        require_ray_worker()
        with profiler.section("backward", cuda=True):
            loss.backward()
        component_values = torch.stack([torch.stack([components[role][key].detach() for key in component_names])
                                        for role in ROLES])
        return ({"choices": choices, "stage_indices": result["stage_indices"]},
                torch.stack([numerators[role].detach() for role in ROLES]), component_values)

    from .prefetch import SamplePrefetcher, forward_role_requests
    phase_requests = forward_role_requests(config["roles"])
    policy = config["data"]["train_stage_augmentation"]
    diagnostics_enabled = (policy["enabled"] or config["roles"]["C"] > config["roles"]["D"] > 0)
    sample_source = resources.enter_context(SamplePrefetcher(
        sampler, config["roles"], training["gradient_accumulation_steps"], training["max_steps"],
        enabled=training.get("prefetch_samples", False)))
    for step in range(1, training["max_steps"] + 1):
        torch.cuda.synchronize(device)
        update_started = time.monotonic()
        free_before, total_bytes = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        profiler = UpdateProfiler(enabled=step <= training.get("profile_steps", 0), device=device)
        if observe_vision:
            configure_batched_vision_sdpa(raw_model, **vision_flags, profile=profiler.enabled)
            batched_vision_sdpa_stats(raw_model, reset=True)
        learning_rate = learning_rate_for_step(training, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        window_numerators = torch.zeros(4, device=device, dtype=torch.float64)
        window_components = torch.zeros((4, len(component_names)), device=device, dtype=torch.float64)
        role_counts = config["roles"]
        actual_counts = dict.fromkeys(ROLES, 0)
        phase_layouts = []
        derived_pairs = 0
        independent_c_rows = 0
        stage_diagnostics = empty_diagnostics()
        if role_counts["D"]:
            for _ in range(training["gradient_accumulation_steps"] // 2):
                # Only explicitly paired C rows produce D. Independent C can
                # use augmented tasks in either forward of the same cycle.
                first = [sample(request) for request in phase_requests[0]]
                paired_indices = [i for i, request in enumerate(phase_requests[0]) if request == "C_paired"]
                c_rows = [first[i] for i in paired_indices]
                independent_c_rows += sum(phase.count("C_independent") for phase in phase_requests)
                result, numerators, components = run_phase(first, phase_requests[0])
                window_numerators += numerators
                window_components += components
                with profiler.section("derive_d"):
                    choices_by_row = dict(zip(result["stage_indices"], result["choices"], strict=True))
                    d_rows = [sampler.make_d(first[index], choices_by_row[index]) for index in paired_indices]
                    for source, derived in zip(c_rows, d_rows, strict=True):
                        _validate_c_d_pair(source, derived)
                    derived_pairs += len(d_rows)
                derived = iter(d_rows)
                second = [next(derived) if request == "D" else sample(request) for request in phase_requests[1]]
                _, numerators, components = run_phase(second, phase_requests[1])
                window_numerators += numerators
                window_components += components
        else:
            for _ in range(training["gradient_accumulation_steps"]):
                examples = [sample(role) for role in ("A", "B", "C") for _ in range(role_counts[role])]
                _, numerators, components = run_phase(examples, phase_requests[0])
                independent_c_rows += role_counts["C"]
                window_numerators += numerators
                window_components += components
        if actual_counts != counts:
            raise RuntimeError(f"Actual supervised role rows differ from the update budget: {actual_counts} != {counts}")
        if derived_pairs != actual_counts["D"] or actual_counts["C"] != derived_pairs + independent_c_rows:
            raise RuntimeError("Every paired C must produce one D; independent C must not produce D")
        observed = torch.tensor([actual_counts[role] for role in ROLES] + [derived_pairs],
                                device=device, dtype=torch.int64)
        global_observed = _sum_across_workers(observed, world_size).tolist()
        global_counts = dict(zip(ROLES, global_observed[:4], strict=True))
        global_derived_pairs = global_observed[4]
        if global_counts != budget["role_rows_per_update_global"]:
            raise RuntimeError("Observed global supervised rows differ from the resolved budget")
        if step == 1:
            accounting = {"optimizer_steps": step, "world_size": world_size,
                          "rank": rank, "actual_rows_per_update_local": actual_counts,
                          "expected_rows_per_update_local": counts,
                          "actual_rows_per_update_global": global_counts,
                          "forward_role_rows_per_worker": phase_layouts,
                          "derived_C_D_pairs_per_update_global": global_derived_pairs,
                          "unpaired_C_rows_per_update_local": independent_c_rows,
                          "D_contract": "One D per paired C, none per independent C; same observations and metadata; detached argmax at C poi_slot."}
            (output_dir / f"batch-accounting-{rank:03d}.json").write_text(json.dumps(accounting, indent=2) + "\n")
            if rank == 0:
                (output_dir / "batch_accounting.json").write_text(json.dumps(accounting, indent=2) + "\n")
        with profiler.section("optimizer", cuda=True):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"], error_if_nonfinite=True)
            require_ray_worker()
            optimizer.step()
        with profiler.section("metrics_reduce", cuda=True):
            global_numerators = _sum_across_workers(window_numerators, world_size)
            global_components = _sum_across_workers(window_components, world_size)
        torch.cuda.synchronize(device)
        update_seconds = time.monotonic() - update_started
        training_compute_seconds += update_seconds
        update_durations.append(update_seconds)
        free_after, _ = torch.cuda.mem_get_info(device)
        peak_allocated_gib = max(peak_allocated_gib, torch.cuda.max_memory_allocated(device) / 2**30)
        peak_reserved_gib = max(peak_reserved_gib, torch.cuda.max_memory_reserved(device) / 2**30)
        means = {role: float(global_numerators[i] / global_masses[i]) if global_masses[i] else 0.0 for i, role in enumerate(ROLES)}
        latest_metrics = {"optimizer_steps": step, "loss": sum(means[role] * config["loss"]["role_weights"][role] for role in ROLES),
                          "gradient_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - started,
                          "learning_rate": learning_rate, "update_seconds": update_seconds,
                          "time/training_step": update_seconds,
                          "time/training_seconds": training_compute_seconds,
                          "time/elapsed_seconds": time.monotonic() - started,
                          "examples_per_second_global": sum(counts.values()) * world_size / update_seconds,
                          "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                          "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                          "gpu_total_gib": total_bytes / 2**30,
                          "gpu_free_before_update_gib": free_before / 2**30,
                          "gpu_free_after_update_gib": free_after / 2**30,
                          "forwards_per_update_per_worker": len(phase_layouts),
                          "derived_C_D_pairs_per_update_global": global_derived_pairs,
                          **{f"role_{role}_loss": means[role] for role in ROLES},
                          **{f"role_{role}_rows_per_update_global": global_counts[role] for role in ROLES},
                          **{f"role_{role}_rows_cumulative_global": global_counts[role] * step for role in ROLES}}
        # Components already include cdf/ceiling coefficients, before the outer
        # role weight. Their sum is role_X_loss, with the same frame/row mass.
        latest_metrics.update({f"role_{role}_{key}_loss": float(global_components[i, j] / global_masses[i])
                               if global_masses[i] else 0.0
                               for i, role in enumerate(ROLES) for j, key in enumerate(component_names)})
        latest_metrics.update(profiler.metrics())
        if profiler.enabled and sample_source.enabled:
            # This CPU duration overlaps GPU work; do not add it to update time.
            latest_metrics["time/profile/sample_prepare_host_seconds"] = sample_source.prepare_seconds
        if observe_vision:
            latest_metrics.update(_vision_runtime_metrics(batched_vision_sdpa_stats(raw_model), "training"))
        if diagnostics_enabled:
            parts = [stage_diagnostics]
            if world_size > 1:
                parts = [None] * world_size
                dist.all_gather_object(parts, stage_diagnostics)
            combined_diagnostics = merge_diagnostics(parts)
            if (combined_diagnostics["c_rows"] != global_counts["C"] or
                    combined_diagnostics["paired_c_rows"] != global_counts["D"] or
                    combined_diagnostics["d_rows"] != global_counts["D"]):
                raise RuntimeError("Stage diagnostics disagree with consumed role rows")
            if rank == 0:
                logger.log_stage_augmentation(combined_diagnostics, step=step)
        if step % training["save_steps"] == 0 or step == training["max_steps"]:
            checkpoint_started = time.monotonic()
            path = output_dir / f"checkpoint-{step:06d}"
            if rank == 0:
                checkpoint = save_weights(raw_model, path, config, step)
                logger.log_checkpoint(path, step=step, final=step == training["max_steps"])
            if world_size > 1:
                dist.barrier()
            latest_metrics["checkpoint_path"] = str(path)
            latest_metrics["time/checkpoint_and_logging_seconds"] = time.monotonic() - checkpoint_started
        if rank == 0:
            with (output_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(latest_metrics) + "\n")
            print(json.dumps(latest_metrics), flush=True)
            logger.log_step(latest_metrics, step=step)
        if config["evaluation"]["enabled"] and (step % config["evaluation"]["every_steps"] == 0 or step == training["max_steps"]):
            # Every rank scores disjoint batches with its unwrapped model;
            # uneven shard lengths must never enter DDP forward collectives.
            if world_size > 1:
                dist.barrier()
            evaluation_started = time.monotonic()
            training_elapsed_seconds = evaluation_started - started
            result = run_evaluation(evaluation_plan, step, "primary")
            render_result = run_evaluation(render_plan, step, "renders") if render_plan is not None else None
            dense_result = result if dense_mode else (
                run_evaluation(dense_plan, step, "dense") if dense_plan is not None else None)
            event = None
            persistence_error = None
            if rank == 0:
                try:
                    result["optimizer_steps"] = step
                    result["training_elapsed_seconds"] = training_elapsed_seconds
                    result["training_compute_seconds"] = training_compute_seconds
                    result["provenance"] = {"mode": "during_training", "optimizer_steps": step}
                    (output_dir / f"evaluation-{step:06d}.json").write_text(json.dumps(result, indent=2) + "\n")
                    event = {"optimizer_steps": step, "plan_sha256": result["plan_sha256"],
                             "elapsed_seconds": result["elapsed_seconds"], "model_calls": result["model_calls"],
                             "training_elapsed_seconds": training_elapsed_seconds,
                             "training_compute_seconds": training_compute_seconds,
                             "task_count": result["task_count"], **result["metrics"],
                             **_evaluation_runtime_metrics(result)}
                    if render_plan is not None:
                        from .rendering import render_evaluation
                        render_settings = config["evaluation"]["renders"]
                        render_result["optimizer_steps"] = step
                        render_result["training_elapsed_seconds"] = training_elapsed_seconds
                        (output_dir / f"render-evaluation-{step:06d}.json").write_text(json.dumps(render_result, indent=2) + "\n")
                        trajectory_count = len({(str(window.trajectory.cache_root), window.trajectory.trajectory_id)
                                                for window in render_plan.windows})
                        bundle = render_evaluation(
                            render_result, render_plan, output_dir=output_dir / "eval_renders", step=step,
                            training_elapsed_seconds=training_elapsed_seconds,
                            max_trajectories=trajectory_count, max_frames=render_settings["max_frames"],
                            replay_fps=render_settings["replay_fps"],
                        )
                        logger.log_evaluation_renders(bundle, step=step)
                        event.update(rendered_trajectories=bundle["rendered_trajectories"],
                                     render_model_calls=render_result["model_calls"],
                                     render_evaluation_seconds=render_result["elapsed_seconds"],
                                     rendering_seconds=bundle["rendering_seconds"])
                        logger.log_evaluation(render_result["metrics"], step=step,
                                              diagnostics=render_result.get("diagnostics"),
                                              evaluation=render_result, purpose="renders")
                    if dense_plan is not None:
                        from .deploy_dense import render_dense_evaluation
                        dense_result.update(optimizer_steps=step, training_elapsed_seconds=training_elapsed_seconds)
                        (output_dir / f"deploy-dense-evaluation-{step:06d}.json").write_text(json.dumps(dense_result) + "\n")
                        dense_bundle = render_dense_evaluation(
                            dense_result, dense_plan, output_dir=output_dir / "deploy_dense", step=step,
                            training_elapsed_seconds=training_elapsed_seconds,
                            max_frames=dense_settings["max_frames"], replay_fps=dense_settings["replay_fps"],
                            provenance={"mode": "during_training"},
                        )
                        logger.log_deploy_dense(dense_bundle, step=step)
                        if not dense_mode:
                            logger.log_evaluation(
                                {**dense_result["metrics"], **_evaluation_runtime_metrics(dense_result),
                                 "elapsed_seconds": dense_result["elapsed_seconds"]}, step=step,
                                diagnostics=dense_result.get("diagnostics"),
                                evaluation=dense_result, purpose="dense")
                        event.update(deploy_dense_model_calls=dense_result["model_calls"],
                                     deploy_dense_reused_primary_predictions=dense_mode,
                                     deploy_dense_evaluation_seconds=dense_result["elapsed_seconds"],
                                     deploy_dense_rendering_seconds=dense_bundle["rendering_seconds"],
                                     deploy_dense_media_pairs=len(dense_bundle["items"]))
                    event["evaluation_and_persistence_seconds"] = time.monotonic() - evaluation_started
                    # The clock covers scoring, rendering, local result writes
                    # and media preparation. The final artifact enqueue and
                    # unified history commit below are outside this timer.
                    event["evaluation_timing_semantics"] = (
                        "Wall time through scoring, rendering, local result persistence and media preparation; "
                        "excludes final evaluation artifact enqueue, unified history commit and asynchronous cloud completion.")
                    logger.log_evaluation({key: value for key, value in event.items() if key != "optimizer_steps"},
                                          step=step, diagnostics=result.get("diagnostics"), evaluation=result)
                    logger.commit_step(step=step)
                    with (output_dir / "evaluation_metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(event) + "\n")
                    print(json.dumps({"evaluation_completed": event}), flush=True)
                except Exception as error:
                    persistence_error = f"{type(error).__name__}: {error}"
            if world_size > 1:
                shared = [event, persistence_error]
                dist.broadcast_object_list(shared, src=0, device=device)
                event, persistence_error = shared
            if persistence_error is not None:
                raise RuntimeError("Evaluation persistence failed: " + persistence_error)
            evaluation_events.append(event)
            latest_metrics.update({f"eval/{key}": value for key, value in event.items() if key != "optimizer_steps"})
        elif rank == 0:
            logger.commit_step(step=step)
        ray_train.report(latest_metrics)

    sample_source.close()
    # Reuse the final real examples to verify both native C and progress outputs.
    verification_examples = [c_rows[0], d_rows[0]] if role_counts["D"] else examples[:1]
    verification_inputs = collate_examples(processor, verification_examples, config["data"]["max_pixels"], device)
    raw_model.eval()

    def verification_predictions() -> dict[str, torch.Tensor]:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = raw_model(verification_inputs, [example.role for example in verification_examples],
                                   [stage_answer_count(example) for example in verification_examples])
        return {key: value.detach().cpu().clone() for key, value in prediction.items()
                if key.endswith("logits") and value is not None}

    before_predictions = verification_predictions()
    # Check a real restore after deliberate parameter perturbation, without
    # restoring any optimizer/RNG/sampler state or claiming model-quality eval.
    final_signature = _sample_signature(raw_model)
    delta = sum(float((final_signature[name] - before).abs().sum()) for name, before in initial_signature.items())
    before_reload = tensor_digest(trainable_state(raw_model))
    with torch.no_grad():
        next(parameter for parameter in raw_model.parameters() if parameter.requires_grad).flatten()[0].add_(1)
    reload_report = load_weights(raw_model, output_dir / f"checkpoint-{training['max_steps']:06d}")
    restored = tensor_digest(trainable_state(raw_model)) == before_reload
    after_predictions = verification_predictions()
    prediction_delta = 0.0
    for name, expected in before_predictions.items():
        actual = after_predictions[name]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        finite = torch.isfinite(expected)
        prediction_delta = max(prediction_delta, float((actual[finite] - expected[finite]).abs().max()))
    if not restored or delta <= 0:
        raise RuntimeError(f"Training verification failed: reload_equal={restored}, parameter_delta={delta}")
    task_names = sorted({task for lane in dataset.by_role.values() for task in lane})
    sampled_counts = torch.tensor([[sampled_tasks[role].get(task, 0) for role in "ABC"] for task in task_names],
                                  device=device, dtype=torch.float64)
    sampled_counts = _sum_across_workers(sampled_counts, world_size).cpu().to(torch.int64)
    per_task_counts = {task: dict(zip("ABC", row)) for task, row in zip(task_names, sampled_counts.tolist(), strict=True)}
    totals = sampled_counts.sum(dim=1)
    sampling_coverage = {
        "available_tasks": len(task_names), "sampled_tasks": int((totals > 0).sum()),
        "minimum_independent_draws_per_task": int(totals.min()),
        "independent_draws_by_role": dict(zip("ABC", sampled_counts.sum(dim=0).tolist())),
        "per_task": per_task_counts,
        "semantics": "Actual independent training A/B/C draws across all workers; D derives from C without resampling.",
        "derived_D_rows": counts["D"] * world_size * len(update_durations),
        **_sampling_profile_coverage(sampled_joint, sampled_poi_slots, device=device, world_size=world_size),
    }
    if rank == 0:
        (output_dir / "sampling_coverage.json").write_text(json.dumps(sampling_coverage, indent=2) + "\n")
    warmup_excluded = min(2, max(0, len(update_durations) - 1))
    measured_durations = update_durations[warmup_excluded:]
    runtime_summary = {
        **runtime, "optimizer_updates": len(update_durations),
        "training_compute_seconds": training_compute_seconds,
        "all_update_mean_seconds": float(np.mean(update_durations)),
        "warmup_updates_excluded": warmup_excluded,
        "warm_update_mean_seconds": float(np.mean(measured_durations)),
        "warm_update_p50_seconds": float(np.quantile(measured_durations, .5)),
        "warm_update_p90_seconds": float(np.quantile(measured_durations, .9)),
        "gpu_peak_allocated_gib": peak_allocated_gib, "gpu_peak_reserved_gib": peak_reserved_gib,
        "actual_role_rows_over_run_global": {role: count * world_size * len(update_durations)
                                             for role, count in counts.items()},
    }
    (output_dir / f"runtime-summary-{rank:03d}.json").write_text(json.dumps(runtime_summary, indent=2) + "\n")
    # A completed W&B run requires every worker's restore/parity check. If a
    # peer fails, Ray tears down the collective instead of logging success.
    if world_size > 1:
        dist.barrier()
    final_metrics = {**latest_metrics, "status": "completed", "checkpoint_reload_equal": restored,
                     "evaluation_events": evaluation_events, "total_elapsed_seconds": time.monotonic() - started,
                     "training_compute_seconds": training_compute_seconds,
                     "runtime_summary": runtime_summary,
                     "sampling_coverage": sampling_coverage,
                     "sampled_parameter_delta_l1": delta, "checkpoint_load": reload_report,
                     "checkpoint_prediction_parity": True, "checkpoint_prediction_max_abs_diff": prediction_delta,
                     "verification_examples": [{"role": example.role, "trajectory_id": example.trajectory_id,
                                                "frame_indices": example.frame_indices, "poi_slot": example.poi_slot,
                                                "frame_weight_order": example.frame_weight_order,
                                                "frame_selection_profile": example.frame_selection_profile,
                                                "query_order_profile": example.query_order_profile}
                                               for example in verification_examples]}
    if rank == 0:
        (output_dir / "result.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    logger.finish(summary=final_metrics)
    ray_train.report({**latest_metrics, "status": "completed", "checkpoint_reload_equal": restored,
                      "sampled_parameter_delta_l1": delta, "checkpoint_prediction_parity": True,
                      "checkpoint_prediction_max_abs_diff": prediction_delta})
