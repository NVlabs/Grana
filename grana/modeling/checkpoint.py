# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grana weights-only artifacts; optimizer, sampler, RNG and step are never loaded."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .inputs import preprocessing_spec

HEADS = ("progress_head", "temporal_head")
FORMAT = "grana-weights-v1"


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_state(model) -> dict[str, torch.Tensor]:
    """All trainable parameters and every complete active auxiliary head."""
    selected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()
            if name in selected or any(name.startswith(head + ".") for head in HEADS)}


def plan_head_load(model, supplied: dict[str, torch.Tensor], allow_fresh_heads=()) -> tuple[dict, dict]:
    """Validate all heads before mutation. Allowance never discards complete heads."""
    allowances = set(allow_fresh_heads)
    if not allowances <= set(HEADS):
        raise ValueError(f"Unknown fresh-head allowance: {sorted(allowances - set(HEADS))}")
    destination = model.state_dict()
    selected, report = {}, {}
    for head in HEADS:
        expected = {key: value for key, value in destination.items() if key.startswith(head + ".")}
        if not expected:
            raise ValueError(f"Model has no active {head}")
        missing = [key for key in expected if key not in supplied]
        mismatched = [key for key, value in expected.items() if key in supplied and value.shape != supplied[key].shape]
        if missing or mismatched:
            if head not in allowances:
                raise ValueError(f"Incomplete or shape-mismatched {head}: missing={missing}, mismatch={mismatched}; explicit whole-head fresh allowance required")
            report[head] = {"outcome": "fresh", "missing": missing, "shape_mismatch": mismatched}
        else:
            selected.update({key: supplied[key] for key in expected})
            report[head] = {"outcome": "loaded", "tensor_count": len(expected)}
    return selected, report


def save_weights(model, path: str | Path, config: dict, step: int) -> dict:
    from safetensors.torch import save_file

    data = config.get("data", {})
    image_preprocessing = preprocessing_spec(
        data.get("max_pixels"), gpu_image_resize=data.get("gpu_image_resize", False))
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    state = trainable_state(model)
    # Resize initialization is random even for frozen rows. Save both readouts.
    added_ids = torch.tensor(list(model.token_ids.values()), device=model.backbone.get_input_embeddings().weight.device)
    state["added_input_embeddings"] = model.backbone.get_input_embeddings().weight.index_select(0, added_ids).detach().cpu().contiguous().clone()
    state["added_output_embeddings"] = model.backbone.get_output_embeddings().weight.index_select(0, added_ids).detach().cpu().contiguous().clone()
    digest = tensor_digest(state)
    save_file(state, str(path / "weights.safetensors"))
    # Loss belongs to the run, not the transferable weights. A new run may
    # fine-tune these tensors with any objective without a migration or opt-in.
    weight_config = {key: value for key, value in config.items() if key != "loss"}
    metadata = {"format": FORMAT, "model": model.model_spec, "weights_sha256": digest,
                "source_step": step, "source_stage": config["training"]["stage"],
                "initialization": "weights_only", "config": weight_config,
                "image_preprocessing": image_preprocessing}
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"path": str(path.resolve()), "weights_sha256": digest, "tensor_count": len(state)}


def load_weights(model, path: str | Path, allow_fresh_heads=()) -> dict:
    from safetensors.torch import load_file

    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("format") != FORMAT:
        raise ValueError("Only Grana weights-v1 checkpoints are supported; legacy checkpoints require a separately verified converter")
    # Tensor shapes alone do not encode LoRA's alpha/rank scaling. Changing
    # alpha while loading the same tensors changes the represented function.
    for field in ("lora_rank", "lora_alpha"):
        if field not in metadata["model"] or metadata["model"][field] != model.model_spec.get(field):
            raise ValueError(f"Checkpoint {field} must match the target adapter architecture")
    supplied = load_file(str(path / "weights.safetensors"))
    if tensor_digest(supplied) != metadata.get("weights_sha256"):
        raise ValueError("Checkpoint tensor digest mismatch")
    if metadata["model"]["token_ids"] != model.token_ids:
        raise ValueError("Checkpoint special-token IDs do not match the target tokenizer")
    selected, report = plan_head_load(model, supplied, allow_fresh_heads)
    # Adapter coverage is strict and independent of pretrain/posttrain objectives.
    destination = model.state_dict()
    adapter_names = [name for name, parameter in model.named_parameters()
                     if parameter.requires_grad and not any(name.startswith(head + ".") for head in HEADS)]
    for name in adapter_names:
        if name not in supplied or supplied[name].shape != destination[name].shape:
            raise ValueError(f"Missing or shape-mismatched adapter tensor: {name}; model sizes cannot initialize one another")
        selected[name] = supplied[name]
    embeddings = (model.backbone.get_input_embeddings(), model.backbone.get_output_embeddings())
    added_ids = torch.tensor(list(model.token_ids.values()), device=embeddings[0].weight.device)
    for key, embedding in zip(("added_input_embeddings", "added_output_embeddings"), embeddings):
        if key not in supplied or supplied[key].shape != (len(added_ids), embedding.weight.shape[1]):
            raise ValueError(f"Missing or shape-mismatched {key}")
    # No tensor mutation occurs until the complete load plan has been checked.
    model.load_state_dict(selected, strict=False)
    with torch.no_grad():
        for key, embedding in zip(("added_input_embeddings", "added_output_embeddings"), embeddings):
            embedding.weight.index_copy_(0, added_ids.to(embedding.weight.device), supplied[key].to(embedding.weight))
    return {"initialization": "weights_only", "source_step": metadata.get("source_step"),
            "source_stage": metadata.get("source_stage"), "heads": report,
            "adapter_tensor_count": len(adapter_names), "weights_sha256": metadata["weights_sha256"]}
