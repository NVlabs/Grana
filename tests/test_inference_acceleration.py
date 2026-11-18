# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the inference-only latency stack (grana.inference.acceleration).

``torch.compile`` is replaced by identity where a compiled path is exercised,
so these tests check wiring, exactness and fallbacks without CUDA or weights.
"""
from contextvars import ContextVar
import copy
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from grana.inference import acceleration
from grana.inference import predictor as inference
from grana.modeling.backends import configure_delta_rule_backend
from grana.modeling.vision_sdpa import configure_batched_vision_sdpa

IMAGE, VIDEO, START, END = 60, 61, 62, 63


def tiny_config():
    config = Qwen3_5Config(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "layer_types": ["linear_attention", "full_attention"],
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2, "linear_key_head_dim": 8,
                     "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4},
        vision_config={"hidden_size": 32, "num_heads": 4, "intermediate_size": 64,
                       "depth": 2, "out_hidden_size": 32, "patch_size": 16,
                       "spatial_merge_size": 2, "num_position_embeddings": 16},
        image_token_id=IMAGE, video_token_id=VIDEO, vision_start_token_id=START, vision_end_token_id=END)
    for item in (config, config.text_config, config.vision_config):
        item._attn_implementation = "sdpa"
    return config


def tiny_model(dtype=torch.float32):
    torch.manual_seed(0)
    # HF selects the norm at construction, independently of the delta-rule
    # backend. Keep this CPU fixture portable when FLA and CUDA are available.
    with patch.object(qwen, "FusedRMSNormGated", None):
        model = qwen.Qwen3_5ForConditionalGeneration(tiny_config()).to(dtype).eval()
    # CPU model: keep CUDA-only kernels (causal-conv1d, FLA) out even when their packages import.
    acceleration.configure_causal_conv1d(model, None, "cpu")
    configure_delta_rule_backend(model, "torch")
    return model


def image_tokens(count):
    return [START, *[IMAGE] * count, END]


def prompt(grids, *, text=(1, 2), batch=1):
    """Token ids, mm types, grids and pixels for ``grids`` (merge size 2) images."""
    ids = list(text)
    for frames, height, width in grids:
        ids += image_tokens(frames * height * width // 4) + [3]
    input_ids = torch.tensor([ids] * batch)
    types = (input_ids == IMAGE).long()
    grid = torch.tensor(list(grids) * batch, dtype=torch.long).reshape(-1, 3)
    patches = sum(frames * height * width for frames, height, width in grids) * batch
    torch.manual_seed(1)
    return input_ids, types, grid, torch.randn(patches, 3 * 2 * 16 * 16)


@pytest.fixture
def identity_compile(monkeypatch):
    calls = []

    def compile_(function, **kwargs):
        calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", compile_)
    return calls


@pytest.fixture
def dynamo_limit():
    import torch._dynamo
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    previous = getattr(config, name)
    yield config, name
    setattr(config, name, previous)


@pytest.fixture(autouse=True)
def restore_compile_failure_policy():
    import torch._dynamo
    config = torch._dynamo.config
    with config.patch(suppress_errors=config.suppress_errors,
                      fail_on_recompile_limit_hit=config.fail_on_recompile_limit_hit):
        yield


# --------------------------------------------------------------------------- host position IDs

ROPE_CASES = {
    "text_only": dict(grids=[], left_pad=0),
    "one_square_image": dict(grids=[(1, 4, 4)], left_pad=0),
    "two_images": dict(grids=[(1, 4, 4), (1, 4, 4)], left_pad=0),
    "non_square": dict(grids=[(1, 4, 8)], left_pad=0),
    "tall_non_square": dict(grids=[(1, 8, 4), (1, 4, 4)], left_pad=0),
    "multi_frame": dict(grids=[(2, 4, 4)], left_pad=0),
    "left_padding": dict(grids=[(1, 4, 4), (1, 4, 8)], left_pad=3),
}


@pytest.mark.parametrize("case", ROPE_CASES, ids=list(ROPE_CASES))
def test_host_rope_index_equals_hf_get_rope_index(case):
    spec = ROPE_CASES[case]
    model = tiny_model().model
    ids, types, grid, _ = prompt(spec["grids"], batch=2)
    mask = torch.ones_like(ids)
    if spec["left_pad"]:
        pad = spec["left_pad"]
        # Row 0 is left padded; row 1 keeps a full prompt.
        ids = torch.cat((torch.zeros(2, pad, dtype=ids.dtype), ids), dim=1)
        types = torch.cat((torch.zeros(2, pad, dtype=types.dtype), types), dim=1)
        mask = torch.cat((torch.zeros(2, pad, dtype=mask.dtype), mask), dim=1)
        mask[1, :pad] = 1
        ids[1, :pad] = 5
    grid_or_none = grid if len(grid) else None
    reference, deltas = model.get_rope_index(ids, types, image_grid_thw=grid_or_none, attention_mask=mask)
    computed = acceleration.host_rope_index(ids, types, grid.reshape(-1, 3), mask, 2)
    assert computed is not None
    positions, rope_deltas = computed
    assert positions.dtype == reference.dtype and positions.shape == reference.shape
    assert torch.equal(positions, reference)
    assert torch.equal(rope_deltas, deltas.to(rope_deltas.dtype)) and rope_deltas.shape == deltas.shape


def test_host_rope_index_without_attention_mask_equals_hf():
    model = tiny_model().model
    ids, types, grid, _ = prompt([(1, 4, 4), (1, 8, 4)])
    reference, deltas = model.get_rope_index(ids, types, image_grid_thw=grid, attention_mask=None)
    positions, rope_deltas = acceleration.host_rope_index(ids, types, grid, None, 2)
    assert torch.equal(positions, reference) and torch.equal(rope_deltas, deltas.to(rope_deltas.dtype))


def test_host_rope_index_keeps_input_id_dtype():
    ids, types, grid, _ = prompt([(1, 4, 4)])
    positions, deltas = acceleration.host_rope_index(ids.int(), types, grid, None, 2)
    assert positions.dtype == torch.int32 and deltas.dtype == torch.int64


def unsupported_cases():
    ids, types, grid, _ = prompt([(1, 4, 4)])
    video = types.clone()
    video[0, 3] = 2
    yield "video_token", (ids, video, grid, None, 2)
    yield "unknown_token_type", (ids, torch.where(types == 1, 7, types), grid, None, 2)
    yield "run_length_mismatch", (ids, types, torch.tensor([[1, 4, 8]]), None, 2)
    yield "missing_grid", (ids, types, torch.zeros(0, 3, dtype=torch.long), None, 2)
    yield "one_dimensional_ids", (ids[0], types[0], grid, None, 2)
    yield "type_shape_mismatch", (ids, types[:, :-1], grid, None, 2)
    yield "bad_grid_shape", (ids, types, grid[:, :2], None, 2)
    yield "mask_shape_mismatch", (ids, types, grid, torch.ones(1, 3, dtype=torch.long), 2)
    yield "bool_merge_size", (ids, types, grid, None, True)
    yield "zero_merge_size", (ids, types, grid, None, 0)
    yield "fully_masked_row", (ids, types, grid, torch.zeros_like(ids), 2)


@pytest.mark.parametrize("arguments", [case for _, case in unsupported_cases()],
                         ids=[name for name, _ in unsupported_cases()])
def test_host_rope_index_returns_none_for_unsupported_inputs(arguments):
    assert acceleration.host_rope_index(*arguments) is None


def test_host_position_ids_are_exact_skip_hf_after_verification_and_patch_only_the_instance():
    reference_model = tiny_model()
    model = copy.deepcopy(reference_model)
    state = acceleration.configure_host_position_ids(model, verify_calls=2)
    module = model.model
    assert "forward" in module.__dict__ and "forward" not in reference_model.model.__dict__
    assert qwen.Qwen3_5Model.forward is type(reference_model.model).forward
    calls = []
    original = module.get_rope_index

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    module.get_rope_index = spy
    ids, types, grid, pixels = prompt([(1, 4, 4), (1, 4, 8)])
    inputs = dict(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid, pixel_values=pixels,
                  attention_mask=torch.ones_like(ids))
    with torch.inference_mode():
        expected = reference_model(**inputs).logits
        for call in range(3):
            assert torch.equal(model(**inputs).logits, expected)
    # Two verification calls compare against HF; the third skips HF entirely.
    assert len(calls) == 2
    assert state == {"calls": 3, "fallbacks": 0, "verified": 2, "mismatches": 0, "enabled": True}


def test_host_position_ids_fall_back_without_mm_types_or_with_explicit_positions():
    reference_model = tiny_model()
    model = copy.deepcopy(reference_model)
    state = acceleration.configure_host_position_ids(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        assert torch.equal(model(input_ids=ids).logits, reference_model(input_ids=ids).logits)
        positions = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)
        assert torch.equal(model(input_ids=ids, position_ids=positions).logits,
                           reference_model(input_ids=ids, position_ids=positions).logits)
    assert state["calls"] == 0 and state["fallbacks"] == 2


def test_host_position_id_mismatch_warns_and_permanently_uses_hf(monkeypatch):
    reference_model = tiny_model()
    model = copy.deepcopy(reference_model)
    state = acceleration.configure_host_position_ids(model)
    real = acceleration.host_rope_index
    host_calls = []

    def wrong(*args):
        host_calls.append(1)
        positions, deltas = real(*args)
        return positions + 1, deltas

    monkeypatch.setattr(acceleration, "host_rope_index", wrong)
    ids, types, grid, pixels = prompt([(1, 4, 4)])
    inputs = dict(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid, pixel_values=pixels)
    with torch.inference_mode():
        expected = reference_model(**inputs).logits
        with pytest.warns(RuntimeWarning, match="differed"):
            assert torch.equal(model(**inputs).logits, expected)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert torch.equal(model(**inputs).logits, expected)
    assert len(host_calls) == 1
    assert state["enabled"] is False and state["mismatches"] == 1 and state["calls"] == 0
    assert state["fallbacks"] == 2


def test_host_position_ids_require_a_qwen_backbone():
    with pytest.raises(ValueError):
        acceleration.configure_host_position_ids(nn.Linear(2, 2))


def test_strict_position_ids_reject_unsupported_inputs_and_mismatches(monkeypatch):
    model = tiny_model()
    state = acceleration.configure_host_position_ids(model, strict=True)
    with torch.inference_mode(), pytest.raises(ValueError, match="requires supported"):
        model(input_ids=torch.tensor([[1, 2, 3]]))
    real = acceleration.host_rope_index
    monkeypatch.setattr(acceleration, "host_rope_index",
                        lambda *args: (real(*args)[0] + 1, real(*args)[1]))
    ids, types, grid, pixels = prompt([(1, 4, 4)])
    with torch.inference_mode(), pytest.raises(RuntimeError, match="verification failed"):
        model(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid, pixel_values=pixels)
    assert state["fallbacks"] == state["calls"] == 0
    assert state["mismatches"] == 1 and state["enabled"] is False


# --------------------------------------------------------------------------- LoRA merge

class Wrapper(nn.Module):
    """The predictor's model shape: a PEFT backbone plus non-LoRA heads."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(32, 4)


def lora_wrapper(dtype=torch.bfloat16, *, second_adapter=False):
    from peft import LoraConfig, get_peft_model
    base = tiny_model(dtype)
    targets = [name for name, module in base.named_modules()
               if "language_model" in name and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj", "down_proj")
               and isinstance(module, nn.Linear)]
    backbone = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=targets, lora_dropout=0.0))
    if second_adapter:
        backbone.add_adapter("other", LoraConfig(r=2, lora_alpha=2, target_modules=targets))
    torch.manual_seed(3)
    with torch.no_grad():
        for name, parameter in backbone.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn_like(parameter) * 0.1)
    return Wrapper(backbone).eval()


def test_merge_lora_rounds_once_from_fp32_and_unloads_everything():
    from peft.tuners.lora import LoraLayer
    model = lora_wrapper()
    assert acceleration.lora_merge_supported(model)
    expected = {}
    for name, layer in model.backbone.named_modules():
        if isinstance(layer, LoraLayer):
            weight = layer.get_base_layer().weight
            update = layer.lora_B["default"].weight.float() @ layer.lora_A["default"].weight.float()
            expected[name.replace("base_model.model.", "")] = (
                weight.float() + update * layer.scaling["default"]).to(torch.bfloat16)
    untouched = {name: value.clone() for name, value in model.named_parameters()
                 if "lora_" not in name and "visual" in name or name.startswith("head.")}
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        summary = acceleration.merge_lora(model)
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert summary == {"method": "fp32_round_once", "tf32": False, "merged_layers": len(expected),
                       "scaling": [2.0], "weight_dtypes": ["bfloat16"]}
    assert isinstance(model.backbone, qwen.Qwen3_5ForConditionalGeneration)
    assert not any(isinstance(module, LoraLayer) for module in model.modules())
    assert not any("lora_" in name for name, _ in model.named_parameters())
    merged = dict(model.backbone.named_modules())
    for name, weight in expected.items():
        assert torch.equal(merged[name].weight, weight), name
    parameters = dict(model.named_parameters())
    for name, value in untouched.items():
        key = name.replace("backbone.base_model.model.", "backbone.")
        assert torch.equal(parameters[key], value), name
    assert not acceleration.lora_merge_supported(model)


@pytest.mark.skipif(not hasattr(torch.backends.cuda.matmul, "fp32_precision"),
                    reason="torch without the fp32_precision TF32 API")
def test_merge_lora_works_when_host_set_tf32_with_the_new_api():
    from peft.tuners.lora import LoraLayer
    from torch.utils._python_dispatch import TorchDispatchMode

    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    model = lora_wrapper()
    layer = next(module for module in model.modules() if isinstance(module, LoraLayer))
    base = layer.get_base_layer()
    update = layer.lora_B["default"].weight.float() @ layer.lora_A["default"].weight.float()
    expected = (base.weight.float() + update * layer.scaling["default"]).to(torch.bfloat16)
    precisions = []

    class CheckMergePrecision(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.mm.default:
                precisions.append(matmul.fp32_precision)
            return func(*args, **(kwargs or {}))

    matmul.fp32_precision = "tf32"
    try:
        # Whether legacy reads raise depends on the Torch build. Verify the
        # precision used for real merge matmuls and the host setting afterward.
        with CheckMergePrecision():
            summary = acceleration.merge_lora(model)
        assert precisions and set(precisions) == {"ieee"}
        assert matmul.fp32_precision == "tf32"
    finally:
        matmul.fp32_precision = previous
    assert summary["merged_layers"] > 0 and summary["tf32"] is False
    # Unload keeps the base Linear, whose weight was merged in place.
    assert torch.equal(base.weight, expected)


def test_merged_lora_weight_does_not_modify_layer():
    from peft.tuners.lora import LoraLayer
    model = lora_wrapper()
    layer = next(module for module in model.modules() if isinstance(module, LoraLayer))
    before = layer.get_base_layer().weight.clone()
    acceleration.merged_lora_weight(layer)
    assert torch.equal(layer.get_base_layer().weight, before)


def test_merge_lora_preserves_fp32_forward():
    model = lora_wrapper(torch.float32)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        before = model.backbone(input_ids=ids).logits
        acceleration.merge_lora(model)
        after = model.backbone(input_ids=ids).logits
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


def test_merge_lora_rejects_unsupported_layouts():
    assert not acceleration.lora_merge_supported(nn.Linear(2, 2))
    assert not acceleration.lora_merge_supported(Wrapper(tiny_model()))
    model = lora_wrapper(second_adapter=True)
    assert not acceleration.lora_merge_supported(model)
    with pytest.raises(ValueError, match="Unsupported LoRA layout"):
        acceleration.merge_lora(model)
    with pytest.raises(ValueError, match="requires PEFT LoRA"):
        acceleration.merge_lora(Wrapper(tiny_model()))


# --------------------------------------------------------------------------- causal-conv1d

def test_causal_conv1d_reports_and_can_force_the_torch_path(monkeypatch):
    model = tiny_model()
    layers = [module for module in model.modules() if isinstance(module, qwen.Qwen3_5GatedDeltaNet)]
    assert len(layers) == 1
    report = acceleration.configure_causal_conv1d(model)
    assert report == {"requested": "auto", "package_available": qwen.causal_conv1d_fn is not None,
                      "active": layers[0].causal_conv1d_fn is not None, "linear_attention_layers": 1}

    def kernel(*args, **kwargs):
        raise AssertionError("not called")

    layers[0].causal_conv1d_fn = kernel
    assert acceleration.configure_causal_conv1d(model, True)["active"] is True
    report = acceleration.configure_causal_conv1d(model, False)
    assert report["requested"] is False and report["active"] is False
    assert layers[0].causal_conv1d_fn is None
    assert layers[0].causal_conv1d_update is qwen.torch_causal_conv1d_update


def test_causal_conv1d_absent_never_fails_unless_required():
    model = tiny_model()
    for layer in model.modules():
        if isinstance(layer, qwen.Qwen3_5GatedDeltaNet):
            layer.causal_conv1d_fn = None
    assert acceleration.configure_causal_conv1d(model)["active"] is False
    with pytest.raises(RuntimeError, match="causal-conv1d"):
        acceleration.configure_causal_conv1d(model, True)
    report = acceleration.configure_causal_conv1d(nn.Linear(2, 2))
    assert report["active"] is False and report["linear_attention_layers"] == 0
    with pytest.raises(RuntimeError):
        acceleration.configure_causal_conv1d(nn.Linear(2, 2), True)


# --------------------------------------------------------------------------- decoder compile

def test_decoder_compile_uses_default_mode_without_cuda_graphs(identity_compile, dynamo_limit):
    config, name = dynamo_limit
    setattr(config, name, 8)
    acceleration.configure_decoder_compile(tiny_model())
    assert getattr(config, name) == acceleration.RECOMPILE_LIMIT
    assert identity_compile == [{"dynamic": False, "fullgraph": False}]
    setattr(config, name, 1000)
    acceleration.configure_decoder_compile(tiny_model())
    assert getattr(config, name) == 1000


def test_decoder_compile_is_exact_and_falls_back_for_cache_grad_and_padding(identity_compile):
    reference_model = tiny_model()
    model = copy.deepcopy(reference_model)
    state = acceleration.configure_decoder_compile(model)
    ids, types, grid, pixels = prompt([(1, 4, 4)])
    inputs = dict(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid, pixel_values=pixels,
                  attention_mask=torch.ones_like(ids))
    with torch.inference_mode():
        assert torch.equal(model(**inputs, use_cache=False).logits,
                           reference_model(**inputs, use_cache=False).logits)
    assert state == {"compiled_calls": 1, "fallbacks": 0}
    with torch.inference_mode():
        # use_cache and a padding mask keep HF's path.
        assert torch.equal(model(**inputs, use_cache=True).logits,
                           reference_model(**inputs, use_cache=True).logits)
        padded = dict(inputs, attention_mask=torch.cat((torch.zeros(1, 1, dtype=torch.long),
                                                        torch.ones(1, ids.shape[1] - 1, dtype=torch.long)), 1))
        assert torch.equal(model(**padded, use_cache=False).logits,
                           reference_model(**padded, use_cache=False).logits)
    assert state == {"compiled_calls": 1, "fallbacks": 2}
    # Enabled autograd must take the original, differentiable path.
    output = model(**inputs, use_cache=False).logits
    assert output.grad_fn is not None
    assert state["fallbacks"] == 3


def test_decoder_compile_requires_qwen_text_model():
    with pytest.raises(ValueError):
        acceleration.configure_decoder_compile(nn.Linear(2, 2))


def test_strict_decoder_rejects_cache_autograd_and_padding(identity_compile):
    model = tiny_model()
    state = acceleration.configure_decoder_compile(model, strict=True)
    ids, types, grid, pixels = prompt([(1, 4, 4)])
    inputs = dict(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid, pixel_values=pixels,
                  attention_mask=torch.ones_like(ids))
    with torch.inference_mode(), pytest.raises(ValueError, match="compiled decoder"):
        model(**inputs, use_cache=True)
    with pytest.raises(ValueError, match="compiled decoder"):
        model(**inputs, use_cache=False)
    inputs["attention_mask"][0, 0] = 0
    with torch.inference_mode(), pytest.raises(ValueError, match="compiled decoder"):
        model(**inputs, use_cache=False)
    assert state == {"compiled_calls": 0, "fallbacks": 0}


def test_padded_batch_decoder_uses_hf_linear_mask_and_preserves_unpadded_path(identity_compile):
    reference = tiny_model()
    model = copy.deepcopy(reference)
    state = acceleration.configure_decoder_compile(model, strict=True, allow_padding=True)
    ids, types, grid, pixels = prompt([(1, 4, 4)], text=(1, 2, 3, 4), batch=2)
    mask = torch.ones_like(ids)
    mask[0, :2] = 0
    inputs = dict(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid,
                  pixel_values=pixels, attention_mask=mask)
    with torch.inference_mode():
        actual = model(**inputs, use_cache=False).logits
        expected = reference(**inputs, use_cache=False).logits
        assert torch.equal(actual, expected)
        # A subsequent unpadded request retains the original compiled function.
        inputs["attention_mask"] = torch.ones_like(ids)
        assert torch.equal(model(**inputs, use_cache=False).logits, reference(**inputs, use_cache=False).logits)
    assert state == {"compiled_calls": 2, "fallbacks": 0, "padded_compiled_calls": 1}
    assert identity_compile == [{"dynamic": False, "fullgraph": False}] * 2


@pytest.mark.parametrize("name", ["TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"])
def test_required_compile_policy_rejects_environment_disable(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(RuntimeError, match=name):
        acceleration.require_compiled_inference()


def test_required_compile_policy_rejects_process_disable():
    import torch._dynamo
    with torch._dynamo.config.patch(disable=True):
        with pytest.raises(RuntimeError, match="torch._dynamo.config.disable"):
            acceleration.require_compiled_inference()


def test_required_compile_policy_raises_at_shape_limit_instead_of_running_eager():
    import torch._dynamo
    from torch._dynamo.exc import FailOnRecompileLimitHit
    with torch._dynamo.config.patch(suppress_errors=True, fail_on_recompile_limit_hit=False,
                                   recompile_limit=1):
        acceleration.require_compiled_inference()
        compiled = torch.compile(lambda x: x.sin() + 1, backend="eager", dynamic=False)
        assert torch.equal(compiled(torch.ones(2)), torch.ones(2).sin() + 1)
        with pytest.raises(FailOnRecompileLimitHit):
            compiled(torch.ones(3))
        assert torch._dynamo.config.suppress_errors is False


# --------------------------------------------------------------------------- vision compile

def test_vision_compile_requires_batched_sdpa(identity_compile):
    with pytest.raises(ValueError, match="batched vision SDPA"):
        acceleration.configure_vision_compile(tiny_model())
    with pytest.raises(ValueError):
        acceleration.configure_vision_compile(nn.Linear(2, 2))


def batched_pair():
    reference_model = tiny_model()
    configure_batched_vision_sdpa(reference_model)
    model = copy.deepcopy(reference_model)
    return reference_model, model


def test_vision_compile_is_exact_caches_tables_and_restores_attention(identity_compile):
    reference_model, model = batched_pair()
    state = acceleration.configure_vision_compile(model)
    attentions = [block.attn for block in model.model.visual.blocks]
    before = [attention.forward for attention in attentions]
    calls = []
    real = model.model.visual.fast_pos_embed_interpolate

    def counted(grid):
        calls.append(tuple(map(tuple, grid.tolist())))
        return real(grid)

    model.model.visual.fast_pos_embed_interpolate = counted
    for grids in ([(1, 4, 4), (1, 4, 4)], [(1, 4, 4), (1, 4, 4)], [(1, 4, 8)]):
        _, _, grid, pixels = prompt(grids)
        with torch.inference_mode():
            expected = reference_model.model.visual(pixels, grid_thw=grid)
            actual = model.model.visual(pixels, grid_thw=grid)
        assert torch.equal(actual.last_hidden_state, expected.last_hidden_state)
        assert torch.equal(actual.pooler_output, expected.pooler_output)
        assert [attention.forward for attention in attentions] == before
    assert calls == [((1, 4, 4), (1, 4, 4)), ((1, 4, 8),)]
    assert state == {"compiled_calls": 3, "fallbacks": 0}


@pytest.mark.parametrize("grids", [[(2, 4, 4)], [(2, 4, 4), (1, 4, 4)], [(3, 4, 8)]],
                         ids=["two_frames", "frames_and_image", "three_frames_non_square"])
def test_vision_compile_attends_per_temporal_frame_like_hf(identity_compile, grids):
    hf_model = tiny_model()
    reference_model, model = batched_pair()
    state = acceleration.configure_vision_compile(model)
    _, _, grid, pixels = prompt(grids)
    with torch.inference_mode():
        hf = hf_model.model.visual(pixels, grid_thw=grid).pooler_output
        batched = reference_model.model.visual(pixels, grid_thw=grid).pooler_output
        actual = model.model.visual(pixels, grid_thw=grid).pooler_output
    # HF splits attention per frame (cu_seqlens = repeat_interleave(h * w, t)).
    assert torch.equal(actual, batched)
    torch.testing.assert_close(actual, hf, rtol=1e-5, atol=1e-5)
    assert state == {"compiled_calls": 1, "fallbacks": 0}


def test_vision_compile_falls_back_for_unequal_frames(identity_compile):
    hf_model = tiny_model()
    _, model = batched_pair()
    state = acceleration.configure_vision_compile(model)
    _, _, grid, pixels = prompt([(2, 4, 4), (1, 4, 8)])
    with torch.inference_mode():
        torch.testing.assert_close(model.model.visual(pixels, grid_thw=grid).pooler_output,
                                   hf_model.model.visual(pixels, grid_thw=grid).pooler_output, rtol=1e-5, atol=1e-5)
    assert state == {"compiled_calls": 0, "fallbacks": 1}


def test_vision_compile_falls_back_for_unequal_images_and_autograd(identity_compile):
    reference_model, model = batched_pair()
    state = acceleration.configure_vision_compile(model)
    _, _, grid, pixels = prompt([(1, 4, 4), (1, 4, 8)])
    with torch.inference_mode():
        expected = reference_model.model.visual(pixels, grid_thw=grid).pooler_output
        assert torch.equal(model.model.visual(pixels, grid_thw=grid).pooler_output, expected)
    _, _, grid, pixels = prompt([(1, 4, 4)])
    output = model.model.visual(pixels, grid_thw=grid).pooler_output
    assert output.grad_fn is not None
    assert state == {"compiled_calls": 0, "fallbacks": 2}


def test_strict_vision_rejects_unequal_images_and_autograd(identity_compile):
    _, model = batched_pair()
    state = acceleration.configure_vision_compile(model, strict=True)
    _, _, grid, pixels = prompt([(1, 4, 4), (1, 4, 8)])
    with torch.inference_mode(), pytest.raises(ValueError, match="equal resized image token counts"):
        model.model.visual(pixels, grid_thw=grid)
    _, _, grid, pixels = prompt([(1, 4, 4)])
    with pytest.raises(ValueError, match="compiled vision tower"):
        model.model.visual(pixels, grid_thw=grid)
    assert state == {"compiled_calls": 0, "fallbacks": 0}


def test_vision_compile_restores_attention_after_compiled_failure(monkeypatch):
    def broken(function, **kwargs):
        def run(*args):
            raise RuntimeError("compile failed")
        return run

    monkeypatch.setattr(torch, "compile", broken)
    _, model = batched_pair()
    acceleration.configure_vision_compile(model)
    attentions = [block.attn for block in model.model.visual.blocks]
    before = [attention.forward for attention in attentions]
    _, _, grid, pixels = prompt([(1, 4, 4)])
    with torch.inference_mode(), pytest.raises(RuntimeError, match="compile failed"):
        model.model.visual(pixels, grid_thw=grid)
    assert [attention.forward for attention in attentions] == before


def test_full_stack_forward_is_bitwise_equal_to_batched_reference(identity_compile):
    reference_model, model = batched_pair()
    counters = [acceleration.configure_host_position_ids(model), acceleration.configure_decoder_compile(model),
                acceleration.configure_vision_compile(model)]
    ids, types, grid, pixels = prompt([(1, 4, 4), (1, 4, 4)])
    inputs = dict(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid, pixel_values=pixels,
                  attention_mask=torch.ones_like(ids), use_cache=False)
    with torch.inference_mode():
        assert torch.equal(model(**inputs).logits, reference_model(**inputs).logits)
    assert [counter.get("fallbacks") for counter in counters] == [0, 0, 0]
    assert counters[0]["calls"] == counters[1]["compiled_calls"] == counters[2]["compiled_calls"] == 1


# --------------------------------------------------------------------------- predictor wiring

def stack_predictor(device="cpu"):
    model = lora_wrapper(torch.float32)
    configure_delta_rule_backend(model.backbone, "auto")
    predictor = inference.GranaPredictor.__new__(inference.GranaPredictor)
    predictor.model, predictor.device = model, torch.device(device)
    predictor._request_images = ContextVar("test-stack", default=None)
    predictor._counters = {}
    return predictor


@pytest.fixture
def required_kernels(monkeypatch):
    """Keep real LoRA/position/compile wiring; stub only unavailable CPU-fixture CUDA kernels."""
    from grana.modeling import backends
    original_convolution = acceleration.configure_causal_conv1d

    def guard(enabled):
        assert enabled is True
        return {"requested": True, "applied": True}

    def convolution(model, enabled, device):
        if torch.device(device).type == "cpu":
            return original_convolution(model, enabled, device)
        assert enabled is True and device.type == "cuda"
        return {"requested": True, "active": True, "package_available": True}

    monkeypatch.setattr(acceleration, "configure_fla_output_kernel_guard", guard)
    monkeypatch.setattr(acceleration, "configure_causal_conv1d", convolution)
    monkeypatch.setattr(backends, "configure_delta_rule_backend", lambda *args: {"resolved": "fla"})


def test_required_stack_enables_all_pieces_and_records_independent_counters(identity_compile, required_kernels):
    from peft.tuners.lora import LoraLayer
    predictor = stack_predictor(device="cuda")
    stack = predictor._configure_inference_stack()
    for name in ("merge_lora", "cpu_position_ids", "batched_vision_sdpa", "reuse_images",
                 "compile_decoder", "compile_vision"):
        assert stack[name] is True
    assert stack["lora_merge"]["merged_layers"] > 0
    assert stack["batched_vision_modules"] == 2
    assert stack["causal_conv1d"]["requested"] is True
    assert stack["fla_output_guard"]["requested"] is True
    assert stack["compile"]["vision_variant"] == acceleration.VISION_COMPILE_VARIANT
    assert not any(isinstance(module, LoraLayer) for module in predictor.model.modules())
    assert identity_compile == [{"dynamic": False, "fullgraph": False}] * 3
    assert predictor._counters["decoder_compile"]["padded_compiled_calls"] == 0
    counters = predictor.runtime_counters()
    assert set(counters) == {"cpu_position_ids", "decoder_compile", "vision_compile"}
    counters["decoder_compile"]["compiled_calls"] = 99
    assert predictor.runtime_counters()["decoder_compile"]["compiled_calls"] == 0


def test_required_merged_stack_matches_unmerged_fp32_prediction(identity_compile, required_kernels):
    plain, fast = stack_predictor(), stack_predictor(device="cuda")
    fast._configure_inference_stack()
    ids, types, grid, pixels = prompt([(1, 4, 4), (1, 4, 4)])
    inputs = dict(input_ids=ids, mm_token_type_ids=types, image_grid_thw=grid, pixel_values=pixels,
                  attention_mask=torch.ones_like(ids))
    with torch.inference_mode():
        expected = plain.model.backbone(**inputs).logits
        actual = fast.model.backbone(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert fast.runtime_counters()["cpu_position_ids"]["calls"] == 1
    assert all(counter["fallbacks"] == 0 for counter in fast.runtime_counters().values())


def test_required_stack_fails_if_convolution_kernel_is_missing(identity_compile, required_kernels, monkeypatch):
    predictor = stack_predictor(device="cuda")
    def missing(*args):
        raise RuntimeError("causal-conv1d unavailable")
    monkeypatch.setattr(acceleration, "configure_causal_conv1d", missing)
    with pytest.raises(RuntimeError, match="causal-conv1d unavailable"):
        predictor._configure_inference_stack()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real torch.compile parity needs CUDA")
def test_real_compile_on_cuda_matches_eager_closely():
    reference_model = tiny_model(torch.bfloat16).cuda()
    configure_batched_vision_sdpa(reference_model)
    model = copy.deepcopy(reference_model)
    counters = [acceleration.configure_host_position_ids(model), acceleration.configure_decoder_compile(model),
                acceleration.configure_vision_compile(model)]
    ids, types, grid, pixels = prompt([(1, 4, 4), (1, 4, 4)])
    inputs = dict(input_ids=ids.cuda(), mm_token_type_ids=types.cuda(), image_grid_thw=grid.cuda(),
                  pixel_values=pixels.cuda().bfloat16(), attention_mask=torch.ones_like(ids).cuda(),
                  use_cache=False)
    with torch.inference_mode():
        expected = reference_model(**inputs).logits.float()
        for _ in range(2):
            torch.testing.assert_close(model(**inputs).logits.float(), expected, rtol=5e-2, atol=5e-2)
    assert counters[1]["compiled_calls"] == counters[2]["compiled_calls"] == 2
    assert counters[1]["fallbacks"] == counters[2]["fallbacks"] == 0


def test_causal_conv1d_auto_uses_torch_convolution_off_cuda():
    model = qwen.Qwen3_5ForConditionalGeneration(tiny_config()).eval()
    for layer in model.modules():
        if isinstance(layer, qwen.Qwen3_5GatedDeltaNet):
            layer.causal_conv1d_fn = object()  # stand-in for an imported CUDA kernel
    report = acceleration.configure_causal_conv1d(model, None, "cpu")
    assert report["active"] is False and report["linear_attention_layers"] > 0
    assert all(layer.causal_conv1d_fn is None for layer in model.modules()
               if isinstance(layer, qwen.Qwen3_5GatedDeltaNet))
    with pytest.raises(RuntimeError, match="CUDA"):
        acceleration.configure_causal_conv1d(model, True, "cpu")


@pytest.fixture
def fla_tuner(monkeypatch):
    # Isolate process state and exercise FLA's optional disk-loader interface
    # without depending on CUDA, Triton, or an installed FLA package.
    configs = [SimpleNamespace(num_warps=warps) for warps in (8, 4, 2)]
    tuner = SimpleNamespace(configs=configs, cache={"unsafe": configs[0], "safe": configs[1]})
    tuner.maybe_load_cached_config = lambda key: tuner.cache.__setitem__(key.autotune_key, configs[0])
    chunk_o = SimpleNamespace(chunk_fwd_kernel_o=SimpleNamespace(fn=tuner))
    monkeypatch.setitem(sys.modules, "fla", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "fla.ops.common", SimpleNamespace(chunk_o=chunk_o))
    return tuner


def test_fla_output_guard_drops_eight_warp_configs_and_cached_ones(fla_tuner):
    report = acceleration.configure_fla_output_kernel_guard()
    assert report["applied"] and report["requested"] is True and report["scope"] == "process"
    assert report["max_num_warps"] == 4 and report["removed_configs"] == 1
    assert [config.num_warps for config in fla_tuner.configs] == [4, 2]
    assert set(fla_tuner.cache) == {"safe"}
    assert acceleration.configure_fla_output_kernel_guard() == report
    key = SimpleNamespace(autotune_key=("from-disk",))
    fla_tuner.maybe_load_cached_config(key)
    assert key.autotune_key not in fla_tuner.cache
    # Reports are snapshots, not a public handle to mutate the process policy.
    report["requested"] = False
    assert acceleration.configure_fla_output_kernel_guard()["requested"] is True


@pytest.mark.parametrize("keyword_arguments", [False, True])
def test_fla_output_guard_forwards_cache_loader_context(fla_tuner, keyword_arguments):
    def load(key, nargs, runtime_kwargs):
        assert nargs == {"q": "tensor"} and runtime_kwargs == {"BT": 64}
        fla_tuner.cache[key.autotune_key] = SimpleNamespace(num_warps=8)

    fla_tuner.maybe_load_cached_config = load
    acceleration.configure_fla_output_kernel_guard()
    key = SimpleNamespace(autotune_key=("fla-060",))
    if keyword_arguments:
        fla_tuner.maybe_load_cached_config(key=key, nargs={"q": "tensor"}, runtime_kwargs={"BT": 64})
    else:
        fla_tuner.maybe_load_cached_config(key, {"q": "tensor"}, {"BT": 64})
    assert key.autotune_key not in fla_tuner.cache


@pytest.mark.parametrize("first", [False, True])
def test_fla_output_guard_rejects_conflicts_in_both_orders(fla_tuner, first):
    report = acceleration.configure_fla_output_kernel_guard(first)
    configs, cache, loader = list(fla_tuner.configs), dict(fla_tuner.cache), fla_tuner.maybe_load_cached_config
    with pytest.raises(RuntimeError, match="separate processes"):
        acceleration.configure_fla_output_kernel_guard(not first)
    assert fla_tuner.configs == configs and fla_tuner.cache == cache
    assert fla_tuner.maybe_load_cached_config is loader
    assert acceleration.configure_fla_output_kernel_guard(first) == report
    assert report["applied"] == first and report["requested"] == first
    if not first:
        assert [config.num_warps for config in configs] == [8, 4, 2]
        key = SimpleNamespace(autotune_key=("unguarded-disk",))
        loader(key)
        assert fla_tuner.cache[key.autotune_key].num_warps == 8


def test_fla_output_guard_serializes_conflicting_initialization(fla_tuner):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)

    def configure_guard(enabled):
        barrier.wait(timeout=5)
        try:
            return acceleration.configure_fla_output_kernel_guard(enabled)
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(configure_guard, [False, True]))
    reports = [item for item in results if isinstance(item, dict)]
    assert len(reports) == 1 and sum(isinstance(item, RuntimeError) for item in results) == 1
    expected = [4, 2] if reports[0]["applied"] else [8, 4, 2]
    assert [config.num_warps for config in fla_tuner.configs] == expected


def test_required_predictor_guard_rejects_prior_process_opt_out(fla_tuner):
    acceleration.configure_fla_output_kernel_guard(False)
    with pytest.raises(RuntimeError, match="process-wide"):
        stack_predictor(device="cuda")._configure_inference_stack()


@pytest.mark.parametrize("unavailable", ["missing_package", "layout", "no_safe_config"])
def test_required_predictor_guard_fails_without_a_safe_implementation(fla_tuner, monkeypatch, unavailable):
    if unavailable == "missing_package":
        monkeypatch.setitem(sys.modules, "fla", None)
    elif unavailable == "layout":
        fla_tuner.configs = None
    else:
        fla_tuner.configs = [SimpleNamespace(num_warps=8)]
    with pytest.raises(RuntimeError, match="Inference requires the FLA output-kernel guard"):
        stack_predictor(device="cuda")._configure_inference_stack()
    assert not hasattr(fla_tuner, "_grana_fla_output_guard")
