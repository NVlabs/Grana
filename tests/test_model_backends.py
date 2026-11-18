# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from torch import nn

from grana.modeling.backends import checkpoint_verification_delta_rule, configure_delta_rule_backend, configure_sdpa_backends


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_reference_is_temporary_and_restores_fla_on_error(make_layer, qwen, fail):
    layer = make_layer().eval()
    layer.chunk_gated_delta_rule = foreign_chunk
    layer.recurrent_gated_delta_rule = foreign_recurrent
    other = make_layer().eval()
    other.chunk_gated_delta_rule = foreign_chunk
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    try:
        with checkpoint_verification_delta_rule(layer) as report:
            assert report == {"mode": "torch_reference", "scope": "checkpoint_integrity_only", "switched_layers": 1}
            assert layer.chunk_gated_delta_rule is qwen.torch_chunk_gated_delta_rule
            assert layer.recurrent_gated_delta_rule is qwen.torch_recurrent_gated_delta_rule
            assert other.chunk_gated_delta_rule is foreign_chunk
            if fail:
                raise RuntimeError("simulated validation failure")
    except RuntimeError as exc:
        assert fail and str(exc) == "simulated validation failure"
    assert layer.chunk_gated_delta_rule is foreign_chunk
    assert layer.recurrent_gated_delta_rule is foreign_recurrent
    assert all(torch.equal(value, before[name]) for name, value in layer.state_dict().items())


def test_checkpoint_reference_rejects_training_mode(make_layer):
    layer = make_layer().train()
    layer.chunk_gated_delta_rule = foreign_chunk
    with pytest.raises(ValueError, match="eval mode"):
        with checkpoint_verification_delta_rule(layer):
            pass
    assert layer.chunk_gated_delta_rule is foreign_chunk


def test_sdpa_policy_survives_checkpoint_recomputation_and_preserves_other_flags():
    from torch.backends import cuda
    from torch.utils.checkpoint import checkpoint

    switches = {"cudnn": (cuda.cudnn_sdp_enabled, cuda.enable_cudnn_sdp),
                "flash": (cuda.flash_sdp_enabled, cuda.enable_flash_sdp),
                "memory_efficient": (cuda.mem_efficient_sdp_enabled, cuda.enable_mem_efficient_sdp),
                "math": (cuda.math_sdp_enabled, cuda.enable_math_sdp)}
    original = {name: getter() for name, (getter, _) in switches.items()}
    try:
        # Preserve a caller's deliberate choices for the remaining backends.
        cuda.enable_cudnn_sdp(True)
        cuda.enable_flash_sdp(False)
        cuda.enable_mem_efficient_sdp(True)
        cuda.enable_math_sdp(True)
        report = configure_sdpa_backends()
        expected = {"cudnn": False, "flash": False, "memory_efficient": True, "math": True}
        assert report["enabled"] == expected
        assert report["scope"] == "process"
        assert json.loads(json.dumps(report)) == report
        assert configure_sdpa_backends() == report
        observed = []

        def layer(value):
            observed.append({name: getter() for name, (getter, _) in switches.items()})
            return value.sin()

        inputs = torch.randn(2, 3, requires_grad=True)
        output = checkpoint(layer, inputs, use_reentrant=False)
        output.sum().backward()
        assert observed == [expected, expected]
        torch.testing.assert_close(inputs.grad, inputs.detach().cos())
    finally:
        for name, (_, setter) in switches.items():
            setter(original[name])


@pytest.fixture
def qwen(monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    # Construct only tiny CPU layers, including on hosts with FLA installed.
    monkeypatch.setattr(modeling_qwen3_5, "FusedRMSNormGated", None)
    return modeling_qwen3_5


@pytest.fixture
def make_layer(qwen):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, layer_types=["linear_attention"],
    )

    def make():
        layer = qwen.Qwen3_5GatedDeltaNet(config, 0)
        layer.causal_conv1d_fn = None
        layer.causal_conv1d_update = qwen.torch_causal_conv1d_update
        return layer

    return make


def foreign_chunk(*args, **kwargs):
    raise AssertionError("Diagnostic backend must not execute")


def foreign_recurrent(*args, **kwargs):
    raise AssertionError("Diagnostic backend must not execute")


foreign_chunk.__module__ = "fla.ops.gated_delta_rule"
foreign_recurrent.__module__ = "fla.ops.gated_delta_rule"


@pytest.mark.parametrize("backend", [None, False, 0, [], {}, "", "Torch", "fla"])
def test_invalid_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="delta_rule_backend.*auto.*torch"):
        configure_delta_rule_backend(nn.Linear(1, 1), backend)


def test_auto_retains_callables_and_records_actual_implementations(make_layer):
    layer = make_layer()
    layer.chunk_gated_delta_rule = foreign_chunk
    layer.recurrent_gated_delta_rule = foreign_recurrent
    base = nn.ModuleDict({"target": layer, "unrelated": nn.Linear(2, 2)})
    report = configure_delta_rule_backend(base)
    assert layer.chunk_gated_delta_rule is foreign_chunk
    assert layer.recurrent_gated_delta_rule is foreign_recurrent
    assert report == {
        "requested": "auto", "resolved": "fla", "layers": ["target"],
        "implementations": {"target": {
            "chunk": "fla.ops.gated_delta_rule.foreign_chunk",
            "recurrent": "fla.ops.gated_delta_rule.foreign_recurrent"}},
    }
    assert json.loads(json.dumps(report)) == report


def test_torch_changes_only_target_instance_and_preserves_parameters(make_layer, qwen):
    first, second = make_layer().eval(), make_layer()
    for layer in (first, second):
        layer.chunk_gated_delta_rule = foreign_chunk
        layer.recurrent_gated_delta_rule = foreign_recurrent
    global_functions = qwen.chunk_gated_delta_rule, qwen.fused_recurrent_gated_delta_rule
    parameters = dict(first.named_parameters())
    states = {name: value.clone() for name, value in first.state_dict().items()}
    unchanged_components = first.norm, first.conv1d, first.causal_conv1d_fn, first.causal_conv1d_update

    report = configure_delta_rule_backend(first, "torch")

    assert report["requested"] == report["resolved"] == "torch"
    assert report["layers"] == ["<root>"]
    prefix = "transformers.models.qwen3_5.modeling_qwen3_5."
    assert report["implementations"] == {"<root>": {
        "chunk": prefix + "torch_chunk_gated_delta_rule",
        "recurrent": prefix + "torch_recurrent_gated_delta_rule"}}
    assert first.chunk_gated_delta_rule is qwen.torch_chunk_gated_delta_rule
    assert first.recurrent_gated_delta_rule is qwen.torch_recurrent_gated_delta_rule
    assert second.chunk_gated_delta_rule is foreign_chunk
    assert second.recurrent_gated_delta_rule is foreign_recurrent
    assert global_functions == (qwen.chunk_gated_delta_rule, qwen.fused_recurrent_gated_delta_rule)
    assert not first.training and second.training
    assert unchanged_components == (first.norm, first.conv1d, first.causal_conv1d_fn, first.causal_conv1d_update)
    assert states.keys() == first.state_dict().keys()
    assert parameters.keys() == dict(first.named_parameters()).keys()
    for name, value in first.state_dict().items():
        assert torch.equal(value, states[name])
    for name, parameter in first.named_parameters():
        assert parameter is parameters[name]
    # Auto reports an explicit prior selection without resetting the instance.
    assert configure_delta_rule_backend(first)["resolved"] == "torch"
    assert first.chunk_gated_delta_rule is qwen.torch_chunk_gated_delta_rule


def test_torch_tiny_forward_and_backward_are_finite_on_cpu(make_layer):
    layer = make_layer()
    configure_delta_rule_backend(layer, "torch")
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    output = layer(hidden)
    assert output.device.type == "cpu" and output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    for gradient in (hidden.grad, layer.in_proj_qkv.weight.grad, layer.out_proj.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()


def test_explicit_torch_requires_target_layers():
    base = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="requires Qwen3_5GatedDeltaNet"):
        configure_delta_rule_backend(base, "torch")
    assert configure_delta_rule_backend(base) == {
        "requested": "auto", "resolved": "none", "layers": [], "implementations": {}}
