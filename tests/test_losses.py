# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinal objectives and optimizer-window scaling, using tiny tensors."""
import math
from types import SimpleNamespace

import pytest
import torch

from grana.training.losses import (
    FRAME_WEIGHTS,
    continuous_to_c51,
    normalized_loss,
    progress_loss,
    progress_loss_components,
    role_loss_components,
    role_numerators,
    stage_loss,
    stage_loss_components,
    window_role_counts,
)


def test_c51_endpoints_and_interpolation():
    values = torch.tensor([0.0, 1.0, 0.5, 2 / 9])
    projected = continuous_to_c51(values)
    assert projected[0, 0] == projected[1, 9] == 1
    torch.testing.assert_close(projected.sum(-1), torch.ones(4))
    torch.testing.assert_close(projected[2, 4:6], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close((projected * torch.linspace(0, 1, 10)).sum(-1), values)


@pytest.mark.parametrize("ceiling_weight", [0.0, 3.0])
def test_progress_loss_against_scalar_reference_and_gradient(ceiling_weight):
    logits = torch.tensor([[[-0.4, 0.1, 0.5, -0.2, 0.4, -0.1, 0.7, -0.3, 0.6, 0.0]]], requires_grad=True)
    cdf_weight = 0.7
    for target in (0.0, 0.04, 0.05, 0.47, 1.0):
        probabilities = [math.exp(x) / sum(math.exp(y) for y in logits.detach().flatten().tolist()) for x in logits.detach().flatten().tolist()]
        position = float(torch.tensor(target)) * 9
        lower, upper = math.floor(position), math.ceil(position)
        distribution = [0.0] * 10
        distribution[lower] += upper - position if lower != upper else 1
        distribution[upper] += position - lower if lower != upper else 0
        ce = -sum(q * math.log(p) for p, q in zip(probabilities, distribution))
        expected = sum(i / 9 * p for i, p in enumerate(probabilities))
        ceiling = ceiling_weight * max(expected - float(torch.tensor(target)) - 1 / 9, 0) if target < 0.05 else 0
        cdf = sum((sum(probabilities[:boundary]) - sum(distribution[:boundary])) ** 2 for boundary in range(1, 10)) / 9
        reference = ce + cdf_weight * cdf + ceiling
        actual = progress_loss(logits, torch.tensor([[target]]), cdf_weight=cdf_weight, ceiling_weight=ceiling_weight)
        assert actual.item() == pytest.approx(reference, abs=2e-6)
        actual.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_stage_ce_cdf_and_masked_classes():
    logits = torch.tensor([[[0.0, 0.0, 0.0, -torch.inf]]], requires_grad=True)
    loss = stage_loss(logits, torch.tensor([[0]]))
    assert loss.item() == pytest.approx(math.log(3) + 5 / 18, rel=1e-6)
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_stage_cdf_uses_valid_task_size_for_mixed_padded_rows():
    counts = [1, 2, 3, 8, 10, 26]
    targets = torch.tensor([0, 1, 0, 3, 4, 25])
    logits = torch.full((len(counts), 26), -torch.inf)
    for row, count in enumerate(counts):
        logits[row, :count] = 0
    logits.requires_grad_()
    parts = stage_loss_components(logits, targets, cdf_weight=1.3)
    references = []
    for count, target in zip(counts, targets.tolist()):
        cdf = sum((boundary / count - int(target < boundary)) ** 2 for boundary in range(1, count)) / max(count - 1, 1)
        references.append(math.log(count) + 1.3 * cdf)
    torch.testing.assert_close(parts["total"], torch.tensor(references))
    # Padding a task to 26 answer letters must not dilute its ordinal penalty.
    torch.testing.assert_close(parts["cdf"][2], torch.tensor(1.3 * 5 / 18))
    parts["total"].sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert parts["total"][0] == 0
    for row, count in enumerate(counts):
        assert torch.count_nonzero(logits.grad[row, count:]) == 0


def test_ten_stage_c_uses_ce_and_nine_cdf_boundaries():
    # All ten native stages participate in both terms and their gradients.
    values = torch.tensor([-.8, .2, .7, -.3, .4, .1, -.5, .8, -.2, .6])
    logits = values.clone().requires_grad_()
    reference = values.clone().requires_grad_()
    target = torch.tensor(4)
    weight = 1.3
    parts = stage_loss_components(logits, target, cdf_weight=weight)
    probabilities = reference.softmax(-1)
    expected_ce = -probabilities[4].log()
    expected_cdf = weight * sum((probabilities[:j + 1].sum() - int(j >= 4)).square()
                                for j in range(9)) / 9
    torch.testing.assert_close(parts["ce"], expected_ce)
    torch.testing.assert_close(parts["cdf"], expected_cdf)
    parts["total"].backward()
    (expected_ce + expected_cdf).backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert (logits.grad[[0, 1, 2, 6, 7, 8, 9]].abs() > 0).all()
    # A K=3 sub-block objective would define a different loss: reject that
    # accidental implementation with this nonuniform ten-stage case.
    subset = stage_loss_components(values[3:6], torch.tensor(1), cdf_weight=weight)
    assert not torch.isclose(parts["ce"], subset["ce"])
    assert not torch.isclose(parts["cdf"], subset["cdf"])


def test_ten_native_stages_have_nonzero_supervision():
    example = SimpleNamespace(role="C", num_stages=10, stage_ids=(6,) * 8,
                              targets=(6,) * 8)
    logits = torch.zeros(1, 8, 26)
    logits[..., 10:] = -torch.inf
    logits.requires_grad_()
    output = {"stage_indices": [0], "stage_logits": logits,
              "progress_indices": [], "progress_logits": None}
    parts = role_loss_components(output, [example])["C"]
    torch.testing.assert_close(parts["ce"], torch.tensor(sum(FRAME_WEIGHTS) * math.log(10)))
    assert parts["cdf"] > 0
    parts["total"].backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[0, :, :10].abs().sum(0) > 0).all()
    assert torch.count_nonzero(logits.grad[..., 10:]) == 0


def test_mixed_stage_counts_match_unpadded_ce_cdf_values_and_gradients():
    generator = torch.Generator().manual_seed(247)
    counts = (10, 3, 8)
    examples = [SimpleNamespace(role="C", num_stages=k,
                                stage_ids=tuple(slot % k for slot in range(8)),
                                targets=tuple(slot % k for slot in range(8)),
                                frame_weight_order=tuple(reversed(range(8))))
                for k in counts]
    values = torch.randn(3, 8, 26, generator=generator)
    for row, count in enumerate(counts):
        values[row, :, count:] = -torch.inf
    padded = values.clone().requires_grad_()
    unpadded = [values[row, :, :count].clone().requires_grad_() for row, count in enumerate(counts)]
    output = {"stage_indices": [0, 1, 2], "stage_logits": padded,
              "progress_indices": [], "progress_logits": None}
    actual = role_loss_components(output, examples, loss_config={"stage": {"cdf_weight": .7}})["C"]
    weights = torch.tensor(tuple(reversed(FRAME_WEIGHTS)))
    references = [stage_loss_components(logits, torch.tensor(example.targets), cdf_weight=.7)
                  for logits, example in zip(unpadded, examples, strict=True)]
    expected = {key: sum((parts[key] * weights).sum() for parts in references)
                for key in actual}
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    actual["total"].backward()
    expected["total"].backward()
    assert torch.isfinite(padded.grad).all()
    for row, (count, reference) in enumerate(zip(counts, unpadded, strict=True)):
        torch.testing.assert_close(padded.grad[row, :, :count], reference.grad)
        assert torch.count_nonzero(padded.grad[row, :, count:]) == 0


@pytest.mark.parametrize("value", [0.0, 0.04, 0.3, 0.5, 1.0])
def test_progress_c51_target_is_stationary_for_ce_cdf(value):
    targets = torch.tensor([value])
    distribution = continuous_to_c51(targets)
    # Exact sparse target probabilities, including genuine -inf logits.
    logits = distribution.log().requires_grad_()
    parts = progress_loss_components(logits, targets)
    assert torch.isfinite(parts["total"]).all()
    torch.testing.assert_close(parts["cdf"], torch.zeros(1), atol=1e-14, rtol=0)
    parts["total"].sum().backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad), atol=6e-8, rtol=0)


@pytest.mark.parametrize("head", ["stage", "progress"])
def test_cdf_penalizes_distant_errors_with_identical_target_probability(head):
    probabilities = torch.zeros(2, 10)
    probabilities[:, 0] = 0.5
    probabilities[0, 1] = 0.5
    probabilities[1, 5] = 0.5
    if head == "stage":
        # Stage validity comes from finite logits, so use negligible mass elsewhere.
        probabilities = probabilities.clamp_min(1e-20)
        parts = stage_loss_components(probabilities.log(), torch.tensor([0, 0]))
    else:
        parts = progress_loss_components(probabilities.log(), torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(parts["ce"][0], parts["ce"][1])
    assert parts["cdf"][1] > parts["cdf"][0]
    torch.testing.assert_close(parts["cdf"], torch.tensor([0.25 / 9, 1.25 / 9]), atol=2e-8, rtol=1e-6)


def test_components_sum_and_role_dispatch_preserve_frame_weights():
    frame_weights = (0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    examples = [SimpleNamespace(role=role, targets=[0] * 8) for role in "ABCD"]
    # Slot 5 alone determines this hand-computed reduction.
    for index, value in ((0, 0.5), (1, 1.0), (2, 1), (3, 0.0)):
        examples[index].targets[5] = value
    output = {"progress_indices": [0, 1, 3], "progress_logits": torch.zeros(3, 8, 10, requires_grad=True),
              "stage_indices": [2], "stage_logits": torch.zeros(1, 8, 3, requires_grad=True)}
    config = {"stage": {"cdf_weight": 0.5},
              "progress": {"cdf_weight": 0.7, "ceiling_weight": 3.0}}
    parts = role_loss_components(output, examples, frame_weights, config)
    numerators = role_numerators(output, examples, frame_weights, config)
    for role, components in parts.items():
        assert set(components) == {"total", "ce", "cdf", "ceiling"}
        torch.testing.assert_close(components["total"], sum(components[name] for name in ("ce", "cdf", "ceiling")))
        torch.testing.assert_close(numerators[role], components["total"])
    torch.testing.assert_close(parts["C"]["ce"], torch.tensor(2 * math.log(3)))
    for role in "ABD":
        torch.testing.assert_close(parts[role]["ce"], torch.tensor(2 * math.log(10)))
    torch.testing.assert_close(parts["D"]["ceiling"], torch.tensor(2 * 3 * (0.5 - 1 / 9)))
    assert parts["A"]["ceiling"] == parts["B"]["ceiling"] == parts["C"]["ceiling"] == 0
    torch.testing.assert_close(parts["C"]["cdf"], torch.tensor(2 * 0.5 / 9))
    sum(numerators.values()).backward()
    for kind in ("stage", "progress"):
        gradient = output[kind + "_logits"].grad
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[:, :5]) == torch.count_nonzero(gradient[:, 6:]) == 0
        assert gradient[:, 5].abs().sum() > 0


def test_occurrence_weight_permutations_preserve_each_role_objective_and_gradient():
    generator = torch.Generator().manual_seed(162)
    examples = [SimpleNamespace(role=role, targets=tuple((slot % 3) if role == "C" else slot / 7
                                                       for slot in range(8))) for role in "ABCD"]
    original = {"progress_indices": [0, 1, 3], "stage_indices": [2],
                "progress_logits": torch.randn(3, 8, 10, generator=generator, requires_grad=True),
                "stage_logits": torch.randn(1, 8, 3, generator=generator, requires_grad=True)}
    permutations = ((7, 6, 5, 4, 3, 2, 1, 0), (0, 1, 2, 7, 6, 5, 4, 3),
                    (0, 1, 2, 7, 6, 5, 4, 3), (7, 6, 5, 4, 3, 2, 1, 0))
    reordered_examples = [SimpleNamespace(role=example.role,
                                         targets=tuple(example.targets[index] for index in permutation),
                                         frame_weight_order=permutation)
                          for example, permutation in zip(examples, permutations, strict=True)]
    reordered = {key: original[key] for key in ("progress_indices", "stage_indices")}
    for kind in ("progress", "stage"):
        reordered[kind + "_logits"] = torch.stack([
            original[kind + "_logits"][row, list(permutations[index])].detach()
            for row, index in enumerate(original[kind + "_indices"])]).requires_grad_()
    expected = role_loss_components(original, examples)
    actual = role_loss_components(reordered, reordered_examples)
    for role in "ABCD":
        for component in ("total", "ce", "cdf", "ceiling"):
            torch.testing.assert_close(actual[role][component], expected[role][component], rtol=2e-7, atol=2e-6)
    sum(row["total"] for row in expected.values()).backward()
    sum(row["total"] for row in actual.values()).backward()
    for kind in ("progress", "stage"):
        for row, index in enumerate(original[kind + "_indices"]):
            torch.testing.assert_close(reordered[kind + "_logits"].grad[row],
                                       original[kind + "_logits"].grad[row, list(permutations[index])])
    # Moving only observations/labels changes the objective with nonuniform
    # weights; this fixture distinguishes occurrence weights from slot weights.
    naive = role_loss_components(reordered, [SimpleNamespace(role=e.role, targets=e.targets)
                                            for e in reordered_examples])
    assert any(abs(naive[role]["total"].item() - expected[role]["total"].item()) > .01 for role in "ABCD")


def test_identity_occurrence_weights_are_exactly_legacy_reduction():
    logits = torch.arange(80, dtype=torch.float32).reshape(1, 8, 10) / 37
    example = SimpleNamespace(role="B", targets=tuple(slot / 7 for slot in range(8)),
                              frame_weight_order=tuple(range(8)))
    output = {"progress_indices": [0], "progress_logits": logits,
              "stage_indices": [], "stage_logits": None}
    parts = progress_loss_components(logits, torch.tensor([example.targets]))
    observed = role_loss_components(output, [example])
    for name, values in parts.items():
        assert torch.equal(observed["B"][name], (values * torch.tensor(FRAME_WEIGHTS)).sum(-1)[0])


@pytest.mark.parametrize("order", [(0,) * 8, tuple(range(7)), (*range(7), 8),
                                    (False, *range(1, 8)), None])
def test_invalid_occurrence_weights_cannot_change_supervision_mass(order):
    example = SimpleNamespace(role="B", targets=(.5,) * 8, frame_weight_order=order)
    output = {"progress_indices": [0], "progress_logits": torch.zeros(1, 8, 10),
              "stage_indices": [], "stage_logits": None}
    with pytest.raises(ValueError, match="frame_weight_order"):
        role_loss_components(output, [example])


def test_ce_cdf_defaults_and_ceiling_is_disabled():
    logits = torch.zeros(1, 10)
    parts = progress_loss_components(logits, torch.tensor([0.0]))
    assert parts["ceiling"] == 0
    assert parts["cdf"] > 0
    torch.testing.assert_close(progress_loss(logits, torch.tensor([0.0])), parts["total"])
    stage_parts = stage_loss_components(torch.zeros(1, 3), torch.tensor([0]))
    assert stage_parts["ceiling"] == 0
    assert stage_parts["cdf"] > 0
    torch.testing.assert_close(stage_loss(torch.zeros(1, 3), torch.tensor([0])), stage_parts["total"])


@pytest.mark.parametrize("logits,target", [
    ([0.0, -torch.inf, 0.0], 0),
    ([-torch.inf, -torch.inf], 0),
    ([0.0, -torch.inf], 1),
    ([0.0, torch.inf], 0),
    ([0.0, torch.nan], 0),
])
def test_stage_rejects_invalid_masks_and_targets(logits, target):
    with pytest.raises(ValueError, match="finite prefix"):
        stage_loss(torch.tensor([logits]), torch.tensor([target]))


@pytest.mark.parametrize("kwargs", [
    {"cdf_weight": -1}, {"cdf_weight": float("nan")},
    {"cdf_weight": float("inf")}, {"cdf_weight": True}, {"ceiling_weight": -0.1},
])
def test_progress_rejects_invalid_loss_weights(kwargs):
    with pytest.raises(ValueError):
        progress_loss(torch.zeros(1, 10), torch.tensor([0.0]), **kwargs)


@pytest.mark.parametrize("accumulation", [2, 4])
@pytest.mark.parametrize("world_size", [1, 2])
def test_independent_role_normalization_matches_single_global_batch(accumulation, world_size):
    counts = window_role_counts({"A": 2, "B": 4, "C": 1, "D": 1}, accumulation)
    mass = sum(FRAME_WEIGHTS)
    denominators = {role: count * mass * world_size for role, count in counts.items()}
    parameter = torch.tensor(2.0, requires_grad=True)
    role_weights = {"A": 1.0, "B": 2.0, "C": 0.25, "D": 3.0}
    # Each rank accumulates its own portion; DDP then takes the rank mean.
    numerators = {role: parameter.square() * count * mass for role, count in counts.items()}
    loss = normalized_loss(numerators, denominators, role_weights, world_size)
    loss.backward()
    assert loss.item() == pytest.approx(4 * sum(role_weights.values()))
    assert parameter.grad.item() == pytest.approx(4 * sum(role_weights.values()))


def test_d_rejects_incomplete_pairs():
    with pytest.raises(ValueError):
        window_role_counts({"A": 2, "B": 2, "C": 1, "D": 1}, 3)
