import copy
from dataclasses import replace
import json
import math

import pytest
import torch

from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.moe_experiment_comparison import run_moe_document_comparison
from ai_journey.moe_report import verify_moe_report
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan


@pytest.fixture(scope="module")
def measured():
    return run_moe_document_comparison(plan(seeds=(53, 54)), TRAIN, HELDOUT)


def test_offline_verification_recomputes_actual_evidence_without_model_execution(
    measured, monkeypatch
):
    import ai_journey.training_models as models

    monkeypatch.setattr(
        models,
        "build_training_model",
        lambda *a, **k: pytest.fail("offline verification must not allocate models"),
    )
    actual = json.loads(json.dumps(measured))
    assert verify_moe_report(actual) == plan(seeds=(53, 54))
    assert "cannot authenticate" in actual["verification_scope"]


@pytest.mark.parametrize(
    "mutation",
    [
        "runtime-alias",
        "step-alias",
        "parameter-alias",
        "objective",
        "dense-aux",
        "flag",
        "count",
        "loss",
        "mask",
        "summary",
        "shared",
        "clip-alias",
        "source",
        "unknown",
        "probability-alias",
        "route-shape",
    ],
)
def test_resealed_impossible_or_aliased_evidence_is_rejected(measured, mutation):
    report = copy.deepcopy(measured)
    trial = report["trials"][0]
    sparse = trial["arms"][1]
    row = sparse["trace"][0]
    if mutation == "runtime-alias":
        report["runtime"]["threads"] = True
    elif mutation == "step-alias":
        row["step"] = True
    elif mutation == "parameter-alias":
        sparse["parameter_count"] = float(sparse["parameter_count"])
    elif mutation == "objective":
        row["objective"] += 0.1
    elif mutation == "dense-aux":
        trial["arms"][0]["trace"][0]["balance_loss"] = 1.0
    elif mutation == "flag":
        row["routing"][0]["accepted_routes"][0][0] = False
    elif mutation == "count":
        row["routing"][0]["accepted_counts"][0] += 1
    elif mutation == "loss":
        sparse["heldout"]["batches"][0]["token_nll"][0] += 0.1
    elif mutation == "mask":
        sparse["heldout"]["batches"][0]["token_mask"][0][0] = False
    elif mutation == "summary":
        report["summary"]["paired_deltas"][0]["per_seed"][0]["delta_nll"] += 0.01
    elif mutation == "shared":
        sparse["initial_shared_sha256"] = "0" * 64
    elif mutation == "clip-alias":
        row["clipped"] = int(row["clipped"])
    elif mutation == "source":
        report["trials"][1]["train_sha256"] = "0" * 64
    elif mutation == "unknown":
        sparse["heldout"]["extra_note"] = "not an executable field"
    elif mutation == "probability-alias":
        row["routing"][0]["probability_sums"][0] = 0
    else:
        row["routing"] = ["invalid"]
    report["report_sha256"] = decoder_comparison_digest(report)
    with pytest.raises(ValueError):
        verify_moe_report(report)


def test_stale_seal_is_rejected_before_accepting_any_analytical_equality(measured):
    report = copy.deepcopy(measured)
    report["runtime"]["threads"] = 1.0
    with pytest.raises(ValueError, match="digest"):
        verify_moe_report(report)


def test_offline_contract_retains_recorded_runtime_while_replay_is_separate(measured):
    report = copy.deepcopy(measured)
    report["runtime"]["torch_version"] = "recorded-other-version"
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_moe_report(report)


@pytest.mark.parametrize(
    "mutation", ["unknown", "moved-group", "omitted", "replacement"]
)
def test_resealed_optimizer_names_must_bind_the_declared_architecture(
    measured, monkeypatch, mutation
):
    import ai_journey.training_models as models

    report = copy.deepcopy(measured)
    groups = report["trials"][0]["arms"][1]["optimizer_groups"]
    original_count = groups["trainable_parameters"]
    if mutation == "unknown":
        groups["decay_names"][1] = "nonexistent.parameter"
    elif mutation == "moved-group":
        groups["no_decay_names"].append(groups["decay_names"].pop(1))
    elif mutation == "omitted":
        groups["decay_names"].pop(1)
    else:
        # A real vector from this architecture still cannot replace a matrix.
        groups["decay_names"][1] = groups["no_decay_names"].pop(0)
    assert groups["trainable_parameters"] == original_count
    report["report_sha256"] = decoder_comparison_digest(report)
    monkeypatch.setattr(
        models,
        "build_training_model",
        lambda *a, **kw: pytest.fail("offline verification must not allocate a model"),
    )
    with pytest.raises(ValueError, match="optimizer"):
        verify_moe_report(report)


@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
@pytest.mark.parametrize("position", ["learned", "rope"])
@pytest.mark.parametrize("layers", [None, (0,), (1,), (0, 1)])
def test_allocation_free_optimizer_schema_matches_independent_actual_models(
    norm, mlp, position, layers
):
    from ai_journey.decoder_variant_optimizer import decoder_optimizer_groups
    from ai_journey.moe_experiment import build_matched_moe_experiment_model
    from ai_journey.moe_report import _expected_optimizer_names, _optimizer_groups

    config = plan().arms[int(layers is not None)].config
    config = replace(
        config,
        gpt2=replace(config.gpt2, layer_count=2),
        norm_kind=norm,
        mlp_kind=mlp,
        position_kind=position,
    )
    if layers is not None:
        config = replace(config, moe_layers=layers)
    model = build_matched_moe_experiment_model(config, 53)
    actual = json.loads(json.dumps(decoder_optimizer_groups(model)))
    expected = _expected_optimizer_names(config)
    assert {key: actual[key] for key in expected} == expected
    _optimizer_groups(actual, config, model.parameter_count)


@pytest.mark.parametrize("arm_index", [0, 1])
def test_resealed_gradient_disappearance_is_not_a_valid_clip_transition(
    measured, arm_index
):
    report = copy.deepcopy(measured)
    row = report["trials"][0]["arms"][arm_index]["trace"][0]
    assert row["gradient_norm_before_clip"] > 0.1
    row["gradient_norm_after_clip"] = 0.0
    report["report_sha256"] = decoder_comparison_digest(report)
    with pytest.raises(ValueError, match="clipping"):
        verify_moe_report(report)


@pytest.mark.parametrize("scale", [0.0, 1e-30, 1e-12, 1e-8, 0.001, 1.0, 10.0, 1e6])
@pytest.mark.parametrize("maximum", [1e-9, 1e-4, 1.0, 1000.0])
def test_norm_transition_matches_actual_float32_clip_grad_norm_oracles(scale, maximum):
    from ai_journey.moe_report import _gradient_clip_transition

    generator = torch.Generator().manual_seed(53)
    parameters = [torch.nn.Parameter(torch.zeros(shape)) for shape in ((17,), (13, 3))]
    for parameter in parameters:
        parameter.grad = torch.randn(parameter.shape, generator=generator) * scale
    precise_before = math.sqrt(
        math.fsum(float(p.grad.double().square().sum()) for p in parameters)
    )
    returned_before = float(
        torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True)
    )
    after = math.sqrt(
        math.fsum(float(p.grad.double().square().sum()) for p in parameters)
    )
    # Dense trials record a double-reduced norm; MoE/GQA record the returned
    # float32 reduction. Both must preserve valid measured artifact semantics.
    _gradient_clip_transition(precise_before, after, maximum)
    _gradient_clip_transition(returned_before, after, maximum)


def test_transition_rejects_vanished_small_control_and_underreported_clipping():
    from ai_journey.moe_report import _gradient_clip_transition

    for before, after, maximum in (
        (1.0, 0.0, 1e-9),
        (0.1, 0.01, 1.0),
        (10.0, 0.1, 1.0),
    ):
        with pytest.raises(ValueError, match="clipping"):
            _gradient_clip_transition(before, after, maximum)
