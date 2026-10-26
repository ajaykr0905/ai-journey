from dataclasses import replace
import json

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.moe_config import MoEConfig
from ai_journey import moe_resources as resources


def arms():
    base = GPT2Config(
        vocab_size=256, context_length=4, embedding_dim=4, head_count=2, layer_count=1
    )
    return DecoderVariantConfig(base), MoEConfig(base)


def controls(**changes):
    return dict(seeds=3, steps=2, batch_size=2, evaluation_windows=5, **changes)


def test_complete_job_has_an_independent_small_model_arithmetic_oracle():
    result = resources.estimate_moe_job(arms(), **controls())
    assert result["model_parameters"] == [1292, 1752]
    assert result["active_parameters"] == [1292, 1456]
    assert result["forward_backward_work_units"] == 3 * (1292 + 1456) * 4 * (
        12 + 4 + 10
    )
    assert result["optimizer_work_units"] == 3 * 10 * (1292 + 1752) * 2
    assert result["work_units"] == (
        result["forward_backward_work_units"] + result["optimizer_work_units"]
    )
    assert result["trial_count"] == 6
    assert result["peak_route_elements"] == 5 * 4 * 4
    assert result["peak_attention_elements"] == 5 * 2 * 4**2
    assert json.loads(json.dumps(result)) == result
    assert resources.require_moe_job_budget(arms(), **controls()) == result


def test_repeated_architectures_and_seeds_are_charged_without_deduplication():
    config = arms()[1]
    single = resources.estimate_moe_job(
        (config,), seeds=1, steps=8, batch_size=8, evaluation_windows=16
    )
    whole = resources.estimate_moe_job(
        (config,) * 16, seeds=16, steps=8, batch_size=8, evaluation_windows=16
    )
    assert whole["work_units"] == 256 * single["work_units"]
    resources.require_moe_job_budget(
        (config,), seeds=1, steps=8, batch_size=8, evaluation_windows=16
    )
    with pytest.raises(ValueError, match="work_units"):
        resources.require_moe_job_budget(
            (config,) * 16, seeds=16, steps=8, batch_size=8, evaluation_windows=16
        )


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("seeds", True),
        ("seeds", 0),
        ("seeds", 17),
        ("steps", 1.0),
        ("steps", 65),
        ("batch_size", 33),
        ("evaluation_windows", 0),
        ("evaluation_windows", 8193),
        ("evaluation_batch_size", False),
        ("evaluation_batch_size", 33),
    ],
)
def test_controls_have_bounded_exact_integer_admission(name, invalid):
    values = controls()
    values[name] = invalid
    with pytest.raises(ValueError, match=name):
        resources.estimate_moe_job(arms(), **values)


@pytest.mark.parametrize("invalid", [(), [], (None,), arms() * 9])
def test_arm_admission_is_bounded_and_explicit(invalid):
    with pytest.raises((TypeError, ValueError)):
        resources.estimate_moe_job(invalid, **controls())


@pytest.mark.parametrize(
    ("constant", "field"),
    [
        ("MAX_JOB_MODEL_PARAMETERS", "peak_model_parameters"),
        ("MAX_JOB_WORK_UNITS", "work_units"),
        ("MAX_JOB_ROUTE_ELEMENTS", "peak_route_elements"),
        ("MAX_JOB_ATTENTION_ELEMENTS", "peak_attention_elements"),
    ],
)
def test_independent_budgets_admit_exact_boundary_and_reject_above_it(
    constant, field, monkeypatch
):
    estimate = resources.estimate_moe_job(arms(), **controls())
    monkeypatch.setattr(resources, constant, estimate[field])
    resources.require_moe_job_budget(arms(), **controls())
    monkeypatch.setattr(resources, constant, estimate[field] - 1)
    with pytest.raises(ValueError, match=field):
        resources.require_moe_job_budget(arms(), **controls())


def test_estimation_and_large_model_rejection_do_not_allocate_tensors(monkeypatch):
    dense, sparse = arms()
    oversized = replace(sparse, gpt2=replace(sparse.gpt2, embedding_dim=4096))

    def forbidden(*args, **kwargs):
        raise AssertionError("resource admission allocated a tensor")

    monkeypatch.setattr(torch, "empty", forbidden)
    assert resources.estimate_moe_job((dense, sparse), **controls())["work_units"] > 0
    with pytest.raises(ValueError, match="peak_model_parameters"):
        resources.require_moe_job_budget((oversized,), **controls())


def test_dense_optimizer_cost_tracks_stored_not_active_experts():
    sparse = arms()[1]
    wider = replace(sparse, expert_count=8)
    small = resources.estimate_moe_job((sparse,), **controls())
    large = resources.estimate_moe_job((wider,), **controls())
    assert large["optimizer_work_units"] == 3 * 10 * large["model_parameters"][0] * 2
    assert large["optimizer_work_units"] > small["optimizer_work_units"]
    assert large["active_parameters"][0] < large["model_parameters"][0]
