import copy
from dataclasses import replace
import json

import pytest

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.moe_config import MoEConfig
from ai_journey.moe_experiment_plan import (
    MoEExperimentArm,
    MoEExperimentPlan,
    moe_experiment_plan_from_mapping,
    require_moe_experiment_work,
)


def plan(**kwargs):
    base = GPT2Config(
        256,
        4,
        8,
        2,
        1,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    return MoEExperimentPlan(
        arms=(
            MoEExperimentArm("baseline", DecoderVariantConfig(base)),
            MoEExperimentArm("top2", MoEConfig(base), 0.01, 0.001),
        ),
        **kwargs,
    )


def test_plan_roundtrip_pins_model_capacity_objective_and_workload_controls():
    original = plan(seeds=(53, 54))
    assert (
        moe_experiment_plan_from_mapping(json.loads(json.dumps(original.to_mapping())))
        == original
    )
    assert len(original.fingerprint()) == 64
    changed_arm = replace(original.arms[1], balance_weight=0.02)
    assert (
        replace(original, arms=(original.arms[0], changed_arm)).fingerprint()
        != original.fingerprint()
    )
    changed_arm = replace(
        original.arms[1], config=replace(original.arms[1].config, capacity_factor=0.5)
    )
    assert (
        replace(original, arms=(original.arms[0], changed_arm)).fingerprint()
        != original.fingerprint()
    )


def test_same_architecture_with_distinct_aux_objectives_is_a_real_intervention():
    original = plan()
    changed = replace(original.arms[1], name="no-balance", balance_weight=0.0)
    actual = replace(original, arms=(*original.arms, changed))
    assert moe_experiment_plan_from_mapping(actual.to_mapping()) == actual
    with pytest.raises(ValueError, match="distinct interventions"):
        replace(
            original, arms=(*original.arms, replace(original.arms[1], name="duplicate"))
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", (True,)),
        ("seeds", (53, 53)),
        ("steps", True),
        ("steps", 33),
        ("batch_size", 9),
        ("evaluation_batch_size", 17),
        ("learning_rate", float("nan")),
        ("learning_rate", 10**1000),
        ("max_gradient_norm", 0),
    ],
)
def test_plan_rejects_invalid_unbounded_or_aliased_controls(field, value):
    with pytest.raises(ValueError):
        plan(**{field: value})


def test_plan_rejects_uncontrolled_dense_changes_and_inactive_router_objectives():
    original = plan()
    with pytest.raises(ValueError, match="dense architecture"):
        replace(
            original,
            arms=(
                original.arms[0],
                replace(
                    original.arms[1],
                    config=replace(original.arms[1].config, norm_kind="rmsnorm"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="inactive"):
        replace(original.arms[0], balance_weight=0.01)
    with pytest.raises(ValueError, match="dense baseline"):
        replace(original, arms=tuple(reversed(original.arms)))


@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "arm", "model", "objective", "optimizer"]
)
def test_plan_parser_rejects_unknown_or_incomplete_controls(mutation):
    payload = copy.deepcopy(plan().to_mapping())
    if mutation == "extra":
        payload["silent_control"] = 1
    elif mutation == "missing":
        del payload["evaluation_batch_size"]
    elif mutation == "arm":
        payload["arms"][0]["comment"] = "ignore me"
    elif mutation == "model":
        payload["arms"][1]["model"]["config"]["expert_count"] = True
    elif mutation == "objective":
        payload["arms"][1]["balance_weight"] = True
    else:
        del payload["optimizer"]["weight_decay"]
    with pytest.raises(ValueError):
        moe_experiment_plan_from_mapping(payload)


def test_work_admission_charges_every_seed_arm_and_heldout_pass():
    original = plan()
    first = require_moe_experiment_work(original, evaluation_windows=100)["work_units"]
    paired = require_moe_experiment_work(
        replace(original, seeds=(53, 54)), evaluation_windows=100
    )["work_units"]
    assert paired == 2 * first
    assert (
        require_moe_experiment_work(original, evaluation_windows=101)["work_units"]
        > first
    )
    assert (
        require_moe_experiment_work(
            replace(
                original,
                arms=(
                    *original.arms,
                    replace(original.arms[1], name="other", balance_weight=0),
                ),
            ),
            evaluation_windows=100,
        )["work_units"]
        > first
    )
    with pytest.raises(ValueError, match="work_units"):
        require_moe_experiment_work(
            plan(seeds=(1, 2, 3, 4), steps=32, batch_size=8), evaluation_windows=8192
        )
