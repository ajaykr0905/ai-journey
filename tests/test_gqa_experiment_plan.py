from dataclasses import replace
import json

import pytest

from ai_journey.gqa_cli import default_gqa_plan
from ai_journey.gqa_experiment_plan import (
    gqa_experiment_plan_from_mapping,
    require_gqa_experiment_budget,
)


def test_plan_roundtrip_and_whole_job_counters_charge_every_unequal_arm_and_seed():
    plan = replace(default_gqa_plan(), seeds=(53, 54))
    assert (
        gqa_experiment_plan_from_mapping(json.loads(json.dumps(plan.to_mapping())))
        == plan
    )
    budget = require_gqa_experiment_budget(plan, evaluation_windows=7)
    assert (
        budget["model_parameters"][0]
        > budget["model_parameters"][1]
        > budget["model_parameters"][2]
    )
    single = require_gqa_experiment_budget(
        replace(plan, seeds=(53,)), evaluation_windows=7
    )
    assert budget["work_units"] == 2 * single["work_units"]
    assert budget["trial_count"] == 6
    audit = require_gqa_experiment_budget(
        plan, evaluation_windows=7, cache_prefix_lengths=(4, 8)
    )
    assert audit["work_units"] == budget["work_units"] + audit["cache_audit_work_units"]
    assert audit["cache_audit_work_units"] > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("kv_heads", (2, 1)),
        ("kv_heads", (4, 4)),
        ("kv_heads", (4, 3)),
        ("kv_heads", (4, True)),
        ("seeds", (True,)),
        ("seeds", (53, 53)),
        ("steps", 17),
        ("batch_size", True),
        ("learning_rate", float("nan")),
    ],
)
def test_invalid_or_aliased_controls_reject(field, value):
    with pytest.raises(ValueError):
        replace(default_gqa_plan(), **{field: value})


def test_large_joint_job_and_unbounded_cache_work_reject():
    plan = replace(default_gqa_plan(), seeds=(1, 2, 3, 4), steps=16, batch_size=8)
    with pytest.raises(ValueError, match="work_units"):
        require_gqa_experiment_budget(plan, evaluation_windows=8192)
    for lengths in ((True,), (9,), (1, 1, 1, 1, 1)):
        with pytest.raises(ValueError, match="cache prefix"):
            require_gqa_experiment_budget(
                default_gqa_plan(), evaluation_windows=2, cache_prefix_lengths=lengths
            )


def test_missing_plan_or_optimizer_fields_do_not_get_silent_defaults():
    payload = default_gqa_plan().to_mapping()
    payload["optimizer"].pop("epsilon")
    with pytest.raises(ValueError, match="optimizer fields"):
        gqa_experiment_plan_from_mapping(payload)
