import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_faults import FaultPlan, InjectedFault


def test_fault_plan_resolves_exact_declared_coordinate():
    plan = FaultPlan([InjectedFault("t", 1, 1, "transient")])
    assert plan.kind_for("t", 1, 1) == "transient"
    assert plan.kind_for("t", 1, 2) is None


def test_fault_plan_rejects_duplicates_and_unknown_kinds():
    fault = InjectedFault("t", 0, 1, "transient")
    with pytest.raises(VerifiableRewardError, match="unique"):
        FaultPlan([fault, fault])
    with pytest.raises(VerifiableRewardError, match="kind"):
        FaultPlan([InjectedFault("t", 0, 1, "crash")])
