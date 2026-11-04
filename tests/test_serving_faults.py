import pytest

from ai_journey.serving_faults import FaultKind


def test_fault_kinds_use_a_fixed_low_cardinality_vocabulary():
    assert {kind.value for kind in FaultKind} == {
        "process_kill",
        "engine_restart",
        "network_pause",
        "device_oom",
    }


def test_fault_kind_rejects_unbounded_labels():
    with pytest.raises(ValueError):
        FaultKind("customer-specific-chaos")
