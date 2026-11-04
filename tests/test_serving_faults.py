import pytest

from ai_journey.serving_faults import FaultEvent, FaultKind, ServingFaultError


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


def test_fault_event_has_an_explicit_bounded_interval():
    event = FaultEvent("fault-1", FaultKind.PROCESS_KILL, "worker-0", 100, 25)

    assert event.end_ns == 125


def test_fault_event_rejects_zero_duration():
    with pytest.raises(ServingFaultError, match="duration"):
        FaultEvent("fault-1", FaultKind.PROCESS_KILL, "worker-0", 100, 0)
