import pytest

from ai_journey.serving_faults import (
    FaultEvent,
    FaultKind,
    ServingFaultError,
    require_nonoverlapping_targets,
    validate_fault_schedule,
)


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


def _fault(fault_id, start_ns, duration_ns=10, target="worker-0"):
    return FaultEvent(fault_id, FaultKind.PROCESS_KILL, target, start_ns, duration_ns)


def test_fault_schedule_accepts_tied_starts_with_unique_ids():
    schedule = (_fault("a", 0), _fault("b", 0, target="worker-1"))

    assert validate_fault_schedule(schedule) == schedule


def test_fault_schedule_rejects_duplicate_ids():
    with pytest.raises(ServingFaultError, match="unique"):
        validate_fault_schedule((_fault("a", 0), _fault("a", 10)))


def test_fault_overlap_is_checked_per_target():
    schedule = (_fault("a", 0, 10), _fault("b", 5, 10, target="worker-1"))

    assert require_nonoverlapping_targets(schedule) == schedule

    with pytest.raises(ServingFaultError, match="overlap"):
        require_nonoverlapping_targets((_fault("a", 0, 10), _fault("b", 5, 10)))
