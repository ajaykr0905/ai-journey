import pytest

from ai_journey.memory_hierarchy_access import PATTERN_NAMES, AccessMeasurement
from ai_journey.memory_hierarchy_config import LatencyControls
from ai_journey.memory_hierarchy_littles_law import LAW, littles_law_section
from ai_journey.memory_hierarchy_stream import MemoryHierarchyError
from ai_journey.memory_hierarchy_sweep import StreamMeasurement
from ai_journey.memory_hierarchy_timing import BatchTiming

LATENCY = LatencyControls(100, "Stated test latency, not a measurement.")


def timing(median, calls=1):
    return BatchTiming(calls, 0, 3, (median - 1, median, median + 1), "injected")


def stream():
    return (
        StreamMeasurement("triad", 1024, "float64", timing(400, calls=4)),
        StreamMeasurement("copy", 4096, "float64", timing(800)),
        StreamMeasurement("triad", 4096, "float64", timing(2000)),
    )


def access():
    return tuple(
        AccessMeasurement(name, 8192, "float64", timing(median))
        for name, median in zip(PATTERN_NAMES, (2000, 16000, 56000, 92000))
    )


def test_rows_multiply_measured_rates_by_the_stated_latency():
    section = littles_law_section(LATENCY, stream(), access())
    bytes_per_ns = 5 * 4096 * 8 / 2000

    assert section["law"] == LAW
    assert section["stated_latency_ns"] == 100
    assert section["latency_note"] == "Stated test latency, not a measurement."
    assert section["latency_measured"] is False
    assert section["bandwidth"] == {
        "source": "stream_triad_largest_working_set",
        "elements": 4096,
        "measured_bytes_per_ns": bytes_per_ns,
        "derived_bytes_in_flight": bytes_per_ns * 100,
    }
    assert section["access_patterns"] == [
        {
            "pattern": name,
            "measured_elements_per_ns": 8192 / median,
            "derived_elements_in_flight": 8192 / median * 100,
        }
        for name, median in zip(PATTERN_NAMES, (2000, 16000, 56000, 92000))
    ]
    random = section["access_patterns"][-1]
    assert random["derived_elements_in_flight"] == pytest.approx(8.904, abs=1e-3)


def test_rows_need_typed_inputs_triad_and_every_pattern_in_order():
    with pytest.raises(MemoryHierarchyError, match="needs triad"):
        littles_law_section(LATENCY, stream()[1:2], access())
    with pytest.raises(MemoryHierarchyError, match="every pattern in order"):
        littles_law_section(LATENCY, stream(), access()[::-1])
    with pytest.raises(TypeError, match="LatencyControls"):
        littles_law_section(100, stream(), access())
    with pytest.raises(TypeError, match="StreamMeasurement"):
        littles_law_section(LATENCY, list(stream()), access())
    with pytest.raises(TypeError, match="AccessMeasurement"):
        littles_law_section(LATENCY, stream(), stream())
