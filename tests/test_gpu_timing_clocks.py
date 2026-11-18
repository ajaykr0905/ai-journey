import math

import pytest

from ai_journey.gpu_timing_clocks import (
    CLOCK_DEFINITIONS,
    CLOCK_DOMAINS,
    CLOCKS,
    DEVICE_DOMAIN,
    DISPATCH,
    DOMAIN_DEFINITIONS,
    EVENT,
    HOST_DOMAIN,
    SYNCHRONIZED,
    ClockSamples,
    GpuTimingError,
    cross_domain_difference,
    event_ns,
    require_same_domain,
)
from ai_journey.matmul_metal_mapping import MatmulMetalError


def test_each_clock_names_one_domain_and_one_definition():
    assert CLOCKS == (EVENT, SYNCHRONIZED, DISPATCH)
    assert CLOCK_DOMAINS == {
        EVENT: DEVICE_DOMAIN,
        SYNCHRONIZED: HOST_DOMAIN,
        DISPATCH: HOST_DOMAIN,
    }
    assert set(CLOCK_DEFINITIONS) == set(CLOCKS)
    assert set(DOMAIN_DEFINITIONS) == {DEVICE_DOMAIN, HOST_DOMAIN}
    assert "no synchronize between the readings" in CLOCK_DEFINITIONS[DISPATCH]
    assert "round(ms * 1e6)" in CLOCK_DEFINITIONS[EVENT]
    assert issubclass(GpuTimingError, MatmulMetalError)


@pytest.mark.parametrize(
    ("milliseconds", "expected"),
    [(0.000001, 1), (0.0225, 22500), (1.2345678, 1234568), (60000.0, 60 * 10**9)],
)
def test_event_ms_round_to_integer_nanoseconds(milliseconds, expected):
    assert event_ns(milliseconds) == expected


@pytest.mark.parametrize(
    "milliseconds", [0.0, -1.0, 4e-7, math.nan, math.inf, 60000.5, 1, True, "1.0"]
)
def test_event_ms_outside_the_contract_are_refused(milliseconds):
    with pytest.raises(GpuTimingError):
        event_ns(milliseconds)


def test_host_clocks_share_a_domain_and_the_event_clock_does_not():
    assert require_same_domain(SYNCHRONIZED, DISPATCH) == HOST_DOMAIN
    assert require_same_domain(EVENT, EVENT) == DEVICE_DOMAIN
    with pytest.raises(GpuTimingError, match="cross_domain_difference"):
        require_same_domain(SYNCHRONIZED, EVENT)
    with pytest.raises(GpuTimingError, match="clock must be one of"):
        require_same_domain("wall", DISPATCH)


def test_a_cross_domain_difference_says_so():
    record = cross_domain_difference(90_000, 60_000)

    assert record == {
        "value_ns": 30_000,
        "cross_domain": True,
        "minuend": SYNCHRONIZED,
        "subtrahend": EVENT,
    }
    assert cross_domain_difference(1, 5)["value_ns"] == -4
    for bad in ((0, 1), (1, 0), (1.0, 1), (True, 1)):
        with pytest.raises(GpuTimingError):
            cross_domain_difference(*bad)


def test_event_samples_keep_raw_ms_beside_their_nanoseconds():
    samples = ClockSamples.from_event_ms((0.5, 0.25, 0.75, 0.125))

    assert samples.samples_ns == (500_000, 250_000, 750_000, 125_000)
    assert samples.domain == DEVICE_DOMAIN
    mapping = samples.to_mapping()
    assert list(mapping) == ["clock", "domain", "samples_ns", "elapsed_ms", "timing"]
    assert mapping["elapsed_ms"] == [0.5, 0.25, 0.75, 0.125]
    assert mapping["timing"]["median_ns"] == 250_000
    assert mapping["timing"]["iqr_ns"] == 500_000 - 125_000


def test_host_samples_carry_no_raw_ms_and_use_nearest_rank_statistics():
    samples = ClockSamples(SYNCHRONIZED, (40, 10, 30, 20, 50))

    assert samples.domain == HOST_DOMAIN
    assert "elapsed_ms" not in samples.to_mapping()
    summary = samples.summary
    assert (summary.p25_ns, summary.median_ns, summary.p75_ns) == (20, 30, 40)


@pytest.mark.parametrize(
    ("clock", "samples", "raw", "message"),
    [
        ("wall", (1,), None, "clock must be one of"),
        (SYNCHRONIZED, (), None, "nonempty"),
        (SYNCHRONIZED, [1], None, "nonempty"),
        (DISPATCH, (1,), (0.000001,), "only the event clock"),
        (EVENT, (1,), None, "one raw ms value"),
        (EVENT, (1, 2), (0.000001,), "one raw ms value"),
        (EVENT, (2,), (0.000001,), "round"),
    ],
)
def test_inconsistent_samples_are_refused(clock, samples, raw, message):
    with pytest.raises(GpuTimingError, match=message):
        ClockSamples(clock, samples, raw)


def test_nonpositive_host_samples_are_refused():
    with pytest.raises(ValueError, match="positive integer"):
        ClockSamples(DISPATCH, (1, 0))
