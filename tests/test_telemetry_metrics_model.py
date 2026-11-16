import math

import pytest

from ai_journey.telemetry_metrics_model import (
    COUNTER,
    GAUGE,
    HISTOGRAM,
    HistogramValue,
    MetricFamily,
    MetricSet,
    Series,
    TelemetryMetricsError,
    integer_histogram,
    metric_family,
    metric_set,
    require_label_name,
    require_label_value,
    require_metric_name,
    sample_value,
    scaled_bounds,
)

SCOPE = {"scope": "local_cpu_lab"}


def counter(name="lab_attempts_total", samples=None, help_text="Lab attempts."):
    samples = samples or [({"outcome": "ok"} | SCOPE, 3)]
    return metric_family(name, COUNTER, help_text, samples)


def histogram(samples=None, buckets=(0.001, 0.25)):
    value = HistogramValue((1.0, 2.0, 3.0), 0.75)
    return metric_family(
        "lab_latency_seconds",
        HISTOGRAM,
        "Lab latency.",
        samples or [(SCOPE, value)],
        buckets=buckets,
    )


def test_builder_sorts_series_converts_integers_and_derives_label_names():
    family = counter(
        samples=[
            ({"scope": "local_cpu_lab", "outcome": "timeout"}, 1),
            ({"outcome": "ok", "scope": "local_cpu_lab"}, 2**53),
        ]
    )

    assert [item.label_values for item in family.series] == [
        ("ok", "local_cpu_lab"),
        ("timeout", "local_cpu_lab"),
    ]
    assert family.series[0].value == float(2**53)
    assert type(family.series[1].value) is float
    assert family.label_names == ("outcome", "scope")
    assert family.sample_names == ("lab_attempts_total",)
    assert family.series[0].label_map() == {"outcome": "ok", "scope": "local_cpu_lab"}
    assert histogram().sample_names == (
        "lab_latency_seconds_bucket",
        "lab_latency_seconds_sum",
        "lab_latency_seconds_count",
    )


@pytest.mark.parametrize(
    "name", ["", "1abc", "a-b", "rule:name", "a b", "x" * 129, "naïve", None]
)
def test_exported_metric_names_follow_the_data_model_without_colons(name):
    with pytest.raises(TelemetryMetricsError, match="invalid metric name"):
        require_metric_name(name)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("__name__", "invalid label name"),
        ("__internal", "invalid label name"),
        ("9lives", "invalid label name"),
        ("a:b", "invalid label name"),
        ("", "invalid label name"),
        (7, "invalid label name"),
        ("le", "reserved"),
        ("quantile", "reserved"),
        ("job", "reserved"),
        ("instance", "reserved"),
    ],
)
def test_label_names_reject_reserved_and_malformed_names(name, message):
    with pytest.raises(TelemetryMetricsError, match=message):
        require_label_name(name)


def test_label_values_allow_quotes_backslashes_and_line_feeds_only():
    assert require_label_value('a "quoted" \\ value\nnext') == (
        'a "quoted" \\ value\nnext'
    )
    for value in ("", "tab\there", "nul\0", "x" * 257, 3):
        with pytest.raises(TelemetryMetricsError):
            require_label_value(value)


def test_sample_values_are_finite_exact_floats():
    assert sample_value(-0.0) == 0.0 and math.copysign(1, sample_value(-0.0)) == 1
    assert sample_value(-(2**53)) == -float(2**53)
    for value in (2**53 + 1, math.nan, math.inf, -math.inf):
        with pytest.raises(TelemetryMetricsError):
            sample_value(value)
    for value in (True, "1", None):
        with pytest.raises(TypeError):
            sample_value(value)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: counter(name="lab_attempts"), "must end in _total"),
        (
            lambda: metric_family("lab_up_total", GAUGE, "Up.", [(SCOPE, 1)]),
            "must not end in _total",
        ),
        (lambda: counter(samples=[(SCOPE, -1)]), "negative"),
        (lambda: counter(help_text=""), "help text"),
        (lambda: counter(help_text=" padded"), "start or end"),
        (lambda: counter(help_text="bell\a"), "control character"),
        (
            lambda: counter(samples=[(SCOPE, 1), ({"outcome": "ok"} | SCOPE, 1)]),
            "differ in label names",
        ),
        (lambda: counter(samples=[(SCOPE, 1), (SCOPE, 2)]), "without repeats"),
        (lambda: counter(samples=[({"le": "1"}, 1)]), "reserved"),
        (lambda: counter(samples=[({"scope": ""}, 1)]), "label value"),
        (
            lambda: metric_family("lab_x", "summary", "Summary.", []),
            "unsupported metric type",
        ),
        (
            lambda: metric_family(
                "lab_x", GAUGE, "Gauge.", [(SCOPE, 1)], buckets=(1.0,)
            ),
            "no bucket bounds",
        ),
        (
            lambda: metric_family(
                "lab_x",
                GAUGE,
                "Gauge.",
                [(SCOPE, HistogramValue((1.0,), 1.0))],
            ),
            "one float",
        ),
    ],
)
def test_families_enforce_type_naming_and_series_rules(build, message):
    with pytest.raises(TelemetryMetricsError, match=message):
        build()


def test_direct_construction_requires_canonical_series_order():
    first = Series((("outcome", "ok"),), 1.0)
    second = Series((("outcome", "timeout"),), 1.0)

    family = MetricFamily("lab_total", COUNTER, "Lab.", (first, second))
    assert family.series == (first, second)
    with pytest.raises(TelemetryMetricsError, match="sorted"):
        MetricFamily("lab_total", COUNTER, "Lab.", (second, first))
    with pytest.raises(TelemetryMetricsError, match="sorted and unique"):
        Series((("scope", "a"), ("outcome", "b")), 1.0)
    with pytest.raises(TelemetryMetricsError, match="finite float"):
        Series((), 1)
    empty = MetricFamily("lab_gap_seconds", GAUGE, "Absent value.", ())
    assert empty.label_names == () and empty.series == ()


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: histogram(buckets=(0.25, 0.001)), "strictly increase"),
        (lambda: histogram(buckets=(0.001, 0.001)), "strictly increase"),
        (lambda: histogram(buckets=(0.001, math.inf)), "finite"),
        (lambda: histogram(buckets=(0.001,)), "one count per bound"),
        (lambda: histogram(samples=[(SCOPE, 1.0)]), "one count per bound"),
        (
            lambda: metric_family(
                "lab_latency_seconds", HISTOGRAM, "Latency.", [], buckets=(1.0,)
            ),
            "needs a series",
        ),
        (
            lambda: metric_family(
                "lab_latency_count",
                HISTOGRAM,
                "Latency.",
                [(SCOPE, HistogramValue((1.0,), 1.0))],
            ),
            "sample suffix",
        ),
    ],
)
def test_histogram_families_need_strict_bounds_and_matching_counts(build, message):
    with pytest.raises(TelemetryMetricsError, match=message):
        build()


@pytest.mark.parametrize(
    ("counts", "total", "message"),
    [
        ((2.0, 1.0), 1.0, "non-decreasing"),
        ((0.5, 1.0), 1.0, "whole numbers"),
        ((1, 1), 1.0, "whole numbers"),
        ((math.nan,), 0.0, "whole numbers"),
        ((float(2**53 + 2),), 1.0, "whole numbers"),
        ((), 0.0, "cumulative counts"),
        ((0.0,), 0.5, "sums to 0"),
        ((1.0,), math.inf, "finite float"),
    ],
)
def test_histogram_values_are_cumulative_whole_counts(counts, total, message):
    with pytest.raises(TelemetryMetricsError, match=message):
        HistogramValue(counts, total)


def test_integer_histogram_buckets_exact_edges_without_float_rounding():
    value = integer_histogram(
        [999, 1000, 1001, 250_000_000, 300_000_000],
        bounds=(1000, 250_000_000),
        scale=10**9,
    )

    assert value.cumulative_counts == (2.0, 4.0, 5.0)
    assert value.count == 5.0
    assert value.sum == (999 + 1000 + 1001 + 250_000_000 + 300_000_000) / 10**9
    assert integer_histogram([], bounds=(1,), scale=1) == HistogramValue(
        (0.0, 0.0), 0.0
    )
    assert scaled_bounds((500, 250_000), scale=10**6) == (0.0005, 0.25)
    for bounds in ((5, 5), (0,), (1.5,)):
        with pytest.raises(TelemetryMetricsError):
            integer_histogram([1], bounds=bounds, scale=1)
    with pytest.raises(TelemetryMetricsError, match="non-negative integers"):
        integer_histogram([-1], bounds=(1,), scale=1)
    with pytest.raises(TelemetryMetricsError, match="scale"):
        scaled_bounds((1,), scale=0)


def test_metric_sets_sort_families_and_reject_sample_name_collisions():
    gauge = metric_family("lab_up", GAUGE, "Up.", [(SCOPE, 1)])
    collected = metric_set([histogram(), gauge, counter()])

    assert [family.name for family in collected.families] == [
        "lab_attempts_total",
        "lab_latency_seconds",
        "lab_up",
    ]
    assert collected.family("lab_up") is gauge
    assert collected.series_count == 3
    with pytest.raises(TelemetryMetricsError, match="no metric family"):
        collected.family("lab_down")
    with pytest.raises(TelemetryMetricsError, match="sorted by unique name"):
        MetricSet((gauge, counter()))
    with pytest.raises(TelemetryMetricsError, match="sorted by unique name"):
        metric_set([gauge, gauge])
    clash = metric_family("lab_latency_seconds_sum", GAUGE, "Clash.", [(SCOPE, 1)])
    with pytest.raises(TelemetryMetricsError, match="both produce"):
        metric_set([histogram(), clash])
    with pytest.raises(TelemetryMetricsError, match="MetricFamily values"):
        MetricSet([gauge])
