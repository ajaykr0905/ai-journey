import pytest
from test_telemetry_metrics_exposition import GOLDEN, lab_metrics

from ai_journey.telemetry_metrics_exposition import (
    MAX_EXPOSITION_BYTES,
    encode_exposition,
    is_canonical_exposition,
    parse_exposition,
)
from ai_journey.telemetry_metrics_model import TelemetryMetricsError

HISTOGRAM_HEAD = "# HELP h_seconds Latency.\n# TYPE h_seconds histogram\n"


def test_parser_round_trips_the_encoder_exactly():
    metrics = lab_metrics()

    assert parse_exposition(encode_exposition(metrics)) == metrics
    assert parse_exposition(GOLDEN) == metrics
    assert is_canonical_exposition(GOLDEN)
    assert parse_exposition("").families == ()


def test_parser_accepts_any_order_and_spelling_but_only_one_is_canonical():
    text = (
        "# TYPE z_up gauge\n"
        "# HELP z_up Up.\n"
        'z_up{scope="b"} 1.0\n'
        'z_up{scope="a"} 4.8e1\n'
        "# HELP a_total Total.\n"
        "# TYPE a_total counter\n"
        "a_total 2391208\n"
    )
    metrics = parse_exposition(text)

    assert [family.name for family in metrics.families] == ["a_total", "z_up"]
    assert [series.value for series in metrics.family("z_up").series] == [48.0, 1.0]
    assert not is_canonical_exposition(text)
    assert encode_exposition(metrics) == (
        "# HELP a_total Total.\n"
        "# TYPE a_total counter\n"
        "a_total 2.391208e+06\n"
        "# HELP z_up Up.\n"
        "# TYPE z_up gauge\n"
        'z_up{scope="a"} 48\n'
        'z_up{scope="b"} 1\n'
    )


def test_parser_unescapes_help_and_label_values():
    text = (
        "# HELP g_info A \\\\ and a\\nbreak.\n"
        "# TYPE g_info gauge\n"
        'g_info{note="q\\"\\\\\\n,}"} 1\n'
    )
    family = parse_exposition(text).family("g_info")

    assert family.help == "A \\ and a\nbreak."
    assert family.series[0].label_map() == {"note": 'q"\\\n,}'}
    assert is_canonical_exposition(text)


def histogram(*lines):
    return HISTOGRAM_HEAD + "".join(line + "\n" for line in lines)


GOOD_SERIES = (
    'h_seconds_bucket{le="0.5"} 1',
    'h_seconds_bucket{le="+Inf"} 2',
    "h_seconds_sum 0.75",
    "h_seconds_count 2",
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("# HELP x Up.\n# TYPE x gauge\nx 1", "end with a line feed"),
        ("# HELP x Up.\r\n# TYPE x gauge\n", "carriage returns"),
        ("# HELP x Up.\n# TYPE x gauge\nx\t1\n", "tabs"),
        ("# HELP x Up.\n\n# TYPE x gauge\n", "blank and comment"),
        ("# a comment\n", "blank and comment"),
        ("#HELP x Up.\n", "blank and comment"),
        ("# HELP x Up.\n# TYPE x gauge\nx 1 1700000000000\n", "timestamps"),
        ("# HELP x Up.\n# TYPE x gauge\nx  1\n", "timestamps"),
        ("# HELP x Up.\n# TYPE x gauge\nx\n", "one space"),
        ("# HELP x Up.\nx 1\n", "precedes its TYPE"),
        ("x 1\n", "precedes its TYPE"),
        ("# HELP x Up.\n", "both HELP and TYPE"),
        ("# TYPE x gauge\n", "both HELP and TYPE"),
        ("# HELP x Up.\n# TYPE x summary\n", "unsupported metric type"),
        ("# HELP x Up.\n# TYPE x untyped\n", "unsupported metric type"),
        ("# HELP x Up.\n# TYPE x gauge\n# TYPE x gauge\n", "repeats its TYPE"),
        ("# HELP x Up.\n# HELP x Up.\n", "repeats its HELP"),
        ("# TYPE x gauge\nx 1\n# HELP x Up.\n", "follows samples"),
        (
            "# HELP x Up.\n# TYPE x gauge\n# HELP y Up.\n# TYPE y gauge\n"
            "# HELP x Again.\n",
            "split into several groups",
        ),
        ("# HELP x\n", "needs a name and text"),
        ("# HELP x \n# TYPE x gauge\n", "help text"),
        ("# HELP rule:x Up.\n", "invalid metric name"),
        ("# HELP x Up.\n# TYPE x gauge\ny 1\n", "does not belong to family x"),
        ("# HELP x_total T.\n# TYPE x_total counter\nx 1\n", "does not belong"),
        ("# HELP x Up.\n# TYPE x gauge\nx{} 1\n", "expected a label name"),
        ('# HELP x Up.\n# TYPE x gauge\nx{a="b",} 1\n', "expected a label name"),
        ('# HELP x Up.\n# TYPE x gauge\nx{a="b" b="c"} 1\n', "closing"),
        ('# HELP x Up.\n# TYPE x gauge\nx{a="b",a="c"} 1\n', "label a repeats"),
        ('# HELP x Up.\n# TYPE x gauge\nx{a="b} 1\n', "unterminated"),
        ('# HELP x Up.\n# TYPE x gauge\nx{a="b\\"} 1\n', "unterminated"),
        ("# HELP x Up.\n# TYPE x gauge\nx{a=b} 1\n", 'needs ="value"'),
        ('# HELP x Up.\n# TYPE x gauge\nx{a="\\t"} 1\n', "unknown escape"),
        ("# HELP x Up\\t.\n# TYPE x gauge\n", "unknown escape"),
        ('# HELP x Up.\n# TYPE x gauge\nx{a=""} 1\n', "label value"),
        ('# HELP x Up.\n# TYPE x gauge\nx{le="1"} 1\n', "reserved"),
        ('# HELP x Up.\n# TYPE x gauge\nx{__name__="x"} 1\n', "invalid label"),
        ("# HELP x Up.\n# TYPE x gauge\nx 0x10\n", "invalid sample value"),
        ("# HELP x Up.\n# TYPE x gauge\nx 1_000\n", "invalid sample value"),
        ("# HELP x Up.\n# TYPE x gauge\nx inf\n", "invalid sample value"),
        ("# HELP x Up.\n# TYPE x gauge\nx 1,5\n", "invalid sample value"),
        ("# HELP x Up.\n# TYPE x gauge\nx NaN\n", "finite"),
        ("# HELP x Up.\n# TYPE x gauge\nx 1e999\n", "finite"),
        ("# HELP x Up.\n# TYPE x gauge\nx 1\nx 2\n", "without repeats"),
        ("# HELP x_total T.\n# TYPE x_total counter\nx_total -1\n", "negative"),
        ("# HELP x T.\n# TYPE x counter\nx 1\n", "must end in _total"),
    ],
)
def test_parser_rejects_malformed_or_ambiguous_exposition(text, message):
    with pytest.raises(TelemetryMetricsError, match=message):
        parse_exposition(text)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ((GOOD_SERIES[1], GOOD_SERIES[0], *GOOD_SERIES[2:]), "must ascend"),
        (('h_seconds_bucket{le="0.5"} 1', *GOOD_SERIES[2:]), r"\+Inf bucket once"),
        ((*GOOD_SERIES[:2], "h_seconds_count 2", "h_seconds_sum 0.75"), "_count"),
        ((*GOOD_SERIES[:3],), "ends before"),
        (
            (*GOOD_SERIES[:2], 'h_seconds_bucket{le="1"} 2', *GOOD_SERIES[2:]),
            "must ascend",
        ),
        (
            (*GOOD_SERIES[:3], 'h_seconds_bucket{le="+Inf"} 2', GOOD_SERIES[3]),
            "ended before",
        ),
        ((*GOOD_SERIES, *GOOD_SERIES), "series repeats"),
        (
            (
                'h_seconds_bucket{le="0.5"} 2',
                'h_seconds_bucket{le="+Inf"} 1',
                "h_seconds_sum 0.75",
                "h_seconds_count 1",
            ),
            "non-decreasing",
        ),
        ((*GOOD_SERIES[:3], "h_seconds_count 3"), "differs from _count"),
        (("h_seconds_bucket 1", *GOOD_SERIES[1:]), "one le label"),
        (
            (*GOOD_SERIES[:2], 'h_seconds_sum{le="1"} 0.75', GOOD_SERIES[3]),
            "takes no le",
        ),
        (
            (*GOOD_SERIES[:2], 'h_seconds_sum{x="1"} 0.75', GOOD_SERIES[3]),
            "does not follow its buckets",
        ),
        (
            (
                *GOOD_SERIES,
                'h_seconds_bucket{phase="b",le="0.25"} 1',
                'h_seconds_bucket{phase="b",le="+Inf"} 2',
                'h_seconds_sum{phase="b"} 0.75',
                'h_seconds_count{phase="b"} 2',
            ),
            "differ in label names|disagree",
        ),
        (
            (
                'h_seconds_bucket{phase="a",le="0.5"} 1',
                'h_seconds_bucket{phase="a",le="+Inf"} 1',
                'h_seconds_sum{phase="a"} 0.5',
                'h_seconds_count{phase="a"} 1',
                'h_seconds_bucket{phase="b",le="0.25"} 1',
                'h_seconds_bucket{phase="b",le="+Inf"} 1',
                'h_seconds_sum{phase="b"} 0.5',
                'h_seconds_count{phase="b"} 1',
            ),
            "disagree on bucket bounds",
        ),
        (('h_seconds_bucket{le="NaN"} 1', *GOOD_SERIES[1:]), "must ascend"),
        (('h_seconds_bucket{le="-Inf"} 1', *GOOD_SERIES[1:]), "finite"),
        (("h_seconds_sum 0", "h_seconds_count 0"), "does not follow"),
        ((), "needs a series"),
    ],
)
def test_parser_rejects_inconsistent_histograms(lines, message):
    with pytest.raises(TelemetryMetricsError, match=message):
        parse_exposition(histogram(*lines))


def test_parser_reads_valid_histograms_with_shared_bounds():
    text = histogram(
        'h_seconds_bucket{phase="a",le="0.5"} 1',
        'h_seconds_bucket{phase="a",le="+Inf"} 1',
        'h_seconds_sum{phase="a"} 0.5',
        'h_seconds_count{phase="a"} 1',
        'h_seconds_bucket{phase="b",le="0.50"} 0',
        'h_seconds_bucket{phase="b",le="+Inf"} 0',
        'h_seconds_sum{phase="b"} 0',
        'h_seconds_count{phase="b"} 0',
    )
    family = parse_exposition(text).family("h_seconds")

    assert family.buckets == (0.5,)
    assert [series.value.count for series in family.series] == [1.0, 0.0]
    assert not is_canonical_exposition(text)
    assert is_canonical_exposition(text.replace('le="0.50"', 'le="0.5"'))


def test_parser_bounds_input_size_and_type():
    with pytest.raises(TelemetryMetricsError, match="exceeds"):
        parse_exposition("#" * (MAX_EXPOSITION_BYTES + 1))
    with pytest.raises(TypeError):
        parse_exposition(GOLDEN.encode())
