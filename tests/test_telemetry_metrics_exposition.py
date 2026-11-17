import math
import random
import struct

import pytest

from ai_journey.telemetry_metrics_exposition import (
    CONTENT_TYPE,
    encode_exposition,
    escape_help,
    escape_label_value,
    flatten_samples,
    format_value,
)
from ai_journey.telemetry_metrics_model import (
    COUNTER,
    GAUGE,
    HISTOGRAM,
    HistogramValue,
    metric_family,
    metric_set,
)

SCOPE = {"scope": "local_cpu_lab"}


def lab_metrics(order=1):
    attempts = [
        ({"outcome": "timeout"} | SCOPE, 0),
        ({"outcome": "ok"} | SCOPE, 48),
        ({"outcome": "connection_refused"} | SCOPE, 6),
    ][::order]
    latency = [
        ({"phase": "pre_fault"} | SCOPE, HistogramValue((1.0, 3.0, 3.0), 0.0025)),
        ({"phase": "post_fault"} | SCOPE, HistogramValue((0.0, 2.0, 2.0), 0.0015)),
    ][::order]
    families = [
        metric_family(
            "lab_attempts_total", COUNTER, "Client attempts by outcome.", attempts
        ),
        metric_family(
            "lab_latency_seconds",
            HISTOGRAM,
            "Success latency.",
            latency,
            buckets=(0.0005, 0.001),
        ),
        metric_family(
            "lab_run_info",
            GAUGE,
            "Run identity; help with a \\ and a\nline feed.",
            [({"note": 'say "hi"\\\n'} | SCOPE, 1)],
        ),
        metric_family("lab_unavailability_seconds", GAUGE, "Absent here.", []),
        metric_family("lab_detection_seconds", GAUGE, "Detection.", [(SCOPE, 0.00576)]),
    ][::order]
    return metric_set(families)


GOLDEN = """\
# HELP lab_attempts_total Client attempts by outcome.
# TYPE lab_attempts_total counter
lab_attempts_total{outcome="connection_refused",scope="local_cpu_lab"} 6
lab_attempts_total{outcome="ok",scope="local_cpu_lab"} 48
lab_attempts_total{outcome="timeout",scope="local_cpu_lab"} 0
# HELP lab_detection_seconds Detection.
# TYPE lab_detection_seconds gauge
lab_detection_seconds{scope="local_cpu_lab"} 0.00576
# HELP lab_latency_seconds Success latency.
# TYPE lab_latency_seconds histogram
lab_latency_seconds_bucket{phase="post_fault",scope="local_cpu_lab",le="0.0005"} 0
lab_latency_seconds_bucket{phase="post_fault",scope="local_cpu_lab",le="0.001"} 2
lab_latency_seconds_bucket{phase="post_fault",scope="local_cpu_lab",le="+Inf"} 2
lab_latency_seconds_sum{phase="post_fault",scope="local_cpu_lab"} 0.0015
lab_latency_seconds_count{phase="post_fault",scope="local_cpu_lab"} 2
lab_latency_seconds_bucket{phase="pre_fault",scope="local_cpu_lab",le="0.0005"} 1
lab_latency_seconds_bucket{phase="pre_fault",scope="local_cpu_lab",le="0.001"} 3
lab_latency_seconds_bucket{phase="pre_fault",scope="local_cpu_lab",le="+Inf"} 3
lab_latency_seconds_sum{phase="pre_fault",scope="local_cpu_lab"} 0.0025
lab_latency_seconds_count{phase="pre_fault",scope="local_cpu_lab"} 3
# HELP lab_run_info Run identity; help with a \\\\ and a\\nline feed.
# TYPE lab_run_info gauge
lab_run_info{note="say \\"hi\\"\\\\\\n",scope="local_cpu_lab"} 1
# HELP lab_unavailability_seconds Absent here.
# TYPE lab_unavailability_seconds gauge
"""


def test_encoder_writes_the_canonical_golden_exposition():
    assert encode_exposition(lab_metrics()) == GOLDEN
    assert CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"


def test_input_order_never_changes_the_encoded_bytes():
    assert encode_exposition(lab_metrics(order=-1)) == encode_exposition(lab_metrics())
    assert encode_exposition(metric_set([])) == ""


def test_escaping_covers_exactly_the_format_0_0_4_characters():
    assert escape_help('a\\b\n"c"') == 'a\\\\b\\n"c"'
    assert escape_label_value('a\\b\n"c"') == 'a\\\\b\\n\\"c\\"'


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (-1.0, "-1"),
        (48.0, "48"),
        (0.25, "0.25"),
        (2.5, "2.5"),
        (0.005, "0.005"),
        (0.0005, "0.0005"),
        (0.0001, "0.0001"),
        (0.00001, "1e-05"),
        (2.6542e-05, "2.6542e-05"),
        (123456.5, "123456.5"),
        (100000.0, "100000"),
        (999999.0, "999999"),
        (1000000.0, "1e+06"),
        (2391208.0, "2.391208e+06"),
        (1e21, "1e+21"),
        (-1.5e300, "-1.5e+300"),
        (5e-324, "5e-324"),
        (0.1 + 0.2, "0.30000000000000004"),
        (math.inf, "+Inf"),
        (-math.inf, "-Inf"),
        (math.nan, "NaN"),
    ],
)
def test_values_use_go_shortest_g_formatting(value, text):
    assert format_value(value) == text


def test_formatted_values_round_trip_exactly():
    generator = random.Random(73)
    values = [
        struct.unpack("<d", generator.getrandbits(64).to_bytes(8, "little"))[0]
        for _ in range(2000)
    ]
    values += [generator.uniform(0, 10) for _ in range(500)]
    values += [float(generator.randrange(2**53)) for _ in range(500)]
    for value in values:
        if math.isfinite(value):
            assert float(format_value(value)) == value
    with pytest.raises(TypeError):
        format_value(1)


def test_flattened_samples_match_the_exposition_lines():
    metrics = lab_metrics()
    samples = flatten_samples(metrics)
    lines = [line for line in GOLDEN.splitlines() if not line.startswith("#")]

    assert len(samples) == len(lines) == 15
    bucket = samples[4]
    assert bucket.name == "lab_latency_seconds_bucket"
    assert bucket.labels[-1] == ("le", "0.0005")
    assert bucket.label_map()["phase"] == "post_fault"
    assert [sample.name for sample in samples[7:9]] == [
        "lab_latency_seconds_sum",
        "lab_latency_seconds_count",
    ]
    with pytest.raises(TypeError):
        flatten_samples(metrics.families)
    with pytest.raises(TypeError):
        encode_exposition(metrics.families)
