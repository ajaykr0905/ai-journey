"""Deterministic Prometheus text exposition format 0.0.4 for metric sets.

``encode_exposition`` writes one ``# HELP`` and one ``# TYPE`` line per family,
families sorted by name and series by label values, labels sorted by name
with ``le`` last on bucket lines, and no timestamps. Help text escapes the
backslash and the line feed; label values also escape the double quote.
Histogram series list cumulative buckets in bound order ending with
``le="+Inf"``, then ``_sum`` and ``_count``. Values and bounds use Go's
shortest ``strconv.FormatFloat(v, 'g', -1, 64)`` spelling, as the Go client
library writes them, so equal metric sets always encode to identical bytes
and a count of 2391208 appears as ``2.391208e+06``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from .serving_metrics import PROMETHEUS_CONTENT_TYPE
from .telemetry_metrics_model import HISTOGRAM, MetricFamily, MetricSet

CONTENT_TYPE = PROMETHEUS_CONTENT_TYPE


def format_value(value: float) -> str:
    """Spell a float as Go's shortest ``'g'`` format, or +Inf, -Inf and NaN.

    Python's ``repr`` already yields the shortest digits that round-trip; Go
    differs only in layout, switching to an exponent below 1e-4 or from 1e6.
    """

    if type(value) is not float:
        raise TypeError("exposition values must be floats")
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == 0:
        return "0"
    sign, digits, exponent = Decimal(repr(value)).as_tuple()
    while len(digits) > 1 and digits[-1] == 0:
        digits, exponent = digits[:-1], exponent + 1
    text = "".join(str(digit) for digit in digits)
    point = len(text) + exponent
    prefix = "-" if sign else ""
    if not -4 <= point - 1 < 6:
        mantissa = text[0] + ("." + text[1:] if len(text) > 1 else "")
        return f"{prefix}{mantissa}e{'-' if point < 1 else '+'}{abs(point - 1):02d}"
    if point <= 0:
        return f"{prefix}0.{'0' * -point}{text}"
    if point >= len(text):
        return prefix + text + "0" * (point - len(text))
    return f"{prefix}{text[:point]}.{text[point:]}"


def escape_help(text: str) -> str:
    """Escape help text as format 0.0.4 requires: backslash and line feed."""

    return text.replace("\\", "\\\\").replace("\n", "\\n")


def escape_label_value(text: str) -> str:
    """Escape a label value: backslash, double quote and line feed."""

    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass(frozen=True)
class Sample:
    """One exposition line: sample name, ordered labels and value."""

    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label_map(self) -> dict[str, str]:
        return dict(self.labels)


def family_samples(family: MetricFamily) -> tuple[Sample, ...]:
    """Expand one family into its sample lines in exposition order."""

    if type(family) is not MetricFamily:
        raise TypeError("family must be MetricFamily")
    if family.kind != HISTOGRAM:
        return tuple(
            Sample(family.name, series.labels, series.value) for series in family.series
        )
    samples = []
    bounds = (*family.buckets, math.inf)
    for series in family.series:
        histogram = series.value
        for bound, count in zip(bounds, histogram.cumulative_counts, strict=True):
            labels = (*series.labels, ("le", format_value(bound)))
            samples.append(Sample(family.name + "_bucket", labels, count))
        samples.append(Sample(family.name + "_sum", series.labels, histogram.sum))
        samples.append(Sample(family.name + "_count", series.labels, histogram.count))
    return tuple(samples)


def flatten_samples(metrics: MetricSet) -> tuple[Sample, ...]:
    """Every sample line of a metric set, in exposition order."""

    if type(metrics) is not MetricSet:
        raise TypeError("metrics must be MetricSet")
    return tuple(
        sample for family in metrics.families for sample in family_samples(family)
    )


def _sample_line(sample: Sample) -> str:
    labels = ",".join(
        f'{name}="{escape_label_value(value)}"' for name, value in sample.labels
    )
    braces = "{" + labels + "}" if labels else ""
    return f"{sample.name}{braces} {format_value(sample.value)}"


def encode_exposition(metrics: MetricSet) -> str:
    """Render a metric set as canonical text exposition 0.0.4.

    Every line ends with a line feed; a family without series still states
    its help and type, so a declared but unmeasured gauge stays visible.
    """

    if type(metrics) is not MetricSet:
        raise TypeError("metrics must be MetricSet")
    lines = []
    for family in metrics.families:
        lines.append(f"# HELP {family.name} {escape_help(family.help)}")
        lines.append(f"# TYPE {family.name} {family.kind}")
        lines.extend(_sample_line(sample) for sample in family_samples(family))
    return "".join(line + "\n" for line in lines)
