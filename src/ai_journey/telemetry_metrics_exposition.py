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

``parse_exposition`` reads that format back strictly and returns the same
validated metric set, so ``parse_exposition(encode_exposition(m)) == m``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal

from .serving_metrics import PROMETHEUS_CONTENT_TYPE
from .telemetry_metrics_model import (
    HISTOGRAM,
    METRIC_TYPES,
    HistogramValue,
    MetricFamily,
    MetricSet,
    TelemetryMetricsError,
    metric_family,
    metric_set,
    require_metric_name,
)

CONTENT_TYPE = PROMETHEUS_CONTENT_TYPE
MAX_EXPOSITION_BYTES = 1024 * 1024
_SAMPLE_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_HELP_ESCAPES = {"\\": "\\", "n": "\n"}
_LABEL_ESCAPES = {"\\": "\\", '"': '"', "n": "\n"}


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


def _unescape(text: str, escapes: dict[str, str], where: str) -> str:
    characters = []
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\":
            following = text[index + 1 : index + 2]
            if following not in escapes:
                raise TelemetryMetricsError(f"{where}: unknown escape \\{following}")
            characters.append(escapes[following])
            index += 2
        else:
            characters.append(character)
            index += 1
    return "".join(characters)


def _number(text: str, where: str) -> float:
    if text in ("+Inf", "-Inf", "NaN") or _NUMBER.fullmatch(text):
        return float(text)
    raise TelemetryMetricsError(f"{where}: invalid sample value {text!r}")


def _labels(line: str, index: int, where: str) -> tuple[list[tuple[str, str]], int]:
    """Parse ``name="value"`` pairs after an opening brace, through ``}``."""

    labels: list[tuple[str, str]] = []
    while True:
        match = _LABEL_NAME.match(line, index)
        if match is None:
            raise TelemetryMetricsError(f"{where}: expected a label name")
        name, index = match.group(), match.end()
        if line[index : index + 2] != '="':
            raise TelemetryMetricsError(f'{where}: label {name} needs ="value"')
        start = index = index + 2
        while index < len(line) and line[index] != '"':
            index += 2 if line[index] == "\\" else 1
        if index >= len(line):
            raise TelemetryMetricsError(f"{where}: unterminated label value")
        if any(name == seen for seen, _ in labels):
            raise TelemetryMetricsError(f"{where}: label {name} repeats")
        labels.append((name, _unescape(line[start:index], _LABEL_ESCAPES, where)))
        index += 1
        if line[index : index + 1] == "}":
            return labels, index + 1
        if line[index : index + 1] != ",":
            raise TelemetryMetricsError(f"{where}: labels need , or a closing }}")
        index += 1


def _sample(line: str, where: str) -> tuple[str, list[tuple[str, str]], float]:
    match = _SAMPLE_NAME.match(line)
    if match is None:
        raise TelemetryMetricsError(f"{where}: expected a sample name")
    name, index = match.group(), match.end()
    labels: list[tuple[str, str]] = []
    if line[index : index + 1] == "{":
        labels, index = _labels(line, index + 1, where)
    if line[index : index + 1] != " ":
        raise TelemetryMetricsError(f"{where}: one space must precede the value")
    fields = line[index + 1 :].split(" ")
    if len(fields) != 1:
        raise TelemetryMetricsError(
            f"{where}: a sample ends at its value; timestamps are not accepted"
        )
    return name, labels, _number(fields[0], where)


@dataclass
class _Draft:
    name: str
    help: str | None = None
    kind: str | None = None
    samples: list[tuple[str, list[tuple[str, str]], float, str]] = field(
        default_factory=list
    )


def _metadata(draft: _Draft, keyword: str, text: str, where: str) -> None:
    if draft.samples:
        raise TelemetryMetricsError(
            f"{where}: {keyword} for {draft.name} follows samples"
        )
    if keyword == "HELP":
        if draft.help is not None:
            raise TelemetryMetricsError(f"{where}: {draft.name} repeats its HELP line")
        draft.help = _unescape(text, _HELP_ESCAPES, where)
    else:
        if draft.kind is not None:
            raise TelemetryMetricsError(f"{where}: {draft.name} repeats its TYPE line")
        if text not in METRIC_TYPES:
            raise TelemetryMetricsError(f"{where}: unsupported metric type {text!r}")
        draft.kind = text


@dataclass
class _HistogramSeries:
    labels: tuple[tuple[str, str], ...]
    bounds: list[float] = field(default_factory=list)
    counts: list[float] = field(default_factory=list)
    sum: float | None = None
    count: float | None = None


def _histogram(draft: _Draft) -> MetricFamily:
    """Group bucket, sum and count lines into series, one series at a time."""

    finished: list[_HistogramSeries] = []
    current: _HistogramSeries | None = None
    for name, labels, value, where in draft.samples:
        suffix = name[len(draft.name) :]
        bounds = [text for label, text in labels if label == "le"]
        key = tuple(sorted(pair for pair in labels if pair[0] != "le"))
        if suffix == "_bucket":
            if len(bounds) != 1:
                raise TelemetryMetricsError(f"{where}: a bucket needs one le label")
            if current is None or current.sum is not None or current.labels != key:
                if current is not None and current.count is None:
                    raise TelemetryMetricsError(
                        f"{where}: a series ended before its _sum and _count"
                    )
                if any(item.labels == key for item in finished):
                    raise TelemetryMetricsError(f"{where}: histogram series repeats")
                current = _HistogramSeries(key)
                finished.append(current)
            bound = _number(bounds[0], where)
            if math.isnan(bound) or current.bounds and bound <= current.bounds[-1]:
                raise TelemetryMetricsError(f"{where}: bucket bounds must ascend")
            current.bounds.append(bound)
            current.counts.append(value)
            continue
        if bounds:
            raise TelemetryMetricsError(f"{where}: {name} takes no le label")
        if current is None or current.labels != key:
            raise TelemetryMetricsError(f"{where}: {name} does not follow its buckets")
        if suffix == "_sum":
            if current.sum is not None or current.bounds[-1] != math.inf:
                raise TelemetryMetricsError(
                    f"{where}: _sum must follow a +Inf bucket once"
                )
            current.sum = value
        elif current.sum is None or current.count is not None:
            raise TelemetryMetricsError(f"{where}: _count must follow _sum once")
        else:
            current.count = value
    if current is not None and current.count is None:
        raise TelemetryMetricsError(f"{draft.name} ends before a series' _count")
    layouts = {tuple(item.bounds) for item in finished}
    if len(layouts) > 1:
        raise TelemetryMetricsError(f"{draft.name} series disagree on bucket bounds")
    if any(item.counts[-1] != item.count for item in finished):
        raise TelemetryMetricsError(f"{draft.name} +Inf bucket differs from _count")
    return metric_family(
        draft.name,
        HISTOGRAM,
        draft.help,
        [
            (dict(item.labels), HistogramValue(tuple(item.counts), item.sum))
            for item in finished
        ],
        buckets=finished[0].bounds[:-1] if finished else (),
    )


def _family(draft: _Draft) -> MetricFamily:
    if draft.kind is None or draft.help is None:
        raise TelemetryMetricsError(f"{draft.name} needs both HELP and TYPE lines")
    if draft.kind == HISTOGRAM:
        return _histogram(draft)
    return metric_family(
        draft.name,
        draft.kind,
        draft.help,
        [(dict(labels), value) for _, labels, value, _ in draft.samples],
    )


def parse_exposition(text: str) -> MetricSet:
    """Parse text exposition 0.0.4 strictly into a validated metric set.

    Only ``# HELP`` and ``# TYPE`` lines and sample lines are accepted, with
    single-space separators and a line feed after every line. Rejected because
    the encoder never writes them or the format leaves them open: blank and
    other comment lines, tabs and carriage returns, timestamps, untyped and
    summary families, a HELP or TYPE line that repeats or follows samples, a
    family split into two groups, unknown escapes, empty label sets, repeated
    labels or series, and histogram series whose buckets do not ascend to
    ``+Inf`` before ``_sum`` and ``_count`` or whose ``+Inf`` bucket is not
    the count. Families and series may come in any order and numbers in any
    decimal spelling; ``is_canonical_exposition`` also requires encoder bytes.
    """

    if type(text) is not str:
        raise TypeError("exposition text must be str")
    if len(text.encode("utf-8", "surrogatepass")) > MAX_EXPOSITION_BYTES:
        raise TelemetryMetricsError(f"exposition exceeds {MAX_EXPOSITION_BYTES} bytes")
    if text and not text.endswith("\n"):
        raise TelemetryMetricsError("exposition must end with a line feed")
    drafts: list[_Draft] = []
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        where = f"line {number}"
        if "\r" in line or "\t" in line:
            raise TelemetryMetricsError(
                f"{where}: tabs and carriage returns are not accepted"
            )
        if line.startswith(("# HELP ", "# TYPE ")):
            name, separator, rest = line[7:].partition(" ")
            if not separator:
                raise TelemetryMetricsError(
                    f"{where}: {line[2:6]} needs a name and text"
                )
            require_metric_name(name)
            if not drafts or drafts[-1].name != name:
                if any(draft.name == name for draft in drafts):
                    raise TelemetryMetricsError(
                        f"{where}: family {name} is split into several groups"
                    )
                drafts.append(_Draft(name))
            _metadata(drafts[-1], line[2:6], rest, where)
            continue
        if not line or line.startswith("#"):
            raise TelemetryMetricsError(
                f"{where}: blank and comment lines are not accepted"
            )
        name, labels, value = _sample(line, where)
        draft = drafts[-1] if drafts else None
        if draft is None or draft.kind is None:
            raise TelemetryMetricsError(f"{where}: {name} precedes its TYPE line")
        names = (
            [draft.name + suffix for suffix in ("_bucket", "_sum", "_count")]
            if draft.kind == HISTOGRAM
            else [draft.name]
        )
        if name not in names:
            raise TelemetryMetricsError(
                f"{where}: {name} does not belong to family {draft.name}"
            )
        draft.samples.append((name, labels, value, where))
    return metric_set(_family(draft) for draft in drafts)


def is_canonical_exposition(text: str) -> bool:
    """Whether text parses strictly and is exactly what the encoder writes."""

    return encode_exposition(parse_exposition(text)) == text
