"""Strict Prometheus metric families for local CPU lab telemetry.

A family is one metric name, one type, help text and a fixed set of series.
The model admits only what the Prometheus data model and the text exposition
format 0.0.4 carry without ambiguity, plus house rules that keep a lab export
small and unmistakable:

* metric names match ``[a-zA-Z_][a-zA-Z0-9_]*``. The data model also allows
  colons but reserves them for recording rules, so exported families do not
  use them;
* label names match the same pattern and never start with ``__``, which
  Prometheus reserves. ``le`` belongs to histogram buckets, ``quantile`` to
  summaries, and the scraper attaches ``job`` and ``instance``, so no series
  declares them;
* label values and help text are non-empty printable text in which a line
  feed is the only allowed control character. Prometheus reads an empty label
  value as an absent label, and help text has no edge whitespace because a
  parser cannot tell it from the separator;
* counters end in ``_total``; gauges and histograms do not, and a histogram
  name does not end in ``_bucket``, ``_sum`` or ``_count``;
* all series of a family carry the same label names, are sorted by label
  values and never repeat, and no two families produce the same sample name;
* sample values are finite floats, as Prometheus stores them. An integer is
  accepted only up to 2**53, where a float represents it exactly;
* histogram bounds are explicit, finite and strictly increasing, and ``+Inf``
  is implicit. Cumulative counts are whole numbers that never decrease, the
  ``+Inf`` bucket is the observation count, and an empty histogram sums to 0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

COUNTER = "counter"
GAUGE = "gauge"
HISTOGRAM = "histogram"
METRIC_TYPES = (COUNTER, GAUGE, HISTOGRAM)
HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")
RESERVED_LABEL_NAMES = frozenset({"le", "quantile", "job", "instance"})
MAX_FAMILIES = 128
MAX_SERIES_PER_FAMILY = 64
MAX_LABELS_PER_SERIES = 8
MAX_BUCKETS = 32
MAX_NAME_CHARS = 128
MAX_LABEL_VALUE_CHARS = 256
MAX_HELP_CHARS = 512
MAX_EXACT_INTEGER = 2**53
_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class TelemetryMetricsError(ValueError):
    """Raised when telemetry inputs, evidence or outputs violate the contract."""


def require_metric_name(value: object) -> str:
    """Return an exported metric name: data-model syntax without colons."""

    if (
        type(value) is not str
        or len(value) > MAX_NAME_CHARS
        or _NAME.fullmatch(value) is None
    ):
        raise TelemetryMetricsError(f"invalid metric name: {value!r}")
    return value


def require_label_name(value: object) -> str:
    """Return a label name that an exported series may declare."""

    if (
        type(value) is not str
        or len(value) > MAX_NAME_CHARS
        or _NAME.fullmatch(value) is None
        or value.startswith("__")
    ):
        raise TelemetryMetricsError(f"invalid label name: {value!r}")
    if value in RESERVED_LABEL_NAMES:
        raise TelemetryMetricsError(f"label name {value!r} is reserved")
    return value


def _require_text(value: object, name: str, limit: int) -> str:
    if type(value) is not str or not 1 <= len(value) <= limit:
        raise TelemetryMetricsError(f"{name} must be 1 to {limit} characters")
    if any(not (character.isprintable() or character == "\n") for character in value):
        raise TelemetryMetricsError(f"{name} contains a control character")
    return value


def require_label_value(value: object) -> str:
    """Return a non-empty printable label value; line feeds are escaped later."""

    return _require_text(value, "label value", MAX_LABEL_VALUE_CHARS)


def sample_value(value: object, name: str = "sample value") -> float:
    """Return a finite float; an integer must be exact in float64.

    Negative zero becomes zero, so a value has one exposition spelling.
    """

    if type(value) is int:
        if abs(value) > MAX_EXACT_INTEGER:
            raise TelemetryMetricsError(f"{name} {value} is not exact in float64")
        return float(value)
    if type(value) is not float:
        raise TypeError(f"{name} must be an int or float")
    if not math.isfinite(value):
        raise TelemetryMetricsError(f"{name} must be finite")
    return value + 0.0


@dataclass(frozen=True)
class HistogramValue:
    """Cumulative bucket counts, the last one for ``+Inf``, and the sum."""

    cumulative_counts: tuple[float, ...]
    sum: float

    def __post_init__(self) -> None:
        counts = self.cumulative_counts
        if type(counts) is not tuple or not 1 <= len(counts) <= MAX_BUCKETS + 1:
            raise TelemetryMetricsError(
                f"a histogram needs 1 to {MAX_BUCKETS + 1} cumulative counts"
            )
        previous = 0.0
        for count in counts:
            if (
                type(count) is not float
                or not count.is_integer()
                or not previous <= count <= MAX_EXACT_INTEGER
            ):
                raise TelemetryMetricsError(
                    "cumulative counts must be non-decreasing whole numbers"
                )
            previous = count
        if type(self.sum) is not float or not math.isfinite(self.sum):
            raise TelemetryMetricsError("histogram sum must be a finite float")
        if counts[-1] == 0 and self.sum != 0:
            raise TelemetryMetricsError("a histogram without observations sums to 0")

    @property
    def count(self) -> float:
        """Observation count, equal to the ``+Inf`` bucket."""

        return self.cumulative_counts[-1]


@dataclass(frozen=True)
class Series:
    """One series: label pairs sorted by name and one value."""

    labels: tuple[tuple[str, str], ...]
    value: float | HistogramValue

    def __post_init__(self) -> None:
        if type(self.labels) is not tuple or len(self.labels) > MAX_LABELS_PER_SERIES:
            raise TelemetryMetricsError(
                f"a series carries at most {MAX_LABELS_PER_SERIES} labels"
            )
        for pair in self.labels:
            if type(pair) is not tuple or len(pair) != 2:
                raise TelemetryMetricsError("labels must be (name, value) pairs")
            require_label_name(pair[0])
            require_label_value(pair[1])
        names = [name for name, _ in self.labels]
        if any(left >= right for left, right in zip(names, names[1:])):
            raise TelemetryMetricsError("label names must be sorted and unique")
        if not isinstance(self.value, HistogramValue) and (
            type(self.value) is not float or not math.isfinite(self.value)
        ):
            raise TelemetryMetricsError("a series value must be a finite float")

    @property
    def label_values(self) -> tuple[str, ...]:
        return tuple(value for _, value in self.labels)

    def label_map(self) -> dict[str, str]:
        return dict(self.labels)


@dataclass(frozen=True)
class MetricFamily:
    """One metric name with its type, help text and sorted series."""

    name: str
    kind: str
    help: str
    series: tuple[Series, ...]
    buckets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        require_metric_name(self.name)
        if self.kind not in METRIC_TYPES:
            raise TelemetryMetricsError(f"unsupported metric type: {self.kind!r}")
        _require_text(self.help, "help text", MAX_HELP_CHARS)
        if self.help != self.help.strip():
            raise TelemetryMetricsError("help text must not start or end with space")
        if (
            type(self.series) is not tuple
            or len(self.series) > MAX_SERIES_PER_FAMILY
            or any(type(item) is not Series for item in self.series)
        ):
            raise TelemetryMetricsError(
                f"a family holds at most {MAX_SERIES_PER_FAMILY} Series values"
            )
        if len({tuple(n for n, _ in item.labels) for item in self.series}) > 1:
            raise TelemetryMetricsError(f"{self.name} series differ in label names")
        keys = [item.label_values for item in self.series]
        if any(left >= right for left, right in zip(keys, keys[1:])):
            raise TelemetryMetricsError(
                f"{self.name} series must be sorted by label values without repeats"
            )
        if type(self.buckets) is not tuple:
            raise TelemetryMetricsError("bucket bounds must be a tuple")
        if self.kind == HISTOGRAM:
            self._check_histogram()
            return
        if self.buckets:
            raise TelemetryMetricsError(f"a {self.kind} has no bucket bounds")
        if any(isinstance(item.value, HistogramValue) for item in self.series):
            raise TelemetryMetricsError(f"a {self.kind} series holds one float")
        if self.kind == COUNTER:
            if not self.name.endswith("_total"):
                raise TelemetryMetricsError(f"counter {self.name} must end in _total")
            if any(item.value < 0 for item in self.series):
                raise TelemetryMetricsError(f"counter {self.name} is negative")
        elif self.name.endswith("_total"):
            raise TelemetryMetricsError(f"gauge {self.name} must not end in _total")

    def _check_histogram(self) -> None:
        if self.name.endswith(("_total", *HISTOGRAM_SUFFIXES)):
            raise TelemetryMetricsError(
                f"histogram {self.name} must not end in a sample suffix"
            )
        if not self.series:
            raise TelemetryMetricsError(
                f"histogram {self.name} needs a series to carry its bounds"
            )
        bounds = self.buckets
        if len(bounds) > MAX_BUCKETS or any(
            type(bound) is not float or not math.isfinite(bound) for bound in bounds
        ):
            raise TelemetryMetricsError(
                f"histogram {self.name} needs at most {MAX_BUCKETS} finite bounds"
            )
        if any(left >= right for left, right in zip(bounds, bounds[1:])):
            raise TelemetryMetricsError(
                f"histogram {self.name} bounds must strictly increase"
            )
        for item in self.series:
            if (
                not isinstance(item.value, HistogramValue)
                or len(item.value.cumulative_counts) != len(bounds) + 1
            ):
                raise TelemetryMetricsError(
                    f"histogram {self.name} series need one count per bound and +Inf"
                )

    @property
    def label_names(self) -> tuple[str, ...]:
        """Label names shared by every series; empty for a family without any."""

        return tuple(name for name, _ in self.series[0].labels) if self.series else ()

    @property
    def sample_names(self) -> tuple[str, ...]:
        """Names that this family's sample lines carry."""

        if self.kind == HISTOGRAM:
            return tuple(self.name + suffix for suffix in HISTOGRAM_SUFFIXES)
        return (self.name,)


def metric_family(
    name: str,
    kind: str,
    help_text: str,
    samples: Iterable[tuple[Mapping[str, str], object]],
    *,
    buckets: Sequence[float] = (),
) -> MetricFamily:
    """Build a family from (labels mapping, value) pairs given in any order.

    Integer values become exact floats and series are sorted canonically, so a
    caller states what was measured and the model fixes one representation.
    """

    built = []
    for labels, value in samples:
        if not isinstance(labels, Mapping):
            raise TypeError("series labels must be a mapping")
        pairs = tuple(sorted(labels.items()))
        if not isinstance(value, HistogramValue):
            value = sample_value(value)
        built.append(Series(pairs, value))
    built.sort(key=lambda item: item.label_values)
    return MetricFamily(
        name,
        kind,
        help_text,
        tuple(built),
        tuple(sample_value(bound, "bucket bound") for bound in buckets),
    )


def scaled_bounds(bounds: Sequence[int], *, scale: int) -> tuple[float, ...]:
    """Express exact integer bounds in base units, such as microseconds as s."""

    if type(scale) is not int or scale < 1:
        raise TelemetryMetricsError("scale must be a positive integer")
    if any(type(bound) is not int or bound < 1 for bound in bounds):
        raise TelemetryMetricsError("integer bounds must be positive integers")
    return tuple(bound / scale for bound in bounds)


def integer_histogram(
    observations: Iterable[int], *, bounds: Sequence[int], scale: int
) -> HistogramValue:
    """Bucket exact integer observations against exact integer bounds.

    Nanosecond latencies are compared with nanosecond bounds, so no float
    rounding decides a bucket. Only the sum is divided by ``scale`` (10**9 for
    seconds), once and correctly rounded.
    """

    scaled_bounds(bounds, scale=scale)
    if any(left >= right for left, right in zip(bounds, bounds[1:])):
        raise TelemetryMetricsError("integer bounds must strictly increase")
    values = list(observations)
    if any(type(value) is not int or value < 0 for value in values):
        raise TelemetryMetricsError("observations must be non-negative integers")
    if len(values) > MAX_EXACT_INTEGER:
        raise TelemetryMetricsError("too many observations for exact counts")
    cumulative = tuple(
        float(sum(value <= bound for value in values)) for bound in bounds
    ) + (float(len(values)),)
    return HistogramValue(cumulative, sample_value(sum(values) / scale, "sum"))


@dataclass(frozen=True)
class MetricSet:
    """Families sorted by name whose names and sample names never collide."""

    families: tuple[MetricFamily, ...]

    def __post_init__(self) -> None:
        if (
            type(self.families) is not tuple
            or len(self.families) > MAX_FAMILIES
            or any(type(family) is not MetricFamily for family in self.families)
        ):
            raise TelemetryMetricsError(
                f"a metric set holds at most {MAX_FAMILIES} MetricFamily values"
            )
        names = [family.name for family in self.families]
        if names != sorted(set(names)):
            raise TelemetryMetricsError("families must be sorted by unique name")
        owners: dict[str, str] = {}
        for family in self.families:
            for name in (family.name, *family.sample_names):
                if owners.setdefault(name, family.name) != family.name:
                    raise TelemetryMetricsError(
                        f"{family.name} and {owners[name]} both produce {name}"
                    )

    def family(self, name: str) -> MetricFamily:
        """Return the family with this exact name."""

        for family in self.families:
            if family.name == name:
                return family
        raise TelemetryMetricsError(f"no metric family named {name!r}")

    @property
    def series_count(self) -> int:
        return sum(len(family.series) for family in self.families)


def metric_set(families: Iterable[MetricFamily]) -> MetricSet:
    """Collect families from several adapters in canonical name order."""

    return MetricSet(tuple(sorted(families, key=lambda family: family.name)))
