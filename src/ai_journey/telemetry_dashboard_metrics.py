"""Metric families from the telemetry-metrics lab that the dashboard charts.

The dashboard may reference only these names. They are the families that the
metrics workstream's ``telemetry_metrics_serving`` and
``telemetry_metrics_worker`` derive from the same sealed local CPU lab
evidence and export in text exposition 0.0.4, with the same types and label
names: counters end in ``_total``, durations are seconds gauges holding the
verified run's value and the success latency is a histogram split by
recovery phase. Every exported series carries ``scope="local_cpu_lab"``, and
every dashboard query selects on it, so a panel cannot show a same-named
series from anywhere else. If the metrics lab renames or retypes a family,
this catalog is the one place to change before the dashboard is regenerated.
``series`` lists the names a PromQL expression can select: a histogram
exposes ``_bucket``, ``_sum`` and ``_count`` series and never its bare name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .telemetry_traces_io import TelemetryError

METRIC_TYPES = ("counter", "gauge", "histogram")
SCOPE_LABEL = "scope"
LAB_SCOPE = "local_cpu_lab"
_NAME = re.compile(r"ai_journey_[a-z][a-z0-9_]*[a-z0-9]")
_LABEL = re.compile(r"[a-z][a-z0-9_]*")
_HELP = re.compile(r"[A-Z][ -~]{0,159}")


@dataclass(frozen=True)
class MetricSpec:
    """One exposed metric: its name, type, label names and meaning."""

    name: str
    type: str
    labels: tuple[str, ...]
    help: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NAME.fullmatch(self.name) is None:
            raise TelemetryError("metric names must be ai_journey_ snake_case")
        if self.type not in METRIC_TYPES:
            raise TelemetryError(
                f"metric type must be one of {', '.join(METRIC_TYPES)}"
            )
        if (self.type == "counter") != self.name.endswith("_total"):
            raise TelemetryError("exactly the counters end in _total")
        if type(self.labels) is not tuple or any(
            type(label) is not str
            or _LABEL.fullmatch(label) is None
            or label == "le"
            or label.startswith("__")
            for label in self.labels
        ):
            raise TelemetryError("metric labels must be snake_case and not le")
        if type(self.help) is not str or _HELP.fullmatch(self.help) is None:
            raise TelemetryError("metric help must be a short printable sentence")

    @property
    def unit(self) -> str:
        """Grafana unit of the metric's samples: seconds or a plain count."""

        return "s" if self.name.endswith("_seconds") else "short"

    @property
    def series(self) -> tuple[str, ...]:
        if self.type == "histogram":
            return tuple(
                f"{self.name}_{suffix}" for suffix in ("bucket", "sum", "count")
            )
        return (self.name,)

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.type,
            "labels": list(self.labels),
            "unit": self.unit,
            "help": self.help,
        }


METRICS = (
    MetricSpec(
        "ai_journey_serving_recovery_attempts_total",
        "counter",
        ("outcome", SCOPE_LABEL),
        "Classified client attempts by outcome",
    ),
    MetricSpec(
        "ai_journey_serving_recovery_executions_total",
        "counter",
        ("lifetime", SCOPE_LABEL),
        "Committed journal executions by server lifetime",
    ),
    MetricSpec(
        "ai_journey_serving_recovery_duplicate_executions_total",
        "counter",
        (SCOPE_LABEL,),
        "At-least-once duplicate executions after a lost response",
    ),
    MetricSpec(
        "ai_journey_serving_recovery_detection_latency_seconds",
        "gauge",
        (SCOPE_LABEL,),
        "SIGKILL to the supervisor observing the exit",
    ),
    MetricSpec(
        "ai_journey_serving_recovery_restart_latency_seconds",
        "gauge",
        (SCOPE_LABEL,),
        "Observed exit to the replacement lifetime being ready",
    ),
    MetricSpec(
        "ai_journey_serving_recovery_unavailability_seconds",
        "gauge",
        (SCOPE_LABEL,),
        "First failed attempt after the kill to the first replacement success",
    ),
    MetricSpec(
        "ai_journey_serving_recovery_time_to_slo_recovery_seconds",
        "gauge",
        (SCOPE_LABEL,),
        "Kill to the end of the first run of consecutive in-SLO successes",
    ),
    MetricSpec(
        "ai_journey_serving_recovery_success_latency_seconds",
        "histogram",
        ("phase", SCOPE_LABEL),
        "Latency of successful client attempts by recovery phase",
    ),
    MetricSpec(
        "ai_journey_worker_recovery_recovery_seconds",
        "gauge",
        (SCOPE_LABEL,),
        "Worker fault injection to completed recovery",
    ),
    MetricSpec(
        "ai_journey_worker_recovery_lost_items_total",
        "counter",
        (SCOPE_LABEL,),
        "Uncheckpointed work items lost to the fault and replayed",
    ),
)
METRIC_NAMES = tuple(metric.name for metric in METRICS)
SERIES = frozenset(series for metric in METRICS for series in metric.series)
