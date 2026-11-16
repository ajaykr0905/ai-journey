"""Grafana dashboard-as-code for the recovery labs' local CPU telemetry.

``build_dashboard`` renders Grafana dashboard JSON (the dashboard model with
``schemaVersion`` 39) from the pinned config and the shared metric catalog,
so the same config always yields the same bytes. Every query panel reads the
Prometheus datasource chosen through the ``datasource`` template variable,
and every catalog metric appears in at least one panel: client attempt
outcomes, executions and duplicates, the detection, restart, unavailability
and SLO recovery durations, request latency quantiles and the worker lab's
recovery time and lost items. The dashboard is titled, tagged, described and
headed as local CPU lab telemetry.

Latency quantiles come from ``histogram_quantile`` over cumulative bucket
counts, so they are bucket-interpolated estimates; the labs' sealed reports
keep the exact nearest-rank percentiles. Durations use ``max`` so a scrape of
several series still shows one value per panel. The JSON has not been
imported into a running Grafana as part of this lab.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .telemetry_traces_config import TelemetryConfig

DASHBOARD_SCHEMA_VERSION = 39
GRID_COLUMNS = 24
DATASOURCE_VARIABLE = "datasource"
DATASOURCE = {"type": "prometheus", "uid": "${" + DATASOURCE_VARIABLE + "}"}
TAGS = ("ai-journey", "local-cpu-lab", "recovery-telemetry")
DASHBOARD_BOUNDARY = (
    "Local CPU lab telemetry: panels read metrics derived from sealed "
    "recovery-lab evidence recorded on one CPU host. This is not production, "
    "GPU, CUDA, distributed or multi-host monitoring, and it makes no SLO, "
    "capacity or alerting claim."
)


@dataclass(frozen=True)
class _Panel:
    title: str
    type: str
    grid: tuple[int, int, int, int]
    description: str
    targets: tuple[tuple[str, str], ...] = ()
    unit: str = "short"


def _quantile_label(quantile: float) -> str:
    return f"p{quantile * 100:g}"


def _panels(config: TelemetryConfig) -> tuple[_Panel, ...]:
    latency = "ai_journey_serving_request_latency_seconds_bucket"
    durations = (
        ("ai_journey_serving_detection_seconds", "fault detection"),
        ("ai_journey_serving_restart_seconds", "server restart"),
        ("ai_journey_serving_unavailability_seconds", "client unavailability"),
        ("ai_journey_serving_slo_recovery_seconds", "SLO recovery"),
    )
    return (
        _Panel(
            "Local CPU lab telemetry",
            "text",
            (0, 0, 24, 4),
            "Evidence boundary of every panel on this dashboard.",
        ),
        _Panel(
            "Serving client attempts by outcome",
            "bargauge",
            (0, 4, 12, 8),
            "Classified client attempts of the serving-recovery lab, by outcome.",
            (("sum by (outcome) (ai_journey_serving_attempts_total)", "{{outcome}}"),),
        ),
        _Panel(
            "Serving executions and duplicates",
            "stat",
            (12, 4, 12, 8),
            "Committed executions and at-least-once duplicate executions.",
            (
                ("sum(ai_journey_serving_executions_total)", "executions"),
                (
                    "sum(ai_journey_serving_duplicate_executions_total)",
                    "duplicate executions",
                ),
            ),
        ),
        _Panel(
            "Serving recovery durations",
            "stat",
            (0, 12, 24, 6),
            "Measured monotonic durations of the last verified serving run.",
            tuple((f"max({name})", label) for name, label in durations),
            unit="s",
        ),
        _Panel(
            "Serving request latency quantiles",
            "timeseries",
            (0, 18, 16, 9),
            "histogram_quantile estimates from cumulative latency buckets; the "
            "sealed report keeps exact nearest-rank percentiles.",
            tuple(
                (
                    f"histogram_quantile({quantile!r}, sum by (le) ({latency}))",
                    _quantile_label(quantile),
                )
                for quantile in config.latency_quantiles
            ),
            unit="s",
        ),
        _Panel(
            "Worker recovery time",
            "stat",
            (16, 18, 8, 5),
            "Fault injection to completed recovery of the worker-recovery lab.",
            (("max(ai_journey_worker_recovery_seconds)", "worker recovery"),),
            unit="s",
        ),
        _Panel(
            "Worker items lost",
            "stat",
            (16, 23, 8, 4),
            "Uncheckpointed work items lost to the fault and then replayed.",
            (("sum(ai_journey_worker_items_lost_total)", "items lost"),),
        ),
    )


_REDUCE = {"calcs": ["lastNotNull"], "fields": "", "values": False}
_OPTIONS = {
    "stat": {
        "colorMode": "none",
        "graphMode": "none",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": _REDUCE,
        "textMode": "value_and_name",
    },
    "bargauge": {
        "displayMode": "basic",
        "orientation": "horizontal",
        "reduceOptions": _REDUCE,
        "showUnfilled": True,
    },
    "timeseries": {
        "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
        "tooltip": {"mode": "multi", "sort": "none"},
    },
}


def _render(panel: _Panel, panel_id: int) -> dict[str, Any]:
    x, y, w, h = panel.grid
    rendered: dict[str, Any] = {
        "id": panel_id,
        "type": panel.type,
        "title": panel.title,
        "description": f"Local CPU lab telemetry. {panel.description}",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
    }
    if panel.type == "text":
        rendered["options"] = {
            "mode": "markdown",
            "content": f"**Local CPU lab telemetry.** {DASHBOARD_BOUNDARY}",
        }
        return rendered
    rendered["datasource"] = dict(DATASOURCE)
    rendered["fieldConfig"] = {"defaults": {"unit": panel.unit}, "overrides": []}
    rendered["options"] = copy.deepcopy(_OPTIONS[panel.type])
    rendered["targets"] = [
        {
            "refId": chr(ord("A") + index),
            "expr": expression,
            "legendFormat": legend,
            "datasource": dict(DATASOURCE),
        }
        for index, (expression, legend) in enumerate(panel.targets)
    ]
    return rendered


def build_dashboard(config: TelemetryConfig) -> dict[str, Any]:
    """Render the recovery telemetry dashboard JSON for one pinned config."""

    if not isinstance(config, TelemetryConfig):
        raise TypeError("config must be TelemetryConfig")
    return {
        "annotations": {"list": []},
        "description": DASHBOARD_BOUNDARY,
        "editable": False,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": [
            _render(panel, index) for index, panel in enumerate(_panels(config), 1)
        ],
        "refresh": config.dashboard_refresh,
        "schemaVersion": DASHBOARD_SCHEMA_VERSION,
        "tags": list(TAGS),
        "templating": {
            "list": [
                {
                    "current": {},
                    "hide": 0,
                    "includeAll": False,
                    "label": "Prometheus",
                    "multi": False,
                    "name": DATASOURCE_VARIABLE,
                    "options": [],
                    "query": "prometheus",
                    "refresh": 1,
                    "regex": "",
                    "type": "datasource",
                }
            ]
        },
        "time": {"from": config.dashboard_time_from, "to": "now"},
        "timepicker": {},
        "timezone": "utc",
        "title": config.dashboard_title,
        "uid": config.dashboard_uid,
        "version": 1,
    }
