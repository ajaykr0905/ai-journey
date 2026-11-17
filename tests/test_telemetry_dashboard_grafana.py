import re
from dataclasses import replace
from pathlib import Path

import pytest

from ai_journey.telemetry_dashboard_grafana import (
    DASHBOARD_BOUNDARY,
    DATASOURCE,
    build_dashboard,
)
from ai_journey.telemetry_dashboard_metrics import (
    METRIC_NAMES,
    METRICS,
    SERIES,
    MetricSpec,
)
from ai_journey.telemetry_dashboard_promql import check_promql
from ai_journey.telemetry_traces_config import load_telemetry_config
from ai_journey.telemetry_traces_io import TelemetryError, artifact_bytes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_telemetry_config(ROOT / "config" / "day-73-telemetry-traces.json")
# The families the telemetry-metrics lab exports that this dashboard charts,
# with type and labels, as telemetry_metrics_serving.py and
# telemetry_metrics_worker.py name them; every series carries a scope label.
EXPOSED = {
    "ai_journey_serving_recovery_attempts_total": ("counter", ("outcome", "scope")),
    "ai_journey_serving_recovery_executions_total": (
        "counter",
        ("lifetime", "scope"),
    ),
    "ai_journey_serving_recovery_duplicate_executions_total": ("counter", ("scope",)),
    "ai_journey_serving_recovery_detection_latency_seconds": ("gauge", ("scope",)),
    "ai_journey_serving_recovery_restart_latency_seconds": ("gauge", ("scope",)),
    "ai_journey_serving_recovery_unavailability_seconds": ("gauge", ("scope",)),
    "ai_journey_serving_recovery_time_to_slo_recovery_seconds": ("gauge", ("scope",)),
    "ai_journey_serving_recovery_success_latency_seconds": (
        "histogram",
        ("phase", "scope"),
    ),
    "ai_journey_worker_recovery_recovery_seconds": ("gauge", ("scope",)),
    "ai_journey_worker_recovery_lost_items_total": ("counter", ("scope",)),
}
LATENCY = "ai_journey_serving_recovery_success_latency_seconds"


def test_catalog_names_the_families_the_metrics_lab_exports():
    assert METRIC_NAMES == tuple(EXPOSED)
    assert {m.name: (m.type, m.labels) for m in METRICS} == EXPOSED
    assert LATENCY not in SERIES and f"{LATENCY}_bucket" in SERIES
    by_name = {metric.name: metric for metric in METRICS}
    assert by_name["ai_journey_worker_recovery_recovery_seconds"].to_mapping() == {
        "name": "ai_journey_worker_recovery_recovery_seconds",
        "type": "gauge",
        "labels": ["scope"],
        "unit": "s",
        "help": "Worker fault injection to completed recovery",
    }


def test_every_query_reads_only_series_scoped_to_the_local_cpu_lab():
    selector = re.compile(r"(ai_journey_[a-z0-9_]+)(\{[^}]*\})?")

    for _, target in queries(build_dashboard(CONFIG)):
        found = selector.findall(target["expr"])
        assert found, target["expr"]
        assert all(matchers == '{scope="local_cpu_lab"}' for _, matchers in found)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (("serving_attempts_total", "counter", (), "Help"), "ai_journey_"),
        (("ai_journey_x_total", "summary", (), "Help"), "metric type"),
        (("ai_journey_x_seconds", "counter", (), "Help"), "_total"),
        (("ai_journey_x_total", "gauge", (), "Help"), "_total"),
        (("ai_journey_x_seconds", "histogram", ("le",), "Help"), "labels"),
        (("ai_journey_x_seconds", "gauge", ("Outcome",), "Help"), "labels"),
        (("ai_journey_x_seconds", "gauge", (), "lowercase help"), "help"),
    ],
)
def test_metric_specs_enforce_prometheus_naming(values, message):
    with pytest.raises(TelemetryError, match=message):
        MetricSpec(*values)


def queries(dashboard):
    return [
        (panel, target)
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]


def test_dashboard_is_deterministic_labeled_and_bound_to_the_config():
    dashboard = build_dashboard(CONFIG)

    assert artifact_bytes(dashboard) == artifact_bytes(build_dashboard(CONFIG))
    assert dashboard["schemaVersion"] == 39 and dashboard["id"] is None
    assert (dashboard["uid"], dashboard["title"]) == (
        CONFIG.dashboard_uid,
        CONFIG.dashboard_title,
    )
    assert dashboard["time"] == {"from": "now-6h", "to": "now"}
    assert dashboard["refresh"] == "1m"
    assert "local-cpu-lab" in dashboard["tags"]
    assert dashboard["description"] == DASHBOARD_BOUNDARY
    header = dashboard["panels"][0]
    assert (
        header["type"] == "text" and DASHBOARD_BOUNDARY in header["options"]["content"]
    )
    assert all(
        panel["description"].startswith("Local CPU lab telemetry.")
        for panel in dashboard["panels"]
    )
    (variable,) = dashboard["templating"]["list"]
    assert (variable["type"], variable["name"], variable["query"]) == (
        "datasource",
        "datasource",
        "prometheus",
    )


def test_every_catalog_metric_is_charted_through_the_datasource_variable():
    dashboard = build_dashboard(CONFIG)
    referenced = set()
    by_series = {series: metric for metric in METRICS for series in metric.series}

    for panel, target in queries(dashboard):
        assert panel["datasource"] == target["datasource"] == DATASOURCE
        summary = check_promql(target["expr"], allowed_metrics=SERIES)
        for series in summary.metrics:
            metric = by_series[series]
            referenced.add(metric.name)
            assert panel["fieldConfig"]["defaults"]["unit"] == metric.unit
    assert referenced == set(METRIC_NAMES)
    assert [panel["id"] for panel in dashboard["panels"]] == list(range(1, 8))


def test_latency_quantiles_follow_the_config():
    config = replace(CONFIG, latency_quantiles=(0.9, 0.999))
    (panel,) = [
        p for p in build_dashboard(config)["panels"] if p["type"] == "timeseries"
    ]

    assert [t["legendFormat"] for t in panel["targets"]] == ["p90", "p99.9"]
    assert [t["refId"] for t in panel["targets"]] == ["A", "B"]
    assert panel["targets"][1]["expr"] == (
        "histogram_quantile(0.999, sum by (le) "
        f'({LATENCY}_bucket{{scope="local_cpu_lab"}}))'
    )


def test_returned_dashboards_do_not_share_mutable_state():
    first = build_dashboard(CONFIG)
    first["panels"][1]["options"]["reduceOptions"]["calcs"].append("mean")

    assert build_dashboard(CONFIG)["panels"][1]["options"]["reduceOptions"][
        "calcs"
    ] == ["lastNotNull"]
    with pytest.raises(TypeError, match="TelemetryConfig"):
        build_dashboard(CONFIG.to_mapping())
