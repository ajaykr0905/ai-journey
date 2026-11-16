import copy
from pathlib import Path

import pytest

from ai_journey.telemetry_dashboard_grafana import build_dashboard
from ai_journey.telemetry_dashboard_metrics import SERIES
from ai_journey.telemetry_dashboard_validate import (
    DashboardValidationError,
    validate_dashboard,
)
from ai_journey.telemetry_traces_config import load_telemetry_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_telemetry_config(ROOT / "config" / "day-73-telemetry-traces.json")


def dashboard():
    return copy.deepcopy(build_dashboard(CONFIG))


def test_the_generated_dashboard_validates_and_summarizes_its_queries():
    summary = validate_dashboard(dashboard())

    assert summary.to_mapping() == {
        "uid": "ai-journey-day-73-recovery",
        "schema_version": 39,
        "panels": 7,
        "targets": 1 + 2 + 4 + 3 + 1 + 1,
        "series": sorted(
            SERIES
            - {
                "ai_journey_serving_recovery_success_latency_seconds_count",
                "ai_journey_serving_recovery_success_latency_seconds_sum",
            }
        ),
        "functions": ["histogram_quantile", "max", "sum"],
    }


def panel(index, **changes):
    return lambda d: d["panels"][index].update(changes)


def target(panel_index, target_index, **changes):
    return lambda d: d["panels"][panel_index]["targets"][target_index].update(changes)


def grid(index, **changes):
    return lambda d: d["panels"][index]["gridPos"].update(changes)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (panel(2, id=2), "repeats panel id 2"),
        (panel(2, id=0), r"id must be an integer in \[1"),
        (panel(2, id=True), "id must be an integer"),
        (
            panel(2, title="Serving client attempts by outcome"),
            "unique printable title",
        ),
        (panel(2, type="graph"), "unknown type graph"),
        (panel(0, targets=[]), "fields do not match a text panel"),
        (lambda d: d["panels"][2].pop("fieldConfig"), "fields do not match a stat"),
        (grid(2, x=11), "overlaps panel 2"),
        (grid(6, y=20), "overlaps panel 6"),
        (grid(2, x=13), "extends past 24 columns"),
        (grid(2, w=0), r"w must be an integer in \[1, 24\]"),
        (grid(2, y=-1), "y must be an integer"),
        (grid(2, h=1.5), "h must be an integer"),
        (lambda d: d["panels"][2]["gridPos"].pop("h"), "exactly x, y, w and h"),
        (panel(2, datasource={"type": "prometheus", "uid": "${other}"}), "declared"),
        (panel(2, datasource={"type": "loki", "uid": "${datasource}"}), "Prometheus"),
        (panel(2, datasource={"type": "prometheus", "uid": "P1"}), "declared"),
        (
            target(2, 0, datasource={"type": "prometheus", "uid": "${spare}"}),
            "declared",
        ),
        (target(2, 1, refId="A"), "unique uppercase refId"),
        (target(2, 0, refId="a"), "unique uppercase refId"),
        (
            target(2, 0, expr="sum(ai_journey_serving_recovery_executions_total"),
            "unclosed",
        ),
        (
            target(2, 0, expr="summ(ai_journey_serving_recovery_executions_total)"),
            "unknown PromQL function",
        ),
        (
            target(2, 0, expr="sum(ai_journey_serving_recovery_execution_total)"),
            "unknown metric",
        ),
        (
            target(
                4, 0, expr="max(ai_journey_serving_recovery_success_latency_seconds)"
            ),
            "unknown metric",
        ),
        (target(2, 0, interval="1m"), "fields do not match a Prometheus target"),
        (
            lambda d: d["panels"][2]["fieldConfig"]["defaults"].update(unit="furlongs"),
            "known unit",
        ),
        (lambda d: d.update(schemaVersion=38), "schemaVersion must be 39"),
        (lambda d: d.update(id=7), "null id"),
        (lambda d: d.update(uid="x" * 41), "uid is invalid"),
        (lambda d: d.update(title="Recovery telemetry"), "local CPU lab"),
        (lambda d: d.update(tags=["ai-journey"]), "local-cpu-lab"),
        (lambda d: d.update(rows=[]), "unknown field rows"),
        (lambda d: d.pop("templating"), "lacks templating"),
        (
            lambda d: d["templating"]["list"][0].update(type="query"),
            "datasource variable",
        ),
        (
            lambda d: d["templating"]["list"].append(d["templating"]["list"][0]),
            "repeats variable",
        ),
        (lambda d: d["templating"]["list"].clear(), "declared datasource variable"),
        (
            lambda d: d["panels"][0]["options"].update(content="Recovery."),
            "text panel must state",
        ),
        (lambda d: d.update(panels=[]), "list of 1 to 64"),
    ],
)
def test_broken_dashboards_are_rejected_with_their_location(mutate, message):
    broken = dashboard()
    mutate(broken)
    with pytest.raises(DashboardValidationError, match=message):
        validate_dashboard(broken)


def test_a_wider_allowed_series_set_is_the_callers_choice():
    custom = dashboard()
    custom["panels"][2]["targets"][0]["expr"] = "sum(up)"

    with pytest.raises(DashboardValidationError, match="unknown metric 'up'"):
        validate_dashboard(custom)
    assert "up" in validate_dashboard(custom, allowed_series=SERIES | {"up"}).series
    with pytest.raises(DashboardValidationError, match="JSON object"):
        validate_dashboard([custom])
