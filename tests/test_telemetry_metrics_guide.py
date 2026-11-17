from pathlib import Path

import pytest
from test_telemetry_metrics_serving import make_serving_run
from test_telemetry_metrics_worker import make_worker_run

from ai_journey.telemetry_metrics_config import load_telemetry_metrics_config
from ai_journey.telemetry_metrics_promql import render
from ai_journey.telemetry_metrics_report import build_metrics_export
from ai_journey.telemetry_metrics_rules import (
    AlertingRule,
    RecordingRule,
    build_rule_file,
)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "telemetry-metrics.md"
PINNED = ROOT / "config" / "day-73-telemetry-metrics.json"


@pytest.fixture(scope="module")
def guide():
    return DOC.read_text(encoding="utf-8")


def test_guide_states_commands_exit_codes_and_the_evidence_boundary(guide):
    for phrase in (
        "python -m ai_journey.serving_recovery_cli run",
        "python -m ai_journey.recovery_cli run",
        "python -m ai_journey.telemetry_metrics_cli export",
        "python -m ai_journey.telemetry_metrics_cli verify",
        "config/day-70-serving-recovery.json",
        "config/day-70-worker-recovery.json",
        "config/day-73-telemetry-metrics.json",
        "| 0 | The export verified and every declared objective was met. |",
        "| 1 | `verify` rejected the export or its evidence. |",
        "| 3 | The export verified, but a declared objective was breached",
        "text exposition format 0.0.4",
        "text/plain; version=0.0.4; charset=utf-8",
        'scope="local_cpu_lab"',
        "strconv.FormatFloat(v, 'g', -1, 64)",
        "without PyYAML",
        "declared lab controls",
        "not GPU, CUDA, distributed, multi-host or production monitoring",
        "not a benchmark",
        "## Limitations",
    ):
        assert phrase in guide, phrase


def test_guide_lists_every_exported_family(guide, tmp_path):
    config = load_telemetry_metrics_config(PINNED)
    export = build_metrics_export(
        config,
        serving_run_dir=make_serving_run(tmp_path),
        worker_run_dir=make_worker_run(tmp_path),
    )

    for family in export.metrics.families:
        assert f"| `{family.name}` | {family.kind} |" in guide, family.name


def test_guide_shows_every_generated_rule_and_pinned_threshold(guide):
    config = load_telemetry_metrics_config(PINNED)
    for rule in build_rule_file(config).rules:
        if type(rule) is RecordingRule:
            assert f"`{rule.record}`" in guide
            continue
        assert type(rule) is AlertingRule
        expression = render(rule.expr)
        row = f"| `{rule.alert}` | `{dict(rule.labels)['slo']}` |"
        assert row in guide, rule.alert
        if not expression.startswith("scope:"):
            assert f"{row} `{expression}` |" in guide, expression
    for name, value in config.to_mapping()["serving_objectives"].items():
        assert f"| `serving_objectives.{name}` | {value} |" in guide
    for name, value in config.to_mapping()["worker_objectives"].items():
        assert f"| `worker_objectives.{name}` | {value} |" in guide
