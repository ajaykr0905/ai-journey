import re
import shlex
from pathlib import Path

from ai_journey import recovery_cli, serving_recovery_cli, telemetry_traces_cli
from ai_journey.telemetry_dashboard_grafana import (
    DASHBOARD_BOUNDARY,
    DASHBOARD_SCHEMA_VERSION,
)
from ai_journey.telemetry_dashboard_metrics import METRICS
from ai_journey.telemetry_traces_config import load_telemetry_config
from ai_journey.telemetry_traces_lab import TELEMETRY_BOUNDARY
from ai_journey.telemetry_traces_otlp import TIME_MAPPING_FORMULA

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "telemetry-traces-dashboards.md"
PREFIX = ["PYTHONPATH=src", "python", "-m"]


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[:3] == PREFIX:
                commands.append((words[3], words[4:]))
    return commands


def flat(text):
    return " ".join(text.split())


def test_documented_commands_parse_and_chain_their_directories():
    commands = documented_commands(GUIDE.read_text(encoding="utf-8"))

    assert [module for module, _ in commands] == [
        "ai_journey.serving_recovery_cli",
        "ai_journey.recovery_cli",
    ] + ["ai_journey.telemetry_traces_cli"] * 4
    serving = serving_recovery_cli._parser().parse_args(commands[0][1])
    worker = recovery_cli._parser().parse_args(commands[1][1])
    export, dashboard, verify_traces, verify_dashboard = (
        telemetry_traces_cli.build_parser().parse_args(argv) for _, argv in commands[2:]
    )
    assert [export.command, dashboard.command] == ["export", "dashboard"]
    assert (export.serving_run_dir, export.worker_run_dir) == (
        serving.run_dir,
        worker.run_dir,
    )
    assert verify_traces.report == export.output_dir / "report.json"
    assert (verify_traces.serving_run_dir, verify_traces.worker_run_dir) == (
        serving.run_dir,
        worker.run_dir,
    )
    assert verify_dashboard.report == dashboard.output_dir / "report.json"
    assert verify_dashboard.serving_run_dir is None
    for lab in (serving, worker):
        assert (ROOT / lab.source).is_file() and (ROOT / lab.config).is_file()
    pinned = Path("config/day-73-telemetry-traces.json")
    assert export.config == dashboard.config == verify_traces.config == pinned
    load_telemetry_config(ROOT / pinned)


def test_guide_states_the_boundaries_mapping_catalog_and_exit_codes():
    guide = GUIDE.read_text(encoding="utf-8")
    text = flat(guide)
    config = load_telemetry_config(ROOT / "config" / "day-73-telemetry-traces.json")

    assert flat(TELEMETRY_BOUNDARY) in text
    assert flat(DASHBOARD_BOUNDARY) in text
    assert TIME_MAPPING_FORMULA in guide
    assert config.anchor_utc in text and str(config.anchor_unix_nano) in text
    assert f"`schemaVersion` {DASHBOARD_SCHEMA_VERSION}" in text
    for metric in METRICS:
        assert f"| `{metric.name}` | {metric.type} |" in guide
    assert (
        telemetry_traces_cli.EXIT_OK,
        telemetry_traces_cli.EXIT_REJECTED,
        telemetry_traces_cli.EXIT_INVALID,
    ) == (0, 1, 2)
    for row in (
        "| 0 | The artifact was written or verified. |",
        "| 1 | Source evidence or a stored artifact was rejected. |",
        "| 2 | Usage errors, unreadable or invalid inputs, or a refused output. |",
    ):
        assert row in guide
