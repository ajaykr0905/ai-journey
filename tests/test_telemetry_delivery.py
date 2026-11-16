import contextlib
import io
import json
import re
import shlex
import tomllib
from pathlib import Path

import pytest

from ai_journey import (
    recovery_cli,
    serving_recovery_cli,
    telemetry_metrics_cli,
    telemetry_traces_cli,
)
from ai_journey.telemetry_dashboard_metrics import METRICS
from ai_journey.telemetry_metrics_exposition import flatten_samples, parse_exposition
from ai_journey.telemetry_metrics_model import LAB_SCOPE, SCOPE_LABEL
from ai_journey.telemetry_metrics_report import EVIDENCE_BOUNDARY
from ai_journey.telemetry_traces_lab import TELEMETRY_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Export recovery telemetry as metrics, traces and a dashboard\n"
METRICS_GUIDE = ROOT / "docs/telemetry-metrics.md"
TRACES_GUIDE = ROOT / "docs/telemetry-traces-dashboards.md"
CLIS = {
    "ai-journey-serving-recovery": serving_recovery_cli,
    "ai-journey-recovery": recovery_cli,
    "ai-journey-telemetry-metrics": telemetry_metrics_cli,
    "ai-journey-telemetry-traces": telemetry_traces_cli,
}
MODULE_FORM = ["PYTHONPATH=src", "python", "-m"]
DEMO = "/tmp/day-73/"


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def documented_commands(text):
    """Each documented lab command as its installed name and arguments."""

    installed = {cli.__name__: command for command, cli in CLIS.items()}
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[:3] == MODULE_FORM:
                words = [installed[words[3]], *words[4:]]
            if words and words[0] in CLIS:
                commands.append((words[0], words[1:]))
    return commands


def localized(argv, base):
    """Move the documented /tmp/day-73 paths under base and resolve the rest."""

    return [
        str(base / word.removeprefix(DEMO))
        if word.startswith(DEMO)
        else str(ROOT / word)
        if word.startswith(("config/", "data/"))
        else word
        for word in argv
    ]


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    """Run the README commands in order through the packaged CLIs."""

    base = tmp_path_factory.mktemp("telemetry-readme")
    results = []
    for command, argv in documented_commands(readme_section()):
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            code = CLIS[command].main(localized(argv, base))
        assert output.getvalue(), (command, argv[0], code, errors.getvalue())
        results.append((code, json.loads(output.getvalue())))
    return base, results


def test_readme_runs_the_guides_commands_as_installed_scripts():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = documented_commands(readme_section())
    metrics_guide = documented_commands(METRICS_GUIDE.read_text(encoding="utf-8"))
    traces_guide = documented_commands(TRACES_GUIDE.read_text(encoding="utf-8"))

    for command, cli in CLIS.items():
        assert project["project"]["scripts"][command] == f"{cli.__name__}:main"
    assert traces_guide[:2] == metrics_guide[:2]
    assert readme == metrics_guide + traces_guide[2:]
    assert [(command, argv[0]) for command, argv in readme] == [
        ("ai-journey-serving-recovery", "run"),
        ("ai-journey-recovery", "run"),
        ("ai-journey-telemetry-metrics", "export"),
        ("ai-journey-telemetry-metrics", "verify"),
        ("ai-journey-telemetry-traces", "export"),
        ("ai-journey-telemetry-traces", "dashboard"),
        ("ai-journey-telemetry-traces", "verify"),
        ("ai-journey-telemetry-traces", "verify"),
    ]


def test_documented_flow_exports_and_verifies_every_artifact(flow):
    _, results = flow
    codes = [code for code, _ in results]
    receipts = [receipt for _, receipt in results]
    serving, worker, metrics, metrics_verified = receipts[:4]
    traces, dashboard, traces_verified, dashboard_verified = receipts[4:]

    assert codes[:2] == [0, 0]
    # Timings depend on this host's load, so a breach is reported, never hidden.
    assert codes[2] in (0, 3)
    assert (codes[2] == 3) == bool(metrics["breached_objectives"])
    assert codes[3:] == [codes[2], 0, 0, 0, 0]
    assert metrics["verified"] is True and metrics_verified == metrics
    assert traces["verified"] is True and dashboard["verified"] is True
    assert (traces_verified, dashboard_verified) == (traces, dashboard)
    assert [trace["run_id"] for trace in traces["traces"]] == [
        serving["run_id"],
        worker["experiment"]["run_id"],
    ]


def test_dashboard_queries_only_series_the_metrics_export_writes(flow):
    base, _ = flow
    exposition = (base / "telemetry" / "metrics.prom").read_text(encoding="utf-8")
    metrics = parse_exposition(exposition)
    dashboard = json.loads((base / "dashboard" / "dashboard.json").read_bytes())
    families = {family.name: family for family in metrics.families}
    written = {name for family in metrics.families for name in family.sample_names}
    queried = {
        name
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        for name in re.findall(r"\bai_journey_[a-z0-9_]+", target["expr"])
    }

    assert queried and queried <= written
    for metric in METRICS:
        family = families[metric.name]
        assert family.kind == metric.type, metric.name
        assert not family.series or family.label_names == tuple(sorted(metric.labels))
    for sample in flatten_samples(metrics):
        assert sample.label_map()[SCOPE_LABEL] == LAB_SCOPE, sample.name


def test_readme_states_the_outputs_exit_codes_and_evidence_boundary(flow):
    _, results = flow
    metrics, dashboard = results[2][1], results[5][1]
    section = " ".join(readme_section().split())

    for phrase in (
        "[telemetry metrics guide](docs/telemetry-metrics.md)",
        "[telemetry traces and dashboards guide]"
        "(docs/telemetry-traces-dashboards.md)",
        "Prometheus text exposition format 0.0.4",
        f"{metrics['families']} families with every series labeled "
        f'`{SCOPE_LABEL}="{LAB_SCOPE}"`',
        f"{metrics['recording_rules']} recording and "
        f"{metrics['alerting_rules']} alerting rules",
        f"(schemaVersion {dashboard['schema_version']}, {dashboard['panels']} "
        f"panels, {dashboard['targets']} queries) whose queries select only "
        "series the metrics export writes",
        "`ai-journey-telemetry-metrics` exits 0 when the export verifies and "
        "every declared objective is met, 1 when `verify` rejects an export or "
        "its evidence, 2 for usage errors, invalid inputs, refused outputs or "
        "evidence that `export` rejects, and 3 when the export verifies but an "
        "objective is breached",
        "`ai-journey-telemetry-traces` exits 0 on success, 1 for rejected "
        "evidence or a rejected stored artifact, and 2 for usage errors, "
        "invalid inputs or a refused output",
        "CI job `telemetry-cpu` runs these commands from an installed wheel "
        "without torch, NumPy or matplotlib",
    ):
        assert phrase in section, phrase
    for claim, boundary in (
        (
            "objectives are declared lab controls, not service-level agreements",
            EVIDENCE_BOUNDARY,
        ),
        (
            "This is not GPU, CUDA, distributed, multi-host or production "
            "monitoring evidence",
            EVIDENCE_BOUNDARY,
        ),
        (
            "Durations are measured monotonic-clock differences; absolute "
            "timestamps come from a declared anchor and are synthetic.",
            TELEMETRY_BOUNDARY,
        ),
    ):
        assert claim in section and claim in " ".join(boundary.split()), claim
    assert (
        telemetry_metrics_cli.EXIT_OK,
        telemetry_metrics_cli.EXIT_REJECTED,
        telemetry_metrics_cli.EXIT_INVALID,
        telemetry_metrics_cli.EXIT_OBJECTIVE_BREACHED,
    ) == (0, 1, 2, 3)
    assert (
        telemetry_traces_cli.EXIT_OK,
        telemetry_traces_cli.EXIT_REJECTED,
        telemetry_traces_cli.EXIT_INVALID,
    ) == (0, 1, 2)
