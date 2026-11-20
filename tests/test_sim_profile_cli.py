import json
import os
import subprocess
import sys

import pytest

from ai_journey import sim_profile_semantics
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.sim_profile_cli import EXIT_INVALID, EXIT_OK, EXIT_REJECTED, PROG, main
from test_sim_profile_fixtures import ROOT, round_targets, small_config

# A None entry in sys.modules makes every import of that name fail.
WITHOUT_TORCH_OR_NUMPY = """
import importlib
import sys
for name in ("torch", "numpy"):
    sys.modules[name] = None
sys.exit(importlib.import_module("ai_journey.sim_profile_cli").main(sys.argv[1:]))
"""


def write_config(path, rounds=()):
    path.write_text(json.dumps(small_config(rounds).to_mapping(), indent=2) + "\n")
    return path


@pytest.fixture(scope="module")
def loop(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sim-profile")
    base_config = write_config(folder / "base.json")
    baseline = folder / "baseline.json"
    assert main(["run", "--config", str(base_config), "--stage", "baseline", "--output", str(baseline)]) == EXIT_OK  # fmt: skip
    report = json.loads(baseline.read_text())
    (target,) = round_targets(report, 1)
    config = write_config(folder / "config.json", [{"target": target, "change": "One change."}])  # fmt: skip
    after = folder / "round-1.json"
    assert main(["run", "--config", str(config), "--stage", "round-1", "--output", str(after)]) == EXIT_OK  # fmt: skip
    rounds = folder / "rounds.json"
    return folder, config, baseline, after, rounds


def test_run_prints_hotspots_workloads_and_a_measured_summary(loop, capsys):
    folder, config, baseline, *_ = loop
    output = folder / "again.json"

    assert main(["run", "--config", str(config), "--stage", "baseline", "--output", str(output)]) == EXIT_OK  # fmt: skip
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("cpu_measurement stage=baseline load_before=")
    assert sum(" hotspot rank=" in line for line in lines) == 10
    assert sum(" workload name=" in line for line in lines) == 3
    assert lines[-1].startswith("sim_profile_snapshot=")
    assert "scope=cpu_wall_clock_this_host" in lines[-1]
    assert lines[-1].endswith(
        "gpu_executed=false cuda_compiled=false hardware_counters=false"
    )
    payload = output.read_bytes()
    assert payload == canonical_json_bytes(json.loads(payload)) + b"\n"


def test_rounds_and_verify_round_trip(loop, capsys):
    _, config, baseline, after, rounds = loop
    command = ["rounds", "--config", str(config), "--snapshot", str(baseline)]

    assert (
        main([*command, "--snapshot", str(after), "--output", str(rounds)]) == EXIT_OK
    )
    out = capsys.readouterr().out.splitlines()
    assert out[2].startswith("cpu_measurement round=1 target=")
    assert out[-1].startswith("sim_profile_rounds=")
    for report in (baseline, after, rounds):
        assert main(["verify", "--report", str(report), "--config", str(config)]) == 0
        assert (
            capsys.readouterr().out.splitlines()[-1].startswith("verified_sim_profile_")
        )


def test_tampered_and_noncanonical_reports_are_rejected(loop, capsys):
    folder, _, baseline, *_ = loop
    report = json.loads(baseline.read_text())
    report["summary"]["total_median_ns"] += 1
    tampered = folder / "tampered.json"
    tampered.write_bytes(canonical_json_bytes(report) + b"\n")
    pretty = folder / "pretty.json"
    pretty.write_text(json.dumps(json.loads(baseline.read_text()), indent=1))

    for path in (tampered, pretty):
        assert main(["verify", "--report", str(path)]) == EXIT_REJECTED
        assert "sim-profile verify rejected" in capsys.readouterr().err


def test_bad_inputs_and_aliased_outputs_exit_with_two(loop, capsys):
    folder, config, baseline, after, _ = loop
    cases = [
        ["run", "--config", str(config), "--stage", "round-2", "--output", str(folder / "x.json")],
        ["run", "--config", str(config), "--stage", "baseline", "--output", str(config)],
        ["rounds", "--config", str(config), "--snapshot", str(baseline), "--output", str(folder / "y.json")],
        ["rounds", "--config", str(config), "--snapshot", str(baseline), "--snapshot", str(after), "--output", str(after)],
        ["verify", "--report", str(folder / "missing.json")],
        ["verify", "--report", str(baseline), "--config", str(folder / "missing.json")],
    ]  # fmt: skip
    for argv in cases:
        assert main(argv) == EXIT_INVALID, argv
        assert "failed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stopped:
        main(["run", "--config", str(config)])
    assert stopped.value.code == 2


def test_a_tree_with_changed_misuse_records_is_rejected(loop, monkeypatch, capsys):
    folder, config, *_ = loop
    monkeypatch.setattr(sim_profile_semantics, "SEMANTICS_SHA256", "0" * 64)

    argv = ["run", "--config", str(config), "--stage", "baseline", "--output", str(folder / "z.json")]  # fmt: skip
    assert main(argv) == EXIT_REJECTED
    assert "changed a semantic" in capsys.readouterr().err
    assert not (folder / "z.json").exists()


def test_the_cli_runs_without_torch_or_numpy(loop):
    _, config, baseline, *_ = loop
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH_OR_NUMPY, "verify", "--report", str(baseline), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )  # fmt: skip

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[-1].startswith(
        "verified_sim_profile_snapshot="
    )
    assert PROG == "ai-journey-sim-profile"
