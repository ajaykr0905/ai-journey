"""Dry-run launch plans and collected evidence cannot manufacture GPU readiness."""

from copy import deepcopy
from uuid import uuid4
from pathlib import Path
import os
import subprocess
import sys
import tomllib

import pytest

from ai_journey.gpu_plan import gpu_record_digest
from ai_journey.gpu_preflight import GpuTopology
from ai_journey.gpu_preflight_cli import main
from ai_journey.gpu_preflight_evidence import (
    build_gpu_run_evidence,
    verify_gpu_run_evidence,
)
from ai_journey.gpu_preflight_io import read_gpu_json, write_gpu_json
from ai_journey.gpu_preflight_launch import build_gpu_launch_plan
from test_gpu_plan import resource_report, simulated_probe
from test_gpu_preflight_worker import rank_record


def two_rank_records(monkeypatch):
    first = rank_record(monkeypatch)
    second = deepcopy(first)
    second["process_id"] += 1
    second["rank"].update(rank=1, local_rank=1, device_ordinal=1)
    second["evidence_sha256"] = gpu_record_digest(
        {key: value for key, value in second.items() if key != "evidence_sha256"}
    )
    return [first, second]


def test_launch_plan_targets_only_the_implemented_preflight_worker(tmp_path):
    path = tmp_path / "resource input.json"
    write_gpu_json(path, resource_report())
    plan = build_gpu_launch_plan(
        resource_report_path=path,
        topology=GpuTopology((0, 1)),
        output_dir=tmp_path / "rank output",
        run_id=str(uuid4()),
    )
    argv = plan["argv"]
    assert argv[1:3] == ["-m", "torch.distributed.run"]
    assert "--standalone" in argv and "--max-restarts=0" in argv
    assert "ai_journey.gpu_preflight_worker" in argv
    assert str(path.resolve()) in argv
    assert not any("ddp_worker" in part for part in argv)
    assert plan["admission"]["preflight_ready"] is False
    assert plan["command_executed"] is plan["training_executed"] is False


def test_collect_requires_every_rank_and_consistent_runtime_provenance(monkeypatch):
    records = two_rank_records(monkeypatch)
    kwargs = dict(
        run_id=records[0]["run_id"],
        topology=GpuTopology((0, 1)),
        resource_report=resource_report(),
    )
    payload = build_gpu_run_evidence(records, **kwargs)
    assert payload["preflight_passed"] is True
    assert verify_gpu_run_evidence(payload) == payload
    with pytest.raises(ValueError, match="every requested rank"):
        build_gpu_run_evidence(records[:1], **kwargs)
    for field, value in (
        ("run_id", str(uuid4())),
        ("process_id", records[0]["process_id"]),
    ):
        changed = deepcopy(records)
        changed[1][field] = value
        changed[1]["evidence_sha256"] = gpu_record_digest(
            {key: item for key, item in changed[1].items() if key != "evidence_sha256"}
        )
        with pytest.raises(ValueError):
            build_gpu_run_evidence(changed, **kwargs)


def test_resealed_gpu_execution_claim_is_rejected(monkeypatch):
    records = two_rank_records(monkeypatch)
    payload = build_gpu_run_evidence(
        records,
        run_id=records[0]["run_id"],
        topology=GpuTopology((0, 1)),
        resource_report=resource_report(),
    )
    payload["gpu_kernel_executed"] = True
    payload["evidence_sha256"] = gpu_record_digest(
        {key: value for key, value in payload.items() if key != "evidence_sha256"}
    )
    with pytest.raises(ValueError, match="rank observations"):
        verify_gpu_run_evidence(payload)


def test_cli_plans_collects_and_verifies_without_launching(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("ai_journey.gpu_preflight_cli.probe_gpu_host", simulated_probe)
    records = two_rank_records(monkeypatch)
    run_id = records[0]["run_id"]
    resource = tmp_path / "resources.json"
    write_gpu_json(resource, resource_report())
    common = [
        "--resource-report",
        str(resource),
        "--devices",
        "0",
        "1",
        "--output-dir",
        str(tmp_path / "ranks"),
        "--run-id",
        run_id,
    ]
    assert main(["plan", *common, "--output", str(tmp_path / "plan.json")]) == 0
    assert read_gpu_json(tmp_path / "plan.json")["command_executed"] is False
    for rank, record in enumerate(records):
        write_gpu_json(tmp_path / "ranks" / run_id / f"rank-{rank}.json", record)
    assert main(["collect", *common, "--output", str(tmp_path / "run.json")]) == 0
    assert main(["verify", "--report", str(tmp_path / "run.json")]) == 0
    capsys.readouterr()


def test_cli_rejects_input_output_collision_before_replacing_source(tmp_path):
    path = tmp_path / "resources.json"
    write_gpu_json(path, resource_report())
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        main(
            [
                "plan",
                "--resource-report",
                str(path),
                "--devices",
                "0",
                "1",
                "--output-dir",
                str(tmp_path),
                "--output",
                str(path),
            ]
        )
    assert path.read_bytes() == original


def test_console_entry_and_fresh_process_module_are_usable():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert (
        metadata["project"]["scripts"]["ai-journey-gpu-preflight"]
        == "ai_journey.gpu_preflight_cli:main"
    )
    result = subprocess.run(
        [sys.executable, "-m", "ai_journey.gpu_preflight_cli", "--help"],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert "{probe,plan,collect,verify}" in result.stdout
