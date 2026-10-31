import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from ai_journey.zero_probe import (
    ZeroProbeError,
    load_zero_probe_report,
    main,
    seal_zero_probe_report,
    verify_zero_probe_report,
)


def test_two_rank_gloo_probe_executes_equivalent_collectives(tmp_path: Path, capsys):
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "zero-probe.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node",
            "2",
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            str(master_port),
            "-m",
            "ai_journey.zero_probe",
            "--output",
            str(output),
        ],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert verify_zero_probe_report(report) is report
    assert report["kind"] == "cpu_gloo_zero_collective_conformance"
    assert report["executed"] is True
    assert report["performance_measured"] is False
    assert report["backend"] == "gloo"
    assert report["world_size"] == 2
    assert report["expected_sum"] == [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]
    assert all(rank["all_reduce"] == report["expected_sum"] for rank in report["ranks"])
    assert all(rank["all_gather"] == report["expected_sum"] for rank in report["ranks"])
    assert "backend=gloo" in completed.stdout
    assert "performance_measured=false" in completed.stdout

    tampered = json.loads(json.dumps(report))
    tampered["ranks"][0]["all_gather"][0] = -1.0
    tampered = seal_zero_probe_report(tampered)
    with pytest.raises(ZeroProbeError, match="collective outputs"):
        verify_zero_probe_report(tampered)

    assert main(["--verify-report", str(output)]) == 0
    assert "verified_zero_probe=" in capsys.readouterr().out


def test_zero_probe_loader_rejects_duplicate_fields(tmp_path: Path):
    report = tmp_path / "probe.json"
    report.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    with pytest.raises(ZeroProbeError, match="duplicate"):
        load_zero_probe_report(report)
