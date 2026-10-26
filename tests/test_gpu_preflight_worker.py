"""Worker observations bind real rank identity to recomputable admission."""

from copy import deepcopy
from uuid import uuid4

import pytest

from ai_journey.gpu_plan import gpu_record_digest
from ai_journey.gpu_preflight import GpuTopology
from ai_journey.gpu_preflight_evidence import verify_gpu_rank_evidence
from ai_journey.gpu_preflight_io import read_gpu_json, write_gpu_json
from ai_journey.gpu_preflight_worker import run_rank_preflight, main
from test_gpu_plan import resource_report, simulated_probe


def rank_record(monkeypatch):
    for name, value in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "ai_journey.gpu_preflight_worker.probe_gpu_host", simulated_probe
    )
    return run_rank_preflight(
        topology=GpuTopology((0, 1)),
        resource_report=resource_report(),
        run_id=str(uuid4()),
    )


def test_worker_binds_actual_process_rank_and_observations(monkeypatch):
    import os

    record = rank_record(monkeypatch)
    assert record["process_id"] == os.getpid()
    assert record["rank"]["device_ordinal"] == 0
    assert record["admission"]["preflight_ready"] is True
    assert record["training_executed"] is False
    assert verify_gpu_rank_evidence(record) == record


@pytest.mark.parametrize("corruption", ["device", "memory", "training", "topology"])
def test_resealed_wrong_provenance_is_rejected(monkeypatch, corruption):
    record = deepcopy(rank_record(monkeypatch))
    if corruption == "device":
        record["rank"]["device_ordinal"] = 1
    elif corruption == "memory":
        record["probe"]["devices"][0]["free_bytes"] = 0
    elif corruption == "training":
        record["training_executed"] = True
    else:
        record["topology"]["devices"] = [1, 0]
    record["evidence_sha256"] = gpu_record_digest(
        {key: value for key, value in record.items() if key != "evidence_sha256"}
    )
    with pytest.raises(ValueError):
        verify_gpu_rank_evidence(record)


def test_worker_writes_blocked_observation_before_failure_exit(tmp_path, monkeypatch):
    record = rank_record(monkeypatch)
    blocked = deepcopy(record["probe"])
    blocked.update(available=False, reason="cuda_unavailable", devices=[])
    monkeypatch.setattr(
        "ai_journey.gpu_preflight_worker.probe_gpu_host", lambda: blocked
    )
    source = tmp_path / "resources.json"
    write_gpu_json(source, resource_report())
    run_id = str(uuid4())
    assert (
        main(
            [
                "--devices",
                "0",
                "1",
                "--dtype",
                "bfloat16",
                "--resource-report",
                str(source),
                "--output-dir",
                str(tmp_path / "ranks"),
                "--run-id",
                run_id,
            ]
        )
        == 2
    )
    observed = read_gpu_json(tmp_path / "ranks" / run_id / "rank-0.json")
    assert observed["admission"]["preflight_ready"] is False
    assert verify_gpu_rank_evidence(observed) == observed


def test_invalid_rank_environment_prevents_host_probe(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)

    def unexpected():
        raise AssertionError("probe must not run")

    monkeypatch.setattr("ai_journey.gpu_preflight_worker.probe_gpu_host", unexpected)
    with pytest.raises(ValueError, match="RANK"):
        run_rank_preflight(
            topology=GpuTopology((0, 1)),
            resource_report=resource_report(),
            run_id=str(uuid4()),
        )


def test_record_io_rejects_duplicate_fields_and_symlink_inputs(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"rank":0,"rank":1}')
    with pytest.raises(ValueError, match="duplicate"):
        read_gpu_json(path)
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        read_gpu_json(link)
