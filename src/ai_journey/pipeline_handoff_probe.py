"""Execute one bounded pipeline activation handoff over local CPU/Gloo ranks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Literal

import torch
import torch.distributed as dist

from .activation_handoff import (
    MAX_ACTIVATION_CONTRACT_BYTES,
    ActivationHandoffContract,
)
from .hybrid_topology import HybridParallelTopology
from .pipeline_partition import PipelinePartitionPlan, partition_transformer_layers


PROBE_SCHEMA = "ai-journey.pipeline-handoff-probe.v1"
_WORLD_SIZE = 2
_FAILURE_MODES = {
    None,
    "malformed_payload",
    "worker_exit",
    "startup_timeout",
    "completion_timeout",
}


class PipelineHandoffProbeError(RuntimeError):
    """Base error carrying observations collected after owned workers are reaped."""

    supervision: dict[str, object]


class PipelineHandoffWorkerFailure(PipelineHandoffProbeError):
    """Raised after either owned handoff worker exits unsuccessfully."""


class PipelineHandoffTimeout(PipelineHandoffProbeError, TimeoutError):
    """Raised after a bounded startup or completion deadline expires."""


@dataclass(frozen=True)
class PipelineHandoffProbeResult:
    """One agreed CPU/Gloo handoff result with an explicit evidence boundary."""

    contract: ActivationHandoffContract
    rank_outcomes: tuple[dict[str, object], ...]

    def to_mapping(self) -> dict[str, object]:
        """Return the public executed-conformance observation."""

        return {
            "schema": PROBE_SCHEMA,
            "executed": True,
            "performance_measured": False,
            "backend": "gloo",
            "device": "cpu",
            "world_size": _WORLD_SIZE,
            "outcome": "success",
            "contract": self.contract.to_mapping(),
            "rank_outcomes": [dict(item) for item in self.rank_outcomes],
            "limitations": [
                "CPU/Gloo transfer correctness only; no NVIDIA GPU or NCCL was used.",
                "No latency, bandwidth, throughput, utilization, memory, or cost was measured.",
            ],
        }


def _bounded_timeout(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 60
    ):
        raise ValueError(f"{name} must be finite and between zero and 60 seconds")
    return float(value)


def _admit_inputs(
    *,
    topology: HybridParallelTopology,
    partition: PipelinePartitionPlan,
    contract: ActivationHandoffContract,
    payload: bytes,
) -> ActivationHandoffContract:
    if not isinstance(topology, HybridParallelTopology):
        raise TypeError("topology must be a HybridParallelTopology")
    if not isinstance(partition, PipelinePartitionPlan):
        raise TypeError("partition must be a PipelinePartitionPlan")
    if not isinstance(contract, ActivationHandoffContract):
        raise TypeError("contract must be an ActivationHandoffContract")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return ActivationHandoffContract.from_mapping(
        contract.to_mapping(),
        topology=topology,
        partition=partition,
        payload=payload,
    )


def _rank_outcomes(receipts: list[dict[str, object]]) -> list[dict[str, object]]:
    outcomes = []
    for probe_rank, role in enumerate(("source", "destination")):
        matching = [item for item in receipts if item.get("probe_rank") == probe_rank]
        if len(matching) != 1 or matching[0].get("role") != role:
            continue
        outcome = matching[0].get("outcome")
        if outcome not in ("success", "failure"):
            continue
        outcomes.append({"probe_rank": probe_rank, "role": role, "outcome": outcome})
    return outcomes


def _load_receipts(directory: Path) -> list[dict[str, object]]:
    receipts = []
    for rank in range(_WORLD_SIZE):
        path = directory / f"rank-{rank}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            receipts.append(value)
    return receipts


def _supervision(
    processes: list[subprocess.Popen[bytes]],
    *,
    phase: str,
    receipts: list[dict[str, object]],
    directory: Path,
) -> dict[str, object]:
    rank_outcomes = _rank_outcomes(receipts)
    agreed = (
        rank_outcomes
        if len(rank_outcomes) == _WORLD_SIZE
        and len({item["outcome"] for item in rank_outcomes}) == 1
        else []
    )
    return {
        "schema": "ai-journey.pipeline-handoff-supervision.v1",
        "phase": phase,
        "outcome": "failure",
        "all_workers_reaped": all(process.poll() is not None for process in processes),
        "workers": [
            {
                "probe_rank": rank,
                "pid": process.pid,
                "returncode": process.returncode,
            }
            for rank, process in enumerate(processes)
        ],
        "rank_outcomes": agreed,
        "worker_logs": [
            (directory / f"rank-{rank}.log").read_text(
                encoding="utf-8", errors="replace"
            )[-8192:]
            for rank in range(len(processes))
        ],
    }


def _reap(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=3)


def _gloo_environment() -> dict[str, str]:
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    if "GLOO_SOCKET_IFNAME" not in environment:
        interfaces = {name for _, name in socket.if_nameindex()}
        for candidate in ("lo0", "lo"):
            if candidate in interfaces:
                environment["GLOO_SOCKET_IFNAME"] = candidate
                break
    return environment


def _wait_for_startup(
    processes: list[subprocess.Popen[bytes]],
    markers: list[Path],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        statuses = [process.poll() for process in processes]
        if any(status is not None for status in statuses):
            raise PipelineHandoffWorkerFailure(
                "a pipeline handoff worker exited during startup"
            )
        if all(marker.is_file() for marker in markers):
            return
        if time.monotonic() >= deadline:
            raise PipelineHandoffTimeout("pipeline handoff startup deadline exceeded")
        time.sleep(0.02)


def _wait_for_completion(
    processes: list[subprocess.Popen[bytes]], *, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        statuses = [process.poll() for process in processes]
        if any(status not in (None, 0) for status in statuses):
            # Give a peer that is publishing the same agreed failure a brief
            # chance to exit before the supervisor reaps any remaining child.
            grace = min(deadline, time.monotonic() + 0.5)
            while any(process.poll() is None for process in processes):
                if time.monotonic() >= grace:
                    break
                time.sleep(0.01)
            raise PipelineHandoffWorkerFailure("the pipeline handoff failed")
        if all(status == 0 for status in statuses):
            return
        if time.monotonic() >= deadline:
            raise PipelineHandoffTimeout(
                "pipeline handoff completion deadline exceeded"
            )
        time.sleep(0.02)


def run_pipeline_handoff_probe(
    *,
    topology: HybridParallelTopology,
    partition: PipelinePartitionPlan,
    contract: ActivationHandoffContract,
    payload: bytes,
    startup_timeout_seconds: float = 30,
    completion_timeout_seconds: float = 30,
    failure_mode: Literal[
        "malformed_payload",
        "worker_exit",
        "startup_timeout",
        "completion_timeout",
    ]
    | None = None,
) -> PipelineHandoffProbeResult:
    """Execute and supervise one admitted two-rank local CPU/Gloo transfer."""

    admitted = _admit_inputs(
        topology=topology,
        partition=partition,
        contract=contract,
        payload=payload,
    )
    startup_timeout = _bounded_timeout(
        startup_timeout_seconds, "startup_timeout_seconds"
    )
    completion_timeout = _bounded_timeout(
        completion_timeout_seconds, "completion_timeout_seconds"
    )
    if failure_mode not in _FAILURE_MODES:
        raise ValueError("failure_mode is unsupported")

    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    failure: PipelineHandoffProbeError | None = None
    phase = "startup"
    with tempfile.TemporaryDirectory(prefix="ai-journey-handoff-") as temporary:
        directory = Path(temporary)
        topology_path = directory / "topology.json"
        contract_path = directory / "contract.json"
        payload_path = directory / "payload.bin"
        topology_path.write_text(
            json.dumps(topology.to_mapping(), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        contract_path.write_bytes(admitted.to_json_bytes())
        payload_path.write_bytes(payload)
        try:
            for rank in range(_WORLD_SIZE):
                log = (directory / f"rank-{rank}.log").open("xb")
                logs.append(log)
                command = [
                    sys.executable,
                    "-m",
                    "ai_journey.pipeline_handoff_probe",
                    "--worker-rank",
                    str(rank),
                    "--rendezvous",
                    (directory / "rendezvous").as_uri(),
                    "--topology",
                    str(topology_path),
                    "--layer-count",
                    str(partition.layer_count),
                    "--contract",
                    str(contract_path),
                    "--payload",
                    str(payload_path),
                    "--started",
                    str(directory / f"rank-{rank}.started"),
                    "--output",
                    str(directory / f"rank-{rank}.json"),
                    "--process-group-timeout",
                    str(math.ceil(startup_timeout + completion_timeout + 5)),
                ]
                if failure_mode is not None:
                    command.extend(["--failure-mode", failure_mode])
                processes.append(
                    subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=_gloo_environment(),
                    )
                )
            _wait_for_startup(
                processes,
                [directory / f"rank-{rank}.started" for rank in range(_WORLD_SIZE)],
                timeout_seconds=startup_timeout,
            )
            phase = "completion"
            _wait_for_completion(processes, timeout_seconds=completion_timeout)
            receipts = _load_receipts(directory)
            rank_outcomes = _rank_outcomes(receipts)
            if len(rank_outcomes) != _WORLD_SIZE or any(
                item["outcome"] != "success" for item in rank_outcomes
            ):
                raise PipelineHandoffWorkerFailure(
                    "pipeline handoff receipts did not agree on success"
                )
            return PipelineHandoffProbeResult(admitted, tuple(rank_outcomes))
        except PipelineHandoffProbeError as error:
            failure = error
            raise
        finally:
            try:
                _reap(processes)
            finally:
                for log in logs:
                    log.close()
            if failure is not None:
                failure.supervision = _supervision(
                    processes,
                    phase=phase,
                    receipts=_load_receipts(directory),
                    directory=directory,
                )


def _send_bytes(payload: bytes, *, tag: int) -> None:
    length = torch.tensor([len(payload)], dtype=torch.int64)
    dist.send(length, dst=1, tag=tag)
    dist.send(torch.tensor(list(payload), dtype=torch.uint8), dst=1, tag=tag + 1)


def _receive_bytes(*, maximum_bytes: int, tag: int) -> bytes:
    length = torch.empty(1, dtype=torch.int64)
    dist.recv(length, src=0, tag=tag)
    size = int(length.item())
    if not 0 < size <= maximum_bytes:
        raise ValueError("received handoff field exceeds its byte bound")
    buffer = torch.empty(size, dtype=torch.uint8)
    dist.recv(buffer, src=0, tag=tag + 1)
    return bytes(buffer.tolist())


def _write_receipt(
    output: Path, *, probe_rank: int, outcome: str, message: str | None
) -> None:
    payload: dict[str, object] = {
        "schema": "ai-journey.pipeline-handoff-rank.v1",
        "probe_rank": probe_rank,
        "role": "source" if probe_rank == 0 else "destination",
        "outcome": outcome,
    }
    if message is not None:
        payload["message"] = message
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _run_worker(args: argparse.Namespace) -> int:
    if args.failure_mode == "startup_timeout" and args.worker_rank == 1:
        time.sleep(args.process_group_timeout + 5)
        return 1
    topology = HybridParallelTopology.from_mapping(
        json.loads(args.topology.read_text(encoding="utf-8"))
    )
    partition = partition_transformer_layers(topology, layer_count=args.layer_count)
    contract_bytes = args.contract.read_bytes()
    payload = args.payload.read_bytes()
    contract = ActivationHandoffContract.from_json_bytes(
        contract_bytes,
        topology=topology,
        partition=partition,
        payload=payload,
    )
    dist.init_process_group(
        "gloo",
        init_method=args.rendezvous,
        rank=args.worker_rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=args.process_group_timeout),
    )
    try:
        args.started.write_text("started\n", encoding="utf-8")
        if args.failure_mode == "worker_exit" and args.worker_rank == 1:
            os._exit(86)
        if args.failure_mode == "completion_timeout" and args.worker_rank == 1:
            time.sleep(args.process_group_timeout + 5)
            return 1
        local_error = None
        try:
            if args.worker_rank == 0:
                transferred = payload
                if args.failure_mode == "malformed_payload":
                    transferred = bytes([payload[0] ^ 0xFF]) + payload[1:]
                _send_bytes(contract_bytes, tag=10)
                _send_bytes(transferred, tag=20)
            else:
                received_contract = _receive_bytes(
                    maximum_bytes=MAX_ACTIVATION_CONTRACT_BYTES, tag=10
                )
                received_payload = _receive_bytes(
                    maximum_bytes=contract.payload_length, tag=20
                )
                ActivationHandoffContract.from_json_bytes(
                    received_contract,
                    topology=topology,
                    partition=partition,
                    payload=received_payload,
                )
        except (TypeError, ValueError, RuntimeError) as error:
            local_error = str(error)
        observations: list[dict[str, object] | None] = [None] * _WORLD_SIZE
        dist.all_gather_object(
            observations,
            {"probe_rank": args.worker_rank, "error": local_error},
        )
        agreed_outcome = (
            "failure"
            if any(
                item is None or item.get("error") is not None for item in observations
            )
            else "success"
        )
        agreed_message = next(
            (
                str(item["error"])
                for item in observations
                if item is not None and item.get("error") is not None
            ),
            None,
        )
        _write_receipt(
            args.output,
            probe_rank=args.worker_rank,
            outcome=agreed_outcome,
            message=agreed_message,
        )
        dist.barrier()
        return 0 if agreed_outcome == "success" else 1
    finally:
        dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--worker-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--started", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-group-timeout", type=int, required=True)
    parser.add_argument(
        "--failure-mode", choices=sorted(mode for mode in _FAILURE_MODES if mode)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one private supervised worker entry point."""

    return _run_worker(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
