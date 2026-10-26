"""Executable acceptance oracles for real multi-process training and recovery."""

from pathlib import Path
import json
from hashlib import sha256
import os
import re
import shutil
import stat
import time

import torch

from .ddp_checkpoint import bounded_checkpoint_load
from .ddp_launch import WorkerFailure, launch_training
from .ddp_contracts import integer
from .ddp_limits import preflight

CRASH_RECOVERY_SCHEMA = "ai-journey-ddp-crash-recovery-v1"
CRASH_RECOVERY_SCOPE = "CPU-local Gloo crash/restart observations; not GPU, power-loss or production failover evidence. Digests validate internal consistency, not timing authenticity."
_STATE_FIELDS = ("model", "optimizer", "rank_rngs", "step")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _state_digest(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        encoded = [
            "tensor",
            str(tensor.dtype),
            list(tensor.shape),
            sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        ]
    elif isinstance(value, dict):
        encoded = [
            "dict",
            [
                [_state_digest(key), _state_digest(value[key])]
                for key in sorted(
                    value, key=lambda key: (type(key).__name__, repr(key))
                )
            ],
        ]
    elif type(value) in (list, tuple):
        encoded = [type(value).__name__, [_state_digest(item) for item in value]]
    elif value is None or type(value) in (bool, int, float, str):
        encoded = [type(value).__name__, value]
    else:
        raise ValueError("unsupported recovery state field")
    return sha256(_canonical(encoded).encode()).hexdigest()


def _fields(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{name} fields differ from the recovery schema")


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid recovery digest")


def crash_recovery_digest(report):
    return sha256(
        _canonical({k: v for k, v in report.items() if k != "report_sha256"}).encode()
    ).hexdigest()


def _crash_metrics(report):
    crash, supervision = report["injection"], report["supervision"]
    ready = max(row["ready_ns"] for row in report["restoration"])
    detected = supervision["failure_observed_ns"]
    return {
        "lost_optimizer_updates": crash["completed_step"]
        - report["last_durable"]["step"],
        "resumed_optimizer_updates": report["total_steps"]
        - report["last_durable"]["step"],
        "failure_detection_upper_ns": detected - crash["observed_ns"],
        "supervisor_cleanup_ns": supervision["reaped_ns"] - detected,
        "restoration_wall_ns": ready - report["resume_started_ns"],
        "resume_completion_wall_ns": report["resume_finished_ns"]
        - report["resume_started_ns"],
        "recovery_ready_lower_ns": ready - detected,
        "recovery_ready_upper_ns": ready - crash["observed_ns"],
    }


def verify_crash_recovery(report, *, directory: Path | None = None):
    """Recompute loss/latency bounds and reject inconsistent local crash evidence."""
    _fields(
        report,
        (
            "schema",
            "scope",
            "source_sha256",
            "spec_sha256",
            "world_size",
            "total_steps",
            "checkpoint_interval",
            "injection",
            "supervision",
            "last_durable",
            "resume_started_ns",
            "resume_finished_ns",
            "control_workers",
            "resumed_workers",
            "restoration",
            "state_sha256",
            "metrics",
            "report_sha256",
        ),
        "report",
    )
    if (
        report["schema"] != CRASH_RECOVERY_SCHEMA
        or report["scope"] != CRASH_RECOVERY_SCOPE
    ):
        raise ValueError("recovery evidence schema or CPU scope differs")
    for field in ("source_sha256", "spec_sha256", "report_sha256"):
        _digest(report[field])
    if crash_recovery_digest(report) != report["report_sha256"]:
        raise ValueError("recovery evidence digest differs")
    world = integer(report["world_size"], "world_size", 1)
    total = integer(report["total_steps"], "total_steps", 1)
    interval = integer(report["checkpoint_interval"], "checkpoint_interval", 1)
    if world > 4 or total > 128 or interval > 128:
        raise ValueError("recovery exceeds bounded CPU protocol")
    crash = report["injection"]
    _fields(
        crash,
        (
            "schema",
            "rank",
            "world_size",
            "pid",
            "completed_step",
            "observed_ns",
            "exit_code",
        ),
        "crash boundary",
    )
    for field in (
        "rank",
        "world_size",
        "pid",
        "completed_step",
        "observed_ns",
        "exit_code",
    ):
        integer(crash[field], field, 0 if field == "rank" else 1)
    if (
        crash["schema"] != "ai-journey-ddp-crash-boundary-v1"
        or crash["world_size"] != world
        or crash["rank"] >= world
        or crash["exit_code"] != 86
        or crash["completed_step"] > total
    ):
        raise ValueError("crash boundary does not match the owned worker")
    durable = report["last_durable"]
    _fields(
        durable,
        ("step", "model_sha256", "checkpoint_sha256"),
        "last durable checkpoint",
    )
    integer(durable["step"], "last_durable_step", 1)
    _digest(durable["model_sha256"])
    _digest(durable["checkpoint_sha256"])
    if durable["step"] != (crash["completed_step"] - 1) // interval * interval:
        raise ValueError("last durable step contradicts the observed crash boundary")
    supervision = report["supervision"]
    _fields(
        supervision,
        (
            "schema",
            "started_ns",
            "failure_observed_ns",
            "reaped_ns",
            "timed_out",
            "workers",
        ),
        "supervision",
    )
    for field in ("started_ns", "failure_observed_ns", "reaped_ns"):
        integer(supervision[field], field, 1)
    if (
        supervision["schema"] != "ai-journey-ddp-supervision-v1"
        or supervision["timed_out"] is not False
        or not supervision["started_ns"]
        <= crash["observed_ns"]
        <= supervision["failure_observed_ns"]
        <= supervision["reaped_ns"]
    ):
        raise ValueError("supervision does not bound the observed crash")
    if type(supervision["workers"]) is not list or len(supervision["workers"]) != world:
        raise ValueError("owned worker observations are incomplete")
    failed_pids = set()
    for rank, worker in enumerate(supervision["workers"]):
        _fields(
            worker, ("rank", "pid", "returncode", "exit_observed_ns"), "owned worker"
        )
        integer(worker["rank"], "rank")
        integer(worker["pid"], "pid", 1)
        integer(worker["exit_observed_ns"], "exit_observed_ns", 1)
        if (
            worker["rank"] != rank
            or type(worker["returncode"]) is not int
            or worker["pid"] in failed_pids
            or not supervision["started_ns"]
            <= worker["exit_observed_ns"]
            <= supervision["reaped_ns"]
        ):
            raise ValueError("owned worker exit evidence is invalid")
        failed_pids.add(worker["pid"])
    failed = supervision["workers"][crash["rank"]]
    if (
        failed["pid"] != crash["pid"]
        or failed["returncode"] != crash["exit_code"]
        or failed["exit_observed_ns"] < crash["observed_ns"]
    ):
        raise ValueError("supervisor did not observe the injected child exit")
    started = integer(report["resume_started_ns"], "resume_started_ns", 1)
    finished = integer(report["resume_finished_ns"], "resume_finished_ns", 1)
    if not supervision["reaped_ns"] <= started < finished:
        raise ValueError("resume window precedes owned-child cleanup")
    identities = set()
    groups = {}
    for name, expected_start in (
        ("control_workers", 0),
        ("resumed_workers", durable["step"]),
    ):
        workers = report[name]
        if type(workers) is not list or len(workers) != world:
            raise ValueError("recovery worker identities are incomplete")
        pids = set()
        for rank, worker in enumerate(workers):
            _fields(
                worker,
                ("rank", "pid", "started_step", "completed_step", "model_sha256"),
                name,
            )
            for field in ("rank", "pid", "started_step", "completed_step"):
                integer(worker[field], field, 1 if field == "pid" else 0)
            _digest(worker["model_sha256"])
            if (
                worker["rank"] != rank
                or worker["started_step"] != expected_start
                or worker["completed_step"] != total
                or worker["pid"] in pids
            ):
                raise ValueError("recovery progress or process identities differ")
            pids.add(worker["pid"])
            identities.add(worker["model_sha256"])
        groups[name] = pids
    if (groups["resumed_workers"] & (failed_pids | groups["control_workers"])) or len(
        identities
    ) != 1:
        raise ValueError("recovery requires fresh workers and matching final models")
    restoration = report["restoration"]
    if type(restoration) is not list or len(restoration) != world:
        raise ValueError("restoration readiness observations are incomplete")
    for rank, witness in enumerate(restoration):
        _fields(
            witness,
            (
                "schema",
                "rank",
                "world_size",
                "pid",
                "restored_step",
                "ready_ns",
                "model_sha256",
            ),
            "restoration boundary",
        )
        for field in ("rank", "world_size", "pid", "restored_step", "ready_ns"):
            integer(witness[field], field, 0 if field == "rank" else 1)
        if (
            witness["schema"] != "ai-journey-ddp-restored-boundary-v1"
            or witness["rank"] != rank
            or witness["world_size"] != world
            or witness["pid"] != report["resumed_workers"][rank]["pid"]
            or witness["restored_step"] != durable["step"]
            or witness["model_sha256"] != durable["model_sha256"]
            or not started <= witness["ready_ns"] <= finished
        ):
            raise ValueError("restoration boundary does not prove the durable restart")
    states = report["state_sha256"]
    _fields(states, ("continuous", "recovered"), "state digests")
    for state in states.values():
        _fields(state, _STATE_FIELDS, "state digests")
        for value in state.values():
            _digest(value)
    if states["continuous"] != states["recovered"]:
        raise ValueError("recovery does not reproduce uninterrupted state")
    if _canonical(report["metrics"]) != _canonical(_crash_metrics(report)):
        raise ValueError("recovery loss or measured timing bounds differ")
    if directory is not None:
        _verify_crash_artifacts(report, directory)
    return report


def _verify_crash_artifacts(report, directory):
    try:
        durable_path = directory / "last-durable.pt"
        durable = bounded_checkpoint_load(durable_path, 64 * 1024 * 1024)
        if (
            durable["step"] != report["last_durable"]["step"]
            or durable["model_sha256"] != report["last_durable"]["model_sha256"]
            or sha256(durable_path.read_bytes()).hexdigest()
            != report["last_durable"]["checkpoint_sha256"]
        ):
            raise ValueError("last durable checkpoint differs")
        for name, filename in (
            ("continuous", "continuous.pt"),
            ("recovered", "recovered.pt"),
        ):
            payload = bounded_checkpoint_load(directory / filename, 64 * 1024 * 1024)
            state = {field: _state_digest(payload[field]) for field in _STATE_FIELDS}
            if (
                state != report["state_sha256"][name]
                or payload["step"] != report["total_steps"]
                or payload["source_sha256"] != report["source_sha256"]
            ):
                raise ValueError("final checkpoint state differs")
        rank = report["injection"]["rank"]
        if _canonical(
            _read_json(directory / "failed" / f"rank-{rank}.crash.json", max_bytes=4096)
        ) != _canonical(report["injection"]):
            raise ValueError("crash boundary witness differs")
        for rank, witness in enumerate(report["restoration"]):
            if _canonical(
                _read_json(
                    directory / "resumed" / f"rank-{rank}.recovery.json", max_bytes=4096
                )
            ) != _canonical(witness):
                raise ValueError("restoration witness differs")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("crash recovery artifacts do not match the report") from error


def _read_json(path: Path, *, max_bytes=128 * 1024):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate recovery JSON field")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite recovery JSON constant")

    descriptor = os.open(
        path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= max_bytes:
            raise ValueError("recovery JSON must be a bounded regular file")
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("recovery JSON exceeds its byte budget")
    return json.loads(payload, object_pairs_hook=unique, parse_constant=constant)


def read_crash_recovery(path: Path, *, directory: Path | None = None):
    return verify_crash_recovery(_read_json(path), directory=directory)


def certify_crash_recovery(
    *,
    source: Path,
    spec: Path,
    directory: Path,
    world_size=2,
    total_steps=4,
    checkpoint_interval=2,
    failure_after_step=3,
    failure_rank=1,
):
    """Measure an owned abrupt loss, fresh restoration, and exact final-state parity."""
    integer(world_size, "world_size", 1)
    integer(total_steps, "total_steps", 1)
    integer(checkpoint_interval, "checkpoint_interval", 1)
    integer(failure_after_step, "failure_after_step", 1)
    integer(failure_rank, "failure_rank")
    if (
        world_size > 4
        or total_steps > 128
        or failure_rank >= world_size
        or not checkpoint_interval < failure_after_step <= total_steps
    ):
        raise ValueError(
            "crash recovery requires a prior checkpoint and bounded scheduled failure"
        )
    if directory.is_symlink() or (
        directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
    ):
        raise ValueError("crash recovery workdir must be new or empty")
    preflight(source, spec, world_size=world_size, steps=total_steps)
    source_sha256 = sha256(source.read_bytes()).hexdigest()
    spec_sha256 = sha256(spec.read_bytes()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    continuous_path, recovered_path = (
        directory / "continuous.pt",
        directory / "recovered.pt",
    )
    control = launch_training(
        source=source,
        spec=spec,
        checkpoint=continuous_path,
        output_dir=directory / "continuous",
        world_size=world_size,
        steps=total_steps,
    )
    try:
        launch_training(
            source=source,
            spec=spec,
            checkpoint=recovered_path,
            output_dir=directory / "failed",
            world_size=world_size,
            steps=total_steps,
            checkpoint_interval=checkpoint_interval,
            failure_rank=failure_rank,
            failure_mode="crash",
            failure_after_step=failure_after_step,
        )
    except WorkerFailure as failure:
        supervision = failure.supervision
    else:
        raise RuntimeError("the requested abrupt worker failure was not observed")
    injection = _read_json(
        directory / "failed" / f"rank-{failure_rank}.crash.json", max_bytes=4096
    )
    durable = bounded_checkpoint_load(recovered_path, 64 * 1024 * 1024)
    shutil.copyfile(recovered_path, directory / "last-durable.pt")
    last_durable = {
        "step": durable["step"],
        "model_sha256": durable["model_sha256"],
        "checkpoint_sha256": sha256(recovered_path.read_bytes()).hexdigest(),
    }
    resume_started = time.monotonic_ns()
    resumed = launch_training(
        source=source,
        spec=spec,
        checkpoint=recovered_path,
        output_dir=directory / "resumed",
        world_size=world_size,
        steps=total_steps - durable["step"],
        resume=True,
        observe_recovery=True,
    )
    resume_finished = time.monotonic_ns()
    expected = bounded_checkpoint_load(continuous_path, 64 * 1024 * 1024)
    actual = bounded_checkpoint_load(recovered_path, 64 * 1024 * 1024)
    if not all(exact_state(expected[name], actual[name]) for name in _STATE_FIELDS):
        raise RuntimeError("crash recovery does not reproduce uninterrupted state")
    if (
        sha256(source.read_bytes()).hexdigest() != source_sha256
        or sha256(spec.read_bytes()).hexdigest() != spec_sha256
    ):
        raise ValueError("recovery source or specification changed during the run")

    def workers(rows):
        return [
            {
                key: row[key]
                for key in (
                    "rank",
                    "pid",
                    "started_step",
                    "completed_step",
                    "model_sha256",
                )
            }
            for row in rows
        ]

    report = {
        "schema": CRASH_RECOVERY_SCHEMA,
        "scope": CRASH_RECOVERY_SCOPE,
        "source_sha256": source_sha256,
        "spec_sha256": spec_sha256,
        "world_size": world_size,
        "total_steps": total_steps,
        "checkpoint_interval": checkpoint_interval,
        "injection": injection,
        "supervision": supervision,
        "last_durable": last_durable,
        "resume_started_ns": resume_started,
        "resume_finished_ns": resume_finished,
        "control_workers": workers(control),
        "resumed_workers": workers(resumed),
        "restoration": [
            _read_json(
                directory / "resumed" / f"rank-{rank}.recovery.json", max_bytes=4096
            )
            for rank in range(world_size)
        ],
        "state_sha256": {
            "continuous": {
                name: _state_digest(expected[name]) for name in _STATE_FIELDS
            },
            "recovered": {name: _state_digest(actual[name]) for name in _STATE_FIELDS},
        },
    }
    report["metrics"] = _crash_metrics(report)
    report["report_sha256"] = crash_recovery_digest(report)
    return verify_crash_recovery(report, directory=directory)


def exact_state(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            exact_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            exact_state(a, b) for a, b in zip(left, right)
        )
    return left == right


def certify_recovery(
    *, source: Path, spec: Path, directory: Path, world_size=2, split_steps=2
) -> dict:
    """Compare an uninterrupted run to a stopped run resumed in fresh processes."""
    continuous_path = directory / "continuous.pt"
    resumed_path = directory / "resumed.pt"
    continuous = launch_training(
        source=source,
        spec=spec,
        checkpoint=continuous_path,
        output_dir=directory / "continuous",
        world_size=world_size,
        steps=split_steps * 2,
    )
    first = launch_training(
        source=source,
        spec=spec,
        checkpoint=resumed_path,
        output_dir=directory / "first",
        world_size=world_size,
        steps=split_steps,
    )
    resumed = launch_training(
        source=source,
        spec=spec,
        checkpoint=resumed_path,
        output_dir=directory / "resumed",
        world_size=world_size,
        steps=split_steps,
        resume=True,
    )
    expected = bounded_checkpoint_load(continuous_path, 64 * 1024 * 1024)
    actual = bounded_checkpoint_load(resumed_path, 64 * 1024 * 1024)
    verified = {
        name: exact_state(expected[name], actual[name])
        for name in ("model", "optimizer", "rank_rngs", "step")
    }
    if not all(verified.values()):
        raise RuntimeError(
            "distributed recovery does not reproduce uninterrupted state"
        )
    if {rank["pid"] for rank in first} & {rank["pid"] for rank in resumed}:
        raise RuntimeError("restart evidence requires fresh worker processes")
    return {
        "world_size": world_size,
        "completed_step": split_steps * 2,
        "exact_state": verified,
        "model_sha256": continuous[0]["model_sha256"],
        "initial_worker_pids": [rank["pid"] for rank in first],
        "resumed_worker_pids": [rank["pid"] for rank in resumed],
    }


def certify_batch_equivalence(
    *, source: Path, spec: Path, directory: Path, world_size=2, steps=2, tolerance=1e-10
) -> dict:
    """Compare DDP against one process with the same effective global batch."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.loads(spec.read_text())
    payload["training"]["micro_batch_size"] *= world_size
    serial_spec = directory / "serial-spec.json"
    serial_spec.write_text(json.dumps(payload))
    distributed_path, serial_path = (
        directory / "distributed.pt",
        directory / "serial.pt",
    )
    distributed = launch_training(
        source=source,
        spec=spec,
        checkpoint=distributed_path,
        output_dir=directory / "distributed",
        world_size=world_size,
        steps=steps,
    )
    serial = launch_training(
        source=source,
        spec=serial_spec,
        checkpoint=serial_path,
        output_dir=directory / "serial",
        world_size=1,
        steps=steps,
    )
    expected = bounded_checkpoint_load(serial_path, 64 * 1024 * 1024)["model"]
    actual = bounded_checkpoint_load(distributed_path, 64 * 1024 * 1024)["model"]
    maximum_error = max(
        (expected[name] - actual[name]).abs().max().item() for name in expected
    )
    if maximum_error > tolerance or [
        metric["tokens"] for metric in distributed[0]["metrics"]
    ] != [metric["tokens"] for metric in serial[0]["metrics"]]:
        raise RuntimeError("DDP does not match the equivalent global-batch oracle")
    return {
        "world_size": world_size,
        "steps": steps,
        "max_parameter_error": maximum_error,
        "absolute_tolerance": tolerance,
        "global_batch_size": payload["training"]["micro_batch_size"]
        * payload["training"]["accumulation_steps"],
        "equivalent": True,
    }
