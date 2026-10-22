"""Integrity-checked evidence for measured local distributed process runs."""

from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import torch

from .ddp_contracts import integer
from .gpt2_training import training_spec_from_mapping, learning_rate_at


LIMITATIONS = [
    "CPU/Gloo processes only; no NVIDIA multi-GPU performance evidence.",
    "Synthetic public fixture; no generalized model-quality conclusion.",
    "Automated execution does not establish learner viewing or understanding.",
]


def digest(payload):
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def build_evidence(source: bytes, spec, ranks: list[dict]) -> dict:
    if not ranks or any(
        rank["metrics"] != ranks[0]["metrics"]
        or rank["model_sha256"] != ranks[0]["model_sha256"]
        for rank in ranks
    ):
        raise ValueError("workers disagree on measured training evidence")
    payload = {
        "schema_version": 1,
        "world_size": len(ranks),
        "source_sha256": sha256(source).hexdigest(),
        "spec": {
            "schema_version": 1,
            "model": asdict(spec.model),
            "training": asdict(spec.training),
            "source": asdict(spec.source),
        },
        "ranks": [
            {key: rank[key] for key in ("rank", "pid", "model_sha256")}
            for rank in ranks
        ],
        "metrics": ranks[0]["metrics"],
        "model_sha256": ranks[0]["model_sha256"],
        "completed_step": ranks[0]["completed_step"],
        "runtime": {
            "device": "cpu",
            "backend": "gloo",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "operating_system": platform.system(),
            "dtype": "torch.float64",
            "torch_threads": 1,
            "torch_config_sha256": sha256(torch.__config__.show().encode()).hexdigest(),
        },
        "limitations": LIMITATIONS.copy(),
    }
    payload["evidence_sha256"] = digest(payload)
    return verify_evidence(payload)


def verify_evidence(payload: dict) -> dict:
    fields = {
        "schema_version",
        "world_size",
        "source_sha256",
        "spec",
        "ranks",
        "metrics",
        "model_sha256",
        "completed_step",
        "runtime",
        "limitations",
        "evidence_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload["schema_version"] != 1
    ):
        raise ValueError("distributed evidence schema does not match")
    unsigned = {
        key: value for key, value in payload.items() if key != "evidence_sha256"
    }
    if digest(unsigned) != payload["evidence_sha256"]:
        raise ValueError("distributed evidence digest does not match")
    world = integer(payload["world_size"], "world_size", 1)
    if world > 4:
        raise ValueError("world size exceeds CPU lab budget")
    spec = training_spec_from_mapping(payload["spec"])
    if (
        payload["source_sha256"] != spec.source.sha256
        or payload["limitations"] != LIMITATIONS
    ):
        raise ValueError("evidence provenance or scope does not match")
    ranks = payload["ranks"]
    if (
        not isinstance(ranks, list)
        or len(ranks) != world
        or {rank["rank"] for rank in ranks} != set(range(world))
        or len({rank["pid"] for rank in ranks}) != world
    ):
        raise ValueError("rank process identities are incomplete")
    for rank in ranks:
        integer(rank["pid"], "pid", 1)
        if rank["model_sha256"] != payload["model_sha256"]:
            raise ValueError("rank model fingerprints do not agree")
    metrics = payload["metrics"]
    completed = integer(payload["completed_step"], "completed_step", 1)
    if (
        not isinstance(metrics, list)
        or not metrics
        or completed > spec.training.optimizer_steps
    ):
        raise ValueError("evidence progress is invalid")
    for expected_step, metric in enumerate(metrics, completed - len(metrics) + 1):
        if (
            metric["step"] != expected_step
            or metric["tokens"] != spec.training.tokens_per_step * world
            or metric["micro_batches"] != spec.training.accumulation_steps
        ):
            raise ValueError("distributed metric accounting does not match")
        if any(
            isinstance(metric[name], bool)
            or not math.isfinite(metric[name])
            or metric[name] < 0
            for name in ("mean_loss", "loss_sum", "gradient_norm", "learning_rate")
        ):
            raise ValueError("distributed metrics must be finite and nonnegative")
        if not math.isclose(
            metric["mean_loss"] * metric["tokens"],
            metric["loss_sum"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or metric["learning_rate"] != learning_rate_at(
            expected_step - 1, spec.training
        ):
            raise ValueError("distributed loss or schedule does not match")
    if (
        metrics[-1]["model_sha256"] != payload["model_sha256"]
        or payload["runtime"]["device"] != "cpu"
        or payload["runtime"]["backend"] != "gloo"
    ):
        raise ValueError("distributed final state or runtime does not match")
    return payload


def write_evidence(path: Path, payload: dict):
    verify_evidence(payload)
    if path.is_symlink():
        raise ValueError("evidence output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_evidence(path: Path):
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 4 * 1024 * 1024
    ):
        raise ValueError("distributed evidence exceeds file constraints")
    return verify_evidence(json.loads(path.read_text(encoding="utf-8")))
