"""Tamper-evident reports for deterministic GPT-2 training runs."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Mapping

import torch

from .decoder_variant_config import DecoderVariantConfig
from .gpt2_config import GPT2Config
from .gqa_config import GQAConfig
from .training_models import expected_training_parameters
from .gpt2_training import (
    GPT2StepMetric,
    GPT2TrainingError,
    GPT2TrainingRun,
    run_training_experiment,
    training_spec_from_mapping,
)

GPT2_TRAINING_EVIDENCE_SCHEMA_VERSION = 2
GPT2_TRAINING_LIMITATIONS = (
    "Scale-limited byte-level CPU run on a tiny GPT-2-style decoder.",
    "This does not establish GPU throughput, distributed scaling, or production readiness.",
    "Automated execution is not evidence that Ajay watched or completed the lesson.",
)
_SHA256 = re.compile(r"[a-f0-9]{64}")
_FIELDS = {
    "schema_version",
    "spec",
    "spec_sha256",
    "corpus",
    "optimizer_groups",
    "parameter_count",
    "progress",
    "model_fingerprints",
    "metrics",
    "runtime",
    "limitations",
    "evidence_sha256",
}


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(canonical.encode()).hexdigest()


def _spec_payload(run: GPT2TrainingRun) -> dict[str, Any]:
    return run.spec.to_mapping()


def _runtime_payload(*, torch_threads: int) -> dict[str, Any]:
    torch_config = torch.__config__.show()
    cpu_capability = getattr(
        torch.backends.cpu, "get_cpu_capability", lambda: "unknown"
    )()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch_threads,
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "cpu_capability": cpu_capability,
        "torch_config_sha256": sha256(torch_config.encode()).hexdigest(),
    }


def build_training_evidence(run: GPT2TrainingRun) -> dict[str, Any]:
    """Build an internally consistent report from one measured run, not an attestation."""

    if not isinstance(run, GPT2TrainingRun):
        raise TypeError("run must be GPT2TrainingRun")
    payload: dict[str, Any] = {
        "schema_version": GPT2_TRAINING_EVIDENCE_SCHEMA_VERSION,
        "spec": _spec_payload(run),
        "spec_sha256": run.spec.fingerprint(),
        "corpus": {
            "sha256": run.corpus_fingerprint,
            "bytes": run.corpus_bytes,
            "discarded_samples_per_epoch": run.discarded_samples_per_epoch,
        },
        "optimizer_groups": {
            **asdict(run.optimizer_groups),
            "decay_names": list(run.optimizer_groups.decay_names),
            "no_decay_names": list(run.optimizer_groups.no_decay_names),
        },
        "parameter_count": run.parameter_count,
        "progress": {
            "started_step": run.started_step,
            "completed_step": run.completed_step,
        },
        "model_fingerprints": {
            "initial": run.initial_model_fingerprint,
            "final": run.final_model_fingerprint,
        },
        "metrics": [asdict(metric) for metric in run.metrics],
        "runtime": _runtime_payload(torch_threads=run.torch_threads),
        "limitations": list(GPT2_TRAINING_LIMITATIONS),
    }
    payload["evidence_sha256"] = _canonical_digest(payload)
    verify_training_evidence(payload)
    return payload


def _require_digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GPT2TrainingError(f"{name} must be a SHA-256 digest")


def _dense_parameter_shapes(config, *, name_count: int) -> dict[str, tuple[int, ...]]:
    """Enumerate unique dense parameters without tensor or model construction.

    Work is bounded by the submitted name count before iterating decoder layers.
    The tied head is not a second parameter and RoPE buffers are not weights.
    """
    prefix = ""
    kv_heads = None
    if type(config) is GQAConfig:
        prefix, kv_heads = "decoder.", config.kv_heads
        config = config.decoder
    if type(config) is GPT2Config:
        base, norm, mlp, position = config, "layernorm", "gelu", "learned"
        variant = False
        hidden = 4 * base.embedding_dim
    elif type(config) is DecoderVariantConfig:
        variant = True
        base, norm, mlp, position = (
            config.gpt2,
            config.norm_kind,
            config.mlp_kind,
            config.position_kind,
        )
        hidden = (
            config.resolved_swiglu_hidden_dim
            if mlp == "swiglu"
            else 4 * base.embedding_dim
        )
    else:
        raise GPT2TrainingError(
            "optimizer evidence requires an explicit dense architecture"
        )
    norm_tensors = 2 if norm == "layernorm" else 1
    tensor_count = (
        1
        + (position == "learned")
        + norm_tensors
        + base.layer_count * (2 * norm_tensors + 4 + (3 if mlp == "swiglu" else 4))
    )
    if tensor_count != name_count:
        raise GPT2TrainingError(
            "training evidence optimizer names differ from architecture"
        )
    width = base.embedding_dim
    qkv_width = 3 * width if kv_heads is None else width + 2 * kv_heads * base.head_dim
    shapes = {}

    def add(name, shape):
        shapes[f"{prefix}{name}"] = shape

    def add_norm(name):
        add(f"{name}.weight", (width,))
        if norm == "layernorm":
            add(f"{name}.bias", (width,))

    def linear(name, output, inputs, *, bias=True):
        add(f"{name}.weight", (output, inputs))
        if bias:
            add(f"{name}.bias", (output,))

    add("wte.weight", (base.vocab_size, width))
    if position == "learned":
        add("wpe.weight", (base.context_length, width))
    for index in range(base.layer_count):
        block = f"h.{index}"
        add_norm(f"{block}.ln_1")
        if variant:
            add_norm(f"{block}.ln_2")
        linear(f"{block}.attn.c_attn", qkv_width, width)
        linear(f"{block}.attn.c_proj", width, width)
        if not variant:
            add_norm(f"{block}.ln_2")
        if mlp == "swiglu":
            linear(f"{block}.mlp.0.gate_proj", hidden, width, bias=False)
            linear(f"{block}.mlp.0.up_proj", hidden, width, bias=False)
            linear(f"{block}.mlp.0.down_proj", width, hidden, bias=False)
        else:
            linear(f"{block}.mlp.c_fc", hidden, width)
            linear(f"{block}.mlp.c_proj", width, hidden)
    add_norm("ln_f")
    return shapes


def verify_training_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify integrity, accounting, schedule, and provenance in a saved report."""

    if not isinstance(payload, Mapping) or set(payload) != _FIELDS:
        raise GPT2TrainingError("training evidence fields do not match the schema")
    if payload["schema_version"] != GPT2_TRAINING_EVIDENCE_SCHEMA_VERSION:
        raise GPT2TrainingError("training evidence schema version is unsupported")
    stored_digest = payload["evidence_sha256"]
    _require_digest(stored_digest, "evidence_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "evidence_sha256"
    }
    if _canonical_digest(unsigned) != stored_digest:
        raise GPT2TrainingError("training evidence digest does not match")
    spec = training_spec_from_mapping(payload["spec"])
    if payload["spec_sha256"] != spec.fingerprint():
        raise GPT2TrainingError("training evidence spec fingerprint does not match")
    corpus = payload["corpus"]
    if not isinstance(corpus, Mapping) or set(corpus) != {
        "sha256",
        "bytes",
        "discarded_samples_per_epoch",
    }:
        raise GPT2TrainingError("training evidence corpus fields are invalid")
    _require_digest(corpus["sha256"], "corpus sha256")
    if (
        any(
            isinstance(corpus[name], bool)
            or not isinstance(corpus[name], int)
            or corpus[name] < 0
            for name in ("bytes", "discarded_samples_per_epoch")
        )
        or corpus["bytes"] < 2
    ):
        raise GPT2TrainingError("training evidence corpus accounting is invalid")
    groups = payload["optimizer_groups"]
    if not isinstance(groups, Mapping) or set(groups) != {
        "decay_names",
        "no_decay_names",
        "decay_parameters",
        "no_decay_parameters",
    }:
        raise GPT2TrainingError("training evidence optimizer groups are invalid")
    if any(
        type(groups[name]) is not list for name in ("decay_names", "no_decay_names")
    ):
        raise GPT2TrainingError("training evidence optimizer names must be lists")
    names = [*groups["decay_names"], *groups["no_decay_names"]]
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
        or any(
            isinstance(groups[name], bool)
            or not isinstance(groups[name], int)
            or groups[name] < 0
            for name in ("decay_parameters", "no_decay_parameters")
        )
    ):
        raise GPT2TrainingError("training evidence optimizer accounting is invalid")
    shapes = _dense_parameter_shapes(spec.model, name_count=len(names))
    for group, matrix in (("decay", True), ("no_decay", False)):
        expected = [
            name for name, shape in shapes.items() if (len(shape) >= 2) == matrix
        ]
        if groups[f"{group}_names"] != expected:
            raise GPT2TrainingError(
                "training evidence optimizer names differ from architecture"
            )
        if groups[f"{group}_parameters"] != sum(
            math.prod(shapes[name]) for name in expected
        ):
            raise GPT2TrainingError(
                "training evidence optimizer parameter count differs from architecture"
            )
    parameter_count = payload["parameter_count"]
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count != expected_training_parameters(spec.model)
        or groups["decay_parameters"] + groups["no_decay_parameters"] != parameter_count
    ):
        raise GPT2TrainingError("training evidence parameter count is invalid")
    progress = payload["progress"]
    if not isinstance(progress, Mapping) or set(progress) != {
        "started_step",
        "completed_step",
    }:
        raise GPT2TrainingError("training evidence progress fields are invalid")
    started, completed = progress["started_step"], progress["completed_step"]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in progress.values()
        )
        or not 0 <= started <= completed <= spec.training.optimizer_steps
    ):
        raise GPT2TrainingError("training evidence progress is invalid")
    fingerprints = payload["model_fingerprints"]
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != {
        "initial",
        "final",
    }:
        raise GPT2TrainingError("training evidence model fingerprints are invalid")
    for name, value in fingerprints.items():
        _require_digest(value, f"model {name} fingerprint")
    metrics = payload["metrics"]
    if not isinstance(metrics, list) or len(metrics) != completed - started:
        raise GPT2TrainingError("training evidence metric count is invalid")
    expected_fields = set(GPT2StepMetric.__dataclass_fields__)
    for offset, metric in enumerate(metrics, start=started + 1):
        if not isinstance(metric, Mapping) or set(metric) != expected_fields:
            raise GPT2TrainingError("training evidence metric fields are invalid")
        if metric["step"] != offset:
            raise GPT2TrainingError("training evidence metric steps are not contiguous")
        if metric["micro_batches"] != spec.training.accumulation_steps:
            raise GPT2TrainingError("training evidence accumulation count is invalid")
        if metric["target_tokens"] != spec.training.tokens_per_step:
            raise GPT2TrainingError("training evidence token accounting is invalid")
        numeric = (
            metric["mean_loss"],
            metric["learning_rate"],
            metric["gradient_norm_before_clip"],
            metric["gradient_norm_after_clip"],
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric
        ):
            raise GPT2TrainingError("training evidence metrics must be finite")
        if metric["gradient_norm_after_clip"] > spec.training.gradient_clip + 1e-6:
            raise GPT2TrainingError("training evidence gradient clipping is invalid")
        if not isinstance(metric["clipped"], bool):
            raise GPT2TrainingError("training evidence clipped flag is invalid")
    runtime = payload["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python",
        "torch",
        "device",
        "torch_threads",
        "operating_system",
        "machine",
        "cpu_capability",
        "torch_config_sha256",
    }:
        raise GPT2TrainingError("training evidence runtime fields are invalid")
    if runtime["device"] != "cpu" or any(
        not isinstance(runtime[name], str) or not runtime[name]
        for name in (
            "python",
            "torch",
            "operating_system",
            "machine",
            "cpu_capability",
        )
    ):
        raise GPT2TrainingError("training evidence runtime is invalid")
    _require_digest(runtime["torch_config_sha256"], "torch config sha256")
    if runtime["torch_threads"] != 1:
        raise GPT2TrainingError("training evidence thread control is invalid")
    if payload["limitations"] != list(GPT2_TRAINING_LIMITATIONS):
        raise GPT2TrainingError("training evidence limitations are invalid")
    return dict(payload)


def verify_training_replay(source: bytes, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run in a compatible runtime and require exact evidence."""

    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    expected = verify_training_evidence(payload)
    current_runtime = _runtime_payload(torch_threads=1)
    if expected["runtime"] != current_runtime:
        raise GPT2TrainingError(
            "training evidence runtime does not match the exact replay environment"
        )
    spec = training_spec_from_mapping(expected["spec"])
    started = expected["progress"]["started_step"]
    count = expected["progress"]["completed_step"] - started
    with tempfile.TemporaryDirectory(prefix="gpt2-training-replay-") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        if started:
            run_training_experiment(
                source, spec, checkpoint_path=checkpoint, step_count=started
            )
        run = run_training_experiment(
            source,
            spec,
            checkpoint_path=checkpoint,
            resume=bool(started),
            step_count=count or None,
        )
    actual = build_training_evidence(run)
    if actual != expected:
        raise GPT2TrainingError("fresh GPT-2 training replay does not match evidence")
    return actual
