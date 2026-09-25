"""Portable checkpoints for the NumPy context language model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ai_journey.context_mlp import ContextMLP, model_fingerprint

SCHEMA_VERSION = 1


def checkpoint_payload(model: ContextMLP, *, step: int) -> dict[str, Any]:
    """Return a JSON-compatible checkpoint payload."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    return {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "model_fingerprint": model_fingerprint(model),
        "parameters": {
            name: values.tolist() for name, values in model.__dict__.items()
        },
    }


def model_from_payload(payload: dict[str, Any]) -> ContextMLP:
    """Validate and restore model parameters from a checkpoint payload."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    parameters = payload.get("parameters")
    expected = set(ContextMLP.__dataclass_fields__)
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ValueError("checkpoint parameter set is invalid")
    model = ContextMLP(
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in parameters.items()
        }
    )
    if model_fingerprint(model) != payload.get("model_fingerprint"):
        raise ValueError("checkpoint fingerprint mismatch")
    return model


def save_checkpoint(path: Path, model: ContextMLP, *, step: int) -> None:
    """Atomically replace a checkpoint with validated JSON."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(checkpoint_payload(model, step=step), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_checkpoint(path: Path) -> tuple[ContextMLP, int]:
    """Load a checkpoint and return its verified model and step."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = model_from_payload(payload)
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step is invalid")
    return model, step
