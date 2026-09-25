"""Portable checkpoints for the NumPy context language model."""

from __future__ import annotations

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
