"""Portable checkpoints for the NumPy context language model."""

from __future__ import annotations

from typing import Any

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
