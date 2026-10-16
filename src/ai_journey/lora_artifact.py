"""Bounded, pickle-free adapter artifacts with atomic publication."""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import torch

from .lora import LoRAError, adapter_state

MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_FACTOR_ELEMENTS = 65536
_DTYPES = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _encode_tensor(value):
    if str(value.dtype) not in _DTYPES or value.numel() > MAX_FACTOR_ELEMENTS:
        raise LoRAError("adapter tensor exceeds the supported dtype or element budget")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": value.reshape(-1).tolist(),
    }


def write_adapter_artifact(path, model):
    """Publish factors and their integrity digest without serializing base weights.

    The hash detects accidental corruption; it is not an authenticity signature.
    A failed encode, flush or replace leaves any previous artifact intact.
    """
    state = adapter_state(model)
    payload = {
        "schema": "lora-artifact-v1",
        "base_fingerprint": state["base_fingerprint"],
        "layers": {
            name: {
                "config": layer["config"],
                "a": _encode_tensor(layer["a"]),
                "b": _encode_tensor(layer["b"]),
            }
            for name, layer in state["layers"].items()
        },
    }
    payload["sha256"] = sha256(_canonical(payload)).hexdigest()
    data = _canonical(payload) + b"\n"
    if len(data) > MAX_ARTIFACT_BYTES:
        raise LoRAError("adapter artifact exceeds the byte budget")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload["sha256"]
