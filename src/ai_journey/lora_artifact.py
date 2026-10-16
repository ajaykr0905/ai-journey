"""Bounded, pickle-free adapter artifacts with atomic publication."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import re

import torch

from .lora import LoRAConfig, LoRAError, adapter_state, load_adapter_state

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


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise LoRAError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _reject_constant(value):
    raise LoRAError(f"nonfinite JSON number: {value}")


def _decode_tensor(value):
    if not isinstance(value, dict) or set(value) != {"shape", "dtype", "values"}:
        raise LoRAError("invalid adapter tensor fields")
    shape, dtype, values = value["shape"], value["dtype"], value["values"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            type(size) is not int or not 1 <= size <= MAX_FACTOR_ELEMENTS
            for size in shape
        )
        or math.prod(shape) > MAX_FACTOR_ELEMENTS
        or not isinstance(dtype, str)
        or dtype not in _DTYPES
        or not isinstance(values, list)
        or len(values) != math.prod(shape)
    ):
        raise LoRAError(
            "adapter tensor dimensions, dtype, or values exceed the contract"
        )
    for number in values:
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise LoRAError("adapter tensor values must be finite numbers")
        try:
            finite = math.isfinite(number)
        except OverflowError:
            finite = False
        if not finite:
            raise LoRAError("adapter tensor values must be finite numbers")
    try:
        tensor = torch.tensor(values, dtype=_DTYPES[dtype]).reshape(shape)
    except (RuntimeError, OverflowError, ValueError) as exc:
        raise LoRAError(
            "adapter values cannot be represented in their declared dtype"
        ) from exc
    if not bool(torch.isfinite(tensor).all()):
        raise LoRAError("adapter values overflow their declared dtype")
    return tensor


def read_adapter_artifact(path):
    """Read a bounded JSON artifact without executing any serialized Python."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise LoRAError("adapter artifact exceeds the byte budget")
    try:
        payload = json.loads(
            data, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise LoRAError("invalid adapter JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "base_fingerprint", "layers", "sha256"}
        or payload["schema"] != "lora-artifact-v1"
        or not isinstance(payload["base_fingerprint"], str)
        or re.fullmatch(r"[a-f0-9]{64}", payload["base_fingerprint"]) is None
        or not isinstance(payload["layers"], dict)
        or not 1 <= len(payload["layers"]) <= 128
    ):
        raise LoRAError("invalid adapter artifact envelope")
    digest = payload.pop("sha256")
    if not isinstance(digest, str) or digest != sha256(_canonical(payload)).hexdigest():
        raise LoRAError("adapter artifact integrity mismatch")
    layers = {}
    for name, layer in payload["layers"].items():
        if (
            not isinstance(name, str)
            or len(name) > 512
            or not isinstance(layer, dict)
            or set(layer) != {"config", "a", "b"}
            or not isinstance(layer["config"], dict)
            or set(layer["config"]) != {"rank", "alpha", "dropout", "seed"}
        ):
            raise LoRAError("invalid adapter target metadata")
        LoRAConfig(**layer["config"])
        layers[name] = {
            "config": layer["config"],
            "a": _decode_tensor(layer["a"]),
            "b": _decode_tensor(layer["b"]),
        }
    return {
        "schema": "lora-state-v1",
        "base_fingerprint": payload["base_fingerprint"],
        "layers": layers,
    }


def restore_adapter_artifact(path, model):
    """Load only after the parsed artifact and receiving model agree exactly."""
    load_adapter_state(model, read_adapter_artifact(path))
