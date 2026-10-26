"""Observe NVIDIA runtime availability without claiming CUDA execution."""

from datetime import datetime, timezone
from hashlib import sha256
import os
import platform

import torch


def validate_gpu_probe(payload: dict) -> dict:
    """Validate serialized host observations before using them for admission."""
    fields = {
        "schema_version",
        "kind",
        "observed_at_utc",
        "system",
        "machine",
        "torch",
        "cuda_runtime",
        "nccl_available",
        "visible_devices_sha256",
        "available",
        "reason",
        "devices",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "nvidia_host_probe"
    ):
        raise ValueError("NVIDIA host probe schema does not match")
    timestamp = datetime.fromisoformat(payload["observed_at_utc"])
    if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("probe observation time must be UTC")
    for name in ("system", "machine", "torch"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError("probe runtime identity is incomplete")
    if payload["cuda_runtime"] is not None and not isinstance(
        payload["cuda_runtime"], str
    ):
        raise ValueError("probe CUDA runtime identity is invalid")
    for name in ("available", "nccl_available"):
        if type(payload[name]) is not bool:
            raise ValueError("probe availability flags must be boolean")
    digest = payload["visible_devices_sha256"]
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("visible device identity must be a SHA-256 digest")
    devices = payload["devices"]
    if not isinstance(devices, list) or len(devices) > 64:
        raise ValueError("probe device inventory is invalid")
    for ordinal, device in enumerate(devices):
        if not isinstance(device, dict) or set(device) != {
            "ordinal",
            "name",
            "total_bytes",
            "free_bytes",
            "compute_capability",
        }:
            raise ValueError("probe device fields do not match")
        if (
            type(device["ordinal"]) is not int
            or device["ordinal"] != ordinal
            or not isinstance(device["name"], str)
            or not device["name"]
        ):
            raise ValueError("probe CUDA ordinals must be contiguous")
        if (
            any(type(device[name]) is not int for name in ("total_bytes", "free_bytes"))
            or not 0 <= device["free_bytes"] <= device["total_bytes"]
            or device["total_bytes"] <= 0
        ):
            raise ValueError("probe device memory observation is invalid")
        capability = device["compute_capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)
        ):
            raise ValueError("probe compute capability is invalid")
    if payload["available"]:
        if (
            payload["reason"] is not None
            or payload["system"] != "Linux"
            or not payload["cuda_runtime"]
            or not payload["nccl_available"]
            or not devices
        ):
            raise ValueError("available NVIDIA probe lacks supporting observations")
    elif payload["reason"] not in (
        "unsupported_operating_system",
        "pytorch_without_cuda",
        "cuda_unavailable",
        "nccl_unavailable",
        "no_visible_devices",
        "device_query_failed",
    ):
        raise ValueError("unavailable NVIDIA probe requires a known reason")
    return payload


def probe_gpu_host() -> dict:
    """Read the actual local runtime; unsupported hosts remain explicitly blocked."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    payload = {
        "schema_version": 1,
        "kind": "nvidia_host_probe",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "nccl_available": bool(
            torch.distributed.is_available() and torch.distributed.is_nccl_available()
        ),
        "visible_devices_sha256": sha256(visible.encode()).hexdigest()
        if visible is not None
        else None,
        "available": False,
        "reason": None,
        "devices": [],
    }
    if payload["system"] != "Linux":
        payload["reason"] = "unsupported_operating_system"
    elif not payload["cuda_runtime"]:
        payload["reason"] = "pytorch_without_cuda"
    elif not torch.cuda.is_available():
        payload["reason"] = "cuda_unavailable"
    elif not payload["nccl_available"]:
        payload["reason"] = "nccl_unavailable"
    else:
        try:
            count = torch.cuda.device_count()
            if not 0 < count <= 64:
                payload["reason"] = "no_visible_devices"
            else:
                for ordinal in range(count):
                    properties = torch.cuda.get_device_properties(ordinal)
                    free, total = torch.cuda.mem_get_info(ordinal)
                    payload["devices"].append(
                        {
                            "ordinal": ordinal,
                            "name": properties.name,
                            "total_bytes": int(total),
                            "free_bytes": int(free),
                            "compute_capability": [properties.major, properties.minor],
                        }
                    )
                payload["available"] = True
        except (RuntimeError, OSError, AttributeError):
            payload.update(available=False, reason="device_query_failed", devices=[])
    return validate_gpu_probe(payload)
