"""Requested single-host NVIDIA topology and torchrun rank contracts."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuTopology:
    """Map local worker ranks to CUDA ordinals after CUDA_VISIBLE_DEVICES."""

    devices: tuple[int, ...]
    backend: str = "nccl"
    dtype: str = "bfloat16"

    def __post_init__(self):
        if (
            not isinstance(self.devices, tuple)
            or not 1 <= len(self.devices) <= 64
            or any(
                type(device) is not int or not 0 <= device < 64
                for device in self.devices
            )
            or len(set(self.devices)) != len(self.devices)
        ):
            raise ValueError("devices must contain one to 64 distinct CUDA ordinals")
        if self.backend != "nccl":
            raise ValueError("NVIDIA preflight requires the NCCL backend")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported NVIDIA preflight dtype")

    @property
    def world_size(self):
        return len(self.devices)

    def device_for_rank(self, local_rank: int) -> int:
        if type(local_rank) is not int or not 0 <= local_rank < self.world_size:
            raise ValueError("local rank is outside the requested topology")
        return self.devices[local_rank]

    def to_mapping(self) -> dict:
        return {
            "devices": list(self.devices),
            "backend": self.backend,
            "dtype": self.dtype,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping):
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"devices", "backend", "dtype"}
            or not isinstance(payload["devices"], list)
        ):
            raise ValueError("GPU topology fields do not match")
        return cls(tuple(payload["devices"]), payload["backend"], payload["dtype"])


def validate_rank_environment(
    environment: Mapping[str, str], topology: GpuTopology
) -> dict:
    """Require a complete single-node torchrun world before hardware access."""
    values = {}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        value = environment.get(name)
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise ValueError(f"{name} must be an unsigned decimal torchrun value")
        values[name] = int(value)
    if (
        values["WORLD_SIZE"] != topology.world_size
        or values["LOCAL_WORLD_SIZE"] != topology.world_size
        or values["RANK"] != values["LOCAL_RANK"]
    ):
        raise ValueError("rank environment does not match the single-host topology")
    device = topology.device_for_rank(values["LOCAL_RANK"])
    return {
        "rank": values["RANK"],
        "local_rank": values["LOCAL_RANK"],
        "world_size": topology.world_size,
        "device_ordinal": device,
    }
