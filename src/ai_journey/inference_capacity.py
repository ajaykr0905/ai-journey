"""Memory admission controls for analytical decoder-serving plans."""

from __future__ import annotations

from dataclasses import dataclass

from .inference_kv import InferenceCapacityError, KVCacheLayout, _positive_integer


@dataclass(frozen=True)
class DeviceMemoryBudget:
    """Declared device bytes reserved for weights, runtime, and KV state."""

    total_bytes: int
    model_bytes: int
    runtime_reserve_bytes: int

    def __post_init__(self) -> None:
        _positive_integer(self.total_bytes, "total_bytes")
        for name in ("model_bytes", "runtime_reserve_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise InferenceCapacityError(f"{name} must be nonnegative")
        if self.model_bytes + self.runtime_reserve_bytes >= self.total_bytes:
            raise InferenceCapacityError(
                "model and runtime reserves must leave positive KV capacity"
            )

    @property
    def kv_bytes(self) -> int:
        return self.total_bytes - self.model_bytes - self.runtime_reserve_bytes


def maximum_resident_sequences(
    budget: DeviceMemoryBudget, layout: KVCacheLayout
) -> int:
    """Return full-context sequences fitting the declared KV allocation."""

    if not isinstance(budget, DeviceMemoryBudget):
        raise TypeError("budget must be DeviceMemoryBudget")
    if not isinstance(layout, KVCacheLayout):
        raise TypeError("layout must be KVCacheLayout")
    per_sequence = layout.total_bytes // layout.batch_size
    return budget.kv_bytes // per_sequence
