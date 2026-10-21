"""Validated model-bound key/value cache for GPT-2 inference."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class KVCache:
    model_sha256: str
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

    def __post_init__(self):
        if (
            not isinstance(self.model_sha256, str)
            or len(self.model_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.model_sha256
            )
        ):
            raise ValueError("cache must be bound to a model SHA-256")
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("cache must contain key/value pairs for every layer")
        reference = self.keys[0]
        for tensor in (*self.keys, *self.values):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 4
                or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype
                or tensor.device.type != "cpu"
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(
                    "cache tensors must have matching finite CPU batch/head/time/width shapes"
                )
        if (
            reference.shape[0] == 0
            or reference.shape[1] == 0
            or reference.shape[3] == 0
        ):
            raise ValueError("cache batch, head and width dimensions must be positive")

    @property
    def length(self) -> int:
        return self.keys[0].shape[2]

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.keys, *self.values)
        )

    def clone(self):
        return KVCache(
            self.model_sha256,
            tuple(tensor.clone() for tensor in self.keys),
            tuple(tensor.clone() for tensor in self.values),
        )
