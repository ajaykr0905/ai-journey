"""Explicit contracts for bounded CPU distributed training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping


def integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Topology:
    rank: int
    world_size: int
    local_rank: int = 0

    def __post_init__(self):
        integer(self.world_size, "world_size", 1)
        integer(self.rank, "rank")
        integer(self.local_rank, "local_rank")
        if self.rank >= self.world_size or self.local_rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")

    @property
    def is_writer(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Topology:
        values = {}
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            raw = environment.get(name)
            if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
                raise ValueError(f"{name} must be an unsigned decimal integer")
            values[name] = int(raw)
        return cls(values["RANK"], values["WORLD_SIZE"], values["LOCAL_RANK"])
