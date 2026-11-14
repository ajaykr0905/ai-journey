"""Explicit contracts for bounded CPU distributed training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping


def integer(value: int, name: str, minimum: int = 0) -> int:
    """Return ``value`` if it is an int, not a bool, of at least ``minimum``.

    Rejecting bools and floats keeps True from passing as rank 1 and 2.0 as a
    world size. Raises ValueError naming the field.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Topology:
    """One process's rank, world size and local rank, validated together.

    Both ranks must lie inside the world; raises ValueError otherwise.
    """

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
        """True only for rank 0, the one rank designated to write outputs."""
        return self.rank == 0

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Topology:
        """Build a topology from torchrun's RANK, WORLD_SIZE and LOCAL_RANK.

        Each must be an unsigned ASCII decimal string, so a malformed launch
        environment fails before any collective. Raises ValueError.
        """
        values = {}
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            raw = environment.get(name)
            if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
                raise ValueError(f"{name} must be an unsigned decimal integer")
            values[name] = int(raw)
        return cls(values["RANK"], values["WORLD_SIZE"], values["LOCAL_RANK"])
