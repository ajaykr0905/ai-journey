"""Wave quantization and tail effect of a grid on one declared device.

With ``A`` resident blocks per SM and ``S`` SMs, one wave runs ``S * A``
blocks, so a grid of ``B`` blocks needs ``ceil(B / (S * A))`` waves and its
last wave may be only partly filled. Utilization assumes equal block run
times and a full wave of SM slots per wave; the hardware block scheduler,
uneven block durations, and concurrent kernels are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .simt_device import SimtDevice, require_int, require_text


@dataclass(frozen=True)
class WaveQuantization:
    """Waves, tail fill, and SM-slot utilization for one grid on one device."""

    device_key: str
    total_blocks: int
    sm_count: int
    active_blocks_per_sm: int

    def __post_init__(self) -> None:
        require_text(self.device_key, "device_key", maximum=64)
        require_int(self.total_blocks, "total_blocks", minimum=1, maximum=2**62)
        require_int(self.sm_count, "sm_count", minimum=1, maximum=4096)
        require_int(
            self.active_blocks_per_sm, "active_blocks_per_sm", minimum=1, maximum=256
        )

    @property
    def slots_per_wave(self) -> int:
        return self.sm_count * self.active_blocks_per_sm

    @property
    def full_waves(self) -> int:
        return self.total_blocks // self.slots_per_wave

    @property
    def tail_blocks(self) -> int:
        return self.total_blocks % self.slots_per_wave

    @property
    def waves(self) -> int:
        return self.full_waves + (1 if self.tail_blocks else 0)

    @property
    def last_wave_fill(self) -> Fraction:
        """Return the occupied share of the last wave's block slots."""

        if not self.tail_blocks:
            return Fraction(1)
        return Fraction(self.tail_blocks, self.slots_per_wave)

    @property
    def sm_slot_utilization(self) -> Fraction:
        """Return blocks run divided by block slots offered across all waves."""

        return Fraction(self.total_blocks, self.waves * self.slots_per_wave)

    def to_mapping(self) -> dict[str, object]:
        return {
            "device": self.device_key,
            "total_blocks": self.total_blocks,
            "sm_count": self.sm_count,
            "active_blocks_per_sm": self.active_blocks_per_sm,
            "slots_per_wave": self.slots_per_wave,
            "full_waves": self.full_waves,
            "tail_blocks": self.tail_blocks,
            "waves": self.waves,
            "last_wave_fill": float(self.last_wave_fill),
            "sm_slot_utilization": float(self.sm_slot_utilization),
        }


def wave_quantization(
    device: SimtDevice, *, total_blocks: int, active_blocks_per_sm: int
) -> WaveQuantization:
    """Quantize a grid into waves of ``sm_count * active_blocks_per_sm`` blocks."""

    if not isinstance(device, SimtDevice):
        raise TypeError("device must be SimtDevice")
    require_int(
        active_blocks_per_sm,
        "active_blocks_per_sm",
        minimum=1,
        maximum=device.architecture.max_blocks_per_sm,
    )
    return WaveQuantization(
        device_key=device.key,
        total_blocks=total_blocks,
        sm_count=device.sm_count,
        active_blocks_per_sm=active_blocks_per_sm,
    )
