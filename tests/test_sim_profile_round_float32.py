"""Round 2: to_float32 rounds through one precompiled Struct."""

import math
import struct

import pytest

from ai_journey.kernel_launch_buffer import DeviceBuffer, to_float32
from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.kernel_launch_reference import seeded_float32
from ai_journey.sim_profile_harness import profile_call


def old_rounding(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_the_rounding_keeps_every_bit_of_the_module_level_struct_calls():
    values = [
        *seeded_float32(75, "round-two", 2000),
        *(x * y for x, y in zip(seeded_float32(1, "a", 500), seeded_float32(2, "b", 500))),
        0.1,
        -0.0,
        1.0 + 2**-24,
        2.0**-149,
        2.0**-150,
        3.4028234663852886e38,
    ]  # fmt: skip

    for value in values:
        assert struct.pack("<d", to_float32(value)) == struct.pack(
            "<d", old_rounding(value)
        )


def test_errors_are_unchanged():
    for bad in (math.inf, -math.inf, math.nan, 1, True, "1.0"):
        with pytest.raises(KernelLaunchError, match="finite float"):
            to_float32(bad)
    with pytest.raises(KernelLaunchError, match="outside the float32 range"):
        to_float32(3.5e38)
    with pytest.raises(KernelLaunchError, match="round it with to_float32"):
        DeviceBuffer("out", "float32", [0.1])


def test_a_rounding_calls_one_pack_and_one_unpack_method():
    records, result = profile_call(lambda: [to_float32(0.1) for _ in range(5)])
    calls = {record.function: record.calls for record in records}

    assert result == [old_rounding(0.1)] * 5
    assert calls["builtins:<method 'pack' of '_struct.Struct' objects>"] == 5
    assert calls["builtins:<method 'unpack' of '_struct.Struct' objects>"] == 5
    assert "builtins:<built-in method _struct.pack>" not in calls
