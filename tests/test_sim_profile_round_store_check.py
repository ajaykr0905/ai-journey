"""Round 4: a float32 store checks exactness without calling to_float32."""

import math

import pytest

from ai_journey.kernel_launch_buffer import AccessRecorder, DeviceBuffer, to_float32
from ai_journey.kernel_launch_config import Dim3, KernelLaunchError, LaunchConfig
from ai_journey.kernel_launch_reference import seeded_float32
from ai_journey.sim_profile_harness import profile_call

CONFIG = LaunchConfig(Dim3(1), Dim3(2))


@pytest.mark.parametrize(
    "value",
    [0.1, 1.0 + 2**-30, 2.0**-150, 3.5e38, -1e39, math.inf, -math.inf, math.nan, 1, True, "0.5", None],
)  # fmt: skip
def test_inexact_or_foreign_values_are_refused_with_the_same_message(value):
    with pytest.raises(KernelLaunchError, match="round it with to_float32"):
        DeviceBuffer("out", "float32", [value])
    out = DeviceBuffer.zeros("out", "float32", 1)
    recorder = AccessRecorder(CONFIG, [out])
    recorder.enter(0)
    with pytest.raises(KernelLaunchError, match="round it with to_float32"):
        out[0] = value
    recorder.leave()
    assert recorder.finish().writes == 0


def test_exact_values_are_stored_bit_for_bit():
    values = [
        *seeded_float32(75, "round-four", 500),
        -0.0,
        2.0**-149,
        3.4028234663852886e38,
    ]
    buffer = DeviceBuffer("out", "float32", values)

    assert [math.copysign(1.0, v) for v in buffer.to_host()] == [
        math.copysign(1.0, v) for v in values
    ]
    assert buffer.to_host() == tuple(values)


def test_a_store_calls_to_float32_no_more():
    values = [to_float32(0.1 * index) for index in range(10)]
    records, _ = profile_call(lambda: DeviceBuffer("out", "float32", values))
    calls = {record.function: record.calls for record in records}

    assert calls["ai_journey.kernel_launch_buffer:_check_value"] == 10
    assert "ai_journey.kernel_launch_buffer:to_float32" not in calls
