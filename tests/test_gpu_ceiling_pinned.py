import json
from pathlib import Path

from ai_journey.gpu_ceiling_config import (
    MAX_RUN_ACCESSES,
    MAX_RUN_BYTES,
    MAX_RUN_CALLS,
    MAX_RUN_FLOPS,
    PlannedWork,
    load_config,
)
from ai_journey.gpu_ceiling_reference import stream_threads
from ai_journey.gpu_ceiling_source import BANK_MAX_STRIDE, FMA_CHAIN_COUNTS

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-74-gpu-ceilings.json"


def test_the_pinned_config_sweeps_every_kernel_the_source_compiles():
    config = load_config(PINNED)

    assert config.name == "day-74-gpu-ceilings" and config.seed == 74
    assert config.fma.chain_counts == FMA_CHAIN_COUNTS
    assert config.fma.fma_steps == (32, 128, 512, 2048, 8192)
    assert config.bank.strides == tuple(2**power for power in range(6))
    assert config.bank.strides[-1] == BANK_MAX_STRIDE
    assert config.bandwidth.buffer_bytes == (1 << 24, 1 << 26, 1 << 28)
    assert config.to_mapping() == json.loads(PINNED.read_text(encoding="utf-8"))


def test_the_pinned_run_is_counted_and_fits_every_work_bound():
    work = load_config(PINNED).planned_work()

    assert work == PlannedWork(
        correctness_cases=20,
        fma_points=30,
        stream_points=6,
        bank_points=12,
        timed_samples=528,
        calls=2184,
        flops=18742096429056,
        stream_bytes=84406173696,
        bank_accesses=670014898176,
    )
    assert work.calls <= MAX_RUN_CALLS and work.flops <= MAX_RUN_FLOPS
    assert work.stream_bytes <= MAX_RUN_BYTES
    assert work.bank_accesses <= MAX_RUN_ACCESSES


def test_every_pinned_correctness_launch_has_threads_past_the_data():
    config = load_config(PINNED)
    cases = config.correctness

    assert cases.threads % config.fma.group_size
    assert cases.threads % config.bank.group_size
    for kernel in ("stream_copy", "stream_read_reduce"):
        walkers = stream_threads(
            cases.stream_float4, config.bandwidth.per_thread(kernel)
        )
        assert walkers % config.bandwidth.group_size, kernel
    assert cases.fma_steps % 8 == 0
