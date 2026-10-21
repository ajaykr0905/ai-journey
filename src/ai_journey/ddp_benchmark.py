"""Bounded CPU serving measurements with warmup exclusion and output parity."""

import math
import platform
import time
import uuid

import torch

from .ddp_contracts import integer
from .ddp_service import GenerationRequest


def percentile(samples, quantile: float) -> float:
    samples = sorted(samples)
    if (
        not samples
        or not 0 < quantile <= 1
        or any(not math.isfinite(value) or value < 0 for value in samples)
    ):
        raise ValueError("percentile requires finite nonnegative samples")
    return samples[max(0, math.ceil(quantile * len(samples)) - 1)]


def benchmark_service(
    service,
    request: GenerationRequest,
    *,
    repeats: int = 5,
    warmup: int = 1,
    clock=time.perf_counter,
) -> dict:
    integer(repeats, "repeats", 1)
    integer(warmup, "warmup")
    if repeats > 100 or warmup > 10:
        raise ValueError("benchmark exceeds the CPU measurement budget")
    prefix = uuid.uuid4().hex
    outputs = []
    for index in range(warmup):
        service.generate(
            GenerationRequest(
                f"{prefix}-warm-{index}", request.prompt, request.sampling
            )
        )
    samples = []
    window_start = clock()
    for index in range(repeats):
        started = clock()
        result = service.generate(
            GenerationRequest(f"{prefix}-{index}", request.prompt, request.sampling)
        )
        elapsed = clock() - started
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError("benchmark clock must be monotonic and finite")
        samples.append(elapsed)
        outputs.append(result["tokens"])
    window_seconds = clock() - window_start
    if window_seconds <= 0:
        raise ValueError("benchmark window must have positive duration")
    if any(output != outputs[0] for output in outputs):
        raise RuntimeError("benchmark output parity failed")
    tokens = sum(len(output) for output in outputs)
    return {
        "repeats": repeats,
        "warmup": warmup,
        "window_seconds": window_seconds,
        "tokens": tokens,
        "tokens_per_second": tokens / window_seconds,
        "latency_seconds": {
            "p50": percentile(samples, 0.5),
            "p95": percentile(samples, 0.95),
            "p99": percentile(samples, 0.99),
        },
        "output_parity": True,
        "model_sha256": service.model_sha256,
        "runtime": {
            "device": "cpu",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
