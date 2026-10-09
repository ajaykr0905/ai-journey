"""Deterministic Day 35 causal-average experiment and evidence gates."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from .causal_average import (
    CausalAverageAudit,
    CausalAverageError,
    CausalGradientAudit,
    CausalWeightAudit,
    audit_equivalence,
    audit_gradient_equivalence,
    audit_weights,
    causal_average_cumsum,
    causal_average_loop,
    causal_average_matmul,
    causal_average_softmax,
    future_influence_error,
    masked_softmax_weights,
    padding_influence_error,
    require_equivalence,
    require_weight_safety,
    triangular_average_weights,
)
from .causal_stream import CausalAverageStream


REPORT_SCHEMA = "ai-journey-day-35-v1"


@dataclass(frozen=True)
class CausalExperimentConfig:
    """Bounded controls for a CPU float64 causal-average audit."""

    batch_size: int = 3
    time: int = 8
    channels: int = 4
    seed: int = 35
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("batch_size", "time", "channels", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in ("batch_size", "time", "channels"):
            if getattr(self, name) <= 0:
                raise CausalAverageError(f"{name} must be positive")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise CausalAverageError("tolerance must be finite and non-negative")


@dataclass(frozen=True)
class CausalExperimentResult:
    """Measured equivalence, safety, gradient, padding, and stream results."""

    config: CausalExperimentConfig
    input_sha256: str
    forward: CausalAverageAudit
    gradient: CausalGradientAudit
    triangular_weights: CausalWeightAudit
    softmax_weights: CausalWeightAudit
    loop_future_error: float
    matmul_future_error: float
    softmax_future_error: float
    cumsum_future_error: float
    padding_error: float
    streaming_error: float


def _tensor_fingerprint(values: torch.Tensor) -> str:
    header = f"{tuple(values.shape)}:{values.dtype}".encode()
    return sha256(header + values.detach().contiguous().numpy().tobytes()).hexdigest()


def run_causal_experiment(
    config: CausalExperimentConfig = CausalExperimentConfig(),
) -> CausalExperimentResult:
    """Run every Day 35 contract on one deterministic synthetic batch."""

    if not isinstance(config, CausalExperimentConfig):
        raise TypeError("config must be CausalExperimentConfig")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    values = torch.randn(
        config.batch_size,
        config.time,
        config.channels,
        generator=generator,
        dtype=torch.float64,
    )
    forward = audit_equivalence(values)
    require_equivalence(forward, tolerance=config.tolerance)
    gradient = audit_gradient_equivalence(values)
    require_equivalence(
        CausalAverageAudit(
            gradient.matmul_error, gradient.softmax_error, gradient.cumsum_error
        ),
        tolerance=config.tolerance,
    )
    triangular = audit_weights(
        triangular_average_weights(config.time, dtype=torch.float64)
    )
    softmax = audit_weights(masked_softmax_weights(config.time, dtype=torch.float64))
    require_weight_safety(triangular, tolerance=config.tolerance)
    require_weight_safety(softmax, tolerance=config.tolerance)

    lengths = torch.tensor(
        [config.time - (index % config.time) for index in range(config.batch_size)]
    )
    padding_error = padding_influence_error(values, lengths)

    stream = CausalAverageStream(config.channels, dtype=torch.float64)
    split = max(1, config.time // 2)
    chunks = [values[0, :split]]
    if split < config.time:
        chunks.append(values[0, split:])
    streamed = torch.cat([stream.update(chunk) for chunk in chunks])
    streaming_error = float(
        (streamed - causal_average_cumsum(values[0])).abs().max().item()
    )

    result = CausalExperimentResult(
        config=config,
        input_sha256=_tensor_fingerprint(values),
        forward=forward,
        gradient=gradient,
        triangular_weights=triangular,
        softmax_weights=softmax,
        loop_future_error=future_influence_error(values, causal_average_loop),
        matmul_future_error=future_influence_error(values, causal_average_matmul),
        softmax_future_error=future_influence_error(values, causal_average_softmax),
        cumsum_future_error=future_influence_error(values, causal_average_cumsum),
        padding_error=padding_error,
        streaming_error=streaming_error,
    )
    for name in (
        "loop_future_error",
        "matmul_future_error",
        "softmax_future_error",
        "cumsum_future_error",
        "padding_error",
        "streaming_error",
    ):
        if getattr(result, name) > config.tolerance:
            raise CausalAverageError(f"{name} exceeds tolerance")
    return result


def _report_fingerprint(core: dict[str, Any]) -> str:
    encoded = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(encoded).hexdigest()


def build_experiment_report(result: CausalExperimentResult) -> dict[str, Any]:
    """Build canonical, self-verifying JSON evidence from a measured result."""

    if not isinstance(result, CausalExperimentResult):
        raise TypeError("result must be CausalExperimentResult")
    core = {"schema": REPORT_SCHEMA, **asdict(result)}
    return {**core, "report_sha256": _report_fingerprint(core)}


def verify_experiment_report(report: dict[str, Any]) -> None:
    """Reject malformed, failing, or manually changed Day 35 evidence."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    expected = {
        "schema",
        "config",
        "input_sha256",
        "forward",
        "gradient",
        "triangular_weights",
        "softmax_weights",
        "loop_future_error",
        "matmul_future_error",
        "softmax_future_error",
        "cumsum_future_error",
        "padding_error",
        "streaming_error",
        "report_sha256",
    }
    if set(report) != expected:
        raise CausalAverageError("experiment report fields are invalid")
    if report["schema"] != REPORT_SCHEMA:
        raise CausalAverageError("experiment report schema is unsupported")
    core = {key: report[key] for key in expected - {"report_sha256"}}
    if report["report_sha256"] != _report_fingerprint(core):
        raise CausalAverageError("experiment report fingerprint mismatch")
    try:
        config = CausalExperimentConfig(**report["config"])
        input_sha256 = report["input_sha256"]
        errors = [
            *report["forward"].values(),
            *report["gradient"].values(),
            report["triangular_weights"]["row_sum_error"],
            report["triangular_weights"]["maximum_future_weight"],
            -report["triangular_weights"]["minimum_weight"],
            report["softmax_weights"]["row_sum_error"],
            report["softmax_weights"]["maximum_future_weight"],
            -report["softmax_weights"]["minimum_weight"],
            report["loop_future_error"],
            report["matmul_future_error"],
            report["softmax_future_error"],
            report["cumsum_future_error"],
            report["padding_error"],
            report["streaming_error"],
        ]
    except (KeyError, TypeError, AttributeError) as exc:
        raise CausalAverageError("experiment report values are malformed") from exc
    if (
        not isinstance(input_sha256, str)
        or len(input_sha256) != 64
        or any(character not in "0123456789abcdef" for character in input_sha256)
    ):
        raise CausalAverageError("input fingerprint is invalid")
    if any(
        isinstance(error, bool)
        or not isinstance(error, (int, float))
        or not isfinite(error)
        or error > config.tolerance
        for error in errors
    ):
        raise CausalAverageError("experiment report contains a failing error metric")


def write_experiment_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically publish a verified experiment report."""

    verify_experiment_report(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(report, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
