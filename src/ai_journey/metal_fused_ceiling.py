"""Take the bandwidth ceiling of a softmax run from a verified gpu-ceilings report.

A softmax moves each element at least twice, so a rate means most next to
what the memory of this host can deliver. The Day 74 gpu-ceilings lab
measures that ceiling on the host itself: the better median GB/s of a
streaming copy and a streaming read-reduce at its largest buffer. This
module reads such a report only through the Day 74 loader, so the Day 74
verifier must accept it first, and a tampered report sets no ceiling.

The ceiling keeps the identity of its report: the digest, the config
digest, the label, the load averages of that separate run, its buffer and
its environment. A fraction of the ceiling is a measurement only when both
the softmax run and the ceiling report are measurements from the same
interpreter, libraries and platform. A fraction above 1 is possible: the
ceiling is the best median of the Day 74 kernels at one buffer size, not a
hardware limit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .gpu_ceiling_io import load_report
from .gpu_ceiling_report import MEASUREMENT_LABEL, SYNTHETIC_LABEL
from .gpu_ceiling_source import STREAM_KERNELS
from .kernel_metal_report import ENVIRONMENT_FIELDS
from .metal_fused_source import MetalFusedError

SOURCE = "gpu_ceilings"
DEFINITION = (
    "the bandwidth ceiling of a verified Day 74 gpu-ceilings report: the better "
    "median GB/s of stream_copy and stream_read_reduce at its largest buffer, on "
    "the host that made it; not a vendor peak and not a hardware limit"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _loads(values: object) -> list[float]:
    if (
        type(values) is not list
        or len(values) != 3
        or not all(
            type(value) is float and math.isfinite(value) and value >= 0
            for value in values
        )
    ):
        raise MetalFusedError("a ceiling load average must be three nonnegative floats")
    return list(values)


@dataclass(frozen=True)
class BandwidthCeiling:
    """The measured bandwidth ceiling of one verified report, with its identity."""

    report_sha256: str
    config_sha256: str
    measured: bool
    label: str
    gb_per_second: float
    kernel: str
    buffer_bytes: int
    load_average_before: list[float]
    load_average_after: list[float]
    environment: dict[str, str]

    def __post_init__(self) -> None:
        for digest in (self.report_sha256, self.config_sha256):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise MetalFusedError("a ceiling digest must be a SHA-256 digest")
        if type(self.measured) is not bool or self.label != (
            MEASUREMENT_LABEL if self.measured else SYNTHETIC_LABEL
        ):
            raise MetalFusedError("the ceiling label must match whether it measured")
        rate = self.gb_per_second
        if type(rate) is not float or not math.isfinite(rate) or rate <= 0:
            raise MetalFusedError("the bandwidth ceiling must be a positive float")
        if self.kernel not in STREAM_KERNELS:
            raise MetalFusedError("the ceiling must come from a Day 74 stream kernel")
        if type(self.buffer_bytes) is not int or self.buffer_bytes <= 0:
            raise MetalFusedError("the ceiling buffer must be a positive byte count")
        _loads(self.load_average_before)
        _loads(self.load_average_after)
        if type(self.environment) is not dict or set(self.environment) != set(
            ENVIRONMENT_FIELDS
        ):
            raise MetalFusedError("the ceiling environment must hold the host fields")

    def same_environment(self, environment: dict[str, str]) -> bool:
        """Whether a run used the interpreter, libraries and platform of this report."""

        return all(
            environment.get(field) == self.environment[field]
            for field in ENVIRONMENT_FIELDS
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the ceiling and its identity as a softmax report records them."""

        return {
            "source": SOURCE,
            "definition": DEFINITION,
            "report_sha256": self.report_sha256,
            "config_sha256": self.config_sha256,
            "measured": self.measured,
            "label": self.label,
            "gb_per_second": self.gb_per_second,
            "kernel": self.kernel,
            "buffer_bytes": self.buffer_bytes,
            "load_average_before": list(self.load_average_before),
            "load_average_after": list(self.load_average_after),
            "environment": dict(self.environment),
        }


def ceiling_from_report(
    report: dict[str, object], summary: dict[str, object]
) -> BandwidthCeiling:
    """Build the ceiling from a report and the summary that its verifier returned."""

    sealed = report["summary"]
    return BandwidthCeiling(
        report_sha256=summary["report_sha256"],
        config_sha256=summary["config_sha256"],
        measured=summary["measured"],
        label=summary["label"],
        gb_per_second=summary["bandwidth_ceiling_gb_per_second"],
        kernel=sealed["bandwidth_ceiling_kernel"],
        buffer_bytes=sealed["bandwidth_ceiling_buffer_bytes"],
        load_average_before=summary["load_average_before"],
        load_average_after=summary["load_average_after"],
        environment=dict(report["environment"]),
    )


def load_ceiling(path: Path) -> BandwidthCeiling:
    """Read one gpu-ceilings report through the verified Day 74 loader."""

    try:
        report, summary = load_report(path)
    except ValueError as error:
        raise MetalFusedError(f"the ceiling report was refused: {error}") from error
    return ceiling_from_report(report, summary)
