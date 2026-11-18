"""Build, load and publish only verified, canonical bottleneck reports.

``report_from_files`` reads the three Day 74 reports through their own
verified loaders, parses the timing input and seals the report. Reads reuse
the bounded reader of ``memory_access_report_io``, and a loaded report must
pass the verifier and be canonical JSON followed by one newline. Given all
four input paths, the loader also rebuilds the report from them, and the
report must equal that rebuild. Writes verify first and then publish through
``memory_access_report_io.write_report``, which refuses an output that
aliases a protected input such as a source report.

Unreadable or malformed files raise ``MatmulBottleneckError``; a report that
was read but fails verification raises ``MatmulBottleneckVerificationError``.
Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .matmul_bottleneck_config import (
    BottleneckConfig,
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
    config_from_mapping,
)
from .matmul_bottleneck_report import build_report
from .matmul_bottleneck_sources import sources_from_files
from .matmul_bottleneck_timing import load_timing
from .matmul_bottleneck_verify import verify_report
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import (
    MAX_REPORT_BYTES,
    parse_json_object,
    read_bounded_bytes,
    write_report,
)


@dataclass(frozen=True)
class InputPaths:
    """The four inputs of one report: three Day 74 reports and the timing."""

    matmul_report: Path
    ceiling_report: Path
    gap_report: Path
    timing: Path

    def __post_init__(self) -> None:
        for path in self.all():
            if not isinstance(path, Path):
                raise TypeError("input paths must be pathlib.Path")

    def all(self) -> tuple[Path, Path, Path, Path]:
        return self.matmul_report, self.ceiling_report, self.gap_report, self.timing


def report_from_files(paths: InputPaths, config: BottleneckConfig) -> dict:
    """Verify the Day 74 sources, parse the timing and seal the report."""

    sources = sources_from_files(
        paths.matmul_report, paths.ceiling_report, paths.gap_report
    )
    return build_report(sources, load_timing(paths.timing), config)


def load_report(
    path: Path,
    *,
    config: BottleneckConfig | None = None,
    inputs: InputPaths | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read, verify and canonical-check one report; return it with its summary.

    The report's own checks run before any input is read, so a tampered
    report is named as such first. The rebuild needs a config, and it uses
    the supplied one or else the embedded one, which the verifier checked.
    """

    try:
        payload = read_bounded_bytes(path, max_bytes=MAX_REPORT_BYTES, label="report")
        report = parse_json_object(payload, label="report")
    except ValueError as error:
        raise MatmulBottleneckError(str(error)) from error
    summary = verify_report(report, config=config)
    if inputs is not None:
        rebuild_config = config or config_from_mapping(report["config"])
        rebuilt = report_from_files(inputs, rebuild_config)
        summary = verify_report(report, config=config, rebuilt=rebuilt)
    if payload != canonical_json_bytes(report) + b"\n":
        raise MatmulBottleneckVerificationError("report file is not canonical JSON")
    return report, summary


def save_report(
    path: Path, report: dict[str, object], *, protected: Iterable[Path] = ()
) -> dict[str, object]:
    """Verify a sealed report, then publish it atomically; return its summary."""

    summary = verify_report(report)
    try:
        write_report(path, report, protected=tuple(protected))
    except ValueError as error:
        raise MatmulBottleneckError(str(error)) from error
    return summary
