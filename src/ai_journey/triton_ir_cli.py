"""Summarize, map, check and verify the Day 80 Triton IR lab from one command line.

``summarize`` checks the fixture folder and prints the manifest, the PTX,
TTIR and TTGIR counts, the pipeline facts and one line for each of the 12
compiles. With ``--output`` it also writes the sealed report. It needs no
triton.

``stage-map`` prints, for each statement of the Day 79 kernel, the TTIR and
TTGIR ops and the PTX families that carry its line.

``check`` compiles every config again with triton for ``sm_80`` and
``sm_90``, compares each stage digest with the dump, and writes a sealed
report. Triton has no macOS wheel, so on the development host ``check``
stops with exit code 3 before it makes the output directory or writes a
byte. A backend injected through ``main`` gives a synthetic report.

``verify`` recomputes a sealed report from the fixture folder on the CPU,
with no triton.

Exit codes: 0 success; 1 a report was read or built but failed verification,
or a regenerated text or metadata differs from the dump (the report is still
written, so the difference can be read); 2 usage errors, unreadable or
invalid inputs and refused outputs; 3 this host cannot compile with triton,
so ``check`` wrote nothing. These are the codes of the Day 73 to 79 labs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .kernel_launch_io import canonical_json_bytes, paths_alias, write_atomic
from .kernel_metal_cli import _refuse_unusable_output
from .triton_ir_manifest import (
    COMMITTED_CONFIG,
    COMMITTED_STAGES,
    COMMITTED_TARGET,
    MANIFEST_NAME,
    load_manifest,
    stage_path,
)
from .triton_ir_report import (
    DIGEST_FIELD,
    RECORDED_LABEL,
    REGENERATION_LABEL,
    analyze,
    build_report,
    seal_report,
)
from .triton_ir_runtime import CompilerBackend, TritonIrUnavailableError
from .triton_ir_verify import TritonIrVerificationError, load_report, verify_report

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-triton-ir"
NAME = "triton-ir"
KINDS = {
    RECORDED_LABEL: "recorded",
    REGENERATION_LABEL: "regenerated",
}


def build_parser() -> argparse.ArgumentParser:
    """Define the four commands, each with explicit paths and no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Read the TTIR, TTGIR and PTX that triton wrote for the Day 79 "
            "matmul kernel, map them to the kernel lines, and check or verify "
            "the sealed report. Nothing runs on a GPU."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser(
        "summarize",
        help="print the counts of the committed texts and of all 12 compiles",
        allow_abbrev=False,
    )
    summarize.add_argument("--fixtures", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    mapping = commands.add_parser(
        "stage-map",
        help="print the ops and PTX families of each kernel statement",
        allow_abbrev=False,
    )
    mapping.add_argument("--fixtures", type=Path, required=True)
    check = commands.add_parser(
        "check",
        help="compile again with triton and compare every stage digest",
        allow_abbrev=False,
    )
    check.add_argument("--fixtures", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed report from the fixtures, without triton",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--fixtures", type=Path, required=True)
    return parser


def _pairs(counts: Mapping[str, int], *, nonzero: bool = False) -> str:
    return " ".join(
        f"{name}={count}" for name, count in counts.items() if count or not nonzero
    )


def _ops(counts: Mapping[str, int]) -> str:
    return ",".join(f"{name}:{count}" for name, count in counts.items()) or "none"


def report_lines(report: Mapping[str, object]) -> list[str]:
    """Return the lines that ``summarize`` and ``check`` print before the digest."""

    manifest = report["manifest"]
    ptx = report["ptx"]
    ttir = report["ttir"]
    ttgir = report["ttgir"]
    pipeline = report["pipeline"]
    loop = ptx["loops"][0]
    registers = ",".join(f"{r['class']}:{r['declared']}" for r in ptx["registers"])
    ranges = ",".join(f"{r['start']}:{r['end']}" for r in ttir["tt_make_range"])
    lines = [
        f"triton_ir manifest triton={manifest['triton']} python={manifest['python']} "
        f"machine={manifest['machine']} "
        f"fp_fusion={manifest['TRITON_DEFAULT_FP_FUSION']} "
        f"source_sha256={report['source']['sha256']} "
        f"manifest_sha256={report['manifest_sha256']}",
        f"triton_ir ptx target={ptx['target']} version={ptx['version']} "
        f"address_size={ptx['address_size']} entry={ptx['entry']} "
        f"params={len(ptx['parameters'])} reqntid={ptx['reqntid']} "
        f"registers={registers} shared_extern={ptx['shared'][0]['name']} "
        f"instructions={ptx['instructions']} predicated={ptx['predicated']}",
        f"triton_ir ptx_families {_pairs(ptx['families'])}",
        f"triton_ir ptx_k_loop label={loop['label']} "
        f"lines={loop['first_line']}-{loop['last_line']} "
        f"instructions={loop['instructions']} {_pairs(loop['families'], nonzero=True)}",
        f"triton_ir ttir ops={ttir['ops']} tt.load={ttir['tt_load']} "
        f"masked={ttir['tt_load_masked']} tt.store={ttir['tt_store']} "
        f"tt.dot={ttir['tt_dot']} scf.for={ttir['scf_for']} "
        f"tt.splat={ttir['tt_splat']} make_range={ranges}",
        f"triton_ir ttgir ops={ttgir['ops']} target={ttgir['target']} "
        f"num_warps={ttgir['num_warps']} "
        f"threads_per_warp={ttgir['threads_per_warp']} "
        f"encodings={','.join(v['kind'] for v in ttgir['encodings'].values())} "
        f"mma_encoding={str(ttgir['mma_encoding']).lower()} "
        f"async_copies={ttgir['async_copy_global_to_local']} "
        f"async_wait={','.join(str(n) for n in ttgir['async_wait'])} "
        f"local_alloc={','.join(ttgir['local_alloc'])}",
        f"triton_ir pipeline config={pipeline['config']} "
        f"num_stages={pipeline['num_stages']} "
        f"buffers={pipeline['buffers_per_operand']} "
        f"shared_bytes={pipeline['shared_bytes']} "
        f"prologue_groups={pipeline['prologue_commit_groups']} "
        f"wait_in_loop={','.join(str(n) for n in pipeline['wait_group_in_k_loop'])} "
        f"fma_per_thread={pipeline['fma_in_k_loop']} "
        f"mma={pipeline['mma_instructions']}",
    ]
    for row in report["comparison"]["compiles"]:
        lines.append(
            f"triton_ir compile target={row['target']} config={row['config']} "
            f"shared_bytes={row['shared_bytes']} ring={row['ring_buffers']} "
            f"ttgir_lines={row['lines']['ttgir']} llir_lines={row['lines']['llir']} "
            f"ptx_lines={row['lines']['ptx']} cubin_bytes={row['cubin_bytes']} "
            f"derived_fma={row['derived_fma_per_thread_per_iteration']}"
        )
    for row in report["comparison"]["targets"]:
        same = row["identical"]
        lines.append(
            f"triton_ir sm_90_vs_sm_80 config={row['config']} "
            f"ttir_same={str(same['ttir']).lower()} "
            f"ttgir_same={str(same['ttgir']).lower()} "
            f"ptx_line_change={row['line_change']['ptx']:+d} "
            f"llir_line_change={row['line_change']['llir']:+d}"
        )
    regeneration = report["regeneration"]
    if regeneration is not None:
        summary = regeneration["summary"]
        lines.append(
            f"triton_ir regeneration backend={regeneration['backend']} "
            f"triton={regeneration['environment']['triton']} "
            f"texts={summary['texts_matched']}/{summary['texts']} "
            f"metadata={summary['metadata_matched']}/{summary['compiles']} "
            f"cubins_sm_80={summary['cubins_by_target']['sm_80']}/6 "
            f"cubins_sm_90={summary['cubins_by_target']['sm_90']}/6"
        )
    return lines


def stage_map_lines(report: Mapping[str, object]) -> list[str]:
    return [
        f"stage_map line={row['line']} key={row['key']} ttir={_ops(row['ttir'])} "
        f"ttgir={_ops(row['ttgir'])} ptx={_ops(row['ptx'])} "
        f"k_loop={_ops(row['ptx_k_loop'])}"
        for row in report["stage_map"]["statements"]
    ]


def summary_line(prefix: str, report: Mapping[str, object]) -> str:
    """Return the one line that names a sealed report and its main counts."""

    summary = report["summary"]
    load = report["host_load"]
    kind = KINDS.get(report["label"], "synthetic")
    return (
        f"{prefix}_{kind}={report[DIGEST_FIELD]} "
        f"ptx_instructions={summary['ptx_instructions']} "
        f"fma={summary['fma_rn_f32']} cp_async={summary['cp_async_all']} "
        f"mma={summary['mma']} compiles={summary['compiles']} "
        f"load_before={','.join(f'{v:.2f}' for v in load['before'])} "
        f"load_after={','.join(f'{v:.2f}' for v in load['after'])} "
        "gpu=false ptx_executed=false"
    )


def _sealed(
    fixtures: Path,
    clock: Callable[[], float],
    load_average: Callable[[], tuple[float, float, float]],
    backend: object | None = None,
) -> dict[str, object]:
    report = seal_report(
        build_report(fixtures, clock=clock, load_average=load_average, backend=backend)
    )
    verify_report(report, fixtures)
    return report


def _stage_texts(fixtures: Path) -> tuple[Path, ...]:
    return tuple(
        fixtures / stage_path(COMMITTED_TARGET, COMMITTED_CONFIG, stage)
        for stage in COMMITTED_STAGES
    )


def _refuse_output(output: Path, fixtures: Path) -> None:
    """Refuse an output that would replace the manifest or a committed text.

    The committed texts are the only copy of the recorded dump in the
    repository, so a report must never take the place of one of them.
    """

    _refuse_unusable_output(output, fixtures / MANIFEST_NAME)
    if any(paths_alias(output, text) for text in _stage_texts(fixtures)):
        raise ValueError("report output must not alias a committed stage text")


def _write(output: Path, fixtures: Path, report: Mapping[str, object]) -> None:
    write_atomic(
        output,
        canonical_json_bytes(report) + b"\n",
        protected=(fixtures / MANIFEST_NAME, *_stage_texts(fixtures)),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: object | None = None,
    clock: Callable[[], float] | None = None,
    load_average: Callable[[], tuple[float, float, float]] | None = None,
) -> int:
    """Run one command and return its exit code; the keywords are for tests.

    A replaced backend makes a check report synthetic. The triton compiler
    backend refuses a replaced clock or load average.
    """

    arguments = build_parser().parse_args(argv)
    clock = clock or time.time
    load_average = load_average or os.getloadavg
    status = EXIT_OK
    try:
        if arguments.command == "summarize":
            if arguments.output is not None:
                _refuse_output(arguments.output, arguments.fixtures)
            report = _sealed(arguments.fixtures, clock, load_average)
            if arguments.output is not None:
                _write(arguments.output, arguments.fixtures, report)
            lines = [*report_lines(report), summary_line("triton_ir", report)]
        elif arguments.command == "stage-map":
            lines = stage_map_lines(analyze(arguments.fixtures))
        elif arguments.command == "verify":
            report = load_report(arguments.report, arguments.fixtures)
            lines = [summary_line("verified_triton_ir", report)]
        else:
            _refuse_output(arguments.output, arguments.fixtures)
            load_manifest(arguments.fixtures)
            chosen = CompilerBackend() if backend is None else backend
            report = _sealed(arguments.fixtures, clock, load_average, chosen)
            _write(arguments.output, arguments.fixtures, report)
            lines = [*report_lines(report), summary_line("triton_ir", report)]
            regenerated = report["regeneration"]["summary"]
            if not regenerated["all_texts_and_metadata_match"]:
                print(
                    f"{NAME} check rejected: {regenerated['texts_matched']} of "
                    f"{regenerated['texts']} texts and "
                    f"{regenerated['metadata_matched']} of "
                    f"{regenerated['compiles']} metadata records match the dump",
                    file=sys.stderr,
                )
                status = EXIT_REJECTED
    except TritonIrUnavailableError as error:
        print(f"{NAME} {arguments.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except TritonIrVerificationError as error:
        print(f"{NAME} {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"{NAME} {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
