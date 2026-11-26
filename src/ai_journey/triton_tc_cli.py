"""Compile, model and verify the Day 82 Tensor Core lab from one command line.

``model`` prints the speedup model of the declared A100 and H100 inputs: the
per-SM rates, the peaks, and for each size the fp32, tf32 and bf16 times,
their bounds and the ratio fp32 : tf32 : bf16. It is a model, not a
measurement. With ``--output`` it also writes the sealed model report. It
needs no triton.

``compile`` compiles the four variants with triton for sm_80 and sm_90,
applies the instruction-class rule to each compile and writes a sealed
report; with ``--dump`` it also writes every TTGIR and PTX text. Triton has
no macOS wheel, so on the development host ``compile`` stops with exit code
3 before it makes a folder or writes a byte. A backend injected through
``main`` gives a synthetic report.

``verify`` recomputes a sealed report from the config on the CPU, and with
``--dump`` it recounts every text of the dump folder. It needs no triton.

Exit codes: 0 success; 1 a report failed verification, or a compile broke
the rule (the report is still written, so the cause can be read); 2 usage
errors, unreadable or invalid inputs and refused outputs; 3 this host cannot
compile with triton, so ``compile`` wrote nothing. These are the codes of
the Day 73 to 81 labs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .kernel_launch_io import canonical_json_bytes, write_atomic
from .kernel_metal_cli import _refuse_unusable_output
from .triton_tc_compile import CompilerBackend, TritonTcUnavailableError
from .triton_tc_config import load_config
from .triton_tc_report import DIGEST_FIELD, build_report, seal_report
from .triton_tc_verify import TritonTcVerificationError, load_report, verify_report

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "python -m ai_journey.triton_tc_cli"
NAME = "triton-tc"
PATH_ORDER = ("fp32", "tf32", "fp16", "bf16")
KINDS = {
    "declared_model": "model",
    "triton_compiler_output": "compiled",
    "synthetic_compile": "synthetic",
}


def build_parser() -> argparse.ArgumentParser:
    """Define the three commands, each with explicit paths and no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Compile the Day 82 Tensor Core matmul offline for sm_80 and sm_90, "
            "check its instruction classes, model the fp32, tf32 and bf16 times "
            "from declared datasheet inputs, and verify the sealed report. "
            "Nothing runs on a GPU."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser(
        "model",
        help="print the speedup model of the declared inputs; not a measurement",
        allow_abbrev=False,
    )
    model.add_argument("--config", type=Path, required=True)
    model.add_argument("--output", type=Path)
    compile_ = commands.add_parser(
        "compile",
        help="compile the four variants with triton and check the class rule",
        allow_abbrev=False,
    )
    compile_.add_argument("--config", type=Path, required=True)
    compile_.add_argument("--output", type=Path, required=True)
    compile_.add_argument("--dump", type=Path)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed report on the CPU, without triton",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--dump", type=Path)
    return parser


def _decimal(value: Mapping[str, object], places: int) -> str:
    return f"{value['numerator'] / value['denominator']:.{places}f}"


def _ratio(ratio: Mapping[str, Mapping[str, object]]) -> str:
    return ":".join(
        "1" if path == "bf16" else _decimal(ratio[path], 2)
        for path in ("fp32", "tf32", "bf16")
    )


def model_lines(report: Mapping[str, object]) -> list[str]:
    """Return the device lines and one line for each device and size."""

    model = report["model"]
    lines = [
        f"triton_tc model label={model['label']} "
        f"config_sha256={report['config_sha256']} "
        f"devices={len(model['devices'])} "
        f"sizes={','.join(str(size) for size in model['sizes'])}"
    ]
    for device in model["devices"]:
        rates = device["fma_per_sm_per_clock"]
        lines.append(
            f"triton_tc device key={device['key']} target={device['target']} "
            f"sms={device['sm_count']} "
            f"bandwidth_gbps={_decimal(device['bandwidth_gbps'], 0)} "
            "fma_per_sm_per_clock="
            + ",".join(f"{path}:{rates[path]}" for path in PATH_ORDER)
            + f" per_clock_ratio={_ratio(device['per_clock_ratio'])} "
            "dense_tflops="
            + ",".join(
                f"{path}:{_decimal(device['dense_tflops'][path], 1)}"
                for path in PATH_ORDER
            )
            + f" peak_ratio={_ratio(device['peak_ratio'])} "
            "implied_clock_mhz="
            + ",".join(
                f"{path}:{_decimal(device['implied_clock_mhz'][path], 1)}"
                for path in PATH_ORDER
            )
        )
        for row in device["rows"]:
            points = row["points"]
            lines.append(
                f"triton_tc model_row device={device['key']} n={row['size']} "
                + " ".join(
                    f"{path}_us={_decimal(points[path]['time_us'], 3)}"
                    for path in ("fp32", "tf32", "bf16")
                )
                + " bound="
                + ",".join(points[path]["bound"] for path in ("fp32", "tf32", "bf16"))
                + f" ratio={_ratio(row['ratio'])} "
                + f"memory_bound={','.join(row['memory_bound']) or 'none'}"
            )
    return lines


def compile_lines(report: Mapping[str, object]) -> list[str]:
    """Return one line for each compile and one for the rule summary."""

    section = report["compile"]
    lines = []
    for row in section["rows"]:
        classes = row["ptx"]["classes"]
        forms = ",".join(
            f"{form['shape']}.{form['a_type']}:{form['count']}"
            for form in row["ptx"]["mma_forms"]
        )
        swizzles = ",".join(
            "/".join(
                f"{key}={value}"
                for key, value in item["parameters"].items()
                if key in ("vec", "perPhase", "maxPhase", "swizzlingByteWidth")
            )
            for item in row["ttgir"]["shared"]
        )
        lines.append(
            f"triton_tc compile variant={row['variant']} target={row['target']} "
            f"observed={row['observed']} expected={row['expected']} "
            f"accepted={str(row['accepted']).lower()} "
            f"ptx_target={row['ptx']['target']} mma_sync={classes['mma_sync']} "
            f"wgmma={classes['wgmma_mma_async']} ldmatrix={classes['ldmatrix']} "
            f"cp_async={classes['cp_async']} fma_rn_f32={classes['fma_rn_f32']} "
            f"mma_forms={forms or 'none'} "
            f"mma_versions={','.join(str(v) for v in row['ttgir']['mma_versions']) or 'none'} "
            f"swizzles={swizzles or 'none'} shared_bytes={row['shared_bytes']}"
            + ("" if row["note"] is None else " note=mma_sync_on_sm_90")
        )
    summary = section["summary"]
    lines.append(
        f"triton_tc compile_summary backend={section['backend']} "
        f"triton={section['environment']['triton']} "
        f"accepted={summary['accepted']}/{summary['compiles']} "
        f"mma_sync_on_sm_90={','.join(summary['mma_sync_on_sm_90']) or 'none'}"
    )
    return lines


def summary_line(prefix: str, report: Mapping[str, object]) -> str:
    """Return the one line that names a sealed report and its host load."""

    load = report["host_load"]
    kind = KINDS[report["label"]]
    return (
        f"{prefix}_{kind}={report[DIGEST_FIELD]} label={report['label']} "
        f"load_before={','.join(f'{v:.2f}' for v in load['before'])} "
        f"load_after={','.join(f'{v:.2f}' for v in load['after'])} "
        "gpu=false ptx_executed=false measured=false"
    )


def _refuse_dump(dump: Path, output: Path, config: Path) -> None:
    """Refuse a dump folder that is not a folder or that would hold an input.

    The dump writes ``index.json`` and one folder per variant, so neither the
    config nor the report may lie inside it.
    """

    if dump.is_symlink() or (dump.exists() and not dump.is_dir()):
        raise ValueError("dump must be a folder that is not a symlink")
    folder = dump.resolve()
    for path, name in ((config, "config"), (output, "report output")):
        if path.resolve().is_relative_to(folder):
            raise ValueError(f"{name} must not be inside the dump folder")


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: object | None = None,
    clock: Callable[[], float] | None = None,
    load_average: Callable[[], tuple[float, float, float]] | None = None,
) -> int:
    """Run one command and return its exit code; the keywords are for tests.

    A replaced backend makes a compile report synthetic. The triton compiler
    backend refuses a replaced clock or load average.
    """

    arguments = build_parser().parse_args(argv)
    clock = clock or time.time
    load_average = load_average or os.getloadavg
    status = EXIT_OK
    try:
        if arguments.command == "model":
            if arguments.output is not None:
                _refuse_unusable_output(arguments.output, arguments.config)
            report = seal_report(
                build_report(arguments.config, clock=clock, load_average=load_average)
            )
            verify_report(report, arguments.config)
            if arguments.output is not None:
                write_atomic(
                    arguments.output,
                    canonical_json_bytes(report) + b"\n",
                    protected=(arguments.config,),
                )
            lines = [*model_lines(report), summary_line("triton_tc", report)]
        elif arguments.command == "verify":
            report = load_report(arguments.report, arguments.config, arguments.dump)
            lines = [
                summary_line("verified_triton_tc", report)
                + f" texts_recounted={str(arguments.dump is not None).lower()}"
            ]
        else:
            _refuse_unusable_output(arguments.output, arguments.config)
            if arguments.dump is not None:
                _refuse_dump(arguments.dump, arguments.output, arguments.config)
            load_config(arguments.config)
            chosen = CompilerBackend() if backend is None else backend
            report = seal_report(
                build_report(
                    arguments.config,
                    clock=clock,
                    load_average=load_average,
                    backend=chosen,
                    dump=arguments.dump,
                )
            )
            verify_report(report, arguments.config, arguments.dump)
            write_atomic(
                arguments.output,
                canonical_json_bytes(report) + b"\n",
                protected=(arguments.config,),
            )
            lines = [*compile_lines(report), summary_line("triton_tc", report)]
            summary = report["compile"]["summary"]
            if not summary["all_accepted"]:
                print(
                    f"{NAME} compile rejected: {summary['accepted']} of "
                    f"{summary['compiles']} compiles pass the instruction-class rule",
                    file=sys.stderr,
                )
                status = EXIT_REJECTED
    except TritonTcUnavailableError as error:
        print(f"{NAME} {arguments.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except TritonTcVerificationError as error:
        print(f"{NAME} {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError, KeyError) as error:
        print(f"{NAME} {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
