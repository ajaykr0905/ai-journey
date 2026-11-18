import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_bottleneck_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    PROG,
    main,
)
from ai_journey.memory_access_plan import canonical_json_bytes
from test_matmul_bottleneck_io import write_inputs
from test_matmul_bottleneck_report import synthetic_report
from test_matmul_bottleneck_sources import recorded_reports
from test_matmul_bottleneck_verify import resealed
from test_matmul_roofline_gap import resealed as resealed_source

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-75-matmul-bottleneck.json"
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_bottleneck_cli import main
code = main(sys.argv[1:])
print("torch imported" if sys.modules.get("torch") else "torch not imported")
sys.exit(code)
"""


def input_arguments(inputs):
    return [
        "--matmul-report",
        str(inputs.matmul_report),
        "--ceiling-report",
        str(inputs.ceiling_report),
        "--gap-report",
        str(inputs.gap_report),
        "--timing",
        str(inputs.timing),
    ]


def classify(inputs, output):
    return [
        "classify",
        *input_arguments(inputs),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
    ]


def test_classify_prints_each_name_and_seals_the_report(tmp_path, capsys):
    inputs = write_inputs(tmp_path)
    output = tmp_path / "bottleneck.json"

    assert main(classify(inputs, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report = synthetic_report()
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert lines[0].startswith("synthetic_timing ceilings fma_gflops=1802.7 ")
    assert "timing_run_load_before=1.00,1.50,2.00" in lines[0]
    assert lines[1].startswith(
        "synthetic_timing bottleneck m=256 n=256 k=256 mapping=x_columns "
        'name="latency or occupancy-bound" reason=below_both_roofs '
    )
    assert lines[3].startswith(
        "synthetic_timing mapping m=256 n=256 k=256 verdict=x_rows_worse_as_modeled "
    )
    assert lines[4].startswith(
        "synthetic_timing bottleneck m=129 n=65 k=1000 mapping=x_columns "
        'name="dispatch-bound" reason=one_call_overhead_dominates '
    )
    assert lines[-3] == (
        'synthetic_timing top kernel=matmul_x_columns name="indeterminate" '
        "points=1/2 reason=tie"
    )
    assert lines[-1] == (
        f"matmul_bottleneck_synthetic_timing={report['report_sha256']} points=2 "
        "sources_checked=true scope=synthetic_backend_or_clock measured=false "
        "counters=false nsight=false instruments=false cuda=false nvidia=false"
    )


def test_verify_recomputes_alone_or_with_all_four_inputs(tmp_path, capsys):
    inputs = write_inputs(tmp_path)
    output = tmp_path / "bottleneck.json"
    assert main(classify(inputs, output)) == EXIT_OK
    capsys.readouterr()

    assert main(["verify", "--report", str(output)]) == EXIT_OK
    alone = capsys.readouterr().out
    assert alone.startswith("verified_matmul_bottleneck_synthetic_timing=")
    assert "sources_checked=false" in alone
    arguments = ["verify", "--report", str(output), "--config", str(CONFIG)]
    assert main([*arguments, *input_arguments(inputs)]) == EXIT_OK
    assert "sources_checked=true" in capsys.readouterr().out


def test_verify_takes_all_four_inputs_or_none(tmp_path, capsys):
    inputs = write_inputs(tmp_path)
    arguments = ["verify", "--report", "r.json", *input_arguments(inputs)[:2]]

    with pytest.raises(SystemExit) as stopped:
        main(arguments)
    assert stopped.value.code == EXIT_INVALID
    assert "together, or none of them" in capsys.readouterr().err


def test_a_tampered_report_or_source_is_rejected_with_code_1(tmp_path, capsys):
    inputs = write_inputs(tmp_path)
    tampered = tmp_path / "tampered.json"
    report = resealed(
        lambda r: r["points"][0]["kernels"]["matmul_x_rows"]["decision"].update(
            bottleneck="compute-bound"
        )
    )
    tampered.write_bytes(canonical_json_bytes(report) + b"\n")

    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert "matmul-bottleneck verify rejected: point 0" in capsys.readouterr().err

    matmul, _, _ = recorded_reports()
    edited = resealed_source(matmul, lambda r: r["summary"].update(points=3))
    inputs.matmul_report.write_bytes(canonical_json_bytes(edited) + b"\n")
    assert main(classify(inputs, tmp_path / "out.json")) == EXIT_REJECTED
    assert "rejected: matmul-metal report" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_invalid_inputs_and_aliased_outputs_exit_with_code_2(tmp_path, capsys):
    inputs = write_inputs(tmp_path)

    assert main(classify(inputs, inputs.timing)) == EXIT_INVALID
    assert "must not alias an input" in capsys.readouterr().err
    assert main(classify(inputs, CONFIG)) == EXIT_INVALID
    capsys.readouterr()
    inputs.timing.write_text('{"kind": "matmul_gpu_timing"}', encoding="utf-8")
    assert main(classify(inputs, tmp_path / "out.json")) == EXIT_INVALID
    assert "timing fields do not match" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == (
        EXIT_INVALID
    )


def test_the_parser_names_the_program_and_refuses_abbreviations(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {PROG} ")
    assert "{classify,verify}" in usage
    with pytest.raises(SystemExit):
        main(["verify", "--rep", "r.json"])


def test_classify_and_verify_run_without_torch(tmp_path):
    inputs = write_inputs(tmp_path)
    output = tmp_path / "bottleneck.json"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    for arguments in (
        classify(inputs, output),
        ["verify", "--report", str(output), *input_arguments(inputs)],
    ):
        completed = subprocess.run(
            [sys.executable, "-c", WITHOUT_TORCH, *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=environment,
        )
        assert completed.returncode == EXIT_OK, completed.stderr
        assert completed.stdout.rstrip().endswith("torch not imported")
