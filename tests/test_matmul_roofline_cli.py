import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_roofline_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    build_parser,
    main,
)
from ai_journey.matmul_roofline_io import load_report
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD
from test_matmul_roofline_gap import (
    CEILINGS,
    MATMUL,
    resealed,
    synthetic_ceilings,
    synthetic_matmul,
    with_environment,
    write_sources,
)

ROOT = Path(__file__).resolve().parents[1]
# A None entry in sys.modules makes every import of torch fail, as on a host
# without PyTorch; the gap command must not need it.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_roofline_cli import main
code = main(sys.argv[1:])
print("torch imported" if sys.modules.get("torch") is not None else "torch not imported")
sys.exit(code)
"""


def gap(directory, matmul=MATMUL, ceilings=CEILINGS):
    """Write both sources into ``directory``; return the gap arguments and paths."""

    sources = write_sources(directory, matmul, ceilings)
    output = directory / "gap.json"
    arguments = [
        "gap",
        "--matmul-report",
        str(sources[0]),
        "--ceiling-report",
        str(sources[1]),
        "--output",
        str(output),
    ]
    return arguments, sources, output


def test_the_parser_requires_explicit_commands_and_paths():
    parser = build_parser()

    assert parser.prog == "ai-journey-matmul-roofline"
    parsed = parser.parse_args(
        [
            "gap",
            "--matmul-report",
            "m.json",
            "--ceiling-report",
            "c.json",
            "--output",
            "g.json",
        ]
    )
    assert (parsed.command, str(parsed.matmul_report), str(parsed.output)) == (
        "gap",
        "m.json",
        "g.json",
    )
    parsed = parser.parse_args(["verify", "--report", "g.json"])
    assert (parsed.matmul_report, parsed.ceiling_report) == (None, None)
    for argv in (
        [],
        ["gap", "--matmul-report", "m.json", "--output", "g.json"],
        [
            "gap",
            "--matmul",
            "m.json",
            "--ceiling-report",
            "c.json",
            "--output",
            "g.json",
        ],
        ["verify"],
    ):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(argv)
        assert stopped.value.code == 2


def test_a_measured_gap_prints_every_ratio_and_verifies(tmp_path, capsys):
    arguments, (matmul, ceilings), output = gap(tmp_path)

    assert main(arguments) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)
    digest = report[DIGEST_FIELD]
    assert lines[-1] == (
        f"matmul_roofline_measurement={digest} points=2 sources_checked=true "
        "scope=apple_gpu_metal_this_host cuda=false nvidia=false"
    )
    assert lines[0].startswith("apple_gpu_metal_measurement ceilings fma_gflops=")
    assert "ceiling_run_load_before=9.48,9.40,8.75" in lines[0]
    assert len(lines) == 2 + 3 * len(report["points"])
    for position, point in enumerate(report["points"]):
        roof, columns, rows = lines[1 + 3 * position : 4 + 3 * position]
        shape = f"m={point['m']} n={point['n']} k={point['k']}"
        assert roof.startswith(f"apple_gpu_metal_measurement roof {shape} ")
        ideal = point["intensities"]["ideal_reuse"]
        assert f"ideal_reuse_bound={ideal['bound']} " in roof
        for line, side in ((columns, "matmul_x_columns"), (rows, "matmul_x_rows")):
            entry = point["sides"][side]
            assert line == (
                f"apple_gpu_metal_measurement gap {shape} mapping={entry['mapping']} "
                f"gflops={entry['median_gflops_per_second']:.1f} "
                f"fma_share={entry['fma_ceiling_share']:.3g} "
                f"no_reuse_roof_share={entry['no_reuse_roof_share']:.3g} "
                f"ideal_reuse_roof_share={entry['ideal_reuse_roof_share']:.3g} "
                f"torch_ratio={entry['torch_ratio']:.3f}"
            )
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert main(["verify", "--report", str(output)]) == EXIT_OK
    assert capsys.readouterr().out == (
        f"verified_{lines[-1]}\n".replace(
            "sources_checked=true", "sources_checked=false"
        )
    )
    bound = [
        "verify",
        "--report",
        str(output),
        "--matmul-report",
        str(matmul),
        "--ceiling-report",
        str(ceilings),
    ]
    assert main(bound) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_a_synthetic_source_prints_synthetic_lines(tmp_path, capsys):
    arguments, _, output = gap(tmp_path, synthetic_matmul(), synthetic_ceilings())

    assert main(arguments) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)
    assert lines[-1] == (
        f"matmul_roofline_synthetic_timing={report[DIGEST_FIELD]} points=2 "
        "sources_checked=true scope=synthetic_backend_or_clock measured=false "
        "cuda=false nvidia=false"
    )
    assert all(line.startswith("synthetic_timing ") for line in lines[:-1])


def test_rejected_reports_exit_1_and_bad_inputs_exit_2(tmp_path, capsys):
    arguments, (matmul, ceilings), output = gap(tmp_path)
    assert main(arguments) == EXIT_OK
    capsys.readouterr()

    tampered = tmp_path / "tampered-matmul.json"
    tampered.write_bytes(
        canonical_json_bytes(
            resealed(MATMUL, lambda report: report["summary"].update(timed_samples=1))
        )
        + b"\n"
    )
    other = tmp_path / "other.json"
    swapped = [*arguments[:2], str(tampered), *arguments[3:-1], str(other)]
    assert main(swapped) == EXIT_REJECTED
    assert capsys.readouterr().err.startswith(
        "matmul-roofline gap rejected: matmul-metal report:"
    )
    assert not other.exists()
    gap_report = json.loads(output.read_text(encoding="ascii"))
    changed = tmp_path / "changed.json"
    changed.write_bytes(
        canonical_json_bytes(resealed(gap_report, lambda r: r.update(measured=False)))
        + b"\n"
    )
    assert main(["verify", "--report", str(changed)]) == EXIT_REJECTED
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(gap_report, indent=1) + "\n", encoding="ascii")
    assert main(["verify", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err

    missing = [
        *arguments[:4],
        str(tmp_path / "missing.json"),
        *arguments[5:-1],
        str(other),
    ]
    assert main(missing) == EXIT_INVALID
    assert capsys.readouterr().err.startswith(
        "matmul-roofline gap failed: gpu-ceilings report:"
    )
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == EXIT_INVALID
    capsys.readouterr()


def test_outputs_that_alias_a_source_or_cannot_be_written_are_refused(tmp_path, capsys):
    arguments, (matmul, ceilings), _ = gap(tmp_path)
    before = matmul.read_bytes(), ceilings.read_bytes()
    link = tmp_path / "link.json"
    link.symlink_to(ceilings)

    for target in (matmul, ceilings, link, tmp_path):
        assert main([*arguments[:-1], str(target)]) == EXIT_INVALID
        assert capsys.readouterr().err.startswith("matmul-roofline gap failed:")
    assert (matmul.read_bytes(), ceilings.read_bytes()) == before


def test_sources_that_do_not_pair_exit_2(tmp_path, capsys):
    ceilings = with_environment(synthetic_ceilings(), {"numpy_version": "0.0.1"})
    arguments, _, output = gap(tmp_path, MATMUL, ceilings)

    assert main(arguments) == EXIT_INVALID
    assert "different software or platforms: numpy_version" in capsys.readouterr().err
    assert not output.exists()


def test_verify_binds_to_both_sources_or_to_neither(tmp_path, capsys):
    arguments, (matmul, ceilings), output = gap(tmp_path)
    assert main(arguments) == EXIT_OK
    capsys.readouterr()

    with pytest.raises(SystemExit) as stopped:
        main(["verify", "--report", str(output), "--matmul-report", str(matmul)])
    assert stopped.value.code == 2
    assert "or neither" in capsys.readouterr().err
    other = tmp_path / "other"
    other.mkdir()
    _, (other_matmul, other_ceilings), _ = gap(
        other, synthetic_matmul(), synthetic_ceilings()
    )
    bound = [
        "verify",
        "--report",
        str(output),
        "--matmul-report",
        str(other_matmul),
        "--ceiling-report",
        str(other_ceilings),
    ]
    assert main(bound) == EXIT_REJECTED
    assert "supplied source reports" in capsys.readouterr().err


def test_gap_and_verify_run_where_torch_cannot_be_imported(tmp_path):
    arguments, (matmul, ceilings), output = gap(tmp_path)

    def run(argv):
        return subprocess.run(
            [sys.executable, "-c", WITHOUT_TORCH, *argv],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

    built = run(arguments)
    assert built.returncode == EXIT_OK, built.stderr
    assert built.stdout.endswith("torch not imported\n")
    checked = run(
        [
            "verify",
            "--report",
            str(output),
            "--matmul-report",
            str(matmul),
            "--ceiling-report",
            str(ceilings),
        ]
    )
    assert checked.returncode == EXIT_OK, checked.stderr
    summary = built.stdout.splitlines()[-2]
    assert checked.stdout == f"verified_{summary}\ntorch not imported\n"
