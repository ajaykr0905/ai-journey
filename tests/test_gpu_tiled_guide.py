import re
import shlex
from pathlib import Path

from ai_journey import gpu_tiled_cli
from ai_journey.gpu_tiled_config import PINNED_PLAN
from ai_journey.gpu_tiled_cuda import (
    CUBLAS_CHECK_NAMES,
    TILED_CHECK_NAMES,
    TILES,
    cublas_source,
    shared_memory_bytes,
    source_sha256,
    tiled_cuda_source,
)
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from test_gpu_tiled_report import REPORT_SHA256

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-tiled-cuda-contract.md"
COMMAND = ["python", "-m", "ai_journey.gpu_tiled_cli"]


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[1 : 1 + len(COMMAND)] == COMMAND:
                commands.append(words[1 + len(COMMAND) :])
    return commands


def test_the_documented_commands_parse_and_chain():
    parser = gpu_tiled_cli.build_parser()
    commands = documented_commands(guide())
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["run", "verify", "sources"]
    assert parsed[0].config is None and parsed[1].config is None
    assert parsed[1].report == parsed[0].output
    assert "PINNED_PLAN" in guide()


def test_the_documented_line_is_the_pinned_summary_line():
    report = {
        "report_sha256": REPORT_SHA256,
        "plan": {"name": PINNED_PLAN.name},
        "summary": {
            "cases": 9,
            "tiles": 3,
            "block_runs": 27,
            "block_runs_matching_replay": 27,
            "block_runs_matching_traffic": 27,
            "race_free_block_runs": 27,
            "cuda_checks_passed": 297,
            "cuda_checks": 297,
            "cublas_checks_passed": 63,
            "cublas_checks": 63,
            "occupancy_estimates": 9,
        },
    }
    line = gpu_tiled_cli.summary_line("gpu_tiled_report", report)

    assert f"```text\n{line}\n```" in guide()
    assert "`verified_gpu_tiled_report=`" in guide()


def test_the_guide_states_each_digest_check_and_bound():
    text = guide()

    for tile in TILES:
        row = f"| {tile} | {tile * tile:,} | {shared_memory_bytes(tile):,} bytes |"
        assert row in text, row
        assert f"`{source_sha256(tiled_cuda_source(tile=tile))}`" in text
    assert f"`{source_sha256(cublas_source())}`" in text
    for name in TILED_CHECK_NAMES:
        assert f"`{name}`" in text, name
    assert f"has {len(CUBLAS_CHECK_NAMES)} checks" in text
    assert f"The {len(TILED_CHECK_NAMES)} checks are" in text
    for index, shape in enumerate(PINNED_PLAN.shapes):
        left, right = matmul_inputs(PINNED_PLAN.seed, CORRECTNESS_STREAM, index, shape)
        row = (
            f"| {index} | {shape.m} x {shape.n} x {shape.k} | "
            f"`{matrix_sha256(fma_replay(left, right))}` |"
        )
        assert row in text, row
        assert f" | {error_bound(shape.k):.3e} |\n" in text


def test_the_guide_states_the_evidence_boundary_and_the_exit_codes():
    text = flat(guide())

    for phrase in (
        "This host has no NVIDIA GPU, no CUDA toolkit, no `nvcc` and no cuBLAS.",
        "the lab compiles, links and runs none of the texts.",
        "Assumption: with `CUBLAS_DEFAULT_MATH`, `cublasSgemm` uses no TF32",
        "This lab did not check it. A later run must check it on the NVIDIA host.",
        "This lab produces no GFLOP/s number.",
        "The lab measured no GPU time.",
        "These times describe one shared host and no GPU.",
        "`0`: the command succeeded.",
        "`1`: the command read or built a report, but the report failed",
        "`2`: the arguments are wrong,",
        "No command needs a GPU, so no exit code marks a missing device.",
    ):
        assert phrase in text, phrase
    assert (
        gpu_tiled_cli.EXIT_OK,
        gpu_tiled_cli.EXIT_REJECTED,
        gpu_tiled_cli.EXIT_INVALID,
    ) == (0, 1, 2)
