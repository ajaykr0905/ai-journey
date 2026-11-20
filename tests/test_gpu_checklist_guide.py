"""Tests that the Day 77 CUDA guide states what the code does. No CUDA runs."""

import re
import shlex
from pathlib import Path

import numpy

from ai_journey import gpu_checklist_cli
from ai_journey.gpu_checklist_blocksim import ChecklistTraffic, run_blocks, vector_flags
from ai_journey.gpu_checklist_check import (
    BLOCKED_CHECK_NAMES,
    COARSENED_CHECK_NAMES,
    check_cuda_source,
)
from ai_journey.gpu_checklist_cuda import (
    CUBLAS_FILE,
    PINNED_BLOCKED,
    PINNED_COARSENED,
    PINNED_KERNELS,
    cuda_source,
    source_files,
    source_sha256,
)
from ai_journey.gpu_checklist_occupancy import (
    declared_registers,
    estimate_mapping,
    live_floats,
)
from ai_journey.gpu_tiled_blocksim import PHASE_STEPS
from ai_journey.memory_access_tiling import MatmulShape, TiledMatmulTraffic

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-checklist-cuda-contract.md"
COMMAND = ["python", "-m", "ai_journey.gpu_checklist_cli"]
SCHEDULES = (
    PHASE_STEPS,
    ("load", "fma", "barrier"),
    ("load", "barrier", "fma"),
    ("load", "fma"),
    ("barrier", "load", "barrier", "fma"),
)


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


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def count(value):
    return f"{value:,}"


def test_the_documented_commands_parse_and_chain():
    parser = gpu_checklist_cli.build_parser()
    commands = documented_commands(guide())
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["sources", "check"]
    assert parsed[1].source_dir == parsed[0].output_dir


def test_the_documented_line_is_the_check_line_of_the_pinned_file(tmp_path, capsys):
    assert gpu_checklist_cli.main(["sources", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert gpu_checklist_cli.main(["check", "--source-dir", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert f"```text\n{lines[1]}\n```" in guide()
    assert lines[1].startswith("cuda_check=coarsened_matmul_t16_c4.cu ")


def test_the_guide_states_each_digest_tile_and_check():
    text = guide()

    for kernel in PINNED_KERNELS:
        rows, columns = kernel.to_mapping()["output_tile"]
        x, y = kernel.block
        row = (
            f"| `{kernel.name}` | {x} x {y} | {rows} x {columns} | "
            f"{count(kernel.shared_memory_bytes)} bytes | "
            f"`{source_sha256(cuda_source(kernel))}` |"
        )
        assert row in text, row
    assert f"`{source_sha256(dict(source_files())[CUBLAS_FILE])}`" in text
    for name in (*COARSENED_CHECK_NAMES, *BLOCKED_CHECK_NAMES):
        assert f"`{name}`" in text, name
    assert (
        f"The coarsened kernel has {len(COARSENED_CHECK_NAMES)} checks, and the "
        f"blocked kernel has {len(BLOCKED_CHECK_NAMES)} checks"
    ) in flat(text)


def test_the_documented_extra_guard_fails_only_the_loads_check():
    guard = "if (tx == 0) { k = 0; }"
    kernel = PINNED_COARSENED[1]
    marker = "        for (int j = 0; j < 4; ++j) {\n            int col"
    source = cuda_source(kernel)
    edited = source.replace(marker, f"        {guard}\n{marker}")
    text = flat(guide())

    assert f"a text can add `{guard}` before the loop over the B tiles." in text
    assert "and the text fails `coarsened_loads` only." in text
    assert source.count(marker) == 1
    result = check_cuda_source(edited, kernel=kernel, shape=MatmulShape(37, 45, 29))
    assert result.failures == ("coarsened_loads",)


def test_the_load_table_is_the_declared_closed_form():
    shape = MatmulShape(1000, 999, 100)
    text = guide()

    for tile in (16, 32):
        day_76 = TiledMatmulTraffic(shape, tile=tile)
        row = (
            f"| Day 76 tile {tile} | {day_76.block_columns} x {day_76.block_rows} | "
            f"{count(day_76.useful_a_loads)} | {count(day_76.useful_b_loads)} | 2 |"
        )
        assert row in text, row
    for kernel in PINNED_KERNELS:
        traffic = ChecklistTraffic(
            kernel, shape, *vector_flags(kernel, shape, (True, True))
        )
        grid = traffic.launch.grid
        reads = traffic.shared_reads / traffic.fma_steps
        row = (
            f"| `{kernel.name}` | {grid.x} x {grid.y} | {count(traffic.a_loads)} | "
            f"{count(traffic.b_loads)} | {reads:g} |"
        )
        assert row in text, row


def test_the_hazard_table_is_the_simulator_count():
    shape = MatmulShape(37, 45, 29)
    generator = numpy.random.default_rng(0)
    left = generator.standard_normal((shape.m, shape.k), dtype=numpy.float32)
    right = generator.standard_normal((shape.k, shape.n), dtype=numpy.float32)
    kernels = (PINNED_COARSENED[1], PINNED_BLOCKED[0])

    for steps in SCHEDULES:
        cells = []
        for kernel in kernels:
            hazards = run_blocks(left, right, kernel=kernel, steps=steps).hazards
            cells.append(
                f"{count(hazards.read_after_write)} and "
                f"{count(hazards.write_after_read)}"
            )
        row = f"| {', '.join(steps)} | {cells[0]} | {cells[1]} |"
        assert row in guide(), row


def test_the_occupancy_table_is_the_declared_estimate():
    text = guide()

    for kernel in PINNED_KERNELS:
        cells = []
        for registers in declared_registers(kernel):
            mapping = estimate_mapping(kernel, registers_per_thread=registers)
            blocks = mapping["active_blocks_per_sm"]
            ratio = (
                mapping["occupancy"]["numerator"] / mapping["occupancy"]["denominator"]
            )
            noun = "block" if blocks == 1 else "blocks"
            cells.append(f"{registers}: {ratio:.3f}, {blocks} {noun}")
        row = f"| `{kernel.name}` | {' | '.join(cells)} |"
        assert row in text, row
    floors = ", ".join(str(live_floats(kernel)) for kernel in PINNED_KERNELS)
    assert floors.replace(", 80", " and 80") in text


def test_the_guide_states_the_evidence_boundary_and_the_exit_codes():
    text = flat(guide())

    for phrase in (
        "This host has no NVIDIA GPU, no CUDA toolkit, no `nvcc`, no cuBLAS and no "
        "Nsight.",
        "Thus the lab compiles, links and runs none of the texts.",
        "The lab measures no GPU time, and it writes no sealed report.",
        "This lab produces no GFLOP/s number and no speedup.",
        "The lab measured no GPU time.",
        "The register counts are assumptions,",
        "These times describe one shared host and no GPU.",
        "They are not measured traffic, and the model has no cache.",
        "`0`: the command succeeded.",
        "`1`: `check` read each file, but a file failed a check.",
        "`2`: the arguments are wrong,",
        "No command needs a GPU, so no exit code marks a missing device.",
    ):
        assert phrase in text, phrase
    assert (
        gpu_checklist_cli.EXIT_OK,
        gpu_checklist_cli.EXIT_REJECTED,
        gpu_checklist_cli.EXIT_INVALID,
    ) == (0, 1, 2)


def test_the_nvidia_preflight_job_covers_the_checklist_files():
    job = ci_job("nvidia-preflight-contracts")
    sources = sorted(ROOT.glob("src/ai_journey/gpu_checklist_*.py"))
    tests = sorted(ROOT.glob("tests/test_gpu_checklist_*.py"))

    # A later lab can add checklist files; these must stay among them.
    parts = ("cuda", "cli", "check", "blocksim", "occupancy")
    assert {ROOT / f"src/ai_journey/gpu_checklist_{part}.py" for part in parts} <= (
        set(sources)
    )
    assert {ROOT / f"tests/test_gpu_checklist_{part}.py" for part in parts} | {
        ROOT / "tests/test_gpu_checklist_guide.py"
    } <= set(tests)
    assert set(sources) <= set(ROOT.glob("src/ai_journey/gpu*.py"))
    assert set(tests) <= set(ROOT.glob("tests/test_gpu*.py"))
    for command in (
        "ruff check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "ruff format --check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m compileall -q src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m pytest -q tests/test_gpu*.py",
    ):
        assert command in job, command
