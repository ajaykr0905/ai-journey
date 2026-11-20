"""Tests that the Day 78 CUDA guide states what the code does. No CUDA runs."""

import re
import shlex
from pathlib import Path

from ai_journey import gpu_reduce_cuda_cli
from ai_journey.gpu_reduce_bound import case_values, summation_depth
from ai_journey.gpu_reduce_config import load_config
from ai_journey.gpu_reduce_cuda import (
    COMBINE_BARRIERS,
    COMBINE_SHARED_BYTES,
    PINNED_KERNELS,
    R1_SHARED,
    WARP_SHUFFLE,
    combine_source,
    cuda_source,
    source_sha256,
)
from ai_journey.gpu_reduce_cuda_check import (
    CHECK_SIZES,
    CORRECTNESS_SIZES,
    GridStrideLoops,
    check_names,
    check_reduce_source,
    combine_coverage,
)
from ai_journey.gpu_reduce_cuda_sim import simulate
from ai_journey.gpu_reduce_source import R1, R3
from ai_journey.matmul_tiled_comparison import VERDICTS
from test_gpu_reduce_cuda_sim import (
    COMBINE_COUNTS,
    FLOAT_PATTERNS,
    FLOAT_SIZES,
    SMALL_SIZES,
    recorded_inputs,
)
from test_matmul_roofline_ci_contract import ci_job

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-reduce-cuda-contract.md"
METAL_GUIDE = ROOT / "docs/gpu-reduction-lab.md"
PINNED_CONFIG = ROOT / "config/day-78-gpu-reduce.json"
COMMAND = ["python", "-m", "ai_journey.gpu_reduce_cuda_cli"]
R1_TEXT = cuda_source(R1_SHARED)
SHUFFLE_TEXT = cuda_source(WARP_SHUFFLE)


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def count(value):
    return f"{value:,}"


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[1 : 1 + len(COMMAND)] == COMMAND:
                commands.append(words[1 + len(COMMAND) :])
    return commands


def test_the_documented_commands_parse_and_chain():
    parser = gpu_reduce_cuda_cli.build_parser()
    commands = documented_commands(guide())
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["sources", "check", "simulate"]
    assert parsed[1].source_dir == parsed[0].output_dir
    assert parsed[2].config == Path("config/day-78-gpu-reduce.json")
    assert gpu_reduce_cuda_cli.PROG == "ai-journey-gpu-reduce-cuda"


def test_the_documented_line_is_the_check_line_of_the_pinned_file(tmp_path, capsys):
    assert gpu_reduce_cuda_cli.main(["sources", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert gpu_reduce_cuda_cli.main(["check", "--source-dir", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert f"```text\n{lines[1]}\n```" in guide()
    assert lines[1].startswith("cuda_check=reduce_warp_shuffle.cu ")
    assert len(CHECK_SIZES) == 14 and len(CORRECTNESS_SIZES) == 9


def test_the_guide_states_each_digest_count_and_check():
    text = guide()
    rows = {
        R1_SHARED: ("reduce_r1_shared_kernel", 256, 9, 262144),
        WARP_SHUFFLE: ("reduce_warp_shuffle_kernel", 512, 4, 131072),
    }

    for kernel, (name, elements, barriers, partials) in rows.items():
        assert kernel.kernel_name == name
        assert (kernel.elements_per_block, kernel.barriers_per_block) == (
            elements,
            barriers,
        )
        assert kernel.partial_count(1 << 26) == partials
        row = (
            f"| `{kernel.file_name}` | `{name}` | {elements} | {barriers} | "
            f"{count(kernel.shared_memory_bytes)} bytes | {count(partials)} | "
            f"`{source_sha256(cuda_source(kernel))}` |"
        )
        assert row in text, row
    assert f"`{source_sha256(combine_source())}`" in text
    assert f"The combine has {count(COMBINE_SHARED_BYTES)} bytes of shared" in flat(
        text
    )
    assert f"each of its threads reaches {COMBINE_BARRIERS} barriers" in flat(text)
    for name in {*check_names(R1_SHARED), *check_names(WARP_SHUFFLE)}:
        assert f"`{name}`" in text, name
    assert (
        f"The R1 text has {len(check_names(R1_SHARED))} checks, and the shuffle "
        f"text has {len(check_names(WARP_SHUFFLE))} checks"
    ) in text
    depths = (summation_depth(R1, 1 << 26), summation_depth(R3, 1 << 26))
    assert depths == (273, 146)
    assert "D is 273 for R1 and 146 for the shuffle kernel" in flat(text)


def test_each_code_excerpt_is_text_of_a_pinned_file():
    excerpts = re.findall(r"```c\n(.*?)```", guide(), flags=re.DOTALL)
    files = {"R1": R1_TEXT, "shuffle": SHUFFLE_TEXT}

    assert len(excerpts) == 3
    assert excerpts[0] in R1_TEXT and excerpts[1] in SHUFFLE_TEXT
    # The combine excerpt is in both files, inside the one combine text.
    assert all(excerpts[2] in source for source in files.values())
    assert excerpts[2] in combine_source()


def mutated(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_each_documented_change_fails_the_named_checks():
    text = flat(guide())
    cases = (
        (
            "a text without the barrier after the load fails `r1_barriers` only.",
            R1_SHARED,
            mutated(R1_TEXT, "0.0f;\n    __syncthreads();\n", "0.0f;\n"),
            ("r1_barriers",),
        ),
        (
            "A text with the mask `0x0000ffffu` fails `shuffle_warp_stage` only.",
            WARP_SHUFFLE,
            mutated(SHUFFLE_TEXT, "0xffffffffu, w, 16u", "0x0000ffffu, w, 16u"),
            ("shuffle_warp_stage",),
        ),
        (
            "A text with the guard `t < 16u` in place of `warp == 0u` fails "
            "`shuffle_warp_stage` only.",
            WARP_SHUFFLE,
            mutated(SHUFFLE_TEXT, "if (warp == 0u)", "if (t < 16u)"),
            ("shuffle_warp_stage",),
        ),
        (
            "A text with a `__syncthreads()` after the last shuffle fails "
            "`shuffle_barriers` and `shuffle_warp_stage`.",
            WARP_SHUFFLE,
            mutated(SHUFFLE_TEXT, "w, 1u));\n", "w, 1u));\n        __syncthreads();\n"),
            ("shuffle_barriers", "shuffle_warp_stage"),
        ),
        (
            "A text with the step `e += 2048u` fails `combine_grid_stride` only.",
            WARP_SHUFFLE,
            mutated(SHUFFLE_TEXT, "e += 4096u", "e += 2048u"),
            ("combine_grid_stride",),
        ),
        (
            "A text that launches the combine as two blocks fails `launch` only.",
            WARP_SHUFFLE,
            mutated(SHUFFLE_TEXT, "<<<1, 1024>>>", "<<<2, 1024>>>"),
            ("launch",),
        ),
        (
            "A text whose tree stops at 16 values fails `shuffle_tree` and "
            "`shuffle_barriers`, because each thread then reaches one more barrier.",
            WARP_SHUFFLE,
            mutated(SHUFFLE_TEXT, "s >= 32u;", "s >= 16u;"),
            ("shuffle_tree", "shuffle_barriers"),
        ),
    )

    for phrase, kernel, source, failures in cases:
        assert phrase in text, phrase
        for n in CHECK_SIZES:
            result = check_reduce_source(source, kernel=kernel, n=n)
            assert result.failures == failures, (phrase, n)
    # More than 3072 partials: the step of 2048 reads some partials twice.
    offsets = ((0, 0), (1, 1024), (2, 2048), (3, 3072))
    stepped = GridStrideLoops(3072, 2048, offsets, 1024, ((0, 0),))
    assert "With more than 3072 partials, its loops read some partials twice." in text
    assert int(combine_coverage(stepped, 3073).max()) == 2
    assert int(combine_coverage(stepped, 3072).max()) == 1


def test_the_race_table_is_the_simulator_count():
    values = case_values(78, 4, "mixed", 1000)
    text = guide()
    strides = ("the load", *(f"stride {s}" for s in R1_SHARED.tree_strides))

    for barrier, label in enumerate(strides):
        cells = []
        for kernel in PINNED_KERNELS:
            if barrier >= kernel.barriers_per_block:
                cells.append("none")
                continue
            run = simulate(values, kernel=kernel, dropped_barriers=(barrier,))
            assert run.hazards.write_after_read == 0
            cells.append(str(run.hazards.read_after_write))
        row = f"| {barrier}, after {label} | {cells[0]} | {cells[1]} |"
        assert row in text, row
    every = [
        simulate(
            values,
            kernel=kernel,
            dropped_barriers=tuple(range(kernel.barriers_per_block)),
        ).hazards.read_after_write
        for kernel in PINNED_KERNELS
    ]
    assert f"| every barrier | {every[0]} | {every[1]} |" in text


PREDICTION = re.compile(
    r"cuda_prediction kernel=(\S+) input=(\S+) pattern=(\S+) n=(\d+) "
    r"output=(\S+) bits=(0x[0-9a-f]{8}) depth=\d+ bound_ratio=(\S+) "
)


def test_the_predicted_table_and_summary_are_the_pinned_simulation():
    lines, failures = gpu_reduce_cuda_cli.simulate_config(PINNED_CONFIG)
    text = guide()
    rows, ratios = {}, {}
    for line in lines[:-1]:
        kernel, label, pattern, n, output, bits, ratio = PREDICTION.match(line).groups()
        rows.setdefault((label, pattern, n), []).append(f"{output} (`{bits}`)")
        ratios[(kernel, label, int(n))] = float(ratio)

    assert failures == [] and len(rows) == 18
    for (label, pattern, n), cells in rows.items():
        row = f"| {label} | {pattern} | {n} | {cells[0]} | {cells[1]} |"
        assert row in text, row
    assert f"```text\n{lines[-1]}\n```" in text
    assert max(ratios, key=ratios.get) == ("reduce_r1_shared", "case-6", 511)
    assert max(ratios.values()) == 0.124
    assert "The largest bound ratio is 0.124, for `reduce_r1_shared` on" in flat(text)


def test_the_metal_comparisons_are_the_tested_counts():
    text = flat(guide())
    inputs = list(recorded_inputs())
    differ = sum(
        simulate(values, kernel=R1_SHARED).output
        != simulate(values, kernel=WARP_SHUFFLE).output
        for values, _ in inputs
    )

    assert len(inputs) == 8 and differ == 2
    for phrase in (
        f"On {len(FLOAT_PATTERNS) * len(FLOAT_SIZES)} seeded inputs of "
        f"{min(FLOAT_SIZES)} to {max(FLOAT_SIZES)} elements, it gives the bits of "
        "the simulator.",
        f"for {len(SMALL_SIZES)} sizes from {min(SMALL_SIZES)} to {max(SMALL_SIZES)}.",
        f"for {len(COMBINE_COUNTS)} partial counts from {min(COMBINE_COUNTS)} to "
        f"{max(COMBINE_COUNTS)}.",
        f"That report holds {len(inputs)} inputs: 6 correctness cases and 2 timing "
        "sizes.",
        f"At all {len(inputs)} recorded inputs, `reduce_r1_shared` gives the bits "
        "of `reduce_r1_sequential`.",
        f"The two orders give different bits at {differ} of the {len(inputs)} "
        "inputs.",
    ):
        assert phrase in text, phrase
    # The one-time check names the pinned run of the Metal guide, and the
    # Metal guide names this guide.
    metal = METAL_GUIDE.read_text(encoding="utf-8")
    pinned = "5905fe9a28f58e242d669a5489b79a0937dab4595f584567d230818ae987df47"
    assert pinned in text and pinned in metal
    assert "\n## CUDA texts for a later NVIDIA run\n" in metal
    assert "`docs/gpu-reduce-cuda-contract.md` writes R1 and R3" in flat(metal)


def test_the_guide_states_the_evidence_boundary_and_the_exit_codes():
    text = flat(guide())

    for phrase in (
        "This host has no NVIDIA GPU, no CUDA toolkit, no `nvcc`, no CUB and no "
        "Nsight.",
        "Thus the lab compiles, links and runs none of the texts.",
        "The lab measures no GPU time, and it writes no sealed report.",
        "SIMD-group functions such as `simd_shuffle_down` are the Metal "
        "counterpart of CUDA warp shuffles, not the same instructions.",
        "These values are predictions from the stated rounding and order. No GPU "
        "run gave them.",
        "This lab produces no GB/s value and no speedup.",
        "The lab measured no GPU time.",
        "They are not CUDA, NVIDIA or warp-level results.",
        "`0`: the command succeeded.",
        "`1`: `check` read each file, but a file failed a check.",
        "`2`: the arguments are wrong,",
        "No command needs a GPU, so no exit code marks a missing device.",
        'Ajay must still watch Lecture 9, "Reductions",',
        "Ajay must still do the later NVIDIA run, or record who does it.",
    ):
        assert phrase in text, phrase
    assert (
        gpu_reduce_cuda_cli.EXIT_OK,
        gpu_reduce_cuda_cli.EXIT_REJECTED,
        gpu_reduce_cuda_cli.EXIT_INVALID,
    ) == (0, 1, 2)


def test_the_later_run_keeps_the_method_of_the_metal_lab():
    plan = flat(guide().split("\n## What a later NVIDIA run must do\n", 1)[1])
    plan = plan.split(" ## ", 1)[0]
    config = load_config(PINNED_CONFIG)
    timing = config.benchmark

    # One device buffer, a moving order, the Day 76 rule and every verdict.
    for phrase in (
        "Check the files with `sha256sum -c SHA256SUMS`.",
        "Do not use `--use_fast_math` or `-ftz=true`.",
        "Check that each `__fadd_rn` is one `add.rn.f32` instruction.",
        "Put 1024 quiet-NaN floats after the input, 64 after the partials and 31 "
        "after the output.",
        "Compare each result with the bits that `simulate` prints for its kernel "
        "and input.",
        "Time each kernel and a library sum on the same device buffer.",
        "The library sums are `cub::DeviceReduce::Sum` and `torch.sum` on the CUDA "
        "device.",
        f"Run {timing.warmup} samples that are not timed and {timing.repeats} timed "
        "samples for each side.",
        "Each sample holds the fewest whole calls that read 2^30 bytes.",
        "In repeat r, time side r mod S first, where S is the number of sides.",
        f"A ratio is `faster` only when it is {config.comparison.min_faster_ratio} "
        "or more and the IQR ranges of the two sides do not overlap.",
        "The same test in the other direction gives `slower`.",
        "Otherwise the verdict is `no_clear_difference`.",
        "Keep each verdict in the report, also each `slower` and each "
        "`no_clear_difference` verdict.",
        "`gamma_4096(u) * sum |x|`.",
        "Record the load averages of the host before and after the timing.",
    ):
        assert phrase in plan, phrase
    assert timing.target_bytes_per_sample == 1 << 30
    assert VERDICTS == ("faster", "slower", "no_clear_difference")


def test_the_nvidia_preflight_job_covers_the_cuda_files():
    job = ci_job("nvidia-preflight-contracts")
    sources = sorted(ROOT.glob("src/ai_journey/gpu_reduce_cuda*.py"))
    tests = sorted(ROOT.glob("tests/test_gpu_reduce_cuda*.py"))

    parts = ("", "_check", "_sim", "_cli")
    assert {ROOT / f"src/ai_journey/gpu_reduce_cuda{part}.py" for part in parts} <= (
        set(sources)
    )
    assert {ROOT / f"tests/test_gpu_reduce_cuda{part}.py" for part in parts} | {
        ROOT / "tests/test_gpu_reduce_cuda_guide.py"
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
