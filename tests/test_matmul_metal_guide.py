import re
import shlex
from pathlib import Path

from ai_journey import matmul_metal_cli
from ai_journey.kernel_metal_reference import MAX_VECTOR_BUFFER
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from ai_journey.matmul_metal_config import (
    MAX_REPLAY_STEPS,
    MAX_RUN_CALLS,
    MAX_RUN_FLOPS,
    load_config,
)
from ai_journey.matmul_metal_mapping import MAPPINGS, matmul_launch, matmul_shape
from ai_journey.matmul_metal_reference import ERROR_BOUND, GAMMA, error_bound
from ai_journey.matmul_metal_report import MEASUREMENT_BOUNDARY
from ai_journey.matmul_metal_source import LOOP_STATEMENT

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-metal-lab.md"
COMMAND = ["python", "-m", "ai_journey.matmul_metal_cli"]
PINNED = "config/day-74-matmul-metal.json"


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(COMMAND)] == COMMAND:
                commands.append(words[len(COMMAND) :])
    return commands


def shape_text(shape):
    return f"{shape.m} x {shape.n} x {shape.k}"


def test_documented_commands_parse_and_operate_the_pinned_config():
    commands = documented_commands(guide())
    parser = matmul_metal_cli.build_parser()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["run", "verify"]
    assert {str(arguments.config) for arguments in parsed} == {PINNED}
    assert parsed[1].report == parsed[0].output
    assert (ROOT / PINNED).is_file()
    assert f"the program name `{parser.prog}`" in flat(guide())


def test_guide_states_the_exit_codes_reasons_and_line_prefixes():
    text = flat(guide())

    for phrase in (
        "0: the command succeeded.",
        "1: the command read a report, but the report failed the checks.",
        "2: a usage error",
        "3: this host cannot run Metal kernels through PyTorch",
    ):
        assert phrase in text, phrase
    assert (
        matmul_metal_cli.EXIT_OK,
        matmul_metal_cli.EXIT_REJECTED,
        matmul_metal_cli.EXIT_INVALID,
        matmul_metal_cli.EXIT_UNAVAILABLE,
    ) == (0, 1, 2, 3)
    for reason in UNAVAILABLE_REASONS:
        assert f"`{reason}`" in text, reason
    for prefix in (
        "`apple_gpu_metal_measurement`",
        "`matmul_metal_measurement=`",
        "`synthetic_timing`",
        "`matmul_metal_synthetic_timing=`",
        "`verified_`",
    ):
        assert prefix in text, prefix


def test_guide_quotes_the_boundary_and_defines_peak_as_a_measured_ceiling():
    text = flat(guide())

    assert flat(MEASUREMENT_BOUNDARY["statement"]) in text
    assert "`host-specific Apple GPU (Metal) measurement`" in flat(text)
    assert "`apple_gpu_metal_this_host`" in text
    assert "`synthetic_backend_or_clock`" in text
    assert "Nothing here runs a CUDA kernel or uses an NVIDIA GPU." in text
    assert "It is not a hardware peak or a vendor peak" in text
    assert "it checks the timings for structure only" in text


def test_guide_states_the_loop_the_bound_and_its_growth_with_k():
    text = flat(guide())

    assert f"`{LOOP_STATEMENT}`" in text
    assert f"`{ERROR_BOUND}`" in text
    assert GAMMA == "gamma_K(u) = K * u / (1 - K * u)"
    assert "`gamma_K(u) = K u / (1 - K u)`" in text
    first, second, third = (
        f"{error_bound(k):.2e}".replace("e-0", "e-") for k in (1, 1000, 4096)
    )
    assert (
        f"about {first} at K = 1, {second} at K = 1000 and {third} at K = 4096"
    ) in text


def test_guide_examples_and_lists_follow_from_the_pinned_config():
    text = flat(guide())
    config = load_config(ROOT / PINNED)
    group = config.launch.group_size

    columns, rows = (
        matmul_launch(mapping, matmul_shape(17, 33, 9), group) for mapping in MAPPINGS
    )
    for name, launch in (("x_columns", columns), ("x_rows", rows)):
        width, height = launch.extent
        groups_x, groups_y = launch.groups
        threads_x, threads_y = launch.threads
        assert (
            f"`{name}`: an extent of {width} x {height}, {groups_x} x {groups_y} "
            f"groups, {threads_x} x {threads_y} threads and "
            f"{launch.padding_threads} padding threads."
        ) in text
    padded = [
        shape
        for shape in config.correctness.shapes
        if all(
            extent % size
            for mapping in MAPPINGS
            for extent, size in zip(matmul_launch(mapping, shape, group).extent, group)
        )
    ]
    listed = " ".join(f"- {shape_text(shape)}" for shape in padded)
    assert f"both axes of both mappings: {listed} The shape 32 x 32 x 16" in text


def test_guide_controls_and_planned_work_match_the_pinned_config():
    text = guide()
    config = load_config(ROOT / PINNED)
    timing = config.benchmark

    assert f"| `seed` | {config.seed} |" in text
    assert f"| `launch.group_size` | {list(config.launch.group_size)} |" in text
    for field in ("warmup", "repeats", "target_flops_per_sample"):
        value = getattr(timing, field)
        assert f"| `benchmark.{field}` | {value} |" in text, field
    assert f"| `benchmark.max_runtime_seconds` | {timing.max_runtime_seconds} |" in text
    assert f"{len(config.correctness.shapes)} shapes, 1 x 1 x 1 to" in text
    work = config.planned_work()
    assert (
        f"the counts are {work.correctness_cases} correctness cases, "
        f"{work.replay_steps} replay steps, {work.benchmark_points} sweep points, "
        f"{work.timed_samples} timed samples, {work.calls} kernel calls and "
        f"{work.flops} FLOPs"
    ) in flat(text)
    limits = [value.bit_length() - 1 for value in (MAX_REPLAY_STEPS, MAX_RUN_CALLS)]
    assert flat(
        f"A run can have at most 2^{limits[0]} replay steps, 2^{limits[1]} calls "
        f"and 2^{MAX_RUN_FLOPS.bit_length() - 1} FLOPs."
    ) in flat(text)
    assert f"at most {MAX_VECTOR_BUFFER} elements" in flat(text)
    squares = [shape.m for shape in timing.shapes if shape.m == shape.n == shape.k]
    assert len(squares) == 6 and len(timing.shapes) - len(squares) == 3
    assert "The pinned sweep has 6 square shapes, from 128 to 4096, and 3" in flat(text)
