import re
import shlex
from pathlib import Path

from ai_journey import gpu_timing_cli
from ai_journey.gpu_timing_clocks import CLOCK_DOMAINS
from ai_journey.gpu_timing_config import MAX_TINY_FLOPS, load_config
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-timing-lab.md"
COMMAND = ["python", "-m", "ai_journey.gpu_timing_cli"]
PINNED = "config/day-75-gpu-timing.json"


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


def test_documented_commands_parse_and_operate_the_pinned_config():
    commands = documented_commands(guide())
    parser = gpu_timing_cli.build_parser()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["run", "verify"]
    assert {str(arguments.config) for arguments in parsed} == {PINNED}
    assert parsed[1].report == parsed[0].output
    assert (ROOT / PINNED).is_file()
    assert f"the program name `{parser.prog}`" in flat(guide())


def test_guide_states_the_exit_codes_and_line_prefixes():
    text = flat(guide())

    for phrase in (
        "0: the command succeeded.",
        "1: the command read a report, but the report failed the checks.",
        "2: a usage error",
        "3: this host cannot run Metal kernels through PyTorch",
    ):
        assert phrase in text, phrase
    assert (
        gpu_timing_cli.EXIT_OK,
        gpu_timing_cli.EXIT_REJECTED,
        gpu_timing_cli.EXIT_INVALID,
        gpu_timing_cli.EXIT_UNAVAILABLE,
    ) == (0, 1, 2, 3)
    assert "`unsupported_operating_system`" in text
    assert "unsupported_operating_system" in UNAVAILABLE_REASONS
    for prefix in (
        "`apple_gpu_metal_measurement`",
        "`gpu_timing_measurement=`",
        "`synthetic_timing`",
        "`gpu_timing_synthetic_timing=`",
        "`verified_`",
    ):
        assert prefix in text, prefix


def test_guide_names_each_clock_with_its_domain():
    text = guide()

    for clock, domain in CLOCK_DOMAINS.items():
        assert f"| `{clock}` | `{domain}` |" in text, clock
    flat_text = flat(text)
    assert "`round(ms * 1e6)`" in flat_text
    assert "`cross_domain: true`" in flat_text
    assert "`1 / median_ns**2`" in flat_text


def test_guide_keeps_the_evidence_boundary():
    text = flat(guide())

    for phrase in (
        "Nothing here runs a CUDA kernel or uses an NVIDIA GPU.",
        "reads no hardware counters",
        "not a vendor peak",
        "is an inference from timings",
        "`host-specific Apple GPU (Metal) measurement`",
        "`apple_gpu_metal_this_host`",
        "`synthetic_backend_or_clock`",
    ):
        assert phrase in text, phrase


def test_guide_controls_and_planned_work_match_the_pinned_config():
    text = guide()
    config = load_config(ROOT / PINNED)
    sweep = config.sweep

    assert f"| `seed` | {config.seed} |" in text
    assert f"| `launch.group_size` | {list(config.launch.group_size)} |" in text
    for field in ("warmup", "repeats", "max_runtime_seconds"):
        assert f"| `sweep.{field}` | {getattr(sweep, field)} |" in text, field
    ceiling = config.reference.fma_ceiling_gflops_per_second
    assert f"| `reference.fma_ceiling_gflops_per_second` | {ceiling} |" in text
    assert f"The Day 74 FMA ceiling is {ceiling:.0f} GFLOP/s" in flat(text)
    sides = [shape.m for shape in sweep.shapes]
    assert all(shape.m == shape.n == shape.k for shape in sweep.shapes)
    listed = ", ".join(str(side) for side in sides[:-1]) + f" and {sides[-1]}"
    assert f"The pinned sweep uses the cubes {listed}." in flat(text)
    assert f"| `sweep.shapes` | {len(sides)} cubes, 1 to {sides[-1]} |" in text
    assert f"the first shape has at most {MAX_TINY_FLOPS} FLOPs" in text
    work = config.planned_work()
    assert (
        f"The planned work has {work.sides} sides, {work.timed_samples} timed "
        f"samples, {work.launches} launches and {work.flops} FLOPs."
    ) in flat(text)
