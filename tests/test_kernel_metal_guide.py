import json
import re
import shlex
from pathlib import Path

from ai_journey import kernel_metal_cli
from ai_journey.kernel_metal_config import MAX_RUN_BYTES, MAX_RUN_CALLS, load_config
from ai_journey.kernel_metal_reference import GRAY_WEIGHTS, MAX_LEVEL_DIFFERENCE
from ai_journey.kernel_metal_report import (
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_SCOPE,
)
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/kernel-metal-lab.md"
COMMAND = ["python", "-m", "ai_journey.kernel_metal_cli"]
PINNED = "config/day-73-kernel-metal.json"


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
    parser = kernel_metal_cli.build_parser()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["run", "verify"]
    assert {str(arguments.config) for arguments in parsed} == {PINNED}
    assert parsed[1].report == parsed[0].output
    assert (ROOT / PINNED).is_file()
    assert parser.prog == " ".join(COMMAND)


def test_guide_states_the_rounding_rule_the_bound_and_the_exit_codes():
    text = flat(guide())

    for weight in GRAY_WEIGHTS:
        assert f"{weight:.4f}" in text
    for step in (
        "Compute `y = 0.2989 R + 0.5870 G + 0.1140 B` in float32",
        "Round half up: `floor(y + 0.5)`.",
        "Clamp the result to 0..255.",
    ):
        assert step in text, step
    assert MAX_LEVEL_DIFFERENCE == 1
    assert "can differ from the reference by at most 1 level" in text
    assert "Each vector-add output must match the reference bit for bit." in text
    for phrase in (
        "0: the command succeeded.",
        "1: the command read a report, but the report failed the checks.",
        "2: a usage error",
        "3: this host cannot run Metal kernels through PyTorch",
    ):
        assert phrase in text, phrase
    assert (
        kernel_metal_cli.EXIT_OK,
        kernel_metal_cli.EXIT_REJECTED,
        kernel_metal_cli.EXIT_INVALID,
        kernel_metal_cli.EXIT_UNAVAILABLE,
    ) == (0, 1, 2, 3)
    for reason in UNAVAILABLE_REASONS:
        assert f"`{reason}`" in text, reason


def test_guide_quotes_the_boundary_and_keeps_the_labels_apart():
    text = flat(guide())

    assert flat(MEASUREMENT_BOUNDARY["statement"]) in text
    assert f"`{MEASUREMENT_LABEL}`" in text
    assert f"`{MEASUREMENT_SCOPE}`" in text
    assert f"`{SYNTHETIC_SCOPE}`" in text
    assert "Nothing here runs a CUDA kernel or uses an NVIDIA GPU." in text
    assert "it checks the timings for structure only" in text


def test_guide_controls_and_extents_match_the_pinned_config():
    text = guide()
    pinned = json.loads((ROOT / PINNED).read_text(encoding="utf-8"))
    config = load_config(ROOT / PINNED)

    assert f"| `seed` | {pinned['seed']} |" in text
    group = pinned["launch"]["vector_add_group_size"]
    assert f"| `launch.vector_add_group_size` | {group} |" in text
    for field in (
        "warmup",
        "repeats",
        "target_bytes_per_sample",
        "max_runtime_seconds",
    ):
        assert f"| `benchmark.{field}` | {pinned['benchmark'][field]} |" in text, field
    work = config.planned_work()
    assert flat(
        f"{work.correctness_cases} correctness cases, {work.benchmark_points} "
        f"sweep points, {work.timed_samples} timed samples, {work.calls} kernel "
        f"calls and {work.accounted_bytes} compulsory bytes"
    ) in flat(text)
    padded = [n for n in config.correctness.vector_add_lengths if n % group]
    listed = ", ".join(str(n) for n in padded[:-1]) + f" and {padded[-1]}"
    assert f"The pinned lengths {listed} are not multiples" in flat(text)
    exact = [n for n in config.correctness.vector_add_lengths if not n % group]
    assert exact == [256]
    assert len(config.benchmark.vector_add_lengths) == 7
    assert len(config.benchmark.rgb_to_gray_shapes) == 6
    assert "The pinned sweep has 7 vector lengths and 6 image shapes." in flat(text)


def test_guide_names_every_line_prefix_and_the_inclusive_bounds():
    text = flat(guide())

    # The summary line has its own prefix; the CLI tests pin both forms.
    for prefix in (
        "`apple_gpu_metal_measurement`",
        "`kernel_metal_measurement=`",
        "`synthetic_timing`",
        "`kernel_metal_synthetic_timing=`",
    ):
        assert prefix in text, prefix
    calls, data = MAX_RUN_CALLS.bit_length() - 1, MAX_RUN_BYTES.bit_length() - 1
    assert (1 << calls, 1 << data) == (MAX_RUN_CALLS, MAX_RUN_BYTES)
    assert f"A run can have at most 2^{calls} calls and 2^{data} bytes." in text
    assert "| `launch.vector_add_group_size` | 256 | 2 to 1024 |" in guide()
    assert "each 2 or more, with at most 1024 threads in total |" in guide()
