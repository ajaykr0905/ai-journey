import re
import shlex
from pathlib import Path

from ai_journey import (
    matmul_tiled_cli,
    matmul_tiled_config,
    matmul_tiled_variants,
    matmul_tiled_verify,
)
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_source import source_sha256
from ai_journey.matmul_tiled_benchmark import global_loads_per_output
from ai_journey.matmul_tiled_comparison import CUBLAS, vendor_label
from ai_journey.matmul_tiled_config import load_config
from ai_journey.matmul_tiled_report import MEASUREMENT_BOUNDARY
from ai_journey.matmul_tiled_source import LOOP_STATEMENT
from ai_journey.matmul_tiled_variants import (
    CORRECTNESS_MODES,
    VARIANTS,
    library_source,
    text_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-tiled-lab.md"
COMMAND = ["python", "-m", "ai_journey.matmul_tiled_cli"]
PINNED = "config/day-76-matmul-tiled.json"


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
    parser = matmul_tiled_cli.build_parser()
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
        "`unsupported_operating_system`",
        "`apple_gpu_metal_measurement`",
        "`matmul_tiled_measurement=`",
        "`synthetic_timing`",
        "`matmul_tiled_synthetic_timing=`",
        "`verified_`",
    ):
        assert phrase in text, phrase
    assert "unsupported_operating_system" in UNAVAILABLE_REASONS
    assert (
        matmul_tiled_cli.EXIT_OK,
        matmul_tiled_cli.EXIT_REJECTED,
        matmul_tiled_cli.EXIT_INVALID,
        matmul_tiled_cli.EXIT_UNAVAILABLE,
    ) == (0, 1, 2, 3)


def test_guide_keeps_the_evidence_boundary_and_names_no_cublas_number():
    text = flat(guide())

    for phrase in (
        "Nothing here runs a CUDA kernel or uses an NVIDIA GPU.",
        "The lab does not measure cuBLAS.",
        "PyTorch does not report which kernel it runs.",
        f'"{vendor_label("2.15.0")}"',
        f'"{CUBLAS}"',
        "The lab reads no hardware counters.",
        "so this cause is an inference",
    ):
        assert phrase in text, phrase
    true = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is True]
    false = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is False]
    assert "The fields " + " and ".join(f"`{name}`" for name in true) in text
    listed = ", ".join(f"`{name}`" for name in false[:-1]) + f" and `{false[-1]}`"
    assert f"These fields are false: {listed}." in text


def test_guide_pins_every_source_digest_and_mode():
    text = guide()
    pinned = load_config(ROOT / PINNED)

    for name, variant in VARIANTS.items():
        assert f"| `{name}` | `{variant.source_sha256}` |" in text, name
    library = text_sha256(library_source(pinned.tiled_variants()))
    assert f"| tiled library of the pinned config | `{library}` |" in text
    assert f"| Day 74 naive source | `{source_sha256()}` |" in text
    for mode in CORRECTNESS_MODES:
        assert f"`{mode}`" in text, mode
    assert f"`{LOOP_STATEMENT}`" in text


def test_guide_traffic_table_matches_the_model_at_4096_cubed():
    text = guide()
    shape = matmul_shape(4096, 4096, 4096)

    assert (
        f"| naive `matmul_x_columns` | {global_loads_per_output(shape, None):.0f} |"
        in (text)
    )
    for name, variant in VARIANTS.items():
        loads = global_loads_per_output(shape, variant.tile)
        row = f"| `{name}` | {loads:.0f} | {variant.threadgroup_bytes} bytes |"
        assert row in text, name


def test_guide_controls_and_planned_work_match_the_pinned_config():
    text = guide()
    config = load_config(ROOT / PINNED)
    benchmark = config.benchmark
    work = config.planned_work()

    assert f"| `seed` | {config.seed} |" in text
    assert f"| `naive.group_size` | {list(config.naive.group_size)} |" in text
    for field in (
        "warmup",
        "repeats",
        "target_flops_per_sample",
        "max_runtime_seconds",
    ):
        assert f"| `benchmark.{field}` | {getattr(benchmark, field)} |" in text, field
    ratio = config.comparison.min_faster_ratio
    assert f"| `comparison.min_faster_ratio` | {ratio} |" in text
    assert f"The ratio is {ratio} or more." in flat(text)
    assert (
        f"The pinned config has {work.correctness_cases} cases, "
        f"{work.replay_steps} replay steps and {work.correctness_launches} "
        f"correctness launches. It also has {work.benchmark_points} sweep points, "
        f"{work.timed_samples} timed samples, {work.calls} calls and "
        f"{work.flops} FLOPs."
    ) in flat(text)


def test_guide_names_the_two_modules_that_the_verifier_shares_with_the_run():
    text = flat(guide())

    # The verifier parses the config and reads the kernel text as the run does.
    assert matmul_tiled_verify.config_from_mapping is (
        matmul_tiled_config.config_from_mapping
    )
    assert matmul_tiled_verify.VARIANTS is matmul_tiled_variants.VARIANTS
    assert "uses only NumPy and its own code" not in text
    assert (
        "It also uses the config parser and the variant registry of the run."
    ) in text
