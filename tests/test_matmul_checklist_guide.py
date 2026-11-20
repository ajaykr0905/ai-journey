import itertools
import re
import shlex
from pathlib import Path

from ai_journey import matmul_checklist_cli
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from ai_journey.matmul_checklist_config import load_config
from ai_journey.matmul_checklist_report import MEASUREMENT_BOUNDARY
from ai_journey.matmul_checklist_steps import CHECKLIST, checklist_sha256
from ai_journey.matmul_checklist_variants import VARIANTS, checklist_library
from ai_journey.matmul_metal_source import source_sha256
from ai_journey.matmul_tiled_comparison import CUBLAS, vendor_label
from ai_journey.matmul_tiled_variants import text_sha256
from test_matmul_checklist_cli import run_argv, stand_in_main, write_config
from test_matmul_checklist_config import make_config_mapping
from test_matmul_checklist_correctness import StandInBackend

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-checklist-lab.md"
COMMAND = ["python", "-m", "ai_journey.matmul_checklist_cli"]
PINNED = "config/day-77-matmul-checklist.json"


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
    parser = matmul_checklist_cli.build_parser()
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
        "`matmul_checklist_measurement=`",
        "`synthetic_timing`",
        "`matmul_checklist_synthetic_timing=`",
        "`verified_`",
    ):
        assert phrase in text, phrase
    assert "unsupported_operating_system" in UNAVAILABLE_REASONS


# The phrase of each kind of line in the guide's numbered list, and the word
# after the label that starts that kind of printed line.
LINE_KINDS = (
    ("checklist step", "checklist"),
    ("load averages", "load_average"),
    ("refused", "refused"),
    ("correctness case", "correctness"),
    ("sweep point, with the median", "benchmark"),
    ("step kernel", "speedup"),
    ("fastest kernel", "best"),
    ("Python warnings", "warnings"),
    ("summary line", "summary"),
)


def documented_line_kinds(text, points):
    """Read the numbered list of printed lines; repeat a per-point item."""

    block = text.split("Then it prints these lines:\n\n", 1)[1].split("\n\n", 1)[0]
    kinds = []
    for item in re.split(r"\n(?=\d+\. )", block):
        item = flat(re.sub(r"^\d+\. ", "", item))
        named = sorted(
            (item.index(phrase), kind) for phrase, kind in LINE_KINDS if phrase in item
        )
        repeats = points if item.startswith("For each sweep point") else 1
        kinds += [kind for _, kind in named] * repeats
    return kinds


def test_the_numbered_list_gives_the_printed_lines_in_their_order(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    backend = StandInBackend(threads={"matmul_s6_group_32x32": 512})

    assert stand_in_main(run_argv(config, tmp_path / "r.json"), backend) == 0
    lines = capsys.readouterr().out.splitlines()
    printed = [
        "summary" if line.startswith("matmul_checklist_") else line.split()[1]
        for line in lines
    ]
    runs = [kind for kind, _ in itertools.groupby(printed)]
    points = len(make_config_mapping()["benchmark"]["shapes"])

    assert runs.count("refused") == 1 and points == 2
    assert documented_line_kinds(guide(), points) == runs


def test_guide_keeps_the_evidence_boundary_and_names_no_cublas_number():
    text = flat(guide())

    for phrase in (
        "Nothing here runs a CUDA kernel or uses an NVIDIA GPU.",
        "The lab does not measure cuBLAS.",
        "The lab reads no hardware counters, so it does not measure occupancy.",
        f'"{vendor_label("2.15.0")}"',
        f'"{CUBLAS}"',
    ):
        assert phrase in text, phrase
    true = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is True]
    false = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is False]
    assert "The fields " + " and ".join(f"`{name}`" for name in true) in text
    listed = ", ".join(f"`{name}`" for name in false[:-1]) + f" and `{false[-1]}`"
    assert f"These fields are false: {listed}." in text


def test_guide_quotes_every_step_and_says_who_confirms_the_items():
    text = flat(guide())

    assert "The item list follows the tracker notes." in text
    assert "Ajay must still check it against Lecture 8." in text
    assert f"`{checklist_sha256()}`" in text
    for step in CHECKLIST:
        assert flat(step.change) in text, step.step
        assert flat(step.hypothesis) in text, step.step


def test_guide_pins_every_kernel_digest_and_the_library_digest():
    text = guide()

    for name, variant in VARIANTS.items():
        if variant.origin == "day_74_naive":
            continue
        assert f"`{name}` | `{variant.source_sha256}` |" in text, name
    assert f"| Day 74 naive source | `{source_sha256()}` |" in text
    library = text_sha256(checklist_library())
    assert f"| checklist library | `{library}` |" in text


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
    assert (
        f"The pinned config has {work.correctness_cases} cases, "
        f"{work.replay_steps} replay steps and {work.correctness_launches} "
        f"correctness launches. It also has {work.benchmark_points} sweep points, "
        f"{work.timed_samples} timed samples, {work.calls} calls and "
        f"{work.flops} FLOPs."
    ) in flat(text)
