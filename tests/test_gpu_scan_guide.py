import itertools
import json
import re
import shlex
from pathlib import Path

from ai_journey import gpu_scan_cli
from ai_journey.gpu_ceiling_source import source_sha256 as ceiling_source_sha256
from ai_journey.gpu_scan_config import load_config
from ai_journey.gpu_scan_plan import PLAN, plan_sha256
from ai_journey.gpu_scan_report import MEASUREMENT_BOUNDARY
from ai_journey.gpu_scan_source import KERNELS, S2_SIMD, add_mapping, library_sha256
from ai_journey.gpu_scan_speedup import vendor_label
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from test_gpu_scan_cli import run_argv, stand_in_main, write_config
from test_gpu_scan_config import make_config_mapping
from test_gpu_scan_correctness import StandInBackend
from test_gpu_scan_recorded import RECORDED_REPORT_JSON

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-scan-lab.md"
COMMAND = ["python", "-m", "ai_journey.gpu_scan_cli"]
PINNED = "config/day-78-gpu-scan.json"


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
    parser = gpu_scan_cli.build_parser()
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
        "`gpu_scan_measurement=`",
        "`synthetic_timing`",
        "`gpu_scan_synthetic_timing=`",
        "`verified_`",
    ):
        assert phrase in text, phrase
    assert "unsupported_operating_system" in UNAVAILABLE_REASONS


# The phrase of each kind of line in the guide's numbered list, and the word
# after the label that starts that kind of printed line.
LINE_KINDS = (
    ("hypothesis of the plan", "plan"),
    ("load averages", "load_average"),
    ("refused", "refused"),
    ("correctness case", "correctness"),
    ("sweep point, with the median", "benchmark"),
    ("kernel of the plan", "speedup"),
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
    backend = StandInBackend(widths={S2_SIMD: 16})

    assert stand_in_main(run_argv(config, tmp_path / "r.json"), backend) == 0
    lines = capsys.readouterr().out.splitlines()
    printed = [
        "summary" if line.startswith("gpu_scan_") else line.split()[1] for line in lines
    ]
    runs = [kind for kind, _ in itertools.groupby(printed)]
    points = len(make_config_mapping()["benchmark"]["lengths"])

    assert runs.count("refused") == 1 and points == 2
    assert documented_line_kinds(guide(), points) == runs


def test_guide_keeps_the_evidence_boundary_and_names_the_vendor_call():
    text = flat(guide())

    for phrase in (
        "Ajay must still watch Lecture 20.",
        "Nothing here runs a CUDA kernel or uses an NVIDIA GPU.",
        "The SIMD-group functions of S2 are the Metal counterpart of CUDA warp "
        "shuffles, not the same instruction.",
        "PyTorch does not report which kernel `torch.cumsum` runs.",
        "The lab reads no hardware counters.",
        f'"{vendor_label("2.15.0")}"',
        "These names are inferences from timings and the copy rate of the same run.",
    ):
        assert phrase in text, phrase
    true = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is True]
    false = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is False]
    assert "The fields " + " and ".join(f"`{name}`" for name in true) in text
    listed = ", ".join(f"`{name}`" for name in false[:-1]) + f" and `{false[-1]}`"
    assert f"These fields are false: {listed}." in text
    for claim in ("watched Lecture 20", "Ajay watched", "occupancy was", "cuBLAS"):
        assert claim not in text, claim


def test_guide_quotes_every_change_and_hypothesis_and_the_plan_digest():
    text = flat(guide())

    assert f"`{plan_sha256()}`" in text
    for entry in PLAN:
        assert flat(entry.change) in text, entry.step
        assert flat(entry.hypothesis) in text, entry.step
    assert "before the first timed run" in text


def test_guide_pins_every_kernel_digest_and_the_kernel_table():
    text = guide()

    for name, kernel in KERNELS.items():
        assert f"| `{name}` | `{kernel.source_sha256}` |" in text, name
        row = (
            f"| `{name}` | {kernel.threads_per_group} | {kernel.elements_per_thread} | "
            f"{kernel.threadgroup_bytes} | {kernel.barriers_per_block} | "
            f"{kernel.block_depth} | {'yes' if kernel.replay else 'no'} |"
        )
        assert row in text, name
    assert f"| `scan_add_offsets` | `{add_mapping()['source_sha256']}` |" in text
    assert f"| scan library | `{library_sha256()}` |" in text
    assert f"| Day 74 ceiling source | `{ceiling_source_sha256()}` |" in text


def test_guide_controls_planned_work_and_recorded_digest_match_the_code():
    text = guide()
    config = load_config(ROOT / PINNED)
    benchmark = config.benchmark
    work = config.planned_work()

    assert f"| `seed` | {config.seed} |" in text
    assert f"| `correctness.runs` | {config.correctness.runs} |" in text
    for field in (
        "warmup",
        "repeats",
        "target_bytes_per_sample",
        "max_runtime_seconds",
    ):
        assert f"| `benchmark.{field}` | {getattr(benchmark, field)} |" in text, field
    ratio = config.comparison.min_faster_ratio
    assert f"| `comparison.min_faster_ratio` | {ratio} |" in text
    assert (
        f"The pinned config has {work.correctness_cases} cases with "
        f"{work.case_elements} elements and {work.correctness_runs} correctness "
        f"runs. It also has {work.benchmark_points} sweep points, "
        f"{work.timed_samples} timed samples, {work.calls} calls and "
        f"{work.bytes} bytes."
    ) in flat(text)
    digest = json.loads(RECORDED_REPORT_JSON)["report_sha256"]
    assert f"`{digest}`" in text
