import itertools
import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, gpu_reduce_cli
from ai_journey.gpu_reduce_bound import summation_depth
from ai_journey.gpu_reduce_config import load_config
from ai_journey.gpu_reduce_plan import PLAN, SIZE_CLASSES, plan_sha256
from ai_journey.gpu_reduce_report import MEASUREMENT_BOUNDARY
from ai_journey.gpu_reduce_source import KERNELS, STEP_KERNELS, library_sha256
from ai_journey.gpu_reduce_speedup import vendor_label
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from test_gpu_reduce_cli import run_argv, stand_in_main, write_config
from test_gpu_reduce_config import make_config_mapping
from test_gpu_reduce_recorded import RECORDED_SHA256
from test_gpu_reduce_speedup import write_ceiling

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-reduction-lab.md"
COMMAND = ["python", "-m", "ai_journey.gpu_reduce_cli"]
CEILING_COMMAND = ["python", "-m", "ai_journey.gpu_ceiling_cli"]
PINNED = "config/day-78-gpu-reduce.json"
CEILING_CONFIG = "config/day-74-gpu-ceilings.json"


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands(text, prefix):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(prefix)] == prefix:
                commands.append(words[len(prefix) :])
    return commands


def test_documented_commands_parse_and_chain_the_ceiling_to_the_run():
    (ceiling,) = documented_commands(guide(), CEILING_COMMAND)
    commands = documented_commands(guide(), COMMAND)
    made = gpu_ceiling_cli.build_parser().parse_args(ceiling)
    parser = gpu_reduce_cli.build_parser()
    run, verify = (parser.parse_args(argv) for argv in commands)

    assert [argv[0] for argv in commands] == ["run", "verify"]
    assert str(made.config) == CEILING_CONFIG and (ROOT / CEILING_CONFIG).is_file()
    assert str(run.config) == str(verify.config) == PINNED
    assert run.ceilings == verify.ceilings == made.output
    assert verify.report == run.output
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
        "`gpu_reduce_measurement=`",
        "`synthetic_timing`",
        "`gpu_reduce_synthetic_timing=`",
        "`verified_`",
        "The CI job `nvidia-preflight-contracts` runs the tests of this lab",
    ):
        assert phrase in text, phrase
    assert "unsupported_operating_system" in UNAVAILABLE_REASONS


# The phrase of each kind of line in the guide's numbered list, and the word
# after the label that starts that kind of printed line.
LINE_KINDS = (
    ("plan step", "plan"),
    ("load averages", "load_average"),
    ("bandwidth ceiling", "ceiling"),
    ("correctness case", "correctness"),
    ("timing size, with the median", "benchmark"),
    ("two step ratios", "speedup"),
    ("fastest kernel", "best"),
    ("order of an earlier step", "same_order"),
    ("Python warnings", "warnings"),
    ("summary line", "summary"),
)


def documented_line_kinds(text, points):
    """Read the numbered list of printed lines; repeat a per-size item."""

    block = text.split("Then it prints\nthese lines:\n\n", 1)[1].split("\n\n", 1)[0]
    kinds = []
    for item in re.split(r"\n(?=\d+\. )", block):
        item = flat(re.sub(r"^\d+\. ", "", item))
        named = sorted(
            (item.index(phrase), kind) for phrase, kind in LINE_KINDS if phrase in item
        )
        repeats = points if item.startswith("For each timing size") else 1
        kinds += [kind for _, kind in named] * repeats
    return kinds


def test_the_numbered_list_gives_the_printed_lines_in_their_order(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    ceilings = write_ceiling(tmp_path)

    assert stand_in_main(run_argv(config, ceilings, tmp_path / "r.json")) == 0
    lines = capsys.readouterr().out.splitlines()
    printed = [
        "summary" if line.startswith("gpu_reduce_") else line.split()[1]
        for line in lines
    ]
    runs = [kind for kind, _ in itertools.groupby(printed)]
    points = len(make_config_mapping()["benchmark"]["sizes"])

    assert points == 2
    assert documented_line_kinds(guide(), points) == runs


def test_guide_keeps_the_evidence_boundary_and_the_simd_group_caveat():
    text = flat(guide())

    for phrase in (
        "Nothing here runs a CUDA kernel or uses an NVIDIA GPU.",
        "The lab reads no hardware counters.",
        "SIMD-group functions are Metal's counterpart of CUDA warp shuffles, not "
        "the same instruction.",
        f'"{vendor_label("2.15.0")}"',
        "It is not a vendor peak, and it is not a hardware limit.",
    ):
        assert phrase in text, phrase
    true = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is True]
    false = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is False]
    assert "The fields " + " and ".join(f"`{name}`" for name in true) in text
    listed = ", ".join(f"`{name}`" for name in false[:-1]) + f" and `{false[-1]}`"
    assert f"These fields are false: {listed}." in text


def test_guide_quotes_every_step_and_keeps_the_learner_work_pending():
    text = flat(guide())

    assert f"`{plan_sha256()}`" in text
    for step in PLAN:
        assert flat(step.change) in text, step.step
        assert flat(step.hypothesis) in text, step.step
    for name, first, last in SIZE_CLASSES:
        assert f"| {name} | {first} | {last} |" in guide(), name
    for phrase in (
        "Ajay must still watch Lecture 9 and Lecture 20.",
        "Ajay must still check it against Lecture 9.",
        'Ajay must still watch Lecture 20, "Scan Algorithm". This lab has no scan '
        "kernel.",
    ):
        assert phrase in text, phrase


def test_guide_pins_every_kernel_digest_the_library_and_the_depths():
    text = guide()

    for name, kernel in KERNELS.items():
        assert f"| `{name}` | `{kernel.source_sha256}` |" in text, name
    assert f"| reduction library | `{library_sha256()}` |" in text
    rows = {"1": 1, "1000003": 1000003, "2^26": 1 << 26}
    for label, n in rows.items():
        depths = " | ".join(str(summation_depth(name, n)) for name in STEP_KERNELS)
        assert f"| {label} | {depths} |" in text, label
    assert f"`{RECORDED_SHA256}`" in text


def test_guide_controls_and_planned_work_match_the_pinned_config():
    text = guide()
    config = load_config(ROOT / PINNED)
    benchmark = config.benchmark
    work = config.planned_work()

    assert f"| `seed` | {config.seed} |" in text
    assert f"| `correctness.runs` | {config.correctness.runs} |" in text
    sizes = ", ".join(map(str, benchmark.sizes[:-1])) + f" and {benchmark.sizes[-1]}"
    assert f"| `benchmark.sizes` | {sizes} |" in text
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
        f"The pinned config has {work.correctness_cases} cases, "
        f"{work.correctness_elements} case elements and "
        f"{work.correctness_launches} correctness launches. It also has "
        f"{work.benchmark_points} timing sizes, {work.timed_samples} timed "
        f"samples, {work.calls} calls and {work.bytes_read} bytes."
    ) in flat(text)
    calls = ", ".join(str(benchmark.calls_per_sample(n)) for n in benchmark.sizes[:-1])
    assert (
        f"a sample holds {calls} and {benchmark.calls_per_sample(benchmark.sizes[-1])}"
        " calls."
    ) in flat(text)
