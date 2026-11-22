import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, metal_fused_cli
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from ai_journey.metal_fused_config import load_config
from ai_journey.metal_fused_plan import PLAN, plan_sha256
from ai_journey.metal_fused_report import MEASUREMENT_BOUNDARY
from ai_journey.metal_fused_source import VARIANTS, library_sha256
from test_metal_fused_cli import inputs, run_argv, stand_in_main

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/metal-fused-softmax-lab.md"
FUSED = ["python", "-m", "ai_journey.metal_fused_cli"]
CEILINGS = ["python", "-m", "ai_journey.gpu_ceiling_cli"]
PINNED = "config/day-79-metal-fused.json"


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented(text, command):
    found = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(command)] == command:
                found.append(words[len(command) :])
    return found


def test_documented_commands_parse_and_operate_the_pinned_config():
    text = guide()
    run, verify = (
        metal_fused_cli.build_parser().parse_args(argv)
        for argv in documented(text, FUSED)
    )
    (ceiling_argv,) = documented(text, CEILINGS)
    ceilings = gpu_ceiling_cli.build_parser().parse_args(ceiling_argv)

    assert (run.command, verify.command) == ("run", "verify")
    assert str(run.config) == str(verify.config) == PINNED
    assert verify.report == run.output
    assert run.ceilings == verify.ceilings == ceilings.output
    assert str(ceilings.config) == "config/day-74-gpu-ceilings.json"
    assert (ROOT / PINNED).is_file()
    assert "RECORDED_REPORT_JSON as T" in text
    assert f"the program name `{metal_fused_cli.PROG}`" in flat(text)


def test_guide_states_the_exit_codes_and_line_prefixes():
    text = flat(guide())

    for phrase in (
        "0: the command succeeded.",
        "1: the command read a report, but the report failed the checks.",
        "2: a usage error",
        "3: this host cannot run Metal kernels through PyTorch",
        "`unsupported_operating_system`",
        "`apple_gpu_metal_measurement`",
        "`metal_fused_measurement=`",
        "`synthetic_timing`",
        "`metal_fused_synthetic_timing=`",
        "`verified_`",
    ):
        assert phrase in text, phrase
    assert "unsupported_operating_system" in UNAVAILABLE_REASONS


# The word after the label that starts each kind of printed line, in the order
# of the guide's numbered list.
LINE_KINDS = (
    "plan",
    "load_average",
    "ceiling",
    "exp_probe",
    "refused",
    "correctness",
    "benchmark",
    "speedup",
    "warnings",
)


def test_the_numbered_list_gives_the_printed_lines_in_their_order(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    assert stand_in_main(run_argv(config, ceilings, tmp_path / "r.json")) == 0
    lines = capsys.readouterr().out.splitlines()

    kinds = [line.split()[1] for line in lines[:-1]]
    order = [LINE_KINDS.index("speedup" if kind == "best" else kind) for kind in kinds]
    assert order == sorted(order)
    assert lines[-1].startswith("metal_fused_synthetic_timing=")
    listed = re.findall(r"^\d+\. One line", guide(), flags=re.MULTILINE)
    assert len(listed) >= 8


def test_guide_pins_every_step_digest_the_library_and_the_plan():
    text = guide()

    for variant in VARIANTS.values():
        assert f"| `{variant.name}` | `{variant.source_sha256}` |" in text
    assert library_sha256() in text
    assert plan_sha256() in text


def test_the_plan_table_states_each_declared_prediction():
    text = guide()
    words = {
        "faster": "faster",
        "slower": "slower",
        "no_clear_difference": "no clear difference",
    }

    for step in PLAN:
        predicted = " or ".join(words[verdict] for verdict in step.predicted)
        row = re.search(rf"^\| {step.step} \|.*\| {predicted} \|$", text, re.MULTILINE)
        assert row is not None, step.step


def test_guide_controls_match_the_pinned_config():
    config = load_config(ROOT / PINNED)
    text = guide()

    assert f"| `seed` | {config.seed} |" in text
    assert f"| `correctness.repeats` | {config.correctness.repeats} |" in text
    assert f"| `benchmark.warmup` | {config.benchmark.warmup} |" in text
    assert f"| `benchmark.repeats` | {config.benchmark.repeats} |" in text
    target = config.benchmark.target_bytes_per_sample
    assert f"| `benchmark.target_bytes_per_sample` | {target} |" in text
    runtime = config.benchmark.max_runtime_seconds
    assert f"| `benchmark.max_runtime_seconds` | {runtime} |" in text
    ratio = config.comparison.min_faster_ratio
    assert f"| `comparison.min_faster_ratio` | {ratio} |" in text
    assert f"| `correctness.cases` | {len(config.correctness.cases)} cases |" in text
    assert f"| `benchmark.shapes` | {len(config.benchmark.shapes)} shapes |" in text


def test_guide_keeps_the_evidence_boundary_and_the_pending_lecture():
    text = flat(guide())

    for field, value in MEASUREMENT_BOUNDARY.items():
        if value is False:
            assert f"`{field}`" in text, field
    assert "Ajay must still watch Lecture 18." in text
    assert "Ajay must still watch Lecture 18 and compare" in text
    assert "not evidence of CUDA, NVIDIA or Triton GPU performance" in text
    assert "no Triton kernel ran on a GPU" in text
