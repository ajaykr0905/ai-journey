import json
import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, metal_tc_cli
from ai_journey.metal_tc_config import load_config
from ai_journey.metal_tc_plan import PLAN, plan_sha256
from ai_journey.metal_tc_report import MEASUREMENT_BOUNDARY
from ai_journey.metal_tc_source import KERNELS, PROBE_VERSION, library_sha256
from test_metal_tc_recorded import RECORDED_REPORT_JSON

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "metal-matrix-precision-lab.md"
TC = ["python", "-m", "ai_journey.metal_tc_cli"]
CEILINGS = ["python", "-m", "ai_journey.gpu_ceiling_cli"]
PINNED = "config/day-82-metal-tc.json"
SIDES = ("S1", "M1", "M2", "M3", "PyTorch float32", "PyTorch float16")


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


def table_rows(text, header):
    """The rows of the Markdown table whose header line starts with ``header``."""

    block = text.split(header, 1)[1].split("\n\n", 1)[0]
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in block.splitlines()[1:]
    ]


def test_documented_commands_parse_and_operate_the_pinned_config():
    text = guide()
    run, verify = (
        metal_tc_cli.build_parser().parse_args(argv) for argv in documented(text, TC)
    )
    (ceiling_argv,) = documented(text, CEILINGS)
    ceilings = gpu_ceiling_cli.build_parser().parse_args(ceiling_argv)

    assert (run.command, verify.command) == ("run", "verify")
    assert str(run.config) == str(verify.config) == PINNED
    assert verify.report == run.output
    assert run.ceilings == verify.ceilings == ceilings.output
    assert str(ceilings.config) == "config/day-74-gpu-ceilings.json"
    load_config(ROOT / PINNED)
    assert f"the program name `{metal_tc_cli.PROG}`" in flat(text)


def test_the_guide_names_the_digests_and_predictions_of_the_code():
    text = guide()

    for name, kernel in KERNELS.items():
        if name != PROBE_VERSION:
            assert f"| `{name}` | {kernel.precision} | {kernel.group_size} |" in text
            assert kernel.source_sha256 in text
    for library in ("main", "bfloat"):
        assert library_sha256(library) in text
    assert plan_sha256() in text
    for step in PLAN:
        assert f"| `{step.step}` |" in text
        assert f"| {' or '.join(step.predicted)} |" in text


def test_the_guide_states_the_exit_codes_and_prefixes():
    text = flat(guide())

    for phrase in (
        "- 0: the command succeeded.",
        "- 3: this host cannot run Metal kernels through PyTorch",
        "`apple_gpu_metal_measurement`",
        "`metal_tc_measurement=`",
        "`metal_tc_synthetic_timing=`",
    ):
        assert phrase in text


def test_the_stated_ratios_follow_the_timing_table():
    text = guide()
    times = {
        row[0]: [float(cell.split()[0]) for cell in row[1:]]
        for row in table_rows(text, "| Side | 512 | 1024 | 2048 | 4096 |\n")
    }
    verdicts = {
        row[0].strip("`"): [float(cell.split()[0]) for cell in row[1:]]
        for row in table_rows(text, "| Step | 512 | 1024 | 2048 | 4096 |\n")
    }
    pairs = {
        "m1_over_s1": ("M1", "S1"),
        "m2_over_m1": ("M2", "M1"),
        "m3_over_s1": ("M3", "S1"),
        "m1_over_torch_float32": ("M1", "PyTorch float32"),
        "torch_float16_over_torch_float32": ("PyTorch float16", "PyTorch float32"),
    }

    assert set(SIDES) <= set(times)
    for step, (first, base) in pairs.items():
        for size, ratio in enumerate(verdicts[step]):
            stated = times[base][size] / times[first][size]
            assert abs(stated - ratio) <= 0.011, (step, size)
    assert "M1 over S1 is 1.44, 1.45, 1.41 and 1.25." in flat(text)


def test_the_guide_describes_the_recorded_report_of_the_test_suite():
    text = flat(guide())
    hypotheses = json.loads(RECORDED_REPORT_JSON)["summary"]["hypotheses"]

    assert "on 2026-11-26 at 18:15 IST" in text
    assert "Eight of its 18 predictions missed" in text
    assert sum(record["missed"] for record in hypotheses.values()) == 8
    assert sum(record["held"] for record in hypotheses.values()) == 10


def test_the_guide_keeps_the_evidence_boundary_and_the_learner_work_honest():
    text = flat(guide())

    assert "Ajay must still watch Lectures 23 and 15." in text
    assert "This guide does not summarize the lectures" in text
    assert "Apple GPUs have no NVIDIA Tensor Cores." in text
    assert "It cannot prove that a timing was observed or that a GPU ran." in text
    assert MEASUREMENT_BOUNDARY["tensor_cores_used"] is False
    for claim in (
        "ran on an NVIDIA",
        "Tensor Cores ran",
        "CUDA kernel ran",
        "PTX ran",
        "occupancy of",
    ):
        assert claim not in text
    assert ";" not in re.sub(r"```.*?```|`[^`]*`", "", guide(), flags=re.DOTALL)
