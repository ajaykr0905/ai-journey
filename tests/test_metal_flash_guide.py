import json
import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, metal_flash_cli
from ai_journey.metal_flash_config import load_config
from ai_journey.metal_flash_plan import PLAN, plan_sha256
from ai_journey.metal_flash_report import MEASUREMENT_BOUNDARY
from ai_journey.metal_flash_source import KERNELS, PROBE_EXP, library_sha256
from test_metal_flash_recorded import RECORDED_REPORT_JSON

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "metal-flash-attention-lab.md"
FLASH = ["python", "-m", "ai_journey.metal_flash_cli"]
CEILINGS = ["python", "-m", "ai_journey.gpu_ceiling_cli"]
PINNED = "config/day-81-metal-flash.json"


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
        metal_flash_cli.build_parser().parse_args(argv)
        for argv in documented(text, FLASH)
    )
    (ceiling_argv,) = documented(text, CEILINGS)
    ceilings = gpu_ceiling_cli.build_parser().parse_args(ceiling_argv)

    assert (run.command, verify.command) == ("run", "verify")
    assert str(run.config) == str(verify.config) == PINNED
    assert verify.report == run.output
    assert run.ceilings == verify.ceilings == ceilings.output
    assert str(ceilings.config) == "config/day-74-gpu-ceilings.json"
    load_config(ROOT / PINNED)
    assert f"the program name `{metal_flash_cli.PROG}`" in flat(text)


def test_the_guide_names_the_digests_and_predictions_of_the_code():
    text = guide()

    for name, kernel in KERNELS.items():
        if name != PROBE_EXP:
            assert f"| `{name}` | {kernel.group_size} |" in text
            assert kernel.source_sha256 in text
    assert library_sha256() in text
    assert plan_sha256() in text
    for step in PLAN:
        assert f"| `{step.step}` |" in text
        assert f"| {' or '.join(step.predicted)} |" in text


def test_the_guide_states_the_exit_codes_prefixes_and_memory_limits():
    text = flat(guide())

    for phrase in (
        "- 0: the command succeeded.",
        "- 3: this host cannot run Metal kernels through PyTorch",
        "`apple_gpu_metal_measurement`",
        "`metal_flash_measurement=`",
        "`metal_flash_synthetic_timing=`",
        "A buffer that lives and dies between two polls.",
        "This result does not prove that no such buffer existed.",
    ):
        assert phrase in text


def test_the_guide_describes_the_recorded_report_of_the_test_suite():
    text = flat(guide())

    assert "2026-11-24 at 08:47 IST" in text
    assert "Three of its six predictions missed" in text
    hypotheses = json.loads(RECORDED_REPORT_JSON)["summary"]["hypotheses"]
    assert sum(record["missed"] for record in hypotheses.values()) == 3


def test_the_guide_keeps_the_evidence_boundary_and_the_learner_work_honest():
    text = flat(guide())

    assert "Ajay must still watch Lecture 12." in text
    assert "This guide does not summarize the lecture" in text
    assert "The lab times forward passes only." in text
    assert "It cannot prove that a timing was observed or that a GPU ran." in text
    assert MEASUREMENT_BOUNDARY["ptx_executed"] is False
    for claim in ("ran on an NVIDIA", "CUDA kernel ran", "PTX ran", "occupancy of"):
        assert claim not in text
    assert ";" not in re.sub(r"```.*?```|`[^`]*`", "", guide(), flags=re.DOTALL)
