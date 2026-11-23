import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, metal_liger_cli
from ai_journey.metal_liger_benchmark import modeled_bytes
from ai_journey.metal_liger_config import load_config
from ai_journey.metal_liger_plan import plan_sha256
from ai_journey.metal_liger_report import MEASUREMENT_BOUNDARY
from ai_journey.metal_liger_source import FAMILY_KERNELS, KERNELS, library_sha256

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "metal-liger-lab.md"
LIGER = ["python", "-m", "ai_journey.metal_liger_cli"]
CEILINGS = ["python", "-m", "ai_journey.gpu_ceiling_cli"]
PINNED = "config/day-80-metal-liger.json"


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
        metal_liger_cli.build_parser().parse_args(argv)
        for argv in documented(text, LIGER)
    )
    (ceiling_argv,) = documented(text, CEILINGS)
    ceilings = gpu_ceiling_cli.build_parser().parse_args(ceiling_argv)

    assert (run.command, verify.command) == ("run", "verify")
    assert str(run.config) == str(verify.config) == PINNED
    assert verify.report == run.output
    assert run.ceilings == verify.ceilings == ceilings.output
    assert str(ceilings.config) == "config/day-74-gpu-ceilings.json"
    load_config(ROOT / PINNED)
    assert f"the program name `{metal_liger_cli.PROG}`" in flat(text)


def test_the_guide_names_the_digests_of_the_code_it_describes():
    text = guide()

    for name in FAMILY_KERNELS.values():
        assert f"| `{name}` |" in text
        assert KERNELS[name].source_sha256 in text
    assert library_sha256() in text
    assert plan_sha256() in text


def test_the_guide_states_the_exit_codes_prefixes_and_the_softmax_bytes():
    text = flat(guide())

    for phrase in (
        "- 0: the command succeeded.",
        "- 3: this host cannot run Metal kernels through PyTorch",
        "`apple_gpu_metal_measurement`",
        "`metal_liger_measurement=`",
        "`metal_liger_synthetic_timing=`",
    ):
        assert phrase in text
    for rows, cols in ((1024, 32000), (1024, 50257)):
        assert f"| {rows} | {cols} | {rows * cols} | {4 * rows * cols} " in text
    assert modeled_bytes(
        "torch_rmsnorm",
        "rmsnorm",
        load_config(ROOT / PINNED).benchmark.shapes["rmsnorm"][0].layout(),
    ) == (40 * 4096 * 1024 + 20 * 4096)


def test_the_guide_keeps_the_evidence_boundary_and_the_learner_work_honest():
    text = flat(guide())

    assert "Ajay must still watch Lecture 28." in text
    assert "This guide does not summarize the lecture" in text
    assert "No code of the Liger Kernel project runs here." in text
    assert "The lab times forward passes only." in text
    assert "It cannot prove that a timing was observed or that a GPU ran." in text
    assert MEASUREMENT_BOUNDARY["liger_kernel_code_executed"] is False
    for claim in ("ran on an NVIDIA", "CUDA kernel ran", "PTX ran", "occupancy of"):
        assert claim not in text
    assert ";" not in re.sub(r"```.*?```|`[^`]*`", "", guide(), flags=re.DOTALL)
