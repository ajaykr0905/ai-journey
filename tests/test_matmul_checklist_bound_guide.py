import json
import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, matmul_checklist_bound_cli, matmul_checklist_cli
from ai_journey.matmul_bottleneck_config import load_config as load_rule_config
from ai_journey.matmul_checklist_bound import MEASUREMENT_BOUNDARY
from ai_journey.matmul_checklist_bound_traffic import intensity, kernel_traffic
from ai_journey.matmul_checklist_steps import ITEMS
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_metal_mapping import matmul_shape
from test_matmul_checklist_bound_recorded import (
    RECORDED_BOUND_SHA256,
    RECORDED_TIMING_JSON,
)

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-performance-checklist.md"
LAB = ROOT / "docs/matmul-checklist-lab.md"
MODULES = {
    "ai_journey.gpu_ceiling_cli": gpu_ceiling_cli,
    "ai_journey.matmul_checklist_cli": matmul_checklist_cli,
    "ai_journey.matmul_checklist_bound_cli": matmul_checklist_bound_cli,
}
HEADINGS = {
    "coalescing": "Coalescing",
    "occupancy": "Occupancy",
    "tiling": "Tiling",
    "arithmetic_intensity": "Arithmetic intensity",
    "memory_vs_compute_bound": "Memory vs compute bound",
    "control_divergence": "Control divergence",
}
PARTS = (
    "On an NVIDIA GPU,",
    "On an Apple GPU,",
    "- In this lab:",
    "- Result:",
    "- Not measured here:",
)


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            assert words[:2] == ["python", "-m"] and words[2] in MODULES, line
            parser = MODULES[words[2]].build_parser()
            commands.append((words[2], parser.parse_args(words[3:])))
    return commands


def test_the_documented_commands_parse_and_feed_each_other_their_reports():
    commands = documented_commands(guide())
    ceilings, checklist, timing, verify_timing, bound, verify = (
        arguments for _, arguments in commands
    )

    assert [module.rsplit(".", 1)[-1] for module, _ in commands] == [
        "gpu_ceiling_cli",
        "matmul_checklist_cli",
        *["matmul_checklist_bound_cli"] * 4,
    ]
    assert [arguments.command for _, arguments in commands] == [
        "run",
        "run",
        "timing",
        "verify-timing",
        "bound",
        "verify",
    ]
    for arguments in (ceilings, checklist, timing, verify_timing):
        assert (ROOT / arguments.config).is_file()
    assert timing.config == checklist.config == verify_timing.config
    assert verify_timing.report == timing.output
    for arguments in (bound, verify):
        assert arguments.checklist_report == checklist.output
        assert arguments.ceiling_report == ceilings.output
        assert arguments.timing_report == timing.output
        assert (ROOT / arguments.rule_config).is_file()
    assert verify.report == bound.output
    prog = matmul_checklist_bound_cli.build_parser().prog
    assert f"the program name `{prog}`" in flat(guide())


def test_the_guide_states_the_exit_codes_and_line_prefixes():
    text = flat(guide())

    for phrase in (
        "0: the command succeeded.",
        "1: the command read a report, but the report or one of its sources failed",
        "2: a usage error",
        "3: this host cannot run Metal kernels through PyTorch",
        "`apple_gpu_metal_measurement`",
        "`matmul_checklist_one_call_measurement=`",
        "`matmul_checklist_bound_measurement=`",
        "`matmul_checklist_one_call_synthetic_timing=`",
        "`matmul_checklist_bound_synthetic_timing=`",
        "`verified_`",
        "`sources_checked=true`",
    ):
        assert phrase in text, phrase


def test_every_item_of_the_plan_has_its_five_parts():
    text = guide()
    sections = re.split(r"^### ", text, flags=re.MULTILINE)
    bodies = {section.split("\n", 1)[0]: section for section in sections[1:]}

    assert set(HEADINGS) == set(ITEMS)
    for item, heading in HEADINGS.items():
        body = bodies[heading]
        positions = [body.index(part) for part in PARTS]
        assert positions == sorted(positions), item
    assert "| Item | Question | Step | Result on this host |" in text
    for heading in HEADINGS.values():
        assert f"\n| {heading} |" in text, heading


def test_the_model_numbers_of_the_guide_follow_the_traffic_model():
    text = guide()
    shape = matmul_shape(4096, 4096, 4096)

    def value(name):
        return float(intensity(kernel_traffic(name, shape)))

    def reads(name):
        model = kernel_traffic(name, shape).to_mapping()
        return model["threadgroup"]["reads_per_fma"]["decimal"].rstrip("0")

    assert f"| no tile | {value('matmul_x_rows'):.3f} | none |" in text
    assert f"| 32 x 32 x 32, 1 x 1 | {value('matmul_tiled_32'):.2f} | 2 |" in text
    assert reads("matmul_tiled_32") == "2."
    assert f"| 32 x 32 x 32, 4 x 1 | {value('matmul_s2_coarse_4x1'):.2f} |" in text
    assert reads("matmul_s2_coarse_4x1") == "1.25"
    assert f"| 64 x 64 x 16, 4 x 4 | {value('matmul_s5_pad4'):.2f} | 0.5 |" in text
    assert f"| 32 x 32 x 16, 4 x 4 | {value('matmul_s6_group_8x8'):.2f} | 0.5 |" in (
        text
    )
    assert f"| 128 x 128 x 16, 4 x 4 | {value('matmul_s6_group_32x32'):.2f} |" in text
    # Every kernel from S3 on keeps a 4 x 4 block for each thread.
    assert {reads(name) for name in list(VARIANTS)[4:]} == {"0.5"}


def test_the_guide_keeps_the_evidence_boundary_of_the_bound():
    text = flat(guide())
    true = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is True]
    false = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is False]

    listed = ", ".join(f"`{name}`" for name in true[:-1]) + f" and `{true[-1]}`"
    assert f"A measured bound report has these true fields: {listed}." in text
    listed = ", ".join(f"`{name}`" for name in false[:-1]) + f" and `{false[-1]}`"
    assert f"These fields are false: {listed}." in text
    for phrase in (
        "Nothing here runs a CUDA kernel or uses an NVIDIA GPU.",
        "Each name is an inference from timings and measured ceilings, not from "
        "hardware counters.",
        "It does not prove that latency or occupancy is the cause.",
    ):
        assert phrase in text, phrase


def test_the_guide_quotes_the_rule_thresholds_and_every_digest():
    text = flat(guide())
    rule = load_rule_config(ROOT / "config/day-75-matmul-bottleneck.json").rule
    checklist = re.search(r"digest is `([0-9a-f]{64})`", flat(LAB.read_text()))

    assert f"near-roof share of {rule.near_roof_share_min} of the rule" in text
    assert f"The tolerance is {round(rule.timing_tolerance * 100)} percent." in text
    assert f"| checklist | `{checklist.group(1)}` |" in text
    for report in ("gpu-ceilings", "one-call timing", "bound"):
        assert re.search(rf"\| {report} \| `[0-9a-f]{{64}}` \|", text), report
    recorded = json.loads(RECORDED_TIMING_JSON)["report_sha256"]
    assert f"with digest `{recorded}`" in text
    assert f"measured bound with digest `{RECORDED_BOUND_SHA256}`" in text
