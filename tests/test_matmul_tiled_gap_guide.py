import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, matmul_tiled_cli, matmul_tiled_gap_cli
from ai_journey.gpu_ceiling_analysis import FIXED_SHARE_LIMIT
from ai_journey.matmul_bottleneck_config import load_config as load_rule_config
from ai_journey.matmul_tiled_gap import (
    DEFINITIONS,
    GAP_ENVIRONMENT_FIELDS,
    MEASUREMENT_BOUNDARY,
    RULE_MODEL,
)
from ai_journey.matmul_tiled_gap_timing import (
    FLOOR_SHAPE,
    REPORT_SCHEMA_VERSION,
    method_mapping,
)
from test_matmul_tiled_gap_recorded import RECORDED_GAP_SHA256, recorded

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-tiled-lab.md"
GAP = ["python", "-m", "ai_journey.matmul_tiled_gap_cli"]
CEILINGS = ["python", "-m", "ai_journey.gpu_ceiling_cli"]
TILED = ["python", "-m", "ai_journey.matmul_tiled_cli"]
TILED_PINNED = "config/day-76-matmul-tiled.json"
CEILING_PINNED = "config/day-74-gpu-ceilings.json"
RULE_PINNED = "config/day-75-matmul-bottleneck.json"


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


def test_documented_gap_commands_parse_and_chain_their_outputs():
    text = guide()
    parser = matmul_tiled_gap_cli.build_parser()
    timing, verify_timing, gap, verify = (
        parser.parse_args(argv) for argv in documented(text, GAP)
    )
    (ceiling_argv,) = documented(text, CEILINGS)
    ceilings = gpu_ceiling_cli.build_parser().parse_args(ceiling_argv)
    run = matmul_tiled_cli.build_parser().parse_args(documented(text, TILED)[0])

    assert [args.command for args in (timing, verify_timing, gap, verify)] == [
        "timing",
        "verify-timing",
        "gap",
        "verify",
    ]
    assert str(ceilings.config) == CEILING_PINNED
    assert str(timing.config) == str(verify_timing.config) == TILED_PINNED
    assert verify_timing.report == timing.output
    assert (gap.tiled_report, gap.ceiling_report, gap.timing_report) == (
        run.output,
        ceilings.output,
        timing.output,
    )
    assert (verify.tiled_report, verify.ceiling_report, verify.timing_report) == (
        gap.tiled_report,
        gap.ceiling_report,
        gap.timing_report,
    )
    assert str(gap.rule_config) == str(verify.rule_config) == RULE_PINNED
    assert verify.report == gap.output
    for pinned in (TILED_PINNED, CEILING_PINNED, RULE_PINNED):
        assert (ROOT / pinned).is_file(), pinned
    assert load_rule_config(ROOT / RULE_PINNED).rule.intensity_model == RULE_MODEL
    assert f"the program name `{parser.prog}`" in flat(text)


def test_the_gap_section_states_the_inference_boundary_and_the_prefixes():
    text = flat(guide())

    for phrase in (
        "Each name is an inference from timings and measured ceilings, not from "
        "hardware counters.",
        "Code 3 is only for `timing`.",
        "`sources_checked=true`",
        "`matmul_tiled_one_call_measurement=`",
        "`matmul_tiled_gap_measurement=`",
        "`matmul_tiled_one_call_synthetic_timing=`",
        "`matmul_tiled_gap_synthetic_timing=`",
        "PyTorch does not name the kernel of `torch.matmul`, so no traffic model "
        "applies to it.",
        "It does not prove that latency or occupancy is the cause.",
        "Only hardware counters can show which cost it is.",
        "Each side runs at a {} x {} x {} floor.".format(*FLOOR_SHAPE),
    ):
        assert phrase in text, phrase
    true = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is True]
    false = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is False]
    listed = ", ".join(f"`{name}`" for name in true[:-1]) + f" and `{true[-1]}`"
    assert f"A measured gap report has these true fields: {listed}." in text
    listed = ", ".join(f"`{name}`" for name in false[:-1]) + f" and `{false[-1]}`"
    assert f"These fields are false: {listed}." in text


def test_the_guide_says_which_schema_version_seals_which_event_text():
    text = flat(guide())
    old = method_mapping(1)["domains"]
    new = method_mapping(REPORT_SCHEMA_VERSION)["domains"]

    assert (
        f"New `timing` reports have `schema_version` {REPORT_SCHEMA_VERSION} and "
        "seal the version 2 event-domain text of the gpu-timing lab."
    ) in text
    assert (
        "The timing reports in this guide and in the tests are version 1 records "
        'with the old "device-side" text.'
    ) in text
    assert "The verifier accepts each version only with its own text." in text
    assert recorded()["schema_version"] == 1
    assert recorded()["method"]["domains"] == old
    assert any(value.startswith("device-side: ") for value in old.values())
    assert not any("device-side" in value for value in new.values())


def test_the_gap_section_quotes_the_report_model_and_pins_the_recorded_digests():
    text = flat(guide())

    assert "`4 * (2 * M * N * K + M * N)` bytes" in text
    assert "4 * (2 * M * N * K + M * N) bytes" in DEFINITIONS["traffic"]
    assert f"`{recorded()['report_sha256']}`" in text
    assert f"`{RECORDED_GAP_SHA256}`" in text


def test_the_gap_section_states_the_ceiling_flag_and_what_the_sources_omit():
    text = flat(guide())
    limit = round(FIXED_SHARE_LIMIT * 100)

    # The flag bounds the fixed cost of the ceiling call. No rate grows with it.
    assert "increases with the work" not in text
    assert (
        "The ceiling report sets `fma_scales_with_work` to true. Thus the fixed "
        f"cost of one call was at most {limit} percent of the time of the ceiling "
        "call."
    ) in text
    # No source names its host, and the timing report has no warning record.
    assert not any(
        "host" in field or "gpu" in field for field in GAP_ENVIRONMENT_FIELDS
    )
    assert "Thus the `gap` command cannot show that one host made all three." in text
    assert "warnings" not in recorded()
    assert "The timing report has no record of Python warnings" in text
