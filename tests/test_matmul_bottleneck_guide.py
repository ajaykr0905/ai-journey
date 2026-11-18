import json
import re
import shlex
from pathlib import Path

from ai_journey import (
    gpu_ceiling_cli,
    matmul_bottleneck_cli,
    matmul_metal_cli,
    matmul_roofline_cli,
)
from ai_journey.matmul_bottleneck_rule import BOTTLENECKS
from test_matmul_bottleneck_recorded import GAP_SHA256, RECORDED_REPORT, REPORT_SHA256

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-bottleneck-lab.md"
CONFIG = ROOT / "config/day-75-matmul-bottleneck.json"
NAIVE = ("matmul_x_columns", "matmul_x_rows")


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented(path, module):
    command = ["python", "-m", module.__name__]
    commands = []
    for block in re.findall(
        r"```bash\n(.*?)```", path.read_text(encoding="utf-8"), re.S
    ):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(command)] == command:
                commands.append(module.build_parser().parse_args(words[len(command) :]))
    return commands


def loads(values):
    return ", ".join(f"{value:.2f}" for value in values[:-1]) + (
        f" and {values[-1]:.2f}"
    )


def test_documented_commands_read_the_outputs_of_the_day_74_guides():
    classify, verify = documented(GUIDE, matmul_bottleneck_cli)
    metal, _ = documented(ROOT / "docs/matmul-metal-lab.md", matmul_metal_cli)
    ceiling, _ = documented(ROOT / "docs/gpu-ceilings-lab.md", gpu_ceiling_cli)
    gap, _ = documented(ROOT / "docs/matmul-roofline-gap.md", matmul_roofline_cli)

    assert (classify.command, verify.command) == ("classify", "verify")
    assert classify.matmul_report == verify.matmul_report == metal.output
    assert classify.ceiling_report == verify.ceiling_report == ceiling.output
    assert classify.gap_report == verify.gap_report == gap.output
    assert classify.timing == verify.timing
    assert classify.config == verify.config == Path(CONFIG.relative_to(ROOT))
    assert verify.report == classify.output
    assert f"The parser uses the program name `{matmul_bottleneck_cli.PROG}`." in flat(
        guide()
    )


def test_the_guide_states_plainly_that_the_names_are_inferences():
    text = flat(guide())

    for phrase in (
        "These names are inferences from timings and measured ceilings. The lab "
        "used no hardware counters, no Nsight and no Instruments.",
        "Every Metal number in this guide is a host-specific Apple GPU (Metal) "
        "measurement.",
        "Nothing here runs a CUDA kernel or uses an NVIDIA GPU.",
        "- 0: the command succeeded.",
        "- 1: the command read a report, but the report failed the checks.",
        "- 2: a usage error, an unreadable or invalid input, inputs that do not "
        "pair, or a refused output.",
    ):
        assert phrase in text
    listed = re.findall(r"^- `([a-z -]+)`: ", guide(), flags=re.M)
    assert tuple(listed) == BOTTLENECKS


def test_the_threshold_table_matches_the_pinned_config():
    config = json.loads(CONFIG.read_bytes())
    rows = dict(
        re.findall(r"^\| `([a-z_]+\.[a-z_]+)` \| ([^|]+) \|", guide(), flags=re.M)
    )
    expected = {
        f"{section}.{name}": value
        for section in ("rule", "mapping")
        for name, value in config[section].items()
    }

    assert set(rows) == set(expected)
    for field, value in expected.items():
        shown = rows[field].strip()
        if type(value) is bool:
            assert shown == str(value).lower()
        elif type(value) is str:
            assert shown == f"`{value}`"
        else:
            assert float(shown) == value


def test_the_results_table_quotes_the_recorded_report():
    text = guide()
    for point in RECORDED_REPORT["points"]:
        shape = f"{point['m']} x {point['n']} x {point['k']}"
        for kernel in NAIVE:
            entry = point["kernels"][kernel]
            numbers = entry["decision"]["numbers"]
            row = (
                f"| {shape} | `{entry['mapping']}` | "
                f"{entry['decision']['bottleneck']} | "
                f"{numbers['overhead_share']:.3f} | "
                f"{numbers['fixed_share_bound']:.3f} | "
                f"{numbers['gpu_to_steady_ratio']:.2f} | "
                f"{numbers['lower_roof_share']:.3f} |"
            )
            assert row in text


def test_the_guide_quotes_the_recorded_digests_ceilings_and_load_averages():
    text = flat(guide())
    sources = RECORDED_REPORT["sources"]
    ceilings = RECORDED_REPORT["ceilings"]

    assert f"`{REPORT_SHA256}`" in text
    assert f"`{GAP_SHA256}`" in text
    assert (
        f"The FMA ceiling was {ceilings['fma_gflops_per_second']:.0f} GFLOP/s" in text
    )
    assert (
        f"bandwidth ceiling was {ceilings['bandwidth_gb_per_second']:.1f} GB/s" in text
    )
    before = sources["gpu_ceilings"]["load_average_before"]
    assert f"The ceiling report recorded {loads(before)} before its run." in text
    timing = sources["timing"]["load_average_before"]
    assert f"The timing file recorded {loads(timing)} before its run." in text
    after = sources["timing"]["load_average_after"]
    assert f"before the timing run and {loads(after)} after it." in text
    ratios = [
        p["mapping"]["x_columns_to_x_rows_rate_ratio"]
        for p in RECORDED_REPORT["points"]
    ]
    assert (
        f"`x_columns` ran {min(ratios):.2f} to {max(ratios):.2f} times as fast" in text
    )
    for kernel in NAIVE:
        top = RECORDED_REPORT["summary"]["top_bottleneck"][kernel]
        assert (top["top"], top["top_points"]) == ("latency or occupancy-bound", 6)
    assert "was `latency or occupancy-bound`, at 6 of 9 shapes." in text
