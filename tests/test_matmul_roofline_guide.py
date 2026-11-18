import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, matmul_metal_cli, matmul_roofline_cli
from ai_journey.matmul_roofline_gap import DEFINITIONS, MEASUREMENT_BOUNDARY
from test_matmul_roofline_recorded import (
    CEILING_SHA256,
    GAP_SHA256,
    MATMUL_SHA256,
    RECORDED_GAP,
)

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-roofline-gap.md"
METAL_GUIDE = ROOT / "docs/matmul-metal-lab.md"
CEILING_GUIDE = ROOT / "docs/gpu-ceilings-lab.md"
NAIVE = ("matmul_x_columns", "matmul_x_rows")


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands(text, module):
    command = ["python", "-m", module]
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(command)] == command:
                commands.append(words[len(command) :])
    return commands


def loads(values):
    return ", ".join(f"{value:.2f}" for value in values[:-1]) + (
        f" and {values[-1]:.2f}"
    )


def test_documented_commands_read_the_outputs_of_the_two_metal_guides():
    commands = documented_commands(guide(), "ai_journey.matmul_roofline_cli")
    parser = matmul_roofline_cli.build_parser()
    gap, verify = (parser.parse_args(argv) for argv in commands)
    (metal_run, _) = (
        matmul_metal_cli.build_parser().parse_args(argv)
        for argv in documented_commands(
            METAL_GUIDE.read_text(encoding="utf-8"), "ai_journey.matmul_metal_cli"
        )
    )
    (ceiling_run, _) = (
        gpu_ceiling_cli.build_parser().parse_args(argv)
        for argv in documented_commands(
            CEILING_GUIDE.read_text(encoding="utf-8"), "ai_journey.gpu_ceiling_cli"
        )
    )

    assert [argv[0] for argv in commands] == ["gap", "verify"]
    assert gap.matmul_report == verify.matmul_report == metal_run.output
    assert gap.ceiling_report == verify.ceiling_report == ceiling_run.output
    assert verify.report == gap.output
    assert f"The parser uses the program name `{parser.prog}`." in flat(guide())


def test_guide_defines_peak_and_states_the_exit_codes_and_requirements():
    text = flat(guide())

    for phrase in (
        'In this guide, "peak" means a ceiling that the gpu-ceilings lab measured '
        "on this host. The command uses no vendor number.",
        "- 0: the command succeeded.",
        "- 1: the command read a report, but the report failed the checks. A source "
        "report that fails its own verifier also gives code 1.",
        "- 2: a usage error, an unreadable or invalid input, two source reports that "
        "do not pair, or a refused output.",
        "the summary line says `sources_checked=true`",
        "the line says `sources_checked=false`",
        "The `verify` command takes both source reports or neither.",
        "It does not import torch and needs no GPU, so it runs on any host with NumPy.",
        "The command refuses such a pair with exit code 2.",
        "this check cannot prove that both reports came from one host.",
        "The matmul-metal report does not record load averages.",
    ):
        assert phrase in text, phrase
    assert matmul_roofline_cli.EXIT_REJECTED == 1
    assert matmul_roofline_cli.EXIT_INVALID == 2


def test_guide_states_the_formulas_that_the_report_defines():
    text = flat(guide())

    for phrase in (
        "The bytes are `4 * (2 * M * N * K + M * N)`.",
        "The intensity is `K / (4 * K + 2)` FLOPs per byte, which is less than 1/4.",
        "The bytes are `4 * (M * K + K * N + M * N)`. No kernel moves fewer bytes.",
        "The FMA share is the median rate divided by the FMA ceiling.",
        "The bandwidth roof is the bandwidth ceiling times the intensity, in GFLOP/s.",
        "The roof share is the median rate divided by the bandwidth roof.",
        "The bound is `bandwidth` when the bandwidth roof is less than the FMA "
        "ceiling. Else the bound is `compute`.",
        "One call does `2 * M * N * K` FLOPs.",
    ):
        assert phrase in text, phrase
    assert "4 * (2 * M * N * K + M * N)" in DEFINITIONS["no_reuse"]
    assert "K / (4 * K + 2)" in DEFINITIONS["no_reuse"]
    assert "4 * (M * K + K * N + M * N)" in DEFINITIONS["ideal_reuse"]


def test_the_tables_hold_the_recorded_gap():
    lines = guide().splitlines()

    for point in RECORDED_GAP["points"]:
        shape = f"{point['m']} x {point['n']} x {point['k']}"
        cells = []
        for model, places in (("no_reuse", 4), ("ideal_reuse", 2)):
            entry = point["intensities"][model]
            ratio = entry["flops_per_byte"]
            roof = entry["bandwidth_roof_gflops_per_second"]
            roof_text = f"{roof:.1f}" if model == "no_reuse" else f"{roof:.0f}"
            cells += [
                f"{ratio['numerator'] / ratio['denominator']:.{places}f}",
                f"{roof_text} GFLOP/s",
            ]
        torch = point["sides"]["torch_matmul"]["fma_ceiling_share"]
        bound = point["intensities"]["ideal_reuse"]["bound"]
        assert f"| {shape} | {' | '.join(cells)} | {bound} | {torch:.3f} |" in lines
        for side in NAIVE:
            entry = point["sides"][side]
            assert (
                f"| {shape} | `{entry['mapping']}` | "
                f"{entry['median_gflops_per_second']:.1f} | "
                f"{entry['fma_ceiling_share']:.3f} | "
                f"{entry['no_reuse_roof_share']:.2f} | "
                f"{entry['ideal_reuse_roof_share']:.4f} | "
                f"{entry['torch_ratio']:.3f} |"
            ) in lines, (shape, side)


def test_the_results_and_the_ceilings_are_those_of_the_recorded_gap():
    text = flat(guide())
    summary = RECORDED_GAP["summary"]
    ceilings = RECORDED_GAP["ceilings"]
    best = summary["best_fma_ceiling_share"]
    largest = RECORDED_GAP["points"][best["matmul_x_rows"]["index"]]
    no_reuse = summary["no_reuse_roof_share_range"]
    ideal = summary["ideal_reuse_roof_share_range"]

    assert best["torch_matmul"]["index"] == best["matmul_x_rows"]["index"]
    for phrase in (
        f"The FMA ceiling was {ceilings['fma_gflops_per_second']:.0f} GFLOP/s, and "
        f"the bandwidth ceiling was {ceilings['bandwidth_gb_per_second']:.1f} GB/s.",
        f"The ridge point was {ceilings['ridge_flops_per_byte']:.1f} FLOPs per byte.",
        "The highest FMA share of `matmul_x_columns` was "
        f"{best['matmul_x_columns']['fma_ceiling_share']:.3f}.",
        "The highest FMA share of `matmul_x_rows` was "
        f"{best['matmul_x_rows']['fma_ceiling_share']:.3f}, at "
        f"{largest['m']} x {largest['n']} x {largest['k']}.",
        f"`torch.matmul` reached {best['torch_matmul']['fma_ceiling_share']:.3f} of "
        "the FMA ceiling at 4096 x 4096 x 4096.",
        f"At all {summary['points']} shapes, both naive kernels ran above the "
        "no-reuse roof.",
        f"`matmul_x_columns` ran {no_reuse['matmul_x_columns'][0]:.2f} to "
        f"{no_reuse['matmul_x_columns'][1]:.2f} times above it, and `matmul_x_rows` "
        f"ran {no_reuse['matmul_x_rows'][0]:.2f} to "
        f"{no_reuse['matmul_x_rows'][1]:.2f} times above it.",
        f"At ideal reuse, `matmul_x_columns` used {ideal['matmul_x_columns'][0]:.4f} "
        f"to {ideal['matmul_x_columns'][1]:.4f} of the bandwidth roof, and "
        f"`matmul_x_rows` used {ideal['matmul_x_rows'][0]:.4f} to "
        f"{ideal['matmul_x_rows'][1]:.4f} of it.",
        f"Its digest is `{GAP_SHA256}`.",
        f"The matmul-metal report has the digest `{MATMUL_SHA256}`.",
        f"The ceiling report has the digest `{CEILING_SHA256}`.",
    ):
        assert phrase in text, phrase
    assert summary["points_above_no_reuse_roof"] == {side: 9 for side in NAIVE}
    bounds = [
        point["intensities"]["ideal_reuse"]["bound"] for point in RECORDED_GAP["points"]
    ]
    assert bounds == ["bandwidth", "bandwidth"] + ["compute"] * 7
    assert [RECORDED_GAP["points"][index]["m"] for index in (0, 1, 2)] == [
        128,
        256,
        512,
    ]


def test_the_load_averages_and_rates_match_the_two_metal_guides():
    text = flat(guide())
    source = RECORDED_GAP["sources"]["gpu_ceilings"]
    before, after = source["load_average_before"], source["load_average_after"]
    ceiling_guide = CEILING_GUIDE.read_text(encoding="utf-8")
    metal_guide = METAL_GUIDE.read_text(encoding="utf-8")
    ceilings = RECORDED_GAP["ceilings"]

    assert (
        f"The ceiling report recorded {loads(before)} before and {loads(after)} "
        "after."
    ) in text
    assert (
        f"| 1 | {', '.join(f'{value:.2f}' for value in before)} | "
        f"{', '.join(f'{value:.2f}' for value in after)} | "
        f"{ceilings['fma_gflops_per_second']:.0f} GFLOP/s | "
        f"{ceilings['bandwidth_gb_per_second']:.1f} GB/s | "
        f"{ceilings['ridge_flops_per_byte']:.1f} FLOPs per byte |"
    ) in ceiling_guide.splitlines()
    for phrase in (
        "The load averages were 7.98, 9.06 and 8.98 before the run. They were 8.24, "
        "8.97 and 8.95 after the run.",
    ):
        assert phrase in flat(metal_guide)
    assert (
        "The `uptime` load averages were 7.98, 9.06 and 8.98 before the matmul-metal "
        "run and 8.24, 8.97 and 8.95 after it."
    ) in text
    rows = metal_guide.splitlines()
    for point in RECORDED_GAP["points"]:
        rates = [
            f"{point['sides'][side]['median_gflops_per_second']:.0f} ("
            for side in (*NAIVE, "torch_matmul")
        ]
        prefix = f"| {point['m']} x {point['n']} x {point['k']} | "
        (row,) = [line for line in rows if line.startswith(prefix)]
        assert [cell.split("(")[0] + "(" for cell in row.split(" | ")[1:4]] == rates


def test_guide_quotes_the_measurement_boundary_and_its_exclusions():
    text = flat(guide())

    assert (
        f'The boundary of a measured gap report states: "'
        f'{MEASUREMENT_BOUNDARY["statement"]}"'
    ) in text
    for phrase in (
        "The gap command times nothing and reads no hardware counters.",
        "A ceiling is a rate of the ceiling lab's own kernels, so a kernel can go "
        "above it.",
        "It is not evidence of CUDA or NVIDIA performance",
        "The `verify` command with both source reports finds the change, because it "
        "builds the gap again from the verified source reports.",
    ):
        assert phrase in text, phrase
    for marker in ("/Users/", "/home/", "/private/"):
        assert marker not in guide()
