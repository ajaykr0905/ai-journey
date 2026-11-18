import dataclasses
import json
import re
import shlex
from pathlib import Path

from ai_journey import (
    gpu_ceiling_cli,
    gpu_timing_cli,
    matmul_bottleneck_cli,
    matmul_metal_cli,
    matmul_roofline_cli,
)
from ai_journey.matmul_bottleneck_config import load_config
from ai_journey.matmul_bottleneck_rule import BOTTLENECKS, PointInputs, classify
from test_matmul_bottleneck_recorded import GAP_SHA256, RECORDED_REPORT, REPORT_SHA256

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-bottleneck-lab.md"
CONFIG = ROOT / "config/day-75-matmul-bottleneck.json"
NAIVE = ("matmul_x_columns", "matmul_x_rows")
CONFIG_RULE = json.loads(CONFIG.read_bytes())["rule"]
CONFIG_PINNED = load_config(CONFIG)


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
    classify, verify, _ = documented(GUIDE, matmul_bottleneck_cli)
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


# Median one-call host times in ns of the event check run on 2026-11-18 at
# 08:54 IST, with and without the torch.mps.Event pair, 21 repeats each.
EVENT_CHECK = {
    (128, "matmul_x_columns"): (343250, 299083),
    (128, "matmul_x_rows"): (484167, 377833),
    (256, "matmul_x_columns"): (797416, 714000),
    (256, "matmul_x_rows"): (1606083, 1479125),
    (512, "matmul_x_columns"): (1849500, 1638708),
    (512, "matmul_x_rows"): (2510333, 2472667),
}


def test_the_guide_quotes_the_event_check_run_against_the_recorded_steady_times():
    text = flat(guide())
    steady = {
        (point["m"], kernel): point["kernels"][kernel]["decision"]["numbers"][
            "steady_ns_per_call"
        ]
        for point in RECORDED_REPORT["points"]
        if point["m"] == point["n"] == point["k"]
        for kernel in NAIVE
    }
    added = [(with_ - without) / 1000 for with_, without in EVENT_CHECK.values()]
    share = {
        key: (1 - steady[key] / with_, 1 - steady[key] / without)
        for key, (with_, without) in EVENT_CHECK.items()
    }
    small = [share[key][1] for key in share if key[0] < 512]
    rows = RECORDED_REPORT["points"][2]["kernels"]["matmul_x_rows"]

    assert "The `torch.mps.Event` pair does not read a GPU clock." in text
    assert (
        f"the event pair added {min(added):.0f} to {max(added):.0f} microseconds"
        in text
    )
    assert (
        f"the overhead shares without the event pair were at least {min(small):.2f}"
        in text
    )
    assert f"the `x_columns` share was {share[512, 'matmul_x_columns'][1]:.2f}." in text
    with_, without = share[512, "matmul_x_rows"]
    assert (
        f"one `x_rows` call took {EVENT_CHECK[512, 'matmul_x_rows'][0] / 1e6:.2f} ms, "
        f"against {rows['inputs']['median_host_ns'] / 1e6:.2f} ms in the recorded "
        f"file. Its overhead share was {with_:.2f} with the event pair and "
        f"{without:.2f} without it." in text
    )
    assert without < CONFIG_RULE["dispatch_overhead_share_min"] <= min(small)


def recorded_results(rule=None, slower_ratio_min=None):
    """Classify the recorded points again, with one threshold replaced."""

    rule = rule or CONFIG_PINNED.rule
    slower = slower_ratio_min or CONFIG_PINNED.mapping.slower_ratio_min
    bound = {
        kernel: call["median_host_ns"]
        for kernel, call in RECORDED_REPORT["sources"]["timing"][
            "fastest_calls"
        ].items()
    }
    names, verdicts = [], []
    for point in RECORDED_REPORT["points"]:
        for kernel in NAIVE:
            numbers = point["kernels"][kernel]["decision"]["numbers"]
            inputs = PointInputs(
                steady_ns_per_call=numbers["steady_ns_per_call"],
                rate_gflops_per_second=numbers["rate_gflops_per_second"],
                gpu_ns=int(numbers["gpu_ns"]),
                host_ns=int(numbers["host_ns"]),
                dispatch_ns=int(numbers["dispatch_ns"]),
                fma_ceiling_gflops_per_second=numbers["fma_ceiling_gflops_per_second"],
                fma_scales_with_work=RECORDED_REPORT["ceilings"][
                    "fma_scales_with_work"
                ],
                bandwidth_roof_gflops_per_second=numbers[
                    "bandwidth_roof_gflops_per_second"
                ],
                fixed_cost_bound_ns=bound[kernel],
            )
            names.append(classify(inputs, rule).bottleneck)
        mapping = point["mapping"]
        verdicts.append(mapping["x_columns_to_x_rows_rate_ratio"] >= slower)
    return names, verdicts


def test_the_recorded_results_keep_their_names_over_each_documented_range():
    recorded = recorded_results()
    rows = dict(
        re.findall(
            r"^\| `([a-z_]+\.[a-z_]+)` \| [0-9.]+ \| ([0-9.]+ to [0-9.]+) \|$",
            guide(),
            flags=re.M,
        )
    )
    numeric = [
        "dispatch_overhead_share_min",
        "timing_tolerance",
        "gpu_agreement_max_ratio",
        "roof_excess_tolerance",
        "near_roof_share_min",
    ]

    assert recorded[0] == [
        point["kernels"][kernel]["decision"]["bottleneck"]
        for point in RECORDED_REPORT["points"]
        for kernel in NAIVE
    ]
    assert all(recorded[1])
    assert set(rows) == {f"rule.{name}" for name in numeric} | {
        "mapping.slower_ratio_min"
    }
    for name in numeric:
        low, high = (float(value) for value in rows[f"rule.{name}"].split(" to "))
        for value in (low, high):
            rule = dataclasses.replace(CONFIG_PINNED.rule, **{name: value})
            assert recorded_results(rule) == recorded, (name, value)
    low, high = (
        float(value) for value in rows["mapping.slower_ratio_min"].split(" to ")
    )
    assert recorded_results(slower_ratio_min=low) == recorded
    assert recorded_results(slower_ratio_min=high) == recorded
    assert recorded_results(slower_ratio_min=high + 0.001) != recorded
    # The top of the dispatch range is a real edge: one step past it, a name changes.
    past = dataclasses.replace(CONFIG_PINNED.rule, dispatch_overhead_share_min=0.52)
    names = recorded_results(past)[0]
    changed = [
        (index, before, after)
        for index, (before, after) in enumerate(zip(recorded[0], names))
        if before != after
    ]
    assert changed == [(4, "dispatch-bound", "indeterminate")]
    assert "A dispatch threshold of 0.52 changes its name to `indeterminate`" in flat(
        guide()
    )


# Digests of the joined run on 2026-11-18 from 09:14:56 IST, under the GPU lock.
JOINED_DIGESTS = {
    "gpu-timing": "45786bd734735ac4b6164b8bdf95e82031784acfdee89cc8774fb0cd8cb11547",
    "gpu-ceilings": "677ec40de39f20f50fc8964e3bbd3972ef84698ed7cd2228fce6adb9d73ca0d6",
    "matmul-metal, cubes": (
        "08b837a9cd8369020351f69110ff67aced937977bb6412aa214b297da4341eb0"
    ),
    "roofline gap, cubes": (
        "a39f31eccedaa4930fa9022c02653c76af9ad6e2cb284d9b19ce6ac4f672368d"
    ),
    "joined bottleneck": (
        "e1a9af3b8558c3d924369fb57c8c9da9cc13ad0a364d4719748cab400d76e8ae"
    ),
}


def test_the_joined_commands_chain_the_gpu_timing_and_cube_reports():
    *_, joined = documented(GUIDE, matmul_bottleneck_cli)
    (cubes,) = documented(GUIDE, matmul_metal_cli)
    (gap,) = documented(GUIDE, matmul_roofline_cli)
    timing, _ = documented(ROOT / "docs/gpu-timing-lab.md", gpu_timing_cli)
    ceiling, _ = documented(ROOT / "docs/gpu-ceilings-lab.md", gpu_ceiling_cli)
    cube_config = json.loads((ROOT / cubes.config).read_bytes())
    day_74 = json.loads((ROOT / "config/day-74-matmul-metal.json").read_bytes())

    assert joined.timing == timing.output
    assert joined.matmul_report == gap.matmul_report == cubes.output
    assert joined.ceiling_report == gap.ceiling_report == ceiling.output
    assert joined.gap_report == gap.output
    assert joined.config == Path(CONFIG.relative_to(ROOT))
    timed = json.loads((ROOT / timing.config).read_bytes())["sweep"]["shapes"]
    assert cube_config["benchmark"]["shapes"] == [
        shape for shape in day_74["benchmark"]["shapes"] if shape in timed
    ]
    assert len(cube_config["benchmark"]["shapes"]) == 6
    assert {**cube_config, "name": None, "benchmark": None} == {
        **day_74,
        "name": None,
        "benchmark": None,
    }


def test_the_guide_quotes_the_joined_run_digests_and_labels_it():
    text = flat(guide())

    for name, digest in JOINED_DIGESTS.items():
        assert f"| {name} | `{digest}` |" in text
    assert "All numbers are host-specific Apple GPU (Metal) measurements" in text
    assert "8.90, 8.80 and 10.84 before the ceiling run" in text
    assert "The top name of each kernel was `latency or occupancy-bound`, at 4" in text
