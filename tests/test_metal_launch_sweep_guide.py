import itertools
import json
import re
import shlex
from functools import partial
from pathlib import Path

from ai_journey import metal_launch_sweep_cli
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from ai_journey.memory_access_benchmark import summarize_timings
from ai_journey.metal_launch_sweep_benchmark import time_shapes
from ai_journey.metal_launch_sweep_config import (
    MAX_RUN_BYTES,
    MAX_RUN_CALLS,
    MAX_RUN_FLOPS,
    load_config,
)
from ai_journey.metal_launch_sweep_correctness import BOUND, LATTICE_POINTS
from ai_journey.metal_launch_sweep_mapping import MAPPING_CASE
from ai_journey.metal_launch_sweep_report import (
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
)
from ai_journey.metal_launch_sweep_runtime import MetalLaunchSweepRuntime, SweepWorkload
from ai_journey.metal_launch_sweep_shapes import REJECTION_REASONS, simd_footprint
from test_metal_launch_sweep_recorded import (
    RECORDED_LOAD_AVERAGES,
    recorded_report,
)

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/metal-launch-sweep.md"
COMMAND = ["python", "-m", "ai_journey.metal_launch_sweep_cli"]
PINNED = "config/day-74-metal-launch-sweep.json"


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def section(title):
    text = guide()
    start = text.index(f"\n## {title}\n")
    following = text.find("\n## ", start + 1)
    return text[start : following if following != -1 else None]


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(COMMAND)] == COMMAND:
                commands.append(words[len(COMMAND) :])
    return commands


def percent(ratio):
    return f"{abs(ratio - 1) * 100:.1f}%"


def shapes_of(point):
    return {"x".join(map(str, shape["group_size"])): shape for shape in point["shapes"]}


def test_documented_commands_parse_and_operate_the_pinned_config():
    commands = documented_commands(guide())
    parser = metal_launch_sweep_cli.build_parser()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["run", "verify"]
    assert {str(arguments.config) for arguments in parsed} == {PINNED}
    assert parsed[1].report == parsed[0].output
    assert (ROOT / PINNED).is_file()
    assert f"The parser names itself `{parser.prog}`." in flat(guide())


def test_guide_names_the_exit_codes_reasons_prefixes_and_rules():
    text = flat(guide())

    for phrase in (
        "0: the command succeeded.",
        "1: the command read a report, but the report failed the checks.",
        "2: a usage error",
        "3: this host cannot run Metal kernels through PyTorch",
    ):
        assert phrase in text, phrase
    assert (
        metal_launch_sweep_cli.EXIT_OK,
        metal_launch_sweep_cli.EXIT_REJECTED,
        metal_launch_sweep_cli.EXIT_INVALID,
        metal_launch_sweep_cli.EXIT_UNAVAILABLE,
    ) == (0, 1, 2, 3)
    for reason in UNAVAILABLE_REASONS:
        assert f"`{reason}`" in text, reason
    for prefix in (
        "`apple_gpu_metal_measurement`",
        "`metal_launch_sweep_measurement=`",
        "`synthetic_timing`",
        "`metal_launch_sweep_synthetic_timing=`",
        "`verified_`",
    ):
        assert prefix in text, prefix
    rules = flat(section("Pipeline limits and the launch-shape rules"))
    positions = [rules.index(f"`{reason}`") for reason in REJECTION_REASONS]
    assert positions == sorted(positions)
    assert MetalLaunchSweepRuntime.requires_whole_simd_groups is False
    assert "the report records `requires_whole_simd_groups` as false" in rules


def test_guide_quotes_the_boundary_and_quotes_no_vendor_peak():
    text = flat(guide())

    assert flat(MEASUREMENT_BOUNDARY["statement"]) in text
    assert MEASUREMENT_LABEL in text
    assert "Nothing here runs a CUDA kernel or uses an NVIDIA GPU." in text
    assert "The lab quotes no vendor peak." in text
    for match in re.finditer(r"peaks?\b", text):
        assert text[match.start() - 7 : match.start()] == "vendor ", match
    assert flat(f"`{BOUND}`") in text
    assert "it checks the timings for structure only" in text


def test_every_interpretation_is_marked_as_a_hypothesis():
    analysis = section("Shape, size and the execution width")
    claims, hypotheses = analysis.split("The following statements are hypotheses.")
    bullets = re.findall(r"^- (.*)$", hypotheses, flags=re.MULTILINE)

    assert len(bullets) == 4
    assert all(bullet.startswith("Hypothesis: ") for bullet in bullets)
    # The measured part relates data to the width but names no hidden cause.
    for word in ("cache", "schedul", "memory system", "fixed cost"):
        assert word not in claims, word


def test_guide_controls_checks_and_planned_work_match_the_pinned_config():
    text = guide()
    pinned = json.loads((ROOT / PINNED).read_text(encoding="utf-8"))
    config = load_config(ROOT / PINNED)

    assert f"| `seed` | {pinned['seed']} |" in text
    for field in ("target_flops_per_sample",):
        assert f"| `matmul.{field}` | {pinned['matmul'][field]} |" in text
    vector = pinned["vector"]
    for field in ("correctness_length", "target_bytes_per_sample"):
        assert f"| `vector.{field}` | {vector[field]} |" in text, field
    for field in ("warmup", "repeats", "max_runtime_seconds"):
        assert f"| `timing.{field}` | {pinned['timing'][field]} |" in text, field
    sizes = ", ".join(str(size) for size in vector["group_sizes"][:-1])
    assert f"| `vector.group_sizes` | {sizes} and {vector['group_sizes'][-1]} |" in text
    work = config.planned_work()
    assert flat(
        f"the checks are {work.mapping_checks} mapping checks, {work.layout_probes} "
        f"layout probes and {work.correctness_launches} correctness launches. The "
        f"sweep has {work.benchmark_points} points, {work.timed_samples} timed "
        f"samples, {work.calls} kernel calls, {work.matmul_flops} matmul FLOPs and "
        f"{work.vector_bytes} vector bytes."
    ) in flat(text)
    bounds = [limit.bit_length() - 1 for limit in (MAX_RUN_CALLS, MAX_RUN_FLOPS)]
    data = MAX_RUN_BYTES.bit_length() - 1
    assert flat(
        f"A run can have at most 2^{bounds[0]} calls, 2^{bounds[1]} matmul FLOPs "
        f"and 2^{data} vector bytes."
    ) in flat(text)
    m, n, k = MAPPING_CASE
    assert f"The case has {m} rows, {n} columns and an inner size of {k}." in flat(text)
    assert f"each group size at {vector['correctness_length']} elements" in text
    assert f"A lattice of up to {LATTICE_POINTS**2} values" in flat(text)
    assert pinned["timing"]["repeats"] % len(pinned["matmul"]["group_shapes"]) == 0


def test_the_footprint_table_follows_the_width_that_the_run_measured():
    text = guide()
    report = recorded_report()
    width = report["pipelines"]["matmul_naive"]["thread_execution_width"]

    for group in load_config(ROOT / PINNED).matmul.group_shapes:
        footprint = simd_footprint(group, width)
        row = (
            f"| {group[0]}x{group[1]} | {group[0] * group[1]} | "
            f"{footprint.simd_groups} | {footprint.rows} | {footprint.columns} |"
        )
        assert row in text, row


def test_the_measured_tables_and_figures_come_from_the_recorded_report():
    text = guide()
    words = flat(text)
    report = recorded_report()

    for point in report["benchmark"]["matmul"]:
        heading = f"### Matmul at {point['m']} x {point['n']} x {point['k']}"
        assert heading in text
        calls = point["calls_per_sample"]
        held = f"{calls} call" + ("s" if calls > 1 else "")
        assert f"Each sample held {held}. The spread was {point['spread']:.3f}" in words
        for label, shape in shapes_of(point).items():
            row = (
                f"| {label} | {shape['threads_per_group']} | "
                f"{shape['rows_per_simd_group']} | "
                f"{shape['median_ns_per_call'] / 1000:.2f} | "
                f"{shape['iqr_ns_per_call'] / 1000:.2f} | "
                f"{shape['median_gflops_per_second']:.1f} | {shape['rank']} |"
            )
            assert row in text, row
        assert point["separated"] is True
    for point in report["benchmark"]["vector"]:
        assert f"### Vector add at {point['length']} elements" in text
        for shape in point["shapes"]:
            row = (
                f"| {shape['group_size'][0]} | {shape['idle_lanes']} | "
                f"{shape['median_ns_per_call'] / 1000:.2f} | "
                f"{shape['iqr_ns_per_call'] / 1000:.2f} | "
                f"{shape['median_gb_per_second']:.2f} | {shape['rank']} |"
            )
            assert row in text, row
        assert point["separated"] is False
    before, after = RECORDED_LOAD_AVERAGES
    assert (
        flat(
            f"load averages of {before[0]}, {before[1]} and {before[2]} before the run "
            f"and {after[0]}, {after[1]} and {after[2]} after it"
        )
        in words
    )
    assert f"`{report['report_sha256']}`" in text
    assert f"The run took {report['run']['elapsed_ns'] / 1e9:.1f} seconds" in words
    summary = report["summary"]
    assert (
        f"{summary['best_matmul_gflops_per_second']:.1f} GFLOP/s for the matmul and "
        f"{summary['best_vector_gb_per_second']:.2f} GB/s for the vector add"
    ) in words
    cases = max(case["max_bound_ratio"] for case in report["correctness"]["matmul"])
    points = max(
        point["accuracy"]["max_bound_ratio"] for point in report["benchmark"]["matmul"]
    )
    assert f"correctness case was {cases:.4f} of its bound" in words
    assert f"timed point was {points:.4f} of its bound" in words


def test_the_width_section_states_only_what_the_recorded_report_shows():
    words = flat(section("Shape, size and the execution width"))
    report = recorded_report()
    small, middle, large = (shapes_of(point) for point in report["benchmark"]["matmul"])
    five = ("16x16", "32x8", "8x32", "64x4", "4x64")

    def spread(shapes):
        medians = [shapes[label]["timing"]["median_ns"] for label in five]
        return max(medians) / min(medians)

    assert (
        f"their rates were within {percent(spread(middle))} and "
        f"{percent(spread(large))} of each other" in words
    )
    below = (
        small["4x64"]["median_gflops_per_second"]
        / small["16x16"]["median_gflops_per_second"]
    )
    assert f"8 rows had the lowest rate of the five, {percent(below)} below" in words
    for shapes in (small, middle, large):
        fastest = min(five, key=lambda label: shapes[label]["timing"]["median_ns"])
        assert shapes[fastest]["rows_per_simd_group"] > 1
    ranks = [shapes["32x32"]["rank"] for shapes in (small, middle, large)]
    assert ranks == [1, 7, 7]
    slower = [
        shapes["32x32"]["timing"]["median_ns"] / shapes["8x8"]["timing"]["median_ns"]
        for shapes in (middle, large)
    ]
    assert f"was {percent(slower[0])} and {percent(slower[1])} slower than 8x8" in words
    short, long = report["benchmark"]["vector"]
    assert f"All nine sizes were within {percent(long['spread'])}" in words
    best = min(shape["timing"]["median_ns"] for shape in long["shapes"])
    for shape in long["shapes"][:2]:
        assert shape["timing"]["median_ns"] / best - 1 < 0.016
    lowest = sorted(short["shapes"], key=lambda shape: shape["rank"])[-3:]
    assert sorted(shape["group_size"][0] for shape in lowest) == [8, 16, 32]
    assert "the sizes 8, 16 and 32 gave the three lowest rates" in words


def first_turns(shapes, repeats):
    """Count the repeats in which each shape runs first, from the timing loop."""

    log = []
    workload = SweepWorkload(
        calls=tuple(partial(log.append, position) for position in range(shapes)),
        synchronize=lambda: None,
        outputs_agree=lambda: True,
        output=lambda: None,
    )
    time_shapes(
        workload,
        calls_per_sample=1,
        warmup=0,
        repeats=repeats,
        clock=itertools.count().__next__,
    )
    starts = log[::shapes]
    return [starts.count(position) for position in range(shapes)]


def median_ns(samples):
    return summarize_timings(tuple(samples)).median_ns


def test_the_guide_states_how_often_each_shape_runs_first():
    config = load_config(ROOT / PINNED)
    repeats = config.timing.repeats
    shapes, sizes = config.matmul.group_shapes, config.vector.group_sizes
    text = flat(guide())

    # The turns rotate, but only whole cycles give every shape each turn.
    assert set(first_turns(len(shapes), repeats)) == {repeats // len(shapes)}
    turns = first_turns(len(sizes), repeats)
    more = [size for size, count in zip(sizes, turns) if count > min(turns)]
    assert (more, max(turns), min(turns)) == ([8, 16, 32], 3, 2)
    assert "When the repeat count is a multiple of the number of shapes" in text
    assert (
        f"It is not a multiple of the {len(sizes)} vector sizes. Thus the sizes 8, "
        f"16 and 32 run first in {max(turns)} repeats, and each other size runs "
        f"first in {min(turns)} repeats."
    ) in text


def test_the_extra_first_turns_leave_the_lowest_vector_rates_unchanged():
    words = flat(section("Shape, size and the execution width"))
    short = recorded_report()["benchmark"]["vector"][0]
    sizes, calls = len(short["shapes"]), short["calls_per_sample"]
    by_turn = [[] for _ in range(sizes)]
    for position, shape in enumerate(short["shapes"]):
        for repeat, sample in enumerate(shape["samples_ns"]):
            by_turn[(position - repeat) % sizes].append(sample)
    per_call = [median_ns(samples) / calls / 1000 for samples in by_turn]
    whole = short["repeats"] - short["repeats"] % sizes

    assert f"a repeat had a median of {per_call[0]:.1f} µs for each call" in words
    assert (
        f"other turns had medians of {min(per_call[1:]):.1f} µs to "
        f"{max(per_call[1:]):.1f} µs"
    ) in words
    # All the repeats, then only the whole cycles, where every size runs first
    # equally often, rank the same three sizes lowest.
    for cut in (None, whole):
        ranked = sorted(
            short["shapes"], key=lambda shape: median_ns(shape["samples_ns"][:cut])
        )
        assert {shape["group_size"][0] for shape in ranked[-3:]} == {8, 16, 32}
    assert (
        f"In the first {whole} repeats, each size ran first {whole // sizes} times, "
        "and the sizes 8, 16 and 32 still gave the three lowest rates."
    ) in words


def iqrs_overlap(first, second):
    """Whether the interquartile ranges of two timed shapes overlap."""

    one, two = first["timing"], second["timing"]
    return one["p25_ns"] <= two["p75_ns"] and two["p25_ns"] <= one["p75_ns"]


def test_the_width_section_hedges_ranks_whose_quartiles_overlap():
    words = flat(section("Shape, size and the execution width"))
    report = recorded_report()
    five = ("16x16", "32x8", "8x32", "64x4", "4x64")

    for point in report["benchmark"]["matmul"]:
        shapes = shapes_of(point)
        fastest = min(five, key=lambda label: shapes[label]["timing"]["median_ns"])
        one_row = [label for label in five if shapes[label]["rows_per_simd_group"] == 1]
        assert any(iqrs_overlap(shapes[fastest], shapes[label]) for label in one_row)
    assert (
        "But at each size, the fastest of the five and a shape with 1 row had "
        "overlapping IQRs, so this run does not separate them."
    ) in words
    middle = shapes_of(report["benchmark"]["matmul"][1])
    assert middle["32x32"]["rank"] == 7
    assert iqrs_overlap(middle["32x32"], middle["4x64"])
    assert "At 1024, its IQR overlapped the IQR of 4x64." in words
    # The separated mark covers only the extreme pair of a point.
    assert "The mark compares only those two shapes." in flat(section("Timing method"))
    # Quartiles and turn medians are not in the tables, so the section names
    # the report's samples and quartiles as a source too.
    assert (
        "comes from the recorded report of the run. The source is the tables "
        "above, or the raw samples and quartiles in the report."
    ) in words
