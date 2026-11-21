"""Tests that every Metal rate is a copy of a recorded rate, and that no GPU rate exists.

The Day 76 and Day 77 guides hold the tables of the two recorded runs. Each
cited rate must stand in the row of its kernel and shape, and each report
digest and load average must stand in its guide.
"""

import json
from pathlib import Path

import pytest

from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.triton_matmul_kernel import AUTOTUNE_CONFIGS, TritonMatmulError
from ai_journey.triton_matmul_performance import (
    CUBLAS,
    CUDA_TEXT,
    INTERPRETER_TIME,
    LATER_GPU_RUN,
    METAL_RATES,
    METAL_REPORTS,
    PERFORMANCE_SHAPES,
    TRITON_GPU,
    MetalRate,
    performance_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {"torch_matmul": "torch.matmul"}


def guide(lab):
    (report,) = [item for item in METAL_REPORTS if item.lab == lab]
    return (ROOT / report.guide).read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def label(shape):
    return f"{shape.m} x {shape.n} x {shape.k}"


def cell(rate):
    return f"{rate.median:.1f} ({rate.low:.1f} to {rate.high:.1f})"


def cells(row):
    return [part.strip() for part in row.strip().strip("|").split("|")]


def day_76_row(shape):
    rows = [
        cells(line)
        for line in guide("day-76").splitlines()
        if line.startswith(f"| {label(shape)} |")
    ]
    (row,) = [row for row in rows if row[-1] == "not measured"]
    return row


def day_77_rows(shape):
    section = guide("day-77").split(f"\n### {label(shape)}\n", 1)[1]
    table = section.split("\n### ", 1)[0]
    return [cells(line) for line in table.splitlines() if line.startswith("| ")]


def test_the_shapes_are_the_benchmark_shapes_that_both_runs_timed():
    def shapes(name):
        config = json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
        return [MatmulShape(*triple) for triple in config["benchmark"]["shapes"]]

    assert list(PERFORMANCE_SHAPES) == shapes("day-77-matmul-checklist.json")
    assert set(PERFORMANCE_SHAPES) <= set(shapes("day-76-matmul-tiled.json"))


def test_each_day_76_rate_is_the_rate_of_its_recorded_table():
    rates = [rate for rate in METAL_RATES if rate.lab == "day-76"]

    assert len(rates) == 2 * len(PERFORMANCE_SHAPES)
    for rate in rates:
        _, _, tiled, vendor, cublas = day_76_row(rate.shape)
        column = vendor if rate.kernel == "torch_matmul" else tiled
        assert column == cell(rate), (rate.kernel, rate.shape)
        assert cublas == "not measured"


def test_each_day_77_rate_is_the_rate_of_its_recorded_table():
    rates = [rate for rate in METAL_RATES if rate.lab == "day-77"]

    assert len(rates) == 3 * len(PERFORMANCE_SHAPES)
    for rate in rates:
        name = f"`{DISPLAY.get(rate.kernel, rate.kernel)}`"
        (row,) = [row for row in day_77_rows(rate.shape) if row[1] == name]
        assert row[2] == cell(rate), (rate.kernel, rate.shape)


def test_each_report_digest_time_and_load_average_stands_in_its_guide():
    for report in METAL_REPORTS:
        text = flat(guide(report.lab))
        date, time, zone = report.run.split()
        before = ", ".join(f"{value:.2f}" for value in report.load_before[:2])
        after = ", ".join(f"{value:.2f}" for value in report.load_after[:2])

        assert f"The report digest is `{report.report_sha256}`." in text
        assert f"This run used the pinned config on {date} at {time} {zone}." in text
        assert (
            f"The report recorded load averages of {before} and "
            f"{report.load_before[2]:.2f} before the run. It recorded {after} "
            f"and {report.load_after[2]:.2f} after the run."
        ) in text


def test_no_gpu_cuda_cublas_or_interpreter_side_has_a_number():
    mapping = performance_mapping()

    assert [row["shape"] for row in mapping["shapes"]] == [
        shape.to_mapping() for shape in PERFORMANCE_SHAPES
    ]
    for row in mapping["shapes"]:
        assert (row["triton_gpu"], row["cuda_text"], row["cublas"]) == (
            TRITON_GPU,
            CUDA_TEXT,
            CUBLAS,
        )
        assert [item["kernel"] for item in row["metal"]] == [
            "matmul_tiled_32",
            "torch_matmul",
            "matmul_s3_block_4x4",
            "matmul_s7_interior",
            "torch_matmul",
        ]
    assert TRITON_GPU == "not measured: no NVIDIA or AMD GPU on this host"
    assert mapping["interpreter_time"] == INTERPRETER_TIME
    assert INTERPRETER_TIME.startswith("not reported: ")
    for report in mapping["metal_reports"]:
        assert report["measurement"] == "host-specific Apple GPU (Metal) measurement"
    assert "no rate here measures or predicts Triton" in mapping["boundary"]


def test_the_later_gpu_run_names_its_shapes_samples_tuning_and_baselines():
    text = " ".join(LATER_GPU_RUN)

    for shape in PERFORMANCE_SHAPES:
        assert label(shape) in text
    for phrase in (
        "triton.autotune with AUTOTUNE_CONFIGS and the key M, N, K",
        "2 calls that are not timed and then 11 timed samples",
        "the Day 76 CUDA text at tile 32",
        "the five Day 77 CUDA texts",
        "sgemm_row_major with cuBLAS",
        "check the Day 76 TF32 assumption",
        "leave TRITON_INTERPRET unset",
        "The tuning times are not results.",
    ):
        assert phrase in text, phrase
    assert len(AUTOTUNE_CONFIGS) == 6


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (("day-76", "matmul_tiled_16", PERFORMANCE_SHAPES[0], 1.0, 1.0, 1.0), "cited"),
        (("day-77", "torch_matmul", MatmulShape(1000, 1000, 1000), 1, 1, 1), "shape"),
        (("day-77", "torch_matmul", PERFORMANCE_SHAPES[0], 5.0, 6.0, 7.0), "inside"),
    ],
)
def test_a_rate_that_is_not_a_cited_recorded_rate_is_refused(fields, message):
    with pytest.raises(TritonMatmulError, match=message):
        MetalRate(*fields)
