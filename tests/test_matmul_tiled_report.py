import itertools
import os
import sys
import warnings

import pytest

from ai_journey.gpu_ceiling_source import GpuCeilingError
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_metal_source import source_sha256
from ai_journey.matmul_tiled_config import config_from_mapping
from ai_journey.matmul_tiled_report import (
    MAX_WARNING_TEXT,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    measure_report,
    warning_records,
)
from ai_journey.matmul_tiled_runtime import NAIVE_KERNEL
from ai_journey.matmul_tiled_source import MatmulTiledError
from ai_journey.matmul_tiled_variants import VARIANTS, library_source, text_sha256
from ai_journey.memory_access_report import report_digest
from test_kernel_metal_runtime import requires_metal
from test_matmul_tiled_config import make_config_mapping
from test_matmul_tiled_correctness import StandInBackend
from test_matmul_tiled_runtime import fake_runtime, tiled_runtime

LOAD = (1.5, 2.25, 3.0)


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def stand_in_report(backend=None, **changes):
    config = config_from_mapping({**make_config_mapping(), **changes})
    return measure_report(
        config,
        backend=backend or StandInBackend(),
        clock=scripted_clock(),
        load_average=lambda: LOAD,
    )


class WarningBackend(StandInBackend):
    """A stand-in whose kernel runs raise warnings, as PyTorch can on first use."""

    def run(self, name, left, right, launch):
        home = os.path.expanduser("~")
        warnings.warn(f"{home}/lib/mps.py: fell back to a Metal kernel", UserWarning)
        if name == NAIVE_KERNEL:
            warnings.warn("x" * (MAX_WARNING_TEXT + 50), DeprecationWarning)
        return super().run(name, left, right, launch)


def test_a_stand_in_report_is_synthetic_sealed_and_complete():
    torch_imported = sys.modules.get("torch") is not None

    report = stand_in_report()

    assert report["report_sha256"] == report_digest(report)
    assert (report["measured"], report["backend"], report["timer"]) == (
        False,
        "injected",
        "injected",
    )
    assert report["measurement_scope"] == "synthetic_backend_or_clock"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["host_load"]["before"] == report["host_load"]["after"] == list(LOAD)
    assert report["warnings"] == {
        "capture": report["warnings"]["capture"],
        "torch_imported_before_capture": torch_imported,
        "count": 0,
        "distinct": 0,
        "records": [],
    }
    assert report["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert report["refusals"] == [] and report["summary"]["refused_variants"] == []
    points = report["benchmark"]["points"]
    assert [list(point["sides"]) for point in points] == [
        [NAIVE_KERNEL, *VARIANTS, TORCH_MATMUL]
    ] * 2
    assert report["summary"]["timed_samples"] == 2 * 5 * 3
    assert report["comparison"]["vendor"] == (
        "vendor GPU matmul: torch.matmul on MPS, PyTorch 2.15.0"
    )
    assert report["run"]["timed_ns"] <= report["run"]["elapsed_ns"]


def test_both_source_digests_and_every_variant_digest_and_limit_are_recorded():
    shaders = stand_in_report()["shaders"]

    assert shaders["naive"]["source_sha256"] == source_sha256()
    assert [kernel["name"] for kernel in shaders["naive"]["kernels"]] == [
        "matmul_x_columns",
        "matmul_x_rows",
    ]
    tiled = shaders["tiled"]
    assert tiled["source"] == library_source(tuple(VARIANTS.values()))
    assert tiled["source_sha256"] == text_sha256(tiled["source"])
    for entry, variant in zip(tiled["variants"], VARIANTS.values()):
        assert entry["source_sha256"] == variant.source_sha256
        assert entry["limits"]["static_threadgroup_memory_bytes"] == (
            variant.threadgroup_bytes
        )
        assert entry["admission"]["admitted"] is True


def test_a_refused_tile_is_recorded_and_never_checked_or_timed():
    report = stand_in_report(StandInBackend(threads={"matmul_tiled_32": 512}))

    assert report["refusals"] == [
        {
            "name": "matmul_tiled_32",
            "admitted": False,
            "reason": "pipeline_allows_fewer_threads_than_the_tile",
            "required_threads": 1024,
            "allowed_threads": 512,
            "declared_threadgroup_bytes": 8192,
            "static_threadgroup_bytes": 8192,
        }
    ]
    assert report["summary"]["refused_variants"] == ["matmul_tiled_32"]
    assert report["summary"]["kernels_checked"] == [
        NAIVE_KERNEL,
        "matmul_tiled_8",
        "matmul_tiled_16",
    ]
    for point in report["benchmark"]["points"]:
        assert "matmul_tiled_32" not in point["sides"]
    for entry in report["comparison"]["points"]:
        assert entry["tiled"]["side"] in ("matmul_tiled_8", "matmul_tiled_16")


def test_warnings_from_the_run_are_sealed_once_each_without_the_home_path():
    report = stand_in_report(WarningBackend())

    captured = report["warnings"]
    assert captured["count"] == 3 * 4 + 3
    assert captured["distinct"] == 2 and len(captured["records"]) == 2
    fallback, long = captured["records"]
    assert fallback["category"] == "UserWarning" and fallback["count"] == 12
    assert fallback["message"] == "~/lib/mps.py: fell back to a Metal kernel"
    assert fallback["file"] == "test_matmul_tiled_report.py"
    assert os.path.expanduser("~") not in str(captured)
    assert long["category"] == "DeprecationWarning" and long["count"] == 3
    assert len(long["message"]) == MAX_WARNING_TEXT
    assert report["summary"]["warnings"] == 15


def test_warning_records_keep_a_bounded_list_with_the_full_count():
    caught = [
        warnings.WarningMessage(f"warning {index}", UserWarning, "/a/b.py", index)
        for index in range(40)
    ]

    record = warning_records(caught, torch_imported=False, home="/a")

    assert (record["count"], record["distinct"], len(record["records"])) == (40, 40, 32)
    assert record["records"][0]["file"] == "b.py"
    with pytest.raises(TypeError, match="bool"):
        warning_records(caught, torch_imported=1, home="/a")


def test_the_mps_switches_come_from_the_backend_and_bad_loads_are_refused():
    backend = StandInBackend()
    backend.mps_switches = {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert stand_in_report(backend)["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    config = config_from_mapping(make_config_mapping())
    with pytest.raises(GpuCeilingError, match="three nonnegative floats"):
        measure_report(
            config,
            backend=StandInBackend(),
            clock=scripted_clock(),
            load_average=lambda: (1.0, 2.0),
        )


def test_a_runtime_that_compiled_other_tiles_cannot_make_the_report():
    runtime = fake_runtime()
    mapping = make_config_mapping()
    mapping["variants"] = ["matmul_tiled_8"]

    assert runtime.tiled_source == library_source(tuple(VARIANTS.values()))
    with pytest.raises(MatmulTiledError, match="another tiled library"):
        measure_report(
            config_from_mapping(mapping),
            backend=runtime,
            clock=scripted_clock(),
            load_average=lambda: LOAD,
        )


def test_a_run_past_its_budget_stops_without_a_report():
    mapping = make_config_mapping()
    mapping["benchmark"]["max_runtime_seconds"] = 1
    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        measure_report(
            config_from_mapping(mapping),
            backend=StandInBackend(),
            clock=scripted_clock(step=10**9),
            load_average=lambda: LOAD,
        )


@requires_metal
def test_a_real_run_is_a_labeled_apple_gpu_measurement():
    runtime = tiled_runtime()
    config = config_from_mapping(make_config_mapping())

    report = measure_report(config, backend=runtime)

    assert report["measured"] is True
    assert report["label"] == "host-specific Apple GPU (Metal) measurement"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert report["backend"] == "torch.mps.compile_shader"
    assert report["warnings"]["torch_imported_before_capture"] is True
    for case in report["correctness"]["cases"]:
        for output in case["outputs"]:
            assert output["bound_ratio"] <= 1.0
