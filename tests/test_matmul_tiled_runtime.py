import functools
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.kernel_metal_geometry import CheckedLaunch
from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS
from ai_journey.matmul_metal_benchmark import agreement_bound
from ai_journey.matmul_metal_mapping import X_COLUMNS, matmul_launch, matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    max_relative_error,
)
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_runtime import (
    NAIVE_KERNEL,
    MatmulTiledRuntime,
    MatmulTiledUnavailableError,
    mps_switches,
)
from ai_journey.matmul_tiled_source import MatmulTiledError, threadgroup_bytes
from ai_journey.matmul_tiled_variants import (
    FEWER_THREADS,
    OTHER_THREADGROUP_BYTES,
    VARIANTS,
)
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

ROOT = Path(__file__).resolve().parents[1]
NAIVE_GROUP = (32, 8)
ALL_VARIANTS = tuple(VARIANTS.values())
SHAPE = matmul_shape(31, 47, 129)


@functools.cache
def tiled_runtime():
    """Compile both libraries once per test process; call only under requires_metal."""

    return MatmulTiledRuntime(ALL_VARIANTS)


def fake_pipelines(threads=None, memory=None):
    """A torch stand-in whose libraries report chosen pipeline limits."""

    threads, memory = threads or {}, memory or {}

    def compile_shader(source):
        pipelines = {}
        for name in re.findall(r"kernel void (\w+)\(", source):
            tile = VARIANTS[name].tile if name in VARIANTS else None
            pipelines[name] = SimpleNamespace(
                max_threads_per_threadgroup=threads.get(name, 1024),
                thread_execution_width=32,
                static_thread_group_memory_length=memory.get(
                    name, 0 if tile is None else threadgroup_bytes(tile)
                ),
            )
        return SimpleNamespace(**pipelines)

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    torch.get_float32_matmul_precision = lambda: "highest"
    return torch


def fake_runtime(**limits):
    return MatmulTiledRuntime(
        ALL_VARIANTS,
        system=darwin,
        import_module=importer(fake_pipelines(**limits)),
        environ={},
    )


def checked(runtime, name, shape=SHAPE, group=None):
    if group is None:
        group = VARIANTS[name].group_size if name in VARIANTS else NAIVE_GROUP
    return CheckedLaunch(matmul_launch(X_COLUMNS, shape, group), runtime.limits[name])


@pytest.mark.parametrize(
    ("module", "reason"),
    [
        (ImportError("no torch"), "torch_not_importable"),
        (fake_torch(built=False), "pytorch_without_mps"),
        (fake_torch(available=False), "mps_unavailable"),
        (fake_torch(compile_shader=False), "compile_shader_unavailable"),
    ],
)
def test_a_host_without_the_requirements_is_named_before_compiling(module, reason):
    with pytest.raises(MatmulTiledUnavailableError, match=reason):
        MatmulTiledRuntime(ALL_VARIANTS, system=darwin, import_module=importer(module))
    with pytest.raises(MatmulTiledUnavailableError, match="unsupported_operating"):
        MatmulTiledRuntime(
            ALL_VARIANTS,
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


@pytest.mark.parametrize(
    ("marker", "label"),
    [
        ("kernel void matmul_x_rows(", "the Day 74 naive source"),
        ("kernel void matmul_tiled_32(", "the tiled source"),
    ],
)
def test_a_rejected_library_is_named_as_an_unavailable_host(marker, label):
    torch = fake_pipelines()
    compile_shader = torch.mps.compile_shader

    def refuse(source):
        if marker in source:
            raise RuntimeError("program_source:9: error: unknown type name")
        return compile_shader(source)

    torch.mps = SimpleNamespace(compile_shader=refuse)

    with pytest.raises(MatmulTiledUnavailableError, match=f"rejected {label}"):
        MatmulTiledRuntime(ALL_VARIANTS, system=darwin, import_module=importer(torch))


def test_every_pipeline_reports_its_limits_and_a_refused_tile_is_data():
    runtime = fake_runtime(
        threads={"matmul_tiled_32": 896}, memory={"matmul_tiled_16": 2064}
    )

    assert list(runtime.limits) == [
        "matmul_x_columns",
        "matmul_x_rows",
        "matmul_tiled_8",
        "matmul_tiled_16",
        "matmul_tiled_32",
    ]
    verdicts = {name: entry.reason for name, entry in runtime.admissions.items()}
    assert verdicts == {
        "matmul_tiled_8": None,
        "matmul_tiled_16": OTHER_THREADGROUP_BYTES,
        "matmul_tiled_32": FEWER_THREADS,
    }
    assert runtime.limits["matmul_tiled_32"].max_threads_per_threadgroup == 896
    assert runtime.float32_matmul_precision == "highest"
    assert runtime.mps_switches == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }


def test_a_refused_variant_or_a_foreign_launch_never_runs():
    runtime = fake_runtime(threads={"matmul_tiled_32": 512})
    left, right = matmul_inputs(76, CORRECTNESS_STREAM, 3, SHAPE)
    naive = checked(runtime, NAIVE_KERNEL)

    with pytest.raises(MatmulTiledError, match="refused by its pipeline"):
        runtime.run("matmul_tiled_32", left, right, naive)
    with pytest.raises(MatmulTiledError, match="in its own group"):
        runtime.run(
            "matmul_tiled_8",
            left,
            right,
            checked(runtime, "matmul_tiled_8", group=(16, 16)),
        )
    with pytest.raises(MatmulTiledError, match="in its own group"):
        runtime.run("matmul_tiled_16", left, right, checked(runtime, "matmul_tiled_8"))
    with pytest.raises(MatmulTiledError, match="neither the naive kernel"):
        runtime.run("matmul_x_rows", left, right, naive)
    with pytest.raises(MatmulTiledError, match="starts with the naive"):
        runtime.workload(
            left, right, {"matmul_tiled_8": checked(runtime, "matmul_tiled_8")}
        )
    with pytest.raises(MatmulTiledError, match="refused by its pipeline"):
        runtime.workload(
            left,
            right,
            {
                NAIVE_KERNEL: naive,
                "matmul_tiled_32": checked(runtime, "matmul_tiled_8"),
            },
        )


def test_the_mps_switches_are_copied_as_tokens_or_unset():
    assert mps_switches({"PYTORCH_MPS_FAST_MATH": "1"}) == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert mps_switches(
        {"PYTORCH_MPS_FAST_MATH": "", "PYTORCH_MPS_PREFER_METAL": "/Users/someone"}
    ) == {"PYTORCH_MPS_FAST_MATH": "unknown", "PYTORCH_MPS_PREFER_METAL": "unknown"}


def test_importing_the_runtime_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.matmul_tiled_runtime import MatmulTiledRuntime\n"
        "from ai_journey.matmul_tiled_variants import VARIANTS\n"
        "try:\n"
        "    MatmulTiledRuntime(tuple(VARIANTS.values()))\n"
        "except Exception as error:\n"
        "    print(type(error).__name__, error)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.stdout.startswith("MatmulTiledUnavailableError ")
    assert f"({reason})" in completed.stdout


@requires_metal
def test_both_libraries_compile_and_each_tile_gets_a_verdict_from_its_pipeline():
    runtime = tiled_runtime()

    assert set(runtime.limits) == {"matmul_x_columns", "matmul_x_rows", *VARIANTS}
    for name, variant in VARIANTS.items():
        limits = runtime.limits[name]
        admission = runtime.admissions[name]
        assert limits.static_threadgroup_memory_bytes == variant.threadgroup_bytes
        assert admission.admitted == (
            limits.max_threads_per_threadgroup >= variant.threads_per_group
        )
    assert runtime.limits[NAIVE_KERNEL].static_threadgroup_memory_bytes == 0


@requires_metal
@pytest.mark.parametrize("shape", [matmul_shape(17, 33, 9), SHAPE])
def test_each_admitted_tile_keeps_its_guard_tail_and_the_bound(shape):
    runtime = tiled_runtime()
    left, right = matmul_inputs(76, CORRECTNESS_STREAM, 1, shape)
    reference, magnitude = float64_reference(left, right)
    replay = fma_replay(left, right).reshape(-1).view(numpy.uint32)
    elements = shape.m * shape.n

    for name in (NAIVE_KERNEL, *VARIANTS):
        if name in VARIANTS and not runtime.admissions[name].admitted:
            continue
        launch = checked(runtime, name, shape)
        buffer = runtime.run(name, left, right, launch)
        assert buffer.size == elements + launch.geometry.padding_threads
        tail = buffer[elements:].view(numpy.uint32)
        assert set(tail.tolist()) <= {FLOAT32_SENTINEL_BITS}, name
        output = buffer[:elements].reshape(shape.m, shape.n)
        assert max_relative_error(output, reference, magnitude) <= error_bound(shape.k)
        # Observed on the development host's GPU; the run records it per case.
        assert numpy.array_equal(buffer[:elements].view(numpy.uint32), replay), name


@requires_metal
def test_the_workload_checks_every_side_against_the_naive_output():
    runtime = tiled_runtime()
    shape = matmul_shape(64, 48, 100)
    left, right = matmul_inputs(76, 1, 0, shape)
    launches = {NAIVE_KERNEL: checked(runtime, NAIVE_KERNEL, shape)}
    launches.update({name: checked(runtime, name, shape) for name in VARIANTS})

    workload = runtime.workload(left, right, launches)
    for call in workload.calls:
        call()
    checks = workload.check()

    assert list(workload.sides) == [NAIVE_KERNEL, *VARIANTS, TORCH_MATMUL]
    assert list(checks) == [*VARIANTS, TORCH_MATMUL]
    for name, result in checks.items():
        assert 0.0 <= result.disagreement <= agreement_bound(shape.k), name
        if name in VARIANTS:
            assert result.same_bits_as_naive is True, name
