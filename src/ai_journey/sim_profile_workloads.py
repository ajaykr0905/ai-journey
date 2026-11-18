"""The three pinned simulator workloads that each profiling stage runs.

Inputs are built once, before any profile or timer starts, so a measured
pass covers only what a lab run repeats: buffer setup, the simulated launch
and the copy back to the host. Each pass runs the Day 73 or Day 74 run
helper unchanged, in the canonical order. Its output digest must equal the
digest of the pure-Python reference, which never touches the simulator, so
a faster simulator that changed one output bit fails the pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .kernel_launch_buffer import values_sha256
from .kernel_launch_config import LaunchConfig, launch_1d, launch_2d
from .kernel_launch_kernels import KernelRun, run_rgb_to_grayscale, run_vector_add
from .kernel_launch_reference import (
    reference_rgb_to_grayscale,
    reference_vector_add,
    rgb_image,
    vector_add_inputs,
)
from .matmul_model_kernels import matmul_launch, run_naive_matmul
from .matmul_model_reference import matmul_inputs, reference_matmul
from .sim_profile_config import (
    NAIVE_MATMUL,
    RGB_TO_GRAYSCALE,
    VECTOR_ADD,
    SimProfileConfig,
    SimProfileError,
)


@dataclass(frozen=True)
class Workload:
    """One pinned simulator pass with prebuilt inputs and its reference digest."""

    name: str
    config: LaunchConfig
    output_dtype: str
    reference_sha256: str
    run: Callable[[], KernelRun]

    def checked_run(self) -> KernelRun:
        """Run one pass and refuse an output that differs from the reference."""

        result = self.run()
        self.check(result)
        return result

    def check(self, result: KernelRun) -> None:
        if type(result) is not KernelRun or (
            result.output_sha256 != self.reference_sha256
        ):
            raise SimProfileError(
                f"workload {self.name} output differs from its reference"
            )


def _vector_add(config: SimProfileConfig) -> Workload:
    spec = config.vector_add
    a, b = vector_add_inputs(spec.input_seed, spec.n)

    def run() -> KernelRun:
        return run_vector_add(a, b, threads_per_block=spec.threads_per_block)

    return Workload(
        name=VECTOR_ADD,
        config=launch_1d(spec.n, spec.threads_per_block),
        output_dtype="float32",
        reference_sha256=values_sha256("float32", reference_vector_add(a, b)),
        run=run,
    )


def _grayscale(config: SimProfileConfig) -> Workload:
    spec = config.rgb_to_grayscale
    rgb = rgb_image(spec.input_seed, spec.width, spec.height)

    def run() -> KernelRun:
        return run_rgb_to_grayscale(
            rgb, width=spec.width, height=spec.height, block=spec.block
        )

    reference = reference_rgb_to_grayscale(rgb, spec.width, spec.height)
    return Workload(
        name=RGB_TO_GRAYSCALE,
        config=launch_2d(spec.width, spec.height, *spec.block),
        output_dtype="uint8",
        reference_sha256=values_sha256("uint8", reference),
        run=run,
    )


def _matmul(config: SimProfileConfig) -> Workload:
    spec = config.naive_matmul
    a, b = matmul_inputs(spec.input_seed, spec.shape)

    def run() -> KernelRun:
        return run_naive_matmul(
            a, b, spec.shape, block=spec.block, mapping=spec.mapping
        )

    return Workload(
        name=NAIVE_MATMUL,
        config=matmul_launch(spec.shape, spec.block, spec.mapping),
        output_dtype="float32",
        reference_sha256=values_sha256("float32", reference_matmul(a, b, spec.shape)),
        run=run,
    )


def build_workloads(config: SimProfileConfig) -> tuple[Workload, ...]:
    """Prepare the vector add, grayscale and naive matmul passes, in that order."""

    if type(config) is not SimProfileConfig:
        raise TypeError("config must be SimProfileConfig")
    return (_vector_add(config), _grayscale(config), _matmul(config))
