"""Run the Day 81 kernels and the PyTorch attention call on the MPS device.

Importing this module never imports torch, so a host without torch can
still verify reports. ``MetalFlashRuntime`` first copies the two PyTorch
MPS switches from the environment, as the Day 76 to 80 runtimes do. It then
runs the Day 73 ``probe_metal``, which reads the operating system before it
imports torch, compiles the library through the public
``torch.mps.compile_shader`` and reads the limits of every pipeline.

A kernel launches only when its pipeline admits it (``admit``): the
pipeline allows the group size of the kernel, reports a SIMD width of 32,
and reports the static threadgroup bytes of its text. The Day 79 refusal
names apply.

The runtime takes and returns NumPy arrays, so the correctness and
benchmark code never touch torch, and a stand-in can replace the runtime on
a host without MPS. Only ``MetalFlashRuntime`` itself makes a measured
report.

``memory`` reads the two MPS allocator counters around one side at a time.
``torch.mps.current_allocated_memory()`` counts the bytes of live tensors
in the PyTorch MPS allocator, rounded to its blocks.
``torch.mps.driver_allocated_memory()`` counts the bytes that the Metal
driver holds for this process, the allocator cache included. The runtime reads
them after ``synchronize`` before and after each step, and a second thread
polls them while the call runs. The polls see only what the allocator and
the driver hold at the moment of each poll, so a buffer that lives between
two polls, or memory that Metal holds outside these counters, does not
show.
"""

from __future__ import annotations

import importlib
import os
import platform
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import CheckedLaunch, PipelineLimits, vector_launch
from .kernel_metal_runtime import BACKEND, probe_metal
from .matmul_tiled_runtime import mps_switches, pipeline_limits
from .metal_fused_runtime import (
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
)
from .metal_flash_plan import TORCH_SDPA
from .metal_flash_reference import FlashInputs, FlashLayout, sentinel_output
from .metal_flash_source import (
    CAUSAL,
    F1,
    HEAD_DIM,
    KERNELS,
    MODES,
    N0,
    PROBE_EXP,
    PROBE_GROUP,
    SIMD_WIDTH,
    VARIANTS,
    MetalFlashError,
    flash_library,
    kernel_names,
)

REFUSALS = (FEWER_THREADS, OTHER_SIMD_WIDTH, OTHER_THREADGROUP_BYTES)
COUNTERS = ("current_bytes", "driver_bytes")
READINGS = ("before", "after_allocation", "after_call", "peak_during_call")


class MetalFlashUnavailableError(MetalFlashError):
    """Raised when this host cannot compile or run the Day 81 kernels."""


@dataclass(frozen=True)
class FlashAdmission:
    """Whether one pipeline admits the group size of its kernel."""

    kernel: str
    limits: PipelineLimits

    def __post_init__(self) -> None:
        if self.kernel not in KERNELS:
            raise MetalFlashError(f"{self.kernel!r} is not a registered kernel")
        if not isinstance(self.limits, PipelineLimits):
            raise TypeError("an admission needs the limits of its pipeline")

    @property
    def reason(self) -> str | None:
        """The first refusal of the pipeline, or ``None``."""

        kernel, limits = KERNELS[self.kernel], self.limits
        if limits.max_threads_per_threadgroup < kernel.group_size:
            return FEWER_THREADS
        if limits.thread_execution_width != SIMD_WIDTH:
            return OTHER_SIMD_WIDTH
        if limits.static_threadgroup_memory_bytes != kernel.threadgroup_bytes:
            return OTHER_THREADGROUP_BYTES
        return None

    @property
    def admitted(self) -> bool:
        """Whether the kernel may launch."""

        return self.reason is None

    def to_mapping(self) -> dict[str, object]:
        """Return the verdict and the numbers behind it."""

        kernel = KERNELS[self.kernel]
        return {
            "kernel": self.kernel,
            "admitted": self.admitted,
            "reason": self.reason,
            "required_threads": kernel.group_size,
            "allowed_threads": self.limits.max_threads_per_threadgroup,
            "thread_execution_width": self.limits.thread_execution_width,
            "declared_threadgroup_bytes": kernel.threadgroup_bytes,
            "static_threadgroup_bytes": self.limits.static_threadgroup_memory_bytes,
        }


def admit(limits: Mapping[str, PipelineLimits]) -> dict[str, FlashAdmission]:
    """Return the verdict of every kernel of the library."""

    return {name: FlashAdmission(name, limits[name]) for name in KERNELS}


def launches(variant: str, mode: str, layout: FlashLayout) -> list[tuple[str, object]]:
    """Return each kernel of one side with its launch geometry, in order."""

    if not isinstance(layout, FlashLayout):
        raise TypeError("layout must be a FlashLayout")
    groups = layout.tokens if variant == N0 else layout.blocks()
    result = []
    for name in kernel_names(variant, mode):
        size = KERNELS[name].group_size
        result.append((name, vector_launch(groups * size, size)))
    return result


def launch_mappings(variant: str, mode: str, layout: FlashLayout) -> list[dict]:
    """Return the launches of one side as a report records them."""

    return [
        {"kernel": name, **geometry.to_mapping()}
        for name, geometry in launches(variant, mode, layout)
    ]


@dataclass(frozen=True)
class FlashWorkload:
    """One timing point's device buffers, bound as no-argument calls.

    ``sides`` holds N0, F1 and the PyTorch call, in plan order. Each call
    only queues work; ``synchronize`` waits for it. ``outputs`` waits too and
    returns each side's latest flat O buffer.
    """

    sides: Mapping[str, Callable[[], object]]
    synchronize: Callable[[], None]
    outputs: Callable[[], Mapping[str, numpy.ndarray]]

    @property
    def calls(self) -> tuple[Callable[[], object], ...]:
        """The side calls in timing order, as the rotating timer takes them."""

        return tuple(self.sides.values())


def _compile(torch: object, source: str) -> object:
    try:
        return torch.mps.compile_shader(source)
    except RuntimeError as error:
        raise MetalFlashUnavailableError(
            f"the Metal compiler rejected the Day 81 library: {error}"
        ) from error


def _require_inputs(inputs: object) -> FlashInputs:
    if not isinstance(inputs, FlashInputs):
        raise TypeError("inputs must be FlashInputs")
    return inputs


def _require_side(variant: str, mode: str) -> None:
    if variant not in VARIANTS:
        raise MetalFlashError(f"{variant!r} is not an attention variant")
    if mode not in MODES:
        raise MetalFlashError(f"{mode!r} is not an attention mode")


class MetalFlashRuntime:
    """The compiled Day 81 library on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        library_source = flash_library()
        self.mps_switches = mps_switches(environ)
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise MetalFlashUnavailableError(availability.message)
        torch = import_module("torch")
        library = _compile(torch, library_source)
        self._torch = torch
        self.library_source = library_source
        self._functions = {name: getattr(library, name) for name in KERNELS}
        self.torch_version = availability.torch_version
        self.limits = {
            name: pipeline_limits(function)
            for name, function in self._functions.items()
        }

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def _to(self, array: numpy.ndarray) -> object:
        array = numpy.ascontiguousarray(array)
        if array.dtype == numpy.uint32:
            # The kernels read uint; int32 carries the same bits into torch.
            array = array.view(numpy.int32)
        return self._torch.from_numpy(array).to("mps")

    def _require_admitted(self, variant: str, mode: str) -> None:
        verdicts = admit(self.limits)
        for name in kernel_names(variant, mode):
            if not verdicts[name].admitted:
                raise MetalFlashError(
                    f"{name} was refused by its pipeline: {verdicts[name].reason}"
                )

    def _side(self, variant: str, mode: str, inputs: FlashInputs, q, k, v, o):
        """Bind one kernel side to device buffers; return the call and its buffers."""

        _require_side(variant, mode)
        self._require_admitted(variant, mode)
        layout = inputs.layout
        rows = (self._to(layout.starts()), self._to(layout.lengths_array()))
        steps = []
        if variant == N0:
            scores = self._torch.empty(
                layout.score_elements, dtype=self._torch.float32, device="mps"
            )
            groups = tuple(self._to(array) for array in layout.query_groups())
            extra = (self._to(layout.score_starts()), *groups)
            buffers = (
                (scores, q, k, *rows, *extra),
                (scores, *rows, *extra),
                (o, scores, v, *rows, *extra),
            )
            held = (scores, *rows, *extra)
        else:
            groups = tuple(self._to(array) for array in layout.block_groups())
            buffers = ((o, q, k, v, *rows, *groups),)
            held = (*rows, *groups)
        for (name, geometry), arguments in zip(
            launches(variant, mode, layout), buffers
        ):
            CheckedLaunch(geometry, self.limits[name])
            steps.append(
                partial(
                    self._functions[name], *arguments, **geometry.launch_arguments()
                )
            )

        def call() -> None:
            for step in steps:
                step()

        return call, held

    def _inputs(self, inputs: FlashInputs) -> tuple[object, object, object]:
        return self._to(inputs.q), self._to(inputs.k), self._to(inputs.v)

    def run(self, inputs: FlashInputs, variant: str, mode: str) -> numpy.ndarray:
        """Run one kernel side once and return its whole O buffer.

        The buffer starts as sentinels, so a store into a gap row shows there.
        """

        inputs = _require_inputs(inputs)
        _require_side(variant, mode)
        self._require_admitted(variant, mode)
        q, k, v = self._inputs(inputs)
        output = self._to(sentinel_output(inputs.layout))
        call, _ = self._side(variant, mode, inputs, q, k, v, output)
        call()
        self.synchronize()
        return output.cpu().numpy()

    def exp_probe(self, arguments: numpy.ndarray) -> numpy.ndarray:
        """Run ``precise::exp`` once at each argument."""

        if (
            not isinstance(arguments, numpy.ndarray)
            or arguments.dtype != numpy.float32
            or arguments.ndim != 1
            or arguments.size % PROBE_GROUP
        ):
            raise MetalFlashError(
                f"probe arguments must be float32, a multiple of {PROBE_GROUP}"
            )
        launch = CheckedLaunch(
            vector_launch(arguments.size, PROBE_GROUP), self.limits[PROBE_EXP]
        )
        x = self._to(arguments)
        y = self._torch.empty_like(x)
        self._functions[PROBE_EXP](y, x, **launch.geometry.launch_arguments())
        self.synchronize()
        return y.cpu().numpy()

    def _torch_side(self, inputs: FlashInputs, mode: str, q, k, v):
        layout = inputs.layout
        shape = (1, layout.sequences, layout.longest, HEAD_DIM)
        views = [array.view(shape) for array in (q, k, v)]
        attention = self._torch.nn.functional.scaled_dot_product_attention
        causal = mode == CAUSAL
        latest: dict[str, object] = {}

        def sdpa() -> None:
            latest["o"] = attention(*views, is_causal=causal)

        return sdpa, latest

    def _require_dense(self, inputs: FlashInputs) -> FlashLayout:
        layout = inputs.layout
        if layout.gap != 0 or not layout.dense:
            raise MetalFlashError("a timing point packs dense sequences with no gap")
        return layout

    def workload(self, inputs: FlashInputs, mode: str) -> FlashWorkload:
        """Place one point's rows on the device and bind the three sides once."""

        inputs = _require_inputs(inputs)
        layout = self._require_dense(inputs)
        q, k, v = self._inputs(inputs)
        outputs = {
            variant: self._torch.empty(
                layout.buffer_elements, dtype=self._torch.float32, device="mps"
            )
            for variant in (N0, F1)
        }
        sides = {
            variant: self._side(variant, mode, inputs, q, k, v, outputs[variant])[0]
            for variant in (N0, F1)
        }
        sdpa, latest = self._torch_side(inputs, mode, q, k, v)
        sides[TORCH_SDPA] = sdpa

        def read() -> dict[str, numpy.ndarray]:
            self.synchronize()
            result = {name: outputs[name].cpu().numpy() for name in (N0, F1)}
            result[TORCH_SDPA] = latest["o"].cpu().numpy().reshape(-1)
            return result

        return FlashWorkload(sides=sides, synchronize=self.synchronize, outputs=read)

    def _read(self) -> dict[str, int]:
        return {
            "current_bytes": int(self._torch.mps.current_allocated_memory()),
            "driver_bytes": int(self._torch.mps.driver_allocated_memory()),
        }

    def _counters(self) -> dict[str, int]:
        self.synchronize()
        return self._read()

    def _observe(self, call: Callable[[], None]) -> tuple[dict[str, int], int]:
        """Run one call while a second thread polls both counters; return the peak."""

        stop = threading.Event()
        peak = dict.fromkeys(COUNTERS, 0)
        polls = [0]

        def take() -> None:
            reading = self._read()
            for name in COUNTERS:
                peak[name] = max(peak[name], reading[name])
            polls[0] += 1

        def poll() -> None:
            while not stop.is_set():
                take()

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        try:
            call()
            self.synchronize()
        finally:
            stop.set()
            thread.join()
        take()
        return dict(peak), polls[0]

    def memory(self, inputs: FlashInputs, mode: str) -> dict[str, dict]:
        """Read both allocator counters around each side, one side at a time.

        Each side starts after ``empty_cache``, allocates its own inputs,
        output and scratch, runs once under the poll and frees everything.
        """

        inputs = _require_inputs(inputs)
        layout = self._require_dense(inputs)
        torch = self._torch
        readings = {}
        for side in (N0, F1, TORCH_SDPA):
            self.synchronize()
            torch.mps.empty_cache()
            before = self._counters()
            q, k, v = self._inputs(inputs)
            held: tuple = ()
            if side == TORCH_SDPA:
                call, latest = self._torch_side(inputs, mode, q, k, v)
            else:
                output = torch.empty(
                    layout.buffer_elements, dtype=torch.float32, device="mps"
                )
                call, held = self._side(side, mode, inputs, q, k, v, output)
                latest = {"o": output}
                del output
            allocated = self._counters()
            peak, polls = self._observe(call)
            readings[side] = {
                "before": before,
                "after_allocation": allocated,
                "after_call": self._counters(),
                "peak_during_call": peak,
                "polls": polls,
            }
            del q, k, v, held, latest, call
            self.synchronize()
            torch.mps.empty_cache()
        return readings
