"""Run the Day 82 kernels and ``torch.matmul`` on the MPS device.

Importing this module never imports torch, so a host without torch can
still verify reports. ``MetalTcRuntime`` first copies the two PyTorch MPS
switches from the environment, as the Day 76 to 81 runtimes do. It then
runs the Day 73 ``probe_metal``, which reads the operating system before it
imports torch, and compiles two libraries through the public
``torch.mps.compile_shader``: the main library (S1, M1, M2 and the version
probe) and the bfloat library (M3).

PyTorch raises ``SyntaxError`` when the Metal compiler rejects a text, and
``RuntimeError`` when a kernel builds no pipeline. PyTorch builds the
pipeline when the kernel is read. A rejected main library makes the host
unavailable. Either error in the bfloat library records M3 as unsupported.

Support is checked at run time and recorded, never assumed:

- The version probe writes ``__METAL_VERSION__`` of the main library.
- M3 is supported only when the bfloat library compiles, its kernel builds a
  pipeline and one 8 x 8 x 8 product of small integers gives the exact
  result. Otherwise M3 is recorded as unsupported with its reason, and it
  never launches.
- The bfloat16 PyTorch call is supported only when the same product on
  bfloat16 tensors gives the exact result.
- A support check that raises an error records its side as unsupported,
  and a timing point places no operand of a type that no side reads.

A kernel launches only when its pipeline admits it (``admit``): the
pipeline allows 128 threads, reports a SIMD width of 32, and reports the
static threadgroup bytes of its text. The runtime takes and returns NumPy
arrays, so the correctness and benchmark code never touch torch, and a
stand-in can replace the runtime on a host without MPS. Only
``MetalTcRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import CheckedLaunch, PipelineLimits, vector_launch
from .kernel_metal_runtime import BACKEND, probe_metal
from .matmul_tiled_runtime import mps_switches, pipeline_limits
from .memory_access_tiling import MatmulShape
from .metal_fused_runtime import (
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
)
from .metal_tc_plan import SIDE_PRECISION, SIDES, TORCH_BFLOAT16, TORCH_SIDES
from .metal_tc_reference import TcOperands, guarded, sentinel_output
from .metal_tc_source import (
    ARG_CASTS,
    BFLOAT16,
    BFLOAT_LIBRARY,
    FLOAT16,
    FLOAT32,
    GROUP,
    KERNEL_NAME,
    KERNELS,
    LIBRARIES,
    M3,
    MAIN_LIBRARY,
    PRECISION,
    PRECISIONS,
    PROBE_VERSION,
    SIMD_WIDTH,
    TILE,
    VARIANTS,
    MetalTcError,
    library_source,
)

REFUSALS = (FEWER_THREADS, OTHER_SIMD_WIDTH, OTHER_THREADGROUP_BYTES)
# A rejected text raises SyntaxError, and a kernel without a pipeline raises
# RuntimeError, so both mean that a library did not build.
BUILD_ERRORS = (SyntaxError, RuntimeError)
COMPILE_FAILED = "the_bfloat_library_did_not_compile"
WRONG_RESULT = "the_support_product_was_not_exact"
CALL_FAILED = "the_bfloat16_call_raised_an_error"
SUPPORT_REASONS = (COMPILE_FAILED, WRONG_RESULT, CALL_FAILED)
SUPPORT_RULE = (
    "M3 is supported when the bfloat library compiles and C = A @ B for 8 x 8 "
    "integers A[i][j] = (3 i + j) mod 5 - 2 and B[i][j] = (i + 2 j) mod 5 - 2 "
    "is exact; the bfloat16 PyTorch call is supported when the same product on "
    "bfloat16 tensors is exact"
)
_PROBE_SIZE = 8


class MetalTcUnavailableError(MetalTcError):
    """Raised when this host cannot compile or run the Day 82 main library."""


def support_operands() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """The small exact product that decides bfloat support."""

    index = numpy.arange(_PROBE_SIZE)
    a = ((3 * index[:, None] + index[None, :]) % 5 - 2).astype(numpy.float32)
    b = ((index[:, None] + 2 * index[None, :]) % 5 - 2).astype(numpy.float32)
    return a, b, (a.astype(numpy.float64) @ b.astype(numpy.float64))


@dataclass(frozen=True)
class Support:
    """Whether one optional side runs on this host, and why not."""

    supported: bool
    reason: str | None

    def __post_init__(self) -> None:
        if type(self.supported) is not bool or self.supported != (self.reason is None):
            raise MetalTcError("an unsupported side needs one reason")
        if self.reason is not None and self.reason not in SUPPORT_REASONS:
            raise MetalTcError(f"{self.reason!r} is not a support reason")

    def to_mapping(self) -> dict[str, object]:
        """Return the support record as a report stores it."""

        return {"supported": self.supported, "reason": self.reason}


@dataclass(frozen=True)
class TcAdmission:
    """Whether one pipeline admits the group size and memory of its kernel."""

    kernel: str
    limits: PipelineLimits

    def __post_init__(self) -> None:
        if self.kernel not in KERNELS:
            raise MetalTcError(f"{self.kernel!r} is not a registered kernel")
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


def admit(limits: Mapping[str, PipelineLimits]) -> dict[str, TcAdmission]:
    """Return the verdict of every kernel whose pipeline compiled."""

    return {name: TcAdmission(name, limits[name]) for name in KERNELS if name in limits}


def needed_precisions(supported: tuple[str, ...]) -> tuple[str, ...]:
    """The input types that the supported sides read, so no other type is made.

    A host without bfloat16 tensors keeps its float32 and float16 sides.
    """

    kinds = {SIDE_PRECISION[side] for side in supported}
    return tuple(precision for precision in PRECISIONS if precision in kinds)


def launch(shape: MatmulShape):
    """One group of 128 threads for each 32 x 32 tile of C, as a 1D launch."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    groups = -(-shape.m // TILE) * -(-shape.n // TILE)
    return vector_launch(groups * GROUP, GROUP)


def launch_mapping(shape: MatmulShape) -> dict[str, object]:
    """Return the launch of a shape as a report records it."""

    return launch(shape).to_mapping()


@dataclass(frozen=True)
class TcWorkload:
    """One timing point's device tensors, bound as no-argument calls.

    ``sides`` holds the seven sides in plan order. An unsupported side keeps
    its place in the rotation as a call that does nothing, and its samples
    are dropped. Each call only queues work; ``synchronize`` waits for it.
    ``outputs`` waits too and returns each supported side's latest C.
    """

    sides: Mapping[str, Callable[[], object]]
    supported: tuple[str, ...]
    synchronize: Callable[[], None]
    outputs: Callable[[], Mapping[str, numpy.ndarray]]

    @property
    def calls(self) -> tuple[Callable[[], object], ...]:
        """The side calls in timing order, as the rotating timer takes them."""

        return tuple(self.sides.values())


def _compile(torch: object, source: str) -> object:
    try:
        return torch.mps.compile_shader(source)
    except BUILD_ERRORS as error:
        raise MetalTcUnavailableError(
            f"the Metal compiler rejected the Day 82 main library: {error}"
        ) from error


def _pipelines(compiled: object, library: str) -> dict[str, object]:
    """Read the kernels of one compiled library, which builds their pipelines."""

    return {
        name: getattr(compiled, name)
        for name, kernel in KERNELS.items()
        if kernel.library == library
    }


def _nothing() -> None:
    return None


class MetalTcRuntime:
    """The compiled Day 82 libraries on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        sources = {name: library_source(name) for name in LIBRARIES}
        self.mps_switches = mps_switches(environ)
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise MetalTcUnavailableError(availability.message)
        torch = import_module("torch")
        self._torch = torch
        self.library_sources = sources
        self.torch_version = availability.torch_version
        functions = _pipelines(_compile(torch, sources[MAIN_LIBRARY]), MAIN_LIBRARY)
        compiled = True
        try:
            bfloat = torch.mps.compile_shader(sources[BFLOAT_LIBRARY])
            functions.update(_pipelines(bfloat, BFLOAT_LIBRARY))
        except BUILD_ERRORS:
            compiled = False
        self._functions = {
            name: functions[name] for name in KERNELS if name in functions
        }
        self.limits = {
            name: pipeline_limits(function)
            for name, function in self._functions.items()
        }
        self.metal_version = self._metal_version()
        self.support = {
            M3: self._kernel_support() if compiled else Support(False, COMPILE_FAILED),
            TORCH_BFLOAT16: self._torch_support(),
        }

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def _tensor(self, values: numpy.ndarray, precision: str) -> object:
        """Copy float32 values that the type stores exactly into a device tensor."""

        dtype = {
            FLOAT32: self._torch.float32,
            FLOAT16: self._torch.float16,
            BFLOAT16: self._torch.bfloat16,
        }[precision]
        host = self._torch.from_numpy(numpy.ascontiguousarray(values))
        return host.to(dtype).to("mps")

    def _metal_version(self) -> int:
        out = self._torch.zeros(1, dtype=self._torch.int32, device="mps")
        self._functions[PROBE_VERSION](out, threads=SIMD_WIDTH, group_size=SIMD_WIDTH)
        self.synchronize()
        return int(out.cpu().item())

    def _kernel_support(self) -> Support:
        a, b, exact = support_operands()
        shape = MatmulShape(*exact.shape, a.shape[1])
        values = TcOperands(shape, a, b)
        try:
            output = self._launch(M3, values)[: exact.size].astype(numpy.float64)
        except (RuntimeError, TypeError):
            return Support(False, CALL_FAILED)
        if not numpy.array_equal(output, exact.reshape(-1)):
            return Support(False, WRONG_RESULT)
        return Support(True, None)

    def _torch_support(self) -> Support:
        a, b, exact = support_operands()
        try:
            left, right = (self._tensor(x, BFLOAT16) for x in (a, b))
            product = self._torch.matmul(left, right).float().cpu().numpy()
        except (RuntimeError, TypeError):
            return Support(False, CALL_FAILED)
        if not numpy.array_equal(product.astype(numpy.float64), exact):
            return Support(False, WRONG_RESULT)
        return Support(True, None)

    def supported(self, side: str) -> bool:
        """Whether a kernel variant or a PyTorch side runs on this host."""

        if side not in SIDES:
            raise MetalTcError(f"{side!r} is not a side of the plan")
        record = self.support.get(side)
        return record is None or record.supported

    def _require_admitted(self, variant: str) -> str:
        if variant not in VARIANTS:
            raise MetalTcError(f"{variant!r} is not a kernel variant")
        name = KERNEL_NAME[variant]
        if name not in self.limits:
            raise MetalTcError(f"{name} did not compile on this host")
        verdict = TcAdmission(name, self.limits[name])
        if not verdict.admitted:
            raise MetalTcError(f"{name} was refused by its pipeline: {verdict.reason}")
        return name

    def _bind(self, variant: str, shape: MatmulShape, c, a, b) -> partial:
        name = self._require_admitted(variant)
        geometry = launch(shape)
        CheckedLaunch(geometry, self.limits[name])
        return partial(
            self._functions[name],
            c,
            a,
            b,
            shape.m,
            shape.n,
            shape.k,
            **geometry.launch_arguments(),
            arg_casts=ARG_CASTS,
        )

    def _launch(self, variant: str, values: TcOperands) -> numpy.ndarray:
        precision = PRECISION[variant]
        a, b = (guarded(x) for x in values.inputs(precision))
        output = self._tensor(sentinel_output(values.shape), FLOAT32)
        call = self._bind(
            variant,
            values.shape,
            output,
            self._tensor(a, precision),
            self._tensor(b, precision),
        )
        call()
        self.synchronize()
        return output.cpu().numpy()

    def run(self, values: TcOperands, variant: str) -> numpy.ndarray:
        """Run one kernel once and return its whole C buffer with the tail.

        Each operand ends in quiet NaNs that no read may reach, and the C
        buffer starts as sentinels, so a store past C shows in the tail.
        """

        if not isinstance(values, TcOperands):
            raise TypeError("values must be TcOperands")
        if not self.supported(variant):
            raise MetalTcError(f"{variant} is not supported on this host")
        return self._launch(variant, values)

    def workload(self, values: TcOperands) -> TcWorkload:
        """Place one point's operands on the device and bind the seven sides once."""

        if not isinstance(values, TcOperands):
            raise TypeError("values must be TcOperands")
        shape = values.shape
        torch = self._torch
        supported = tuple(side for side in SIDES if self.supported(side))
        tensors = {}
        for precision in needed_precisions(supported):
            a, b = values.inputs(precision)
            tensors[precision] = (
                self._tensor(a, precision).view(shape.m, shape.k),
                self._tensor(b, precision).view(shape.k, shape.n),
            )
        outputs = {
            variant: torch.empty(shape.m * shape.n, dtype=torch.float32, device="mps")
            for variant in VARIANTS
        }
        latest: dict[str, object] = {}
        sides: dict[str, Callable[[], object]] = {}
        for side in SIDES:
            if side not in supported:
                sides[side] = _nothing
            elif side in VARIANTS:
                a, b = tensors[PRECISION[side]]
                sides[side] = self._bind(side, shape, outputs[side], a, b)
            else:
                sides[side] = self._torch_call(side, tensors, latest)

        def read() -> dict[str, numpy.ndarray]:
            self.synchronize()
            result = {}
            for side in supported:
                value = outputs[side] if side in VARIANTS else latest[side]
                result[side] = value.float().cpu().numpy().reshape(-1)
            return result

        return TcWorkload(
            sides=sides,
            supported=supported,
            synchronize=self.synchronize,
            outputs=read,
        )

    def _torch_call(self, side: str, tensors: dict, latest: dict) -> Callable:
        precision = {
            name: kind for name, kind in zip(TORCH_SIDES, PRECISIONS, strict=True)
        }[side]
        a, b = tensors[precision]
        matmul = self._torch.matmul

        def call() -> None:
            latest[side] = matmul(a, b)

        return call
