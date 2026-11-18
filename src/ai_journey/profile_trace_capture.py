"""Profile the Day 74 naive Metal matmul kernels and ``torch.matmul`` on MPS.

The capture reuses the Day 74 lab end to end: ``MatmulMetalRuntime`` probes
the host, compiles both naive kernels and binds one workload per shape;
``matmul_inputs`` draws the operands; and every launch is checked against
its own pipeline. For each shape, the steps are:

1. Run ``warmup`` untimed iterations of the same body, without the profiler,
   so one-time costs such as pipeline and graph builds stay out of it.
2. Run ``iterations`` iterations under ``torch.profiler`` with
   ``ProfilerActivity.CPU``, in the annotation layout of
   ``profile_trace_layout``.
3. Check the latest outputs as Day 74 does: the two naive kernels must hold
   the same bits, and ``torch.matmul`` must agree within its bound.
4. Export the Chrome trace, replace the host name and the export path in
   it, and check that it parses strictly and has the layout.

``ProfilerActivity.CPU`` sees only host-side work: the dispatch calls and
the host's wait inside ``torch.mps.synchronize()``. It records no GPU kernel
and no GPU kernel time; this torch build has no MPS profiler activity. Each
side also records a pair of ``torch.mps.Event(enable_timing=True)`` events
around its dispatch, and their elapsed time is the only device-side figure
of the capture. Torch is imported only when a real session starts.
"""

from __future__ import annotations

import importlib
import json
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from .kernel_metal_geometry import CheckedLaunch
from .matmul_metal_benchmark import agreement_bound
from .matmul_metal_mapping import MAPPINGS, matmul_launch
from .matmul_metal_reference import BENCHMARK_STREAM, matmul_inputs
from .matmul_metal_runtime import SIDES, MatmulMetalRuntime, WorkloadCheck
from .matmul_metal_source import KERNEL_FOR_MAPPING
from .memory_access_tiling import MatmulShape
from .profile_trace_config import ProfileTraceConfig
from .profile_trace_events import (
    ProfileTraceError,
    decode_trace_json,
    parse_trace_bytes,
)
from .profile_trace_layout import (
    DISPATCH,
    ITERATION,
    SYNCHRONIZE,
    check_layout,
    phase_region,
    side_order,
    side_region,
    trace_file_name,
)

REDACTED_HOST = "redacted-host"
ACTIVITY = "torch.profiler.ProfilerActivity.CPU"
DEVICE_TIMER = "torch.mps.Event(enable_timing=True)"
NANOSECONDS_PER_MILLISECOND = 1_000_000
# The fields of a torch export that could name the host or a local path.
_REDACTED_FIELDS = ("host_name", "traceName")


class ProfileTraceCheckError(ProfileTraceError):
    """Raised when the outputs after a capture disagree beyond their bound."""


@dataclass(frozen=True)
class CapturedShape:
    """One shape's stored trace bytes, device event times and output check.

    ``device_ns`` holds, per side, the elapsed time between the two MPS
    events of each profiled iteration, in iteration order.
    """

    index: int
    shape: MatmulShape
    file_name: str
    trace_bytes: bytes
    device_ns: dict[str, tuple[int, ...]]
    naive_same_bits: bool
    torch_disagreement: float
    agreement_bound: float


def redact_export(payload: bytes, file_name: str) -> bytes:
    """Replace the host name and the export path of one torch export.

    Torch writes each field once as ``"key": "value"``. The replacement keeps
    every other byte, and the result must decode to the same document with
    only those two values changed.
    """

    document = decode_trace_json(payload)
    replacements = {"host_name": REDACTED_HOST, "traceName": file_name}
    redacted = payload
    for key in _REDACTED_FIELDS:
        if key not in document:
            continue
        pattern = re.compile(rb'"' + key.encode() + rb'": "(?:[^"\\]|\\.)*"')
        if len(pattern.findall(redacted)) != 1:
            raise ProfileTraceError(f"export must hold {key} exactly once")
        value = json.dumps(replacements[key]).encode("ascii")
        redacted = pattern.sub(lambda _: b'"' + key.encode() + b'": ' + value, redacted)
    expected = {
        key: (replacements[key] if key in replacements else value)
        for key, value in document.items()
    }
    if decode_trace_json(redacted) != expected:
        raise ProfileTraceError("redaction changed more than the host fields")
    return redacted


class TorchProfilerSession:
    """The real capture: the Day 74 runtime, ``torch.profiler`` and MPS events.

    Only this session makes a measured capture.
    """

    measured = True

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], MatmulMetalRuntime] = MatmulMetalRuntime,
        import_module: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self.runtime = runtime_factory()
        self._torch = import_module("torch")
        self._profiler = import_module("torch.profiler")
        self.limits = self.runtime.limits
        self.torch_version = self.runtime.torch_version
        self.float32_matmul_precision = self.runtime.float32_matmul_precision

    def workload(self, left, right, launches):
        """Bind one shape's device buffers through the Day 74 runtime."""

        return self.runtime.workload(left, right, launches)

    def annotate(self, name: str) -> AbstractContextManager[object]:
        """Open one named host span in the current profile."""

        return self._profiler.record_function(name)

    def device_events(self) -> tuple[object, object]:
        """Return two fresh timing events on the MPS stream."""

        event = self._torch.mps.Event
        return event(enable_timing=True), event(enable_timing=True)

    @staticmethod
    def elapsed_ns(start: object, stop: object) -> int:
        """Return the elapsed time between two recorded events in nanoseconds."""

        milliseconds = float(start.elapsed_time(stop))
        return round(milliseconds * NANOSECONDS_PER_MILLISECOND)

    @contextmanager
    def profile(self) -> Iterator[Callable[[], bytes]]:
        """Profile the body with the CPU activity; yield a reader of its export."""

        profiler = self._profiler
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as session:
            exported: list[bytes] = []
            yield lambda: exported[0]
        with tempfile.TemporaryDirectory(prefix="profile-trace-") as directory:
            path = Path(directory) / "trace.json"
            session.export_chrome_trace(str(path))
            exported.append(path.read_bytes())


def _launches(session: object, shape: MatmulShape, group: tuple[int, int]):
    return {
        KERNEL_FOR_MAPPING[mapping]: CheckedLaunch(
            matmul_launch(mapping, shape, group),
            session.limits[KERNEL_FOR_MAPPING[mapping]],
        )
        for mapping in MAPPINGS
    }


def _iterate(
    session: object, workload: object, count: int, annotate: bool
) -> dict[str, list[int]]:
    """Run ``count`` iterations of the capture body; return the event times."""

    def region(name: str) -> AbstractContextManager[object]:
        return session.annotate(name) if annotate else _quiet()

    device: dict[str, list[int]] = {side: [] for side in SIDES}
    for number in range(count):
        with region(ITERATION):
            for side in side_order(number):
                with region(side_region(side)):
                    start, stop = session.device_events()
                    start.record()
                    with region(phase_region(side, DISPATCH)):
                        workload.sides[side]()
                    stop.record()
                    with region(phase_region(side, SYNCHRONIZE)):
                        workload.synchronize()
                device[side].append(session.elapsed_ns(start, stop))
    return device


@contextmanager
def _quiet() -> Iterator[None]:
    yield None


def _check(result: object, shape: MatmulShape) -> tuple[bool, float, float]:
    if not isinstance(result, WorkloadCheck):
        raise TypeError("a workload check must return WorkloadCheck")
    bound = agreement_bound(shape.k)
    if not result.naive_same_bits:
        raise ProfileTraceCheckError(
            f"shape {shape.m} x {shape.n} x {shape.k}: the naive kernels differ"
        )
    if not 0 <= result.torch_disagreement <= bound:
        raise ProfileTraceCheckError(
            f"shape {shape.m} x {shape.n} x {shape.k}: torch.matmul is "
            f"{result.torch_disagreement:.3g} from the naive output; the bound is "
            f"{bound:.3g}"
        )
    return True, float(result.torch_disagreement), bound


def capture_shape(
    session: object, config: ProfileTraceConfig, index: int
) -> CapturedShape:
    """Warm up, profile and check one shape, then return its redacted trace."""

    shape = config.shapes[index]
    left, right = matmul_inputs(config.seed, BENCHMARK_STREAM, index, shape)
    workload = session.workload(
        left, right, _launches(session, shape, config.group_size)
    )
    _iterate(session, workload, config.warmup, annotate=False)
    with session.profile() as exported:
        device = _iterate(session, workload, config.iterations, annotate=True)
    same, disagreement, bound = _check(workload.check(), shape)
    name = trace_file_name(index, shape)
    payload = redact_export(exported(), name)
    check_layout(parse_trace_bytes(payload), config.iterations)
    for side, values in device.items():
        if any(type(value) is not int or value < 0 for value in values):
            raise ProfileTraceError(f"{side} device event times must be >= 0 ns")
    return CapturedShape(
        index=index,
        shape=shape,
        file_name=name,
        trace_bytes=payload,
        device_ns={side: tuple(values) for side, values in device.items()},
        naive_same_bits=same,
        torch_disagreement=disagreement,
        agreement_bound=bound,
    )


def capture_traces(
    config: ProfileTraceConfig, session: object
) -> tuple[CapturedShape, ...]:
    """Capture every configured shape in order with one session."""

    if not isinstance(config, ProfileTraceConfig):
        raise TypeError("config must be a ProfileTraceConfig")
    return tuple(
        capture_shape(session, config, index) for index in range(len(config.shapes))
    )
