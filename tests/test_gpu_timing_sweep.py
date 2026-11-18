import pytest

from ai_journey.gpu_timing_clocks import DISPATCH, EVENT, SYNCHRONIZED
from ai_journey.gpu_timing_source import EMPTY_KERNEL, empty_launch
from ai_journey.gpu_timing_sweep import measure_sweep
from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.matmul_metal_mapping import MatmulMetalError, matmul_shape
from ai_journey.matmul_metal_runtime import SIDES, MatmulWorkload, WorkloadCheck
from ai_journey.matmul_metal_source import KERNELS
from test_gpu_timing_benchmark import FakeDevice
from test_gpu_timing_runtime import timing_runtime
from test_kernel_metal_runtime import requires_metal

LIMITS = PipelineLimits(
    max_threads_per_threadgroup=1024,
    thread_execution_width=32,
    static_threadgroup_memory_bytes=0,
)
SHAPES = (matmul_shape(1, 1, 1), matmul_shape(16, 16, 16), matmul_shape(64, 32, 128))
# Virtual GPU costs: a fixed 5 us plus a rate per FLOP for each side.
FIXED_NS = 5_000
NS_PER_FLOP = {"matmul_x_columns": 0.02, "matmul_x_rows": 0.05, "torch_matmul": 0.01}


class StandInBackend:
    """A virtual-clock stand-in for ``GpuTimingRuntime``; its reports are synthetic."""

    backend = "stand-in"
    torch_version = "2.15.0"
    float32_matmul_precision = "highest"

    def __init__(self, check=None):
        self.device = FakeDevice(dispatch_ns=3_000, wake_ns=40_000)
        self.limits = {name: LIMITS for name in (*KERNELS, EMPTY_KERNEL)}
        self.result = check or WorkloadCheck(True, 0.0)
        self.workloads = []
        self.empty_launches = []

    def clock(self):
        return self.device.clock()

    def synchronize(self):
        self.device.synchronize()

    def event_pair(self):
        return self.device.event_pair()

    def empty_side(self, launch):
        self.empty_launches.append(launch)
        self.device.gpu_ns[EMPTY_KERNEL] = FIXED_NS
        return self.device.side(EMPTY_KERNEL)

    def workload(self, left, right, launches):
        flops = 2 * left.shape[0] * left.shape[1] * right.shape[1]
        self.workloads.append((left.shape, right.shape, dict(launches)))
        for side in SIDES:
            cost = FIXED_NS + round(NS_PER_FLOP[side] * flops)
            self.device.gpu_ns[side] = cost
        return MatmulWorkload(
            sides={side: self.device.side(side) for side in SIDES},
            synchronize=self.device.synchronize,
            check=lambda: self.result,
        )


def sweep(backend, shapes=SHAPES, **kwargs):
    return measure_sweep(
        backend,
        seed=75,
        shapes=shapes,
        group_size=(32, 8),
        warmup=1,
        repeats=3,
        clock=backend.clock,
        **kwargs,
    )


def test_the_empty_kernel_runs_first_with_its_one_thread_launch():
    backend = StandInBackend()

    empty, points = sweep(backend)

    assert backend.empty_launches[0].geometry == empty_launch()
    assert empty.timing.implementation == EMPTY_KERNEL
    assert empty.timing.samples(EVENT).samples_ns == (3_000 + FIXED_NS,) * 3
    assert empty.timing.samples(DISPATCH).samples_ns == (3_000,) * 3
    mapping = empty.to_mapping()
    assert mapping["kernel"]["name"] == EMPTY_KERNEL
    assert mapping["launch"]["launched_threads"] == 1
    launches = [entry for entry in backend.device.log if entry[0] == "launch"]
    assert launches[0] == ("launch", EMPTY_KERNEL)
    assert len(points) == 3


def test_every_shape_times_the_three_day_74_sides_on_seeded_operands():
    backend = StandInBackend()

    _, points = sweep(backend)

    for point, shape, (left, right, launches) in zip(points, SHAPES, backend.workloads):
        assert point.shape == shape
        assert left == (shape.m, shape.k) and right == (shape.k, shape.n)
        assert set(launches) == set(KERNELS)
        assert tuple(item.implementation for item in point.timings) == SIDES
        rows = point.timing("matmul_x_rows")
        expected = 3_000 + FIXED_NS + round(0.05 * shape.flops)
        assert rows.samples(EVENT).samples_ns == (expected,) * 3
        assert rows.samples(SYNCHRONIZED).samples_ns == (expected + 40_000,) * 3
    mapping = points[2].to_mapping()
    assert (mapping["m"], mapping["n"], mapping["k"]) == (64, 32, 128)
    assert mapping["flops_per_call"] == 2 * 64 * 32 * 128
    assert list(mapping["sides"]) == list(SIDES)
    assert set(mapping["launches"]) == {"x_columns", "x_rows"}
    assert mapping["agreement_bound"] > mapping["torch_disagreement"] == 0.0


@pytest.mark.parametrize(
    ("check", "message"),
    [
        (WorkloadCheck(False, 0.0), "differ in bits"),
        (WorkloadCheck(True, 1.0), "agreement bound"),
    ],
)
def test_a_shape_with_wrong_outputs_stops_the_run(check, message):
    with pytest.raises(MatmulMetalError, match=message):
        sweep(StandInBackend(check))


def test_a_sweep_needs_bounded_matmul_shapes():
    with pytest.raises(MatmulMetalError, match="1 to 16"):
        sweep(StandInBackend(), shapes=())
    with pytest.raises(TypeError, match="MatmulShape"):
        sweep(StandInBackend(), shapes=((1, 1, 1),))


@requires_metal
def test_a_short_sweep_on_this_gpu_has_the_timing_structure():
    runtime = timing_runtime()

    empty, points = measure_sweep(
        runtime,
        seed=75,
        shapes=SHAPES,
        group_size=(32, 8),
        warmup=1,
        repeats=3,
    )

    for timing in (empty.timing, *(item for p in points for item in p.timings)):
        assert timing.timer == "time.perf_counter_ns"
        assert timing.median_ns(DISPATCH) <= timing.median_ns(SYNCHRONIZED)
        assert all(value > 0 for value in timing.samples(EVENT).elapsed_ms)
