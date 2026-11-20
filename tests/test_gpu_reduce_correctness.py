import itertools
from fractions import Fraction

import numpy
import pytest

from ai_journey.gpu_reduce_bound import (
    case_values,
    exact_sums,
    partial_count,
    summation_depth,
)
from ai_journey.gpu_reduce_config import ReduceCase, config_from_mapping
from ai_journey.gpu_reduce_correctness import check_case, check_cases
from ai_journey.gpu_reduce_runtime import (
    OUTPUT_SLOTS,
    PARTIAL_GUARD,
    SIDES,
    KernelAdmission,
    ReduceWorkload,
    ReductionBuffers,
    SideResult,
)
from ai_journey.gpu_reduce_source import (
    FIXED_GROUPS,
    GRID_STRIDE,
    GROUP_SIZE,
    KERNELS,
    R2,
    R3,
    R5,
    STEP_KERNELS,
    GpuReduceError,
    library_source,
)
from ai_journey.kernel_metal_benchmark import RunBudget
from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.kernel_metal_reference import array_sha256, float32_sentinels
from test_gpu_reduce_config import make_config_mapping
from test_gpu_reduce_runtime import reduce_runtime
from test_kernel_metal_runtime import requires_metal


def group_sums(name, values):
    """Each pass-1 group's sum in float64, rounded to float32."""

    kernel = KERNELS[name]
    groups = partial_count(name, values.size)
    if kernel.mapping == GRID_STRIDE:
        threads = FIXED_GROUPS * GROUP_SIZE
        padded = numpy.zeros(-(-values.size // threads) * threads)
        padded[: values.size] = values
        per_thread = padded.reshape(-1, threads).sum(axis=0)
        return per_thread.reshape(groups, GROUP_SIZE).sum(axis=1).astype(numpy.float32)
    span = GROUP_SIZE if kernel.mapping == "one_element_per_thread" else 2 * GROUP_SIZE
    padded = numpy.zeros(groups * span)
    padded[: values.size] = values
    return padded.reshape(groups, span).sum(axis=1).astype(numpy.float32)


class StandInBackend:
    """A NumPy stand-in for ``GpuReduceRuntime``; its reports are synthetic.

    Each kernel returns the float32 rounding of the exact sum and its group
    sums as partials. ``leak`` kernels write past their partials, ``hole``
    kernels leave a partial unwritten, ``spill`` kernels write past their
    output, ``scale`` kernels multiply the result, ``drift`` kernels move it
    by one ulp, and ``flaky`` kernels change a partial bit on every other run.
    """

    backend = "stand-in"
    torch_version = "2.15.0"

    def __init__(self, *, leak=(), hole=(), spill=(), scale=None, drift=(), flaky=()):
        self.limits = {
            name: PipelineLimits(1024, 32, kernel.threadgroup_bytes)
            for name, kernel in KERNELS.items()
        }
        self.admissions = {
            name: KernelAdmission(KERNELS[name], self.limits[name]) for name in KERNELS
        }
        self.mps_switches = {
            "PYTORCH_MPS_FAST_MATH": None,
            "PYTORCH_MPS_PREFER_METAL": None,
        }
        self.library_source = library_source()
        self.leak, self.hole, self.spill = set(leak), set(hole), set(spill)
        self.scale, self.drift, self.flaky = dict(scale or {}), set(drift), set(flaky)
        self.runs, self.uploads, self.workloads = [], [], []
        self._count = itertools.count()

    def synchronize(self):
        pass

    def _total(self, name, values):
        total = numpy.float32(values.astype(numpy.float64).sum())
        if name in self.drift:
            total = numpy.nextafter(total, numpy.float32(numpy.inf))
        if name in self.scale:
            total = numpy.float32(total * numpy.float32(self.scale[name]))
        return total

    def run(self, name, values):
        self.runs.append(name)
        partials = group_sums(name, values)
        if name in self.flaky and next(self._count) % 2:
            partials[0] = numpy.nextafter(partials[0], numpy.float32(numpy.inf))
        buffer = numpy.concatenate([partials, float32_sentinels(PARTIAL_GUARD)])
        output = float32_sentinels(OUTPUT_SLOTS)
        output[0] = self._total(name, values)
        if name in self.leak:
            buffer[partials.size] = 0.0
        if name in self.hole:
            buffer[0] = float32_sentinels(1)[0]
        if name in self.spill:
            output[1] = 0.0
        return ReductionBuffers(buffer, output)

    def upload(self, values):
        self.uploads.append(values.size)
        return values

    def workload(self, source, n):
        self.workloads.append(n)
        values = source[:n]

        def side():
            return None

        def check():
            return {
                name: SideResult(float(self._total(name, values)), True)
                for name in SIDES
            }

        return ReduceWorkload(
            sides={name: side for name in SIDES},
            synchronize=self.synchronize,
            check=check,
        )


def run_case(backend, pattern="normal", n=33, index=1, runs=2):
    return check_case(
        backend, seed=78, index=index, case=ReduceCase(pattern, n), runs=runs
    )


def test_every_kernel_records_its_launch_depth_bound_and_measured_bits():
    backend = StandInBackend()

    result = run_case(backend, "mixed", 300007)
    mapping = result.to_mapping()
    values = case_values(78, 1, "mixed", 300007)
    sums = exact_sums(values)

    assert backend.runs == [name for name in STEP_KERNELS for _ in range(2)]
    assert mapping["input_sha256"] == array_sha256(values)
    assert mapping["reference"] == sums.reference
    assert mapping["exact_in_any_order"] is False
    assert [output["kernel"] for output in mapping["outputs"]] == list(STEP_KERNELS)
    for output in mapping["outputs"]:
        name = output["kernel"]
        assert output["partials"] == partial_count(name, 300007)
        assert output["launch"]["groups"] == [output["partials"]]
        assert output["depth"] == summation_depth(name, 300007)
        assert output["same_bits_every_run"] is True
        assert output["output"] == float(numpy.float32(sums.reference))
        assert output["error"] == float(abs(Fraction(output["output"]) - sums.total))
        assert 0.0 <= output["bound_ratio"] <= 1.0
        assert (output["partials_guard"], output["output_guard"]) == (64, 31)
    assert mapping["same_order"] == [
        {"kernel": R3, "same_order_as": R2, "same_bits": True}
    ]


@pytest.mark.parametrize("pattern", ["ones", "halves"])
def test_an_exact_case_records_the_exact_sum_for_every_kernel(pattern):
    result = run_case(StandInBackend(), pattern, 1000003)

    assert result.sums.exact_in_any_order
    for output in result.to_mapping()["outputs"]:
        assert output["exact_result"] is True and output["error"] == 0.0


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (StandInBackend(leak={R2}), "wrote past its 67 partials"),
        (StandInBackend(hole={R5}), "left a partial slot unwritten"),
        (StandInBackend(spill={R3}), "wrote past its one output"),
        (StandInBackend(scale={R2: 1e4}), "from the exact sum; the bound"),
        (StandInBackend(scale={R2: float("nan")}), "a read past n adds a NaN"),
    ],
)
def test_a_broken_guard_or_bound_stops_the_case(backend, message):
    with pytest.raises(GpuReduceError, match=message):
        run_case(backend, "normal", 33793)


def test_an_exact_case_off_by_one_ulp_stops_even_inside_the_bound():
    with pytest.raises(GpuReduceError, match="and the sum is exact"):
        run_case(StandInBackend(drift={R5}), "ones", 1000003)


def test_changed_bits_between_runs_and_orders_are_recorded_not_refused():
    result = run_case(StandInBackend(flaky={R5}, drift={R3}), "normal", 300007, runs=3)
    outputs = {output["kernel"]: output for output in result.to_mapping()["outputs"]}

    assert outputs[R5]["same_bits_every_run"] is False
    assert all(
        outputs[name]["same_bits_every_run"] for name in STEP_KERNELS if name != R5
    )
    assert result.same_order() == [
        {"kernel": R3, "same_order_as": R2, "same_bits": False}
    ]


def test_every_configured_case_runs_in_order_under_the_budget():
    config = config_from_mapping(make_config_mapping())
    backend = StandInBackend()
    ticks = itertools.count(0, 1_000)

    results = check_cases(
        backend,
        seed=config.seed,
        controls=config.correctness,
        budget=RunBudget(60, ticks.__next__),
    )

    assert [result.case for result in results] == list(config.correctness.cases)
    assert [result.index for result in results] == list(range(6))
    assert len(backend.runs) == 6 * 6 * 2
    with pytest.raises(TypeError, match="RunBudget"):
        check_cases(backend, seed=78, controls=config.correctness, budget=None)


@requires_metal
@pytest.mark.parametrize(
    ("pattern", "n"), [("positive", 1000003), ("halves", 1000003), ("normal", 31)]
)
def test_every_kernel_passes_its_bound_and_repeats_its_bits_on_this_gpu(pattern, n):
    result = run_case(reduce_runtime(), pattern, n, index=2, runs=3)

    for output in result.to_mapping()["outputs"]:
        assert output["bound_ratio"] <= 1.0, output
        # Observed on the development host; a run records it per case.
        assert output["same_bits_every_run"] is True, output
    assert result.same_order()[0]["same_bits"] is True
