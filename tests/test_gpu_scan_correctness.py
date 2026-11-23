import numpy
import pytest

from ai_journey.gpu_scan_correctness import check_case, runnable
from ai_journey.gpu_scan_plan import COPY, VENDOR
from ai_journey.gpu_scan_reference import (
    CORRECTNESS_STREAM,
    INTEGERS,
    NORMAL,
    REPLAYS,
    error_bound,
    exact_prefix,
    float64_reference,
    float64_sha256,
    scan_inputs,
)
from ai_journey.gpu_scan_runtime import (
    ScanAdmission,
    ScanWorkload,
    padded_copy_input,
)
from ai_journey.gpu_scan_source import (
    ADD_OFFSETS,
    KERNELS,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
    guard_elements,
)
from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.kernel_metal_reference import array_sha256, float32_sentinels
from test_gpu_scan_runtime import scan_runtime
from test_kernel_metal_runtime import requires_metal


def rounded(values):
    """The float64 sum rounded once: what an exact order would give."""

    return float64_reference(values)[0].astype(numpy.float32)


class StandInBackend:
    """A NumPy stand-in for ``ScanRuntime``; its reports are synthetic.

    S0 and S1 return their CPU replays, the bits the development GPU gave in
    every case. S2 and ``torch.cumsum`` return the float64 sum rounded once.
    ``leak`` kernels write into their tail, ``scale`` kernels multiply their
    output, ``wobble`` sides move one element by one ulp on odd runs, and
    ``nudge`` sides move it on every run of an input that is not all integers.
    """

    backend = "stand-in"
    torch_version = "2.15.0"

    def __init__(self, *, threads=None, memory=None, widths=None, **faults):
        threads, memory, widths = threads or {}, memory or {}, widths or {}
        self.limits = {
            name: PipelineLimits(
                threads.get(name, 1024),
                widths.get(name, 32),
                memory.get(name, KERNELS[name].threadgroup_bytes),
            )
            for name in KERNELS
        }
        for helper in (ADD_OFFSETS, COPY):
            self.limits[helper] = PipelineLimits(1024, 32, 0)
        self.admissions = {
            name: ScanAdmission(KERNELS[name], self.limits[name]) for name in KERNELS
        }
        self.mps_switches = {
            "PYTORCH_MPS_FAST_MATH": None,
            "PYTORCH_MPS_PREFER_METAL": None,
        }
        self.leak = set(faults.get("leak", ()))
        self.scale = dict(faults.get("scale", {}))
        self.wobble = set(faults.get("wobble", ()))
        self.nudge = set(faults.get("nudge", ()))
        self.vendor_nan = faults.get("vendor_nan", False)
        self.runs = []

    def synchronize(self):
        pass

    def _data(self, name, values):
        data = REPLAYS[name](values) if name in REPLAYS else rounded(values)
        if name in self.scale:
            data = data * numpy.float32(self.scale[name])
        count = sum(1 for side, _ in self.runs if side == name)
        whole = numpy.array_equal(values, numpy.round(values))
        if (name in self.wobble and count % 2) or (name in self.nudge and not whole):
            data[-1] = numpy.nextafter(data[-1], numpy.float32(numpy.inf))
        self.runs.append((name, values.size))
        return data

    def scan(self, name, values):
        if not self.admissions[name].admitted:
            raise GpuScanError(f"{name} was refused by its pipeline")
        tail = float32_sentinels(guard_elements(values.size))
        if name in self.leak:
            tail[-1] = 0.0
        return numpy.concatenate([self._data(name, values), tail])

    def cumsum(self, values):
        data = self._data(VENDOR, values)
        if self.vendor_nan:
            data[0] = numpy.nan
        return data

    def workload(self, values, names):
        for name in names:
            if not self.admissions[name].admitted:
                raise GpuScanError(f"{name} was refused by its pipeline")

        def side():
            return None

        def check():
            outputs = {name: self._data(name, values) for name in (*names, VENDOR)}
            outputs[COPY] = padded_copy_input(values)
            return outputs

        return ScanWorkload(
            sides={name: side for name in (*names, VENDOR, COPY)},
            synchronize=self.synchronize,
            check=check,
        )


def run_case(backend, length=1025, pattern=NORMAL, index=1, runs=3, names=None):
    return check_case(
        backend,
        seed=78,
        index=index,
        length=length,
        pattern=pattern,
        names=tuple(KERNELS) if names is None else names,
        runs=runs,
    )


def test_every_side_records_each_run_its_bound_and_its_measured_bits():
    backend = StandInBackend()

    case = run_case(backend, 4097).to_mapping()
    values = scan_inputs(78, CORRECTNESS_STREAM, 1, 4097, NORMAL)
    reference, _ = float64_reference(values)

    assert [output["side"] for output in case["outputs"]] == [*KERNELS, VENDOR]
    assert case["levels"] == [4097, 5]
    assert case["input_sha256"] == array_sha256(values)
    assert case["reference_sha256"] == float64_sha256(reference)
    assert case["exact_sha256"] is None
    tail = float32_sentinels(guard_elements(4097))
    for output in case["outputs"][:3]:
        name = output["side"]
        assert output["guard_elements"] == guard_elements(4097) == 2047
        assert output["bound"] == error_bound(name, 4097)
        assert 0.0 < output["bound_ratio"] < 0.2
        assert output["deterministic"] and len(output["run_sha256"]) == 3
        assert output["exact_bits"] is None
        if name in REPLAYS:
            replay = REPLAYS[name](values)
            assert case["replay_sha256"][name] == array_sha256(replay)
            assert output["buffer_sha256"] == array_sha256(
                numpy.concatenate([replay, tail])
            )
            assert (output["replay_bits"], output["replay_differences"]) == (True, 0)
        else:
            assert (output["replay_bits"], output["replay_differences"]) == (None, None)
    vendor = case["outputs"][3]
    assert (vendor["bound"], vendor["bound_ratio"], vendor["guard_elements"]) == (
        None,
        None,
        None,
    )
    assert set(case["replay_sha256"]) == {S0_HILLIS_STEELE, S1_BLELLOCH}
    assert [level["blocks"] for level in case["launches"][S1_BLELLOCH]] == [5, 1]
    assert case["launches"][S1_BLELLOCH][0]["scan"]["threads"] == [5 * 512]
    assert case["launches"][S0_HILLIS_STEELE][0]["add"]["threads"] == [4352]
    assert case["launches"][S2_SIMD][1]["add"] is None


def test_an_integer_case_needs_the_exact_sums_and_measures_the_vendor():
    backend = StandInBackend()

    case = run_case(backend, 70001, INTEGERS).to_mapping()
    values = scan_inputs(78, CORRECTNESS_STREAM, 1, 70001, INTEGERS)

    assert case["exact_sha256"] == array_sha256(exact_prefix(values))
    assert [output["exact_bits"] for output in case["outputs"]] == [True] * 4
    assert all(output["max_relative_error"] == 0.0 for output in case["outputs"])
    # One ulp on one run stays within the bound but misses an exact sum.
    with pytest.raises(GpuScanError, match="missed the exact integer sums"):
        run_case(StandInBackend(wobble={S2_SIMD}), 70001, INTEGERS)


def test_a_change_of_bits_from_run_to_run_is_measured_not_refused():
    backend = StandInBackend(wobble={S2_SIMD, VENDOR})

    case = run_case(backend, 1000).to_mapping()
    outputs = {output["side"]: output for output in case["outputs"]}

    assert outputs[S2_SIMD]["deterministic"] is False
    assert outputs[VENDOR]["deterministic"] is False
    assert outputs[S0_HILLIS_STEELE]["deterministic"] is True
    first, second, third = outputs[S2_SIMD]["run_sha256"]
    assert first == third != second
    assert outputs[S2_SIMD]["buffer_sha256"] == first


@pytest.mark.parametrize(
    ("faults", "message"),
    [
        ({"leak": {S1_BLELLOCH}}, "wrote into its guard tail"),
        ({"scale": {S0_HILLIS_STEELE: 1.001}}, "above its bound"),
        ({"vendor_nan": True}, "not finite"),
    ],
)
def test_a_wrong_output_stops_the_case(faults, message):
    with pytest.raises(GpuScanError, match=message):
        run_case(StandInBackend(**faults), 1025)


def test_a_refused_kernel_never_runs_and_has_no_record():
    backend = StandInBackend(widths={S2_SIMD: 16}, threads={S1_BLELLOCH: 256})

    case = run_case(backend, 33).to_mapping()

    assert backend.admissions[S2_SIMD].reason == "simd_width_is_not_32"
    assert backend.admissions[S1_BLELLOCH].reason == (
        "pipeline_allows_fewer_threads_than_the_group"
    )
    assert runnable(backend, KERNELS) == (S0_HILLIS_STEELE,)
    assert [output["side"] for output in case["outputs"]] == [S0_HILLIS_STEELE, VENDOR]
    assert list(case["launches"]) == [S0_HILLIS_STEELE]
    assert {name for name, _ in backend.runs} == {S0_HILLIS_STEELE, VENDOR}
    with pytest.raises(GpuScanError, match="not a registered"):
        runnable(backend, ("scan_s9_other",))


@pytest.mark.parametrize("runs", [1, 6, True])
def test_a_case_needs_two_to_five_runs(runs):
    with pytest.raises(GpuScanError, match="2 to 5 times"):
        run_case(StandInBackend(), runs=runs)


@requires_metal
@pytest.mark.parametrize(
    ("length", "pattern"), [(1, NORMAL), (1000, NORMAL), (65537, INTEGERS)]
)
def test_this_gpu_passes_every_rule_and_holds_the_replay_bits(length, pattern):
    case = run_case(scan_runtime(), length, pattern).to_mapping()

    for output in case["outputs"][:3]:
        assert output["deterministic"] is True, output["side"]
        assert output["replay_bits"] in (True, None), output["side"]
        assert output["bound_ratio"] <= 1.0
    assert case["outputs"][3]["side"] == VENDOR
