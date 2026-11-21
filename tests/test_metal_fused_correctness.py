import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_fused_config import CorrectnessCase
from ai_journey.metal_fused_correctness import (
    RULES,
    check_case,
    check_exp_probe,
    runnable,
)
from ai_journey.metal_fused_plan import TORCH_SOFTMAX
from ai_journey.metal_fused_reference import (
    CORRECTNESS_STREAM,
    EXP_PROBE_COUNT,
    OFFSET_PLUS_1000,
    UNIFORM_80,
    error_bounds,
    float32_sha256,
    float64_sha256,
    sentinel_buffer,
    softmax_inputs,
    softmax_reference,
)
from ai_journey.metal_fused_runtime import FusedWorkload, admit
from ai_journey.metal_fused_source import (
    F1,
    F2,
    F3,
    KERNELS,
    STEP_VARIANTS,
    U0,
    MetalFusedError,
    fused_library,
)
from test_kernel_metal_runtime import requires_metal
from test_metal_fused_emulator import TINY, emulate
from test_metal_fused_runtime import fused_runtime

RAGGED = CorrectnessCase((1, 31, 33, 1000, 7), UNIFORM_80)
OFFSET = CorrectnessCase((64, 64), OFFSET_PLUS_1000, 64)


def recorded_limits(**changes):
    limits = {
        name: PipelineLimits(1024, 32, kernel.threadgroup_bytes)
        for name, kernel in KERNELS.items()
    }
    limits.update(changes)
    return limits


class StandInBackend:
    """Emulates every step on the CPU; not the real runtime, so never a measurement.

    ``faults`` maps a step to one way of going wrong: ``gap`` stores into a
    gap slot, ``nan`` writes a NaN, ``off`` moves one output by 2**-10,
    ``drift`` changes a bit in the second and later runs, ``torch_off`` moves
    the torch output at the timing points, and ``probe`` breaks the probe.
    """

    def __init__(self, faults=None, limits=None, torch_version="2.15.0"):
        self.faults = dict(faults or {})
        self.limits = limits or recorded_limits()
        self.torch_version = torch_version
        self.mps_switches = {
            "PYTORCH_MPS_FAST_MATH": None,
            "PYTORCH_MPS_PREFER_METAL": None,
        }
        self.library_source = fused_library()
        self.runs = []
        self.calls = {}

    def run(self, variant, values, layout, required):
        assert admit(self.limits, required)[variant].admitted
        count = sum(1 for name, _ in self.runs if name == variant)
        self.runs.append((variant, layout))
        output = emulate(variant, values, layout)
        fault = self.faults.get(variant)
        valid = layout.valid()
        if fault == "gap":
            output[valid[-1] + 1] = 0.5
        elif fault == "nan":
            output[valid[0]] = numpy.nan
        elif fault == "off":
            output[valid[-1]] += numpy.float32(2.0**-10)
        elif fault == "drift" and count:
            bits = output.view(numpy.uint32)
            bits[valid[-1]] ^= 1
        return output

    def exp_probe(self, arguments):
        with numpy.errstate(under="ignore"):
            results = numpy.exp(arguments.astype(numpy.float64)).astype(numpy.float32)
        results[numpy.abs(results) < TINY] = 0
        if self.faults.get("probe"):
            results[-9] = numpy.float32(1.001)
        return results

    def workload(self, values, layout, names):
        assert layout.gap == 0
        sides = {}
        for name in (*names, TORCH_SOFTMAX):
            self.calls[name] = 0

            def call(name=name):
                self.calls[name] += 1

            sides[name] = call

        def outputs():
            results = {name: emulate(name, values, layout) for name in names}
            reference = softmax_reference(values, layout).output
            results[TORCH_SOFTMAX] = reference.astype(numpy.float32)
            if self.faults.get("torch_off"):
                results[TORCH_SOFTMAX][0] += numpy.float32(2.0**-8)
            for name in names:
                if self.faults.get(name) == "off":
                    results[name][-1] += numpy.float32(2.0**-10)
            return results

        return FusedWorkload(sides=sides, synchronize=lambda: None, outputs=outputs)


def run_case(backend, case=RAGGED, names=STEP_VARIANTS, repeats=2, index=0):
    return check_case(
        backend,
        seed=79,
        index=index,
        case=case,
        names=names,
        repeats=repeats,
        required=1024,
    )


def test_a_case_records_each_step_with_digests_the_verifier_can_rebuild():
    backend = StandInBackend()
    record = run_case(backend).to_mapping()

    layout = RAGGED.layout()
    values = softmax_inputs(79, CORRECTNESS_STREAM, 0, layout, UNIFORM_80)
    reference = softmax_reference(values, layout)
    assert record["input_sha256"] == float32_sha256(values)
    assert record["reference_sha256"] == float64_sha256(reference.output)
    assert record["layout"] == layout.to_mapping()
    assert record["launch"]["group_size"] == [64]
    assert record["launch"]["groups"] == [5]
    assert record["naive_float32_finite"] is True
    assert [output["kernel"] for output in record["outputs"]] == list(STEP_VARIANTS)
    for output in record["outputs"]:
        bounds = error_bounds(output["kernel"], reference)
        assert output["bound_sha256"] == float64_sha256(bounds.elements)
        assert output["row_sum_bound_sha256"] == float64_sha256(bounds.row_sums)
        assert output["identical_repeats"] is True
        assert 0 <= output["max_bound_ratio"] <= 1
        assert 0 <= output["max_row_sum_ratio"] <= 1
        assert 0 <= output["differing_from_rounded_reference"] <= layout.elements
    assert record["same_bits"] == [{"variant": F1, "as": U0, "same": True}]
    assert len(backend.runs) == 2 * len(STEP_VARIANTS)


def test_the_offset_case_records_that_no_max_subtraction_would_fail():
    record = run_case(StandInBackend(), case=OFFSET, index=1).to_mapping()

    assert record["naive_float32_finite"] is False
    assert record["case"] == {"rows": 2, "cols": 64, "inputs": OFFSET_PLUS_1000}


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("gap", "escaped into a gap"),
        ("nan", "not finite"),
        ("off", "exceeds its bound"),
    ],
)
def test_a_case_that_breaks_a_rule_stops_the_run(fault, message):
    with pytest.raises(MetalFusedError, match=f"{F2}: .*{message}"):
        run_case(StandInBackend(faults={F2: fault}))


def test_a_step_that_changes_bits_between_runs_is_measured_not_refused():
    record = run_case(StandInBackend(faults={F1: "drift"}), repeats=3).to_mapping()

    drift = {
        output["kernel"]: output["identical_repeats"] for output in record["outputs"]
    }
    assert drift == {U0: True, F1: False, F2: True, F3: True}
    assert record["repeats"] == 3


def test_a_length_one_row_matches_the_rounded_reference_exactly():
    case = CorrectnessCase((1, 1, 1), UNIFORM_80, 1)

    record = run_case(StandInBackend(), case=case).to_mapping()

    layout = case.layout()
    expected = sentinel_buffer(layout)
    expected[layout.valid()] = 1.0
    for output in record["outputs"]:
        assert output["differing_from_rounded_reference"] == 0
        assert output["buffer_sha256"] == float32_sha256(expected)
        assert output["max_abs_error"] == 0.0


def test_same_bits_is_recorded_only_when_both_steps_ran():
    record = run_case(StandInBackend(), names=(F1, F2, F3)).to_mapping()

    assert record["same_bits"] == []


def test_the_exp_probe_is_measured_once_and_stops_a_broken_assumption():
    record = check_exp_probe(StandInBackend()).to_mapping()

    assert record["count"] == EXP_PROBE_COUNT and record["holds"] is True
    assert record["exp_ulps"] == 4 and record["max_ulps"] <= 0.5
    with pytest.raises(MetalFusedError, match="breaks the assumption"):
        check_exp_probe(StandInBackend(faults={"probe": True}))


def test_only_admitted_steps_may_run():
    refused = admit(recorded_limits(**{F3: PipelineLimits(512, 32, 256)}), 1024)

    assert runnable(refused, (U0, F1, F2)) == (U0, F1, F2)
    with pytest.raises(MetalFusedError, match="refused by its pipeline"):
        runnable(refused, (U0, F3))
    with pytest.raises(MetalFusedError, match="registered step"):
        runnable(refused, (TORCH_SOFTMAX,))
    assert "measured" in RULES


@requires_metal
def test_every_step_passes_the_rules_on_this_gpu():
    runtime = fused_runtime()

    record = run_case(runtime, repeats=2).to_mapping()

    assert [output["kernel"] for output in record["outputs"]] == list(STEP_VARIANTS)
    for output in record["outputs"]:
        assert output["max_bound_ratio"] <= 1
        assert type(output["identical_repeats"]) is bool
    assert type(record["same_bits"][0]["same"]) is bool
    assert check_exp_probe(runtime).to_mapping()["holds"] is True
