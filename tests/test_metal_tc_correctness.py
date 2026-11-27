import numpy
import pytest

from ai_journey.metal_tc_config import TcCase
from ai_journey.metal_tc_correctness import check_case
from ai_journey.metal_tc_reference import SENTINEL_BITS
from ai_journey.metal_tc_source import M2, M3, S1, VARIANTS, MetalTcError
from test_kernel_metal_runtime import requires_metal
from test_metal_tc_emulator import TcEmulator
from test_metal_tc_runtime import tc_runtime

CASE = TcCase(33, 65, 17)


def test_a_case_records_digests_bounds_and_each_kernel_output():
    record = check_case(TcEmulator(), seed=82, index=2, case=CASE, repeats=2)

    assert set(record["outputs"]) == set(VARIANTS)
    assert set(record["inputs"]) == set(record["bound_sha256"])
    assert record["launch"]["groups"] == [2 * 3]
    for entry in record["outputs"].values():
        assert entry["identical_repeats"] is True
        assert 0 <= entry["max_bound_ratio"] <= 1
    assert record["m1_same_bits_as_s1"] is True
    outputs = record["outputs"]
    assert outputs[M3]["max_relative_error"] > outputs[M2]["max_relative_error"]
    assert outputs[M2]["max_relative_error"] > outputs[S1]["max_relative_error"]


def test_an_unsupported_kernel_has_no_output_in_the_case():
    record = check_case(
        TcEmulator(bfloat=False), seed=82, index=0, case=CASE, repeats=2
    )

    assert M3 not in record["outputs"]
    assert set(record["outputs"]) == set(VARIANTS) - {M3}


def _store_past_c(side, output):
    output = output.copy()
    output[-1] = 0.0
    return output


def _poison(side, output):
    output = output.copy()
    output[0] = numpy.nan
    return output


def _drift(side, output):
    output = output.copy()
    output[3] += 1e-3
    return output


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_store_past_c, "escaped into the tail"),
        (_poison, "not finite"),
        (_drift, "exceeds its bound"),
    ],
)
def test_a_broken_output_stops_the_run(corrupt, message):
    with pytest.raises(MetalTcError, match=message):
        check_case(TcEmulator(corrupt=corrupt), seed=82, index=2, case=CASE, repeats=2)


def test_a_repeat_that_changes_bits_is_measured_not_refused():
    counter = {"calls": 0}

    def flicker(side, output):
        counter["calls"] += 1
        if side == S1 and counter["calls"] == 1:
            output = output.copy()
            output[0] = numpy.nextafter(output[0], numpy.float32(numpy.inf))
        return output

    record = check_case(
        TcEmulator(corrupt=flicker), seed=82, index=2, case=CASE, repeats=2
    )

    assert record["outputs"][S1]["identical_repeats"] is False
    assert record["m1_same_bits_as_s1"] is False


@requires_metal
def test_this_gpu_passes_a_ragged_case_on_every_supported_kernel():
    runtime = tc_runtime()
    record = check_case(runtime, seed=82, index=3, case=TcCase(100, 129, 70), repeats=2)

    for variant in VARIANTS:
        if runtime.supported(variant):
            assert record["outputs"][variant]["max_bound_ratio"] <= 1
    assert SENTINEL_BITS == 0x7FC00000
