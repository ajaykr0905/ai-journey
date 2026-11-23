from pathlib import Path

import numpy
import pytest

from ai_journey.metal_liger_config import load_config
from ai_journey.metal_liger_correctness import check_case, check_probes
from ai_journey.metal_liger_source import CROSS_ENTROPY, FAMILIES, MetalLigerError
from test_kernel_metal_runtime import requires_metal
from test_metal_liger_emulator import LigerEmulator
from test_metal_liger_runtime import liger_runtime

CONFIG = load_config(
    Path(__file__).resolve().parents[1] / "config" / "day-80-metal-liger.json"
)


def _cases():
    return [
        (family, index, case)
        for family in FAMILIES
        for index, case in enumerate(CONFIG.correctness.cases[family])
    ]


def _check(backend, family, index, case, repeats=2):
    return check_case(
        backend,
        seed=CONFIG.seed,
        index=index,
        case=case,
        repeats=repeats,
        required=1024,
    )


@pytest.mark.parametrize(("family", "index", "case"), _cases()[::3])
def test_the_stand_in_passes_the_day_80_cases_and_repeats_its_bits(family, index, case):
    record = _check(LigerEmulator(), family, index, case).to_mapping()

    assert record["family"] == family and record["index"] == index
    assert record["output"]["identical_repeats"] is True
    assert 0 <= record["output"]["max_bound_ratio"] <= 1
    assert (record["naive_float32_finite"] is None) is (family != CROSS_ENTROPY)
    assert record["launch"]["threads_per_group"] == record["layout"]["group_size"]


def test_the_naive_cross_entropy_fails_on_the_large_logit_case():
    case = CONFIG.correctness.cases[CROSS_ENTROPY][1]

    assert case.inputs == "offset_plus_1000"
    assert _check(LigerEmulator(), CROSS_ENTROPY, 1, case).naive_float32_finite is False


def _corrupt(change):
    def corrupt(family, output):
        output = output.copy()
        change(output)
        return output

    return LigerEmulator(corrupt=corrupt)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda out: out.__setitem__(-1, 0.0), "escaped into a gap"),
        (lambda out: out.__setitem__(0, numpy.inf), "not finite"),
        (lambda out: out.__setitem__(0, out[0] + 1.0), "exceeds its bound"),
    ],
)
def test_a_broken_output_stops_the_run(change, message):
    family, index, case = _cases()[0]

    with pytest.raises(MetalLigerError, match=message):
        _check(_corrupt(change), family, index, case)


def test_a_repeat_that_changes_bits_within_the_bound_is_kept_as_data():
    calls = {"count": 0}

    def corrupt(family, output):
        calls["count"] += 1
        if calls["count"] == 2:
            output = output.copy()
            output[0] = numpy.nextafter(output[0], numpy.float32(numpy.inf))
        return output

    family, index, case = _cases()[0]
    record = _check(LigerEmulator(corrupt=corrupt), family, index, case)

    assert record.output.identical_repeats is False


def test_a_probe_that_breaks_its_assumption_stops_the_run():
    class LooseRsqrt(LigerEmulator):
        def probe(self, function, arguments):
            result = super().probe(function, arguments)
            return result * numpy.float32(1.001) if function == "rsqrt" else result

    records = check_probes(LigerEmulator())
    assert set(records) == {"exp", "rsqrt", "log"}
    assert all(record["holds"] for record in records.values())
    with pytest.raises(MetalLigerError, match="rsqrt probe breaks"):
        check_probes(LooseRsqrt())


@requires_metal
@pytest.mark.parametrize(("family", "index", "case"), _cases())
def test_every_day_80_case_passes_on_the_apple_gpu(family, index, case):
    record = _check(liger_runtime(), family, index, case, repeats=3)

    assert record.output.max_bound_ratio <= 1
    assert record.output.identical_repeats is True


@requires_metal
def test_the_apple_gpu_meets_every_probe_assumption():
    records = check_probes(liger_runtime())

    assert records["rsqrt"]["max_ulps"] <= 2 and records["log"]["exact_zero"] is True
