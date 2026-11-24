import numpy
import pytest

from ai_journey.metal_flash_config import FlashCase
from ai_journey.metal_flash_correctness import check_case, check_probe
from ai_journey.metal_flash_source import F1, MODES, N0, VARIANTS, MetalFlashError
from test_metal_flash_emulator import FlashEmulator

CASE = FlashCase((1, 31, 33, 70), "wide_scores")


def test_the_emulator_passes_every_rule_and_records_each_side():
    record = check_case(FlashEmulator(), seed=81, index=0, case=CASE, repeats=2)
    mapping = record.to_mapping()

    assert mapping["layout"]["gap"] == 3
    assert set(mapping) >= {"q_sha256", "k_sha256", "v_sha256", "modes"}
    for mode in MODES:
        entry = mapping["modes"][mode]
        assert entry["unscaled_exp_overflows"] is True
        for variant in VARIANTS:
            output = entry["outputs"][variant]
            assert output["identical_repeats"] is True
            assert 0 <= output["max_bound_ratio"] <= 1
            assert output["launches"][0]["kernel"].startswith(f"flash_{variant}")


def store_in_gap(side, output):
    output = output.copy()
    output[-1] = 0.0
    return output


def far_off(side, output):
    output = output.copy()
    output[0] += 1.0
    return output


def not_finite(side, output):
    output = output.copy()
    output[1] = numpy.nan
    return output


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [(store_in_gap, "gap row"), (far_off, "exceeds its bound"), (not_finite, "finite")],
)
def test_a_side_that_breaks_a_rule_stops_the_run(corrupt, match):
    with pytest.raises(MetalFlashError, match=match):
        check_case(
            FlashEmulator(corrupt=corrupt), seed=81, index=0, case=CASE, repeats=2
        )


def test_repeats_that_differ_are_recorded_as_a_measurement():
    calls = {"count": 0}

    def wobble(side, output):
        calls["count"] += 1
        if side == F1 and calls["count"] % 2 == 0:
            output = output.copy()
            output[0] = numpy.nextafter(output[0], numpy.float32(numpy.inf))
        return output

    record = check_case(
        FlashEmulator(corrupt=wobble), seed=81, index=0, case=CASE, repeats=2
    )
    outputs = record.to_mapping()["modes"]["full"]["outputs"]
    assert outputs[N0]["identical_repeats"] is True
    assert outputs[F1]["identical_repeats"] is False


def test_a_wrong_buffer_shape_is_refused():
    class Short(FlashEmulator):
        def run(self, inputs, variant, mode):
            return super().run(inputs, variant, mode)[:-1]

    with pytest.raises(MetalFlashError, match="no float32 buffer"):
        check_case(Short(), seed=81, index=0, case=CASE, repeats=2)


def test_an_exp_probe_that_breaks_the_bound_assumption_stops_the_run():
    class Loose(FlashEmulator):
        def exp_probe(self, arguments):
            return super().exp_probe(arguments) * numpy.float32(1.001)

    assert check_probe(FlashEmulator())["holds"] is True
    with pytest.raises(MetalFlashError, match="exp probe"):
        check_probe(Loose())
