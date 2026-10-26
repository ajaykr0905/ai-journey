import copy
from dataclasses import replace

import pytest

from ai_journey.decoder_variant_plan import (
    DecoderExperimentPlan,
    decoder_plan_from_mapping,
)
from test_decoder_variant_integrity import config


def plan(**controls):
    base = config()
    return DecoderExperimentPlan(
        variants=(("baseline", base), ("rmsnorm", replace(base, norm_kind="rmsnorm"))),
        declared_changes=("norm_kind",),
        **controls,
    )


def test_plan_roundtrip_preserves_identity_and_all_controls():
    expected = plan()
    assert decoder_plan_from_mapping(expected.to_mapping()) == expected
    assert len(expected.fingerprint()) == 64
    assert (
        replace(expected, learning_rate=0.002).fingerprint() != expected.fingerprint()
    )


def test_plan_rejects_unexpected_interventions_and_unknown_fields():
    expected = plan()
    with pytest.raises(ValueError, match="undeclared"):
        replace(
            expected,
            variants=(
                expected.variants[0],
                ("combined", replace(expected.variants[1][1], position_kind="rope")),
            ),
        )
    for field in ("schema", "optimizer", "declared_changes"):
        payload = copy.deepcopy(expected.to_mapping())
        payload.pop(field)
        with pytest.raises(ValueError, match="fields"):
            decoder_plan_from_mapping(payload)
    payload = expected.to_mapping() | {"silent_extra_control": 1}
    with pytest.raises(ValueError, match="fields"):
        decoder_plan_from_mapping(payload)


@pytest.mark.parametrize(
    "controls",
    [
        {"seeds": (True,)},
        {"seeds": (1, 1)},
        {"steps": 0},
        {"learning_rate": float("nan")},
        {"evaluation_batch_size": 33},
    ],
)
def test_plan_rejects_invalid_or_unbounded_controls(controls):
    with pytest.raises(ValueError, match="plan"):
        plan(**controls)


def test_plan_rejects_inactive_controls_that_masquerade_as_architecture_changes():
    controls = plan()
    baseline = controls.variants[0]
    fake_arm = ("inactive-rope", replace(baseline[1], rope_base=20000.0))
    with pytest.raises(ValueError, match="inactive RoPE"):
        replace(
            controls, variants=(baseline, fake_arm), declared_changes=("rope_base",)
        )
    # A real RoPE arm can declare and vary its active frequency base.
    active = ("rope", replace(baseline[1], position_kind="rope", rope_base=20000.0))
    declared = replace(
        controls,
        variants=(baseline, active),
        declared_changes=("position_kind", "rope_base"),
    )
    assert decoder_plan_from_mapping(declared.to_mapping()) == declared
