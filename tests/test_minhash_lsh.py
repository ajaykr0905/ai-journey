import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_lsh import LSHBanding, band_keys


def test_banding_exposes_rows_threshold_and_collision_curve():
    banding = LSHBanding(num_hashes=128, bands=16)

    assert banding.rows_per_band == 8
    assert banding.approximate_threshold == pytest.approx(0.7071067812)
    assert banding.collision_probability(0.0) == 0.0
    assert banding.collision_probability(1.0) == 1.0
    assert banding.collision_probability(0.8) > banding.collision_probability(0.6)


@pytest.mark.parametrize(
    ("hashes", "bands"),
    [(15, 1), (513, 1), (16, 0), (16, 17), (18, 4), (16, True)],
)
def test_bands_must_exactly_partition_the_signature(hashes, bands):
    with pytest.raises(TokenizerError, match="LSH"):
        LSHBanding(hashes, bands)


@pytest.mark.parametrize("similarity", [True, -0.1, 1.1, float("nan")])
def test_collision_curve_rejects_invalid_similarity(similarity):
    with pytest.raises(TokenizerError, match="similarity"):
        LSHBanding(16, 4).collision_probability(similarity)


def test_band_keys_bind_band_position_and_signature_values():
    banding = LSHBanding(16, 4)
    signature = tuple(range(16))

    keys = band_keys(signature, banding)

    assert len(keys) == 4
    assert [band for band, _ in keys] == [0, 1, 2, 3]
    assert all(len(digest) == 64 for _, digest in keys)
    changed = band_keys(tuple(range(15)) + (99,), banding)
    assert changed[:3] == keys[:3]
    assert changed[3] != keys[3]


@pytest.mark.parametrize(
    "signature", [tuple(range(15)), tuple(range(17)), (0,) * 15 + (-1,)]
)
def test_band_keys_reject_signatures_outside_the_contract(signature):
    with pytest.raises(TokenizerError, match="does not match"):
        band_keys(signature, LSHBanding(16, 4))


def test_band_keys_require_a_banding_contract():
    with pytest.raises(TypeError, match="LSHBanding"):
        band_keys((0,) * 16, object())
