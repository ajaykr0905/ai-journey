import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_lsh import LSHBanding


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
