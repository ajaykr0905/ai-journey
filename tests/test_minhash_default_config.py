from pathlib import Path

from ai_journey.minhash_config import load_minhash_policy
from ai_journey.minhash_lsh import LSHBanding


ROOT = Path(__file__).resolve().parents[1]


def test_day66_policy_pins_candidate_recall_and_exact_acceptance_threshold():
    policy = load_minhash_policy(ROOT / "config" / "day-66-minhash-dedup.json")
    banding = LSHBanding(policy.num_hashes, policy.bands)

    assert policy.shingle_width == 5
    assert policy.num_hashes == 128
    assert policy.seed == 20261109
    assert policy.bands == 16
    assert policy.duplicate_threshold == 0.8
    assert banding.rows_per_band == 8
    assert banding.collision_probability(policy.duplicate_threshold) > 0.94
    assert banding.collision_probability(0.5) < 0.07
