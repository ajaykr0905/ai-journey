import pytest

from ai_journey.serving_analysis import ServingAnalysisError, nearest_rank_percentile


def test_nearest_rank_percentile_is_deterministic_for_small_samples():
    values = (40, 10, 30, 20)

    assert nearest_rank_percentile(values, 50) == 20
    assert nearest_rank_percentile(values, 95) == 40
    assert nearest_rank_percentile(values, 100) == 40


@pytest.mark.parametrize("values,percentile", [((), 50), ((1,), 0), ((-1,), 50)])
def test_nearest_rank_percentile_rejects_undefined_inputs(values, percentile):
    with pytest.raises((ServingAnalysisError, TypeError)):
        nearest_rank_percentile(values, percentile)
