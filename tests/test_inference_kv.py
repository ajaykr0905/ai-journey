import pytest

from ai_journey.inference_kv import InferenceCapacityError, KVCacheLayout


def test_7b_mha_cache_at_32k_context_and_batch_16_is_256_gib():
    layout = KVCacheLayout(
        layers=32,
        kv_heads=32,
        head_dim=128,
        context_tokens=32_768,
        batch_size=16,
    )

    assert layout.bytes_per_token_per_sequence == 524_288
    assert layout.total_bytes == 256 * 1024**3


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("layers", True, TypeError),
        ("kv_heads", 0, InferenceCapacityError),
        ("head_dim", -1, InferenceCapacityError),
        ("bytes_per_element", 0, InferenceCapacityError),
    ],
)
def test_kv_layout_rejects_ambiguous_or_nonpositive_dimensions(field, value, exception):
    values = {
        "layers": 32,
        "kv_heads": 32,
        "head_dim": 128,
        "context_tokens": 32_768,
        "batch_size": 16,
    }
    values[field] = value

    with pytest.raises(exception):
        KVCacheLayout(**values)
