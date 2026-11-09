import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_hash import hash_shingle, hash_shingles


def test_shingle_hash_has_stable_64_bit_golden_value():
    assert hash_shingle("public\x1fdata") == 13_664_215_493_144_161_150
    assert 0 <= hash_shingle("public\x1fdata") < 2**64


def test_hash_set_is_unique_and_order_independent():
    first = hash_shingles(("b", "a", "b"))
    second = hash_shingles(("a", "b"))

    assert first == second
    assert len(first) == 2


@pytest.mark.parametrize("value", ["", None, 1])
def test_shingle_hash_rejects_invalid_values(value):
    with pytest.raises(TokenizerError, match="shingle"):
        hash_shingle(value)


@pytest.mark.parametrize("value", [(), [], ("",), ("valid", 1)])
def test_hash_set_requires_a_nonempty_text_tuple(value):
    with pytest.raises(TokenizerError, match="shingles"):
        hash_shingles(value)
