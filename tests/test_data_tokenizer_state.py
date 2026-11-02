import pytest

from ai_journey.bpe import ByteTokenizer, TokenizerError
from ai_journey.tokenizer_io import (
    tokenizer_fingerprint,
    tokenizer_from_state,
    tokenizer_state,
)


def test_in_memory_tokenizer_state_restores_exact_encoding_identity():
    original = ByteTokenizer.train("repeat repeat repeat", vocab_size=280)

    restored = tokenizer_from_state(tokenizer_state(original))

    assert restored.encode("repeat") == original.encode("repeat")
    assert tokenizer_fingerprint(restored) == tokenizer_fingerprint(original)


@pytest.mark.parametrize(
    "state",
    [
        {"kind": "byte", "merges": [], "extra": True},
        {"kind": "unknown", "merges": []},
        {"kind": "byte", "merges": [[999, 1]]},
    ],
)
def test_in_memory_tokenizer_state_rejects_noncanonical_payloads(state):
    with pytest.raises(TokenizerError):
        tokenizer_from_state(state)
