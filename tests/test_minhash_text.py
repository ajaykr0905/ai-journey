import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_text import tokenize_for_minhash, word_shingles


def test_tokenization_normalizes_unicode_case_and_punctuation():
    assert tokenize_for_minhash("ＣＡＦÉ, café!") == ("café", "café")


def test_shingles_are_unique_sorted_and_use_token_boundaries():
    assert word_shingles("B a b a", width=2) == ("a\x1fb", "b\x1fa")


def test_short_document_is_retained_as_one_shingle():
    assert word_shingles("Only two", width=5) == ("only\x1ftwo",)


@pytest.mark.parametrize("width", [True, 0, 33, 2.5])
def test_shingle_width_is_strictly_bounded(width):
    with pytest.raises(TokenizerError, match="shingle width"):
        word_shingles("valid words", width=width)


@pytest.mark.parametrize("text", ["", "   ", "---"])
def test_tokenization_rejects_text_without_words(text):
    with pytest.raises(TokenizerError, match="MinHash text"):
        tokenize_for_minhash(text)
