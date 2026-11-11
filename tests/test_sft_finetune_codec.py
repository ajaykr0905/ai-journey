import pytest

from ai_journey.sft_finetune_codec import UTF8ByteCodec
from ai_journey.sft_finetune_contract import SFTFineTuneError


def test_utf8_byte_codec_round_trips_ascii_and_unicode_without_fitting():
    codec = UTF8ByteCodec()
    text = "Hello, नमस्ते 🌍"
    encoded = codec.encode(text)
    assert codec.decode(encoded) == text
    assert codec.vocab_size == 256
    assert (
        codec.fingerprint()
        == "40aff2e9d2d8922e47afd4648e6967497158785fbd1da870e7110266bf944880"
    )


def test_utf8_byte_codec_rejects_invalid_ids_and_partial_sequences():
    codec = UTF8ByteCodec()
    with pytest.raises(SFTFineTuneError):
        codec.decode([256])
    with pytest.raises(SFTFineTuneError):
        codec.decode([0xE2])
