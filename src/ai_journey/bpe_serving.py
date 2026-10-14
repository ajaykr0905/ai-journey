"""Bounded byte-token generation that finishes as strict UTF-8, never lossy text."""

import codecs
import math

import torch

from .bpe import SpecialTokenizer, TokenizerError
from .bpe_corpus import EOS_MARKER
from .tokenizer_codec import BPECodec
from .transformer_lab import DecoderLanguageModel


def _missing_continuations(buffer):
    if not buffer:
        return 0
    length = 2 if buffer[0] < 0xE0 else 3 if buffer[0] < 0xF0 else 4
    return length - len(buffer)


def utf8_candidates(tokenizer, pending: bytes, remaining_tokens: int):
    """Conservatively guarantee completion using the available byte fallback ids."""
    allowed = []
    for token in range(tokenizer.vocab_size):
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        decoder.setstate((pending, 0))
        try:
            decoder.decode(tokenizer.token_bytes(token), final=False)
        except UnicodeError:
            continue
        buffer = decoder.getstate()[0]
        if _missing_continuations(buffer) <= remaining_tokens:
            allowed.append(token)
    return tuple(allowed)


@torch.no_grad()
def generate_utf8(
    model: DecoderLanguageModel,
    codec: BPECodec,
    prompt: str,
    *,
    new_tokens=32,
    seed=40,
    temperature=1.0,
    top_k=None,
):
    """Sample only valid UTF-8 continuations, with EOS and caller-state isolation.

    Constraining the distribution changes sampling semantics. This is not an
    unconstrained language-model quality benchmark or GPT tokenizer compatibility.
    """
    if not isinstance(model, DecoderLanguageModel) or not isinstance(codec, BPECodec):
        raise TypeError("generation requires a decoder and BPECodec")
    tokenizer = codec.tokenizer
    if not isinstance(tokenizer, SpecialTokenizer) or tokenizer.names != (EOS_MARKER,):
        raise TokenizerError("UTF-8 serving requires exactly one reserved EOS marker")
    if (
        model.config.vocab_size != codec.vocab_size
        or model.token_embedding.weight.device.type != "cpu"
    ):
        raise TokenizerError("codec must match the bounded CPU model vocabulary")
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= 1024:
        raise TokenizerError("prompt must contain 1..1024 UTF-8 bytes")
    if (
        type(new_tokens) is not int
        or not 1 <= new_tokens <= 128
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise TokenizerError("generation token or seed budget is invalid")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise TokenizerError("temperature must be positive and finite")
    if top_k is not None and (
        type(top_k) is not int or not 1 <= top_k <= codec.vocab_size
    ):
        raise TokenizerError("top_k must be within the vocabulary")
    ids = list(codec.encode(prompt))
    generator = torch.Generator().manual_seed(seed)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    modes = [(module, module.training) for module in model.modules()]
    emitted = []
    stopped = False
    try:
        model.eval()
        for index in range(new_tokens):
            context = torch.tensor([ids[-model.config.block_size :]], dtype=torch.long)
            logits, _ = model(context)
            scores = logits[0, -1] / temperature
            if not bool(torch.isfinite(scores).all()):
                raise TokenizerError("generation logits must remain finite")
            allowed = utf8_candidates(
                tokenizer, decoder.getstate()[0], new_tokens - index - 1
            )
            mask = torch.zeros(codec.vocab_size, dtype=torch.bool)
            mask[list(allowed)] = True
            scores = scores.masked_fill(~mask, float("-inf"))
            if top_k is not None:
                selected = scores.topk(min(top_k, len(allowed))).indices
                keep = torch.zeros_like(mask)
                keep[selected] = True
                scores = scores.masked_fill(~keep, float("-inf"))
            token = int(
                torch.multinomial(scores.softmax(dim=0), 1, generator=generator)
            )
            ids.append(token)
            if token == tokenizer.vocab_size - 1:
                stopped = True
                break
            emitted.append(decoder.decode(tokenizer.token_bytes(token), final=False))
        emitted.append(decoder.decode(b"", final=True))
    finally:
        for module, mode in modes:
            module.training = mode
    return {
        "token_ids": ids,
        "text": prompt + "".join(emitted),
        "eos": stopped,
        "sampling_policy": "strict-utf8-with-byte-fallback-completion-budget",
    }
