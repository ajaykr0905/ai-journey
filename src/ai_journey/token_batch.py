"""Length-aware token batches for the decoder's padded-input contract."""

from dataclasses import dataclass

import torch

from .bpe import SpecialTokenizer, TokenizerError


@dataclass(frozen=True)
class EncodedBatch:
    token_ids: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor


def encode_batch(
    tokenizer,
    texts: tuple[str, ...],
    *,
    pad_id=0,
    max_tokens=1024,
    truncation="error",
    allowed=frozenset(),
) -> EncodedBatch:
    if not isinstance(texts, tuple) or not 1 <= len(texts) <= 64:
        raise TokenizerError("batch must contain 1..64 texts")
    if type(pad_id) is not int or not 0 <= pad_id < tokenizer.vocab_size:
        raise TokenizerError("pad_id must belong to the vocabulary")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise TokenizerError("max_tokens must be in [1, 4096]")
    if truncation not in {"error", "left", "right"}:
        raise TokenizerError("truncation must be error, left, or right")
    rows = []
    for text in texts:
        values = (
            tokenizer.encode(text, allowed=allowed)
            if isinstance(tokenizer, SpecialTokenizer)
            else tokenizer.encode(text)
        )
        if not values:
            raise TokenizerError("decoder request text must encode at least one token")
        if len(values) > max_tokens:
            if truncation == "error":
                raise TokenizerError("request exceeds the token budget")
            values = (
                values[-max_tokens:] if truncation == "left" else values[:max_tokens]
            )
        rows.append(values)
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    width = int(lengths.max())
    values = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for index, row in enumerate(rows):
        values[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return EncodedBatch(
        values, lengths, torch.arange(width)[None, :] < lengths[:, None]
    )


def decode_batch(tokenizer, batch: EncodedBatch) -> tuple[str, ...]:
    if not isinstance(batch, EncodedBatch):
        raise TypeError("batch must be EncodedBatch")
    values, lengths = batch.token_ids, batch.lengths
    if (
        values.ndim != 2
        or values.dtype != torch.long
        or lengths.ndim != 1
        or lengths.dtype != torch.long
        or len(lengths) != len(values)
        or bool(((lengths <= 0) | (lengths > values.shape[1])).any())
    ):
        raise TokenizerError("token batch shapes or lengths are invalid")
    return tuple(
        tokenizer.decode(values[index, : int(length)].tolist())
        for index, length in enumerate(lengths)
    )
