"""Greedy pre/post generations from identical held-out ChatML prefixes."""

from __future__ import annotations

from hashlib import sha256

import torch

from .sft_finetune_codec import UTF8ByteCodec
from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_dataset import FineTuneExample
from .transformer_lab import DecoderLanguageModel


def generation_prompt(example: FineTuneExample, *, block_size: int) -> tuple[int, ...]:
    if not isinstance(example, FineTuneExample) or example.split != "validation":
        raise SFTFineTuneError("generation prompt requires a validation example")
    prefix = example.text[: example.assistant_spans[0][0]]
    tokens = UTF8ByteCodec().encode(prefix)[-block_size:]
    if not tokens:
        raise SFTFineTuneError("generation prompt cannot be empty")
    return tokens


@torch.no_grad()
def greedy_generation(
    model: DecoderLanguageModel, example: FineTuneExample, *, new_tokens: int
) -> dict:
    if type(new_tokens) is not int or not 1 <= new_tokens <= 64:
        raise SFTFineTuneError("new_tokens must be in [1, 64]")
    prompt = generation_prompt(example, block_size=model.config.block_size)
    result = model.generate(
        torch.tensor([prompt], dtype=torch.long), new_tokens=new_tokens, top_k=1
    )[0].tolist()
    generated = result[len(prompt) :]
    payload = bytes(generated)
    return {
        "example_id": example.example_id,
        "prompt_sha256": sha256(bytes(prompt)).hexdigest(),
        "token_ids": generated,
        "utf8_preview": payload.decode("utf-8", errors="replace"),
        "sha256": sha256(payload).hexdigest(),
    }
