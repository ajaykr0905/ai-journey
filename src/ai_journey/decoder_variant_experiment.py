"""Matched-token, CPU-only training comparisons for decoder architectures."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import nn
from torch.nn import functional as F

from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_model import DecoderVariantModel


@dataclass(frozen=True)
class VariantResult:
    """Deterministic loss measurements for one architecture arm."""

    name: str
    architecture_fingerprint: str
    parameter_count: int
    initial_train_nll: float
    final_train_nll: float
    heldout_nll: float


@dataclass(frozen=True)
class ComparisonResult:
    """Evidence that every arm used the same bytes, windows, and seed."""

    seed: int
    steps: int
    batch_size: int
    context_length: int
    train_sha256: str
    heldout_sha256: str
    batch_offsets_sha256: str
    heldout_windows: int
    variants: tuple[VariantResult, ...]


def _build_matched_model(
    config: DecoderVariantConfig, seed: int
) -> DecoderVariantModel:
    """Keep homologous modules identically initialized across architecture arms.

    Seeding only the constructor per arm is insufficient: a missing position
    table or a different feed-forward block shifts the RNG stream for later
    modules. Reapply each standard PyTorch initializer with a seed derived
    from the module's stable name. Skip the tied output head so the token
    embedding keeps its Embedding initialization and weight alias.
    """

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DecoderVariantModel(config).cpu()
        for name, module in model.named_modules():
            if name == "lm_head" or not isinstance(
                module, (nn.Embedding, nn.Linear, nn.LayerNorm)
            ):
                continue
            digest = sha256(seed.to_bytes(8, "big") + b"\0" + name.encode()).digest()
            torch.manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
            module.reset_parameters()
    return model


def run_variant_comparison(
    *,
    train_bytes: bytes,
    heldout_bytes: bytes,
    variants: Mapping[str, DecoderVariantConfig],
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    evaluation_windows: int = 32,
) -> ComparisonResult:
    """Train each tiny decoder on identical byte windows, then measure held-out NLL.

    This is a deterministic CPU protocol, not a claim about GPU throughput or
    general model quality. Only input hashes and measurements enter the result.
    Dropout must be disabled; shared named modules receive identical initial
    values so architecture is the sole intended intervention.
    """

    if not isinstance(train_bytes, bytes) or not isinstance(heldout_bytes, bytes):
        raise TypeError("train_bytes and heldout_bytes must be bytes")
    if train_bytes == heldout_bytes:
        raise ValueError("heldout_bytes must differ from train_bytes")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("variants must be a non-empty mapping")
    for name, config in variants.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(config, DecoderVariantConfig)
        ):
            raise ValueError("variants require non-empty names and decoder configs")
    configs = tuple(variants.values())
    if len({config.fingerprint() for config in configs}) != len(configs):
        raise ValueError("variants must have distinct architecture contracts")
    base = configs[0].gpt2
    if any(config.gpt2 != base for config in configs):
        raise ValueError("all variants must share one GPT-2 base configuration")
    if base.vocab_size < 256:
        raise ValueError("byte experiments require vocab_size of at least 256")
    if any(
        rate != 0
        for rate in (
            base.embedding_dropout,
            base.attention_dropout,
            base.residual_dropout,
        )
    ):
        raise ValueError("dropout must be zero for a matched comparison")
    for name, value in (
        ("seed", seed),
        ("steps", steps),
        ("batch_size", batch_size),
        ("evaluation_windows", evaluation_windows),
    ):
        if (
            type(value) is not int
            or value < (0 if name == "seed" else 1)
            or (name == "seed" and value > 2**63 - 1)
        ):
            raise ValueError(f"{name} must be a valid non-negative/positive integer")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("learning_rate must be finite and positive")
    try:
        learning_rate = float(learning_rate)
    except OverflowError as exc:
        raise ValueError("learning_rate must be finite and positive") from exc
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if (
        len(train_bytes) <= base.context_length
        or len(heldout_bytes) <= base.context_length
    ):
        raise ValueError("train and held-out bytes must each exceed context_length")

    train_tokens = torch.tensor(list(train_bytes), dtype=torch.long)
    heldout_tokens = torch.tensor(list(heldout_bytes), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_start = len(train_bytes) - base.context_length - 1
    offsets = torch.randint(max_start + 1, (steps, batch_size), generator=generator)
    offset_digest = sha256(
        json.dumps(offsets.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    heldout_count = min(evaluation_windows, len(heldout_bytes) - base.context_length)

    def windows(
        tokens: torch.Tensor, starts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = starts[:, None] + torch.arange(base.context_length + 1)
        selected = tokens[positions]
        return selected[:, :-1], selected[:, 1:]

    initial_inputs, initial_targets = windows(train_tokens, offsets[0])
    heldout_starts = torch.arange(heldout_count)
    heldout_inputs, heldout_targets = windows(heldout_tokens, heldout_starts)

    def nll(
        model: DecoderVariantModel, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits = model(inputs)
        return F.cross_entropy(logits.reshape(-1, base.vocab_size), targets.reshape(-1))

    results: list[VariantResult] = []
    # fork_rng restores the caller's CPU RNG state; no CUDA state is touched.
    with torch.random.fork_rng(devices=[]):
        for name, config in variants.items():
            model = _build_matched_model(config, seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            with torch.no_grad():
                model.eval()
                initial_loss = float(nll(model, initial_inputs, initial_targets))
            if not math.isfinite(initial_loss):
                raise FloatingPointError(f"non-finite initial loss for {name}")
            model.train()
            for starts in offsets:
                inputs, targets = windows(train_tokens, starts)
                optimizer.zero_grad(set_to_none=True)
                loss = nll(model, inputs, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite training loss for {name}")
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                final_loss = float(nll(model, initial_inputs, initial_targets))
                heldout_loss = float(nll(model, heldout_inputs, heldout_targets))
            if not math.isfinite(final_loss) or not math.isfinite(heldout_loss):
                raise FloatingPointError(f"non-finite evaluation loss for {name}")
            results.append(
                VariantResult(
                    name=name,
                    architecture_fingerprint=config.fingerprint(),
                    parameter_count=model.parameter_count,
                    initial_train_nll=initial_loss,
                    final_train_nll=final_loss,
                    heldout_nll=heldout_loss,
                )
            )

    return ComparisonResult(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        context_length=base.context_length,
        train_sha256=sha256(train_bytes).hexdigest(),
        heldout_sha256=sha256(heldout_bytes).hexdigest(),
        batch_offsets_sha256=offset_digest,
        heldout_windows=heldout_count,
        variants=tuple(results),
    )
