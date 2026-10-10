"""Bounded CPU certification of attention, padding and cached decoder parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite

import torch

from .attention_audit import (
    audit_attention_causality,
    audit_attention_gradients,
    diagnose_attention_weights,
    per_head_reference,
)
from .transformer_cache import decode, prefill
from .transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    TransformerLabError,
)


@dataclass(frozen=True)
class TransformerCertificationConfig:
    model: TransformerConfig = TransformerConfig(
        vocab_size=11, block_size=8, embedding_dim=8, head_count=2, layer_count=2
    )
    seed: int = 37
    tolerance: float = 1e-10

    def __post_init__(self):
        if not isinstance(self.model, TransformerConfig):
            raise TypeError("model must be TransformerConfig")
        if self.model.normalization_mode != "layer_norm" or self.model.block_size < 2:
            raise TransformerLabError(
                "certification requires layer_norm and block_size >= 2"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise TransformerLabError("seed must be an integer in [0, 2**32)")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise TransformerLabError("tolerance must be finite and nonnegative")
        if (
            self.model.embedding_dim > 64
            or self.model.block_size > 64
            or self.model.layer_count > 4
            or self.model.vocab_size > 1024
        ):
            raise TransformerLabError("certification is bounded to small CPU models")


def certify_transformer(
    config: TransformerCertificationConfig = TransformerCertificationConfig(),
) -> dict:
    """Run independent audits on CPU float64 without changing caller RNG state."""
    if not isinstance(config, TransformerCertificationConfig):
        raise TypeError("config must be TransformerCertificationConfig")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = DecoderLanguageModel(config.model).double().eval()
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        inputs = torch.randn(
            2,
            min(5, config.model.block_size),
            config.model.embedding_dim,
            generator=generator,
            dtype=torch.float64,
        )
        attention_results = []
        for block in model.blocks:
            gradient = audit_attention_gradients(block.attention, inputs)
            causality = audit_attention_causality(block.attention, inputs)
            diagnostics = diagnose_attention_weights(
                per_head_reference(block.attention, inputs).weights
            )
            attention_results.append(
                {
                    "gradients": asdict(gradient),
                    "causality": asdict(causality),
                    "probabilities": asdict(diagnostics),
                }
            )
        tokens = torch.randint(
            config.model.vocab_size,
            (2, config.model.block_size + 3),
            generator=generator,
        )
        prefix = min(3, config.model.block_size - 1)
        logits, cache = prefill(model, tokens[:, :prefix])
        reference, _ = model(tokens[:, :prefix])
        cache_error = float((logits - reference).abs().max().detach())
        decoded, cache = decode(model, tokens[:, prefix:], cache)
        expected = []
        for position in range(prefix, tokens.shape[1]):
            start = max(0, position + 1 - config.model.block_size)
            output, _ = model(tokens[:, start : position + 1])
            expected.append(output[:, -1:])
        cache_error = max(
            cache_error,
            float((decoded - torch.cat(expected, dim=1)).abs().max().detach()),
        )
        padded = tokens[:, : min(5, config.model.block_size)]
        lengths = torch.tensor([padded.shape[1], 1], dtype=torch.long)
        padded_logits, _ = model(padded, lengths=lengths)
        single, _ = model(padded[1:2, :1])
        padding_error = float((padded_logits[1:2, :1] - single).abs().max().detach())
        padding_error = max(
            padding_error, float(padded_logits[1, 1:].abs().max().detach())
        )
        for result in attention_results:
            errors = [
                *result["gradients"].values(),
                result["causality"]["prefix_error"],
                result["causality"]["future_gradient"],
            ]
            if any(not isfinite(error) or error > config.tolerance for error in errors):
                raise TransformerLabError("attention certification failed")
        if any(
            not isfinite(error) or error > config.tolerance
            for error in (cache_error, padding_error)
        ):
            raise TransformerLabError("decoder certification failed")
        return {
            "config": asdict(config),
            "input_sha256": sha256(
                inputs.numpy().tobytes() + tokens.numpy().tobytes()
            ).hexdigest(),
            "attention": attention_results,
            "cache_rollover_error": cache_error,
            "padding_error": padding_error,
            "parameter_count": model.parameter_count,
            "runtime": {
                "device": "cpu",
                "dtype": "float64",
                "torch_version": torch.__version__,
                "threads": torch.get_num_threads(),
            },
        }
