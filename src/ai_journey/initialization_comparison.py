"""Controlled fixed-normal versus Kaiming initialization experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import nn

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    StepMetric,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
    build_optimizer,
    evaluate_nll,
    expected_initialization_std,
    model_fingerprint,
    seed_everything,
    train_steps,
)


@dataclass(frozen=True)
class ParameterInitializationAudit:
    """Observed and theoretical statistics for one initialized weight matrix."""

    name: str
    module_type: str
    shape: tuple[int, ...]
    fan_in: int | None
    expected_std: float
    observed_mean: float
    observed_std: float
    relative_std_error: float
    bias_is_zero: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_model_initialization(
    model: DecoderLanguageModel,
) -> tuple[ParameterInitializationAudit, ...]:
    """Inspect every explicitly initialized matrix in stable module order."""

    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be a DecoderLanguageModel")
    audits: list[ParameterInitializationAudit] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            weight = module.weight.detach()
            expected = expected_initialization_std(module, model.config)
            observed = float(weight.std(unbiased=False))
            bias_is_zero = None
            fan_in = None
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                bias_is_zero = module.bias is None or bool(
                    torch.count_nonzero(module.bias.detach()) == 0
                )
            audits.append(
                ParameterInitializationAudit(
                    name=f"{name}.weight",
                    module_type=type(module).__name__,
                    shape=tuple(weight.shape),
                    fan_in=fan_in,
                    expected_std=expected,
                    observed_mean=float(weight.mean()),
                    observed_std=observed,
                    relative_std_error=abs(observed - expected) / expected,
                    bias_is_zero=bias_is_zero,
                )
            )
    return tuple(audits)


@dataclass(frozen=True)
class KaimingVariantResult:
    """Training evidence for one controlled initialization policy."""

    name: str
    model_config: TransformerConfig
    initialization_audit: tuple[ParameterInitializationAudit, ...]
    initial_train_nll: float
    initial_validation_nll: float
    trace: tuple[StepMetric, ...]
    final_train_nll: float
    final_validation_nll: float
    model_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_config": asdict(self.model_config),
            "initialization_audit": [
                item.to_dict() for item in self.initialization_audit
            ],
            "initial_train_nll": self.initial_train_nll,
            "initial_validation_nll": self.initial_validation_nll,
            "trace": [asdict(metric) for metric in self.trace],
            "final_train_nll": self.final_train_nll,
            "final_validation_nll": self.final_validation_nll,
            "model_fingerprint": self.model_fingerprint,
        }


@dataclass(frozen=True)
class KaimingComparisonResult:
    """Fixed-normal and Kaiming runs with all training controls held constant."""

    corpus_fingerprint: str
    training_config: TrainingConfig
    variants: tuple[KaimingVariantResult, KaimingVariantResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_fingerprint": self.corpus_fingerprint,
            "training_config": asdict(self.training_config),
            "variants": [variant.to_dict() for variant in self.variants],
        }


def _run_variant(
    name: str,
    corpus: TokenCorpus,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
) -> KaimingVariantResult:
    seed_everything(training_config.seed)
    model = DecoderLanguageModel(model_config)
    audit = audit_model_initialization(model)
    initial_train = evaluate_nll(model, corpus.train_tokens)
    initial_validation = evaluate_nll(model, corpus.validation_tokens)
    cursor = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    trace = train_steps(
        model,
        cursor,
        build_optimizer(model, training_config),
        training_config,
    )
    return KaimingVariantResult(
        name=name,
        model_config=model_config,
        initialization_audit=audit,
        initial_train_nll=initial_train,
        initial_validation_nll=initial_validation,
        trace=trace,
        final_train_nll=evaluate_nll(model, corpus.train_tokens),
        final_validation_nll=evaluate_nll(model, corpus.validation_tokens),
        model_fingerprint=model_fingerprint(model),
    )


def run_kaiming_comparison(
    corpus: TokenCorpus,
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
) -> KaimingComparisonResult:
    """Run matched fixed-normal and Kaiming transformer training curves."""

    if model_config.vocab_size != corpus.vocab_size:
        raise TransformerLabError("model vocabulary does not match corpus codec")
    if model_config.initialization_mode != "fixed_normal":
        raise TransformerLabError("comparison baseline must use fixed_normal")
    variants = (
        _run_variant("fixed_normal", corpus, model_config, training_config),
        _run_variant(
            "kaiming_normal",
            corpus,
            replace(model_config, initialization_mode="kaiming_normal"),
            training_config,
        ),
    )
    return KaimingComparisonResult(
        corpus_fingerprint=corpus.fingerprint(),
        training_config=training_config,
        variants=variants,
    )
