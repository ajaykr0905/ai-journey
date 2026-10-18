"""Matched full and low-rank preference optimization on a tiny CPU decoder."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import math
import platform

import torch

from .lora import (
    LoRAConfig,
    frozen_fingerprint,
    inject_adapters,
    merge_adapters,
    parameter_summary,
)
from .lora_experiment import ADAPTER_TARGETS
from .lora_experiment import isolated_cpu_runtime
from .preference_data import (
    PreferenceDataError,
    PreferenceDataset,
    PreferenceFold,
    authored_preference_fixture,
    fold_evidence,
    grouped_preference_folds,
    preference_codec,
    preference_schedule_evidence,
    preference_training_batches,
)
from .preference_optimization import (
    evaluate_candidate_logprob,
    evaluate_preferences,
    train_preference_batches,
)
from .sft_data import (
    CHAT_FIXTURE_ID,
    ConversationSplit,
    authored_chat_fixture,
    render_conversation,
    sft_schedule_evidence,
    sft_training_batches,
)
from .sft_experiment import train_sft_batches
from .transformer_lab import (
    CharacterCodec,
    DecoderLanguageModel,
    TransformerConfig,
    model_fingerprint,
)


@dataclass(frozen=True)
class PreferenceComparisonConfig:
    seeds: tuple[int, ...] = (44, 45)
    fold_count: int = 3
    fold_seed: int = 4403
    block_size: int = 32
    embedding_dim: int = 16
    head_count: int = 2
    sft_batch_size: int = 4
    sft_steps: int = 4
    preference_batch_size: int = 2
    preference_steps: int = 4
    sft_learning_rate: float = 0.01
    preference_learning_rate: float = 0.01
    beta: float = 0.2
    adapter_scale: float = 2.0

    def __post_init__(self):
        if not isinstance(self.seeds, tuple) or not 2 <= len(self.seeds) <= 4:
            raise PreferenceDataError("seeds require two to four paired values")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise PreferenceDataError("seeds must be integers in [0, 2**32)")
        if len(set(self.seeds)) != len(self.seeds):
            raise PreferenceDataError("seeds must be distinct")
        for name, low, high in (
            ("fold_count", 2, 6),
            ("fold_seed", 0, 2**32 - 1),
            ("block_size", 8, 128),
            ("embedding_dim", 4, 32),
            ("head_count", 1, 8),
            ("sft_batch_size", 1, 16),
            ("sft_steps", 1, 32),
            ("preference_batch_size", 1, 8),
            ("preference_steps", 1, 32),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise PreferenceDataError(
                    f"{name} must be an integer in [{low}, {high}]"
                )
        if self.embedding_dim % self.head_count:
            raise PreferenceDataError("embedding_dim must divide evenly across heads")
        for name, high in (
            ("sft_learning_rate", 0.1),
            ("preference_learning_rate", 0.1),
            ("beta", 10.0),
            ("adapter_scale", 8.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= high
            ):
                raise PreferenceDataError(f"{name} must be finite and in (0, {high}]")
        if self.work_units > 80_000_000:
            raise PreferenceDataError(
                "preference comparison exceeds the CPU work budget"
            )

    @property
    def work_units(self):
        return (
            len(self.seeds)
            * (self.sft_steps + self.fold_count * 4 * self.preference_steps)
            * max(self.sft_batch_size, self.preference_batch_size)
            * self.block_size
            * self.embedding_dim**2
        )

    def model_config(self, vocab_size):
        return TransformerConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            embedding_dim=self.embedding_dim,
            head_count=self.head_count,
            layer_count=1,
            feed_forward_expansion=2,
            dropout=0.0,
        )


def preference_config_dict(config):
    if not isinstance(config, PreferenceComparisonConfig):
        raise TypeError("config must be PreferenceComparisonConfig")
    return {**asdict(config), "seeds": list(config.seeds)}


def preference_config_from_dict(value):
    if (
        not isinstance(value, dict)
        or set(value) != {field.name for field in fields(PreferenceComparisonConfig)}
        or not isinstance(value["seeds"], list)
    ):
        raise PreferenceDataError(
            "config requires exactly all preference controls and a seeds list"
        )
    return PreferenceComparisonConfig(**{**value, "seeds": tuple(value["seeds"])})


@dataclass(frozen=True)
class PreferenceContext:
    dataset: PreferenceDataset
    folds: tuple[PreferenceFold, ...]
    sft_split: ConversationSplit
    codec: CharacterCodec

    def evidence(self) -> dict:
        return {
            "preference_fixture": self.dataset.evidence(),
            "folds": fold_evidence(self.folds),
            "sft_fixture_id": CHAT_FIXTURE_ID,
            "sft_fixture": self.sft_split.evidence(),
            "vocabulary": list(self.codec.tokens),
        }


def prepare_preference_context(config: PreferenceComparisonConfig) -> PreferenceContext:
    """Create fixed data, leakage-safe folds, and one shared character codec."""
    if not isinstance(config, PreferenceComparisonConfig):
        raise TypeError("config must be PreferenceComparisonConfig")
    dataset = authored_preference_fixture()
    folds = grouped_preference_folds(
        dataset, fold_count=config.fold_count, seed=config.fold_seed
    )
    sft_split = authored_chat_fixture()
    sft_text = "".join(
        render_conversation(conversation).text
        for conversation in sft_split.train + sft_split.validation
    )
    codec = preference_codec(dataset, extra_text=sft_text)
    return PreferenceContext(dataset, folds, sft_split, codec)


def warm_preference_base(
    config: PreferenceComparisonConfig, context: PreferenceContext, *, seed: int
) -> tuple[DecoderLanguageModel, dict]:
    """Build one deterministic assistant-only SFT reference for all fold arms."""
    if not isinstance(config, PreferenceComparisonConfig):
        raise TypeError("config must be PreferenceComparisonConfig")
    if not isinstance(context, PreferenceContext):
        raise TypeError("context must be PreferenceContext")
    if seed not in config.seeds:
        raise PreferenceDataError("warmup seed must be declared in the protocol")
    with isolated_cpu_runtime(seed):
        model = DecoderLanguageModel(config.model_config(len(context.codec.tokens)))
        initial = model_fingerprint(model)
        batches = sft_training_batches(
            context.sft_split.train,
            context.codec,
            block_size=config.block_size,
            batch_size=config.sft_batch_size,
            steps=config.sft_steps,
            seed=seed ^ 0x5F7,
        )
        trace = train_sft_batches(
            model, batches, learning_rate=config.sft_learning_rate
        )
        final = model_fingerprint(model)
    return model, {
        "initial_sha256": initial,
        "final_sha256": final,
        "schedule": sft_schedule_evidence(batches),
        "training": trace,
    }


PREFERENCE_ARMS = ("frozen", "full", "lora-rank-1", "lora-rank-2")

PREFERENCE_LIMITATIONS = (
    "Authored grounded-reading pairs and assistant-only warmup chats; tiny one-layer "
    "CPU character decoder; response scores are length-normalized before a DPO-style "
    "objective. Grouped folds and paired seeds are descriptive, not population "
    "confidence intervals. This is offline preference optimization, not reward-model "
    "training, on-policy reinforcement learning, PPO, human annotation, chain-of-thought "
    "evaluation, or evidence about large models, GPUs, production, or learner completion."
)


def _arm_model(base, name, config, seed):
    if name == "frozen":
        return copy.deepcopy(base).requires_grad_(False), None
    if name == "full":
        return copy.deepcopy(base).requires_grad_(True), None
    if name.startswith("lora-rank-"):
        rank = int(name.rsplit("-", 1)[1])
        adapter = LoRAConfig(
            rank=rank,
            alpha=rank * config.adapter_scale,
            dropout=0.0,
            seed=seed,
        )
        return inject_adapters(base, ADAPTER_TARGETS, adapter), adapter
    raise PreferenceDataError(f"unknown preference arm: {name}")


def run_preference_fold(
    base: DecoderLanguageModel,
    fold: PreferenceFold,
    config: PreferenceComparisonConfig,
    context: PreferenceContext,
    *,
    seed: int,
) -> dict:
    """Run matched frozen, dense, and two LoRA policies on one held-out fold."""
    if not isinstance(base, DecoderLanguageModel):
        raise TypeError("base must be DecoderLanguageModel")
    if not isinstance(fold, PreferenceFold) or not isinstance(
        context, PreferenceContext
    ):
        raise TypeError("fold and context must use preference experiment types")
    source_before = model_fingerprint(base)
    reference = copy.deepcopy(base).requires_grad_(False)
    schedule = preference_training_batches(
        fold.train,
        batch_size=config.preference_batch_size,
        steps=config.preference_steps,
        seed=seed ^ (fold.index + 1) ^ 0xD90,
    )
    probe_pair = fold.train[0]
    trials = []
    for name in PREFERENCE_ARMS:
        policy, adapter = _arm_model(base, name, config, seed)
        frozen_before = frozen_fingerprint(policy)
        initial_errors = [
            abs(
                evaluate_candidate_logprob(policy, probe_pair, label, context.codec)
                - evaluate_candidate_logprob(base, probe_pair, label, context.codec)
            )
            for label in ("chosen", "rejected")
        ]
        initial_error = max(initial_errors)
        if initial_error != 0:
            raise PreferenceDataError(
                "preference arms must start from the same policy function"
            )
        training = train_preference_batches(
            policy,
            reference,
            schedule,
            context.codec,
            learning_rate=config.preference_learning_rate,
            beta=config.beta,
        )
        frozen_after = frozen_fingerprint(policy)
        if frozen_before != frozen_after:
            raise PreferenceDataError("frozen preference state changed during training")
        heldout = evaluate_preferences(
            policy, reference, fold.heldout, context.codec, beta=config.beta
        )
        merge = None
        if adapter is not None:
            policy.eval()
            dense = merge_adapters(policy)
            merge_errors = [
                abs(
                    evaluate_candidate_logprob(policy, probe_pair, label, context.codec)
                    - evaluate_candidate_logprob(
                        dense, probe_pair, label, context.codec
                    )
                )
                for label in ("chosen", "rejected")
            ]
            merge_error = max(merge_errors)
            if not math.isfinite(merge_error) or merge_error > 1e-10:
                raise PreferenceDataError("merged preference adapter is not equivalent")
            merge = {
                "probe_max_abs_error": merge_error,
                "dense_sha256": model_fingerprint(dense),
                "parameters": parameter_summary(dense),
            }
        trials.append(
            {
                "name": name,
                "adapter": asdict(adapter) if adapter is not None else None,
                "parameters": parameter_summary(policy),
                "initial_function_max_abs_error": initial_error,
                "final_sha256": model_fingerprint(policy),
                "frozen_before_sha256": frozen_before,
                "frozen_after_sha256": frozen_after,
                "training": training,
                "heldout": heldout,
                "merge": merge,
            }
        )
    if model_fingerprint(base) != source_before:
        raise PreferenceDataError("source base changed while running fold arms")
    return {
        "fold": fold.evidence(),
        "schedule": preference_schedule_evidence(schedule),
        "reference_sha256": model_fingerprint(reference),
        "trials": trials,
    }


def run_preference_comparison(config=None) -> dict:
    """Run every declared seed, group fold, and matched adaptation arm."""
    if config is None:
        config = PreferenceComparisonConfig()
    if not isinstance(config, PreferenceComparisonConfig):
        raise TypeError("config must be PreferenceComparisonConfig")
    context = prepare_preference_context(config)
    seeds = []
    for seed in config.seeds:
        with isolated_cpu_runtime(seed):
            base, warmup = warm_preference_base(config, context, seed=seed)
            source_before = model_fingerprint(base)
            fold_results = [
                run_preference_fold(base, fold, config, context, seed=seed)
                for fold in context.folds
            ]
            if model_fingerprint(base) != source_before:
                raise PreferenceDataError("warmed source changed between fold runs")
            seeds.append(
                {
                    "seed": seed,
                    "warmup": warmup,
                    "source_after_sha256": model_fingerprint(base),
                    "folds": fold_results,
                }
            )
    return {
        "config": preference_config_dict(config),
        "context": context.evidence(),
        "model": asdict(config.model_config(len(context.codec.tokens))),
        "adapter_targets": list(ADAPTER_TARGETS),
        "seeds": seeds,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
        },
        "limitations": PREFERENCE_LIMITATIONS,
    }
