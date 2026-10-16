"""Bounded, paired, deterministic CPU experiments on an existing tiny decoder."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
from dataclasses import asdict, dataclass
import math
import platform
import random

import numpy as np
import torch
from torch.nn import functional as F

from .lora import frozen_fingerprint, parameter_summary
from .lora_data import (
    FIXTURE_ID,
    LoRAExperimentError,
    bounded_integer,
    heldout_evidence,
    heldout_targets,
    schedule_evidence,
    synthetic_fixture,
    training_batches,
)
from .transformer_lab import DecoderLanguageModel, TransformerConfig, model_fingerprint


@dataclass(frozen=True)
class ComparisonConfig:
    seeds: tuple[int, ...] = (11, 29)
    block_size: int = 8
    embedding_dim: int = 16
    head_count: int = 2
    batch_size: int = 4
    pretrain_steps: int = 12
    finetune_steps: int = 8
    pretrain_learning_rate: float = 0.01
    finetune_learning_rate: float = 0.01
    adapter_scale: float = 2.0
    evaluation_batch_size: int = 32

    def __post_init__(self):
        if not isinstance(self.seeds, tuple) or not 2 <= len(self.seeds) <= 4:
            raise LoRAExperimentError("seeds require two to four distinct paired seeds")
        for seed in self.seeds:
            bounded_integer(seed, "seed", 0, 2**32 - 1)
        if len(set(self.seeds)) != len(self.seeds):
            raise LoRAExperimentError("seeds must be distinct")
        for name, low, high in (
            ("block_size", 2, 32),
            ("embedding_dim", 4, 32),
            ("head_count", 1, 4),
            ("batch_size", 1, 8),
            ("pretrain_steps", 1, 64),
            ("finetune_steps", 1, 64),
            ("evaluation_batch_size", 1, 64),
        ):
            bounded_integer(getattr(self, name), name, low, high)
        if self.embedding_dim % self.head_count:
            raise LoRAExperimentError("embedding_dim must be divisible by head_count")
        for name, high in (
            ("pretrain_learning_rate", 0.1),
            ("finetune_learning_rate", 0.1),
            ("adapter_scale", 8),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= high
            ):
                raise LoRAExperimentError(f"{name} must be finite and in (0, {high}]")
        if self.work_units > 10_000_000:
            raise LoRAExperimentError("comparison exceeds the combined CPU work budget")

    @property
    def work_units(self):
        return (
            len(self.seeds)
            * (self.pretrain_steps + 3 * self.finetune_steps)
            * self.batch_size
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


@contextmanager
def isolated_cpu_runtime(seed):
    """Restore caller RNGs and CPU controls on success and all exception paths."""
    bounded_integer(seed, "seed", 0, 2**32 - 1)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        with torch.device("cpu"):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        torch.set_default_dtype(dtype)


def _finite_model(model):
    if any(
        parameter.device.type != "cpu"
        or parameter.dtype != torch.float64
        or not bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    ):
        raise LoRAExperimentError(
            "comparison parameters must be finite float64 CPU tensors"
        )


def train_on_batches(model, batches, *, learning_rate):
    """Use the exact supplied schedule, including forward-only frozen controls."""
    _finite_model(model)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise LoRAExperimentError("learning_rate must be finite and in (0, 0.1]")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    optimizer = (
        torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
        if parameters
        else None
    )
    losses, target_tokens = [], 0
    model.train()
    for batch in batches:
        inputs = torch.tensor(
            [row.inputs for row in batch], dtype=torch.long, device="cpu"
        )
        targets = torch.tensor(
            [row.targets for row in batch], dtype=torch.long, device="cpu"
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with nullcontext() if optimizer is not None else torch.no_grad():
            _, loss = model(inputs, targets)
        if loss is None or not bool(torch.isfinite(loss)):
            raise LoRAExperimentError("training loss must be finite")
        losses.append(float(loss.detach()))
        target_tokens += targets.numel()
        if optimizer is not None:
            loss.backward()
            if any(
                p.grad is None or not bool(torch.isfinite(p.grad).all())
                for p in parameters
            ):
                raise LoRAExperimentError(
                    "trainable gradients must be present and finite"
                )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            _finite_model(model)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    return {
        "losses": losses,
        "consumed_target_tokens": target_tokens,
        "optimizer_steps": len(losses) if optimizer is not None else 0,
    }


def evaluate_heldout(model, documents, codec, *, batch_size=32):
    """Score only the last logit of each document-local, unique target context."""
    bounded_integer(batch_size, "evaluation batch_size", 1, 64)
    _finite_model(model)
    targets = heldout_targets(documents, codec, block_size=model.config.block_size)
    losses = [None] * len(targets)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            for width in range(1, model.config.block_size + 1):
                indexes = [
                    i for i, row in enumerate(targets) if len(row.context) == width
                ]
                for start in range(0, len(indexes), batch_size):
                    selected = indexes[start : start + batch_size]
                    inputs = torch.tensor(
                        [targets[i].context for i in selected],
                        dtype=torch.long,
                        device="cpu",
                    )
                    labels = torch.tensor(
                        [targets[i].target for i in selected],
                        dtype=torch.long,
                        device="cpu",
                    )
                    logits, _ = model(inputs)
                    values = F.cross_entropy(logits[:, -1], labels, reduction="none")
                    for index, value in zip(selected, values.tolist(), strict=True):
                        losses[index] = value
    finally:
        for module, mode in modes:
            module.training = mode
    total = math.fsum(losses)
    mean = total / len(targets)
    try:
        perplexity = math.exp(mean)
    except OverflowError as exc:
        raise LoRAExperimentError("held-out perplexity overflow") from exc
    if any(not math.isfinite(value) for value in (total, mean, perplexity)):
        raise LoRAExperimentError("held-out metrics must be finite")
    return {
        "nll_sum": total,
        "target_tokens": len(targets),
        "mean_nll": mean,
        "perplexity": perplexity,
    }


LIMITATIONS = (
    "Authored synthetic character grammar; tiny one-layer CPU decoder; paired seeds "
    "are descriptive, not population confidence. Equal batches and target exposure "
    "do not equal compute cost. No claim that LoRA always wins, that these results "
    "represent large models, or that parameter counts measure RAM or latency."
)


def config_dict(config):
    return {**asdict(config), "seeds": list(config.seeds)}


def _probe(model, batch):
    model.eval()
    with torch.inference_mode():
        logits, _ = model(
            torch.tensor([row.inputs for row in batch], dtype=torch.long, device="cpu")
        )
    return logits


def _trial(name, model, base, batches, split, codec, config):
    initial = model_fingerprint(model)
    counts = parameter_summary(model)
    frozen_before = frozen_fingerprint(model)
    probe_error = float(
        (_probe(model, batches[0]) - _probe(base, batches[0])).abs().max()
    )
    if probe_error != 0:
        raise LoRAExperimentError(
            "comparison arms must start from the same base function"
        )
    trace = train_on_batches(
        model, batches, learning_rate=config.finetune_learning_rate
    )
    frozen_after = frozen_fingerprint(model)
    if frozen_before != frozen_after:
        raise LoRAExperimentError("frozen model state changed during adaptation")
    return {
        "name": name,
        "parameters": counts,
        "initial_sha256": initial,
        "final_sha256": model_fingerprint(model),
        "frozen_before_sha256": frozen_before,
        "frozen_after_sha256": frozen_after,
        "initial_function_max_abs_error": probe_error,
        "training": trace,
        "heldout": evaluate_heldout(
            model, split.validation, codec, batch_size=config.evaluation_batch_size
        ),
    }


def _run_comparison(config, arms):
    if not isinstance(config, ComparisonConfig):
        raise LoRAExperimentError("comparison requires ComparisonConfig")
    split = synthetic_fixture()
    codec = split.codec()
    model_config = config.model_config(len(codec.tokens))
    seeds = []
    for seed in config.seeds:
        with isolated_cpu_runtime(seed):
            base = DecoderLanguageModel(model_config)
            initial = model_fingerprint(base)
            pretrain = training_batches(
                split.pretrain,
                codec,
                block_size=config.block_size,
                batch_size=config.batch_size,
                steps=config.pretrain_steps,
                seed=seed ^ 0xB45E,
            )
            adaptation = training_batches(
                split.train,
                codec,
                block_size=config.block_size,
                batch_size=config.batch_size,
                steps=config.finetune_steps,
                seed=seed ^ 0xA11CE,
            )
            trace = train_on_batches(
                base, pretrain, learning_rate=config.pretrain_learning_rate
            )
            pretrained = model_fingerprint(base)
            trials = []
            for name in arms:
                model = copy.deepcopy(base)
                if name == "frozen":
                    model.requires_grad_(False)
                trials.append(
                    _trial(name, model, base, adaptation, split, codec, config)
                )
            if pretrained != model_fingerprint(base):
                raise LoRAExperimentError("forked trials mutated the pretrained source")
            seeds.append(
                {
                    "seed": seed,
                    "initial_sha256": initial,
                    "pretrained_sha256": pretrained,
                    "source_after_sha256": model_fingerprint(base),
                    "pretrain_schedule": schedule_evidence(pretrain),
                    "adaptation_schedule": schedule_evidence(adaptation),
                    "pretraining": trace,
                    "trials": trials,
                }
            )
    return {
        "config": config_dict(config),
        "fixture": {"id": FIXTURE_ID, **split.evidence()},
        "model": asdict(model_config),
        "heldout": heldout_evidence(
            split.validation, codec, block_size=config.block_size
        ),
        "seeds": seeds,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
        },
        "limitations": LIMITATIONS,
    }


def run_dense_comparison(config=ComparisonConfig()):
    """Pretrain once per seed, then compare dense adaptation with a frozen fork."""
    return _run_comparison(config, ("frozen", "full"))
