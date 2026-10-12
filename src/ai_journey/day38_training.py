"""Bounded, reproducible full-transformer training on verified public text."""

from __future__ import annotations

import json
import os
import platform
import random
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gpt_bigram import CorpusSource, validate_corpus_bytes
from .transformer_lab import (
    BatchCursor,
    CharacterCodec,
    DecoderLanguageModel,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
    build_optimizer,
    load_training_checkpoint,
    model_fingerprint,
    run_transformer_experiment,
    seed_everything,
)

DAY38_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Day38DataConfig:
    """Explicit limits for a scale-limited public-corpus experiment."""

    block_size: int = 32
    train_token_limit: int = 65_536
    validation_token_limit: int = 8_192
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        for name in ("block_size", "train_token_limit", "validation_token_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= self.block_size if name != "block_size" else value <= 0:
                requirement = (
                    "positive" if name == "block_size" else "greater than block_size"
                )
                raise TransformerLabError(f"{name} must be {requirement}")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not isfinite(self.validation_fraction)
            or not 0 < self.validation_fraction < 1
        ):
            raise TransformerLabError(
                "validation_fraction must be finite and between zero and one"
            )


@dataclass(frozen=True)
class BoundedCorpus:
    """A disjoint ordered split with explicit full- and used-token counts."""

    corpus: TokenCorpus
    source: CorpusSource
    total_token_count: int
    split_index: int
    full_train_token_count: int
    full_validation_token_count: int

    @property
    def used_train_token_count(self) -> int:
        return len(self.corpus.train_tokens)

    @property
    def used_validation_token_count(self) -> int:
        return len(self.corpus.validation_tokens)


@dataclass(frozen=True)
class Day38TrainingConfig:
    """Architecture, optimizer, and sampling controls for the full model."""

    data: Day38DataConfig = Day38DataConfig()
    embedding_dim: int = 32
    head_count: int = 4
    layer_count: int = 2
    dropout: float = 0.2
    steps: int = 80
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 38
    sample_prompt: str = "\n"
    sample_tokens: int = 200
    sample_temperature: float = 0.8
    sample_top_k: int | None = 20
    sample_seed: int = 3_800

    def __post_init__(self) -> None:
        if not isinstance(self.data, Day38DataConfig):
            raise TypeError("data must be Day38DataConfig")
        for name in (
            "embedding_dim",
            "head_count",
            "layer_count",
            "steps",
            "batch_size",
            "seed",
            "sample_tokens",
            "sample_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in ("seed", "sample_seed"):
            if not 0 <= getattr(self, name) < 2**32:
                raise TransformerLabError(f"{name} must be in NumPy's range [0, 2**32)")
        for name in (
            "embedding_dim",
            "head_count",
            "layer_count",
            "steps",
            "batch_size",
            "sample_tokens",
        ):
            if getattr(self, name) <= 0:
                raise TransformerLabError(f"{name} must be positive")
        if self.embedding_dim % self.head_count:
            raise TransformerLabError("embedding_dim must be divisible by head_count")
        if not isinstance(self.sample_prompt, str) or not self.sample_prompt:
            raise TransformerLabError("sample_prompt must be a non-empty string")
        if self.sample_top_k is not None and (
            isinstance(self.sample_top_k, bool)
            or not isinstance(self.sample_top_k, int)
            or self.sample_top_k <= 0
        ):
            raise TransformerLabError("sample_top_k must be a positive integer")
        for name in (
            "dropout",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "sample_temperature",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise TransformerLabError(f"{name} must be finite")
        if not 0 <= self.dropout < 1:
            raise TransformerLabError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise TransformerLabError(
                "learning_rate and gradient_clip must be positive"
            )
        if self.weight_decay < 0:
            raise TransformerLabError("weight_decay must be non-negative")
        if self.sample_temperature <= 0:
            raise TransformerLabError("sample_temperature must be positive")

    def model_config(self, vocab_size: int) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=vocab_size,
            block_size=self.data.block_size,
            embedding_dim=self.embedding_dim,
            head_count=self.head_count,
            layer_count=self.layer_count,
            dropout=self.dropout,
        )

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(
            steps=self.steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            seed=self.seed,
        )


@dataclass(frozen=True)
class Day38Experiment:
    """Measured training result plus a sample replayed from the final checkpoint."""

    source: CorpusSource
    config: Day38TrainingConfig
    total_token_count: int
    split_index: int
    full_train_token_count: int
    full_validation_token_count: int
    used_train_token_count: int
    used_validation_token_count: int
    corpus_fingerprint: str
    codec_fingerprint: str
    model_fingerprint: str
    parameter_count: int
    completed_steps: int
    initial_train_nll: float
    final_train_nll: float
    initial_validation_nll: float
    final_validation_nll: float
    trace: tuple[dict[str, Any], ...]
    sample: str
    sample_token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = asdict(self.source)
        payload["config"]["data"] = asdict(self.config.data)
        payload["sample_token_ids"] = list(self.sample_token_ids)
        payload["trace"] = list(self.trace)
        return payload


@contextmanager
def _preserve_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic)


def prepare_bounded_corpus(
    text: str, source: CorpusSource, config: Day38DataConfig
) -> BoundedCorpus:
    """Verify source bytes and return deterministic, disjoint bounded partitions."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(source, CorpusSource):
        raise TypeError("source must be CorpusSource")
    if not isinstance(config, Day38DataConfig):
        raise TypeError("config must be Day38DataConfig")
    validate_corpus_bytes(text.encode("utf-8"), source)
    codec = CharacterCodec.from_text(text)
    tokens = torch.tensor(codec.encode(text), dtype=torch.long)
    split_index = int(len(tokens) * (1 - config.validation_fraction))
    full_train = tokens[:split_index]
    full_validation = tokens[split_index:]
    minimum = config.block_size + 1
    if len(full_train) < minimum or len(full_validation) < minimum:
        raise TransformerLabError(
            "source split must provide more than block_size tokens per partition"
        )
    train = full_train[: config.train_token_limit].clone()
    validation = full_validation[: config.validation_token_limit].clone()
    if len(train) < minimum or len(validation) < minimum:
        raise TransformerLabError(
            "bounded partitions must provide more than block_size tokens"
        )
    corpus = TokenCorpus(
        codec=codec,
        train_tokens=train,
        validation_tokens=validation,
        source_sha256=source.sha256,
    )
    return BoundedCorpus(
        corpus=corpus,
        source=source,
        total_token_count=len(tokens),
        split_index=split_index,
        full_train_token_count=len(full_train),
        full_validation_token_count=len(full_validation),
    )


def run_day38_experiment(
    text: str,
    source: CorpusSource,
    config: Day38TrainingConfig,
    *,
    checkpoint_path: Path,
    resume: bool = False,
) -> Day38Experiment:
    """Train the complete decoder stack and replay a sample from its checkpoint."""

    if not isinstance(config, Day38TrainingConfig):
        raise TypeError("config must be Day38TrainingConfig")
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be pathlib.Path")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")
    bounded = prepare_bounded_corpus(text, source, config.data)
    corpus = bounded.corpus
    model_config = config.model_config(corpus.vocab_size)
    training_config = config.training_config()
    corpus.codec.encode(config.sample_prompt)
    with _preserve_random_state():
        result = run_transformer_experiment(
            corpus,
            model_config=model_config,
            training_config=training_config,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )
        seed_everything(training_config.seed)
        model = DecoderLanguageModel(model_config)
        optimizer = build_optimizer(model, training_config)
        cursor = BatchCursor(
            corpus.train_tokens,
            block_size=model_config.block_size,
            batch_size=training_config.batch_size,
            seed=training_config.seed,
        )
        restored_step = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            cursor=cursor,
            training_config=training_config,
            corpus_fingerprint=corpus.fingerprint(),
        )
        if restored_step != result.completed_steps:
            raise RuntimeError("checkpoint step does not match the experiment")
        if model_fingerprint(model) != result.model_fingerprint:
            raise RuntimeError("checkpoint model does not match the experiment")
        prompt = torch.tensor(
            [corpus.codec.encode(config.sample_prompt)], dtype=torch.long
        )
        sample_generator = torch.Generator(device="cpu").manual_seed(config.sample_seed)
        generated = model.generate(
            prompt,
            new_tokens=config.sample_tokens,
            temperature=config.sample_temperature,
            top_k=config.sample_top_k,
            generator=sample_generator,
        )
        sample_ids = tuple(int(value) for value in generated[0].tolist())
    return Day38Experiment(
        source=source,
        config=config,
        total_token_count=bounded.total_token_count,
        split_index=bounded.split_index,
        full_train_token_count=bounded.full_train_token_count,
        full_validation_token_count=bounded.full_validation_token_count,
        used_train_token_count=bounded.used_train_token_count,
        used_validation_token_count=bounded.used_validation_token_count,
        corpus_fingerprint=result.corpus_fingerprint,
        codec_fingerprint=result.codec_fingerprint,
        model_fingerprint=result.model_fingerprint,
        parameter_count=result.parameter_count,
        completed_steps=result.completed_steps,
        initial_train_nll=result.initial_train_nll,
        final_train_nll=result.final_train_nll,
        initial_validation_nll=result.initial_validation_nll,
        final_validation_nll=result.final_validation_nll,
        trace=tuple(asdict(metric) for metric in result.trace),
        sample=corpus.codec.decode(sample_ids),
        sample_token_ids=sample_ids,
    )


def _evidence_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(canonical.encode()).hexdigest()


def build_day38_report(
    experiment: Day38Experiment, *, elapsed_seconds: float
) -> dict[str, Any]:
    """Build a tamper-evident report with runtime and scope limitations."""

    if not isinstance(experiment, Day38Experiment):
        raise TypeError("experiment must be Day38Experiment")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise TransformerLabError("elapsed_seconds must be finite and non-negative")
    payload = {
        "schema_version": DAY38_REPORT_SCHEMA_VERSION,
        "experiment": experiment.to_dict(),
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "intraop_threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
            "elapsed_seconds": float(elapsed_seconds),
        },
        "finding": (
            "training-and-validation-nll-decreased"
            if experiment.final_train_nll < experiment.initial_train_nll
            and experiment.final_validation_nll < experiment.initial_validation_nll
            else "declared-loss-improvement-not-observed"
        ),
        "limitations": [
            "bounded CPU character-level experiment",
            "sample quality is diagnostic rather than a model-quality claim",
            "no GPU, distributed, production, or learner-completion evidence",
        ],
    }
    payload["evidence_fingerprint"] = _evidence_fingerprint(payload)
    verify_day38_report(payload)
    return payload


def verify_day38_report(report: dict[str, Any]) -> None:
    """Reject malformed, nonfinite, or fingerprint-mismatched Day 38 evidence."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    expected = {
        "schema_version",
        "experiment",
        "runtime",
        "finding",
        "limitations",
        "evidence_fingerprint",
    }
    if set(report) != expected:
        raise TransformerLabError("report fields are invalid")
    if report["schema_version"] != DAY38_REPORT_SCHEMA_VERSION:
        raise TransformerLabError("unsupported report schema")
    fingerprint = report["evidence_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise TransformerLabError("evidence_fingerprint must be a SHA-256 digest")
    unsigned = {
        key: value for key, value in report.items() if key != "evidence_fingerprint"
    }
    try:
        expected_fingerprint = _evidence_fingerprint(unsigned)
    except (TypeError, ValueError) as exc:
        raise TransformerLabError("report values must be finite JSON data") from exc
    if fingerprint != expected_fingerprint:
        raise TransformerLabError("evidence fingerprint mismatch")
    experiment = report["experiment"]
    runtime = report["runtime"]
    if not isinstance(experiment, dict) or not isinstance(runtime, dict):
        raise TransformerLabError("experiment and runtime must be objects")
    metrics = (
        "initial_train_nll",
        "final_train_nll",
        "initial_validation_nll",
        "final_validation_nll",
    )
    if any(
        isinstance(experiment.get(name), bool)
        or not isinstance(experiment.get(name), (int, float))
        or not isfinite(experiment[name])
        for name in metrics
    ):
        raise TransformerLabError("loss metrics must be finite numbers")
    config = experiment.get("config")
    if not isinstance(config, dict):
        raise TransformerLabError("experiment config must be an object")
    sample = experiment.get("sample")
    sample_ids = experiment.get("sample_token_ids")
    expected_sample_length = len(config.get("sample_prompt", "")) + config.get(
        "sample_tokens", -1
    )
    if (
        not isinstance(sample, str)
        or not isinstance(sample_ids, list)
        or len(sample) != expected_sample_length
        or len(sample_ids) != expected_sample_length
    ):
        raise TransformerLabError("sample evidence length is inconsistent")
    elapsed = runtime.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not isfinite(elapsed)
        or elapsed < 0
    ):
        raise TransformerLabError(
            "runtime elapsed_seconds must be finite and non-negative"
        )


def write_day38_report(path: Path, report: dict[str, Any]) -> None:
    """Verify and durably publish a complete JSON evidence snapshot."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    verify_day38_report(report)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
