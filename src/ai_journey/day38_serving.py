"""Read-only operations on verified Day 38 training artifacts."""

from dataclasses import dataclass
from pathlib import Path

from .day38_training import (
    Day38DataConfig,
    Day38TrainingConfig,
    _preserve_random_state,
    load_day38_report,
    prepare_bounded_corpus,
)
from .gpt_bigram import CorpusSource, load_verified_corpus
from .transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TokenCorpus,
    TransformerLabError,
    build_optimizer,
    expected_parameter_count,
    load_training_checkpoint,
    model_fingerprint,
)


@dataclass
class TrainedArtifact:
    """Weights and corpus bound to a verified training report."""

    model: DecoderLanguageModel
    corpus: TokenCorpus
    report: dict


def load_trained_artifact(
    report_path: Path,
    checkpoint_path: Path,
    corpus_path: Path,
    *,
    max_parameters: int = 1_000_000,
) -> TrainedArtifact:
    """Validate source, codec, sampler, step, and weights before returning a model."""
    report = load_day38_report(report_path)
    evidence = report["experiment"]
    settings = dict(evidence["config"])
    data = Day38DataConfig(**settings.pop("data"))
    config = Day38TrainingConfig(data=data, **settings)
    architecture = config.model_config(evidence["vocab_size"])
    if (
        isinstance(max_parameters, bool)
        or not isinstance(max_parameters, int)
        or max_parameters <= 0
        or expected_parameter_count(architecture) > max_parameters
    ):
        raise TransformerLabError("trained model exceeds the parameter budget")
    source = CorpusSource(**evidence["source"])
    text = load_verified_corpus(corpus_path, source)
    corpus = prepare_bounded_corpus(text, source, data).corpus
    if (
        corpus.fingerprint() != evidence["corpus_fingerprint"]
        or corpus.codec.fingerprint() != evidence["codec_fingerprint"]
    ):
        raise TransformerLabError("training report does not match the corpus")
    with _preserve_random_state():
        model = DecoderLanguageModel(architecture)
        training = config.training_config()
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(
            corpus.train_tokens,
            block_size=data.block_size,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        step = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            cursor=cursor,
            training_config=training,
            corpus_fingerprint=corpus.fingerprint(),
        )
        if (
            step != evidence["completed_steps"]
            or model_fingerprint(model) != evidence["model_fingerprint"]
        ):
            raise TransformerLabError("checkpoint does not match the training report")
    model.eval()
    return TrainedArtifact(model, corpus, report)
