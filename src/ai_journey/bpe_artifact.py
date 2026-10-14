"""Read-only verification of BPE-trained weights, data, codec, and loss evidence."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from .bpe import RegexTokenizer, SpecialTokenizer, TokenizerError
from .bpe_corpus import EOS_MARKER, build_bpe_corpus
from .tokenizer_cli import read_documents
from .tokenizer_io import load_tokenizer, tokenizer_fingerprint
from .transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TokenCorpus,
    TransformerConfig,
    TrainingConfig,
    build_optimizer,
    evaluate_nll,
    expected_parameter_count,
    load_training_checkpoint,
    model_fingerprint,
)


@dataclass(frozen=True)
class BPEArtifact:
    model: DecoderLanguageModel
    corpus: TokenCorpus
    report: dict


def load_bpe_artifact(
    report_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    train_path: Path,
    validation_path: Path,
):
    with report_path.open("rb") as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise TokenizerError("BPE report exceeds 1 MiB")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TokenizerError("duplicate BPE report field")
            result[key] = value
        return result

    report = json.loads(payload, object_pairs_hook=unique)
    if (
        not isinstance(report, dict)
        or set(report) != {"schema_version", "experiment", "runtime_scope"}
        or type(report["schema_version"]) is not int
        or report["schema_version"] != 1
    ):
        raise TokenizerError("BPE report fields are invalid")
    evidence = report["experiment"]
    if not isinstance(evidence, dict):
        raise TokenizerError("BPE experiment evidence must be an object")
    architecture = TransformerConfig(**evidence["model_config"])
    training = TrainingConfig(**evidence["training_config"])
    if (
        expected_parameter_count(architecture) > 100000
        or architecture.block_size > 32
        or architecture.vocab_size > 513
        or training.steps > 100
    ):
        raise TokenizerError("BPE artifact exceeds the CPU serving budget")
    tokenizer = load_tokenizer(tokenizer_path)
    if not isinstance(tokenizer, SpecialTokenizer) or tokenizer.names != (EOS_MARKER,):
        raise TokenizerError("BPE artifact requires its registered EOS tokenizer")
    train, validation = read_documents(train_path), read_documents(validation_path)
    if sum(len(text.encode()) for text in (*train, *validation)) > 64 * 1024:
        raise TokenizerError("BPE artifact data exceeds 64 KiB")
    base = tokenizer.tokenizer
    corpus = build_bpe_corpus(
        train,
        validation,
        vocab_size=base.vocab_size,
        pattern=base.pattern if isinstance(base, RegexTokenizer) else None,
    )
    if max(len(corpus.train_tokens), len(corpus.validation_tokens)) > 4096:
        raise TokenizerError("BPE artifact token partitions exceed 4096 tokens")
    if (
        tokenizer_fingerprint(tokenizer) != corpus.codec.fingerprint()
        or evidence["codec_fingerprint"] != corpus.codec.fingerprint()
        or evidence["corpus_fingerprint"] != corpus.fingerprint()
    ):
        raise TokenizerError("BPE artifact data or codec identity mismatch")
    with torch.random.fork_rng(devices=[]):
        model = DecoderLanguageModel(architecture)
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(
            corpus.train_tokens,
            block_size=architecture.block_size,
            batch_size=training.batch_size,
            seed=training.seed,
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
            type(evidence["completed_steps"]) is not int
            or step != evidence["completed_steps"]
            or step != training.steps
            or model_fingerprint(model) != evidence["model_fingerprint"]
            or evidence["parameter_count"] != model.parameter_count
        ):
            raise TokenizerError(
                "BPE weights do not match the completed training evidence"
            )
        for name, tokens in (
            ("final_train_nll", corpus.train_tokens),
            ("final_validation_nll", corpus.validation_tokens),
        ):
            saved = evidence[name]
            if (
                isinstance(saved, bool)
                or not isinstance(saved, (int, float))
                or not math.isfinite(saved)
                or not math.isclose(
                    saved, evaluate_nll(model, tokens), rel_tol=1e-6, abs_tol=1e-6
                )
            ):
                raise TokenizerError(
                    "BPE reported loss does not match the trained weights"
                )
    model.eval()
    return BPEArtifact(model, corpus, report)
