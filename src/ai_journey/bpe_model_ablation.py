"""Matched-seed CPU decoder experiments varying only the requested BPE budget."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from time import perf_counter

import torch

from .bpe import TokenizerError
from .bpe_corpus import build_bpe_corpus
from .bpe_likelihood import score_documents, summarize_likelihood
from .day38_training import _preserve_random_state
from .tokenizer_training import training_segments
from .transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TransformerConfig,
    TrainingConfig,
    build_optimizer,
    model_fingerprint,
    seed_everything,
    train_steps,
)


@dataclass(frozen=True)
class VocabularyProtocol:
    vocab_sizes: tuple[int, ...] = (256, 272, 304)
    seeds: tuple[int, ...] = (41, 42, 43)
    steps: int = 20
    block_size: int = 8
    batch_size: int = 4
    hypothesis: str = "More BPE merges reduce content token count; held-out bits per byte need not improve."

    def __post_init__(self):
        if (
            not isinstance(self.vocab_sizes, tuple)
            or len(self.vocab_sizes) != 3
            or len(set(self.vocab_sizes)) != 3
            or self.vocab_sizes[0] != 256
            or any(
                type(size) is not int or not 256 <= size <= 512
                for size in self.vocab_sizes
            )
        ):
            raise TokenizerError(
                "protocol requires byte baseline and three distinct budgets in [256, 512]"
            )
        if (
            not isinstance(self.seeds, tuple)
            or not 2 <= len(self.seeds) <= 5
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds
            )
        ):
            raise TokenizerError(
                "protocol requires 2 to 5 distinct bounded integer seeds"
            )
        for name, lower, upper in (
            ("steps", 1, 100),
            ("block_size", 2, 32),
            ("batch_size", 1, 16),
        ):
            if (
                type(getattr(self, name)) is not int
                or not lower <= getattr(self, name) <= upper
            ):
                raise TokenizerError(f"{name} exceeds the bounded CPU protocol")
        if (
            not isinstance(self.hypothesis, str)
            or not 1 <= len(self.hypothesis) <= 1000
        ):
            raise TokenizerError("protocol requires a predeclared bounded hypothesis")


def protocol_fingerprint(protocol: VocabularyProtocol) -> str:
    return sha256(json.dumps(asdict(protocol), sort_keys=True).encode()).hexdigest()


def _trunk_fingerprint(model):
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        if name not in {"token_embedding.weight", "lm_head.weight"}:
            digest.update(name.encode())
            digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_vocabulary_ablation(train, validation, protocol=VocabularyProtocol()):
    """Fit train-only tokenizers, train decoders, and return held-out evidence.

    Equal steps/batch/context match token exposure, not byte exposure or FLOPs.
    Vocabulary-dependent embedding/head sizes and context-in-bytes are reported
    confounders, not evidence for a uniquely optimal vocabulary size.
    """
    if not isinstance(protocol, VocabularyProtocol):
        raise TypeError("protocol must be VocabularyProtocol")
    training_segments(train)
    training_segments(validation)
    if (
        len(train) > 100
        or len(validation) > 100
        or sum(len(text.encode()) for text in (*train, *validation)) > 65536
    ):
        raise TokenizerError("ablation text exceeds the bounded CPU budget")
    if set(train) & set(validation):
        raise TokenizerError("training text leaks into held-out documents")
    corpora = [
        build_bpe_corpus(train, validation, vocab_size=size)
        for size in protocol.vocab_sizes
    ]
    if any(
        len(corpus.train_tokens) <= protocol.block_size
        or len(corpus.train_tokens) > 8192
        for corpus in corpora
    ):
        raise TokenizerError(
            "training partitions must fit the bounded token/context budget"
        )
    if any(len(corpus.validation_tokens) > 8192 for corpus in corpora):
        raise TokenizerError("held-out partitions exceed the token budget")
    if len({corpus.vocab_size for corpus in corpora}) != 3:
        raise TokenizerError(
            "training text cannot realize three distinct vocabulary sizes"
        )
    trials = []
    with _preserve_random_state():
        for seed in protocol.seeds:
            reference = None
            for requested, corpus in zip(protocol.vocab_sizes, corpora):
                seed_everything(seed)
                model = DecoderLanguageModel(
                    TransformerConfig(
                        vocab_size=corpus.vocab_size,
                        block_size=protocol.block_size,
                        embedding_dim=16,
                        head_count=2,
                        layer_count=1,
                    )
                )
                if reference is None:
                    reference = {
                        name: value.clone()
                        for name, value in model.state_dict().items()
                    }
                else:
                    with torch.no_grad():
                        for name, value in model.state_dict().items():
                            if name in {"token_embedding.weight", "lm_head.weight"}:
                                value[:256].copy_(reference[name][:256])
                                value[-1].copy_(reference[name][-1])
                            else:
                                value.copy_(reference[name])
                initial_trunk = _trunk_fingerprint(model)
                initial_model = model_fingerprint(model)
                initial = summarize_likelihood(
                    score_documents(model, corpus.codec, validation)
                )
                training = TrainingConfig(
                    steps=protocol.steps, batch_size=protocol.batch_size, seed=seed
                )
                cursor = BatchCursor(
                    corpus.train_tokens,
                    block_size=protocol.block_size,
                    batch_size=protocol.batch_size,
                    seed=seed,
                )
                optimizer = build_optimizer(model, training)
                started = perf_counter()
                trace = train_steps(model, cursor, optimizer, training)
                elapsed = perf_counter() - started
                final = summarize_likelihood(
                    score_documents(model, corpus.codec, validation)
                )
                trials.append(
                    {
                        "seed": seed,
                        "requested_vocab_size": requested,
                        "actual_vocab_size_with_eos": corpus.vocab_size,
                        "parameter_count": model.parameter_count,
                        "corpus_fingerprint": corpus.fingerprint(),
                        "codec_fingerprint": corpus.codec.fingerprint(),
                        "initial_trunk_fingerprint": initial_trunk,
                        "initial_model_fingerprint": initial_model,
                        "final_model_fingerprint": model_fingerprint(model),
                        "training_seconds": elapsed,
                        "training_target_tokens": protocol.steps
                        * protocol.batch_size
                        * protocol.block_size,
                        "training_config": asdict(training),
                        "model_config": asdict(model.config),
                        "trace": [asdict(item) for item in trace],
                        "initial": initial,
                        "final": final,
                    }
                )
    return {
        "schema": "bpe-model-ablation-v1",
        "protocol": asdict(protocol),
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "source_fingerprint": corpora[0].source_sha256,
        "evaluation_policy": "EOS prefix per document; stride one; reset positions on rollover; exclude EOS from content bits per byte",
        "limitations": [
            "bounded CPU demonstration, not a UD benchmark",
            "canonical segmentation, not marginal string likelihood",
            "equal token exposure, unequal byte exposure and FLOPs",
            "vocabulary changes embedding/head parameters and context byte coverage",
            "no GPU, distributed, production, or learner evidence",
        ],
        "trials": trials,
    }
