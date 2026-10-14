"""Controlled vocabulary-size experiments on aligned held-out translations."""

from hashlib import sha256
import json
from time import perf_counter

from .bpe import ByteTokenizer, TokenizerError
from .parallel_corpus import ParallelSplit, parallel_folds
from .tokenizer_io import tokenizer_fingerprint
from .tokenizer_metrics import compression_metrics
from .tokenizer_training import train_documents


def _partition_identity(records):
    return sha256(
        json.dumps(
            [(row.sentence_id, row.text) for row in records], ensure_ascii=False
        ).encode()
    ).hexdigest()


def run_compression_experiment(
    split: ParallelSplit, *, vocab_sizes=(272, 288, 304), pattern=None
):
    """Vary only vocabulary target within each source-language family.

    No model-quality claim is made: token compression is not perplexity.
    Elapsed times are local CPU observations, not portable benchmark claims.
    """
    if not isinstance(split, ParallelSplit):
        raise TypeError("experiment requires an aligned ParallelSplit")
    if (
        not isinstance(vocab_sizes, tuple)
        or len(vocab_sizes) != 3
        or len(set(vocab_sizes)) != 3
    ):
        raise TokenizerError("experiment requires three distinct vocabulary targets")
    for target in vocab_sizes:
        ByteTokenizer.train("", vocab_size=target)
    for left, right in (
        (split.first_train, split.second_train),
        (split.first_validation, split.second_validation),
    ):
        if not left or tuple(row.sentence_id for row in left) != tuple(
            row.sentence_id for row in right
        ):
            raise TokenizerError("translation partitions must remain aligned")
    train_ids = [row.sentence_id for row in split.first_train]
    heldout_ids = [row.sentence_id for row in split.first_validation]
    if (
        len(set(train_ids)) != len(train_ids)
        or len(set(heldout_ids)) != len(heldout_ids)
        or set(train_ids) & set(heldout_ids)
    ):
        raise TokenizerError("partition ids must be unique and disjoint")
    train = (split.first_train, split.second_train)
    heldout = (split.first_validation, split.second_validation)
    evaluations = []
    for family, records in enumerate(train):
        documents = tuple(row.text for row in records)
        for target in vocab_sizes:
            started = perf_counter()
            tokenizer = train_documents(documents, vocab_size=target, pattern=pattern)
            training_seconds = perf_counter() - started
            evaluations.append(
                {
                    "training_partition": family,
                    "requested_vocabulary": target,
                    "actual_vocabulary": tokenizer.vocab_size,
                    "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                    "training_seconds": training_seconds,
                    "heldout": [
                        compression_metrics(
                            tokenizer, tuple(row.text for row in partition)
                        ).as_dict()
                        for partition in heldout
                    ],
                }
            )
    return {
        "schema_version": 1,
        "experiment": "parallel-heldout-vocabulary-compression",
        "pattern": pattern,
        "vocab_sizes": list(vocab_sizes),
        "partition_sha256": [
            _partition_identity(partition) for partition in (*train, *heldout)
        ],
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "byte_baseline": [
            compression_metrics(
                ByteTokenizer(), tuple(row.text for row in partition)
            ).as_dict()
            for partition in heldout
        ],
        "evaluations": evaluations,
        "limitations": "Local CPU tokenizer compression only; no model-quality, GPU, or learner-completion evidence.",
    }


def run_compression_cross_validation(
    first, second, *, vocab_sizes=(272, 288, 304), pattern=None
):
    """Use PUD's recommended ten-fold training protocol, not its test labels."""
    folds = [
        run_compression_experiment(split, vocab_sizes=vocab_sizes, pattern=pattern)
        for split in parallel_folds(first, second, fold_count=10)
    ]
    aggregate = []
    for trial in range(6):
        rows = [fold["evaluations"][trial] for fold in folds]
        heldout = []
        for language in range(2):
            metrics = [row["heldout"][language] for row in rows]
            totals = {
                field: sum(metric[field] for metric in metrics)
                for field in (
                    "documents",
                    "utf8_bytes",
                    "characters",
                    "tokens",
                    "byte_fallback_tokens",
                )
            }
            totals["bytes_per_token"] = totals["utf8_bytes"] / totals["tokens"]
            totals["characters_per_token"] = totals["characters"] / totals["tokens"]
            heldout.append(totals)
        aggregate.append(
            {
                "training_partition": rows[0]["training_partition"],
                "requested_vocabulary": rows[0]["requested_vocabulary"],
                "heldout": heldout,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "parallel-ten-fold-vocabulary-compression",
        "fold_count": 10,
        "folds": folds,
        "aggregate": aggregate,
        "limitations": "Token compression on public text only; no parsing score or language-model quality claim.",
    }
