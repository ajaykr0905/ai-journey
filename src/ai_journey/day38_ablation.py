"""Run the predeclared Day 38 dropout comparison on bounded public data."""

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from .ablation_protocol import (
    build_ablation_report,
    load_ablation_protocol,
    run_ablation,
)
from .day38_inference_cli import protect_output
from .day38_training import (
    Day38DataConfig,
    _preserve_random_state,
    prepare_bounded_corpus,
)
from .gpt_bigram import (
    CorpusSource,
    GPTBigramError,
    TINY_SHAKESPEARE,
    load_verified_corpus,
)
from .serving_benchmark import _publish
from .transformer_lab import (
    CharacterCodec,
    TransformerLabError,
    expected_parameter_count,
)


def run_dropout_ablation(
    text: str,
    source: CorpusSource,
    protocol_path: Path,
    *,
    train_tokens: int = 4096,
    validation_tokens: int = 1024,
) -> dict:
    """Keep initialization and sampled batches matched within each trial seed."""
    protocol = load_ablation_protocol(
        protocol_path, vocab_size=len(CharacterCodec.from_text(text).tokens)
    )
    if protocol.independent_variable != "model.dropout":
        raise TransformerLabError(
            "Day 38 comparison requires model.dropout as its only variable"
        )
    if len(protocol.arms) * len(protocol.trial_seeds) > 12:
        raise TransformerLabError("comparison permits at most 12 bounded trials")
    for arm in protocol.arms:
        if (
            expected_parameter_count(arm.model_config) > 1_000_000
            or arm.model_config.block_size > 128
            or arm.training_config.steps > 500
            or arm.training_config.batch_size > 64
        ):
            raise TransformerLabError("ablation exceeds the CPU experiment budget")
    data = Day38DataConfig(
        block_size=protocol.arms[0].model_config.block_size,
        train_token_limit=train_tokens,
        validation_token_limit=validation_tokens,
    )
    if train_tokens > 100_000 or validation_tokens > 20_000:
        raise TransformerLabError("ablation exceeds the token budget")
    corpus = prepare_bounded_corpus(text, source, data).corpus
    started = perf_counter()
    with _preserve_random_state():
        result = run_ablation(corpus, protocol)
    report = {
        "schema": "day38-dropout-ablation-v1",
        "source": asdict(source),
        "data": asdict(data),
        "corpus_fingerprint": corpus.fingerprint(),
        "elapsed_seconds": perf_counter() - started,
        "ablation": build_ablation_report(result),
    }
    report["evidence_fingerprint"] = sha256(
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=4096)
    parser.add_argument("--validation-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        protect_output(args.output, (args.corpus, args.protocol))
        text = load_verified_corpus(args.corpus, TINY_SHAKESPEARE)
        report = run_dropout_ablation(
            text,
            TINY_SHAKESPEARE,
            args.protocol,
            train_tokens=args.train_tokens,
            validation_tokens=args.validation_tokens,
        )
        _publish(args.output, report)
    except (GPTBigramError, TransformerLabError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
