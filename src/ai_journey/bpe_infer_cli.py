"""Serve strict UTF-8 from a fully verified local BPE training artifact."""

import argparse
from pathlib import Path

from .bpe import TokenizerError
from .bpe_artifact import load_bpe_artifact
from .bpe_serving import generate_utf8
from .bpe_serving_benchmark import benchmark_bpe_serving
from .serving_benchmark import _publish
from .transformer_lab import model_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "checkpoint", "tokenizer", "train", "validation", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument(
        "--operation", choices=("generate", "benchmark"), default="generate"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        inputs = [
            getattr(args, name)
            for name in ("report", "checkpoint", "tokenizer", "train", "validation")
        ]
        if args.output.resolve() in {path.resolve() for path in inputs}:
            raise TokenizerError(
                "generation output must preserve all training artifacts"
            )
        artifact = load_bpe_artifact(*inputs)
        if args.operation == "benchmark":
            benchmark = benchmark_bpe_serving(
                artifact,
                args.prompt,
                new_tokens=args.new_tokens,
                repeats=args.repeats,
                seed=args.seed,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            _publish(args.output, benchmark)
            print(f"Verified paired BPE serving benchmark; wrote {args.output}")
            return 0
        sample = generate_utf8(
            artifact.model,
            artifact.corpus.codec,
            args.prompt,
            new_tokens=args.new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            cached=args.cached,
        )
        _publish(
            args.output,
            {
                "schema_version": 1,
                "model_sha256": model_fingerprint(artifact.model),
                "codec_sha256": artifact.corpus.codec.fingerprint(),
                "corpus_sha256": artifact.corpus.fingerprint(),
                "controls": {
                    name: getattr(args, name)
                    for name in (
                        "prompt",
                        "new_tokens",
                        "seed",
                        "temperature",
                        "top_k",
                        "cached",
                    )
                },
                "sample": sample,
            },
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Verified strict UTF-8 sample; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
