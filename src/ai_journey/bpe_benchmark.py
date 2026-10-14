"""Reference-checked local encoder timings, not portable throughput claims."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter_ns

from .bpe import ByteTokenizer, TokenizerError
from .bpe_fast import encode_fast
from .serving_benchmark import _publish
from .tokenizer_cli import read_documents
from .tokenizer_io import load_tokenizer, tokenizer_fingerprint


def benchmark_encoding(tokenizer, documents, *, repeats=3):
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("encoder benchmark requires a byte or regex tokenizer")
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise TokenizerError("benchmark repeats must be in [1, 20]")
    if (
        not isinstance(documents, tuple)
        or not 1 <= len(documents) <= 64
        or any(not isinstance(text, str) or not text for text in documents)
    ):
        raise TokenizerError("benchmark requires 1..64 nonempty documents")
    size_bytes = sum(len(text.encode()) for text in documents)
    if size_bytes > 128 * 1024:
        raise TokenizerError("benchmark documents exceed 128 KiB")
    expected = tuple(tokenizer.encode(text) for text in documents)
    for text, ids in zip(documents, expected, strict=True):
        if tokenizer.decode(ids) != text or encode_fast(tokenizer, text) != ids:
            raise TokenizerError("encoder benchmark reference parity failed")
    timings = {"reference": [], "heap": []}
    for repeat in range(repeats):
        order = ("reference", "heap") if repeat % 2 == 0 else ("heap", "reference")
        for name in order:
            started = perf_counter_ns()
            result = tuple(
                tokenizer.encode(text)
                if name == "reference"
                else encode_fast(tokenizer, text)
                for text in documents
            )
            elapsed = perf_counter_ns() - started
            if result != expected or elapsed <= 0:
                raise TokenizerError(
                    "encoder benchmark parity or timing contract failed"
                )
            timings[name].append(elapsed)
    return {
        "schema_version": 1,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "documents_sha256": sha256(
            json.dumps(documents, ensure_ascii=False).encode()
        ).hexdigest(),
        "token_sequences_sha256": sha256(json.dumps(expected).encode()).hexdigest(),
        "utf8_bytes": size_bytes,
        "tokens": sum(map(len, expected)),
        "vocabulary_storage_bytes": sum(
            len(tokenizer.token_bytes(token)) for token in range(tokenizer.vocab_size)
        ),
        "elapsed_ns": timings,
        "python_version": platform.python_version(),
        "scope": "Local CPU observations after one parity warmup; no speed gate or production claim.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.limit <= 64 or args.output.resolve() in (
            args.corpus.resolve(),
            args.tokenizer.resolve(),
        ):
            raise TokenizerError("benchmark limits or output path are invalid")
        report = benchmark_encoding(
            load_tokenizer(args.tokenizer),
            read_documents(args.corpus)[: args.limit],
            repeats=args.repeats,
        )
        _publish(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified both encoders; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
