"""Operate local public-safe tokenizer checkpoints without external services."""

import argparse
import json
from pathlib import Path

from .bpe import DEFAULT_PATTERN, TokenizerError
from .bpe_conformance import build_training_audit, verify_training_audit
from .bpe_conformance_io import load_training_audit, save_training_audit
from .serving_benchmark import _publish
from .tokenizer_io import load_tokenizer, save_tokenizer, tokenizer_fingerprint
from .tokenizer_metrics import compression_metrics
from .tokenizer_training import train_documents


def read_documents(path: Path):
    with path.open("rb") as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise TokenizerError("text input exceeds 1 MiB")
    documents = tuple(line for line in payload.decode("utf-8").splitlines() if line)
    if not documents:
        raise TokenizerError("text input contains no documents")
    return documents


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--vocab-size", type=int, default=320)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--regex", action="store_true")
    encode = commands.add_parser("encode")
    encode.add_argument("--tokenizer", type=Path, required=True)
    encode.add_argument("--text", required=True)
    decode = commands.add_parser("decode")
    decode.add_argument("--tokenizer", type=Path, required=True)
    decode.add_argument("--ids", required=True, help="JSON array of token ids")
    audit = commands.add_parser("audit")
    audit.add_argument("--tokenizer", type=Path, required=True)
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    training_audit = commands.add_parser("audit-training")
    training_audit.add_argument("--tokenizer", type=Path, required=True)
    training_audit.add_argument("--corpus", type=Path, required=True)
    training_audit.add_argument("--output", type=Path, required=True)
    training_audit.add_argument("--min-frequency", type=int, default=2)
    verify_audit = commands.add_parser("verify-training-audit")
    verify_audit.add_argument("--tokenizer", type=Path, required=True)
    verify_audit.add_argument("--corpus", type=Path, required=True)
    verify_audit.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            if args.corpus.resolve() == args.output.resolve():
                raise TokenizerError("output must not overwrite the input corpus")
            tokenizer = train_documents(
                read_documents(args.corpus),
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                pattern=DEFAULT_PATTERN if args.regex else None,
            )
            save_tokenizer(args.output, tokenizer)
            print(
                json.dumps(
                    {
                        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                        "vocab_size": tokenizer.vocab_size,
                    }
                )
            )
        else:
            tokenizer = load_tokenizer(args.tokenizer)
            if args.command == "encode":
                print(json.dumps(tokenizer.encode(args.text)))
            elif args.command == "decode":
                print(tokenizer.decode(json.loads(args.ids)))
            elif args.command == "audit":
                if args.output.resolve() in (
                    args.corpus.resolve(),
                    args.tokenizer.resolve(),
                ):
                    raise TokenizerError("audit output must preserve input artifacts")
                metrics = compression_metrics(tokenizer, read_documents(args.corpus))
                _publish(
                    args.output,
                    {
                        "schema_version": 1,
                        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                        "metrics": metrics.as_dict(),
                    },
                )
            elif args.command == "audit-training":
                if args.output.resolve() in (
                    args.corpus.resolve(),
                    args.tokenizer.resolve(),
                ):
                    raise TokenizerError(
                        "training audit output must preserve input artifacts"
                    )
                report = build_training_audit(
                    read_documents(args.corpus),
                    tokenizer,
                    min_frequency=args.min_frequency,
                )
                save_training_audit(args.output, report)
                print(
                    json.dumps(
                        {
                            "audit_sha256": report["audit_sha256"],
                            "verified_merges": len(report["merges"]),
                        }
                    )
                )
            else:
                result = verify_training_audit(
                    load_training_audit(args.audit),
                    read_documents(args.corpus),
                    tokenizer,
                )
                print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
