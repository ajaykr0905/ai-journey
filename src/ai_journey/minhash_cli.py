"""Build reproducible MinHash deduplication evidence for Project 1 data."""

import argparse
import json
from pathlib import Path

from .bpe import TokenizerError
from .minhash_config import load_minhash_policy
from .minhash_corpus import load_project1_corpus
from .minhash_dedup import deduplicate_with_minhash
from .minhash_io import publish_minhash_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="deduplicate a verified Project 1 dataset"
    )
    build.add_argument("--dataset", required=True, type=Path)
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    return parser


def _summary(report: dict) -> str:
    return json.dumps(
        {
            "input_documents": report["metrics"]["input_documents"],
            "kept_documents": report["metrics"]["kept_documents"],
            "removed_documents": report["metrics"]["removed_documents"],
            "removed_fraction": report["metrics"]["removed_fraction"],
            "report_sha256": report["report_sha256"],
        },
        sort_keys=True,
        allow_nan=False,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset = args.dataset.resolve()
        config = args.config.resolve()
        output = args.output.resolve()
        if output in {dataset, config}:
            raise TokenizerError("MinHash output must not alias an input")
        corpus = load_project1_corpus(dataset)
        policy = load_minhash_policy(config)
        result = deduplicate_with_minhash(corpus.documents, policy)
        report = publish_minhash_artifacts(output, corpus, policy, result)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
