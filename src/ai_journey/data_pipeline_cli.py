"""Build and independently verify the reproducible Day 59 dataset."""

import argparse
import json
from pathlib import Path

from .bpe import TokenizerError
from .data_pipeline import build_data_pipeline
from .data_pipeline_config import load_data_pipeline_config
from .data_pipeline_io import publish_pipeline_artifacts
from .data_pipeline_source import load_pipeline_source
from .data_pipeline_verify import replay_pipeline_artifacts


def _summary(report: dict) -> str:
    return json.dumps(
        {
            "total_tokens": report["total_tokens"],
            "total_utf8_bytes": report["total_utf8_bytes"],
            "bytes_per_token": report["bytes_per_token"],
            "token_reduction_fraction": report["token_reduction_fraction"],
        },
        sort_keys=True,
        allow_nan=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="build immutable public dataset artifacts"
    )
    build.add_argument("--snapshot", required=True, type=Path)
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify and replay dataset artifacts")
    verify.add_argument("--dataset", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            snapshot = args.snapshot.resolve()
            config_path = args.config.resolve()
            output = args.output.resolve()
            if output in {snapshot, config_path}:
                raise TokenizerError("dataset output must not alias an input")
            source = load_pipeline_source(snapshot)
            config = load_data_pipeline_config(config_path)
            result = build_data_pipeline(source.documents, config)
            report = publish_pipeline_artifacts(output, result, source)
            observations = report["tokenizer"]["report"]
        else:
            observations = replay_pipeline_artifacts(args.dataset.resolve())
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(_summary(observations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
