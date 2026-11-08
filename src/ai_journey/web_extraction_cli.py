"""Build or independently verify replayable HTML-to-text evidence."""

import argparse
import json
from pathlib import Path

from .bpe import TokenizerError
from .web_corpus import extract_web_corpus
from .web_extraction_config import load_extraction_policy
from .web_extraction_io import publish_extraction_artifacts
from .web_extraction_verify import verify_extraction_artifacts
from .web_snapshot import load_web_snapshot_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="extract a pinned HTML snapshot")
    build.add_argument("--snapshot", required=True, type=Path)
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify published extraction evidence")
    verify.add_argument("--evidence", required=True, type=Path)
    return parser


def _summary(report: dict) -> str:
    return json.dumps(
        {
            "accepted_pages": report["counts"]["accepted_pages"],
            "rejected_pages": report["counts"]["rejected_pages"],
            "extracted_characters": report["counts"]["extracted_characters"],
            "report_sha256": report["report_sha256"],
        },
        sort_keys=True,
        allow_nan=False,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            snapshot_path = args.snapshot.resolve()
            config_path = args.config.resolve()
            output = args.output.resolve()
            if output in {snapshot_path, config_path}:
                raise TokenizerError("HTML extraction output must not alias an input")
            snapshot = load_web_snapshot_evidence(snapshot_path)
            policy = load_extraction_policy(config_path)
            result = extract_web_corpus(snapshot.pages, policy)
            report = publish_extraction_artifacts(
                output,
                result,
                policy,
                snapshot_sha256=snapshot.sha256,
            )
        else:
            report = verify_extraction_artifacts(args.evidence.resolve())
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
