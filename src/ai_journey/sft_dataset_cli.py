"""Build verified, reproducible supervised fine-tuning datasets."""

import argparse
import json
from pathlib import Path

from .sft_dataset_build import SFTDatasetPolicy, build_sft_dataset
from .sft_dataset_io import publish_sft_dataset
from .sft_dataset_record import SFTDatasetError
from .sft_dataset_replay import replay_sft_dataset
from .sft_dataset_source import SFTSourceSpec, load_dolly_snapshot
from .sft_dataset_verify import verify_sft_dataset


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SFTDatasetError(f"duplicate config field: {key}")
        value[key] = item
    return value


def load_build_config(path: Path) -> tuple[SFTSourceSpec, SFTDatasetPolicy]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise SFTDatasetError("SFT config must be a regular file")
    payload = path.read_bytes()
    if not 1 <= len(payload) <= 64 * 1024:
        raise SFTDatasetError("SFT config exceeds its byte budget")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SFTDatasetError("SFT config is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {"source", "policy"}:
        raise SFTDatasetError("SFT config requires source and policy objects")
    return (
        SFTSourceSpec.from_mapping(value["source"]),
        SFTDatasetPolicy.from_mapping(value["policy"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="build a dataset from pinned public bytes"
    )
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify published dataset evidence")
    verify.add_argument("--evidence", required=True, type=Path)
    replay = commands.add_parser(
        "replay", help="rebuild evidence from pinned source bytes"
    )
    replay.add_argument("--source", required=True, type=Path)
    replay.add_argument("--evidence", required=True, type=Path)
    return parser


def _summary(manifest: dict) -> str:
    return json.dumps(
        {
            "examples": manifest["metrics"]["examples"],
            "train_examples": manifest["metrics"]["train_examples"],
            "validation_examples": manifest["metrics"]["validation_examples"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        sort_keys=True,
        allow_nan=False,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            source_path = args.source.resolve()
            config_path = args.config.resolve()
            output = args.output.resolve()
            if output in {source_path, config_path}:
                raise SFTDatasetError("SFT output must not alias an input")
            source, policy = load_build_config(config_path)
            records = load_dolly_snapshot(source_path, source)
            build = build_sft_dataset(records, policy)
            manifest = publish_sft_dataset(output, source, policy, build)
        elif args.command == "verify":
            manifest = verify_sft_dataset(args.evidence.resolve())
        else:
            manifest = replay_sft_dataset(
                args.source.resolve(), args.evidence.resolve()
            )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
