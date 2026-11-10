"""Exact source replay for published SFT dataset evidence."""

from pathlib import Path

from .sft_dataset_build import SFTDatasetPolicy, build_sft_dataset
from .sft_dataset_record import SFTDatasetError
from .sft_dataset_report import build_sft_manifest
from .sft_dataset_source import SFTSourceSpec, load_dolly_snapshot
from .sft_dataset_verify import verify_sft_dataset


def replay_sft_dataset(source_path: Path, evidence: Path) -> dict:
    """Rebuild artifacts from pinned bytes and require exact public evidence."""
    if not isinstance(source_path, Path) or not isinstance(evidence, Path):
        raise TypeError("source_path and evidence must be pathlib.Path values")
    manifest = verify_sft_dataset(evidence)
    source = SFTSourceSpec.from_mapping(manifest["source"])
    policy = SFTDatasetPolicy.from_mapping(manifest["policy"])
    records = load_dolly_snapshot(source_path, source)
    build = build_sft_dataset(records, policy)
    expected_manifest = build_sft_manifest(source, policy, build)
    if expected_manifest != manifest:
        raise SFTDatasetError("replayed SFT manifest does not match published evidence")
    if (evidence / "dataset.jsonl").read_bytes() != build.artifacts.dataset_jsonl:
        raise SFTDatasetError("replayed message dataset bytes do not match evidence")
    if (evidence / "chatml.jsonl").read_bytes() != build.artifacts.chatml_jsonl:
        raise SFTDatasetError("replayed ChatML dataset bytes do not match evidence")
    return manifest
