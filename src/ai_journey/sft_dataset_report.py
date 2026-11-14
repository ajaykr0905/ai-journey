"""Canonical sealed manifests for public SFT dataset builds."""

from dataclasses import asdict
from hashlib import sha256

from .sft_dataset_build import SFTDatasetBuild, SFTDatasetPolicy
from .sft_dataset_identity import canonical_json_bytes
from .sft_dataset_source import SFTSourceSpec


def manifest_sha256(manifest: dict) -> str:
    """Seal over every manifest field except manifest_sha256 itself."""
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")
    unsealed = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    return sha256(canonical_json_bytes(unsealed)).hexdigest()


def build_sft_manifest(
    source: SFTSourceSpec, policy: SFTDatasetPolicy, build: SFTDatasetBuild
) -> dict:
    """Describe source, policy, pipeline counts, metrics and artifacts, then seal.

    input_records adds the quality rejections back, so every source row is
    accounted for.
    """
    if not isinstance(source, SFTSourceSpec):
        raise TypeError("source must be SFTSourceSpec")
    if not isinstance(policy, SFTDatasetPolicy):
        raise TypeError("policy must be SFTDatasetPolicy")
    if not isinstance(build, SFTDatasetBuild):
        raise TypeError("build must be SFTDatasetBuild")
    manifest = {
        "schema_version": 1,
        "source": asdict(source),
        "policy": policy.as_mapping(),
        "pipeline": {
            "input_records": build.dedup.input_records + len(build.quality.rejected),
            "quality_accepted_records": len(build.quality.accepted),
            "quality_rejected_records": len(build.quality.rejected),
            "quality_rejection_counts": build.quality.rejection_counts,
            "exact_duplicate_records": build.dedup.duplicate_records,
            "duplicate_counts": [list(value) for value in build.dedup.duplicate_counts],
            "selection_candidate_records": build.selection.candidate_records,
            "selection_quotas": dict(build.selection.quotas),
        },
        "metrics": build.metrics,
        "artifacts": build.artifacts.descriptors(),
        "attribution": (
            "Databricks Dolly 15k by Databricks contributors; licensed under "
            "Creative Commons Attribution-ShareAlike 3.0 Unported"
        ),
        "evidence_boundary": (
            "deterministic CPU construction from a pinned public snapshot; no claim "
            "of learner authorship, lecture viewing, model quality, or production use"
        ),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest
