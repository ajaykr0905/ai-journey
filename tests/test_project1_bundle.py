import os
from copy import deepcopy

import pytest

from ai_journey.project1_bundle import (
    build_artifact_manifest,
    seal_bundle_index,
    verify_artifact_manifest,
    verify_bundle_index,
)


def test_artifact_manifest_hashes_sorted_bounded_regular_files(tmp_path):
    (tmp_path / "report.json").write_bytes(b"report")
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")

    manifest = build_artifact_manifest(tmp_path, ["report.json", "checkpoint.pt"])

    assert list(manifest) == ["checkpoint.pt", "report.json"]
    assert manifest["report.json"]["bytes"] == 6
    verify_artifact_manifest(tmp_path, manifest)


def test_artifact_manifest_rejects_paths_aliases_duplicates_and_oversize(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(target)
    hardlink = tmp_path / "hardlink"
    os.link(target, hardlink)

    with pytest.raises(ValueError, match="normalized"):
        build_artifact_manifest(tmp_path, ["../target"])
    with pytest.raises(ValueError, match="unique"):
        build_artifact_manifest(tmp_path, ["target", "target"])
    with pytest.raises(ValueError, match="non-symlink"):
        build_artifact_manifest(tmp_path, ["link"])
    with pytest.raises(ValueError, match="max_file_bytes"):
        build_artifact_manifest(tmp_path, ["target"], max_file_bytes=2)
    with pytest.raises(ValueError, match="hardlink aliases"):
        build_artifact_manifest(tmp_path, ["target", "hardlink"])


def test_artifact_verification_detects_changed_bytes(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"before")
    manifest = build_artifact_manifest(tmp_path, ["report.json"])
    path.write_bytes(b"after")

    with pytest.raises(ValueError, match="does not match"):
        verify_artifact_manifest(tmp_path, manifest)


def test_bundle_index_seals_contract_preflight_journal_and_artifacts():
    artifacts = {"report.json": {"bytes": 6, "sha256": "e" * 64}}
    bundle = seal_bundle_index(
        run_id="day-64-project1-0123456789abcdef",
        contract_sha256="a" * 64,
        preflight_sha256="b" * 64,
        journal_sha256="c" * 64,
        artifacts=artifacts,
    )

    assert verify_bundle_index(bundle) == bundle
    assert "does not prove W&B" in bundle["evidence_boundary"]


def test_bundle_index_rejects_tampering_even_when_artifact_shape_is_plausible():
    bundle = seal_bundle_index(
        run_id="day-64-project1-0123456789abcdef",
        contract_sha256="a" * 64,
        preflight_sha256="b" * 64,
        journal_sha256="c" * 64,
        artifacts={"report.json": {"bytes": 6, "sha256": "e" * 64}},
    )
    changed = deepcopy(bundle)
    changed["artifacts"]["report.json"]["bytes"] = 7

    with pytest.raises(ValueError, match="bundle digest"):
        verify_bundle_index(changed)
