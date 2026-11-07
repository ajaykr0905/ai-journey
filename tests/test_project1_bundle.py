import os

import pytest

from ai_journey.project1_bundle import (
    build_artifact_manifest,
    verify_artifact_manifest,
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
