import os
import stat

import pytest

from ai_journey.telemetry_traces_io import (
    TelemetryError,
    TelemetryVerificationError,
    artifact_bytes,
    canonical_json_bytes,
    canonical_sha256,
    create_output_directory,
    load_canonical_artifact,
    publish_new_file,
    read_bounded_file,
    sha256_hex,
    strict_json_loads,
)


def test_canonical_bytes_are_sorted_compact_ascii_and_digest_stable():
    value = {"b": [1, 2.5, None], "a": "é", "c": True}

    assert canonical_json_bytes(value) == b'{"a":"\\u00e9","b":[1,2.5,null],"c":true}'
    assert artifact_bytes(value) == canonical_json_bytes(value) + b"\n"
    assert canonical_sha256(value) == sha256_hex(canonical_json_bytes(value))
    assert issubclass(TelemetryVerificationError, TelemetryError)
    with pytest.raises(TypeError, match="bytes"):
        sha256_hex("text")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"x": {1, 2}}])
def test_values_without_a_canonical_json_form_are_rejected(value):
    with pytest.raises(TelemetryError, match="canonical JSON"):
        canonical_json_bytes(value)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b'{"a":1,"a":2}', "duplicate JSON key: a"),
        (b'{"a":NaN}', "non-finite JSON constant"),
        (b"[Infinity]", "non-finite JSON constant"),
        (b"[1e999]", "overflows"),
        (b"[" + b"9" * 5000 + b"]", "too long"),
        (b"\xff", "strict UTF-8"),
        (b"[1,", "strict UTF-8"),
        (b"[" * 100_000, "strict UTF-8"),
    ],
)
def test_strict_loads_rejects_ambiguous_or_unparseable_json(data, message):
    with pytest.raises(TelemetryError, match=message):
        strict_json_loads(data, label="sample")


def test_strict_loads_requires_bytes():
    assert strict_json_loads(b'{"a":[1,2.5]}', label="sample") == {"a": [1, 2.5]}
    with pytest.raises(TypeError, match="sample must be bytes"):
        strict_json_loads('{"a":1}', label="sample")


def test_bounded_reads_accept_only_regular_files_within_the_bound(tmp_path):
    exact = tmp_path / "exact.json"
    exact.write_bytes(b"x" * 8)
    assert read_bounded_file(exact, max_bytes=8, label="exact") == b"x" * 8
    with pytest.raises(TelemetryError, match="7-byte bound"):
        read_bounded_file(exact, max_bytes=7, label="exact")

    link = tmp_path / "link.json"
    link.symlink_to(exact)
    with pytest.raises(TelemetryError, match="non-symlink"):
        read_bounded_file(link, max_bytes=8, label="link")
    with pytest.raises(TelemetryError, match="regular file"):
        read_bounded_file(tmp_path, max_bytes=8, label="directory")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(TelemetryError, match="regular file"):
        read_bounded_file(fifo, max_bytes=8, label="fifo")
    with pytest.raises(TelemetryError, match="readable"):
        read_bounded_file(tmp_path / "missing.json", max_bytes=8, label="missing")
    with pytest.raises(TelemetryError, match="max_bytes"):
        read_bounded_file(exact, max_bytes=0, label="exact")
    with pytest.raises(TypeError, match="pathlib"):
        read_bounded_file(str(exact), max_bytes=8, label="exact")


@pytest.mark.parametrize(
    "data",
    [b'{"a":1}', b'{"a": 1}\n', b'{"b":1,"a":2}\n', b'{"a":1}\n\n', b'{"a":1e0}\n'],
)
def test_artifacts_must_be_exactly_canonical_with_one_newline(tmp_path, data):
    path = tmp_path / "artifact.json"
    path.write_bytes(data)
    with pytest.raises(TelemetryError, match="canonical"):
        load_canonical_artifact(path, max_bytes=64, label="artifact")
    path.write_bytes(b'{"a":1.0}\n')
    assert load_canonical_artifact(path, max_bytes=64, label="artifact") == {"a": 1.0}


def test_publication_is_complete_private_and_never_overwrites(tmp_path):
    target = tmp_path / "artifact.json"
    publish_new_file(target, b"first\n", max_bytes=16)

    assert target.read_bytes() == b"first\n"
    assert stat.S_IMODE(target.stat().st_mode) & 0o077 == 0
    with pytest.raises(TelemetryError, match="refusing to overwrite"):
        publish_new_file(target, b"second\n", max_bytes=16)
    assert target.read_bytes() == b"first\n"
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "nowhere")
    with pytest.raises(TelemetryError, match="refusing to overwrite"):
        publish_new_file(dangling, b"x", max_bytes=16)
    with pytest.raises(TelemetryError, match="16-byte bound"):
        publish_new_file(tmp_path / "large.json", b"x" * 17, max_bytes=16)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "artifact.json",
        "dangling.json",
    ]


def test_publication_refuses_a_symlinked_or_missing_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(TelemetryError, match="real directory"):
        publish_new_file(alias / "artifact.json", b"x", max_bytes=8)
    with pytest.raises(TelemetryError, match="real directory"):
        publish_new_file(tmp_path / "missing" / "artifact.json", b"x", max_bytes=8)
    assert list(real.iterdir()) == []
    with pytest.raises(TypeError, match="bytes"):
        publish_new_file(real / "artifact.json", "x", max_bytes=8)


def test_output_directories_are_new_and_outside_every_source(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    created = create_output_directory(tmp_path / "a" / "b", sources=(source,))
    assert created.is_dir()

    with pytest.raises(TelemetryError, match="already exist"):
        create_output_directory(created, sources=(source,))
    for inside in (source / "telemetry", source / "x" / ".." / "y"):
        with pytest.raises(TelemetryError, match="inside a source"):
            create_output_directory(inside, sources=(source,))
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(TelemetryError, match="inside a source"):
        create_output_directory(alias / "telemetry", sources=(source,))
    assert list(source.iterdir()) == []
    with pytest.raises(TypeError, match="source directories"):
        create_output_directory(tmp_path / "c", sources=(str(source),))
