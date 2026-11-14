import json
import os
from pathlib import Path

import pytest

from ai_journey.memory_access_pattern import MemoryAccessError
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import build_analysis_report, seal_report
from ai_journey.memory_access_report_io import (
    MAX_PLAN_BYTES,
    MAX_REPORT_BYTES,
    load_plan,
    load_report,
    read_json_object,
    write_report,
)
from test_memory_access_plan import make_plan_mapping
from test_memory_access_report import make_plan


def sealed_report():
    return seal_report(build_analysis_report(make_plan()))


def write_plan(path: Path) -> Path:
    path.write_text(json.dumps(make_plan_mapping(), indent=2), encoding="utf-8")
    return path


def test_report_round_trips_through_an_atomic_canonical_file(tmp_path: Path):
    path = tmp_path / "nested" / "analysis.json"
    report = sealed_report()

    write_report(path, report)

    assert load_report(path) == report
    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert not list(path.parent.glob(f".{path.name}.*"))
    write_report(path, report)
    assert load_report(path) == report


def test_plan_loader_accepts_formatted_plans_and_rejects_duplicate_keys(
    tmp_path: Path,
):
    plan_path = write_plan(tmp_path / "plan.json")

    assert load_plan(plan_path).to_mapping() == make_plan_mapping()
    plan_path.write_text('{"name": "a", "name": "b"}', encoding="utf-8")
    with pytest.raises(MemoryAccessError, match="duplicate JSON field: name"):
        load_plan(plan_path)
    plan_path.write_bytes(b"{" + b" " * MAX_PLAN_BYTES)
    with pytest.raises(MemoryAccessError, match="exceeds"):
        load_plan(plan_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"value": NaN}', "non-finite"),
        (b'{"value": Infinity}', "non-finite"),
        (b'{"value": 1,}', "strict UTF-8 JSON"),
        (b'{"value": "\xff"}', "strict UTF-8 JSON"),
        (b"[1, 2]", "root must be a JSON object"),
        (b'{"a": {"b": 1, "b": 2}}', "duplicate JSON field: b"),
    ],
)
def test_strict_reader_rejects_ambiguous_json(tmp_path: Path, payload, message):
    path = tmp_path / "input.json"
    path.write_bytes(payload)

    with pytest.raises(MemoryAccessError, match=message):
        read_json_object(path, max_bytes=1024, label="input")


def test_reader_refuses_symlinks_directories_and_missing_files(tmp_path: Path):
    target = write_plan(tmp_path / "plan.json")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    for path in (link, tmp_path, tmp_path / "missing.json"):
        with pytest.raises(MemoryAccessError, match="regular non-symlink"):
            read_json_object(path, max_bytes=MAX_PLAN_BYTES, label="plan")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_reader_rejects_a_fifo_without_blocking(tmp_path: Path):
    fifo = tmp_path / "plan.fifo"
    os.mkfifo(fifo)

    with pytest.raises(MemoryAccessError, match="regular non-symlink"):
        read_json_object(fifo, max_bytes=MAX_PLAN_BYTES, label="plan")


def test_loaded_reports_must_match_their_digest_and_canonical_bytes(tmp_path: Path):
    path = tmp_path / "analysis.json"
    report = sealed_report()

    tampered = json.loads(canonical_json_bytes(report))
    tampered["summary"]["patterns"] += 1
    path.write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(MemoryAccessError, match="digest"):
        load_report(path)

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(MemoryAccessError, match="not canonical"):
        load_report(path)

    path.write_bytes(b"{" + b" " * MAX_REPORT_BYTES)
    with pytest.raises(MemoryAccessError, match="exceeds"):
        load_report(path)


def test_writer_refuses_unsealed_reports_and_keeps_existing_output(tmp_path: Path):
    path = tmp_path / "analysis.json"
    path.write_bytes(b"previous\n")
    report = sealed_report()
    drifted = {**report, "measured": True}

    for invalid in (build_analysis_report(make_plan()), drifted, []):
        with pytest.raises(MemoryAccessError, match="sealed"):
            write_report(path, invalid)

    assert path.read_bytes() == b"previous\n"
    assert not list(tmp_path.glob(".analysis.json.*"))


def test_writer_never_replaces_a_protected_input_or_a_symlink(tmp_path: Path):
    plan_path = write_plan(tmp_path / "plan.json")
    before = plan_path.read_bytes()
    report = sealed_report()
    hardlink = tmp_path / "hardlink.json"
    os.link(plan_path, hardlink)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(plan_path)
    relative = tmp_path / "sub" / ".." / "plan.json"
    (tmp_path / "sub").mkdir()

    for output in (plan_path, hardlink, symlink, relative):
        with pytest.raises(MemoryAccessError, match="protected input"):
            write_report(output, report, protected=(plan_path,))
    other = tmp_path / "other.json"
    other.write_bytes(b"other\n")
    unrelated_link = tmp_path / "unrelated-link.json"
    unrelated_link.symlink_to(other)
    with pytest.raises(MemoryAccessError, match="non-symlink"):
        write_report(unrelated_link, report, protected=(plan_path,))
    with pytest.raises(MemoryAccessError, match="non-symlink"):
        write_report(tmp_path / "sub", report)

    assert plan_path.read_bytes() == before
    assert other.read_bytes() == b"other\n"
    with pytest.raises(TypeError, match="pathlib"):
        write_report(str(tmp_path / "x.json"), report)


def test_writer_refuses_symlink_loops_instead_of_crashing(tmp_path: Path):
    plan_path = write_plan(tmp_path / "plan.json")
    loop = tmp_path / "loop.json"
    loop.symlink_to(loop)
    output = tmp_path / "analysis.json"
    report = sealed_report()

    with pytest.raises(MemoryAccessError, match="non-symlink"):
        write_report(loop, report, protected=(plan_path,))
    write_report(output, report, protected=(loop,))

    assert load_report(output) == report
    assert loop.is_symlink()
