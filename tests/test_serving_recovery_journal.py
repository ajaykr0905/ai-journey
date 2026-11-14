import json
import stat
import threading

import pytest

from ai_journey.serving_recovery_identity import idempotency_key, server_lifetime_id
from ai_journey.serving_recovery_journal import (
    ExecutionJournal,
    read_execution_journal,
)

RUN = "serving-recovery-test-0123456789abcdef"
LIFETIME = server_lifetime_id(RUN, 1)
MODEL = "c" * 64


def journal(path, max_records=64):
    return ExecutionJournal(
        path, lifetime_id=LIFETIME, model_sha256=MODEL, max_records=max_records
    )


def append(writer, index, attempt=1):
    return writer.append(
        idempotency_key=idempotency_key(RUN, index),
        attempt=attempt,
        request_sha256="a" * 64,
        response_sha256="b" * 64,
    )


def read(path, *, allow_torn_tail=False, lifetime=LIFETIME):
    return read_execution_journal(
        path, lifetime_id=lifetime, max_records=64, allow_torn_tail=allow_torn_tail
    )


def test_journal_commits_one_canonical_line_per_execution(tmp_path):
    path = tmp_path / "journal.jsonl"
    with journal(path) as writer:
        first = append(writer, 0)
        second = append(writer, 0, attempt=2)
        assert writer.records_written == 2

    contents = read(path)
    assert contents.records == (first, second)
    assert contents.torn_tail_bytes == 0
    assert [record.sequence for record in contents.records] == [0, 1]
    lines = path.read_bytes().splitlines()
    assert json.loads(lines[1])["attempt"] == 2
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_concurrent_appends_remain_whole_and_contiguous(tmp_path):
    path = tmp_path / "journal.jsonl"
    with journal(path, max_records=64) as writer:
        threads = [
            threading.Thread(
                target=lambda base=base: [
                    append(writer, base * 8 + i) for i in range(8)
                ]
            )
            for base in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    contents = read(path)
    assert len(contents.records) == 64
    assert len({record.idempotency_key for record in contents.records}) == 64


def test_writer_refuses_existing_paths_full_journals_and_closed_handles(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="already exists"):
        journal(path)

    bounded = journal(tmp_path / "bounded.jsonl", max_records=1)
    append(bounded, 0)
    with pytest.raises(ValueError, match="full"):
        append(bounded, 1)
    bounded.close()
    with pytest.raises(ValueError, match="closed"):
        append(bounded, 1)


def test_reader_tolerates_exactly_one_torn_final_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    with journal(path) as writer:
        records = (append(writer, 0), append(writer, 1))
    path.write_bytes(path.read_bytes() + b'{"attempt":1,"idemp')

    contents = read(path, allow_torn_tail=True)
    assert contents.records == records
    assert contents.torn_tail_bytes == len(b'{"attempt":1,"idemp')
    with pytest.raises(ValueError, match="unterminated"):
        read(path)

    path.write_bytes(path.read_bytes() + b"\n" + path.read_bytes().splitlines()[0])
    with pytest.raises(ValueError, match="strict"):
        read(path, allow_torn_tail=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda lines: [b""] + lines, "empty"),
        (lambda lines: [lines[0].replace(b":", b": ", 1)] + lines[1:], "canonical"),
        (lambda lines: [lines[1], lines[0]], "contiguous"),
        (
            lambda lines: [
                lines[0],
                lines[0].replace(b'"sequence":0', b'"sequence":1'),
            ],
            "repeats",
        ),
        (
            lambda lines: [lines[0].replace(b'"attempt":1', b'"attempt":true')],
            "integer",
        ),
        (lambda lines: [lines[0][:-1] + b',"x":1}'], "fields"),
    ],
)
def test_reader_rejects_corrupt_committed_records(tmp_path, mutate, message):
    path = tmp_path / "journal.jsonl"
    with journal(path) as writer:
        append(writer, 0)
        append(writer, 1)
    lines = path.read_bytes().splitlines()
    path.write_bytes(b"".join(line + b"\n" for line in mutate(lines)))

    with pytest.raises((TypeError, ValueError), match=message):
        read(path, allow_torn_tail=True)


def test_reader_binds_records_to_their_lifetime_and_model(tmp_path):
    path = tmp_path / "journal.jsonl"
    with journal(path) as writer:
        append(writer, 0)
    with pytest.raises(ValueError, match="another lifetime"):
        read(path, lifetime=server_lifetime_id(RUN, 2))

    other = tmp_path / "other.jsonl"
    with ExecutionJournal(
        other, lifetime_id=LIFETIME, model_sha256="d" * 64, max_records=4
    ) as writer:
        append(writer, 1)
    other.write_bytes(
        path.read_bytes() + other.read_bytes().replace(b'"sequence":0', b'"sequence":1')
    )
    with pytest.raises(ValueError, match="model identity"):
        read(other)
