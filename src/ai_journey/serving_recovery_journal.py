"""Append-only per-lifetime execution journal for the serving recovery lab.

The server appends one canonical JSON line per executed request before it sends
the response, so every acknowledged response has a committed record. The
trailing newline commits a record. After a SIGKILL the reader tolerates exactly
one torn final line (bytes after the last newline) and reports its length; any
other malformed content is rejected. Records use unbuffered ``os.write`` calls,
so they survive a process kill; no per-record fsync is issued, so host power
loss is outside this journal's durability claim.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .serving_recovery_config import (
    ServingRecoveryError,
    canonical_json_bytes,
    read_bounded_file,
    require_digest,
    require_int,
    strict_json_loads,
)
from .serving_recovery_identity import require_idempotency_key, require_lifetime_id

MAX_RECORD_BYTES = 1024
MAX_JOURNAL_RECORDS = 16_640
MAX_ATTEMPT = 65


@dataclass(frozen=True)
class ExecutionRecord:
    """One request execution committed by one server lifetime."""

    lifetime_id: str
    sequence: int
    idempotency_key: str
    attempt: int
    request_sha256: str
    response_sha256: str
    model_sha256: str

    def __post_init__(self) -> None:
        require_lifetime_id(self.lifetime_id)
        require_int(self.sequence, "sequence", 0, MAX_JOURNAL_RECORDS - 1)
        require_idempotency_key(self.idempotency_key)
        require_int(self.attempt, "attempt", 1, MAX_ATTEMPT)
        for name in ("request_sha256", "response_sha256", "model_sha256"):
            require_digest(getattr(self, name), name)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }


_RECORD_FIELDS = frozenset(
    {"schema_version"} | {f.name for f in fields(ExecutionRecord)}
)


def record_from_mapping(payload: object) -> ExecutionRecord:
    """Parse one exact journal record mapping."""

    if type(payload) is not dict or set(payload) != _RECORD_FIELDS:
        raise ServingRecoveryError("journal record fields do not match schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ServingRecoveryError("journal record schema is unsupported")
    return ExecutionRecord(
        **{key: value for key, value in payload.items() if key != "schema_version"}
    )


class ExecutionJournal:
    """Thread-safe exclusive writer for one server lifetime's executions."""

    def __init__(
        self,
        path: Path,
        *,
        lifetime_id: str,
        model_sha256: str,
        max_records: int,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("journal path must be pathlib.Path")
        self.lifetime_id = require_lifetime_id(lifetime_id)
        self.model_sha256 = require_digest(model_sha256, "model_sha256")
        self.max_records = require_int(
            max_records, "max_records", 1, MAX_JOURNAL_RECORDS
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            self._descriptor: int | None = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise ServingRecoveryError("journal path already exists") from exc
        self._lock = threading.Lock()
        self._count = 0

    @property
    def records_written(self) -> int:
        with self._lock:
            return self._count

    def append(
        self,
        *,
        idempotency_key: str,
        attempt: int,
        request_sha256: str,
        response_sha256: str,
    ) -> ExecutionRecord:
        """Commit one execution with a single newline-terminated append."""

        with self._lock:
            if self._descriptor is None:
                raise ServingRecoveryError("journal is closed")
            if self._count >= self.max_records:
                raise ServingRecoveryError("journal is full")
            record = ExecutionRecord(
                lifetime_id=self.lifetime_id,
                sequence=self._count,
                idempotency_key=idempotency_key,
                attempt=attempt,
                request_sha256=request_sha256,
                response_sha256=response_sha256,
                model_sha256=self.model_sha256,
            )
            line = memoryview(canonical_json_bytes(record.to_mapping()) + b"\n")
            while line:
                line = line[os.write(self._descriptor, line) :]
            self._count += 1
            return record

    def close(self) -> None:
        with self._lock:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None

    def __enter__(self) -> ExecutionJournal:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class JournalContents:
    """Committed records of one lifetime plus the length of any torn tail."""

    lifetime_id: str
    records: tuple[ExecutionRecord, ...]
    torn_tail_bytes: int


def read_execution_journal(
    path: Path,
    *,
    lifetime_id: str,
    max_records: int,
    allow_torn_tail: bool,
) -> JournalContents:
    """Read committed records; tolerate one torn final line only when allowed."""

    require_lifetime_id(lifetime_id)
    require_int(max_records, "max_records", 1, MAX_JOURNAL_RECORDS)
    if type(allow_torn_tail) is not bool:
        raise TypeError("allow_torn_tail must be a boolean")
    data = read_bounded_file(
        path, max_bytes=(max_records + 1) * MAX_RECORD_BYTES, name="journal"
    )
    *lines, tail = data.split(b"\n")
    if len(lines) > max_records:
        raise ServingRecoveryError("journal exceeds its record bound")
    records = []
    executed = set()
    for sequence, line in enumerate(lines):
        if not line or len(line) > MAX_RECORD_BYTES:
            raise ServingRecoveryError(f"journal line {sequence} is empty or oversized")
        record = record_from_mapping(
            strict_json_loads(line, name=f"journal line {sequence}")
        )
        if canonical_json_bytes(record.to_mapping()) != line:
            raise ServingRecoveryError(f"journal line {sequence} is not canonical")
        if record.lifetime_id != lifetime_id:
            raise ServingRecoveryError("journal record belongs to another lifetime")
        if record.sequence != sequence:
            raise ServingRecoveryError("journal sequence is not contiguous")
        if records and record.model_sha256 != records[0].model_sha256:
            raise ServingRecoveryError("journal model identity changed")
        if (record.idempotency_key, record.attempt) in executed:
            raise ServingRecoveryError("journal repeats one client attempt")
        executed.add((record.idempotency_key, record.attempt))
        records.append(record)
    if tail:
        if not allow_torn_tail:
            raise ServingRecoveryError("journal ends with an unterminated record")
        if len(tail) > MAX_RECORD_BYTES:
            raise ServingRecoveryError("journal tail is longer than one record")
    return JournalContents(lifetime_id, tuple(records), len(tail))
