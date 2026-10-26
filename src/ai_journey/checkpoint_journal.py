"""Bounded immutable checkpoint bytes behind an atomically published manifest.

Candidates prove byte integrity only. The training loader must independently
validate every candidate's full run, tensor and optimizer contract before use.
No version or caller file is automatically deleted by this module.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile

SCHEMA = "ai-journey-checkpoint-journal-v1"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_BLOB_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_VERSIONS = 128


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"journal {name} is outside its integer budget")


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("journal requires a lowercase SHA-256 identity")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seal(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_bytes(path, budget):
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= budget:
            raise ValueError("journal input exceeds its regular-file byte budget")
        value = handle.read(budget + 1)
        after = os.fstat(handle.fileno())
    if (
        len(value) != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError("journal input changed during its bounded read")
    return value


@dataclass(frozen=True)
class CheckpointCandidate:
    step: int
    path: Path
    sha256: str


class CheckpointJournal:
    """Publish immutable versions and enumerate byte-intact same-run candidates.

    All operations take a nonblocking advisory lease on a stable lock inode.
    Budgets and run identity are durable; reopening cannot silently change them.
    A full journal rejects new publication without removing an older version.
    """

    def __init__(
        self,
        directory: Path,
        *,
        run_sha256: str,
        max_versions: int = 16,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ):
        if not isinstance(directory, Path):
            raise TypeError("journal directory must be pathlib.Path")
        _digest(run_sha256)
        _integer(max_versions, "max_versions", 1, MAX_VERSIONS)
        _integer(max_total_bytes, "max_total_bytes", 1, MAX_TOTAL_BYTES)
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError("journal directory must be a non-symlink directory")
        self.directory = directory
        self.run_sha256 = run_sha256
        self.max_versions = max_versions
        self.max_total_bytes = max_total_bytes
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._locked():
            manifest = self.directory / "manifest.json"
            if manifest.exists() or manifest.is_symlink():
                self._manifest()
            else:
                if any(
                    path.name != ".checkpoint-journal.lock"
                    for path in self.directory.iterdir()
                ):
                    raise ValueError("new journal must not adopt a nonempty directory")
                self._write_manifest([])

    @contextmanager
    def _locked(self):
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError("journal directory changed")
        descriptor = os.open(
            self.directory / ".checkpoint-journal.lock",
            os.O_RDWR | os.O_CREAT | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("journal lease must be an unaliased regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("checkpoint journal is busy") from error
            yield
        finally:
            os.close(descriptor)

    def _manifest(self):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate journal manifest field")
                result[key] = value
            return result

        value = json.loads(
            _read_bytes(self.directory / "manifest.json", MAX_MANIFEST_BYTES),
            object_pairs_hook=unique,
        )
        if type(value) is not dict or set(value) != {
            "schema",
            "run_sha256",
            "max_versions",
            "max_total_bytes",
            "entries",
            "manifest_sha256",
        }:
            raise ValueError("journal manifest fields differ")
        submitted = {
            key: part for key, part in value.items() if key != "manifest_sha256"
        }
        if (
            value["schema"] != SCHEMA
            or value["run_sha256"] != self.run_sha256
            or type(value["max_versions"]) is not int
            or value["max_versions"] != self.max_versions
            or type(value["max_total_bytes"]) is not int
            or value["max_total_bytes"] != self.max_total_bytes
            or value["manifest_sha256"] != _seal(submitted)
        ):
            raise ValueError("journal run, budget or manifest seal differs")
        entries = value["entries"]
        if type(entries) is not list or len(entries) > self.max_versions:
            raise ValueError("journal version count exceeds its budget")
        previous, total = -1, 0
        for entry in entries:
            if type(entry) is not dict or set(entry) != {
                "step",
                "filename",
                "sha256",
                "size_bytes",
            }:
                raise ValueError("journal entry fields differ")
            _integer(entry["step"], "step", 0, 1_000_000)
            _integer(entry["size_bytes"], "size_bytes", 1, MAX_BLOB_BYTES)
            _digest(entry["sha256"])
            if entry["step"] <= previous or entry["filename"] != self._filename(
                entry["step"], entry["sha256"]
            ):
                raise ValueError(
                    "journal entries must have ordered, content-bound paths"
                )
            previous = entry["step"]
            total += entry["size_bytes"]
        if total > self.max_total_bytes:
            raise ValueError("journal checkpoint bytes exceed their budget")
        return entries

    @staticmethod
    def _filename(step, digest):
        return f"step-{step:07d}-{digest}.pt"

    def _write_manifest(self, entries):
        value = {
            "schema": SCHEMA,
            "run_sha256": self.run_sha256,
            "max_versions": self.max_versions,
            "max_total_bytes": self.max_total_bytes,
            "entries": entries,
        }
        value["manifest_sha256"] = _seal(value)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(_canonical(value).encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / "manifest.json")
            temporary = None
            _sync_directory(self.directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def publish(self, checkpoint: Path, *, step: int) -> CheckpointCandidate:
        """Commit one complete immutable blob before its manifest becomes visible."""
        if not isinstance(checkpoint, Path):
            raise TypeError("journal checkpoint must be pathlib.Path")
        _integer(step, "step", 0, 1_000_000)
        with self._locked():
            entries = self._manifest()
            payload = _read_bytes(checkpoint, min(MAX_BLOB_BYTES, self.max_total_bytes))
            digest = sha256(payload).hexdigest()
            entry = {
                "step": step,
                "filename": self._filename(step, digest),
                "sha256": digest,
                "size_bytes": len(payload),
            }
            if entries and step <= entries[-1]["step"]:
                if entries[-1] == entry:
                    self._verified(entry)
                    return CheckpointCandidate(
                        step, self.directory / entry["filename"], digest
                    )
                raise ValueError("journal publication must advance the committed step")
            if (
                len(entries) >= self.max_versions
                or sum(row["size_bytes"] for row in entries) + len(payload)
                > self.max_total_bytes
            ):
                raise ValueError(
                    "journal publication exceeds its durable retention budget"
                )
            destination = self.directory / entry["filename"]
            owned = {"manifest.json", ".checkpoint-journal.lock", entry["filename"]}
            owned.update(row["filename"] for row in entries)
            if any(path.name not in owned for path in self.directory.iterdir()):
                raise ValueError(
                    "journal has uncommitted or unknown files; inspect before publishing"
                )
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.directory, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    self._verified(entry)
                _sync_directory(self.directory)
                self._write_manifest([*entries, entry])
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return CheckpointCandidate(step, destination, digest)

    def _verified(self, entry):
        payload = _read_bytes(self.directory / entry["filename"], entry["size_bytes"])
        if (
            len(payload) != entry["size_bytes"]
            or sha256(payload).hexdigest() != entry["sha256"]
        ):
            raise ValueError("journal checkpoint bytes differ from the committed entry")

    @property
    def recorded_step(self) -> int | None:
        """Manifest high-water only: this does not prove any version restorable."""
        with self._locked():
            entries = self._manifest()
            return entries[-1]["step"] if entries else None

    def candidates(self) -> tuple[CheckpointCandidate, ...]:
        """Return byte-intact versions newest first, never uncommitted orphans.

        Corrupt/missing blob entries are skipped. A corrupt or wrong-run manifest
        fails closed; no directory scan can silently select another run's bytes.
        Full semantic/optimizer validation remains the training loader's job.
        """
        with self._locked():
            valid = []
            for entry in reversed(self._manifest()):
                try:
                    self._verified(entry)
                except (OSError, ValueError):
                    continue
                valid.append(
                    CheckpointCandidate(
                        entry["step"],
                        self.directory / entry["filename"],
                        entry["sha256"],
                    )
                )
            return tuple(valid)
