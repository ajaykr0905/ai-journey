"""Immutable public-source identity and bounded Dolly snapshot loading."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from .sft_dataset_record import DollyRecord, SFTDatasetError


_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SFTDatasetError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


@dataclass(frozen=True)
class SFTSourceSpec:
    """Pinned Dolly snapshot identity: revision, file, size, SHA-256 and license.

    Only the CC-BY-SA-3.0 Databricks Dolly 15k file at a full commit revision is
    accepted, so evidence cannot quietly switch sources.
    """

    dataset_id: str
    revision: str
    filename: str
    sha256: str
    bytes: int
    license: str
    license_url: str
    source_url: str

    def __post_init__(self) -> None:
        if self.dataset_id != "databricks/databricks-dolly-15k":
            raise SFTDatasetError("source dataset_id must name Databricks Dolly 15k")
        if not isinstance(self.revision, str) or not _REVISION.fullmatch(self.revision):
            raise SFTDatasetError("source revision must be a full commit digest")
        if self.filename != "databricks-dolly-15k.jsonl":
            raise SFTDatasetError("source filename is not the pinned Dolly snapshot")
        if not isinstance(self.sha256, str) or not _DIGEST.fullmatch(self.sha256):
            raise SFTDatasetError("source sha256 must be lowercase hexadecimal")
        if type(self.bytes) is not int or not 1 <= self.bytes <= 32 * 1024 * 1024:
            raise SFTDatasetError(
                "source byte count is outside the public snapshot budget"
            )
        if self.license != "CC-BY-SA-3.0":
            raise SFTDatasetError(
                "source license must preserve Dolly attribution terms"
            )
        for label, value, prefix in (
            ("license_url", self.license_url, "https://"),
            ("source_url", self.source_url, "https://huggingface.co/"),
        ):
            if not isinstance(value, str) or not value.startswith(prefix):
                raise SFTDatasetError(f"{label} must be a public HTTPS URL")
        if self.revision not in self.source_url or self.filename not in self.source_url:
            raise SFTDatasetError(
                "source_url must bind the pinned revision and filename"
            )

    @classmethod
    def from_mapping(cls, value: object) -> "SFTSourceSpec":
        """Build the spec from config or manifest JSON with exactly its eight fields."""
        fields = {
            "dataset_id",
            "revision",
            "filename",
            "sha256",
            "bytes",
            "license",
            "license_url",
            "source_url",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise SFTDatasetError("source policy fields are invalid")
        return cls(**value)


def load_source_spec(path: Path) -> SFTSourceSpec:
    """Load a spec from a regular, non-symlink JSON file of at most 16 KiB.

    Duplicate fields and invalid UTF-8 JSON raise SFTDatasetError.
    """
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise SFTDatasetError("source policy must be a regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SFTDatasetError("unable to read source policy") from exc
    if not 1 <= len(payload) <= 16 * 1024:
        raise SFTDatasetError("source policy exceeds its byte budget")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SFTDatasetError("source policy is not strict UTF-8 JSON") from exc
    return SFTSourceSpec.from_mapping(value)


def load_dolly_snapshot(path: Path, spec: SFTSourceSpec) -> tuple[DollyRecord, ...]:
    """Load every record after verifying exact public snapshot bytes."""
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise SFTDatasetError("Dolly snapshot must be a regular file")
    if not isinstance(spec, SFTSourceSpec):
        raise TypeError("spec must be SFTSourceSpec")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SFTDatasetError("unable to read Dolly snapshot") from exc
    if len(payload) != spec.bytes or sha256(payload).hexdigest() != spec.sha256:
        raise SFTDatasetError(
            "Dolly snapshot identity does not match the source policy"
        )
    if b"\x00" in payload:
        raise SFTDatasetError("Dolly snapshot must be NUL-free UTF-8 JSONL")
    records = []
    try:
        for line in payload.splitlines():
            value = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object)
            records.append(DollyRecord.from_mapping(value))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SFTDatasetError("Dolly snapshot contains invalid JSONL") from exc
    if not 500 <= len(records) <= 20_000:
        raise SFTDatasetError(
            "Dolly snapshot record count is outside the dataset budget"
        )
    return tuple(records)
