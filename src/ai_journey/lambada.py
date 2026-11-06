"""Bounded LAMBADA last-word evaluation for the public transformer lab."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


class LambadaError(ValueError):
    """Raised when LAMBADA inputs or evidence violate their contract."""


@dataclass(frozen=True)
class LambadaSource:
    """Pinned identity for the public OpenAI-processed LAMBADA test set."""

    repository: str
    revision: str
    dataset_server_url: str
    sample_sha256: str
    sample_bytes: int
    test_rows: int
    source_file_sha256: str
    license: str

    @classmethod
    def from_mapping(cls, payload: dict) -> "LambadaSource":
        fields = {
            "repository",
            "revision",
            "dataset_server_url",
            "sample_sha256",
            "sample_bytes",
            "test_rows",
            "source_file_sha256",
            "license",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise LambadaError("LAMBADA source fields do not match")
        for field in ("repository", "dataset_server_url", "license"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise LambadaError(f"source {field} must be nonempty text")
        if payload["repository"] != "EleutherAI/lambada_openai":
            raise LambadaError("source repository must identify lambada_openai")
        if payload["license"] != "MIT":
            raise LambadaError("source license must remain MIT")
        for field, length in (
            ("revision", 40),
            ("sample_sha256", 64),
            ("source_file_sha256", 64),
        ):
            value = payload[field]
            if (
                not isinstance(value, str)
                or len(value) != length
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise LambadaError(f"source {field} must be a lowercase digest")
        for field in ("sample_bytes", "test_rows"):
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LambadaError(f"source {field} must be a positive integer")
        return cls(**payload)

    def verify_sample(self, source: bytes) -> None:
        if not isinstance(source, bytes):
            raise TypeError("sample response must be bytes")
        if (
            len(source) != self.sample_bytes
            or sha256(source).hexdigest() != self.sample_sha256
        ):
            raise LambadaError("sample response does not match the pinned snapshot")


@dataclass(frozen=True)
class LambadaExample:
    """One bounded narrative and its exact final-word target."""

    row_index: int
    text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise LambadaError("example row index must be nonnegative")
        if not isinstance(self.text, str) or not self.text.strip():
            raise LambadaError("example text must be nonempty")
        encoded = self.text.encode("utf-8")
        if len(encoded) > 32 * 1024 or "\x00" in self.text:
            raise LambadaError("example text exceeds the input contract")

    @property
    def digest(self) -> str:
        payload = {"row_index": self.row_index, "text": self.text}
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
