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


def examples_from_dataset_server(source: bytes) -> tuple[LambadaExample, ...]:
    """Parse a bounded Hugging Face dataset-server first-rows response."""
    if not isinstance(source, bytes):
        raise TypeError("dataset response must be bytes")
    if not 0 < len(source) <= 2 * 1024 * 1024:
        raise LambadaError("dataset response exceeds the input budget")
    try:
        payload = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LambadaError("dataset response is not valid UTF-8 JSON") from error
    required = {"dataset", "config", "split", "features", "rows", "truncated"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["dataset"] != "EleutherAI/lambada_openai"
        or payload["config"] != "default"
        or payload["split"] != "test"
        or type(payload["truncated"]) is not bool
        or not isinstance(payload["features"], list)
        or not isinstance(payload["rows"], list)
        or not 1 <= len(payload["rows"]) <= 1000
    ):
        raise LambadaError("dataset response metadata does not match LAMBADA")
    examples = []
    for entry in payload["rows"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"row_idx", "row", "truncated_cells"}
            or isinstance(entry["row_idx"], bool)
            or not isinstance(entry["row_idx"], int)
            or entry["row_idx"] < 0
            or entry["truncated_cells"] != []
            or not isinstance(entry["row"], dict)
            or set(entry["row"]) != {"text"}
        ):
            raise LambadaError("dataset row envelope is incomplete or truncated")
        examples.append(LambadaExample(entry["row_idx"], entry["row"]["text"]))
    if len({example.row_index for example in examples}) != len(examples) or len(
        {example.digest for example in examples}
    ) != len(examples):
        raise LambadaError("dataset response contains duplicate examples")
    return tuple(examples)
