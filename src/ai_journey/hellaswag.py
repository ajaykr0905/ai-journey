"""Bounded HellaSwag evaluation for the public GPT-2 systems lab."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


class HellaSwagError(ValueError):
    """Raised when benchmark data or evidence violates its contract."""


_EXAMPLE_FIELDS = {
    "ind",
    "activity_label",
    "ctx_a",
    "ctx_b",
    "ctx",
    "endings",
    "source_id",
    "split",
    "split_type",
    "label",
}


@dataclass(frozen=True)
class HellaSwagExample:
    """One validated four-way HellaSwag validation example."""

    index: int
    activity_label: str
    context_a: str
    context_b: str
    context: str
    endings: tuple[str, str, str, str]
    label: int
    source_id: str
    split: str
    split_type: str

    @classmethod
    def from_mapping(cls, payload: dict) -> "HellaSwagExample":
        if not isinstance(payload, dict) or set(payload) != _EXAMPLE_FIELDS:
            raise HellaSwagError("HellaSwag example fields do not match")
        index = payload["ind"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise HellaSwagError("example index must be a nonnegative integer")
        strings = {
            field: payload[field]
            for field in (
                "activity_label",
                "ctx_a",
                "ctx_b",
                "ctx",
                "source_id",
                "split",
                "split_type",
            )
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in strings.values()
        ):
            raise HellaSwagError("example text fields must be nonempty strings")
        endings = payload["endings"]
        if (
            not isinstance(endings, list)
            or len(endings) != 4
            or any(not isinstance(value, str) or not value.strip() for value in endings)
            or len(set(endings)) != 4
        ):
            raise HellaSwagError("example must contain four distinct nonempty endings")
        label = payload["label"]
        if isinstance(label, int) and not isinstance(label, bool):
            parsed_label = label
        elif isinstance(label, str) and label in {"0", "1", "2", "3"}:
            parsed_label = int(label)
        else:
            raise HellaSwagError("example label must select one of four endings")
        if parsed_label not in range(4):
            raise HellaSwagError("example label must select one of four endings")
        canonical_context = f"{strings['ctx_a'].rstrip()} {strings['ctx_b'].lstrip()}"
        if strings["ctx"].strip() != canonical_context:
            raise HellaSwagError("example context does not match ctx_a and ctx_b")
        if strings["split"] != "val" or strings["split_type"] not in {
            "indomain",
            "zeroshot",
        }:
            raise HellaSwagError("example must come from a known validation split")
        return cls(
            index=index,
            activity_label=strings["activity_label"],
            context_a=strings["ctx_a"],
            context_b=strings["ctx_b"],
            context=strings["ctx"],
            endings=tuple(endings),
            label=parsed_label,
            source_id=strings["source_id"],
            split=strings["split"],
            split_type=strings["split_type"],
        )

    @property
    def canonical_context(self) -> str:
        return f"{self.context_a.rstrip()} {self.context_b.lstrip()}"

    @property
    def digest(self) -> str:
        payload = {
            "index": self.index,
            "activity_label": self.activity_label,
            "context_a": self.context_a,
            "context_b": self.context_b,
            "context": self.context,
            "endings": self.endings,
            "label": self.label,
            "source_id": self.source_id,
            "split": self.split,
            "split_type": self.split_type,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class HellaSwagSource:
    """Pinned public validation-set identity without a runtime data dependency."""

    repository: str
    revision: str
    validation_url: str
    validation_sha256: str
    validation_bytes: int
    validation_rows: int
    license: str

    @classmethod
    def from_mapping(cls, payload: dict) -> "HellaSwagSource":
        fields = {
            "repository",
            "revision",
            "validation_url",
            "validation_sha256",
            "validation_bytes",
            "validation_rows",
            "license",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise HellaSwagError("HellaSwag source fields do not match")
        for field in ("repository", "validation_url", "license"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise HellaSwagError(f"source {field} must be a nonempty string")
        for field in ("revision", "validation_sha256"):
            value = payload[field]
            if (
                not isinstance(value, str)
                or len(value) != 40 + 24 * (field == "validation_sha256")
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise HellaSwagError(f"source {field} must be a lowercase digest")
        for field in ("validation_bytes", "validation_rows"):
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HellaSwagError(f"source {field} must be a positive integer")
        if payload["license"] != "MIT":
            raise HellaSwagError("HellaSwag source license must remain MIT")
        return cls(**payload)


def examples_from_dataset_server(source: bytes) -> tuple[HellaSwagExample, ...]:
    """Parse a bounded Hugging Face dataset-server first-rows response."""
    if not isinstance(source, bytes):
        raise TypeError("dataset response must be bytes")
    if not 0 < len(source) <= 2 * 1024 * 1024:
        raise HellaSwagError("dataset response exceeds the input budget")
    try:
        payload = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HellaSwagError("dataset response must be valid UTF-8 JSON") from error
    required = {"dataset", "config", "split", "features", "rows", "truncated"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["dataset"] != "Rowan/hellaswag"
        or payload["config"] != "default"
        or payload["split"] != "validation"
        or type(payload["truncated"]) is not bool
        or not isinstance(payload["features"], list)
        or not isinstance(payload["rows"], list)
        or not 1 <= len(payload["rows"]) <= 1000
    ):
        raise HellaSwagError("dataset response metadata does not match HellaSwag")
    examples = []
    for entry in payload["rows"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"row_idx", "row", "truncated_cells"}
            or isinstance(entry["row_idx"], bool)
            or not isinstance(entry["row_idx"], int)
            or entry["row_idx"] < 0
            or entry["truncated_cells"] != []
        ):
            raise HellaSwagError("dataset row envelope is incomplete or truncated")
        example = HellaSwagExample.from_mapping(entry["row"])
        if entry["row_idx"] != example.index:
            raise HellaSwagError("dataset row index does not match example index")
        examples.append(example)
    if len({example.index for example in examples}) != len(examples) or len(
        {example.digest for example in examples}
    ) != len(examples):
        raise HellaSwagError("dataset response contains duplicate examples")
    return tuple(examples)
