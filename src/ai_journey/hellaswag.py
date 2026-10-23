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
