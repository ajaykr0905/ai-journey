"""Canonical identities for normalized SFT records and artifact values."""

from dataclasses import asdict
from hashlib import sha256
import json

from .sft_dataset_record import DollyRecord, SFTDatasetError


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise SFTDatasetError("value cannot be encoded as canonical JSON") from exc


def record_sha256(record: DollyRecord) -> str:
    if not isinstance(record, DollyRecord):
        raise TypeError("record must be DollyRecord")
    return sha256(canonical_json_bytes(asdict(record))).hexdigest()


def record_id(record: DollyRecord) -> str:
    """Return a stable public identifier without exposing source row positions."""
    return f"dolly-{record_sha256(record)[:24]}"
