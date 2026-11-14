"""Canonical identities for normalized SFT records and artifact values."""

from dataclasses import asdict
from hashlib import sha256
import json

from .sft_dataset_record import DollyRecord, SFTDatasetError


def canonical_json_bytes(value: object) -> bytes:
    """The one JSON encoding behind SFT identities, artifacts and the manifest seal.

    Values JSON cannot represent exactly, such as NaN, raise SFTDatasetError.
    """
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
    """Content identity: SHA-256 of a normalized record's canonical JSON fields.

    Equal content gives equal digests whatever its source position.
    """
    if not isinstance(record, DollyRecord):
        raise TypeError("record must be DollyRecord")
    return sha256(canonical_json_bytes(asdict(record))).hexdigest()


def record_id(record: DollyRecord) -> str:
    """Return a stable public identifier without exposing source row positions."""
    return record_id_from_sha256(record_sha256(record))


def record_id_from_sha256(digest: str) -> str:
    """Derive record_id from a record_sha256 digest the caller already holds.

    Ranking and sorting thousands of records would otherwise rehash every
    record for each key it needs.
    """
    return f"dolly-{digest[:24]}"
