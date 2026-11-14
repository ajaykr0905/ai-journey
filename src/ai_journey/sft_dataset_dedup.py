"""Order-independent exact deduplication for normalized SFT records."""

from collections import Counter
from dataclasses import dataclass

from .sft_dataset_identity import record_id, record_id_from_sha256, record_sha256
from .sft_dataset_record import DollyRecord, SFTDatasetError


@dataclass(frozen=True)
class SFTDedupResult:
    """Unique records sorted by identity, with counts that must reconcile.

    input_records is kept plus duplicate records, and each duplicate_counts
    entry is (record_id, occurrences); construction raises SFTDatasetError
    otherwise.
    """

    records: tuple[DollyRecord, ...]
    input_records: int
    duplicate_records: int
    duplicate_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            not self.records
            or self.input_records != len(self.records) + self.duplicate_records
        ):
            raise SFTDatasetError("deduplication counts do not reconcile")
        identities = tuple(record_id(record) for record in self.records)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities
        ):
            raise SFTDatasetError(
                "deduplicated records must have unique sorted identities"
            )
        if (
            sum(count - 1 for _, count in self.duplicate_counts)
            != self.duplicate_records
        ):
            raise SFTDatasetError("duplicate evidence does not reconcile")


def deduplicate_records(records: tuple[DollyRecord, ...]) -> SFTDedupResult:
    """Drop exact duplicates by content identity, independently of input order.

    Records whose normalized fields hash alike are duplicates; one per digest is
    kept, sorted by record_id.
    """
    if (
        not isinstance(records, tuple)
        or not records
        or any(not isinstance(record, DollyRecord) for record in records)
    ):
        raise SFTDatasetError("deduplication requires a nonempty Dolly record tuple")
    by_digest = {}
    counts = Counter()
    for record in records:
        digest = record_sha256(record)
        counts[digest] += 1
        by_digest.setdefault(digest, record)
    unique = tuple(
        by_digest[digest] for digest in sorted(by_digest, key=record_id_from_sha256)
    )
    duplicate_counts = tuple(
        sorted(
            (record_id_from_sha256(digest), count)
            for digest, count in counts.items()
            if count > 1
        )
    )
    return SFTDedupResult(
        records=unique,
        input_records=len(records),
        duplicate_records=len(records) - len(unique),
        duplicate_counts=duplicate_counts,
    )
