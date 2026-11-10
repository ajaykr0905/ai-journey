"""Seeded, order-independent selection of an exact SFT dataset size."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256

from .sft_dataset_identity import record_id, record_sha256
from .sft_dataset_quota import category_quotas
from .sft_dataset_record import DollyRecord, SFTDatasetError


@dataclass(frozen=True)
class SFTSelectionResult:
    records: tuple[DollyRecord, ...]
    candidate_records: int
    quotas: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.records or self.candidate_records < len(self.records):
            raise SFTDatasetError("selection counts do not reconcile")
        actual = Counter(record.category for record in self.records)
        if dict(self.quotas) != dict(sorted(actual.items())):
            raise SFTDatasetError("selected category counts do not match quotas")
        identities = [record_id(record) for record in self.records]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise SFTDatasetError("selected records must have unique sorted identities")


def _selection_rank(record: DollyRecord, seed: int) -> str:
    return sha256(f"{seed}:{record_sha256(record)}".encode()).hexdigest()


def select_records(
    records: tuple[DollyRecord, ...], *, target: int, seed: int
) -> SFTSelectionResult:
    if (
        not isinstance(records, tuple)
        or not records
        or any(not isinstance(record, DollyRecord) for record in records)
    ):
        raise SFTDatasetError("selection requires a nonempty Dolly record tuple")
    identities = [record_sha256(record) for record in records]
    if len(identities) != len(set(identities)):
        raise SFTDatasetError(
            "selection candidates must already be exactly deduplicated"
        )
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise SFTDatasetError("selection seed must be an unsigned 32-bit integer")
    counts = Counter(record.category for record in records)
    quotas = category_quotas(dict(counts), target)
    selected = []
    for category, quota in quotas.items():
        candidates = sorted(
            (record for record in records if record.category == category),
            key=lambda record: (_selection_rank(record, seed), record_id(record)),
        )
        selected.extend(candidates[:quota])
    selected.sort(key=record_id)
    return SFTSelectionResult(
        records=tuple(selected),
        candidate_records=len(records),
        quotas=tuple(quotas.items()),
    )
