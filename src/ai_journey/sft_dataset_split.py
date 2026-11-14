"""Deterministic category-stratified SFT train and validation partitions."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256

from .sft_dataset_identity import record_id, record_sha256
from .sft_dataset_quota import category_quotas
from .sft_dataset_record import DollyRecord, SFTDatasetError


@dataclass(frozen=True)
class SFTDatasetSplit:
    """Disjoint train and validation records, each sorted by identity.

    An empty partition, unsorted ids or a shared content identity raise
    SFTDatasetError.
    """

    train: tuple[DollyRecord, ...]
    validation: tuple[DollyRecord, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.validation:
            raise SFTDatasetError(
                "SFT train and validation partitions must be nonempty"
            )
        train_ids = tuple(record_id(record) for record in self.train)
        validation_ids = tuple(record_id(record) for record in self.validation)
        if train_ids != tuple(sorted(train_ids)) or validation_ids != tuple(
            sorted(validation_ids)
        ):
            raise SFTDatasetError("SFT partitions must be sorted by stable identity")
        if set(train_ids).intersection(validation_ids):
            raise SFTDatasetError("SFT partitions must not share content identities")


def _split_rank(record: DollyRecord, seed: int) -> str:
    return sha256(f"split:{seed}:{record_sha256(record)}".encode()).hexdigest()


def split_records(
    records: tuple[DollyRecord, ...], *, validation_count: int, seed: int
) -> SFTDatasetSplit:
    """Hold out validation_count records by category quota and seeded rank.

    Every category keeps a validation record; ranks hash in a "split:" domain,
    so they stay independent of selection ranks even if the seeds match.
    """
    if (
        not isinstance(records, tuple)
        or not records
        or any(not isinstance(record, DollyRecord) for record in records)
    ):
        raise SFTDatasetError("splitting requires a nonempty Dolly record tuple")
    identities = [record_sha256(record) for record in records]
    if len(identities) != len(set(identities)):
        raise SFTDatasetError("split input contains duplicate content identities")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise SFTDatasetError("split seed must be an unsigned 32-bit integer")
    counts = Counter(record.category for record in records)
    if type(validation_count) is not int or not len(counts) <= validation_count < len(
        records
    ):
        raise SFTDatasetError(
            "validation count must retain every category and training data"
        )
    quotas = category_quotas(dict(counts), validation_count)
    validation = []
    for category, quota in quotas.items():
        candidates = sorted(
            (record for record in records if record.category == category),
            key=lambda record: (_split_rank(record, seed), record_id(record)),
        )
        validation.extend(candidates[:quota])
    validation_ids = {record_sha256(record) for record in validation}
    train = [
        record for record in records if record_sha256(record) not in validation_ids
    ]
    return SFTDatasetSplit(
        train=tuple(sorted(train, key=record_id)),
        validation=tuple(sorted(validation, key=record_id)),
    )
