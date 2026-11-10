"""Explicit, auditable quality admission for SFT source records."""

from collections import Counter
from dataclasses import dataclass

from .sft_dataset_record import DollyRecord, SFTDatasetError


@dataclass(frozen=True)
class SFTQualityPolicy:
    min_response_bytes: int = 1
    max_prompt_bytes: int = 16 * 1024
    max_response_bytes: int = 16 * 1024
    reject_identity_answers: bool = True

    def __post_init__(self) -> None:
        for name, low, high in (
            ("min_response_bytes", 1, 1024),
            ("max_prompt_bytes", 128, 64 * 1024),
            ("max_response_bytes", 128, 64 * 1024),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise SFTDatasetError(f"{name} must be an integer in [{low}, {high}]")
        if self.min_response_bytes > self.max_response_bytes:
            raise SFTDatasetError("response byte bounds are inverted")
        if type(self.reject_identity_answers) is not bool:
            raise SFTDatasetError("reject_identity_answers must be boolean")


@dataclass(frozen=True)
class SFTQualityResult:
    accepted: tuple[DollyRecord, ...]
    rejected: tuple[tuple[int, str], ...]

    @property
    def rejection_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(reason for _, reason in self.rejected).items()))


def quality_rejection(record: DollyRecord, policy: SFTQualityPolicy) -> str | None:
    if not isinstance(record, DollyRecord) or not isinstance(policy, SFTQualityPolicy):
        raise TypeError("quality admission requires a DollyRecord and SFTQualityPolicy")
    prompt_bytes = len(record.instruction.encode("utf-8")) + len(
        record.context.encode("utf-8")
    )
    response_bytes = len(record.response.encode("utf-8"))
    if prompt_bytes > policy.max_prompt_bytes:
        return "prompt_too_large"
    if (
        policy.reject_identity_answers
        and not record.context
        and record.instruction.casefold() == record.response.casefold()
    ):
        return "identity_answer"
    if response_bytes < policy.min_response_bytes:
        return "response_too_small"
    if response_bytes > policy.max_response_bytes:
        return "response_too_large"
    return None


def filter_quality(
    records: tuple[DollyRecord, ...], policy: SFTQualityPolicy
) -> SFTQualityResult:
    if (
        not isinstance(records, tuple)
        or not records
        or any(not isinstance(record, DollyRecord) for record in records)
    ):
        raise SFTDatasetError("quality input must be a nonempty Dolly record tuple")
    if not isinstance(policy, SFTQualityPolicy):
        raise TypeError("policy must be SFTQualityPolicy")
    accepted = []
    rejected = []
    for index, record in enumerate(records):
        reason = quality_rejection(record, policy)
        if reason is None:
            accepted.append(record)
        else:
            rejected.append((index, reason))
    if not accepted:
        raise SFTDatasetError("quality policy rejected every source record")
    return SFTQualityResult(tuple(accepted), tuple(rejected))
