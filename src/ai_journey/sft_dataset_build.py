"""Compose quality, deduplication, selection, split, and export into one build."""

from dataclasses import asdict, dataclass

from .sft_dataset_dedup import SFTDedupResult, deduplicate_records
from .sft_dataset_export import SFTDatasetArtifacts, export_split
from .sft_dataset_metrics import dataset_metrics, validate_dataset_metrics
from .sft_dataset_quality import SFTQualityPolicy, SFTQualityResult, filter_quality
from .sft_dataset_record import DollyRecord, SFTDatasetError
from .sft_dataset_select import SFTSelectionResult, select_records
from .sft_dataset_split import SFTDatasetSplit, split_records


@dataclass(frozen=True)
class SFTDatasetPolicy:
    example_count: int = 500
    validation_count: int = 50
    selection_seed: int = 67
    split_seed: int = 6701
    quality: SFTQualityPolicy = SFTQualityPolicy()

    def __post_init__(self) -> None:
        if type(self.example_count) is not int or not 8 <= self.example_count <= 5_000:
            raise SFTDatasetError("example_count must be an integer in [8, 5000]")
        if (
            type(self.validation_count) is not int
            or not 1 <= self.validation_count < self.example_count
        ):
            raise SFTDatasetError("validation_count must leave nonempty training data")
        for name in ("selection_seed", "split_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2**32 - 1:
                raise SFTDatasetError(f"{name} must be an unsigned 32-bit integer")
        if not isinstance(self.quality, SFTQualityPolicy):
            raise SFTDatasetError("quality must be an SFTQualityPolicy")

    def as_mapping(self) -> dict:
        return {
            "example_count": self.example_count,
            "quality": asdict(self.quality),
            "selection_seed": self.selection_seed,
            "split_seed": self.split_seed,
            "validation_count": self.validation_count,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SFTDatasetPolicy":
        fields = {
            "example_count",
            "validation_count",
            "selection_seed",
            "split_seed",
            "quality",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or not isinstance(value["quality"], dict)
        ):
            raise SFTDatasetError("SFT dataset policy fields are invalid")
        quality_fields = {
            "min_response_bytes",
            "max_prompt_bytes",
            "max_response_bytes",
            "reject_identity_answers",
        }
        if set(value["quality"]) != quality_fields:
            raise SFTDatasetError("SFT quality policy fields are invalid")
        return cls(
            **{key: item for key, item in value.items() if key != "quality"},
            quality=SFTQualityPolicy(**value["quality"]),
        )


@dataclass(frozen=True)
class SFTDatasetBuild:
    quality: SFTQualityResult
    dedup: SFTDedupResult
    selection: SFTSelectionResult
    split: SFTDatasetSplit
    metrics: dict
    artifacts: SFTDatasetArtifacts

    def __post_init__(self) -> None:
        if self.dedup.input_records != len(self.quality.accepted):
            raise SFTDatasetError("quality and deduplication counts do not reconcile")
        if self.selection.candidate_records != len(self.dedup.records):
            raise SFTDatasetError("deduplication and selection counts do not reconcile")
        if self.metrics["examples"] != len(self.selection.records):
            raise SFTDatasetError("selection and metric counts do not reconcile")
        validate_dataset_metrics(self.metrics)


def build_sft_dataset(
    records: tuple[DollyRecord, ...], policy: SFTDatasetPolicy
) -> SFTDatasetBuild:
    if not isinstance(policy, SFTDatasetPolicy):
        raise TypeError("policy must be SFTDatasetPolicy")
    quality = filter_quality(records, policy.quality)
    dedup = deduplicate_records(quality.accepted)
    selection = select_records(
        dedup.records, target=policy.example_count, seed=policy.selection_seed
    )
    split = split_records(
        selection.records,
        validation_count=policy.validation_count,
        seed=policy.split_seed,
    )
    metrics = dataset_metrics(split)
    artifacts = export_split(split)
    return SFTDatasetBuild(quality, dedup, selection, split, metrics, artifacts)
