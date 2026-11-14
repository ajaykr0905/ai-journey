"""End-to-end deterministic cleaning, splitting, sharding, and tokenization."""

from dataclasses import dataclass

from .bpe import ByteTokenizer, TokenizerError
from .data_cleaning import clean_documents
from .data_dedup import DeduplicationResult, deduplicate_documents
from .data_documents import DataDocument
from .data_quality import QualityDecision, QualityPolicy, assess_quality
from .data_sharding import DataShard, shard_documents
from .data_split import DataSplit, split_documents
from .data_tokenization import TokenizationReport, build_tokenization_report


@dataclass(frozen=True)
class DataPipelineConfig:
    """Day 59 controls; defaults equal config/day-59-data-pipeline.json.

    Vocabulary 256-4096, min_frequency of at least 2, positive counts and a
    32-bit seed keep the run small and valid; violations raise TokenizerError.
    """

    quality: QualityPolicy = QualityPolicy()
    validation_count: int = 100
    train_shards: int = 4
    validation_shards: int = 1
    seed: int = 59
    vocabulary_size: int = 320
    min_frequency: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.quality, QualityPolicy):
            raise TypeError("pipeline quality must be a QualityPolicy")
        values = (
            self.validation_count,
            self.train_shards,
            self.validation_shards,
            self.seed,
            self.vocabulary_size,
            self.min_frequency,
        )
        if any(type(value) is not int for value in values):
            raise TokenizerError("pipeline controls must be integers")
        if (
            self.validation_count < 1
            or self.train_shards < 1
            or self.validation_shards < 1
            or not 0 <= self.seed < 2**32
            or not 256 <= self.vocabulary_size <= 4_096
            or self.min_frequency < 2
        ):
            raise TokenizerError("pipeline controls are outside supported bounds")


@dataclass(frozen=True)
class DataPipelineResult:
    """Every stage's output, rejections and duplicates included, for the report."""

    config: DataPipelineConfig
    input_documents: int
    changed_documents: int
    quality: tuple[QualityDecision, ...]
    deduplication: DeduplicationResult
    split: DataSplit
    train_shards: tuple[DataShard, ...]
    validation_shards: tuple[DataShard, ...]
    tokenizer: ByteTokenizer
    tokenization: TokenizationReport


def build_data_pipeline(
    documents: tuple[DataDocument, ...], config: DataPipelineConfig
) -> DataPipelineResult:
    """Execute the complete in-memory pipeline with explicit rejected records."""
    if not isinstance(config, DataPipelineConfig):
        raise TypeError("pipeline requires a DataPipelineConfig")
    cleaned = clean_documents(documents)
    decisions = tuple(assess_quality(document, config.quality) for document in cleaned)
    admitted = tuple(decision.document for decision in decisions if decision.accepted)
    if len(admitted) < config.validation_count + 1:
        raise TokenizerError("quality admission leaves insufficient split data")
    deduplication = deduplicate_documents(admitted)
    if len(deduplication.documents) < config.validation_count + 1:
        raise TokenizerError("deduplication leaves insufficient split data")
    split = split_documents(
        deduplication.documents,
        validation_count=config.validation_count,
        seed=config.seed,
    )
    if config.train_shards > len(split.train) or config.validation_shards > len(
        split.validation
    ):
        raise TokenizerError("shard counts must leave every shard nonempty")
    train_shards = shard_documents(
        split.train, shard_count=config.train_shards, seed=config.seed
    )
    validation_shards = shard_documents(
        split.validation,
        shard_count=config.validation_shards,
        seed=config.seed,
    )
    tokenizer, tokenization = build_tokenization_report(
        split.train,
        split.validation,
        vocabulary_size=config.vocabulary_size,
        min_frequency=config.min_frequency,
    )
    return DataPipelineResult(
        config,
        len(documents),
        sum(document.changed for document in cleaned),
        decisions,
        deduplication,
        split,
        train_shards,
        validation_shards,
        tokenizer,
        tokenization,
    )
