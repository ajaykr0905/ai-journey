"""Strict bounded JSON configuration for the Day 59 data pipeline."""

import json
from pathlib import Path

from .bpe import TokenizerError
from .data_pipeline import DataPipelineConfig
from .data_quality import QualityPolicy


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError("duplicate data pipeline config field")
        result[key] = value
    return result


def load_data_pipeline_config(path: Path) -> DataPipelineConfig:
    """Load an exact schema without accepting JSON aliases or excess input."""
    if not isinstance(path, Path):
        raise TypeError("config path must be pathlib.Path")
    try:
        with path.open("rb") as stream:
            payload = stream.read(64 * 1024 + 1)
        if len(payload) > 64 * 1024:
            raise TokenizerError("data pipeline config exceeds 64 KiB")
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError("unable to read data pipeline config") from exc
    expected = {
        "schema_version",
        "quality",
        "validation_count",
        "train_shards",
        "validation_shards",
        "seed",
        "vocabulary_size",
        "min_frequency",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise TokenizerError("data pipeline config fields are invalid")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise TokenizerError("unsupported data pipeline config version")
    quality = data["quality"]
    if not isinstance(quality, dict) or set(quality) != {
        "min_characters",
        "max_characters",
        "min_alphanumeric_fraction",
        "max_repeated_run",
    }:
        raise TokenizerError("data quality config fields are invalid")
    return DataPipelineConfig(
        quality=QualityPolicy(**quality),
        validation_count=data["validation_count"],
        train_shards=data["train_shards"],
        validation_shards=data["validation_shards"],
        seed=data["seed"],
        vocabulary_size=data["vocabulary_size"],
        min_frequency=data["min_frequency"],
    )
