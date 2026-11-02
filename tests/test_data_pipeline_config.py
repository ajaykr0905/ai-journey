import json
from pathlib import Path

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_pipeline_config import load_data_pipeline_config


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_day59_config_loads_exact_pipeline_controls():
    config = load_data_pipeline_config(ROOT / "config/day-59-data-pipeline.json")

    assert config.seed == 59
    assert config.validation_count == 100
    assert config.train_shards == 4
    assert config.vocabulary_size == 320
    assert config.quality.min_characters == 20


def test_config_rejects_unknown_and_duplicate_fields(tmp_path):
    original = (ROOT / "config/day-59-data-pipeline.json").read_text()
    unknown = tmp_path / "unknown.json"
    unknown.write_text(original.replace('"schema_version": 1,', '"extra": 1,'))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(original.replace('"seed": 59,', '"seed": 59, "seed": 60,'))

    with pytest.raises(TokenizerError, match="fields"):
        load_data_pipeline_config(unknown)
    with pytest.raises(TokenizerError, match="duplicate"):
        load_data_pipeline_config(duplicate)


def test_config_rejects_controls_outside_runtime_bounds(tmp_path):
    payload = json.loads((ROOT / "config/day-59-data-pipeline.json").read_text())
    payload["validation_count"] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(TokenizerError, match="outside supported bounds"):
        load_data_pipeline_config(path)
