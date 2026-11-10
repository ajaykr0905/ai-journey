from pathlib import Path

from ai_journey.sft_dataset_cli import load_build_config


ROOT = Path(__file__).resolve().parents[1]


def test_day_67_config_pins_public_source_license_and_exact_dataset_size():
    source, policy = load_build_config(ROOT / "config/day-67-sft-dataset.json")
    assert source.dataset_id == "databricks/databricks-dolly-15k"
    assert source.revision == "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
    assert (
        source.sha256
        == "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    )
    assert source.bytes == 13_085_339
    assert source.license == "CC-BY-SA-3.0"
    assert source.revision in source.source_url
    assert policy.example_count == 500
    assert policy.validation_count == 50
    assert policy.selection_seed != policy.split_seed


def test_day_67_config_keeps_quality_limits_explicit_and_bounded():
    _, policy = load_build_config(ROOT / "config/day-67-sft-dataset.json")
    assert policy.quality.min_response_bytes == 1
    assert policy.quality.max_prompt_bytes == 16 * 1024
    assert policy.quality.max_response_bytes == 16 * 1024
    assert policy.quality.reject_identity_answers is True
