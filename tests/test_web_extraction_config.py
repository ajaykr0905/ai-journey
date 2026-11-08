import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.web_extraction_config import load_extraction_policy


def _config():
    return {
        "schema_version": 1,
        "policy": {
            "minimum_pages": 100,
            "minimum_text_characters": 120,
            "maximum_replacement_fraction": 0.005,
        },
    }


def test_config_loads_exact_extraction_policy(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()))

    policy = load_extraction_policy(path)

    assert policy.minimum_pages == 100
    assert policy.minimum_text_characters == 120
    assert policy.maximum_replacement_fraction == 0.005


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(schema_version=2),
        lambda value: value["policy"].pop("minimum_pages"),
        lambda value: value["policy"].update(minimum_pages=True),
    ],
)
def test_config_rejects_schema_or_value_drift(tmp_path, mutation):
    value = _config()
    mutation(value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))

    with pytest.raises(TokenizerError):
        load_extraction_policy(path)


def test_config_rejects_duplicate_keys_and_symlinks(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"schema_version":1,"schema_version":1,"policy":{}}')
    with pytest.raises(TokenizerError, match="duplicate"):
        load_extraction_policy(path)

    target = tmp_path / "target.json"
    target.write_text(json.dumps(_config()))
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(TokenizerError, match="nonsymlink"):
        load_extraction_policy(path)
