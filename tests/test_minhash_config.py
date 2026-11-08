import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_config import load_minhash_policy


def _payload():
    return {
        "schema_version": 1,
        "policy": {
            "shingle_width": 5,
            "num_hashes": 128,
            "seed": 20261109,
            "bands": 32,
            "duplicate_threshold": 0.8,
        },
    }


def test_config_loads_exact_reproducible_policy(tmp_path):
    path = tmp_path / "minhash.json"
    path.write_text(json.dumps(_payload()))

    policy = load_minhash_policy(path)

    assert policy.shingle_width == 5
    assert policy.num_hashes == 128
    assert policy.seed == 20261109
    assert policy.bands == 32
    assert policy.duplicate_threshold == 0.8


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "policy": _payload()["policy"]},
        {"schema_version": 1, "policy": {}},
        {**_payload(), "extra": True},
    ],
)
def test_config_rejects_unknown_versions_and_fields(tmp_path, payload):
    path = tmp_path / "minhash.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TokenizerError, match="version|fields"):
        load_minhash_policy(path)


def test_config_rejects_duplicate_json_fields(tmp_path):
    path = tmp_path / "minhash.json"
    path.write_text('{"schema_version":1,"schema_version":1,"policy":{}}')
    with pytest.raises(TokenizerError, match="duplicate"):
        load_minhash_policy(path)


def test_config_rejects_symlinks_and_oversized_files(tmp_path):
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_payload()))
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(TokenizerError, match="nonsymlink"):
        load_minhash_policy(link)

    target.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(TokenizerError, match="64 KiB"):
        load_minhash_policy(target)


def test_config_requires_pathlib_path():
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_minhash_policy("minhash.json")
