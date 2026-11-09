"""Strict MinHash deduplication configuration loading."""

import json
from pathlib import Path

from .bpe import TokenizerError
from .minhash_dedup import MinHashDedupPolicy


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise TokenizerError("duplicate MinHash config field")
        value[key] = item
    return value


def load_minhash_policy(path: Path) -> MinHashDedupPolicy:
    """Read one bounded, nonsymlink JSON policy with exact fields."""
    if not isinstance(path, Path):
        raise TypeError("MinHash config path must be pathlib.Path")
    if not path.is_file() or path.is_symlink():
        raise TokenizerError("MinHash config must be a regular nonsymlink file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(64 * 1024 + 1)
        if len(payload) > 64 * 1024:
            raise TokenizerError("MinHash config exceeds 64 KiB")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError("unable to read MinHash config") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "policy"}:
        raise TokenizerError("MinHash config fields are invalid")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise TokenizerError("unsupported MinHash config version")
    policy = value["policy"]
    expected = {
        "shingle_width",
        "num_hashes",
        "seed",
        "bands",
        "duplicate_threshold",
    }
    if not isinstance(policy, dict) or set(policy) != expected:
        raise TokenizerError("MinHash policy fields are invalid")
    try:
        return MinHashDedupPolicy(**policy)
    except TypeError as exc:
        raise TokenizerError("MinHash policy values are invalid") from exc
