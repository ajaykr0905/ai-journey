"""Strict configuration loading for reproducible HTML extraction runs."""

import json
from pathlib import Path

from .bpe import TokenizerError
from .web_corpus import ExtractionPolicy


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError("duplicate HTML extraction config field")
        result[key] = value
    return result


def load_extraction_policy(path: Path) -> ExtractionPolicy:
    if not isinstance(path, Path):
        raise TypeError("HTML extraction config path must be pathlib.Path")
    if not path.is_file() or path.is_symlink():
        raise TokenizerError("HTML extraction config must be a regular nonsymlink file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(64 * 1024 + 1)
        if len(payload) > 64 * 1024:
            raise TokenizerError("HTML extraction config exceeds 64 KiB")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError("unable to read HTML extraction config") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "policy"}:
        raise TokenizerError("HTML extraction config fields are invalid")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise TokenizerError("unsupported HTML extraction config version")
    policy = value["policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "minimum_pages",
        "minimum_text_characters",
        "maximum_replacement_fraction",
    }:
        raise TokenizerError("HTML extraction policy fields are invalid")
    try:
        return ExtractionPolicy(**policy)
    except TypeError as exc:
        raise TokenizerError("HTML extraction policy values are invalid") from exc
