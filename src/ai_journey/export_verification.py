"""Verify exported partitions; optional pinned snapshot proves text provenance."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .bpe import TokenizerError
from .tokenizer_sources import HINDI, ENGLISH, load_public_corpus


def verify_export(
    manifest_path: Path, train_path: Path, validation_path: Path, *, source_path=None
):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TokenizerError("duplicate export manifest field")
            result[key] = value
        return result

    with manifest_path.open("rb") as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise TokenizerError("export manifest exceeds 1 MiB")
    manifest = json.loads(payload, object_pairs_hook=unique)
    fields = {
        "schema_version",
        "source",
        "source_url",
        "license_url",
        "attribution",
        "split_seed",
        "unused_sentences",
        "partitions",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != fields
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise TokenizerError("export manifest fields are invalid")
    source = next(
        (source for source in (HINDI, ENGLISH) if asdict(source) == manifest["source"]),
        None,
    )
    if (
        source is None
        or manifest["source_url"] != source.url
        or manifest["license_url"] != source.license_url
    ):
        raise TokenizerError("export manifest source pin or license is invalid")
    if (
        type(manifest["split_seed"]) is not int
        or manifest["split_seed"] != 40
        or type(manifest["unused_sentences"]) is not int
        or manifest["unused_sentences"] < 0
    ):
        raise TokenizerError("export split controls are invalid")
    partitions = manifest["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation"}:
        raise TokenizerError("export requires train and validation partitions")
    seen = set()
    texts = {}
    for name, path in (("train", train_path), ("validation", validation_path)):
        partition = partitions[name]
        if not isinstance(partition, dict) or set(partition) != {
            "sentence_ids",
            "sha256",
            "size_bytes",
        }:
            raise TokenizerError("export partition fields are invalid")
        ids = partition["sentence_ids"]
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 100
            or any(
                not isinstance(value, str)
                or not value
                or any(char.isspace() for char in value)
                for value in ids
            )
        ):
            raise TokenizerError("export sentence ids are invalid")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise TokenizerError("export partition ids must be unique and disjoint")
        seen.update(ids)
        size = partition["size_bytes"]
        if type(size) is not int or not 1 <= size <= 64 * 1024:
            raise TokenizerError("export partition exceeds 64 KiB")
        with path.open("rb") as stream:
            data = stream.read(size + 1)
        if len(data) != size or sha256(data).hexdigest() != partition["sha256"]:
            raise TokenizerError("export partition content identity mismatch")
        lines = data.decode("utf-8").split("\n")
        if (
            lines[-1] != ""
            or len(lines) - 1 != len(ids)
            or any(not line.strip() for line in lines[:-1])
        ):
            raise TokenizerError(
                "export text must contain exactly one line per sentence id"
            )
        texts.update(zip(ids, lines[:-1], strict=True))
    if source_path is not None:
        original = {
            row.sentence_id: row.text for row in load_public_corpus(source_path, source)
        }
        if len(seen) + manifest["unused_sentences"] != len(original) or any(
            original.get(key) != text for key, text in texts.items()
        ):
            raise TokenizerError(
                "export text or ids do not match the pinned source snapshot"
            )
    return manifest
