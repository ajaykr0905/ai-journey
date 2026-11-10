"""Independent structural and cryptographic verification of SFT artifacts."""

from hashlib import sha256
import json
from pathlib import Path

from .sft_dataset_build import SFTDatasetPolicy
from .sft_dataset_format import ChatExample, ChatTurn
from .sft_dataset_metrics import validate_dataset_metrics
from .sft_dataset_record import SFTDatasetError
from .sft_dataset_report import manifest_sha256
from .sft_dataset_source import SFTSourceSpec
from .sft_dataset_template import RenderedChat, render_chatml


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SFTDatasetError(f"duplicate artifact field: {key}")
        value[key] = item
    return value


def _nonfinite(value):
    raise SFTDatasetError(f"nonfinite artifact number: {value}")


def _read_bounded(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise SFTDatasetError("SFT artifact must be a regular file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise SFTDatasetError("unable to read SFT artifact") from exc
    if len(payload) > limit:
        raise SFTDatasetError("SFT artifact exceeds its byte budget")
    return payload


def _load_json(payload: bytes, label: str):
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SFTDatasetError(f"invalid {label} JSON") from exc


def _verify_descriptors(directory: Path, manifest: dict) -> dict[str, bytes]:
    if set(manifest["artifacts"]) != {"dataset.jsonl", "chatml.jsonl"}:
        raise SFTDatasetError("SFT artifact descriptors are invalid")
    payloads = {}
    for name, descriptor in manifest["artifacts"].items():
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"bytes", "sha256"}
            or type(descriptor["bytes"]) is not int
            or not 1 <= descriptor["bytes"] <= 64 * 1024 * 1024
            or not isinstance(descriptor["sha256"], str)
            or len(descriptor["sha256"]) != 64
        ):
            raise SFTDatasetError("SFT artifact descriptor values are invalid")
        payload = _read_bounded(directory / name, descriptor["bytes"])
        if (
            len(payload) != descriptor["bytes"]
            or sha256(payload).hexdigest() != descriptor["sha256"]
            or not payload.endswith(b"\n")
        ):
            raise SFTDatasetError("SFT artifact identity mismatch")
        payloads[name] = payload
    return payloads


def _dataset_examples(payload: bytes) -> list[tuple[str, ChatExample]]:
    examples = []
    seen = set()
    for line in payload.splitlines():
        row = _load_json(line, "dataset row")
        if not isinstance(row, dict) or set(row) != {
            "category",
            "example_id",
            "messages",
            "source_sha256",
            "split",
        }:
            raise SFTDatasetError("SFT dataset row fields are invalid")
        messages = row["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or any(
                not isinstance(message, dict) or set(message) != {"content", "role"}
                for message in messages
            )
        ):
            raise SFTDatasetError("SFT dataset messages are invalid")
        example = ChatExample(
            example_id=row["example_id"],
            source_sha256=row["source_sha256"],
            category=row["category"],
            messages=tuple(ChatTurn(**message) for message in messages),
        )
        if example.example_id != f"dolly-{example.source_sha256[:24]}":
            raise SFTDatasetError("SFT example identity is not bound to source content")
        if row["split"] not in {"train", "validation"} or example.example_id in seen:
            raise SFTDatasetError("SFT example split or uniqueness is invalid")
        seen.add(example.example_id)
        examples.append((row["split"], example))
    return examples


def _verify_order(examples: list[tuple[str, ChatExample]]) -> None:
    splits = [split for split, _ in examples]
    if splits != sorted(splits, key={"train": 0, "validation": 1}.get):
        raise SFTDatasetError("SFT dataset partitions are not contiguous")
    for partition in ("train", "validation"):
        identities = [
            example.example_id for split, example in examples if split == partition
        ]
        if identities != sorted(identities):
            raise SFTDatasetError("SFT dataset partition identities are not sorted")


def _verify_chatml(payload: bytes, examples: list[tuple[str, ChatExample]]) -> None:
    rows = [_load_json(line, "ChatML row") for line in payload.splitlines()]
    if len(rows) != len(examples):
        raise SFTDatasetError("SFT ChatML and message row counts differ")
    for row, (partition, example) in zip(rows, examples, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "assistant_spans",
            "example_id",
            "split",
            "text",
        }:
            raise SFTDatasetError("SFT ChatML row fields are invalid")
        spans = row["assistant_spans"]
        if (
            not isinstance(spans, list)
            or not spans
            or any(not isinstance(span, list) or len(span) != 2 for span in spans)
        ):
            raise SFTDatasetError("SFT assistant spans are invalid")
        rendered = RenderedChat(
            row["example_id"], row["text"], tuple(tuple(span) for span in spans)
        )
        if row["split"] != partition or rendered != render_chatml(example):
            raise SFTDatasetError("SFT ChatML row does not match its message example")


def _verify_manifest(manifest: object) -> tuple[SFTSourceSpec, SFTDatasetPolicy]:
    fields = {
        "schema_version",
        "source",
        "policy",
        "pipeline",
        "metrics",
        "artifacts",
        "attribution",
        "evidence_boundary",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != fields
        or manifest["schema_version"] != 1
        or not isinstance(manifest["manifest_sha256"], str)
        or manifest_sha256(manifest) != manifest["manifest_sha256"]
        or not isinstance(manifest["attribution"], str)
        or "Databricks" not in manifest["attribution"]
        or not isinstance(manifest["evidence_boundary"], str)
        or "no claim" not in manifest["evidence_boundary"]
    ):
        raise SFTDatasetError("SFT manifest fields or seal are invalid")
    source = SFTSourceSpec.from_mapping(manifest["source"])
    policy = SFTDatasetPolicy.from_mapping(manifest["policy"])
    validate_dataset_metrics(manifest["metrics"])
    pipeline = manifest["pipeline"]
    pipeline_fields = {
        "input_records",
        "quality_accepted_records",
        "quality_rejected_records",
        "quality_rejection_counts",
        "exact_duplicate_records",
        "duplicate_counts",
        "selection_candidate_records",
        "selection_quotas",
    }
    if (
        not isinstance(pipeline, dict)
        or set(pipeline) != pipeline_fields
        or any(
            type(pipeline[name]) is not int or pipeline[name] < 0
            for name in (
                "input_records",
                "quality_accepted_records",
                "quality_rejected_records",
                "exact_duplicate_records",
                "selection_candidate_records",
            )
        )
        or pipeline["input_records"]
        != pipeline["quality_accepted_records"] + pipeline["quality_rejected_records"]
        or pipeline["quality_accepted_records"]
        != pipeline["selection_candidate_records"] + pipeline["exact_duplicate_records"]
        or not isinstance(pipeline["quality_rejection_counts"], dict)
        or sum(pipeline["quality_rejection_counts"].values())
        != pipeline["quality_rejected_records"]
        or not isinstance(pipeline["selection_quotas"], dict)
        or sum(pipeline["selection_quotas"].values()) != policy.example_count
        or not isinstance(pipeline["duplicate_counts"], list)
        or sum(row[1] - 1 for row in pipeline["duplicate_counts"])
        != pipeline["exact_duplicate_records"]
    ):
        raise SFTDatasetError("SFT pipeline evidence does not reconcile")
    return source, policy


def verify_sft_dataset(directory: Path) -> dict:
    """Verify inventory, seals, metrics, messages, templates, and loss spans."""
    if not isinstance(directory, Path):
        raise TypeError("directory must be pathlib.Path")
    if not directory.is_dir() or directory.is_symlink():
        raise SFTDatasetError("SFT evidence root must be a regular directory")
    if {path.name for path in directory.iterdir()} != {
        "manifest.json",
        "dataset.jsonl",
        "chatml.jsonl",
    }:
        raise SFTDatasetError("SFT evidence inventory is invalid")
    manifest = _load_json(
        _read_bounded(directory / "manifest.json", 1024 * 1024), "manifest"
    )
    _, policy = _verify_manifest(manifest)
    payloads = _verify_descriptors(directory, manifest)
    examples = _dataset_examples(payloads["dataset.jsonl"])
    _verify_order(examples)
    _verify_chatml(payloads["chatml.jsonl"], examples)
    metrics = manifest["metrics"]
    if (
        len(examples) != policy.example_count
        or len(examples) != metrics["examples"]
        or sum(split == "train" for split, _ in examples) != metrics["train_examples"]
        or sum(split == "validation" for split, _ in examples)
        != metrics["validation_examples"]
        or metrics["validation_examples"] != policy.validation_count
    ):
        raise SFTDatasetError("SFT artifact counts do not match policy and metrics")
    return manifest
