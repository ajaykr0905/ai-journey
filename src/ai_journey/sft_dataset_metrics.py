"""Reconciled size, task, and supervision metrics for SFT partitions."""

from collections import Counter
import math

from .sft_dataset_format import format_dolly_record
from .sft_dataset_record import SFTDatasetError
from .sft_dataset_split import SFTDatasetSplit


def _distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise SFTDatasetError("metric distributions require observations")
    result = {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "total": sum(values),
    }
    if not math.isfinite(result["mean"]):
        raise SFTDatasetError("metric distribution mean must be finite")
    return result


def dataset_metrics(split: SFTDatasetSplit) -> dict:
    """Counts and UTF-8 size distributions over both partitions, for the manifest.

    Prompt sizes measure the formatted user turn, context included.
    """
    if not isinstance(split, SFTDatasetSplit):
        raise TypeError("split must be SFTDatasetSplit")
    rows = split.train + split.validation
    prompts = [
        len(format_dolly_record(record).messages[0].content.encode("utf-8"))
        for record in rows
    ]
    responses = [len(record.response.encode("utf-8")) for record in rows]
    return {
        "examples": len(rows),
        "train_examples": len(split.train),
        "validation_examples": len(split.validation),
        "context_examples": sum(bool(record.context) for record in rows),
        "category_examples": dict(
            sorted(Counter(record.category for record in rows).items())
        ),
        "prompt_utf8_bytes": _distribution(prompts),
        "assistant_utf8_bytes": _distribution(responses),
    }


def validate_dataset_metrics(metrics: object) -> dict:
    """Reject metrics whose counts or size distributions do not reconcile.

    examples must equal train plus validation and the category total; each
    distribution needs 0 < min <= max <= total and mean == total / examples.
    Raises SFTDatasetError.
    """
    fields = {
        "examples",
        "train_examples",
        "validation_examples",
        "context_examples",
        "category_examples",
        "prompt_utf8_bytes",
        "assistant_utf8_bytes",
    }
    if not isinstance(metrics, dict) or set(metrics) != fields:
        raise SFTDatasetError("SFT dataset metric fields are invalid")
    counts = [
        metrics["examples"],
        metrics["train_examples"],
        metrics["validation_examples"],
        metrics["context_examples"],
    ]
    if (
        any(type(value) is not int or value < 0 for value in counts)
        or metrics["examples"]
        != metrics["train_examples"] + metrics["validation_examples"]
        or metrics["context_examples"] > metrics["examples"]
        or not isinstance(metrics["category_examples"], dict)
        or not metrics["category_examples"]
        or any(
            not isinstance(key, str) or type(value) is not int or value < 1
            for key, value in metrics["category_examples"].items()
        )
        or sum(metrics["category_examples"].values()) != metrics["examples"]
    ):
        raise SFTDatasetError("SFT dataset counts do not reconcile")
    for name in ("prompt_utf8_bytes", "assistant_utf8_bytes"):
        value = metrics[name]
        if (
            not isinstance(value, dict)
            or set(value) != {"min", "max", "mean", "total"}
            or type(value["min"]) is not int
            or type(value["max"]) is not int
            or type(value["total"]) is not int
            or not isinstance(value["mean"], (int, float))
            or isinstance(value["mean"], bool)
            or not 0 < value["min"] <= value["max"] <= value["total"]
            or not math.isfinite(value["mean"])
            or value["mean"] != value["total"] / metrics["examples"]
        ):
            raise SFTDatasetError(f"{name} distribution is invalid")
    return metrics
