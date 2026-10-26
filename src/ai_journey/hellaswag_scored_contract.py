"""Semantic contracts for serialized completion-only HellaSwag scores."""

from hashlib import sha256
import json
import math

from .hellaswag import HellaSwagError, probabilities_from_scores


def require_scoring_digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HellaSwagError(f"{name} must be a lowercase SHA-256 digest")


def _finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def validate_scored_rows(records, protocol) -> str:
    """Recompute decisions, probabilities and identities without rerunning a model."""
    fields = {
        "row_index",
        "example_index",
        "example_sha256",
        "label",
        "prediction",
        "correct",
        "scores",
        "probabilities",
        "confidence",
        "normalization",
        "split_type",
    }
    for row in records:
        if not isinstance(row, dict) or set(row) != fields:
            raise HellaSwagError("scored row fields do not match")
        if any(
            type(row[name]) is not int or row[name] < 0
            for name in ("row_index", "example_index")
        ):
            raise HellaSwagError("scored row identity is invalid")
        require_scoring_digest(row["example_sha256"], "example_sha256")
        if any(
            type(row[name]) is not int or not 0 <= row[name] < 4
            for name in ("label", "prediction")
        ):
            raise HellaSwagError("scored row labels must identify one of four endings")
        if row["normalization"] != "completion_mean_nll" or row["split_type"] not in (
            "indomain",
            "zeroshot",
        ):
            raise HellaSwagError("scored row normalization or split type is invalid")
        scores = row["scores"]
        if not isinstance(scores, list) or len(scores) != 4:
            raise HellaSwagError("scored row requires all four candidate scores")
        for score in scores:
            if not isinstance(score, dict) or set(score) != {
                "tokens",
                "loss_sum",
                "mean_nll",
            }:
                raise HellaSwagError("candidate score fields do not match")
            if (
                type(score["tokens"]) is not int
                or not 1 <= score["tokens"] <= 1024
                or not all(
                    _finite_number(score[name]) for name in ("loss_sum", "mean_nll")
                )
            ):
                raise HellaSwagError(
                    "candidate score token or loss accounting is invalid"
                )
            if not math.isclose(
                score["loss_sum"] / score["tokens"],
                score["mean_nll"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise HellaSwagError("candidate mean NLL does not match token totals")
        prediction = min(range(4), key=lambda index: scores[index]["mean_nll"])
        if (
            row["prediction"] != prediction
            or type(row["correct"]) is not bool
            or row["correct"] != (prediction == row["label"])
        ):
            raise HellaSwagError(
                "scored row outcome does not match its candidate losses"
            )
        expected = probabilities_from_scores(scores)
        probabilities = row["probabilities"]
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != 4
            or any(
                not _finite_number(value)
                or not math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)
                for value, target in zip(probabilities, expected, strict=True)
            )
        ):
            raise HellaSwagError(
                "candidate probabilities do not match their NLL scores"
            )
        if not _finite_number(row["confidence"]) or not math.isclose(
            row["confidence"], probabilities[prediction], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise HellaSwagError(
                "candidate confidence does not match the predicted ending"
            )
    if (
        [row["row_index"] for row in records]
        != list(range(protocol.start, protocol.start + protocol.count))
        or len({row["example_index"] for row in records}) != len(records)
        or len({row["example_sha256"] for row in records}) != len(records)
    ):
        raise HellaSwagError(
            "scored rows do not cover the selected protocol identities"
        )
    identity = [
        {"row_index": row["row_index"], "example_sha256": row["example_sha256"]}
        for row in records
    ]
    return sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
