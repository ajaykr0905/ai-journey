"""Sealed validation for deterministic SFT experiment reports."""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256

from .sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from .sft_finetune_experiment import LIMITATIONS

SCHEMA = "ai-journey-sft-finetune-v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def report_sha256(report: dict) -> str:
    return sha256(
        _canonical({key: value for key, value in report.items() if key != "sha256"})
    ).hexdigest()


def build_finetune_report(result: dict) -> dict:
    try:
        owned = json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise SFTFineTuneError(f"experiment result is not finite JSON: {exc}") from exc
    report = {"schema": SCHEMA, "result": owned}
    report["sha256"] = report_sha256(report)
    return validate_finetune_report(report)


def validate_finetune_report(report: dict) -> dict:
    try:
        if not isinstance(report, dict) or set(report) != {
            "schema",
            "result",
            "sha256",
        }:
            raise SFTFineTuneError("report requires schema, result, and sha256")
        if report["schema"] != SCHEMA or _DIGEST.fullmatch(report["sha256"]) is None:
            raise SFTFineTuneError("report identity is invalid")
        if report_sha256(report) != report["sha256"]:
            raise SFTFineTuneError("report digest does not match its content")
        result = report["result"]
        expected = {
            "config",
            "dataset",
            "schedule_sha256",
            "initial_model",
            "final_model",
            "training",
            "comparison",
            "runtime",
            "limitations",
        }
        if not isinstance(result, dict) or set(result) != expected:
            raise SFTFineTuneError("experiment result fields are invalid")
        config = SFTFineTuneConfig.from_mapping(result["config"])
        if result["limitations"] != LIMITATIONS:
            raise SFTFineTuneError("report limitations are incomplete")
        for digest in (
            result["dataset"]["sha256"],
            result["schedule_sha256"],
            result["initial_model"]["model_sha256"],
            result["final_model"]["model_sha256"],
            result["training"]["final_model_sha256"],
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise SFTFineTuneError("report contains an invalid digest")
        if result["training"]["optimizer_steps"] != config.steps:
            raise SFTFineTuneError("training trace does not cover configured steps")
        if (
            result["training"]["final_model_sha256"]
            != result["final_model"]["model_sha256"]
        ):
            raise SFTFineTuneError("training and final model identities differ")
        before = result["comparison"]["before"]["score"]
        after = result["comparison"]["after"]["score"]
        for score in (before, after):
            if score["target_tokens"] <= 0 or not all(
                math.isfinite(score[name]) and score[name] >= 0
                for name in ("nll_sum", "mean_nll", "perplexity")
            ):
                raise SFTFineTuneError("comparison scores must be finite and covered")
            if not math.isclose(
                score["mean_nll"], score["nll_sum"] / score["target_tokens"]
            ):
                raise SFTFineTuneError("comparison mean NLL is inconsistent")
        expected_delta = after["mean_nll"] - before["mean_nll"]
        if result["comparison"]["delta_mean_nll"] != expected_delta:
            raise SFTFineTuneError("comparison delta is inconsistent")
        if result["runtime"]["device"] != "cpu" or result["runtime"]["threads"] != 1:
            raise SFTFineTuneError(
                "runtime controls are not deterministic CPU controls"
            )
        return report
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise SFTFineTuneError(f"invalid fine-tuning report: {exc}") from exc
