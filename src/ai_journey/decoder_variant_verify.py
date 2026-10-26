"""Replay a bounded CPU decoder comparison without changing its artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path

import torch

from .decoder_variant_cli import (
    MAX_WORK_UNITS,
    REPORT_SCHEMA_VERSION,
    REPORT_SCOPE,
    _read_bytes,
    _standard_variants,
)
from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_experiment import run_variant_comparison
from .decoder_variant_resources import expected_decoder_variant_parameter_count
from .gpt2_config import GPT2Config

MAX_REPORT_BYTES = 64 * 1024
_REPORT_FIELDS = {
    "schema_version",
    "device",
    "scope",
    "baseline_config",
    "learning_rate",
    "evaluation_windows",
    "torch_version",
    "seed",
    "steps",
    "batch_size",
    "context_length",
    "train_sha256",
    "heldout_sha256",
    "batch_offsets_sha256",
    "heldout_windows",
    "variants",
}
_VARIANT_FIELDS = {
    "name",
    "architecture_fingerprint",
    "parameter_count",
    "initial_train_nll",
    "final_train_nll",
    "heldout_nll",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    return parser


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not permitted: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_report(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"report must be a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        payload = stream.read(MAX_REPORT_BYTES + 1)
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError(f"report exceeds {MAX_REPORT_BYTES} bytes: {path}")
    report = json.loads(
        payload.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if type(report) is not dict:
        raise ValueError("report must be a JSON object")
    return report


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _digest(name: str, value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _baseline(payload: object) -> DecoderVariantConfig:
    field_names = {field.name for field in fields(GPT2Config)}
    if type(payload) is not dict or set(payload) != field_names:
        raise ValueError("baseline_config must contain every GPT2Config field")
    if type(payload["vocab_size"]) is not int or payload["vocab_size"] != 256:
        raise ValueError("baseline_config vocab_size must be 256")
    limits = (
        ("context_length", 2, 64),
        ("embedding_dim", 4, 64),
        ("head_count", 1, 16),
        ("layer_count", 1, 2),
    )
    values = {
        name: _bounded_int(name, payload[name], minimum, maximum)
        for name, minimum, maximum in limits
    }
    expected = GPT2Config(
        vocab_size=256,
        **values,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    if (
        any(
            type(payload[name]) is not float
            for name in (
                "layer_norm_epsilon",
                "embedding_dropout",
                "attention_dropout",
                "residual_dropout",
            )
        )
        or type(payload["activation_function"]) is not str
        or payload != asdict(expected)
    ):
        raise ValueError("baseline_config differs from the bounded CLI contract")
    return DecoderVariantConfig(expected)


def _validate_report(report: dict[str, object]) -> dict[str, DecoderVariantConfig]:
    if set(report) != _REPORT_FIELDS:
        raise ValueError("report schema fields do not match version 2")
    if type(report["schema_version"]) is not int or report["schema_version"] != (
        REPORT_SCHEMA_VERSION
    ):
        raise ValueError("schema_version must be 2")
    if report["device"] != "cpu" or report["scope"] != REPORT_SCOPE:
        raise ValueError("report device or scope is not the CPU comparison contract")
    if report["torch_version"] != str(torch.__version__):
        raise ValueError("torch_version must match the current PyTorch version")

    baseline = _baseline(report["baseline_config"])
    base = baseline.gpt2
    if (
        type(report["context_length"]) is not int
        or report["context_length"] != base.context_length
    ):
        raise ValueError("context_length does not match baseline_config")
    _bounded_int("seed", report["seed"], 0, 2**63 - 1)
    steps = _bounded_int("steps", report["steps"], 1, 64)
    batch_size = _bounded_int("batch_size", report["batch_size"], 1, 32)
    evaluation_windows = _bounded_int(
        "evaluation_windows", report["evaluation_windows"], 1, 256
    )
    _bounded_int("heldout_windows", report["heldout_windows"], 1, evaluation_windows)
    learning_rate = report["learning_rate"]
    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise ValueError("learning_rate must be finite and in (0, 0.1]")
    if (
        steps
        * batch_size
        * base.context_length
        * base.layer_count
        * (base.embedding_dim)
        > MAX_WORK_UNITS
    ):
        raise ValueError(f"report exceeds {MAX_WORK_UNITS} work units")
    for name in ("train_sha256", "heldout_sha256", "batch_offsets_sha256"):
        _digest(name, report[name])

    variants = _standard_variants(baseline)
    records = report["variants"]
    if type(records) is not list or len(records) != len(variants):
        raise ValueError("variants must contain exactly five arms")
    for record, (name, config) in zip(records, variants.items(), strict=True):
        if type(record) is not dict or set(record) != _VARIANT_FIELDS:
            raise ValueError("variants contain an invalid arm record")
        if record["name"] != name:
            raise ValueError("variants must be in canonical arm order")
        if record["architecture_fingerprint"] != config.fingerprint():
            raise ValueError(f"architecture fingerprint differs for {name}")
        if type(record["parameter_count"]) is not int or record[
            "parameter_count"
        ] != expected_decoder_variant_parameter_count(config):
            raise ValueError(f"parameter_count differs for {name}")
        for field in ("initial_train_nll", "final_train_nll", "heldout_nll"):
            if type(record[field]) is not float or not math.isfinite(record[field]):
                raise ValueError(f"{field} must be a finite measured loss")
    return variants


def main(argv: Sequence[str] | None = None) -> int:
    """Recompute every measurement and compare it with an unchanged report."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = _read_report(args.report)
        variants = _validate_report(report)
        train_bytes = _read_bytes(args.train)
        heldout_bytes = _read_bytes(args.heldout)
        if sha256(train_bytes).hexdigest() != report["train_sha256"]:
            raise ValueError("train_sha256 does not match supplied training bytes")
        if sha256(heldout_bytes).hexdigest() != report["heldout_sha256"]:
            raise ValueError("heldout_sha256 does not match supplied held-out bytes")
        result = run_variant_comparison(
            train_bytes=train_bytes,
            heldout_bytes=heldout_bytes,
            variants=variants,
            seed=report["seed"],
            steps=report["steps"],
            batch_size=report["batch_size"],
            learning_rate=report["learning_rate"],
            evaluation_windows=report["evaluation_windows"],
        )
        expected = asdict(result)
        expected["variants"] = list(expected["variants"])
        expected.update(
            schema_version=REPORT_SCHEMA_VERSION,
            device="cpu",
            scope=REPORT_SCOPE,
            baseline_config=asdict(variants["baseline"].gpt2),
            learning_rate=report["learning_rate"],
            evaluation_windows=report["evaluation_windows"],
            torch_version=str(torch.__version__),
        )
        if report != expected:
            raise ValueError("report does not match deterministic replay")
    except (OSError, TypeError, ValueError, FloatingPointError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"verified={args.report} device=cpu variants={len(variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
