"""Bounded durable storage for source-backed BPE training audits."""

import json
from pathlib import Path

from .bpe import TokenizerError
from .bpe_conformance import audit_fingerprint
from .serving_benchmark import _publish


def _validate_seal(report: dict) -> None:
    if (
        not isinstance(report, dict)
        or type(report.get("audit_sha256")) is not str
        or report["audit_sha256"] != audit_fingerprint(report)
    ):
        raise TokenizerError("training audit fingerprint mismatch")


def save_training_audit(path: Path, report: dict) -> None:
    if not isinstance(path, Path):
        raise TypeError("training audit path must be pathlib.Path")
    _validate_seal(report)
    _publish(path, report)


def load_training_audit(path: Path) -> dict:
    if not isinstance(path, Path):
        raise TypeError("training audit path must be pathlib.Path")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TokenizerError("duplicate training audit field")
            result[key] = value
        return result

    try:
        with path.open("rb") as stream:
            payload = stream.read(2 * 1024 * 1024 + 1)
        if len(payload) > 2 * 1024 * 1024:
            raise TokenizerError("training audit exceeds 2 MiB")
        report = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError("unable to read training audit") from exc
    _validate_seal(report)
    return report
