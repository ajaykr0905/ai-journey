"""Self-authenticating evidence records for GPT-2 interoperability runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping

from .gpt2_audit import LogitParityResult
from .gpt2_config import GPT2Config
from .gpt2_source import GPT2Source


class GPT2EvidenceError(ValueError):
    """Raised when GPT-2 evidence is incomplete or has been altered."""


_SHA256 = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class GPT2Evidence:
    """Complete provenance and parity evidence for one imported checkpoint."""

    model_id: str
    revision: str
    architecture_fingerprint: str
    state_fingerprint: str
    parameter_count: int
    token_ids: tuple[int, ...]
    compared_values: int
    max_absolute_error: float
    mean_absolute_error: float
    top1_matches: int
    top1_total: int
    reference_digest: str
    candidate_digest: str
    torch_version: str
    python_version: str

    def __post_init__(self) -> None:
        for name in (
            "architecture_fingerprint",
            "state_fingerprint",
            "reference_digest",
            "candidate_digest",
        ):
            if not isinstance(getattr(self, name), str) or not _SHA256.fullmatch(
                getattr(self, name)
            ):
                raise GPT2EvidenceError(f"{name} must be a SHA-256 digest")
        if not self.model_id or not self.revision:
            raise GPT2EvidenceError("model identity must be non-empty")
        if not self.token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.token_ids
        ):
            raise GPT2EvidenceError("token_ids must contain non-negative integers")
        for name in ("parameter_count", "compared_values", "top1_total"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GPT2EvidenceError(f"{name} must be a positive integer")
        if (
            isinstance(self.top1_matches, bool)
            or not isinstance(self.top1_matches, int)
            or not 0 <= self.top1_matches <= self.top1_total
        ):
            raise GPT2EvidenceError("top1_matches must be in [0, top1_total]")
        for name in ("max_absolute_error", "mean_absolute_error"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise GPT2EvidenceError(f"{name} must be finite and non-negative")
        if not self.torch_version or not self.python_version:
            raise GPT2EvidenceError("runtime versions must be non-empty")


def build_evidence(
    source: GPT2Source,
    config: GPT2Config,
    *,
    state_fingerprint: str,
    parameter_count: int,
    parity: LogitParityResult,
    torch_version: str,
    python_version: str,
) -> GPT2Evidence:
    """Bind source, architecture, learned state, runtime, and parity results."""

    if not isinstance(source, GPT2Source):
        raise TypeError("source must be GPT2Source")
    if not isinstance(config, GPT2Config):
        raise TypeError("config must be GPT2Config")
    if not isinstance(parity, LogitParityResult):
        raise TypeError("parity must be LogitParityResult")
    expected_values = len(parity.token_ids) * config.vocab_size
    if parity.compared_values != expected_values:
        raise GPT2EvidenceError("parity did not cover every probe vocabulary logit")
    if parity.top1_total != len(parity.token_ids):
        raise GPT2EvidenceError("parity top-1 total does not match the probe length")
    return GPT2Evidence(
        model_id=source.model_id,
        revision=source.revision,
        architecture_fingerprint=config.fingerprint(),
        state_fingerprint=state_fingerprint,
        parameter_count=parameter_count,
        token_ids=parity.token_ids,
        compared_values=parity.compared_values,
        max_absolute_error=parity.max_absolute_error,
        mean_absolute_error=parity.mean_absolute_error,
        top1_matches=parity.top1_matches,
        top1_total=parity.top1_total,
        reference_digest=parity.reference_digest,
        candidate_digest=parity.candidate_digest,
        torch_version=torch_version,
        python_version=python_version,
    )


def evidence_payload(evidence: GPT2Evidence) -> dict[str, Any]:
    """Return a canonical payload with a digest over every evidence field."""

    if not isinstance(evidence, GPT2Evidence):
        raise TypeError("evidence must be GPT2Evidence")
    core = {"schema_version": 1, "evidence": asdict(evidence)}
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    return {**core, "evidence_sha256": sha256(encoded).hexdigest()}


def validate_evidence_payload(payload: Mapping[str, Any]) -> GPT2Evidence:
    """Validate an exact evidence schema and its canonical content digest."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if set(payload) != {"schema_version", "evidence", "evidence_sha256"}:
        raise GPT2EvidenceError("evidence payload fields are not exact")
    if payload["schema_version"] != 1:
        raise GPT2EvidenceError("unsupported evidence schema version")
    values = payload["evidence"]
    if not isinstance(values, Mapping):
        raise GPT2EvidenceError("evidence must be an object")
    expected_fields = set(GPT2Evidence.__dataclass_fields__)
    if set(values) != expected_fields:
        raise GPT2EvidenceError("evidence fields are not exact")
    prepared = dict(values)
    if not isinstance(prepared["token_ids"], (list, tuple)):
        raise GPT2EvidenceError("token_ids must be an array")
    prepared["token_ids"] = tuple(prepared["token_ids"])
    evidence = GPT2Evidence(**prepared)
    expected = evidence_payload(evidence)["evidence_sha256"]
    if payload["evidence_sha256"] != expected:
        raise GPT2EvidenceError("evidence SHA-256 mismatch")
    return evidence
