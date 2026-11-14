"""Immutable public input identities for scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .scaling_experiment_protocol import ExperimentPoint
from .scaling_laws import ScalingLawError

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ExperimentInputs:
    """The immutable public inputs that every run identity binds.

    The source URL must be HTTP(S), digests lowercase SHA-256 and the code
    revision a full Git SHA; violations raise ScalingLawError.
    """

    source_url: str
    source_sha256: str
    tokenizer_sha256: str
    code_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_url, str) or not self.source_url.startswith(
            ("https://", "http://")
        ):
            raise ScalingLawError("source_url must be an HTTP(S) URL")
        for name in ("source_sha256", "tokenizer_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ScalingLawError(f"{name} must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.code_revision, str)
            or _GIT_SHA.fullmatch(self.code_revision) is None
        ):
            raise ScalingLawError("code_revision must be a full lowercase Git SHA")


def run_identity(point: ExperimentPoint, inputs: ExperimentInputs) -> str:
    """Return a stable digest for one coordinate and all immutable inputs."""

    if not isinstance(point, ExperimentPoint):
        raise TypeError("point must be ExperimentPoint")
    if not isinstance(inputs, ExperimentInputs):
        raise TypeError("inputs must be ExperimentInputs")
    payload = {
        "code_revision": inputs.code_revision,
        "parameter_count": point.parameter_count,
        "seed": point.seed,
        "source_sha256": inputs.source_sha256,
        "source_url": inputs.source_url,
        "tokenizer_sha256": inputs.tokenizer_sha256,
        "train_tokens": point.train_tokens,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
