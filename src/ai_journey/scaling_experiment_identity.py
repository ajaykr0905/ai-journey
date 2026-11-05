"""Immutable public input identities for scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .scaling_laws import ScalingLawError

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ExperimentInputs:
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
