"""Installed commands for importing and verifying public GPT-2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Sequence

import torch
from torch import Tensor

from .gpt2_audit import audit_logit_parity
from .gpt2_evidence import GPT2EvidenceError, build_evidence, evidence_payload
from .gpt2_evidence_io import load_evidence, write_evidence
from .gpt2_interop import (
    fingerprint_hugging_face_state,
    load_hugging_face_weights,
    normalize_hugging_face_checkpoint,
)
from .gpt2_model import GPT2LanguageModel
from .gpt2_source import load_checkpoint_state_dict, load_source_manifest


class GPT2CLIError(ValueError):
    """Raised when an import or verification gate fails."""


def load_reference_probe(
    path: str | Path, *, max_bytes: int = 16 * 1024 * 1024
) -> tuple[Tensor, Tensor]:
    """Load a bounded weights-only token/logit reference probe."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise GPT2CLIError("reference probe must be a regular non-symlink file")
    if source.stat().st_size <= 0 or source.stat().st_size > max_bytes:
        raise GPT2CLIError("reference probe exceeds the byte limit")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise GPT2CLIError(
            "reference probe is not a weights-only tensor archive"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"token_ids", "logits"}:
        raise GPT2CLIError("reference probe must contain only token_ids and logits")
    token_ids = payload["token_ids"]
    logits = payload["logits"]
    if not isinstance(token_ids, Tensor) or not isinstance(logits, Tensor):
        raise GPT2CLIError("reference probe values must be tensors")
    return token_ids, logits


def import_checkpoint(
    *,
    source_path: str | Path,
    artifact_directory: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    max_absolute_error: float,
) -> dict[str, object]:
    """Import the pinned checkpoint, enforce parity, and publish evidence."""

    if max_absolute_error < 0:
        raise GPT2CLIError("max_absolute_error must be non-negative")
    source = load_source_manifest(source_path)
    config, checkpoint = load_checkpoint_state_dict(source, artifact_directory)
    state = normalize_hugging_face_checkpoint(checkpoint, config)
    state_fingerprint = fingerprint_hugging_face_state(state, config)
    model = GPT2LanguageModel(config)
    load_hugging_face_weights(model, state)
    token_ids, reference_logits = load_reference_probe(reference_path)
    parity = audit_logit_parity(model, token_ids, reference_logits)
    if parity.max_absolute_error > max_absolute_error:
        raise GPT2CLIError(
            f"maximum logit error {parity.max_absolute_error} exceeds "
            f"{max_absolute_error}"
        )
    if parity.top1_matches != parity.top1_total:
        raise GPT2CLIError("top-1 parity is incomplete")
    evidence = build_evidence(
        source,
        config,
        state_fingerprint=state_fingerprint,
        parameter_count=model.parameter_count,
        parity=parity,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
    )
    write_evidence(output_path, evidence)
    return evidence_payload(evidence)


def verify_report(path: str | Path, *, max_absolute_error: float) -> dict[str, object]:
    """Verify report integrity and the declared parity gates."""

    if max_absolute_error < 0:
        raise GPT2CLIError("max_absolute_error must be non-negative")
    evidence = load_evidence(path)
    if evidence.max_absolute_error > max_absolute_error:
        raise GPT2CLIError(
            f"reported maximum logit error {evidence.max_absolute_error} exceeds "
            f"{max_absolute_error}"
        )
    if evidence.top1_matches != evidence.top1_total:
        raise GPT2CLIError("reported top-1 parity is incomplete")
    return evidence_payload(evidence)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_parser = commands.add_parser("import", help="import and audit GPT-2")
    import_parser.add_argument("--source", required=True)
    import_parser.add_argument("--artifacts", required=True)
    import_parser.add_argument("--reference", required=True)
    import_parser.add_argument("--output", required=True)
    import_parser.add_argument("--max-absolute-error", type=float, default=1e-3)
    verify_parser = commands.add_parser("verify", help="verify saved evidence")
    verify_parser.add_argument("--report", required=True)
    verify_parser.add_argument("--max-absolute-error", type=float, default=1e-3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed GPT-2 interoperability command."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "import":
            payload = import_checkpoint(
                source_path=arguments.source,
                artifact_directory=arguments.artifacts,
                reference_path=arguments.reference,
                output_path=arguments.output,
                max_absolute_error=arguments.max_absolute_error,
            )
        else:
            payload = verify_report(
                arguments.report,
                max_absolute_error=arguments.max_absolute_error,
            )
    except (GPT2CLIError, GPT2EvidenceError, OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
