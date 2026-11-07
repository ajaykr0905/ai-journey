"""Bounded admission checks for Project 1 training runs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .gpt2_training import GPT2TrainingSpec
from .project1_contract import Project1RunContract, Project1RunError
from .training_models import expected_training_parameters


@dataclass(frozen=True)
class Project1Preflight:
    """Inspectable workload admitted before model construction."""

    parameter_count: int
    source_bytes: int
    optimizer_steps: int
    tokens_per_step: int
    total_target_tokens: int


def admit_training_run(
    contract: Project1RunContract,
    spec: GPT2TrainingSpec,
    source: bytes,
) -> Project1Preflight:
    """Reject identity or resource mismatches before allocating model tensors."""

    if not isinstance(contract, Project1RunContract):
        raise TypeError("contract must be Project1RunContract")
    if not isinstance(spec, GPT2TrainingSpec):
        raise TypeError("spec must be GPT2TrainingSpec")
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    if contract.spec_sha256 != spec.fingerprint():
        raise Project1RunError("contract does not match the training spec")
    source_digest = sha256(source).hexdigest()
    if (
        contract.source_sha256 != spec.source.sha256
        or source_digest != spec.source.sha256
    ):
        raise Project1RunError("contract, spec, and source digests differ")
    if not 2 <= len(source) <= contract.max_source_bytes:
        raise Project1RunError("source exceeds the admitted byte range")
    parameters = expected_training_parameters(spec.model)
    if parameters > contract.max_parameters:
        raise Project1RunError("model exceeds the admitted parameter count")
    training = spec.training
    return Project1Preflight(
        parameter_count=parameters,
        source_bytes=len(source),
        optimizer_steps=training.optimizer_steps,
        tokens_per_step=training.tokens_per_step,
        total_target_tokens=training.total_target_tokens,
    )
