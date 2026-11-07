"""Bounded admission checks for Project 1 training runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

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
    parameter_bytes: int
    gradient_bytes: int
    adamw_state_bytes: int
    persistent_training_bytes: int


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
    parameter_bytes = parameters * 4
    gradient_bytes = parameters * 4
    adamw_state_bytes = parameters * 8
    return Project1Preflight(
        parameter_count=parameters,
        source_bytes=len(source),
        optimizer_steps=training.optimizer_steps,
        tokens_per_step=training.tokens_per_step,
        total_target_tokens=training.total_target_tokens,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        adamw_state_bytes=adamw_state_bytes,
        persistent_training_bytes=(
            parameter_bytes + gradient_bytes + adamw_state_bytes
        ),
    )


def seal_preflight(
    contract: Project1RunContract, preflight: Project1Preflight
) -> dict[str, Any]:
    """Seal analytical admission evidence with an explicit claim boundary."""

    if not isinstance(contract, Project1RunContract):
        raise TypeError("contract must be Project1RunContract")
    if not isinstance(preflight, Project1Preflight):
        raise TypeError("preflight must be Project1Preflight")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_sha256": contract.fingerprint(),
        "admission": asdict(preflight),
        "evidence_boundary": (
            "Analytical CPU FP32 parameter, gradient, and AdamW-state bytes only; "
            "excludes activations, allocator overhead, runtime, GPU, and throughput."
        ),
    }
    payload["preflight_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def verify_preflight(
    payload: Mapping[str, Any], contract: Project1RunContract
) -> Project1Preflight:
    """Recompute the preflight seal and exact byte-accounting relationships."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "contract_sha256",
        "admission",
        "evidence_boundary",
        "preflight_sha256",
    }:
        raise Project1RunError("preflight fields do not match schema")
    if (
        payload["schema_version"] != 1
        or payload["contract_sha256"] != contract.fingerprint()
    ):
        raise Project1RunError("preflight does not match its run contract")
    unsigned = {
        key: value for key, value in payload.items() if key != "preflight_sha256"
    }
    expected = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if payload["preflight_sha256"] != expected:
        raise Project1RunError("preflight digest does not match")
    try:
        preflight = Project1Preflight(**payload["admission"])
    except (TypeError, ValueError) as exc:
        raise Project1RunError("preflight admission is invalid") from exc
    if (
        preflight.parameter_bytes != preflight.parameter_count * 4
        or preflight.gradient_bytes != preflight.parameter_count * 4
        or preflight.adamw_state_bytes != preflight.parameter_count * 8
        or preflight.persistent_training_bytes != preflight.parameter_count * 16
        or preflight.total_target_tokens
        != preflight.optimizer_steps * preflight.tokens_per_step
    ):
        raise Project1RunError("preflight accounting relationships do not match")
    return preflight
