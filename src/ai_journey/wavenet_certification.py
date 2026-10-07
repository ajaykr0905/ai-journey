"""Operational certification checks for the independently rebuilt WaveNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .wavenet import HierarchicalLanguageModel, WaveNetError
from .wavenet_rebuild import RebuiltWaveNet


@dataclass(frozen=True)
class ParameterInventoryEntry:
    """One registered parameter's public structural metadata."""

    name: str
    shape: tuple[int, ...]
    elements: int
    dtype: str
    requires_gradient: bool


def parameter_inventory(model: RebuiltWaveNet) -> tuple[ParameterInventoryEntry, ...]:
    """Return a stable, complete inventory of registered rebuild parameters."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    entries = tuple(
        ParameterInventoryEntry(
            name=name,
            shape=tuple(parameter.shape),
            elements=parameter.numel(),
            dtype=str(parameter.dtype),
            requires_gradient=parameter.requires_grad,
        )
        for name, parameter in model.named_parameters()
    )
    if not entries:
        raise WaveNetError("rebuild has no registered parameters")
    return entries


@dataclass(frozen=True)
class ParameterManifestAudit:
    """Agreement between the compiled plan and registered parameter manifest."""

    tensors: int
    registered_elements: int
    planned_elements: int
    shape_mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            not self.shape_mismatches
            and self.registered_elements == self.planned_elements
        )


def audit_parameter_manifest(model: RebuiltWaveNet) -> ParameterManifestAudit:
    """Prove registered shapes and element counts match the compiled plan."""

    inventory = parameter_inventory(model)
    manifest = model.parameter_manifest()
    mismatches = tuple(
        entry.name for entry in inventory if manifest.get(entry.name) != entry.shape
    )
    return ParameterManifestAudit(
        tensors=len(inventory),
        registered_elements=sum(entry.elements for entry in inventory),
        planned_elements=model.plan.parameter_count,
        shape_mismatches=mismatches,
    )


@dataclass(frozen=True)
class StorageIndependenceAudit:
    """Evidence that reference and rebuild parameters do not alias storage."""

    reference_tensors: int
    rebuild_tensors: int
    shared_storage_pairs: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.shared_storage_pairs


def audit_storage_independence(
    reference: HierarchicalLanguageModel, rebuilt: RebuiltWaveNet
) -> StorageIndependenceAudit:
    """Reject a supposed independent rebuild that shares parameter storage."""

    if not isinstance(reference, HierarchicalLanguageModel):
        raise TypeError("reference must be HierarchicalLanguageModel")
    if not isinstance(rebuilt, RebuiltWaveNet):
        raise TypeError("rebuilt must be RebuiltWaveNet")
    if reference.config != rebuilt.config:
        raise WaveNetError("reference and rebuild configurations must match")
    reference_parameters = tuple(reference.named_parameters())
    rebuilt_parameters = tuple(rebuilt.named_parameters())
    shared = tuple(
        f"{reference_name}:{rebuilt_name}"
        for reference_name, reference_parameter in reference_parameters
        for rebuilt_name, rebuilt_parameter in rebuilt_parameters
        if reference_parameter.untyped_storage().data_ptr()
        == rebuilt_parameter.untyped_storage().data_ptr()
    )
    return StorageIndependenceAudit(
        reference_tensors=len(reference_parameters),
        rebuild_tensors=len(rebuilt_parameters),
        shared_storage_pairs=shared,
    )


@dataclass(frozen=True)
class ParameterFinitenessAudit:
    """Finite-value coverage across every rebuild parameter tensor."""

    tensors: int
    elements: int
    nonfinite_parameters: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.nonfinite_parameters


def audit_parameter_finiteness(model: RebuiltWaveNet) -> ParameterFinitenessAudit:
    """Identify any registered parameter containing NaN or infinity."""

    inventory = parameter_inventory(model)
    parameters = dict(model.named_parameters())
    nonfinite = tuple(
        entry.name
        for entry in inventory
        if not bool(torch.isfinite(parameters[entry.name]).all())
    )
    return ParameterFinitenessAudit(
        tensors=len(inventory),
        elements=sum(entry.elements for entry in inventory),
        nonfinite_parameters=nonfinite,
    )


@dataclass(frozen=True)
class ParameterStatistics:
    """Scale diagnostics for one registered rebuild tensor."""

    name: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    l2_norm: float


def parameter_statistics(model: RebuiltWaveNet) -> tuple[ParameterStatistics, ...]:
    """Summarize every parameter without retaining tensor references."""

    inventory = parameter_inventory(model)
    parameters = dict(model.named_parameters())
    if not audit_parameter_finiteness(model).passed:
        raise WaveNetError("parameter statistics require finite model state")
    summaries: list[ParameterStatistics] = []
    with torch.no_grad():
        for entry in inventory:
            values = parameters[entry.name].detach().double()
            summaries.append(
                ParameterStatistics(
                    name=entry.name,
                    minimum=float(values.min()),
                    maximum=float(values.max()),
                    mean=float(values.mean()),
                    standard_deviation=float(values.std(unbiased=False)),
                    l2_norm=float(torch.linalg.vector_norm(values)),
                )
            )
    return tuple(summaries)
