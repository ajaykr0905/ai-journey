"""Operational certification checks for the independently rebuilt WaveNet."""

from __future__ import annotations

from dataclasses import dataclass

from .wavenet import WaveNetError
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
