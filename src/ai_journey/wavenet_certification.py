"""Operational certification checks for the independently rebuilt WaveNet."""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256

import torch
from torch.nn import functional as F

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


@dataclass(frozen=True)
class ActivationSnapshot:
    """Detached summary and digest for one primitive forward boundary."""

    name: str
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    digest: str


def _activation_snapshot(name: str, value: torch.Tensor) -> ActivationSnapshot:
    detached = value.detach().cpu().contiguous()
    numeric = detached.double()
    return ActivationSnapshot(
        name=name,
        shape=tuple(detached.shape),
        minimum=float(numeric.min()),
        maximum=float(numeric.max()),
        mean=float(numeric.mean()),
        standard_deviation=float(numeric.std(unbiased=False)),
        digest=sha256(detached.numpy().tobytes()).hexdigest(),
    )


def trace_rebuild_activations(
    model: RebuiltWaveNet, contexts: torch.Tensor
) -> tuple[ActivationSnapshot, ...]:
    """Capture every primitive activation boundary without changing model mode."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, None)
    mode = model.training
    snapshots: list[ActivationSnapshot] = []
    try:
        model.eval()
        with torch.no_grad():
            hidden = model.embedding_weight[contexts]
            snapshots.append(_activation_snapshot("embedding", hidden))
            for index, spec in enumerate(model.plan.stages):
                hidden = model._stage_forward(hidden, index=index, spec=spec)
                snapshots.append(_activation_snapshot(f"stage_{index + 1}", hidden))
            logits = hidden[:, 0, :] @ model.output_weight.transpose(0, 1)
            logits = logits + model.output_bias
            snapshots.append(_activation_snapshot("logits", logits))
    finally:
        model.train(mode)
    return tuple(snapshots)


@dataclass(frozen=True)
class ActivationFinitenessAudit:
    """Finite-value status at every recorded forward boundary."""

    boundaries: int
    nonfinite_boundaries: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.nonfinite_boundaries


def audit_activation_finiteness(
    model: RebuiltWaveNet, contexts: torch.Tensor
) -> ActivationFinitenessAudit:
    """Locate the first and subsequent boundaries affected by nonfinite state."""

    snapshots = trace_rebuild_activations(model, contexts)
    nonfinite = tuple(
        snapshot.name
        for snapshot in snapshots
        if not all(
            math.isfinite(value)
            for value in (
                snapshot.minimum,
                snapshot.maximum,
                snapshot.mean,
                snapshot.standard_deviation,
            )
        )
    )
    return ActivationFinitenessAudit(
        boundaries=len(snapshots), nonfinite_boundaries=nonfinite
    )


@dataclass(frozen=True)
class SaturationMeasurement:
    """Fraction of one tanh stage at or beyond a declared magnitude."""

    stage: str
    elements: int
    saturated_fraction: float
    threshold: float


def measure_activation_saturation(
    model: RebuiltWaveNet, contexts: torch.Tensor, *, threshold: float = 0.95
) -> tuple[SaturationMeasurement, ...]:
    """Measure saturation after each hierarchical tanh activation."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, None)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise WaveNetError("threshold must be finite and in (0, 1]")
    mode = model.training
    measurements: list[SaturationMeasurement] = []
    try:
        model.eval()
        with torch.no_grad():
            hidden = model.embedding_weight[contexts]
            for index, spec in enumerate(model.plan.stages):
                hidden = model._stage_forward(hidden, index=index, spec=spec)
                fraction = float((hidden.abs() >= threshold).double().mean())
                measurements.append(
                    SaturationMeasurement(
                        stage=f"stage_{index + 1}",
                        elements=hidden.numel(),
                        saturated_fraction=fraction,
                        threshold=float(threshold),
                    )
                )
    finally:
        model.train(mode)
    return tuple(measurements)


@dataclass(frozen=True)
class BatchInvarianceAudit:
    """Agreement for one example evaluated alone and inside a batch."""

    batch_size: int
    max_abs_logit_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_abs_logit_error <= self.tolerance


def audit_eval_batch_invariance(
    model: RebuiltWaveNet,
    contexts: torch.Tensor,
    *,
    tolerance: float = 1e-6,
) -> BatchInvarianceAudit:
    """Prove evaluation logits do not depend on neighboring batch examples."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, None)
    if not contexts.shape[0]:
        raise WaveNetError("contexts must contain at least one example")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise WaveNetError("tolerance must be non-negative and finite")
    mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            single_logits, _ = model(contexts[:1])
            batch_logits, _ = model(contexts)
        error = float((single_logits[0] - batch_logits[0]).abs().max())
    finally:
        model.train(mode)
    return BatchInvarianceAudit(
        batch_size=int(contexts.shape[0]),
        max_abs_logit_error=error,
        tolerance=float(tolerance),
    )


@dataclass(frozen=True)
class TopKParityAudit:
    """Ranked prediction agreement between reference and rebuild."""

    examples: int
    k: int
    mismatched_examples: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatched_examples


def audit_top_k_parity(
    reference: HierarchicalLanguageModel,
    rebuilt: RebuiltWaveNet,
    contexts: torch.Tensor,
    *,
    k: int = 3,
) -> TopKParityAudit:
    """Compare ordered top-k token predictions for every supplied example."""

    if not isinstance(reference, HierarchicalLanguageModel):
        raise TypeError("reference must be HierarchicalLanguageModel")
    if not isinstance(rebuilt, RebuiltWaveNet):
        raise TypeError("rebuilt must be RebuiltWaveNet")
    if reference.config != rebuilt.config:
        raise WaveNetError("reference and rebuild configurations must match")
    rebuilt._validate_inputs(contexts, None)
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 1 <= k <= rebuilt.config.vocab_size:
        raise WaveNetError("k must be within the vocabulary")
    reference_mode, rebuilt_mode = reference.training, rebuilt.training
    try:
        reference.eval()
        rebuilt.eval()
        with torch.no_grad():
            reference_logits, _ = reference(contexts)
            rebuilt_logits, _ = rebuilt(contexts)
            reference_top = reference_logits.topk(k, dim=-1).indices
            rebuilt_top = rebuilt_logits.topk(k, dim=-1).indices
        mismatches = tuple(
            index
            for index in range(contexts.shape[0])
            if not torch.equal(reference_top[index], rebuilt_top[index])
        )
    finally:
        reference.train(reference_mode)
        rebuilt.train(rebuilt_mode)
    return TopKParityAudit(
        examples=int(contexts.shape[0]), k=k, mismatched_examples=mismatches
    )


@dataclass(frozen=True)
class PerExampleLossParityAudit:
    """Example-level loss agreement hidden by a matching batch mean."""

    examples: int
    max_abs_error: float
    mismatched_examples: tuple[int, ...]
    tolerance: float

    @property
    def passed(self) -> bool:
        return not self.mismatched_examples


def audit_per_example_loss_parity(
    reference: HierarchicalLanguageModel,
    rebuilt: RebuiltWaveNet,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    *,
    tolerance: float = 1e-6,
) -> PerExampleLossParityAudit:
    """Compare unreduced cross-entropy rather than only aggregate loss."""

    if not isinstance(reference, HierarchicalLanguageModel):
        raise TypeError("reference must be HierarchicalLanguageModel")
    if not isinstance(rebuilt, RebuiltWaveNet):
        raise TypeError("rebuilt must be RebuiltWaveNet")
    if reference.config != rebuilt.config:
        raise WaveNetError("reference and rebuild configurations must match")
    rebuilt._validate_inputs(contexts, targets)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise WaveNetError("tolerance must be non-negative and finite")
    reference_mode, rebuilt_mode = reference.training, rebuilt.training
    try:
        reference.eval()
        rebuilt.eval()
        with torch.no_grad():
            reference_logits, _ = reference(contexts)
            rebuilt_logits, _ = rebuilt(contexts)
            errors = (
                F.cross_entropy(reference_logits, targets, reduction="none")
                - F.cross_entropy(rebuilt_logits, targets, reduction="none")
            ).abs()
        mismatches = tuple(
            index for index, error in enumerate(errors) if float(error) > tolerance
        )
        maximum = float(errors.max()) if errors.numel() else 0.0
    finally:
        reference.train(reference_mode)
        rebuilt.train(rebuilt_mode)
    return PerExampleLossParityAudit(
        examples=int(targets.numel()),
        max_abs_error=maximum,
        mismatched_examples=mismatches,
        tolerance=float(tolerance),
    )
