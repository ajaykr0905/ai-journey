"""Operational certification checks for the independently rebuilt WaveNet."""

from __future__ import annotations

import copy
import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .wavenet import (
    HierarchicalLanguageModel,
    WaveNetDataset,
    WaveNetError,
    WaveNetTrainingStep,
)
from .wavenet_rebuild import (
    RebuiltWaveNet,
    rebuild_model_fingerprint,
    reference_parameter_pairs,
    sample_rebuild,
)


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


@dataclass(frozen=True)
class ProbabilitySimplexAudit:
    """Probability normalization and bounds for rebuild predictions."""

    examples: int
    max_row_sum_error: float
    minimum_probability: float
    maximum_probability: float
    all_finite: bool
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.all_finite
            and self.minimum_probability >= 0
            and self.maximum_probability <= 1
            and self.max_row_sum_error <= self.tolerance
        )


def audit_probability_simplex(
    model: RebuiltWaveNet,
    contexts: torch.Tensor,
    *,
    tolerance: float = 1e-6,
) -> ProbabilitySimplexAudit:
    """Verify softmax outputs form one finite simplex row per example."""

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
            logits, _ = model(contexts)
            probabilities = torch.softmax(logits.double(), dim=-1)
            row_errors = (probabilities.sum(dim=-1) - 1).abs()
        return ProbabilitySimplexAudit(
            examples=int(contexts.shape[0]),
            max_row_sum_error=float(row_errors.max()),
            minimum_probability=float(probabilities.min()),
            maximum_probability=float(probabilities.max()),
            all_finite=bool(torch.isfinite(probabilities).all()),
            tolerance=float(tolerance),
        )
    finally:
        model.train(mode)


@dataclass(frozen=True)
class GradientStatistics:
    """Magnitude and finite-value evidence for one parameter gradient."""

    name: str
    l2_norm: float
    max_abs_value: float
    finite: bool


def measure_gradient_statistics(
    model: RebuiltWaveNet, contexts: torch.Tensor, targets: torch.Tensor
) -> tuple[GradientStatistics, ...]:
    """Measure every loss gradient while restoring caller gradient state."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, targets)
    mode = model.training
    parameters = tuple(model.named_parameters())
    previous_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in parameters
    }
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        loss.backward()
        statistics = tuple(
            GradientStatistics(
                name=name,
                l2_norm=float(torch.linalg.vector_norm(parameter.grad.detach())),
                max_abs_value=float(parameter.grad.detach().abs().max()),
                finite=bool(torch.isfinite(parameter.grad).all()),
            )
            for name, parameter in parameters
            if parameter.grad is not None
        )
    finally:
        for name, parameter in parameters:
            previous = previous_gradients[name]
            parameter.grad = None if previous is None else previous
        model.train(mode)
    return statistics


@dataclass(frozen=True)
class GradientCosineMeasurement:
    """Directional agreement for one mapped parameter gradient."""

    name: str
    cosine_similarity: float


def measure_gradient_cosines(
    reference: HierarchicalLanguageModel,
    rebuilt: RebuiltWaveNet,
    contexts: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[GradientCosineMeasurement, ...]:
    """Compare gradient direction for every mapped parameter pair."""

    pairs = reference_parameter_pairs(reference, rebuilt)
    rebuilt._validate_inputs(contexts, targets)
    reference_parameters = tuple(reference.named_parameters())
    rebuilt_parameters = tuple(rebuilt.named_parameters())
    previous = {
        id(parameter): None
        if parameter.grad is None
        else parameter.grad.detach().clone()
        for _, parameter in reference_parameters + rebuilt_parameters
    }
    reference_mode, rebuilt_mode = reference.training, rebuilt.training
    try:
        reference.eval()
        rebuilt.eval()
        reference.zero_grad(set_to_none=True)
        rebuilt.zero_grad(set_to_none=True)
        _, reference_loss = reference(contexts, targets)
        _, rebuilt_loss = rebuilt(contexts, targets)
        assert reference_loss is not None and rebuilt_loss is not None
        reference_loss.backward()
        rebuilt_loss.backward()
        measurements: list[GradientCosineMeasurement] = []
        for name, reference_parameter, rebuilt_parameter in pairs:
            assert reference_parameter.grad is not None
            assert rebuilt_parameter.grad is not None
            left = reference_parameter.grad.detach().reshape(-1).double()
            right = rebuilt_parameter.grad.detach().reshape(-1).double()
            left_norm = torch.linalg.vector_norm(left)
            right_norm = torch.linalg.vector_norm(right)
            if float(left_norm) == 0 and float(right_norm) == 0:
                cosine = 1.0
            elif float(left_norm) == 0 or float(right_norm) == 0:
                cosine = 0.0
            else:
                cosine = float(torch.dot(left, right) / (left_norm * right_norm))
            measurements.append(
                GradientCosineMeasurement(name=name, cosine_similarity=cosine)
            )
    finally:
        for _, parameter in reference_parameters + rebuilt_parameters:
            stored = previous[id(parameter)]
            parameter.grad = None if stored is None else stored
        reference.train(reference_mode)
        rebuilt.train(rebuilt_mode)
    return tuple(measurements)


@dataclass(frozen=True)
class GradientCoverageAudit:
    """Coverage of registered parameters by one representative loss graph."""

    registered_parameters: int
    parameters_with_gradients: int
    missing_gradients: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing_gradients


def audit_gradient_coverage(
    model: RebuiltWaveNet, contexts: torch.Tensor, targets: torch.Tensor
) -> GradientCoverageAudit:
    """Detect registered tensors disconnected from the training loss."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, targets)
    parameters = tuple(model.named_parameters())
    previous = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in parameters
    }
    mode = model.training
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        loss.backward()
        missing = tuple(
            name for name, parameter in parameters if parameter.grad is None
        )
    finally:
        for name, parameter in parameters:
            stored = previous[name]
            parameter.grad = None if stored is None else stored
        model.train(mode)
    return GradientCoverageAudit(
        registered_parameters=len(parameters),
        parameters_with_gradients=len(parameters) - len(missing),
        missing_gradients=missing,
    )


@dataclass(frozen=True)
class GradientClippingAudit:
    """Observed global gradient norm before and after clipping."""

    pre_clip_norm: float
    post_clip_norm: float
    max_norm: float
    clipped: bool

    @property
    def passed(self) -> bool:
        return self.post_clip_norm <= self.max_norm + 1e-6


def audit_gradient_clipping(
    model: RebuiltWaveNet,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_norm: float,
) -> GradientClippingAudit:
    """Exercise PyTorch clipping on rebuild gradients without retaining changes."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, targets)
    if (
        isinstance(max_norm, bool)
        or not isinstance(max_norm, (int, float))
        or not math.isfinite(max_norm)
        or max_norm <= 0
    ):
        raise WaveNetError("max_norm must be positive and finite")
    parameters = tuple(model.parameters())
    previous = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    mode = model.training
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(parameters, float(max_norm)))
        post_clip = float(
            torch.sqrt(
                sum(
                    parameter.grad.detach().double().square().sum()
                    for parameter in parameters
                    if parameter.grad is not None
                )
            )
        )
    finally:
        for parameter, stored in zip(parameters, previous, strict=True):
            parameter.grad = None if stored is None else stored
        model.train(mode)
    return GradientClippingAudit(
        pre_clip_norm=pre_clip,
        post_clip_norm=post_clip,
        max_norm=float(max_norm),
        clipped=pre_clip > max_norm,
    )


@dataclass(frozen=True)
class OptimizerStepParityAudit:
    """Parameter agreement after one matched optimizer transition."""

    parameter_tensors: int
    max_abs_parameter_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_abs_parameter_error <= self.tolerance


def audit_optimizer_step_parity(
    reference: HierarchicalLanguageModel,
    rebuilt: RebuiltWaveNet,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    *,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
) -> OptimizerStepParityAudit:
    """Compare one SGD update and restore both caller model states."""

    pairs = reference_parameter_pairs(reference, rebuilt)
    rebuilt._validate_inputs(contexts, targets)
    for name, value, positive in (
        ("learning_rate", learning_rate, True),
        ("tolerance", tolerance, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise WaveNetError(f"{name} must be {qualifier} and finite")
    reference_state = copy.deepcopy(reference.state_dict())
    rebuilt_state = copy.deepcopy(rebuilt.state_dict())
    reference_mode, rebuilt_mode = reference.training, rebuilt.training
    try:
        reference.eval()
        rebuilt.eval()
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=float(learning_rate)
        )
        rebuilt_optimizer = torch.optim.SGD(
            rebuilt.parameters(), lr=float(learning_rate)
        )
        for model, optimizer in (
            (reference, reference_optimizer),
            (rebuilt, rebuilt_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(contexts, targets)
            assert loss is not None
            loss.backward()
            optimizer.step()
        maximum = max(
            float(
                (reference_parameter.detach() - rebuilt_parameter.detach()).abs().max()
            )
            for _, reference_parameter, rebuilt_parameter in pairs
        )
    finally:
        reference.load_state_dict(reference_state)
        rebuilt.load_state_dict(rebuilt_state)
        reference.zero_grad(set_to_none=True)
        rebuilt.zero_grad(set_to_none=True)
        reference.train(reference_mode)
        rebuilt.train(rebuilt_mode)
    return OptimizerStepParityAudit(
        parameter_tensors=len(pairs),
        max_abs_parameter_error=maximum,
        tolerance=float(tolerance),
    )


@dataclass(frozen=True)
class GradientResetAudit:
    """Evidence that gradient clearing releases every accumulated gradient."""

    populated_before_reset: int
    populated_after_reset: int
    parameters_unchanged: bool

    @property
    def passed(self) -> bool:
        return (
            self.populated_before_reset > 0
            and self.populated_after_reset == 0
            and self.parameters_unchanged
        )


def audit_gradient_reset(
    model: RebuiltWaveNet, contexts: torch.Tensor, targets: torch.Tensor
) -> GradientResetAudit:
    """Backpropagate, clear with set-to-none, and preserve caller state."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, targets)
    parameters = tuple(model.named_parameters())
    previous = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in parameters
    }
    mode = model.training
    fingerprint = rebuild_model_fingerprint(model)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        loss.backward()
        before = sum(parameter.grad is not None for _, parameter in parameters)
        model.zero_grad(set_to_none=True)
        after = sum(parameter.grad is not None for _, parameter in parameters)
        unchanged = rebuild_model_fingerprint(model) == fingerprint
    finally:
        for name, parameter in parameters:
            stored = previous[name]
            parameter.grad = None if stored is None else stored
        model.train(mode)
    return GradientResetAudit(
        populated_before_reset=before,
        populated_after_reset=after,
        parameters_unchanged=unchanged,
    )


def rebuild_batch_fingerprint(
    model: RebuiltWaveNet, contexts: torch.Tensor, targets: torch.Tensor
) -> str:
    """Bind one validated input batch to stable shapes, dtypes, and values."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, targets)
    digest = sha256(model.plan.fingerprint().encode())
    for name, tensor in (("contexts", contexts), ("targets", targets)):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RepeatInferenceAudit:
    """Determinism evidence across repeated evaluation forwards."""

    repeats: int
    max_abs_logit_error: float
    distinct_digests: int

    @property
    def passed(self) -> bool:
        return self.max_abs_logit_error == 0 and self.distinct_digests == 1


def audit_repeat_inference(
    model: RebuiltWaveNet, contexts: torch.Tensor, *, repeats: int = 3
) -> RepeatInferenceAudit:
    """Require bitwise-stable logits across repeated evaluation calls."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, None)
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError("repeats must be an integer")
    if repeats < 2:
        raise WaveNetError("repeats must be at least two")
    mode = model.training
    outputs: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(repeats):
                logits, _ = model(contexts)
                outputs.append(logits.detach().cpu().clone())
    finally:
        model.train(mode)
    baseline = outputs[0]
    maximum = max(float((baseline - output).abs().max()) for output in outputs[1:])
    digests = {
        sha256(output.contiguous().numpy().tobytes()).hexdigest() for output in outputs
    }
    return RepeatInferenceAudit(
        repeats=repeats,
        max_abs_logit_error=maximum,
        distinct_digests=len(digests),
    )


@dataclass(frozen=True)
class InferenceRngAudit:
    """Caller CPU RNG preservation across deterministic evaluation."""

    before_digest: str
    after_digest: str

    @property
    def passed(self) -> bool:
        return self.before_digest == self.after_digest


def audit_inference_rng_isolation(
    model: RebuiltWaveNet, contexts: torch.Tensor
) -> InferenceRngAudit:
    """Prove evaluation does not consume the caller's random stream."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, None)
    before = torch.get_rng_state().clone()
    mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(contexts)
    finally:
        model.train(mode)
    after = torch.get_rng_state().clone()
    return InferenceRngAudit(
        before_digest=sha256(before.numpy().tobytes()).hexdigest(),
        after_digest=sha256(after.numpy().tobytes()).hexdigest(),
    )


@dataclass(frozen=True)
class InputImmutabilityAudit:
    """Mutation status for caller-owned contexts and targets."""

    contexts_unchanged: bool
    targets_unchanged: bool

    @property
    def passed(self) -> bool:
        return self.contexts_unchanged and self.targets_unchanged


def audit_input_immutability(
    model: RebuiltWaveNet, contexts: torch.Tensor, targets: torch.Tensor
) -> InputImmutabilityAudit:
    """Run forward/backward and prove caller input tensors are untouched."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, targets)
    original_contexts = contexts.clone()
    original_targets = targets.clone()
    parameters = tuple(model.named_parameters())
    previous = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in parameters
    }
    try:
        model.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        loss.backward()
    finally:
        for name, parameter in parameters:
            stored = previous[name]
            parameter.grad = None if stored is None else stored
    return InputImmutabilityAudit(
        contexts_unchanged=torch.equal(contexts, original_contexts),
        targets_unchanged=torch.equal(targets, original_targets),
    )


@dataclass(frozen=True)
class DatasetTokenCoverageAudit:
    """Vocabulary coverage observed across dataset inputs and targets."""

    vocabulary_size: int
    observed_token_ids: tuple[int, ...]
    missing_token_ids: tuple[int, ...]
    boundary_token_seen: bool

    @property
    def coverage_fraction(self) -> float:
        return len(self.observed_token_ids) / self.vocabulary_size


def audit_dataset_token_coverage(
    model: RebuiltWaveNet, dataset: WaveNetDataset
) -> DatasetTokenCoverageAudit:
    """Report unexercised vocabulary entries before model certification."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    if not isinstance(dataset, WaveNetDataset):
        raise TypeError("dataset must be WaveNetDataset")
    if dataset.context_size != model.config.context_size:
        raise WaveNetError("dataset context_size does not match the rebuild")
    if dataset.vocab_size != model.config.vocab_size:
        raise WaveNetError("dataset vocabulary does not match the rebuild")
    observed = tuple(
        sorted(
            set(dataset.contexts.reshape(-1).tolist()) | set(dataset.targets.tolist())
        )
    )
    missing = tuple(
        index for index in range(dataset.vocab_size) if index not in observed
    )
    return DatasetTokenCoverageAudit(
        vocabulary_size=dataset.vocab_size,
        observed_token_ids=observed,
        missing_token_ids=missing,
        boundary_token_seen=0 in observed,
    )


@dataclass(frozen=True)
class SampleReproducibilityAudit:
    """Equality of two isolated sampling runs with identical controls."""

    seed: int
    text: str
    token_ids: tuple[int, ...]
    terminated: bool
    identical: bool

    @property
    def passed(self) -> bool:
        return self.identical


def audit_sample_reproducibility(
    model: RebuiltWaveNet,
    vocabulary_tokens: tuple[str, ...],
    *,
    seed: int,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> SampleReproducibilityAudit:
    """Repeat a seeded bounded sample and compare the complete result."""

    first = sample_rebuild(
        model,
        vocabulary_tokens,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    second = sample_rebuild(
        model,
        vocabulary_tokens,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return SampleReproducibilityAudit(
        seed=seed,
        text=first.text,
        token_ids=first.token_ids,
        terminated=first.terminated,
        identical=first == second,
    )


@dataclass(frozen=True)
class SampleTerminationAudit:
    """Termination and length behavior across a fixed seed panel."""

    samples: int
    terminated_samples: int
    termination_fraction: float
    maximum_observed_tokens: int
    max_new_tokens: int
    out_of_bounds_samples: int

    @property
    def passed(self) -> bool:
        return self.out_of_bounds_samples == 0


def audit_sample_termination(
    model: RebuiltWaveNet,
    vocabulary_tokens: tuple[str, ...],
    *,
    seeds: tuple[int, ...],
    max_new_tokens: int = 20,
) -> SampleTerminationAudit:
    """Measure boundary-token termination without allowing unbounded samples."""

    if not isinstance(seeds, tuple) or not seeds:
        raise TypeError("seeds must be a non-empty tuple of integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds must contain only integers")
    samples = tuple(
        sample_rebuild(
            model,
            vocabulary_tokens,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        for seed in seeds
    )
    terminated = sum(sample.terminated for sample in samples)
    lengths = tuple(len(sample.token_ids) for sample in samples)
    out_of_bounds = sum(length > max_new_tokens for length in lengths)
    return SampleTerminationAudit(
        samples=len(samples),
        terminated_samples=terminated,
        termination_fraction=terminated / len(samples),
        maximum_observed_tokens=max(lengths),
        max_new_tokens=max_new_tokens,
        out_of_bounds_samples=out_of_bounds,
    )


@dataclass(frozen=True)
class TrainingTraceAudit:
    """Continuity and finite-value status for an optimization trace."""

    expected_steps: int
    observed_steps: int
    missing_or_reordered_steps: tuple[int, ...]
    nonfinite_steps: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return (
            self.observed_steps == self.expected_steps
            and not self.missing_or_reordered_steps
            and not self.nonfinite_steps
        )


def audit_training_trace(
    trace: tuple[WaveNetTrainingStep, ...],
    *,
    expected_start: int,
    expected_steps: int,
) -> TrainingTraceAudit:
    """Reject incomplete, reordered, or nonfinite rebuild training evidence."""

    if not isinstance(trace, tuple) or any(
        not isinstance(item, WaveNetTrainingStep) for item in trace
    ):
        raise TypeError("trace must be a tuple of WaveNetTrainingStep values")
    for name, value in (
        ("expected_start", expected_start),
        ("expected_steps", expected_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if expected_start < 0 or expected_steps <= 0:
        raise WaveNetError("expected_start and expected_steps are out of range")
    expected = tuple(range(expected_start, expected_start + expected_steps))
    observed = tuple(item.step for item in trace)
    ordering_errors = tuple(
        expected_step
        for index, expected_step in enumerate(expected)
        if index >= len(observed) or observed[index] != expected_step
    )
    nonfinite = tuple(
        item.step
        for item in trace
        if not math.isfinite(item.loss)
        or not math.isfinite(item.gradient_norm)
        or item.gradient_norm < 0
    )
    return TrainingTraceAudit(
        expected_steps=expected_steps,
        observed_steps=len(trace),
        missing_or_reordered_steps=ordering_errors,
        nonfinite_steps=nonfinite,
    )


@dataclass(frozen=True)
class ParameterDelta:
    """Magnitude of one parameter change between two rebuild states."""

    name: str
    l2_delta: float
    max_abs_delta: float
    changed: bool


def measure_parameter_deltas(
    baseline: RebuiltWaveNet, candidate: RebuiltWaveNet
) -> tuple[ParameterDelta, ...]:
    """Compare compatible rebuild states without retaining tensor aliases."""

    if not isinstance(baseline, RebuiltWaveNet) or not isinstance(
        candidate, RebuiltWaveNet
    ):
        raise TypeError("baseline and candidate must be RebuiltWaveNet")
    if baseline.config != candidate.config:
        raise WaveNetError("baseline and candidate configurations must match")
    baseline_parameters = dict(baseline.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if baseline_parameters.keys() != candidate_parameters.keys():
        raise WaveNetError("baseline and candidate parameter manifests must match")
    measurements: list[ParameterDelta] = []
    with torch.no_grad():
        for name, baseline_parameter in baseline_parameters.items():
            difference = (
                candidate_parameters[name].detach().double()
                - baseline_parameter.detach().double()
            )
            maximum = float(difference.abs().max())
            measurements.append(
                ParameterDelta(
                    name=name,
                    l2_delta=float(torch.linalg.vector_norm(difference)),
                    max_abs_delta=maximum,
                    changed=maximum > 0,
                )
            )
    return tuple(measurements)


@dataclass(frozen=True)
class ModelFootprint:
    """Exact persistent tensor storage owned by the rebuild module."""

    parameter_bytes: int
    buffer_bytes: int
    total_bytes: int
    parameter_elements: int


def measure_model_footprint(model: RebuiltWaveNet) -> ModelFootprint:
    """Count parameter and registered-buffer storage without estimating runtime memory."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return ModelFootprint(
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
        total_bytes=parameter_bytes + buffer_bytes,
        parameter_elements=model.parameter_count,
    )


@dataclass(frozen=True)
class InferenceBenchmark:
    """Local CPU timing distribution for one fixed inference batch."""

    batch_size: int
    repeats: int
    median_seconds: float
    minimum_seconds: float
    median_examples_per_second: float


def benchmark_rebuild_inference(
    model: RebuiltWaveNet,
    contexts: torch.Tensor,
    *,
    warmups: int = 2,
    repeats: int = 5,
) -> InferenceBenchmark:
    """Time bounded CPU inference without claiming cross-machine comparability."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, None)
    if not contexts.shape[0]:
        raise WaveNetError("contexts must contain at least one example")
    for name, value, minimum in (("warmups", warmups, 0), ("repeats", repeats, 1)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise WaveNetError(f"{name} must be at least {minimum}")
    mode = model.training
    timings: list[float] = []
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(warmups):
                model(contexts)
            for _ in range(repeats):
                started = time.perf_counter()
                model(contexts)
                timings.append(time.perf_counter() - started)
    finally:
        model.train(mode)
    median = statistics.median(timings)
    return InferenceBenchmark(
        batch_size=int(contexts.shape[0]),
        repeats=repeats,
        median_seconds=median,
        minimum_seconds=min(timings),
        median_examples_per_second=int(contexts.shape[0]) / median,
    )


@dataclass(frozen=True)
class RebuildCertificationResult:
    """Composed structural, numerical, and reproducibility certification."""

    model_fingerprint: str
    batch_fingerprint: str
    parameter_manifest: ParameterManifestAudit
    storage_independence: StorageIndependenceAudit
    parameter_finiteness: ParameterFinitenessAudit
    activation_finiteness: ActivationFinitenessAudit
    batch_invariance: BatchInvarianceAudit
    top_k_parity: TopKParityAudit
    per_example_loss_parity: PerExampleLossParityAudit
    probability_simplex: ProbabilitySimplexAudit
    gradient_coverage: GradientCoverageAudit
    optimizer_step_parity: OptimizerStepParityAudit
    repeat_inference: RepeatInferenceAudit
    inference_rng: InferenceRngAudit
    input_immutability: InputImmutabilityAudit
    footprint: ModelFootprint

    @property
    def failed_gates(self) -> tuple[str, ...]:
        gates = {
            "parameter_manifest": self.parameter_manifest.passed,
            "storage_independence": self.storage_independence.passed,
            "parameter_finiteness": self.parameter_finiteness.passed,
            "activation_finiteness": self.activation_finiteness.passed,
            "batch_invariance": self.batch_invariance.passed,
            "top_k_parity": self.top_k_parity.passed,
            "per_example_loss_parity": self.per_example_loss_parity.passed,
            "probability_simplex": self.probability_simplex.passed,
            "gradient_coverage": self.gradient_coverage.passed,
            "optimizer_step_parity": self.optimizer_step_parity.passed,
            "repeat_inference": self.repeat_inference.passed,
            "inference_rng": self.inference_rng.passed,
            "input_immutability": self.input_immutability.passed,
        }
        return tuple(name for name, passed in gates.items() if not passed)

    @property
    def passed(self) -> bool:
        return not self.failed_gates


def certify_rebuild(
    reference: HierarchicalLanguageModel,
    rebuilt: RebuiltWaveNet,
    contexts: torch.Tensor,
    targets: torch.Tensor,
) -> RebuildCertificationResult:
    """Run the complete bounded reliability gate for a mapped rebuild."""

    return RebuildCertificationResult(
        model_fingerprint=rebuild_model_fingerprint(rebuilt),
        batch_fingerprint=rebuild_batch_fingerprint(rebuilt, contexts, targets),
        parameter_manifest=audit_parameter_manifest(rebuilt),
        storage_independence=audit_storage_independence(reference, rebuilt),
        parameter_finiteness=audit_parameter_finiteness(rebuilt),
        activation_finiteness=audit_activation_finiteness(rebuilt, contexts),
        batch_invariance=audit_eval_batch_invariance(rebuilt, contexts),
        top_k_parity=audit_top_k_parity(reference, rebuilt, contexts),
        per_example_loss_parity=audit_per_example_loss_parity(
            reference, rebuilt, contexts, targets
        ),
        probability_simplex=audit_probability_simplex(rebuilt, contexts),
        gradient_coverage=audit_gradient_coverage(rebuilt, contexts, targets),
        optimizer_step_parity=audit_optimizer_step_parity(
            reference, rebuilt, contexts, targets
        ),
        repeat_inference=audit_repeat_inference(rebuilt, contexts),
        inference_rng=audit_inference_rng_isolation(rebuilt, contexts),
        input_immutability=audit_input_immutability(rebuilt, contexts, targets),
        footprint=measure_model_footprint(rebuilt),
    )


CERTIFICATION_REPORT_SCHEMA = 1


def _certification_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode()).hexdigest()


def certification_report_payload(result: RebuildCertificationResult) -> dict[str, Any]:
    """Build canonical, tamper-evident certification evidence."""

    if not isinstance(result, RebuildCertificationResult):
        raise TypeError("result must be RebuildCertificationResult")
    payload: dict[str, Any] = {
        "schema_version": CERTIFICATION_REPORT_SCHEMA,
        "curriculum_day": 33,
        "model_fingerprint": result.model_fingerprint,
        "batch_fingerprint": result.batch_fingerprint,
        "result": asdict(result),
        "gates": {
            "parameter_manifest": result.parameter_manifest.passed,
            "storage_independence": result.storage_independence.passed,
            "parameter_finiteness": result.parameter_finiteness.passed,
            "activation_finiteness": result.activation_finiteness.passed,
            "batch_invariance": result.batch_invariance.passed,
            "top_k_parity": result.top_k_parity.passed,
            "per_example_loss_parity": result.per_example_loss_parity.passed,
            "probability_simplex": result.probability_simplex.passed,
            "gradient_coverage": result.gradient_coverage.passed,
            "optimizer_step_parity": result.optimizer_step_parity.passed,
            "repeat_inference": result.repeat_inference.passed,
            "inference_rng": result.inference_rng.passed,
            "input_immutability": result.input_immutability.passed,
        },
    }
    payload["evidence_fingerprint"] = _certification_digest(payload)
    return payload


def verify_certification_report(payload: dict[str, Any]) -> None:
    """Reject incomplete, failed, or modified certification evidence."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    candidate = copy.deepcopy(payload)
    fingerprint = candidate.pop("evidence_fingerprint", None)
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise WaveNetError("evidence_fingerprint must be a SHA-256 hex digest")
    if _certification_digest(candidate) != fingerprint:
        raise WaveNetError("certification evidence fingerprint mismatch")
    if candidate.get("schema_version") != CERTIFICATION_REPORT_SCHEMA:
        raise WaveNetError("unsupported certification report schema")
    if candidate.get("curriculum_day") != 33:
        raise WaveNetError("certification report must identify curriculum Day 33")
    gates = candidate.get("gates")
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise WaveNetError("certification report contains a failed gate")


def write_certification_report(path: Path, result: RebuildCertificationResult) -> None:
    """Atomically publish verified rebuild certification evidence."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    payload = certification_report_payload(result)
    verify_certification_report(payload)
    serialized = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_certification_report(path: Path) -> dict[str, Any]:
    """Load and verify persisted rebuild certification evidence."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise WaveNetError("certification report is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise WaveNetError("certification report root must be an object")
    verify_certification_report(payload)
    return payload
