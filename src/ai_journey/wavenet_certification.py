"""Operational certification checks for the independently rebuilt WaveNet."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from hashlib import sha256

import torch
from torch.nn import functional as F

from .wavenet import HierarchicalLanguageModel, WaveNetError
from .wavenet_rebuild import (
    RebuiltWaveNet,
    rebuild_model_fingerprint,
    reference_parameter_pairs,
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
