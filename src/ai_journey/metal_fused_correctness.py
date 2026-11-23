"""Correctness of every softmax step on seeded cases, with determinism measured.

The run first measures the exp probe: ``precise::exp`` at the declared
arguments must meet ``EXP_PROBE_RULE``, the assumption of every error bound,
or the run stops. Each case then draws seeded rows in a gapped layout,
computes the float64 reference, and runs every admitted step ``repeats``
times on the MPS device, each run into a fresh buffer of sentinels.

Four rules apply to the output of every repeat, and a case that breaks one
stops the run, so a report never holds a failed check:

- Every gap slot keeps its sentinel bits.
- Every output element is finite.
- Every element is within its step's bound of the reference.
- Every row sum is within its bound of 1. That bound adds the element
  bounds of the row, so this rule also checks the arithmetic of the bounds.

A race or a missed barrier can spoil one repeat only, so the rules check
each repeat, not only the first. Two facts are measurements, not rules.
``identical_repeats`` records whether every repeat gave the bits of the
first, so a repeat that changed bits within the bounds is kept as data, and
``same_bits`` records whether a step that declares the order of an earlier
step kept that step's bits. The record holds the errors of the first repeat.
``differing_from_rounded_reference`` counts the elements whose bits differ
from the float64 reference rounded to float32; when it is 0, a verifier can
rebuild the whole buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .kernel_metal_geometry import vector_launch
from .metal_fused_config import CorrectnessCase
from .metal_fused_plan import PLAN
from .metal_fused_reference import (
    CORRECTNESS_STREAM,
    EXP_ERROR,
    EXP_PROBE,
    EXP_PROBE_COUNT,
    EXP_PROBE_RULE,
    EXP_ULPS,
    MAX_CASE_INDEX,
    SENTINEL_BITS,
    RowLayout,
    error_bounds,
    exp_probe_arguments,
    exp_probe_errors,
    float32_sha256,
    float64_sha256,
    naive_float32_finite,
    row_sum_errors,
    softmax_inputs,
    softmax_reference,
)
from .metal_fused_source import STEP_VARIANTS, MetalFusedError, require_count

RULES = (
    "every gap slot keeps its sentinel in every repeat; every output is finite; "
    "every element is within its bound of the reference; every row sum is within "
    "its bound of 1; identical_repeats, same_bits and "
    "differing_from_rounded_reference are measured"
)


def runnable(admissions: dict, names: object) -> tuple[str, ...]:
    """Return ``names`` once each one may launch; a refused step is an error.

    A refusal is already a record of the run, so asking to launch the step
    is a mistake of the caller.
    """

    chosen = tuple(names)
    for name in chosen:
        if name not in STEP_VARIANTS:
            raise MetalFusedError(f"{name!r} is not a registered step")
        if not admissions[name].admitted:
            raise MetalFusedError(f"{name} was refused by its pipeline")
    return chosen


@dataclass(frozen=True)
class ProbeRecord:
    """The measured exp probe and the digest of the results behind it."""

    results_sha256: str
    errors: dict[str, object]

    def to_mapping(self) -> dict[str, object]:
        """Return the probe as a report records it."""

        return {
            "arguments": EXP_PROBE,
            "count": EXP_PROBE_COUNT,
            "arguments_sha256": float32_sha256(exp_probe_arguments()),
            "rule": EXP_PROBE_RULE,
            "exp_ulps": EXP_ULPS,
            "exp_error": EXP_ERROR,
            "results_sha256": self.results_sha256,
            **self.errors,
        }


def check_exp_probe(backend: object) -> ProbeRecord:
    """Run the probe once; stop when a result breaks the assumption of the bounds."""

    results = backend.exp_probe(exp_probe_arguments())
    probe = exp_probe_errors(results)
    if not probe.holds:
        raise MetalFusedError(
            "the exp probe breaks the assumption of the bounds: max relative "
            f"error {probe.max_relative_error!r}, tiny max error "
            f"{probe.tiny_max_abs_error!r}"
        )
    return ProbeRecord(float32_sha256(results), probe.to_mapping())


@dataclass(frozen=True)
class FusedOutput:
    """What one step's output showed in one case."""

    kernel: str
    buffer_sha256: str
    identical_repeats: bool
    max_abs_error: float
    max_bound_ratio: float
    max_row_sum_error: float
    max_row_sum_ratio: float
    differing_from_rounded_reference: int
    bound_sha256: str
    row_sum_bound_sha256: str

    def to_mapping(self) -> dict[str, object]:
        """Return the output as a report records it."""

        return {
            "kernel": self.kernel,
            "buffer_sha256": self.buffer_sha256,
            "identical_repeats": self.identical_repeats,
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
            "max_row_sum_error": self.max_row_sum_error,
            "max_row_sum_ratio": self.max_row_sum_ratio,
            "differing_from_rounded_reference": self.differing_from_rounded_reference,
            "bound_sha256": self.bound_sha256,
            "row_sum_bound_sha256": self.row_sum_bound_sha256,
        }


@dataclass(frozen=True)
class FusedCase:
    """One correctness case: its rows, reference and every step's output."""

    index: int
    case: CorrectnessCase
    layout: RowLayout
    repeats: int
    input_sha256: str
    reference_sha256: str
    naive_float32_finite: bool
    outputs: tuple[FusedOutput, ...]

    def same_bits(self) -> list[dict[str, object]]:
        """For each step that declares an earlier step's order: did it keep its bits?"""

        digests = {output.kernel: output.buffer_sha256 for output in self.outputs}
        return [
            {
                "variant": step.variant,
                "as": step.same_bits_as,
                "same": digests[step.variant] == digests[step.same_bits_as],
            }
            for step in PLAN
            if step.same_bits_as is not None
            and step.variant in digests
            and step.same_bits_as in digests
        ]

    def to_mapping(self) -> dict[str, object]:
        """Return the case with its digests, launch and outputs."""

        size = self.layout.group_size
        return {
            "index": self.index,
            "case": self.case.to_mapping(),
            "layout": self.layout.to_mapping(),
            "launch": vector_launch(self.layout.rows * size, size).to_mapping(),
            "repeats": self.repeats,
            "input_sha256": self.input_sha256,
            "reference_sha256": self.reference_sha256,
            "naive_float32_finite": self.naive_float32_finite,
            "outputs": [output.to_mapping() for output in self.outputs],
            "same_bits": self.same_bits(),
        }


def _rules(run: numpy.ndarray, slots, reference, bounds, where: str) -> tuple:
    """Apply the four rules to one repeat; return its errors, or stop the run.

    ``slots`` holds the valid slots in row order and the mask of the gaps.
    """

    valid, gaps = slots
    if numpy.any(run[gaps].view(numpy.uint32) != SENTINEL_BITS):
        raise MetalFusedError(f"{where}: a store escaped into a gap slot")
    packed = run[valid]
    if not numpy.all(numpy.isfinite(packed)):
        raise MetalFusedError(f"{where}: an output is not finite")
    error = numpy.abs(packed.astype(numpy.float64) - reference.output)
    ratio = float(numpy.max(error / bounds.elements))
    if not ratio <= 1.0:
        raise MetalFusedError(f"{where}: an element exceeds its bound by {ratio!r}")
    sums = row_sum_errors(packed, reference.layout)
    sum_ratio = float(numpy.max(sums / bounds.row_sums))
    if not sum_ratio <= 1.0:
        raise MetalFusedError(f"{where}: a row sum exceeds its bound by {sum_ratio!r}")
    return packed, error, ratio, sums, sum_ratio


def _buffer(result: object, layout: RowLayout, label: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (layout.buffer_elements,)
    ):
        raise MetalFusedError(f"{label} returned no float32 buffer of its layout")
    return numpy.ascontiguousarray(result)


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    case: CorrectnessCase,
    names: tuple[str, ...],
    repeats: int,
    required: int,
) -> FusedCase:
    """Run one case on every named step, ``repeats`` times each, under the rules."""

    require_count(index, "case index", 0, MAX_CASE_INDEX)
    if not isinstance(case, CorrectnessCase):
        raise TypeError("case must be a CorrectnessCase")
    layout = case.layout()
    values = softmax_inputs(seed, CORRECTNESS_STREAM, index, layout, case.inputs)
    reference = softmax_reference(values, layout)
    rounded = reference.output.astype(numpy.float32).view(numpy.uint32)
    valid = layout.valid()
    gaps = numpy.ones(layout.buffer_elements, dtype=bool)
    gaps[valid] = False
    label = f"case {index} ({layout.rows} rows, longest {layout.longest})"
    outputs = []
    for name in names:
        where = f"{label}: {name}"
        runs = [
            _buffer(backend.run(name, values, layout, required), layout, where)
            for _ in range(repeats)
        ]
        bounds = error_bounds(name, reference)
        checked = [
            _rules(
                run,
                (valid, gaps),
                reference,
                bounds,
                f"{where}: repeat {number} of {repeats}",
            )
            for number, run in enumerate(runs, start=1)
        ]
        first = runs[0]
        packed, error, ratio, sums, sum_ratio = checked[0]
        first_bits = first.view(numpy.uint32)
        outputs.append(
            FusedOutput(
                kernel=name,
                buffer_sha256=float32_sha256(first),
                identical_repeats=all(
                    numpy.array_equal(run.view(numpy.uint32), first_bits)
                    for run in runs[1:]
                ),
                max_abs_error=float(error.max()),
                max_bound_ratio=ratio,
                max_row_sum_error=float(sums.max()),
                max_row_sum_ratio=sum_ratio,
                differing_from_rounded_reference=int(
                    numpy.count_nonzero(packed.view(numpy.uint32) != rounded)
                ),
                bound_sha256=float64_sha256(bounds.elements),
                row_sum_bound_sha256=float64_sha256(bounds.row_sums),
            )
        )
    return FusedCase(
        index=index,
        case=case,
        layout=layout,
        repeats=repeats,
        input_sha256=float32_sha256(values),
        reference_sha256=float64_sha256(reference.output),
        naive_float32_finite=naive_float32_finite(values, layout),
        outputs=tuple(outputs),
    )
