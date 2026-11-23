"""Correctness of N0 and F1 on seeded ragged cases, with determinism measured.

The run first measures the Day 79 exp probe: ``precise::exp`` must meet
the probe rule, or the run stops, because the bound assumes it. Each case
then draws seeded sequences in a gapped layout and computes the float64
reference of every row in each mode. N0 and F1 run ``repeats`` times in
each mode on the MPS device, each run into a fresh O buffer of sentinels.

Three rules apply to the output of every repeat, and a case that breaks one
stops the run, so a report never holds a failed check:

- Every gap row keeps its sentinel bits.
- Every output value is finite.
- Every value is within its declared bound of the float64 reference.

A race or a missed barrier can spoil one repeat only, so the rules check
each repeat. Two facts are measurements, not rules: ``identical_repeats``
records whether every repeat gave the bits of the first, and
``differing_from_rounded_reference`` counts the values whose bits differ
from the reference rounded to float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .metal_flash_config import FlashCase
from .metal_flash_reference import (
    CORRECTNESS_STREAM,
    MAX_CASE_INDEX,
    FlashReference,
    flash_inputs,
    flash_reference,
    float64_digest,
    gather_rows,
    unscaled_exp_overflows,
)
from .metal_flash_runtime import launch_mappings
from .metal_flash_source import (
    HEAD_DIM,
    MODES,
    VARIANTS,
    MetalFlashError,
    require_size,
)
from .metal_fused_correctness import check_exp_probe
from .metal_fused_reference import SENTINEL_BITS, float32_sha256
from .metal_fused_source import MetalFusedError

RULES = (
    "every gap row keeps its sentinel in every repeat; every output is finite; "
    "every value is within its bound of the reference; identical_repeats and "
    "differing_from_rounded_reference are measured"
)


def check_probe(backend: object) -> dict[str, object]:
    """Run the Day 79 exp probe once; stop when it breaks the bound's assumption."""

    try:
        return check_exp_probe(backend).to_mapping()
    except MetalFlashError:
        raise
    except MetalFusedError as error:
        raise MetalFlashError(str(error)) from error


@dataclass(frozen=True)
class FlashOutput:
    """What one kernel side's output showed in one case and mode."""

    buffer_sha256: str
    identical_repeats: bool
    max_abs_error: float
    max_bound_ratio: float
    differing_from_rounded_reference: int

    def to_mapping(self) -> dict[str, object]:
        """Return the output as a report records it."""

        return {
            "buffer_sha256": self.buffer_sha256,
            "identical_repeats": self.identical_repeats,
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
            "differing_from_rounded_reference": self.differing_from_rounded_reference,
        }


def _buffer(result: object, size: int, where: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (size,)
    ):
        raise MetalFlashError(f"{where} returned no float32 buffer of its layout")
    return numpy.ascontiguousarray(result)


def compare(
    output: numpy.ndarray, reference: FlashReference, bound: numpy.ndarray, where: str
) -> tuple[numpy.ndarray, float, int]:
    """Check the rows of one flat O buffer; return the errors, ratio and count."""

    rows = gather_rows(output, reference.rows)
    if not numpy.all(numpy.isfinite(rows)):
        raise MetalFlashError(f"{where}: an output is not finite")
    error = numpy.abs(rows.astype(numpy.float64) - reference.output)
    ratio = float(numpy.max(error / bound))
    if not ratio <= 1.0:
        raise MetalFlashError(f"{where}: a value exceeds its bound by {ratio!r}")
    rounded = reference.output.astype(numpy.float32).view(numpy.uint32)
    differing = int(numpy.count_nonzero(rows.view(numpy.uint32) != rounded))
    return error, ratio, differing


def check_side(
    backend: object,
    inputs: object,
    reference: FlashReference,
    variant: str,
    repeats: int,
    label: str,
) -> FlashOutput:
    """Run one kernel side ``repeats`` times on one case, under the rules."""

    layout = inputs.layout
    gaps = numpy.ones(layout.buffer_rows, dtype=bool)
    gaps[layout.valid_rows()] = False
    runs = []
    for number in range(1, repeats + 1):
        where = f"{label} {variant}: repeat {number} of {repeats}"
        run = _buffer(
            backend.run(inputs, variant, reference.mode),
            layout.buffer_elements,
            where,
        )
        gap_bits = run.reshape(-1, HEAD_DIM)[gaps].view(numpy.uint32)
        if numpy.any(gap_bits != SENTINEL_BITS):
            raise MetalFlashError(f"{where}: a store escaped into a gap row")
        runs.append((run, *compare(run, reference, reference.bound, where)))
    first, error, ratio, differing = runs[0]
    bits = first.view(numpy.uint32)
    return FlashOutput(
        buffer_sha256=float32_sha256(first),
        identical_repeats=all(
            numpy.array_equal(run.view(numpy.uint32), bits) for run, *_ in runs[1:]
        ),
        max_abs_error=float(error.max()),
        max_bound_ratio=ratio,
        differing_from_rounded_reference=differing,
    )


@dataclass(frozen=True)
class FlashCaseRecord:
    """One correctness case: inputs, references, bounds and kernel outputs."""

    index: int
    case: FlashCase
    repeats: int
    digests: dict[str, str]
    modes: dict[str, dict[str, object]]

    def to_mapping(self) -> dict[str, object]:
        """Return the case with its digests, launches and outputs."""

        layout = self.case.layout()
        return {
            "index": self.index,
            "case": self.case.to_mapping(),
            "layout": layout.to_mapping(),
            "repeats": self.repeats,
            **self.digests,
            "modes": self.modes,
        }


def check_case(
    backend: object, *, seed: int, index: int, case: FlashCase, repeats: int
) -> FlashCaseRecord:
    """Run one case on N0 and F1 in each mode ``repeats`` times, under the rules."""

    require_size(index, "case index", 0, MAX_CASE_INDEX)
    if not isinstance(case, FlashCase):
        raise TypeError("case must be a FlashCase")
    layout = case.layout()
    inputs = flash_inputs(seed, CORRECTNESS_STREAM, index, layout, case.inputs)
    label = f"case {index} ({layout.sequences} sequences, longest {layout.longest})"
    modes = {}
    for mode in MODES:
        reference = flash_reference(inputs, mode)
        modes[mode] = {
            "reference_sha256": float64_digest(reference.output),
            "bound_sha256": float64_digest(reference.bound),
            "max_score": reference.max_score,
            "unscaled_exp_overflows": unscaled_exp_overflows(reference),
            "outputs": {
                variant: {
                    "launches": launch_mappings(variant, mode, layout),
                    **check_side(
                        backend, inputs, reference, variant, repeats, f"{label} {mode}"
                    ).to_mapping(),
                }
                for variant in VARIANTS
            },
        }
    return FlashCaseRecord(
        index=index,
        case=case,
        repeats=repeats,
        digests=inputs.digests(),
        modes=modes,
    )
