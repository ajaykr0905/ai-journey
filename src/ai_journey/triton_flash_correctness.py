"""Correctness of the flash attention text against float64, per config and family.

Each run gets three float32 buffers: O with ``rows * D`` values per head, and m
and l with ``rows`` values per head, where ``rows`` is N padded to whole query
blocks. Every slot starts as the Day 73 quiet-NaN sentinel. After a run:

* every guard row, past N, must keep its sentinel bits, so a store that
  escaped its mask fails;
* every value of O, m and l must be finite, so a row that no program stored
  fails;
* every l must be at least 1, because the row max adds ``exp(0) = 1``;
* the largest output error of each head must stay within its declared bound
  against float64 naive attention, and so must the error of ``m + log(l)``
  against the float64 log-sum-exp.

The order of the additions inside ``tl.dot`` and ``tl.sum`` is not fixed, so
the harness holds each output to a bound, not to bit equality. A run that
breaks a rule raises ``TritonFlashError``, so a report never holds a failed
run. A record keeps every output value in base64, so a verifier recomputes
each error from the values.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy

from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .triton_flash_kernel import (
    FlashConfig,
    TritonFlashError,
    key_blocks,
    padded_rows,
    query_blocks,
    require_config,
)
from .triton_flash_reference import (
    FAMILIES,
    AttentionCase,
    array_sha256,
    attention_bounds,
    attention_inputs,
    naive_attention,
    require_case,
)


def encode_float32(array: numpy.ndarray) -> str:
    """Return the base64 text of the C-order little-endian float32 bytes."""

    little = numpy.ascontiguousarray(array, dtype="<f4")
    return base64.b64encode(little.tobytes()).decode("ascii")


def sentinel_buffer(count: int) -> numpy.ndarray:
    """Return ``count`` float32 quiet NaNs with the Day 73 sentinel bits."""

    return numpy.full(count, FLOAT32_SENTINEL_BITS, dtype=numpy.uint32).view(
        numpy.float32
    )


@dataclass(frozen=True)
class RunRecord:
    """One config and family of one case: its launch, values and errors."""

    config: str
    family: str
    programs: int
    key_blocks: int
    guard_rows: int
    output_sha256: str
    maxima_sha256: str
    sums_sha256: str
    output: str
    maxima: str
    sums: str
    output_bounds: tuple[float, ...]
    lse_bounds: tuple[float, ...]
    max_output_error: float
    output_bound_ratio: float
    max_lse_error: float
    lse_bound_ratio: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "config": self.config,
            "family": self.family,
            "programs": self.programs,
            "key_blocks": self.key_blocks,
            "guard_rows": self.guard_rows,
            "output_sha256": self.output_sha256,
            "maxima_sha256": self.maxima_sha256,
            "sums_sha256": self.sums_sha256,
            "output": self.output,
            "maxima": self.maxima,
            "sums": self.sums,
            "output_bounds": list(self.output_bounds),
            "lse_bounds": list(self.lse_bounds),
            "max_output_error": self.max_output_error,
            "output_bound_ratio": self.output_bound_ratio,
            "max_lse_error": self.max_lse_error,
            "lse_bound_ratio": self.lse_bound_ratio,
        }


@dataclass(frozen=True)
class OperandRecord:
    """The digests of one family's float32 inputs.

    The float64 reference goes through ``numpy.matmul``, whose order of
    additions depends on the BLAS build, so its bits differ between hosts. A
    record keeps no digest of it, and a verifier rebuilds it on its own host.
    """

    family: str
    q_sha256: str
    k_sha256: str
    v_sha256: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "family": self.family,
            "q_sha256": self.q_sha256,
            "k_sha256": self.k_sha256,
            "v_sha256": self.v_sha256,
        }


@dataclass(frozen=True)
class CaseRecord:
    """One attention case, checked for every config in both families."""

    index: int
    case: AttentionCase
    operands: tuple[OperandRecord, ...]
    runs: tuple[RunRecord, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "index": self.index,
            "case": self.case.to_mapping(),
            "scale": self.case.scale,
            "operands": [operand.to_mapping() for operand in self.operands],
            "runs": [run.to_mapping() for run in self.runs],
        }


def _buffer(result: object, length: int, label: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (length,)
    ):
        raise TritonFlashError(f"{label} returned no float32 buffer of {length}")
    return numpy.array(result, copy=True)


def _ratio(error: float, bound: float) -> float:
    if error == 0:
        return 0.0
    return error / bound if bound > 0 else float("inf")


def check_run(
    backend: object,
    *,
    family: str,
    inputs: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
    reference: tuple[numpy.ndarray, numpy.ndarray],
    case: AttentionCase,
    config: FlashConfig,
    label: str,
) -> RunRecord:
    """Run one config on one family's inputs and enforce every rule."""

    chosen, checked = require_config(config), require_case(case)
    heads, n, dim = checked.batch_heads, checked.sequence, checked.head_dim
    rows = padded_rows(n, chosen)
    where = f"{label} {family} {chosen.name}"
    result = backend.run(
        *inputs,
        causal=checked.causal,
        config=chosen,
        output=sentinel_buffer(heads * rows * dim),
        maxima=sentinel_buffer(heads * rows),
        sums=sentinel_buffer(heads * rows),
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise TritonFlashError(f"{where}: a backend must return O, m and l")
    output = _buffer(result[0], heads * rows * dim, where).reshape(heads, rows, dim)
    maxima = _buffer(result[1], heads * rows, where).reshape(heads, rows)
    sums = _buffer(result[2], heads * rows, where).reshape(heads, rows)
    for buffer in (output, maxima, sums):
        if numpy.any(buffer[:, n:].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
            raise TritonFlashError(f"{where}: a store escaped its mask into the guard")
    output, maxima, sums = output[:, :n], maxima[:, :n], sums[:, :n]
    if not all(numpy.isfinite(item).all() for item in (output, maxima, sums)):
        raise TritonFlashError(
            f"{where}: a value of O, m or l is not finite, so no program stored "
            "it or the softmax overflowed"
        )
    if not (sums >= 1).all():
        raise TritonFlashError(f"{where}: a running sum l is below 1")
    expected, lse = reference
    bounds = attention_bounds(*inputs, checked.causal, chosen.block_n)
    rebuilt = maxima.astype(numpy.float64) + numpy.log(sums.astype(numpy.float64))
    output_errors = [
        float(numpy.abs(output[head] - expected[head]).max()) for head in range(heads)
    ]
    lse_errors = [
        float(numpy.abs(rebuilt[head] - lse[head]).max()) for head in range(heads)
    ]
    output_ratio = max(
        _ratio(error, bound.output) for error, bound in zip(output_errors, bounds)
    )
    lse_ratio = max(
        _ratio(error, bound.lse) for error, bound in zip(lse_errors, bounds)
    )
    if not output_ratio <= 1:
        raise TritonFlashError(
            f"{where}: an output error is {output_ratio!r} times its bound"
        )
    if not lse_ratio <= 1:
        raise TritonFlashError(
            f"{where}: a log-sum-exp error is {lse_ratio!r} times its bound"
        )
    return RunRecord(
        config=chosen.name,
        family=family,
        programs=heads * query_blocks(n, chosen),
        key_blocks=key_blocks(n, chosen),
        guard_rows=rows - n,
        output_sha256=array_sha256(output),
        maxima_sha256=array_sha256(maxima),
        sums_sha256=array_sha256(sums),
        output=encode_float32(output),
        maxima=encode_float32(maxima),
        sums=encode_float32(sums),
        output_bounds=tuple(bound.output for bound in bounds),
        lse_bounds=tuple(bound.lse for bound in bounds),
        max_output_error=max(output_errors),
        output_bound_ratio=output_ratio,
        max_lse_error=max(lse_errors),
        lse_bound_ratio=lse_ratio,
    )


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    case: AttentionCase,
    configs: tuple[FlashConfig, ...],
) -> CaseRecord:
    """Run every config on both families of one case, config by config."""

    checked = require_case(case)
    if type(configs) is not tuple or not configs:
        raise TritonFlashError("configs must be a non-empty tuple")
    label = (
        f"case {index} (BH={checked.batch_heads} N={checked.sequence} "
        f"D={checked.head_dim} causal={checked.causal})"
    )
    operands, prepared = [], {}
    for family in FAMILIES:
        inputs = attention_inputs(family, seed, index, checked)
        reference = naive_attention(*inputs, checked.causal)
        prepared[family] = (inputs, reference)
        operands.append(OperandRecord(family, *(array_sha256(item) for item in inputs)))
    runs = [
        check_run(
            backend,
            family=family,
            inputs=prepared[family][0],
            reference=prepared[family][1],
            case=checked,
            config=config,
            label=label,
        )
        for config in configs
        for family in FAMILIES
    ]
    return CaseRecord(index, checked, tuple(operands), tuple(runs))
