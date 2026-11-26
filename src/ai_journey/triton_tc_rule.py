"""The instruction-class rule of the four variants on sm_80 and sm_90.

The rule states which math instructions the compile of each variant must
hold, from the PTX and the TTGIR of that compile:

- ``fp32-ieee`` on both targets: scalar ``fma.rn.f32`` and no MMA. IEEE
  float32 has no Tensor Core path, so ``tl.dot`` runs on the FMA lanes;
- ``fp32-tf32``, ``fp16`` and ``bf16`` on sm_80: ``mma.sync`` and no
  ``wgmma``, because sm_80 has no ``wgmma``;
- the same three on sm_90: ``wgmma`` is the class that the lab expects, as
  Triton picks its MMA version 3 for a 4-warp 128 x 128 tile on Hopper. The
  rule also accepts ``mma.sync`` there and records the choice: a Hopper GPU
  runs ``mma.sync`` too, so the choice is the compiler's, and the lab must
  report what Triton 3.9 does, not what the lab expects.

Each compile must also pass these checks: every MMA takes the variant's
operand type (``tf32``, ``f16`` or ``bf16``) and accumulates in ``f32``; the
TTGIR agrees with the PTX (``mma.sync`` with an ``nvidia_mma`` encoding of
version 2, ``wgmma`` with version 3, scalar FMA with none); the static text
holds a whole number of tile products of its class; in a Tensor Core
variant, scalar FMAs hold less than one tile product; and each shared buffer
of the TTGIR fits in the shared memory bytes of the metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from .triton_tc_compile import TARGETS, CompiledVariant
from .triton_tc_kernel import TritonTcError, require_variant
from .triton_tc_ptx import analyze_ptx, static_tile_products
from .triton_tc_ttgir import analyze_ttgir

OBSERVED = ("scalar_fma", "mma_sync", "wgmma", "mixed", "none")
# The class of each observed class in the PTX rule table.
PTX_CLASS = {
    "scalar_fma": "fma_rn_f32",
    "mma_sync": "mma_sync",
    "wgmma": "wgmma_mma_async",
}
MMA_VERSION = {"scalar_fma": (), "mma_sync": (2,), "wgmma": (3,)}
CHECKS = (
    "target",
    "class",
    "operand_types",
    "mma_encoding",
    "tile_products",
    "no_scalar_dot",
    "shared_fits",
)


def allowed_classes(variant: str, target: str) -> tuple[str, ...]:
    """Return the accepted classes, the expected one first."""

    declared = require_variant(variant)
    if target not in TARGETS:
        raise TritonTcError(f"target must be one of {tuple(TARGETS)}")
    if not declared.tensor_cores:
        return ("scalar_fma",)
    return ("mma_sync",) if target == "sm_80" else ("wgmma", "mma_sync")


def observed_class(classes: Mapping[str, int]) -> str:
    """Name the math class of a text from its PTX class counts."""

    sync, group = classes["mma_sync"] > 0, classes["wgmma_mma_async"] > 0
    if sync and group:
        return "mixed"
    if group:
        return "wgmma"
    if sync:
        return "mma_sync"
    return "scalar_fma" if classes["fma_rn_f32"] > 0 else "none"


def _whole(value: Fraction) -> bool:
    return value > 0 and value.denominator == 1


def check_compile(compiled: CompiledVariant) -> dict[str, object]:
    """Apply the rule to one compile; return its analyses, checks and verdict."""

    if type(compiled) is not CompiledVariant:
        raise TypeError("compiled must be CompiledVariant")
    variant = require_variant(compiled.variant)
    ptx = analyze_ptx(compiled.ptx)
    ttgir = analyze_ttgir(compiled.ttgir)
    allowed = allowed_classes(compiled.variant, compiled.target)
    observed = observed_class(ptx.classes)
    products = static_tile_products(ptx)
    wanted = variant.mma_operand_type
    largest = max((buffer.bytes for buffer in ttgir.buffers), default=0)
    checks = {
        "target": ptx.target.startswith(compiled.target)
        and ttgir.target == f"cuda:{TARGETS[compiled.target]}",
        "class": observed in allowed,
        "operand_types": all(
            (form.a_type, form.b_type, form.d_type) == (wanted, wanted, "f32")
            for form in ptx.mma_forms
        )
        and (wanted is None) == (not ptx.mma_forms),
        "mma_encoding": observed in MMA_VERSION
        and ttgir.mma_versions == MMA_VERSION[observed],
        "tile_products": observed in PTX_CLASS
        and _whole(products[PTX_CLASS[observed]]),
        "no_scalar_dot": not variant.tensor_cores or products["fma_rn_f32"] < 1,
        "shared_fits": largest <= compiled.shared_bytes,
    }
    note = None
    if compiled.target == "sm_90" and variant.tensor_cores and observed == "mma_sync":
        note = "triton chose mma.sync on sm_90; the rule accepts and records it"
    return {
        "variant": compiled.variant,
        "target": compiled.target,
        "expected": allowed[0],
        "allowed": list(allowed),
        "observed": observed,
        "checks": checks,
        "accepted": all(checks.values()),
        "note": note,
        "shared_bytes": compiled.shared_bytes,
        "largest_buffer_bytes": largest,
        "buffer_bytes": sum(buffer.bytes for buffer in ttgir.buffers),
        "tile_products": {
            name: {"numerator": value.numerator, "denominator": value.denominator}
            for name, value in products.items()
        },
        "ptx": ptx.to_mapping(),
        "ttgir": {
            "target": ttgir.target,
            "num_warps": ttgir.num_warps,
            "mma_versions": list(ttgir.mma_versions),
            "mma": [item.to_mapping() for item in ttgir.mma],
            "shared": [item.to_mapping() for item in ttgir.shared],
            "dot_operands": ttgir.to_mapping()["dot_operands"],
            "dots": [dict(item) for item in ttgir.dots],
            "buffers": [item.to_mapping() for item in ttgir.buffers],
            "ops": dict(ttgir.ops),
        },
    }


def rule_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Count the accepted compiles and name the class of each one."""

    return {
        "compiles": len(rows),
        "accepted": sum(bool(row["accepted"]) for row in rows),
        "all_accepted": all(row["accepted"] for row in rows) and bool(rows),
        "observed": {
            f"{row['variant']}@{row['target']}": row["observed"] for row in rows
        },
        "mma_sync_on_sm_90": [
            row["variant"] for row in rows if row["note"] is not None
        ],
    }
