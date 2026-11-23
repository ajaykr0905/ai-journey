"""Map the lines of the Day 79 kernel to the TTIR, TTGIR and PTX that carry them.

Each stage text keeps the location of the kernel line that produced each op:
``loc(#locN)`` in TTIR and TTGIR, and ``.loc`` lines in PTX. ``stage_map``
reads those locations and, for each statement in ``STATEMENTS``, counts the
TTIR ops, the TTGIR ops and the PTX instruction families that name its line,
for the whole entry and for one pass of the PTX K loop. Nothing here is typed
by hand: a change of the compiler output changes the map.

``pipeline_facts`` derives what ``num_stages`` did to the loop, and why the
``tl.dot`` became FMA instructions and no ``mma``, from the TTGIR, the PTX
and the compile metadata of one config. Each fact is a count or a check that
a test can pin, not a claim about a GPU run: nothing here ran on a GPU.
"""

from __future__ import annotations

from collections import Counter

from .triton_ir_manifest import CompiledConfig, TritonIrError
from .triton_ir_mlir import MlirModule, ttgir_summary
from .triton_ir_ptx import PtxSummary
from .triton_matmul_kernel import (
    DOT,
    INPUT_PRECISION,
    LOAD_MASKS,
    STORE_MASK,
    TRITON_SOURCE,
)

THREADS_PER_WARP = 32
OPERANDS = 2
STATEMENTS = (
    ("k_loop", "for k0 in range(0, K, BLOCK_K):"),
    ("a_mask", LOAD_MASKS[0]),
    ("b_mask", LOAD_MASKS[1]),
    ("a_load", "a = tl.load(a_ptrs, mask=a_mask, other=0.0)"),
    ("b_load", "b = tl.load(b_ptrs, mask=b_mask, other=0.0)"),
    ("dot", DOT),
    ("a_advance", "a_ptrs += BLOCK_K * stride_ak"),
    ("b_advance", "b_ptrs += BLOCK_K * stride_bk"),
    ("c_mask", STORE_MASK),
    ("store", "tl.store(c_ptrs, acc, mask=c_mask)"),
)


def statement_line(statement: str, text: str = TRITON_SOURCE) -> int:
    """Return the 1-based line of the one source line that is ``statement``."""

    found = [
        number
        for number, line in enumerate(text.split("\n"), start=1)
        if line.strip() == statement
    ]
    if len(found) != 1:
        raise TritonIrError(f"the kernel text must hold {statement!r} once")
    return found[0]


def _ops(module: MlirModule, line: int) -> dict[str, int]:
    return dict(
        sorted(Counter(op.name for op in module.ops if op.source_line == line).items())
    )


def _families(instructions) -> dict[str, int]:
    return dict(sorted(Counter(item.family for item in instructions).items()))


def _loop(ptx: PtxSummary) -> tuple[int, int]:
    if len(ptx.loops) != 1:
        raise TritonIrError("the stage map needs a PTX entry with exactly one loop")
    loop = ptx.loops[0]
    return loop.first_line, loop.last_line


def stage_map(
    ttir: MlirModule, ttgir: MlirModule, ptx: PtxSummary
) -> dict[str, object]:
    """Return, for each statement, the ops and instruction families of its line."""

    first, last = _loop(ptx)
    rows = []
    for key, statement in STATEMENTS:
        line = statement_line(statement)
        mine = [item for item in ptx.instructions if item.source_line == line]
        rows.append(
            {
                "key": key,
                "line": line,
                "statement": statement,
                "ttir": _ops(ttir, line),
                "ttgir": _ops(ttgir, line),
                "ptx": _families(mine),
                "ptx_k_loop": _families(
                    [item for item in mine if first < item.line <= last]
                ),
            }
        )
    lines = sorted(
        {op.source_line for op in (*ttir.ops, *ttgir.ops) if op.source_line}
        | {item.source_line for item in ptx.instructions if item.source_line}
    )
    return {
        "statements": rows,
        "lines": [
            {
                "line": line,
                "ttir_ops": sum(op.source_line == line for op in ttir.ops),
                "ttgir_ops": sum(op.source_line == line for op in ttgir.ops),
                "ptx_instructions": sum(
                    item.source_line == line for item in ptx.instructions
                ),
            }
            for line in lines
        ],
        "without_source_line": {
            "ttir_ops": sum(op.source_line is None for op in ttir.ops),
            "ttgir_ops": sum(op.source_line is None for op in ttgir.ops),
            "ptx_instructions": sum(
                item.source_line is None for item in ptx.instructions
            ),
        },
    }


def _wait_operands(ptx: PtxSummary, inside: bool) -> list[int]:
    first, last = _loop(ptx)
    return [
        int(item.operands.rstrip(";").strip())
        for item in ptx.instructions
        if item.family == "cp.async.wait_group"
        and (first < item.line <= last) == inside
    ]


def pipeline_facts(
    record: CompiledConfig, ttgir: MlirModule, ptx: PtxSummary
) -> dict[str, object]:
    """Derive the cp.async ring and the FMA path of one compile from its texts."""

    config = record.config
    summary = ttgir_summary(ttgir)
    rings = [int(shape.split("x")[0]) for shape in summary["local_alloc"]]
    if len(rings) != OPERANDS or len(set(rings)) != 1:
        raise TritonIrError("the TTGIR must hold one shared ring for A and one for B")
    loop = ptx.loops[0].to_mapping()["families"] if len(ptx.loops) == 1 else None
    if loop is None:
        raise TritonIrError("the pipeline facts need a PTX entry with exactly one loop")
    families = ptx.families
    threads = record.num_warps * THREADS_PER_WARP
    multiply_adds = config.block_m * config.block_n * config.block_k
    buffers = rings[0]
    groups = families["cp.async.commit_group"]
    return {
        "config": config.name,
        "target": record.target,
        "num_stages": record.num_stages,
        "buffers_per_operand": buffers,
        "buffers_are_stages_minus_one": buffers == record.num_stages - 1,
        "tile_bytes": config.tile_bytes,
        "ring_bytes": buffers * config.tile_bytes,
        "shared_bytes": record.shared_bytes,
        "ring_fills_shared": buffers * config.tile_bytes == record.shared_bytes,
        "commit_groups": groups,
        "commit_groups_in_k_loop": loop["cp.async.commit_group"],
        "prologue_commit_groups": groups - loop["cp.async.commit_group"],
        "wait_group_in_k_loop": _wait_operands(ptx, True),
        "wait_group_after_k_loop": _wait_operands(ptx, False),
        "cp_async_per_thread_per_iteration": loop["cp.async"],
        "tile_elements_per_thread": config.tile_bytes // 4 // threads,
        "threads": threads,
        "reqntid": ptx.reqntid,
        "input_precision": INPUT_PRECISION,
        "mma_encoding": summary["mma_encoding"],
        "dot_parent_kinds": sorted(
            {item["parent_kind"] for item in summary["dot_operands"]}
        ),
        "mma_instructions": families["mma"],
        "fma_in_k_loop": loop["fma.rn.f32"],
        "fma_per_thread_per_iteration": multiply_adds // threads,
        "fma_matches_tile": loop["fma.rn.f32"] * threads == multiply_adds,
        "ld_global": families["ld.global"],
    }
