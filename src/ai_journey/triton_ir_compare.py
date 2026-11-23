"""Compare the six configs and the two targets of the recorded dump.

The manifest gives each of the 12 compiles its shared memory bytes, warps,
stages, stage line counts, stage digests and cubin size. ``compare`` turns
those into one row for each compile and one row for each config pair across
``sm_80`` and ``sm_90``. Only one compile has its texts in the repository,
so only that compile has counted instructions. A number that needs a stage
text of another compile is named in ``NEEDS_FULL_DUMPS`` instead of guessed.

Two numbers of each row are derived, not counted: the ring depth
``shared_bytes / tile_bytes``, and the FMA count of one thread in one K loop
pass if ``tl.dot`` lowers to one FMA for each multiply-add. The committed
PTX confirms the second number for its config only.
"""

from __future__ import annotations

from hashlib import sha256

from .triton_ir_manifest import (
    COMMITTED_CONFIG,
    COMMITTED_TARGET,
    STAGES,
    TARGETS,
    TritonIrError,
    TritonIrManifest,
)
from .triton_ir_ptx import PtxSummary
from .triton_matmul_kernel import AUTOTUNE_CONFIGS

THREADS_PER_WARP = 32
SM_80_TARGET = 'ttg.target = "cuda:80"'
SM_90_TARGET = 'ttg.target = "cuda:90"'
NEEDS_FULL_DUMPS = (
    "the instruction families, registers and loops of the 11 PTX texts that "
    "are not committed, the sm_90 PTX of every config among them",
    "the layouts, rings and async waits of the 11 TTGIR texts that are not "
    "committed",
    "every LLVM IR text: the repository keeps only its digest and line count",
    "the register use and spills of any cubin: ptxas reports them, and no "
    "ptxas ran on this host",
)


def compile_rows(manifest: TritonIrManifest) -> list[dict[str, object]]:
    """Return one row of recorded and derived numbers for each compile."""

    rows = []
    for record in manifest.records:
        config = record.config
        threads = record.num_warps * THREADS_PER_WARP
        ring, rest = divmod(record.shared_bytes, config.tile_bytes)
        rows.append(
            {
                "target": record.target,
                "config": config.name,
                "block": [config.block_m, config.block_n, config.block_k],
                "num_warps": record.num_warps,
                "num_stages": record.num_stages,
                "threads": threads,
                "tile_bytes": config.tile_bytes,
                "shared_bytes": record.shared_bytes,
                "ring_buffers": ring if rest == 0 else None,
                "ring_is_stages_minus_one": rest == 0 and ring == record.num_stages - 1,
                "lines": {stage: record.stage(stage).lines for stage in STAGES},
                "cubin_bytes": record.cubin_bytes,
                "derived_fma_per_thread_per_iteration": (
                    config.block_m * config.block_n * config.block_k // threads
                ),
                "committed_texts": record.target == COMMITTED_TARGET
                and config.name == COMMITTED_CONFIG,
            }
        )
    return rows


def target_rows(
    manifest: TritonIrManifest, committed_ttgir: str
) -> list[dict[str, object]]:
    """Return, for each config, what changes from ``sm_80`` to ``sm_90``.

    For the committed config the TTGIR of ``sm_90`` can be checked without
    its text: the digest of the ``sm_80`` text with only its target attribute
    changed must be the recorded ``sm_90`` digest.
    """

    if committed_ttgir.count(SM_80_TARGET) != 1:
        raise TritonIrError("the committed TTGIR must name its target once")
    retargeted = sha256(
        committed_ttgir.replace(SM_80_TARGET, SM_90_TARGET).encode("ascii")
    ).hexdigest()
    rows = []
    for config in AUTOTUNE_CONFIGS:
        old = manifest.record(TARGETS[0], config.name)
        new = manifest.record(TARGETS[1], config.name)
        rows.append(
            {
                "config": config.name,
                "identical": {
                    **{
                        stage: old.stage(stage).sha256 == new.stage(stage).sha256
                        for stage in STAGES
                    },
                    "cubin": old.cubin_sha256 == new.cubin_sha256,
                },
                "line_change": {
                    stage: new.stage(stage).lines - old.stage(stage).lines
                    for stage in STAGES
                },
                "shared_bytes_equal": old.shared_bytes == new.shared_bytes,
                "cubin_bytes_change": new.cubin_bytes - old.cubin_bytes,
                "ttgir_is_retargeted_sm_80": (
                    retargeted == new.stage("ttgir").sha256
                    if config.name == COMMITTED_CONFIG
                    else None
                ),
            }
        )
    return rows


def compare(
    manifest: TritonIrManifest, committed_ttgir: str, committed_ptx: PtxSummary
) -> dict[str, object]:
    """Return the compile rows, the target rows and the committed counts."""

    rows = compile_rows(manifest)
    committed = next(row for row in rows if row["committed_texts"])
    loop = committed_ptx.loops[0].to_mapping()["families"]
    return {
        "compiles": rows,
        "targets": target_rows(manifest, committed_ttgir),
        "committed": {
            "target": COMMITTED_TARGET,
            "config": COMMITTED_CONFIG,
            "ptx_families": committed_ptx.families,
            "ptx_k_loop_families": loop,
            "derived_fma_confirmed": loop["fma.rn.f32"]
            == committed["derived_fma_per_thread_per_iteration"],
        },
        "needs_full_dumps": list(NEEDS_FULL_DUMPS),
    }
