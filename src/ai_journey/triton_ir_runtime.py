"""Compile the Day 79 kernel text again with triton and compare it with the dump.

``CompilerBackend`` imports triton only when a command makes it, never when
this module is imported. Triton has no macOS wheel, so on the development
host it stops with ``TritonIrUnavailableError`` before anything is compiled
or written, and the check command exits 3. On a Linux host with the triton
3.9.0 wheel it compiles ``TRITON_SOURCE`` the way the recorded dump script
did: ``ASTSource`` with the declared signature and the config's constexprs,
``GPUTarget("cuda", 80 or 90, 32)``, and the config's warps and stages. A
compile for a declared target needs no GPU, and none runs.

``regenerate`` compares each regenerated stage text with the manifest by
digest and line count, and each compile's metadata with the recorded one.
The session owner found that a second compile gave byte-identical TTIR,
TTGIR, LLVM IR and PTX for every compile, while the ``sm_80`` cubins differed
between the two runs. So a text must match, and a cubin digest is recorded
with its match but is not a rule.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from .triton_ir_manifest import (
    STAGES,
    TritonIrError,
    TritonIrManifest,
)
from .triton_matmul_kernel import KERNEL_NAME, TritonConfig, source_sha256
from .triton_matmul_runtime import compile_text

COMPILER_BACKEND = "triton_compiler"
INJECTED_BACKEND = "injected"
FP_FUSION_VARIABLE = "TRITON_DEFAULT_FP_FUSION"
JIT_TYPE = "JITFunction"
ENVIRONMENT_FIELDS = ("python", "system", "machine", "triton", "fp_fusion")
REASONS = ("triton_not_importable", "compiler_not_importable", "interpreter_enabled")
CAPABILITIES = {"sm_80": 80, "sm_90": 90}
THREADS_PER_WARP = 32
POINTERS = ("a_ptr", "b_ptr", "c_ptr")
INTEGERS = (
    "M",
    "N",
    "K",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
)
CONSTEXPRS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M")


class TritonIrUnavailableError(TritonIrError):
    """Raised when this host cannot compile with triton; nothing was compiled."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}")
        self.reason = reason
        super().__init__(f"{detail} ({reason})")


def signature() -> dict[str, str]:
    """Return the ``ASTSource`` signature that the recorded dump used."""

    return {
        **dict.fromkeys(POINTERS, "*fp32"),
        **dict.fromkeys(INTEGERS, "i32"),
        **dict.fromkeys(CONSTEXPRS, "constexpr"),
    }


@dataclass(frozen=True)
class CompiledStages:
    """The stage texts and the metadata of one regenerated compile."""

    texts: Mapping[str, str]
    shared_bytes: int
    num_warps: int
    num_stages: int
    cubin: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.texts, Mapping) or tuple(self.texts) != STAGES:
            raise TritonIrError(f"a compile must give the stages {STAGES} in order")
        if not all(type(text) is str for text in self.texts.values()):
            raise TritonIrError("each stage text must be a string")
        for name in ("shared_bytes", "num_warps", "num_stages"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TritonIrError(f"{name} must be a non-negative integer")
        if type(self.cubin) is not bytes:
            raise TritonIrError("cubin must be bytes")


def _import(name: str, reason: str) -> object:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise TritonIrUnavailableError(
            reason, f"{name} cannot be imported: {error}"
        ) from error


class CompilerBackend:
    """Compile the pinned text with the installed triton for a declared target."""

    name = COMPILER_BACKEND

    def __init__(self) -> None:
        triton = _import("triton", "triton_not_importable")
        compiler = _import("triton.compiler", "compiler_not_importable")
        backends = _import("triton.backends.compiler", "compiler_not_importable")
        os.environ.setdefault(FP_FUSION_VARIABLE, "1")
        text, namespace = compile_text()
        kernel = namespace[KERNEL_NAME]
        if type(kernel).__name__ != JIT_TYPE:
            raise TritonIrUnavailableError(
                "interpreter_enabled",
                f"triton.jit gives {type(kernel).__name__}, so TRITON_INTERPRET "
                "turns the compiler off",
            )
        self.source_sha256 = source_sha256(text)
        self._triton = triton
        self._source = compiler.ASTSource
        self._target = backends.GPUTarget
        self._kernel = kernel
        self.environment = {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "triton": str(triton.__version__),
            "fp_fusion": os.environ[FP_FUSION_VARIABLE],
        }

    def compile(self, target: str, config: TritonConfig) -> CompiledStages:
        source = self._source(
            fn=self._kernel,
            signature=signature(),
            constexprs=config.meta_parameters(),
        )
        compiled = self._triton.compile(
            source,
            target=self._target("cuda", CAPABILITIES[target], THREADS_PER_WARP),
            options={"num_warps": config.num_warps, "num_stages": config.num_stages},
        )
        return CompiledStages(
            texts={stage: compiled.asm[stage] for stage in STAGES},
            shared_bytes=int(compiled.metadata.shared),
            num_warps=int(compiled.metadata.num_warps),
            num_stages=int(compiled.metadata.num_stages),
            cubin=bytes(compiled.asm["cubin"]),
        )


def _environment(backend: object) -> dict[str, str]:
    environment = getattr(backend, "environment", None)
    if (
        not isinstance(environment, Mapping)
        or tuple(environment) != ENVIRONMENT_FIELDS
        or not all(type(value) is str for value in environment.values())
    ):
        raise TritonIrError(
            f"a backend must name its environment: {', '.join(ENVIRONMENT_FIELDS)}"
        )
    return dict(environment)


def regenerate(manifest: TritonIrManifest, backend: object) -> dict[str, object]:
    """Compile every recorded config again and compare it with the manifest."""

    if type(manifest) is not TritonIrManifest:
        raise TypeError("manifest must be TritonIrManifest")
    environment = _environment(backend)
    compiler = type(backend) is CompilerBackend
    if compiler and backend.source_sha256 != manifest.source_sha256:
        raise TritonIrError("the backend compiled another text than the dump")
    rows = []
    for record in manifest.records:
        compiled = backend.compile(record.target, record.config)
        if type(compiled) is not CompiledStages:
            raise TritonIrError("a backend must return CompiledStages")
        stages = {}
        for stage in STAGES:
            text = compiled.texts[stage]
            recorded = record.stage(stage)
            digest = sha256(text.encode("utf-8")).hexdigest()
            lines = len(text.splitlines())
            stages[stage] = {
                "recorded_sha256": recorded.sha256,
                "regenerated_sha256": digest,
                "recorded_lines": recorded.lines,
                "regenerated_lines": lines,
                "match": digest == recorded.sha256 and lines == recorded.lines,
            }
        cubin = sha256(compiled.cubin).hexdigest()
        rows.append(
            {
                "target": record.target,
                "config": record.name,
                "stages": stages,
                "metadata": {
                    "shared_bytes": compiled.shared_bytes,
                    "num_warps": compiled.num_warps,
                    "num_stages": compiled.num_stages,
                    "match": (
                        compiled.shared_bytes,
                        compiled.num_warps,
                        compiled.num_stages,
                    )
                    == (record.shared_bytes, record.num_warps, record.num_stages),
                },
                "cubin": {
                    "recorded_sha256": record.cubin_sha256,
                    "regenerated_sha256": cubin,
                    "recorded_bytes": record.cubin_bytes,
                    "regenerated_bytes": len(compiled.cubin),
                    "match": cubin == record.cubin_sha256,
                },
            }
        )
    return {
        "backend": COMPILER_BACKEND if compiler else INJECTED_BACKEND,
        "environment": environment,
        "compiles": rows,
        "summary": regeneration_summary(rows),
    }


def regeneration_summary(rows: list[Mapping[str, object]]) -> dict[str, object]:
    """Count the matching texts, metadata and cubins; texts and metadata rule."""

    texts = [stage["match"] for row in rows for stage in row["stages"].values()]
    metadata = [row["metadata"]["match"] for row in rows]
    cubins = [row["cubin"]["match"] for row in rows]
    return {
        "texts_matched": sum(texts),
        "texts": len(texts),
        "metadata_matched": sum(metadata),
        "compiles": len(metadata),
        "cubins_matched": sum(cubins),
        "cubins_by_target": {
            target: sum(
                row["cubin"]["match"] for row in rows if row["target"] == target
            )
            for target in CAPABILITIES
        },
        "all_texts_and_metadata_match": all(texts) and all(metadata),
    }
