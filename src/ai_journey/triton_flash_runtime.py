"""Load the pinned flash attention text into the Triton CPU interpreter, on demand.

``InterpreterBackend`` imports triton and torch when a command makes it, and
never when this module is imported. It names the first missing requirement in
``TritonFlashUnavailableError``, so the check command can exit 3 before it
writes anything. Triton has no macOS wheel, so on the development host the
backend always stops there.

The backend runs only the CPU interpreter. It sets ``TRITON_INTERPRET=1`` when
the variable is unset, and it refuses a host where ``triton.jit`` does not
return an ``InterpretedFunction``. The interpreter runs each program instance
in turn as Python and NumPy code on the CPU, so a run checks the indexing, the
masks, the key loop and the online softmax of the text. It is not a GPU run,
and its time is not a performance number.

``compile_text`` follows the Day 79 lab: it registers the text in
``linecache`` under a name that holds the digest of the text, and then
executes it in a fresh namespace. ``triton.jit`` and the interpreter read the
kernel source through ``inspect``, so they read the exact pinned text, and a
second text can never replace the source of a kernel compiled before it.
"""

from __future__ import annotations

import importlib
import linecache
import os
import platform

import numpy

from .triton_flash_kernel import (
    KERNEL_NAME,
    LAUNCHER_NAME,
    TRITON_SOURCE,
    FlashConfig,
    TritonFlashError,
    float32_scale,
    padded_rows,
    require_config,
    source_sha256,
)
from .triton_matmul_runtime import INTERPRET_VARIABLE, INTERPRETED_TYPE, REASONS

INTERPRETER_BACKEND = "triton_interpreter"
SOURCE_PREFIX = "<ai_journey.triton_flash_kernel.TRITON_SOURCE"


class TritonFlashUnavailableError(TritonFlashError):
    """Raised when this host cannot run the Triton CPU interpreter; nothing ran."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}")
        self.reason = reason
        super().__init__(f"{detail} ({reason})")


def _import(name: str, reason: str) -> object:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise TritonFlashUnavailableError(
            reason, f"{name} cannot be imported: {error}"
        ) from error


def source_filename(text: str) -> str:
    """Name one text in ``linecache`` by its digest, so no two texts share it."""

    return f"{SOURCE_PREFIX} sha256={source_sha256(text)}>"


def compile_text() -> tuple[str, dict[str, object]]:
    """Execute ``TRITON_SOURCE`` in a fresh namespace; return the text and it."""

    text = TRITON_SOURCE
    filename = source_filename(text)
    lines = text.splitlines(keepends=True)
    linecache.cache[filename] = (len(text), None, lines, filename)
    namespace: dict[str, object] = {"__name__": "ai_journey_triton_flash_text"}
    exec(compile(text, filename, "exec"), namespace)
    return text, namespace


class InterpreterBackend:
    """Run ``launch_flash_attention`` of the pinned text in the CPU interpreter."""

    name = INTERPRETER_BACKEND

    def __init__(self) -> None:
        triton = _import("triton", "triton_not_importable")
        torch = _import("torch", "torch_not_importable")
        if INTERPRET_VARIABLE not in os.environ:
            os.environ[INTERPRET_VARIABLE] = "1"
        text, namespace = compile_text()
        kernel = namespace[KERNEL_NAME]
        if type(kernel).__name__ != INTERPRETED_TYPE:
            raise TritonFlashUnavailableError(
                "interpreter_disabled",
                f"{INTERPRET_VARIABLE}={os.environ[INTERPRET_VARIABLE]!r} gives "
                f"{type(kernel).__name__}, and this lab runs only the CPU "
                "interpreter",
            )
        self.kernel = kernel
        self.source_sha256 = source_sha256(text)
        self._launch = namespace[LAUNCHER_NAME]
        self._torch = torch
        self.environment = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
            "numpy": numpy.__version__,
            "torch": str(torch.__version__),
            "triton": str(triton.__version__),
            "triton_interpret": os.environ[INTERPRET_VARIABLE],
        }

    def run(
        self,
        q: numpy.ndarray,
        k: numpy.ndarray,
        v: numpy.ndarray,
        *,
        causal: bool,
        config: FlashConfig,
        output: numpy.ndarray,
        maxima: numpy.ndarray,
        sums: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Launch the text on CPU tensors that share memory with the buffers.

        Each buffer holds ``rows = padded_rows(N, config)`` rows per head, and
        the rows past N are the guard. The interpreter raises its own errors,
        and this method names them as lab errors, so a command reports a
        failed check.
        """

        chosen = require_config(config)
        torch = self._torch
        heads, sequence, dim = q.shape
        try:
            self._launch(
                *(torch.from_numpy(numpy.ascontiguousarray(x)) for x in (q, k, v)),
                *(torch.from_numpy(buffer) for buffer in (output, maxima, sums)),
                heads,
                sequence,
                dim,
                float32_scale(dim),
                padded_rows(sequence, chosen),
                causal=causal,
                **chosen.launcher_arguments(),
            )
        except Exception as error:
            raise TritonFlashError(
                f"the Triton interpreter failed on {chosen.name}: {error}"
            ) from error
        return output, maxima, sums
