"""Load the pinned Triton text into the CPU interpreter, only when a run needs it.

``InterpreterBackend`` imports triton and torch when a command makes it, and
never when this module is imported. It names the first missing requirement in
``TritonMatmulUnavailableError``, so the check command can exit 3 before it
writes anything. Triton has no macOS wheel, so on the development host the
backend always stops there.

The backend runs only the CPU interpreter. It sets ``TRITON_INTERPRET=1`` when
the variable is unset, and it refuses a host where ``triton.jit`` does not
return an ``InterpretedFunction``, for example when ``TRITON_INTERPRET=0``.
The interpreter runs each program instance in turn as Python and NumPy code on
the CPU: a ``tl.load`` with a mask reads only the masked lanes, and ``tl.dot``
is ``numpy.matmul(a, b, dtype=float32) + acc`` in Triton 3.9.0. Thus a run
checks the indexing, the masks and the K loop of the text. It is not a GPU
run, and its time is not a performance number.

``compile_text`` executes ``TRITON_SOURCE`` in a fresh namespace. It first
registers the text in ``linecache`` under ``SOURCE_FILENAME``, so
``inspect.getsource`` reads the kernel as a file: ``triton.jit`` and the
interpreter read the source of the function that they wrap.

``CI_WHEELS`` and ``ci_install_mapping`` record what the PyPI metadata of
triton 3.9.0 stated on 2026-11-22: one CPython 3.11 wheel for Linux x86_64 and
one for Linux aarch64, ``Requires-Python >=3.11,<3.16``, and no dependency on
Python 3.11. A resolver run confirmed the x86_64 wheel for a Linux CI host.
"""

from __future__ import annotations

import importlib
import linecache
import os
import platform
from dataclasses import dataclass

import numpy

from .triton_matmul_kernel import (
    KERNEL_NAME,
    LAUNCHER_NAME,
    TRITON_SOURCE,
    TritonConfig,
    TritonMatmulError,
    require_config,
    source_sha256,
)

INTERPRETER_BACKEND = "triton_interpreter"
INTERPRET_VARIABLE = "TRITON_INTERPRET"
INTERPRETED_TYPE = "InterpretedFunction"
SOURCE_FILENAME = "<ai_journey.triton_matmul_kernel.TRITON_SOURCE>"
ENVIRONMENT_FIELDS = (
    "python",
    "implementation",
    "system",
    "machine",
    "numpy",
    "torch",
    "triton",
    "triton_interpret",
)
REASONS = ("triton_not_importable", "torch_not_importable", "interpreter_disabled")
PYPI_VERSION = "3.9.0"
PYPI_SOURCE = "https://pypi.org/pypi/triton/3.9.0/json"
PYPI_READ_ON = "2026-11-22"
REQUIRES_PYTHON = ">=3.11,<3.16"
REQUIRES_DIST = ('importlib-metadata; python_version < "3.10"',)
CORE_METADATA_SHA256 = (
    "5dfdd39550d7c33bb319d2427993a7cd259ea1d9209213938e5cab6dfc7373ae"
)
RESOLVER_CHECK = (
    "uv 0.9.7: uv pip compile with --python-version 3.11, --python-platform "
    "x86_64-manylinux_2_28 and --only-binary :all: resolved triton==3.9.0 and "
    "no other package"
)


class TritonMatmulUnavailableError(TritonMatmulError):
    """Raised when this host cannot run the Triton CPU interpreter; nothing ran."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}")
        self.reason = reason
        super().__init__(f"{detail} ({reason})")


@dataclass(frozen=True)
class WheelRecord:
    """One wheel file of the PyPI release, as its JSON metadata names it."""

    filename: str
    sha256: str
    size: int
    uploaded: str

    @property
    def tags(self) -> dict[str, object]:
        """Split the file name into its PEP 427 python, abi and platform tags."""

        _, version, python, abi, platforms = self.filename.removesuffix(".whl").split(
            "-"
        )
        return {
            "version": version,
            "python": python,
            "abi": abi,
            "platforms": platforms.split("."),
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size": self.size,
            "uploaded": self.uploaded,
            "tags": self.tags,
        }


CI_WHEELS = (
    WheelRecord(
        "triton-3.9.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "1622118a038b0c6e16fa4a0305da62c8b3c15904ee1cc4d4e65846a25969e41e",
        285777556,
        "2026-10-27T18:28:00.919896Z",
    ),
    WheelRecord(
        "triton-3.9.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "feaaec13578ca84301d561d034cb08b3ae293d7f0b7ab6b8ddbb2d6796363716",
        263373771,
        "2026-10-27T18:38:49.077989Z",
    ),
)


def ci_install_mapping() -> dict[str, object]:
    """Return the recorded PyPI facts that a Linux CI install of triton needs."""

    return {
        "source": PYPI_SOURCE,
        "read_on": PYPI_READ_ON,
        "version": PYPI_VERSION,
        "requires_python": REQUIRES_PYTHON,
        "requires_dist": list(REQUIRES_DIST),
        "core_metadata_sha256": CORE_METADATA_SHA256,
        "wheels": [wheel.to_mapping() for wheel in CI_WHEELS],
        "resolver_check": RESOLVER_CHECK,
    }


def _import(name: str, reason: str) -> object:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise TritonMatmulUnavailableError(
            reason, f"{name} cannot be imported: {error}"
        ) from error


def compile_text() -> tuple[str, dict[str, object]]:
    """Execute ``TRITON_SOURCE`` in a fresh namespace; return the text and it.

    The text is read once, so the caller hashes the exact text that ran. The
    linecache entry has no modification time, so ``linecache.checkcache``
    keeps it although no file has that name.
    """

    text = TRITON_SOURCE
    lines = text.splitlines(keepends=True)
    linecache.cache[SOURCE_FILENAME] = (len(text), None, lines, SOURCE_FILENAME)
    namespace: dict[str, object] = {"__name__": "ai_journey_triton_matmul_text"}
    exec(compile(text, SOURCE_FILENAME, "exec"), namespace)
    return text, namespace


class InterpreterBackend:
    """Run ``launch_matmul`` of the pinned text in the Triton CPU interpreter."""

    name = INTERPRETER_BACKEND

    def __init__(self) -> None:
        triton = _import("triton", "triton_not_importable")
        torch = _import("torch", "torch_not_importable")
        if INTERPRET_VARIABLE not in os.environ:
            os.environ[INTERPRET_VARIABLE] = "1"
        text, namespace = compile_text()
        kernel = namespace[KERNEL_NAME]
        if type(kernel).__name__ != INTERPRETED_TYPE:
            raise TritonMatmulUnavailableError(
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
        left: numpy.ndarray,
        right: numpy.ndarray,
        config: TritonConfig,
        buffer: numpy.ndarray,
    ) -> numpy.ndarray:
        """Launch the text on CPU tensors that share memory with the arrays.

        ``buffer`` holds C and its guard tail as one flat float32 array. The
        kernel gets row-major strides, so a store past the last row of C lands
        in the tail.
        """

        chosen = require_config(config)
        torch = self._torch
        rows, inner = left.shape
        columns = right.shape[1]
        self._launch(
            torch.from_numpy(numpy.ascontiguousarray(left)),
            torch.from_numpy(numpy.ascontiguousarray(right)),
            torch.from_numpy(buffer),
            rows,
            columns,
            inner,
            **chosen.launcher_arguments(),
        )
        return buffer
