"""The CPU runtime a memory-hierarchy report was measured on, minus identifiers.

The record names the interpreter, the NumPy build and the BLAS library it was
built against, the operating system, the machine architecture and the
logical CPU count: enough to explain why two hosts' rates differ, and nothing
that names a host. Host names, user names, paths, serial numbers, network
addresses and environment variables are never recorded, although
``platform.uname`` reads the host name and NumPy's build record lists build
paths: only the fields above are copied out. Any value that is not a short
version or platform token is recorded as ``unknown``, so the record cannot
smuggle free text into a report.
"""

from __future__ import annotations

import os
import platform
import re

import numpy

TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
ENVIRONMENT_FIELDS = (
    "python_version",
    "python_implementation",
    "numpy_version",
    "numpy_blas",
    "numpy_blas_version",
    "platform_system",
    "platform_machine",
    "cpu_count",
)


def _token(value: object) -> str:
    if isinstance(value, str) and TOKEN.fullmatch(value) is not None:
        return value
    return "unknown"


def _blas() -> dict[str, object]:
    config = getattr(numpy.__config__, "CONFIG", None)
    try:
        blas = config["Build Dependencies"]["blas"]
    except (KeyError, TypeError):
        return {}
    return blas if isinstance(blas, dict) else {}


def host_environment() -> dict[str, object]:
    """Describe the interpreter, NumPy build and CPU without naming the host.

    The BLAS name and version matter because the matmul compute ceiling is
    whatever that library achieves; NumPy builds without the build-time
    configuration record them as ``unknown``.
    """

    blas = _blas()
    count = os.cpu_count()
    return {
        "python_version": _token(platform.python_version()),
        "python_implementation": _token(platform.python_implementation()),
        "numpy_version": _token(numpy.__version__),
        "numpy_blas": _token(blas.get("name")),
        "numpy_blas_version": _token(blas.get("version")),
        "platform_system": _token(platform.system()),
        "platform_machine": _token(platform.machine()),
        "cpu_count": count if type(count) is int and count > 0 else None,
    }
