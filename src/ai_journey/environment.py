"""Runtime diagnostics that are safe to print in logs and CI."""

from __future__ import annotations

import platform
import sys
from typing import Any


def environment_report() -> dict[str, Any]:
    """Return Python and optional PyTorch/CUDA capability information.

    The report intentionally contains no environment-variable values, usernames,
    filesystem paths, hostnames, tokens, or browser state.
    """

    report: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "python_3_11_or_newer": sys.version_info >= (3, 11),
    }

    try:
        import torch
    except ImportError:
        report.update(
            {
                "torch_available": False,
                "cuda_available": False,
                "message": (
                    "PyTorch is not installed; CPU-only NumPy exercises remain "
                    "available and Day 9 uses finite differences."
                ),
            }
        )
        return report

    cuda_available = bool(torch.cuda.is_available())
    report.update(
        {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "message": (
                "CUDA is available to PyTorch."
                if cuda_available
                else "PyTorch is installed, but CUDA is not available; using CPU."
            ),
        }
    )
    return report
