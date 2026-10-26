"""Isolated numerical controls for exact CPU decoder experiments."""

from contextlib import contextmanager

import torch


@contextmanager
def isolated_decoder_runtime():
    """Pin float32 CPU execution and restore ambient controls on every exit.

    Only the CPU generator is saved or seeded: accelerator RNG state belongs
    to the caller even when an experiment runs on a GPU-equipped machine.
    """

    rng = torch.get_rng_state()
    dtype = torch.get_default_dtype()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.device("cpu"):
            yield
    finally:
        torch.set_rng_state(rng)
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
