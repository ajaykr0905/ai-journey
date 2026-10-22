"""Resource preflight before allocating CPU lab models or launching processes."""

from hashlib import sha256
from pathlib import Path

from .ddp_contracts import integer
from .gpt2_model import expected_gpt2_parameter_count
from .gpt2_training import load_training_spec
from .ddp_data import BatchPlan


def preflight(source: Path, spec_path: Path, *, world_size: int, steps: int):
    integer(world_size, "world_size", 1)
    integer(steps, "steps", 1)
    if world_size > 4 or steps > 128:
        raise ValueError("CPU lab permits four workers and 128 updates per invocation")
    spec = load_training_spec(spec_path)
    if (
        source.is_symlink()
        or not source.is_file()
        or not 1 < source.stat().st_size <= 1024 * 1024
    ):
        raise ValueError("public source must be a regular file within the 1 MiB budget")
    data = source.read_bytes()
    if sha256(data).hexdigest() != spec.source.sha256:
        raise ValueError("public source digest does not match specification")
    if (
        expected_gpt2_parameter_count(spec.model) > 1_000_000
        or spec.model.context_length > 128
    ):
        raise ValueError("model exceeds CPU parameter or context budget")
    if (
        spec.training.tokens_per_step * world_size > 8192
        or steps > spec.training.optimizer_steps
    ):
        raise ValueError("requested training exceeds token or schedule budget")
    BatchPlan(
        (len(data) - 1) // spec.training.sequence_length,
        world_size,
        spec.training.micro_batch_size,
        spec.training.accumulation_steps,
    )
    return spec
