"""Conservative admission checks covering every decoder comparison arm."""

from .gpt2_config import GPT2Config

MAX_WORK_UNITS = 200_000
MAX_TOTAL_WORK_UNITS = 1_000_000
MAX_ATTENTION_ELEMENTS = 8_000_000


def require_comparison_budget(
    base: GPT2Config,
    *,
    steps: int,
    batch_size: int,
    evaluation_windows: int,
    variant_count: int = 5,
) -> None:
    """Bound training, evaluation and peak attention independently.

    Work units are an admission proxy rather than measured FLOPs. Training
    receives a factor of three for its forward/backward/update work; both
    diagnostic training forwards and held-out evaluation are also included.
    """

    for value in (steps, batch_size, evaluation_windows, variant_count):
        if type(value) is not int or value < 1:
            raise ValueError("comparison budget controls must be positive integers")
    width_work = base.context_length * base.layer_count * base.embedding_dim
    training = steps * batch_size * width_work
    if training > MAX_WORK_UNITS:
        raise ValueError(f"requested experiment exceeds {MAX_WORK_UNITS} work units")
    total = variant_count * (
        3 * training + (2 * batch_size + evaluation_windows) * width_work
    )
    if total > MAX_TOTAL_WORK_UNITS:
        raise ValueError("training and evaluation exceed total comparison work budget")
    attention = (
        max(batch_size, evaluation_windows) * base.head_count * base.context_length**2
    )
    if attention > MAX_ATTENTION_ELEMENTS:
        raise ValueError("comparison exceeds peak attention element budget")
