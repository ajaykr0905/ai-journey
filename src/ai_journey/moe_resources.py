"""Allocation-free whole-job admission for bounded CPU MoE comparisons.

Work units are declared parameter-token proxies, not measured FLOPs or timing.
They include both held-out passes, every repeated arm/seed, and dense AdamW
updates of idle experts. Routing/attention elements are admission proxies,
not a complete training-memory estimate or a GPU-readiness claim.
"""

from .checkpoint_archive import MAX_CHECKPOINT_RECORDS
from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_resources import expected_decoder_variant_parameter_count
from .moe_config import (
    MoEConfig,
    expected_moe_active_parameters,
    expected_moe_parameter_count,
)

MAX_JOB_MODEL_PARAMETERS = 5_000_000
MAX_JOB_WORK_UNITS = 100_000_000
MAX_JOB_ROUTE_ELEMENTS = 262_144
MAX_JOB_ATTENTION_ELEMENTS = 8_000_000
MAX_JOB_CHECKPOINT_RECORDS = MAX_CHECKPOINT_RECORDS


def expected_checkpoint_records(config):
    """Bound full model/AdamW/RNG archive records, not bytes or allocator peaks.

    Unique parameters have one model storage and three AdamW storages.
    Tied heads add no storage; 32 more records cover RNG/archive metadata.
    """
    if type(config) not in (DecoderVariantConfig, MoEConfig):
        raise TypeError("checkpoint record estimate requires a dense or MoE config")
    layers = config.gpt2.layer_count
    tensors = 12 * layers + 4
    if config.norm_kind == "rmsnorm":
        tensors -= 2 * layers + 1
    if config.position_kind == "rope":
        tensors -= 1
    if config.mlp_kind == "swiglu":
        tensors -= layers
    if isinstance(config, MoEConfig):
        expert_tensors = 3 if config.mlp_kind == "swiglu" else 4
        tensors += len(config.moe_layers) * (
            (config.expert_count - 1) * expert_tensors + 1
        )
    return 4 * tensors + 32


def _control(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"MoE job {name} must be an integer in [1, {maximum}]")
    return value


def estimate_moe_job(
    arms,
    *,
    seeds,
    steps,
    batch_size,
    evaluation_windows,
    evaluation_batch_size=16,
):
    """Charge all sequential trials without deduplicating equal architectures."""
    if not isinstance(arms, tuple) or not 1 <= len(arms) <= 16:
        raise ValueError("MoE job requires one to 16 declared arms")
    if any(type(arm) not in (DecoderVariantConfig, MoEConfig) for arm in arms):
        raise TypeError("MoE job arms must be exact dense or MoE configurations")
    _control(seeds, "seeds", 16)
    _control(steps, "steps", 64)
    _control(batch_size, "batch_size", 32)
    _control(evaluation_windows, "evaluation_windows", 8192)
    _control(evaluation_batch_size, "evaluation_batch_size", 32)
    stored, active, forward_work, optimizer_work = [], [], [], []
    route_elements, attention_elements = [], []
    peak_batch = max(batch_size, min(evaluation_batch_size, evaluation_windows))
    for arm in arms:
        sparse = isinstance(arm, MoEConfig)
        parameters = (
            expected_moe_parameter_count(arm)
            if sparse
            else expected_decoder_variant_parameter_count(arm)
        )
        active_parameters = (
            expected_moe_active_parameters(arm) if sparse else parameters
        )
        stored.append(parameters)
        active.append(active_parameters)
        forward_work.append(
            active_parameters
            * arm.gpt2.context_length
            * (3 * steps * batch_size + 2 * batch_size + 2 * evaluation_windows)
        )
        optimizer_work.append(10 * parameters * steps)
        route_elements.append(
            peak_batch
            * arm.gpt2.context_length
            * arm.expert_count
            * len(arm.moe_layers)
            if sparse
            else 0
        )
        attention_elements.append(
            peak_batch * arm.gpt2.head_count * arm.gpt2.context_length**2
        )
    forward_total = seeds * sum(forward_work)
    optimizer_total = seeds * sum(optimizer_work)
    return {
        "model_parameters": stored,
        "active_parameters": active,
        "seed_count": seeds,
        "trial_count": seeds * len(arms),
        "forward_backward_work_units": forward_total,
        "optimizer_work_units": optimizer_total,
        "work_units": forward_total + optimizer_total,
        "peak_model_parameters": max(stored),
        "peak_route_elements": max(route_elements),
        "peak_attention_elements": max(attention_elements),
        "peak_checkpoint_records": max(
            expected_checkpoint_records(arm) for arm in arms
        ),
    }


def require_moe_job_budget(arms, **controls):
    """Reject any complete job exceeding an explicit independent CPU budget."""
    estimate = estimate_moe_job(arms, **controls)
    for field, limit in (
        ("peak_model_parameters", MAX_JOB_MODEL_PARAMETERS),
        ("work_units", MAX_JOB_WORK_UNITS),
        ("peak_route_elements", MAX_JOB_ROUTE_ELEMENTS),
        ("peak_attention_elements", MAX_JOB_ATTENTION_ELEMENTS),
        ("peak_checkpoint_records", MAX_JOB_CHECKPOINT_RECORDS),
    ):
        if estimate[field] > limit:
            raise ValueError(f"MoE job exceeds {field} budget of {limit}")
    return estimate
