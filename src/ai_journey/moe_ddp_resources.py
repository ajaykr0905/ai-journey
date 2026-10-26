"""Allocation-free complete-window admission; proxies are not measured FLOPs."""

from .moe_config import expected_moe_parameter_count, expected_moe_active_parameters
from .moe_resources import expected_checkpoint_records


def estimate_moe_ddp_job(config):
    model = config.model
    stored, active = (
        expected_moe_parameter_count(model),
        expected_moe_active_parameters(model),
    )
    local_positions = (
        config.micro_batch_size * config.sequence_length * config.accumulation_steps
    )
    return {
        "stored_parameters_per_rank": stored,
        "global_positions_per_update": config.world_size * local_positions,
        "retained_route_elements_per_rank": local_positions
        * model.expert_count
        * len(model.moe_layers),
        "retained_attention_elements_per_rank": config.accumulation_steps
        * config.micro_batch_size
        * model.gpt2.head_count
        * config.sequence_length**2
        * model.gpt2.layer_count,
        "whole_job_work_units": config.world_size
        * config.optimizer_steps
        * (3 * active * local_positions + 10 * stored),
        "checkpoint_records": expected_checkpoint_records(model),
        "dtype": "float64",
        "capacity_scope": "rank_microbatch",
    }


def require_moe_ddp_budget(config):
    report = estimate_moe_ddp_job(config)
    for name, limit in (
        ("stored_parameters_per_rank", 1_000_000),
        ("global_positions_per_update", 8192),
        ("retained_route_elements_per_rank", 262144),
        ("retained_attention_elements_per_rank", 8_000_000),
        ("whole_job_work_units", 100_000_000),
        ("checkpoint_records", 16384),
    ):
        if report[name] > limit:
            raise ValueError(f"MoE DDP exceeds {name} budget {limit}")
    return report
