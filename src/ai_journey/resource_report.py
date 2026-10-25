"""Deterministic reports for analytical transformer resource plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .resource_accounting import (
    DecoderSpec,
    TrainingWorkload,
    attention_workspace_memory,
    checkpointed_activation_memory,
    compute_cost,
    compute_runtime,
    distributed_training_memory,
    forward_flops,
    memory_capacity,
    optimizer_update_flops,
    parameter_memory,
    training_state_memory,
    training_step_flops,
)


@dataclass(frozen=True)
class ResourcePlan:
    """Inputs required to reproduce one analytical resource estimate."""

    model: DecoderSpec
    workload: TrainingWorkload
    parameter_dtype: str
    gradient_dtype: str
    optimizer_state_dtype: str
    activation_dtype: str
    master_weights: bool
    checkpoint_segments: int
    save_attention_probabilities: bool
    sharding_strategy: str
    device_capacity_bytes: int
    reserve_fraction: float
    accelerator_peak_flops: int
    model_flops_utilization: float
    hourly_cost_per_accelerator: float


def build_resource_report(plan: ResourcePlan) -> dict[str, object]:
    """Build a JSON-safe report from named, replayable assumptions."""

    model = plan.model
    workload = plan.workload
    microbatch_forward = forward_flops(
        model,
        batch_size=workload.micro_batch_size,
        sequence_length=workload.sequence_length,
    )
    step = training_step_flops(model, workload)
    optimizer = optimizer_update_flops(model)
    total_flops = (step["total"] + optimizer) * workload.optimizer_steps
    activations = checkpointed_activation_memory(
        model,
        batch_size=workload.micro_batch_size,
        sequence_length=workload.sequence_length,
        dtype=plan.activation_dtype,
        checkpoint_segments=plan.checkpoint_segments,
    )
    attention = attention_workspace_memory(
        model,
        batch_size=workload.micro_batch_size,
        sequence_length=workload.sequence_length,
        dtype=plan.activation_dtype,
        save_probabilities=plan.save_attention_probabilities,
    )
    per_device = distributed_training_memory(
        model,
        parameter_dtype=plan.parameter_dtype,
        gradient_dtype=plan.gradient_dtype,
        optimizer_state_dtype=plan.optimizer_state_dtype,
        master_weights=plan.master_weights,
        activation_bytes=activations["peak"] + attention["total"],
        data_parallel_size=workload.data_parallel_size,
        strategy=plan.sharding_strategy,
    )
    capacity = memory_capacity(
        per_device["total"],
        device_capacity_bytes=plan.device_capacity_bytes,
        reserve_fraction=plan.reserve_fraction,
    )
    runtime = compute_runtime(
        total_flops,
        accelerator_peak_flops=plan.accelerator_peak_flops,
        accelerator_count=workload.data_parallel_size,
        model_flops_utilization=plan.model_flops_utilization,
    )
    cost = compute_cost(
        runtime["seconds"],
        accelerator_count=workload.data_parallel_size,
        hourly_cost_per_accelerator=plan.hourly_cost_per_accelerator,
    )
    return {
        "schema_version": 1,
        "kind": "analytical_planning_estimate",
        "measured": False,
        "model": asdict(model),
        "workload": asdict(workload),
        "assumptions": {
            key: value
            for key, value in asdict(plan).items()
            if key not in {"model", "workload"}
        },
        "parameters": {
            "count": model.parameter_count,
            "groups": model.parameter_groups(),
            "storage": parameter_memory(model, dtype=plan.parameter_dtype),
        },
        "flops": {
            "microbatch_forward": microbatch_forward,
            "optimizer_step": step,
            "optimizer_update": optimizer,
            "training_total": total_flops,
        },
        "memory": {
            "training_state": training_state_memory(
                model,
                parameter_dtype=plan.parameter_dtype,
                gradient_dtype=plan.gradient_dtype,
                optimizer_state_dtype=plan.optimizer_state_dtype,
                master_weights=plan.master_weights,
            ),
            "checkpointed_activations": activations,
            "attention_workspace": attention,
            "per_device": per_device,
            "capacity": capacity,
        },
        "runtime": runtime,
        "cost": cost,
        "limitations": [
            "Analytical estimate only; no GPU allocation or kernel was measured.",
            "Allocator fragmentation, communication, and temporary kernel workspaces are excluded.",
            "Runtime and cost depend on the declared MFU and price assumptions.",
        ],
    }
