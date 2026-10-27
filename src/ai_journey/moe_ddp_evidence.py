"""Check bounded replicated expert observations; seals are not authentication."""

from hashlib import sha256
import json
import math

import torch

from .checkpoint_archive import load_checkpoint_archive, MAX_CHECKPOINT_RECORDS
from .checkpoint_journal import _read_bytes
from .moe_checkpoint import _digest, restore_moe_payload
from .moe_ddp_checkpoint import MAX_BYTES, SCHEMA as CHECKPOINT_SCHEMA
from .moe_ddp_checkpoint import moe_ddp_training_contract
from .moe_ddp_data import public_moe_ddp_schedule
from .moe_ddp_plan import (
    MAX_REPORT_BYTES,
    canonical,
    public_moe_ddp_plan_from_mapping,
    source_admission,
    unique_fields,
)
from .moe_ddp_worker import RECEIPT_SCHEMA
from .moe_route_audit import validate_moe_routing
from .moe_training import build_moe_optimizer, validate_moe_optimizer
from .training_models import build_training_model, training_model_fingerprint

SCHEMA = "ai-journey-public-moe-ddp-evidence-v1"
LIMITATIONS = [
    "Bounded replicated CPU/Gloo float64 work; not expert parallelism or GPU evidence.",
    "No throughput, production, distributed-scale or learner-completion claim.",
    "JSON seals check internal consistency, not authorship or measurement authenticity.",
    "Static verification is not training replay; optional archive verification restores locally.",
]
_RANK_FIELDS = {
    "schema",
    "rank",
    "world_size",
    "pid",
    "plan_sha256",
    "started_step",
    "completed_step",
    "model_sha256",
    "checkpoint_identity",
    "metrics",
}
_METRIC_FIELDS = {
    "step",
    "target_tokens",
    "ce_loss",
    "balance_loss",
    "router_z_loss",
    "objective",
    "gradient_norm_before_clip",
    "gradient_norm_after_clip",
    "model_sha256",
    "attempted_counts",
    "accepted_counts",
    "capacity_scope",
}


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"MoE evidence {name} is outside its integer range")
    return value


def _number(value):
    if type(value) not in (int, float) or not 0 <= value <= 1e100:
        raise ValueError(
            "MoE evidence measurement must be a bounded nonnegative number"
        )
    return float(value)


def _sha(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ValueError("MoE evidence identity must be a lowercase SHA-256")
    return value


def _identity(value):
    if type(value) is not dict or set(value) != {"payload_sha256", "archive_sha256"}:
        raise ValueError("MoE evidence checkpoint identities differ")
    for item in value.values():
        _sha(item)


def _global_rows(plan, source_bytes, workers):
    """Recompute shared observations and capacity accounting from all receipts."""
    run = plan.run
    if type(workers) is not list or len(workers) != run.world_size:
        raise ValueError("MoE evidence worker count differs")
    first = workers[0]
    if type(first) is not dict or set(first) != _RANK_FIELDS:
        raise ValueError("MoE worker receipt fields differ")
    start = _integer(first["started_step"], 0, run.optimizer_steps - 1, "start")
    end = _integer(first["completed_step"], start + 1, run.optimizer_steps, "end")
    _sha(first["model_sha256"])
    _identity(first["checkpoint_identity"])
    pids = set()
    for rank, worker in enumerate(workers):
        if (
            type(worker) is not dict
            or set(worker) != _RANK_FIELDS
            or worker["schema"] != RECEIPT_SCHEMA
            or type(worker["rank"]) is not int
            or worker["rank"] != rank
            or type(worker["world_size"]) is not int
            or worker["world_size"] != run.world_size
            or worker["plan_sha256"] != plan.fingerprint()
            or type(worker["started_step"]) is not int
            or type(worker["completed_step"]) is not int
            or any(
                worker[key] != first[key]
                for key in (
                    "started_step",
                    "completed_step",
                    "model_sha256",
                    "checkpoint_identity",
                )
            )
            or type(worker["metrics"]) is not list
            or len(worker["metrics"]) != end - start
        ):
            raise ValueError("MoE worker receipts disagree on their declared execution")
        pids.add(_integer(worker["pid"], 1, 2**31 - 1, "worker PID"))
    if len(pids) != run.world_size:
        raise ValueError("MoE evidence workers must have distinct process IDs")
    rows = []
    layers = run.model.moe_layers
    experts = run.model.expert_count
    tokens = run.micro_batch_size * run.sequence_length
    total = tokens * run.accumulation_steps * run.world_size
    capacity = max(
        run.model.min_capacity,
        math.ceil(run.model.capacity_factor * tokens * 2 / experts),
    )
    for index in range(end - start):
        observed, schedules, shared = [], [], None
        attempted = [[0] * experts for _ in layers]
        accepted = [[0] * experts for _ in layers]
        probabilities = [[0.0] * experts for _ in layers]
        for rank, worker in enumerate(workers):
            metric = worker["metrics"][index]
            if type(metric) is not dict or set(metric) != _METRIC_FIELDS | {
                "routing",
                "data_schedule",
            }:
                raise ValueError("MoE worker metric fields differ")
            current = {name: metric[name] for name in _METRIC_FIELDS}
            if shared is None:
                shared = current
            elif canonical(current) != canonical(shared):
                raise ValueError("MoE global metric observations disagree across ranks")
            schedule = public_moe_ddp_schedule(
                plan, source_bytes, rank=rank, update_index=start + index
            )
            if canonical(metric["data_schedule"]) != canonical(schedule):
                raise ValueError(
                    "MoE evidence sample schedule differs from the source cursor"
                )
            schedules.append(schedule)
            routing = metric["routing"]
            if type(routing) is not list or len(routing) != run.accumulation_steps:
                raise ValueError("MoE evidence microbatch route count differs")
            for batch in routing:
                if type(batch) is not list or len(batch) != len(layers):
                    raise ValueError("MoE evidence layer route count differs")
                for position, (layer, stats) in enumerate(
                    zip(layers, batch, strict=True)
                ):
                    validate_moe_routing(stats, expert_count=experts)
                    if (
                        stats["layer_index"] != layer
                        or stats["valid_tokens"] != tokens
                        or stats["capacity"] != capacity
                        or stats["capacity_applied"] is not True
                        or stats["token_indices"] != list(range(tokens))
                        or len(stats["accepted_per_token"]) != run.micro_batch_size
                        or any(
                            len(row) != run.sequence_length
                            for row in stats["accepted_per_token"]
                        )
                    ):
                        raise ValueError(
                            "MoE evidence differs from full-token training capacity"
                        )
                    for expert in range(experts):
                        attempted[position][expert] += stats["attempted_counts"][expert]
                        accepted[position][expert] += stats["accepted_counts"][expert]
                        probabilities[position][expert] += stats["probability_sums"][
                            expert
                        ]
            observed.append(routing)
        if (
            type(shared["step"]) is not int
            or shared["step"] != start + index + 1
            or type(shared["target_tokens"]) is not int
            or shared["target_tokens"] != total
            or shared["capacity_scope"] != "rank_microbatch"
            or canonical(shared["attempted_counts"]) != canonical(attempted)
            or canonical(shared["accepted_counts"]) != canonical(accepted)
        ):
            raise ValueError(
                "MoE evidence global progress or route conservation differs"
            )
        for name in (
            "ce_loss",
            "balance_loss",
            "router_z_loss",
            "objective",
            "gradient_norm_before_clip",
            "gradient_norm_after_clip",
        ):
            _number(shared[name])
        balance = (
            experts
            * math.fsum(
                attempted[layer][expert] * probabilities[layer][expert]
                for layer in range(len(layers))
                for expert in range(experts)
            )
            / (2 * len(layers) * total**2)
        )
        expected = (
            shared["ce_loss"]
            + run.balance_weight * shared["balance_loss"]
            + run.z_weight * shared["router_z_loss"]
        )
        if (
            not math.isclose(
                shared["balance_loss"], balance, rel_tol=1e-12, abs_tol=1e-12
            )
            or shared["objective"] != expected
            or shared["gradient_norm_after_clip"]
            > min(shared["gradient_norm_before_clip"], run.max_gradient_norm)
            * (1 + 2e-8)
            + 1e-12
        ):
            raise ValueError("MoE evidence objective or clipping accounting differs")
        _sha(shared["model_sha256"])
        rows.append({**shared, "rank_routing": observed, "rank_schedule": schedules})
    if rows[-1]["model_sha256"] != first["model_sha256"]:
        raise ValueError("MoE evidence final model observation differs")
    return rows


def build_moe_ddp_evidence(plan, source, workers):
    source_admission(plan, source)
    workers = json.loads(canonical(workers))
    metrics = _global_rows(plan, len(source), workers)
    first = workers[0]
    result = {
        "schema": SCHEMA,
        "plan": plan.to_mapping(),
        "plan_sha256": plan.fingerprint(),
        "source_bytes": len(source),
        "source_sha256": sha256(source).hexdigest(),
        "started_step": first["started_step"],
        "completed_step": first["completed_step"],
        "model_sha256": first["model_sha256"],
        "checkpoint_identity": first["checkpoint_identity"],
        "runtime": {
            "device": "cpu",
            "backend": "gloo",
            "dtype": "float64",
            "threads": 1,
            "world_size": plan.run.world_size,
            "torch_version": str(torch.__version__),
        },
        "workers": workers,
        "metrics": metrics,
        "limitations": LIMITATIONS,
    }
    result["evidence_sha256"] = sha256(canonical(result).encode()).hexdigest()
    validate_moe_ddp_evidence(result)
    return result


def validate_moe_ddp_evidence(report):
    fields = {
        "schema",
        "plan",
        "plan_sha256",
        "source_bytes",
        "source_sha256",
        "started_step",
        "completed_step",
        "model_sha256",
        "checkpoint_identity",
        "runtime",
        "workers",
        "metrics",
        "limitations",
        "evidence_sha256",
    }
    if type(report) is not dict or set(report) != fields or report["schema"] != SCHEMA:
        raise ValueError("MoE evidence fields differ")
    encoded = canonical(report).encode()
    if len(encoded) > MAX_REPORT_BYTES:
        raise ValueError("MoE evidence exceeds its retained byte budget")
    if (
        report["evidence_sha256"]
        != sha256(
            canonical(
                {
                    key: value
                    for key, value in report.items()
                    if key != "evidence_sha256"
                }
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("MoE evidence seal differs")
    plan = public_moe_ddp_plan_from_mapping(report["plan"])
    if (
        report["plan_sha256"] != plan.fingerprint()
        or report["source_sha256"] != plan.source.sha256
    ):
        raise ValueError("MoE evidence source/plan identity differs")
    _integer(report["source_bytes"], 2, 1024 * 1024, "source byte count")
    runtime = report["runtime"]
    if (
        type(runtime) is not dict
        or set(runtime)
        != {"device", "backend", "dtype", "threads", "world_size", "torch_version"}
        or runtime["device"] != "cpu"
        or runtime["backend"] != "gloo"
        or runtime["dtype"] != "float64"
        or type(runtime["threads"]) is not int
        or runtime["threads"] != 1
        or type(runtime["world_size"]) is not int
        or runtime["world_size"] != plan.run.world_size
        or type(runtime["torch_version"]) is not str
        or not 1 <= len(runtime["torch_version"]) <= 128
        or report["limitations"] != LIMITATIONS
    ):
        raise ValueError("MoE evidence runtime or limitations differ")
    expected = _global_rows(plan, report["source_bytes"], report["workers"])
    if canonical(report["metrics"]) != canonical(expected):
        raise ValueError("MoE evidence global rows differ from rank observations")
    for key in (
        "started_step",
        "completed_step",
        "model_sha256",
        "checkpoint_identity",
    ):
        if canonical(report[key]) != canonical(report["workers"][0][key]):
            raise ValueError("MoE evidence boundary differs from its rank observations")
    return report


def verify_moe_ddp_archive(report, path):
    """Restore on one CPU only; do not imply replay or live multi-rank evidence."""
    validate_moe_ddp_evidence(report)
    plan = public_moe_ddp_plan_from_mapping(report["plan"])
    payload = load_checkpoint_archive(
        path,
        max_bytes=MAX_BYTES,
        max_records=MAX_CHECKPOINT_RECORDS,
        expected_sha256=report["checkpoint_identity"]["archive_sha256"],
    )
    if type(payload) is not dict or set(payload) != {
        "schema",
        "world_size",
        "config",
        "step",
        "data_cursor",
        "source_sha256",
        "data_contract",
        "inner",
        "rank_rngs",
        "checkpoint_sha256",
    }:
        raise ValueError("MoE evidence checkpoint fields differ")
    if (
        payload["schema"] != CHECKPOINT_SCHEMA
        or type(payload["world_size"]) is not int
        or payload["world_size"] != plan.run.world_size
        or _digest(payload["config"]) != _digest(plan.run.to_mapping())
        or type(payload["step"]) is not int
        or payload["step"] != report["completed_step"]
        or _digest(payload["data_cursor"])
        != _digest({"next_update_index": payload["step"]})
        or payload["source_sha256"] != plan.source.sha256
        or _digest(payload["data_contract"]) != _digest(plan.data_contract())
        or payload["checkpoint_sha256"]
        != report["checkpoint_identity"]["payload_sha256"]
        or payload["checkpoint_sha256"]
        != _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        )
    ):
        raise ValueError("MoE evidence checkpoint contract or payload seal differs")
    rngs = payload["rank_rngs"]
    if type(rngs) is not list or len(rngs) != plan.run.world_size:
        raise ValueError("MoE evidence checkpoint rank RNG count differs")
    for rng in rngs:
        if (
            not isinstance(rng, torch.Tensor)
            or rng.device.type != "cpu"
            or rng.dtype != torch.uint8
            or rng.shape != torch.get_rng_state().shape
        ):
            raise ValueError("MoE evidence checkpoint rank RNG schema differs")
        torch.Generator(device="cpu").set_state(rng)
    with torch.random.fork_rng(devices=[]):
        model = build_training_model(plan.run.model, max_parameters=1_000_000).double()
        optimizer = build_moe_optimizer(
            model, learning_rate=plan.run.learning_rate, config=plan.run.optimizer
        )
        step = restore_moe_payload(
            payload["inner"],
            model=model,
            optimizer=optimizer,
            training_contract=moe_ddp_training_contract(plan.run, plan.data_contract()),
            source_sha256=plan.source.sha256,
        )
        if (
            step != report["completed_step"]
            or validate_moe_optimizer(model, optimizer) != step
            or any(
                group["lr"] != plan.run.learning_rate
                for group in optimizer.param_groups
            )
            or training_model_fingerprint(model) != report["model_sha256"]
        ):
            raise ValueError("MoE evidence restored checkpoint state differs")
    return {
        "archive_sha256": report["checkpoint_identity"]["archive_sha256"],
        "step": step,
        "scope": "single_cpu_typed_restore_not_training_replay",
    }


def read_moe_ddp_evidence(path):
    return validate_moe_ddp_evidence(
        json.loads(_read_bytes(path, MAX_REPORT_BYTES), object_pairs_hook=unique_fields)
    )
