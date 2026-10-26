"""Strict controlled KV-grouping plans and complete CPU job admission."""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math

from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_optimizer import DecoderOptimizerConfig
from .decoder_variant_resources import expected_decoder_variant_parameter_count
from .gqa_config import GQAConfig, expected_gqa_parameter_count
from .gqa_resources import require_gqa_execution_budget
from .training_models import model_config_from_mapping, model_config_mapping

MAX_EXPERIMENT_WORK_UNITS = 100_000_000


@dataclass(frozen=True)
class GQAExperimentPlan:
    decoder: DecoderVariantConfig
    kv_heads: tuple[int, ...]
    seeds: tuple[int, ...] = (53,)
    steps: int = 2
    batch_size: int = 2
    evaluation_batch_size: int = 4
    learning_rate: float = 0.001
    max_gradient_norm: float = 1.0
    optimizer: DecoderOptimizerConfig = DecoderOptimizerConfig()

    def __post_init__(self):
        if type(self.decoder) is not DecoderVariantConfig:
            raise ValueError(
                "controlled GQA experiment requires an exact dense decoder"
            )
        base = self.decoder.gpt2
        if (
            self.decoder.position_kind != "rope"
            or base.vocab_size != 256
            or not 2 <= base.context_length <= 16
            or not 4 <= base.embedding_dim <= 32
            or not 2 <= base.head_count <= 8
            or not 1 <= base.layer_count <= 2
            or any(
                (base.embedding_dropout, base.attention_dropout, base.residual_dropout)
            )
        ):
            raise ValueError(
                "controlled GQA experiment exceeds its zero-dropout byte protocol"
            )
        if (
            type(self.kv_heads) is not tuple
            or not 2 <= len(self.kv_heads) <= 4
            or any(type(value) is not int for value in self.kv_heads)
            or self.kv_heads != tuple(sorted(set(self.kv_heads), reverse=True))
            or self.kv_heads[0] != base.head_count
        ):
            raise ValueError(
                "GQA arms require descending unique KV heads with full-MHA first"
            )
        for value in self.kv_heads:
            GQAConfig(self.decoder, value)
        if (
            type(self.seeds) is not tuple
            or not 1 <= len(self.seeds) <= 4
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                type(seed) is not int or not 0 <= seed < 2**63 for seed in self.seeds
            )
        ):
            raise ValueError("GQA seeds must be bounded unique exact integers")
        for name, maximum in (
            ("steps", 16),
            ("batch_size", 8),
            ("evaluation_batch_size", 16),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid GQA experiment {name}")
        for name, maximum in (("learning_rate", 0.1), ("max_gradient_norm", 100.0)):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not 0 < value <= maximum
                or not math.isfinite(value)
            ):
                raise ValueError(f"invalid GQA experiment {name}")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.optimizer, DecoderOptimizerConfig):
            raise ValueError("GQA experiment requires explicit optimizer controls")

    @property
    def arms(self):
        return tuple(GQAConfig(self.decoder, value) for value in self.kv_heads)

    def to_mapping(self):
        return {
            **asdict(self),
            "decoder": model_config_mapping(self.decoder),
            "kv_heads": list(self.kv_heads),
            "seeds": list(self.seeds),
        }

    def fingerprint(self):
        return sha256(
            json.dumps(
                self.to_mapping(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


def gqa_experiment_plan_from_mapping(payload):
    if type(payload) is not dict or set(payload) != {
        field.name for field in fields(GQAExperimentPlan)
    }:
        raise ValueError("GQA experiment plan fields differ")
    if type(payload["kv_heads"]) is not list or type(payload["seeds"]) is not list:
        raise ValueError("GQA plan arrays must retain their exact schema")
    if type(payload["optimizer"]) is not dict or set(payload["optimizer"]) != {
        field.name for field in fields(DecoderOptimizerConfig)
    }:
        raise ValueError("GQA optimizer fields differ")
    return GQAExperimentPlan(
        **{
            **payload,
            "decoder": model_config_from_mapping(payload["decoder"]),
            "optimizer": DecoderOptimizerConfig(**payload["optimizer"]),
            "kv_heads": tuple(payload["kv_heads"]),
            "seeds": tuple(payload["seeds"]),
        }
    )


def require_gqa_experiment_budget(plan, *, evaluation_windows, cache_prefix_lengths=()):
    """Charge all arms/seeds and optional replay-backed prefix audits up front.

    Parameter-token/update/hash work units are declared admission proxies, not
    FLOPs, timing, complete allocator peaks or GPU memory. Cache/score inventories
    additionally use the core per-operation bounds without reducing query heads.
    """
    if not isinstance(plan, GQAExperimentPlan):
        raise ValueError("GQA job requires a validated plan")
    if type(evaluation_windows) is not int or not 1 <= evaluation_windows <= 8192:
        raise ValueError("invalid GQA evaluation window count")
    if (
        type(cache_prefix_lengths) is not tuple
        or len(cache_prefix_lengths) > 4
        or any(
            type(value) is not int or not 1 <= value <= plan.decoder.gpt2.context_length
            for value in cache_prefix_lengths
        )
    ):
        raise ValueError("invalid GQA cache prefix workload")
    context = plan.decoder.gpt2.context_length
    reference = expected_decoder_variant_parameter_count(plan.decoder)
    parameters = [expected_gqa_parameter_count(config) for config in plan.arms]
    if max(parameters) > 2_000_000:
        raise ValueError("GQA job exceeds stored parameter budget")
    executions = [
        require_gqa_execution_budget(
            config,
            batch_size=max(
                plan.batch_size, min(plan.evaluation_batch_size, evaluation_windows)
            ),
            sequence_length=context,
        )
        for config in plan.arms
    ]
    forward = len(plan.seeds) * sum(
        value * context * (3 * plan.steps * plan.batch_size + 2 * evaluation_windows)
        for value in parameters
    )
    optimizer = len(plan.seeds) * sum(10 * value * plan.steps for value in parameters)
    initialization = len(plan.seeds) * len(parameters) * 2 * reference
    # Each prefix: full and incremental forward tokens, plus two model-state
    # hashes per one-token cached step (cached_forward + storage audit).
    cache = len(plan.seeds) * sum(
        4 * value * sum(cache_prefix_lengths) for value in parameters
    )
    total = forward + optimizer + initialization + cache
    if total > MAX_EXPERIMENT_WORK_UNITS:
        raise ValueError("GQA job exceeds complete work_units budget")
    return {
        "model_parameters": parameters,
        "seed_count": len(plan.seeds),
        "trial_count": len(parameters) * len(plan.seeds),
        "forward_backward_work_units": forward,
        "optimizer_work_units": optimizer,
        "initialization_work_units": initialization,
        "cache_audit_work_units": cache,
        "work_units": total,
        "peak_initializer_model_parameters": 2 * reference,
        "peak_score_elements": max(row["score_elements"] for row in executions),
        "peak_native_cache_bytes": max(row["kv_cache_bytes"] for row in executions),
        "checkpoint_records": max(row["checkpoint_records"] for row in executions),
    }
