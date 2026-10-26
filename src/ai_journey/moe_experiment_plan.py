"""Explicit bounded controls for dense-versus-MoE document experiments."""

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
import re

from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_optimizer import DecoderOptimizerConfig
from .moe_config import MoEConfig, expected_moe_parameter_count, moe_config_from_mapping
from .moe_resources import require_moe_job_budget
from .training_models import model_config_from_mapping, model_config_mapping

PLAN_SCHEMA = "ai-journey-moe-experiment-plan-v1"


def _number(value, name, minimum, maximum):
    if (
        type(value) not in {int, float}
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError(f"invalid finite experiment {name}")
    return float(value)


@dataclass(frozen=True)
class MoEExperimentArm:
    name: str
    config: DecoderVariantConfig
    balance_weight: float = 0.0
    z_weight: float = 0.0

    def __post_init__(self):
        if (
            type(self.name) is not str
            or re.fullmatch(r"[a-z][a-z0-9-]{0,47}", self.name) is None
            or type(self.config) not in {DecoderVariantConfig, MoEConfig}
        ):
            raise ValueError(
                "experiment arms require bounded names and concrete configs"
            )
        for name in ("balance_weight", "z_weight"):
            object.__setattr__(self, name, _number(getattr(self, name), name, 0, 1))
        if not isinstance(self.config, MoEConfig) and (
            self.balance_weight or self.z_weight
        ):
            raise ValueError("dense arms cannot apply inactive router objectives")

    def to_mapping(self):
        return {
            "name": self.name,
            "model": (
                {"kind": "moe", "config": self.config.to_mapping()}
                if isinstance(self.config, MoEConfig)
                else model_config_mapping(self.config)
            ),
            "balance_weight": self.balance_weight,
            "z_weight": self.z_weight,
        }


@dataclass(frozen=True)
class MoEExperimentPlan:
    arms: tuple[MoEExperimentArm, ...]
    seeds: tuple[int, ...] = (53,)
    steps: int = 2
    batch_size: int = 2
    evaluation_batch_size: int = 4
    learning_rate: float = 0.001
    max_gradient_norm: float = 1.0
    optimizer: DecoderOptimizerConfig = DecoderOptimizerConfig()

    def __post_init__(self):
        if (
            type(self.arms) is not tuple
            or not 2 <= len(self.arms) <= 4
            or any(not isinstance(arm, MoEExperimentArm) for arm in self.arms)
            or self.arms[0].name != "baseline"
            or type(self.arms[0].config) is not DecoderVariantConfig
            or any(not isinstance(arm.config, MoEConfig) for arm in self.arms[1:])
            or len({arm.name for arm in self.arms}) != len(self.arms)
        ):
            raise ValueError(
                "plan requires a dense baseline then one to three unique MoE arms"
            )
        identities = [
            json.dumps(arm.to_mapping() | {"name": ""}, sort_keys=True)
            for arm in self.arms
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("experiment arms must deliver distinct interventions")
        baseline = self.arms[0].config
        for arm in self.arms[1:]:
            if any(
                getattr(arm.config, field.name) != getattr(baseline, field.name)
                for field in fields(DecoderVariantConfig)
            ):
                raise ValueError("MoE arms must share every dense architecture control")
            if (
                arm.config.expert_count > 8
                or expected_moe_parameter_count(arm.config) > 2_000_000
            ):
                raise ValueError(
                    "MoE arm exceeds the bounded expert/parameter contract"
                )
        base = baseline.gpt2
        if (
            base.vocab_size != 256
            or not 2 <= base.context_length <= 32
            or not 4 <= base.embedding_dim <= 32
            or not 1 <= base.layer_count <= 2
            or any(
                (base.embedding_dropout, base.attention_dropout, base.residual_dropout)
            )
        ):
            raise ValueError("experiment requires bounded zero-dropout byte decoders")
        if (
            baseline.resolved_swiglu_hidden_dim is not None
            and baseline.resolved_swiglu_hidden_dim > 4 * base.embedding_dim
        ):
            raise ValueError("experiment exceeds the bounded SwiGLU width")
        if (
            type(self.seeds) is not tuple
            or not 1 <= len(self.seeds) <= 4
            or any(
                type(seed) is not int or not 0 <= seed <= 2**63 - 1
                for seed in self.seeds
            )
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("experiment requires one to four distinct bounded seeds")
        for name, maximum in (
            ("steps", 32),
            ("batch_size", 8),
            ("evaluation_batch_size", 16),
        ):
            if (
                type(getattr(self, name)) is not int
                or not 1 <= getattr(self, name) <= maximum
            ):
                raise ValueError(f"invalid bounded experiment {name}")
        object.__setattr__(
            self,
            "learning_rate",
            _number(self.learning_rate, "learning_rate", 1e-9, 0.1),
        )
        object.__setattr__(
            self,
            "max_gradient_norm",
            _number(self.max_gradient_norm, "max_gradient_norm", 1e-9, 1000),
        )
        if not isinstance(self.optimizer, DecoderOptimizerConfig):
            raise TypeError("experiment optimizer must be DecoderOptimizerConfig")
        require_moe_experiment_work(self, evaluation_windows=1)

    def to_mapping(self):
        from dataclasses import asdict

        return {
            "schema": PLAN_SCHEMA,
            "arms": [arm.to_mapping() for arm in self.arms],
            "seeds": list(self.seeds),
            "steps": self.steps,
            "batch_size": self.batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "learning_rate": self.learning_rate,
            "max_gradient_norm": self.max_gradient_norm,
            "optimizer": asdict(self.optimizer),
        }

    def fingerprint(self):
        return sha256(
            json.dumps(
                self.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()


def require_moe_experiment_work(plan, *, evaluation_windows):
    """Bound every arm/seed and both held-out passes, not only one trial.

    This conservative work proxy is an admission control, not a FLOP, timing,
    memory or GPU claim. Charge the actual document-local window count before
    constructing any experiment model, including every partially filled tail.
    """

    return require_moe_job_budget(
        tuple(arm.config for arm in plan.arms),
        seeds=len(plan.seeds),
        steps=plan.steps,
        batch_size=plan.batch_size,
        evaluation_windows=evaluation_windows,
        evaluation_batch_size=plan.evaluation_batch_size,
    )


def moe_experiment_plan_from_mapping(payload):
    if (
        type(payload) is not dict
        or set(payload)
        != {field.name for field in fields(MoEExperimentPlan)} | {"schema"}
        or payload["schema"] != PLAN_SCHEMA
    ):
        raise ValueError("MoE experiment plan fields or schema differ")
    if type(payload["arms"]) is not list or type(payload["seeds"]) is not list:
        raise ValueError("experiment arms and seeds must be lists")
    if type(payload["optimizer"]) is not dict or set(payload["optimizer"]) != {
        field.name for field in fields(DecoderOptimizerConfig)
    }:
        raise ValueError("experiment optimizer fields differ")
    arms = []
    for arm in payload["arms"]:
        if (
            type(arm) is not dict
            or set(arm) != {"name", "model", "balance_weight", "z_weight"}
            or type(arm["model"]) is not dict
        ):
            raise ValueError("experiment arm fields differ")
        model = arm["model"]
        if model.get("kind") == "moe":
            if set(model) != {"kind", "config"}:
                raise ValueError("MoE model mapping fields differ")
            config = moe_config_from_mapping(model["config"])
        else:
            config = model_config_from_mapping(model)
        arms.append(
            MoEExperimentArm(
                arm["name"], config, arm["balance_weight"], arm["z_weight"]
            )
        )
    return MoEExperimentPlan(
        arms=tuple(arms),
        seeds=tuple(payload["seeds"]),
        optimizer=DecoderOptimizerConfig(**payload["optimizer"]),
        **{
            key: value
            for key, value in payload.items()
            if key not in {"schema", "arms", "seeds", "optimizer"}
        },
    )
