"""Strict, fingerprinted plans for bounded decoder architecture experiments."""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
import re

from .decoder_variant_budget import require_comparison_budget
from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_optimizer import DecoderOptimizerConfig
from .gpt2_config import GPT2Config

PLAN_SCHEMA = "ai-journey-decoder-plan-v1"
_ARCHITECTURE_FIELDS = {
    "norm_kind",
    "mlp_kind",
    "position_kind",
    "rope_base",
    "swiglu_hidden_dim",
}


@dataclass(frozen=True)
class DecoderExperimentPlan:
    variants: tuple[tuple[str, DecoderVariantConfig], ...]
    declared_changes: tuple[str, ...] = ("norm_kind", "mlp_kind", "position_kind")
    seeds: tuple[int, ...] = (52,)
    steps: int = 1
    batch_size: int = 2
    learning_rate: float = 0.001
    evaluation_batch_size: int = 16
    max_gradient_norm: float = 1.0
    optimizer: DecoderOptimizerConfig = DecoderOptimizerConfig()

    def __post_init__(self):
        if type(self.variants) is not tuple or not 2 <= len(self.variants) <= 8:
            raise ValueError("plan requires two to eight architecture arms")
        if any(
            type(row) is not tuple
            or len(row) != 2
            or not isinstance(row[0], str)
            or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", row[0]) is None
            or not isinstance(row[1], DecoderVariantConfig)
            for row in self.variants
        ):
            raise ValueError(
                "plan arms require bounded names and decoder configurations"
            )
        if (
            len({name for name, _ in self.variants}) != len(self.variants)
            or self.variants[0][0] != "baseline"
        ):
            raise ValueError("plan requires unique arm names with baseline first")
        if (
            type(self.declared_changes) is not tuple
            or not self.declared_changes
            or len(set(self.declared_changes)) != len(self.declared_changes)
            or not set(self.declared_changes) <= _ARCHITECTURE_FIELDS
        ):
            raise ValueError(
                "plan declared_changes must identify distinct architecture controls"
            )
        baseline = self.variants[0][1]
        if baseline != DecoderVariantConfig(baseline.gpt2):
            raise ValueError("plan baseline must use the unchanged GPT-2 architecture")
        if len({variant.fingerprint() for _, variant in self.variants}) != len(
            self.variants
        ):
            raise ValueError("plan architecture arms must be distinct")
        for _, variant in self.variants[1:]:
            if variant.gpt2 != baseline.gpt2:
                raise ValueError("plan arms must share the GPT-2 base controls")
            changes = {
                name
                for name in _ARCHITECTURE_FIELDS
                if getattr(variant, name) != getattr(baseline, name)
            }
            if not changes <= set(self.declared_changes):
                raise ValueError("plan arm changes undeclared architecture controls")
            hidden = variant.resolved_swiglu_hidden_dim
            if hidden is not None and hidden > 4 * baseline.gpt2.embedding_dim:
                raise ValueError("plan SwiGLU width exceeds the bounded CPU contract")
        base = baseline.gpt2
        for name, minimum, maximum in (
            ("context_length", 2, 64),
            ("embedding_dim", 4, 64),
            ("head_count", 1, 16),
            ("layer_count", 1, 2),
        ):
            if not minimum <= getattr(base, name) <= maximum:
                raise ValueError(f"plan {name} exceeds the bounded CPU contract")
        if base.vocab_size != 256 or any(
            (base.embedding_dropout, base.attention_dropout, base.residual_dropout)
        ):
            raise ValueError("plan requires 256 byte tokens and zero dropout")
        if (
            type(self.seeds) is not tuple
            or not 1 <= len(self.seeds) <= 4
            or any(
                type(seed) is not int or not 0 <= seed <= 2**63 - 1
                for seed in self.seeds
            )
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError(
                "plan seeds must contain one to four distinct bounded integers"
            )
        for name, maximum in (
            ("steps", 64),
            ("batch_size", 32),
            ("evaluation_batch_size", 32),
        ):
            if (
                type(getattr(self, name)) is not int
                or not 1 <= getattr(self, name) <= maximum
            ):
                raise ValueError(f"invalid plan {name}")
        for name, maximum in (("learning_rate", 0.1), ("max_gradient_norm", 1000)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(f"invalid plan {name}")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.optimizer, DecoderOptimizerConfig):
            raise TypeError("plan optimizer must be DecoderOptimizerConfig")
        require_comparison_budget(
            base,
            steps=self.steps,
            batch_size=self.batch_size,
            evaluation_windows=self.evaluation_batch_size,
            variant_count=len(self.variants),
        )

    def to_mapping(self) -> dict:
        return {
            "schema": PLAN_SCHEMA,
            "variants": [
                {"name": name, "config": asdict(config)}
                for name, config in self.variants
            ],
            "declared_changes": list(self.declared_changes),
            "seeds": list(self.seeds),
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "evaluation_batch_size": self.evaluation_batch_size,
            "max_gradient_norm": self.max_gradient_norm,
            "optimizer": asdict(self.optimizer),
        }

    def fingerprint(self) -> str:
        return sha256(
            json.dumps(
                self.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()


def decoder_plan_from_mapping(payload: dict) -> DecoderExperimentPlan:
    names = {field.name for field in fields(DecoderExperimentPlan)} | {"schema"}
    if (
        type(payload) is not dict
        or set(payload) != names
        or payload["schema"] != PLAN_SCHEMA
    ):
        raise ValueError("decoder plan fields or schema are invalid")
    if (
        type(payload["variants"]) is not list
        or type(payload["seeds"]) is not list
        or type(payload["declared_changes"]) is not list
    ):
        raise ValueError("plan variants, seeds and declared_changes must be lists")
    variants = []
    for row in payload["variants"]:
        if (
            type(row) is not dict
            or set(row) != {"name", "config"}
            or type(row["config"]) is not dict
            or set(row["config"])
            != {field.name for field in fields(DecoderVariantConfig)}
        ):
            raise ValueError("plan arm fields are invalid")
        config = dict(row["config"])
        if type(config["gpt2"]) is not dict or set(config["gpt2"]) != {
            field.name for field in fields(GPT2Config)
        }:
            raise ValueError("plan GPT-2 config must contain every field")
        config["gpt2"] = GPT2Config(**config["gpt2"])
        variants.append((row["name"], DecoderVariantConfig(**config)))
    values = {
        name: value
        for name, value in payload.items()
        if name not in {"schema", "variants", "seeds", "declared_changes", "optimizer"}
    }
    return DecoderExperimentPlan(
        variants=tuple(variants),
        seeds=tuple(payload["seeds"]),
        declared_changes=tuple(payload["declared_changes"]),
        optimizer=DecoderOptimizerConfig(**payload["optimizer"]),
        **values,
    )
