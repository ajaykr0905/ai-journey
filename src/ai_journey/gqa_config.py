"""New tagged grouped-query contracts; legacy decoder serialization is untouched."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from collections.abc import Mapping

from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_resources import expected_decoder_variant_parameter_count
from .moe_config import MoEConfig, expected_moe_parameter_count


@dataclass(frozen=True)
class GQAConfig:
    decoder: DecoderVariantConfig | MoEConfig
    kv_heads: int

    def __post_init__(self):
        if type(self.decoder) not in (DecoderVariantConfig, MoEConfig):
            raise TypeError("GQA decoder must be an exact dense or MoE config")
        if self.decoder.position_kind != "rope":
            raise ValueError("grouped-query decoder requires rotary positions")
        if (
            type(self.kv_heads) is not int
            or not 1 <= self.kv_heads <= self.gpt2.head_count
            or self.gpt2.head_count % self.kv_heads
        ):
            raise ValueError("kv_heads must divide query head_count")

    @property
    def gpt2(self):
        return self.decoder.gpt2

    @property
    def position_kind(self):
        return self.decoder.position_kind

    @property
    def rope_base(self):
        return self.decoder.rope_base

    @property
    def expert_count(self):
        return self.decoder.expert_count if isinstance(self.decoder, MoEConfig) else 0

    def to_mapping(self):
        return {
            "schema_version": 1,
            "decoder": {
                "kind": "moe_decoder"
                if isinstance(self.decoder, MoEConfig)
                else "decoder_variant",
                "config": self.decoder.to_mapping()
                if isinstance(self.decoder, MoEConfig)
                else asdict(self.decoder),
            },
            "kv_heads": self.kv_heads,
        }

    def fingerprint(self):
        payload = {"decoder": self.decoder.fingerprint(), "kv_heads": self.kv_heads}
        return sha256(
            b"grouped-query-v1\0"
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def gqa_config_from_mapping(payload: Mapping) -> GQAConfig:
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "decoder", "kv_heads"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("grouped-query config fields/schema do not match")
    nested = payload["decoder"]
    if not isinstance(nested, Mapping) or nested.get("kind") not in (
        "decoder_variant",
        "moe_decoder",
    ):
        raise ValueError("grouped-query config must wrap one dense or MoE architecture")
    from .training_models import model_config_from_mapping

    return GQAConfig(model_config_from_mapping(nested), payload["kv_heads"])


def expected_gqa_parameter_count(config: GQAConfig) -> int:
    if not isinstance(config, GQAConfig):
        raise TypeError("config must be GQAConfig")
    base = (
        expected_moe_parameter_count(config.decoder)
        if isinstance(config.decoder, MoEConfig)
        else expected_decoder_variant_parameter_count(config.decoder)
    )
    width, kv_width = config.gpt2.embedding_dim, config.kv_heads * config.gpt2.head_dim
    return base + config.gpt2.layer_count * 2 * (kv_width - width) * (width + 1)


def cache_head_count(value) -> int:
    """Resolve stored KV heads, not the query heads used by score work."""
    config = getattr(value, "config", value)
    if isinstance(config, GQAConfig):
        return config.kv_heads
    from .training_models import base_model_config

    return base_model_config(config).head_count
