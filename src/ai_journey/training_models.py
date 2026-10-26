"""Strict architecture contracts shared by training and checkpoint consumers."""

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256

import torch

from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_model import DecoderVariantModel
from .decoder_variant_resources import expected_decoder_variant_parameter_count
from .gpt2_config import GPT2Config
from .gpt2_model import GPT2LanguageModel, expected_gpt2_parameter_count
from .moe_config import MoEConfig, expected_moe_parameter_count, moe_config_from_mapping
from .moe_model import MoEDecoderModel

TrainingModelConfig = GPT2Config | DecoderVariantConfig | MoEConfig
TrainingModel = GPT2LanguageModel | DecoderVariantModel | MoEDecoderModel

# Parameter counts do not bound nonpersistent T-by-T causal masks or the number
# of Python decoder blocks. Keep these construction costs bounded independently.
MAX_CPU_MODEL_LAYERS = 64
MAX_CPU_ATTENTION_MASK_BYTES = 64 * 1024 * 1024


def base_model_config(value: TrainingModelConfig | TrainingModel) -> GPT2Config:
    """Resolve base dimensions without assuming every architecture has flat fields."""
    if isinstance(value, (GPT2LanguageModel, DecoderVariantModel)):
        value = value.config
    if isinstance(value, DecoderVariantConfig):
        return value.gpt2
    if isinstance(value, GPT2Config):
        return value
    raise TypeError("model contract must be GPT2Config or DecoderVariantConfig")


def expected_training_parameters(config: TrainingModelConfig) -> int:
    if isinstance(config, MoEConfig):
        return expected_moe_parameter_count(config)
    if isinstance(config, DecoderVariantConfig):
        return expected_decoder_variant_parameter_count(config)
    return expected_gpt2_parameter_count(config)


def build_training_model(
    config: TrainingModelConfig, *, max_parameters: int = 5_000_000
) -> TrainingModel:
    """Bound parameters, decoder-block work and peak causal-mask construction."""
    if type(max_parameters) is not int or max_parameters <= 0:
        raise ValueError("max_parameters must be a positive integer")
    if expected_training_parameters(config) > max_parameters:
        raise ValueError("model exceeds the declared parameter budget")
    if isinstance(config, MoEConfig):
        from .checkpoint_archive import MAX_CHECKPOINT_RECORDS
        from .moe_resources import expected_checkpoint_records

        if expected_checkpoint_records(config) > MAX_CHECKPOINT_RECORDS:
            raise ValueError("MoE model exceeds the bounded checkpoint-record budget")
    base = config.gpt2 if isinstance(config, DecoderVariantConfig) else config
    if base.layer_count > MAX_CPU_MODEL_LAYERS:
        raise ValueError(
            f"CPU model construction exceeds the {MAX_CPU_MODEL_LAYERS}-layer work budget"
        )
    # Each block retains a bool mask. torch.tril(torch.ones(..., dtype=bool))
    # also overlaps one input/output pair while constructing the final block:
    # (L - 1) retained masks + the two temporary/current masks = (L + 1) T².
    peak_attention_bytes = (base.layer_count + 1) * base.context_length**2
    if peak_attention_bytes > MAX_CPU_ATTENTION_MASK_BYTES:
        raise ValueError(
            f"CPU attention-mask construction requires {peak_attention_bytes} bytes, exceeding the {MAX_CPU_ATTENTION_MASK_BYTES}-byte budget"
        )
    if isinstance(config, MoEConfig):
        return MoEDecoderModel(config)
    return (
        DecoderVariantModel(config)
        if isinstance(config, DecoderVariantConfig)
        else GPT2LanguageModel(config)
    )


def model_config_mapping(config: TrainingModelConfig) -> dict:
    """Retain legacy GPT-2 mappings and explicitly tag modern architecture choices."""
    base_model_config(config)
    if isinstance(config, MoEConfig):
        return {"kind": "moe_decoder", "config": config.to_mapping()}
    if isinstance(config, DecoderVariantConfig):
        return {"kind": "decoder_variant", "config": asdict(config)}
    return asdict(config)


def _base_from_mapping(payload: Mapping) -> GPT2Config:
    if not isinstance(payload, Mapping) or set(payload) != set(
        GPT2Config.__dataclass_fields__
    ):
        raise ValueError("GPT-2 model configuration fields do not match")
    return GPT2Config(**payload)


def model_config_from_mapping(payload: Mapping) -> TrainingModelConfig:
    """Parse one complete architecture without silently ignoring configuration keys."""
    if not isinstance(payload, Mapping):
        raise ValueError("model configuration must be a mapping")
    if "kind" not in payload:
        return _base_from_mapping(payload)
    if set(payload) == {"kind", "config"} and payload["kind"] == "moe_decoder":
        return moe_config_from_mapping(payload["config"])
    if set(payload) != {"kind", "config"} or payload["kind"] != "decoder_variant":
        raise ValueError("unsupported model architecture contract")
    values = payload["config"]
    if not isinstance(values, Mapping) or set(values) != set(
        DecoderVariantConfig.__dataclass_fields__
    ):
        raise ValueError("decoder variant configuration fields do not match")
    return DecoderVariantConfig(
        **{**values, "gpt2": _base_from_mapping(values["gpt2"])}
    )


def training_model_fingerprint(model: TrainingModel) -> str:
    """Hash architecture and tensor state while retaining baseline fingerprint bytes."""
    if not isinstance(model, (GPT2LanguageModel, DecoderVariantModel)):
        raise TypeError("model must be GPT2LanguageModel or DecoderVariantModel")
    digest = sha256()
    if isinstance(model, MoEDecoderModel):
        digest.update(b"moe-decoder-v1\0")
        digest.update(model.config.fingerprint().encode())
    elif isinstance(model, DecoderVariantModel):
        digest.update(b"decoder-variant-v1\0")
        digest.update(model.config.fingerprint().encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
