"""Validated model-bound key/value cache for GPT-2 inference."""

from dataclasses import dataclass
import math
import torch
from .gpt2_training import gpt2_model_fingerprint
from .training_models import base_model_config
from .gqa_config import cache_head_count


@dataclass(frozen=True)
class KVCache:
    model_sha256: str
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

    def __post_init__(self):
        if (
            not isinstance(self.model_sha256, str)
            or len(self.model_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.model_sha256
            )
        ):
            raise ValueError("cache must be bound to a model SHA-256")
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("cache must contain key/value pairs for every layer")
        reference = self.keys[0]
        for tensor in (*self.keys, *self.values):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 4
                or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype
                or tensor.device.type != "cpu"
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(
                    "cache tensors must have matching finite CPU batch/head/time/width shapes"
                )
        if (
            reference.shape[0] == 0
            or reference.shape[1] == 0
            or reference.shape[3] == 0
        ):
            raise ValueError("cache batch, head and width dimensions must be positive")

    @property
    def length(self) -> int:
        return self.keys[0].shape[2]

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.keys, *self.values)
        )

    def clone(self):
        return KVCache(
            self.model_sha256,
            tuple(tensor.clone() for tensor in self.keys),
            tuple(tensor.clone() for tensor in self.values),
        )


def cached_attention(attention, inputs: torch.Tensor, past=None):
    """Attend new queries over cached prefix keys with offset-aware causal masking."""
    if attention.training:
        raise ValueError("cached attention requires evaluation mode")
    batch, time, channels = inputs.shape
    query, key, value = attention.c_attn(inputs).split(channels, dim=2)
    shape = batch, time, attention.head_count, attention.head_dim
    query, key, value = [
        tensor.view(shape).transpose(1, 2) for tensor in (query, key, value)
    ]
    offset = 0
    if past is not None:
        old_key, old_value = past
        if (
            old_key.shape != old_value.shape
            or old_key.shape[:2] != key.shape[:2]
            or old_key.shape[3] != key.shape[3]
            or old_key.dtype != key.dtype
        ):
            raise ValueError("cached attention prefix is incompatible")
        offset = old_key.shape[2]
        key = torch.cat((old_key, key), dim=2)
        value = torch.cat((old_value, value), dim=2)
    if key.shape[2] > attention.context_length:
        raise ValueError("cached attention exceeds context capacity")
    mask = torch.arange(key.shape[2])[None, :] <= (offset + torch.arange(time))[:, None]
    scores = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dim)
    weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1)
    output = (weights @ value).transpose(1, 2).contiguous().view(batch, time, channels)
    return attention.c_proj(output), (key.detach(), value.detach())


@torch.inference_mode()
def cached_forward(model, token_ids: torch.Tensor, cache: KVCache | None = None):
    """Advance a validated baseline or modern decoder with model-bound cache state."""
    base = base_model_config(model)
    if any(module.training for module in model.modules()):
        raise ValueError("cached decoder requires every module in evaluation mode")
    if (
        token_ids.dtype != torch.long
        or token_ids.ndim != 2
        or token_ids.numel() == 0
        or token_ids.device.type != "cpu"
    ):
        raise ValueError("cached decoder requires nonempty CPU integer token batches")
    if token_ids.min() < 0 or token_ids.max() >= base.vocab_size:
        raise ValueError("cached decoder token id exceeds vocabulary")
    fingerprint = gpt2_model_fingerprint(model)
    offset = 0 if cache is None else cache.length
    if offset + token_ids.shape[1] > base.context_length:
        raise ValueError("cached decoder context capacity exceeded")
    if cache is not None:
        if (
            cache.model_sha256 != fingerprint
            or len(cache.keys) != len(model.h)
            or cache.keys[0].shape[:2] != (token_ids.shape[0], cache_head_count(model))
            or cache.keys[0].shape[-1] != base.head_dim
            or cache.keys[0].dtype != model.wte.weight.dtype
        ):
            raise ValueError("cache does not belong to this decoder and batch")
    if getattr(model.config, "position_kind", None) == "rope":
        past = None if cache is None else tuple(zip(cache.keys, cache.values))
        output, pairs = model.forward_with_cache(token_ids, past_key_values=past)
        return output, KVCache(
            fingerprint,
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
        )
    positions = torch.arange(offset, offset + token_ids.shape[1])
    hidden = model.wte(token_ids) + model.wpe(positions)
    keys, values = [], []
    for index, block in enumerate(model.h):
        past = None if cache is None else (cache.keys[index], cache.values[index])
        attended, state = cached_attention(block.attn, block.ln_1(hidden), past)
        hidden = hidden + attended
        hidden = hidden + block.mlp(block.ln_2(hidden))
        keys.append(state[0])
        values.append(state[1])
    return model.lm_head(model.ln_f(hidden)), KVCache(
        fingerprint, tuple(keys), tuple(values)
    )
