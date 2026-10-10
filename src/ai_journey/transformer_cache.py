"""Validated, restartable inference caches for the small decoder lab."""

from __future__ import annotations

import re

import torch
from torch import Tensor

from .transformer_lab import TransformerLabError


class LayerKV:
    """Own immutable copies of batched head-major keys and values."""

    __slots__ = ("_keys", "_values")

    def __init__(self, keys: Tensor, values: Tensor) -> None:
        if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
            raise TypeError("keys and values must be tensors")
        if keys.ndim != 4 or any(size <= 0 for size in keys.shape):
            raise TransformerLabError(
                "keys must have nonempty [batch, heads, time, dim] shape"
            )
        if (
            keys.shape != values.shape
            or keys.dtype != values.dtype
            or keys.device != values.device
        ):
            raise TransformerLabError(
                "keys and values must match shape, dtype, and device"
            )
        if (
            not keys.is_floating_point()
            or not torch.isfinite(keys).all()
            or not torch.isfinite(values).all()
        ):
            raise TransformerLabError(
                "cache tensors must be finite floating-point values"
            )
        self._keys = keys.detach().clone()
        self._values = values.detach().clone()

    @property
    def keys(self) -> Tensor:
        return self._keys.clone()

    @property
    def values(self) -> Tensor:
        return self._values.clone()


class DecoderCache:
    """Own token history and per-layer state bound to one model version."""

    __slots__ = ("_tokens", "_layers", "_config_digest", "_model_digest")

    def __init__(
        self,
        tokens: Tensor,
        layers: tuple[LayerKV, ...],
        *,
        config_digest: str,
        model_digest: str,
    ) -> None:
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 2
            or tokens.dtype != torch.long
        ):
            raise TypeError("tokens must be a two-dimensional torch.long tensor")
        if any(size <= 0 for size in tokens.shape) or torch.any(tokens < 0):
            raise TransformerLabError("cache tokens must be nonempty and nonnegative")
        if (
            not isinstance(layers, tuple)
            or not layers
            or not all(isinstance(layer, LayerKV) for layer in layers)
        ):
            raise TypeError("layers must be a nonempty tuple of LayerKV states")
        shape = layers[0]._keys.shape
        for layer in layers:
            if (
                layer._keys.shape != shape
                or layer._keys.dtype != layers[0]._keys.dtype
                or layer._keys.device != tokens.device
            ):
                raise TransformerLabError(
                    "cache layers must agree on shape, dtype, and token device"
                )
        if (shape[0], shape[2]) != tuple(tokens.shape):
            raise TransformerLabError("cache layers must match token batch and time")
        for digest in (config_digest, model_digest):
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise TransformerLabError("cache identity must be a SHA-256 digest")
        self._tokens = tokens.detach().clone()
        self._layers = tuple(LayerKV(layer._keys, layer._values) for layer in layers)
        self._config_digest = config_digest
        self._model_digest = model_digest

    @property
    def tokens(self) -> Tensor:
        return self._tokens.clone()

    @property
    def layers(self) -> tuple[LayerKV, ...]:
        return tuple(LayerKV(layer._keys, layer._values) for layer in self._layers)

    @property
    def config_digest(self) -> str:
        return self._config_digest

    @property
    def model_digest(self) -> str:
        return self._model_digest
