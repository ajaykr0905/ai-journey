"""Strict preregistration contract for analytical inference-capacity reports."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .inference_capacity import DeviceMemoryBudget
from .inference_kv import InferenceCapacityError, KVCacheLayout, _positive_integer
from .inference_speculative import SpeculativePlan


@dataclass(frozen=True)
class InferenceProtocol:
    name: str
    parameter_count: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    context_tokens: int
    request_lengths: tuple[int, ...]
    bytes_per_element: int
    total_device_bytes: int
    model_bytes: int
    runtime_reserve_bytes: int
    block_tokens: int
    speculative: SpeculativePlan

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or re.fullmatch(r"[a-z0-9-]{1,64}", self.name) is None
        ):
            raise InferenceCapacityError("name must be a bounded lowercase identifier")
        _positive_integer(self.parameter_count, "parameter_count")
        if type(self.request_lengths) is not tuple or not self.request_lengths:
            raise InferenceCapacityError("request_lengths must be a nonempty tuple")
        _layout = self.kv_layout
        _budget = self.device_budget
        _positive_integer(self.query_heads, "query_heads")
        if self.kv_heads > self.query_heads or self.query_heads % self.kv_heads:
            raise InferenceCapacityError("kv_heads must divide query_heads")
        _positive_integer(self.block_tokens, "block_tokens")
        if any(
            type(length) is not int or not 1 <= length <= self.context_tokens
            for length in self.request_lengths
        ):
            raise InferenceCapacityError("request lengths must fit the model context")
        if not isinstance(self.speculative, SpeculativePlan):
            raise TypeError("speculative must be SpeculativePlan")

    @property
    def kv_layout(self) -> KVCacheLayout:
        return KVCacheLayout(
            self.layers,
            self.kv_heads,
            self.head_dim,
            self.context_tokens,
            len(self.request_lengths),
            self.bytes_per_element,
        )

    @property
    def device_budget(self) -> DeviceMemoryBudget:
        return DeviceMemoryBudget(
            self.total_device_bytes,
            self.model_bytes,
            self.runtime_reserve_bytes,
        )
