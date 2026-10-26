"""Exact stored/cache inventories and bounded grouped-query CPU work admission."""

import torch

from .checkpoint_archive import MAX_CHECKPOINT_RECORDS
from .gqa_config import GQAConfig, expected_gqa_parameter_count
from .grouped_rotary_attention import MAX_GQA_NATIVE_CACHE_BYTES, MAX_GQA_SCORE_ELEMENTS
from .moe_config import MoEConfig
from .moe_resources import expected_checkpoint_records

MAX_GQA_CACHE_BYTES = MAX_GQA_NATIVE_CACHE_BYTES


def gqa_parameter_inventory(config: GQAConfig) -> dict:
    if not isinstance(config, GQAConfig):
        raise TypeError("config must be GQAConfig")
    base, decoder = config.gpt2, config.decoder
    width, layers = base.embedding_dim, base.layer_count
    expert = (
        3 * width * decoder.resolved_swiglu_hidden_dim
        if decoder.mlp_kind == "swiglu"
        else 8 * width**2 + 5 * width
    )
    moe_layers = len(decoder.moe_layers) if isinstance(decoder, MoEConfig) else 0
    experts = decoder.expert_count if isinstance(decoder, MoEConfig) else 0
    kv_width = config.kv_heads * base.head_dim
    groups = {
        "token_embedding": base.vocab_size * width,
        "attention": layers * 2 * (width + 1) * (width + kv_width),
        "block_norms": layers * width * (2 if decoder.norm_kind == "rmsnorm" else 4),
        "final_norm": width * (1 if decoder.norm_kind == "rmsnorm" else 2),
        "dense_ffn": (layers - moe_layers) * expert,
        "expert_ffn": moe_layers * experts * expert,
        "routers": moe_layers * width * experts,
        "output_head_alias": 0,
    }
    if sum(groups.values()) != expected_gqa_parameter_count(config):
        raise ValueError("grouped-query parameter inventory does not reconcile")
    return groups


def estimate_gqa_execution(
    config: GQAConfig,
    *,
    batch_size: int,
    sequence_length: int,
    query_tokens: int | None = None,
    dtype: str = "float32",
) -> dict:
    if not isinstance(config, GQAConfig):
        raise TypeError("config must be GQAConfig")
    if (
        type(batch_size) is not int
        or not 1 <= batch_size <= 64
        or type(sequence_length) is not int
        or not 1 <= sequence_length <= config.gpt2.context_length
    ):
        raise ValueError("grouped-query batch/sequence dimensions exceed their bounds")
    query = sequence_length if query_tokens is None else query_tokens
    if type(query) is not int or not 1 <= query <= sequence_length:
        raise ValueError("query_tokens must fit the total sequence")
    if dtype not in ("float32", "float64"):
        raise ValueError("grouped-query CPU dtype must be float32 or float64")
    element = 4 if dtype == "float32" else 8
    return {
        "parameter_count": expected_gqa_parameter_count(config),
        "parameter_bytes": expected_gqa_parameter_count(config) * element,
        "kv_cache_bytes": 2
        * config.gpt2.layer_count
        * batch_size
        * sequence_length
        * config.kv_heads
        * config.gpt2.head_dim
        * element,
        "score_elements": batch_size * config.gpt2.head_count * query * sequence_length,
        "query_heads": config.gpt2.head_count,
        "kv_heads": config.kv_heads,
        "query_tokens": query,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "dtype": dtype,
        "checkpoint_records": expected_checkpoint_records(config.decoder),
    }


def require_gqa_execution_budget(config, **kwargs):
    estimate = estimate_gqa_execution(config, **kwargs)
    if estimate["kv_cache_bytes"] > MAX_GQA_CACHE_BYTES:
        raise ValueError("grouped-query exceeds the native KV-cache byte budget")
    if estimate["score_elements"] > MAX_GQA_SCORE_ELEMENTS:
        raise ValueError("grouped-query exceeds the query-head score-work budget")
    if estimate["checkpoint_records"] > MAX_CHECKPOINT_RECORDS:
        raise ValueError("grouped-query exceeds the checkpoint-record budget")
    return estimate


def audit_gqa_cache_storage(model, cache) -> dict:
    """Inspect actual CPU backing stores, not just tensor logical dimensions."""
    from .training_models import training_model_fingerprint

    if not isinstance(
        model.config, GQAConfig
    ) or cache.model_sha256 != training_model_fingerprint(model):
        raise ValueError("cache must belong to this grouped-query model")
    if len(cache.keys) != model.config.gpt2.layer_count or len(cache.values) != len(
        cache.keys
    ):
        raise ValueError("cache layer count does not match the grouped-query model")
    reference = cache.keys[0]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 4:
        raise ValueError("cache must have rank-four CPU tensors")
    expected = require_gqa_execution_budget(
        model.config,
        batch_size=reference.shape[0],
        sequence_length=reference.shape[2],
        query_tokens=1,
        dtype=str(reference.dtype).removeprefix("torch."),
    )
    expected_shape = (
        reference.shape[0],
        model.config.kv_heads,
        reference.shape[2],
        model.config.gpt2.head_dim,
    )
    logical, stores = 0, {}
    for tensor in (*cache.keys, *cache.values):
        if (
            tensor.shape != expected_shape
            or tensor.dtype != model.wte.weight.dtype
            or tensor.device.type != "cpu"
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(
                "cache tensor schema does not match grouped-query contract"
            )
        size = tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        if storage.nbytes() != size:
            raise ValueError("cache retains storage outside its native KV dimensions")
        logical += size
        stores[storage.data_ptr()] = storage.nbytes()
    if logical != expected["kv_cache_bytes"]:
        raise ValueError("actual cache bytes do not match the analytical inventory")
    return {
        "model_sha256": cache.model_sha256,
        "logical_bytes": logical,
        "retained_storage_bytes": sum(stores.values()),
        "unique_backing_stores": len(stores),
        "kv_heads": model.config.kv_heads,
        "sequence_length": reference.shape[2],
    }
