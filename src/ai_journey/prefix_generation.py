"""Real prefix-aware prefill over the existing baseline/modern/native-GQA cache."""

from contextlib import contextmanager

import torch

from .ddp_cache import cached_forward
from .prefix_cache import PrefixKVStore, prefix_identity


@contextmanager
def prefix_prefill(model, tokens, *, store, namespace="default", observation=None):
    """Yield raw next-token logits and KV state while keeping the lease charged."""
    if not isinstance(store, PrefixKVStore):
        raise TypeError("prefix prefill requires a PrefixKVStore")
    if observation is not None and type(observation) is not dict:
        raise TypeError("prefix observation must be a caller-owned dictionary")
    identity = prefix_identity(model, namespace=namespace)
    receipt = {}
    with store.acquire(identity, tokens, observation=receipt) as match:
        try:
            reused = 0 if match is None else match.prefix_length
            suffix = tokens[reused:]
            if not suffix:
                logits, cache = match.last_logits, match.cache
            else:
                output, cache = cached_forward(
                    model,
                    torch.tensor([suffix], dtype=torch.long),
                    None if match is None else match.cache,
                )
                logits = output[:, -1, :]
            published = False
            if suffix:
                published = store.publish(identity, tokens, cache, logits)
            if observation is not None:
                observation.update(
                    outcome=receipt["outcome"],
                    effective_prompt_tokens=len(tokens),
                    reused_prompt_tokens=reused,
                    forwarded_prefill_tokens=len(suffix),
                    prefix_published=published,
                    architecture_sha256=identity.architecture_sha256,
                    native_kv_heads=identity.kv_heads,
                )
            yield logits.unsqueeze(1), cache
        finally:
            # The owned driver must likewise drop its yielded aliases first.
            # Do not retain tensor copies in a suspended frame or traceback
            # after the store closes its scope and releases their charge.
            match = logits = cache = output = None
