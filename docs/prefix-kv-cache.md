# Bounded CPU prefix snapshots

`PrefixKVStore(max_bytes=..., max_entries=...)` is a process-local, thread-safe
LRU for verified prefill states. It does not cache generated responses or sampler
RNG. `prefix_identity(model, namespace=...)` binds the complete architecture,
weights, dtype, native KV heads, context policy and trusted operator namespace.
The namespace is not authentication or a client-supplied tenant claim.

Publish a real `cached_forward` prefill with its exact token tuple, KV state and
raw `logits[:, -1, :]`. Publication owns compact independent copies. A lease
obtained with `with store.acquire(identity, prompt) as match` owns isolated copies;
do not retain lease values after leaving that context. Exact stored prompts can
reuse the saved raw last-token logits. Branches reuse the longest common prefix.
An interior-prefix request leaves its final token for computation because there
are no saved logits for that earlier position.

Admission reserves compact tensor bytes before cloning. The byte ceiling covers
resident KV/logit snapshots, publication reservations and in-scope leased copies,
not model weights, inference workspace, Python metadata or allocator RSS. Entry
count and context bounds separately limit metadata. Legacy KV views may retain
larger packed-QKV storage than their logical size; owned contiguous copies avoid
charging that hidden backing as if it were compact. Native GQA KV heads are never
expanded for storage. Lack of room bypasses caching rather than overcommitting.

LRU eviction or `clear()` can remove residents while isolated active leases stay
valid and charged until their context exits. Exceptions release reservations and
leases. The store itself neither performs inference nor changes existing HTTP
schemas. CPU parity and byte accounting are evidence, not a speedup or GPU claim.

`generate_tokens(..., use_cache=True, prefix_store=store)` performs real reusable
prefill, preserving its existing return schema. An optional caller-owned
`prefix_observation` dictionary records hit/miss/bypass, effective prompt tokens,
reused prefix tokens and actual forwarded prefill tokens. Sampling uses a new
request-local RNG and applies filters to raw logits; cache reuse never restores
another request's sampler. Only successful prompt prefill is published, never
generated continuations. A later cancelled/failed decode can leave that valid
prefill resident, while its active lease is always released.

The current generator left-truncates overlong prompts and resets positions when
its context fills. Identity explicitly binds that policy. Prefix reuse operates
on the effective initial context only; rollover still rebuilds the same complete
window as uncached generation, rather than slicing old KV and changing semantics.

The existing `ai-journey-ddp serve` operator command accepts
`--prefix-cache-bytes 8388608 --prefix-cache-entries 64 --prefix-namespace default`.
Zero bytes is the default and disables prefix reuse. The namespace comes only
from the trusted operator, never a request field or asserted tenant identity.
This loopback-only service is not an authenticated multi-tenant boundary.

`GenerationService(..., prefix_store=store)` enables the same integration in
Python. Fresh request IDs still consume their full prompt/output token admission
charge and perform new decoding. Whole-response replay remains a separate cache.
`/metrics` exposes fixed hit/miss/bypass counters, reused/forwarded prefill token
counts and bounded pool gauges without request IDs, prompts, model IDs or namespace
labels. Token-generation metrics count prefix-hit decoding as new work, not replay.
Draining clears resident snapshots; active copies remain charged until decoding
or cancellation unwinds, and final admission cleanup clears any late publication.
