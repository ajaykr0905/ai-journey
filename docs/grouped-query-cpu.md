# Grouped-query rotary CPU lab

`GQAConfig(decoder, kv_heads)` wraps an exact `DecoderVariantConfig` or `MoEConfig`
using RoPE. K/V heads must divide the query-head count. Existing configuration
dataclasses, serialized defaults and baseline fingerprint bytes are unchanged;
the new `grouped_query` tag and `decoder.` tensor namespace are explicit.

`GroupedQueryModel` reuses existing attention-adjacent blocks, experts and tied
head. Queries keep their original head count; K/V projections have only
`kv_heads * head_dim` channels each. Computation expands groups temporarily for
the CPU reference, but returned caches retain native K/V heads and own compact
backing storage. Reducing K/V heads does **not** reduce the query-score matrix.

Use `build_training_model` for parameter, block, mask-construction and checkpoint
record admission. Full/cached grouped forward additionally bounds native cache
inventory to 64 MiB and query-score work to eight million elements before
embedding/projection. These are independent reference limits, not a full
allocator peak or GPU memory estimate. `gqa_parameter_inventory` reconciles all
stored parameters; `estimate_gqa_execution` models cache/score dimensions;
`audit_gqa_cache_storage(model, cache)` inspects actual retained CPU backing bytes.

`train_gqa_step` uses the same explicit CPU AdamW grouping/controls, clipping and
failure rollback as the expert reference. Dense wrappers have zero auxiliary
terms; expert wrappers retain live balance/z losses and training-only capacity
with provably idle expert gradients zero-filled. Normal eval remains dropless.
Masks control routing/loss eligibility, not attention-key masking.

`save_gqa_checkpoint`, `load_gqa_checkpoint` and `load_gqa_model_checkpoint` reuse
the bounded, sealed, schema-checked archive protocol. Callers supply original
source bytes/digest and the complete training contract, including batch order.
KV grouping and rotary base are bound even when tensor shapes could otherwise
match. Frozen loading validates optimizer/restart state, preserves caller RNG,
and enables model-bound cache generation. Old GPT-2 weight import does not imply
GQA compatibility and is rejected for this explicit architecture.

Numerical/gradient/cache/storage/restart checks establish CPU reference behavior.
They establish no GPU execution, inference-speed improvement, distributed MoE
scaling, public benchmark improvement, production readiness or learner activity.
