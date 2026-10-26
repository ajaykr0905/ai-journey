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

## Operate the dense grouped-query control

`config/gqa-cpu-lab.json` declares a new version-three training specification:
two query heads, one K/V head, RMSNorm/SwiGLU/RoPE and four tiny CPU updates on
the existing SHA-pinned, CC0 synthetic public fixture. Version-one GPT-2 and
version-two decoder specifications are unchanged. This generic training route
accepts dense GQA only; expert wrappers require their explicit auxiliary-loss
training interface and are rejected before allocation.

```bash
ai-journey-gpt2-train train --source data/day-19-demo-names.txt \
  --spec config/gqa-cpu-lab.json --checkpoint /tmp/gqa-training.pt \
  --report /tmp/gqa-training.json
ai-journey-gpt2-train verify --report /tmp/gqa-training.json
ai-journey-gpt2-train replay --source data/day-19-demo-names.txt \
  --report /tmp/gqa-training.json
ai-journey-ddp recover-crash --source data/day-19-demo-names.txt \
  --spec config/gqa-cpu-lab.json --output /tmp/gqa-recovery.json \
  --workdir /tmp/gqa-recovery-owned
ai-journey-ddp verify-crash-recovery --report /tmp/gqa-recovery.json \
  --workdir /tmp/gqa-recovery-owned
```

The dedicated CI job builds an isolated wheel and runs these installed commands,
including real two-rank Gloo loss/restart and artifact verification. The crash
control observes step three, restores step two, and verifies exact final state;
it is neither GPU nor expert-parallel execution.

Numerical/gradient/cache/storage/restart checks establish CPU reference behavior.
They establish no GPU execution, inference-speed improvement, distributed MoE
scaling, public benchmark improvement, production readiness or learner activity.
