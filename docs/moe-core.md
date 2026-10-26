# Bounded CPU mixture-of-experts reference

`MoEConfig` replaces selected decoder FFNs with a bias-free top-two router and
GELU or SwiGLU experts. The inherited attention, norms, positions and tied token
head retain their existing implementation. Total stored parameter count includes
every expert; `expected_moe_active_parameters` is only a per-token upper bound,
not optimizer storage or a memory estimate. Use `build_training_model` to enforce
the existing parameter, block and causal-mask construction budgets.
The model factory and checkpoint writer additionally reject architectures whose
conservative model/AdamW archive estimate exceeds 16,384 records. MoE readers
explicitly admit up to that record count while retaining byte, compression,
duplicate-record and strict post-load state checks; legacy readers keep 1,024.

## Routing and capacity

Exact gate ties prefer lower expert IDs. Selected weights sum to one before
capacity. Training admits routes token-major, first expert before second, using
`max(min_capacity, ceil(capacity_factor * valid_tokens * 2 / expert_count))` slots
per expert. Dropped routes contribute zero without renormalizing survivors.
Capacity is per forward, not invariant to batch/chunk partitioning. Eval is
explicitly dropless; cache decoding requires eval mode. Training capacity probes
must be reported separately from ordinary held-out evaluation.

The optional boolean token mask excludes rows before router projection, capacity,
auxiliary denominators and firing counts. It is a routing/loss eligibility mask,
not an attention-key padding mask. Public document evaluators use right padding.
The model returns logits normally; `forward_training` additionally returns live
balance/z losses, exact detached route evidence and idle-expert parameter names.
No mutable last-forward diagnostic state is stored on the model.

The balancing term is `E * sum(mean_probability[e] * attempted_fraction[e])`,
where attempted fractions count both selected routes before capacity. The z term
is mean squared `logsumexp` over valid router rows; model auxiliary terms average
across MoE layers. Empty masked rows contribute neither exposure nor loss.

## CPU updates and exact restart

`build_moe_optimizer` delegates to the dense comparison's explicit AdamW builder.
`train_moe_step` minimizes CE plus explicitly weighted balance and z terms. Only
parameters of experts receiving zero accepted routes may have absent gradients
filled with zeros. AdamW still updates/decays every parameter each step, including
idle experts. This is sparse forward compute, **not** sparse optimizer memory,
sparse weight decay, expert parallelism or a GPU kernel. Active missing/nonfinite
gradients remain errors. Failed updates restore weights, optimizer, RNG, existing
gradients and mixed module modes; successful updates advance the caller RNG.

`save_moe_checkpoint` atomically seals model, optimizer, global update count,
CPU RNG, source digest and the complete caller-declared training contract.
The caller owns data scheduling: the contract must declare batch order and the
saved step must identify its next batch. No sampler is invented or implicitly
recovered. Loading requires the same controls, tensor schema and source digest;
malformed archives and rejected restores cannot partially mutate live state.

`load_moe_model_checkpoint` additionally requires original source bytes, validates
the complete restart state, preserves caller RNG and returns a frozen dropless
eval model. Archive and parameter byte/work limits still apply. Self-sealing
detects internal corruption; it does not authenticate authorship, public licensing
or learner activity. The existing GPT-2 trainer rejects MoE to prevent silently
omitting auxiliary objectives; use the explicit MoE update/experiment path.

All contracts here are CPU reference checks. They establish no GPU throughput,
expert-parallel scaling, production readiness or general model-quality result.
