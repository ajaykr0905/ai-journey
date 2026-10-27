# Replicated expert optimizer windows on CPU

This opt-in library is separate from the plain-CE DDP training worker. It runs
replicated float64 CPU/Gloo models: every rank stores every expert. It is not
expert parallelism, GPU validation, a speed result or production recovery proof.

Construct an exact `MoEDDPConfig(MoEConfig(...), ...)`, establish the matching
`ddp_runtime.process_group`, then construct `MoEDDPTrainingSession(config)`.
Its `step(window)` accepts a tuple of exactly `accumulation_steps` tuples:
`(inputs, targets, token_mask)`, each of declared `[micro_batch_size, sequence_length]`
shape with CPU long tokens and a CPU boolean mask. All input IDs must be valid;
only selected targets enter CE. One rank may select zero tokens, but the complete
world must select at least one. The mask controls routing/loss eligibility, not
attention-key masking; this inherits the existing decoder's causal semantics.

One outer DDP forward owns all microbatch graphs and one backward. Temporary
router hooks capture live full-softmax probability sums and squared logsumexp
sums, and are removed on success or exception. Detached routing evidence remains
auditable separately; detached probability sums are never used as gradients.
`find_unused_parameters=True` handles different expert-use sets on different
ranks. Only experts with zero **globally accepted** routes may receive synthetic
zero gradients. Every stored parameter still receives dense AdamW decay, moments
and a counter on every successful update, including idle experts.

## Objective and capacity contract

Let `N` be globally selected targets, `W` ranks, `L` expert layers, `E` experts,
`P[r,l,e]` live local probability sums and `Z[r,l]` local squared-logsumexp sums.
Global attempted-route frequency is `f[l,e] = attempted[l,e] / (2*N)`.
Each rank differentiates:

```text
W*CE_sum[r]/N
  + balance_weight*W*E*sum(f*P[r])/(L*N)
  + z_weight*W*sum(Z[r])/(L*N)
```

DDP averages gradients, producing the token-weighted global CE and global
attempt-frequency auxiliary objective. Reported metrics are globally reduced,
not local-rank means. Capacity is explicitly first-come **per rank/microbatch**;
changing microbatch partitioning may change drops and the objective. The oracle
therefore runs those same declared groups serially; it does not pool capacity.

## Admission and lifecycle

Limits are two ranks, one million stored parameters per rank, context 128,
microbatch four, eight retained microbatches/window, 16 scheduled updates and
8,192 computed positions/update. Complete-job work is bounded to 100M declared
parameter-token work units; all retained microbatch routing and attention
elements are charged, including masked positions' dense work. Independent
checkpoint-record limits apply before construction. These are conservative
admission proxies, not measured FLOPs or complete allocator-peak accounting.

Default model initialization uses the declared CPU seed without changing caller
RNG. Library callers own their rank-local training RNG and process-group scope.
An exception reached by a live process restores its local model, optimizer, CPU
RNG, gradients and module modes. Pre-forward admission failures leave the session
usable. Once a forward is invoked, only an explicitly common rejection **after
every backward completed** can raise `MoEDDPRetryableError` and retain usability.
Tests prove two further same-wrapper serial-equivalent updates after rank-local
gradient poisoning, clipping failure, post-step exception and finite model or
moment divergence. Correct the cause before retrying; this is not an automatic
retry loop.

Forward, pre-backward, interrupted-backward and transport failures instead raise
`MoEDDPFatalError` and poison the session. Further calls reject immediately. Real
rank-local forward and backward exceptions exercise this boundary; live tensor
rollback is not a guessed reducer reset. Transport exceptions never qualify as
common semantic rejection, even after a successful backward. Use fresh processes
and a valid durable checkpoint. No statement here claims recovery of a dead
worker's memory or communicator.

## Replicated stochastic checkpoint boundary

`save_moe_ddp_checkpoint(path, session=session, source=source_bytes,
data_contract=...)` collectively stores the existing sealed MoE payload inside
`ai-journey-replicated-moe-checkpoint-v1`, adding exact run/world/data contracts,
all rank-local CPU RNG states and `data_cursor.next_update_index`. This zero-based
next update index equals the count of completed optimizer updates. The caller's
nonempty finite JSON data contract must describe its scheduling algorithm; no
unrecorded sampler state is inferred. New sessions enforce their declared fixed
learning rate, invariant optimizer controls and ordered groups.

`load_moe_ddp_checkpoint(...)` admits the complete outer run and common loaded
seal across ranks before restoring. It reuses the existing typed MoE optimizer /
architecture restoration and then selects each rank's saved CPU RNG. Rejected
archive, cursor, source, world, data schedule, optimizer controls, moments or RNG
preserve caller state. Tests run uninterrupted and split execution in genuinely
fresh two-rank processes with dropout enabled; weights, optimizer, RNG and metrics
match exactly in that fixed local protocol, not across arbitrary platforms.

Saving returns `MoEDDPCheckpointIdentity(payload_sha256, archive_sha256)`. The
payload seal hashes canonical submitted metadata/tensors; the archive SHA hashes
serialized file bytes. They are deliberately distinct. Only `archive_sha256`
belongs in the loader's `expected_archive_sha256` argument; passing the payload
seal rejects before mutation. Both identities detect changes, not authorship.
