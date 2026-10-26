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
RNG, gradients and module modes. Currently every post-forward failure poisons the
session: use fresh processes, not a tensor restore as a guessed reducer reset.
No statement here claims recovery of a dead worker's memory or communicator.
