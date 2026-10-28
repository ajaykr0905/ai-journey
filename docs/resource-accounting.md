# Transformer resource accounting

The Day 51 planner turns explicit architecture, workload, tensor dtype,
sharding, capacity, utilization, and price assumptions into reproducible JSON.
It is planning evidence. It does not allocate a GPU, execute a kernel, measure
throughput, or prove that a lecture or manual calculation was completed.

## Run the pinned 7B exercise

```bash
ai-journey-resource-plan \
  --config config/day-51-resource-plan.json \
  --output artifacts/day-51-resource-plan.json

ai-journey-resource-verify \
  --report artifacts/day-51-resource-plan.json
```

The output must not alias the protected config, including resolved symlink paths
or existing hardlinks. Disjoint existing reports can still be atomically replaced;
the planner never treats its input plan as an output report.

The pinned plan uses a 6,738,415,616-parameter dense SwiGLU decoder, sequence
length 2,048, and a global batch of eight formed by one sequence on each of
eight data-parallel ranks. It models 1,000 optimizer steps with BF16 parameters,
gradients, and saved activations; FP32 Adam moments and master weights; uniform
layer checkpointing; and ZeRO-3-style state sharding.

For those inputs, the deterministic report calculates:

- 16,384 tokens per optimizer step and 16,384,000 training tokens;
- 702,346,076,659,712,000 modeled training FLOPs;
- 14,489,755,648 bytes of modeled peak memory per device;
- 77,309,411,328 usable bytes after reserving 10% of an 80 GiB device;
- 803.9676 seconds at the declared 35% model-FLOPs utilization;
- 1.7866 accelerator-hours and an illustrative cost of $5.3598 at $3.00 per
  accelerator-hour.

The capacity, peak-FLOPs, utilization, and price values are inputs, not detected
hardware or a provider quote. Replace them with an approved environment and
current price before making a purchase or launch decision.

## Accounting boundaries

Projection FLOPs count a multiply-add as two operations. Dense attention counts
both the query-key and probability-value matrix multiplications, including the
full square work of ordinary causal kernels. Backward work defaults to twice the
forward work, and optimizer updates use an explicit operations-per-parameter
budget.

Memory is decomposed into parameters, gradients, optimizer moments, master
weights, checkpoint boundaries, one recomputed activation window, and dense
attention workspace. DDP replicates state; the ZeRO stages shard progressively
more state across the declared data-parallel size. KV-cache storage is available
as a separate inference primitive and scales with layer count, sequence length,
batch size, KV heads, and dtype width.

The estimate excludes allocator fragmentation, communication buffers, temporary
kernel workspaces, input batches, framework state, and non-model processes. The
10% reserve is a planning margin, not proof that the configuration will execute.
Reports are recomputed during verification and carry a canonical SHA-256 digest;
the digest detects changes but is not a digital signature.
