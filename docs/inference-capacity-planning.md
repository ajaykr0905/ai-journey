# Analytical inference capacity planning

The Day 61 lab turns a declared decoder shape and workload into reproducible
memory and work estimates. It does not run a model server or claim measured GPU
performance.

## Reproduce the pinned plan

```bash
ai-journey-inference build \
  --protocol config/day-61-inference-plan.json \
  --output /tmp/day-61-inference-report.json
ai-journey-inference verify \
  --report /tmp/day-61-inference-report.json
```

Both commands print the same SHA-256 digest. Verification reconstructs the
protocol, recomputes every derived field, checks the evidence boundaries, and
compares canonical report bytes.

## Pinned calculation

The scenario declares a 7-billion-parameter decoder with 32 layers, 32 query
heads, head dimension 128, bf16-sized two-byte cache elements, 32,768-token
context, and batch 16.

| Cache layout | KV heads | Full cache |
|---|---:|---:|
| Multi-head attention | 32 | 256 GiB |
| Grouped-query attention | 8 | 64 GiB |

The formula is `2 × layers × KV heads × head dimension × element bytes ×
context × batch`; the leading two represents keys and values. The declared
80-GiB device budget reserves 14 GiB for weights and 6 GiB for runtime state,
leaving 60 GiB for KV data. The 64-GiB GQA cache is therefore rejected instead
of being rounded into an apparently valid batch.

The report also records whole-page allocation, separate projection and
attention FLOPs for prefill and decode, and exact speculative-decoding
expectations. These remain analytical inputs. Allocator behavior, usable GPU
memory, latency, throughput, acceptance rate, and serving cost require an
executed workload with hardware and software evidence.
