# Hybrid parallelism planning and CPU/Gloo conformance

Build one Sealed Report with explicit bounded topology, shard, stage,
microbatch, communication, activation, payload, and output inputs:

```console
ai-journey-hybrid-parallel \
  --world-size 8 \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --sequence-parallel-size 2 \
  --tensor-shape 2,3 \
  --tensor-values 0,1,2,3,4,5 \
  --token-ids 11,12,13,14,15 \
  --layer-count 5 \
  --schedule-policy 1f1b \
  --microbatch-count 2 \
  --element-size-bytes 4 \
  --pipeline-payload-bytes 16 \
  --source-rank 0 \
  --destination-rank 2 \
  --activation-shape 4 \
  --activation-dtype float32 \
  --payload-hex 000102030405060708090a0b0c0d0e0f \
  --output hybrid-parallel-report.json
```

The generation command starts and supervises two local worker processes for one
real CPU/Gloo activation handoff. It publishes the report atomically only after
the analytical inputs, executed observation, derived fields, and seal pass
validation. Choose a new output path for each run.

Verify the retained report offline without rerunning Gloo or modifying it:

```console
ai-journey-hybrid-parallel-verify \
  --report hybrid-parallel-report.json
```

## Evidence Boundary

The topology, tensor and sequence shards, pipeline partition, schedule,
Pipeline Bubble metrics, and communication totals are deterministic analytical
evidence. The only executed distributed observation is one bounded local
CPU/Gloo activation handoff.

No NVIDIA GPU or NCCL is used. This is not performance, throughput, utilization, memory, cost, multi-node, or production-readiness evidence.
It is also not evidence that Ajay watched a lecture or completed a manual exercise.
