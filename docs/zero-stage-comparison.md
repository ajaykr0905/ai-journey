# ZeRO stage comparison and collective conformance

Day 57 adds two separate evidence paths for data-parallel and ZeRO reasoning:

- a deterministic analytical comparison of stages 0–3 from one pinned model,
  topology, dtype, activation, and gradient-bucket plan; and
- an executed two-rank CPU/Gloo check that all-reduce produces the same summed
  tensor as reduce-scatter followed by all-gather.

The paths stay separate because analytical capacity estimates and executed
collective correctness answer different questions.

## Build and verify the pinned comparison

```bash
mkdir -p artifacts/day-57-zero
ai-journey-zero-compare build \
  --config config/day-57-zero-comparison.json \
  --output artifacts/day-57-zero/comparison.json
ai-journey-zero-compare verify \
  --report artifacts/day-57-zero/comparison.json
```

The plan uses the repository's 6,738,415,616-parameter decoder, eight declared
data-parallel ranks, BF16 parameters and gradients, FP32 Adam moments and master
parameters, 25 MiB gradient buckets, and 1,012,924,416 bytes of modeled saved
activations and attention workspace. Capacity admission assumes an 80 GiB
device with a 10% reserve, leaving 77,309,411,328 usable bytes. The topology
bandwidth and latency values are assumptions. The code does not inspect the
host or purchase hardware.

The sealed report digest is
`e7ff705e7a8165241b8fe34bc7ea8ab5208b6dcccc75dbd6cf6b5d51ae7c0acc`.
The four stage results are:

| Stage | Sharded state | Modeled maximum rank peak | Modeled ring wire per rank | Fits reserved 80 GiB |
| ---: | --- | ---: | ---: | --- |
| 0 | None | 108,827,574,272 bytes | 23,584,454,656 bytes | No |
| 1 | Optimizer and master parameters | 38,074,210,304 bytes | 35,376,681,984 bytes | Yes |
| 2 | Gradients, optimizer, and master parameters | 26,308,197,376 bytes | 23,584,454,656 bytes | Yes |
| 3 | Parameters, gradients, optimizer, and master parameters | 26,281,982,976 bytes | 23,584,454,656 bytes | Yes |

Stage 1 has more modeled communication because the schedule includes a full
gradient all-reduce and an all-gather of updated parameter shards. Stages 2 and
3 use reduce-scatter for gradients. Stage 3 additionally all-gathers parameter
shards for computation. These are logical flat-ring estimates, not measured
latency or throughput. The admission result is a deterministic plan check, not
evidence that any listed stage ran on an 80 GiB accelerator.

## Execute the CPU collective check

Choose an unused local port and run two Gloo ranks:

```bash
python -m torch.distributed.run \
  --nproc-per-node 2 \
  --master-addr 127.0.0.1 \
  --master-port 29571 \
  -m ai_journey.zero_probe \
  --output artifacts/day-57-zero/cpu-gloo-probe.json

ai-journey-zero-probe --verify-report \
  artifacts/day-57-zero/cpu-gloo-probe.json
```

Each rank contributes a deterministic eight-element tensor. The probe executes
all-reduce, reduce-scatter, and all-gather, then verifies that both communication
paths reconstruct `[12, 14, 16, 18, 20, 22, 24, 26]`. The report records every
rank input and output, the CPU/Gloo backend, the PyTorch version, and a canonical
SHA-256 digest. Verification recomputes the expected sum from the rank inputs
and rejects duplicate JSON fields, altered outputs, non-finite values, missing
ranks, and integrity drift.

## Evidence boundary

The analytical comparison has `measured: false`. The collective probe has
`executed: true` and `performance_measured: false`. It is CPU/Gloo correctness
evidence only: no NVIDIA GPU, NCCL, multi-node network, utilization, memory,
latency, bandwidth, monetary cost, or production scale was measured. This work
does not satisfy the Week 9 real multi-GPU gate. Ajay must still provide an
approved 2+ NVIDIA GPU environment and retain the commands, logs,
hardware/software manifest, utilization, memory, cost, and limitations from
that run.
