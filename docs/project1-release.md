# Project 1 release and recovery evidence

Project 1 is a reproducible transformer training and serving lab built from public
data. The repository now contains deterministic model, training, evaluation,
fine-tuning, serving-control, reward-verification, and recovery components. The
Day 70 release gate adds a real local worker process kill, controller-side failure
detection, restart from a content-sealed checkpoint, exact loss accounting, and an
independently verified report.

## Run the checked-in recovery experiment

Use a new directory because the operator refuses to overwrite evidence:

```bash
ai-journey-recovery run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-worker-recovery.json \
  --run-dir /tmp/day-70-worker-recovery

ai-journey-recovery verify \
  --run-dir /tmp/day-70-worker-recovery
```

The controller launches a separate Python worker without a shell. The first
attempt publishes progress after item 11 and waits. The controller observes that
admitted boundary, sends a real process kill, detects the nonzero exit, verifies
the item-8 checkpoint, and launches a distinct attempt from that checkpoint. The
report records the actual monotonic detection, restart-delay, and recovery times.
It also reconciles three lost uncheckpointed items, three replayed items, zero
omitted items, and zero duplicated committed items.

## Architecture and evaluations

The flagship uses a decoder-only transformer core with deterministic data and
checkpoint controls. The repository includes public-data GPT-2 interoperability,
assistant-only SFT, LoRA and full-fine-tune comparisons, HellaSwag and LAMBADA
evaluation paths, contamination checks, serving admission controls, and
verifiable-reward fixtures. The recovery experiment operates on a small
content-private CPU workload so CI can execute the complete process lifecycle.
It does not substitute for the model-serving or multi-GPU milestone.

## Compute cost in INR

External cloud and GPU spend for the checked-in CPU fixtures is **₹0**. The
existing host, electricity, engineering time, and employer-provided compute were
not metered, so a total INR cost is unavailable and is not inferred. No paid GPU
run was launched by this automation.

## What went wrong and what remains blocked

- The current environment does not provide the required real NVIDIA multi-GPU
  run, utilization and memory measurements, or distributed worker-failure trace.
- The repository has bounded serving controls and CPU measurements, but no
  verified vLLM or SGLang deployment on supported GPU hardware.
- The Project 1 training fixtures demonstrate reproducibility and failure
  handling, not useful pretrained-model quality or production readiness.
- Recovery time depends on this host and workload. It must not be generalized to
  another machine, model, dataset, process topology, or serving stack.

Ajay still needs to run the GPU and serving milestone with a hardware/software
manifest, exact commands, utilization, memory, throughput, p50/p95/p99 latency,
cost, injected failure, checkpoint or data loss, and measured recovery time. The
interpretability video, personal write-up, and public blog post also require
learner-authored evidence and are not claimed by this repository change.
