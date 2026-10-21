# GPT-2 training controls

This workflow tests optimizer behavior on the repository's public CC0 name-like
fixture. It uses a 7,536-parameter byte-level GPT-2-style decoder so accumulation,
scheduling, weight decay, clipping, rollback, and checkpoint replay can run on CPU.
It does not train or fine-tune the 124M checkpoint.

## Run the checked protocol

```bash
ai-journey-gpt2-train train \
  --source data/day-19-demo-names.txt \
  --spec config/gpt2-training.json \
  --checkpoint /tmp/gpt2-training.pt \
  --report /tmp/gpt2-training.json
```

The protocol uses micro-batches of two sequences and accumulates four of them before
each optimizer update. Each micro-batch mean loss is divided by four before
backpropagation. The resulting effective batch is eight sequences, or 64 target
tokens per update. Twelve updates consume 768 target tokens.

AdamW applies weight decay to matrix-shaped parameters and excludes biases and
normalization vectors. The learning rate warms up for two updates, then follows a
cosine decay from `0.003` to `0.0003`. Gradients are clipped at a global norm of
`1.0`. Each run pins one PyTorch intra-op thread and restores the caller setting
afterward, so host thread defaults cannot change the evidence. A failed update
restores model, optimizer, cursor, random-number, gradient, and module-mode state
before returning the error.

## Verify and replay

```bash
ai-journey-gpt2-train verify \
  --report artifacts/gpt2-training-2026-10-21/training-evidence.json

ai-journey-gpt2-train replay \
  --source data/day-19-demo-names.txt \
  --report artifacts/gpt2-training-2026-10-21/training-evidence.json
```

`verify` checks the report digest, source and spec fingerprints, parameter grouping,
step sequence, token accounting, schedule values, clipping bounds, and runtime
metadata. `replay` repeats the full experiment in a temporary directory and requires
the complete regenerated report to match.

The checked run recorded mean loss `12.200964450836182` on update 1 and
`10.875006675720215` on update 12. Every update crossed the clipping threshold. This
is a small deterministic systems check, not a model-quality result. It provides no
GPU, distributed, throughput, production, lecture-viewing, or learner-completion
evidence.
