# Controlled public-data MoE experiments

The bounded reference compares a dense modern decoder with an otherwise matched
top-two expert decoder. The default is an 8-channel, one-layer RMSNorm/SwiGLU/RoPE
byte control, not a pretrained model or a benchmark-quality architecture study.
Only cleared public inputs should be supplied. No command downloads data.

```bash
ai-journey-moe-experiment plan --output artifacts/moe-plan.json
# Review all model, expert, capacity, optimizer and objective controls first.
ai-journey-moe-experiment run --plan artifacts/moe-plan.json \
  --train public-train.txt --heldout public-heldout.txt \
  --output artifacts/moe-report.json
ai-journey-moe-experiment verify --report artifacts/moe-report.json
ai-journey-moe-experiment replay --report artifacts/moe-report.json \
  --train public-train.txt --heldout public-heldout.txt
```

Output parents must already exist. Atomic writers refuse to replace existing
files, including symlinks. Inputs are bounded regular non-symlink files; duplicate
JSON keys, non-finite values, undeclared fields and exact document leakage are
rejected. Every nonempty line is one document; no training window crosses a
document. Short training lines without a complete context window are not sampled.

Arms share every dense architecture control, named-module initialization,
training schedule, token exposure and explicit matrix-decay/vector-no-decay
AdamW policy. Router balance and z-loss coefficients are declared separately
from the model configuration; traces retain CE and each objective term. Forward
expert compute is sparse, but the reference optimizer updates and decays all
stored parameters, including idle experts. Stored and active parameter counts
are not interchangeable.

Whole-job admission charges every arm and seed, both held-out passes, optimizer
updates and independent route/attention/archive controls before model allocation.
Work units are conservative admission proxies, not FLOPs, memory guarantees,
timings or GPU readiness. Limits include four arms/seeds, 32 training steps,
eight examples per training batch, 32-token contexts, eight experts, two layers
and 8192 held-out targets; combined work can reject individually valid controls.

## Pinned public snapshot

```bash
ai-journey-moe-experiment public --source en-pud \
  --snapshot public/en_pud-ud-test.conllu --plan artifacts/moe-plan.json \
  --output artifacts/moe-public.json
ai-journey-moe-experiment verify --report artifacts/moe-public.json
ai-journey-moe-experiment replay --report artifacts/moe-public.json \
  --snapshot public/en_pud-ud-test.conllu
```

English/Hindi PUD reuse the repository's immutable source revision, exact byte
size/SHA-256 and CC-BY-SA-3.0 identity checks. Whole sentences are selected by
content hash and split seed. The default 16 training/four held-out documents are
a bounded protocol, not representative population evidence or a claim that the
UD test partition is a training benchmark. Reports retain pins/license URLs,
selected identities and measured metadata, not source text or local paths.

## Interpretation and restart

Normal evaluation is dropless and excludes padded tokens from routing exposure.
Every within-document target is scored once. Reports preserve batch masks, raw
token losses, expert choices/accepted flags and exact first-come accounting.
The Python `probe_moe_training_capacity` function separately evaluates unchanged
weights in explicit zero-dropout training-capacity mode across declared batch
groupings. Its dropping/loss strata can change with grouping; these are not
ordinary inference scores, causal effects or throughput measurements.

Paired-seed NLL deltas and load-distribution drift are descriptive. Expert indices
are not aligned across independently initialized seeds. Seed spread is not a
population confidence interval, and equal token exposure is not equal compute.
No GPU, expert-parallel, production-scale or architecture-superiority claim is made.

`verify` recomputes internal schema, canonical seal, losses, route conservation,
objective/clipping, exposure, resource admission and summaries without executing
a model. Hashes do not authenticate collected observations. `replay` separately
reruns all measurements from supplied sources and requires exact equality under
the recorded PyTorch version and isolated float32/one-thread CPU protocol. This
successful control is not a cross-platform reproducibility guarantee.

Add `--progress-directory artifacts/moe-progress` to `run` or `public` to retain
an immutable artifact per completed seed. Reissue the same plan, sources and
directory to reuse those owned artifacts. The interrupted seed starts again
from its beginning: no partial model/optimizer state is resumed. Corruption,
changed source/schedule/runtime and noncontiguous progress are rejected before
new training. Use only your own progress and replay the final report; seals do
not establish that fabricated measurements were actually collected.
