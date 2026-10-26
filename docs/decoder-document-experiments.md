# Document-isolated decoder comparisons

The original schema-2 byte-window smoke command remains available. The document
mode scores every within-document held-out target once, matches training batches
across architectures, records finite optimizer transitions and initial/final
tensor fingerprints, and supports one to four declared paired seeds on CPU.

```bash
ai-journey-decoder-compare plan --output artifacts/decoder-plan.json
# Review/edit the plan before the run; unknown controls and undeclared
# architecture interventions are rejected. Every nonempty input line is one
# document. Repeated documents and overlap across the two files are rejected.
ai-journey-decoder-compare documents \
  --plan artifacts/decoder-plan.json \
  --train public-train.txt --heldout public-heldout.txt \
  --output artifacts/decoder-documents.json
ai-journey-decoder-verify documents \
  --report artifacts/decoder-documents.json \
  --train public-train.txt --heldout public-heldout.txt
```

Use independently cleared public data. Sources are capped at one MiB each, the
partitions at 2048 documents, held-out scoring at 8192 targets, and plans at eight
arms, four seeds, 64 steps, 32 examples per training batch and 32 examples per
evaluation batch. Additional work and attention budgets reject expensive plans
before model allocation. Explicit SwiGLU width cannot exceed four times model
width. Short training documents remain in source evidence but are not sampled
when they contain no complete context window.

Reports contain hashes and measurements, not source text or local source paths.
Writes publish a complete new report without overwriting an existing path.
Verification checks the same PyTorch version and pinned float32 CPU protocol,
then reruns all seeds and compares every recorded tensor fingerprint, update,
coverage count, metric and descriptive summary. It does not change artifacts.
Seed spread is descriptive; equal target exposure does not equal compute cost,
and this bounded experiment establishes no GPU, distributed-scale or population
generalization claim.
