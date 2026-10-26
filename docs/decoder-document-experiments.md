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
width. A learned-position arm cannot vary an inactive RoPE frequency base and
claim it as an architecture intervention. Short training documents remain in source evidence but are not sampled
when they contain no complete context window.

Reports contain hashes and measurements, not source text or local source paths.
Writes publish a complete new report without overwriting an existing path.
Verification checks the same PyTorch version and pinned float32 CPU protocol,
then reruns all seeds and compares every recorded tensor fingerprint, update,
coverage count, metric and descriptive summary. It does not change artifacts.
Seed spread is descriptive; equal target exposure does not equal compute cost,
and this bounded experiment establishes no GPU, distributed-scale or population
generalization claim.

## Pinned public snapshots

The public mode reuses this repository's immutable English/Hindi Universal
Dependencies PUD pins and CC-BY-SA-3.0 license metadata. Supply the exact offline
CoNLL-U snapshot already described by `tokenizer_sources`; size and SHA-256 are
verified before parsing or training. Whole sentences are assigned by content
hash and split seed, independently of annotation order. Reports retain the
revision, source/license URLs, split controls and selected document hashes.

```bash
ai-journey-decoder-compare public --source en-pud \
  --snapshot public/en_pud-ud-test.conllu \
  --plan artifacts/decoder-plan.json --output artifacts/decoder-public.json
ai-journey-decoder-verify public --source en-pud \
  --snapshot public/en_pud-ud-test.conllu --report artifacts/decoder-public.json
```

The defaults select 16 training and four held-out documents; explicit counts
are capped at 128 and 32. Selection is a bounded protocol, not a claim that the
selected subset represents the source corpus or that a UD test partition is a
training benchmark. No network request is made by either command.

## Restart an interrupted comparison

Add `--progress-directory artifacts/decoder-progress` to the `documents` command
to publish an immutable artifact after each completed seed. Reissue the same
plan and inputs with that directory to reuse complete seeds. The seed that was
interrupted starts again from its beginning; this option does not restore a
partially updated model. Completed artifacts are never overwritten and unrelated
directory contents are preserved.

Restart checks the plan, source hashes, document identities, schedule, finite
data and update coverage before reusing evidence. Use only progress from your
own run: hashes detect corruption, not a maliciously fabricated measurement.
Always run `ai-journey-decoder-verify documents` on the final report; it recomputes
all seeds and validates the reused measurements against executable code.
