# Day 33 — Rebuild and halfway evidence review

## Curriculum input

The canonical plan schedules a rebuild day and halfway review. Its hands-on task
is to rebuild the hierarchical character model, while the support lesson covers
word embeddings and Word2Vec. The workbook still marks Day 33 as not started.
Accordingly, this repository records an independently verified engineering
implementation, not evidence that Ajay watched the lesson, performed a
blank-file rebuild, or wrote a personal reflection.

## Verified rebuild

The Day 33 implementation reconstructs the Day 32 hierarchy from registered raw
`Parameter` objects and tensor operations. It performs embedding indexing,
time-axis grouping, matrix multiplication, normalization, activation, dropout,
and the output projection without using the reference model's high-level
embedding, linear, or normalization modules.

The evidence gates compare this independent path with the Day 32 reference after
an exact parameter transfer. They cover logits, loss, every parameter gradient,
and one centered finite-difference probe. Additional checks cover seeded
initialization, complete-dataset metrics, deterministic fitting, transactional
checkpoints, exact interrupted-versus-uninterrupted training with dropout,
bounded sampling, and memorization of a small training slice. A canonical report
fingerprint detects accidental evidence changes; it is not a digital signature.

Run the CPU experiment:

```bash
python scripts/run_day_33.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-33-wavenet-rebuild.json
```

For the faster CI profile, add `--steps 2 --batch-size 4 --context-size 4`,
`--embedding-dim 4 --hidden-dim 12 --group-factors 2 2`,
`--overfit-examples 1 --overfit-steps 20`, and
`--minimum-overfit-improvement 0.1`. Invalid input or a failed evidence gate
cannot replace an existing valid report.

Run the independent reliability certification:

```bash
python scripts/certify_day_33.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-33-certification.json
```

The certification adds structural parameter and storage checks, activation and
gradient diagnostics, per-example prediction parity, optimizer-transition
parity, input/RNG/repeatability controls, bounded sampling checks, an exact model
footprint, and atomic tamper-evident evidence. Its local CPU timing measurement
is diagnostic only and is not included as a cross-machine pass threshold.

## Evidence boundary

This is a small deterministic CPU experiment over public demo names. It verifies
functional equivalence and restart behavior for the declared implementation; it
does not establish audio WaveNet fidelity, language-model quality, GPU or
distributed performance, production readiness, or Word2Vec understanding. Ajay
still needs to complete the rebuild independently, record his own differences
and mistakes, confirm or skip the support lesson, and supply the personal
halfway review requested by the curriculum.
