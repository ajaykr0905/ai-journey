# Day 24 — Transformer baseline and rebuild evidence boundary

## Curriculum input

The canonical plan schedules no new primary lecture. It asks the learner to rebuild
the makemore MLP from a blank file, deliberately overfit a 100-example subset, and
use 3Blue1Brown Calculus Chapter 7 as support material. The workbook still marks the
day as not started, so the repository does not claim that Ajay completed those
manual or viewing tasks.

## Verified repository capability

The Week 4 workstream now has a small decoder-only PyTorch transformer that can be
trained on the committed public character corpus. The implementation includes a
causal attention mask, pre-normalized residual blocks, deterministic shuffled
batches, held-out negative log-likelihood, clipped-gradient metrics, seeded
generation, and AdamW parameter grouping.

The checkpoint contains the model, optimizer, batch cursor, and PyTorch RNG state.
Loading validates the model configuration, training configuration, corpus
fingerprint, schema version, and model fingerprint. The regression suite proves
that one resumed step produces the same metric and model bytes as uninterrupted
training, including with dropout enabled.

Run the baseline on CPU:

```bash
python scripts/run_day_24.py \
  --corpus data/day-19-demo-names.txt \
  --steps 100 \
  --output artifacts/day-24-transformer.json \
  --checkpoint artifacts/day-24-transformer.pt
```

Run the exact 100-window capacity probe:

```bash
python scripts/check_day_24_overfit.py \
  --corpus data/day-19-demo-names.txt \
  --examples 100 \
  --output artifacts/day-24-overfit.json
```

These commands create repository evidence, not learner evidence. Ajay still needs
to provide his own blank-file MLP rebuild, the command and seed from his own
100-example run, an explanation of the observed overfitting, and confirmation or
an explicit skip for the Chapter 7 support lesson.
