# Day 25 — Activation and gradient diagnostics

## Curriculum input

The canonical plan schedules the first section of Karpathy's activations,
gradients, and BatchNorm lesson. Its hands-on task is to reproduce activation
histograms and learn to recognize the distribution shifts caused by poor
initialization. The workbook still marks the day as not started, so this
repository does not claim that Ajay watched the lecture or completed a manual
type-along.

## Verified repository capability

The transformer lab now makes initialization scale an explicit, fingerprinted
model setting. The Day 25 experiment changes only that setting between a
baseline and a deliberately stressed model, while holding the corpus, seed,
architecture, optimizer, batch order, and training steps fixed.

Each variant records:

- initial and final GELU activation histograms for every transformer block;
- initial and final named parameter-gradient histograms;
- non-finite, out-of-range, zero, near-zero, RMS, and standard-deviation metrics;
- explicit distribution-health checks and missing-gradient detection;
- first-step update-to-parameter RMS ratios;
- loss traces, model fingerprints, and the stressed-to-baseline activation RMS
  contrast.

Run the CPU comparison:

```bash
python scripts/run_day_25.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-25-diagnostics.json \
  --plot artifacts/day-25-activations.svg \
  --gradient-plot artifacts/day-25-gradients.svg
```

The command exits nonzero if the stressed initialization does not produce the
configured minimum activation-RMS contrast. Histogram ranges and health limits
are serialized in the JSON report so the result can be reviewed without relying
on a plot alone.

## Evidence boundary

This is a deterministic CPU experiment on the committed public character corpus.
It is not evidence of GPU training, distributed scale, BatchNorm implementation,
or learner activity. Ajay still needs to provide his own lecture confirmation,
type-along or notebook run, interpretation of the activation plots, and
confirmation or an explicit skip for the Calculus Chapter 8 support lesson.
