# Day 26 — Fan-in Kaiming initialization

## Curriculum input

The canonical plan schedules the middle section of Karpathy's activations,
gradients, and BatchNorm lesson. The hands-on task is to implement Kaiming
initialization and compare loss curves with a fixed-scale initialization. The
workbook still marks the day as not started, so this repository does not claim
that Ajay watched the lecture or completed a manual type-along.

## Controlled experiment

The experiment changes one model setting: linear weight matrices use either a
fixed normal standard deviation or `gain / sqrt(fan_in)`. Embeddings retain the
same fixed-scale initialization in both variants. The corpus, split, seed,
architecture, optimizer, batch order, training steps, and evaluation procedure
remain fixed.

Each run records:

- theoretical and observed standard deviations for every initialized matrix;
- zero-bias checks and the maximum relative scale error;
- per-step loss and gradient norm, loss reduction, best step, and mean loss;
- exhaustive train and held-out negative log-likelihood before and after training;
- runtime versions, deterministic-kernel state, model and corpus fingerprints;
- a self-verifying JSON report plus deterministic loss-curve and audit SVGs.

Run the CPU comparison:

```bash
python scripts/run_day_26.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-26-kaiming.json \
  --loss-plot artifacts/day-26-loss-curves.svg \
  --audit-plot artifacts/day-26-initialization.svg
```

The command exits nonzero when an observed initialization scale misses its
configured tolerance, either variant fails to reduce loss, or the predeclared
Kaiming mean-loss improvement is absent. These gates describe this small,
deterministic experiment; they are not a claim that Kaiming initialization is
universally superior for every transformer architecture.

Add `--require-kaiming-validation-improvement` when a run must reject a
training-only gain that does not transfer to held-out NLL. The default report
still records that tradeoff without hiding the negative result.

## Evidence boundary

This is a CPU-only comparison on the committed public character corpus. It does
not prove GPU performance, distributed scale, production behavior, or learner
activity. Ajay still needs to provide his own lecture confirmation, independent
Kaiming implementation or notebook run, interpretation of both loss curves, and
confirmation or an explicit skip for Calculus Chapter 9.
