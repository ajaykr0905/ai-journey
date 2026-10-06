# Day 32 — Hierarchical character modeling with module containers

## Curriculum input

The canonical plan schedules Karpathy's WaveNet lesson, with emphasis on
tree-like receptive fields, `nn.Module` containers, and the development workflow
used to restructure a flat model. The support lesson covers encoder-only
transformers. The workbook still marks Day 32 as not started, so this repository
does not claim that Ajay watched either video or completed the type-along.

## Verified implementation

The Day 32 model starts with fixed-width, boundary-padded character contexts.
Each registered stage concatenates adjacent time steps, applies a learned linear
projection, normalizes the final dimension, and applies `tanh`. For the default
eight-token context, grouping factors `(2, 2, 2)` reduce the time axis from
`8 → 4 → 2 → 1`. Runtime shape evidence verifies those transformations instead
of relying on a diagram alone.

The implementation also provides record-disjoint train/validation data,
deterministic shuffled batches, full-dataset NLL and perplexity, bounded seeded
sampling, exact-state checkpoints, and atomic self-verifying JSON reports.
Checkpoint tests compare an interrupted run with an uninterrupted control,
including model weights, optimizer moments, batch position, and dropout RNG.

Run the CPU experiment:

```bash
python scripts/run_day_32.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-32-wavenet.json
```

For a faster smoke run, keep the same data path and use `--steps 4`,
`--context-size 4`, `--embedding-dim 4`, `--hidden-dim 12`, and
`--group-factors 2 2`. The grouping factors must multiply to the context size;
invalid settings fail before replacing an existing report.

## Evidence boundary

This is a small deterministic CPU character-model experiment over public demo
names. It demonstrates hierarchical tensor grouping, registered container
behavior, and exact local restart. It is not an audio WaveNet reproduction, a
quality comparison with transformers, a GPU benchmark, or production evidence.
Ajay still needs to confirm or explicitly skip both scheduled videos, complete
the type-along in his own words, and explain how the module containers change
parameter registration and tensor shapes.
