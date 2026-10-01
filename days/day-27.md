# Day 27 — Scratch BatchNorm and the train/eval trap

## Curriculum input

The canonical plan schedules the final section of Karpathy's activations,
gradients, and BatchNorm lesson. The hands-on task is to implement BatchNorm and
deliberately omit `model.eval()` to observe the resulting inference error. The
workbook still marks the day as not started, so this repository does not claim
that Ajay watched the lecture or completed the manual exercise.

## Verified experiment

The implementation normalizes the final tensor dimension, learns affine scale
and bias, and keeps persistent running mean, variance, and sample count. Its
training output and state updates are checked against PyTorch, including the
biased batch variance used for normalization and the unbiased variance used for
running-state updates.

The transformer can select scratch BatchNorm for every normalization boundary.
The selected policy and running statistics are checkpointed, fingerprinted, and
restored during exact-restart tests. The controlled experiment holds the corpus,
seed, architecture, optimizer, and weights fixed while comparing correct eval
mode with the deliberate train-mode mistake. It also measures whether a changed
companion example changes the same target's logits in train and eval modes.

Run the CPU experiment:

```bash
python scripts/run_day_27.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-27-batchnorm.json \
  --plot artifacts/day-27-batchnorm.svg
```

The command writes a self-fingerprinting JSON report and deterministic SVG. It
exits nonzero unless training reduces loss, the mode mistake causes a measurable
NLL gap, train-mode outputs depend on batch composition, and eval-mode coupling
stays below the configured tolerance.

## Evidence boundary

This is a deterministic CPU experiment on a small public character corpus. The
measured train/eval gap demonstrates a failure mode for this configuration; it
does not establish universal BatchNorm quality or production performance. Ajay
still needs to confirm the primary lecture, independently run or implement the
exercise, record what changed when `model.eval()` was omitted, and confirm or
explicitly skip the higher-order-derivatives support lecture.
