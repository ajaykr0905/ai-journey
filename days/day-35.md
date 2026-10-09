# Day 35: Causal averaging before learned attention

Curriculum focus: the second segment of Karpathy's GPT build introduces the
lower-triangular averaging matrix that lets each token aggregate only its own
prefix. The repository implementation is independently written and does not
assert that either scheduled lecture was watched or that Ajay completed the
manual exercise.

## Run the verified experiment

```bash
ai-journey-day-35 \
  --output artifacts/day-35-causal-average.json
```

The command generates a seeded CPU float64 batch and computes every prefix mean
with an explicit loop, a normalized lower-triangular matrix, masked softmax over
zero logits, and a cumulative sum. It gates forward and input-gradient agreement,
row normalization, exact zero future mass, non-negative weights, future-token
perturbation isolation, padding isolation, and chunked streaming equivalence.
The JSON report is content-addressed and published atomically.

Streaming inference keeps only a count and channel-wise sum. Complete snapshots
are copied by value, validated before restore, fingerprinted, and atomically
written. This is a uniform causal average, not learned query-key attention. The
default run is a small CPU numerical audit, not a model-quality, GPU, distributed,
or performance result.

## Learner evidence still required

- Confirm or explicitly skip the scheduled Karpathy and MIT 6.S191 videos.
- Implement the three curriculum forms independently and record the command,
  seed, and report path.
- Explain why the normalized triangular row for position `t` contains `t + 1`
  copies of `1 / (t + 1)`.
- Perturb one future token by hand and explain why earlier outputs do not change.
