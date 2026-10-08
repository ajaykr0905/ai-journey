# Day 34: Tiny Shakespeare bigram baseline

Curriculum focus: the first segment of Karpathy's GPT build covers text loading,
character encoding, a bigram baseline, and the training-loop skeleton. The
repository implementation is independently written and does not assert that the
lecture was watched or that Ajay completed a type-along.

## Run the verified baseline

```bash
ai-journey-day-34 \
  --corpus artifacts/tinyshakespeare.txt \
  --download \
  --output artifacts/day-34-bigram.json \
  --checkpoint artifacts/day-34-bigram.pt
```

The download is bounded by the pinned byte count and accepted only when its
SHA-256 digest matches the declared public source. Training uses an ordered 90/10
train/validation split, a seeded random-window sampler, a learned character-to-
next-character logit table, AdamW, exhaustive partition evaluation, and a seeded
bounded sample. The checkpoint binds the exact corpus digest, vocabulary,
configuration, model, optimizer, and sampler RNG state. The JSON report carries
a canonical content digest and records the full update trace.

This is a CPU character-bigram baseline. It is not a transformer, a quality
benchmark, a GPU result, or evidence of statistical significance. The generated
sample is a deterministic diagnostic. Tiny Shakespeare's underlying works are
public domain; the exact compilation is fetched from Andrej Karpathy's char-rnn
repository and is not committed to this repository.

## Learner evidence still required

- Confirm the primary and support videos were watched, or explicitly record a skip.
- Type the baseline independently and record the command, seed, and output paths.
- Explain why the embedding table has shape `(vocabulary, vocabulary)`.
- Compare initial and final train/validation NLL and interpret any generalization gap.
