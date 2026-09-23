# Day 19 — Count-Based Character Bigram Language Model

**Date:** 2026-09-23

## Goal

Build the first character language-model baseline: count which character
follows which, convert each count row into a conditional probability
distribution, and generate ten seeded samples. The canonical exercise is a
type-along. This repository adds an independently written deterministic
reference, data manifest, loop oracle, and automated smoke tests. Ajay's own
typed implementation remains separate learner evidence.

## From words to supervised transitions

A bigram model uses exactly one previous character as context. The boundary
token `.` represents both the start and the end of a word. For `ada`, the
training pairs are:

```text
. → a
a → d
d → a
a → .
```

The vocabulary fixes `.` at index zero and sorts the remaining characters. A
square matrix `N` stores the transition counts:

```text
N[previous_character, next_character]
```

Every word of length `L` contributes `L + 1` transitions. The checked-in demo
corpus has 20 words and 100 characters, so the model must count exactly 120
transitions.

## Row normalization and broadcasting

The model needs a conditional distribution for every previous token. The row
total keeps one column dimension:

```text
counts shape:       (18, 18)
row totals shape:   (18, 1)
probability shape:  (18, 18)

probabilities = counts / counts.sum(axis=1, keepdims=True)
```

NumPy compares trailing dimensions. The second dimension is compatible because
`1` can broadcast to `18`; the first dimensions already match. Conceptually,
each scalar row total is reused across that row. No count matrix is duplicated.

An independent nested-loop oracle divides every cell by its row total. The
validator requires the vectorized and loop results to agree within `1e-12` and
requires every probability row to be finite, non-negative, and sum to one.

## Autoregressive sampling

Sampling starts at boundary id zero:

1. Read the probability row for the current character.
2. Draw one next-character id from that categorical distribution.
3. Stop if the draw is the boundary token.
4. Otherwise append the character and use it as the next context.

The generator uses NumPy's `Generator` with an explicit seed. Each sample trace
records all token ids and the exact probability used for every draw. A maximum
length guard reports truncation honestly instead of pretending that a boundary
was sampled.

With the checked-in corpus and seed `1909`, the model smoke test reports:

```text
corpus SHA-256:       f3670d269756a7b8800389c29599c25b2eccc63ca16e7b4f84b532d21f973411
words:                20
characters:           100
vocabulary size:      18
observed transitions: 120
non-zero bigram types: 58
broadcast/oracle error: 0.0
terminated samples:   10/10
```

Generated strings are samples from this tiny empirical model. They are not a
quality benchmark, and matching name-like text does not establish
generalization.

## Data manifest

The demo dataset is deliberately small, synthetic, and public-safe. It contains
no private or employer data. The manifest records its source path, provenance,
CC0 dedication, encoding contract, SHA-256 digest, record counts, and exact
vocabulary. Tests recompute the manifest from the bytes and fail if the corpus
or metadata changes without review.

Files:

- `data/day-19-demo-names.txt`
- `data/day-19-demo-names.manifest.json`

This is the first concrete Week 3 tokenizer/data-manifest artifact. The
boundary-aware character vocabulary is the tokenizer for this baseline, and
the deterministic run is the model smoke test. It does not yet satisfy the
week's full transformer baseline milestone.

## Reproducible experiment

```bash
python scripts/run_day_19.py \
  --output artifacts/day-19-bigram.md \
  --json-output artifacts/day-19-bigram.json \
  --manifest-output artifacts/day-19-corpus-manifest.json
```

The Markdown report includes manifest fields, matrix shapes, invariant results,
seeded samples, frequent transitions, and a Mermaid graph. The full JSON report
contains counts, probabilities, sample traces, metrics, and the manifest. A
custom one-word-per-line corpus can be supplied with `--corpus`.

## What the evidence does and does not establish

The repository verifies corpus validation, canonical serialization, a stable
character vocabulary, boundary-aware pair extraction, exact integer counting,
row-wise broadcasting, an independent normalization oracle, model invariants,
seeded categorical sampling, trace reconciliation, manifest integrity, reports,
and the command-line workflow. It does not establish lecture completion,
Ajay's type-along work, manual understanding, model quality, neural training, or
the completed Week 3 gate.

## Learner evidence still needed

1. Confirm or explicitly skip the assigned makemore segment and calculus
   support video.
2. Type the bigram count model independently rather than copying this reference.
3. Run ten samples and record the command, seed, output paths, and generated
   strings.
4. Explain why `keepdims=True` changes `(18,)` into `(18, 1)` and why the latter
   broadcasts across count-matrix rows.
5. Hand-count all transitions for one short word and locate the matching cells.
6. Explain in Ajay's own words how a derivative can be understood as an
   instantaneous rate of change after the support lesson.

No lecture viewing, type-along, personal explanation, or milestone pass is
claimed here.

## Sources

- Curriculum lecture: Andrej Karpathy, *The spelled-out intro to language
  modeling: building makemore*:
  https://www.youtube.com/watch?v=PaCmpygFfXo
- Course index and lecture description: Andrej Karpathy, *Neural Networks: Zero
  to Hero*: https://github.com/karpathy/nn-zero-to-hero
- Lecture notebook: Andrej Karpathy,
  `makemore_part1_bigrams.ipynb`:
  https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb
- Reference project scope: Andrej Karpathy, *makemore*:
  https://github.com/karpathy/makemore
- Broadcasting rules: NumPy documentation:
  https://numpy.org/doc/stable/user/basics.broadcasting.html
- Seeded categorical sampling API: NumPy documentation:
  https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
- Scheduled support lesson: 3Blue1Brown, *The paradox of the derivative*:
  https://www.youtube.com/watch?v=9vKqVkMQHKk

The workbook assigns the lectures. Listing them records curriculum provenance;
it does not prove that Ajay watched or completed them. The code here was written
independently and does not copy the reference notebook or repository.
