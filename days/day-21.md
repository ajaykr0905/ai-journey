# Day 21: Dataset shift and record-boundary integrity

**Date:** 2026-09-25
**Curriculum phase:** Karpathy Zero to Hero
**Scheduled activity:** Rebuild the makemore bigram model from a blank file, swap
the dataset to Indian city names, and review the chain rule and product rule.

The blank-file rebuild and lecture are learner activities. They remain pending
until Ajay completes and records them. The repository work for this day builds a
separate deterministic analysis around the dataset swap.

## Engineering question

A model can run correctly on its training corpus while behaving poorly after a
dataset change. A preprocessing bug can also create relationships between
independent records that should never interact. This lab asks:

1. Which characters appear only after the dataset swap?
2. How much do the smoothed transition distributions differ?
3. How does each corpus-trained model score both corpora?
4. Which transitions appear if independent names are concatenated without
   per-record boundaries?

## Shared vocabulary and smoothing

Both corpora use one boundary-first vocabulary formed from their character
union. This avoids treating a missing training character as an indexing error
during cross-corpus evaluation. For count matrix `N` and smoothing value
`alpha > 0`, each row is normalized as:

```text
P(next=j | current=i) = (N[i, j] + alpha)
                        / sum_k(N[i, k] + alpha)
```

Every model/corpus pairing then receives a boundary-aware mean negative
log-likelihood and perplexity. The default experiment uses `alpha = 1.0`.

## Checked-in result

The baseline contains 20 synthetic name-like records. The shifted dataset
contains 30 lowercased Indian city names and has a checked-in SHA-256 manifest.

| Metric | Result |
|---|---:|
| Shared vocabulary | 25 tokens |
| Character-set Jaccard similarity | 0.708333 |
| Characters seen only after the swap | `bcfgjuw` |
| Transition Jensen-Shannon divergence | 0.032334244 |
| Baseline model on baseline | NLL 2.254358388; perplexity 9.529177 |
| Baseline model on cities | NLL 3.071423473; perplexity 21.572589 |
| City model on baseline | NLL 2.888667127; perplexity 17.969343 |
| City model on cities | NLL 2.480273936; perplexity 11.944536 |

The baseline model's higher city-corpus loss is evidence of distribution shift
inside this tiny experiment. It is not a model-quality or generalization claim.

## Boundary negative control

Correct encoding gives every independent word its own start and end boundary.
The negative control concatenates all words into one stream. With 30 cities,
that mistake creates exactly 29 last-character-to-next-first-character events.

| Encoding | City transitions | Cross-record events |
|---|---:|---:|
| Per-record boundaries | 229 | 0 by construction |
| Naive packed stream | 200 | 29 |

The packed stream has fewer transitions because each internal record boundary
loses two boundary transitions and gains one direct cross-record transition.

## Open-source connection

PyTorch TorchTitan issue
[#4801](https://github.com/pytorch/torchtitan/issues/4801) reports that a sparse
attention path can cross packed-document boundaries after position IDs reset.
That issue concerns transformer masking and compressed key/value behavior. This
Day 21 lab does not implement or fix that system. It provides a small CPU-only
analogue of the same invariant: independent records must remain isolated, and a
negative control should demonstrate what leakage looks like.

Our active TorchTitan pull request
[#4864](https://github.com/pytorch/torchtitan/pull/4864) addresses a different
reliability layer: ensuring CPU-safe RL tests are actually collected. Together,
the two examples motivate a portfolio rule: define the invariant, construct a
minimal deterministic failure, and make the regression discoverable in CI.

vLLM issue
[#27433](https://github.com/vllm-project/vllm/issues/27433) tracks batch-invariant
inference validation. Its official paths require supported GPU environments for
meaningful coverage. No vLLM hardware result is claimed here.

## Reproduce

```bash
python scripts/run_day_21.py \
  --output artifacts/day-21-corpus-shift.md \
  --json-output artifacts/day-21-corpus-shift.json
```

The Markdown report contains the comparison table and boundary diagram. The
JSON report records both corpora, the shared vocabulary, four evaluations, and
both structural boundary audits.

## What the tests establish

The repository checks smoothing validation, stable shared-vocabulary ordering,
character coverage, probability invariants, a hand-computed toy loss,
Jensen-Shannon identity and symmetry, exact boundary counts, deterministic
experiment output, JSON serialization, report wording, and CLI failure modes.

It does not establish lecture viewing, Ajay's blank-file rebuild, transformer
attention correctness, vLLM batch invariance, production reliability, GPU
performance, or upstream issue resolution.

## Learner evidence still needed

1. Rebuild the count-based bigram model from a blank file without copying the
   repository implementation.
2. Retrain it on the city corpus and record the exact command and output.
3. Diff the learner rebuild against the reference and write a short gap log.
4. Explain why one boundary token around the whole file leaks information
   between records.
5. Explain the chain rule and product rule in Ajay's own words.
6. Confirm or explicitly skip the scheduled calculus lesson.

## Sources

- Andrej Karpathy, *The spelled-out intro to language modeling: building
  makemore*: https://github.com/karpathy/nn-zero-to-hero
- PyTorch TorchTitan packed-document issue:
  https://github.com/pytorch/torchtitan/issues/4801
- vLLM batch-invariance tracking issue:
  https://github.com/vllm-project/vllm/issues/27433
- 3Blue1Brown, *Visualizing the chain rule and product rule*:
  https://www.youtube.com/watch?v=YG15m2VwSjA

The curriculum assigns the learning activities. Listing them records
provenance; it does not prove completion. The code and tests were written
independently and do not copy upstream implementations.
