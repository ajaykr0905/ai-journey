# Compare tokenizer budgets with trained decoders

This bounded CPU experiment trains 18 small decoders: three vocabulary budgets,
three initialization seeds, and two public languages. It compares canonical BPE
sequence likelihood, not parsing accuracy or marginal probability over every
possible tokenization. No SentencePiece implementation or equivalence is claimed.

## Run and independently replay

Install this checkout with `python -m pip install -e .`. Obtain the exact Hindi
and English snapshots using the [BPE lab instructions](bpe-training-serving.md#obtain-and-verify-public-data).
The loader verifies both immutable byte counts and SHA-256 hashes. Both sources
are CC-BY-SA 3.0; credit Parallel Universal Dependencies, CoNLL 2017, DFKI,
Google, and the UD community. Source revisions, attribution, and license URLs
are recorded in the output. No bulk corpus is committed.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ai-journey-bpe-model-compare \
  --hindi artifacts/bpe/hindi.conllu --english artifacts/bpe/english.conllu \
  --protocol config/day-41-vocabulary-ablation.json \
  --output artifacts/bpe/model-comparison.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ai-journey-bpe-model-verify \
  --report artifacts/bpe/model-comparison.json \
  --hindi artifacts/bpe/hindi.conllu --english artifacts/bpe/english.conllu
```

The default selects ten training and ten held-out sentences with translation-aligned
IDs and split seed 40; 980 sentences per source remain unused. Tokenizers fit
training text only. The protocol predeclares budgets 256/272/304, seeds 41/42/43,
20 AdamW updates, batch size four, and eight-token context. Each arm has one
16-dimensional, two-head transformer layer. EOS adds one vocabulary entry.
Shared trunk weights, the 256 byte embeddings/output rows, and EOS rows match
at initialization within each seed. New merged-token rows necessarily differ.

The verifier checks the envelope and paired matrix, then retrains all 18 models
from the verified sources. It requires exact model fingerprints, losses, traces,
IDs, and summaries. Host metadata and elapsed times are observations, not replay
identities. Use the recorded PyTorch version for exact replay; different numerical
stacks can produce different fingerprints. The digest alone is not authentication.
The report preserves all seeds, including regressions, rather than selecting one.

## Read the metrics

Each document begins from EOS. A stride-one rolling context predicts each content
token once and a final EOS separately. No context crosses documents. Positions
reset on context rollover. Content bits per byte is summed content NLL divided
by `UTF-8 bytes * ln(2)`; EOS loss has no artificial byte denominator. Reports
retain document identities, byte counts, content tokens, and separate EOS NLL.
This differs from the older `evaluate_nll` window-averaged training diagnostic.

Token perplexities across different tokenizers are not directly comparable; see
[Hugging Face's fixed-context evaluation explanation](https://huggingface.co/transformers/v3.1.0/perplexity.html).
The uniform-token baseline is `content_tokens * log2(vocabulary_size) / UTF8_bytes`.
It exposes a score reduction caused by compression with no learning at all.
Training gain is measured against the same arm's initialized model, not another
tokenizer's initial score. Seed sample SD and paired-delta standard error describe
initialization variation only, not dataset uncertainty or statistical significance.

## Observed CPU smoke result, 15 October 2026

Python 3.13.9, PyTorch 2.14.1, CPU, one Torch thread. The complete 18-trial report
was exactly replayed. Values below average three seeds; lower trained BPB is
better, and positive training gain means improvement within that arm.

| Language | Budget, before EOS | Parameters | Uniform BPB | Trained BPB | Gain from initialization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hindi | 256 | 11,664 | 8.0056 | 5.9622 | 2.0725 |
| Hindi | 272 | 12,176 | 3.6117 | 2.9614 | 0.6479 |
| Hindi | 304 | 13,200 | 2.8795 | 2.5018 | 0.3820 |
| English | 256 | 11,664 | 8.0056 | 6.1853 | 1.8130 |
| English | 272 | 12,176 | 6.4429 | 5.3827 | 1.0595 |
| English | 304 | 13,200 | 5.9318 | 5.1683 | 0.7664 |

Larger BPE vocabularies lowered the absolute canonical-sequence BPB here, while
the byte baseline improved more during training. Compression explains much of
the cross-arm difference. Equal token exposure is **not** equal byte exposure,
parameter count, byte-context coverage, or compute. The ten-sentence training
sample and twenty updates cannot identify an optimal vocabulary or establish
language quality. Do not compare Hindi and English scores as a language ranking.
This sample is not the source treebanks' recommended ten-fold benchmark, nor GPU,
distributed, serving-throughput, production, or learner-completion evidence.

Ajay's video confirmation, own tokenizer rebuild, independent run, and personal
tokenization blog/explanation remain separate learner evidence.
