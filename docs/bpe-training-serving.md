# Reproduce the bounded BPE lab

Install with `python -m pip install -e .` in a Python 3.11+ virtual environment.
The following pipeline uses public Hindi PUD text, a train-only tokenizer,
exact checkpoint identity, and strict UTF-8 sampling. It is a small CPU smoke
test, not a language-quality, GPU, distributed-scale, or production benchmark.

## Obtain and verify public data

```bash
mkdir -p artifacts/bpe
curl --fail --location --max-time 60 \
  https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-PUD/33e5ecd280d97c7b744993a1e12230559c94fc89/hi_pud-ud-test.conllu \
  --output artifacts/bpe/hindi.conllu
curl --fail --location --max-time 60 \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-PUD/6e328e8195c04a297c1e375d21fe663be1ce6178/en_pud-ud-test.conllu \
  --output artifacts/bpe/english.conllu
ai-journey-corpus-export \
  --hindi artifacts/bpe/hindi.conllu --english artifacts/bpe/english.conllu \
  --train-output artifacts/bpe/train.txt \
  --validation-output artifacts/bpe/validation.txt \
  --manifest artifacts/bpe/data.json
```

The exporter checks both immutable snapshot sizes and SHA-256 digests before
extracting `# text` records. IDs remain translation-aligned and partitions are
disjoint. The default sample uses ten training and ten held-out sentences;
the unused sentences are reported, not silently counted as training evidence.

Both treebanks are CC-BY-SA 3.0. Credit Parallel Universal Dependencies,
CoNLL 2017, DFKI, Google, and the UD community; preserve source attribution and
license requirements when redistributing extracted text. See the pinned
[Hindi license](https://github.com/UniversalDependencies/UD_Hindi-PUD/blob/33e5ecd280d97c7b744993a1e12230559c94fc89/LICENSE.txt),
[English license](https://github.com/UniversalDependencies/UD_English-PUD/blob/6e328e8195c04a297c1e375d21fe663be1ce6178/LICENSE.txt)
and their repository READMEs for contributor credits. Bulk data is not committed
here. For corpus-wide compression, use `ai-journey-tokenizer-compare` with
`--cross-validation`: the source recommends ten-fold training evaluation.
Compression ratios are not parsing scores or language-model perplexity.

## Train and serve

```bash
ai-journey-bpe-train \
  --train artifacts/bpe/train.txt --validation artifacts/bpe/validation.txt \
  --manifest artifacts/bpe/data.json --source-snapshot artifacts/bpe/hindi.conllu \
  --checkpoint artifacts/bpe/model.pt --tokenizer artifacts/bpe/codec.json \
  --output artifacts/bpe/training.json --steps 5
ai-journey-bpe-infer \
  --report artifacts/bpe/training.json --checkpoint artifacts/bpe/model.pt \
  --tokenizer artifacts/bpe/codec.json --train artifacts/bpe/train.txt \
  --validation artifacts/bpe/validation.txt --prompt नमस्ते --new-tokens 4 \
  --top-k 1 --cached --output artifacts/bpe/sample.json
```

Manifest-only validation detects corrupted partition bytes and overlapping IDs;
it is not proof of authorship or provenance. Providing the pinned source snapshot
also verifies every exported ID and exact sentence text. The trainer fits BPE on
training text only and reserves EOS explicitly. Artifact loading rechecks data,
codec, model, completed steps, and final train/held-out losses before serving.

Rerun the identical model-training command with `--resume` to restore its exact
saved state. A completed run performs no further updates. Changing the declared
training configuration is rejected rather than treated as an exact restart.

For paired cache/full-forward timings, replace `--cached` in the inference
command with `--operation benchmark --repeats 3`. Both paths must emit identical
tokens and preserve the trained weights. Timings include UTF-8 masking and cache
identity checks, exclude artifact loading, and do not establish a portable speedup.

## Resume tokenizer learning

```bash
ai-journey-tokenizer-resume --corpus artifacts/bpe/train.txt \
  --tokenizer artifacts/bpe/base-codec.json --state artifacts/bpe/bpe-state.json \
  --vocab-size 270
ai-journey-tokenizer-resume --corpus artifacts/bpe/train.txt \
  --tokenizer artifacts/bpe/base-codec.json --state artifacts/bpe/bpe-state.json \
  --vocab-size 288 --resume
ai-journey-bpe-encode-benchmark --tokenizer artifacts/bpe/base-codec.json \
  --corpus artifacts/bpe/train.txt --output artifacts/bpe/encoding.json
```

Restart validates the corpus, segmentation, minimum frequency, digest, and
canonical merge prefix. It reconstructs that prefix; loading is not a constant-time
operation. Repeated pieces retain their multiplicities during weighted training.
The encoder benchmark uses an ordinary byte/regex codec, not the EOS wrapper.

The tokenizer is independently written, not GPT/tiktoken-compatible. Regex
segmentation uses Python's Unicode rules and must be trusted configuration.
Strict UTF-8 masks alter the sampling distribution; EOS stops without rendering
a fake marker. Byte expansions, corpus sizes, model dimensions, and sampling
budgets are bounded. Modern uncompressed `torch.save` ZIP checkpoints default
to a 64 MiB read budget. No lecture viewing, personal exercise, or reflection is
inferred from these automated checks.
