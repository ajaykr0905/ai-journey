# Paired CPU LoRA comparison

This offline experiment pretrains the existing `DecoderLanguageModel` on an
authored synthetic character grammar, then forks the same pretrained weights
into four arms: a frozen control, full fine-tuning, rank-1 LoRA, and rank-2 LoRA.
The checked-in protocol uses two paired seeds, a single decoder layer, width 16,
two attention heads, and zero model/adapter dropout. It does not establish that
LoRA always wins or that a tiny synthetic experiment represents large models.

## Run and replay

From the repository root, using an environment with the project dependencies:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m ai_journey.lora_cli \
  --config config/lora-comparison.json --output artifacts/lora-comparison.json

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -c \
  'from ai_journey.lora_cli import verify_main; raise SystemExit(verify_main())' \
  --report artifacts/lora-comparison.json
```

When the parent project's console scripts are installed, the equivalent commands
are `ai-journey-lora-compare --config config/lora-comparison.json --output PATH`
and `ai-journey-lora-verify --report PATH`. Both entry functions return 0 for
success and 2 for invalid inputs or failed execution/verification.

The comparison command validates its evidence and writes a canonical JSON report.
The verify command independently rebuilds every pretrained base and every arm.
It exactly compares model/frozen/merged fingerprints, training traces, metrics,
and paired summaries. A recomputed report digest does not make forged metrics
valid. Only `result.runtime` and the top-level report digest are excluded from
exact replay comparison. Runtime scope must still be CPU, float64, one thread;
Python/PyTorch version strings are descriptive metadata. Numeric differences on
another backend can fail exact replay even when version metadata is accepted.

## Data and scoring contract

`lora_data.synthetic_fixture()` generates 8 pretraining, 8 adaptation-training,
and 4 validation documents, all with distinct IDs and full texts. The vocabulary
is fit on the two training partitions, and validation characters must already be
covered. There are no downloads, credentials, personal data, or external corpora.
The partitions share artificial grammar and vocabulary; document separation is
not evidence of task diversity or natural-language generalization.

Training samples contiguous input/target windows within one document. Every arm
receives the same immutable schedule and consumes the same number of next-token
targets. The frozen control performs forward passes on those batches but makes
zero optimizer steps. The checked-in protocol uses 384 pretraining targets and
256 adaptation targets per arm per seed. Equal target exposure does not imply
equal computation or optimizer memory.

Validation scores every target index from 1 through `len(document)-1` exactly
once: 376 targets in total per arm. Each prediction uses up to `block_size` prior
characters from that document and only the final output logit. The first character
is context rather than an unconditional target. Context windows overlap while
scored targets remain unique, and position embeddings restart in each window.
No context crosses a document boundary. Short prefixes and final remainders are
included. NLL is the sum of target losses divided by the total target count;
perplexity is its exponential.

## Controlled adaptation and interpretation

Both attention projections in block 0 are adapted:
`attention.query_key_value` and `attention.projection`. LoRA uses AdamW with zero
weight decay and the same learning rate, batches, and gradient-clipping threshold
as full fine-tuning. Alpha is `rank * adapter_scale`, keeping alpha/rank fixed at
2 across the two ranks. Rank-1 trains 96 parameters; rank-2 trains 192. Reported
counts distinguish total, trainable, and frozen parameters; they do not measure
RAM or latency. Merge equivalence is checked on a training probe, so held-out
targets are not scored again to test the export.

The report binds fixture text/order, vocabulary, document splits, batch schedules,
and held-out coverage to digests. It records each initialized/pretrained source,
each fork's initial/final weights, and frozen state before and after adaptation.
The source and frozen state must remain unchanged. The runner preserves caller
Python, NumPy, and CPU PyTorch RNG states, dtype, thread count, and deterministic
algorithm settings on both success and failure.

Each of the six arm pairs reports per-seed NLL differences, their mean, sample
standard deviation, standard error, observed range, and candidate wins/ties.
Delta is candidate minus reference, so negative favors the candidate. These
statistics describe these few seeds on one synthetic split; they are not
confidence intervals or a basis for population-level claims. No best rank is
selected using validation results.

## Input bounds and verification

Config input is capped at 16 KiB, report input at 256 KiB. Duplicate JSON keys,
nonfinite literals, unknown/missing config fields, unpaired seeds, and out-of-range
controls are rejected. Combined training work units are capped at 10 million;
model dimensions, steps, seed count, and evaluation batches are also bounded.
These are allocation/execution limits, not a portable time benchmark.

Reports contain no timestamps or elapsed-time observations, so repeated runs in
the same numerical environment produce identical bytes. Validation, training,
staged-write, and replacement failures preserve prior output. Publication fsyncs
the temporary file before atomic replacement; directory-entry crash durability
is not claimed. Output cannot alias the config through a path, symlink, or
hardlink. Verification reads reports without modifying them.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s tests -p 'test_lora*.py' -v
```

The tests cover analytical uniform-model likelihood, document isolation, exact
target coverage, matched training exposure, frozen-state mutation detection,
paired uncertainty, canonical publication, rehashed metric/weight/trace forgeries,
bounded input rejection, failure cleanup, and fresh-process comparison/replay.
