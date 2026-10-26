# Controlled public-data KV-grouping experiments

This bounded float32, single-thread CPU protocol isolates KV grouping in a dense
RoPE decoder. The default compares four query heads with 4/2/1 KV heads, one
seed and two optimizer steps. It is a numerical engineering control, not a
trained language-model benchmark or evidence of GPU, speed or quality gains.

```sh
ai-journey-gqa-experiment plan --output gqa-plan.json
ai-journey-gqa-experiment public --source en-pud --snapshot english.conllu \
  --plan gqa-plan.json --output gqa-public.json
ai-journey-gqa-experiment verify --report gqa-public.json
ai-journey-gqa-experiment replay --report gqa-public.json --snapshot english.conllu
```

Use an existing exact immutable PUD snapshot accepted by the repository's
public-source loader, which verifies its pinned bytes, URL and license metadata.
There is no network download or unpinned source fallback. The `run` command also
accepts explicit public-safe `--train` and `--heldout` document-line files; its
`replay` requires both original files. Outputs are new-path-only and atomic.

Initialization starts from the existing reproducible dense full-MHA reference.
Common tensors, Q rows/bias and output projections are copied exactly. Each
compact K/V weight and bias is the mean over the corresponding contiguous
full-MHA head group. With KV=Q this is the identity mapping; with fewer KV heads
it is an explicit intervention, **not** identical full QKV initialization or
homologous K/V tensors. Training documents, sampled batches, seed, optimizer
controls, clipping and complete held-out coverage match across treatments.

Reports retain raw held-out losses/masks and document lengths/identities,
decomposed optimizer transitions, exact parameter/group inventories and declared
training score/cache dimensions. Fewer KV heads reduce stored projection and
native KV dimensions but do not reduce the query-head attention score matrix.
Whole-job admission charges every arm/seed, both held-out passes, initialization
and dense AdamW updates; resource units are explicit admission proxies, not FLOPs,
timing or a complete allocator peak. Initializer co-resident model parameters
are separately disclosed. The current protocol deliberately excludes MoE so KV
grouping does not acquire a second changed architecture or auxiliary objective.

Offline verification constructs no model and recomputes geometry, masks,
parameter/optimizer bindings, objectives, work bounds and paired summaries.
Hashes cannot authenticate a measurement or prove the initialization mapping;
full source replay reruns the actual models and requires exact recorded CPU
runtime and observations. Seed deltas are descriptive, not population confidence
or architecture-superiority evidence. Existing MoE/decoder reports are unchanged.

## Inspect actual native KV storage and numerical cache parity

```sh
ai-journey-gqa-experiment cache --report gqa-public.json --snapshot english.conllu \
  --prefix-count 4 --output gqa-cache.json
ai-journey-gqa-experiment cache-verify --report gqa-cache.json
ai-journey-gqa-experiment cache-replay --report gqa-cache.json --snapshot english.conllu
```

For document-file reports, use both original `--train` and `--heldout` instead
of `--snapshot`. The operation fully replays the declared training comparison
before accepting its final models. It inspects up to four held-out document
prefixes (bounded by context length and available documents), advancing one
token at a time from an empty native KV cache without context overflow.
Whole-job admission includes this complete replay and extra full/incremental
prefix forwards and model-state hashing before any model allocation.

Each step retains all 256 raw float32 full/cached logits, their recomputed error
and fixed absolute/relative tolerance, native KV shapes, actual compact backing
store bytes, and declared per-layer query-score dimensions. Storage is inspected
through the core cache auditor, not inferred from a logically expanded view.
The report's denominators count actual prefixes, growth steps and compared
logits; summaries recompute from those raw observations. Smaller native caches
do not imply fewer query-head score elements, better NLL, faster inference or
lower total peak process memory. No timing or GPU measurement is performed.
`cache-verify` checks internal arithmetic without a model; `cache-replay` repeats
both training and every cache observation against original source/runtime. A
fully fabricated but self-consistent record still requires actual replay.
