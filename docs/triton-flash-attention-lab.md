# A simplified flash attention forward pass in Triton

Day 81 of the curriculum follows GPU MODE Lecture 12, "Flash Attention", and
Lecture 36, "CUTLASS and Flash Attention 3". The tracker notes name the topics:
online softmax, tiling and IO-awareness. The exercise is "Implement a
simplified Flash Attention forward pass in Triton. Compare memory usage vs
naive attention." This lab is the Triton half of Day 81.

The lab does these steps:

1. It writes one Triton kernel text and its host launcher, with a SHA-256
   digest.
2. It writes a NumPy copy of the same blockwise algorithm and a float64 naive
   attention reference.
3. It holds each output to a declared error bound against the float64
   reference, in causal and non-causal cases, with ragged lengths and large
   scores.
4. It states a memory and traffic model of naive and flash attention, per
   head.
5. It seals one report, and an independent verifier checks the report on any
   CPU.

Nothing here runs on a GPU. The host has no NVIDIA GPU, and Triton has no
macOS wheel. Thus the kernel can run only in the Triton CPU interpreter on a
Linux host. The CI job `triton-flash-contracts` runs the interpreter tests on
a Linux runner. That job did not run yet. The lab reports no
Triton GPU time and tunes no config.

Ajay must still watch Lecture 12 and Lecture 36. This lab does not replace
them.

## Commands

```bash
PYTHONPATH=src python -m ai_journey.triton_flash_cli model
PYTHONPATH=src python -m ai_journey.triton_flash_cli check \
  --config config/day-81-triton-flash.json \
  --output /tmp/day-81-triton-flash.json
PYTHONPATH=src python -m ai_journey.triton_flash_cli verify \
  --report /tmp/day-81-triton-flash.json \
  --config config/day-81-triton-flash.json
```

The parser uses the program name `ai-journey-triton-flash`. An installed wheel
gives the same commands under that name.

- `model` prints one line for each row of the memory table. It needs no
  triton.
- `check` loads the plan, runs each case in the Triton CPU interpreter and
  builds the report. It verifies the report and then writes it atomically. It
  never writes over the plan.
- `verify` recomputes a report on the CPU. With `--config`, it also requires
  that the report comes from that plan. It never imports triton or torch.

The exit codes are the codes of the Day 73 to 80 labs:

- 0: the command succeeded.
- 1: the command read or built a report, but the report failed verification.
- 2: a usage error, an unreadable or invalid input, a refused output, or a
  check that broke a rule. Such a check writes no report.
- 3: this host cannot run the Triton CPU interpreter. Then `check` writes
  nothing.

## Hosts without triton

On the development Mac, `check` exits with code 3. It prints the reason
`triton_not_importable`, and it does not make the output directory. The
`model` and `verify` commands work on that host.

The backend imports triton and torch only when `check` makes the backend. It
sets `TRITON_INTERPRET=1` when the variable is unset. It refuses a host where
`triton.jit` does not give an `InterpretedFunction`.

The tests that need triton skip on a host without them. Other tests run the
exact pinned text under a small NumPy model of `triton.language`. That model
covers only the calls of the text, and it is not the Triton interpreter. Under
the model, the text matches the NumPy blockwise loop for every declared config,
within twice the declared error bound of each head. Both sides call
`numpy.matmul`, so on another BLAS build the bits can differ.

## The algorithm step by step

`triton_flash_kernel.TRITON_SOURCE` holds the kernel `flash_attention_forward`
and the launcher `launch_flash_attention`. Its SHA-256 digest is
`f0fda76e5cc299f6895e6ffe9a10eb054eeb44b35c48952b24d05d4e5f601a8c`. Q, K, V
and O are row-major float32 blocks of shape `[BH, N, D]`, where BH counts the
heads of all batches.

The launch grid is `(cdiv(N, BLOCK_M), BH)`. Thus one program computes one
query block of one head. Each program does these steps:

1. It loads its `BLOCK_M x D` tile of Q once. A row past N loads 0.
2. It sets the running max `m` to minus infinity, the running sum `l` to 0 and
   the output accumulator `acc` to 0, all in float32.
3. For each key block of `BLOCK_N` rows, it loads K as a `D x BLOCK_N` tile and
   V as a `BLOCK_N x D` tile. A row past N loads 0.
4. It computes the score tile `s = (Q K^T) * scale`, with `scale = 1 / sqrt(D)`
   rounded to float32.
5. It sets each score of a key past N to minus infinity. With `CAUSAL`, it
   also sets each score of a key after the query to minus infinity.
6. It finds the new row max `m_new` and the factor `alpha = exp(m - m_new)`.
7. It multiplies `l` and `acc` by `alpha`, so the old terms use the new max.
8. It adds the row sums of `p = exp(s - m_new)` to `l`, and `p V` to `acc`.
9. After the last key block, it divides `acc` by `l` and stores O, m and l.
   The store masks keep each write inside the first N rows.

This is the key loop of the text:

```python
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        # K is loaded as a HEAD_DIM x BLOCK_N tile, so no transpose.
        kt_ptrs = k_ptr + head + offs_n[None, :] * stride_xn + offs_d[:, None]
        v_ptrs = v_ptr + head + offs_n[:, None] * stride_xn + offs_d[None, :]
        kt = tl.load(kt_ptrs, mask=col_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)
        s = tl.dot(q, kt, input_precision="ieee") * scale
        if CAUSAL:
            keep = col_mask[None, :] & (offs_n[None, :] <= offs_m[:, None])
        else:
            keep = col_mask[None, :]
        s = tl.where(keep, s, float("-inf"))
        # Rescale the old state when this block raises the row max.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc, input_precision="ieee")
        m_i = m_new
```

The first key block holds key 0. Each query row can see key 0, so `m` is
finite after the first block. Thus the loop never computes
`-inf - -inf`. The row max adds `exp(0) = 1` to `l`, so each `l` is at least 1.

Both `tl.dot` calls set `input_precision="ieee"`. Thus a later GPU run stays
off TF32. The kernel also stores `m` and `l` for each row. The sum `m + log(l)`
is the log-sum-exp that a backward pass needs.

The kernel makes three simplifications:

- The causal kernel visits the key blocks above the diagonal and masks them.
  A full kernel skips these blocks.
- The kernel uses `tl.exp`. Many kernels fold the scale and `log2(e)` into one
  factor and use `tl.exp2`.
- D must be 16, 32, 64 or 128, because `tl.arange` needs a power of two.

## Declared configs

`FLASH_CONFIGS` declares four configs. The lab records them and tunes nothing.
The tile bytes count one Q tile, one K tile, one V tile and one score tile in
float32. The compiler decides what it keeps in registers or shared memory, so
the value is not the shared memory of a launch.

| Config | BLOCK_M x BLOCK_N | Warps | Stages | Tile bytes, D = 64 | Tile bytes, D = 128 |
| --- | --- | --- | --- | --- | --- |
| `m128_n64_w8_s3` | 128 x 64 | 8 | 3 | 98,304 | 163,840 |
| `m64_n64_w4_s3` | 64 x 64 | 4 | 3 | 65,536 | 114,688 |
| `m64_n32_w4_s4` | 64 x 32 | 4 | 4 | 40,960 | 73,728 |
| `m16_n16_w1_s1` | 16 x 16 | 1 | 1 | 13,312 | 25,600 |

The first two configs are larger tiles for a later GPU run. The last two
configs give the interpreter many short blocks.

## Why IO-awareness matters

A GPU has a small, fast memory on the chip and a large, slow memory outside
it. Naive attention writes the N x N score matrix S to the slow memory. Then
it reads S, writes the probability matrix P and reads P again. For a long
sequence, these N x N transfers can take most of the time of the kernel.

Flash attention keeps each score tile on the chip for one loop step. The
online softmax makes this possible, because it needs only `m` and `l` from the
earlier key blocks. Thus the kernel never writes S or P to the slow memory.

The price is a second read of K and V for each query block. The traffic model
below shows this trade. The FlashAttention paper states it as `O(N^2 d^2 / M)`
accesses for an on-chip memory of M values.

## The memory model

The model counts float32 bytes of one head. It uses these rules:

- Naive attention allocates Q, K, V and O, `4 * N * d * 4` bytes. It also
  materializes S and P, `2 * N * N * 4` bytes.
- Flash attention allocates the same Q, K, V and O. It adds only `m` and `l`,
  `2 * N * 4` bytes.
- Naive traffic is `(4 * N * N + 4 * N * d) * 4` bytes, with no cache.
- Flash traffic is `(2 * N * d + 2 * N * d * cdiv(N, 128) + 2 * N) * 4` bytes.
  It reads K and V once for each query block of 128 rows.

This is a model of allocated intermediates and of bytes moved. It is not a
measured GPU footprint. An allocator, a workspace, a cache or a fused library
kernel can change the real numbers.

| N | d | Naive MiB | Flash MiB | Memory ratio | Naive traffic MiB | Flash traffic MiB | Traffic ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 64 | 2.50 | 0.50 | 4.96 | 4.50 | 1.25 | 3.59 |
| 1024 | 64 | 9.00 | 1.01 | 8.93 | 17.00 | 4.51 | 3.77 |
| 2048 | 64 | 34.00 | 2.02 | 16.87 | 66.00 | 17.02 | 3.88 |
| 4096 | 64 | 132.00 | 4.03 | 32.74 | 260.00 | 66.03 | 3.94 |
| 8192 | 64 | 520.00 | 8.06 | 64.50 | 1032.00 | 260.06 | 3.97 |
| 512 | 128 | 3.00 | 1.00 | 2.99 | 5.00 | 2.50 | 2.00 |
| 1024 | 128 | 10.00 | 2.01 | 4.98 | 18.00 | 9.01 | 2.00 |
| 2048 | 128 | 36.00 | 4.02 | 8.96 | 68.00 | 34.02 | 2.00 |
| 4096 | 128 | 136.00 | 8.03 | 16.93 | 264.00 | 132.03 | 2.00 |
| 8192 | 128 | 528.00 | 16.06 | 32.87 | 1040.00 | 520.06 | 2.00 |

The memory ratio grows with N, because only naive attention holds N x N
values. The traffic ratio stays near `2 * 128 / d`, because flash attention
reads K and V again for each query block. A larger BLOCK_M lowers this
traffic, but it needs more on-chip memory.

A script traced the NumPy allocations on the host CPU with `tracemalloc`. It
ran on 2026-11-23 at 19:02 UTC, with load averages 7.74, 9.70 and 10.68. For
N = 1024 and d = 64, the naive form had a peak of 5,245,830 bytes. The
blockwise loop had a peak of 411,260 bytes. One N x N float32 array is 4,194,304 bytes. The NumPy
naive form reuses S for P in place, so it holds one N x N array, not two. These
numbers are host CPU allocations, not a GPU footprint. A test checks the same
property with seed 74.

## Correctness and the bound

Each case draws float32 Q, K and V from
`numpy.random.default_rng([seed, stream, index])`. Family `normal` holds
standard-normal values. Family `large` multiplies Q and K by 8, so the scores
reach several hundred. Without the max shift, `exp(s)` overflows float32 above
88.7. The online softmax still gives finite rows.

The pinned plan `config/day-81-triton-flash.json` has six cases, with seed 81:

| BH | N | D | Causal |
| ---: | ---: | ---: | --- |
| 1 | 1 | 16 | no |
| 2 | 17 | 16 | yes |
| 2 | 70 | 32 | no |
| 1 | 129 | 64 | yes |
| 1 | 77 | 64 | no |
| 1 | 40 | 128 | yes |

The plan requires that each config meets a short last query block and a short
last key block. Each case runs with all four configs and both families, so a
check holds 48 runs.

The bound of each head comes from a first-order rounding analysis, times a
safety factor of 2. It is not a proof. With `u = 2**-24`, the terms are:

- `sigma = gamma_D * a_max + 2u * S` bounds the error of one score. Here
  `a_max` is the largest `(|q| . |k|) * scale`, and S is the largest `|s|`.
- `E = 2 sigma + 4u * S + 8u * (n_kb + 1)` bounds the log error of one weight.
  It allows 8u for each exponential, and `n_kb` is the count of key blocks.
- `R = 2 gamma_(N + n_kb) + 2u * (n_kb + 1)` bounds the sums and the rescale
  products.
- The output bound is `2 * ((exp(2E) - 1) + R + u) * max|V|`.
- The log-sum-exp bound is `2 * (E + R)`.

The bound is loose. In the NumPy stand-in on this Mac, the worst output ratio of the
pinned plan is 0.0014, and the worst log-sum-exp ratio is 0.0152. The last
digits depend on the BLAS library under NumPy, so another host can differ a
little. Under the
NumPy model of `triton.language`, a text without the rescale step breaks the
bound. The factor is about 760 in the case N = 129 with D = 64.

The bound is for the NumPy exponential of the CPU interpreter. No interpreter
run checked the bound. A GPU `tl.exp` uses an approximate base-2
exponential, so a later GPU run needs its own bound.

Each run gets O, m and l buffers with whole query blocks of rows. Every slot
holds a quiet-NaN sentinel before the run. After the run, the rows past N must keep their
sentinels, and every value in the first N rows must be finite. A text without
its store mask fails this rule.

## The report and the verifier

The report holds the plan, the pinned text, the configs, the rules, every run
and the memory table. It also holds every output value of O, m and l in
base64. A SHA-256 seal covers the canonical JSON. The seal detects drift. It is
not a signature.

The verifier uses other code than the run. It does these steps:

1. It draws Q, K and V again from the seed and checks their digests.
2. It decodes every stored value and checks its digest.
3. It computes a float64 reference by a second formula, `P = exp(S - LSE)`.
4. It computes each error and each bound again.
5. It requires that each recomputed ratio is at most 1.
6. It computes the memory table again from the closed forms of the rule.

The float64 reference goes through `numpy.matmul`, and its order of additions
depends on the BLAS build. Thus a recomputed error or bound must match its
record within a relative tolerance of 1e-6, not bit for bit.

Only a report from the interpreter backend gets the interpreter label. A
report from the NumPy stand-in of the tests gets the synthetic label, and the
verifier keeps it so.

## The honest boundary

- The host has no NVIDIA GPU, no CUDA and no Triton wheel. No GPU compiled or
  ran the text.
- The kernel can run only in the Triton CPU interpreter on a Linux host with
  triton 3.9.0 and CPU torch. No such run happened.
- The CI job `triton-flash-contracts` installs triton 3.9.0 and CPU torch,
  and sets `TRITON_INTERPRET=1`. It runs the tests and fails on a skipped
  test. Then it runs `check` on the pinned plan and verifies the report. That
  job did not run yet.
- Ajay must still read the log of the first run of that job.
- The interpreter runs each program in turn as Python and NumPy code. Its time
  says nothing about a GPU, so the lab reports no time.
- The memory section is a model. It is not a measured GPU footprint.
- The lab tunes no config. It only records the declared configs.

## Reading notes on FlashAttention-2 and FlashAttention-3

These notes are a summary from memory. Nobody checked them against the
papers. The lab implements none of these features, and Ajay must still read
both papers.

FlashAttention-2 (Dao, 2023) changes the work split:

- It runs the query blocks of one head in parallel, as this grid does.
- It splits the query rows between the warps of a block, not the key rows. This
  removes the exchange of partial results between warps.
- It cuts the non-matmul work, because matmul units are much faster than the
  other units.
- It skips the key blocks above the diagonal of a causal mask.

FlashAttention-3 (Shah and others, 2024) targets the Hopper H100 GPU:

- Warp specialization: producer warps load tiles, and consumer warps compute.
- TMA: a hardware unit, the Tensor Memory Accelerator, copies tiles between
  the slow memory and shared memory.
- Asynchronous WGMMA: the matmuls of one block overlap the softmax of another
  block.
- FP8: low-precision matmuls with block scaling, to keep the error small.

The title of Lecture 36 names CUTLASS and FlashAttention-3. Ajay must still
watch it for the CUTLASS details.

## What Ajay must still do

- Ajay must still watch Lecture 12 and Lecture 36.
- Ajay must still read the log of the first run of the CI job
  `triton-flash-contracts`.
- Ajay must still read the FlashAttention-2 and FlashAttention-3 papers.
