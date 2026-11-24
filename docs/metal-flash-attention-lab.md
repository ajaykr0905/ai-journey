# Flash attention on the Apple GPU

Day 81 of the curriculum (Track B) follows GPU MODE Lecture 12, "Flash
Attention". Ajay must still watch Lecture 12. This guide does not summarize
the lecture, and it makes no claim about what the lecture shows.

Triton has no macOS build, so this lab writes attention forward kernels in
Metal. The kernels run on the Apple GPU through `torch.mps.compile_shader`.
One run compares three sides on the same device rows:

1. N0, naive attention in three kernels. The first kernel writes the scores
   `S = 0.125 Q K^T` to an N x N device buffer. The second kernel turns each
   row of S into a softmax in place. The third kernel writes `O = P V`.
2. F1, a flash-attention forward kernel. One threadgroup takes one block of
   32 queries. It loads K and V in tiles of 32 rows into threadgroup memory.
   It keeps a running max and a running sum for each query. S and P never go
   to device memory.
3. `torch.nn.functional.scaled_dot_product_attention` on the MPS device,
   with `is_causal` in the causal mode.

Each kernel has a full variant and a causal variant. The head dimension is
64, and the scale is `1 / sqrt(64) = 0.125`. Each sequence has its own
length, so the same kernels take dense and ragged batches.

Nothing here runs a CUDA kernel, uses an NVIDIA GPU, runs PTX or runs a
Triton kernel on a GPU. The lab times forward passes only. PyTorch does not
report which kernels its call runs. The lab reads no hardware counters. Every
timing is a host-specific Apple GPU (Metal) measurement at the load averages
that the report records.

## What flash attention changes

Attention for one head computes `softmax(Q K^T / sqrt(d)) V`. For N tokens,
the scores hold N x N floats. N0 writes them to device memory, reads and
writes them again in the softmax and reads them once more for `P V`. That
traffic is 16 bytes for each visible pair of a query and a key.

The FlashAttention paper (T. Dao and others, 2022) describes a different
order of the same arithmetic. A kernel takes the keys in tiles. For each tile,
it computes the scores, updates the running max `m`, multiplies the old sum
`l` and the old output by `exp(m_old - m_new)`, and adds the new terms. After
the last tile, it divides the output by `l`. This is the online softmax of
Day 79, with a vector of 64 values in place of one sum. Ajay must still check
this description against Lecture 12.

## Commands

The `run` command needs a gpu-ceilings report from the Day 74 lab. The
recorded Day 74 report of the test suite names macOS 27.2, and this host has
macOS 27.2.1. A measured run refuses a ceiling from another platform, so the
pinned run used a fresh Day 74 run:

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
```

Then the `run` command makes a report, and the `verify` command recomputes
it:

```bash
python -m ai_journey.metal_flash_cli run \
  --config config/day-81-metal-flash.json \
  --ceilings /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-81-metal-flash.json
python -m ai_journey.metal_flash_cli verify \
  --report /tmp/day-81-metal-flash.json \
  --config config/day-81-metal-flash.json \
  --ceilings /tmp/day-74-gpu-ceilings.json
```

On a shared host, run each timed command under the GPU lock. The parser uses
the program name `ai-journey-metal-flash`. An installed wheel gives the same
commands under that name.

The `run` command prints these lines:

1. One line for each step of the plan, with its predicted verdict.
2. One line with the load averages before and after the run.
3. One line with both ceilings and the digest of their report.
4. One line for the exp probe.
5. One line for each correctness case and mode, with the bound ratio and the
   repeat result of N0 and F1.
6. One line for each timing point, with the median time, the IQR and the
   GFLOP/s of each side, and the memory of each side.
7. One line for each timing point, with the ratio, the verdict and the result
   of each hypothesis.
8. One line with the count of Python warnings that the run caught.
9. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`metal_flash_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `metal_flash_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the summary
line with the prefix `verified_`. It never imports torch. With `--config`, it
requires the exact config of the run. With `--ceilings`, it requires the
exact ceiling report of the run.

The exit codes are the codes of the Day 73 to 80 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

## The kernels

| Kernel | Group | Threadgroup bytes | Barriers | Text digest |
| --- | --- | --- | --- | --- |
| `flash_n0_scores` | 256 | 256 | 1 | `70d5a1b87e7d12d44fca9c154d6b9d2203fa076cdde717a7a4cb842012c6a68c` |
| `flash_n0_softmax` | 256 | 64 | 2 | `6b4312452989d65df14605d9785b3eb05261c21613c1ce08349b1bbeff3a3d0d` |
| `flash_n0_pv` | 64 | 0 | 0 | `4064d4e534c1324ab559ff4de1c3155f9763d8a334060e99fc191eb143bdd822` |
| `flash_n0_scores_causal` | 256 | 256 | 1 | `e29642489cc8b6dcad4dc3d774de4af23fa800712a104207a73e2c43c6880cf8` |
| `flash_n0_softmax_causal` | 256 | 64 | 2 | `62dca87a332165243c3513eaaf0e2bd535167c895080a087a630d963f774e2a4` |
| `flash_n0_pv_causal` | 64 | 0 | 0 | `13d7a9f80f34c66ce735800f6327c8526a50514eacb950fcf4118ddfcf916c6c` |
| `flash_f1` | 128 | 16384 | 2 | `bd4c160e839abc38c5469b97c11ef44f5233917c3b0051f32638e4451fb86876` |
| `flash_f1_causal` | 128 | 16384 | 2 | `2ebbcb676b46632e910ce44c754d73d8baca1e1ed0b092429e37dd644def7b65` |

The library digest is
`2787a67f6dc7cbabddea66fad281db77a73292e856094ad9fc9ab75a77229bc4`. It also
holds the probe kernel `flash_probe_exp` for `precise::exp`.

N0 runs one threadgroup for each query row. The scores kernel copies the
query row into threadgroup memory, and each thread computes the scores of its
keys with 64 `fma` operations. The softmax kernel uses `simd_max`,
`simd_sum` and one threadgroup step for the max and for the sum.

F1 runs 128 threads for each block of 32 queries. Four threads share one
query row, and thread `part` owns the columns `part + 4 t` of the head. Each
thread adds its 16 products, and two `simd_shuffle_xor` steps add the four
parts. The K and V tiles take 16384 bytes of threadgroup memory.

These rules keep each kernel correct:

- Every barrier is in the kernel body or in the K/V tile loop. The loop bound
  comes from the sequence length and the query block only, so every thread of
  a group reaches every barrier. A test reads each kernel text and checks
  this rule.
- A query row at or past the sequence length writes nothing. A key row at or
  past the length is never read.
- The kernels call `precise::exp` and `precise::divide`, so the compile
  options do not change the math functions.

## The plan

The plan names three steps. Each step compares a first side with a base side
by the ratio rule of Day 76.

| Step | First side | Base side | Predicted verdict |
| --- | --- | --- | --- |
| `n0_over_torch` | N0 | PyTorch call | slower |
| `f1_over_n0` | F1 | N0 | faster |
| `f1_over_torch` | F1 | PyTorch call | slower |

The plan digest is
`23a502f6a118c208a02fbc5ed7b36496edabe8788c9afc624e3b9b0bdfa0cbb5`. The
pinned config names this digest, so a config for other hypotheses cannot run.

A scratch prototype of F1 ran before the lab wrote the plan. At 512, 2048 and
4096 tokens, it took about 2.5 to 3 times the time of the PyTorch call. The
third hypothesis records this fact, so its prediction came after a first
look at the data. The other two predictions came before any timing of N0.

## Correctness

Each case draws seeded sequences in a gapped layout. Three guard rows follow
each sequence in Q, K, V and O. Each input guard row holds `2**100`. A query
that reads a guard row as a key gets a finite score near plus or minus
`2**100`. A positive score gives the guard row all the weight, and the output
breaks the bound. A negative score gives it a weight of 0, so that query
cannot show the read. About half of the queries get a positive score, so the
check relies on the many queries of each case. Before the run, each output
row holds a quiet NaN with one known bit pattern.

The pinned cases cover these sequences and inputs:

- Sequences of 1, 31, 33, 100 and 257 tokens in one ragged batch.
- Sequences of 1000 and 2049 tokens with `wide_scores` inputs. There, q and k
  come from 5 times a standard normal. The scores reach past 88.7, so a
  float32 exp with no max subtraction overflows.
- Sequences of 64 and 700 tokens with values near 1000.
- Two sequences of 1024 tokens.

The reference is float64 from the float32 inputs. It adds the 64 products of
each score in column order and uses the Day 79 `exp64`. It adds the keys in
order, and a test checks that NumPy keeps this order. So each IEEE 754 host
computes the same float64 bits.

The declared bound is a perturbation bound for each output value. Its scale
is `a = sum_j P_j |v_j|`, the softmax mean of `|v|`. The bound counts these
errors:

- The roundings of each score, from the norms of q and k.
- One `precise::exp` call for each weight and one for each rescale.
- The roundings of the sums, the rescales and the final division.
- The error of the float64 reference.

One bound covers N0 and F1, because F1 has the more exp calls and roundings.
The bound assumes that `precise::exp` has a relative error of at most 4
units of `2**-23`. The run measures this assumption first with the Day 79
probe, and a failed probe stops the run.

N0 and F1 run twice on each case in each mode. Three rules apply to the
output of each run, and a case that breaks one stops the run:

1. Each gap row keeps its sentinel bits.
2. Each output is finite.
3. Each value is within its bound of the reference.

Determinism is a measurement, not a rule. The run records whether the
repeats of a kernel gave the same bits.

## Timing

The timing is the Day 74 and Day 76 method, with the Day 78 rotating timer:

1. Each side runs 2 untimed warmup samples.
2. Each timed sample calls `torch.mps.synchronize()`, reads
   `time.perf_counter_ns`, launches a batch of calls, synchronizes again and
   reads the clock again. One call of N0 is its three launches.
3. Repeat `r` runs side `r mod 3` first, so each side runs first in 4 of the
   12 repeats.

Each shape holds 32768 tokens: 64 sequences of 512 tokens, 32 of 1024, 16
of 2048, 8 of 4096 and 4 of 8192. A batch holds the fewest whole calls whose
compulsory bytes reach 64 MiB. The compulsory bytes are 16 x 64 bytes for
each token: read Q, K and V once and write O once. Each shape gives 2 calls
for each sample.

Each GFLOP/s counts 4 x 64 operations for each visible pair of a query and a
key. A causal point has `L (L + 1) / 2` pairs for each sequence. The count
leaves out the exp, the max and the sum.

After the last sample, the run checks 32 query rows of each sequence. The
rows spread evenly from the first row to the last. N0 and F1 must be within
the declared bound. The PyTorch call must be within a sanity bound of
`2**-14 a`, or the run stops.

The ratio rule is the rule of Day 76, with the declared `min_faster_ratio` of
1.1:

- `faster`: the ratio is 1.1 or more, and the p75 sample time of the first
  side is below the p25 time of the base side.
- `slower`: the inverse ratio is 1.1 or more, and the p75 sample time of the
  base side is below the p25 time of the first side.
- `no_clear_difference`: any other result.

## Memory

Before the timing of each point, the run measures each side alone. The run
calls `torch.mps.empty_cache()`. Then the side allocates its own inputs,
output and scratch, runs one call and frees everything. The run reads two
counters:

- `torch.mps.current_allocated_memory()`: the bytes of live tensors in the
  PyTorch MPS allocator, rounded up to its blocks.
- `torch.mps.driver_allocated_memory()`: the bytes that the Metal driver
  holds for this process. This count includes the allocator cache and the
  buffers of the MPS frameworks.

The run reads both counters before the side allocates, after it allocates and
after the call. A second thread polls both counters while the call runs, and
the run keeps the largest value of each.

The counters cannot show these things:

- A buffer that lives and dies between two polls.
- Threadgroup memory, registers and other on-chip storage. The K and V tiles
  of F1 are in threadgroup memory.
- Memory that Metal or the operating system holds outside these counters.
- Which kernel PyTorch ran.

The driver counter grows in large heaps, so it is coarse.

Each side must allocate these bytes by its construction:

- N0: Q, K, V and O, the score buffer of `4 L * L` bytes for each sequence,
  and its index buffers.
- F1: Q, K, V and O and its index buffers.
- The PyTorch call: Q, K and V, and the O that the call returns. Its scratch
  is not known by construction, because PyTorch names no kernel.

## One run on the development host

This run used the pinned config on 2026-11-24 at 09:02 IST. The host was a
shared Darwin arm64 host with an Apple GPU, macOS 27.2.1, CPython 3.11.14,
NumPy 1.26.4 and PyTorch 2.15.0. The environment did not set
`PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. The run held the GPU
lock. Other jobs used the CPU.

Both ceilings came from a fresh Day 74 run at 08:47 IST under the same lock.
The bandwidth ceiling is 107.71 GB/s, the median of `stream_read_reduce` at
256 MiB. The FMA ceiling is 4665.5 GFLOP/s. That run recorded load averages
of 11.73, 14.03 and 16.29 before and 10.22, 13.63 and 16.12 after. Its digest
is `cd75564918285983fe22b4106b5084b44ef42ed9313ca2a8ef19fa164e68fd86`. An
earlier Day 74 run at 00:12 IST gave 123.67 GB/s at load averages near 12 to
15. The second run wrote over its file, and no report cites it.

The Day 81 report recorded load averages of 8.87, 8.36 and 11.07 before the
run. It recorded 11.94, 9.21 and 11.24 after the run. The `uptime` command
printed 8.77, 8.34 and 11.08 before the command and 11.38, 9.14 and 11.21
after it.

The report digest is
`c9790d0451875b74f2193cf9c1e472356cf798a8cdd81affa9ed54fe6a0a348c`. The run
took 41.9 seconds after the library compiled. The run caught 0 Python
warnings, and the capture began before the import of torch. Each pipeline
allowed 1024 threads, reported a SIMD width of 32 and reported its declared
threadgroup bytes. The CPU verifier took about 8 seconds.

### Correctness results

- All 4 cases met the three rules in both modes and in both runs.
- The largest bound ratio was 0.0088 for N0 and 0.0088 for F1. The bound is
  loose: each measured error is less than 1 percent of its bound.
- Each kernel gave the same bits in both runs of each case.
- The exp probe measured 1.406 units in the last place, within its
  assumption.
- For the case with `wide_scores` inputs, a float32 exp of the largest score
  with no max subtraction overflows. N0 and F1 gave finite outputs within the
  bound.

### The timing table

Each time is the median time of one call in milliseconds, with the IQR in
parentheses. The ratio divides the median time of the base side by the
median time of the first side. "F1 of FMA" divides the GFLOP/s of F1 by the
FMA ceiling.

| Shape | Mode | N0 ms | F1 ms | PyTorch ms | GFLOP/s N0 / F1 / PyTorch | F1 of FMA | F1 over N0 | F1 over PyTorch | N0 over PyTorch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 x 512 | full | 14.28 (0.29) | 4.62 (0.23) | 1.84 (0.32) | 301 / 929 / 2333 | 0.199 | 3.09 | 0.40 | 0.13 |
| 64 x 512 | causal | 7.42 (0.31) | 2.52 (0.28) | 0.98 (0.11) | 290 / 855 / 2187 | 0.183 | 2.95 | 0.39 | 0.13 |
| 32 x 1024 | full | 29.25 (0.72) | 8.92 (0.40) | 3.30 (0.31) | 294 / 963 / 2602 | 0.206 | 3.28 | 0.37 | 0.11 |
| 32 x 1024 | causal | 14.37 (0.24) | 4.95 (0.26) | 1.75 (0.23) | 299 / 869 / 2455 | 0.186 | 2.90 | 0.35 | 0.12 |
| 16 x 2048 | full | 61.48 (1.61) | 17.78 (0.30) | 6.10 (0.47) | 279 / 966 / 2815 | 0.207 | 3.46 | 0.34 | 0.10 |
| 16 x 2048 | causal | 29.00 (0.50) | 9.36 (0.22) | 3.12 (0.19) | 296 / 918 / 2755 | 0.197 | 3.10 | 0.33 | 0.11 |
| 8 x 4096 | full | 134.68 (4.23) | 35.36 (1.07) | 12.44 (0.53) | 255 / 972 / 2763 | 0.208 | 3.81 | 0.35 | 0.09 |
| 8 x 4096 | causal | 60.90 (4.08) | 18.71 (1.93) | 6.37 (0.71) | 282 / 919 / 2696 | 0.197 | 3.26 | 0.34 | 0.10 |
| 4 x 8192 | full | 285.65 (15.42) | 77.50 (6.75) | 26.12 (6.01) | 241 / 887 / 2631 | 0.190 | 3.69 | 0.34 | 0.09 |
| 4 x 8192 | causal | 135.89 (3.35) | 38.65 (1.54) | 14.09 (1.42) | 253 / 889 / 2439 | 0.191 | 3.52 | 0.36 | 0.10 |

The verdicts were `faster` for F1 over N0 at all 10 points. They were
`slower` for N0 and for F1 over the PyTorch call at all 10 points. The
predictions held at 30 of 30 step results.

### The memory table

Each value is in MiB. "Allocated" is the growth of the live tensor count
after the side allocated. "Driver peak" is the largest polled driver count
during the call, less the driver count before the side allocated. The causal
points gave the same values, except for a driver peak of 32.2 MiB for the
PyTorch call at 512 tokens.

| Shape | N0 allocated | N0 driver peak | N0 by construction | F1 allocated | F1 driver peak | PyTorch allocated | PyTorch driver peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 x 512 | 96.3 | 1064.0 | 96.3 | 32.0 | 40.0 | 24.0 | 32.0 |
| 32 x 1024 | 160.3 | 1064.0 | 160.3 | 32.0 | 40.0 | 24.0 | 32.0 |
| 16 x 2048 | 288.3 | 1064.0 | 288.3 | 32.0 | 40.0 | 24.0 | 32.0 |
| 8 x 4096 | 544.3 | 552.0 | 544.3 | 32.0 | 40.0 | 24.0 | 32.0 |
| 4 x 8192 | 1056.3 | 1064.0 | 1056.3 | 32.0 | 40.0 | 24.0 | 32.0 |

By construction, F1 and the PyTorch call need 32.0 MiB at each shape: Q, K,
V and O of 8 MiB each. The PyTorch call allocated its Q, K and V, 24 MiB,
before the call. The live tensor count then grew by 8.0 MiB during the call,
the size of its O.

### How to read the results

- F1 took 0.26 to 0.34 of the time of N0. In the full mode, the ratio was
  3.1 at 512 tokens, 3.8 at 4096 tokens and 3.7 at 8192 tokens.
- F1 ran at 855 to 972 GFLOP/s, about 0.2 of the FMA ceiling. F1 uses scalar
  `fma` for both products. The lab cannot show which resource limits F1,
  because it reads no counter.
- The PyTorch call ran at 2187 to 2815 GFLOP/s, about 2.5 to 3.0 times the
  rate of F1. This lab cannot tell which kernel PyTorch ran.
- The live tensor count of N0 grew by its bytes by construction, the score
  buffer included. At 512 tokens the two counts are equal. At the other shapes
  the live count is 384 to 720 bytes larger, which agrees with the rounding of
  the allocator to its blocks. The driver count grew in heaps of about 1 GiB, so it
  shows 1064 MiB even when the scores need 64 MiB.
- For the PyTorch call, no counter showed an N x N buffer. At 8192 tokens,
  the scores of the 4 sequences need 1024 MiB in float32. The polls saw no
  growth past the 32 MiB of Q, K, V and O. This result does not prove that
  no such buffer existed. A buffer that lived between two polls, or memory
  outside the two counters, does not show.
- The causal mode took about half the time of the full mode for each side.
  It has about half the pairs.

## The boundary of the evidence

These are wall-clock timings and allocator readings of one Apple GPU on one
shared host at the load averages above. They do not measure or predict CUDA,
NVIDIA, Triton or FlashAttention performance on other hardware. The lab timed
forward passes only. It read no hardware counter, so it observed no cache
hit, stall or occupancy. Each byte and operation count is a model. The
PyTorch side names a call, not a kernel.

The CPU verifier recomputes the config, the plan, each input, reference and
bound digest, and each launch and admission. From the raw samples, it
recomputes each statistic, rate, ratio, verdict and ceiling fraction. From
the raw readings, it recomputes each memory difference and each count of
bytes by construction. It cannot recompute the fields that only the GPU run
saw:

- The raw samples of each side. The verifier checks their form only.
- The digest and errors of each output that differs from the rounded
  reference. The verifier checks them against the bound.
- `identical_repeats`, because the CPU cannot run the GPU again.
- The results of the exp probe. The verifier checks them against the rule.
- The pipeline limits. The verifier checks their form and rebuilds the
  admissions from them.
- Every allocator reading and poll count. The verifier checks their form.
- The PyTorch checks, the warnings, the MPS switches and the load averages.
  The verifier checks their form.

The report holds the same list in its `verification` section. A matching
digest shows internal consistency. It cannot prove that a timing was
observed or that a GPU ran.

## The recorded report for CI

`tests/test_metal_flash_recorded.py` holds a short real report from this
host, byte for byte. The run used two correctness cases, two points of 4
sequences of 256 tokens and three repeats, on 2026-11-24 at 08:47 IST. The
`uptime` command printed load averages of 9.80, 13.49 and 16.05 before the
run. At these small sizes, the cost of each synchronize call sets most of
the time. Three of its six predictions missed, and the report records that
result. The report is there to be verified on a Linux host without torch,
not to show speed.

The CI job `metal-flash-contracts` runs the tests with CPU-only torch. The
tests do not depend on the operating system of the host. Each test that needs
MPS skips where MPS is not available. In that job, the installed `run` must
stop with exit code 3 and write nothing. The installed `verify` must accept
the recorded report against its recorded config.

## Work that Ajay must still do

- Ajay must still watch Lecture 12.
- Ajay must still check the FlashAttention facts in this guide against the
  lecture and the paper.
- Ajay must still read the log of the first run of the CI job
  `metal-flash-contracts`.
- Ajay must still compare F1 with a CUDA flash-attention kernel. That
  comparison needs a host with an NVIDIA GPU.
