# Fused softmax kernels on the Apple GPU

Day 79 of the curriculum (Track B) follows GPU MODE Lecture 18, "Fused
Kernels". Ajay must still watch Lecture 18. This guide does not summarize the
lecture, and it makes no claim about what the lecture shows.

This lab builds one float32 softmax over the rows of a batch in four steps:

1. U0, unfused: three kernels. The first kernel writes the max of each row,
   the second kernel writes the sum of `exp(x - max)`, and the third kernel
   writes the outputs. The max and the sum go through device memory.
2. F1, fused: one kernel. Each thread loads its part of the row once into
   registers. Two trees in threadgroup memory give the max and the sum.
3. F2: F1 with SIMD-group reductions. The kernel uses `simd_max` and
   `simd_sum` and one threadgroup step for each value.
4. F3, online softmax: one pass keeps a running max and a running sum, then
   one merge step and one write pass.

Each step has a digest and a hypothesis that the lab declared before any
timing. One run checks every step against a float64 reference and times
every step with `torch.softmax(x, dim=-1)` on the same device rows. U0 is the
baseline, and `torch.softmax` is the vendor side.

Nothing here runs a CUDA kernel, uses an NVIDIA GPU or runs a Triton kernel
on a GPU. PyTorch does not report which kernel `torch.softmax` runs. The lab
reads no hardware counters. Every timing is a host-specific Apple GPU (Metal)
measurement at the load averages that the report records.

## Commands

The `run` command needs a gpu-ceilings report from the Day 74 lab. The pinned
run used the recorded Day 74 report of the test suite. This command writes it
to a file:

```bash
PYTHONPATH=src:tests python -c "from pathlib import Path; from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as T; Path('/tmp/day-74-gpu-ceilings.json').write_text(T + chr(10), encoding='ascii')"
```

A fresh Day 74 run gives a new ceiling for the same file:

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
```

Then the `run` command makes a report, and the `verify` command recomputes
it:

```bash
python -m ai_journey.metal_fused_cli run \
  --config config/day-79-metal-fused.json \
  --ceilings /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-79-metal-fused.json
python -m ai_journey.metal_fused_cli verify \
  --report /tmp/day-79-metal-fused.json \
  --config config/day-79-metal-fused.json \
  --ceilings /tmp/day-74-gpu-ceilings.json
```

The parser uses the program name `ai-journey-metal-fused`. The package does
not install this command as a console script yet, so use `python -m`.

The `run` command reads the ceiling report through the Day 74 verifier. Then
it compiles the kernels, checks them, times them, checks its own report and
writes the report. It prints these lines:

1. One line for each plan step, with its base and its predicted verdicts.
2. One line with the load averages before and after the run.
3. One line with the bandwidth ceiling and the digest of its report.
4. One line with the exp probe: the largest error of `precise::exp`.
5. One line for each step that its pipeline refused, with the reason.
6. One line for each correctness case, with the bound ratio of each step,
   the repeat result and the bit result of F1.
7. One line for each timing point, with the median GB/s and the IQR range of
   each side.
8. For each timing point, one line for each step, with its ratios and the
   result of its hypothesis. Then one line with the fastest step.
9. One line with the count of Python warnings that the run caught.
10. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`metal_fused_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `metal_fused_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the same
summary line with the prefix `verified_`. With `--config`, `verify` also
requires that the report comes from that exact config. With `--ceilings`, it
requires that the run used that exact ceiling report. The `verify` command
never imports torch.

The exit codes are the codes of the Day 73 to 77 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

On a Linux host, `run` names the reason `unsupported_operating_system`. It
writes the reason to standard error, prints nothing to standard output and
makes no output file.

## Why fusion saves memory traffic

A softmax must read each element once and write each output once. For `n`
elements, that is `8 n` bytes of float32 data. The lab calls these bytes the
compulsory bytes, and it divides them by the median time to get the
effective GB/s of every side.

U0 reads the row in each of its three kernels and writes the row once. A
model with no cache thus gives U0 `16 n` bytes. U0 also moves `44 r` bytes for
`r` rows: the first slot and the length of each row in each kernel, the max
and the sum. A fused step reads the row once into registers and writes it
once. The model gives it `8 n + 8 r` bytes, about half the bytes of U0.

The model counts device bytes with no cache. It is a model, not a
measurement. A cache can serve a second read of a row, and no counter shows
whether it did.

## The online softmax update

A thread of F3 keeps a running max `m` and a running sum `d` of
`exp(v - m)` over its values. At first, `m` is the most negative finite
float32 and `d` is 0. For each new value `v`:

- If `v > m`, the thread rescales the old sum to the new max:
  `d = d * exp(m - v) + 1`, and then `m = v`.
- Otherwise, the thread adds the new term: `d = d + exp(v - m)`.

The kernel computes `exp(min(m, v) - max(m, v))` once for both cases. Two
partial pairs `(m1, d1)` and `(m2, d2)` merge into one pair:
`M = max(m1, m2)` and `D = d1 * exp(m1 - M) + d2 * exp(m2 - M)`. Each SIMD
group merges its 32 pairs, and one threadgroup step merges the SIMD groups.
The write pass then gives `exp(x - M) / D` for each element.

The update comes from M. Milakov and N. Gimelshein, "Online normalizer
calculation for softmax" (2018). In this lab, F3 also keeps the row in
registers, so F3 reads each element once, as F1 and F2 do. Its change is one
threadgroup step in place of two, at the cost of two `precise::exp` calls for
each element in place of one.

## Kernels

Every kernel runs one threadgroup per row. The group size G is a power of two
from 32 to 1024. It is the fewest threads whose 16 slots hold the longest row
of the batch. Thread `lane` owns the columns `lane + i * G` for `i < 16`. A
row of 16384 elements thus needs 1024 threads.

The first element of the row of group `r` is at `starts[r]`, and the row
holds `lengths[r]` elements. Dense, packed ragged and gapped layouts thus use
the same kernels. A column at or past the length of the row is never read and
never written.

These rules keep each kernel correct:

- Every barrier sits outside each condition on the thread. A loop around a
  barrier takes its bound from G only, so every thread reaches every barrier.
  A test reads the kernel text and checks this rule.
- An idle slot holds the most negative finite float32, never an infinity. No
  subtraction can thus make a NaN.
- The max subtraction keeps each `exp` argument at 0 or below, so no `exp`
  overflows.
- The kernels call `precise::exp` and `precise::divide`, so the compile
  options do not change the math functions.

The digests of the step texts are:

| Step | Kernel | Text digest |
| --- | --- | --- |
| U0 | `softmax_u0_unfused` | `c38c9faf73e14e5695bd758ad3f4d3547b17d571877f091d3264d729bb71997c` |
| F1 | `softmax_f1_threadgroup` | `4a41f2e138c5a3e499293b771f2e105f4c13920ba291b466b75cab657ebe665a` |
| F2 | `softmax_f2_simd` | `a8380a8b2ea58e8a84ed8c8a1025c7cb5ea7f4f1111d1abde374135e0c0af4d8` |
| F3 | `softmax_f3_online` | `169845113557a1bf67e8fff9320b28ca920547ecc54f175b435039b7fb7124c5` |

The library digest is
`c8c1b7ff06c16966a8a0b9e646868e62055faa6ca247e4c0864595a7e9cd7521`. The plan
digest is
`198870239a4a52adb106ed2396e7f9a99d1c2c52ba9770e9fea820e28b032990`. The
pinned config names this digest, so a config for other hypotheses cannot run.

## The plan

| Step | Base | Change | Predicted verdict over the base |
| --- | --- | --- | --- |
| U0 | `torch.softmax` | three kernels, the max and the sum in device memory | slower |
| F1 | U0 | one kernel, the row in registers, two threadgroup trees | faster |
| F2 | F1 | `simd_max` and `simd_sum`, one threadgroup step for each value | faster or no clear difference |
| F3 | F2 | one online pass for the max and the sum, one merge step | no clear difference or slower |

The hypotheses give these reasons:

- U0 reads each row three times and writes it once. It also launches three
  kernels for each call.
- F1 moves about half the modeled bytes of U0, with one launch in place of
  three. F1 adds the same values in the same order as U0, so its outputs can
  keep the U0 bits.
- A tree over G threads needs `log2(G) + 1` barriers for each value, and F2
  needs one. The loads of the row probably set most of the time.
- F3 moves the bytes of F2 with one barrier in place of two, but it calls
  `precise::exp` twice for each element.

## Correctness

Each case draws seeded rows in a gapped layout. Thirty-two guard slots follow
each row in the input and in the output. Each input gap holds `2**100`, so a
kernel that reads a gap gets a wrong max. Before the run, each output gap
holds a quiet NaN with one known bit pattern, so a store into a gap shows.

The cases of the pinned config cover these rows and inputs:

- Rows of 1, 31, 33, 1000 and 4096 elements, and one ragged batch of 1, 31,
  33, 1000, 4096, 16384, 2049 and 7 elements.
- Standard normal inputs, and uniform inputs from -80 to 80.
- Inputs of 1000 or -1000 plus a standard normal. With no max subtraction,
  `exp` overflows or each term is 0, so the float32 softmax gives NaN.

The reference is the float64 softmax of the float32 inputs, with the max
subtraction. Its exp is `exp64`, which uses only IEEE additions, products and
`ldexp`. The libm of a host can round a float64 `exp` to the other neighbor,
and one changed bit changes a digest. With `exp64`, each IEEE 754 host
computes the same float64 bits, within 1 ulp of the exact `exp`.

Each element of each step has a declared bound for the order of that step.
The bound counts the roundings of each term of the sum, the rounding of
`x - max` and the error of each `precise::exp` call. A result below `2**-126`
can flush to zero, so each bound adds `2**-125`. The rows must sum to 1
within the sum of the element bounds of the row.

The bound assumes that `precise::exp` has an error of at most 4 units of
`2**-23`. The run measures this assumption first with the exp probe: 262144
arguments from -104 to 0 and eight extremes. A failed probe stops the run.

Each step runs twice on each case. Four rules apply to the output of each
run, and a case that breaks one stops the run:

1. Each gap slot keeps its sentinel bits.
2. Each output is finite.
3. Each element is within its bound of the reference.
4. Each row sum is within its bound of 1.

A race can spoil one run only, so the rules check the second run too. Two
facts are measurements, not rules. The run records if the two outputs of a
step have the same bits, and it records if F1 kept the bits of U0.

## Timing

The timing is the Day 74 and Day 76 method:

1. Each side runs 5 untimed warmup samples.
2. Each timed sample calls `torch.mps.synchronize()`, reads
   `time.perf_counter_ns`, launches a batch of calls, synchronizes again and
   reads the clock again. One call of U0 is its three launches.
3. Repeat `r` runs side `r mod 5` first and then each other side once, in
   order. The 50 repeats give each side each position 10 times.

A batch holds the fewest whole calls whose compulsory bytes reach 256 MiB.
Every side of a point runs the same batch. A ratio of two median rates is thus
the inverse ratio of the two median sample times.

After the last sample, the latest output of each step must be within its
bound of the reference. The output of `torch.softmax` must be within the
sanity bound `2**-16 * y_ref + 2**-125`, or the run stops. PyTorch names no
kernel, so no derived bound applies to it.

A ragged point draws 2048 lengths from 1 to 4096 from the seed. In the pinned
run, the longest row has 4089 elements. `torch.softmax` has no ragged input
on MPS, so it gets the padded 2048 x 4089 matrix. Each slot past a row holds
the most negative finite float32. The effective GB/s of `torch.softmax`
counts the 4128797 elements of the rows, not the 8374272 elements that it
gets.

## The speedup rule

Each ratio divides two median sample times of the same point of the same run.
The rule of Day 76 names a verdict with the declared `min_faster_ratio` of
1.1:

- `faster`: the ratio is 1.1 or more, and the p75 sample time of the first
  side is below the p25 time of the second side.
- `slower`: the inverse ratio is 1.1 or more, and the p75 sample time of the
  second side is below the p25 time of the first side.
- `no_clear_difference`: any other result.

A prediction holds at a point when the verdict over the base is one of the
predicted verdicts.

## One run on the development host

This run used the pinned config on 2026-11-22 at 11:36 IST. The host was a
shared Darwin arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14, NumPy
1.26.4 and PyTorch 2.15.0. The environment did not set
`PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. The run held the GPU
lock that the timed GPU jobs on this host take. Other jobs used the CPU.

The report recorded load averages of 8.00, 8.05 and 8.17 before the run. It
recorded 7.72, 7.99 and 8.14 after the run. The `uptime` command printed the
same values before and after the run.

The report digest is
`352e9d14fb8be6ef942191b5618660be42f57e0b1b8818dabb3c85dfd61bfafc`. The run
took 7.64 seconds after the library compiled, and the command took about 10
seconds with the torch import. The report has 93765 bytes, and the run caught
0 Python warnings. The capture began before the import of torch.

Each pipeline allowed 1024 threads and reported a SIMD width of 32. The
static threadgroup memory of each pipeline was equal to the declared bytes,
so the run refused no step.

### Correctness results

- All 32 outputs (8 cases, 4 steps) met the four rules in both runs. The
  largest error was 0.706 of its bound, for U0 and F1 in the ragged case.
- The largest relative error of the exp probe was 1.354 units of `2**-23`,
  or 1.406 units in the last place. The bound assumes 4 units. All 42008
  results below `2**-126` were 0.
- Each step gave the same bits in both runs of each case.
- F1 kept the bits of U0 in all 8 cases, as its step declared.
- For the two offset cases, the float32 softmax with no max subtraction gave
  NaN. Each step gave finite outputs within its bound.

### The table

Each rate is the median effective GB/s of 50 samples: the compulsory bytes of
a sample over its time. The range in parentheses comes from the IQR of the
sample times. "Modeled GB/s" divides the modeled bytes of the step by the
same median time. "Of the ceiling" divides the modeled GB/s by the ceiling of
82.28 GB/s. "Over its base", "Over U0" and "Over torch" give the ratio and
the result of the rule.

#### 4096 x 128 (G = 32, 64 calls in each sample)

| Step | Median GB/s (IQR range) | Modeled GB/s | Of the ceiling | Over its base | Over U0 | Over torch | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | 60.3 (58.7 to 61.9) | 123.2 | 1.497 | 0.330, slower | | 0.330, slower | held |
| F1 | 60.5 (59.0 to 61.9) | 61.0 | 0.742 | 1.004, no clear difference | 1.004, no clear difference | 0.332, slower | missed |
| F2 | 125.7 (121.8 to 130.7) | 126.7 | 1.540 | 2.077, faster | 2.085, faster | 0.689, slower | held |
| F3 | 106.2 (100.3 to 110.2) | 107.0 | 1.301 | 0.845, slower | 1.761, faster | 0.582, slower | held |
| torch | 182.5 (173.5 to 190.0) | | | | 3.027, faster | | |

#### 4096 x 1024 (G = 64, 8 calls in each sample)

| Step | Median GB/s (IQR range) | Modeled GB/s | Of the ceiling | Over its base | Over U0 | Over torch | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | 54.8 (52.4 to 56.6) | 109.9 | 1.336 | 0.674, slower | | 0.674, slower | held |
| F1 | 106.2 (101.1 to 111.8) | 106.3 | 1.292 | 1.937, faster | 1.937, faster | 1.305, faster | held |
| F2 | 104.5 (98.4 to 110.3) | 104.6 | 1.271 | 0.984, no clear difference | 1.906, faster | 1.284, faster | held |
| F3 | 104.9 (97.7 to 110.4) | 105.0 | 1.276 | 1.004, no clear difference | 1.914, faster | 1.289, faster | held |
| torch | 81.4 (78.6 to 85.5) | | | | 1.485, faster | | |

#### 1024 x 4096 (G = 256, 8 calls in each sample)

| Step | Median GB/s (IQR range) | Modeled GB/s | Of the ceiling | Over its base | Over U0 | Over torch | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | 55.9 (52.9 to 57.1) | 111.8 | 1.359 | 0.566, slower | | 0.566, slower | held |
| F1 | 106.3 (99.8 to 109.9) | 106.3 | 1.292 | 1.903, faster | 1.903, faster | 1.077, no clear difference | held |
| F2 | 104.1 (98.2 to 110.1) | 104.1 | 1.265 | 0.979, no clear difference | 1.863, faster | 1.055, no clear difference | held |
| F3 | 107.9 (102.4 to 111.2) | 108.0 | 1.312 | 1.037, no clear difference | 1.932, faster | 1.094, no clear difference | held |
| torch | 98.7 (94.8 to 103.7) | | | | 1.766, faster | | |

#### 256 x 16384 (G = 1024, 8 calls in each sample)

| Step | Median GB/s (IQR range) | Modeled GB/s | Of the ceiling | Over its base | Over U0 | Over torch | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | 57.2 (54.2 to 57.8) | 114.4 | 1.391 | 0.853, slower | | 0.853, slower | held |
| F1 | 93.4 (88.8 to 97.1) | 93.4 | 1.135 | 1.632, faster | 1.632, faster | 1.392, faster | held |
| F2 | 100.8 (96.2 to 104.7) | 100.8 | 1.225 | 1.080, no clear difference | 1.762, faster | 1.503, faster | held |
| F3 | 101.3 (98.8 to 104.3) | 101.3 | 1.231 | 1.005, no clear difference | 1.770, faster | 1.510, faster | held |
| torch | 67.1 (65.3 to 69.1) | | | | 1.172, faster | | |

#### Ragged: 2048 rows of 1 to 4096 elements (G = 256, 9 calls in each sample)

| Step | Median GB/s (IQR range) | Modeled GB/s | Of the ceiling | Over its base | Over U0 | Over torch | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | 54.5 (51.5 to 55.9) | 109.2 | 1.327 | 1.056, no clear difference | | 1.056, no clear difference | missed |
| F1 | 101.9 (99.0 to 106.0) | 101.9 | 1.239 | 1.869, faster | 1.869, faster | 1.974, faster | held |
| F2 | 105.4 (98.1 to 110.5) | 105.4 | 1.281 | 1.034, no clear difference | 1.933, faster | 2.042, faster | held |
| F3 | 104.6 (97.1 to 109.1) | 104.6 | 1.272 | 0.992, no clear difference | 1.919, faster | 2.026, faster | held |
| torch | 51.6 (50.4 to 52.6) | | | | 0.947, no clear difference | | |

The predictions held at 18 of 20 points: 4 steps at 5 points each.

### What helped

- Fusion (F1 over U0): at the four points of 4.1 to 4.2 million elements, F1
  was 1.632 to 1.937 times as fast as U0. The model gives F1 about half the
  bytes of U0, so the measured ratio is near the modeled one.
- SIMD reductions (F2 over F1) at 4096 x 128: F2 was 2.077 times as fast as
  F1. At this point, G is 32, so each group is one SIMD group, and each
  thread holds only 4 elements. The trees of F1 then probably cost more time
  than the loads.

At the four large points, all four steps ran at 93.4 to 114.4 modeled GB/s.
U0 took 1.632 to 1.937 times the time of a fused step for about twice the
modeled bytes. This result agrees with steps whose time the device bytes set,
but no counter shows it.

### What did not help

- Fusion at 4096 x 128: F1 showed no clear difference from U0, with a ratio of
  1.004. The hypothesis missed there. The 2 MiB input probably stays in a
  cache between the three passes of U0, and the trees of F1 cost time. These
  are inferences, not observations.
- SIMD reductions at the large points: F2 showed no clear difference from F1,
  with ratios of 0.979 to 1.080. The hypothesis allowed this result.
- The online pass (F3 over F2): at 4096 x 128, F3 was slower than F2, with a
  ratio of 0.845. At the other four points, the rule found no clear
  difference, with ratios of 0.992 to 1.037. The hypothesis allowed both
  results. The barrier that F3 saves did not show in the times. The second
  `precise::exp` call of F3 can be the cost at 4096 x 128, but no counter
  shows it.

### The vendor side

- At 4096 x 128, `torch.softmax` was the fastest side, with 182.5 GB/s. That
  is 2.218 times the recorded ceiling and 1.45 times the rate of F2. The 2 MiB
  input and the 2 MiB output probably stay in a cache over the 64 calls of a
  sample. This rate thus probably measures a cache, not device memory.
- At 4096 x 1024 and 256 x 16384, the fused steps were 1.284 to 1.510 times as
  fast as `torch.softmax`.
- At 1024 x 4096, the rule found no clear difference between the fused steps
  and `torch.softmax`.
- At the ragged point, the fused steps were 1.974 to 2.042 times as fast as
  `torch.softmax`. But `torch.softmax` processed the padded matrix, about
  twice the elements of the rows. U0 showed no clear difference from it, and
  the U0 hypothesis missed there.

### The bandwidth ceiling and rates above it

The ceiling is 82.28 GB/s, from `stream_read_reduce` at a 4 MiB buffer. It
comes from the recorded Day 74 report of 2026-11-17, a separate run at load
averages near 9.5. The fused steps reached 1.135 to 1.312 of this ceiling at
the four large points.

A fraction above 1 does not mean that a kernel moved more data than the
memory can. The Day 74 guide reports 98.9 to 100.8 GB/s for its pinned runs
at 256 MiB, and those reports are not test fixtures. The recorded ceiling is
thus below what device memory gave in this run. It is the best median of the
Day 74 kernels at one small buffer, not a hardware limit.

### Limits of this run

- One run on one shared host gives each ratio. The load averages were near 8
  from other jobs.
- The steps form one chain, so each ratio measures one change on top of the
  changes before it.
- The rule calls a ratio clear only when the IQR ranges do not overlap. Each
  range was 4.4 to 12.1 percent of its median, so a small change stays
  hidden.
- The byte counts are a model with no cache. The lab cannot see the cache
  hits of a second read.
- The ceiling comes from a separate run at another time and load.

## Triton in the interpreter

Triton has no macOS wheel and no Metal backend, so the lab cannot run Triton
on this Apple GPU. The optional test `tests/test_metal_fused_triton.py` holds
Triton text of three patterns. They are the unfused U0, the fused one-block
row of F1 and F2, and the online update of F3. With `TRITON_INTERPRET=1`,
Triton runs each kernel on the CPU with NumPy.

The test uses the cases of the pinned config, their gapped layouts and the
float64 reference. NumPy sets the order of each `tl.sum`, so the test uses the
sanity bound `2**-16 * y_ref + 2**-125` and not the per-order bounds. Without
the flag or without `triton`, the module skips.

On 2026-11-22, the 24 checks passed in a `python:3.11-slim` container with
Triton 3.9.0, CPU PyTorch 2.15.0 and NumPy 1.26.4. The container mounted the
worktree read-only. These are CPU correctness checks only. They measure no
time, and no Triton kernel ran on a GPU.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a GPU ran
the timed work.

The verifier rebuilds each fact that follows from the seed and the digests,
with its own code:

1. It draws the rows of each case and point again, and it builds the gapped
   and ragged layouts again.
2. It computes the float64 reference and the bound of each element and row
   sum for each step again.
3. It derives each pipeline verdict again from the recorded limits.
4. It derives each statistic, rate, modeled byte count, ratio, verdict,
   fraction of the ceiling and held hypothesis from the raw samples.
5. It accepts an output that claims the bits of the rounded reference only
   when the output hash and errors match that reference.

Only a report from `torch.mps.compile_shader` and `time.perf_counter_ns`
passes as a measurement. It also needs a measured ceiling from the same
interpreter, libraries and platform. The test
`tests/test_metal_fused_recorded.py` holds one short measured report from the
development host, so a Linux host without torch or MPS can check a real
report. The verifier computes each float64 value with `exp64`, not with the
libm of the host. The digests of that report thus do not depend on the libm.
A test verifies the report with a changed `numpy.exp`. No Linux host ran the
recorded test for this guide.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 79 | 0 to 4294967295 |
| `correctness.cases` | 8 cases | 1 to 16 cases that cover each kernel path |
| `correctness.repeats` | 2 | 2 to 4 |
| `benchmark.inputs` | `normal` | a named distribution |
| `benchmark.shapes` | 5 shapes | 1 to 8 shapes, up to 16384 elements in a row |
| `benchmark.warmup` | 5 | 0 to 10 |
| `benchmark.repeats` | 50 | 3 to 50, a multiple of 5 |
| `benchmark.target_bytes_per_sample` | 268435456 | 1 KiB to 1 GiB, at most 4096 calls in a sample |
| `benchmark.max_runtime_seconds` | 300 | 1 to 600 |
| `comparison.min_faster_ratio` | 1.1 | above 1 and at most 4 |

## Evidence boundary and exclusions

The evidence boundary of a measured report holds true or false fields and one
statement. The fields `apple_gpu_metal_this_host` and `host_specific` are
true. These fields are false: `cuda_kernel_executed`, `nvidia_gpu_used`,
`triton_gpu_kernel_executed`, `vendor_kernel_known`, `vendor_peak_claimed`,
`hardware_counters_used`, `occupancy_measured`, `distributed_claim`,
`production_claim` and `portable_performance_claim`.

The lab does not record the GPU model, the clock frequency, the thermal state
or the other jobs on the host. Each statement about a cache in this guide is
an inference from timings, not from hardware counters. The results come from
one Apple GPU on one host. They are not evidence of CUDA, NVIDIA or Triton
GPU performance, lecture viewing, completed exercises or learner
understanding. Ajay must still watch Lecture 18 and compare its fused kernels
with these four steps.

## Sources

- GPU MODE lectures, https://github.com/gpu-mode/lectures: Lecture 18, "Fused
  Kernels".
- M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax",
  arXiv:1805.02867, 2018, for the online update.
- Apple, Metal Shading Language Specification, for `simd_max`, `simd_sum`,
  `precise::exp`, `precise::divide` and `threadgroup_barrier`:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- PyTorch 2.15.0, `torch.mps.compile_shader` and `torch.softmax`.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 3.1, for the bound `gamma_n`.
- Triton 3.9.0 and its interpreter mode, `TRITON_INTERPRET=1`.
