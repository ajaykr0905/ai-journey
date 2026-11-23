# Sum reductions on the Apple GPU: from R0 to R5

Day 78 of the curriculum follows GPU MODE Lecture 9, "Reductions", and Lecture
20, "Scan Algorithm". The tracker notes name parallel reduction, warp shuffles
and divergence. The exercise has three parts:

1. Write a sum reduction kernel.
2. Make it faster with warp shuffles.
3. Benchmark both kernels.

This lab does the exercise for a float32 sum on a Mac with an Apple GPU. It
declares six steps, R0 to R5. Each step makes one change to the step before
it, and each step states a hypothesis before the first timed run. One run
checks every kernel against the exact sum and a derived error bound. The same
run times every kernel and `torch.sum` on one device buffer.

Metal has no warps. A SIMD group of 32 lanes is the Metal counterpart of a
CUDA warp. SIMD-group functions are Metal's counterpart of CUDA warp shuffles,
not the same instruction. Nothing here runs a CUDA kernel or uses an NVIDIA
GPU. The lab reads no hardware counters. Every timing is a host-specific
Apple GPU (Metal) measurement at the load averages that the report records.

This lab has no scan kernel. Ajay must still watch Lecture 9 and Lecture 20.

## Commands

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
python -m ai_journey.gpu_reduce_cli run \
  --config config/day-78-gpu-reduce.json \
  --ceilings /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-78-gpu-reduce.json
python -m ai_journey.gpu_reduce_cli verify \
  --report /tmp/day-78-gpu-reduce.json \
  --config config/day-78-gpu-reduce.json \
  --ceilings /tmp/day-74-gpu-ceilings.json
```

The first command makes a fresh bandwidth ceiling with the Day 74 lab. The
parser uses the program name `ai-journey-gpu-reduce`. An installed wheel gives
the same commands under that name.

The `run` command reads the ceiling report, compiles the kernels, checks
them, times them, checks its own report and writes the report. Then it prints
these lines:

1. One line for each plan step, with the base kernel and the predicted
   verdicts of each size class.
2. One line with the load averages before and after the run.
3. One line with the bandwidth ceiling and the digest of its report.
4. One line for each correctness case, with the bound ratio of each kernel.
   The line also tells if the repeated runs of each kernel kept their bits.
5. One line for each timing size, with the median GB/s and the IQR range of
   each side.
6. For each timing size, one line for each step, with the two step ratios,
   the two fractions and the result of the hypothesis. Then one line with the
   fastest kernel at that size.
7. One line for the step that keeps the order of an earlier step.
8. One line with the count of Python warnings that the run caught.
9. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`gpu_reduce_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `gpu_reduce_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the same summary
line with the prefix `verified_`. With `--config`, `verify` also requires that
the report comes from that exact config. With `--ceilings`, it requires that
the report used that exact gpu-ceilings report.

The exit codes are the codes of the Day 73 to 77 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

On the development host, one run of the pinned config took 22 seconds, with
the torch import. The run wrote a report of about 124 KB.

## Hosts without MPS

The `run` command needs an Apple GPU on macOS. It also needs a PyTorch build
with an MPS device and `torch.mps.compile_shader`. The runtime uses the probe
of the Day 73 Metal lab. The probe reads the operating system first, and it
imports torch only on macOS. It names the first requirement that is missing,
for example `unsupported_operating_system` on a Linux host.

When a requirement is missing, `run` writes the reason to standard error and
exits with code 3. It prints nothing to standard output, and it does not make
the output directory or the output file. The `verify` command and the report
reader never import torch, so they work on any host with NumPy.

The CI job `nvidia-preflight-contracts` runs the tests of this lab on Linux
with CPU-only torch. Its globs `src/ai_journey/gpu*.py` and
`tests/test_gpu*.py` include these files. The tests that need the GPU skip on
such a host and report the reason. The other tests use a NumPy stand-in for
the GPU and one short measured report from the development host.

## The plan

The step list follows the curriculum notes. Ajay must still check it against
Lecture 9. The module `gpu_reduce_plan.py` declares the steps. Every hypothesis
entered the repository before the first timed run of a Day 78 kernel. The
pinned config and each report seal the digest of the plan,
`b2134229a87d220e40772e732ca3d3e5d6c6a6ae82a90a475eb0405d87a7217f`. A config
for other hypotheses or other kernels cannot run.

The steps form one chain:

- The base of R0 is `torch.sum`, the vendor side.
- The base of each later step is the step before it. Thus its ratio to its
  base is the speedup of its own change.
- Its ratio to R0 is the speedup of every change so far.
- R3 declares that it adds the same pairs in the same order as R2. The run
  records if each R3 result kept the bits of R2.

A prediction gives one or two verdicts of the Day 76 ratio rule for each size
class. The plan holds the bounds of the classes. The class names state a
hypothesis about each range, not a measured fact.

| Class | First n | Last n | Hypothesis of the plan |
| --- | --- | --- | --- |
| small | 1 | 262143 | the host dispatch of a call sets the time |
| medium | 262144 | 4194303 | a cache can hold the input between calls |
| large | 4194304 | 67108864 | the device loads set the time |

| Step | Kernel | Base | small | medium | large |
| --- | --- | --- | --- | --- | --- |
| R0 | `reduce_r0_interleaved` | `torch.sum` | faster, no clear difference | slower | slower |
| R1 | `reduce_r1_sequential` | R0 | no clear difference, faster | faster | faster |
| R2 | `reduce_r2_first_add` | R1 | no clear difference | faster | faster |
| R3 | `reduce_r3_simd_shuffle` | R2 | no clear difference | no clear difference, faster | no clear difference |
| R4 | `reduce_r4_simd_sum` | R3 | no clear difference | no clear difference, faster | no clear difference |
| R5 | `reduce_r5_grid_stride` | R4 | no clear difference, slower | no clear difference | no clear difference |

The plan states the change and the hypothesis of each step in these words.

R0, the interleaved tree:

- "Load one element per thread. Then add pairs in threadgroup memory with
  interleaved addressing. At stride s, only the threads with t % (2 s) == 0
  add, and s doubles from 1 to 128."
- "R0 is slower than torch.sum from 2^18 elements. At each level, the modulo
  test keeps most lanes of every SIMD group idle. Every SIMD group still takes
  the branch, and a group waits at 9 barriers for each 256 elements. Below
  2^18 elements, the host dispatch of a call probably sets the time. R0
  dispatches two short kernels, so it is faster than torch.sum or shows no
  clear difference."

R1, sequential addressing:

- "Add pairs with sequential addressing: at stride s, the threads t < s add
  shared[t + s], and s halves from 128 to 1."
- "R1 is faster than R0 by 1.1 or more from 2^18 elements. While s is 32 or
  more, a whole SIMD group takes the branch or skips it. Thus only the last 5
  levels diverge, and no thread calculates a modulo. Below 2^18 elements, the
  dispatch probably sets the time, so R1 shows no clear difference from R0 or
  is faster."

R2, the first add during the load:

- "Give each thread two elements 256 apart, and add them during the global
  load. One group then covers 512 elements, and the grid has half the groups."
- "R2 is faster than R1 by 1.1 or more from 2^18 elements. Each thread has two
  loads in flight before its first barrier, so more loads are in flight. Also,
  the tree work for each element halves. Below 2^18 elements, R2 shows no clear
  difference from R1."

R3, SIMD-group shuffles for the last 32 values:

- "Stop the threadgroup tree at 32 values. Then add them in SIMD group 0 with
  simd_shuffle_down at deltas 16, 8, 4, 2 and 1, with no barrier."
- "R3 removes 5 of the 9 barriers of a group. From 2^22 elements, the device
  loads probably set the time of R2, so R3 shows no clear difference from R2.
  From 2^18 to 2^22 elements, a cache can serve the input, so the barriers can
  matter, and R3 is faster or shows no clear difference. Below 2^18 elements,
  R3 shows no clear difference. R3 adds the same pairs in the same order as R2,
  so each result keeps the bits of R2."

R4, `simd_sum` in each SIMD group:

- "Add the two loaded values, and add each SIMD group with simd_sum. Write the
  8 group sums to threadgroup memory, and wait at one barrier. Then SIMD group
  0 adds the 8 sums with simd_sum."
- "R4 waits at 1 barrier for each group instead of 4. From 2^22 elements, the
  loads probably set the time, so R4 shows no clear difference from R3. From
  2^18 to 2^22 elements, R4 is faster or shows no clear difference. Below 2^18
  elements, R4 shows no clear difference. simd_sum does not state its order of
  additions. Thus the bound of R4 allows any order in a SIMD group, and its
  bits can differ from R3."

R5, a fixed grid with a grid-stride loop:

- "Launch a fixed grid of 1024 groups. Each thread adds the elements gid, gid +
  262144, gid + 2 * 262144 and so on in one loop. Then it reduces as R4 does.
  The combine pass thus always adds 1024 partials."
- "R5 shows no clear difference from R4 from 2^18 elements. Both read each
  element once, and the combine pass of R4 reads at most 131072 partials, 1/512
  of the input. Below 2^18 elements, most R5 threads have no element, so R5 is
  slower or shows no clear difference. Each R5 thread adds up to n / 262144
  elements in one chain, so the bound of R5 grows with n."

## Kernels

The source is in the Metal Shading Language, in `gpu_reduce_source.py`, and
`torch.mps.compile_shader` compiles it at run time. Each reduction has two
passes:

1. Pass 1 is the kernel of one step. Each group of 256 threads writes one
   partial sum.
2. Pass 2 is `reduce_combine` for every step. One group of 1024 threads adds
   the partials in a fixed order and writes the sum.

No kernel uses an atomic. Thus the launch fixes the order of every addition,
and a CPU bound can follow that order. Each kernel obeys these rules:

- Every barrier is outside each condition on the thread. Each loop bound
  depends only on the group size. Thus every thread of a group reaches every
  barrier.
- A load at n or past n gives `0.0f`, and an addition of 0 is exact.
- A SIMD-group function runs only where all 32 lanes of its SIMD group are
  active.
- The tree loops of R0 to R3 take their bound from `threads_per_threadgroup`,
  as the CUDA kernels of Harris take it from `blockDim.x`. Thus the compiler
  does not see the bound as a constant. The runtime launches each kernel only
  with its declared group size.

| Kernel | Elements for each thread | Group tree | Group | Threadgroup bytes | Partials at n = 2^26 |
| --- | --- | --- | --- | --- | --- |
| `reduce_r0_interleaved` | 1 | interleaved, 8 levels | 256 | 1024 | 262144 |
| `reduce_r1_sequential` | 1 | sequential, 8 levels | 256 | 1024 | 262144 |
| `reduce_r2_first_add` | 2, 256 apart | sequential, 8 levels | 256 | 1024 | 131072 |
| `reduce_r3_simd_shuffle` | 2, 256 apart | 3 levels, then 5 shuffles | 256 | 1024 | 131072 |
| `reduce_r4_simd_sum` | 2, 256 apart | `simd_sum`, then `simd_sum` of 8 | 256 | 32 | 131072 |
| `reduce_r5_grid_stride` | n / 262144, in a loop | `simd_sum`, then `simd_sum` of 8 | 256 | 32 | 1024 |
| `reduce_combine` | partials / 1024 | sequential, 10 levels | 1024 | 4096 | none |

Each digest covers the text of one kernel. The library is the two-line header
and the seven kernels, in this order.

| Source | SHA-256 |
| --- | --- |
| `reduce_r0_interleaved` | `40303012a3f4b3bcdd1909023d2ee8af860d79dbbde9cbf0714af7ece8f8f6c2` |
| `reduce_r1_sequential` | `b230db3f18d0db381928728e77c376adc34532c49d421d4cd36c4adc6e244173` |
| `reduce_r2_first_add` | `d52e657352c87e5cd8ae8baf6ce45cc92c08cb5ed2f887d7f7f58f1c48f414a1` |
| `reduce_r3_simd_shuffle` | `9ddceaf22450689e0761dbfc5a81ffca455866fc2772e14eff229bebb875f31a` |
| `reduce_r4_simd_sum` | `8ee154263d19714e61dd6849d78ff74afdbb5e702194248f5c7c65b69dd53db0` |
| `reduce_r5_grid_stride` | `fb97a05461e22dafc3a0c345051a73bcba9088d9e9e14dcefb29114917713460` |
| `reduce_combine` | `551bea995a2f88e2d91426e40862f8bf65e58c2de9e76a1a7d537b006233d770` |
| reduction library | `753b102d673402b6ce260b644639d1f09a8c81f16647e831b2e35c8ac54f7edc` |

### Divergence, and why R1 removes it

A SIMD group runs its 32 lanes together. Sometimes the lanes of one SIMD
group take different sides of a branch. Then the group runs each side, and
the lanes of the other side wait. This is control divergence.

In R0, at stride s, only the threads with `t % (2 s) == 0` add. At s = 1, half
the lanes of every SIMD group add, and the other half wait. At s = 2, one lane
in four adds. Until s reaches 32, every one of the 8 SIMD groups takes the
branch with half or more of its lanes idle. Each thread also calculates
`t % (2 s)`, and the compiler does not see s as a constant.

In R1, at stride s, the threads `t < s` add. While s is 32 or more, s is a
multiple of 32. Thus the first s / 32 SIMD groups take the branch with all
their lanes, and the other SIMD groups skip it. Only the last 5 levels, from
s = 16 to s = 1, diverge, and only in SIMD group 0. No thread calculates a
modulo.

### Why SIMD-group shuffles remove barriers

A threadgroup barrier stops every thread of the group until each thread
arrives. It also makes the writes to threadgroup memory visible. A tree in
threadgroup memory thus needs one barrier at each level.

In R2, the last 5 levels use the lanes of SIMD group 0 only. Every thread of
the group still waits at those 5 barriers. The lanes of one SIMD group run
together, so one lane can read the value of another lane with
`simd_shuffle_down`. Such a read needs no threadgroup memory and no barrier.

- R3 stops the tree at 32 values. SIMD group 0 adds them with 5 shuffles, at
  deltas 16, 8, 4, 2 and 1. A group waits at 4 barriers instead of 9.
- R4 adds the 32 lanes of each SIMD group with one `simd_sum`. Then it writes
  8 sums to threadgroup memory, waits at 1 barrier, and adds the 8 sums with
  `simd_sum` in SIMD group 0.

The shuffles of R3 add the same pairs as the last 5 levels of the R2 tree, in
the same order. The Metal Shading Language Specification does not state the
order of the additions in `simd_sum`. Thus the bound of R4 and R5 allows any
order inside a SIMD group.

### R5 and the fixed grid

R5 launches 1024 groups of 256 threads for every n. Thread `gid` adds the
elements `gid`, `gid + 262144`, `gid + 2 * 262144` and so on. The same grid
thus covers any n, and the combine pass always adds 1024 partials. At small n,
most threads have no element. At large n, each thread adds a long chain, and
that chain makes the bound of R5 larger.

## Correctness

The correctness cases are 13 seeded inputs with seed 78. They hold five
patterns:

- `normal`: standard-normal values
- `positive`: uniform values in [0, 1)
- `mixed`: a sign times a value in [1, 2) times 2^e, for e from -24 to 24
- `ones`: every value is 1
- `halves`: k / 2, for k from -8 to 8

The sizes include 1, 31, 33, 257, 511, 513, 300007, 1000003 and 2000003. The
config parser requires these properties of the cases:

- Each pattern has a case.
- Some n is less than 32, so one SIMD group holds the whole input.
- Some n ends its last 512-element group in the second half of the group.
- Some n gives each kernel except R5 more than 3072 partials. Thus thread 0
  of the combine runs its loop of four accumulators.

### The exact sum

Each float32 value is an integer times 2^-149. Thus the exact sum is an
integer times 2^-149 too. The run adds the integer significands of each binary
exponent, and Python integers join the exponents without a rounding. The
float64 reference is that exact sum, rounded once.

### The bound of each order

Any order of additions is a binary tree. If a value passes through at most D
roundings on its path to the root, the error of the sum is at most
`gamma_D(u) * sum |x|`. Here `u = 2^-24` and `gamma_D = D u / (1 - D u)`. This
bound is from section 4.2 of Higham. The lab counts D from four parts:

1. The chain of one thread: 0 for one element, 1 for two elements, and
   `ceil(n / 262144) - 1` for the R5 loop. Any order of c values in one thread
   has a depth of at most c - 1. Thus the bound holds if the compiler changes
   the order of a loop.
2. The tree of one group: 8 for the threadgroup trees of R0 to R3. A
   `simd_sum` does not state its order, so it counts as the worst tree. That
   is 31 for 32 lanes and 7 for the 8 group sums, because an addition of an
   exact zero rounds nothing.
3. The chain of the combine: `ceil(G / 1024) - 1` for G partials.
4. The tree of the combine: 10 levels for 1024 threads.

| n | R0 | R1 | R2 | R3 | R4 | R5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 18 | 18 | 19 | 19 | 49 | 48 |
| 1000003 | 21 | 21 | 20 | 20 | 50 | 51 |
| 2^26 | 273 | 273 | 146 | 146 | 176 | 303 |

The model needs results that are normal numbers. Each input value is a
multiple of 2^g for the smallest such g. Every partial sum is a multiple of
2^g too. The lab refuses an input with g below -126, so no nonzero partial sum
is subnormal, and a flush to zero changes no result.

When `sum |x|` is at most `2^24 * 2^g`, every partial sum of every order is
exact in float32. Then the result must equal the exact sum. The cases with n =
1, all ones and the halves are such exact cases.

PyTorch does not state the order of `torch.sum`. Its timed results get a
declared tolerance, `gamma_4096(u) * sum |x|`. This tolerance is not a derived
bound.

### Guards and repeated runs

Each case runs each kernel 3 times on fresh buffers. A case stops the run in
these conditions:

- A partial slot of the launch is not finite, so a group did not write it.
- The tail of 64 floats past the partials or the tail of 31 floats past the
  output lost a quiet-NaN sentinel.
- The result is not finite. The input has a tail of 1024 quiet-NaN sentinels,
  so a read past n gives NaN.
- The result of one of the runs is outside its bound, or it is not exact in
  an exact case. The run checks each of the 3 results, not only the first.

The record states if every run of a kernel gave the same bits in both whole
buffers. A different result inside the bound does not stop the run. It is a
measured fact.

## Timing

The run draws one seeded standard-normal input of 2^26 values and puts it on
the device once. At each size n, every side reads the first n values of that
buffer. The sides are the six kernels and `torch.sum(x[:n], 0, out=out)` with
a preallocated output. The timing uses the Day 74 and Day 76 method:

1. Each side runs 2 samples that are not timed.
2. Each timed sample calls `torch.mps.synchronize()` and reads
   `time.perf_counter_ns`.
3. The sample then launches a batch of calls, calls `torch.mps.synchronize()`
   again, and reads the clock again.
4. The first side of repeat r is side r mod 7, so the order moves by one side
   at each repeat. Each side has 14 timed samples, so each side takes each
   place twice.

A batch holds the fewest whole calls that read 2^30 bytes. At the pinned sizes,
a sample holds 4096, 256, 16, 4 and 27 calls. A kernel call is two launches:
pass 1 and the combine. A rate is decimal GB/s: 4 n bytes for each call, times
the calls of a sample, divided by the median sample time in nanoseconds. The
rate does not count the partials of pass 2. The IQR of the sample times gives
the range of each rate.

After the last sample at a size, the run checks the latest result of each
side against the exact sum of the prefix. A kernel must stay within the bound
of its order. `torch.sum` must stay within the declared tolerance. Each side
then runs once more, and the result of that repeat must pass the same check.
The record states if the repeat kept the bits. Other bits inside the bound do
not stop the run.

## The bandwidth ceiling

The fraction of the ceiling compares each rate with what this host's memory
gave the Day 74 kernels. The ceiling comes from a gpu-ceilings report that
the Day 74 verifier accepts. It is the better median GB/s of `stream_copy` and
`stream_read_reduce` at the largest buffer of that report. It is not a vendor
peak, and it is not a hardware limit.

A measured run refuses a synthetic ceiling report. It also refuses a ceiling
report from another interpreter, other libraries or another platform. The
report keeps the digest, the label and the load averages of the ceiling run.

## The speedup rule

For each timing size and each step, the report gives these values from the
same run:

- the median GB/s and the range from the IQR of the samples
- the fraction of the bandwidth ceiling and the fraction of `torch.sum`
- the step ratio to the base of the step
- the step ratio to R0
- the result of the hypothesis for the size class of n

A step ratio divides two median rates of the same size. The rule of the Day 76
lab calls the first side faster when two conditions are true:

- The ratio is 1.1 or more. The config sets this minimum.
- The 75th percentile time of the first side is less than the 25th percentile
  time of the second side.

The rule calls the first side slower when the same test is true in the other
direction. Otherwise the result is `no_clear_difference`. The vendor side has
the label "vendor GPU sum: torch.sum on MPS, PyTorch 2.15.0". The label names
the call and the PyTorch version, not a kernel.

## One run on the development host

This run used the pinned config on 2026-11-21 from 02:20:26 to 02:20:48 IST.
The host was a shared Darwin arm64 host with an Apple GPU, macOS 27.2, CPython
3.11.14, NumPy 1.26.4 and PyTorch 2.15.0. The environment did not set
`PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. The run held the GPU
lock that the timed GPU jobs on this host take. Other jobs used the CPU.

A fresh Day 74 run of `config/day-74-gpu-ceilings.json` gave the ceiling. It
ran from 02:20:07 to 02:20:26 IST in the same hold of the lock. These are its
facts:

- The digest of the ceiling report is
  `9d8932489e65a317980f29fe099e5039f84a58f375487680fcd3d4d4ac667ea1`.
- The ceiling is 87.77 GB/s, from `stream_copy` at 256 MiB.
- `stream_read_reduce` gave 83.51 GB/s at 256 MiB.
- That report recorded load averages of 12.29, 11.37 and 12.75 before its run,
  and 11.93, 11.33 and 12.72 after it.
- The `uptime` command printed 12.40, 11.38 and 12.76 before the ceiling run,
  and 12.58, 11.47 and 12.76 after it.

The reduction report recorded load averages of 12.58, 11.47 and 12.76 before
the run, and 12.51, 11.51 and 12.74 after the run. Before and after the run,
the `uptime` command printed the same values as the report. The report digest is
`5905fe9a28f58e242d669a5489b79a0937dab4595f584567d230818ae987df47`. The values
are host-specific Apple GPU (Metal) measurements from one run, not portable
figures.

Each pipeline allowed 1024 threads, reported a SIMD width of 32 and held the
declared threadgroup bytes. Thus the run admitted every kernel. The run caught
0 Python warnings, and the capture began before the import of torch.

### Median GB/s of each side

| n | R0 | R1 | R2 | R3 | R4 | R5 | `torch.sum` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 65536 | 13.4 | 23.5 | 28.4 | 33.8 | 38.6 | 24.8 | 36.9 |
| 1048576 | 19.3 | 50.8 | 90.1 | 141.2 | 199.1 | 283.8 | 331.9 |
| 16777216 | 19.7 | 56.2 | 93.6 | 110.1 | 107.3 | 106.6 | 101.5 |
| 67108864 | 19.9 | 55.2 | 100.9 | 108.2 | 107.7 | 108.0 | 108.1 |
| 10000019 | 19.9 | 55.3 | 95.2 | 97.5 | 99.0 | 93.6 | 18.7 |

The tables below give each rate with the range from the IQR of its 14
samples. "Over its base" and "Over R0" give the step ratio and the result of
the rule. "Of the ceiling" divides by 87.77 GB/s.

### n = 65536, class small

| Step | Kernel | Median GB/s (IQR range) | Of the ceiling | Over its base | Over R0 | Of `torch.sum` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `reduce_r0_interleaved` | 13.4 (13.2 to 13.4) | 0.152 | 0.362, slower | | 0.362 | missed |
| R1 | `reduce_r1_sequential` | 23.5 (23.1 to 23.7) | 0.267 | 1.755, faster | 1.755, faster | 0.636 | held |
| R2 | `reduce_r2_first_add` | 28.4 (28.2 to 29.0) | 0.324 | 1.213, faster | 2.129, faster | 0.771 | missed |
| R3 | `reduce_r3_simd_shuffle` | 33.8 (33.7 to 34.2) | 0.385 | 1.188, faster | 2.528, faster | 0.916 | missed |
| R4 | `reduce_r4_simd_sum` | 38.6 (38.2 to 38.8) | 0.440 | 1.144, faster | 2.891, faster | 1.047 | missed |
| R5 | `reduce_r5_grid_stride` | 24.8 (24.4 to 24.9) | 0.282 | 0.642, slower | 1.855, faster | 0.672 | held |
| vendor | `torch.sum` | 36.9 (36.1 to 37.4) | 0.420 | | | 1.000 | |

### n = 1048576, class medium

| Step | Kernel | Median GB/s (IQR range) | Of the ceiling | Over its base | Over R0 | Of `torch.sum` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `reduce_r0_interleaved` | 19.3 (19.2 to 19.6) | 0.220 | 0.058, slower | | 0.058 | held |
| R1 | `reduce_r1_sequential` | 50.8 (50.2 to 52.7) | 0.578 | 2.630, faster | 2.630, faster | 0.153 | held |
| R2 | `reduce_r2_first_add` | 90.1 (87.4 to 91.1) | 1.026 | 1.774, faster | 4.667, faster | 0.271 | held |
| R3 | `reduce_r3_simd_shuffle` | 141.2 (138.7 to 142.1) | 1.609 | 1.568, faster | 7.316, faster | 0.425 | held |
| R4 | `reduce_r4_simd_sum` | 199.1 (193.6 to 233.7) | 2.269 | 1.410, faster | 10.319, faster | 0.600 | held |
| R5 | `reduce_r5_grid_stride` | 283.8 (233.9 to 297.4) | 3.234 | 1.425, faster | 14.707, faster | 0.855 | missed |
| vendor | `torch.sum` | 331.9 (246.3 to 354.3) | 3.781 | | | 1.000 | |

### n = 16777216, class large

| Step | Kernel | Median GB/s (IQR range) | Of the ceiling | Over its base | Over R0 | Of `torch.sum` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `reduce_r0_interleaved` | 19.7 (19.5 to 19.8) | 0.225 | 0.194, slower | | 0.194 | held |
| R1 | `reduce_r1_sequential` | 56.2 (54.1 to 56.4) | 0.640 | 2.848, faster | 2.848, faster | 0.553 | held |
| R2 | `reduce_r2_first_add` | 93.6 (88.8 to 100.5) | 1.067 | 1.667, faster | 4.749, faster | 0.922 | held |
| R3 | `reduce_r3_simd_shuffle` | 110.1 (101.9 to 110.7) | 1.254 | 1.176, faster | 5.582, faster | 1.084 | missed |
| R4 | `reduce_r4_simd_sum` | 107.3 (102.6 to 112.8) | 1.222 | 0.974, no clear difference | 5.439, faster | 1.056 | held |
| R5 | `reduce_r5_grid_stride` | 106.6 (85.1 to 110.6) | 1.214 | 0.994, no clear difference | 5.404, faster | 1.050 | held |
| vendor | `torch.sum` | 101.5 (93.2 to 106.0) | 1.157 | | | 1.000 | |

### n = 67108864, class large

| Step | Kernel | Median GB/s (IQR range) | Of the ceiling | Over its base | Over R0 | Of `torch.sum` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `reduce_r0_interleaved` | 19.9 (19.5 to 20.0) | 0.227 | 0.184, slower | | 0.184 | held |
| R1 | `reduce_r1_sequential` | 55.2 (53.9 to 56.5) | 0.628 | 2.768, faster | 2.768, faster | 0.510 | held |
| R2 | `reduce_r2_first_add` | 100.9 (93.1 to 102.0) | 1.150 | 1.829, faster | 5.064, faster | 0.934 | held |
| R3 | `reduce_r3_simd_shuffle` | 108.2 (104.0 to 110.3) | 1.233 | 1.073, no clear difference | 5.432, faster | 1.001 | held |
| R4 | `reduce_r4_simd_sum` | 107.7 (102.2 to 112.8) | 1.227 | 0.995, no clear difference | 5.405, faster | 0.996 | held |
| R5 | `reduce_r5_grid_stride` | 108.0 (99.7 to 110.0) | 1.230 | 1.003, no clear difference | 5.421, faster | 0.999 | held |
| vendor | `torch.sum` | 108.1 (100.8 to 111.4) | 1.231 | | | 1.000 | |

### n = 10000019, class large

| Step | Kernel | Median GB/s (IQR range) | Of the ceiling | Over its base | Over R0 | Of `torch.sum` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `reduce_r0_interleaved` | 19.9 (19.6 to 20.0) | 0.226 | 1.063, no clear difference | | 1.063 | missed |
| R1 | `reduce_r1_sequential` | 55.3 (53.6 to 56.1) | 0.630 | 2.785, faster | 2.785, faster | 2.961 | held |
| R2 | `reduce_r2_first_add` | 95.2 (84.4 to 96.4) | 1.085 | 1.721, faster | 4.792, faster | 5.096 | held |
| R3 | `reduce_r3_simd_shuffle` | 97.5 (93.9 to 107.3) | 1.111 | 1.025, no clear difference | 4.910, faster | 5.222 | held |
| R4 | `reduce_r4_simd_sum` | 99.0 (91.5 to 103.8) | 1.128 | 1.015, no clear difference | 4.983, faster | 5.299 | held |
| R5 | `reduce_r5_grid_stride` | 93.6 (89.5 to 100.2) | 1.067 | 0.946, no clear difference | 4.714, faster | 5.013 | held |
| vendor | `torch.sum` | 18.7 (18.5 to 19.0) | 0.213 | | | 1.000 | |

### Correctness results

All 78 outputs stayed within their bounds: 13 cases and 6 kernels. The
largest bound ratio was 0.124, for R1 on the positive case with n = 511. In
the three exact cases, every kernel gave the exact sum. Each kernel gave the
same bits in its 3 runs of each case. At each timing size, a repeat of each
side kept the bits of its timed result, also for `torch.sum`. R3 kept the bits
of R2 in all 13 cases and at all 5 sizes, as the plan predicted.

### What helped

- Sequential addressing (R1): R1 was 2.630 to 2.848 times as fast as R0 at
  the four sizes from 2^20, and 1.755 times at 2^16. R0 stayed at 19.3 to 19.9
  GB/s at those four sizes, 0.220 to 0.227 of the ceiling.
- The first add during the load (R2): R2 was 1.667 to 1.829 times as fast as
  R1 at the four sizes from 2^20. At 2^16, it was 1.213 times as fast.
- SIMD-group shuffles (R3): R3 was faster than R2 at 2^16, 2^20 and 2^24, with
  ratios of 1.188, 1.568 and 1.176. At 2^26 and at the ragged size, the rule
  found no clear difference.
- `simd_sum` (R4): R4 was faster than R3 at 2^16 and 2^20, with ratios of
  1.144 and 1.410. From 2^24, the rule found no clear difference.
- The fixed grid (R5): R5 was 1.425 times as fast as R4 at 2^20, at 283.8
  GB/s.

The fastest kernel was R4 at 2^16 and at the ragged size, R5 at 2^20 and R3
at 2^24 and 2^26. It reached 2.891 to 14.707 times the R0 rate. From 2^24,
R2 to R5 all reached 93.6 to 110.1 GB/s.

### What did not help

- At 2^24 and 2^26, the ratios of R3, R4 and R5 over their bases were 0.974
  to 1.176. At the ragged size, they were 0.946 to 1.025. At 2^26, the ratios
  were 1.073, 0.995 and 1.003, with no clear difference.
  The data suggest that the device loads set the time there, but no counter
  shows it.
- The fixed grid (R5) was slower than R4 at 2^16, with a ratio of 0.642.
  Three quarters of its threads had no element there.

The predictions held at 23 of 30 points: 6 steps at 5 sizes. They missed at
these points:

- At 2^16, R0 was slower than `torch.sum`, and R2, R3 and R4 were each faster
  than their bases. The plan said that the host dispatch sets the time there.
  A call took 6.8 to 19.6 microseconds, and the kernels differed.
- At 2^20, R5 was faster than R4. At 2^24, R3 was faster than R2.
- At the ragged size, R0 showed no clear difference from `torch.sum`, because
  `torch.sum` was slow there.

### Rates above the ceiling

From 2^24, R2 to R5 reached 1.067 to 1.254 of the ceiling, and `torch.sum`
reached 1.157 and 1.231 of it at 2^24 and 2^26. A fraction above 1 shows that
these kernels read faster than the Day 74 kernels did in their run. The ceiling
is the best median of those kernels, not a hardware limit. The Day 74 copy also
writes each byte, and its read-reduce gave 83.51 GB/s. The run cannot tell
which difference matters.

At 2^20, the rates of R2 to R5 and `torch.sum` were 90.1 to 331.9 GB/s, up to
3.781 of the ceiling. The input of 4 MiB probably stays in a cache between the
256 calls of a sample. No counter shows this, so it is an inference.

At the ragged size of 10000019, `torch.sum` gave 18.7 GB/s. One call took
2.14 milliseconds there. At 16777216 elements, one call of `torch.sum` took
0.66 milliseconds. PyTorch does not report which kernel `torch.sum` ran, so
the lab cannot tell why.

### Limits of this run

- One run on one shared host gives each ratio. The load averages were near 12
  from other jobs.
- The steps form one chain, so each ratio measures one change on top of the
  changes before it.
- R1 changes the addressing and also removes the modulo. The run cannot
  separate the two effects.
- R3 removes 5 barriers and also the threadgroup traffic of 5 levels. The run
  cannot separate the two effects.
- The rates count the bytes of the input, not the partials or other traffic.
- The ceiling comes from a separate run with its own load averages.
- The lab reads no hardware counters. It does not measure divergence,
  occupancy, cache hits or bank conflicts. A limit that this guide names is an
  inference from timings and the measured ceiling.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The report records the items in this list:

- the config and its digest, and the host environment without host names
- the declared plan with every hypothesis and kernel digest
- the two MPS switches and the load averages before and after the run
- the Python warnings of the run, with the home directory as `~`
- the digest, limits and verdict of each kernel, and the ceiling record
- the correctness cases, the raw samples, the medians, the IQRs and the checks
- the speedup rows, the result of each hypothesis and the evidence boundary

The run captures warnings from a point before the runtime imports torch. In a
fresh process, the capture thus holds each warning that torch raised in that
process, up to the end of the run.

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk, and moves it into place with `os.replace`. It
refuses an output that aliases the config or the ceiling report. The readers
accept only regular files that are not symbolic links, and they refuse
duplicate JSON keys.

The verifier uses its own code, not the code of the run. It also uses three
modules that measure nothing: the config parser, the kernel registry and the
declared plan. It does not import torch. It does these checks:

1. It draws each input again from the seed and hashes it.
2. It sums each input exactly with `frexp` and Python fractions, a method
   that the run does not use.
3. It counts the depth of each order again and calculates each bound in
   exact rationals.
4. It derives each pipeline verdict again from the recorded limits.
5. It requires the sealed plan to equal the declared plan, byte for byte.
6. It calculates each statistic, rate, ratio, verdict and result of a
   hypothesis again from the raw samples.

The timings cannot be observed again, so the verifier checks them for
structure only. The tests change one field of a report at a time, seal the
report again, and require the verifier to refuse it.

The repository does not keep the report of the pinned run. The tests keep one
short report of a recorded config from 2026-11-21 at 02:17 IST, with digest
`54666a185b431d201d2c3a7719077f401df8a142957974fdbd21153f42c96be9`. Its
ceiling is the recorded Day 74 report of 2026-11-17, 82.28 GB/s at 4 MiB. A
host without torch can check it with the verifier.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 78 | 0 to 2^32 - 1 |
| `plan_sha256` | the digest of the declared plan | only that digest |
| `correctness.cases` | 13 [pattern, n] pairs | 1 to 64 different pairs with n up to 2^22 and at most 2^24 elements in all, which run every pattern and every path |
| `correctness.runs` | 3 | 2 to 5 runs of each kernel on each case |
| `benchmark.sizes` | 65536, 1048576, 16777216, 67108864 and 10000019 | 1 to 16 different sizes up to 2^26 |
| `benchmark.warmup` | 2 | 0 to 10 samples that are not timed |
| `benchmark.repeats` | 14 | 3 to 50 timed samples for each side |
| `benchmark.target_bytes_per_sample` | 1073741824 | 1 KiB to 1 GiB, with at most 4096 calls in a sample |
| `benchmark.max_runtime_seconds` | 300 | 1 to 600 |
| `comparison.min_faster_ratio` | 1.1 | more than 1.0 and at most 4.0 |

The config parser counts the planned work before a run. The pinned config has
13 cases, 7301371 case elements and 468 correctness launches. It also has 5
timing sizes, 490 timed samples, 492688 calls and 601996566976 bytes.

## Evidence boundary and exclusions

The evidence boundary of a measured report holds true or false fields and one
statement. The fields `apple_gpu_metal_this_host` and `host_specific` are true.
These fields are false: `cuda_kernel_executed`, `nvidia_gpu_used`,
`warp_level_claim`, `vendor_kernel_known`, `vendor_peak_claimed`,
`hardware_counters_used`, `occupancy_measured`, `distributed_claim`,
`production_claim` and `portable_performance_claim`.

The statement gives these facts:

- The timings are wall-clock timings on the Apple GPU of the host that made
  the report, at the recorded load averages.
- The kernels came from `torch.mps.compile_shader`, and `torch.sum` ran on the
  MPS device. PyTorch does not report which kernel `torch.sum` ran.
- `simd_shuffle_down` and `simd_sum` are the Metal counterpart of CUDA warp
  shuffles, not the same instructions.
- The lab read no hardware counter, so a limit named from these timings is an
  inference.
- The bandwidth ceiling comes from a separate Day 74 run on this host. It is
  not a vendor peak.
- No CUDA kernel ran, and the lab used no NVIDIA GPU. The timings do not
  measure or predict CUDA, NVIDIA, warp-level, distributed or production
  performance.

The lab does not record the GPU model, the clock frequency, the thermal state
or the other jobs on the host. The results come from one Apple GPU on one
host. They are not evidence of lecture viewing, completed exercises or learner
understanding.

## CUDA texts for a later NVIDIA run

The guide `docs/gpu-reduce-cuda-contract.md` writes R1 and R3 as CUDA C++ text
for a later run on an NVIDIA GPU. Each text adds the pairs of its Metal twin in
the same order, and a CPU simulator predicts the bits of each result. This host
compiled and ran none of the texts.

## Learner work

These items stay with the learner:

- Ajay must still watch Lecture 9, "Reductions".
- Ajay must still watch Lecture 20, "Scan Algorithm". This lab has no scan
  kernel.
- Ajay must still check the six steps of the plan against Lecture 9.

## Sources

- GPU MODE lectures, https://github.com/gpu-mode/lectures: Lecture 9,
  "Reductions", and Lecture 20, "Scan Algorithm".
- M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA, 2007. It shows
  the interleaved tree, sequential addressing, the first add during the load
  and a last warp with no barriers.
- W. W. Hwu, D. B. Kirk and I. El Hajj, Programming Massively Parallel
  Processors, 4th ed., Morgan Kaufmann, 2022, chapter 10, for reduction and
  control divergence.
- Apple, Metal Shading Language Specification, for `threadgroup_barrier`,
  `simd_shuffle_down`, `simd_sum` and the thread position attributes:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 4.2, for the bound of a summation order.
- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader`, and
  the MPS environment variables page of the PyTorch documentation.
