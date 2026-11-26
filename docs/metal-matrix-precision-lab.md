# Matmul with and without the Apple GPU matrix path

Day 82 of the curriculum follows GPU MODE Lecture 23, "Tensor Cores", and
Lecture 15, "CUTLASS". Ajay must still watch Lectures 23 and 15. This guide
does not summarize the lectures, and it makes no claim about what the
lectures show.

Apple GPUs have no NVIDIA Tensor Cores. The matrix path that this lab uses
on the Apple GPU is `simdgroup_matrix`. The 32 threads of one SIMD group hold
one 8 x 8 matrix together, and `simdgroup_multiply_accumulate` multiplies two
of them into a third. This lab times a matmul with that path and without it,
on the same tiles, in float32, half and bfloat.

Nothing here runs a CUDA kernel, uses an NVIDIA GPU or a Tensor Core, runs
PTX or runs a Triton kernel on a GPU. The lab reads no hardware counters.
PyTorch does not report which kernels its call runs. Every timing is a
host-specific Apple GPU (Metal) measurement at the load averages that the
report records.

## What the lab compares

One run compares seven sides on the same square operands:

1. S1, a tiled matmul with float32 inputs and scalar `fma` only.
2. M1, the tiles of S1 with float32 inputs on the matrix path.
3. M2, the tiles of S1 with half inputs on the matrix path.
4. M3, the tiles of S1 with bfloat inputs on the matrix path.
5. `torch.matmul` on the MPS device with float32, float16 and bfloat16
   tensors.

Each kernel adds its products into float32 accumulators and writes a
float32 C. A PyTorch call writes C in the type of its inputs.

The tiles are the same in the four kernels. One group of 128 threads owns
one 32 x 32 tile of C. The group walks K in steps of 16. At each step, the
group copies a 32 x 16 tile of A and a 16 x 32 tile of B into threadgroup
memory. A slot past the matrices holds 0. Each row of a tile holds 4 more
slots of padding. The lab does not test a swizzled layout.

S1 gives each thread a 2 x 4 block of C, at the rows `tr + 16 i` and the
columns `tc + 8 j` of the tile. Each SIMD group of M1, M2 and M3 owns a
16 x 16 block of C as 2 x 2 accumulators of `simdgroup_float8x8`. Each
thread holds 8 float accumulators in all four kernels.

## Commands

The `run` command needs a gpu-ceilings report from the Day 74 lab. A
measured run refuses a ceiling from another interpreter, library or
platform. The recorded Day 74 report of the test suite names macOS 27.2, and
this host has macOS 27.2.1. The Day 81 lab saved a fresh Day 74 report from
this host, with the same interpreter, libraries and platform. The pinned
Day 82 run used that report. On another host, make a fresh report first:

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
```

Then the `run` command makes a report, and the `verify` command recomputes
it:

```bash
python -m ai_journey.metal_tc_cli run \
  --config config/day-82-metal-tc.json \
  --ceilings /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-82-metal-tc.json
python -m ai_journey.metal_tc_cli verify \
  --report /tmp/day-82-metal-tc.json \
  --config config/day-82-metal-tc.json \
  --ceilings /tmp/day-74-gpu-ceilings.json
```

On a shared host, run each timed command under the GPU lock. The parser uses
the program name `ai-journey-metal-tc`. This branch adds no entry point for
that name, so use `python -m ai_journey.metal_tc_cli`.

The `run` command prints these lines:

1. One line for each step of the plan, with its predicted verdicts.
2. One line with the load averages before and after the run.
3. One line with both ceilings, the ridge point and the digest of their
   report.
4. One line with the Metal language version and the bfloat support.
5. One line for each correctness case, with the bound ratio, the relative
   error and the repeat result of each kernel.
6. One line for each size, with the TFLOP/s, the IQR and the share of the
   FMA ceiling of each side.
7. One line for each size, with the ratio and the verdict of each step.
8. One line with the count of Python warnings that the run caught.
9. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`metal_tc_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `metal_tc_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the summary
line with the prefix `verified_`. It never imports torch. With `--config`, it
requires the exact config of the run. With `--ceilings`, it requires the
exact ceiling report of the run.

The exit codes are the codes of the Day 73 to 81 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

## The kernels

| Kernel | Input | Group | Threadgroup bytes | Barriers | Text digest |
| --- | --- | --- | --- | --- | --- |
| `tc_s1_float` | float32 | 128 | 4864 | 2 + 0 | `ba1c7a853364ef75986c21b539ef07e5768331a998e9aec26c7c9b2a2b502d4f` |
| `tc_m1_float` | float32 | 128 | 5888 | 2 + 2 | `3442d3672b6025910ded3ed1a31705f89b13f5aa34447dd0e5bfb37f824862dd` |
| `tc_m2_half` | float16 | 128 | 3456 | 2 + 2 | `65cff25e1ceb6030b1cafd5a69f393a8f4a2e7391f96d337145d728087311ea7` |
| `tc_m3_bfloat` | bfloat16 | 128 | 3456 | 2 + 2 | `6a6a35477de0356297fff5a72eb2370f05a8a95c170f8ed8786a4a685ba6bf50` |

The barrier column gives the threadgroup barriers and then the SIMD-group
barriers. The main library holds S1, M1, M2 and the probe kernel
`tc_probe_version`. Its digest is
`93c29857581d44f58be7b5e8cf6357734e2b6e90ed1fced726e7021da67cc5cc`. M3
compiles in the bfloat library of its own, with the digest
`2ba3e64ba708a41c22f6f6f11ff71309e93a0cab059b8405d7a078766947ef68`.

These rules keep each kernel correct:

- A threadgroup barrier is only in the K loop. The bound of that loop is K,
  so every thread of a group reaches every barrier.
- A SIMD-group barrier is only in the edge store of an MMA kernel. Its
  branch reads the position of the fragment only, so every lane of the SIMD
  group takes it. A test reads each kernel text and checks both rules.
- A fragment inside C goes to device memory with `simdgroup_store`. A
  fragment that crosses an edge goes to a staging slot in threadgroup
  memory, and each lane stores only its values inside C.

The runtime checks the support of the bfloat path on the host:

1. The probe kernel writes `__METAL_VERSION__` of the main library.
2. M3 runs only when the bfloat library compiles and one 8 x 8 x 8 product
   of small integers is exact.
3. The bfloat16 PyTorch call runs only when the same product on bfloat16
   tensors is exact.

When a check fails, the report records the side as unsupported, with its
reason. The side keeps its turn in the rotation but runs nothing, and no
step with that side gets a verdict.

## The plan

The plan names nine steps. Each step compares a first side with a base side
by the ratio rule of Day 76.

| Step | First side | Base side | Predicted verdict |
| --- | --- | --- | --- |
| `m1_over_s1` | M1 | S1 | faster |
| `m2_over_m1` | M2 | M1 | faster or no_clear_difference |
| `m3_over_m1` | M3 | M1 | faster or no_clear_difference |
| `m3_over_s1` | M3 | S1 | faster |
| `torch_float16_over_torch_float32` | PyTorch float16 | PyTorch float32 | faster or no_clear_difference |
| `torch_bfloat16_over_torch_float32` | PyTorch bfloat16 | PyTorch float32 | faster or no_clear_difference |
| `m1_over_torch_float32` | M1 | PyTorch float32 | slower |
| `m2_over_torch_float16` | M2 | PyTorch float16 | slower |
| `m3_over_torch_bfloat16` | M3 | PyTorch bfloat16 | slower |

The plan digest is
`69a30fcd7855fadccf99d7028dccf726b7d3812cbaac6ff87573bdd61521ec1a`. The
pinned config names this digest, so a config for other hypotheses cannot
run.

A scratch prototype ran before the lab wrote the plan, at 1024 and 2048. Its
log is in the evidence folder of the run, and no verifier checked it. These
facts from the prototype are in the hypotheses:

- With 64 x 64 tiles, M1 reached only about 0.7 TFLOP/s, so the plan uses
  32 x 32 tiles.
- With 32 x 32 tiles, M1 took about 0.65 to 0.75 of the time of S1.
- MMA chains with their operands in registers ran at 4.6 to 4.8 TFLOP/s for
  float, half and bfloat.

So most predictions came after a first look at the data.

## Correctness

Each case draws standard-normal float32 operands from its own seeded
stream. M2 reads them rounded to half, and M3 reads them rounded to bfloat,
each to the nearest value with ties to even. A quiet NaN follows each
operand, so a read past an operand poisons an output. Before each run, the
C buffer holds `m n + 64` quiet NaNs with one known bit pattern.

The pinned cases, as `m x n x k`, cover these edges of the tiling:

- `1 x 1 x 1`: one output.
- `7 x 9 x 5`: a K below one step of 16.
- `33 x 65 x 17` and `100 x 129 x 70`: fragments that cross the edges of C,
  and a K that is not a multiple of 16.
- `64 x 96 x 48`: full tiles only, so each store is direct.
- `257 x 131 x 1000`: a K of 1000.

The reference is the float64 product of the float32 draws. It adds the
exact products of each output in increasing k order, so the bits do not
depend on the BLAS of the host. The same loop gives the scale
`S = |A| @ |B|`.

The declared bound of an output comes from the unit roundoff `u` of the
input type and from K:

```text
|C - R| <= (gamma_K(2**-24) (1 + u)**2 + 2 u + u**2 + gamma_K(2**-53)) S
           + e (1 + u) (1 + gamma_K(2**-24)) (R_i + C_j)
           + K e**2 (1 + gamma_K(2**-24)) + 2 (K + 1) 2**-126
```

Here `gamma_K(v) = K v / (1 - K v)`, `R_i` is the sum of row i of `|A|`, and
`C_j` is the sum of column j of `|B|`. The type sets `u` and `e`:

- float32: `u = 0` and `e = 2**-126`, because the kernels read the draws.
- half: `u = 2**-11` and `e = 2**-14`, the smallest normal half.
- bfloat: `u = 2**-8` and `e = 2**-126`.

The `e` terms cover a subnormal operand and a device that flushes it to
zero. The bound assumes that the float32 accumulators round each product
and each sum at most once, in any order. The matrix path names no order, so
this assumption is a rule of the run, not a fact about the hardware.

Each kernel runs twice on each case. Three rules apply to the output of each
run, and a case that breaks one stops the run:

1. Each tail value of C keeps its sentinel bits.
2. Each output is finite.
3. Each output is within the bound of its input type.

Two facts are measurements, not rules. The run records whether the two runs
of a kernel gave the same bits. It also records whether M1 gave the bits of
S1 on the same float32 operands.

## Timing

The timing is the Day 74 and Day 76 method, with the Day 78 rotating timer:

1. Each side runs 2 untimed warmup samples.
2. Each timed sample calls `torch.mps.synchronize()`, reads
   `time.perf_counter_ns`, launches a batch of calls, synchronizes again and
   reads the clock again.
3. Repeat `r` runs side `r mod 7` first, so each side takes each position in
   3 of the 21 repeats.

The sizes are 512, 1024, 2048 and 4096. A batch holds the fewest whole
calls whose `2 n**3` operations reach `2**34`: 64, 8, 1 and 1 calls. Each
side of a size uses the same batch, so a ratio of median times compares
equal work.

After the last sample, the run checks 16 rows of C, spread evenly from the
first row to the last, at every column. The kernels must be within their
declared bounds. A PyTorch call names no kernel, so the lab applies a sanity
bound only. That bound is `v S + 2**-100`, with `v` of `2**-12`, `2**-6` and
`2**-4` for float32, float16 and bfloat16.

The ratio rule is the rule of Day 76, with the declared `min_faster_ratio` of
1.1. The ratio divides the median sample time of the base side by the median
sample time of the first side.

- `faster`: the ratio is 1.1 or more, and the p75 sample time of the first
  side is below the p25 time of the base side.
- `slower`: the inverse ratio is 1.1 or more, and the p75 sample time of the
  base side is below the p25 time of the first side.
- `no_clear_difference`: any other result.

## The ceiling join

Both ceilings come from one verified Day 74 gpu-ceilings report. The FMA
ceiling is the best median rate of the Day 74 float32 FMA chains. The
bandwidth ceiling is the better median rate of the two Day 74 stream
kernels. Neither ceiling is a vendor peak or a hardware limit.

A call of size n reads A and B once and writes C once, at least. That is
`(2 w + c) n**2` bytes, with `w` bytes for each input value and `c` bytes
for each output value. The roofline of a side is `min(F, I B)`, with the FMA
ceiling `F`, the intensity `I` in FLOP/byte and the bandwidth ceiling `B`.

## The pinned run

This run used the pinned config on 2026-11-26 at 18:14 IST. The host was a
shared Darwin arm64 host with an Apple GPU, macOS 27.2.1, CPython 3.11.14,
NumPy 1.26.4 and PyTorch 2.15.0. The environment did not set
`PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. The run held the GPU
lock. Other jobs used the CPU.

The ceilings came from the Day 74 run of 2026-11-24 at 08:47 IST, with the
digest `cd75564918285983fe22b4106b5084b44ef42ed9313ca2a8ef19fa164e68fd86`.
The FMA ceiling is 4665.5 GFLOP/s. The bandwidth ceiling is 107.71 GB/s.
The ridge point is 43.3 FLOP/byte. That run recorded load averages of 11.73,
14.03 and 16.29 before it.

The Day 82 report recorded load averages of 7.09, 5.19 and 5.35 before the
run. It recorded 5.86, 5.02 and 5.29 after the run. The `uptime` command
printed 4.83, 4.72 and 5.19 before the command and 5.95, 5.05 and 5.30 after
it.

The report digest is
`2b18c5b9922280590c7a4e8f277a38961202beba64046a9f036485b8ab349550`. The
run took 16.5 seconds after the libraries compiled. The run caught 0 Python
warnings, and the capture began before the import of torch. The Metal
language version was 400. The runtime found support for M3 and for the
bfloat16 PyTorch call. Each pipeline allowed 1024 threads, reported a SIMD
width of 32 and reported its declared threadgroup bytes. The CPU verifier
took about 3 seconds.

### Correctness results

Each cell gives the largest bound ratio and then the largest relative error
`|C - R| / S` of the case.

| Case | S1 | M1 | M2 | M3 |
| --- | --- | --- | --- | --- |
| 1 x 1 x 1 | 0.026 / 1.5e-09 | 0.026 / 1.5e-09 | 0.065 / 7.4e-05 | 0.349 / 2.7e-03 |
| 7 x 9 x 5 | 0.305 / 9.1e-08 | 0.305 / 9.1e-08 | 0.303 / 3.3e-04 | 0.481 / 3.8e-03 |
| 33 x 65 x 17 | 0.141 / 1.4e-07 | 0.141 / 1.4e-07 | 0.366 / 4.1e-04 | 0.363 / 2.8e-03 |
| 100 x 129 x 70 | 0.046 / 1.9e-07 | 0.046 / 1.9e-07 | 0.199 / 2.3e-04 | 0.248 / 1.9e-03 |
| 64 x 96 x 48 | 0.057 / 1.6e-07 | 0.057 / 1.6e-07 | 0.215 / 2.4e-04 | 0.249 / 1.9e-03 |
| 257 x 131 x 1000 | 0.004 / 2.4e-07 | 0.004 / 2.4e-07 | 0.050 / 6.0e-05 | 0.069 / 5.4e-04 |

- All 6 cases met the three rules in both runs of each kernel.
- The largest bound ratio was 0.48, for M3 at 7 x 9 x 5. At small K, the
  input rounding term `2 u S` sets most of the bound of half and bfloat.
- The largest relative error was 3.8e-3 for bfloat, 4.1e-4 for half and
  2.4e-7 for float32. The unit roundoffs of the three types are `2**-8`,
  `2**-11` and `2**-24`.
- Each kernel gave the same bits in both runs of each case.
- M1 gave the bits of S1 in all 6 cases. At all 4 timing sizes, M1 and the
  float32 PyTorch call also gave the bits of S1. The lab cannot show the
  order of operations inside the matrix path. It shows only that the bits
  were equal for these operands.

### The timing tables

Each time is the median time of one call in milliseconds, with the IQR in
parentheses.

| Side | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- |
| S1 | 0.212 (0.004) | 1.504 (0.007) | 12.199 (0.412) | 98.171 (1.306) |
| M1 | 0.147 (0.004) | 1.040 (0.010) | 8.681 (0.077) | 78.656 (2.130) |
| M2 | 0.147 (0.007) | 1.002 (0.005) | 10.129 (0.094) | 95.811 (2.719) |
| M3 | 0.145 (0.006) | 1.003 (0.017) | 10.030 (0.122) | 95.502 (1.466) |
| PyTorch float32 | 0.100 (0.014) | 0.585 (0.009) | 4.431 (0.077) | 34.036 (1.089) |
| PyTorch float16 | 0.088 (0.008) | 0.535 (0.013) | 4.059 (0.050) | 30.399 (1.013) |
| PyTorch bfloat16 | 0.089 (0.008) | 0.533 (0.009) | 4.025 (0.082) | 30.019 (0.341) |

Each cell gives the achieved TFLOP/s and then its share of the FMA ceiling.

| Side | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- |
| S1 | 1.27 / 0.27 | 1.43 / 0.31 | 1.41 / 0.30 | 1.40 / 0.30 |
| M1 | 1.82 / 0.39 | 2.06 / 0.44 | 1.98 / 0.42 | 1.75 / 0.37 |
| M2 | 1.83 / 0.39 | 2.14 / 0.46 | 1.70 / 0.36 | 1.43 / 0.31 |
| M3 | 1.85 / 0.40 | 2.14 / 0.46 | 1.71 / 0.37 | 1.44 / 0.31 |
| PyTorch float32 | 2.68 / 0.57 | 3.67 / 0.79 | 3.88 / 0.83 | 4.04 / 0.87 |
| PyTorch float16 | 3.04 / 0.65 | 4.01 / 0.86 | 4.23 / 0.91 | 4.52 / 0.97 |
| PyTorch bfloat16 | 3.02 / 0.65 | 4.03 / 0.86 | 4.27 / 0.91 | 4.58 / 0.98 |

Each cell gives the ratio and the verdict of the step at one size.

| Step | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- |
| `m1_over_s1` | 1.44 faster | 1.45 faster | 1.41 faster | 1.25 faster |
| `m2_over_m1` | 1.00 no clear difference | 1.04 no clear difference | 0.86 slower (missed) | 0.82 slower (missed) |
| `m3_over_m1` | 1.01 no clear difference | 1.04 no clear difference | 0.87 slower (missed) | 0.82 slower (missed) |
| `m3_over_s1` | 1.46 faster | 1.50 faster | 1.22 faster | 1.03 no clear difference (missed) |
| `torch_float16_over_torch_float32` | 1.14 faster | 1.09 no clear difference | 1.09 no clear difference | 1.12 faster |
| `torch_bfloat16_over_torch_float32` | 1.13 faster | 1.10 no clear difference | 1.10 faster | 1.13 faster |
| `m1_over_torch_float32` | 0.68 slower | 0.56 slower | 0.51 slower | 0.43 slower |
| `m2_over_torch_float16` | 0.60 slower | 0.53 slower | 0.40 slower | 0.32 slower |
| `m3_over_torch_bfloat16` | 0.61 slower | 0.53 slower | 0.40 slower | 0.31 slower |

The predictions held at 31 of 36 step results. All five misses came from
the half and bfloat kernels at 2048 and 4096.

### How to explain each speedup ratio

The intensity of each side is 85 FLOP/byte or more, and the ridge point is
43.3 FLOP/byte. So the roofline of every side at every size is the FMA
ceiling. The roofline model calls every side compute-bound. By this model,
the bytes of the operands do not set the ratios below.

- M1 over S1 is 1.44, 1.45, 1.41 and 1.25. By the model of the text, the two
  kernels do the same count of fma steps, and they gave the same bits. For 8
  values of k, one SIMD group of M1 reads 4 fragments, 256 values, from
  threadgroup memory for 2048 fma steps. One thread of S1 reads 6 values for
  8 fma steps, 6 times as many for each step. This count is a model of the
  text. The lab reads no counter, so it cannot show which resource limits S1.
- M2 over M1 and M3 over M1 are about 1.0 at 512 and 1024, and 0.82 to 0.87
  at 2048 and 4096. Half and bfloat halve the bytes of the tiles, but every
  side is compute-bound by the model. The scratch MMA chains ran at 4.6 to
  4.8 TFLOP/s for the three types. So this lab expected no gain from the type
  alone. The lab cannot show why M2 and M3 lost speed at the large sizes.
- M3 over S1 is the pair of the hands-on task: bf16 with the matrix path
  over fp32 without it. It is 1.46, 1.50, 1.22 and 1.03. M3 over M1 is 1.04
  or less at each size, so the gain comes from the matrix path, not from
  bfloat. At 4096, the loss of M3 against M1 removes the gain.
- The float16 and bfloat16 PyTorch calls ran 1.09 to 1.14 times as fast as
  the float32 call. At 4096, the float32 call reached 0.87 of the FMA
  ceiling. If the FMA ceiling also limits half math, a faster type can gain
  at most 1 / 0.866 = 1.15 there. The bfloat16 call reached 0.98 of the
  ceiling, so the measured 1.13 is near that limit.
- The kernels ran at 0.31 to 0.68 of the speed of the PyTorch call of the
  same type. The gap grew with the size. This lab cannot tell which kernel
  PyTorch ran.

The bf16 over fp32 ratios of this host, 1.03 to 1.50, are not Tensor Core
ratios. The hands-on task of Day 82 asks for a matmul with and without
Tensor Cores. This host has none, so this lab cannot measure that ratio. The lab
did not measure a half or bfloat FMA ceiling. The scratch MMA chains are the
only sign that the three types share one arithmetic rate here.

## The boundary of the evidence

These are wall-clock timings of one Apple GPU on one shared host at the load
averages above. They do not measure or predict CUDA, NVIDIA, Tensor Core,
CUTLASS or Triton performance on other hardware. The lab read no hardware
counter, so it observed no cache hit, stall, bank conflict or occupancy.
Each byte and operation count is a model. The PyTorch sides name a call, not
a kernel.

The CPU verifier recomputes the config, the plan and each operand,
reference, scale and bound digest. It recomputes each launch and each
admission. From the raw samples, it recomputes each statistic, rate,
roofline, ratio, verdict and ceiling share. It cannot recompute the fields
that only the GPU run saw:

- The raw samples of each side. The verifier checks their form only.
- The digest and the errors of each kernel output. The verifier checks the
  errors against the bounds and the scale that it rebuilds.
- `identical_repeats` and the flags of equal bits. The verifier checks that
  they agree with the digests and the errors.
- The Metal language version and the support records. The verifier checks
  their form.
- The pipeline limits. The verifier checks their form and rebuilds the
  admissions from them.
- The PyTorch checks, the warnings, the MPS switches and the load averages.
  The verifier checks their form.

The report holds the same list in its `verification` section. A matching
digest shows internal consistency. It cannot prove that a timing was
observed or that a GPU ran.

## The recorded report for CI

`tests/test_metal_tc_recorded.py` holds a short real report from this host,
byte for byte. The run used five correctness cases, the sizes 64 and 256 and
seven repeats, on 2026-11-26 at 18:15 IST. The `uptime` command printed load
averages of 4.80, 4.87 and 5.22 before and after the run. At these small
sizes, the cost of each synchronize call sets most of the time. Eight of its
18 predictions missed, and the report records that result. The report is
there to be verified on a host without torch, not to show speed.

The tests do not depend on the operating system of the host. Each test that
needs MPS skips where MPS is not available. Under an emulation of a Linux
host with no MPS, each test passed or skipped. This branch adds no CI job.

## Work that Ajay must still do

- Ajay must still watch Lectures 23 and 15.
- Ajay must still check the Tensor Core and CUTLASS facts of the lectures
  against the NVIDIA documentation. This guide states none of them.
- Ajay must still add a CI job that runs these tests with CPU-only torch, and
  read the log of its first run.
- Ajay must still run the same comparison on a GPU with Tensor Cores. That
  comparison needs a host with an NVIDIA GPU.
