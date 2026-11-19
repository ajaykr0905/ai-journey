# Tiled matmul with threadgroup memory on the Apple GPU

Day 76 of the curriculum follows GPU MODE Lecture 5, "Going Further with CUDA
for Python Programmers", and Lecture 16, "On Hands Profiling". The exercise is
a tiled matmul with shared memory. It asks for three numbers: naive, tiled and
cuBLAS. This lab is the Metal half of that exercise, for a Mac with an Apple
GPU.

The lab compiles three tiled float32 kernels at run time, with tile sizes 8,
16 and 32. It checks them against the Day 74 float64 reference and the Day 74
error bound. Then it times them in one run with the faster naive Day 74 kernel
and with `torch.matmul` on the MPS device.

Nothing here runs a CUDA kernel or uses an NVIDIA GPU. The lab does not measure
cuBLAS. The third number comes from `torch.matmul` on the MPS device, and
PyTorch does not report which kernel it runs. Every timing is a host-specific
Apple GPU (Metal) measurement at the load averages that the report records.

## Commands

```bash
python -m ai_journey.matmul_tiled_cli run \
  --config config/day-76-matmul-tiled.json \
  --output /tmp/day-76-matmul-tiled.json
python -m ai_journey.matmul_tiled_cli verify \
  --report /tmp/day-76-matmul-tiled.json \
  --config config/day-76-matmul-tiled.json
```

The parser uses the program name `ai-journey-matmul-tiled`. This change adds
no console script, so run the module with `python -m`.

The `run` command compiles the kernels, checks them, times them, checks its own
report and writes the report. It prints these lines:

1. One line with the load averages before and after the run.
2. One line for each tile that its pipeline refused, with the reason.
3. One line for each correctness case, with the bound and the largest relative
   error of each kernel. The line also tells if each kernel held the bits of
   the CPU replay.
4. One line for each sweep point, with the median GFLOP/s and the IQR range of
   each side.
5. One line for each sweep point with the three numbers and the two step
   ratios.
6. One line with the count of Python warnings that the run caught.
7. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`matmul_tiled_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `matmul_tiled_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the same summary
line with the prefix `verified_`. With `--config`, `verify` also requires that
the report comes from that exact config.

The exit codes are:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

On the development host, one run of the pinned config took 38 seconds, with the
torch import. The run wrote a report of about 65 KB. The `verify` command took
about 3 seconds for that report.

## Hosts without MPS

The `run` command needs an Apple GPU on macOS. It also needs a PyTorch build
with an MPS device and `torch.mps.compile_shader`. The runtime uses the probe
of the Day 73 Metal lab. The probe reads the operating system first, and it
imports torch only on macOS. It names the first requirement that is missing,
for example `unsupported_operating_system` on a Linux host.

When a requirement is missing, `run` writes the reason to standard error and
exits with code 3. It does not make the output directory or the output file.
The `verify` command and the report reader never import torch, so they work on
any host with NumPy.

The tests that need the GPU skip on other hosts and report the reason. The
other tests use a NumPy stand-in for the GPU and short measured reports from
the development host. The stand-in returns the CPU replay of the fma chain,
which is what the GPU returned on that host.

## What tiling changes

The lab computes `C = A @ B`, with A of shape (M, K), B of shape (K, N) and C of
shape (M, N). All three matrices are row-major float32. The naive kernel runs
one thread for each element of C. Each thread reads its row of A and its column
of B from device memory, so each output costs 2K global loads.

A tiled kernel gives one T x T threadgroup one T x T tile of C. It walks K in
phases of T steps. In each phase, these steps occur:

1. Each thread loads one element of A and one element of B into two
   threadgroup arrays. Thus the group loads one T x T tile of each matrix.
2. The group waits at a barrier until every load is complete.
3. Each thread reads T elements of each tile from threadgroup memory, and it
   applies one fma step for each pair.
4. The group waits at a second barrier, and then the next phase writes the
   tiles again.

One staged element thus serves T outputs. The global loads for each output
decrease from 2K to about 2K / T. The traffic model of the Day 74 memory-access
lab gives these counts. At 4096 x 4096 x 4096, the counts are:

| Kernel | Global loads for each output | Threadgroup memory |
| --- | --- | --- |
| naive `matmul_x_columns` | 8192 | 0 bytes |
| `matmul_tiled_8` | 1024 | 512 bytes |
| `matmul_tiled_16` | 512 | 2048 bytes |
| `matmul_tiled_32` | 256 | 8192 bytes |

The model counts loads from device memory. It does not model caches, so it
does not see the reuse that the GPU caches can give the naive kernel. The
report records the modeled loads of each kernel at each sweep point.

## Metal threadgroup memory and CUDA shared memory

Metal threadgroup memory is the counterpart of CUDA shared memory. The
threadgroup is the counterpart of the CUDA block. The table gives the other
terms that the kernels use.

| CUDA | Metal in this lab |
| --- | --- |
| `__shared__ float As[T][T];` | `threadgroup float a_tile[T][T];` |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `threadIdx` | `thread_position_in_threadgroup` |
| `blockIdx * blockDim + threadIdx` | `thread_position_in_grid` |
| block | threadgroup |
| warp | SIMD group |
| shared memory size as a launch parameter | none: each tile size has its own kernel |

In PyTorch 2.15.0, a launch through `torch.mps.compile_shader` takes four
keywords: `threads`, `group_size`, `arg_casts` and `error_buf_idx`. None of them
sets the size of threadgroup memory. Thus each tile size is its own kernel with
arrays of a fixed size, `2 * T * T * 4` bytes.

## Kernels

The kernel source, in the Metal Shading Language, comes from
`matmul_tiled_source.py`. One template makes each kernel: `matmul_tiled_8`,
`matmul_tiled_16` and `matmul_tiled_32`. Each kernel uses x for the column of C
and y for the row, as the faster Day 74 mapping does. The buffers and the
32-bit scalars in slots 3 to 5 are the same as in the Day 74 kernels.

Three rules keep the arithmetic of a tiled kernel equal to the CPU replay:

- In each phase, `start` is the first k index of the phase, and the inner
  loop runs `i < min(T, k - start)`. Thus each output gets exactly K steps of
  `sum = fma(a_tile[local.y][i], b_tile[i][local.x], sum);` in increasing k.
- A slot past the end of K, M or N holds 0. The inner loop never reads a slot
  past K. A slot past M or N feeds only the output of a padding thread.
- Only the store has a guard. Thus every thread of a group gets to both
  barriers of every phase, as `threadgroup_barrier` requires.

With the first rule, exact bits do not depend on the identity
`fma(0, 0, sum) == sum`. That identity is false when `sum` is -0.0, because
the result is +0.0.

The registry in `matmul_tiled_variants.py` holds each kernel with its tile, its
source text, its digest and its correctness mode. Each digest covers the text
of one kernel. A run compiles the kernels that its config names into one
library, in config order, after a two-line header. The Day 74 naive source
stays unchanged, as a second library. A later lab can add a variant, and no
digest in this table changes.

| Source | SHA-256 |
| --- | --- |
| `matmul_tiled_8` | `21b53bf8ce1c123d1755b12512dedf1dfa1a5cc2509d476ac1019538a924e1ba` |
| `matmul_tiled_16` | `9e79d70317e309e13e043d0c015574670bfe09f0d28cf23abc4300f98a2eeb13` |
| `matmul_tiled_32` | `f5ae7c6f03bcd1772c6eec5aed127916cb962b70bc7f99179d9f115a589ced30` |
| tiled library of the pinned config | `08b88577018024d54344a00f991f387225f1085047caac66f347dc5e719a5fa5` |
| Day 74 naive source | `beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc` |

The three Day 76 kernels declare the mode `k_order_fma`. This mode states that
the source runs the fma chain of the CPU replay. It does not state that the
GPU output has the bits of the replay. The run measures that property for each
kernel in each case. The other mode, `reordered_sum`, is for a later kernel
that adds the products in a different order.

## Pipeline limits and refusals

The runtime reads the limits of each compiled pipeline: the largest
threadgroup, the SIMD width and the static threadgroup memory. A tile gets a
refusal when its pipeline allows fewer than T * T threads. A tile also gets a
refusal when the static threadgroup memory of its pipeline is not
`2 * T * T * 4` bytes. The report records each refusal with its numbers. A
refused tile never runs, and the run continues with the other tiles.

On the development host, each pipeline allowed 1024 threads and reported a SIMD
width of 32. The static threadgroup memory was 512, 2048 and 8192 bytes, equal
to the declared bytes. Thus the pinned run refused no tile.

## Correctness

The correctness cases are the 9 shapes of the Day 74 config, with seed 76. The
config parser requires these properties for each tile size T:

- One shape has an M that is not a multiple of T.
- One shape has an N that is not a multiple of T.
- One shape has a K that is not a multiple of T.
- One shape has a K that is more than T and not a multiple of T.

Thus each tile pads both axes, and some phase after the first is short. Each
case runs the naive kernel and each admitted tile once. Each output buffer is
longer than C by its padding threads, and the tail holds sentinel values
before the launch. Two rules apply, and a case that breaks a rule stops the
run:

- The tail must keep its sentinel values.
- The largest relative error must not be more than the Day 74 bound,
  `(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)`.

The run compares the bits of each output with the CPU replay of the Day 74 lab.
The record counts the elements whose bits differ, and `replay_bits` is true
only when the count is 0. A different count does not stop the run. It is a
result.

In the pinned run, all 36 outputs held the bits of the replay: 9 cases and 4
kernels. Each guard tail kept its sentinel values. The largest relative error
was 0.457 of the bound, at 1 x 1 x 1. At 1000 x 999 x 100, the largest relative
error of each kernel was 2.99e-7.

## Benchmark

The benchmark times five sides on the same device operands at each sweep
point. The sides are the naive `matmul_x_columns`, the three tiles and
`torch.matmul(a, b, out=c)` with a preallocated output. The benchmark uses the
Day 74 method and the Day 74 sweep timer:

1. Each side runs 2 samples that are not timed.
2. Each timed sample calls `torch.mps.synchronize()` and reads
   `time.perf_counter_ns`.
3. The sample then launches a batch of calls, calls `torch.mps.synchronize()`
   again, and reads the clock again.
4. The first side of repeat r is side r mod 5, so the order moves by one side
   at each repeat. Each side has 11 timed samples.
5. After the last sample, the output of each side must agree with the naive
   output within `4 * gamma_K(2**-24) * (|A| @ |B|)`, as the device calculates
   it in float32.

A batch holds the fewest whole calls that reach 2^30 FLOPs. At the pinned
shapes, one call is enough. The GFLOP/s of a side is the FLOPs of one sample
divided by the median sample time in nanoseconds. The IQR of the sample times
gives the range of each rate. The budget is 300 seconds, and the run stops
between two samples after the budget ends.

## The three numbers and the step ratios

Each sweep point gives three numbers from the same run:

- Naive: `matmul_x_columns`, the faster Day 74 kernel.
- Tiled: the admitted tile with the highest median GFLOP/s at that point. A
  tie goes to the tile that the config lists first.
- Vendor: "vendor GPU matmul: torch.matmul on MPS, PyTorch 2.15.0". The label
  names the call and the PyTorch version, not a kernel.

The cuBLAS field of each point says "not measured, no NVIDIA GPU or cuBLAS on
this host". It holds no number.

The report gives two step ratios for each point: tiled over naive, and vendor
over tiled. Each ratio divides two median rates of the same point in the same
run. The rule calls the first side faster when two conditions are true:

- The ratio is 1.1 or more. The config sets this minimum.
- The 75th percentile time of the first side is less than the 25th percentile
  time of the second side.

The rule calls the first side slower when the same test is true in the other
direction. Otherwise the result is `no_clear_difference`.

## One run on the development host

This run used the pinned config on 2026-11-19 at 11:43 IST. The host
was a shared Darwin arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14,
NumPy 1.26.4 and PyTorch 2.15.0. The PyTorch float32 matmul precision was
`highest`. The environment did not set `PYTORCH_MPS_FAST_MATH` or
`PYTORCH_MPS_PREFER_METAL`. The run held the GPU lock that the timed GPU jobs on
this host take. Other jobs used the CPU.

The report recorded load averages of 9.93, 11.62 and 14.06 before the run. It
recorded 9.70, 11.40 and 13.90 after the run. The `uptime` command printed
10.45, 11.75 and 14.12 before the lock, and 9.57, 11.34 and 13.87 after the
run.

The report digest is
`371d65363ce5c53813c34a6edfc9db4725445bf4ffea605b7c81e0076fc03427`. Each cell
gives the median GFLOP/s of 11 samples. The range in parentheses comes from
the IQR of the sample times. The values are host-specific Apple GPU (Metal)
measurements from one run, not portable figures.

| M x N x K | Naive | Tiled (tile 32 at each point) | Vendor | cuBLAS |
| --- | --- | --- | --- | --- |
| 1000 x 1000 x 1000 | 369.1 (358.0 to 375.3) | 324.2 (322.4 to 327.9) | 2160.0 (1981.8 to 2275.5) | not measured |
| 1024 x 1024 x 1024 | 380.2 (370.3 to 381.4) | 338.5 (332.6 to 342.8) | 2294.1 (2002.0 to 2380.0) | not measured |
| 2048 x 2048 x 2048 | 387.3 (382.5 to 393.6) | 339.6 (337.1 to 348.3) | 3608.8 (3464.3 to 3656.4) | not measured |
| 4096 x 4096 x 4096 | 369.4 (357.1 to 371.6) | 353.5 (333.1 to 359.1) | 3940.1 (3703.5 to 4010.8) | not measured |
| 2000 x 3000 x 1000 | 393.8 (390.2 to 398.3) | 342.3 (339.8 to 345.4) | 3448.9 (3269.8 to 3483.6) | not measured |

The step ratios come from the same run:

| M x N x K | Tiled / naive | Result | Vendor / tiled | Result |
| --- | --- | --- | --- | --- |
| 1000 x 1000 x 1000 | 0.878 | slower | 6.662 | faster |
| 1024 x 1024 x 1024 | 0.890 | slower | 6.778 | faster |
| 2048 x 2048 x 2048 | 0.877 | slower | 10.626 | faster |
| 4096 x 4096 x 4096 | 0.957 | no clear difference | 11.146 | faster |
| 2000 x 3000 x 1000 | 0.869 | slower | 10.077 | faster |

The medians of the smaller tiles were lower at each point:

| M x N x K | Tile 8 | Tile 16 | Tile 32 |
| --- | --- | --- | --- |
| 1000 x 1000 x 1000 | 269.1 | 301.5 | 324.2 |
| 1024 x 1024 x 1024 | 273.5 | 314.8 | 338.5 |
| 2048 x 2048 x 2048 | 265.4 | 306.4 | 339.6 |
| 4096 x 4096 x 4096 | 260.6 | 313.3 | 353.5 |
| 2000 x 3000 x 1000 | 270.9 | 311.2 | 342.3 |

These are the results of the run:

- The tiled kernels were not faster than the naive kernel at any point. The
  best tile, 32, was slower at 4 points.
- At 4096 x 4096 x 4096, the rule found no clear difference. There, naive over
  tiled was 1.045, less than 1.1, and the IQR ranges overlapped.
- A larger tile gave a higher rate at each point. The model order was the same:
  a larger tile needs fewer global loads.
- `torch.matmul` was 6.7 to 11.1 times as fast as the best tile.
- In the benchmark checks, each tile and `torch.matmul` had the same bits as
  the naive kernel at each point.
- The run caught 0 Python warnings. The capture began before the import of
  torch, so it covered every warning that torch could raise in the process.

The model gives the tiles 8 to 32 times fewer global loads than the naive
kernel, but they were not faster on this host. A probable cause is that the GPU
caches already served most of the repeated loads of the naive kernel. Then the
barriers and the reads of threadgroup memory can cost more time than the tiles
save. The lab reads no hardware counters, so this cause is an inference. The
next lab, Day 77, will apply a performance checklist to these kernels and
record the speedup of each change.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The report records the items in this list:

- the config and its digest, and the host environment without host names
- the two MPS switches and the load averages before and after the run
- the Python warnings of the run, with the home directory as `~`
- the Day 74 source digest, the tiled library text and its digest
- the digest, mode, pipeline limits and verdict of each tile, and the refusals
- the correctness cases, the raw samples, the medians, the IQRs and the checks
- the three numbers, the step ratios, the run totals and the evidence boundary

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk, and moves it into place with `os.replace`. It
refuses an output that aliases the config. The readers accept only regular
files that are not symbolic links, and they refuse duplicate JSON keys.

The verifier uses NumPy and its own code. It also uses the config parser and
the variant registry of the run. The parser checks the bounds, and the
registry gives the kernel text. The verifier calls no function that measures,
and it does not import torch. It does these checks:

1. It draws the operands again from the seed and calculates the reference
   again.
2. It replays the fma chain with a rounding to odd at each step. The run uses
   a different method, so the two methods check each other.
3. It rebuilds the digest of each output that records the bits of the replay,
   with the sentinel tail.
4. It derives the verdict of each tile again from the recorded limits.
5. It calculates each statistic, rate, check, number and step result again
   from the raw samples.

An output that records other bits cannot be rebuilt. For that output, the
verifier requires a different digest, a count of 1 or more differing elements
and an error within the bound. The timings cannot be observed again, so the
verifier checks them for structure only.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 76 | 0 to 2^32 - 1 |
| `naive.group_size` | [32, 8] | [x, y], with at most 1024 threads in total |
| `variants` | the three Day 76 tiles | 1 to 8 different registered variants |
| `correctness.shapes` | the 9 Day 74 shapes | 1 to 16 different [m, n, k] triples |
| `benchmark.shapes` | 1000^3, 1024^3, 2048^3, 4096^3 and 2000 x 3000 x 1000 | 1 to 16 different [m, n, k] triples |
| `benchmark.warmup` | 2 | 0 to 10 samples that are not timed |
| `benchmark.repeats` | 11 | 3 to 50 timed samples for each side |
| `benchmark.target_flops_per_sample` | 1073741824 | 2^20 to 2^40, with at most 4096 calls in a sample |
| `benchmark.max_runtime_seconds` | 300 | 1 to 600 |
| `comparison.min_faster_ratio` | 1.1 | more than 1.0 and at most 4.0 |

The config parser counts the planned work before a run. The pinned config has
9 cases, 160709684 replay steps and 36 correctness launches. It also has 5
sweep points, 275 timed samples, 325 calls and 11099809909760 FLOPs. These
counts include each named tile, so they are an upper limit when a tile gets a
refusal.

## Evidence boundary and exclusions

The evidence boundary of a measured report holds true or false fields and one
statement. The fields `apple_gpu_metal_this_host` and `host_specific` are true.
These fields are false: `cuda_kernel_executed`, `nvidia_gpu_used`,
`cublas_measured`, `vendor_kernel_known`, `vendor_peak_claimed`,
`distributed_claim`, `production_claim` and `portable_performance_claim`.

The statement gives these facts:

- The timings are wall-clock timings on the Apple GPU of the host that made
  the report, at the recorded load averages.
- The naive and tiled kernels came from `torch.mps.compile_shader`, and
  `torch.matmul` ran on the MPS device.
- PyTorch does not report which kernel `torch.matmul` ran. Thus the vendor
  number names the call, not a kernel.
- No CUDA kernel ran. The lab used no NVIDIA GPU and did not measure cuBLAS.
- Each ratio compares two sides of one run on this host.
- The timings do not measure or predict CUDA, NVIDIA, cuBLAS, distributed or
  production performance.

The lab reads no hardware counters. It does not record the GPU model, the clock
frequency, the thermal state or the other jobs on the host. The rates count the
FLOPs of the matmul, not the instructions of the GPU. The results come from one
Apple GPU on one host. They are not evidence of CUDA, NVIDIA or cuBLAS
performance, lecture viewing, completed exercises or learner understanding.

## Gap to the measured ceilings

The tiles were not faster than the naive kernel. This part of the lab compares
each side with the ceilings that the [gpu-ceilings lab](gpu-ceilings-lab.md)
measured on the same host. It also gives the Day 75 bottleneck name of each
side, from the rule of the [bottleneck lab](matmul-bottleneck-lab.md).

Each name is an inference from timings and measured ceilings, not from
hardware counters. The lab read no counter, so it did not observe a stall, a
cache hit or the occupancy of the GPU. Every Metal number in this part is a
host-specific Apple GPU (Metal) measurement.

### Gap commands

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
python -m ai_journey.matmul_tiled_gap_cli timing \
  --config config/day-76-matmul-tiled.json \
  --output /tmp/day-76-one-call-timing.json
python -m ai_journey.matmul_tiled_gap_cli verify-timing \
  --report /tmp/day-76-one-call-timing.json \
  --config config/day-76-matmul-tiled.json
python -m ai_journey.matmul_tiled_gap_cli gap \
  --tiled-report /tmp/day-76-matmul-tiled.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --timing-report /tmp/day-76-one-call-timing.json \
  --rule-config config/day-75-matmul-bottleneck.json \
  --output /tmp/day-76-matmul-tiled-gap.json
python -m ai_journey.matmul_tiled_gap_cli verify \
  --report /tmp/day-76-matmul-tiled-gap.json \
  --rule-config config/day-75-matmul-bottleneck.json \
  --tiled-report /tmp/day-76-matmul-tiled.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --timing-report /tmp/day-76-one-call-timing.json
```

The tiled report is the output of the `run` command above. The gap parser uses
the program name `ai-journey-matmul-tiled-gap`. This change adds no console
script, so run the module with `python -m`.

The `timing` command needs an Apple GPU, as the `run` command does. The `gap`
and `verify` commands run no kernel and do not import torch. With the three
reports and `--rule-config`, `verify` also builds the gap again and requires
the same bytes. Its summary line then says `sources_checked=true`. The `verify`
command takes the three report paths together or none of them.

The commands use the exit codes of the `run` and `verify` commands. Code 3 is
only for `timing`. A measured line has the prefix `apple_gpu_metal_measurement`.
The summary lines have the prefixes `matmul_tiled_one_call_measurement=` and
`matmul_tiled_gap_measurement=`. A synthetic report uses
`matmul_tiled_one_call_synthetic_timing=` or `matmul_tiled_gap_synthetic_timing=`
instead.

### One-call timing

The Day 75 rule needs three times for one call of each side:

- `median_gpu_ns`: the interval between two MPS events around the call.
- `median_host_ns`: the host time from an idle queue until the call is
  complete.
- `median_dispatch_ns`: the host time to issue the call, before the host
  waits.

The tiled report times batches between two synchronizations, so it does not
hold these times. The `timing` command measures them with the method of the
[gpu-timing lab](gpu-timing-lab.md). Each sample holds one launch and reads one
clock, and the order of the sides and the clocks shifts at each repeat.

The `timing` command uses the tiled config, so it times the same sides on the
operands of the tiled run. These steps occur:

1. Each side runs at a 1 x 1 x 1 floor. The host time of this call is an upper
   limit for a fixed cost that each call of the side pays.
2. Each side runs at each sweep shape. On each clock, it runs 2 samples that
   are not timed and 11 timed samples.
3. After each shape, each output must agree with the naive output within the
   bound of the benchmark. If an output does not agree, the run stops.

The verifier calculates each statistic again from the raw samples. It does
not import torch.

New `timing` reports have `schema_version` 2 and seal the version 2
event-domain text of the gpu-timing lab. The timing reports in this guide and
in the tests are version 1 records with the old "device-side" text. The
verifier accepts each version only with its own text.

### Inputs of the gap

The `gap` command reads three sealed reports. Each report must pass its own
verifier before the command uses a number from it:

- The tiled report gives the median sample time and the median rate of each
  side.
- The gpu-ceilings report gives the FMA ceiling and the bandwidth ceiling.
- The timing report gives the one-call medians and the fastest host time of
  each side.

The command refuses reports that do not pair:

- The three reports must name the same Python, NumPy, PyTorch and platform
  tokens.
- The tiled report and the timing report must come from one config. They must
  also have the same float32 matmul precision, MPS switches and admitted
  tiles.
- The rule config must use the `ideal_reuse` traffic model.

### Values for each side

For each sweep point and side, the gap report gives these values:

- The FMA share: the median rate divided by the measured FMA ceiling.
- The modeled traffic and intensity of the kernel. The naive kernel uses the
  no-reuse model, `4 * (2 * M * N * K + M * N)` bytes. Each tile uses
  `TiledMatmulTraffic` of the memory-access lab. The models count loads from
  device memory, and they model no cache.
- The attainable rate of `MeasuredRoofline` at the modeled intensity. This rate
  is the lower of the FMA ceiling and the bandwidth ceiling times the
  intensity. The modeled share is the median rate divided by this rate.
- The Day 75 name from `matmul_bottleneck_rule.classify`, with the thresholds
  of `config/day-75-matmul-bottleneck.json`.

PyTorch does not name the kernel of `torch.matmul`, so no traffic model
applies to it. The rule uses the ideal-reuse roof for each side. That model
moves each element one time, and no kernel moves fewer bytes. Thus its roof is
an upper limit for each side, also for `torch.matmul`.

A modeled share above 1 has two possible causes:

- The kernel moved fewer device bytes than the model counts. For example, a
  cache served repeated loads.
- The measured bandwidth ceiling is less than the true limit of the host.

### One gap run on the development host

The gap used the pinned tiled report of the run above. Two more runs held the
GPU lock on 2026-11-19 from 13:05:05 IST:

1. The gpu-ceilings command ran the pinned Day 74 config for 16 seconds.
2. The `timing` command ran the pinned tiled config for 88 seconds.

Then the `gap` command and the `verify` command with all inputs ran on the
CPU. Each command took about 4 seconds.

The `uptime` command printed 9.81, 10.52 and 13.03 before the ceiling run. It
printed 9.80, 10.49 and 12.97 after that run, and 6.95, 9.38 and 12.30 after
the timing run. The timing report recorded 10.14, 10.55 and 12.98 before its
run. The tiled run at 11:43 IST recorded 9.93, 11.62 and 14.06 before its run.

The FMA ceiling was 4836.8 GFLOP/s, and the bandwidth ceiling was 124.58 GB/s.
Thus the ridge point was 38.82 FLOPs for each byte. The ceiling report sets
`fma_scales_with_work` to true. Thus the fixed cost of one call was at most 5
percent of the time of the ceiling call. In this run, it was 0.20 percent.

| Report | SHA-256 |
| --- | --- |
| tiled | `371d65363ce5c53813c34a6edfc9db4725445bf4ffea605b7c81e0076fc03427` |
| gpu-ceilings | `8485a7be3370e07964941fa448ed1f93664d22fd48c2f5767a3b7478e4b88161` |
| one-call timing | `da32f7a857d98b4b4da645cd564189734eaf98a5d316619c52eb242fba688592` |
| gap | `be916ab18444905fafc32c28f2660be780b85eec89ed014dcd1d390a583bb652` |

The table gives the range of each value over the 5 sweep points:

| Side | FMA share | Modeled FLOPs for each byte | Modeled attainable GFLOP/s | Modeled share |
| --- | --- | --- | --- | --- |
| naive `matmul_x_columns` | 0.076 to 0.081 | 0.250 | 31.1 | 11.9 to 12.7 |
| `matmul_tiled_8` | 0.054 to 0.057 | 1.99 to 2.00 | 248.2 to 248.9 | 1.05 to 1.10 |
| `matmul_tiled_16` | 0.062 to 0.065 | 3.94 to 3.99 | 490.5 to 497.4 | 0.61 to 0.64 |
| `matmul_tiled_32` | 0.067 to 0.073 | 7.69 to 7.97 | 958.3 to 992.8 | 0.34 to 0.36 |
| `torch.matmul` | 0.447 to 0.815 | no model | no model | no model |

The rule gave these names. The number in parentheses is the FMA share:

| M x N x K | Naive | Tile 8 | Tile 16 | Tile 32 | `torch.matmul` |
| --- | --- | --- | --- | --- | --- |
| 1000 x 1000 x 1000 | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | indeterminate (step 1) |
| 1024 x 1024 x 1024 | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound (0.474) |
| 2048 x 2048 x 2048 | latency or occupancy-bound | indeterminate (step 1) | indeterminate (step 1) | indeterminate (step 1) | compute-bound (0.746) |
| 4096 x 4096 x 4096 | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | compute-bound (0.815) |
| 2000 x 3000 x 1000 | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | latency or occupancy-bound | compute-bound (0.713) |

These are the results of the gap run:

- Each kernel used at most 0.082 of the FMA ceiling. The top name of the naive
  kernel and of each tile was `latency or occupancy-bound`.
- That name is a residual. It says that no measured roof explains the rate. It
  does not prove that latency or occupancy is the cause.
- The naive kernel ran 11.9 to 12.7 times as fast as the roof of its no-reuse
  traffic. To move all of these bytes from device memory, it would need about
  12 times the measured bandwidth ceiling.
- Tile 8 ran at 1.05 to 1.10 of its modeled roof. Tiles 16 and 32 ran at 0.61
  to 0.64 and at 0.34 to 0.36 of their modeled roofs.
- `torch.matmul` used 0.713 to 0.815 of the FMA ceiling at the three largest
  shapes. The rule named these points `compute-bound`.
- At 2048 x 2048 x 2048, the steady time of each tile was 6.1 to 7.0 percent
  more than its one-call host time. Thus step 1 named these points
  `indeterminate`. The naive kernel was 4.7 percent more, within the tolerance
  of 5 percent.

Step 1 also named `torch.matmul` at 1000 x 1000 x 1000 `indeterminate`. Its
steady time was 11.6 percent more than its one-call host time. The steady
times come from the tiled run at 11:43 IST, and the host times come from the
timing run at 13:05 IST. The loads were different, so the two times can differ
by more than the tolerance.

These results agree with the probable cause in the run above. A cache probably
served most of the repeated loads of the naive kernel. Tile 8 ran near the roof
of its modeled traffic, so the device bandwidth can limit it. Tiles 16 and 32
ran at no more than 0.64 of their modeled roofs and below 0.08 of the FMA
ceiling. A cost that the labs did not measure probably limits them, for example
the barriers or the reads of threadgroup memory. Only hardware counters can
show which cost it is.

The repository does not keep these four reports. The tests keep one short
timing report from 2026-11-19 at 13:04 IST, with digest
`ec4525e7fd121ee74774e2dda992a8b9f48ab5f8dccd86d887a338af496750d6`. It times
the recorded tiled config of the tests. With the recorded tiled and
gpu-ceilings reports, it makes one measured gap with digest
`5d70138982e33407fe0f58437834091dce3e1b7b17aa3759544af9af18cbf7ca`. A host
without torch can build this gap again and check it.

### Evidence boundary of the gap

A measured gap report has these true fields: `apple_gpu_metal_this_host`,
`host_specific` and `inference_from_timings`. These fields are false:
`hardware_counters_used`, `nsight_used`, `instruments_used`,
`vendor_peak_used`, `traffic_measured`, `cuda_kernel_executed`,
`nvidia_gpu_used`, `cublas_measured`, `distributed_claim`, `production_claim`
and `portable_performance_claim`.

The gap has these limits:

- The three reports come from separate runs, at different loads. A name can
  change when the load changes.
- The reports record no host name and no GPU model. Thus the `gap` command
  cannot show that one host made all three. It compares only the software and
  platform tokens.
- The timing report has no record of Python warnings, as in the gpu-timing
  lab. Only the tiled report records the warnings of its run.
- An MPS event interval holds the signal latency of its events. Thus
  `median_gpu_ns` is longer than the kernel.
- The traffic comes from models that count no cache. The lab did not measure
  bytes.
- A measured ceiling is a rate that the kernels of the ceiling lab reached. It
  is not a vendor peak and not a hardware limit.
- No CUDA kernel ran, the lab used no NVIDIA GPU, and it did not measure
  cuBLAS. The values do not measure or predict CUDA, NVIDIA, cuBLAS,
  distributed or production performance.

## Sources

- GPU MODE lectures, https://github.com/gpu-mode/lectures: Lecture 5, "Going
  Further with CUDA for Python Programmers", and Lecture 16, "On Hands
  Profiling".
- NVIDIA, CUDA C++ Best Practices Guide v13.5, section 10.2.3.2, "Shared Memory
  in Matrix Multiplication (C=AB)", for the tiled kernel and its loads.
- Apple, Metal Shading Language Specification, for the `threadgroup` address
  space, `threadgroup_barrier` and `fma`:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader`, and
  the MPS environment variables page of the PyTorch documentation.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 3.1, for the bound `gamma_K`.
- S. Boldo and G. Melquiond, "Emulation of FMA and correctly rounded sums:
  proved algorithms using rounding to odd", IEEE Transactions on Computers
  57(4), 2008, for the rounding to odd.
