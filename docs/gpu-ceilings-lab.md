# Measured GPU ceilings on the Apple GPU

The Day 74 lab asks how far below peak a kernel runs. This lab supplies that
peak for an Apple GPU, and it measures the peak on the host that runs the
lab. Day 74 also covers shared memory and bank conflicts, so the lab times
threadgroup memory reads too. In this lab, "peak" means a ceiling measured on this host.
It is never a vendor number. Metal kernels that `torch.mps.compile_shader`
compiles at run time measure three ceilings:

- the float32 FMA ceiling, in GFLOP/s
- the bandwidth ceiling of a streaming copy and a streaming read-reduce, in GB/s
- the cost of threadgroup memory reads at strides 1 to 32, with and without
  one word of padding in each row

Each number is a host-specific Apple GPU (Metal) measurement, and each run
records the load averages of the host. Nothing here runs a CUDA kernel or
uses an NVIDIA GPU.

## Commands

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
python -m ai_journey.gpu_ceiling_cli verify \
  --report /tmp/day-74-gpu-ceilings.json \
  --config config/day-74-gpu-ceilings.json
```

The argument parser uses the program name `ai-journey-gpu-ceilings`. The
package does not install a command with that name, so use `python -m`.

`run` compiles the kernels, checks them, times them, checks its own report,
and writes the report. It prints these lines:

- one line with the load averages before and after the run
- one line for the correctness cases
- one line for each FMA point, stream point and bank point
- one line for each ceiling, the ridge point and the bank finding
- one summary line with the report digest

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`, and the summary line has the prefix
`gpu_ceiling_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `gpu_ceiling_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the same
summary line with the prefix `verified_`. With `--config`, `verify` also
requires that the report comes from that exact config.

The exit codes are the codes of the Day 73 Metal lab:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. PyTorch reports an MPS device without enough memory as a
  device error.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured anything.

On the development host, one run of the pinned config took about 15 seconds,
with the torch import. It wrote a report of about 82 KB.

## Requirements and hosts without MPS

The `run` command needs an Apple GPU on macOS. It also needs a PyTorch build
with an MPS device and `torch.mps.compile_shader`. The development host used
PyTorch 2.15.0. The lab uses the probe of the Day 73 Metal lab. The probe
reads the operating system first, and it imports torch only on macOS. It
names the first requirement that is missing:

- `unsupported_operating_system`: the host does not run macOS.
- `torch_not_importable`: Python cannot import torch.
- `pytorch_without_mps`: the PyTorch build has no MPS backend.
- `mps_unavailable`: PyTorch has an MPS backend, but no MPS device is ready.
- `compile_shader_unavailable`: PyTorch has no `torch.mps.compile_shader`.

When a requirement is missing, `run` writes the reason to standard error and
exits with code 3. The `verify` command and the report reader never import
torch, so they work on any host with NumPy. The tests that need the GPU skip
on other hosts and report the reason. A recorded report from the
development host lets the other tests check a real measurement on any host.

## Kernels

The Metal source is in `gpu_ceiling_source.py`. Each kernel gets its thread
index from `thread_position_in_grid`, as in the Day 73 lab. The scalars `n`,
`trips`, `threads` and `stride` are 32-bit `constant uint&` values. The
runtime passes them with `arg_casts`, as the Day 73 lab does.

### Compute ceiling

Each `fma_chains_<c>` kernel runs `c` independent float32 chains in
registers, for `c` in 1, 2, 4, 8, 16 and 32. Each loop trip applies
`x = fma(x, a, b)` eight times to each chain. The loop reads no memory and
writes no memory. After the loop, each thread writes the XOR of the bit
patterns of its chains. This one output depends on each chain, so the
compiler cannot remove a chain. Each FMA counts as 2 FLOPs.

The constants let the CPU compute each output exactly:

- The initial values of the chains of thread `i` come from row `i % 32` of a
  seeded 32 x 32 table. Each initial value is `m * 2^-23` for an integer `m`,
  so it is in [1, 2).
- The multiplier is `a = (2^24 - 4173) * 2^-24`, and the addend is
  `b = 12561923 * 2^-35`. Both are exact float32 values with 24 significant
  bits.
- For each `x` in [1, 2), the exact value of `x * a + b` is a multiple of
  2^-47 below 4. Float64 holds such a value exactly. Thus one rounding to
  float32 gives the fused result.
- Each step moves `x` toward `b / (1 - a)`, about 1.47, so `x` stays in
  [1, 2) at each step.

Rounding stops a chain when the distance from `x` to `b / (1 - a)` is less
than 1/4173. A step then changes `x` by less than half a unit in the last
place, so the rounded result is `x` again. A chain with the farthest initial
value stops after about 31000 steps. Thus each step count is 16384 or less.
At 16384 steps, about 97% of the chains still move. A kernel that skips a
trip then gives a different output, and the check stops the run.

If a kernel rounds the product before the add, the result changes in about
one step in eight. A later step seldom removes such a change, so most chains
change within 256 steps. Thus a kernel that rounds each product before the
add fails the check.

### Bandwidth ceiling

`stream_copy` copies a buffer of float4 values. `stream_read_reduce` reads
the same kind of buffer and writes one float32 partial sum for each thread.
Both kernels walk the buffer with a grid stride: thread `t` reads the values
`t`, `t + threads`, `t + 2 * threads` and so on. Thus each SIMD group reads
32 consecutive float4 values at each step. Each load is a float4. The copy
also stores float4 values, and the read-reduce stores one float32 for each
thread. The read-reduce data are whole numbers below 1024, so each float32
sum is exact in any order.

### Threadgroup-memory banks

`bank_rows_32` and `bank_rows_33` read a tile in threadgroup memory. Lane `l`
of each SIMD group reads the logical word `l * stride` of a tile with 32
columns. In `bank_rows_32`, a row holds 32 words. In `bank_rows_33`, a row
holds 33 words, so each row has one word of padding. Each loop trip adds the
same number of rows to the address of each lane and then reads 16 rows. Thus
each load has the same bank pattern across the lanes. The addresses change
from trip to trip, so the compiler must issue each load in each trip.

The two kernels run the same instructions. Only the row length, an immediate
value, is different. The tile has 112 rows, so the deepest read stays in the
tile. The bank pattern needs SIMD groups of 32 lanes, so the runtime refuses
a bank pipeline that reports another width.

The Day 74 memory-access lab models CUDA shared memory as 32 banks of 4-byte
words. In that model, the unpadded stride `s` gives an `s`-way conflict for
each power of two `s` up to 32. The padded layout has no conflicts at these
strides. The report shows the model degree beside each measured time. The
model describes CUDA hardware. It is not a fact about Apple GPUs.

## Correctness

Before it times anything, the run checks each kernel on a small launch:

1. The last threadgroup of the launch has threads past the data. The FMA and
   bank launches have 1000 threads. The copy and read-reduce walks have 1025
   and 257 threads.
2. Before the launch, the output buffer holds quiet-NaN sentinels. A tail of
   the buffer is outside the data.
3. After the launch, the tail must still hold its sentinels.
4. The rest of the buffer must match the CPU rebuild bit for bit.

The CPU rebuilds are exact. An FMA output is the XOR of the exact fused
chains. A copy equals its source. A read-reduce output is the exact integer
sum of each walk. A bank output is the wrapping uint32 sum of the tile words
that the lane read. A case that fails stops the run, so a report never holds
a failed check.

The timed points get a check too. After the last sample of a point, its
latest output must match its CPU rebuild. Thus each rate counts work that the
kernel did.

## Timing

Each timed sample has these steps:

1. Call `torch.mps.synchronize()` and read the clock.
2. Launch the calls of the sample.
3. Call `torch.mps.synchronize()` again and read the clock again.

Each point first runs `warmup` samples that are not timed. The timed samples
then rotate over all points of a sweep, and the order reverses at each
repeat. Thus slow changes in clocks, heat or load spread over all points.

An FMA sample and a bank sample each hold one call. A stream sample holds the
fewest whole calls whose bytes reach `target_bytes_per_sample`, with at most
4096 calls. Only `time.perf_counter_ns` with the real `CeilingRuntime` makes a
measurement. Such a report has the label
`host-specific Apple GPU (Metal) measurement` and the scope
`apple_gpu_metal_this_host`. A report from a stand-in backend or an injected
clock has `measured: false` and the scope `synthetic_backend_or_clock`.

## Ceilings and the bank finding

The lab gets each ceiling from the recorded medians:

- The FMA ceiling is the highest median GFLOP/s of all FMA points.
- The bandwidth ceiling is the higher median GB/s of the copy and the
  read-reduce at the largest buffer. A copy reads and writes each byte of its
  buffer. A read-reduce reads each byte once and writes 4 bytes for each
  thread.
- The ridge point is the FMA ceiling divided by the bandwidth ceiling, in
  FLOPs per byte.

The FMA ceiling is the best median rate of all FMA points in one run. The
bandwidth ceiling is the best median rate at the largest buffer, so a smaller
buffer can give a higher rate. Runs 1 and 2 below show this. Neither ceiling
is a vendor peak or a hardware limit, and a faster kernel can exist.

A ceiling is valid only if the fixed cost of a call is small. For each chain
count, the lab fits the line `time = fixed + flops / rate` to its points by
least squares, in exact rational arithmetic. The FMA ceiling
`scales_with_work` when the fixed time of its chain count is at most 5% of
the median time of the ceiling call.

The bank finding compares each time per access with the unpadded time at
stride 1. A time per access is the median call time divided by the loads of
all threads. It is a throughput figure for the whole GPU, not the latency of
one load. A ratio of 1.25 or more counts as a slowdown. The finding has one
of these values:

- `no_clear_conflict_pattern`: no ratio reaches 1.25.
- `stride_slowdown_removed_by_padding`: the unpadded ratio rises at each
  stride step and reaches 1.25, and each padded ratio stays below 1.25.
- `stride_slowdown_reduced_by_padding`: the unpadded ratio rises at each
  stride step and reaches 1.25. Padding makes the largest stride at least
  1.25 times faster, but a padded ratio also reaches 1.25.
- `no_single_pattern`: any other result.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk, and moves it into place with `os.replace`. It
refuses an output that aliases the config through a path, a symbolic link or
a hard link. The readers accept only regular files that are not symbolic
links. The limit is 64 KiB for a config and 1 MiB for a report. The readers
refuse duplicate JSON keys, `NaN`, `Infinity`, unknown fields and missing
fields.

The verifier uses only NumPy and its own code. It does not call the modules
that made the report. It shares the config parser, which only checks bounds.
It rebuilds the correctness facts on the CPU with other methods than the run:

1. It computes the FMA chains in integers. The exact sum `x * a + b` is an
   integer multiple of 2^-47, and the verifier rounds it half to even at 24
   bits.
2. It draws the copy and read-reduce sources again from the seed and sums the
   partials with a weighted count.
3. It sums the bank tile words with Python integers.
4. It adds the sentinel tail to each rebuilt buffer and requires each buffer
   to match its stored digest.

The run uses float64 for the FMA chains, so the run and the verifier check
each other. The verifier cannot observe a timing again, so it checks the
timings for structure only. It recomputes the statistics, rates, batch sizes,
fits, ceilings, ratios, totals and summary from the recorded samples. It also
checks the launch geometry, the planned work, the shader source, the pipeline
limits, the load averages and the labels.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 74 | 0 to 2^32 - 1 |
| `correctness.threads` | 1000 | 1 to 65536, not a multiple of a group size |
| `correctness.fma_steps` | 256 | a multiple of 8, up to 16384 |
| `correctness.stream_float4` | 4099 | 1 to 2^20 float4 values, with copy and read-reduce threads past the data |
| `correctness.bank_trips` | 9 | 1 to 65536 |
| `fma.group_size` | 256 | 1 to 1024 |
| `fma.threads` | 1048576 | 1 to 2^24 |
| `fma.chain_counts` | 1, 2, 4, 8, 16, 32 | increasing values from that list |
| `fma.fma_steps` | 32 to 8192 | 3 to 8 increasing multiples of 8, up to 16384 |
| `bandwidth.group_size` | 256 | 1 to 1024 |
| `bandwidth.buffer_bytes` | 16, 64 and 256 MiB | 1 to 8 increasing multiples of 16 bytes, up to 1 GiB |
| `bandwidth.copy_float4_per_thread` | 4 | 1 to 1024 |
| `bandwidth.reduce_float4_per_thread` | 16 | 1 to 1024 |
| `bandwidth.target_bytes_per_sample` | 1073741824 | 1 KiB to 1 GiB, with at most 4096 calls in a sample |
| `bank.group_size` | 256 | a multiple of 32, up to 1024 |
| `bank.threads` | 1048576 | whole threadgroups, up to 2^24 |
| `bank.strides` | 1, 2, 4, 8, 16, 32 | 2 to 32 increasing strides from 1 to 32, the first of them 1 |
| `bank.trips` | 256 | 1 to 65536 |
| `timing.warmup` | 2 | 0 to 10 samples that are not timed |
| `timing.repeats` | 11 | 3 to 50 timed samples for each point |
| `timing.max_runtime_seconds` | 300 | 1 to 600 |

The config parser counts the planned work before a run, with the batch sizes
of the run. For the pinned config, the counts are 20 correctness cases, 30
FMA points, 6 stream points and 12 bank points. The run has 528 timed samples
and 2184 kernel calls. The work is 18742096429056 FLOPs, 84406173696 stream
bytes and 670014898176 threadgroup loads. A run can have at most 2^22 calls,
2^46 FLOPs, 2^38 stream bytes and 2^40 loads. The time budget covers the checks and the
sweeps, but not the torch import or the kernel compile.

## Three runs on the development host

These runs used the pinned config on 2026-11-17, from 11:29 to 11:32 IST. The
host was a shared Darwin arm64 host with an Apple GPU, macOS 27.2, CPython
3.11.14, NumPy 1.26.4 and PyTorch 2.15.0. Other jobs ran on the host at the
same time, and the load averages show it. All values are host-specific Apple
GPU (Metal) measurements from one host, not portable figures. To get figures
for your host, run the command on it.

| Run | Load averages before (1, 5, 15 min) | Load averages after | FMA ceiling | Bandwidth ceiling | Ridge point |
| --- | --- | --- | --- | --- | --- |
| 1 | 10.66, 9.62, 8.81 | 9.95, 9.50, 8.78 | 4618 GFLOP/s | 100.7 GB/s | 45.8 FLOPs per byte |
| 2 | 7.98, 9.09, 8.71 | 7.73, 9.01, 8.68 | 4632 GFLOP/s | 100.8 GB/s | 45.9 FLOPs per byte |
| 3 | 7.83, 9.01, 8.68 | 8.80, 9.14, 8.73 | 4637 GFLOP/s | 98.9 GB/s | 46.9 FLOPs per byte |

### Compute

In each run, the FMA ceiling came from 32 chains at 2048 steps per chain. The
best single samples of that point gave 4674 to 4694 GFLOP/s. The table shows
the median GFLOP/s of run 1 for each chain count and step count:

| Chains | 32 steps | 128 steps | 512 steps | 2048 steps | 8192 steps |
| --- | --- | --- | --- | --- | --- |
| 1 | 348 | 1133 | 2294 | 3492 | 3932 |
| 2 | 487 | 1706 | 3202 | 3970 | 4274 |
| 4 | 1041 | 2570 | 3761 | 4158 | 4503 |
| 8 | 1113 | 3175 | 4230 | 4467 | 4548 |
| 16 | 2450 | 3922 | 4381 | 4551 | 4571 |
| 32 | 2891 | 4018 | 4543 | 4618 | 4550 |

The rate rose with the work of a call and with the number of chains:

- Each step count has 4 times the work of the step count before it. At 32
  chains, the median time grew by 2.88, 3.54, 3.93 and 4.06 times in run 1.
  At the two largest points, the time grew about as fast as the work.
- At 1 chain, the same steps grew the time by only 1.23, 1.98, 2.63 and 3.55
  times. A fixed cost of each call set most of the time of the small points.
- The fits gave a fixed cost of 67 to 244 µs for each call, in the three runs
  and the six chain counts. At the ceiling point, this cost was 0.22% to
  0.47% of the call time. Thus `scales_with_work` was true in each run.
- At 8192 steps, 1 chain gave 3932 GFLOP/s, which is 85% of the ceiling. With
  2^20 threads, many SIMD groups probably hide part of the latency of one
  chain.
- From 8 chains, the medians at 2048 and 8192 steps were 4467 to 4618 GFLOP/s
  in run 1. Their spread is less than 4%.

### Bandwidth

| Kernel | 16 MiB | 64 MiB | 256 MiB |
| --- | --- | --- | --- |
| `stream_copy`, run 1 | 102.3 GB/s | 101.1 GB/s | 100.7 GB/s |
| `stream_read_reduce`, run 1 | 96.5 GB/s | 97.4 GB/s | 98.0 GB/s |

In each run, the copy gave the bandwidth ceiling at 256 MiB: 98.9 to 100.8
GB/s. The read-reduce gave 96.0 to 99.2 GB/s at the same size. Across the
three runs, all 18 stream medians were from 89.7 to 102.7 GB/s, so the rate
changed little from 16 MiB to 256 MiB. The best single copy samples at 256
MiB gave 102.1 to 113.7 GB/s. For comparison, the Day 73 Metal lab reports
92 to 116 GB/s for its vector add at 16777216 elements.

### Threadgroup memory

The table shows run 1. A ratio compares a time per access with the unpadded
time at stride 1. The model degree comes from the 32-bank model of the
Day 74 memory-access lab.

| Stride | Unpadded ps | Unpadded ratio | Model degree | Padded ps | Padded ratio | Model degree | Padding speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.30 | 1.00 | 1 | 2.07 | 1.60 | 1 | 0.62 |
| 2 | 2.81 | 2.17 | 2 | 2.87 | 2.21 | 1 | 0.98 |
| 4 | 4.85 | 3.74 | 4 | 4.92 | 3.80 | 1 | 0.99 |
| 8 | 7.53 | 5.81 | 8 | 5.32 | 4.11 | 1 | 1.41 |
| 16 | 14.05 | 10.85 | 16 | 6.41 | 4.95 | 1 | 2.19 |
| 32 | 27.83 | 21.49 | 32 | 6.39 | 4.93 | 1 | 4.35 |

The data show these results about this Apple GPU:

- Without padding, the time per access rose at each stride step in each run.
  At stride 32, it was 20.2 to 21.5 times the stride-1 time.
- At stride 2, the unpadded ratio was 2.01 to 2.17, near the model degree
  of 2.
- From stride 4, each unpadded ratio was smaller than the model degree. The
  kernel also does adds and address work for each load, and that work does
  not depend on the stride. That work can explain part of the difference.
- At strides 8, 16 and 32, padding made the reads 1.37 to 4.35 times faster
  in the three runs.
- At strides 2 and 4, padding did not help: the speedup was 0.96 to 1.02. The
  model predicts no conflicts for the padded layout at these strides, so this
  host does not match the model there.
- At stride 1, the padded layout was 1.55 to 1.60 times slower than the
  unpadded layout. Both layouts read 32 consecutive words in each load at
  stride 1. In the unpadded layout, the first word of each load is at a
  multiple of 128 bytes in the tile. In the padded layout, this is true for
  only one load in 32.
- In each run, the finding was `stride_slowdown_reduced_by_padding`.

Thus the stride has a strong effect on threadgroup memory reads on this
Apple GPU. One word of padding reduces the effect at large strides. The
data do not match the model of 32 banks of 4-byte words in detail. This lab
does not identify the bank structure of this GPU.

## How far below the measured ceiling

A later lab can compare a kernel with these ceilings instead of a vendor
number. When the verifier accepts a report, the `summary` field of the
report holds the two ceilings in `fma_ceiling_gflops_per_second` and
`bandwidth_ceiling_gb_per_second`. For a kernel that does `I` FLOPs for each
byte it moves, the attainable rate is the lower of the two values below:

- the FMA ceiling
- the bandwidth ceiling times `I`

With the ceilings of run 1, the attainable rate at 8 FLOPs per byte is 806
GFLOP/s. At 46 FLOPs per byte or more, the attainable rate is 4618 GFLOP/s.
The fraction of the ceiling is the achieved rate divided by the attainable
rate. A fraction from these ceilings describes this host only.

The module `gpu_ceiling_roofline.py` does this comparison in Python:

- `load_roofline` reads a report through the verified loader, and
  `roofline_from_report` verifies a report that is already in memory.
- `attainable_gflops_per_second` gives the lower ceiling at an intensity, and
  `bound` names the ceiling that limits the kernel.
- `below_ceiling` gives the achieved rate, the attainable rate, the fraction,
  the bound, and the digest and label of the ceiling report.

A tampered report gives no roofline, because the verifier rejects it first. A
synthetic report gives a roofline with `measured: false`. A fraction can be
more than 1, because a ceiling is a rate that the kernels of this lab
reached, not a hardware limit.

## Evidence boundary and exclusions

The boundary of a measured report states: "These are wall-clock timings on
the Apple GPU of the host that made this report. The kernels came from
torch.mps.compile_shader. Each ceiling is the best median rate that these
kernels reached in this run. It is not a vendor peak and not a hardware
limit. The timings are host-specific Apple GPU (Metal) measurements. No CUDA
kernel ran, and no NVIDIA GPU was used. Bandwidth counts the bytes that each
kernel must read and write, not the hardware traffic. These timings do not
measure or predict CUDA, NVIDIA, distributed or production performance."

The lab reads no hardware counters. It does not record the GPU model, the
clock frequency, the thermal state or the other jobs on the host. The
bandwidths count compulsory bytes, not the hardware traffic. The report is
evidence from one Apple GPU on one host. It is not evidence of CUDA or NVIDIA
performance, lecture viewing, completed exercises or learner understanding.

## Sources

- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader` in
  `torch/mps/__init__.py`.
- Apple, Metal Shading Language Specification: the `fma` function, the
  `threadgroup` address space and the `thread_index_in_threadgroup`
  attribute: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- `docs/memory-access-model.md`, for the 32-bank model of CUDA shared memory
  and its sources in the CUDA guides.
- `docs/kernel-metal-lab.md`, for the probe, the exit codes and the timing
  method that this lab reuses.
