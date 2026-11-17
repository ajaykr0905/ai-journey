# Metal launch-shape sweep on the Apple GPU

Day 74 of the curriculum is about occupancy. This lab is the launch-shape
half of the Day 74 lab, for a Mac with an Apple GPU. It sweeps the
threadgroup shapes of a naive float32 matmul and the group sizes of a vector
add. It checks every shape, times every shape and ranks the shapes.

Every timing is a measurement on the Apple GPU of one host. Nothing here runs
a CUDA kernel or uses an NVIDIA GPU. The lab quotes no vendor peak. A best
rate is the highest median rate that one run measured on this host.

The lab reuses the Day 73 Metal lab. It reuses the probe, the runtime, the
`vector_add` kernel, the launch geometry, the vector-add check and the timing
helpers. It also uses the Day 73 launch simulator and the shared JSON, read
and write helpers.

## Commands

```bash
python -m ai_journey.metal_launch_sweep_cli run \
  --config config/day-74-metal-launch-sweep.json \
  --output /tmp/day-74-metal-launch-sweep.json
python -m ai_journey.metal_launch_sweep_cli verify \
  --report /tmp/day-74-metal-launch-sweep.json \
  --config config/day-74-metal-launch-sweep.json
```

The parser names itself `ai-journey-metal-launch-sweep`. The package does not
install that command yet.

The `run` command compiles the kernels, checks them, times them, checks its
own report and writes the report. It prints one line for each pipeline, each
shape, each correctness sweep and each timed shape. It also prints a ranking
line for each sweep point, a line with the load averages and a summary line
with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`metal_launch_sweep_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `metal_launch_sweep_synthetic_timing=` instead. The
`verify` command recomputes a report on the CPU and prints the same summary
line with the prefix `verified_`.

The exit codes are those of the Day 73 Metal lab:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured anything.

## Requirements and hosts without MPS

The `run` command needs an Apple GPU on macOS. It also needs a PyTorch build
with an MPS device and `torch.mps.compile_shader`. The development host used
PyTorch 2.15.0. Before `run` measures anything, the Day 73 probe reads the
operating system and names the first requirement that is missing:

- `unsupported_operating_system`: the host does not run macOS.
- `torch_not_importable`: Python cannot import torch.
- `pytorch_without_mps`: the PyTorch build has no MPS backend.
- `mps_unavailable`: PyTorch has an MPS backend, but no MPS device is ready.
- `compile_shader_unavailable`: PyTorch has no `torch.mps.compile_shader`.

When a requirement is missing, `run` writes the reason to standard error and
exits with code 3. The `verify` command and the report reader never import
torch, so they work on any host with NumPy. The tests that need the GPU skip
on other hosts and report the reason.

## Pipeline limits and the launch-shape rules

The API shows threadgroup limits only through each compiled pipeline. The
lab reads `max_threads_per_threadgroup` and `thread_execution_width` from
three pipelines: `matmul_naive`, the Day 73 `vector_add` and
`simd_layout_probe`. On the development host, each pipeline gave 1024 threads
and a width of 32.

The lab checks each shape against the pipeline of its kernel. It applies two
rules in this order:

1. The threads of one threadgroup must not exceed
   `max_threads_per_threadgroup`. Otherwise the reason is
   `exceeds_max_threads_per_threadgroup`.
2. Where the runtime requires whole SIMD groups, the threads must be a whole
   multiple of `thread_execution_width`. Otherwise the reason is
   `not_whole_simd_groups`.

The lab records a rejected shape with its reason and never launches it. In
the pinned run, the pipelines admitted all 16 shapes. The tests use stand-in
limits to show the rejected path from the run to the verifier.

A Metal pipeline requires whole SIMD groups only when its descriptor sets
`threadGroupSizeIsMultipleOfThreadExecutionWidth`. The PyTorch 2.15.0
binaries hold one selector that makes a compute pipeline:
`newComputePipelineStateWithFunction:error:`. That selector takes no
descriptor, and the binaries hold no
`threadGroupSizeIsMultipleOfThreadExecutionWidth` selector. Thus this runtime
does not apply rule 2, and the report records `requires_whole_simd_groups` as
false. On the development host, groups of 8, 16 and 48 threads ran and gave
exact outputs.

The grid always holds whole threadgroups, as in the Day 73 lab. The group
count on each axis is `ceil(extent / group_size)`. The threads past the data
are padding threads, and the bounds guard of each kernel stops them.

## Kernels

The source of `matmul_naive` is in `metal_launch_sweep_source.py`. It
computes `C = A B` in float32, with one thread for each element of `C`. Each
thread gets its index from `thread_position_in_grid`, which is equal to the
CUDA index `blockIdx * blockDim + threadIdx`. The x axis covers the `n`
columns, and the y axis covers the `m` rows. Thus adjacent x threads read
adjacent elements of one row of `B` and the same element of `A`.

The kernel uses no threadgroup memory and no SIMD-group functions. Thus the
shape cannot change the arithmetic of a thread, and every shape must give the
same `C` bit for bit. The 1D sweep runs the Day 73 `vector_add` kernel. It
reads 8 bytes and writes 4 bytes for each element.

The two kernels do different amounts of work for each byte. At 2048 x 2048
x 2048, the matmul does about 341 FLOPs for each compulsory byte. The vector
add does 1 FLOP for each 12 compulsory bytes. The lab does not measure which
unit of the GPU limits each kernel.

## SIMD layout

The probe kernel runs as one threadgroup of each admitted shape. Each thread
writes its SIMD group, its lane, its x and y position and the SIMD width that
it sees. The run compares this record with the linear layout, word for word:

1. The linear index of a thread is `x + gx * y`, where `gx` is the x size of
   the threadgroup.
2. The SIMD group is the linear index divided by the width.
3. The lane is the linear index modulo the width.

On the development host, all 16 shapes matched the linear layout. A shape
with an x size below the width thus puts more than one output row into one
SIMD group. The probe shows which SIMD group and lane ran each thread. It
does not show when or where the GPU scheduled a SIMD group.

| Shape | Threads | SIMD groups | Rows per SIMD group | Columns per SIMD group |
| --- | --- | --- | --- | --- |
| 8x8 | 64 | 2 | 4 | 8 |
| 16x16 | 256 | 8 | 2 | 16 |
| 32x32 | 1024 | 32 | 1 | 32 |
| 32x8 | 256 | 8 | 1 | 32 |
| 8x32 | 256 | 8 | 4 | 8 |
| 64x4 | 256 | 8 | 1 | 32 |
| 4x64 | 256 | 8 | 8 | 4 |

The vector sizes 8 and 16 are below the width. Their only SIMD group has 24
and 16 idle lanes. The size 48 fills one SIMD group and leaves 16 idle lanes
in the second.

## Checks before the timing

A run checks the kernels before it times them:

1. The Day 73 launch simulator runs a per-thread copy of the matmul for each
   admitted shape. The case has 13 rows, 19 columns and an inner size of 3.
   Each element must have one writer, and no thread may read or write out of
   bounds. Both a canonical and a seeded thread order must give the exact
   integer product.
2. The probe must report the linear layout for each admitted shape.
3. Each matmul case runs once with each admitted shape. The guard tail of
   each output buffer must keep its sentinels. All shapes must give the same
   `C` bit for bit.
4. Each element of `C` must lie within `(gamma_K(2**-24) + gamma_K(2**-53)) *
   (|A| @ |B|)` of the float64 product. The bound holds for every summation
   order and for fused multiply-adds.
5. The Day 73 vector-add check runs each group size at 65537 elements. The
   Metal kernel, PyTorch on MPS and PyTorch on the CPU must match the float32
   sum bit for bit.

A check that fails stops the run, so a report never holds a failed check. In
the pinned run, the largest error of a correctness case was 0.2206 of its
bound. The largest error of a timed point was 0.0164 of its bound.

## Timing method

For each sweep point, the run places one set of inputs on the device and
binds one call for each admitted shape. Then it does these steps:

1. Each shape runs `warmup` samples that are not timed.
2. Each timed sample calls `torch.mps.synchronize()` and reads the clock.
3. The sample launches a batch of calls with one shape, calls
   `torch.mps.synchronize()` again and reads the clock again.
4. Repeat `r` runs each shape once. Shape `r` modulo the number of shapes
   runs first. Thus slow drift in the clock, the heat or the load does not
   favor the shapes that run first.
5. After the last sample, the outputs of all shapes must be equal. The
   matmul output must also meet the bound, and the vector output must equal
   the float32 sum.

A batch holds the fewest whole calls that reach the target of the sample.
The matmul target counts `2 m n k` FLOPs for each call. The vector target
counts 12 compulsory bytes for each element. The report keeps the raw
samples and gives the nearest-rank median and IQR of each shape. It also
gives the median time of each call, the rate and the rank.

The ranking orders the shapes by their median sample. The spread is the
slowest median divided by the fastest median. The report marks a point as
`separated` when the IQR of the fastest shape and the IQR of the slowest
shape do not overlap. Without that mark, the ranking of a point is inside the
noise of the run.

## One run on the development host

The run used the pinned config on 2026-11-17 at 11:18 IST, while it held the
GPU lock that the timed runs on this host share. The host was a shared Darwin
arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14, NumPy 1.26.4 and
PyTorch 2.15.0. Other jobs ran on the host at the same time, as the load
averages show.

The `uptime` command gave load averages of 6.58, 9.01 and 8.39 before the run
and 7.23, 9.03 and 8.41 after it. The run took 10.8 seconds after the
compile. It wrote a report of 56 KB with the digest
`0d5ea2c1065145ceef6c82cd5191cc71767b9496151984b21559055d391e73b7`. The tests
keep that report, and `verify` accepts it on any host.

The values come from one run on one host. They are not portable figures. All
rates are host-specific Apple GPU (Metal) measurements at these load
averages. To get figures for your host, run the command on it.

### Matmul at 256 x 256 x 256

Each sample held 32 calls. The spread was 1.150, and the IQRs of the fastest
and the slowest shape did not overlap.

| Shape | Threads | Rows per SIMD group | Median (µs) | IQR (µs) | GFLOP/s | Rank |
| --- | --- | --- | --- | --- | --- | --- |
| 8x8 | 64 | 4 | 110.26 | 1.88 | 304.3 | 2 |
| 16x16 | 256 | 2 | 111.91 | 2.28 | 299.8 | 3 |
| 32x32 | 1024 | 1 | 106.86 | 2.92 | 314.0 | 1 |
| 32x8 | 256 | 1 | 113.90 | 2.67 | 294.6 | 5 |
| 8x32 | 256 | 4 | 112.75 | 3.26 | 297.6 | 4 |
| 64x4 | 256 | 1 | 116.47 | 4.17 | 288.1 | 6 |
| 4x64 | 256 | 8 | 122.91 | 2.34 | 273.0 | 7 |

### Matmul at 1024 x 1024 x 1024

Each sample held 1 call. The spread was 1.041, and the IQRs of the fastest
and the slowest shape did not overlap.

| Shape | Threads | Rows per SIMD group | Median (µs) | IQR (µs) | GFLOP/s | Rank |
| --- | --- | --- | --- | --- | --- | --- |
| 8x8 | 64 | 4 | 5508.88 | 46.04 | 389.8 | 1 |
| 16x16 | 256 | 2 | 5550.12 | 31.00 | 386.9 | 2 |
| 32x32 | 1024 | 1 | 5732.54 | 62.96 | 374.6 | 7 |
| 32x8 | 256 | 1 | 5595.29 | 56.92 | 383.8 | 6 |
| 8x32 | 256 | 4 | 5563.46 | 55.58 | 386.0 | 3 |
| 64x4 | 256 | 1 | 5572.96 | 50.33 | 385.3 | 4 |
| 4x64 | 256 | 8 | 5591.38 | 195.54 | 384.1 | 5 |

### Matmul at 2048 x 2048 x 2048

Each sample held 1 call. The spread was 1.022, and the IQRs of the fastest
and the slowest shape did not overlap.

| Shape | Threads | Rows per SIMD group | Median (µs) | IQR (µs) | GFLOP/s | Rank |
| --- | --- | --- | --- | --- | --- | --- |
| 8x8 | 64 | 4 | 41679.04 | 320.08 | 412.2 | 1 |
| 16x16 | 256 | 2 | 41799.12 | 314.71 | 411.0 | 3 |
| 32x32 | 1024 | 1 | 42602.71 | 399.58 | 403.3 | 7 |
| 32x8 | 256 | 1 | 41976.58 | 392.38 | 409.3 | 5 |
| 8x32 | 256 | 4 | 41755.21 | 117.46 | 411.4 | 2 |
| 64x4 | 256 | 1 | 41809.25 | 115.46 | 410.9 | 4 |
| 4x64 | 256 | 8 | 42053.29 | 116.92 | 408.5 | 6 |

### Vector add at 1048576 elements

Each sample held 11 calls. The spread was 1.406, and the IQRs of the fastest
and the slowest size overlapped.

| Group size | Idle lanes | Median (µs) | IQR (µs) | GB/s | Rank |
| --- | --- | --- | --- | --- | --- |
| 8 | 24 | 152.55 | 56.95 | 82.48 | 9 |
| 16 | 16 | 128.35 | 48.11 | 98.03 | 8 |
| 32 | 0 | 124.58 | 44.16 | 101.01 | 7 |
| 48 | 16 | 118.43 | 33.06 | 106.25 | 6 |
| 64 | 0 | 108.48 | 35.10 | 116.00 | 1 |
| 128 | 0 | 114.66 | 42.08 | 109.74 | 3 |
| 256 | 0 | 118.01 | 20.64 | 106.62 | 5 |
| 512 | 0 | 110.04 | 30.83 | 114.35 | 2 |
| 1024 | 0 | 115.13 | 37.71 | 109.29 | 4 |

### Vector add at 16777216 elements

Each sample held 1 call. The spread was 1.036, and the IQRs of the fastest
and the slowest size overlapped.

| Group size | Idle lanes | Median (µs) | IQR (µs) | GB/s | Rank |
| --- | --- | --- | --- | --- | --- |
| 8 | 24 | 1788.04 | 133.21 | 112.60 | 3 |
| 16 | 16 | 1802.83 | 149.46 | 111.67 | 6 |
| 32 | 0 | 1802.08 | 131.00 | 111.72 | 5 |
| 48 | 16 | 1803.88 | 123.17 | 111.61 | 7 |
| 64 | 0 | 1794.04 | 79.46 | 112.22 | 4 |
| 128 | 0 | 1785.54 | 104.25 | 112.75 | 2 |
| 256 | 0 | 1775.29 | 101.67 | 113.40 | 1 |
| 512 | 0 | 1838.54 | 122.08 | 109.50 | 9 |
| 1024 | 0 | 1820.21 | 117.46 | 110.61 | 8 |

The best rates of this run were 412.2 GFLOP/s for the matmul and 116.00 GB/s
for the vector add. These are the highest medians that this run measured.
They are not vendor peaks.

## Shape, size and the execution width

This section relates the measured data to the execution width of 32. Each
statement before the hypotheses comes from the tables above.

The x size of a 2D shape sets the rows that one SIMD group computes. Shapes
with an x size of 32 or more put 1 row into each SIMD group. The shapes 16x16,
8x8, 8x32 and 4x64 put 2, 4, 4 and 8 rows into each SIMD group.

Five shapes have 256 threads. They have the same size, but a different split
over rows and columns. At 1024 and at 2048, their rates were within 0.8% and
0.7% of each other. At 256, the shape 4x64 with 8 rows had the lowest rate of
the five, 8.9% below 16x16. A shape with 1 row per SIMD group was never the
fastest of these five shapes.

The effect of the threadgroup size changed with the matrix size. The shape
32x32 has 1024 threads, the most that each pipeline allows. It was the
fastest shape at 256, but it was the slowest at 1024 and at 2048. There, it
was 4.1% and 2.2% slower than 8x8.

At 16777216 elements, this run found no effect of the group size outside the
noise. All nine sizes were within 3.6%, and their extreme IQRs overlapped. The
sizes 8 and 16 leave 24 and 16 of the 32 lanes idle, but they were within 1.6%
of the best size.

At 1048576 elements, the sizes 8, 16 and 32 gave the three lowest rates. Their
IQRs overlap the IQRs of the other sizes, so this run does not separate them.

The following statements are hypotheses. The lab reads no hardware counters.
It cannot observe how the GPU schedules threadgroups or SIMD groups. Thus the
data does not prove or disprove these statements.

- Hypothesis: the caches of the GPU serve the reads of `A` and `B` equally
  well for each split of rows. Thus the split does not limit this naive
  matmul at 1024 and 2048.
- Hypothesis: at 256, one call takes about 110 µs, and the grid holds 64 to
  1024 threadgroups. A fixed cost for each threadgroup would favor the 64
  large threadgroups of 32x32 at this size.
- Hypothesis: a threadgroup of 1024 threads must fit on a GPU core as a whole.
  This could leave fewer threads active on a core at 1024 and 2048 than the
  smaller shapes do.
- Hypothesis: the memory system limits the vector add at 16777216 elements,
  so idle lanes do not change its rate.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk and moves it into place with `os.replace`. It
refuses an output that aliases the config through a path, a symbolic link or
a hard link. The readers accept only regular files that are not symbolic
links, up to 64 KiB for a config and 1 MiB for a report.

The verifier uses NumPy, its own code and some helpers of the Day 73
verifier. It does not call the modules that made the report. It shares the
config parser, which only checks bounds. It does these steps:

1. It admits each shape again from the recorded limits and the SIMD rule,
   and it computes each SIMD footprint again.
2. It computes the mapping counts and the exact product of the simulated case
   again. It also computes the words that the probe must have returned.
3. It draws the matmul inputs again and checks their digests. It holds each
   recorded lattice value of `C` to its own float64 product and bound.
4. It draws the vector inputs again and computes each vector-add digest
   again, with the sentinel tail of each Metal buffer.
5. It computes the batch sizes, statistics, rates, ranks, spreads, totals and
   planned work again.

The GPU rounding sets each matmul output, so the verifier cannot compute the
digest of `C` again. A lattice of up to 25 values of `C` takes its place. The
verifier cannot observe a timing again, so it checks the timings for
structure only. Each section must match the rebuilt section as canonical
JSON, so `1`, `1.0` and `true` stay different.

A measured report must hold the load averages from before and after its run.
If the host gives no load averages, `run` stops with code 2 before the first
launch.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 74 | 0 to 2^32 - 1 |
| `matmul.group_shapes` | 7 shapes, 8x8 to 4x64 | 1 to 16 different [x, y] pairs, each 2 or more, with at most 1024 threads in total |
| `matmul.correctness_cases` | [1, 1, 1], [37, 53, 29] and [64, 64, 16] | 1 to 8 different [m, n, k] triples of 1 to 512 |
| `matmul.sizes` | [256, 256, 256], [1024, 1024, 1024] and [2048, 2048, 2048] | 1 to 4 triples of 1 to 4096, each with more FLOPs than the one before |
| `matmul.target_flops_per_sample` | 1073741824 | 2^20 to 2^40, with at most 4096 calls in a sample |
| `vector.group_sizes` | 8, 16, 32, 48, 64, 128, 256, 512 and 1024 | 1 to 16 increasing sizes of 2 to 1024 |
| `vector.correctness_length` | 65537 | 1 to 2^25, not a multiple of a group size |
| `vector.lengths` | 1048576 and 16777216 | 1 to 4 increasing lengths of 1 to 2^25 |
| `vector.target_bytes_per_sample` | 134217728 | 1 KiB to 1 GiB, with at most 4096 calls in a sample |
| `timing.warmup` | 2 | 0 to 10 samples that are not timed |
| `timing.repeats` | 21 | 3 to 50 timed samples for each shape |
| `timing.max_runtime_seconds` | 300 | 1 to 600 |

The correctness cases must leave padding threads on each axis of each shape.
A shape with 1 thread on an axis has no padding on that axis, so the parser
refuses it. The repeat count 21 is a multiple of the 7 matmul shapes, so each
shape takes each turn 3 times.

The config parser counts the planned work before a run, as if each pipeline
admits each shape. For the pinned config, the checks are 7 mapping checks, 16
layout probes and 30 correctness launches. The sweep has 5 points, 819 timed
samples, 7958 kernel calls, 3284576239616 matmul FLOPs and 70325895168 vector
bytes. A run can have at most 2^22 calls, 2^44 matmul FLOPs and 2^38 vector
bytes. The time budget covers the checks and the sweep, but not the torch
import or the compile.

## Evidence boundary and exclusions

The boundary of a measured report states: "These are wall-clock timings on
the Apple GPU of the host that made this report. The kernels came from
torch.mps.compile_shader. The timings are host-specific Apple GPU (Metal)
measurements. No CUDA kernel ran, and no NVIDIA GPU was used. A best rate is
the highest median rate that this run measured, not a vendor peak. The layout
probe shows which SIMD group and lane ran each thread, not how the GPU
schedules SIMD groups. These timings do not measure or predict CUDA, NVIDIA,
distributed or production performance."

The lab reads no hardware counters. It does not record the GPU model, the
clock frequency, the thermal state or the other jobs on the host. It records
only the load averages. The report is evidence from one Apple GPU on one
host. It is not evidence of CUDA or NVIDIA performance, lecture viewing,
completed exercises or learner understanding.

## Sources

- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader` in
  `torch/mps/__init__.py`, and the Objective-C selectors in
  `torch/lib/libtorch_cpu.dylib`.
- Apple, Metal Shading Language Specification, the attributes
  `thread_position_in_grid`, `thread_index_in_threadgroup`,
  `simdgroup_index_in_threadgroup`, `thread_index_in_simdgroup` and
  `threads_per_simdgroup`:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Apple, Metal documentation, `MTLComputePipelineState`
  `maxTotalThreadsPerThreadgroup` and `threadExecutionWidth`:
  https://developer.apple.com/documentation/metal/mtlcomputepipelinestate
- Apple, Metal documentation, `MTLComputePipelineDescriptor`
  `threadGroupSizeIsMultipleOfThreadExecutionWidth`:
  https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
  SIAM, 2002, section 3.1, the bound `gamma_K(u) = K u / (1 - K u)`.
