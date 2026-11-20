# A performance checklist for the Metal matmul on the Apple GPU

Day 77 of the curriculum follows GPU MODE Lecture 8, "CUDA Performance
Checklist", and Lecture 2, "Recap Ch. 1-3 from the PMPP book". The tracker
notes name the checklist: coalescing, occupancy, tiling, arithmetic intensity
and memory vs compute bound. The exercise has three parts:

1. Print the checklist.
2. Apply every item to your matmul.
3. Record the speedup after each change.

This lab does the exercise for the float32 matmul of the Day 74 and Day 76
labs, on a Mac with an Apple GPU. It declares eight steps, S0 to S7. Each step
applies one item with one change, and it states a hypothesis before the run.
One run checks every kernel against the Day 74 error bound and times every
kernel with `torch.matmul` on the same device operands.

Nothing here runs a CUDA kernel or uses an NVIDIA GPU. The lab does not measure
cuBLAS. PyTorch does not report which kernel `torch.matmul` runs. The lab reads
no hardware counters, so it does not measure occupancy. Every timing is a
host-specific Apple GPU (Metal) measurement at the load averages that the
report records.

## Commands

```bash
python -m ai_journey.matmul_checklist_cli run \
  --config config/day-77-matmul-checklist.json \
  --output /tmp/day-77-matmul-checklist.json
python -m ai_journey.matmul_checklist_cli verify \
  --report /tmp/day-77-matmul-checklist.json \
  --config config/day-77-matmul-checklist.json
```

The parser uses the program name `ai-journey-matmul-checklist`. An installed
wheel gives the same commands under that name.

The `run` command compiles the kernels, checks them, times them, checks its own
report and writes the report. Then it prints these lines:

1. One line for each checklist step, with the item, the base kernel and the
   predicted verdicts. The lines come from the checklist that the report
   seals.
2. One line with the load averages before and after the run.
3. One line for each kernel that its pipeline refused, with the reason.
4. One line for each correctness case, with the bound and the largest relative
   error of each kernel. The line also tells if each kernel held the bits of
   the CPU replay.
5. One line for each sweep point, with the median GFLOP/s and the IQR range of
   each side.
6. For each sweep point, one line for each step kernel, with the two step
   ratios, the fraction of `torch.matmul` and the result of the hypothesis.
   Then one line with the fastest kernel at that point.
7. One line with the count of Python warnings that the run caught.
8. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`matmul_checklist_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `matmul_checklist_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the same summary
line with the prefix `verified_`. With `--config`, `verify` also requires that
the report comes from that exact config.

The exit codes are the codes of the Day 73 to 76 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

On the development host, one run of the pinned config took 58 seconds, with
the torch import. The run wrote a report of about 149 KB. The `verify` command
took about 3 seconds for that report.

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

The tests that need the GPU skip on other hosts and report the reason. The
other tests use a NumPy stand-in for the GPU and one short measured report
from the development host. The stand-in returns the CPU replay of the fma
chain, which is what the GPU returned on that host.

## The checklist

The item list follows the tracker notes. Ajay must still check it against Lecture 8.
The lab adds one more item, control divergence, for the ragged shape. The
module `matmul_checklist_steps.py` declares the steps. Every hypothesis
entered the repository before the first timed run of a Day 77 kernel. The
pinned config and each report seal the digest of the plan,
`cb5e3164e0d37716699dc9381b587faeeb40c2ca4f1a52d75b8ec586379323dc`. A config
for other hypotheses cannot run.

The steps form one chain. Each step changes the kernel that the step before
it keeps. Thus its ratio to that base kernel is the speedup of its own
change. Its ratio to S0 is the speedup of every change so far. These rules
apply:

- S0 and S1 add no new kernel. The Day 74 and Day 76 labs made those changes,
  and the run times their kernels again as evidence.
- The base of S0 is `matmul_x_rows`, the mapping without coalescing, so the S0
  ratio is the coalescing evidence of this run.
- S6 adds two group sizes beside the S5 kernel. The chain keeps the S5 kernel,
  so S7 changes the S5 kernel.
- A hypothesis predicts one or two verdicts of the Day 76 ratio rule. It holds
  at a point when the verdict of the kernel over its base is one of them.

| Step | Item | Status | Base kernel | New kernel | Predicted verdicts |
| --- | --- | --- | --- | --- | --- |
| S0 | coalescing | already applied | `matmul_x_rows` | `matmul_x_columns` | faster |
| S1 | tiling | already applied | `matmul_x_columns` | `matmul_tiled_32` | slower, no clear difference |
| S2 | arithmetic intensity | new kernel | `matmul_tiled_32` | `matmul_s2_coarse_4x1` | faster |
| S3 | arithmetic intensity | new kernel | `matmul_s2_coarse_4x1` | `matmul_s3_block_4x4` | faster |
| S4 | memory vs compute bound | new kernel | `matmul_s3_block_4x4` | `matmul_s4_vec4` | no clear difference |
| S5 | tiling | new kernel | `matmul_s4_vec4` | `matmul_s5_pad4` | no clear difference, slower |
| S6 | occupancy | new kernel | `matmul_s5_pad4` | `matmul_s6_group_8x8` | slower |
| S6 | occupancy | new kernel | `matmul_s5_pad4` | `matmul_s6_group_32x32` | no clear difference, slower |
| S7 | control divergence | new kernel | `matmul_s5_pad4` | `matmul_s7_interior` | faster |

The plan states the change and the hypothesis of each step in these words.

S0, coalescing:

- "Map x to the columns of C, so that the lanes of a SIMD group read
  consecutive elements of B (Day 74)."
- "matmul_x_columns is faster than matmul_x_rows at every point. On Day 74,
  matmul_x_rows ran at 0.30 to 0.63 of the matmul_x_columns rate. Its lanes
  read elements of A that are K floats apart."

S1, tiling:

- "Stage 32 x 32 tiles of A and B in threadgroup memory, with one output for
  each thread (Day 76)."
- "Tiling alone does not make the kernel faster. On Day 76, tile 32 ran at
  0.87 to 0.96 of the naive rate. The barriers and the threadgroup reads
  probably cost more time than the device loads that the tiles save."

S2, arithmetic intensity:

- "Give each thread 4 outputs in one column of the 32 x 32 tile, with 4
  register accumulators, in groups of 32 x 8 threads."
- "Each B value from threadgroup memory now serves 4 fma steps, so threadgroup
  reads fall from 2 to 1.25 for each fma. Each barrier also covers 4 times the
  work of a thread. S2 is faster than S1 by 1.1 or more."

S3, arithmetic intensity:

- "Give each thread a 4 x 4 block of outputs from 64 x 64 tiles, with K phases
  of 16."
- "Each value from threadgroup memory now serves 4 fma steps, so threadgroup
  reads fall from 1.25 to 0.5 for each fma. The model also halves the device
  loads for each output. S3 is faster than S2 by 1.1 or more."

S4, memory vs compute bound:

- "Stage A and B with float4 loads and stores when K and N are multiples of 4,
  and with scalars otherwise."
- "The no-cache model gives S3 15.9 FLOPs for each byte at 4096 cubed. That is
  below the ridge point of 38.8 from the Day 76 gap run, so the model calls S3
  memory bound. On Day 76, the naive kernel ran 12 times above its no-cache
  roof, so caches probably serve most repeated loads. Each thread of S3 also
  stages only 8 values for the 256 fma steps of a phase. S4 shows no clear
  difference from S3."

S5, tiling:

- "Add 4 floats of padding to each tile row in threadgroup memory, as a
  control for bank conflicts."
- "At each k step, the lanes of a SIMD group of S4 read two A words and 16
  consecutive B words. The 32-bank model of the Day 74 memory-access lab finds
  no conflict there, so padding cannot remove one. Padding moves the B rows
  off their 64-byte boundaries, and the gpu-ceilings lab measured slower
  consecutive reads off 128-byte boundaries. S5 is not faster than S4."

S6, occupancy:

- "Run the S5 kernel in 8 x 8 and 32 x 32 groups, with tiles of 32 x 32 and
  128 x 128."
- "No occupancy counter exists here, so this step records the time and the
  pipeline limits only. The 8 x 8 group doubles the modeled device loads for
  each output and holds only 2 SIMD groups, so it is slower than S5. The 32 x
  32 group halves those loads, but each group holds 1024 threads with 16
  accumulators, so it is not faster than S5."

S7, control divergence:

- "Give interior groups a path with no guards: no zero-fill selects, a
  constant k-loop bound and no store guards."
- "At the ragged shape, only 78 of the 1504 groups lie at an edge, so
  divergence costs little there. In the interior path, the constant loop bound
  lets the compiler unroll the k loop. S7 is faster than S5 by 1.1 or more."

## Kernels

The lab times ten kernels from three labs. Their source is in the Metal
Shading Language, and `torch.mps.compile_shader` compiles it at run time:

- `matmul_x_columns` and `matmul_x_rows` come from the Day 74 source,
  unchanged. They run one thread for each output in the naive group of the
  config.
- `matmul_tiled_32` is the Day 76 kernel text, unchanged.
- The seven Day 77 kernels come from one template in
  `matmul_checklist_source.py`. A `BlockDesign` fixes every size and three
  switches, so each step changes only the fields of its own change.

A Day 77 kernel gives one threadgroup one BM x BN tile of C. It walks K in
phases of BK. Each thread keeps a TM x TN block of outputs in registers. Thread
(x, y) of a group holds the outputs at rows `row0 + y + GY * i` and columns
`column0 + x + GX * j`, where the group has GX x GY threads. Thus the threads
of one SIMD group store neighboring columns together.

These rules keep each stored output equal to the CPU replay in principle:

- Each output takes exactly K steps of `sum = fma(a, b, sum)` in increasing k
  from 0. A guarded phase runs `min(BK, k - start)` steps, so the loop never
  reads a zero-filled slot past K.
- A slot past M in the A tile or past N in the B tile holds 0. It feeds only
  outputs that the kernel never stores.
- Only the stores have guards. Each barrier sits in the phase body, outside
  any condition on the thread. Only two conditions enclose a barrier: the
  interior test of S7 and the K tail. Each of them is the same for every
  thread of a group.

The design has three switches:

- `vector_loads`: stage four floats with one `float4` load and one `float4`
  store when K (for A) or N (for B) is a multiple of 4. Each vector address is
  then 16-byte aligned. Other shapes use the scalar path, and a `float4` past
  the matrix is zero.
- `pad`: add floats to each tile row in threadgroup memory. The padding is a
  multiple of 4, so the rows stay 16-byte aligned.
- `interior_path`: some groups take a fast path. Such a group has its tile
  inside M and N, and K and N are multiples of 4. It runs each full phase with
  no zero-fill select and a constant loop bound, and it stores with no guard.
  Edge groups and the K tail run the guarded code.

Each kernel declares its tiles as `float4` arrays and reads them as `float`
values. Thus a vector store is always aligned, and a kernel without vector
loads uses the same memory as one with them. The runtime binds only fresh MPS
tensors at storage offset 0, so each buffer is 16-byte aligned.

| Kernel | Tile (BM x BN x BK) | Outputs for each thread | Group | Threadgroup bytes |
| --- | --- | --- | --- | --- |
| `matmul_x_columns` | none | 1 x 1 | the naive group | 0 |
| `matmul_x_rows` | none | 1 x 1 | the naive group | 0 |
| `matmul_tiled_32` | 32 x 32 x 32 | 1 x 1 | 32 x 32 | 8192 |
| `matmul_s2_coarse_4x1` | 32 x 32 x 32 | 4 x 1 | 32 x 8 | 8192 |
| `matmul_s3_block_4x4` | 64 x 64 x 16 | 4 x 4 | 16 x 16 | 8192 |
| `matmul_s4_vec4` | 64 x 64 x 16 | 4 x 4 | 16 x 16 | 8192 |
| `matmul_s5_pad4` | 64 x 64 x 16 | 4 x 4 | 16 x 16 | 9472 |
| `matmul_s6_group_8x8` | 32 x 32 x 16 | 4 x 4 | 8 x 8 | 4864 |
| `matmul_s6_group_32x32` | 128 x 128 x 16 | 4 x 4 | 32 x 32 | 18688 |
| `matmul_s7_interior` | 64 x 64 x 16 | 4 x 4 | 16 x 16 | 9472 |

The registry in `matmul_checklist_variants.py` holds each kernel with its
origin, its text and its digest. Each digest covers the text of one kernel, so
a new kernel moves no other digest. The Day 74 kernels share the digest of
the Day 74 source. The run compiles the Day 74 source as one library. It
compiles the Day 76 kernel and the Day 77 kernels as a second library, after
the two-line Day 76 header.

| Source | SHA-256 |
| --- | --- |
| `matmul_tiled_32` | `f5ae7c6f03bcd1772c6eec5aed127916cb962b70bc7f99179d9f115a589ced30` |
| `matmul_s2_coarse_4x1` | `e2520d4f573c267360eb3833306bad7fbfe88325a4060b72b051f59bf54c144b` |
| `matmul_s3_block_4x4` | `d4530e74d254196749d27543ccf78238c0edaf5b551c28555095f5e77a23bc9d` |
| `matmul_s4_vec4` | `6df4d1063583f53372a700e08fb895858a42371f2e3d0f79bb48bd13fe4eac05` |
| `matmul_s5_pad4` | `8567cddd838fcc4aa82000c0f434864b5d4bd1c494e21d5937f787400e784028` |
| `matmul_s6_group_8x8` | `919930b29ccdd65de0b096cc22c1c5b0683e9f9c03c0791469098039511ae2bc` |
| `matmul_s6_group_32x32` | `fc609adecff160fd1e17f2ad85ba7b85785bf48cd040b5ef0a281fa4f090dcc0` |
| `matmul_s7_interior` | `b5053c2f0c52043d9c93ce8359a491878fd7e8ef44f2c85b6e901019841225ab` |
| checklist library | `d0a8f46a8831029a8894edca48da9b8d61646571c40c36d2f502d6f30cea027d` |
| Day 74 naive source | `beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc` |

## Pipeline limits and refusals

The runtime reads the limits of each compiled pipeline: the largest
threadgroup, the SIMD width and the static threadgroup memory. A pipeline
refuses a kernel in two cases:

- The pipeline allows fewer threads than the group of the kernel.
- The static threadgroup memory of the pipeline differs from the bytes that
  the kernel declares.

The report records each refusal with its numbers. A refused kernel never
runs, and the run continues with the other kernels. A step whose base kernel
got a refusal has no verdict.

These limits are the only occupancy facts that the pipelines give. They do
not tell how many groups a GPU core holds at one time, and the lab reads no
counter that does. Thus the S6 step records the time of each group size, not
its occupancy.

## Correctness

The correctness cases are the 9 shapes of the Day 74 config and one more
shape, 130 x 196 x 1000, with seed 77. The Day 74 shapes never stage on the
vector path with an interior group, so the added shape runs that path. The
config parser requires these properties:

- Each Day 74 kernel pads both axes of its group in some case.
- For each tiled kernel, some case has an M, some an N and some a K that is
  not a multiple of the tile. Some K above the K phase is not a multiple of
  it, so a later phase ends early.
- For each kernel with vector loads, some case stages A and some stages B on
  the scalar path. Some case stages both on the vector path, with an edge on M
  or N and a short K phase.
- For the interior kernel, some case of the vector path holds an interior
  group, an edge group and a short K phase.

Each case runs each admitted kernel once. Each output buffer is longer than C
by every spare output slot of its thread grid. Before the kernel runs, that
tail holds sentinel values. A store that escaped its guard can reach only the
tail or another element of C. Two rules apply, and a case that breaks a rule
stops the run:

- The tail must keep its sentinel values.
- The largest relative error must not be more than the Day 74 bound,
  `(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)`.

The run compares the bits of each output with the CPU replay of the Day 74
lab. The record counts the elements whose bits differ, and `replay_bits` is
true only when the count is 0. A different count does not stop the run. It is
a result.

## Timing

The benchmark times eleven sides on the same device operands at each sweep
point: the ten kernels and `torch.matmul(a, b, out=c)` with a preallocated
output. The benchmark uses the Day 74 and Day 76 method:

1. Each side runs 2 samples that are not timed.
2. Each timed sample calls `torch.mps.synchronize()` and reads
   `time.perf_counter_ns`.
3. The sample then launches a batch of calls, calls `torch.mps.synchronize()`
   again, and reads the clock again.
4. The first side of repeat r is side r mod 11, so the order moves by one side
   at each repeat. Each side has 11 timed samples.
5. After the last sample, the output of each side must agree with the S0
   output within `4 * gamma_K(2**-24) * (|A| @ |B|)`, as the device calculates
   it in float32.

A batch holds the fewest whole calls that reach 2^30 FLOPs. At the pinned
shapes, one call is enough. The GFLOP/s of a side is the FLOPs of one sample
divided by the median sample time in nanoseconds. The IQR of the sample times
gives the range of each rate. One budget of 300 seconds covers the
correctness cases and the sweep. The run stops between two samples after the
budget ends, and it then writes no report.

Each point also records two numbers of a traffic model with no cache, for
each kernel:

- The global element loads for each output: `2 * K` for a Day 74 kernel, and
  `(ceil(N / BN) * M * K + ceil(M / BM) * K * N) / (M * N)` for a tiled kernel.
- The FLOPs for each byte: `2 * K / (4 * (loads + 1))`, with 4 bytes for each
  element and one store for each output.

The model does not see the GPU caches. It is not a measurement.

## The speedup rule

For each sweep point and each kernel that a step adds, the report gives these
values from the same run:

- the median GFLOP/s and the range from the IQR of the samples
- the step ratio to the base kernel of the step
- the step ratio to S0, `matmul_x_columns`
- the fraction of the `torch.matmul` median at that point
- the result of the hypothesis at that point

A step ratio divides two median rates of the same point. The rule of the
Day 76 lab calls the first side faster when two conditions are true:

- The ratio is 1.1 or more. The config sets this minimum.
- The 75th percentile time of the first side is less than the 25th percentile
  time of the second side.

The rule calls the first side slower when the same test is true in the other
direction. Otherwise the result is `no_clear_difference`. The vendor side has
the label "vendor GPU matmul: torch.matmul on MPS, PyTorch 2.15.0". The label
names the call and the PyTorch version, not a kernel. The cuBLAS field says
"not measured, no NVIDIA GPU or cuBLAS on this host". It holds no number.

## One run on the development host

This run used the pinned config on 2026-11-20 at 11:29 IST. The host was a
shared Darwin arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14, NumPy
1.26.4 and PyTorch 2.15.0. The PyTorch float32 matmul precision was `highest`.
The environment did not set `PYTORCH_MPS_FAST_MATH` or
`PYTORCH_MPS_PREFER_METAL`. The run held the GPU lock that the timed GPU jobs
on this host take. Other jobs used the CPU.

The report recorded load averages of 19.21, 21.17 and 23.01 before the run. It
recorded 16.63, 20.39 and 22.64 after the run. The `uptime` command printed
16.11, 20.65 and 22.85 before the lock, and 18.58, 20.73 and 22.75 after the
run.

The report digest is
`130322a7b46db6bc7e9e9bf135a39225e2c139097cd167b74ceb22fd54b2e062`. The values
are host-specific Apple GPU (Metal) measurements from one run, not portable
figures. Each rate is the median GFLOP/s of 11 samples. The range in
parentheses comes from the IQR of the sample times. "Over its base" and "Over
S0" give the step ratio and the result of the rule.

Each pipeline allowed 1024 threads and reported a SIMD width of 32. The static
threadgroup memory of each pipeline was equal to the declared bytes, so the
run refused no kernel. All 100 outputs held the bits of the replay: 10 cases
and 10 kernels. Each guard tail kept its sentinel values. The largest relative
error was 0.477 of the bound, at 1 x 1 x 1. In the benchmark checks, each side
had the same bits as `matmul_x_columns` at each point. That result includes
`torch.matmul`. The run caught 0 Python warnings, and the capture began before
the import of torch.

### 1024 x 1024 x 1024

| Step | Kernel | Median GFLOP/s (IQR range) | Over its base | Over S0 | Of `torch.matmul` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 base | `matmul_x_rows` | 178.6 (172.7 to 181.4) | | | | |
| S0 | `matmul_x_columns` | 354.3 (309.4 to 370.4) | 1.983, faster | 1.000 | 0.158 | held |
| S1 | `matmul_tiled_32` | 323.9 (311.7 to 341.7) | 0.914, no clear difference | 0.914, no clear difference | 0.144 | held |
| S2 | `matmul_s2_coarse_4x1` | 629.1 (623.7 to 653.3) | 1.942, faster | 1.776, faster | 0.280 | held |
| S3 | `matmul_s3_block_4x4` | 920.3 (571.7 to 1049.5) | 1.463, no clear difference | 2.598, faster | 0.410 | missed |
| S4 | `matmul_s4_vec4` | 1218.7 (1151.8 to 1318.7) | 1.324, faster | 3.440, faster | 0.543 | missed |
| S5 | `matmul_s5_pad4` | 1292.4 (951.7 to 1445.7) | 1.060, no clear difference | 3.648, faster | 0.575 | held |
| S6 | `matmul_s6_group_8x8` | 1145.5 (643.6 to 1400.0) | 0.886, no clear difference | 3.233, faster | 0.510 | missed |
| S6 | `matmul_s6_group_32x32` | 920.8 (854.1 to 1031.8) | 0.712, no clear difference | 2.599, faster | 0.410 | held |
| S7 | `matmul_s7_interior` | 1312.0 (1145.2 to 1469.4) | 1.015, no clear difference | 3.704, faster | 0.584 | missed |
| vendor | `torch.matmul` | 2246.2 (1840.6 to 2534.7) | | | 1.000 | |

### 2048 x 2048 x 2048

| Step | Kernel | Median GFLOP/s (IQR range) | Over its base | Over S0 | Of `torch.matmul` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 base | `matmul_x_rows` | 147.2 (144.8 to 148.5) | | | | |
| S0 | `matmul_x_columns` | 389.6 (387.7 to 398.1) | 2.647, faster | 1.000 | 0.115 | held |
| S1 | `matmul_tiled_32` | 348.1 (345.7 to 349.9) | 0.893, slower | 0.893, slower | 0.103 | held |
| S2 | `matmul_s2_coarse_4x1` | 708.7 (691.0 to 721.1) | 2.036, faster | 1.819, faster | 0.209 | held |
| S3 | `matmul_s3_block_4x4` | 1177.9 (1116.9 to 1222.9) | 1.662, faster | 3.023, faster | 0.347 | held |
| S4 | `matmul_s4_vec4` | 1553.9 (1516.2 to 1579.1) | 1.319, faster | 3.988, faster | 0.458 | missed |
| S5 | `matmul_s5_pad4` | 1524.8 (1516.7 to 1565.6) | 0.981, no clear difference | 3.913, faster | 0.450 | held |
| S6 | `matmul_s6_group_8x8` | 1582.2 (1546.5 to 1635.9) | 1.038, no clear difference | 4.061, faster | 0.467 | missed |
| S6 | `matmul_s6_group_32x32` | 1142.3 (1106.5 to 1170.2) | 0.749, slower | 2.932, faster | 0.337 | held |
| S7 | `matmul_s7_interior` | 1774.7 (1668.4 to 1819.4) | 1.164, faster | 4.555, faster | 0.523 | held |
| vendor | `torch.matmul` | 3390.3 (2710.8 to 3505.9) | | | 1.000 | |

### 4096 x 4096 x 4096

| Step | Kernel | Median GFLOP/s (IQR range) | Over its base | Over S0 | Of `torch.matmul` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 base | `matmul_x_rows` | 169.1 (168.9 to 170.0) | | | | |
| S0 | `matmul_x_columns` | 296.5 (288.0 to 316.6) | 1.753, faster | 1.000 | 0.084 | held |
| S1 | `matmul_tiled_32` | 338.7 (337.0 to 341.2) | 1.142, faster | 1.142, faster | 0.096 | missed |
| S2 | `matmul_s2_coarse_4x1` | 612.0 (596.2 to 620.6) | 1.807, faster | 2.064, faster | 0.174 | held |
| S3 | `matmul_s3_block_4x4` | 1160.2 (1153.2 to 1195.8) | 1.896, faster | 3.914, faster | 0.329 | held |
| S4 | `matmul_s4_vec4` | 1566.6 (1554.9 to 1587.4) | 1.350, faster | 5.284, faster | 0.445 | missed |
| S5 | `matmul_s5_pad4` | 1544.9 (1530.1 to 1555.2) | 0.986, no clear difference | 5.211, faster | 0.439 | held |
| S6 | `matmul_s6_group_8x8` | 1582.8 (1562.9 to 1621.2) | 1.025, no clear difference | 5.339, faster | 0.449 | missed |
| S6 | `matmul_s6_group_32x32` | 1204.0 (1186.2 to 1217.3) | 0.779, slower | 4.061, faster | 0.342 | held |
| S7 | `matmul_s7_interior` | 1825.8 (1815.1 to 1832.2) | 1.182, faster | 6.159, faster | 0.518 | held |
| vendor | `torch.matmul` | 3521.4 (3250.9 to 3698.8) | | | 1.000 | |

### 2000 x 3000 x 1000

| Step | Kernel | Median GFLOP/s (IQR range) | Over its base | Over S0 | Of `torch.matmul` | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 base | `matmul_x_rows` | 116.3 (114.9 to 117.7) | | | | |
| S0 | `matmul_x_columns` | 383.4 (366.1 to 393.5) | 3.296, faster | 1.000 | 0.126 | held |
| S1 | `matmul_tiled_32` | 341.0 (336.3 to 346.9) | 0.889, slower | 0.889, slower | 0.112 | held |
| S2 | `matmul_s2_coarse_4x1` | 737.6 (714.7 to 742.2) | 2.163, faster | 1.924, faster | 0.242 | held |
| S3 | `matmul_s3_block_4x4` | 1334.0 (1293.8 to 1368.5) | 1.808, faster | 3.479, faster | 0.438 | held |
| S4 | `matmul_s4_vec4` | 1538.5 (1501.2 to 1570.2) | 1.153, faster | 4.013, faster | 0.506 | missed |
| S5 | `matmul_s5_pad4` | 1542.2 (1470.0 to 1570.1) | 1.002, no clear difference | 4.023, faster | 0.507 | held |
| S6 | `matmul_s6_group_8x8` | 1579.5 (1478.7 to 1614.7) | 1.024, no clear difference | 4.120, faster | 0.519 | missed |
| S6 | `matmul_s6_group_32x32` | 1113.4 (1051.5 to 1149.5) | 0.722, slower | 2.904, faster | 0.366 | held |
| S7 | `matmul_s7_interior` | 1665.8 (1505.8 to 1737.9) | 1.080, no clear difference | 4.345, faster | 0.547 | missed |
| vendor | `torch.matmul` | 3042.8 (2966.5 to 3291.3) | | | 1.000 | |

### What helped

- Coalescing (S0): `matmul_x_columns` was 1.75 to 3.30 times as fast as
  `matmul_x_rows` at each point.
- Thread coarsening (S2): 4 outputs for each thread made the kernel 1.81 to
  2.16 times as fast as tile 32 at each point.
- Register blocking (S3): the 4 x 4 block was 1.66 to 1.90 times as fast as S2
  at three points. At 1024 x 1024 x 1024, the ratio was 1.463, but the IQR
  ranges overlapped, so the rule found no clear difference.
- Vector staging (S4): `float4` loads and stores made the kernel 1.15 to 1.35
  times as fast as S3 at each point. The hypothesis predicted no clear
  difference, so it failed at all four points.
- The interior path (S7): it was 1.16 and 1.18 times as fast as S5 at 2048 and
  4096 cubed. At 1024 cubed and at the ragged shape, the rule found no clear
  difference.

S7 was the fastest kernel at each point. It reached 1312.0 to 1825.8 GFLOP/s,
3.70 to 6.16 times the S0 rate. That is 0.518 to 0.584 of the `torch.matmul`
rate at the same point.

### What did not help

- Tiling alone (S1): tile 32 was slower than S0 at 2048 cubed and at the
  ragged shape, as on Day 76. At 1024 cubed, the rule found no clear
  difference. At 4096 cubed, tile 32 was 1.142 times as fast as S0, so the
  hypothesis failed there. At that point, `matmul_x_columns` ran at
  296.5 GFLOP/s. That rate is below its 389.6 GFLOP/s at 2048 cubed and below
  the 369.4 GFLOP/s of the Day 76 run.
- Padding (S5): the ratio to S4 was 0.981 to 1.060, with no clear difference
  at any point. This result agrees with the hypothesis. It does not show the
  bank structure of the GPU.
- Smaller groups (S6, 8 x 8): the ratio to S5 was 0.886 to 1.038, with no
  clear difference at any point. The hypothesis predicted a slower kernel, so
  it failed at all four points.
- Larger groups (S6, 32 x 32): the kernel was slower than S5 at three points,
  with ratios of 0.722 to 0.779. At 1024 cubed, the IQR ranges overlapped. Of
  the four kernels of S5 to S7, this kernel has the fewest modeled device
  loads, but it was the slowest at each point.

The predictions held at 24 of 36 points: 9 step kernels at 4 points each.
They failed most for S4 and for the 8 x 8 groups, at all four points each.

### Memory vs compute bound

The Day 76 gap run measured an FMA ceiling of 4836.8 GFLOP/s and a bandwidth
ceiling of 124.58 GB/s on this host. That run used other loads, so the
comparisons in this list are approximate:

- At 4096 cubed, S7 reached 0.377 of the FMA ceiling, and `torch.matmul`
  reached 0.728 of it.
- The no-cache model gives S7 15.88 FLOPs for each byte at 4096 cubed. Its
  modeled roof is thus 1977.8 GFLOP/s, and S7 reached 0.92 of it.
- The 8 x 8 group kernel uses 32 x 32 tiles, at 7.97 FLOPs for each byte. It
  reached 1582.8 GFLOP/s, 1.59 times its modeled roof of 992.8 GFLOP/s. Thus
  the GPU caches probably served part of its repeated loads.
- The 32 x 32 group kernel has the highest modeled intensity, 31.51 FLOPs for
  each byte. It was the slowest of the four kernels of S5 to S7.

Thus the no-cache model does not predict the order of these kernels. The rates
of S5, S6 and S7 stay well below the FMA ceiling. The lab reads no hardware
counters, so it cannot name the limit of these kernels. Any name is an
inference from timings and measured ceilings. Possible limits include the
threadgroup reads of the k loop and the barriers, but the run cannot tell
which one applies.

### Limits of this run

- One run on one shared host gives each ratio. The load averages were near 20
  from other jobs.
- The steps form one chain, so each ratio measures one change on top of the
  changes before it. A change can have another effect on another base kernel.
- The S7 change removes the guards and also gives the k loop a constant bound.
  The run cannot separate the two effects.
- With a fixed 4 x 4 block of outputs, the group size of S6 also sets the tile
  size. The run cannot separate occupancy from tile size, and no occupancy
  counter exists here.
- The rule calls a ratio clear only when the IQR ranges do not overlap. Wide
  ranges at 1024 cubed can hide a change there.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The report records the items in this list:

- the config and its digest, and the host environment without host names
- the declared checklist with every hypothesis and the digest of the plan
- the two MPS switches and the load averages before and after the run
- the Python warnings of the run, with the home directory as `~`
- the two source digests, and the digest, limits and verdict of each kernel
- the correctness cases, the raw samples, the medians, the IQRs and the checks
- the model numbers, the speedup rows, the result of each hypothesis and the
  evidence boundary

The run captures warnings from a point before the runtime imports torch. In a
fresh process, the capture thus holds each warning that torch raised up to the
end of the run. The report also records `PYTORCH_MPS_FAST_MATH` and
`PYTORCH_MPS_PREFER_METAL` from the environment, because each switch can
change what `torch.matmul` runs.

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk, and moves it into place with `os.replace`. It
refuses an output that aliases the config. The readers accept only regular
files that are not symbolic links, and they refuse duplicate JSON keys.

The verifier uses the CPU methods of the Day 76 verifier, not the code of the
run. It also uses three modules of the run that measure nothing: the config
parser, the variant registry and the declared checklist. It does not import
torch. It does these checks:

1. It draws the operands again from the seed and calculates the reference
   again.
2. It replays the fma chain with a rounding to odd at each step. The run uses
   a different method, so the two methods check each other.
3. It rebuilds the digest of each output that records the bits of the replay,
   with the sentinel tail of its launch.
4. It derives each pipeline verdict again from the recorded limits.
5. It requires the sealed checklist to equal the declared plan, byte for byte.
6. It calculates each statistic, rate, check, model number, ratio, verdict and
   result of a hypothesis again from the raw samples.

An output that records other bits cannot be rebuilt. For that output, the
verifier requires a different digest, a count of 1 or more differing elements
and an error within the bound. The timings cannot be observed again, so the
verifier checks them for structure only.

The repository does not keep the report of the pinned run. The tests keep one
short report of a recorded config from 2026-11-20 at 11:26 IST, with digest
`2da27f7cfe023cea5c77395eeab67c6eee5825cf28467c44c1407020616e5fb5`. A host
without torch can check it with the verifier. Its rates mostly show the cost
of the synchronize calls.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 77 | 0 to 2^32 - 1 |
| `checklist_sha256` | the digest of the declared plan | only that digest |
| `naive.group_size` | [32, 8] | [x, y], with at most 1024 threads in total |
| `correctness.shapes` | the 9 Day 74 shapes and 130 x 196 x 1000 | 1 to 16 different [m, n, k] triples that run every path |
| `benchmark.shapes` | 1024^3, 2048^3, 4096^3 and 2000 x 3000 x 1000 | 1 to 16 different [m, n, k] triples |
| `benchmark.warmup` | 2 | 0 to 10 samples that are not timed |
| `benchmark.repeats` | 11 | 3 to 50 timed samples for each side |
| `benchmark.target_flops_per_sample` | 1073741824 | 2^20 to 2^40, with at most 4096 calls in a sample |
| `benchmark.max_runtime_seconds` | 300 | 1 to 600 |
| `comparison.min_faster_ratio` | 1.1 | more than 1.0 and at most 4.0 |

The config parser counts the planned work before a run. The pinned config has
10 cases, 186189684 replay steps and 100 correctness launches. It also has 4
sweep points, 484 timed samples, 572 calls and 24133581801472 FLOPs. These
counts include each kernel, so they are an upper limit when a pipeline
refuses a kernel.

## Evidence boundary and exclusions

The evidence boundary of a measured report holds true or false fields and one
statement. The fields `apple_gpu_metal_this_host` and `host_specific` are true.
These fields are false: `cuda_kernel_executed`, `nvidia_gpu_used`,
`cublas_measured`, `vendor_kernel_known`, `vendor_peak_claimed`,
`hardware_counters_used`, `occupancy_measured`, `distributed_claim`,
`production_claim` and `portable_performance_claim`.

The statement gives these facts:

- The timings are wall-clock timings on the Apple GPU of the host that made
  the report, at the recorded load averages.
- The checklist kernels came from `torch.mps.compile_shader`, and
  `torch.matmul` ran on the MPS device.
- PyTorch does not report which kernel `torch.matmul` ran. Thus the vendor
  number names the call, not a kernel.
- The lab read no hardware counter, so the occupancy step records times and
  pipeline limits, not occupancy.
- No CUDA kernel ran. The lab used no NVIDIA GPU and did not measure cuBLAS.
- Each ratio compares two sides of one run on this host. The timings do not
  measure or predict CUDA, NVIDIA, cuBLAS, distributed or production
  performance.

The lab does not record the GPU model, the clock frequency, the thermal state
or the other jobs on the host. The rates count the FLOPs of the matmul, not
the instructions of the GPU. A bottleneck name in this guide is an inference
from timings and measured ceilings, not from hardware counters. The results
come from one Apple GPU on one host. They are not evidence of CUDA, NVIDIA or
cuBLAS performance, lecture viewing, completed exercises or learner
understanding.

## Sources

- GPU MODE lectures, https://github.com/gpu-mode/lectures: Lecture 8, "CUDA
  Performance Checklist", and Lecture 2, "Recap Ch. 1-3 from the PMPP book".
- W. W. Hwu, D. B. Kirk and I. El Hajj, Programming Massively Parallel
  Processors, 4th ed., Morgan Kaufmann, 2022, chapters 4 to 6, for control
  divergence, occupancy, tiling, coalescing and thread coarsening.
- Apple, Metal Shading Language Specification, for the `threadgroup` address
  space, `threadgroup_barrier`, `float4` and `fma`:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader`, and
  the MPS environment variables page of the PyTorch documentation.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 3.1, for the bound `gamma_K`.
- S. Boldo and G. Melquiond, "Emulation of FMA and correctly rounded sums:
  proved algorithms using rounding to odd", IEEE Transactions on Computers
  57(4), 2008, for the rounding to odd.
