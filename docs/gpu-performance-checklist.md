# A GPU performance checklist, applied on an Apple GPU

Day 77 of the curriculum follows GPU MODE Lecture 8, "CUDA Performance
Checklist". The exercise asks for three things:

1. Print the checklist.
2. Apply every item to your matmul.
3. Record the speedup after each change.

The [checklist lab](matmul-checklist-lab.md) applies the items to the float32
Metal matmul and records the speedups. This page is the printable checklist.
For each item, it gives these facts:

1. What the item means on an NVIDIA GPU.
2. What the item maps to on an Apple GPU through Metal.
3. How the lab applied the item.
4. The result of the pinned run on the development host.
5. What this host cannot measure.

The second part of the page answers the item "memory vs compute bound" with
measurements. The module `matmul_checklist_bound_cli` joins three things: the
checklist rates, the ceilings of the [gpu-ceilings lab](gpu-ceilings-lab.md)
and a traffic model of each kernel. It then applies the rule of the
[bottleneck lab](matmul-bottleneck-lab.md).

Nothing here runs a CUDA kernel or uses an NVIDIA GPU. The NVIDIA parts give
documented CUDA concepts, not measurements. The lab reads no hardware
counters, so it does not measure occupancy, bank conflicts or bytes. Every
Metal number is a host-specific Apple GPU (Metal) measurement at the load
averages that its report records.

## The checklist on one page

| Item | Question | Step | Result on this host |
| --- | --- | --- | --- |
| Coalescing | Do the lanes of a group read consecutive addresses? | S0 | Faster at 4 of 4 points, 1.753 to 3.296 times |
| Occupancy | Do enough groups stay on a core to hide latency? | S6 | 8 x 8 groups: no clear difference. 32 x 32 groups: slower at 3 points |
| Tiling | Does each device load serve many outputs? | S1, S5 | Tile 32: not faster at 3 points. Padding: no clear difference |
| Arithmetic intensity | Does each value that a thread reads serve many fma steps? | S2, S3 | S2: faster at 4 points. S3: faster at 3 points. 1.463 to 2.163 times |
| Memory vs compute bound | Which measured ceiling limits the kernel? | S4 and the bound lab | At most 0.378 of the FMA ceiling. The traffic model does not explain the order |
| Control divergence | Do all lanes of a group take the same path? | S7 | Faster at 2 points, 1.164 and 1.182 times |

Each ratio divides the median rate of a step kernel by the median rate of its
base kernel, at one point of the pinned run. The Day 76 rule calls a ratio
faster or slower only when two conditions are true. The ratio is 1.1 or more
in that direction, and the IQR ranges of the two sides do not overlap. The
pinned run has four points: 1024, 2048 and 4096 cubed, and 2000 x 3000 x 1000.

## The items

### Coalescing

On an NVIDIA GPU, a warp holds 32 threads. When the threads of a warp read
global memory, the hardware combines the reads into the fewest 32-byte
transactions that cover them. Consecutive and aligned 4-byte addresses thus
need 4 transactions for the warp. A stride of 8 floats or more needs 32
transactions.

On an Apple GPU, Metal puts the threads of a threadgroup into SIMD groups. On
this host, each pipeline reported a SIMD width of 32. The same rule applies in
principle: consecutive lanes should read consecutive addresses.

- In this lab: S0 maps x to the columns of C in `matmul_x_columns`. The lanes
  of a SIMD group then read consecutive elements of B. The base kernel
  `matmul_x_rows` maps x to the rows of C, so its lanes read elements of A that
  are K floats apart.
- Result: `matmul_x_columns` was 1.753 to 3.296 times as fast as
  `matmul_x_rows`. The traffic model counts the same device bytes for both
  kernels, so the model cannot show this change.
- Not measured here: the lab counts no memory transactions. The Day 74
  memory-access lab models them, but this host has no counter that reads them.

### Occupancy

On an NVIDIA GPU, occupancy is the ratio of the active warps on a
multiprocessor to the largest number of warps that it can hold. The registers
of each thread, the shared memory of each block and the threads of each block
set this limit. More resident warps can hide more memory latency. Nsight
Compute reports the occupancy that a kernel reached.

On an Apple GPU, each compute pipeline reports the largest threadgroup that it
allows, its SIMD width and its static threadgroup memory. These limits are the
only occupancy facts that the lab can read. They do not tell how many groups a
GPU core holds at one time.

- In this lab: S6 runs the S5 design in groups of 8 x 8 and 32 x 32 threads.
  Each thread keeps 4 x 4 outputs, so the tiles become 32 x 32 and 128 x 128.
  Every pipeline allowed 1024 threads, and the static threadgroup memory of
  each pipeline was equal to the declared bytes.
- Result: for the 8 x 8 groups, the ratio to S5 was 0.886 to 1.038, with no
  clear difference at each point. For the 32 x 32 groups, the ratio was 0.712
  to 0.779, slower at 3 points.
- Not measured here: the lab reads no occupancy counter. The group size also
  sets the tile size, so the run cannot separate occupancy from tile size.

### Tiling

On an NVIDIA GPU, a block stages tiles of A and B in `__shared__` memory and
waits at `__syncthreads()`. Each value that the block reads from global memory
then serves many outputs. Shared memory has 32 banks of 4-byte words. When the
lanes of a warp read different words of one bank, the reads occur in turn. One
word of padding for each tile row can remove such conflicts.

On an Apple GPU, the counterpart of shared memory is the `threadgroup` address
space of Metal, with `threadgroup_barrier(mem_flags::mem_threadgroup)`. The
gpu-ceilings lab has a bank kernel that reads tiles with 32-word rows. In the
ceiling run of this page, it read 20.41 times as slowly at a stride of 32
words as at a stride of 1. With 33-word rows, the largest slowdown was 4.79.

- In this lab: S1 stages 32 x 32 tiles with one output for each thread, as the
  Day 76 kernel does. S5 adds 4 floats of padding to each tile row, as a
  control for bank conflicts.
- Result: S1 was 0.889 to 1.142 times as fast as S0. It was slower at 2
  points, and it was faster only at 4096 cubed. The model gives S1 about 0.031
  of the device bytes of S0, but S1 was not faster at 3 points. S5 was 0.981 to
  1.060 times as fast as S4, with no clear difference at each point.
- Not measured here: the lab reads no bank-conflict counter. Thus the S5 result
  does not show the bank structure of this GPU.

### Arithmetic intensity

On an NVIDIA GPU, the arithmetic intensity of a kernel is its FLOPs for each
byte that it moves. Thread coarsening and register tiling give each thread more
outputs. Each value that a thread reads from shared memory then serves more fma
steps. Each barrier and each index calculation also serve more work.

On an Apple GPU, the same changes apply. Each thread keeps its block of outputs
in registers.

- In this lab: S2 gives each thread 4 outputs in one column of the 32 x 32
  tile. The model gives 1.25 threadgroup reads for each fma, against 2 for S1.
  S3 gives each thread a 4 x 4 block of a 64 x 64 tile, with K phases of 16.
  The model gives 0.5 threadgroup reads for each fma and about half of the
  device bytes of S2.
- Result: S2 was 1.807 to 2.163 times as fast as S1, faster at each point. S3
  was 1.463 to 1.896 times as fast as S2, faster at 3 points. At 1024 cubed,
  the IQR ranges overlapped.
- Not measured here: the lab reads no register count, spill count or
  instruction count.

### Memory vs compute bound

On an NVIDIA GPU, the roofline model compares the intensity of a kernel with
the ridge point. The ridge point is the peak FLOP rate divided by the peak
bandwidth. A kernel with less intensity than the ridge point is memory bound,
and a kernel with more intensity is compute bound. Nsight Compute draws a
roofline from hardware counters.

On an Apple GPU, this lab uses no vendor peak. The gpu-ceilings lab measures
an FMA ceiling and a bandwidth ceiling with its own kernels on this host.
`gpu_ceiling_roofline.MeasuredRoofline` takes the two ceilings from a verified
report.

- In this lab: S4 stages A and B with `float4` loads and stores. The model
  gives S4 the device bytes of S3, but a quarter of the device load operations
  of S3.
  The bound lab below gives each kernel its modeled intensity, its attainable
  rate, its share of the FMA ceiling and its Day 75 name.
- Result: S4 was 1.153 to 1.350 times as fast as S3, faster at each point. The
  hypothesis predicted no clear difference. No kernel used more than 0.378 of
  the FMA ceiling. The modeled intensity of each kernel is below the ridge
  point, but the model does not explain the order of the kernels.
- Not measured here: the lab counts no bytes and reads no cache counter. Each
  byte count comes from a model.

### Control divergence

On an NVIDIA GPU, when the threads of a warp take different branches, the warp
runs the paths in turn. A guard on each load and store also adds instructions
when every thread takes the same path.

On an Apple GPU, the threads of a SIMD group also run concurrently, 32 lanes on
this host. This lab does not measure the cost of a branch there. It measures
the time of the S7 change only.

- In this lab: S7 gives interior groups a path with no guards. It has no
  zero-fill selects, a constant k-loop bound and no store guards. Edge groups
  and the K tail keep the guarded code.
- Result: S7 was 1.164 and 1.182 times as fast as S5 at 2048 and 4096 cubed.
  At 1024 cubed and at the ragged shape, the rule found no clear difference.
- Not measured here: the S7 change removes the guards and also gives the k loop
  a constant bound. The run cannot separate the two effects, and it reads no
  divergence counter.

## Memory vs compute bound, measured

The bound lab names the limit of each kernel at each sweep point. Each name is
an inference from timings and measured ceilings, not from hardware counters.
The lab read no counter, so it did not observe a stall, a cache hit, a bank
conflict or the occupancy of the GPU.

### Bound commands

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
python -m ai_journey.matmul_checklist_cli run \
  --config config/day-77-matmul-checklist.json \
  --output /tmp/day-77-matmul-checklist.json
python -m ai_journey.matmul_checklist_bound_cli timing \
  --config config/day-77-matmul-checklist.json \
  --output /tmp/day-77-one-call-timing.json
python -m ai_journey.matmul_checklist_bound_cli verify-timing \
  --report /tmp/day-77-one-call-timing.json \
  --config config/day-77-matmul-checklist.json
python -m ai_journey.matmul_checklist_bound_cli bound \
  --checklist-report /tmp/day-77-matmul-checklist.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --timing-report /tmp/day-77-one-call-timing.json \
  --rule-config config/day-75-matmul-bottleneck.json \
  --output /tmp/day-77-matmul-checklist-bound.json
python -m ai_journey.matmul_checklist_bound_cli verify \
  --report /tmp/day-77-matmul-checklist-bound.json \
  --rule-config config/day-75-matmul-bottleneck.json \
  --checklist-report /tmp/day-77-matmul-checklist.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --timing-report /tmp/day-77-one-call-timing.json
```

The parser uses the program name `ai-journey-matmul-checklist-bound`. The
package has no console script for these commands, so run the module with
`python -m`.

The `timing` command needs an Apple GPU, as the checklist `run` command does.
The `bound`, `verify` and `verify-timing` commands run no kernel and do not
import torch. With the three reports and `--rule-config`, `verify` also builds
the bound again and requires the same bytes. Its summary line then says
`sources_checked=true`. The `verify` command takes the three report paths
together or none of them.

The commands use the exit codes of the Day 73 to 77 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report or one of its sources failed
  the checks.
- 2: a usage error, an unreadable or invalid input, sources that do not pair,
  or a refused output. Code 2 also marks a run that failed a check, had a
  device error or exceeded its time budget.
- 3: this host cannot run Metal kernels through PyTorch, so `timing` stopped
  before it measured or wrote anything.

A measured line has the prefix `apple_gpu_metal_measurement`. The summary
lines have the prefixes `matmul_checklist_one_call_measurement=` and
`matmul_checklist_bound_measurement=`. A synthetic report uses
`matmul_checklist_one_call_synthetic_timing=` or
`matmul_checklist_bound_synthetic_timing=` instead. A `verify` command adds
the prefix `verified_`.

### One-call timing

The Day 75 rule needs three times for one call of each side:

- `median_gpu_ns`: the interval between two MPS events around the call.
- `median_host_ns`: the host time from an idle queue until the call is
  complete.
- `median_dispatch_ns`: the host time to issue the call, before the host
  waits.

The checklist report times batches between two synchronizations, so it does
not hold these times. The `timing` command measures them with the method of
the [gpu-timing lab](gpu-timing-lab.md), as the Day 76 gap lab does. Each
sample holds one launch and reads one clock. The order of the sides and the
clocks moves by one at each repeat. The command uses the checklist config, so
it times the same 11 sides on the operands of the checklist run:

1. Each side runs at a 1 x 1 x 1 floor. The host time of this call is an upper
   limit for a fixed cost that each call of the side pays.
2. Each side runs at each sweep shape. On each clock, it runs 2 samples that
   are not timed and 11 timed samples.
3. After each shape, each output must agree with the S0 output within the
   bound of the checklist run. If an output does not agree, the run stops.

The report seals the version 2 event-domain text of the gpu-timing lab. That
text says that an event interval can also hold host time between the two
`record()` calls. The report also records the Python warnings of the run, as
the checklist report does.

### The traffic model

The module `matmul_checklist_bound_traffic.py` extends the Day 74
`TiledMatmulTraffic` to the tiles of Day 77. A Day 77 tile has BM x BN outputs,
K phases of BK and TM x TN outputs for each thread. For one kernel and one
shape, the model counts these values in exact integers:

- Device loads: `ceil(N / BN) * M * K + ceil(M / BM) * K * N` elements. A tile
  slot past M, N or K holds a zero and costs no load.
- Device load operations: one for each element on the scalar path, or one for
  each four elements on the `float4` path.
- Device stores: `M * N` elements, 4 bytes for each element.
- Threadgroup stores: every tile slot of every phase.
- Threadgroup reads: TM + TN reads at each of the K steps of each launched
  thread, so `(TM + TN) / (TM * TN)` reads for each fma.
- Barriers: two for each phase.

The two Day 74 kernels keep the no-reuse model, `2 * M * N * K` loads with no
threadgroup traffic. The model sees no cache, no coalescing and no bank
conflict. It also counts the steps of padding threads, which never store.

The tests check the counts against hand counts and against a walk of the
loops of each kernel text. For the Day 76 tile and the Day 74 kernels, the
counts must equal those of `TiledMatmulTraffic` and `NaiveMatmulTraffic`. The
loads for each output must also equal the model numbers of the checklist
report.

| Kernels | Tile and thread block | FLOPs for each byte at 4096 cubed | Threadgroup reads for each fma |
| --- | --- | --- | --- |
| `matmul_x_columns`, `matmul_x_rows` | no tile | 0.250 | none |
| `matmul_tiled_32` | 32 x 32 x 32, 1 x 1 | 7.97 | 2 |
| `matmul_s2_coarse_4x1` | 32 x 32 x 32, 4 x 1 | 7.97 | 1.25 |
| `matmul_s3_block_4x4` to `matmul_s5_pad4`, `matmul_s7_interior` | 64 x 64 x 16, 4 x 4 | 15.88 | 0.5 |
| `matmul_s6_group_8x8` | 32 x 32 x 16, 4 x 4 | 7.97 | 0.5 |
| `matmul_s6_group_32x32` | 128 x 128 x 16, 4 x 4 | 31.51 | 0.5 |

### Inputs and values

The `bound` command reads three sealed reports. Each report must pass its own
verifier before the command uses a number from it:

- The checklist report gives the median, p25 and p75 sample times and the
  median rate of each side.
- The gpu-ceilings report gives the FMA ceiling and the bandwidth ceiling.
- The timing report gives the one-call medians and the fastest host time of
  each side.

The command refuses reports that do not pair:

- The three reports must name the same Python, NumPy, PyTorch and platform
  tokens.
- The checklist report and the timing report must come from one config. They
  must also have the same float32 matmul precision, MPS switches and admitted
  kernels.
- The rule config must use the `ideal_reuse` traffic model.

For each sweep point and side, the bound report gives these values:

- The FMA share: the median rate divided by the measured FMA ceiling.
- The modeled traffic and intensity of the kernel.
- The attainable rate of `MeasuredRoofline` at the modeled intensity: the
  lower of the FMA ceiling and the bandwidth ceiling times the intensity. The
  modeled share is the median rate divided by this rate.
- The modeled device bandwidth: the modeled bytes divided by the median time
  of one call. It is the bandwidth that the rate needs if the model holds.
- The modeled threadgroup reads for each nanosecond, at the median rate.
- The Day 75 name from `matmul_bottleneck_rule.classify`, with the thresholds
  of `config/day-75-matmul-bottleneck.json` and the ideal-reuse roof.

The ideal-reuse model moves each element one time, and no kernel moves fewer
bytes. Thus its roof is an upper limit for each side, also for `torch.matmul`.
PyTorch does not name the kernel of `torch.matmul`, so no traffic model
applies to it.

For each step kernel and point, the report then compares the model of the
kernel with the model of its base kernel. It gives the ratios of the modeled
device bytes and of the load operations. It also gives the threadgroup reads
for each fma of both kernels. Beside them, it gives the step ratio and verdict
of the Day 76 rule from the checklist samples, and the result of the
hypothesis.

### One bound run on the development host

The bound used the pinned checklist report of the
[checklist lab](matmul-checklist-lab.md), from 2026-11-20 at 11:29 IST. Two
more runs held the GPU lock on the same day, each in a fresh process:

1. At 12:02:33 IST, the gpu-ceilings command ran the pinned Day 74 config for
   15 seconds.
2. At 12:02:48 IST, the `timing` command ran the pinned checklist config for
   115 seconds. The run used 113 seconds of its budget of 300 seconds.

Then the `bound` command and the `verify` command with all inputs ran on the
CPU. Each command took about 4 seconds. The timing run caught 0 Python
warnings, and the capture began before the import of torch.

The `uptime` command printed 17.24, 15.91 and 16.75 before the ceiling run. It
printed 14.53, 15.38 and 16.54 after that run, which was also before the timing
run. It printed 11.37, 13.82 and 15.77 after the timing run. The reports
recorded these load averages:

| Report | Before | After |
| --- | --- | --- |
| checklist, 11:29 IST | 19.21, 21.17, 23.01 | 16.63, 20.39, 22.64 |
| gpu-ceilings, 12:02 IST | 17.24, 15.91, 16.75 | 15.19, 15.52, 16.60 |
| one-call timing, 12:02 IST | 14.53, 15.38, 16.54 | 11.37, 13.82, 15.77 |

| Report | SHA-256 |
| --- | --- |
| checklist | `130322a7b46db6bc7e9e9bf135a39225e2c139097cd167b74ceb22fd54b2e062` |
| gpu-ceilings | `1061242d74c4abb028693219415db4ff853ffe536c25235f239e37bacfe86480` |
| one-call timing | `5a953eb4afa1ec487302862f150194e2f057f82516d16adf4a4f12e3a3d6a252` |
| bound | `a5bb01db93ab23aa7d8560d8656fd2ec74536118f96eb073b8407b57d12ce227` |

The FMA ceiling was 4835.1 GFLOP/s, from 32 chains of 8192 fma steps. The
bandwidth ceiling was 104.31 GB/s, from `stream_copy` on a buffer of 256 MiB.
Thus the ridge point was 46.35 FLOPs for each byte. The ceiling report sets
`fma_scales_with_work` to true. The Day 76 gap run measured 4836.8 GFLOP/s and
124.58 GB/s at lower loads.

The table gives the range of each value over the 4 sweep points. The names
column counts the points of each Day 75 name.

| Side | FMA share | Modeled share | Points above the modeled roof | Day 75 names |
| --- | --- | --- | --- | --- |
| `matmul_x_columns` | 0.061 to 0.081 | 11.37 to 14.95 | 4 | 1 latency or occupancy-bound, 3 indeterminate |
| `matmul_x_rows` | 0.024 to 0.037 | 4.46 to 6.85 | 4 | 2 latency or occupancy-bound, 2 indeterminate |
| `matmul_tiled_32` | 0.067 to 0.072 | 0.394 to 0.420 | 0 | 4 latency or occupancy-bound |
| `matmul_s2_coarse_4x1` | 0.127 to 0.153 | 0.736 to 0.903 | 0 | 3 latency or occupancy-bound, 1 indeterminate |
| `matmul_s3_block_4x4` | 0.190 to 0.276 | 0.569 to 0.836 | 0 | 1 latency or occupancy-bound, 3 indeterminate |
| `matmul_s4_vec4` | 0.252 to 0.324 | 0.753 to 0.964 | 0 | 2 latency or occupancy-bound, 2 indeterminate |
| `matmul_s5_pad4` | 0.267 to 0.320 | 0.799 to 0.966 | 0 | 1 latency or occupancy-bound, 3 indeterminate |
| `matmul_s6_group_8x8` | 0.237 to 0.327 | 1.39 to 1.93 | 4 | 4 indeterminate |
| `matmul_s6_group_32x32` | 0.190 to 0.249 | 0.293 to 0.366 | 0 | 4 indeterminate |
| `matmul_s7_interior` | 0.271 to 0.378 | 0.811 to 1.103 | 3 | 1 latency or occupancy-bound, 3 indeterminate |
| `torch.matmul` | 0.465 to 0.728 | no model | no model | 1 latency or occupancy-bound, 3 indeterminate |

The step table sets the modeled change of each step beside the measured
change. Each range covers the 4 sweep points.

| Step | Kernel | Device bytes over the base | Load operations over the base | Threadgroup reads for each fma | Ratio over the base | Verdicts |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | `matmul_x_columns` | 1.000 | 1.000 | none | 1.753 to 3.296 | 4 faster |
| S1 | `matmul_tiled_32` | 0.031 to 0.032 | 0.031 | none to 2 | 0.889 to 1.142 | 2 slower, 1 no clear difference, 1 faster |
| S2 | `matmul_s2_coarse_4x1` | 1.000 | 1.000 | 2 to 1.25 | 1.807 to 2.163 | 4 faster |
| S3 | `matmul_s3_block_4x4` | 0.502 to 0.512 | 0.500 to 0.504 | 1.25 to 0.5 | 1.463 to 1.896 | 3 faster, 1 no clear difference |
| S4 | `matmul_s4_vec4` | 1.000 | 0.250 | 0.5 to 0.5 | 1.153 to 1.350 | 4 faster |
| S5 | `matmul_s5_pad4` | 1.000 | 1.000 | 0.5 to 0.5 | 0.981 to 1.060 | 4 no clear difference |
| S6 | `matmul_s6_group_8x8` | 1.954 to 1.992 | 1.984 to 2.000 | 0.5 to 0.5 | 0.886 to 1.038 | 4 no clear difference |
| S6 | `matmul_s6_group_32x32` | 0.504 to 0.520 | 0.500 to 0.505 | 0.5 to 0.5 | 0.712 to 0.779 | 3 slower, 1 no clear difference |
| S7 | `matmul_s7_interior` | 1.000 | 1.000 | 0.5 to 0.5 | 1.015 to 1.182 | 2 faster, 2 no clear difference |

### What the bound run shows

These results come from the run above. Each statement about a limit is an
inference from timings, measured ceilings and a model:

- No kernel came near the FMA ceiling. The highest FMA share of a kernel was
  0.378, for S7 at 4096 cubed. `torch.matmul` reached 0.465 to 0.728.
- The modeled intensity of each kernel is below the ridge point of 46.35. The
  highest one is 31.51, for the 32 x 32 group kernel at 4096 cubed. Thus the
  model alone calls each kernel memory bound.
- The rates do not agree with that call. The 8 x 8 group kernel has about 2
  times the modeled device bytes of S5. But the rule found no clear difference
  between them at each point. It ran 1.39 to 1.93 times as fast as its
  modeled roof, so caches probably served part of its loads.
- The 32 x 32 group kernel has about half of the modeled device bytes of S5,
  but it was slower at 3 points. S4 has the modeled bytes of S3, but it was
  faster at each point.
- S4, S5 and S7 ran at 0.753 to 1.103 of their modeled roofs. The controls
  above show that this closeness to the bandwidth roof probably does not set
  their rates.
- The changes that helped reduced operations more than modeled bytes. S2 cut
  the threadgroup reads for each fma from 2 to 1.25, and S4 cut the device
  load operations to a quarter. S7 removed guard instructions. Only S3 cut
  both the device bytes and the threadgroup reads.
- At the median rates, the tiled kernels made 230.1 to 465.9 modeled
  threadgroup reads in each nanosecond. At a stride of 1, the bank kernel of
  the same ceiling run made 779.3 reads in each nanosecond. If the model
  holds, the kernels thus used 0.30 to 0.60 of that rate.

The run cannot tell which limit applies. Possible limits are the issue rate of
the instructions, the latency of threadgroup reads and barriers, and the
registers and occupancy of each group. Only hardware counters can show which
one applies.

### The Day 75 names

The rule named 16 of the 44 points `latency or occupancy-bound`. That name is
a residual. It says that no measured roof explains the rate. It does not prove
that latency or occupancy is the cause.

The rule named the other 28 points `indeterminate`, at step 1. At these
points, the checklist run took 1.055 to 1.283 times the one-call host time of
the timing run for each call. The tolerance is 5 percent. At all 44 points,
the checklist run was slower than the timing run, by 1.3 to 28.3 percent. The
checklist run had load averages near 20, and the timing run had load averages
near 14.5. The two times can thus differ by more than the tolerance.

No point got the name `memory-bound`, `compute-bound` or `dispatch-bound`.
`torch.matmul` reached 0.701 and 0.728 of the FMA ceiling at 2048 and 4096
cubed. These shares are above the near-roof share of 0.7 of the rule, but
step 1 gave `indeterminate` at both points.

### Limits of the bound

- The three reports come from separate runs, at different loads. A name can
  change when the load changes.
- The reports record no host name and no GPU model. Thus the `bound` command
  cannot show that one host made all three. It compares only the software and
  platform tokens.
- An MPS event interval holds the signal latency of its events. Thus
  `median_gpu_ns` is longer than the kernel.
- The traffic comes from a model that counts no cache. The lab did not
  measure bytes.
- A measured ceiling is a rate that the kernels of the ceiling lab reached. It
  is not a vendor peak and not a hardware limit. The bank kernel reads with
  other instructions than the matmul kernels, so its rate is only a reference.

The repository does not keep the four reports of this run. The tests keep one
short timing report from 2026-11-20 at 12:17 IST, with digest
`f4d25ead6ecd6a00c1433aa9440a4fca723d4cb565abf9e56fe2c281bdd95a94`. It times
the recorded checklist config of the tests. With the recorded checklist and
gpu-ceilings reports, it makes one measured bound with digest
`10c4012f0db032f37e5c3edd504c6f719ebff534578f91e77205b99eb3fbb9e2`. A host
without torch can build this bound again and check it.

### Evidence boundary of the bound

A measured bound report has these true fields: `apple_gpu_metal_this_host`,
`host_specific` and `inference_from_timings`. These fields are false:
`hardware_counters_used`, `nsight_used`, `instruments_used`,
`occupancy_measured`, `vendor_peak_used`, `traffic_measured`,
`cuda_kernel_executed`, `nvidia_gpu_used`, `cublas_measured`,
`distributed_claim`, `production_claim` and `portable_performance_claim`.

The results come from one Apple GPU on one shared host. They do not measure or
predict CUDA, NVIDIA, cuBLAS, distributed or production performance. They are
not evidence of lecture viewing, completed exercises or learner understanding.

## Sources

- GPU MODE lectures, https://github.com/gpu-mode/lectures: Lecture 8, "CUDA
  Performance Checklist", and Lecture 2, "Recap Ch. 1-3 from the PMPP book".
- NVIDIA, CUDA C++ Best Practices Guide v13.5, for coalesced access to global
  memory, shared memory and its banks, occupancy, and branching and
  divergence. Section 10.2.3.2 gives shared memory in matrix multiplication.
- NVIDIA, Nsight Compute documentation, for achieved occupancy and the roofline
  chart.
- W. W. Hwu, D. B. Kirk and I. El Hajj, Programming Massively Parallel
  Processors, 4th ed., Morgan Kaufmann, 2022, chapters 4 to 6, for control
  divergence, occupancy, tiling, coalescing and thread coarsening.
- S. Williams, A. Waterman and D. Patterson, "Roofline: an insightful visual
  performance model for multicore architectures", Communications of the ACM
  52(4), 2009.
- Apple, Metal Shading Language Specification, for the `threadgroup` address
  space, `threadgroup_barrier`, SIMD groups and `float4`:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader`, and
  `torch.mps.Event`.
