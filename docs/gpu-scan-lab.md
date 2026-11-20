# Parallel scan on the Apple GPU with Metal

Day 78 of the curriculum names GPU MODE Lecture 20, "Scan Algorithm". Ajay must
still watch Lecture 20. Ajay must still compare the kernels of the lecture with
the three kernels of this lab.

This lab writes three float32 inclusive scans in the Metal Shading Language.
It checks each scan against a float64 reference with a declared bound for each
element. One run then times the scans, `torch.cumsum` and a copy kernel on the
same input. Each scan has a hypothesis, and the repository received each
hypothesis before the first timed run.

Nothing here runs a CUDA kernel or uses an NVIDIA GPU. The SIMD-group functions
of S2 are the Metal counterpart of CUDA warp shuffles, not the same
instruction. PyTorch does not report which kernel `torch.cumsum` runs. The lab
reads no hardware counters. Every timing is a host-specific Apple GPU (Metal)
measurement at the load averages that the report records.

## Commands

```bash
python -m ai_journey.gpu_scan_cli run \
  --config config/day-78-gpu-scan.json \
  --output /tmp/day-78-gpu-scan.json
python -m ai_journey.gpu_scan_cli verify \
  --report /tmp/day-78-gpu-scan.json \
  --config config/day-78-gpu-scan.json
```

The parser uses the program name `ai-journey-gpu-scan`. The package does not
install a console script for this lab, so run the module with `python -m`.

The `run` command compiles the kernels, checks them, times them, checks its own
report and writes the report. Then it prints these lines:

1. One line for each hypothesis of the plan, with the kernel, its base and the
   predicted verdicts at each pinned length. The lines come from the plan that
   the report seals.
2. One line with the load averages before and after the run.
3. One line for each kernel that its pipeline refused, with the reason.
4. One line for each correctness case, with the largest relative error of each
   side. The line also tells which bits each side held and if its runs gave
   the same bits.
5. One line for each sweep point, with the median GB/s and the IQR range of
   each side.
6. For each sweep point, one line for each kernel of the plan. The line gives
   its three ratios, the fraction of the copy rate and the result of the
   hypothesis. Then one line with the fastest kernel at that point.
7. One line with the count of Python warnings that the run caught.
8. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`gpu_scan_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `gpu_scan_synthetic_timing=` instead.

The `verify` command calculates a report again on the CPU and prints the same
summary line with the prefix `verified_`. With `--config`, `verify` also
requires that the report comes from that exact config.

The exit codes are the codes of the Day 73 to 77 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

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
other tests use a NumPy stand-in for the GPU and one short measured report from
the development host. The Linux CI job `nvidia-preflight-contracts` collects
these tests, because their names start with `test_gpu`.

## The kernels

Each kernel gives one threadgroup one block of 1024 elements. It writes the
inclusive prefix sums of the block and writes the block total to a second
buffer:

- `scan_s0_hillis_steele`: 1024 threads with one element each run the 10
  Hillis-Steele steps. Step k adds the value 2^k slots back. Each step reads one
  threadgroup buffer and writes the other, so one barrier for each step is
  enough.
- `scan_s1_blelloch`: 512 threads with two elements each run the up-sweep and
  the down-sweep of Blelloch in one threadgroup buffer. Element j then takes
  the exclusive sum of element j + 1, so no extra add rounds the result.
- `scan_s2_simd`: each SIMD group of 32 threads scans its 32 elements with
  `simd_prefix_inclusive_sum`. SIMD group 0 scans the 32 totals in the same
  way and moves them up one lane with `simd_shuffle_up`. Each thread then adds
  the total of the SIMD groups before its own.

A longer input runs as a multi-block scan in levels:

1. The kernel scans each block of the input and writes each block total.
2. The same kernel scans the block totals. While the totals fill more than one
   block, it scans them in blocks again.
3. The add kernel `scan_add_offsets` adds the scanned sum of the earlier blocks
   to each element of a later block, from the top level down.

A scan of N elements thus makes 1 dispatch when N is 1024 or less. It makes 3
dispatches up to 2^20 elements and 5 dispatches up to 2^24 elements.

These rules apply to every kernel:

- Each thread loads its value or 0, so a ragged last block scans like a full
  block. Only the stores have guards.
- Each barrier is the line `threadgroup_barrier(mem_flags::mem_threadgroup);`
  alone, in the kernel body or in a loop with the same bounds in every thread.
  No barrier sits under a condition on the thread. A kernel with a barrier has
  one function and no early return. It does not use `break`, `continue` or
  `goto`, and no loop body can change its loop variable. Thus every thread
  reaches every barrier. The function `uniform_barriers` reads each kernel
  text and refuses any other barrier.
- Each scalar reaches its kernel as a 32-bit integer through `arg_casts`, as in
  the Day 73 to 77 Metal labs.

| Kernel | Threads in a group | Elements for each thread | Threadgroup bytes | Barriers in a block | Declared block depth | CPU replay |
| --- | --- | --- | --- | --- | --- | --- |
| `scan_s0_hillis_steele` | 1024 | 1 | 8192 | 11 | 10 | yes |
| `scan_s1_blelloch` | 512 | 2 | 4096 | 22 | 19 | yes |
| `scan_s2_simd` | 1024 | 1 | 128 | 2 | 62 | no |

The block depth is the largest count of float32 additions on a path from an
input to an output of one block. A max-plus model in the tests calculates the
depth of S0 and S1 from their orders. Metal does not state the order of a
SIMD-group prefix sum. Thus S2 declares the worst order: 31 adds in its SIMD
group, 30 adds for the earlier totals and 1 last add. The bound assumes that
the SIMD-group function adds each value once.

The pipeline of each kernel reports its limits. A pipeline refuses a kernel in
three cases. It allows fewer threads than the group of the kernel, or its
static threadgroup memory differs from the declared bytes. S2 also needs a
SIMD width of 32. The report records each refusal, and a refused kernel never
runs.

Each digest covers the text of one kernel. The library is the two-line header
of the Day 76 lab, then the three scans and the add kernel. The copy reference
is the `stream_copy` kernel of the Day 74 ceiling source, unchanged.

| Source | SHA-256 |
| --- | --- |
| `scan_s0_hillis_steele` | `a8db0486becd559294f413bc7f32b71d62502bb874e1fd1da4d6d3cb102249fe` |
| `scan_s1_blelloch` | `d2087a1a14eb30c7b2b4da1e3646fdabe6ab5b948ca8c70b57a80c76b7840f85` |
| `scan_s2_simd` | `ba41e335c10dd9dc5910678ffe6fa64449baae4856228f2c2250ad87be729dab` |
| `scan_add_offsets` | `1151ad028c5dce620771ed8a5b31d4a728e0e2d751809a3b72c8adcd1d55b131` |
| scan library | `39df7e35f24dcf5fd41ef7553563cc75eb5fab2ec9a501e192457d11ff5be1c5` |
| Day 74 ceiling source | `735583b6a215ae98d3092c68ab7f3e5f6eeaf7ca4bcc5bae32827447cecac008` |

## The plan

The module `gpu_scan_plan.py` declares one hypothesis for each kernel. Each
hypothesis names the base of its kernel and predicts one or two verdicts of
the Day 76 ratio rule at each pinned length. The pinned lengths are 2^16, 2^20,
2^24 and 5000003. The length 5000003 leaves a partial block at each of its
three levels.

The plan also binds the digest of each kernel text. Its digest is
`13f7eead88f966d08046b82eaa2693e6956feac22558321f5d65d256ffafec66`. The pinned
config and each report seal that digest, so a config for other hypotheses or
other kernel texts cannot run.

Before the plan entered the repository, two untimed scripts ran scans on the
development GPU. Neither script read a clock or wrote a report.

- The first script ran a prototype text of the three scans and the add
  kernel. It ran each scan and `torch.cumsum` once on normal values at each
  of 9 lengths from 1 to 1048577. S0 and S1 held the bits of their CPU replays
  at each length. S2 held the bits of none of the four orders that the script
  tried for it, except at 1 element.
- The second script ran the library text of this lab. It ran each side three
  times on 20 inputs: normal values and small integers at each of 10 lengths
  from 1 to 1048577. Each scan gave the same bits in its three runs, kept its
  sentinel tail and stayed within its bound. S0 and S1 held the bits of their
  CPU replays. Each side gave the exact sums of each integer input.
  `torch.cumsum` gave other bits from run to run on the normal inputs of 65537
  and 1048577 elements.

The tests of the plan commit also compiled the library on that GPU, but they
ran no kernel. The plan uses one fact from the second script: `torch.cumsum`
gave other bits from run to run at 65537 and 1048577 elements. The three scans
kept their bits.

| Step | Kernel | Base | 2^16 | 2^20 | 2^24 | 5000003 |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | `scan_s0_hillis_steele` | `torch.cumsum` | slower | slower | slower | slower |
| S1 | `scan_s1_blelloch` | `scan_s0_hillis_steele` | no clear difference | no clear difference, slower | no clear difference, slower | no clear difference, slower |
| S2 | `scan_s2_simd` | `scan_s0_hillis_steele` | no clear difference | faster | faster | faster |

The plan states the change and the hypothesis of each step in these words.

S0:

- "Scan each block of 1024 elements with the 10 Hillis-Steele steps in two
  threadgroup buffers, with one barrier at each step. Then scan the block sums
  with the same kernel and add the sum of the earlier blocks to each element."
- "PyTorch does not report which kernel torch.cumsum runs. An untimed check
  before this plan ran each side three times. torch.cumsum gave other bits
  from run to run at 65537 and 1048577 elements, and the three scans of this
  lab did not. Sums in an order that changes from run to run can come from a
  single-pass scan. Such a scan reads and writes each element once, in one
  dispatch. S0 moves about 16 bytes for each element in 3 or 5 dispatches. On
  Day 73, the median time of torch.add was 0.85 to 1.24 times that of a Metal
  vector add. Thus S0 is slower than torch.cumsum at every length."

S1:

- "Scan each block with 512 threads in one threadgroup buffer: an up-sweep and
  a down-sweep of 10 levels each, with a barrier at each level. Element j takes
  the exclusive sum of element j + 1, and the last element takes the block
  total."
- "S1 makes about 2 adds for each element, and S0 makes about 9. But S1 waits
  at 22 barriers in each block, and S0 waits at 11. At each level, the lanes
  of S1 read words 2 to 512 words apart. On Day 74, threadgroup reads at
  strides of 2 to 32 words took 2.0 to 21.5 times as long as reads at stride 1.
  Both kernels make the same dispatches and move the same modeled bytes. Thus
  S1 is not faster than S0 at any length, and at 2^16 the run shows no clear
  difference."

S2:

- "Scan each run of 32 elements with simd_prefix_inclusive_sum. SIMD group 0
  scans the 32 run totals and moves them up one lane with simd_shuffle_up.
  Each thread adds the total of the runs before its own, with 2 barriers in
  each block."
- "SIMD-group functions are the Metal counterpart of CUDA warp shuffles, not
  the same instruction. S2 replaces the 10 threadgroup steps and 11 barriers of
  S0 with two SIMD-group scans and 2 barriers. At 2^16, the dispatches probably
  set the time, so the run shows no clear difference. S0 makes about 31
  threadgroup accesses for each element. If each access takes the Day 74 time
  of 1.3 ps for a stride-1 threadgroup read, S0 spends about 42 us on them at
  2^20. Its 16.8 MB of modeled traffic take 168 us at the Day 74 copy ceiling
  of about 100 GB/s. Thus S2 is faster than S0 by 1.1 or more at 2^20, 2^24 and
  the ragged length."

## Correctness

Each case draws its input from the seed with `numpy.random.default_rng([seed,
0, index])`. The pinned config has 11 cases and runs each side 3 times on each
case. Two patterns test two properties:

- `standard_normal`: float32 normal values. Each output must stay within the
  bound of its order. The CPU replays the float32 additions of S0 and S1 in
  their orders. The run records if the GPU held the bits of the replay.
- `small_integers`: whole numbers from -4 to 4, at most 2^22 of them. Each
  partial sum is an integer of at most 2^24 in size, so float32 holds it
  exactly. Thus every order of additions gives the exact prefix sums, and each
  output must equal them bit for bit.

The bound of element i is `c * sum(|x[j]|, j <= i)` with `c = (gamma_D(2**-24) +
gamma_N(2**-53)) / (1 - gamma_N(2**-53))` and `gamma_k(u) = k u / (1 - k u)`.
D is the rounding depth of the kernel at length N. A scan of L levels has
`D = L * block_depth + L - 1`, because each add of a scanned sum adds one more
addition. The second term covers the error of the float64 reference, and the
last factor covers the float64 rounding of the sum of sizes.

Each output buffer starts as sentinel values. The buffer is longer than the
data by the spare slots of the last block and by one more block. Three rules
stop the run when a kernel breaks one:

- Each run of a kernel keeps its sentinel tail.
- Each element of each run stays within the bound.
- On an integer case, each run equals the exact prefix sums.

These properties are measurements, and they never stop the run:

- `deterministic`: all runs of a side gave the same bits.
- `replay_bits`: the first run of S0 or S1 held the bits of its CPU replay. The
  record also counts the elements that differ. S2 has no replay.
- The largest relative error of `torch.cumsum` and its exact bits on each
  integer case. PyTorch does not report the order of `torch.cumsum`, so it has
  no declared bound. Only a value that is not finite stops the run.

The config parser requires cases that run a ragged block at each level:

- a length below 1024 that is not a multiple of 32
- a length of two or more levels that is not a multiple of 1024
- a length of three levels with a partial block at level 2
- a case of each pattern, and an integer case of two or more levels

## Timing

The benchmark times five sides on the same input at each sweep point: the
three scans, `torch.cumsum(x, 0, out=y)` with a preallocated output, and the
Day 74 `stream_copy`. The copy moves `ceil(N / 4)` float4 values from a
zero-padded copy of the input, 4 for each thread in groups of 256. The method
is the method of the Day 74 to 77 labs:

1. Each side runs 2 samples that are not timed.
2. Each timed sample calls `torch.mps.synchronize()` and reads
   `time.perf_counter_ns`.
3. The sample then makes a batch of calls, calls `torch.mps.synchronize()`
   again, and reads the clock again. One call of a scan queues every dispatch
   of one scan.
4. The first side of repeat r is side r mod 5, so the order moves by one side
   at each repeat. Each side has 11 timed samples.
5. After the last sample, the CPU checks the latest output of each side. A
   kernel must stay within its bound, and the copy must hold its input.

A batch holds the fewest whole calls whose compulsory bytes reach 2^30 bytes.
The compulsory bytes of a scan are 8 bytes for each element: one float32 read
and one float32 write. Every side of a point uses the same batch. The GB/s of a
side divides the bytes of one sample by the median sample time in nanoseconds.
The scans and `torch.cumsum` use 8 N bytes for each call. The copy uses its own
32 bytes for each float4.

Each point also records the modeled bytes of one scan with no cache:

- Each level reads and writes its elements and writes its block totals.
- Each add reads and writes the elements after the first block, and it reads
  each earlier block total once.

The three scans share these modeled bytes, about 16 bytes for each element.
The model is not a measurement.

One budget of 300 seconds covers the correctness cases and the sweep. The run
stops between two samples after the budget ends, and it then writes no report.

## The ratio rule

For each sweep point and each kernel, the report gives these values from the
same run:

- the median GB/s and the range from the IQR of the samples
- the ratio to the base of the kernel in the plan, and to S0 and
  `torch.cumsum`
- the fraction of the copy rate at that point
- the modeled fraction: the modeled bytes of one scan over the median time,
  over the copy rate
- the result of the hypothesis at that point

A ratio divides two median rates of the same point. The rule of the Day 76 lab
calls the first side faster when two conditions are true:

- The ratio is 1.1 or more. The config sets this minimum.
- The 75th percentile time of the first side is less than the 25th percentile
  time of the second side.

The rule calls the first side slower when the same test is true in the other
direction. Otherwise the result is `no_clear_difference`. The vendor side has
the label "vendor GPU scan: torch.cumsum on MPS, PyTorch 2.15.0". The label
names the call and the PyTorch version, not a kernel. The copy is a reference
for the device traffic of the same run, and it takes part in no verdict.

## One run on the development host

This run used the pinned config on 2026-11-21 from 02:19:16 to 02:19:34 IST.
The host was a shared Darwin arm64 host with an Apple GPU, macOS 27.2, CPython
3.11.14, NumPy 1.26.4 and PyTorch 2.15.0. The environment did not set
`PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. The run held the GPU lock
that the timed GPU jobs on this host take. Other jobs ran on the host at the
same time, and the load averages show them.

The report recorded load averages of 10.46, 10.89 and 12.69 before the run. It
recorded 11.51, 11.12 and 12.74 after the run.

The run took 18 seconds, with the torch import. Its report has about 106 KB.
The report digest is
`9bb0b555d905767aeaed647e86e4df78c74217db5c243823679bf0e0397fb8e6`. The values
are host-specific Apple GPU (Metal) measurements from one run, not portable
figures.

### Correctness results

Each pipeline allowed 1024 threads and reported a SIMD width of 32. The static
threadgroup memory of each pipeline was equal to the declared bytes, so the run
refused no kernel. These are the results of the 11 cases:

- Each run of each kernel kept its sentinel tail and stayed within its bound.
  The largest error was 0.101 of the bound, for S0 at 31 elements. For S1 it
  was 0.052 and for S2 it was 0.016, both at 1000 elements.
- S0 and S1 held the bits of their CPU replays in each of the 8 normal cases.
- Each side gave the exact sums in each of the 3 integer cases. That result
  includes `torch.cumsum`.
- Each kernel gave the same bits in all 3 runs of each case.
- `torch.cumsum` gave other bits from run to run in the normal cases of 65536,
  1048577 and 5000003 elements. Its largest relative error was 5.87e-08.

After the timing, each timed kernel output stayed within its bound, and the copy
held its input.

### Rates and ratios

Each rate is the median GB/s of 11 samples, with 8 bytes for each element. The
range in parentheses comes from the IQR of the sample times. A ratio cell gives
the ratio and the result of the rule. The copy cell gives the fraction of the
`stream_copy` rate at the same point.

2^16 elements, 2048 calls in each sample:

| Side | Median GB/s (IQR range) | Over its base | Over S0 | Over `torch.cumsum` | Of the copy rate | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 `scan_s0_hillis_steele` | 31.93 (31.62 to 32.73) | 0.812, slower | | 0.812, slower | 0.237 | held |
| S1 `scan_s1_blelloch` | 27.54 (26.79 to 27.78) | 0.862, slower | 0.862, slower | 0.700, slower | 0.205 | missed |
| S2 `scan_s2_simd` | 51.49 (48.93 to 52.79) | 1.612, faster | 1.612, faster | 1.309, faster | 0.383 | missed |
| `torch.cumsum` | 39.32 (36.56 to 41.20) | | | | 0.292 | |
| `stream_copy` | 134.47 (118.18 to 140.97) | | | | 1.000 | |

2^20 elements, 128 calls in each sample:

| Side | Median GB/s (IQR range) | Over its base | Over S0 | Over `torch.cumsum` | Of the copy rate | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 `scan_s0_hillis_steele` | 63.44 (54.64 to 66.09) | 2.714, no clear difference | | 2.714, no clear difference | 0.398 | missed |
| S1 `scan_s1_blelloch` | 72.46 (71.13 to 77.56) | 1.142, faster | 1.142, faster | 3.100, no clear difference | 0.455 | missed |
| S2 `scan_s2_simd` | 100.62 (94.38 to 105.06) | 1.586, faster | 1.586, faster | 4.304, faster | 0.631 | held |
| `torch.cumsum` | 23.38 (11.58 to 92.55) | | | | 0.147 | |
| `stream_copy` | 159.38 (146.97 to 165.37) | | | | 1.000 | |

2^24 elements, 8 calls in each sample:

| Side | Median GB/s (IQR range) | Over its base | Over S0 | Over `torch.cumsum` | Of the copy rate | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 `scan_s0_hillis_steele` | 42.98 (40.29 to 43.94) | 10.210, faster | | 10.210, faster | 0.537 | missed |
| S1 `scan_s1_blelloch` | 47.08 (45.92 to 48.18) | 1.095, no clear difference | 1.095, no clear difference | 11.183, faster | 0.588 | held |
| S2 `scan_s2_simd` | 47.78 (46.36 to 50.59) | 1.112, faster | 1.112, faster | 11.351, faster | 0.597 | held |
| `torch.cumsum` | 4.21 (3.93 to 6.27) | | | | 0.053 | |
| `stream_copy` | 80.00 (76.14 to 87.03) | | | | 1.000 | |

5000003 elements, 27 calls in each sample:

| Side | Median GB/s (IQR range) | Over its base | Over S0 | Over `torch.cumsum` | Of the copy rate | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- |
| S0 `scan_s0_hillis_steele` | 44.01 (40.91 to 44.65) | 8.667, faster | | 8.667, faster | 0.568 | missed |
| S1 `scan_s1_blelloch` | 44.58 (43.29 to 46.04) | 1.013, no clear difference | 1.013, no clear difference | 8.780, faster | 0.576 | held |
| S2 `scan_s2_simd` | 45.35 (44.17 to 45.91) | 1.030, no clear difference | 1.030, no clear difference | 8.931, faster | 0.586 | missed |
| `torch.cumsum` | 5.08 (4.46 to 5.68) | | | | 0.066 | |
| `stream_copy` | 77.42 (75.44 to 79.41) | | | | 1.000 | |

S2 was the fastest kernel at each point. The predictions held at 5 of 12
points: 3 kernels at 4 lengths each.

### What the hypotheses got right and wrong

- S0 and `torch.cumsum`: S0 was slower at 2^16, as the plan predicted. At 2^24
  and at 5000003, S0 was 10.2 and 8.7 times as fast as `torch.cumsum`. At 2^20,
  the ratio was 2.714, but the IQR range of `torch.cumsum` was 11.58 to 92.55
  GB/s, so the rule found no clear difference. The hypothesis expected a
  single-pass vendor scan, and it failed at 3 of 4 lengths.
- S1 and S0: S1 showed no clear difference at 2^24 and at 5000003, as
  predicted. It was slower at 2^16 and faster at 2^20, so the hypothesis failed
  at those two lengths.
- S2 and S0: S2 was faster at 2^20 and 2^24, as predicted. At 2^16, S2 was
  1.612 times as fast, so the dispatches did not set the whole time there. At
  5000003, the ratio was 1.030, so the rule found no clear difference.

At 2^16, one call of S0 took 16.4 us, and one call of S2 took 10.2 us. Both
kernels make the same 3 dispatches. Thus the block work of S0 probably caused
most of the difference of 6.2 us.

### Memory bound or not

The copy reference gives a ceiling for device traffic in the same run. It is
not a hardware limit. These comparisons use it:

- At 2^24 and at 5000003, the three scans reached 0.537 to 0.597 of the copy
  rate. Their modeled traffic is about 2 times their 8 N bytes. Their modeled
  rate was 1.08 to 1.20 times the copy rate.
- Thus the scans moved their modeled bytes at about the copy rate or faster.
  Device traffic probably set most of their time at these lengths. The modeled
  rates above 1 suggest that caches served part of the traffic, or that the
  copy does not reach the limit of the device. The run cannot tell which.
- At 2^20, S2 reached 0.631 of the copy rate, and S0 reached 0.398. The ratio
  of S2 over S0 was 1.586, so the block work set a large part of the S0 time
  at this length.
- `torch.cumsum` reached 0.053 of the copy rate at 2^24 and 0.066 at 5000003.
  Its 8 N bytes alone do not explain its time. PyTorch does not report its
  kernel, so the run cannot name its limit.

These names are inferences from timings and the copy rate of the same run. The
lab reads no hardware counters, so it observed no stall, cache hit, bank
conflict or occupancy.

### Limits of this run

- One run on one shared host gives each ratio. The load averages were near 11
  from other jobs.
- The copy rate changed with the length, from 77.42 to 159.38 GB/s. Thus each
  fraction applies to its own point only.
- At 2^20, the samples of `torch.cumsum` spread widely. Their IQR range covers a
  factor of 8.
- The in-order replay shows the bits of S0 and S1, but the SIMD-group order of
  S2 stays unknown. S2 thus has only its worst-order bound.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The report records the items in this list:

- the config and its digest, and the host environment without host names
- the declared plan with every hypothesis, the kernel digests and the plan
  digest
- the two MPS switches and the load averages before and after the run
- the Python warnings of the run, with the home directory as `~`
- the digest, limits and verdict of each kernel, and the limits of the add and
  copy kernels
- the correctness cases with the digest of each run, the raw samples, the
  medians, the IQRs and the checks
- the model bytes, the ratio rows, the result of each hypothesis and the
  evidence boundary

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk, and moves it into place with `os.replace`. It
refuses an output that aliases the config. The readers accept only regular
files that are not symbolic links, and they refuse duplicate JSON keys.

The verifier uses its own CPU code, not the code of the run. It also uses three
modules of the run that measure nothing: the config parser, the kernel
registry and the declared plan. It does not import torch. It does these checks:

1. It draws each input again from the seed and sums it again in float64.
2. It replays S0 as whole shifted rows and S1 as a list of level sums. The run
   uses other code for the same orders, so the two forms check each other.
3. It rebuilds the digest of each output that holds the replay bits or the
   exact sums, with its sentinel tail. When the first run holds the replay
   bits, it calculates the error of that run again.
4. It derives each pipeline verdict, thread grid, guard tail and bound again.
5. It requires the sealed plan to equal the declared plan, byte for byte.
6. It calculates each statistic, rate, ratio, verdict and result of a
   hypothesis again from the raw samples.

The recorded error of an output is the largest error of all its runs. When the
first run holds the replay bits, this error must equal the error of that run
if all runs gave the same bits. If a later run gave other bits, this error
must not be less than the error of the first run.

An output with other bits in its first run cannot be rebuilt. For that output,
the verifier requires a different digest, a count of 1 or more differing
elements and an error within the bound. The timed outputs have no digest.
Thus the verifier checks only the fields and the bound of each check after the
timing. It also checks that a side with the S0 bits has the S0 error. The
timings cannot be observed again, so the verifier checks them for structure
only. The tests change each section of a sealed report, seal it again and
require the verifier to refuse it.

The repository does not keep the report of the pinned run. The tests keep one
short report of a recorded config from 2026-11-21 at 02:16 IST, with digest
`2517c7825d88ce1994d4944004dbd5df9f74c45f49ed1bda236b668851c1b3bb`. A host
without torch can check it with the verifier. Its rates mostly show the cost
of the synchronize calls.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 78 | 0 to 2^32 - 1 |
| `plan_sha256` | the digest of the declared plan | only that digest |
| `correctness.cases` | 11 cases from 1 to 5000003 elements | 1 to 32 different [length, pattern] pairs that run a ragged block at each level |
| `correctness.runs` | 3 | 2 to 5 runs of each side |
| `benchmark.lengths` | 2^16, 2^20, 2^24 and 5000003 | 1 to 16 different lengths of 1 to 2^24 |
| `benchmark.warmup` | 2 | 0 to 10 samples that are not timed |
| `benchmark.repeats` | 11 | 3 to 50 timed samples for each side |
| `benchmark.target_bytes_per_sample` | 1073741824 | 1 KiB to 1 GiB, with at most 4096 calls in a sample |
| `benchmark.max_runtime_seconds` | 300 | 1 to 600 |
| `comparison.min_faster_ratio` | 1.1 | more than 1.0 and at most 4.0 |

The config parser counts the planned work before a run. The pinned config has
11 cases with 7169903 elements and 132 correctness runs. It also has 4 sweep
points, 220 timed samples, 143715 calls and 279579697800 bytes. These counts
include each kernel, so they are an upper limit when a pipeline refuses a
kernel.

## Evidence boundary and exclusions

The evidence boundary of a measured report holds true or false fields and one
statement. The fields `apple_gpu_metal_this_host` and `host_specific` are true.
These fields are false: `cuda_kernel_executed`, `nvidia_gpu_used`,
`warp_shuffle_executed`, `vendor_kernel_known`, `vendor_peak_claimed`,
`hardware_counters_used`, `occupancy_measured`, `distributed_claim`,
`production_claim` and `portable_performance_claim`.

The statement gives these facts:

- The timings are wall-clock timings on the Apple GPU of the host that made
  the report, at the recorded load averages.
- The scan kernels and the Day 74 copy came from `torch.mps.compile_shader`, and
  `torch.cumsum` ran on the MPS device.
- PyTorch does not report which kernel `torch.cumsum` ran. Thus the vendor
  number names the call, not a kernel.
- The SIMD-group functions of S2 are the Metal counterpart of CUDA warp
  shuffles, not the same instruction.
- The lab read no hardware counter, so a bound name is an inference from the
  timings and the copy rate of the same run.
- No CUDA kernel ran, and the lab used no NVIDIA GPU. Each ratio compares two
  sides of one run on this host.

The lab does not record the GPU model, the clock frequency, the thermal state
or the other jobs on the host. The rates count the compulsory bytes of a scan,
not the traffic of the hardware. The results come from one Apple GPU on one
host. They are not evidence of CUDA, NVIDIA, warp-level, distributed or
production performance. They are not evidence of lecture viewing, completed
exercises or learner understanding.

## Sources

- GPU MODE lectures, https://github.com/gpu-mode/lectures: Lecture 20, "Scan
  Algorithm".
- W. D. Hillis and G. L. Steele, "Data Parallel Algorithms", Communications of
  the ACM 29(12), 1986, for the step-by-step scan of S0.
- G. E. Blelloch, "Prefix Sums and Their Applications", Carnegie Mellon
  University report CMU-CS-90-190, 1990, for the up-sweep and down-sweep of S1.
- W. W. Hwu, D. B. Kirk and I. El Hajj, Programming Massively Parallel
  Processors, 4th ed., Morgan Kaufmann, 2022, chapter 11, for the Kogge-Stone,
  Brent-Kung and hierarchical scans.
- Apple, Metal Shading Language Specification, for `threadgroup_barrier`,
  `simd_prefix_inclusive_sum` and `simd_shuffle_up`:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader`, and
  the documentation of `torch.cumsum`.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 4.2, for the summation bound `gamma_D`.
