# Naive matmul model lab

This lab is the model half of the Day 74 lab. The day pairs GPU MODE Lecture
4, `Intro to Compute and Memory Architecture`, with an exercise: "Write a naive
matmul kernel. Measure its GFLOPs. Note how far below peak it is." The lab
writes the naive kernel as a per-thread Python function. The Day 73 launch
simulator runs it thread by thread on the CPU.

The lab then counts the global memory requests of each warp, the FLOPs and the
bytes. It puts the kernel on a roofline at given ceilings, and it estimates the
occupancy of the CUDA version. No GPU runs, and the lab compiles no CUDA source
and measures no time. The lab imports only the Python standard library, so it
runs without torch or NumPy.

This half measures no GFLOP/s. In this lab, "peak" means a ceiling that a run
measured on this host. This half measured no ceiling, so it states no peak. Its
roofline function takes the ceilings as inputs, and a later run can give it
ceilings measured on this host.

## Commands

Run the pinned plan, then check the report:

```bash
PYTHONPATH=src python -m ai_journey.matmul_model_cli run \
  --config config/day-74-matmul-model.json \
  --output /tmp/day-74-matmul-model-report.json

PYTHONPATH=src python -m ai_journey.matmul_model_cli verify \
  --report /tmp/day-74-matmul-model-report.json \
  --config config/day-74-matmul-model.json
```

`run` prints one line:

```text
matmul_model_report=6fac533806b4746c46cca614bad20d9cd04eda07bb8a8d7eacd76951fc40ebb8 runs=16 runs_matching_reference=16 order_independent_launches=4/4 cross_checked_requests=13900 probe_traces=8/8 memory_bound_points=10/16 cuda_checks=16/16 simulated_on_cpu=true gpu_executed=false cuda_compiled=false
```

`verify` prints the same line with the prefix `verified_matmul_model_report=`.
The exit codes are:

- `0`: the command succeeded.
- `1`: the command read or built a report, but the report failed verification.
- `2`: the arguments are wrong, the plan is invalid, the report file cannot be
  read, or the command refused the output path.

`--config` is optional for `verify`. With it, `verify` also requires that the
report embeds that exact plan. `run` refuses an output path that names the
plan, also through a symlink or a hard link.

## Kernel

`matmul_model_kernels` writes the exercise kernel in the per-thread style of
the Day 73 kernels:

```python
def naive_matmul_kernel(t, a, b, c, m, n, k):
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    _guarded_dot(a, b, c, row, col, m, n, k)


def _guarded_dot(a, b, c, row, col, m, n, k):
    if row < m and col < n:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + col]))
        c[row * n + col] = total
```

`A` is `m x k`, `B` is `k x n` and `C` is `m x n`. All three are row-major
float32 matrices. The thread takes `row` from y and `col` from x. The guard
tests both axes, so a thread outside `C` touches no memory. The second kernel,
`naive_matmul_rows_kernel`, takes `row` from x and `col` from y. Each element
of `C` then gets the same sum in the same order.

The run helper sizes the grid with `ceil_div` on each axis. It runs the
kernel in the canonical order and in each seeded permuted order of the plan.
No thread writes an element that another thread reads or writes. Thus every
order gives the same bits.

## Reference and summation order

`matmul_model_reference` makes `A` and `B` from the SHA-256 counter stream of
Day 73, with the stream names `matmul-a` and `matmul-b`. Each value is a
float32 value of either sign with a magnitude in [2^-8, 2^8). The reference
computes each element of `C` with plain loops and without the simulator. It
applies this rule:

1. Set the sum to +0.0.
2. For `i` = 0, 1, ..., `k - 1` in turn, round `A[row, i] * B[i, col]` to
   float32.
3. Add that product to the sum, and round the sum to float32.

Each rounding goes to the nearest float32 value, ties to even. Python forms
each product and each sum in double before `to_float32` rounds it. A product
of two float32 values is exact in double. Double has 53 bits, and 53 is at
least 2 * 24 + 2. Thus a sum that rounds to double and then to float32 is the
correctly rounded float32 sum (S. A. Figueroa, "When is double rounding
innocuous?", 1995).

The verifier rounds with integers instead, and it gets the same bits for each
pinned element. A fused multiply-add rounds once for each step, and another
order rounds other partial sums. Either can give other bits, so the order is
part of the rule.

The reference also rounds each exact dot product once, and it counts the
elements where the stated order gives other bits:

| Case | Elements | Elements that differ | Largest difference |
| --- | --- | --- | --- |
| `ragged`, 37 x 45 x 29 | 1,665 | 1,145 | 787 ulps |
| `aligned`, 32 x 64 x 16 | 2,048 | 1,231 | 224 ulps |

The terms have both signs, so some dot products cancel to a small value. A
small value has a small ulp. Thus a small absolute difference can be many ulps.

## Warp access model

A warp holds 32 consecutive linear thread IDs of one block, x fastest. The
model requires `blockDim.x` to be a multiple of 32. Thus each warp is 32
consecutive x-threads with one `threadIdx.y`. In loop step `i`, a warp loads
one element of `A` and one element of `B` in each lane. After the loop, it
stores one element of `C` in each lane.

The lane addresses follow the index expressions of the kernel:

- `A[row, i]` is at byte `(row * k + i) * 4`.
- `B[i, col]` is at byte `(i * n + col) * 4`.
- `C[row, col]` is at byte `(row * n + col) * 4`.

The first element of each matrix is at byte 0 of its own allocation. The
runtime aligns each allocation to at least 256 bytes. A lane that the guard
stops issues no address. For each request, the model counts the distinct
32-byte sectors and 128-byte lines. It gives each request one class:

- `broadcast`: every active lane reads one address.
- `coalesced`: the active lanes, in lane order, read consecutive 4-byte
  elements.
- `strided`: any other pattern.

The model has no cache. It counts each request alone, as if no cache holds a
sector between requests. The memory_access lab counts the same way.

For the pinned plan, the block is 32 x 8. The table gives the totals of the
`aligned` case, 32 x 64 x 16, with 64 full warps in each mapping:

| Mapping | A loads | B loads | C stores | Sector bytes |
| --- | --- | --- | --- | --- |
| x over columns | 1,024 broadcasts, 1 sector each | 1,024 coalesced, 4 sectors each | 64 coalesced, 4 sectors each | 172,032 |
| x over rows | 1,024 strided, 32 sectors each | 1,024 broadcasts, 1 sector each | 64 strided, 32 sectors each | 1,146,880 |

When x runs over columns, the B loads coalesce and the A loads are
broadcasts. When x runs over rows, the A loads and the C stores are `strided`
requests, and the B loads are broadcasts. Both mappings request the same
270,336 bytes. With x over rows, the sectors move 20/3 times as many bytes.

The model divides the FLOPs by the sector bytes of each launch:

| Case | x over columns | x over rows |
| --- | --- | --- |
| `ragged` | 0.291943 FLOP per byte | 0.057417 FLOP per byte |
| `aligned` | 0.380952 FLOP per byte | 0.057143 FLOP per byte |

The report lists every step of two probe warps for each launch: the warps of
`C[0, 0]` and of `C[m - 1, n - 1]`. In the `ragged` case, `n` is 45, so each
row of `B` is 180 bytes. These rows do not align to 32 bytes. For the warp of
`C[0, 0]` with x over columns, step 1 gives these requests:

- `A`: all 32 lanes read byte 4, one broadcast in 1 sector and 1 line.
- `B`: the lanes read bytes 180 to 304, coalesced, in 5 sectors and 2 lines.

With x over rows, the same step reads `A` at bytes 4, 120, 236 and so on.
The lanes of that request are 116 bytes apart, in 32 sectors and 29 lines.
The warp of `C[36, 44]` keeps 13 lanes with x over columns and 5 lanes with x
over rows.

## Cross-checks

Three checks compare the access model with other code:

- The memory_access lab builds each request again from its matrix layouts and
  its `row_access`, `column_access` and `broadcast_access` generators. Its
  `analyze_coalescing` then counts the sectors and lines. All 13,900 requests
  of the pinned plan agree.
- The SIMT lab's coverage model counts the full, partly masked and fully
  masked warps of each launch. Its counts equal the counts of the access
  model.
- `matmul_model_trace` runs the kernel functions, lane by lane, with buffers
  that record each index in program order. Each probe warp loads and stores
  what the model states, in step order.

The tests show failures too. A model that moves one address fails the first
check. A kernel with a reversed loop or a wrong `B` index fails the trace.

## FLOP and byte accounting

`matmul_model_traffic` counts 2 * m * n * k FLOPs: one multiply and one add
for each term of each dot product. Two declared models count the global
traffic:

- No reuse: each thread that the guard admits loads its `k` elements of `A`
  and its `k` elements of `B`. Then it stores its one element of `C`. The
  bytes are 4 * (2 * m * n * k + m * n).
- Ideal reuse: each matrix moves once. The bytes are 4 * (m * k + k * n +
  m * n).

The intensity of a model is its FLOPs divided by its bytes. Without reuse,
the intensity is k / (4 * k + 2). This value is below 1/4 for every shape. No
kernel moves fewer bytes to and from global memory than the ideal-reuse count.
Thus the ideal-reuse intensity is the upper bound for every kernel of a shape.
For a square shape of size `s`, the ideal-reuse intensity is s / 6.

| Shape | FLOPs | No-reuse intensity | Ideal-reuse intensity |
| --- | --- | --- | --- |
| 37 x 45 x 29 | 96,570 | 0.245763 | 5.971432 |
| 32 x 64 x 16 | 65,536 | 0.242424 | 4.571429 |
| 1024 x 1024 x 1024 | 2,147,483,648 | 0.249878 | 170.666667 |
| 4096 x 4096 x 4096 | 137,438,953,472 | 0.249969 | 682.666667 |

The first two shapes are the simulated cases. The plan declares the last two
for the accounting only. Each model counts requested elements, not sectors.
The access model shows that a strided warp moves a 32-byte sector for each
4-byte element. Thus the sector bytes of a poor mapping can be more than the
no-reuse bytes. Both models agree with `NaiveMatmulTraffic` of the
memory_access lab and with `roofline.matmul_workload`.

## Roofline

`matmul_model_roofline.roofline(intensity, ceilings)` takes a compute ceiling
in GFLOP/s and a bandwidth ceiling in GB/s. Both units count 10^9 per second.
It returns these values:

- The attainable rate: the smaller of the compute ceiling and the intensity
  times the bandwidth ceiling.
- The ridge point: the compute ceiling divided by the bandwidth ceiling.
- The bound: `memory` below the ridge point, `compute` above it, and `ridge`
  on it.
- The fraction of the compute ceiling that the attainable rate reaches.

The code holds no hardware number. The plan gives each ceiling pair as a
labeled decimal string. The pinned plan declares two pairs, `declared-ridge-4`
and `declared-ridge-16`. Each pair has a bandwidth ceiling of 1 GB/s and a
compute ceiling of 4 or 16 GFLOP/s. These values are model inputs, not
measurements, and they describe no device.

With a bandwidth ceiling of 1 GB/s, the attainable rate below the ridge point
equals the intensity. Without reuse, the naive kernel is memory-bound at both
ridge points. It reaches less than 1 / (4 * ridge point) of the compute
ceiling:

| Shape | Ridge point 4 | Ridge point 16 |
| --- | --- | --- |
| 37 x 45 x 29 | 0.061441 | 0.015360 |
| 32 x 64 x 16 | 0.060606 | 0.015152 |
| 1024 x 1024 x 1024 | 0.062469 | 0.015617 |
| 4096 x 4096 x 4096 | 0.062492 | 0.015623 |

With ideal reuse, every shape is compute-bound at ridge point 4. At ridge
point 16, the two small shapes stay memory-bound. They reach 0.373215 and
0.285714 of the compute ceiling. The two large shapes are compute-bound.

## Occupancy

`matmul_model_occupancy` applies the occupancy rules of the SIMT lab to the
CUDA version of the kernel. It uses the record for compute capability 8.0 of
that lab. A test requires that the record repeats each launch limit of Day 73.
The plan declares these inputs:

- 256 threads per block, from the 32 x 8 block of the plan.
- One register count per thread: 32, 40, 64 or 128.
- 0 bytes of static and dynamic shared memory.

The lab does not compile the kernel, so it cannot read the register count of a
compiler. Each count is an assumption. The 8.0 record reserves 1 KB of shared
memory for each block. Thus shared memory alone admits 164 blocks, and it never
sets the limit.

| Registers per thread | Registers per warp | Blocks per SM | Warps per SM | Occupancy | Limit |
| --- | --- | --- | --- | --- | --- |
| 32 | 1,024 | 8 | 64 | 1.000000 | warps, registers |
| 40 | 1,280 | 6 | 48 | 0.750000 | registers |
| 64 | 2,048 | 4 | 32 | 0.500000 | registers |
| 128 | 4,096 | 2 | 16 | 0.250000 | registers |

These values are theoretical occupancy from the rules of an occupancy
calculator. They are not achieved occupancy, and they do not show speed.

## CUDA source

`matmul_model_cuda.cuda_source` writes CUDA C++ text for the kernel and its
host launcher. The text uses the launch math of the simulator. It gives one
thread to each element of `C`, takes `row` from y and `col` from x, and guards
both axes. Each grid axis is `cdiv(extent, block)`.

The loop body is
`sum = __fadd_rn(sum, __fmul_rn(a[row * k + i], b[i * n + col]));`. CUDA never
merges these two intrinsics into a fused multiply-add. Thus the text keeps the
roundings of the reference.

The text is source for a later run on an NVIDIA GPU. This lab does not compile
or run it. The development host has no NVIDIA GPU, no CUDA toolkit and no
`nvcc`.

`check_cuda_source` reads the text without a compiler. It uses the helpers of
the Day 73 checker. It runs 8 named checks:

- `cdiv_definition`: `cdiv` returns `(a + b - 1) / b`.
- `matmul_signature`: the `__global__` signature occurs once, with the
  expected parameter types.
- `matmul_index`: the `row` expression on y and the `col` expression on x.
- `matmul_guard`: each memory access is inside `if (row < m && col < n)`.
- `matmul_loop`: the loop is `for (int i = 0; i < k; ++i)`.
- `matmul_accumulate`: the sum is `0.0f` before the loop, and the loop body
  is the pinned `__fadd_rn` statement.
- `matmul_store`: `c[row * n + col] = sum;` follows the loop.
- `matmul_grid`: the launcher divides `n` and `m` by the block with `cdiv`.
  The check evaluates the `dim3` expressions for the extents of each case. It
  requires the launch configuration of the simulator.

Each function must hold exactly its pinned statements. The text must hold the
three definitions and no other code. Thus a macro that redefines `__fadd_rn`
fails every check. The checker reads comments as a compiler does. Text with a
backslash, a `??` trigraph or an unterminated `/*` comment fails every check.
For each check, a mutation test shows that the check rejects its defect.

The pinned text has the SHA-256
`a97031329a75820abe14a8cd5a3a55dbe728bbedd64871a79f8d5e2e7b815fc0`. The report
holds the full text.

## Report and verification

`run` writes canonical JSON (sorted keys, compact separators and ASCII
escapes) with a SHA-256 seal over all other fields. The seal detects drift. It
is not a signature. The report holds no floating-point value. Each ratio is an
exact fraction with a decimal string, rounded half to even to 6 places. The
report holds these sections:

- `plan`, `limits`, `rule` and `geometry`: the declared inputs, the named
  limits, the summation rule and the warp geometry.
- `cases`: for each case, the input digests, the reference digest, the
  rounding counts and one section for each mapping. A mapping section holds the
  launch, one row for each order, the access totals, the cross-checks and the
  probe warps.
- `accounting`: the FLOPs, both traffic models and the roofline points of
  each shape.
- `occupancy`: the 8.0 record and one estimate for each register count.
- `cuda_source`: the text, its digest and the checks for each case.
- `summary` and `evidence_boundary`.

`verify` trusts no number in the report. It does these steps:

1. Check the field set, the seal and the fixed evidence boundary.
2. Parse the embedded plan. With `--config`, require that plan.
3. Compute the case, accounting, occupancy and summary sections again with its
   own code, and name the first field that differs. Its code makes the input
   streams again, rounds with integers and encodes the float32 bits. It counts
   every warp's addresses, sectors and lines. It computes the roofline in
   fractions and the occupancy with its own copy of the 8.0 values.
4. Read the CUDA text with `check_cuda_source`, and compare the limits, the
   rule, the geometry and the 8.0 record with the lab constants.
5. Require that the report equals a fresh build, byte for byte.

The reader accepts only regular non-symlink files: plans of at most 64 KiB and
reports of at most 1 MiB. It rejects duplicate keys, `NaN`, unknown or missing
fields, and booleans in place of integers. A simulated case has at most 64
loop steps. Thus the report of each valid plan stays under the 1 MiB limit.

## Pinned results

The pinned plan `config/day-74-matmul-model.json` uses the input seed 74 and
the permuted order seeds 1, 2 and 3. It declares two cases:

- `ragged`, 37 x 45 x 29. With x over columns, the grid is 2 x 5 blocks and
  the launch has 2,560 threads: 1,665 useful and 895 idle. With x over rows,
  the grid is 2 x 6 blocks and the launch has 3,072 threads: 1,665 useful and
  1,407 idle.
- `aligned`, 32 x 64 x 16. Both mappings launch 2,048 threads and no idle
  thread.

In all 16 runs, the output matches the reference bit for bit. No run has an
out-of-bounds access or a race. All 8 probe traces match the model, and all 16
CUDA checks pass. The reference digests are
`41d50935c365946c5bce24e34eeb75d2e1589ab7ff729920591204c344950cd8` for
`ragged` and `5d3baafab74b128d9c5039142a041e2e0c6b4d855e7706e703688c046867c35f`
for `aligned`. The report digest is
`6fac533806b4746c46cca614bad20d9cd04eda07bb8a8d7eacd76951fc40ebb8`.

## Boundaries

The simulator runs one thread at a time. It does not model warp scheduling,
divergence, caches or timing. The access model counts requests without a
cache, so it gives no DRAM traffic of a real GPU. A GPU with L1 and L2 caches
can serve many of these sectors from a cache.

The lab measures no GFLOP/s. The roofline uses declared ceilings, and the
occupancy uses declared register counts. Neither is a measurement of a device.
The lab does not compile or run the CUDA text, so it gives no NVIDIA, CUDA,
production or distributed evidence. The report is not evidence of lecture
viewing, a completed exercise or learner understanding.
