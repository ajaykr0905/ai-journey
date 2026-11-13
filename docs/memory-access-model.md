# GPU memory-access model and CPU matmul baseline

This lab models how one warp's memory requests turn into global-memory sectors
and shared-memory bank wavefronts, applies those models to a shared-tile
transpose and to shared-memory tiled matmul, and classifies tile traffic on a
declared roofline. Separately, it measures real float32 matmul baselines on the
CPU of the host that runs it. There is no GPU on the development host. Nothing
here executes a CUDA kernel, and no number in the analytical report is a
measurement.

## Commands

```bash
ai-journey-memory-access analyze \
  --config config/day-74-memory-access-plan.json \
  --output /tmp/day-74-memory-access-analysis.json
ai-journey-memory-access verify \
  --report /tmp/day-74-memory-access-analysis.json \
  --config config/day-74-memory-access-plan.json

ai-journey-memory-access benchmark \
  --config config/day-74-memory-access-plan.json \
  --output /tmp/day-74-memory-access-measurement.json
ai-journey-memory-access verify \
  --report /tmp/day-74-memory-access-measurement.json \
  --config config/day-74-memory-access-plan.json
```

`python -m ai_journey.memory_access_cli` accepts the same arguments. Exit code
0 means success, 1 means a readable report failed verification, and 2 covers
usage errors, unreadable or invalid inputs, and refused outputs. `--config` on
`verify` is optional; with it, the report must have been built from that exact
plan.

Plans are read from regular non-symlink files of at most 64 KiB, reports of at
most 1 MiB. Duplicate JSON keys, `NaN`/`Infinity`, unknown or missing fields,
and booleans posing as integers are rejected. Reports are canonical JSON
(sorted keys, compact separators, ASCII escapes) with a SHA-256 digest over all
other fields. The digest detects drift and tampering; it is not a signature and
cannot prove that a timing was observed. Writers self-verify, publish through a
fsynced temporary file and `os.replace`, and refuse an output that aliases the
plan through a resolved path, symlink, or hardlink.

## Declared geometry

| Quantity | Value | Source |
| --- | --- | --- |
| Warp | 32 lanes, consecutive thread IDs | Programming Guide 12.6, 4.1 |
| Global sector | 32 bytes | Best Practices Guide 13.5, 10.2.1 |
| Line | 128-byte aligned segment | Best Practices Guide 13.5, 10.2.1; Programming Guide 12.6, 5.3.2 |
| Shared memory | 32 banks of 4-byte words | Best Practices Guide 13.5, 10.2.3.1; Programming Guide 12.6, 16.4.3 |
| Access sizes | 1, 2, 4, 8, 16 bytes, naturally aligned | Programming Guide 12.6, 5.3.2 |

The geometry is a fixed declared contract; a plan that changes any value is
rejected. Natural alignment means each access lies inside one sector and one
line.

## Models

**Warp patterns.** Generators emit per-lane byte addresses for contiguous,
strided, offset-base, broadcast, row and column walks over a row-major array
with an explicit leading dimension, and a seeded permutation. The permutation
uses a SHA-256 counter stream rather than `random`, so a seed is portable. An
active-lane mask makes lanes inactive; inactive lanes request nothing.

**Coalescing.** On compute capability 6.0 and newer, a warp's accesses
coalesce into the number of 32-byte transactions needed to serve its active
lanes, and a partly used sector is still transferred in full. The model
reports sectors, 128-byte lines, requested bytes (every active lane), unique
bytes, transferred bytes (sectors x 32), and efficiency (unique / transferred).

**Shared-memory banks.** Successive 32-bit words map to successive banks.
Lanes that touch any byte of the same word do not conflict: reads are served
by broadcast, and broadcasts across banks coalesce into one multicast. The
conflict degree is the largest number of distinct words any bank must serve,
which is also the number of serialized wavefronts. Loads and stores count the
same way; which lane's store wins is undefined and not modeled. 1-, 2- and
4-byte accesses are modeled exactly. 8- and 16-byte shared-memory accesses are
rejected, because neither guide states how they split into bank requests on
compute capability 5.x and newer.

**Tile transpose.** A 32 x 32 shared tile is written by rows and read by
columns. Per tile the model counts row-write and column-read wavefronts for any
leading dimension. For an N x N matrix (N a multiple of 128 elements, so every
warp request is identically aligned) it compares a naive kernel that stores
output columns with the tiled kernel that stores output rows.

**Tiled matmul traffic.** For C[M, N] = A[M, K] B[K, N], a naive kernel loads
2 M N K operand elements with no cache reuse modeled. A T x T thread block
loads 2 T^2 ceil(K/T) tile slots per block over ceil(M/T) ceil(N/T) blocks.
Slots outside A or B in partial edge tiles are padded slots that a
bounds-checked kernel zero-fills without a global load, so only useful loads
are traffic. Both kernels store M N elements and do 2 M N K useful FLOPs. The
model also reports shared bytes per block (2 T^2 element bytes), phases,
threads per block (T^2, so T <= 32 for the 1024-thread block limit), and the
reuse factor over naive loads.

**Roofline classification.** Naive, tiled, and compulsory traffic become
`roofline.KernelWorkload`s and are classified with `roofline.analyze_roofline`.
The compulsory workload (`roofline.matmul_workload`) reads each operand once.
Lower-bound times are roofline arithmetic, not runtimes.

**Inner-loop banks.** Thread (tx, ty) of a T x T block has ID tx + ty T. In
inner-loop step i a lane reads A_tile[ty][i] and B_tile[i][tx]. The model
counts wavefronts for every warp and step for a row-major A tile, a row-major
B tile, a column-major B tile, and a column-major B tile padded to T + 1.

## Pinned analytical results

The checked-in plan declares 16 warp patterns, 4 transposes, and a float32
sweep over T in {8, 16, 32} for 1024^3, 1000^3, and 4096^3 matmuls.

- Contiguous float32 needs 4 sectors and 1 line at 100% efficiency. A 4-byte
  base offset needs 5 sectors (80%), stride 2 needs 8 (50%), and strides of 8
  or more need 32 (12.5%). A broadcast moves 1 sector. A half-warp mask halves
  the request to 2 sectors. A permuted span still needs 4 sectors.
- As shared-memory requests, word stride 2 is 2-way conflicted, stride 8 is
  8-way, stride 32 is 32-way, and stride 33 is conflict-free.
- A 33-column float32 row starts 4 bytes off a sector boundary (5 sectors);
  padding the leading dimension to 64 restores 4 sectors.
- The unpadded 32 x 32 float32 tile needs 1,024 column-read wavefronts per tile
  (32-way); a leading dimension of 33 needs 32. For float16, one element of
  padding still leaves 48 wavefronts (2-way); one 32-bit word (two elements)
  removes every conflict.
- For a 4096 x 4096 float32 transpose, naive column stores need 32 sectors per
  request and transfer 536,870,912 bytes; tiled row stores need 4 and transfer
  67,108,864.
- In the inner loop, A-tile reads are broadcasts and B-tile row reads are
  conflict-free. A column-major B tile is 2-way conflicted at T = 8, 8-way at
  T = 16, and 32-way at T = 32; padding it to T + 1 removes the conflicts.
- The declared ceilings give a ridge of 9.5635 FLOPs per byte. Naive traffic
  sits at about 0.25 FLOPs per byte. Tiled traffic rises to about T / 4: 7.877
  for T = 32 at 1024^3, still below the ridge, so all nine tiled estimates are
  memory-bound. Compulsory traffic at 1024^3 is compute-bound at 170.67. At
  1000^3, T = 16 and T = 32 carry 2,024,064 and 3,108,864 padded slots.

## CPU matmul baseline

`benchmark` runs the plan's float32 cases (128^3, 256^3, 512^3 with NumPy and
torch on CPU, 2 warmups and 7 repeats, seed 74). Warmups are untimed. Each
repeat times one matmul call between two `time.perf_counter_ns` readings, and
result checks run after the clock stops. Every warmup and measured result must
lie elementwise within (gamma_K(2^-24) + gamma_K(2^-53)) (|A| |B|) of a float64
reference, where gamma_K(u) = K u / (1 - K u), the classical inner-product
rounding bound. The report keeps raw samples and nearest-rank statistics
(sorted samples[ceil(p n / 100) - 1], median = p50, IQR = p75 - p25, no
interpolation), FLOPs (2 M N K), GFLOP/s at the median and best sample, input
digests, and the runtime: Python, NumPy and torch versions, platform system and
machine, `os.cpu_count()`, torch threads, and torch float32 matmul precision.
No host name, user name, path, or serial number is recorded.

CI runs the installed `benchmark` on
`config/day-74-memory-access-ci-smoke.json` to keep the job short. That plan
equals the pinned plan except for its name and a benchmark of one 32^3 case per
backend with 1 warmup and 3 repeats, so its timings are a smoke check, not a
baseline.

Every timing is a wall-clock CPU measurement on the host that produced the
report. Timings vary with load, threads, and BLAS builds, and they say nothing
about GPU performance. Verification cannot recompute them; it checks schemas,
the plan binding, the CPU-only boundary, statistics recomputed from raw
samples, FLOP and GFLOP/s arithmetic, correctness bounds, and summaries.

The `benchmark` command always times with `time.perf_counter_ns`. In Python,
`main`, `run_benchmark` and `measure_matmul` also accept an injected clock so
tests can script durations. A report timed that way records timer `injected`,
`measured: false`, scope `synthetic_injected_clock` and a boundary with
`cpu_measurement_this_host: false`, and its lines print as `synthetic_timing`.
`verify` accepts it only as synthetic timing, never as a CPU measurement, and
rejects any report whose timer, `measured` flag, scope and boundary disagree.

## Verification

`verify` recomputes every analytical number without the model modules: pattern
lane addresses from their declared parameters; sectors, lines, and banks from
the bytes each lane touches; transposes and inner loops by enumerating lane
addresses; tile traffic by clipping each partial block and phase; roofline
times from the declared ceilings. It then rebuilds the report from its embedded
plan and requires byte-identical canonical JSON.

## Sources

- NVIDIA, CUDA C++ Best Practices Guide v13.5, sections 10.2.1 (coalesced
  access, including 10.2.1.1, 10.2.1.2 and 10.2.1.4), 10.2.3.1 (shared memory
  and memory banks), 10.2.3.2 and 10.2.3.3 (shared memory in matrix
  multiplication, padded transposed tile):
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA, CUDA C++ Programming Guide v12.6, sections 2.2, 4.1, 5.3.2, 16.2,
  16.4.3, 16.5.3 and 16.6.4:
  https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html
- Declared roofline ceilings: NVIDIA A100 specifications table, A100 80GB SXM
  column, FP32 19.5 TFLOPS and GPU memory bandwidth 2,039 GB/s, read as decimal
  units (accessed 2026-11-13): https://www.nvidia.com/en-us/data-center/a100/.
  These are datasheet peaks for non-tensor-core FP32 math, declared as inputs;
  they were not detected or measured.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 3.1 (inner-product rounding bound used by the CPU check).

## Exclusions and boundaries

The models exclude L1 and L2 caching (hits, evictions, and reuse between
requests or thread blocks), hardware prefetching, TLB and address translation,
64-bit shared-memory bank modes and 8- or 16-byte shared-memory accesses,
asynchronous global-to-shared copies such as `cp.async` and TMA, tensor cores,
register blocking, occupancy, launch geometry, and latency hiding. The
analytical report does not prove GPU availability, kernel performance, or
production readiness. The measurement report is CPU evidence from one host
only. Neither report is evidence of lecture viewing, completed exercises, or
learner understanding.
