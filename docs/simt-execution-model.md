# SIMT execution-model analysis

The Day 71 lab turns declared CUDA architecture limits, declared product specs,
and declared kernel shapes into exact launch, indexing, occupancy,
latency-hiding, wave, and roofline numbers. It is an analytical calculator:
no GPU, CUDA kernel, or latency was executed or measured to produce any of
these numbers, so none of them is achieved occupancy, bandwidth, or run time.

## Reproduce the pinned report

```bash
ai-journey-simt build \
  --config config/day-71-simt-execution-plan.json \
  --output /tmp/day-71-simt-report.json

ai-journey-simt verify \
  --report /tmp/day-71-simt-report.json
```

Until the console script is installed, `PYTHONPATH=src python -m
ai_journey.simt_cli` accepts the same arguments. Both commands print one line
with the same SHA-256 digest, `devices=3 sweep_cells=40 memory_rows=18
arithmetic_rows=12`, and `measured=false gpu_executed=false`. They return 0 on
success, 1 for an invalid plan, report, or path, and 2 for usage errors.

`build` loads a bounded plan that rejects duplicate JSON keys, NaN, symlinks,
and unknown or missing fields at every level, then refuses to write the report
over the plan, including through a symlink or hard link. Reports are canonical
JSON (sorted keys, compact separators, ASCII) published through a temporary
file, `fsync`, and `os.replace`. `verify` checks the fixed evidence boundary,
the exclusions, and the digest; requires the declared hardware to equal the
pinned source-backed records; re-derives every derived field with its own
integer and fraction arithmetic, naming the first mismatching path; and finally
compares the report with a fresh build. The digest detects drift; it is not a
signature.

## Declared hardware and sources

Architecture limits per compute capability come from the NVIDIA CUDA C++
Programming Guide v12.6 (archive 12.6.3): Table 21 (Technical Specifications
per Compute Capability), Table 4 (64 FP32 results per clock per SM on 7.x and
8.0), and sections 16.6 and 16.7 (4 warp schedulers per SM, shared memory
capacities, 48 KB default and opt-in per-block limits, and the 1 KB per-block
system reservation on 8.0). Allocation rules come from `cuda_occupancy.h` in the
CUDA 12.6 toolkit (cuda_cudart 12.6.77): registers are allocated per warp in
256-register units across 4 sub-partitions, and shared memory in 256-byte
units on 7.x and 128-byte units on 8.x.

| Capability | Threads / warps / blocks per SM | Registers per SM, block, thread | Shared memory capacities (KB) | Per-block default / opt-in / reserved |
|---|---|---|---|---|
| 7.0 | 2048 / 64 / 32 | 65,536 / 65,536 / 255 | 0, 8, 16, 32, 64, 96 | 48 KB / 96 KB / 0 |
| 7.5 | 1024 / 32 / 16 | 65,536 / 65,536 / 255 | 32, 64 | 48 KB / 64 KB / 0 |
| 8.0 | 2048 / 64 / 32 | 65,536 / 65,536 / 255 | 0, 8, 16, 32, 64, 100, 132, 164 | 48 KB / 163 KB / 1 KB |

Product specs come from NVIDIA datasheets (FP32 peak and DRAM bandwidth) and
architecture whitepapers (SM count and boost clock). A device is rejected
unless SMs x FP32 lanes x 2 x boost clock reproduces its datasheet FP32 figure
within the figure's published rounding.

| Device | SMs | Boost clock | Derived FP32 | Datasheet FP32 | DRAM bandwidth | Ridge point |
|---|---:|---:|---:|---:|---:|---:|
| Tesla V100 SXM2 (7.0) | 80 | 1530 MHz | 15.6672 TFLOP/s | 15.7 TFLOPS | 900 GB/s | 17.408 FLOP/byte |
| T4 (7.5) | 40 | 1590 MHz | 8.1408 TFLOP/s | 8.1 TFLOPS | 300 GB/s | 27.136 FLOP/byte |
| A100 SXM4 80GB (8.0) | 108 | 1410 MHz | 19.49184 TFLOP/s | 19.5 TFLOPS | 2,039 GB/s | 9.5595 FLOP/byte |

Sources: NVIDIA V100 Tensor Core GPU datasheet (JAN20) and Tesla V100 GPU
Architecture whitepaper WP-08608-001_v1.1, Table 1; NVIDIA T4 Tensor Core GPU
datasheet (MAR19) and Turing GPU Architecture whitepaper WP-09183-001_v01,
Table 5; NVIDIA A100 Tensor Core GPU datasheet (JUN21) and A100 Tensor Core GPU
Architecture whitepaper V1.0, Table 4. That A100 table describes the SXM4 board
in its 40 GB form; the 80 GB SXM datasheet column lists the same 19.5 TFLOPS
FP32 figure, which the derived check reproduces from those SM and clock
values. The T4 datasheet lists 300 GB/sec while the Turing whitepaper lists
320 GB/sec; the plan declares the datasheet figure, and every T4
bandwidth-derived number scales linearly with that choice. H100
SXM5 is not pinned: Table 3 of NVIDIA's "NVIDIA Hopper Architecture In-Depth"
technical blog (March 2022) lists its boost clock as not finalized, and the
H100 product page lists FP32 and bandwidth but no clock, so its derived FP32
peak cannot be checked.

Latencies are declared inputs. Guide section 5.2.3 states that most arithmetic
instructions take typically 4 clock cycles on compute capability 7.x, so 4
warp schedulers need 16 independent instructions in flight, and that off-chip
memory latency is typically hundreds of clock cycles. The plan therefore
tabulates arithmetic latency only for 7.0 and 7.5 and uses 400 and 800 SM
cycles as memory sensitivity points inside that range, not as device
latencies. Register budgets and kernel byte and FLOP counts are labeled
planning inputs; nothing was compiled.

## Modeled rules

- **Launch geometry.** Blocks obey x, y <= 1024, z <= 64, and the device's
  threads-per-block limit; grids obey x <= 2^31-1 and y, z <= 65535. Grids are
  sized by ceil division. Warps hold consecutive thread IDs (`x + y * Dx`), so
  a block that is not a warp multiple leaves idle lanes in its last warp.
- **Indexing and coverage.** 1D threads map to `blockIdx.x * blockDim.x +
  threadIdx.x`; 2D threads map to row-major `row * width + column`. The report
  counts covered and uncovered elements, threads a bounds guard masks, warps
  that are fully active, partially masked, or fully masked, and the writes an
  unguarded kernel would send beyond the buffer or onto another row's elements.
  Tests check these closed forms against brute-force thread enumeration.
- **Occupancy.** Following `cudaOccMaxActiveBlocksPerMultiprocessor`: warp
  slots, the block limit, registers (per-warp allocation rounded to 256, warps
  fitted per sub-partition, and the per-block check that rounds warps up to the
  sub-partition count), and shared memory (static + dynamic + reserved bytes
  rounded to the allocation unit, checked against the default or opt-in
  per-block limit, and divided into the configured carveout, which is raised to
  the smallest capacity that holds one block). Active blocks are the minimum
  limit, every tied resource is reported as limiting, and launches that cannot
  place one block are rejected with every failed check. Two deliberate
  differences from the header: register counts above the guide's 255 maximum
  are rejected (the header admits 256), and its compute capability 9.0 barrier
  limit is not modeled.
- **Latency hiding.** Little's law: each SM must keep `bandwidth x latency /
  (SMs x clock)` bytes in flight; warps needed are that divided by bytes per
  lane x active lanes x independent loads; the sustainable bandwidth fraction
  at a resident-warp count is capped at one. Arithmetic pipelines need
  `rate x latency` independent warp-instructions, with the FP32 rate at FP32
  lanes / warp size and the issue-bound rate at one instruction per scheduler
  per cycle.
- **Waves.** A wave runs SMs x resident blocks per SM; the report gives full
  waves, tail blocks, last-wave fill, and SM-slot utilization, assuming equal
  block run times.
- **Throughput.** Peaks, ridge points, and bottleneck classification reuse the
  roofline lab's `RooflineHardware`, `KernelWorkload`, and `analyze_roofline`.

## Pinned results

The 2^24-element float32 vector add launches 65,536 blocks of 256 threads with
no masked lanes. At the declared 16 registers it reaches 100% theoretical
occupancy on all three devices, limited by warp slots, and runs 103 waves on
the V100, 410 on the T4, and 76 on the A100 (99.4%, 99.9%, and 99.8% SM-slot
utilization). Its 1/12 FLOP per byte is memory-bound everywhere, with roofline
lower bounds of 223.7 us, 671.1 us, and 98.7 us; these are calculations, not
observed run times.

The 1920 x 1080 RGB-to-grayscale image with 16 x 16 blocks launches a 120 x 68
grid. Its 15,360 surplus threads fill 480 warps that the guard masks entirely,
with no partially masked warp. A floor-division grid of 120 x 67 would leave
15,360 pixels uncovered. At 1.25 FLOPs per byte the kernel is memory-bound on
every device.

In the occupancy sweep, 256-thread blocks on the A100 fall from 100% to 50%,
25%, and 12.5% theoretical occupancy at 32, 64, 128, and 255 registers. The T4
stays at 100% through 64 registers because its 32-warp limit binds first. Six
of the 40 cells cannot be resident; for example, 1,024 threads at 128
registers need 131,072 registers in one block.

At the declared 800-cycle point the A100 needs 10,711.85 bytes in flight per
SM. One 4-byte load per lane needs 84 warps, more than the 64 an SM holds, so
at most 76.5% of the declared bandwidth is sustainable; two independent loads
per lane (the vector add) need 42 warps and 16-byte loads need 21. Every V100
and T4 row stays within the warp limit. The arithmetic table reproduces the
guide's 16 warps for 4 cycles on 4 schedulers, and needs 8 warps for the FP32
pipeline at one independent instruction per warp or 2 warps at four.

A fixed grid of 1,024 blocks at 64 registers fits 4 blocks per SM on every
device: 3 waves and 79.0% SM-slot utilization on the A100, 4 waves and 80.0% on
the V100, and 7 waves and 91.4% on the T4.

## Boundaries

The model excludes partitioned global caching (cuda_occupancy.h applies it only
to compute capability 5.2, 5.3, 6.1 and 6.2), thread block clusters and
distributed shared memory, block-barrier limits, driver carveout selection
heuristics, cache effects, coalescing and DRAM efficiency below the declared
peak, instruction mix, divergence cost, scheduler policy, clock behavior other
than the declared boost clock, launch overhead, and concurrent kernels. It does
not prove GPU availability, achieved occupancy, measured bandwidth or latency,
production readiness, lecture viewing, a completed exercise, or learner
understanding.
