# Day 77 checklist CUDA contract

This lab is the CUDA half of Day 77. It applies two items of a CUDA
performance checklist to the Day 76 tiled matmul: thread coarsening and 2D
register blocking. It writes each kernel as CUDA C++ text for a later run on an
NVIDIA GPU.

This host has no NVIDIA GPU, no CUDA toolkit, no `nvcc`, no cuBLAS and no
Nsight. Thus the lab compiles, links and runs none of the texts. It checks the
texts without a compiler, and it runs the schedule of each kernel in a CPU
simulator. It also estimates a declared occupancy. The lab measures no GPU time,
and it writes no sealed report.

## Commands

Write the source files, and then check them again from the disk:

```bash
PYTHONPATH=src python -m ai_journey.gpu_checklist_cli sources \
  --output-dir /tmp/day-77-cuda-sources

PYTHONPATH=src python -m ai_journey.gpu_checklist_cli check \
  --source-dir /tmp/day-77-cuda-sources
```

`sources` writes five kernel files, the Day 76 cuBLAS file and a `SHA256SUMS`
file. It prints one line for each source file. `check` reads each source file
from the directory. It checks each file for the 18 shapes of
`config/day-74-matmul-metal.json`: the nine correctness cases and the nine
benchmark points. It prints one line for each file, as in this example:

```text
cuda_check=coarsened_matmul_t16_c4.cu sha256=a4a6110ff1be048cfa79490bf185a0a725a05249bc68a6aad482cc662c8bf74c pinned=true checks=198/198 shapes=18 compiled=false
```

`pinned=true` means that the file has its pinned digest. A new comment in a file
changes its digest, but the file can still pass each check. The exit codes are:

- `0`: the command succeeded.
- `1`: `check` read each file, but a file failed a check. The command names
  each failed check on standard error.
- `2`: the arguments are wrong, a file is missing, unreadable or not ASCII, or
  the command refused the output path.

No command needs a GPU, so no exit code marks a missing device. The package has
no console script for this lab, so the commands use `python -m`.

On the development host, `sources` took 0.81 s and `check` took 0.51 s of wall
time on the CPU. The 1-minute load average was 23.14 before and after the two
commands. These times describe one shared host and no GPU.

## Thread-coarsened kernel

`gpu_checklist_cuda.coarsened_cuda_source(kernel)` writes one kernel and its
host launcher. The coarsening factor F is `CoarsenedKernel.coarsening`. One
T x T block computes a T x (F * T) tile of `C = A @ B`. Each thread computes F
elements of one row of `C`, T columns apart. `A` is `m x k`, `B` is `k x n` and
`C` is `m x n`. All three are row-major float32 matrices.

The kernel runs `cdiv(k, T)` phases. Each phase loads one tile of `A` once and
uses it for F tiles of `B`. In each phase, each thread does these steps:

1. It loads one element of `A` into `As[ty][tx]`, or 0 outside `A`.
2. It does steps 3 to 5 for each of the F tiles of `B`.
3. It loads one element of `B` into `Bs[ty][tx]`, or 0 outside `B`. Then it
   waits at `__syncthreads()`, so each load is in shared memory before a
   thread reads a slot.
4. It applies `sum[j] = __fmaf_rn(As[ty][i], Bs[i][tx], sum[j])` while
   `i < min(T, k - phase * T)`.
5. It waits at `__syncthreads()` again, so no thread writes a slot that
   another thread still reads.

This is the loop over the tiles of `B` in the T = 16, F = 4 text:

```c
        for (int j = 0; j < 4; ++j) {
            int col = col0 + j * 16;
            Bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
            __syncthreads();
            for (int i = 0; i < min(16, k - phase * 16); ++i) {
                sum[j] = __fmaf_rn(As[ty][i], Bs[i][tx], sum[j]);
            }
            __syncthreads();
        }
```

The coarsening cuts the loads of `A`. In the Day 76 kernel, each of the
`cdiv(n, T)` block columns reads all of `A` once. Here, `cdiv(n, F * T)` block
columns read it. The loads of `B` do not change.

## 2D register-blocked kernel

`gpu_checklist_cuda.blocked_cuda_source(kernel)` writes one kernel and its
host launcher. One block computes a BM x BN tile of `C`, and each thread
computes a TM x TN micro-tile in registers. In each step of k, a thread reads
TM values of `A` and TN values of `B` from shared memory. Then it does TM * TN
fma operations. Thus the blocked kernels need fewer shared loads for each fma.

The kernel runs `cdiv(k, BK)` phases. Each phase stages a BM x BK tile of `A`,
transposed, and a BK x BN tile of `B` in shared memory. Each thread stages one
group of 4 consecutive floats of `A` and one of `B`. The text uses one `float4`
load for a group only if these two conditions are true:

- The row length is a multiple of 4: `k` for `A` and `n` for `B`.
- The base pointer is 16-byte aligned.

Then the element offset of each group is a multiple of 4, so the address is
16-byte aligned. Each group also stays inside its row. Otherwise the text uses
4 scalar loads, each with its own guard. A group outside its matrix gives 4
zeros, and the thread loads nothing for it. This is the load of `B` in the
64 x 64 text:

```c
        if (b_vector) {
            float4 v = (b_row < k && b_col < n)
                ? *reinterpret_cast<const float4*>(&b[b_row * n + b_col])
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            Bs[b_r][b_c] = v.x;
            Bs[b_r][b_c + 1] = v.y;
            Bs[b_r][b_c + 2] = v.z;
            Bs[b_r][b_c + 3] = v.w;
        } else {
            for (int q = 0; q < 4; ++q) {
                Bs[b_r][b_c + q] =
                    (b_row < k && b_col + q < n) ? b[b_row * n + b_col + q] : 0.0f;
            }
        }
```

The flag `b_vector` is `n % 4 == 0` and a 16-byte aligned `b`. `cudaMalloc`
aligns each address that it returns to at least 256 bytes. One barrier follows
the loads, and a second barrier follows the inner loop. Each thread stores the
elements of its micro-tile one at a time, each with the guard
`row < m && col < n`.

## Arithmetic, guards and digests

Both kernels keep the Day 76 arithmetic. Each sum of `C` is +0.0 before the
first phase. `__fmaf_rn` rounds each step once to float32, with ties to even,
and the terms come in increasing k. Each inner loop stops at the end of k, so no
padded slot enters a sum. That is the loop of `matmul_metal_reference.fma_replay`,
so each output must hold the bits of that replay.

Each global load has a guard with a zero fill, and each store has a guard. No
barrier is inside a guard, so every thread of a block reaches each barrier. The
texts use no `#pragma unroll`. They expect that `nvcc` unrolls the loops with a
constant trip count, so that the accumulator arrays stay in registers.

| Kernel | Block | C tile | Shared memory | SHA-256 of the text |
| --- | --- | --- | --- | --- |
| `coarsened_matmul_t16_c2` | 16 x 16 | 16 x 32 | 2,048 bytes | `5a57f3c70b5e10cd958d6b5b23e804cee6fe91bf5294ff87ae3cbc9b6b1d6ab7` |
| `coarsened_matmul_t16_c4` | 16 x 16 | 16 x 64 | 2,048 bytes | `a4a6110ff1be048cfa79490bf185a0a725a05249bc68a6aad482cc662c8bf74c` |
| `coarsened_matmul_t32_c2` | 32 x 32 | 32 x 64 | 8,192 bytes | `41e1bbc8274250492ab9316d6721822a982baa5a6cebac126c17cfd1e7494550` |
| `blocked_matmul_m64_n64_k16_t4x4` | 16 x 16 | 64 x 64 | 8,192 bytes | `71fb4757e0e4f52fc4d59dfebb42a6863fc3fb2dc4f6215cd058060b795d2081` |
| `blocked_matmul_m128_n128_k8_t8x8` | 16 x 16 | 128 x 128 | 8,192 bytes | `b8438783e768956b556e375380ebd97bd0123ce32cec6e773cee41dfbc8eee0c` |

The Day 76 cuBLAS text `sgemm_row_major.cu` keeps its digest
`9a86b34a84f3550412c2971a11b3972f6d1595b67193812afdb9583e1c36f889`. Each
launcher sets the grid to `cdiv(n, W)` by `cdiv(m, H)` blocks, where `W` and `H`
are the columns and rows of the C tile. Each text defines its own `cdiv`, so a
later run compiles each file as its own translation unit.

## Checks without a compiler

`gpu_checklist_check.check_cuda_source` reads a text with the helpers of the
Day 76 checker. It reads comments as a compiler reads them. It refuses a line
splice, a trigraph, and a space or a comment inside an operator. It accepts only
the three pinned functions. Then it parses the kernel body into a tree of
statements, loops and guards. It also parses the pinned text with the same
parser. Each check compares one part of the two trees, so a changed statement
fails the check of its part.

The coarsened kernel has 11 checks, and the blocked kernel has 12 checks:

- `cdiv_definition` and the signature checks `coarsened_signature` and
  `blocked_signature`: the pinned `cdiv` and the pinned kernel signature.
- `coarsened_shared_memory` and `blocked_shared_memory`: the two
  `__shared__ float` tiles. The check reads their extents from the text. The
  size must be the declared size, and it must be inside the 48 KB static limit
  of compute capability 8.0.
- `coarsened_index` and `blocked_index`: the thread and block index statements.
- `blocked_vector_alignment`: the two float4 flags, each float4 load in the
  branch of its flag, and the statements that give its offset.
- `coarsened_phase_loop` and `blocked_phase_loop`: one phase loop with the
  pinned header, between the setup statements and the stores.
- `coarsened_loads` and `blocked_loads`: the guarded loads and their zero fill.
  For the coarsened kernel, this check also covers the loop over the B tiles.
- `coarsened_barriers` and `blocked_barriers`: the barriers of the pinned
  text, and three rules that do not use the pinned text.
- `coarsened_inner_loop` and `blocked_inner_loop`: the header
  `for (int i = 0; i < min(T, k - phase * T); ++i)`, with BK in place of T for
  the blocked kernel.
- `coarsened_fma_order` and `blocked_fma_order`: the +0.0 initial value, the
  fma statements of the inner loop, and no other use of an accumulator.
- `coarsened_store_guard` and `blocked_store_guard`: the guarded stores. Only
  the store touches `c`, and no thread leaves the kernel early.
- `coarsened_grid` and `blocked_grid`: the host `dim3` expressions, with C
  unsigned division, give the launch of the kernel model for the shape.

These are the three barrier rules:

1. No barrier is inside an `if`.
2. Each loop around a barrier uses only `k`, `m`, `n`, constants and the
   variables of such loops. Thus every thread reaches each barrier equally
   often.
3. A static scan runs each loop body twice, so the last access of one iteration
   meets the first access of the next. It requires a barrier between each write
   of a shared tile and a later read of it. It also requires a barrier between
   each read and a later write.

The scan reads each assignment to a shared element as a write. It reads each
other use of a shared tile as a read. The tests change one statement at a time.
For example, a text without the first barrier fails `coarsened_barriers` only.
A text with `__fmaf_rz` fails `coarsened_fma_order` only. A text with
`bool b_vector = true;` fails `blocked_vector_alignment` only.

## CPU block simulator

`gpu_checklist_blocksim.run_blocks` holds all blocks of a launch as numpy
float32 arrays. It runs the steps of each text in order, phase by phase. A
product of two float32 values is exact in float64. The simulator rounds each
exact sum once to float32 with the rounding helper of
`matmul_metal_reference`.

The nine Day 74 correctness cases with the seed 74 give 45 runs of the five
kernels. These are the results:

- All 45 runs hold the bits of `fma_replay` for their case.
- In each run, the counted loads, float4 loads, slot writes, shared reads, fma
  steps and stores equal the closed forms of
  `gpu_checklist_blocksim.ChecklistTraffic`.
- Each run stores each element of `C` once, and it logs no hazard.
- In each run, each slot of a tile has one writer in each phase, and no float4
  load is unsafe.

A float4 load is unsafe if its offset is not a multiple of 4, or if a float of
it is outside its row. In the blocked runs, cases 2, 6 and 7 use float4 loads
for `A` and `B`. Cases 4 and 8 use them for `A` only. The other cases use scalar
loads only.

Each block column reads all of `A` once, and each block row reads all of `B`
once. For the case 1000 x 999 x 100, the closed forms give these loads:

| Kernel | Grid | Loads of A | Loads of B | Shared loads per fma |
| --- | --- | --- | --- | --- |
| Day 76 tile 16 | 63 x 63 | 6,300,000 | 6,293,700 | 2 |
| Day 76 tile 32 | 32 x 32 | 3,200,000 | 3,196,800 | 2 |
| `coarsened_matmul_t16_c2` | 32 x 63 | 3,200,000 | 6,293,700 | 2 |
| `coarsened_matmul_t16_c4` | 16 x 63 | 1,600,000 | 6,293,700 | 2 |
| `coarsened_matmul_t32_c2` | 16 x 32 | 1,600,000 | 3,196,800 | 2 |
| `blocked_matmul_m64_n64_k16_t4x4` | 16 x 16 | 1,600,000 | 1,598,400 | 0.5 |
| `blocked_matmul_m128_n128_k8_t8x8` | 8 x 8 | 800,000 | 799,200 | 0.25 |

The grid is the number of block columns by the number of block rows. These are
counts of a model. They are not measured traffic, and the model has no cache.

The simulator also logs the shared memory accesses of one block between
barriers. A read of a slot that another thread wrote in the same interval is a
read-after-write hazard. A write of a slot that another thread read in the same
interval is a write-after-read hazard. A test counts each hazard again with
Python sets. The table gives the hazards of the case 37 x 45 x 29 when a phase
step list changes. For the coarsened kernel, the list repeats for each tile of
`B`.

| Phase steps | Coarsened T = 16, F = 4 | Blocked 64 x 64 |
| --- | --- | --- |
| load, barrier, fma, barrier | 0 and 0 | 0 and 0 |
| load, fma, barrier | 34,800 and 0 | 55,680 and 0 |
| load, barrier, fma | 0 and 28,560 | 0 and 30,720 |
| load, fma | 55,680 and 30,720 | 55,680 and 30,720 |
| barrier, load, barrier, fma | 0 and 3,840 | 0 and 0 |

Each cell gives the read-after-write count and then the write-after-read count.
Each schedule gives the same output bits, because the simulator runs each step
for all threads before the next step. The last row shows a risk of the
coarsened kernel. A barrier before each load of `B` does not protect the load
of `A` that opens the next phase. That load meets the reads of the last tile of
`B`. Thus the text keeps the barrier after each inner loop.

## Declared occupancy

`gpu_checklist_occupancy` follows `gpu_tiled_occupancy`. It uses the calculator
rules of `simt_occupancy` and the compute capability 8.0 record. Each block
declares its threads and the static shared memory of its two tiles. The lab
compiles no kernel, so each register count is an assumption. Each count must at
least hold the live floats of one thread: 4, 6, 4, 24 and 80 for the five
kernels. Each cell gives the theoretical occupancy and the resident blocks per
SM:

| Kernel | Lowest count | Middle count | Highest count |
| --- | --- | --- | --- |
| `coarsened_matmul_t16_c2` | 32: 1.000, 8 blocks | 40: 0.750, 6 blocks | 64: 0.500, 4 blocks |
| `coarsened_matmul_t16_c4` | 32: 1.000, 8 blocks | 40: 0.750, 6 blocks | 64: 0.500, 4 blocks |
| `coarsened_matmul_t32_c2` | 32: 1.000, 2 blocks | 40: 0.500, 1 block | 64: 0.500, 1 block |
| `blocked_matmul_m64_n64_k16_t4x4` | 48: 0.625, 5 blocks | 64: 0.500, 4 blocks | 96: 0.250, 2 blocks |
| `blocked_matmul_m128_n128_k8_t8x8` | 96: 0.250, 2 blocks | 128: 0.250, 2 blocks | 168: 0.125, 1 block |

Shared memory limits no estimate. With the 1 KB that the driver keeps for each
block, a block allocates 3,072 or 9,216 bytes, so 54 or 18 blocks fit. The
registers and the warp slots set each value. At the declared counts, the blocked
kernels have fewer resident warps, but they need fewer shared and global loads
for each fma. These values are theoretical occupancy, not achieved occupancy,
and they predict no speed.

## What a later NVIDIA run must do

This lab produces no GFLOP/s number and no speedup. A later run on an NVIDIA
GPU must do these steps to time the five kernels against cuBLAS:

1. Run `sources`, and copy the six source files and `SHA256SUMS` to the NVIDIA
   host. Check the files with `sha256sum -c SHA256SUMS`.
2. Record the GPU name, its compute capability, the driver version, the CUDA
   toolkit version, the cuBLAS version and `NVIDIA_TF32_OVERRIDE`.
3. Compile each `.cu` file alone with `nvcc` for the compute capability of the
   GPU, with `-Xptxas -v`. Do not use `--use_fast_math` or `-ftz=true`.
4. Record the registers per thread and the spill bytes that `ptxas` reports
   for each kernel. Calculate the occupancy again with the reported counts.
5. Make the operands of each case with `matmul_metal_reference.matmul_inputs`
   from the seed 74 and the case index. Allocate each buffer with `cudaMalloc`.
6. Run each kernel on each correctness case. Each output must have the replay
   digest of its case in `docs/gpu-tiled-cuda-contract.md`.
7. Run `sgemm_row_major` on each case. Each output must stay inside the error
   bound of its case. Check the TF32 assumption of Day 76 on that host.
8. Time each kernel and each cuBLAS call after warm-up calls, for example with
   CUDA events. Use the benchmark shapes of `config/day-74-matmul-metal.json`.
9. Record the load averages of the host before and after the timing.
10. Divide `2 * m * n * k` by each median time to get GFLOP/s. Divide the rate
    of each kernel by the cuBLAS rate for the same shape.
11. Seal the times, the host records and the correctness results in a report
    that a verifier checks.

A later run that changes a text must run `check` on it again and record its new
digest. Only a profiler can give the achieved occupancy, the bank conflicts of
shared memory and the cache traffic. A later run that uses Nsight Compute must
record these values as measurements of that GPU.

## What this lab does not claim

- The lab compiled, linked and ran no CUDA or cuBLAS text. It does not know
  that `nvcc` accepts the texts, or that `nvcc` unrolls their loops.
- The lab used no NVIDIA GPU, no CUDA toolkit, no cuBLAS, no Nsight and no
  hardware counter.
- The lab measured no GPU time. It states no GFLOP/s, no speedup over Day 76
  or cuBLAS, no vendor peak and no fraction of a peak.
- That an NVIDIA GPU gives the replay bits is a prediction from the stated
  rounding of `__fmaf_rn`. No GPU run observed it.
- The load counts come from a model with no cache. They are not measured
  traffic.
- The register counts are assumptions, so the occupancy values are not achieved
  occupancy.
- The CPU times are wall times on one shared host. They are not GPU
  measurements, and they say nothing about production or distributed
  performance.

## Sources

- W. W. Hwu, D. B. Kirk and I. El Hajj, Programming Massively Parallel
  Processors, 4th ed., Morgan Kaufmann, 2022, sections 6.3 and 6.4, for thread
  coarsening and the checklist of optimizations.
- NVIDIA, CUTLASS documentation, "Efficient GEMM in CUDA", for the block tile
  and the thread tile of a register-blocked GEMM.
- NVIDIA, CUDA C++ Programming Guide, device memory accesses, for the 16-byte
  alignment of a `float4` access and the 256-byte alignment of `cudaMalloc`.
- NVIDIA, CUDA Math API, for `__fmaf_rn`.
- NVIDIA, `cuda_occupancy.h` of CUDA 12.6, for the calculator rules that
  `simt_occupancy` follows.
