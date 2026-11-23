# Day 78 reduction CUDA contract

This lab is the CUDA half of Day 78. The Metal half,
`docs/gpu-reduction-lab.md`, times six float32 sum reductions on the Apple GPU
of the development host. This lab writes two of those steps as CUDA C++ text for
a later run on an NVIDIA GPU:

- `reduce_r1_shared` is the CUDA twin of the Metal step R1,
  `reduce_r1_sequential`. It adds the values of a block in shared memory.
- `reduce_warp_shuffle` is the CUDA twin of the Metal step R3,
  `reduce_r3_simd_shuffle`. Its last warp adds 32 values with
  `__shfl_down_sync`.

This host has no NVIDIA GPU, no CUDA toolkit, no `nvcc`, no CUB and no Nsight.
Thus the lab compiles, links and runs none of the texts. It checks the texts
without a compiler, and it runs the schedule of each text in a CPU simulator.
The lab measures no GPU time, and it writes no sealed report.

A warp is a group of 32 threads on an NVIDIA GPU. A SIMD group of 32 lanes is
the Metal counterpart of a warp. SIMD-group functions such as
`simd_shuffle_down` are the Metal counterpart of CUDA warp shuffles, not the
same instructions.

## Commands

Write the source files, and check them again from the disk. Then predict the
result of each kernel for each input of the pinned Day 78 config:

```bash
PYTHONPATH=src python -m ai_journey.gpu_reduce_cuda_cli sources \
  --output-dir /tmp/day-78-cuda-sources

PYTHONPATH=src python -m ai_journey.gpu_reduce_cuda_cli check \
  --source-dir /tmp/day-78-cuda-sources

PYTHONPATH=src python -m ai_journey.gpu_reduce_cuda_cli simulate \
  --config config/day-78-gpu-reduce.json
```

`sources` writes the two source files and a `SHA256SUMS` file. It prints one
line for each source file. `check` reads each source file from the directory.
It checks each file for the 14 sizes of `config/day-78-gpu-reduce.json`: the 9
correctness sizes and the 5 timing sizes. It prints one line for each file, as
in this example:

```text
cuda_check=reduce_warp_shuffle.cu sha256=db3ddc27468f7f8eea0a0c29601116a2ab64681e6f1709802c69504b01a4bf81 pinned=true checks=196/196 sizes=14 compiled=false
```

`pinned=true` means that the file has its pinned digest. A new comment in a file
changes its digest, but the file can still pass each check.

`simulate` reads a Day 78 config. It runs both CPU schedules on each correctness
case and on each timing size of that config. It prints one line for each kernel
and input, and then one summary line. The summary line gives the SHA-256 digest
of all predictions and the first 12 digits of the config digest.

The exit codes are:

- `0`: the command succeeded.
- `1`: `check` read each file, but a file failed a check. Or `simulate` found a
  result outside its bound, a count other than the model count, or a race. The
  command names each failure on standard error.
- `2`: the arguments are wrong, a file is missing, unreadable or not ASCII, the
  config is not valid, or the command refused the output path.

No command needs a GPU, so no exit code marks a missing device. The parser uses
the program name `ai-journey-gpu-reduce-cuda`. The package has no console script
for this lab, so the commands use `python -m`.

## The two texts

Each text is one file and one translation unit. It holds four definitions in
this order:

1. `cdiv`, the ceiling division of the Day 73 to 77 texts.
2. The pass-1 kernel. Each block of 256 threads writes one partial sum.
3. `reduce_combine_kernel`, the twin of the Metal kernel `reduce_combine`. One
   block of 1024 threads adds the partial sums and writes `out[0]`.
4. The launcher. It sets the grid and launches pass 1 and then the combine on
   one stream. It gives the combine the count of partial sums, `grid.x`.

No kernel uses an atomic. Thus the launch fixes the order of every addition.

| File | Pass-1 kernel | Elements per block | Barriers per block | Shared memory | Partials at n = 2^26 | SHA-256 of the text |
| --- | --- | --- | --- | --- | --- | --- |
| `reduce_r1_shared.cu` | `reduce_r1_shared_kernel` | 256 | 9 | 1,024 bytes | 262,144 | `86e4ed8683896e63c03038972bfd627837ea24841176ab6110b00bca0cdb5948` |
| `reduce_warp_shuffle.cu` | `reduce_warp_shuffle_kernel` | 512 | 4 | 1,024 bytes | 131,072 | `db3ddc27468f7f8eea0a0c29601116a2ab64681e6f1709802c69504b01a4bf81` |

Both files hold the same text of the combine kernel. The digest of that text is
`f6190e3a7352134e2abb08f3e70beab86c69206055913432938d5e4887b51cf4`. The combine
has 4,096 bytes of shared memory, and each of its threads reaches 11 barriers.

Each launcher sets the grid to `cdiv(n, E)` blocks of 256 threads, where E is
the number of elements per block. The texts take n from 1 to 2^26. Then the
largest index, `blockIdx.x * 512 + t + 256`, stays far below 2^32, so no
unsigned index wraps.

### R1 with shared memory

Each thread loads one element into shared memory, or `0.0f` at or past n. A
`__syncthreads()` follows the load. Then the block adds pairs with sequential
addressing. At stride s, the threads `t < s` add `shared[t + s]`, and s halves
from 128 to 1:

```c
    for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (t < s) {
            shared[t] = __fadd_rn(shared[t], shared[t + s]);
        }
        __syncthreads();
    }
```

A `__syncthreads()` ends each level, and no barrier is inside a guard. Thread 0
then writes the partial sum of its block.

### The warp-shuffle reduction

Each thread adds two elements 256 apart during the load, as R2 and R3 do. The
tree in shared memory adds pairs at strides 128, 64 and 32, with a barrier after
each level. Then the tree leaves 32 values, one for each lane of warp 0. Warp 0
adds them with five shuffles and no barrier:

```c
    if (warp == 0u) {
        float w = shared[lane];
        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 16u));
        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 8u));
        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 4u));
        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 2u));
        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 1u));
        if (lane == 0u) {
            partials[blockIdx.x] = w;
        }
    }
```

At offset d, lane l adds the value of lane l + d. After the offsets 16, 8, 4, 2
and 1, lane 0 holds the sum of all 32 values. Lane 0 adds the same pairs in the
same order as the last 5 levels of the R2 tree.

### Why the last warp needs no barrier

The CUDA C++ Programming Guide states these facts about `__shfl_down_sync`:

- A shuffle exchanges a variable between the threads of one warp, without shared
  memory.
- The mask names the threads that take part. Each named thread must call the
  same shuffle with the same mask.
- The source lane is the lane of the caller plus the offset. The upper lanes of
  the offset keep their own value.

The guard `warp == 0u` selects threads 0 to 31, because `warp = t / 32u`. The
block has 256 threads, so warp 0 is a whole warp. Thus each lane that the mask
`0xffffffff` names calls each shuffle. The values move in registers, so no
access to shared memory needs a barrier.

The Harris kernels of 2007 add the last warp in shared memory through a
`volatile` pointer, with no barrier. That code expects the 32 threads of a warp
to run in lockstep. From compute capability 7.0, the guide says that code must
not expect this. Such code needs `__syncwarp()`. The shuffles need no such call:
the guide states that the mask makes the named threads converge before the
hardware runs the shuffle.

### The combine

Thread t of the combine adds the partials t, t + 1024, t + 2048 and so on. It
uses a grid-stride loop with four accumulators, and then a loop for the rest:

```c
    for (; e + 3072u < count; e += 4096u) {
        a0 = __fadd_rn(a0, partials[e]);
        a1 = __fadd_rn(a1, partials[e + 1024u]);
        a2 = __fadd_rn(a2, partials[e + 2048u]);
        a3 = __fadd_rn(a3, partials[e + 3072u]);
    }
    for (; e < count; e += 1024u) {
        a0 = __fadd_rn(a0, partials[e]);
    }
```

The launch has exactly 1024 threads, so the stride of the loop is the number of
threads of the grid. Thus the loops read each partial once. Each thread then
writes `(a0 + a1) + (a2 + a3)` to shared memory. The block adds the 1024 values
with sequential addressing, and thread 0 writes `out[0]`.

## Rounding and order

Each addition is `__fadd_rn`. The CUDA Math API states that `__fadd_rn` adds in
round-to-nearest-even mode and that it never merges into a multiply-add. Thus
each text states every rounding and the order of every addition. That order is
the declared order of the Metal twins:

- `reduce_r1_shared` adds the pairs of `reduce_r1_sequential` and
  `reduce_combine`, in their order.
- `reduce_warp_shuffle` adds the pairs of `reduce_r3_simd_shuffle` and
  `reduce_combine`. The Day 78 plan declares that R3 keeps the order of R2, so
  the kernel also adds the pairs of `reduce_r2_first_add`.

Thus the bound of each text is the bound of its Metal twin, with the same depth.
At n = 2^26, D is 273 for R1 and 146 for the shuffle kernel. No text follows R4
or R5, because the Metal Shading Language Specification does not state the
order of `simd_sum`.

The PTX ISA states that the code optimizer treats an add with an explicit
rounding modifier conservatively. The texts expect that `nvcc` writes each
`__fadd_rn` as one `add.rn.f32` instruction. A later run must check this in the
PTX of each file.

## Checks without a compiler

`gpu_reduce_cuda_check.check_reduce_source` reads a text with the helpers of the
Day 76 and 77 checkers. It reads comments as a compiler reads them. It refuses a
line splice, a trigraph, and a space or a comment inside an operator. It accepts
only the four pinned definitions. Then it parses each kernel body into the Day
77 tree of statements, loops and guards, and it parses the pinned text the same
way.

Each statement and each loop or guard header belongs to one part of a kernel,
and each part has one check. A check compares its part of the two trees, with
the loops and guards above each item. An item names each of those by its part
and its place. Thus a changed loop header moves no item of another part.

The R1 text has 13 checks, and the shuffle text has 14 checks:

- `cdiv_definition`: the pinned `cdiv`.
- `r1_shared_memory` and `shuffle_shared_memory`: one `__shared__ float` array
  with one float for each thread of the block. Its size must be inside the 48 KB
  static limit of compute capability 8.0.
- `r1_index` and `shuffle_index`: the thread, lane and warp indices.
- `r1_loads` and `shuffle_loads`: the guarded loads. Each read of `x` must be
  `i < n ? x[i] : 0.0f` for its own index. In the shuffle kernel, the two loads
  of a thread are 256 elements apart. In both kernels, the grid loads each
  element once.
- `r1_tree` and `shuffle_tree`: the tree loop. With 256 threads, its header must
  give the strides 128 to 1, or 128, 64 and 32.
- `r1_barriers` and `shuffle_barriers`: the barriers of the pinned text and the
  barrier rules below.
- `shuffle_warp_stage`: the statements of warp 0 and the warp rules below.
- `r1_partial_store` and `shuffle_partial_store`: one store of the partial sum.
  Only thread 0 of the block runs it, as the last statement of the kernel.
- `combine_grid_stride`: the two loops of the combine. For the partials of the
  size, the loops must read each partial once and nothing at or past the count.
  The guard of the first loop must cover its largest offset.
- `combine_tree`, `combine_barriers` and `combine_store`: the tree of 1024
  values, its 11 barriers and the store of `out[0]` by thread 0.
- `order`: the parts of each kernel come in the pinned order.
- `launch`: the host `dim3` expressions, with C unsigned division, give the
  launch of the kernel model for n. The combine must be one block of 1024
  threads, and its count must be `grid.x`.

These are the barrier rules:

1. No barrier is inside an `if`.
2. Only a tree loop holds a barrier. Its header uses only `blockDim.x`,
   constants and its own variable, so every thread reaches each barrier equally
   often.
3. No thread leaves a kernel early. Each thread reaches the declared count of
   barriers: 9 in R1, 4 in the shuffle kernel and 11 in the combine.

These are the warp rules:

1. Each shuffle is `w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, d))`.
2. Each shuffle is inside the one guard `warp == 0u`. The index statements
   define `lane = t % 32u` and `warp = t / 32u`, so the guard selects threads 0
   to 31.
3. The offsets give lane 0 each of the 32 values once. The rule uses the upper
   lanes as the guide states: each keeps its own value.
4. The statements of warp 0 hold no barrier.

The tests change one statement at a time. For example, a text without the
barrier after the load fails `r1_barriers` only. A text with the mask
`0x0000ffffu` fails `shuffle_warp_stage` only. A text with the guard `t < 16u`
in place of `warp == 0u` fails `shuffle_warp_stage` only.

A text with a `__syncthreads()` after the last shuffle fails `shuffle_barriers`
and `shuffle_warp_stage`. A text with the step `e += 2048u` fails
`combine_grid_stride` only. With more than 3072 partials, its loops read some
partials twice. A text that launches the combine as two blocks fails `launch`
only.

A text whose tree stops at 16 values fails `shuffle_tree` and
`shuffle_barriers`, because each thread then reaches one more barrier. The
checker proves nothing about `nvcc`. A text that passes can still fail to
compile, and only a later NVIDIA run can show that it does not.

## CPU simulation

`gpu_reduce_cuda_sim.simulate` holds every block of pass 1 as numpy float32
arrays. It runs the steps of a text in order. The steps are the guarded loads,
one tree level for each barrier interval and the five shuffles of warp 0. Then
it runs the combine. numpy adds two float32 values with one rounding to nearest
even, the rounding of `__fadd_rn`. Each step adds the pairs of the text in its
order, so each result has the bits of that order.

The tests compare the simulator with the Metal kernels in three ways:

1. A thread-by-thread replay of the MSL text of R1, R2, R3 and `reduce_combine`
   runs in plain Python. On 8 seeded inputs of 1 to 4099 elements, it gives the
   bits of the simulator.
2. The same replay runs on symbolic values, so it gives the tree of additions.
   Each simulated tree equals the tree of its Metal twin for 11 sizes from 1 to
   4099. The R2 tree equals the R3 tree. The combine tree equals the Metal
   combine tree for 9 partial counts from 1 to 9000.
3. The simulated results have the bits of the recorded measured report of the
   Metal lab. That report holds 8 inputs: 6 correctness cases and 2 timing
   sizes.

At all 8 recorded inputs, `reduce_r1_shared` gives the bits of
`reduce_r1_sequential`. At all 8 inputs, `reduce_warp_shuffle` gives the bits of
`reduce_r2_first_add` and `reduce_r3_simd_shuffle`. The two orders give
different bits at 2 of the 8 inputs. Thus the bits can tell the two orders
apart.

A one-time check on 2026-11-21 compared the simulator with the report of the
pinned Metal run, digest
`5905fe9a28f58e242d669a5489b79a0937dab4595f584567d230818ae987df47`. The
repository does not keep that report. At all 18 inputs of that run, 13 cases and
5 timing sizes, the bits matched in the same way. The two orders gave different
bits at 9 of the 18 inputs.

These bits come from the Apple GPU of one host. At these inputs, the Metal
results agree with the declared orders. They do not show what an NVIDIA GPU
gives.

In each run, the counted loads equal n, and zero fills fill the rest of the
grid. Each block writes one partial, and the combine makes one read for each
partial. Each thread reaches the declared barriers.
`CudaReduceRun.matches_model` compares these counts with the closed forms of
the text. These counts come from a model of the text, not from a GPU.

### Races when a barrier is missing

An access log follows the shared memory of one block, from barrier to barrier.
A read of a slot that another thread wrote in the same interval is a
read-after-write race. A write of a slot that another thread read in the same
interval is a write-after-read race. The shared indices do not depend on
`blockIdx`, so the log of one block stands for every block. With each barrier in
place, no run logs a race.

The table gives the read-after-write races of one block when a run drops one
barrier. Barrier 0 follows the load, and barrier k follows level k of the tree.
No run logs a write-after-read race.

| Dropped barrier | `reduce_r1_shared` | `reduce_warp_shuffle` |
| --- | --- | --- |
| 0, after the load | 128 | 128 |
| 1, after stride 128 | 64 | 64 |
| 2, after stride 64 | 32 | 32 |
| 3, after stride 32 | 16 | 0 |
| 4, after stride 16 | 8 | none |
| 5, after stride 8 | 4 | none |
| 6, after stride 4 | 2 | none |
| 7, after stride 2 | 1 | none |
| 8, after stride 1 | 0 | none |
| every barrier | 255 | 224 |

After a dropped barrier, the next level, at stride s, reads s slots that other
threads wrote. The last barrier of each tree guards no access in this model,
because the reader of each slot wrote it. In R1, thread 0 reads slot 0. In the
shuffle kernel, lane l reads slot l. The texts keep that barrier, as the Metal
kernels do, because each level of the loop ends with one barrier.

A dropped barrier changes no simulated value, because the simulator runs each
step for all threads before the next step. On a GPU, the value after a race
depends on the order of the warps.

## Predicted results

`simulate` predicts the result of each kernel for each input of the pinned
config. Each value is a float32 value, followed by its IEEE 754 bits in
hexadecimal. Each prediction stays inside the bound of its order. The three
exact cases give the exact sum. The largest bound ratio is 0.124, for
`reduce_r1_shared` on the positive case with n = 511, as in the Metal run.

| Input | Pattern | n | `reduce_r1_shared` | `reduce_warp_shuffle` |
| --- | --- | --- | --- | --- |
| case-0 | normal | 1 | -0.3776976764202118 (`0xbec16197`) | -0.3776976764202118 (`0xbec16197`) |
| case-1 | normal | 31 | 3.9395437240600586 (`0x407c217c`) | 3.9395437240600586 (`0x407c217c`) |
| case-2 | normal | 33 | 1.9293007850646973 (`0x3ff6f354`) | 1.9293007850646973 (`0x3ff6f354`) |
| case-3 | normal | 257 | 4.14474630355835 (`0x4084a1c3`) | 4.144746780395508 (`0x4084a1c4`) |
| case-4 | normal | 1000003 | 720.9532470703125 (`0x44343d02`) | 720.953125 (`0x44343d00`) |
| case-5 | normal | 2000003 | 1898.933349609375 (`0x44ed5dde`) | 1898.93310546875 (`0x44ed5ddc`) |
| case-6 | positive | 511 | 256.71368408203125 (`0x43805b5a`) | 256.7137145996094 (`0x43805b5b`) |
| case-7 | positive | 1000003 | 500196.5 (`0x48f43c90`) | 500196.5 (`0x48f43c90`) |
| case-8 | mixed | 513 | -92146896.0 (`0xccafc19a`) | -92146904.0 (`0xccafc19b`) |
| case-9 | mixed | 300007 | 2778781952.0 (`0x4f25a0d9`) | 2778781696.0 (`0x4f25a0d8`) |
| case-10 | mixed | 1000003 | -8780908544.0 (`0xd002d882`) | -8780908544.0 (`0xd002d882`) |
| case-11 | ones | 1000003 | 1000003.0 (`0x49742430`) | 1000003.0 (`0x49742430`) |
| case-12 | halves | 1000003 | 2504.0 (`0x451c8000`) | 2504.0 (`0x451c8000`) |
| timing-0 | timing | 65536 | -92.49160766601562 (`0xc2b8fbb4`) | -92.49160766601562 (`0xc2b8fbb4`) |
| timing-1 | timing | 1048576 | 1615.73974609375 (`0x44c9f7ac`) | 1615.739501953125 (`0x44c9f7aa`) |
| timing-2 | timing | 16777216 | 3865.58740234375 (`0x45719966`) | 3865.587890625 (`0x45719968`) |
| timing-3 | timing | 67108864 | -6782.974609375 (`0xc5d3f7cc`) | -6782.9775390625 (`0xc5d3f7d2`) |
| timing-4 | timing | 10000019 | 4596.720703125 (`0x458fa5c4`) | 4596.720703125 (`0x458fa5c4`) |

The inputs `case-0` to `case-12` are the correctness cases of the config, in its
order. The inputs `timing-0` to `timing-4` are the first n values of the one
timing input. The summary line of the pinned config is:

```text
gpu_reduce_cuda_predictions=a4afcb780fbf1002b74cbd808bf676553ade8b6a699d26299cd5d10be96824e9 config=c13145af0b74 kernels=2 inputs=18 within_bounds=36/36 race_free=36/36 compiled=false
```

These values are predictions from the stated rounding and order. No GPU run
gave them. A later NVIDIA run must give the same bits if `nvcc` keeps each
`__fadd_rn` and its order.

## What a later NVIDIA run must do

This lab produces no GB/s value and no speedup. A later run on an NVIDIA GPU
must do these steps to time the two kernels against a library sum:

1. Run `sources`, and copy the two source files and `SHA256SUMS` to the NVIDIA
   host. Check the files with `sha256sum -c SHA256SUMS`.
2. Record the GPU name, its compute capability, the driver version, the CUDA
   toolkit version, the CUB version and the PyTorch version.
3. Compile each `.cu` file alone with `nvcc` for the compute capability of the
   GPU, with `-Xptxas -v`. Do not use `--use_fast_math` or `-ftz=true`.
4. Write the PTX of each file with `nvcc -ptx`. Check that each `__fadd_rn` is
   one `add.rn.f32` instruction.
5. Make the inputs from the seed 78 as the Metal lab does. Use
   `gpu_reduce_bound.case_values` for each correctness case. Use
   `gpu_reduce_bound.benchmark_values` and its prefixes for the timing sizes.
6. Allocate each buffer with `cudaMalloc`, with the guard tails of the Metal
   lab. Put 1024 quiet-NaN floats after the input, 64 after the partials and 31
   after the output.
7. Run each launcher once for each input. Check the status of
   `cudaGetLastError` and `cudaDeviceSynchronize`. Each guard tail must keep its
   sentinels.
8. Compare each result with the bits that `simulate` prints for its kernel and
   input. Record each result with other bits.
9. Time each kernel and a library sum on the same device buffer. The library
   sums are `cub::DeviceReduce::Sum` and `torch.sum` on the CUDA device.
   Allocate the temporary storage of CUB once, before the timing.
10. Use the timing method of the Metal lab. Run 2 samples that are not timed
    and 14 timed samples for each side. Each sample holds the fewest whole calls
    that read 2^30 bytes. Synchronize the device before and after each sample.
11. In repeat r, time side r mod S first, where S is the number of sides. Thus
    the order moves by one side at each repeat, as in the Metal lab.
12. Record the load averages of the host before and after the timing.
13. Divide 4 n bytes times the calls of a sample by the median sample time to
    get GB/s. Divide the rate of each kernel by the rate of each library sum at
    the same size.
14. Apply the ratio rule of the Day 76 Metal lab. A ratio is `faster` only when
    it is 1.1 or more and the IQR ranges of the two sides do not overlap.
15. The same test in the other direction gives `slower`. Otherwise the verdict
    is `no_clear_difference`. Keep each verdict in the report, also each
    `slower` and each `no_clear_difference` verdict.
16. Neither library states its order of additions. Check each library result
    against the declared tolerance of the Metal lab, `gamma_4096(u) * sum |x|`.
    Record if each repeat keeps its bits.
17. Seal the times, the host records and the correctness results in a report
    that a verifier checks.

A later run that changes a text must run `check` on it again and record its new
digest. Only a profiler can give the achieved occupancy, the divergence, the
bank conflicts of shared memory and the memory traffic. A later run that uses
Nsight Compute must record these values as measurements of that GPU.

## What this lab does not claim

- The lab compiled, linked and ran no CUDA text. It does not know that `nvcc`
  accepts the texts, or that `nvcc` writes each `__fadd_rn` as one instruction.
- The lab used no NVIDIA GPU, no CUDA toolkit, no CUB, no Nsight and no hardware
  counter.
- The lab measured no GPU time. It states no GB/s value, no speedup over the
  Metal kernels, CUB or `torch.sum`, no vendor peak and no fraction of a peak.
- That an NVIDIA GPU gives the predicted bits is a prediction from the stated
  rounding and order. No GPU run observed it.
- The matches with the Metal reports are host-specific Apple GPU (Metal)
  results. They agree with the declared orders on one host. They are not CUDA,
  NVIDIA or warp-level results.
- `simd_shuffle_down` and `__shfl_down_sync` are counterparts, not the same
  instruction. The Metal timings do not measure or predict the speed of a warp
  shuffle.
- The counts come from a model of the texts. They are not measured traffic or
  occupancy, and the model has no cache.

## Learner work

These items stay with the learner:

- Ajay must still watch Lecture 9, "Reductions", and check both texts against
  its kernels.
- Ajay must still do the later NVIDIA run, or record who does it. No one has
  compiled or run these texts.

## Sources

- M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA, 2007. It shows
  sequential addressing, the first add during the load and a last warp with no
  barrier.
- J. Luitjens, "Faster Parallel Reductions on Kepler", NVIDIA Technical Blog,
  2014, for reductions with warp shuffles.
- W. W. Hwu, D. B. Kirk and I. El Hajj, Programming Massively Parallel
  Processors, 4th ed., Morgan Kaufmann, 2022, chapter 10, for reduction trees
  and control divergence.
- NVIDIA, CUDA C++ Programming Guide, for the warp shuffle functions,
  `__syncthreads()`, `__syncwarp()` and the independent thread scheduling of
  compute capability 7.0.
- NVIDIA, CUDA Math API, for `__fadd_rn`.
- NVIDIA, PTX ISA, for the rounding modifiers of `add`.
- NVIDIA, CUB documentation, for `cub::DeviceReduce::Sum`.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 4.2, for the bound of a summation order.
