# Fused LLM-training kernels on the Apple GPU

Day 80 of the curriculum (Track B) follows GPU MODE Lecture 28, "Liger
Kernel: Efficient Triton Kernels for LLM Training". Ajay must still watch
Lecture 28. This guide does not summarize the lecture, and it makes no claim
about what the lecture shows.

Triton has no macOS build, so this lab writes three fused forward kernels in
Metal. The kernels run on the Apple GPU through `torch.mps.compile_shader`:

1. `liger_rmsnorm`: `y = x / sqrt(mean(x^2) + eps) * w` for each row.
2. `liger_swiglu`: `y = silu(a) * b`, element by element, in one pass.
3. `liger_cross_entropy`: the loss of each row, `logsumexp(x) - x[t]`. The
   kernel writes one float for each row and never writes the softmax.

Each kernel has a digest and a hypothesis that the lab declared before any
timing. One run checks each kernel against a float64 reference. It then times
each kernel against the unfused PyTorch sequence on the same device rows.

Nothing here runs a CUDA kernel, uses an NVIDIA GPU or runs a Triton kernel on
a GPU. No code of the Liger Kernel project runs here. The lab times forward
passes only. PyTorch does not report which kernels its sequences run. The lab
reads no hardware counters. Every timing is a host-specific Apple GPU (Metal)
measurement at the load averages that the report records.

## What Liger Kernel fuses, and why

The Liger Kernel project writes Triton kernels for the layers of LLM training.
Its paper and its README list kernels for RMSNorm, LayerNorm, RoPE, SwiGLU,
GeGLU and cross-entropy. They also list a fused linear cross-entropy. This
guide takes these facts from the paper and the README only. Ajay must still
check them against Lecture 28.

In eager PyTorch, each operator of a layer usually runs as a separate kernel.
Such a kernel reads its inputs from device memory and writes its output to
device memory. The next kernel then reads that output again. A fused kernel keeps the
intermediate values in registers, so it reads each input once and writes each
output once. A layer such as RMSNorm moves few bytes for each FLOP, so the
bytes probably set most of its time.

Cross-entropy at the end of a language model has a large cost. The logits
have one row for each token and one column for each word of the vocabulary.
The PyTorch call can write a matrix of log-probabilities with the shape of the
logits. A fused kernel with the online softmax update needs one pass over the
logits and writes one loss for each row.

The Liger paper and README also describe two more steps. These steps are
outside this lab:

- The cross-entropy kernel computes the gradient in the forward pass and
  writes it in place of the logits.
- The fused linear cross-entropy computes the logits in chunks, so the full
  logits matrix never exists.

This lab has forward kernels only. It does not compute a gradient, and it
does not fuse the final linear layer.

## Commands

The `run` command needs a gpu-ceilings report from the Day 74 lab. The
recorded Day 74 report of the test suite names macOS 27.2, and this host has
macOS 27.2.1. A measured run refuses a ceiling from another platform, so the
pinned run used a fresh Day 74 run:

```bash
python -m ai_journey.gpu_ceiling_cli run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
```

Then the `run` command makes a report, and the `verify` command recomputes
it:

```bash
python -m ai_journey.metal_liger_cli run \
  --config config/day-80-metal-liger.json \
  --ceilings /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-80-metal-liger.json
python -m ai_journey.metal_liger_cli verify \
  --report /tmp/day-80-metal-liger.json \
  --config config/day-80-metal-liger.json \
  --ceilings /tmp/day-74-gpu-ceilings.json
```

On a shared host, run each timed command under the GPU lock. The parser uses
the program name `ai-journey-metal-liger`. An installed wheel gives the same
commands under that name.

The `run` command prints these lines:

1. One line for each family of the plan, with its predicted verdicts.
2. One line with the load averages before and after the run.
3. One line with the bandwidth ceiling and the digest of its report.
4. One line for each math probe: `exp`, `rsqrt` and `log`.
5. One line for each correctness case, with its bound ratio and its repeat
   result.
6. One line for each timing point, with the median GB/s and the IQR range of
   each side.
7. One line for each timing point, with the ratio, the verdict and the result
   of the hypothesis.
8. One line with the count of Python warnings that the run caught.
9. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`metal_liger_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `metal_liger_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the summary
line with the prefix `verified_`. It never imports torch. With `--config`, it
requires the exact config of the run. With `--ceilings`, it requires the
exact ceiling report of the run.

The exit codes are the codes of the Day 73 to 79 Metal labs:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. Such a run writes no report.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured or wrote anything.

## The kernels

Each kernel runs one threadgroup per row. The group size G is the fewest
threads, a power of two from 32, whose 16 slots hold the longest row. G is at
most 1024. The first element of the row of group `r` is at `starts[r]`, and
the row holds `lengths[r]` elements. Dense, ragged and gapped layouts thus use
the same kernels.

| Kernel | Threadgroup bytes | Barriers | Text digest |
| --- | --- | --- | --- |
| `liger_rmsnorm` | 128 | 1 | `181f7549d9245c85b8db746115a8db3483aa660234bf2de65dbb687c02bfd42f` |
| `liger_swiglu` | 0 | 0 | `8e85f3c621f926057d21e416eb46ef6d04b7e661546734e5dfa3037d5bbbb509` |
| `liger_cross_entropy` | 256 | 1 | `d5aaccbb567d50ebbbc089ac81e8d5a1abd0d0316e4e655cef8b1dd56fbffa3f` |

The library digest is
`7ef7eeb1aad59640b2eb8c07cfc3768db098d6ca7af3afe2b322706dc74e62ca`. It also
holds three probe kernels for `precise::exp`, `precise::rsqrt` and
`precise::log`.

RMSNorm keeps the 16 slots of each thread in registers, so it reads `x` once.
Each thread adds its squares with `fma`. Then `simd_sum` and one threadgroup
step give the sum of the row. The kernel computes `precise::rsqrt(sum / n +
eps)` with `eps = 1e-6` and writes `(x * inv) * w`.

SwiGLU computes `a / (1 + exp(-a)) * b` for each element. No thread waits for
another thread, so the kernel has no barrier.

Cross-entropy keeps a running max `m` and a running sum `d` of `exp(v - m)`.
This is the online update of Day 79. For each logit `v`:

- If `v > m`, the thread rescales the old sum: `d = d * exp(m - v) + 1`, and
  then `m = v`.
- Otherwise, the thread adds the new term: `d = d + exp(v - m)`.

Each SIMD group merges its 32 pairs `(m, d)`, and one threadgroup step merges
the SIMD groups. Thread 0 then writes `(max - x[t]) + log(sum)`. The kernel
subtracts the target logit from the max first. Both terms are 0 or more, so
the rounding error does not grow with the magnitude of the logits.

These rules keep each kernel correct:

- Every barrier is outside each condition on the thread, so every thread
  reaches every barrier. A test reads each kernel text and checks this rule.
- A kernel never reads or writes a column at or past the length of the row.
- The kernels call `precise::` math, so the compile options do not change the
  math functions.

## The plan

| Family | PyTorch sequence on MPS | Predicted verdict of the kernel over PyTorch |
| --- | --- | --- |
| RMSNorm | `x.pow(2).mean(-1, keepdim=True)`, then `torch.rsqrt(ms + eps)`, then `x * inv` and `* w` | faster |
| SwiGLU | `torch.nn.functional.silu(a)`, then `* b` | faster |
| cross-entropy | `torch.nn.functional.cross_entropy(logits, targets, reduction='none')` | faster or no clear difference |

The plan digest is
`43d5f63eb9d93415d0b95244f7150d0652aa73a7845d98da7658993ff3f13165`. The
pinned config names this digest, so a config for other hypotheses cannot run.

The hypotheses give these reasons:

- The PyTorch RMSNorm sequence runs six operators. The model gives it about
  40 bytes for each element, and it gives the kernel about 12.
- The PyTorch SwiGLU sequence writes `silu(a)` and reads it again. The model
  gives it 20 bytes for each element, and it gives the kernel 12.
- If PyTorch writes the log-probabilities, it moves at least 8 bytes for each
  logit, and the kernel moves 4. The kernel also calls `precise::exp` once for
  each logit.

## The memory that the cross-entropy kernel does not write

For `r` rows and a vocabulary of `V` words, the logits hold `n = r V` floats.
A softmax or a matrix of log-probabilities in float32 has `4 n` bytes. The
fused kernel writes `4 r` bytes of losses.

| Rows | Vocabulary | Logits | Softmax bytes that the kernel does not write | Loss bytes that it writes |
| --- | --- | --- | --- | --- |
| 1024 | 32000 | 32768000 | 131072000 (125 MiB) | 4096 |
| 1024 | 50257 | 51463168 | 205852672 (196.3 MiB) | 4096 |

The report records these counts as `unwritten_softmax_bytes`. They are counts
of a model. The lab does not measure the memory that PyTorch allocates.

## Correctness

Each case draws seeded rows in a gapped layout. Thirty-two guard slots follow
each row in the input and in the output. Each input gap holds `2**100`, so a
kernel that reads a gap gets a wrong sum or a wrong max. Before the run, each
output slot holds a quiet NaN with one known bit pattern. The cross-entropy
output holds one loss for each row and then 32 sentinel slots.

The cases of each family cover these rows and inputs:

- Rows of 1, 31, 33 and 1000 elements in one ragged batch.
- Uniform inputs from -80 to 80, and inputs of 1000 or -1000 plus a standard
  normal.
- For cross-entropy, rows of 32000 and 50257 logits. With logits near 1000,
  a float32 cross-entropy with no max subtraction is not finite.

The reference is float64 from the float32 inputs. Its exp is the Day 79
`exp64`, and its log is `log64`. Both use only IEEE operations, `frexp` and
`ldexp`, so each IEEE 754 host computes the same float64 bits.

Each kernel has a declared error bound:

- RMSNorm: the relative error comes from the roundings of the sum of squares,
  one division, one add of eps, `precise::rsqrt` and two products.
- SwiGLU: the relative error comes from `precise::exp`, one add, one division
  and one product. A term of `2**-120 |b|` covers the inputs below -88.7,
  where `exp(-a)` overflows and the kernel returns 0.
- Cross-entropy: the absolute error comes from the online sum, the log, the
  subtraction from the max and the last add. The sum term grows with the
  range of the row and with the columns of each thread.

Each bound also adds the error of the float64 reference and a floor of
`2**-125`. The bounds assume these errors of the precise math. The run
measures each assumption first, and a failed probe stops the run:

| Function | Assumed relative error | Probe arguments |
| --- | --- | --- |
| `precise::exp` | 4 units of `2**-23` | the Day 79 probe: 262144 arguments |
| `precise::rsqrt` | 2 units of `2**-23` | 65536 arguments from `2**-20` to `2**40` |
| `precise::log` | 4 units of `2**-23` of `log x` | 65536 arguments from 1 to `2**17`, and near 1 |

Each kernel runs twice on each case. Three rules apply to the output of each
run, and a case that breaks one stops the run:

1. Each gap slot keeps its sentinel bits.
2. Each output is finite.
3. Each value is within its bound of the reference.

Determinism is a measurement, not a rule. The run records if the repeats of a
kernel gave the same bits.

## Timing

The timing is the Day 74 and Day 76 method, with the Day 78 rotating timer:

1. Each side runs 3 untimed warmup samples.
2. Each timed sample calls `torch.mps.synchronize()`, reads
   `time.perf_counter_ns`, launches a batch of calls, synchronizes again and
   reads the clock again.
3. Repeat `r` runs side `r mod 2` first, so each side runs first in 15 of the
   30 repeats.

A batch holds the fewest whole calls whose compulsory bytes reach 256 MiB.
Both sides of a point run the same batch. The compulsory bytes are these:

- RMSNorm: `8 n + 4 H`, to read `x` once, write `y` once and read the `H`
  weights once.
- SwiGLU: `12 n`, to read `a` and `b` once and write `y` once.
- Cross-entropy: `4 n + 8 r`, to read each logit once, read one target and
  write one loss for each row.

The model with no cache counts each operator of a PyTorch sequence as one
kernel that reads its inputs and writes its output. It reads a broadcast
operand once for each output element. The model of the PyTorch cross-entropy
assumes `log_softmax` and then `nll_loss`. PyTorch does not name its MPS
kernels, so that side of the model is an assumption.

After the last sample, the output of each kernel must be within its bound.
The output of PyTorch must be within a sanity bound of `2**-16` relative, or
the run stops.

The ratio rule is the rule of Day 76, with the declared `min_faster_ratio` of
1.1:

- `faster`: the ratio is 1.1 or more, and the p75 sample time of the kernel is
  below the p25 time of PyTorch.
- `slower`: the inverse ratio is 1.1 or more, and the p75 sample time of
  PyTorch is below the p25 time of the kernel.
- `no_clear_difference`: any other result.

## One run on the development host

This run used the pinned config on 2026-11-23 at 16:01 IST. The host was a
shared Darwin arm64 host with an Apple GPU, macOS 27.2.1, CPython 3.11.14,
NumPy 1.26.4 and PyTorch 2.15.0. The environment did not set
`PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. The run held the GPU
lock. Other jobs used the CPU.

The ceiling report came from a fresh Day 74 run at 15:37 IST under the same
lock. Its ceiling is 124.61 GB/s, the median of `stream_read_reduce` at 256
MiB. That run recorded load averages of 5.61, 6.67 and 6.75 before and 5.80,
6.65 and 6.74 after. Its digest is
`93fefcf83d3561e07fa135281b02d071b2073ef98c2e718c8317a8585acf0f81`.

The Day 80 report recorded load averages of 7.73, 8.02 and 7.68 before the
run. It recorded 7.10, 7.87 and 7.63 after the run. The `uptime` command
printed 8.06, 8.08 and 7.71 before the command and 7.45, 7.92 and 7.66 after
it.

The report digest is
`b2a7743f665c0a7a2efade9e5fe6e21adcb3a50465bb136362078b1a2b264831`. The run
took 10.7 seconds after the library compiled. The run caught 0 Python
warnings, and the capture began before the import of torch. Each pipeline
allowed 1024 threads, reported a SIMD width of 32 and reported its declared
threadgroup bytes. The CPU verifier took about 8 seconds.

### Correctness results

- All 12 cases met the three rules in both runs.
- The largest bound ratios were 0.101 for RMSNorm, 0.296 for SwiGLU and 0.068
  for cross-entropy.
- Each kernel gave the same bits in both runs of each case.
- The probes measured 1.406 units in the last place for `exp`, 0.500 for
  `rsqrt` and 1.841 for `log`. Each probe met its assumption.
- For the two cross-entropy cases with logits near 1000 and -1000, the naive
  float32 cross-entropy was not finite. The kernel gave finite losses within
  the bound.

### The table

Each median and IQR is the time of one call: the sample time divided by the
calls in a sample. Each GB/s divides the compulsory bytes by the median time.
The range in parentheses comes from the IQR of the sample times. "Of the
ceiling" divides the GB/s by 124.61. The ratio divides the median time of
PyTorch by the median time of the kernel.

| Family | Shape | Calls | Kernel median (IQR), us | Kernel GB/s (range) | Of the ceiling | PyTorch median (IQR), us | PyTorch GB/s (range) | Ratio, verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RMSNorm | 4096 x 1024 | 8 | 287.6 (23.6) | 116.7 (110.1 to 119.3) | 0.936 | 1664.6 (24.9) | 20.2 (20.0 to 20.3) | 5.79, faster |
| RMSNorm | 4096 x 4096 | 2 | 1160.4 (57.5) | 115.7 (111.5 to 117.1) | 0.928 | 6496.2 (198.8) | 20.7 (20.2 to 20.8) | 5.60, faster |
| RMSNorm | 4096 x 2049 | 4 | 625.7 (57.5) | 107.3 (103.5 to 113.6) | 0.861 | 3325.1 (76.1) | 20.2 (20.0 to 20.5) | 5.31, faster |
| SwiGLU | 4096 x 1024 | 6 | 452.0 (47.9) | 111.3 (103.3 to 114.6) | 0.894 | 710.4 (43.7) | 70.9 (67.7 to 72.0) | 1.57, faster |
| SwiGLU | 4096 x 4096 | 2 | 1742.9 (113.9) | 115.5 (111.3 to 118.8) | 0.927 | 2754.2 (196.7) | 73.1 (69.2 to 74.2) | 1.58, faster |
| SwiGLU | 4096 x 2049 | 3 | 864.2 (55.1) | 116.5 (111.1 to 118.3) | 0.935 | 1413.6 (71.9) | 71.2 (69.5 to 73.1) | 1.64, faster |
| cross-entropy | 1024 x 32000 | 3 | 1039.3 (63.2) | 126.1 (120.2 to 127.5) | 1.012 | 4130.5 (112.6) | 31.7 (31.1 to 32.0) | 3.97, faster |
| cross-entropy | 1024 x 50257 | 2 | 1678.4 (136.0) | 122.7 (117.2 to 127.0) | 0.984 | 6679.0 (201.4) | 30.8 (30.3 to 31.3) | 3.98, faster |

The predictions held at 8 of 8 points. The 2049-column shapes are dense. Their
rows do not fill each SIMD group equally, but the ragged rows appear in the
correctness cases only.

### How to read the table

- Each kernel ran at 0.86 to 1.01 of the ceiling. The time of these kernels
  probably comes from the device bytes, but no counter shows it.
- The cross-entropy kernel at 32000 columns ran at 1.012 of the ceiling. The
  ceiling is the best median of two Day 74 kernels at one buffer size. It is
  not a hardware limit, so a fraction above 1 is possible.
- The modeled GB/s of the RMSNorm kernel is about 175, above the ceiling. The
  model reads `w` once for each element, and a cache probably serves most of
  those reads. The model thus counts too many bytes for this kernel.
- The PyTorch RMSNorm sequence ran at 101 to 103 modeled GB/s, 0.81 to 0.83
  of the ceiling. That result agrees with six operators that each move their bytes
  through device memory.
- The SwiGLU ratio of 1.57 to 1.64 is near the modeled ratio of 20 to 12.
- The PyTorch cross-entropy ran at about 62 modeled GB/s. This lab cannot
  tell which kernels PyTorch ran, so it cannot explain that number.

## The boundary of the evidence

These are wall-clock timings of one Apple GPU on one shared host at the load
averages above. They do not measure or predict CUDA, NVIDIA, Triton or Liger
Kernel performance. The lab did not run the Triton kernels of Liger Kernel.
It timed forward passes only, with no gradient and no fused linear layer. It
read no hardware counter, so it observed no cache hit, stall or occupancy.
Each byte count is a model. The PyTorch side names a sequence of calls, not a
kernel.

The CPU verifier recomputes the config, the plan, each input, reference and
bound digest, and each launch and admission. From the raw samples, it
recomputes each statistic, rate, ratio, verdict and ceiling fraction. It cannot recompute the
fields that only the GPU run saw:

- The raw samples of each side. The verifier checks their form only.
- The digest and errors of each output that differs from the rounded
  reference. The verifier checks them against the bound.
- `identical_repeats`, because the CPU cannot run the GPU again.
- The results of each probe. The verifier checks them against the rule.
- The pipeline limits, the PyTorch checks, the warnings, the MPS switches and
  the load averages. The verifier checks their form.

The report holds the same list in its `verification` section. A matching
digest shows internal consistency. It cannot prove that a timing was
observed or that a GPU ran.

## The recorded report for CI

`tests/test_metal_liger_recorded.py` holds a short real report from this
host, byte for byte. The run used four correctness cases, three small points
and four repeats, on 2026-11-23 at 16:02 IST. The `uptime` command printed
load averages of 6.94, 7.75 and 7.60 before and after the run. At these small
sizes, the cost of each synchronize call sets most of the time. Two of its
three predictions missed, and the report records that result. The report is
there to be verified on a Linux host without torch, not to show speed.

## Work that Ajay must still do

- Ajay must still watch Lecture 28.
- Ajay must still check the Liger Kernel facts in this guide against the
  lecture, the paper and the README.
- Ajay must still compare these forward kernels with the Triton kernels of
  Liger Kernel. That comparison needs a host with an NVIDIA GPU.
