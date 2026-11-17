# Gap between the naive Metal matmuls and the measured ceilings

Day 74 asks for a naive matmul kernel, its GFLOP/s and the distance from its
rate to peak. Three labs supply the parts:

- The [matmul-metal lab](matmul-metal-lab.md) times two naive Metal kernels
  and `torch.matmul` on the Apple GPU.
- The [gpu-ceilings lab](gpu-ceilings-lab.md) measures an FMA ceiling and a
  bandwidth ceiling on the same GPU.
- The [matmul model lab](matmul-model-lab.md) counts the bytes of a matmul
  under two traffic models.

The gap command reads one sealed report from each of the two Metal labs. It
puts each naive kernel on a roofline from the measured ceilings and the model
byte counts. In this guide, "peak" means a ceiling that the gpu-ceilings lab
measured on this host. The command uses no vendor number. It runs no kernel,
and nothing here runs a CUDA kernel or uses an NVIDIA GPU.

## Commands

```bash
python -m ai_journey.matmul_roofline_cli gap \
  --matmul-report /tmp/day-74-matmul-metal.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-74-matmul-roofline.json
python -m ai_journey.matmul_roofline_cli verify \
  --report /tmp/day-74-matmul-roofline.json \
  --matmul-report /tmp/day-74-matmul-metal.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json
```

The two source paths are the output paths in the guides of the two Metal
labs. The parser uses the program name `ai-journey-matmul-roofline`. An
installed wheel gives the same commands under that name.

The `gap` command verifies both source reports with their own verifiers. Then
it seals the gap report, checks it and writes it. It prints these lines:

- one line with the two ceilings and the load averages of the ceiling run
- for each sweep point, one line with the two intensities, their roofs and
  the rate of `torch.matmul`
- for each sweep point and mapping, one line with the four ratios
- one summary line with the report digest

In a measured gap, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`matmul_roofline_measurement=`. When a source report is synthetic, the lines
use `synthetic_timing` and `matmul_roofline_synthetic_timing=` instead.

The `verify` command recomputes a gap report from the inputs in the report.
With `--matmul-report` and `--ceiling-report`, `verify` also builds the gap
again from the two source reports. The report must then have the same bytes,
and the summary line says `sources_checked=true`. Without the source reports,
the line says `sources_checked=false`. The `verify` command takes both source
reports or neither.

The exit codes are:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks. A source
  report that fails its own verifier also gives code 1.
- 2: a usage error, an unreadable or invalid input, two source reports that do
  not pair, or a refused output.

The command needs NumPy for the verifiers of the two Metal labs. It does not
import torch and needs no GPU, so it runs on any host with NumPy.

## Inputs and checks

Both source reports must pass their own verifiers before the command uses a
number from them. The verifiers recompute the correctness results, the
statistics, the rates and the ceilings. Thus a tampered source report sets no
value.

The two reports must also name the same Python, NumPy and PyTorch versions,
the same operating system and the same machine type. A ceiling from other
software tells nothing about these kernels. The command refuses such a pair
with exit code 2. The reports do not name the host, so this check cannot prove
that both reports came from one host.

The gap is a measurement only when both source reports are measurements. Then
the gap has the label `host-specific Apple GPU (Metal) measurement` and the
scope `apple_gpu_metal_this_host`. A gap with a synthetic source report has
`measured: false` and the scope `synthetic_backend_or_clock`.

The two source reports come from separate runs. The ceiling report records the
load averages before and after its run, and the gap report keeps them. The
matmul-metal report does not record load averages.

## Ratios

The command takes these inputs for each sweep point of the matmul-metal
report:

- the shape `M x N x K` and the number of calls in each sample
- the median sample time of each naive kernel and of `torch.matmul`

The median rate of a side is the FLOPs of one sample divided by its median
sample time. One call does `2 * M * N * K` FLOPs.

The intensities come from the two traffic models of the matmul model lab.
Each float32 element has 4 bytes:

- No reuse: each thread loads its K elements of A and its K elements of B,
  and it stores one element of C. The bytes are `4 * (2 * M * N * K + M * N)`.
  The intensity is `K / (4 * K + 2)` FLOPs per byte, which is less than 1/4.
- Ideal reuse: each element of A, B and C moves one time. The bytes are
  `4 * (M * K + K * N + M * N)`. No kernel moves fewer bytes.

For each side and intensity, the command calculates these values:

- The FMA share is the median rate divided by the FMA ceiling.
- The bandwidth roof is the bandwidth ceiling times the intensity, in GFLOP/s.
- The roof share is the median rate divided by the bandwidth roof.
- The bound is `bandwidth` when the bandwidth roof is less than the FMA
  ceiling. Else the bound is `compute`.

The ratio to `torch.matmul` is the median sample time of `torch.matmul`
divided by the median sample time of the naive kernel. All sides of a point
run the same number of calls in each sample, so this ratio is also the ratio
of the two rates. The matmul-metal report gives the same ratio.

A roof share above 1 at no reuse shows that the kernel ran faster than the
bandwidth ceiling times the no-reuse intensity. Thus the kernel probably moved
fewer bytes than the no-reuse model counts, and caches served part of the
loads. The labs read no hardware counters, so they do not measure this traffic.

## Gap on the development host

These values come from two measured reports of the development host on
2026-11-17:

- The matmul-metal report comes from the pinned run in the matmul-metal lab
  guide. That run started at 11:40 IST.
- The ceiling report comes from run 1 in the gpu-ceilings lab guide. That run
  started at 11:29 IST.

The `uptime` load averages were 10.72, 9.61 and 8.81 before the ceiling run.
The ceiling report recorded 10.66, 9.62 and 8.81 before and 9.95, 9.50 and
8.78 after. The `uptime` load averages were 7.98, 9.06 and 8.98 before the
matmul-metal run and 8.24, 8.97 and 8.95 after it. All values are
host-specific Apple GPU (Metal) measurements from one shared host, not
portable figures.

The tests keep this gap report as a fixture and check it on any host. Its
digest is
`28181026bef6f0029e769ef3bff25572ab94a7bc179d4a00f1f11372a180c3ca`. The
matmul-metal report has the digest
`a4c843ceaa5f018af364a8cea50503f7a180b522da9ecf09b31cb43d8e94aeaf`. The
ceiling report has the digest
`0a7563f44a0cbe6956e5716b97b515beed9e68bec545ede6f3c29b350366b305`.

The FMA ceiling was 4618 GFLOP/s, and the bandwidth ceiling was 100.7 GB/s.
The ridge point was 45.8 FLOPs per byte.

The first table gives the roofs of each shape. The last column divides the
median rate of `torch.matmul` by the FMA ceiling.

| M x N x K | No-reuse FLOPs per byte | No-reuse roof | Ideal-reuse FLOPs per byte | Ideal-reuse roof | Bound at ideal reuse | `torch.matmul` FMA share |
| --- | --- | --- | --- | --- | --- | --- |
| 128 x 128 x 128 | 0.2490 | 25.1 GFLOP/s | 21.33 | 2149 GFLOP/s | bandwidth | 0.033 |
| 256 x 256 x 256 | 0.2495 | 25.1 GFLOP/s | 42.67 | 4298 GFLOP/s | bandwidth | 0.166 |
| 512 x 512 x 512 | 0.2498 | 25.2 GFLOP/s | 85.33 | 8596 GFLOP/s | compute | 0.325 |
| 1024 x 1024 x 1024 | 0.2499 | 25.2 GFLOP/s | 170.67 | 17193 GFLOP/s | compute | 0.519 |
| 2048 x 2048 x 2048 | 0.2499 | 25.2 GFLOP/s | 341.33 | 34385 GFLOP/s | compute | 0.800 |
| 4096 x 4096 x 4096 | 0.2500 | 25.2 GFLOP/s | 682.67 | 68771 GFLOP/s | compute | 0.858 |
| 4096 x 1024 x 1024 | 0.2499 | 25.2 GFLOP/s | 227.56 | 22924 GFLOP/s | compute | 0.756 |
| 1024 x 4096 x 2048 | 0.2499 | 25.2 GFLOP/s | 292.57 | 29473 GFLOP/s | compute | 0.792 |
| 2000 x 3000 x 1000 | 0.2499 | 25.2 GFLOP/s | 272.73 | 27474 GFLOP/s | compute | 0.770 |

The second table gives the four ratios of each shape and mapping.

| M x N x K | Mapping | GFLOP/s | FMA share | No-reuse roof share | Ideal-reuse roof share | Ratio to `torch.matmul` |
| --- | --- | --- | --- | --- | --- | --- |
| 128 x 128 x 128 | `x_columns` | 213.1 | 0.046 | 8.49 | 0.0991 | 1.416 |
| 128 x 128 x 128 | `x_rows` | 134.1 | 0.029 | 5.34 | 0.0624 | 0.891 |
| 256 x 256 x 256 | `x_columns` | 301.7 | 0.065 | 12.00 | 0.0702 | 0.394 |
| 256 x 256 x 256 | `x_rows` | 122.5 | 0.027 | 4.87 | 0.0285 | 0.160 |
| 512 x 512 x 512 | `x_columns` | 362.8 | 0.079 | 14.42 | 0.0422 | 0.241 |
| 512 x 512 x 512 | `x_rows` | 165.1 | 0.036 | 6.56 | 0.0192 | 0.110 |
| 1024 x 1024 x 1024 | `x_columns` | 381.3 | 0.083 | 15.15 | 0.0222 | 0.159 |
| 1024 x 1024 x 1024 | `x_rows` | 152.6 | 0.033 | 6.06 | 0.0089 | 0.064 |
| 2048 x 2048 x 2048 | `x_columns` | 406.4 | 0.088 | 16.14 | 0.0118 | 0.110 |
| 2048 x 2048 x 2048 | `x_rows` | 160.8 | 0.035 | 6.38 | 0.0047 | 0.044 |
| 4096 x 4096 x 4096 | `x_columns` | 384.9 | 0.083 | 15.28 | 0.0056 | 0.097 |
| 4096 x 4096 x 4096 | `x_rows` | 187.7 | 0.041 | 7.46 | 0.0027 | 0.047 |
| 4096 x 1024 x 1024 | `x_columns` | 406.4 | 0.088 | 16.15 | 0.0177 | 0.116 |
| 4096 x 1024 x 1024 | `x_rows` | 153.2 | 0.033 | 6.09 | 0.0067 | 0.044 |
| 1024 x 4096 x 2048 | `x_columns` | 401.4 | 0.087 | 15.94 | 0.0136 | 0.110 |
| 1024 x 4096 x 2048 | `x_rows` | 158.7 | 0.034 | 6.30 | 0.0054 | 0.043 |
| 2000 x 3000 x 1000 | `x_columns` | 401.0 | 0.087 | 15.93 | 0.0146 | 0.113 |
| 2000 x 3000 x 1000 | `x_rows` | 120.5 | 0.026 | 4.79 | 0.0044 | 0.034 |

These are the results:

- The highest FMA share of `matmul_x_columns` was 0.088. Thus the best naive
  kernel ran at about one eleventh of the measured FMA ceiling.
- The highest FMA share of `matmul_x_rows` was 0.041, at 4096 x 4096 x 4096.
- `torch.matmul` reached 0.858 of the FMA ceiling at 4096 x 4096 x 4096.
- At all 9 shapes, both naive kernels ran above the no-reuse roof.
  `matmul_x_columns` ran 8.49 to 16.15 times above it, and `matmul_x_rows`
  ran 4.79 to 7.46 times above it.
- At ideal reuse, `matmul_x_columns` used 0.0056 to 0.0991 of the bandwidth
  roof, and `matmul_x_rows` used 0.0027 to 0.0624 of it.
- From 512 x 512 x 512 up, the ideal-reuse roof was above the FMA ceiling.
  Thus, on the roofline at ideal reuse, the FMA ceiling is the lower roof of
  these shapes, and the FMA share gives their gap.
- At 128 x 128 x 128 and 256 x 256 x 256, the ideal-reuse roof was below the
  FMA ceiling. Thus, on the roofline, the bandwidth ceiling is the lower roof
  of these two shapes even at ideal reuse.

The ratios to `torch.matmul` are the ratios in the matmul-metal lab guide. The
gap command adds the two ceilings and the two traffic models to them.

## Reports and verification

A gap report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature. The report holds these sections:

- the kind, digests, label and measured flag of each source report
- the load averages of the ceiling run
- the environment tokens that both source reports share
- the two ceilings, the ridge point and the `fma_scales_with_work` flag of the
  ceiling report
- the inputs, intensities, roofs, bounds and ratios of each sweep point
- a summary with the best FMA share of each side and the range of each ratio

The verifier calculates each derived value again from the inputs in the
report, with its own code. It calculates the bytes and the intensities in
exact integer fractions. It calculates the rates, roofs, shares and ratios
with the same float operations as the command, so it gets the same bits.

The inputs in a gap report cannot prove that they are correct. For example,
change one median sample time, calculate all values again and seal the report
again. That report passes the checks without the source reports. The `verify`
command with both source reports finds the change, because it builds the gap
again from the verified source reports.

## Evidence boundary and exclusions

The boundary of a measured gap report states: "These ratios compare two
verified reports from the Apple GPU of one host. The matmul rates and the
ceilings are host-specific Apple GPU (Metal) measurements from two separate
runs. Each ceiling is a rate that the ceiling lab's own kernels reached, not a
vendor peak and not a hardware limit. The intensities come from the byte
counts of the model lab, not from measured traffic. No CUDA kernel ran, and no
NVIDIA GPU was used. These ratios do not measure or predict CUDA, NVIDIA,
distributed or production performance."

The gap command times nothing and reads no hardware counters. A ceiling is a
rate of the ceiling lab's own kernels, so a kernel can go above it. The gap is
evidence from one Apple GPU on one host. It is not evidence of CUDA or NVIDIA
performance, lecture viewing, completed exercises or learner understanding.
