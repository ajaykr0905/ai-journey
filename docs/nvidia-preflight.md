# NVIDIA operational preflight

`GpuTopology` declares a single-host NCCL request. Its device ordinals refer to
the CUDA device list **after** `CUDA_VISIBLE_DEVICES` filtering; local rank zero
uses the first declared ordinal. Duplicate devices and unsupported dtypes are
rejected. The supported request dtypes are float32, float16, and bfloat16.

Worker validation requires `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and
`LOCAL_WORLD_SIZE` from a complete single-node torchrun environment. Multi-node
execution is outside this contract. These validations prepare a hardware check;
they do not implement CUDA training or establish that NCCL communication works.

`probe_gpu_host()` reads the actual operating system, PyTorch/CUDA build, NCCL
availability, visible CUDA device properties and free/total memory. Darwin and
CPU-only builds return an explicit unavailable reason. Device-query failures
cannot produce partial success. The visibility-mask identity is hashed; raw
hardware serials, hostnames, and unrelated environment variables are not recorded.
Property discovery does not measure GPU kernels, training, or communication.

`admit_gpu_resources()` consumes a verified resource-accounting report rather than
inventing a second memory model. Its data-parallel size and parameter dtype must
match the topology. Every requested device is checked separately against the
smaller of declared and observed capacity, with the declared reserve subtracted
from observed free memory. Accumulation contributes to global batch and token
accounting; it does not multiply per-microbatch activation memory. Native bfloat16
requires compute capability 8 or later. An analytical fit alone remains blocked
until actual host observations support the request. Sharding estimates remain
assumptions; this admission check does not implement those training strategies.
