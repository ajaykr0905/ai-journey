# NVIDIA operational preflight

`GpuTopology` declares a single-host NCCL request. Its device ordinals refer to
the CUDA device list **after** `CUDA_VISIBLE_DEVICES` filtering; local rank zero
uses the first declared ordinal. Duplicate devices and unsupported dtypes are
rejected. The supported request dtypes are float32, float16, and bfloat16.

Worker validation requires `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and
`LOCAL_WORLD_SIZE` from a complete single-node torchrun environment. Multi-node
execution is outside this contract. These validations prepare a hardware check;
they do not implement CUDA training or establish that NCCL communication works.
