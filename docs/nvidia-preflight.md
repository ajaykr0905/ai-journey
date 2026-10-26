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

The implemented `ai_journey.gpu_preflight_worker` module reads the torchrun rank
environment, probes this host, and writes a verified record to
`OUTPUT_DIR/RUN_ID/rank-RANK.json`. A blocked observation is still written, and
the worker exits with status 2. No process group, CUDA tensor, kernel, or training
loop is started. Records bind the canonical run UUID, actual process ID, rank
mapping, resource report, and host observations. Verification recomputes memory
admission rather than trusting a stored success flag. Digests detect drift;
they are not hardware attestation or third-party signatures.

## Prepare and verify a run

After installing the package, prepare the existing eight-device analytical
scenario without launching any workers:

```bash
ai-journey-resource-plan --config config/day-51-resource-plan.json \
  --output /tmp/nvidia-resources.json
ai-journey-gpu-preflight probe --output /tmp/nvidia-host.json
ai-journey-gpu-preflight plan --resource-report /tmp/nvidia-resources.json \
  --devices 0 1 2 3 4 5 6 7 --dtype bfloat16 \
  --output-dir /tmp/nvidia-preflight --output /tmp/nvidia-plan.json
```

The plan contains a fresh run UUID, expected rank-record paths, memory admission,
and an argv array for `python -m torch.distributed.run --standalone`. It targets
the implemented preflight worker with zero automatic restarts. Paths remain
individual argv elements, including paths containing spaces. No shell command is
executed by `plan`. Generate the plan on the intended Linux NVIDIA host so its
Python path and runtime observations are relevant. Darwin reports an unavailable
host and leaves `preflight_ready` false.

If an authorized operator later executes that argv array on the intended host,
collect the resulting records using the UUID printed in the plan:

```bash
ai-journey-gpu-preflight collect --resource-report /tmp/nvidia-resources.json \
  --devices 0 1 2 3 4 5 6 7 --dtype bfloat16 --run-id UUID_FROM_PLAN \
  --output-dir /tmp/nvidia-preflight --output /tmp/nvidia-run.json
ai-journey-gpu-preflight verify --report /tmp/nvidia-run.json
```

Collection requires every requested rank, distinct process IDs, the same run,
topology, resource contract, runtime build and visible-device identity. Missing
records remain a failure. Verification also accepts an individual rank record.
A valid blocked observation can pass integrity verification while retaining
`preflight_passed: false`; this is different from passing hardware admission.
Even a passed preflight retains `training_executed`, `gpu_kernel_executed`, and
`collective_executed` as false. CUDA training, NCCL collective checks, multi-GPU
performance and fault recovery require separate implemented and measured work.
