# AI Journey

The shared language for reproducible distributed-training and serving evidence in this repository.

## Reproducibility and evidence

**Run Contract**:
The immutable inputs and execution constraints that identify a reproducible run.
_Avoid_: Run setup, experiment settings

**Admitted Checkpoint**:
A checkpoint accepted only after its identity, schema, bounds, and integrity satisfy the run contract.
_Avoid_: Valid checkpoint, good checkpoint

**Exact Restart**:
A resumed run whose next committed state matches the corresponding state of an uninterrupted control.
_Avoid_: Successful restart, equivalent recovery

**Sealed Report**:
A canonical evidence artifact whose derived fields and digest can be recomputed independently.
_Avoid_: Verified report, trusted report

**Evidence Boundary**:
The explicit limit on the claims supported by an executed result.
_Avoid_: Disclaimer, caveat

**Hardware-Gated Validation**:
Validation that remains unavailable until declared hardware and software prerequisites are directly verified.
_Avoid_: GPU-ready validation, hardware support

## Parallel execution

**Hybrid Parallel Topology**:
A factorization of participating ranks across data, tensor, pipeline, and sequence dimensions.
_Avoid_: Parallel setup, device layout

**Pipeline Schedule**:
The dependency-ordered execution slots for forward and backward microbatches across pipeline stages.
_Avoid_: Pipeline timeline, execution plan

**Pipeline Bubble**:
A pipeline-schedule slot in which a stage is idle while the scheduled run is still incomplete.
_Avoid_: Overhead, wasted time
