"""The declared rule that names the top bottleneck of one measured point.

Each name is an inference from timings and the measured Day 74 ceilings. No
hardware counter, Nsight or Instruments trace takes part, so the rule cannot
see stalls, cache hits or occupancy. It can only say which measured limit, if
any, explains a measured time. The steps run in this order, and the first
step that applies sets the name:

1. ``indeterminate`` when the times contradict each other: a dispatch time
   above the one-call host time, or a one-call GPU time or steady-state time
   per call above it by more than the timing tolerance.
2. ``dispatch-bound`` when at least the pinned share of one call's host time
   lies outside the steady-state time per call. The timings cannot split that
   time into host dispatch, completion wait, GPU launch latency or a lower
   GPU clock in an isolated call, so the name covers all of them.
3. ``indeterminate`` when the one-call GPU time and the steady-state time per
   call differ by more than the pinned ratio, so the timing input does not
   corroborate the Day 74 rate.
4. ``indeterminate`` when the rate sits above the FMA ceiling or above the
   bandwidth roof by more than the pinned tolerance: that ceiling, or the
   traffic model of the roof, does not hold for the point.
5. When the rate reaches the pinned share of the lower roof, the name is
   ``memory-bound`` if the bandwidth roof is the lower roof, else
   ``compute-bound``. A compute ceiling that the ceiling lab flags as not
   scaling with work gives ``indeterminate`` instead, if the config asks.
6. Else ``latency or occupancy-bound``: the rate sits below both roofs. That
   name is a residual. It says that neither measured roof explains the rate,
   and an unmeasured limit such as on-chip cache bandwidth also fits it.

A measured ceiling is a rate that the ceiling lab's own kernels reached, so
the true limit can be higher. "Below a roof" therefore stays true against the
true limit, but "near a roof" needs a ceiling that scales with work.
"""

from __future__ import annotations

from dataclasses import dataclass

from .matmul_bottleneck_config import BottleneckRule, MatmulBottleneckError

DISPATCH_BOUND = "dispatch-bound"
MEMORY_BOUND = "memory-bound"
COMPUTE_BOUND = "compute-bound"
LATENCY_BOUND = "latency or occupancy-bound"
INDETERMINATE = "indeterminate"
BOTTLENECKS = (
    DISPATCH_BOUND,
    MEMORY_BOUND,
    COMPUTE_BOUND,
    LATENCY_BOUND,
    INDETERMINATE,
)
# Each reason code, the name it gives and the sentence that explains it.
REASONS = {
    "dispatch_time_above_host_time": (
        INDETERMINATE,
        "the dispatch time of one call exceeds its host time",
    ),
    "gpu_time_above_host_time": (
        INDETERMINATE,
        "the GPU time of one call exceeds its host time by more than the tolerance",
    ),
    "steady_time_above_host_time": (
        INDETERMINATE,
        "the steady-state time per call exceeds the one-call host time by more "
        "than the tolerance",
    ),
    "one_call_overhead_dominates": (
        DISPATCH_BOUND,
        "most of the one-call host time lies outside the steady-state time per call",
    ),
    "gpu_time_disagrees_with_steady_time": (
        INDETERMINATE,
        "the one-call GPU time and the steady-state time per call differ by more "
        "than the agreement ratio",
    ),
    "rate_above_fma_ceiling": (
        INDETERMINATE,
        "the rate exceeds the measured FMA ceiling by more than the tolerance",
    ),
    "rate_above_bandwidth_roof": (
        INDETERMINATE,
        "the rate exceeds the bandwidth roof by more than the tolerance, so the "
        "traffic model overcounts the bytes or the ceiling is low",
    ),
    "fma_ceiling_does_not_scale_with_work": (
        INDETERMINATE,
        "the rate is near the FMA ceiling, but the ceiling lab flags that ceiling "
        "as fixed-cost bound",
    ),
    "near_fma_ceiling": (
        COMPUTE_BOUND,
        "the rate reaches the near-roof share of the FMA ceiling, the lower roof",
    ),
    "near_bandwidth_roof": (
        MEMORY_BOUND,
        "the rate reaches the near-roof share of the bandwidth roof, the lower roof",
    ),
    "below_both_roofs": (
        LATENCY_BOUND,
        "the rate is below the near-roof share of the lower roof, so neither "
        "measured roof explains it",
    ),
}


@dataclass(frozen=True)
class PointInputs:
    """The measured inputs of one kernel at one shape.

    ``steady_ns_per_call`` is the Day 74 median sample time over its calls per
    sample, and ``rate_gflops_per_second`` is the Day 74 median rate. The roof
    is the bandwidth ceiling times the intensity of the config's traffic model.
    """

    steady_ns_per_call: float
    rate_gflops_per_second: float
    gpu_ns: int
    host_ns: int
    dispatch_ns: int
    fma_ceiling_gflops_per_second: float
    fma_scales_with_work: bool
    bandwidth_roof_gflops_per_second: float

    def __post_init__(self) -> None:
        for name in (
            "steady_ns_per_call",
            "rate_gflops_per_second",
            "fma_ceiling_gflops_per_second",
            "bandwidth_roof_gflops_per_second",
        ):
            value = getattr(self, name)
            if type(value) is not float or not value > 0 or value == float("inf"):
                raise MatmulBottleneckError(f"{name} must be a positive finite float")
        for name in ("gpu_ns", "host_ns", "dispatch_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise MatmulBottleneckError(f"{name} must be a positive integer")
        if type(self.fma_scales_with_work) is not bool:
            raise MatmulBottleneckError("fma_scales_with_work must be a boolean")


@dataclass(frozen=True)
class Decision:
    """One name, the reason code that set it, and every number behind it.

    ``deciding`` names the quantity of the step that set the name, its value
    and the limit it met or missed.
    """

    bottleneck: str
    reason: str
    numbers: dict[str, float]
    deciding: dict[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "bottleneck": self.bottleneck,
            "reason": self.reason,
            "explanation": REASONS[self.reason][1],
            "numbers": dict(self.numbers),
            "deciding": dict(self.deciding),
        }


def point_numbers(inputs: PointInputs) -> dict[str, float]:
    """Derive every share and ratio that the rule compares.

    The overhead share splits into two parts that the rule records but does
    not compare: ``host_outside_gpu_share``, the host time outside the GPU
    timeline of the call, and ``gpu_excess_share``, the GPU time above the
    steady-state time. A large second part can be a GPU launch latency or a
    lower clock in an isolated call, and only a counter could tell them apart.
    """

    host = inputs.host_ns
    steady = inputs.steady_ns_per_call
    fma = inputs.fma_ceiling_gflops_per_second
    roof = inputs.bandwidth_roof_gflops_per_second
    rate = inputs.rate_gflops_per_second
    lower = roof if roof < fma else fma
    return {
        "steady_ns_per_call": steady,
        "gpu_ns": float(inputs.gpu_ns),
        "host_ns": float(host),
        "dispatch_ns": float(inputs.dispatch_ns),
        "overhead_share": 1.0 - steady / host,
        "host_outside_gpu_share": 1.0 - inputs.gpu_ns / host,
        "gpu_excess_share": (inputs.gpu_ns - steady) / host,
        "dispatch_share": inputs.dispatch_ns / host,
        "gpu_to_steady_ratio": inputs.gpu_ns / steady,
        "rate_gflops_per_second": rate,
        "fma_ceiling_gflops_per_second": fma,
        "bandwidth_roof_gflops_per_second": roof,
        "fma_ceiling_share": rate / fma,
        "bandwidth_roof_share": rate / roof,
        "lower_roof_share": rate / lower,
    }


def _decision(
    reason: str, numbers: dict[str, float], quantity: str, limit: float
) -> Decision:
    return Decision(
        bottleneck=REASONS[reason][0],
        reason=reason,
        numbers=numbers,
        deciding={"quantity": quantity, "value": numbers[quantity], "limit": limit},
    )


def classify(inputs: PointInputs, rule: BottleneckRule) -> Decision:
    """Apply the ordered rule to one point; the first step that applies wins."""

    if type(inputs) is not PointInputs or type(rule) is not BottleneckRule:
        raise TypeError("classify takes PointInputs and a BottleneckRule")
    numbers = point_numbers(inputs)
    host = numbers["host_ns"]
    slack = 1.0 + rule.timing_tolerance
    if inputs.dispatch_ns > inputs.host_ns:
        return _decision("dispatch_time_above_host_time", numbers, "dispatch_ns", host)
    if numbers["gpu_ns"] > host * slack:
        return _decision("gpu_time_above_host_time", numbers, "gpu_ns", host * slack)
    if numbers["steady_ns_per_call"] > host * slack:
        return _decision(
            "steady_time_above_host_time", numbers, "steady_ns_per_call", host * slack
        )
    if numbers["overhead_share"] >= rule.dispatch_overhead_share_min:
        return _decision(
            "one_call_overhead_dominates",
            numbers,
            "overhead_share",
            rule.dispatch_overhead_share_min,
        )
    ratio = numbers["gpu_to_steady_ratio"]
    limit = rule.gpu_agreement_max_ratio
    if ratio > limit or ratio * limit < 1.0:
        return _decision(
            "gpu_time_disagrees_with_steady_time", numbers, "gpu_to_steady_ratio", limit
        )
    excess = 1.0 + rule.roof_excess_tolerance
    if numbers["fma_ceiling_share"] > excess:
        return _decision("rate_above_fma_ceiling", numbers, "fma_ceiling_share", excess)
    if numbers["bandwidth_roof_share"] > excess:
        return _decision(
            "rate_above_bandwidth_roof", numbers, "bandwidth_roof_share", excess
        )
    near = rule.near_roof_share_min
    if numbers["lower_roof_share"] < near:
        return _decision("below_both_roofs", numbers, "lower_roof_share", near)
    if (
        numbers["bandwidth_roof_gflops_per_second"]
        < numbers["fma_ceiling_gflops_per_second"]
    ):
        return _decision("near_bandwidth_roof", numbers, "lower_roof_share", near)
    if rule.require_fma_scales_with_work and not inputs.fma_scales_with_work:
        return _decision(
            "fma_ceiling_does_not_scale_with_work", numbers, "lower_roof_share", near
        )
    return _decision("near_fma_ceiling", numbers, "lower_roof_share", near)
