"""Strict, bounded experiment preregistration for analytical scaling plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .scaling_admission import CorpusAdmissionPolicy, assess_corpus_admission
from .scaling_budget import build_isoflop_sweep
from .scaling_corpus import plan_corpus_reuse
from .scaling_laws import ComputeOptimalTarget, ScalingLawError, _positive_integer
from .scaling_runtime import RuntimeAssumptions, estimate_runtime_and_cost
from .scaling_schedule import (
    TrainingBatchGeometry,
    align_token_target,
    validate_overshoot,
)

MAX_SCALING_PROTOCOL_BYTES = 64 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ScalingLawError("duplicate scaling protocol field")
        result[key] = value
    return result


@dataclass(frozen=True)
class ScalingProtocol:
    """A preregistered analytical scaling plan, checked end to end when built.

    Construction runs every derived calculation, so excess batch overshoot or
    an unfundable IsoFLOP size fails before a report exists; corpus admission
    is evaluated but may still be negative.
    """

    name: str
    parameter_count: int
    tokens_per_parameter: str
    training_flops_per_parameter_token: str
    unique_train_tokens: int
    batch: TrainingBatchGeometry
    maximum_overshoot_fraction: float
    isoflop_parameter_counts: tuple[int, ...]
    runtime: RuntimeAssumptions
    corpus_policy: CorpusAdmissionPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ScalingLawError("protocol name must be nonempty")
        _positive_integer(self.unique_train_tokens, "unique_train_tokens")
        if not isinstance(self.batch, TrainingBatchGeometry):
            raise TypeError("batch must be TrainingBatchGeometry")
        if (
            isinstance(self.maximum_overshoot_fraction, bool)
            or not isinstance(self.maximum_overshoot_fraction, (int, float))
            or not math.isfinite(self.maximum_overshoot_fraction)
        ):
            raise ScalingLawError("maximum_overshoot_fraction must be finite")
        target = ComputeOptimalTarget(
            self.parameter_count,
            self.tokens_per_parameter,
            self.training_flops_per_parameter_token,
        )
        validate_overshoot(
            align_token_target(target.token_target, self.batch),
            maximum_fraction=self.maximum_overshoot_fraction,
        )
        reuse = plan_corpus_reuse(
            target_tokens=target.token_target,
            unique_train_tokens=self.unique_train_tokens,
        )
        assess_corpus_admission(reuse, self.corpus_policy)
        build_isoflop_sweep(
            compute_budget_flops=target.training_flops,
            parameter_counts=self.isoflop_parameter_counts,
            optimal_tokens_per_parameter=self.tokens_per_parameter,
            training_flops_per_parameter_token=(
                self.training_flops_per_parameter_token
            ),
        )
        estimate_runtime_and_cost(target.training_flops, self.runtime)


def load_scaling_protocol(path: Path) -> ScalingProtocol:
    """Load an exact versioned schema without aliases or unknown fields."""

    if not isinstance(path, Path):
        raise TypeError("protocol path must be pathlib.Path")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_SCALING_PROTOCOL_BYTES + 1)
        if len(payload) > MAX_SCALING_PROTOCOL_BYTES:
            raise ScalingLawError("scaling protocol exceeds 64 KiB")
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        expected = {
            "schema_version",
            "name",
            "parameter_count",
            "tokens_per_parameter",
            "training_flops_per_parameter_token",
            "unique_train_tokens",
            "batch",
            "maximum_overshoot_fraction",
            "isoflop_parameter_counts",
            "runtime",
            "corpus_policy",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ScalingLawError("scaling protocol fields are invalid")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ScalingLawError("unsupported scaling protocol version")
        batch = data["batch"]
        if not isinstance(batch, dict) or set(batch) != {
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "data_parallel_size",
        }:
            raise ScalingLawError("scaling batch fields are invalid")
        sizes = data["isoflop_parameter_counts"]
        if not isinstance(sizes, list):
            raise ScalingLawError("isoflop_parameter_counts must be a list")
        runtime = data["runtime"]
        if not isinstance(runtime, dict) or set(runtime) != {
            "sustained_flops_per_second",
            "accelerator_count",
            "hourly_cost_per_accelerator",
        }:
            raise ScalingLawError("scaling runtime fields are invalid")
        corpus_policy = data["corpus_policy"]
        if not isinstance(corpus_policy, dict) or set(corpus_policy) != {
            "maximum_epochs",
            "minimum_distinct_token_fraction",
        }:
            raise ScalingLawError("scaling corpus policy fields are invalid")
        return ScalingProtocol(
            name=data["name"],
            parameter_count=data["parameter_count"],
            tokens_per_parameter=data["tokens_per_parameter"],
            training_flops_per_parameter_token=data[
                "training_flops_per_parameter_token"
            ],
            unique_train_tokens=data["unique_train_tokens"],
            batch=TrainingBatchGeometry(**batch),
            maximum_overshoot_fraction=data["maximum_overshoot_fraction"],
            isoflop_parameter_counts=tuple(sizes),
            runtime=RuntimeAssumptions(**runtime),
            corpus_policy=CorpusAdmissionPolicy(**corpus_policy),
        )
    except ScalingLawError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        KeyError,
        TypeError,
    ) as error:
        raise ScalingLawError("unable to load scaling protocol") from error
