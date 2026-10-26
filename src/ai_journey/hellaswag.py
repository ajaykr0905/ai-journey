"""Bounded HellaSwag evaluation for the public GPT-2 systems lab."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import platform
import random


class HellaSwagError(ValueError):
    """Raised when benchmark data or evidence violates its contract."""


HELLASWAG_LIMITATIONS = [
    "A pinned 100-row validation slice is not the official 10,042-example score.",
    "The byte-level CPU lab model is not the pretrained GPT-2 124M checkpoint.",
    "Accuracy uncertainty is descriptive for this bounded slice only.",
    "Automated execution does not prove lecture viewing or learner understanding.",
]


_EXAMPLE_FIELDS = {
    "ind",
    "activity_label",
    "ctx_a",
    "ctx_b",
    "ctx",
    "endings",
    "source_id",
    "split",
    "split_type",
    "label",
}


@dataclass(frozen=True)
class HellaSwagExample:
    """One validated four-way HellaSwag validation example."""

    index: int
    activity_label: str
    context_a: str
    context_b: str
    context: str
    endings: tuple[str, str, str, str]
    label: int
    source_id: str
    split: str
    split_type: str

    @classmethod
    def from_mapping(cls, payload: dict) -> "HellaSwagExample":
        if not isinstance(payload, dict) or set(payload) != _EXAMPLE_FIELDS:
            raise HellaSwagError("HellaSwag example fields do not match")
        index = payload["ind"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise HellaSwagError("example index must be a nonnegative integer")
        strings = {
            field: payload[field]
            for field in (
                "activity_label",
                "ctx_a",
                "ctx_b",
                "ctx",
                "source_id",
                "split",
                "split_type",
            )
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in strings.values()
        ):
            raise HellaSwagError("example text fields must be nonempty strings")
        endings = payload["endings"]
        if (
            not isinstance(endings, list)
            or len(endings) != 4
            or any(not isinstance(value, str) or not value.strip() for value in endings)
            or len(set(endings)) != 4
        ):
            raise HellaSwagError("example must contain four distinct nonempty endings")
        label = payload["label"]
        if isinstance(label, int) and not isinstance(label, bool):
            parsed_label = label
        elif isinstance(label, str) and label in {"0", "1", "2", "3"}:
            parsed_label = int(label)
        else:
            raise HellaSwagError("example label must select one of four endings")
        if parsed_label not in range(4):
            raise HellaSwagError("example label must select one of four endings")
        canonical_context = f"{strings['ctx_a'].rstrip()} {strings['ctx_b'].lstrip()}"
        if strings["ctx"].strip() != canonical_context:
            raise HellaSwagError("example context does not match ctx_a and ctx_b")
        if strings["split"] != "val" or strings["split_type"] not in {
            "indomain",
            "zeroshot",
        }:
            raise HellaSwagError("example must come from a known validation split")
        return cls(
            index=index,
            activity_label=strings["activity_label"],
            context_a=strings["ctx_a"],
            context_b=strings["ctx_b"],
            context=strings["ctx"],
            endings=tuple(endings),
            label=parsed_label,
            source_id=strings["source_id"],
            split=strings["split"],
            split_type=strings["split_type"],
        )

    @property
    def canonical_context(self) -> str:
        return f"{self.context_a.rstrip()} {self.context_b.lstrip()}"

    @property
    def digest(self) -> str:
        payload = {
            "index": self.index,
            "activity_label": self.activity_label,
            "context_a": self.context_a,
            "context_b": self.context_b,
            "context": self.context,
            "endings": self.endings,
            "label": self.label,
            "source_id": self.source_id,
            "split": self.split,
            "split_type": self.split_type,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class HellaSwagSource:
    """Pinned public validation-set identity without a runtime data dependency."""

    repository: str
    revision: str
    validation_url: str
    validation_sha256: str
    validation_bytes: int
    validation_rows: int
    sample_url: str
    sample_sha256: str
    sample_bytes: int
    license: str

    @classmethod
    def from_mapping(cls, payload: dict) -> "HellaSwagSource":
        fields = {
            "repository",
            "revision",
            "validation_url",
            "validation_sha256",
            "validation_bytes",
            "validation_rows",
            "sample_url",
            "sample_sha256",
            "sample_bytes",
            "license",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise HellaSwagError("HellaSwag source fields do not match")
        for field in ("repository", "validation_url", "sample_url", "license"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise HellaSwagError(f"source {field} must be a nonempty string")
        for field in ("revision", "validation_sha256", "sample_sha256"):
            value = payload[field]
            expected_length = 40 if field == "revision" else 64
            if (
                not isinstance(value, str)
                or len(value) != expected_length
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise HellaSwagError(f"source {field} must be a lowercase digest")
        for field in ("validation_bytes", "validation_rows", "sample_bytes"):
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HellaSwagError(f"source {field} must be a positive integer")
        if payload["license"] != "MIT":
            raise HellaSwagError("HellaSwag source license must remain MIT")
        return cls(**payload)

    def verify_sample(self, source: bytes) -> None:
        if not isinstance(source, bytes):
            raise TypeError("sample response must be bytes")
        if (
            len(source) != self.sample_bytes
            or sha256(source).hexdigest() != self.sample_sha256
        ):
            raise HellaSwagError(
                "sample response does not match the pinned public snapshot"
            )


@dataclass(frozen=True)
class HellaSwagProtocol:
    """Complete bounded controls for one reproducible validation run."""

    source: HellaSwagSource
    start: int
    count: int
    world_size: int
    bootstrap_samples: int
    bootstrap_seed: int

    @classmethod
    def from_mapping(cls, payload: dict) -> "HellaSwagProtocol":
        fields = {
            "schema_version",
            "source",
            "start",
            "count",
            "world_size",
            "bootstrap_samples",
            "bootstrap_seed",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
        ):
            raise HellaSwagError("HellaSwag protocol schema does not match")
        source = HellaSwagSource.from_mapping(payload["source"])
        controls = {
            field: payload[field]
            for field in (
                "start",
                "count",
                "world_size",
                "bootstrap_samples",
                "bootstrap_seed",
            )
        }
        for name, (minimum, maximum) in {
            "start": (0, source.validation_rows - 1),
            "count": (1, 1000),
            "world_size": (1, 16),
            "bootstrap_samples": (100, 10000),
            "bootstrap_seed": (0, 2**32 - 1),
        }.items():
            value = controls[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise HellaSwagError(f"protocol {name} is outside its budget")
        if controls["start"] + controls["count"] > source.validation_rows:
            raise HellaSwagError("protocol slice exceeds the validation set")
        return cls(source=source, **controls)

    @property
    def fingerprint(self) -> str:
        payload = protocol_mapping(self)
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def protocol_mapping(protocol: HellaSwagProtocol) -> dict:
    if not isinstance(protocol, HellaSwagProtocol):
        raise TypeError("protocol must be HellaSwagProtocol")
    return {
        "schema_version": 1,
        "source": protocol.source.__dict__,
        "start": protocol.start,
        "count": protocol.count,
        "world_size": protocol.world_size,
        "bootstrap_samples": protocol.bootstrap_samples,
        "bootstrap_seed": protocol.bootstrap_seed,
    }


@dataclass(frozen=True)
class HellaSwagRecord:
    """Dataset row ordinal paired with the benchmark's independent example ID."""

    row_index: int
    example: HellaSwagExample


def records_from_dataset_server(source: bytes) -> tuple[HellaSwagRecord, ...]:
    """Parse a bounded Hugging Face dataset-server first-rows response."""
    if not isinstance(source, bytes):
        raise TypeError("dataset response must be bytes")
    if not 0 < len(source) <= 2 * 1024 * 1024:
        raise HellaSwagError("dataset response exceeds the input budget")
    try:
        payload = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HellaSwagError("dataset response must be valid UTF-8 JSON") from error
    required = {"dataset", "config", "split", "features", "rows", "truncated"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["dataset"] != "Rowan/hellaswag"
        or payload["config"] != "default"
        or payload["split"] != "validation"
        or type(payload["truncated"]) is not bool
        or not isinstance(payload["features"], list)
        or not isinstance(payload["rows"], list)
        or not 1 <= len(payload["rows"]) <= 1000
    ):
        raise HellaSwagError("dataset response metadata does not match HellaSwag")
    records = []
    for entry in payload["rows"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"row_idx", "row", "truncated_cells"}
            or isinstance(entry["row_idx"], bool)
            or not isinstance(entry["row_idx"], int)
            or entry["row_idx"] < 0
            or entry["truncated_cells"] != []
        ):
            raise HellaSwagError("dataset row envelope is incomplete or truncated")
        example = HellaSwagExample.from_mapping(entry["row"])
        records.append(HellaSwagRecord(entry["row_idx"], example))
    if (
        len({record.row_index for record in records}) != len(records)
        or len({record.example.index for record in records}) != len(records)
        or len({record.example.digest for record in records}) != len(records)
    ):
        raise HellaSwagError("dataset response contains duplicate examples")
    return tuple(records)


def select_records(
    records, *, start: int = 0, count: int = 100
) -> tuple[HellaSwagRecord, ...]:
    """Select an exact contiguous validation slice independent of response order."""
    for name, value, minimum, maximum in (
        ("start", start, 0, 10041),
        ("count", count, 1, 1000),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise HellaSwagError(f"{name} is outside the evaluation budget")
    records = tuple(records)
    if not records or any(
        not isinstance(record, HellaSwagRecord) for record in records
    ):
        raise HellaSwagError("selection requires validated HellaSwag records")
    selected = tuple(
        sorted(
            (record for record in records if start <= record.row_index < start + count),
            key=lambda record: record.row_index,
        )
    )
    if [record.row_index for record in selected] != list(range(start, start + count)):
        raise HellaSwagError("dataset response does not cover the requested slice")
    return selected


def records_digest(records) -> str:
    records = tuple(records)
    if not records or any(
        not isinstance(record, HellaSwagRecord) for record in records
    ):
        raise HellaSwagError("digest requires validated HellaSwag records")
    payload = [
        {"row_index": record.row_index, "example_sha256": record.example.digest}
        for record in records
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def encode_example(
    example: HellaSwagExample,
    *,
    max_context_bytes: int = 4096,
    max_ending_bytes: int = 1024,
) -> tuple[bytes, tuple[bytes, bytes, bytes, bytes]]:
    """Encode the canonical prompt and space-prefixed candidate continuations."""
    if not isinstance(example, HellaSwagExample):
        raise TypeError("example must be HellaSwagExample")
    for name, value, limit in (
        ("max_context_bytes", max_context_bytes, 16384),
        ("max_ending_bytes", max_ending_bytes, 4096),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= limit
        ):
            raise HellaSwagError(f"{name} is outside the text budget")
    prompt = example.canonical_context.rstrip().encode("utf-8")
    endings = tuple(f" {ending.lstrip()}".encode("utf-8") for ending in example.endings)
    if not prompt or len(prompt) > max_context_bytes:
        raise HellaSwagError("encoded context exceeds the text budget")
    if any(not ending.strip() or len(ending) > max_ending_bytes for ending in endings):
        raise HellaSwagError("encoded ending exceeds the text budget")
    return prompt, endings


def score_record(model, record: HellaSwagRecord) -> dict:
    """Score one HellaSwag row by completion-only mean negative log likelihood."""
    from .ddp_evaluation import score_choices

    if not isinstance(record, HellaSwagRecord):
        raise TypeError("record must be HellaSwagRecord")
    prompt, endings = encode_example(record.example)
    scoring = score_choices(model, prompt, endings)
    prediction = scoring["prediction"]
    probabilities = probabilities_from_scores(scoring["scores"])
    return {
        "row_index": record.row_index,
        "example_index": record.example.index,
        "example_sha256": record.example.digest,
        "label": record.example.label,
        "prediction": prediction,
        "correct": prediction == record.example.label,
        "scores": scoring["scores"],
        "probabilities": probabilities,
        "confidence": probabilities[prediction],
        "normalization": scoring["normalization"],
        "split_type": record.example.split_type,
    }


def probabilities_from_scores(scores) -> list[float]:
    """Convert normalized candidate losses to a stable four-way distribution."""
    scores = tuple(scores)
    if (
        len(scores) != 4
        or any(
            not isinstance(score, dict) or "mean_nll" not in score for score in scores
        )
        or any(
            isinstance(score["mean_nll"], bool)
            or not isinstance(score["mean_nll"], (int, float))
            or not math.isfinite(score["mean_nll"])
            or score["mean_nll"] < 0
            for score in scores
        )
    ):
        raise HellaSwagError("four finite nonnegative candidate losses are required")
    logits = [-float(score["mean_nll"]) for score in scores]
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def evaluate_records(
    model, records, *, bootstrap_samples: int = 2000, bootstrap_seed: int = 49
) -> dict:
    """Evaluate every selected row once with accuracy, NLL and calibration."""
    records = tuple(records)
    if not records or any(
        not isinstance(record, HellaSwagRecord) for record in records
    ):
        raise HellaSwagError("evaluation requires validated HellaSwag records")
    if len({record.row_index for record in records}) != len(records):
        raise HellaSwagError("evaluation rows must be unique")
    results = [score_record(model, record) for record in records]
    return summarize_scores(
        results, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed
    )


def summarize_scores(
    results, *, bootstrap_samples: int = 2000, bootstrap_seed: int = 49
) -> dict:
    """Reduce validated per-row scores into one benchmark summary."""
    from .ddp_metrics import calibration_metrics

    results = tuple(results)
    if not results or len({result.get("row_index") for result in results}) != len(
        results
    ):
        raise HellaSwagError("summary requires unique scored rows")
    correct = sum(result["correct"] for result in results)
    mean_correct_nll = math.fsum(
        result["scores"][result["label"]]["mean_nll"] for result in results
    ) / len(results)
    return {
        "examples": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "accuracy_interval": bootstrap_accuracy(
            results, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "mean_correct_nll": mean_correct_nll,
        "calibration": calibration_metrics(
            [result["probabilities"] for result in results],
            [result["label"] for result in results],
        ),
        "split_types": split_type_metrics(results),
        "records": results,
    }


def split_type_metrics(results) -> dict:
    """Report in-domain and zero-shot strata without hiding absent groups."""
    results = tuple(results)
    if not results:
        raise HellaSwagError("split metrics require scored records")
    output = {}
    for split_type in ("indomain", "zeroshot"):
        group = [result for result in results if result.get("split_type") == split_type]
        if not group:
            output[split_type] = None
            continue
        correct = sum(result["correct"] for result in group)
        output[split_type] = {
            "examples": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "mean_correct_nll": math.fsum(
                result["scores"][result["label"]]["mean_nll"] for result in group
            )
            / len(group),
        }
    if sum(value is not None for value in output.values()) == 0:
        raise HellaSwagError("scored records contain no known split types")
    return output


def bootstrap_accuracy(results, *, samples: int = 2000, seed: int = 49) -> dict:
    """Compute a deterministic 95% nonparametric interval for accuracy."""
    results = tuple(results)
    if not results or any(
        type(result.get("correct")) is not bool for result in results
    ):
        raise HellaSwagError("bootstrap requires scored boolean outcomes")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or not 100 <= samples <= 10000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise HellaSwagError("bootstrap controls exceed the evaluation budget")
    generator = random.Random(seed)
    outcomes = [int(result["correct"]) for result in results]
    estimates = sorted(
        math.fsum(generator.choice(outcomes) for _ in outcomes) / len(outcomes)
        for _ in range(samples)
    )
    lower = estimates[math.floor(0.025 * (samples - 1))]
    upper = estimates[math.ceil(0.975 * (samples - 1))]
    return {
        "method": "nonparametric_percentile",
        "confidence": 0.95,
        "samples": samples,
        "seed": seed,
        "lower": lower,
        "upper": upper,
    }


def shard_records(
    records, *, rank: int, world_size: int
) -> tuple[HellaSwagRecord, ...]:
    """Partition validation rows without duplication or padding."""
    records = tuple(records)
    if not records or any(
        not isinstance(record, HellaSwagRecord) for record in records
    ):
        raise HellaSwagError("sharding requires validated HellaSwag records")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or not 1 <= world_size <= 16
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < world_size
    ):
        raise HellaSwagError("shard topology is invalid")
    ordered = sorted(records, key=lambda record: record.row_index)
    return tuple(ordered[rank::world_size])


def merge_evaluations(shards, expected_records) -> dict:
    """Merge complete disjoint shard results against selected dataset identities."""
    expected_records = tuple(expected_records)
    expected = {record.row_index: record.example.digest for record in expected_records}
    if not expected or len(expected) != len(expected_records):
        raise HellaSwagError("expected evaluation rows must be unique")
    shards = tuple(shards)
    if not shards or any(not isinstance(shard, dict) for shard in shards):
        raise HellaSwagError("at least one shard evaluation is required")
    rows = []
    for shard in shards:
        shard_rows = shard.get("records")
        if not isinstance(shard_rows, (list, tuple)) or not shard_rows:
            raise HellaSwagError("shard evaluation has no scored rows")
        rows.extend(shard_rows)
    observed = [row.get("row_index") for row in rows]
    if len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise HellaSwagError("shard evaluations do not cover the selected rows exactly")
    if any(row.get("example_sha256") != expected[row["row_index"]] for row in rows):
        raise HellaSwagError("shard example identity does not match the selected data")
    return summarize_scores(sorted(rows, key=lambda row: row["row_index"]))


def compare_evaluations(baseline: dict, candidate: dict) -> dict:
    """Compare paired benchmark outcomes with an exact two-sided McNemar test."""
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise TypeError("evaluations must be dictionaries")
    before = baseline.get("records")
    after = candidate.get("records")
    if not isinstance(before, (list, tuple)) or not isinstance(after, (list, tuple)):
        raise HellaSwagError("paired evaluations require scored rows")
    before_by_row = {row.get("row_index"): row for row in before}
    after_by_row = {row.get("row_index"): row for row in after}
    if (
        len(before_by_row) != len(before)
        or len(after_by_row) != len(after)
        or set(before_by_row) != set(after_by_row)
        or not before_by_row
    ):
        raise HellaSwagError("paired evaluations do not cover identical rows")
    pairs = []
    for row_index in sorted(before_by_row):
        left, right = before_by_row[row_index], after_by_row[row_index]
        if any(
            left.get(field) != right.get(field) for field in ("example_sha256", "label")
        ):
            raise HellaSwagError("paired evaluation identities do not match")
        pairs.append((bool(left["correct"]), bool(right["correct"])))
    baseline_only = sum(left and not right for left, right in pairs)
    candidate_only = sum(right and not left for left, right in pairs)
    discordant = baseline_only + candidate_only
    if discordant:
        tail = (
            sum(
                math.comb(discordant, value)
                for value in range(0, min(baseline_only, candidate_only) + 1)
            )
            / 2**discordant
        )
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "examples": len(pairs),
        "both_correct": sum(left and right for left, right in pairs),
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_wrong": sum(not left and not right for left, right in pairs),
        "accuracy_delta": candidate["accuracy"] - baseline["accuracy"],
        "mean_correct_nll_delta": candidate["mean_correct_nll"]
        - baseline["mean_correct_nll"],
        "mcnemar_exact_p_value": p_value,
    }


def run_evaluation(model, response: bytes, protocol: HellaSwagProtocol) -> dict:
    """Run a pinned validation slice through disjoint local evaluation shards."""
    from .gpt2_model import GPT2LanguageModel
    from .decoder_variant_model import DecoderVariantModel
    from .gpt2_training import gpt2_model_fingerprint
    from .training_models import (
        base_model_config,
        expected_training_parameters,
        model_config_mapping,
    )
    import torch

    if not isinstance(model, (GPT2LanguageModel, DecoderVariantModel)):
        raise TypeError("model must be GPT2LanguageModel or DecoderVariantModel")
    if base_model_config(model).vocab_size != 256:
        raise HellaSwagError("byte evaluation requires vocab_size 256")
    if expected_training_parameters(model.config) > 1_000_000:
        raise HellaSwagError("evaluation model exceeds the CPU parameter budget")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise HellaSwagError("bounded evaluation requires CPU model parameters")
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if len(dtypes) != 1 or not dtypes <= {"torch.float32", "torch.float64"}:
        raise HellaSwagError(
            "bounded evaluation requires uniform float32 or float64 parameters"
        )
    if not isinstance(protocol, HellaSwagProtocol):
        raise TypeError("protocol must be HellaSwagProtocol")
    protocol.source.verify_sample(response)
    records = select_records(
        records_from_dataset_server(response),
        start=protocol.start,
        count=protocol.count,
    )
    shards = [
        evaluate_records(
            model,
            shard_records(records, rank=rank, world_size=protocol.world_size),
            bootstrap_samples=protocol.bootstrap_samples,
            bootstrap_seed=(protocol.bootstrap_seed + rank) % 2**32,
        )
        for rank in range(protocol.world_size)
        if shard_records(records, rank=rank, world_size=protocol.world_size)
    ]
    evaluation = merge_evaluations(shards, records)
    evaluation["accuracy_interval"] = bootstrap_accuracy(
        evaluation["records"],
        samples=protocol.bootstrap_samples,
        seed=protocol.bootstrap_seed,
    )
    return {
        "schema_version": 2,
        "protocol_sha256": protocol.fingerprint,
        "sample_sha256": sha256(response).hexdigest(),
        "records_sha256": records_digest(records),
        "model_config": model_config_mapping(model.config),
        "model_sha256": gpt2_model_fingerprint(model),
        "parameter_count": expected_training_parameters(model.config),
        "model_dtype": next(iter(dtypes)),
        "torch_threads": torch.get_num_threads(),
        "world_size": protocol.world_size,
        "evaluation": evaluation,
    }


def _payload_digest(payload: dict) -> str:
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def seal_evidence(report: dict, protocol: HellaSwagProtocol) -> dict:
    """Seal measured benchmark output with protocol, runtime and scope limitations."""
    import torch

    if (
        not isinstance(report, dict)
        or report.get("protocol_sha256") != protocol.fingerprint
    ):
        raise HellaSwagError("benchmark report does not match the protocol")
    canonical_report = json.loads(json.dumps(report, allow_nan=False))
    payload = {
        "schema_version": 1,
        "benchmark": "hellaswag_validation",
        "protocol": protocol_mapping(protocol),
        "run": canonical_report,
        "runtime": {
            "device": "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "operating_system": platform.system(),
            "machine": platform.machine(),
        },
        "limitations": HELLASWAG_LIMITATIONS.copy(),
    }
    payload["evidence_sha256"] = _payload_digest(payload)
    return verify_evidence(payload)


def verify_evidence(payload: dict) -> dict:
    """Reject modified HellaSwag evidence or metrics inconsistent with its rows."""
    fields = {
        "schema_version",
        "benchmark",
        "protocol",
        "run",
        "runtime",
        "limitations",
        "evidence_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["benchmark"] != "hellaswag_validation"
    ):
        raise HellaSwagError("HellaSwag evidence schema does not match")
    unsigned = {
        key: value for key, value in payload.items() if key != "evidence_sha256"
    }
    if _payload_digest(unsigned) != payload["evidence_sha256"]:
        raise HellaSwagError("HellaSwag evidence digest does not match")
    protocol = HellaSwagProtocol.from_mapping(payload["protocol"])
    run = payload["run"]
    runtime = payload["runtime"]
    if (
        not isinstance(run, dict)
        or run.get("protocol_sha256") != protocol.fingerprint
        or run.get("sample_sha256") != protocol.source.sample_sha256
        or run.get("world_size") != protocol.world_size
        or payload["limitations"] != HELLASWAG_LIMITATIONS
        or not isinstance(runtime, dict)
        or set(runtime) != {"device", "torch", "python", "operating_system", "machine"}
        or any(not isinstance(value, str) or not value for value in runtime.values())
        or runtime.get("device") != "cpu"
    ):
        raise HellaSwagError("HellaSwag evidence provenance does not match")
    from .training_models import (
        base_model_config,
        expected_training_parameters,
        model_config_from_mapping,
    )

    if type(run.get("schema_version")) is not int or run["schema_version"] not in (
        1,
        2,
    ):
        raise HellaSwagError("HellaSwag run schema does not match")
    run_fields = {
        "schema_version",
        "protocol_sha256",
        "sample_sha256",
        "records_sha256",
        "model_config",
        "model_sha256",
        "world_size",
        "evaluation",
    }
    if run["schema_version"] == 2:
        run_fields |= {"parameter_count", "model_dtype", "torch_threads"}
    if set(run) != run_fields:
        raise HellaSwagError("HellaSwag run fields do not match")
    from .hellaswag_scored_contract import require_scoring_digest, validate_scored_rows

    for name in ("protocol_sha256", "sample_sha256", "records_sha256", "model_sha256"):
        require_scoring_digest(run[name], name)
    config = model_config_from_mapping(run.get("model_config"))
    if (
        base_model_config(config).vocab_size != 256
        or expected_training_parameters(config) > 1_000_000
    ):
        raise HellaSwagError(
            "HellaSwag model configuration exceeds byte CPU constraints"
        )
    if run["schema_version"] == 2 and (
        type(run.get("parameter_count")) is not int
        or run["parameter_count"] != expected_training_parameters(config)
        or run.get("model_dtype") not in ("torch.float32", "torch.float64")
        or type(run.get("torch_threads")) is not int
        or run["torch_threads"] <= 0
    ):
        raise HellaSwagError(
            "HellaSwag model accounting does not match its architecture"
        )
    evaluation = run.get("evaluation")
    records = evaluation.get("records") if isinstance(evaluation, dict) else None
    if not isinstance(records, (list, tuple)) or len(records) != protocol.count:
        raise HellaSwagError("HellaSwag evidence row count does not match")
    if validate_scored_rows(records, protocol) != run["records_sha256"]:
        raise HellaSwagError(
            "HellaSwag selected-record digest does not match scored rows"
        )
    expected = summarize_scores(
        records,
        bootstrap_samples=protocol.bootstrap_samples,
        bootstrap_seed=protocol.bootstrap_seed,
    )
    expected["records"] = records
    if evaluation != expected:
        raise HellaSwagError("HellaSwag aggregate metrics do not match scored rows")
    return payload
