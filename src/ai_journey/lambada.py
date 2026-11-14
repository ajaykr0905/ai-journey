"""Bounded LAMBADA last-word evaluation for the public transformer lab."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import platform
import random
import re


class LambadaError(ValueError):
    """Raised when LAMBADA inputs or evidence violate their contract."""


LAMBADA_LIMITATIONS = [
    "A bounded test slice is not the official 5,153-example benchmark score.",
    "Byte-level exact match is not directly comparable to token-level GPT-2 results.",
    "Exact-byte overlap checks do not detect paraphrased or translated contamination.",
    "Automated execution does not prove lecture viewing or learner understanding.",
]


@dataclass(frozen=True)
class LambadaSource:
    """Pinned identity for the public OpenAI-processed LAMBADA test set."""

    repository: str
    revision: str
    dataset_server_url: str
    sample_sha256: str
    sample_bytes: int
    test_rows: int
    source_file_sha256: str
    license: str

    @classmethod
    def from_mapping(cls, payload: dict) -> "LambadaSource":
        """Validate the pinned lambada_openai identity or raise LambadaError.

        Fields must match exactly, digests must be lowercase hex, sizes positive
        and the license MIT.
        """
        fields = {
            "repository",
            "revision",
            "dataset_server_url",
            "sample_sha256",
            "sample_bytes",
            "test_rows",
            "source_file_sha256",
            "license",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise LambadaError("LAMBADA source fields do not match")
        for field in ("repository", "dataset_server_url", "license"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise LambadaError(f"source {field} must be nonempty text")
        if payload["repository"] != "EleutherAI/lambada_openai":
            raise LambadaError("source repository must identify lambada_openai")
        if payload["license"] != "MIT":
            raise LambadaError("source license must remain MIT")
        for field, length in (
            ("revision", 40),
            ("sample_sha256", 64),
            ("source_file_sha256", 64),
        ):
            value = payload[field]
            if (
                not isinstance(value, str)
                or len(value) != length
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise LambadaError(f"source {field} must be a lowercase digest")
        for field in ("sample_bytes", "test_rows"):
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LambadaError(f"source {field} must be a positive integer")
        return cls(**payload)

    def verify_sample(self, source: bytes) -> None:
        """Reject a dataset response whose size or SHA-256 differs from the pin.

        Evaluation checks this before parsing the rows it will score.
        """
        if not isinstance(source, bytes):
            raise TypeError("sample response must be bytes")
        if (
            len(source) != self.sample_bytes
            or sha256(source).hexdigest() != self.sample_sha256
        ):
            raise LambadaError("sample response does not match the pinned snapshot")


@dataclass(frozen=True)
class LambadaExample:
    """One bounded narrative and its exact final-word target."""

    row_index: int
    text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise LambadaError("example row index must be nonnegative")
        if not isinstance(self.text, str) or not self.text.strip():
            raise LambadaError("example text must be nonempty")
        encoded = self.text.encode("utf-8")
        if len(encoded) > 32 * 1024 or "\x00" in self.text:
            raise LambadaError("example text exceeds the input contract")

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the row index and exact text, its identity."""
        payload = {"row_index": self.row_index, "text": self.text}
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class LambadaProtocol:
    """Complete controls for one bounded last-word evaluation."""

    source: LambadaSource
    start: int
    count: int
    bootstrap_samples: int
    bootstrap_seed: int
    max_context_bytes: int
    max_target_bytes: int

    @classmethod
    def from_mapping(cls, payload: dict) -> "LambadaProtocol":
        """Validate a schema-1 protocol whose slice fits the test set.

        Each control has a budget (up to 1000 rows, 32 KiB contexts and 256-byte
        targets); violations raise LambadaError.
        """
        fields = {
            "schema_version",
            "source",
            "start",
            "count",
            "bootstrap_samples",
            "bootstrap_seed",
            "max_context_bytes",
            "max_target_bytes",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
        ):
            raise LambadaError("LAMBADA protocol schema does not match")
        source = LambadaSource.from_mapping(payload["source"])
        controls = {
            field: payload[field] for field in fields - {"schema_version", "source"}
        }
        limits = {
            "start": (0, source.test_rows - 1),
            "count": (1, 1000),
            "bootstrap_samples": (100, 10_000),
            "bootstrap_seed": (0, 2**32 - 1),
            "max_context_bytes": (1, 32 * 1024),
            "max_target_bytes": (1, 256),
        }
        for name, (minimum, maximum) in limits.items():
            value = controls[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise LambadaError(f"protocol {name} is outside its budget")
        if controls["start"] + controls["count"] > source.test_rows:
            raise LambadaError("protocol slice exceeds the test set")
        return cls(source=source, **controls)

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 of the canonical protocol mapping that runs record."""
        return sha256(
            json.dumps(
                protocol_mapping(self), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


def protocol_mapping(protocol: LambadaProtocol) -> dict:
    """Return the schema-1 JSON form that fingerprints and sealed evidence use."""
    if not isinstance(protocol, LambadaProtocol):
        raise TypeError("protocol must be LambadaProtocol")
    return {
        "schema_version": 1,
        "source": protocol.source.__dict__,
        "start": protocol.start,
        "count": protocol.count,
        "bootstrap_samples": protocol.bootstrap_samples,
        "bootstrap_seed": protocol.bootstrap_seed,
        "max_context_bytes": protocol.max_context_bytes,
        "max_target_bytes": protocol.max_target_bytes,
    }


def examples_from_dataset_server(source: bytes) -> tuple[LambadaExample, ...]:
    """Parse a bounded Hugging Face dataset-server first-rows response."""
    if not isinstance(source, bytes):
        raise TypeError("dataset response must be bytes")
    if not 0 < len(source) <= 2 * 1024 * 1024:
        raise LambadaError("dataset response exceeds the input budget")
    try:
        payload = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LambadaError("dataset response is not valid UTF-8 JSON") from error
    required = {"dataset", "config", "split", "features", "rows", "truncated"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["dataset"] != "EleutherAI/lambada_openai"
        or payload["config"] != "default"
        or payload["split"] != "test"
        or type(payload["truncated"]) is not bool
        or not isinstance(payload["features"], list)
        or not isinstance(payload["rows"], list)
        or not 1 <= len(payload["rows"]) <= 1000
    ):
        raise LambadaError("dataset response metadata does not match LAMBADA")
    examples = []
    for entry in payload["rows"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"row_idx", "row", "truncated_cells"}
            or isinstance(entry["row_idx"], bool)
            or not isinstance(entry["row_idx"], int)
            or entry["row_idx"] < 0
            or entry["truncated_cells"] != []
            or not isinstance(entry["row"], dict)
            or set(entry["row"]) != {"text"}
        ):
            raise LambadaError("dataset row envelope is incomplete or truncated")
        examples.append(LambadaExample(entry["row_idx"], entry["row"]["text"]))
    if len({example.row_index for example in examples}) != len(examples) or len(
        {example.digest for example in examples}
    ) != len(examples):
        raise LambadaError("dataset response contains duplicate examples")
    return tuple(examples)


def select_examples(
    examples, *, start: int = 0, count: int = 100
) -> tuple[LambadaExample, ...]:
    """Select an exact contiguous test slice independent of response order."""
    for name, value, minimum, maximum in (
        ("start", start, 0, 5152),
        ("count", count, 1, 1000),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise LambadaError(f"{name} is outside the evaluation budget")
    examples = tuple(examples)
    if not examples or any(not isinstance(item, LambadaExample) for item in examples):
        raise LambadaError("selection requires validated LAMBADA examples")
    selected = tuple(
        sorted(
            (item for item in examples if start <= item.row_index < start + count),
            key=lambda item: item.row_index,
        )
    )
    if [item.row_index for item in selected] != list(range(start, start + count)):
        raise LambadaError("dataset response does not cover the requested slice")
    return selected


def examples_digest(examples) -> str:
    """Return one digest over each selected example's row index and digest."""
    examples = tuple(examples)
    if not examples or any(not isinstance(item, LambadaExample) for item in examples):
        raise LambadaError("digest requires validated LAMBADA examples")
    payload = [
        {"row_index": item.row_index, "example_sha256": item.digest}
        for item in examples
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class LambadaItem:
    """A source example split into context bytes and one final-word target."""

    row_index: int
    example_sha256: str
    prompt: bytes
    target: bytes


def encode_example(
    example: LambadaExample,
    *,
    max_context_bytes: int = 4096,
    max_target_bytes: int = 64,
) -> LambadaItem:
    """Split the final non-whitespace word without normalizing source text."""
    if not isinstance(example, LambadaExample):
        raise TypeError("example must be LambadaExample")
    for name, value, maximum in (
        ("max_context_bytes", max_context_bytes, 32 * 1024),
        ("max_target_bytes", max_target_bytes, 256),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise LambadaError(f"{name} is outside the text budget")
    match = re.fullmatch(r"(?s)(.+\s)(\S+)", example.text)
    if match is None:
        raise LambadaError("example must contain context and one final word")
    prompt = match.group(1).encode("utf-8")
    target = match.group(2).encode("utf-8")
    if len(prompt) > max_context_bytes or len(target) > max_target_bytes:
        raise LambadaError("encoded example exceeds the text budget")
    return LambadaItem(example.row_index, example.digest, prompt, target)


def score_target(model, item: LambadaItem) -> dict:
    """Measure final-word likelihood without leaking target bytes into context."""
    import torch

    from .ddp_evaluation import token_nll
    from .training_models import base_model_config

    if not isinstance(item, LambadaItem):
        raise TypeError("item must be LambadaItem")
    modes = [(module, module.training) for module in model.modules()]
    rng = torch.get_rng_state().clone()
    losses = []
    context = list(item.prompt)
    try:
        model.eval()
        with torch.inference_mode():
            for target in item.target:
                inputs = torch.tensor(
                    [context[-base_model_config(model).context_length :]],
                    dtype=torch.long,
                )
                logits = model(inputs)[:, -1:, :]
                loss = token_nll(logits, torch.tensor([[target]])).item()
                if not math.isfinite(loss) or loss < 0:
                    raise LambadaError(
                        "target likelihood must be finite and nonnegative"
                    )
                losses.append(loss)
                context.append(target)
    finally:
        torch.set_rng_state(rng)
        for module, mode in modes:
            module.training = mode
    loss_sum = math.fsum(losses)
    return {
        "target_bytes": len(item.target),
        "loss_sum": loss_sum,
        "mean_nll": loss_sum / len(item.target),
        "bits_per_byte": loss_sum / len(item.target) / math.log(2),
    }


def generate_target(model, item: LambadaItem) -> dict:
    """Greedily generate exactly the target byte length from context only."""
    import torch

    from .training_models import base_model_config

    if not isinstance(item, LambadaItem):
        raise TypeError("item must be LambadaItem")
    modes = [(module, module.training) for module in model.modules()]
    rng = torch.get_rng_state().clone()
    context = list(item.prompt)
    generated = bytearray()
    try:
        model.eval()
        with torch.inference_mode():
            for _ in item.target:
                inputs = torch.tensor(
                    [context[-base_model_config(model).context_length :]],
                    dtype=torch.long,
                )
                logits = model(inputs)[:, -1, :]
                if logits.shape != (1, 256) or not torch.isfinite(logits).all():
                    raise LambadaError(
                        "generation logits must cover finite byte scores"
                    )
                predicted = int(torch.argmax(logits, dim=-1).item())
                generated.append(predicted)
                context.append(predicted)
    finally:
        torch.set_rng_state(rng)
        for module, mode in modes:
            module.training = mode
    prediction = bytes(generated)
    prefix = 0
    for expected, actual in zip(item.target, prediction):
        if expected != actual:
            break
        prefix += 1
    return {
        "prediction_sha256": sha256(prediction).hexdigest(),
        "generated_bytes": len(prediction),
        "exact_match": prediction == item.target,
        "matching_prefix_bytes": prefix,
    }


def score_example(
    model,
    example: LambadaExample,
    *,
    max_context_bytes: int = 4096,
    max_target_bytes: int = 64,
) -> dict:
    """Score one row while keeping benchmark narrative and target text private."""
    item = encode_example(
        example,
        max_context_bytes=max_context_bytes,
        max_target_bytes=max_target_bytes,
    )
    return {
        "row_index": item.row_index,
        "example_sha256": item.example_sha256,
        "prompt_sha256": sha256(item.prompt).hexdigest(),
        "target_sha256": sha256(item.target).hexdigest(),
        "likelihood": score_target(model, item),
        "generation": generate_target(model, item),
    }


def bootstrap_exact_match(results, *, samples: int = 2000, seed: int = 63) -> dict:
    """Compute a deterministic percentile interval for last-word accuracy."""
    results = tuple(results)
    try:
        outcomes = [result["generation"]["exact_match"] for result in results]
    except (KeyError, TypeError) as error:
        raise LambadaError("bootstrap requires scored LAMBADA rows") from error
    if not outcomes or any(type(outcome) is not bool for outcome in outcomes):
        raise LambadaError("bootstrap requires boolean exact-match outcomes")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or not 100 <= samples <= 10_000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise LambadaError("bootstrap controls exceed the evaluation budget")
    generator = random.Random(seed)
    values = [int(outcome) for outcome in outcomes]
    estimates = sorted(
        math.fsum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return {
        "method": "nonparametric_percentile",
        "confidence": 0.95,
        "samples": samples,
        "seed": seed,
        "lower": estimates[math.floor(0.025 * (samples - 1))],
        "upper": estimates[math.ceil(0.975 * (samples - 1))],
    }


def summarize_scores(
    results, *, bootstrap_samples: int = 2000, bootstrap_seed: int = 63
) -> dict:
    """Aggregate byte-weighted likelihood and exact last-word accuracy."""
    results = tuple(results)
    if not results or any(not isinstance(result, dict) for result in results):
        raise LambadaError("summary requires scored LAMBADA rows")
    rows = [result.get("row_index") for result in results]
    if any(isinstance(row, bool) or not isinstance(row, int) for row in rows) or len(
        set(rows)
    ) != len(rows):
        raise LambadaError("summary rows must be unique integers")
    try:
        target_bytes = sum(result["likelihood"]["target_bytes"] for result in results)
        loss_sum = math.fsum(result["likelihood"]["loss_sum"] for result in results)
        exact_matches = sum(result["generation"]["exact_match"] for result in results)
        prefix_bytes = sum(
            result["generation"]["matching_prefix_bytes"] for result in results
        )
    except (KeyError, TypeError) as error:
        raise LambadaError("summary rows do not match the score contract") from error
    if (
        isinstance(target_bytes, bool)
        or not isinstance(target_bytes, int)
        or target_bytes <= 0
        or not math.isfinite(loss_sum)
        or loss_sum < 0
        or not 0 <= exact_matches <= len(results)
        or not 0 <= prefix_bytes <= target_bytes
    ):
        raise LambadaError("summary totals are invalid")
    mean_nll = loss_sum / target_bytes
    return {
        "examples": len(results),
        "target_bytes": target_bytes,
        "loss_sum": loss_sum,
        "mean_nll": mean_nll,
        "bits_per_byte": mean_nll / math.log(2),
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_matches / len(results),
        "exact_match_interval": bootstrap_exact_match(
            results, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "matching_prefix_bytes": prefix_bytes,
        "matching_prefix_ratio": prefix_bytes / target_bytes,
        "records": list(results),
    }


def audit_contamination(
    training_source: bytes,
    examples,
    *,
    context_suffix_bytes: int = 128,
) -> dict:
    """Detect exact passage and long-context overlap with the training source."""
    if not isinstance(training_source, bytes):
        raise TypeError("training source must be bytes")
    if not 0 < len(training_source) <= 16 * 1024 * 1024:
        raise LambadaError("training source exceeds the audit budget")
    if (
        isinstance(context_suffix_bytes, bool)
        or not isinstance(context_suffix_bytes, int)
        or not 32 <= context_suffix_bytes <= 4096
    ):
        raise LambadaError("contamination suffix is outside the audit budget")
    examples = tuple(examples)
    if (
        not examples
        or len(examples) > 1000
        or any(not isinstance(example, LambadaExample) for example in examples)
        or len({example.row_index for example in examples}) != len(examples)
    ):
        raise LambadaError("contamination audit requires unique bounded examples")
    full_matches = []
    context_matches = []
    for example in examples:
        item = encode_example(
            example,
            max_context_bytes=32 * 1024,
            max_target_bytes=256,
        )
        if example.text.encode("utf-8") in training_source:
            full_matches.append(example.row_index)
        if (
            len(item.prompt) >= context_suffix_bytes
            and item.prompt[-context_suffix_bytes:] in training_source
        ):
            context_matches.append(example.row_index)
    contaminated = sorted(set(full_matches) | set(context_matches))
    return {
        "method": "exact_bytes",
        "training_sha256": sha256(training_source).hexdigest(),
        "training_bytes": len(training_source),
        "examples": len(examples),
        "context_suffix_bytes": context_suffix_bytes,
        "full_text_match_rows": full_matches,
        "context_suffix_match_rows": context_matches,
        "contaminated_rows": contaminated,
        "passed": not contaminated,
    }


def run_evaluation(
    model,
    response: bytes,
    training_source: bytes,
    protocol: LambadaProtocol,
) -> dict:
    """Evaluate one pinned LAMBADA slice with explicit contamination evidence."""
    import torch

    from .decoder_variant_model import DecoderVariantModel
    from .gpt2_model import GPT2LanguageModel
    from .gpt2_training import gpt2_model_fingerprint
    from .training_models import (
        base_model_config,
        expected_training_parameters,
        model_config_mapping,
    )

    if not isinstance(model, (GPT2LanguageModel, DecoderVariantModel)):
        raise TypeError("model must be GPT2LanguageModel or DecoderVariantModel")
    if not isinstance(protocol, LambadaProtocol):
        raise TypeError("protocol must be LambadaProtocol")
    if base_model_config(model).vocab_size != 256:
        raise LambadaError("byte evaluation requires vocab_size 256")
    if expected_training_parameters(model.config) > 1_000_000:
        raise LambadaError("evaluation model exceeds the CPU parameter budget")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise LambadaError("bounded evaluation requires CPU model parameters")
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if len(dtypes) != 1 or not dtypes <= {"torch.float32", "torch.float64"}:
        raise LambadaError("evaluation requires uniform float32 or float64 parameters")
    protocol.source.verify_sample(response)
    examples = select_examples(
        examples_from_dataset_server(response),
        start=protocol.start,
        count=protocol.count,
    )
    contamination = audit_contamination(training_source, examples)
    records = [
        score_example(
            model,
            example,
            max_context_bytes=protocol.max_context_bytes,
            max_target_bytes=protocol.max_target_bytes,
        )
        for example in examples
    ]
    evaluation = summarize_scores(
        records,
        bootstrap_samples=protocol.bootstrap_samples,
        bootstrap_seed=protocol.bootstrap_seed,
    )
    return {
        "schema_version": 1,
        "protocol_sha256": protocol.fingerprint,
        "sample_sha256": sha256(response).hexdigest(),
        "examples_sha256": examples_digest(examples),
        "training_sha256": sha256(training_source).hexdigest(),
        "model_config": model_config_mapping(model.config),
        "model_sha256": gpt2_model_fingerprint(model),
        "parameter_count": expected_training_parameters(model.config),
        "model_dtype": next(iter(dtypes)),
        "torch_threads": torch.get_num_threads(),
        "contamination": contamination,
        "evaluation": evaluation,
    }


def _payload_digest(payload: dict) -> str:
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def evaluation_runtime(report: dict) -> dict:
    """Return the runtime fields sealed with evidence.

    Verification requires a CPU device and the run's dtype and thread count;
    versions and platform names are descriptive.
    """
    import torch

    return {
        "device": "cpu",
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "model_dtype": report["model_dtype"],
        "torch_threads": report["torch_threads"],
    }


def seal_evidence(report: dict, protocol: LambadaProtocol) -> dict:
    """Seal one measured run with its protocol, runtime and claim boundary."""
    if (
        not isinstance(report, dict)
        or not isinstance(protocol, LambadaProtocol)
        or report.get("protocol_sha256") != protocol.fingerprint
    ):
        raise LambadaError("LAMBADA report does not match the protocol")
    canonical_report = json.loads(json.dumps(report, allow_nan=False))
    claim_status = (
        "bounded_result"
        if canonical_report["contamination"]["passed"]
        else "contamination_detected"
    )
    payload = {
        "schema_version": 1,
        "benchmark": "lambada_openai_last_word",
        "protocol": protocol_mapping(protocol),
        "run": canonical_report,
        "runtime": evaluation_runtime(canonical_report),
        "claim_status": claim_status,
        "limitations": LAMBADA_LIMITATIONS.copy(),
    }
    payload["evidence_sha256"] = _payload_digest(payload)
    return verify_evidence(payload)


def verify_evidence(payload: dict) -> dict:
    """Reject modified evidence and metrics inconsistent with scored rows."""
    fields = {
        "schema_version",
        "benchmark",
        "protocol",
        "run",
        "runtime",
        "claim_status",
        "limitations",
        "evidence_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload["schema_version"] != 1
        or payload["benchmark"] != "lambada_openai_last_word"
    ):
        raise LambadaError("LAMBADA evidence schema does not match")
    unsigned = {
        key: value for key, value in payload.items() if key != "evidence_sha256"
    }
    if _payload_digest(unsigned) != payload["evidence_sha256"]:
        raise LambadaError("LAMBADA evidence digest does not match")
    protocol = LambadaProtocol.from_mapping(payload["protocol"])
    run = payload["run"]
    contamination = run.get("contamination") if isinstance(run, dict) else None
    evaluation = run.get("evaluation") if isinstance(run, dict) else None
    runtime = payload["runtime"]
    if (
        not isinstance(run, dict)
        or not isinstance(contamination, dict)
        or run.get("protocol_sha256") != protocol.fingerprint
        or run.get("sample_sha256") != protocol.source.sample_sha256
        or run.get("training_sha256") != contamination.get("training_sha256")
        or not isinstance(evaluation, dict)
        or evaluation.get("examples") != protocol.count
        or not isinstance(runtime, dict)
        or set(runtime)
        != {
            "device",
            "torch",
            "python",
            "operating_system",
            "machine",
            "model_dtype",
            "torch_threads",
        }
        or runtime.get("device") != "cpu"
        or runtime.get("model_dtype") != run.get("model_dtype")
        or runtime.get("torch_threads") != run.get("torch_threads")
        or payload["limitations"] != LAMBADA_LIMITATIONS
    ):
        raise LambadaError("LAMBADA evidence constraints do not match")
    contaminated = sorted(
        set(contamination.get("full_text_match_rows", []))
        | set(contamination.get("context_suffix_match_rows", []))
    )
    expected_status = "bounded_result" if not contaminated else "contamination_detected"
    if (
        contamination.get("contaminated_rows") != contaminated
        or contamination.get("passed") is not (not contaminated)
        or payload["claim_status"] != expected_status
    ):
        raise LambadaError("contamination claim is inconsistent")
    expected_evaluation = summarize_scores(
        evaluation.get("records", []),
        bootstrap_samples=protocol.bootstrap_samples,
        bootstrap_seed=protocol.bootstrap_seed,
    )
    if evaluation != expected_evaluation:
        raise LambadaError("LAMBADA metrics do not match scored rows")
    return payload


def compare_evaluations(baseline: dict, candidate: dict) -> dict:
    """Compare paired last-word outcomes on identical benchmark rows."""
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise TypeError("evaluations must be dictionaries")
    before = baseline.get("records")
    after = candidate.get("records")
    if not isinstance(before, (list, tuple)) or not isinstance(after, (list, tuple)):
        raise LambadaError("paired evaluations require scored rows")
    before_by_row = {row.get("row_index"): row for row in before}
    after_by_row = {row.get("row_index"): row for row in after}
    if len(before_by_row) != len(before) or set(before_by_row) != set(after_by_row):
        raise LambadaError("paired evaluations must cover identical unique rows")
    pairs = []
    for row_index in sorted(before_by_row):
        left, right = before_by_row[row_index], after_by_row[row_index]
        if any(
            left.get(field) != right.get(field)
            for field in ("example_sha256", "prompt_sha256", "target_sha256")
        ):
            raise LambadaError("paired evaluation identities do not match")
        pairs.append(
            (
                left["generation"]["exact_match"],
                right["generation"]["exact_match"],
            )
        )
    baseline_only = sum(left and not right for left, right in pairs)
    candidate_only = sum(not left and right for left, right in pairs)
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = (
            sum(
                math.comb(discordant, index)
                for index in range(min(baseline_only, candidate_only) + 1)
            )
            / 2**discordant
        )
        p_value = min(1.0, 2 * tail)
    return {
        "examples": len(pairs),
        "both_exact": sum(left and right for left, right in pairs),
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_wrong": sum(not left and not right for left, right in pairs),
        "exact_match_accuracy_delta": candidate["exact_match_accuracy"]
        - baseline["exact_match_accuracy"],
        "mean_nll_delta": candidate["mean_nll"] - baseline["mean_nll"],
        "matching_prefix_ratio_delta": candidate["matching_prefix_ratio"]
        - baseline["matching_prefix_ratio"],
        "mcnemar_exact_p_value": p_value,
    }
