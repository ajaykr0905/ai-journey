"""Validate complete paired trial matrices and expose seed-level uncertainty."""

from dataclasses import asdict
import math
import re
from statistics import fmean, stdev

from .bpe import TokenizerError
from .bpe_likelihood import DocumentLikelihood, summarize_likelihood
from .bpe_model_ablation import VocabularyProtocol, protocol_fingerprint
from .transformer_lab import TransformerConfig, TrainingConfig, expected_parameter_count


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _score(value):
    documents = tuple(DocumentLikelihood(**row) for row in value["documents"])
    for row in documents:
        if (
            not _digest(row.text_sha256)
            or any(
                type(count) is not int or count <= 0
                for count in (row.utf8_bytes, row.codepoints, row.content_tokens)
            )
            or row.utf8_bytes < row.codepoints
            or any(
                type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                for loss in (row.content_nll, row.eos_nll)
            )
        ):
            raise TokenizerError("invalid document likelihood evidence")
    if value != summarize_likelihood(documents):
        raise TokenizerError("document likelihood totals or denominators were changed")
    return tuple((row.text_sha256, row.utf8_bytes, row.codepoints) for row in documents)


def summarize_ablation(report: dict) -> dict:
    """Never silently average missing, duplicated, or incomparable seed arms.

    The sample SD and standard error describe initialization-seed variation only.
    With two to five seeds these are diagnostics, not a confidence/quality claim.
    """
    try:
        raw = report["protocol"]
        protocol = VocabularyProtocol(
            **{
                **raw,
                "vocab_sizes": tuple(raw["vocab_sizes"]),
                "seeds": tuple(raw["seeds"]),
            }
        )
        if (
            report["schema"] != "bpe-model-ablation-v1"
            or report["protocol_fingerprint"] != protocol_fingerprint(protocol)
            or not _digest(report["source_fingerprint"])
        ):
            raise TokenizerError("invalid ablation schema or protocol identity")
        matrix = {}
        heldout = None
        codecs, trunks = {}, {}
        for trial in report["trials"]:
            seed, budget = trial["seed"], trial["requested_vocab_size"]
            key = (seed, budget)
            if (
                type(seed) is not int
                or type(budget) is not int
                or seed not in protocol.seeds
                or budget not in protocol.vocab_sizes
                or key in matrix
            ):
                raise TokenizerError("trial matrix has duplicate or unplanned arms")
            actual = trial["actual_vocab_size_with_eos"]
            if type(actual) is not int or not 257 <= actual <= budget + 1:
                raise TokenizerError("actual vocabulary exceeds requested budget")
            architecture = TransformerConfig(
                vocab_size=actual,
                block_size=protocol.block_size,
                embedding_dim=16,
                head_count=2,
                layer_count=1,
            )
            training = TrainingConfig(
                steps=protocol.steps, batch_size=protocol.batch_size, seed=seed
            )
            if (
                trial["model_config"] != asdict(architecture)
                or trial["training_config"] != asdict(training)
                or trial["parameter_count"] != expected_parameter_count(architecture)
                or trial["training_target_tokens"]
                != protocol.steps * protocol.batch_size * protocol.block_size
            ):
                raise TokenizerError(
                    "trial controls or parameter/exposure counts changed"
                )
            for field in (
                "corpus_fingerprint",
                "codec_fingerprint",
                "initial_trunk_fingerprint",
                "initial_model_fingerprint",
                "final_model_fingerprint",
            ):
                if not _digest(trial[field]):
                    raise TokenizerError("invalid trial identity")
            identity = (trial["codec_fingerprint"], trial["corpus_fingerprint"], actual)
            if (
                codecs.setdefault(budget, identity) != identity
                or trunks.setdefault(seed, trial["initial_trunk_fingerprint"])
                != trial["initial_trunk_fingerprint"]
            ):
                raise TokenizerError(
                    "paired trials do not share tokenizer data or initial trunk"
                )
            for phase in ("initial", "final"):
                ids = _score(trial[phase])
                if heldout is None:
                    heldout = ids
                if ids != heldout:
                    raise TokenizerError(
                        "held-out document identities differ across trials"
                    )
            trace = trial["trace"]
            if len(trace) != protocol.steps or any(
                item["step"] != index + 1
                or any(
                    type(item[field]) not in (int, float)
                    or not math.isfinite(item[field])
                    or item[field] < 0
                    for field in ("loss", "gradient_norm")
                )
                for index, item in enumerate(trace)
            ):
                raise TokenizerError("training trace is incomplete or nonfinite")
            if (
                type(trial["training_seconds"]) not in (int, float)
                or not math.isfinite(trial["training_seconds"])
                or trial["training_seconds"] < 0
            ):
                raise TokenizerError("invalid training duration")
            matrix[key] = trial
        if (
            set(matrix)
            != {
                (seed, size) for seed in protocol.seeds for size in protocol.vocab_sizes
            }
            or len({row[2] for row in codecs.values()}) != 3
        ):
            raise TokenizerError(
                "trial matrix must contain three realized arms for every seed"
            )
        arms = []
        for budget in protocol.vocab_sizes:
            rows = [matrix[seed, budget] for seed in protocol.seeds]
            scores = [row["final"]["content_bits_per_byte"] for row in rows]
            deltas = [
                score - matrix[seed, 256]["final"]["content_bits_per_byte"]
                for seed, score in zip(protocol.seeds, scores)
            ]
            arms.append(
                {
                    "requested_vocab_size": budget,
                    "actual_vocab_size_with_eos": rows[0]["actual_vocab_size_with_eos"],
                    "parameter_count": rows[0]["parameter_count"],
                    "seeds": list(protocol.seeds),
                    "mean_bits_per_byte": fmean(scores),
                    "seed_sd_bits_per_byte": stdev(scores),
                    "paired_deltas_vs_byte": deltas,
                    "mean_paired_delta": fmean(deltas),
                    "paired_delta_standard_error": stdev(deltas)
                    / math.sqrt(len(deltas)),
                    "paired_delta_range": [min(deltas), max(deltas)],
                    "bytes_per_content_token": rows[0]["final"][
                        "bytes_per_content_token"
                    ],
                }
            )
        return {
            "arms": arms,
            "uncertainty_scope": "initialization seeds only; not dataset uncertainty or statistical significance",
            "comparison_scope": "within-language canonical BPE segmentation at fixed token exposure; no optimal-vocabulary claim",
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise TokenizerError(f"invalid ablation evidence: {exc}") from exc
