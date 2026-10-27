"""Consistency checks for fixed, actual CPU prefix-serving readiness evidence."""

import copy
from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace

from .decoder_variant_config import DecoderVariantConfig
from .gpt2_config import GPT2Config
from .gqa_config import GQAConfig
from .http_load_report import _validate_observation
from .training_models import model_config_mapping

SCOPE = "actual loopback CPU observations from an untrained seeded control; digest verification establishes consistency, not measurement authenticity or speedup"
BUDGET = 32768
SNAPSHOT_FIELDS = (
    'ai_journey_prefix_attempts_total{outcome="hit"}',
    'ai_journey_prefix_attempts_total{outcome="miss"}',
    'ai_journey_prefix_attempts_total{outcome="bypass"}',
    "ai_journey_prefix_reused_prompt_tokens_total",
    "ai_journey_prefix_forwarded_prefill_tokens_total",
    "ai_journey_prefix_resident_bytes",
    "ai_journey_prefix_leased_bytes",
    "ai_journey_prefix_reserved_bytes",
    "ai_journey_prefix_active_leases",
    "ai_journey_prefix_entries",
    "ai_journey_prefix_max_bytes",
    "ai_journey_prefix_max_entries",
    "ai_journey_serving_generated_tokens_total",
    "ai_journey_serving_cache_hits_total",
)


def control_config():
    return GQAConfig(
        DecoderVariantConfig(
            GPT2Config(
                256,
                8,
                16,
                4,
                2,
                embedding_dropout=0,
                attention_dropout=0,
                residual_dropout=0,
            ),
            position_kind="rope",
        ),
        1,
    )


def schedule():
    for label, prompt, temperature, reused in (
        ("cold", [1, 2, 3, 4], 0, 0),
        ("warm", [1, 2, 3, 4], 0, 4),
        ("branch", [1, 2, 3, 5], 0, 3),
        ("interior", [1, 2], 0, 1),
        ("miss", [6, 7, 6, 5], 0, 0),
        ("seeded", [1, 2, 3, 4], 0.8, 4),
    ):
        yield {
            "label": label,
            "request": {
                "request_id": f"prefix-{label}",
                "prompt": prompt,
                "sampling": {
                    "max_new_tokens": 8,
                    "temperature": temperature,
                    "seed": 53,
                },
            },
            "reused": reused,
        }


def seal_prefix_report(report):
    result = copy.deepcopy(report)
    result.pop("report_sha256", None)
    result["report_sha256"] = sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return result


def _snapshot(value):
    if (
        type(value) is not dict
        or set(value) != set(SNAPSHOT_FIELDS)
        or any(
            type(item) is not int or not 0 <= item < 2**63 for item in value.values()
        )
    ):
        raise ValueError("prefix metric snapshot has invalid typed fields")
    if (
        value["ai_journey_prefix_max_bytes"] != BUDGET
        or value["ai_journey_prefix_max_entries"] != 8
    ):
        raise ValueError("prefix snapshot differs from the fixed readiness budget")
    if (
        sum(
            value[f"ai_journey_prefix_{kind}_bytes"]
            for kind in ("resident", "leased", "reserved")
        )
        > BUDGET
        or value["ai_journey_prefix_entries"] > 8
    ):
        raise ValueError("recorded prefix storage exceeds admission")
    if any(
        value[f"ai_journey_prefix_{name}"]
        for name in ("leased_bytes", "reserved_bytes", "active_leases")
    ):
        raise ValueError("settled readiness observations retain a prefix lease")
    if value["ai_journey_serving_cache_hits_total"] != 0:
        raise ValueError("readiness must generate fresh responses, never replay them")


def verify_prefix_report(report, *, directory=None):
    fields = {
        "schema",
        "scope",
        "model_config",
        "model_sha256",
        "architecture_sha256",
        "source_sha256",
        "spec_sha256",
        "checkpoint_sha256",
        "runtime",
        "rows",
        "cancellation",
        "final_metrics",
        "child_exit_code",
        "report_sha256",
    }
    if (
        type(report) is not dict
        or set(report) != fields
        or report["schema"] != "ai-journey-prefix-readiness-v1"
        or report["scope"] != SCOPE
    ):
        raise ValueError("prefix readiness report schema/scope differs")
    if seal_prefix_report(report)["report_sha256"] != report["report_sha256"]:
        raise ValueError("prefix readiness digest differs")
    for field in (
        "model_sha256",
        "architecture_sha256",
        "source_sha256",
        "spec_sha256",
        "checkpoint_sha256",
    ):
        if (
            type(report[field]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", report[field]) is None
        ):
            raise ValueError("prefix readiness requires SHA-256 bindings")
    config = model_config_mapping(control_config())
    if json.dumps(
        report["model_config"], sort_keys=True, separators=(",", ":")
    ) != json.dumps(config, sort_keys=True, separators=(",", ":")):
        raise ValueError(
            "prefix readiness requires the fixed native grouped-query control"
        )
    if (
        sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        != report["architecture_sha256"]
    ):
        raise ValueError("prefix readiness architecture digest differs")
    runtime = report["runtime"]
    if (
        type(runtime) is not dict
        or set(runtime)
        != {"python", "torch", "device", "dtype", "checkpoint_step", "trained"}
        or runtime["device"] != "cpu"
        or runtime["dtype"] != "float64"
        or type(runtime["checkpoint_step"]) is not int
        or runtime["checkpoint_step"] != 0
        or runtime["trained"] is not False
        or any(
            type(runtime[key]) is not str or not 1 <= len(runtime[key]) <= 128
            for key in ("python", "torch")
        )
    ):
        raise ValueError("prefix readiness runtime/control claim differs")
    expected = list(schedule())
    if type(report["rows"]) is not list or len(report["rows"]) != len(expected):
        raise ValueError("prefix readiness workload differs")
    previous = None
    published = set()
    last_finished = 0
    for index, (row, planned) in enumerate(zip(report["rows"], expected)):
        if (
            type(row) is not dict
            or set(row)
            != {"label", "request", "observation", "control_tokens", "before", "after"}
            or row["label"] != planned["label"]
            or json.dumps(row["request"], sort_keys=True)
            != json.dumps(planned["request"], sort_keys=True)
        ):
            raise ValueError("prefix readiness request differs from its fixed schedule")
        observation = row["observation"]
        _validate_observation(
            observation,
            {"request": row["request"], "cancel_after_tokens": None},
            SimpleNamespace(streaming=index != 0, max_new_tokens=8, timeout_seconds=5),
        )
        if (
            observation["outcome"] != "completed"
            or observation["finish_reason"] != "length"
            or observation["model_sha256"] != report["model_sha256"]
            or type(row["control_tokens"]) is not list
            or any(
                type(token) is not int or not 0 <= token < 256
                for token in row["control_tokens"]
            )
            or observation["tokens"] != row["control_tokens"]
        ):
            raise ValueError("prefix output differs from its recorded uncached control")
        if (
            planned["label"] == "warm"
            and row["control_tokens"] != report["rows"][0]["control_tokens"]
        ):
            raise ValueError(
                "identical greedy requests contradict cold/warm output parity"
            )
        if observation["started_ns"] < last_finished:
            raise ValueError("fixed readiness requests overlap or move backwards")
        last_finished = observation["finished_ns"]
        before, after = row["before"], row["after"]
        _snapshot(before)
        _snapshot(after)
        if previous is not None and before != previous:
            raise ValueError(
                "prefix readiness snapshots do not form one sequential run"
            )
        if index == 0 and any(
            before[name]
            for name in SNAPSHOT_FIELDS
            if name
            not in ("ai_journey_prefix_max_bytes", "ai_journey_prefix_max_entries")
        ):
            raise ValueError("prefix readiness must begin with an empty fresh service")
        deltas = {
            "ai_journey_prefix_reused_prompt_tokens_total": planned["reused"],
            "ai_journey_prefix_forwarded_prefill_tokens_total": len(
                planned["request"]["prompt"]
            )
            - planned["reused"],
            "ai_journey_serving_generated_tokens_total": 8,
            **{
                f'ai_journey_prefix_attempts_total{{outcome="{outcome}"}}': int(
                    outcome == ("hit" if planned["reused"] else "miss")
                )
                for outcome in ("hit", "miss", "bypass")
            },
        }
        if any(after[name] - before[name] != amount for name, amount in deltas.items()):
            raise ValueError(
                "prefix observations contradict actual forwarded/generated work"
            )
        published.add(tuple(planned["request"]["prompt"]))
        # This fixed control has two layers, one native KV head, head_dim four
        # and 256 raw logits, all float64. Its tiny schedule cannot evict.
        expected_bytes = sum(
            8 * (2 * 2 * 1 * len(tokens) * 4 + 256) for tokens in published
        )
        if (
            after["ai_journey_prefix_entries"] != len(published)
            or after["ai_journey_prefix_resident_bytes"] != expected_bytes
        ):
            raise ValueError(
                "recorded resident storage differs from native compact snapshots"
            )
        previous = after
    cancellation = report["cancellation"]
    cancel_request = {
        "request_id": "prefix-cancel",
        "prompt": [1, 2, 3, 4],
        "sampling": {"max_new_tokens": 8, "temperature": 0, "seed": 53},
    }
    _validate_observation(
        cancellation,
        {"request": cancel_request, "cancel_after_tokens": 1},
        SimpleNamespace(streaming=True, max_new_tokens=8, timeout_seconds=5),
    )
    if (
        cancellation["outcome"] != "cancelled"
        or cancellation["error_kind"] != "client_closed_stream"
        or cancellation["started_ns"] < last_finished
        or any(
            type(token) is not int or not 0 <= token < 256
            for token in cancellation["tokens"]
        )
        or cancellation["tokens"] != report["rows"][0]["control_tokens"][:1]
    ):
        raise ValueError(
            "prefix readiness cancellation contradicts its later greedy control receipt"
        )
    final = report["final_metrics"]
    _snapshot(final)
    if (
        not 1
        <= final["ai_journey_serving_generated_tokens_total"]
        - previous["ai_journey_serving_generated_tokens_total"]
        <= 8
        or final['ai_journey_prefix_attempts_total{outcome="hit"}']
        != previous['ai_journey_prefix_attempts_total{outcome="hit"}'] + 1
        or final["ai_journey_prefix_reused_prompt_tokens_total"]
        != previous["ai_journey_prefix_reused_prompt_tokens_total"] + 4
    ):
        raise ValueError("cancelled fresh decode lacks prefix/generated work evidence")
    if any(
        final[name] != previous[name]
        for name in SNAPSHOT_FIELDS
        if name
        not in (
            'ai_journey_prefix_attempts_total{outcome="hit"}',
            "ai_journey_prefix_reused_prompt_tokens_total",
            "ai_journey_serving_generated_tokens_total",
        )
    ):
        raise ValueError("client cancellation changed unrelated prefix evidence")
    if type(report["child_exit_code"]) is not int or report["child_exit_code"] != 0:
        raise ValueError("owned serving child did not exit cleanly")
    if directory is not None:
        directory = Path(directory)
        from .hellaswag_io import read_evaluation_bytes, parse_evaluation_json

        artifacts = {}
        for name, field in (
            ("source.txt", "source_sha256"),
            ("spec.json", "spec_sha256"),
            ("checkpoint.pt", "checkpoint_sha256"),
        ):
            path = directory / name
            raw = read_evaluation_bytes(
                path, maximum=4 * 1024 * 1024, name="prefix readiness artifact"
            )
            if sha256(raw).hexdigest() != report[field]:
                raise ValueError("retained prefix readiness artifact differs")
            artifacts[name] = raw
        spec_mapping = parse_evaluation_json(
            artifacts["spec.json"], name="prefix readiness spec"
        )
        if (
            type(spec_mapping) is not dict
            or spec_mapping.get("model") != config
            or type(spec_mapping.get("source")) is not dict
            or spec_mapping["source"].get("sha256") != report["source_sha256"]
        ):
            raise ValueError("retained prefix spec architecture differs")
        from .ddp_checkpoint import bounded_checkpoint_load
        from .ddp_pipeline import load_serving_model
        from .gpt2_training import training_spec_from_mapping
        from .training_models import training_model_fingerprint

        checkpoint = directory / "checkpoint.pt"
        payload = bounded_checkpoint_load(
            checkpoint,
            4 * 1024 * 1024,
            expected_sha256=report["checkpoint_sha256"],
        )
        if (
            type(payload) is not dict
            or type(payload.get("step")) is not int
            or payload["step"] != 0
        ):
            raise ValueError(
                "retained readiness checkpoint is not the declared untrained control"
            )
        model = load_serving_model(
            checkpoint,
            training_spec_from_mapping(spec_mapping),
            artifacts["source.txt"],
            expected_sha256=report["checkpoint_sha256"],
        )
        if training_model_fingerprint(model) != report["model_sha256"]:
            raise ValueError(
                "actual retained checkpoint tensors differ from reported model"
            )
        import torch
        from .ddp_generation import SamplingConfig, generate_tokens

        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for row in report["rows"]:
                request = row["request"]
                actual = generate_tokens(
                    model,
                    request["prompt"],
                    SamplingConfig(**request["sampling"]),
                    use_cache=False,
                )
                if actual["tokens"] != row["control_tokens"]:
                    raise ValueError(
                        "retained checkpoint does not reproduce recorded uncached output"
                    )
        finally:
            torch.set_num_threads(threads)
    return report
