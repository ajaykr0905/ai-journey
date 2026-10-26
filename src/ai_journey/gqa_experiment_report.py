"""Offline semantic checks and actual source replay for controlled GQA trials."""

from hashlib import sha256
from itertools import islice
import math

import torch

from .decoder_variant_comparison import decoder_comparison_digest
from .gqa_experiment import (
    GQA_EXPERIMENT_SCHEMA,
    GQA_INITIALIZATION,
    GQA_SCOPE,
    run_gqa_comparison,
    summarize_gqa_trials,
)
from .gqa_experiment_plan import (
    gqa_experiment_plan_from_mapping,
    require_gqa_experiment_budget,
)
from .gqa_resources import gqa_parameter_inventory, require_gqa_execution_budget
from .gqa_config import expected_gqa_parameter_count
from .moe_report import (
    _digest,
    _equal,
    _fields,
    _float,
    _integer,
    _score,
    _expected_optimizer_names,
    _gradient_clip_transition,
    canonical_moe_json,
)


def _coverage(report, plan):
    docs = report["documents"]
    _fields(
        docs,
        "train_documents heldout_documents train_window_count heldout_target_tokens train_lengths heldout_lengths",
        "GQA documents",
    )
    identities = []
    for partition in ("train", "heldout"):
        digests, lengths = docs[f"{partition}_documents"], docs[f"{partition}_lengths"]
        if (
            type(digests) is not list
            or type(lengths) is not list
            or not 1 <= len(digests) == len(lengths) <= 2048
        ):
            raise ValueError("GQA document identity/length exposure differs")
        for digest, length in zip(digests, lengths, strict=True):
            _digest(digest)
            _integer(length, 2, 1024 * 1024)
        if sum(lengths) > 1024 * 1024 + 2048:
            raise ValueError("GQA source length exceeds document admission")
        identities.extend(digests)
    if len(identities) > 2048 or len(set(identities)) != len(identities):
        raise ValueError("GQA sources duplicate or leak document content")
    context = plan.decoder.gpt2.context_length
    _equal(
        docs["train_window_count"],
        sum(max(0, length - context) for length in docs["train_lengths"]),
        "GQA training windows",
    )
    _integer(docs["train_window_count"], 1, 1024 * 1024)
    targets = sum(length - 1 for length in docs["heldout_lengths"])
    _integer(targets, 1, 8192)
    _equal(docs["heldout_target_tokens"], targets, "GQA target exposure")
    windows = sum(
        math.ceil((length - 1) / context) for length in docs["heldout_lengths"]
    )
    _equal(report["evaluation_windows"], windows, "GQA evaluation windows")
    _equal(
        report["resource_admission"],
        require_gqa_experiment_budget(plan, evaluation_windows=windows),
        "GQA full-job admission",
    )
    return targets, windows


def _evaluation(score, report, plan, targets, windows):
    _score(
        score,
        plan.decoder,
        targets=targets,
        windows=windows,
        batch_size=plan.evaluation_batch_size,
    )
    context = plan.decoder.gpt2.context_length
    lengths = iter(
        min(context, length - 1 - start)
        for length in report["documents"]["heldout_lengths"]
        for start in range(0, length - 1, context)
    )
    for batch in score["batches"]:
        rows = list(islice(lengths, plan.evaluation_batch_size))
        width = max(rows)
        _equal(
            [batch["shape"], batch["token_mask"]],
            [
                [len(rows), width],
                [[index < count for index in range(width)] for count in rows],
            ],
            "GQA complete document window geometry",
        )


def verify_gqa_experiment_report(report):
    _fields(
        report,
        "schema plan runtime train_sha256 heldout_sha256 documents evaluation_windows resource_admission trials summary verification_scope report_sha256",
        "GQA report",
    )
    if (
        report["schema"] != GQA_EXPERIMENT_SCHEMA
        or report["verification_scope"] != GQA_SCOPE
    ):
        raise ValueError("GQA schema or verification scope differs")
    _equal(
        report["report_sha256"], decoder_comparison_digest(report), "GQA report digest"
    )
    plan = gqa_experiment_plan_from_mapping(report["plan"])
    runtime = report["runtime"]
    _fields(runtime, "device dtype threads torch_version", "GQA runtime")
    _equal(
        [runtime["device"], runtime["dtype"], runtime["threads"]],
        ["cpu", "float32", 1],
        "GQA runtime controls",
    )
    if (
        type(runtime["torch_version"]) is not str
        or not 1 <= len(runtime["torch_version"]) <= 100
    ):
        raise ValueError("GQA recorded runtime version is invalid")
    for key in ("train_sha256", "heldout_sha256"):
        _digest(report[key])
    if report["train_sha256"] == report["heldout_sha256"]:
        raise ValueError("GQA train and heldout source identities overlap")
    targets, windows = _coverage(report, plan)
    trials = report["trials"]
    if type(trials) is not list or len(trials) != len(plan.seeds):
        raise ValueError("GQA seed exposure differs")
    schedules = []
    for trial, seed in zip(trials, plan.seeds, strict=True):
        _fields(trial, "seed batch_schedule_sha256 arms", "GQA trial")
        _equal(trial["seed"], seed, "GQA seed")
        _digest(trial["batch_schedule_sha256"])
        if type(trial["arms"]) is not list or len(trial["arms"]) != len(plan.arms):
            raise ValueError("GQA arm exposure differs")
        shared = []
        for arm, config in zip(trial["arms"], plan.arms, strict=True):
            _fields(
                arm,
                "kv_heads architecture_sha256 parameter_count parameter_inventory training_execution initialization initial_model_sha256 final_model_sha256 initial_heldout heldout optimizer_groups trace",
                "GQA arm",
            )
            _equal(
                [
                    arm["kv_heads"],
                    arm["architecture_sha256"],
                    arm["parameter_count"],
                    arm["parameter_inventory"],
                ],
                [
                    config.kv_heads,
                    config.fingerprint(),
                    expected_gqa_parameter_count(config),
                    gqa_parameter_inventory(config),
                ],
                "GQA architecture inventory",
            )
            initialization = arm["initialization"]
            _equal(
                arm["training_execution"],
                require_gqa_execution_budget(
                    config,
                    batch_size=plan.batch_size,
                    sequence_length=plan.decoder.gpt2.context_length,
                ),
                "GQA declared per-operation training inventory",
            )
            _fields(
                initialization,
                "reference_mha_sha256 exact_shared_sha256 policy",
                "GQA initialization",
            )
            if initialization["policy"] != GQA_INITIALIZATION:
                raise ValueError("GQA initialization treatment differs")
            for key in ("reference_mha_sha256", "exact_shared_sha256"):
                _digest(initialization[key])
            shared.append(
                (
                    initialization["reference_mha_sha256"],
                    initialization["exact_shared_sha256"],
                )
            )
            for key in ("initial_model_sha256", "final_model_sha256"):
                _digest(arm[key])
            for key in ("initial_heldout", "heldout"):
                _evaluation(arm[key], report, plan, targets, windows)
                schedules.append(arm[key]["batch_schedule_sha256"])
            expected_groups = {
                key: [f"decoder.{name}" for name in names]
                for key, names in _expected_optimizer_names(plan.decoder).items()
            }
            expected_groups["trainable_parameters"] = expected_gqa_parameter_count(
                config
            )
            _equal(
                arm["optimizer_groups"],
                expected_groups,
                "GQA optimizer exact architecture binding",
            )
            if type(arm["trace"]) is not list or len(arm["trace"]) != plan.steps:
                raise ValueError("GQA optimizer step exposure differs")
            for step, row in enumerate(arm["trace"], 1):
                _fields(
                    row,
                    "step ce_loss balance_loss router_z_loss objective gradient_norm_before_clip gradient_norm_after_clip clipped target_tokens routing",
                    "GQA update",
                )
                _equal(
                    [
                        row["step"],
                        row["target_tokens"],
                        row["routing"],
                    ],
                    [
                        step,
                        plan.batch_size * plan.decoder.gpt2.context_length,
                        [],
                    ],
                    "GQA update controls",
                )
                for key in ("balance_loss", "router_z_loss"):
                    # A live inactive objective may be signed zero. Preserve
                    # those measured bytes while rejecting aliases/nonzero aux.
                    _float(row[key])
                    if row[key] != 0.0:
                        raise ValueError("GQA dense auxiliary objectives must be zero")
                for key in (
                    "ce_loss",
                    "objective",
                    "gradient_norm_before_clip",
                    "gradient_norm_after_clip",
                ):
                    _float(row[key])
                _equal(row["objective"], row["ce_loss"], "GQA dense objective")
                before, after = (
                    row["gradient_norm_before_clip"],
                    row["gradient_norm_after_clip"],
                )
                _equal(
                    row["clipped"],
                    before > plan.max_gradient_norm,
                    "GQA clipping decision",
                )
                _gradient_clip_transition(before, after, plan.max_gradient_norm)
        if len(set(shared)) != 1:
            raise ValueError(
                "GQA treatments do not share the declared reference/Q tensors"
            )
    if len(set(schedules)) != 1:
        raise ValueError("GQA evaluation schedule differs across controlled arms")
    _equal(report["summary"], summarize_gqa_trials(plan, trials), "GQA paired summary")
    return plan


def replay_gqa_experiment_report(report, train_bytes, heldout_bytes):
    plan = verify_gqa_experiment_report(report)
    if (
        report["runtime"]["torch_version"] != str(torch.__version__)
        or type(train_bytes) is not bytes
        or type(heldout_bytes) is not bytes
        or sha256(train_bytes).hexdigest() != report["train_sha256"]
        or sha256(heldout_bytes).hexdigest() != report["heldout_sha256"]
    ):
        raise ValueError(
            "GQA replay requires the recorded CPU runtime and source bytes"
        )
    actual = run_gqa_comparison(plan, train_bytes, heldout_bytes)
    if canonical_moe_json(actual) != canonical_moe_json(report):
        raise ValueError("GQA report differs from actual source replay")
    return plan
