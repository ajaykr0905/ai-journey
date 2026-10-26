"""Plan, run, verify and replay explicitly bounded public-safe CPU MoE trials."""

import argparse
from dataclasses import asdict
from pathlib import Path
import re

from .decoder_variant_cli import _read_bytes
from .decoder_variant_comparison import decoder_comparison_digest
from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_public import PUBLIC_SOURCES, public_decoder_partitions
from .decoder_variant_report_io import write_new_report
from .decoder_variant_verify import _read_report
from .gpt2_config import GPT2Config
from .moe_config import MoEConfig
from .moe_experiment_comparison import run_moe_document_comparison
from .moe_experiment_plan import (
    MoEExperimentArm,
    MoEExperimentPlan,
    moe_experiment_plan_from_mapping,
)
from .moe_experiment_restart import run_restarting_moe_comparison
from .moe_replay import replay_moe_report
from .moe_report import canonical_moe_json, verify_moe_report

PUBLIC_SCHEMA = "ai-journey-moe-public-v1"
MAX_REPORT_BYTES = 4 * 1024 * 1024


def default_moe_plan():
    base = GPT2Config(
        256,
        8,
        8,
        2,
        1,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    dense = DecoderVariantConfig(
        base, norm_kind="rmsnorm", mlp_kind="swiglu", position_kind="rope"
    )
    sparse = MoEConfig(
        base,
        norm_kind=dense.norm_kind,
        mlp_kind=dense.mlp_kind,
        position_kind=dense.position_kind,
    )
    return MoEExperimentPlan(
        arms=(
            MoEExperimentArm("baseline", dense),
            MoEExperimentArm("top2", sparse, balance_weight=0.01),
        )
    )


def _run(plan, train, heldout, progress):
    return (
        run_restarting_moe_comparison(plan, train, heldout, progress)
        if progress
        else run_moe_document_comparison(plan, train, heldout)
    )


def run_public_moe_comparison(
    plan,
    snapshot,
    source_name,
    *,
    train_documents=16,
    heldout_documents=4,
    split_seed=53,
    progress_directory=None,
):
    if type(source_name) is not str or source_name not in PUBLIC_SOURCES:
        raise ValueError("unsupported pinned MoE public source")
    train, heldout, evidence = public_decoder_partitions(
        snapshot,
        PUBLIC_SOURCES[source_name],
        train_documents=train_documents,
        heldout_documents=heldout_documents,
        split_seed=split_seed,
    )
    report = {
        "schema": PUBLIC_SCHEMA,
        "public_source": source_name,
        "public_corpus": evidence,
        "comparison": _run(plan, train, heldout, progress_directory),
    }
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_public_moe_report(report)
    return report


def verify_public_moe_report(report):
    if (
        type(report) is not dict
        or set(report)
        != {"schema", "public_source", "public_corpus", "comparison", "report_sha256"}
        or report["schema"] != PUBLIC_SCHEMA
    ):
        raise ValueError("public MoE report fields or schema differ")
    name = report["public_source"]
    if type(name) is not str or name not in PUBLIC_SOURCES:
        raise ValueError("unsupported pinned MoE public source")
    if report["report_sha256"] != decoder_comparison_digest(report):
        raise ValueError("public MoE report digest differs")
    evidence, source = report["public_corpus"], PUBLIC_SOURCES[name]
    if type(evidence) is not dict or set(evidence) != {
        "source",
        "source_url",
        "license_url",
        "split_seed",
        "train_documents",
        "heldout_documents",
        "train_document_sha256",
        "heldout_document_sha256",
    }:
        raise ValueError("public MoE corpus fields differ")
    if canonical_moe_json(
        [evidence["source"], evidence["source_url"], evidence["license_url"]]
    ) != canonical_moe_json([asdict(source), source.url, source.license_url]):
        raise ValueError("public MoE source differs from its immutable pin/license")
    for field, maximum in (
        ("split_seed", 2**63 - 1),
        ("train_documents", 128),
        ("heldout_documents", 32),
    ):
        if (
            type(evidence[field]) is not int
            or not (0 if field == "split_seed" else 1) <= evidence[field] <= maximum
        ):
            raise ValueError("public MoE partition controls are invalid")
    selected = []
    for field, count_field in (
        ("train_document_sha256", "train_documents"),
        ("heldout_document_sha256", "heldout_documents"),
    ):
        values = evidence[field]
        if (
            type(values) is not list
            or len(values) != evidence[count_field]
            or any(
                type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in values
            )
        ):
            raise ValueError("public MoE partition identities differ")
        selected.extend(values)
    if len(set(selected)) != len(selected):
        raise ValueError("public MoE partitions contain duplicates or leakage")
    plan = verify_moe_report(report["comparison"])
    documents = report["comparison"]["trials"][0]["documents"]
    if (
        len(documents["train_documents"]) != evidence["train_documents"]
        or len(documents["heldout_documents"]) != evidence["heldout_documents"]
    ):
        raise ValueError(
            "public MoE source counts differ from measured document exposure"
        )
    return plan


def replay_public_moe_report(report, snapshot):
    plan = verify_public_moe_report(report)
    evidence = report["public_corpus"]
    train, heldout, expected = public_decoder_partitions(
        snapshot,
        PUBLIC_SOURCES[report["public_source"]],
        train_documents=evidence["train_documents"],
        heldout_documents=evidence["heldout_documents"],
        split_seed=evidence["split_seed"],
    )
    if canonical_moe_json(expected) != canonical_moe_json(evidence):
        raise ValueError(
            "public MoE source partitions differ from pinned snapshot replay"
        )
    replay_moe_report(report["comparison"], train, heldout)
    return plan


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan", help="write a new explicit bounded plan")
    create.add_argument("--output", type=Path, required=True)
    run = commands.add_parser(
        "run", help="train on explicitly cleared document-line bytes"
    )
    run.add_argument("--train", type=Path, required=True)
    run.add_argument("--heldout", type=Path, required=True)
    public = commands.add_parser("public", help="train from an exact offline PUD pin")
    public.add_argument("--snapshot", type=Path, required=True)
    public.add_argument("--source", choices=sorted(PUBLIC_SOURCES), required=True)
    public.add_argument("--train-documents", type=int, default=16)
    public.add_argument("--heldout-documents", type=int, default=4)
    public.add_argument("--split-seed", type=int, default=53)
    for command in (run, public):
        command.add_argument("--plan", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--progress-directory", type=Path)
    verify = commands.add_parser(
        "verify", help="check internal evidence offline without model execution"
    )
    verify.add_argument("--report", type=Path, required=True)
    replay = commands.add_parser(
        "replay", help="rerun all measurements against actual source bytes"
    )
    replay.add_argument("--report", type=Path, required=True)
    replay.add_argument("--train", type=Path)
    replay.add_argument("--heldout", type=Path)
    replay.add_argument("--snapshot", type=Path)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"plan", "run", "public"}:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("output already exists; choose a new path")
        if args.command == "plan":
            write_new_report(args.output, default_moe_plan().to_mapping())
            print(f"plan={args.output} protocol=float32-cpu")
            return 0
        if args.command in {"run", "public"}:
            plan = (
                moe_experiment_plan_from_mapping(_read_report(args.plan))
                if args.plan
                else default_moe_plan()
            )
            if args.command == "run":
                report = _run(
                    plan,
                    _read_bytes(args.train),
                    _read_bytes(args.heldout),
                    args.progress_directory,
                )
                verify_moe_report(report)
            else:
                report = run_public_moe_comparison(
                    plan,
                    args.snapshot,
                    args.source,
                    train_documents=args.train_documents,
                    heldout_documents=args.heldout_documents,
                    split_seed=args.split_seed,
                    progress_directory=args.progress_directory,
                )
            write_new_report(args.output, report)
            print(
                f"report={args.output} seeds={len(plan.seeds)} arms={len(plan.arms)} protocol=float32-cpu"
            )
            return 0
        report = _read_report(args.report, max_bytes=MAX_REPORT_BYTES)
        public = report.get("schema") == PUBLIC_SCHEMA
        if args.command == "verify":
            plan = (
                verify_public_moe_report(report)
                if public
                else verify_moe_report(report)
            )
            print(f"internally_verified={args.report} seeds={len(plan.seeds)}")
            return 0
        if public:
            if (
                args.snapshot is None
                or args.train is not None
                or args.heldout is not None
            ):
                raise ValueError(
                    "public MoE replay requires only --snapshot source input"
                )
            plan = replay_public_moe_report(report, args.snapshot)
        else:
            if args.snapshot is not None or args.train is None or args.heldout is None:
                raise ValueError("document MoE replay requires --train and --heldout")
            plan = replay_moe_report(
                report, _read_bytes(args.train), _read_bytes(args.heldout)
            )
        print(f"replayed={args.report} seeds={len(plan.seeds)} arms={len(plan.arms)}")
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        FloatingPointError,
        KeyError,
        RecursionError,
        OverflowError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
