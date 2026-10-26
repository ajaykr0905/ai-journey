"""Bounded controlled GQA CPU experiments; offline verification and actual replay."""

import argparse
from pathlib import Path

from .decoder_variant_cli import _read_bytes
from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_public import PUBLIC_SOURCES, public_decoder_partitions
from .decoder_variant_report_io import write_new_report
from .decoder_variant_verify import _read_report
from .gpt2_config import GPT2Config
from .gqa_experiment import run_gqa_comparison
from .gqa_experiment_plan import GQAExperimentPlan, gqa_experiment_plan_from_mapping
from .gqa_experiment_report import (
    verify_gqa_experiment_report,
    replay_gqa_experiment_report,
)
from .gqa_experiment_public import (
    PUBLIC_GQA_SCHEMA,
    run_public_gqa_comparison,
    verify_public_gqa_report,
    replay_public_gqa_report,
)
from .gqa_cache_evidence import (
    run_gqa_cache_evidence,
    verify_gqa_cache_evidence,
    replay_gqa_cache_evidence,
)
from .moe_report import canonical_moe_json


def default_gqa_plan():
    base = GPT2Config(
        256,
        8,
        8,
        4,
        1,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    return GQAExperimentPlan(
        DecoderVariantConfig(
            base, norm_kind="rmsnorm", mlp_kind="swiglu", position_kind="rope"
        ),
        (4, 2, 1),
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--train", type=Path, required=True)
    run.add_argument("--heldout", type=Path, required=True)
    public = commands.add_parser("public")
    public.add_argument("--snapshot", type=Path, required=True)
    public.add_argument("--source", choices=sorted(PUBLIC_SOURCES), required=True)
    public.add_argument("--train-documents", type=int, default=16)
    public.add_argument("--heldout-documents", type=int, default=4)
    public.add_argument("--split-seed", type=int, default=53)
    for command in (run, public):
        command.add_argument("--plan", type=Path)
        command.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--report", type=Path, required=True)
    replay.add_argument("--snapshot", type=Path)
    replay.add_argument("--train", type=Path)
    replay.add_argument("--heldout", type=Path)
    cache = commands.add_parser(
        "cache", help="replay training and inspect actual native KV growth"
    )
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--prefix-count", type=int, default=4)
    cache_replay = commands.add_parser(
        "cache-replay", help="rerun every cache measurement"
    )
    for command in (cache, cache_replay):
        command.add_argument("--report", type=Path, required=True)
        command.add_argument("--snapshot", type=Path)
        command.add_argument("--train", type=Path)
        command.add_argument("--heldout", type=Path)
    cache_verify = commands.add_parser(
        "cache-verify", help="recompute storage and raw logit residuals offline"
    )
    cache_verify.add_argument("--report", type=Path, required=True)
    return parser


def _cache_inputs(source_report, args):
    if source_report.get("schema") == PUBLIC_GQA_SCHEMA:
        verify_public_gqa_report(source_report)
        if args.snapshot is None or args.train is not None or args.heldout is not None:
            raise ValueError("public cache evidence requires only --snapshot")
        evidence = source_report["public_corpus"]
        train, heldout, actual = public_decoder_partitions(
            args.snapshot,
            PUBLIC_SOURCES[source_report["public_source"]],
            train_documents=evidence["train_documents"],
            heldout_documents=evidence["heldout_documents"],
            split_seed=evidence["split_seed"],
        )
        if canonical_moe_json(actual) != canonical_moe_json(evidence):
            raise ValueError(
                "cache source partitions differ from pinned public snapshot"
            )
        return train, heldout
    if args.snapshot is not None or args.train is None or args.heldout is None:
        raise ValueError("document cache evidence requires --train and --heldout")
    return _read_bytes(args.train), _read_bytes(args.heldout)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"cache", "cache-verify", "cache-replay"}:
            if args.command == "cache" and (
                args.output.exists() or args.output.is_symlink()
            ):
                raise ValueError("output already exists; choose a new path")
            report = _read_report(args.report, max_bytes=16 * 1024 * 1024)
            if args.command == "cache-verify":
                verify_gqa_cache_evidence(report)
                print(f"cache_internally_verified={args.report}")
            else:
                source = report if args.command == "cache" else report["source_report"]
                train, heldout = _cache_inputs(source, args)
                if args.command == "cache":
                    measured = run_gqa_cache_evidence(
                        source, train, heldout, prefix_count=args.prefix_count
                    )
                    write_new_report(args.output, measured)
                    print(
                        f"cache_report={args.output} observations={len(measured['observations'])}"
                    )
                else:
                    replay_gqa_cache_evidence(report, train, heldout)
                    print(f"cache_replayed={args.report}")
            return 0
        if args.command in {"plan", "run", "public"}:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("output already exists; choose a new path")
            plan = (
                gqa_experiment_plan_from_mapping(_read_report(args.plan))
                if getattr(args, "plan", None)
                else default_gqa_plan()
            )
            if args.command == "plan":
                write_new_report(args.output, plan.to_mapping())
                print(f"plan={args.output}")
                return 0
            if args.command == "public":
                report = run_public_gqa_comparison(
                    plan,
                    args.snapshot,
                    args.source,
                    train_documents=args.train_documents,
                    heldout_documents=args.heldout_documents,
                    split_seed=args.split_seed,
                )
            else:
                report = run_gqa_comparison(
                    plan, _read_bytes(args.train), _read_bytes(args.heldout)
                )
                verify_gqa_experiment_report(report)
            write_new_report(args.output, report)
            print(
                f"report={args.output} seeds={len(plan.seeds)} arms={len(plan.kv_heads)} protocol=float32-cpu"
            )
            return 0
        report = _read_report(args.report, max_bytes=4 * 1024 * 1024)
        public = report.get("schema") == PUBLIC_GQA_SCHEMA
        if args.command == "verify":
            plan = (
                verify_public_gqa_report(report)
                if public
                else verify_gqa_experiment_report(report)
            )
            print(f"internally_verified={args.report} seeds={len(plan.seeds)}")
            return 0
        if public:
            if (
                args.snapshot is None
                or args.train is not None
                or args.heldout is not None
            ):
                raise ValueError("public GQA replay requires only --snapshot")
            replay_public_gqa_report(report, args.snapshot)
        else:
            if args.snapshot is not None or args.train is None or args.heldout is None:
                raise ValueError("document GQA replay requires --train and --heldout")
            replay_gqa_experiment_report(
                report, _read_bytes(args.train), _read_bytes(args.heldout)
            )
        print(f"replayed={args.report}")
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        RecursionError,
        OverflowError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
