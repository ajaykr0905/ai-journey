"""Plan, run and exactly replay bounded document-level decoder experiments."""

import argparse
from hashlib import sha256
from pathlib import Path

import torch

from .decoder_variant_cli import _read_bytes, _standard_variants
from .decoder_variant_comparison import (
    DOCUMENT_REPORT_SCHEMA,
    decoder_comparison_digest,
    run_decoder_document_comparison,
)
from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_plan import DecoderExperimentPlan, decoder_plan_from_mapping
from .decoder_variant_public import (
    PUBLIC_SOURCES,
    run_public_decoder_comparison,
    verify_public_decoder_report,
)
from .decoder_variant_report_io import write_new_report
from .decoder_variant_verify import _read_report
from .gpt2_config import GPT2Config


def _default_plan():
    base = DecoderVariantConfig(
        GPT2Config(
            vocab_size=256,
            context_length=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )
    )
    return DecoderExperimentPlan(variants=tuple(_standard_variants(base).items()))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="write a new editable bounded experiment plan"
    )
    plan.add_argument("--output", type=Path, required=True)
    run = commands.add_parser(
        "run", help="run every planned seed and document-isolated arm"
    )
    run.add_argument("--plan", type=Path)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="replay a document report without changing artifacts"
    )
    verify.add_argument("--report", type=Path, required=True)
    for command in (run, verify):
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--heldout", type=Path, required=True)
    public_run = commands.add_parser(
        "run-public", help="compare a verified pinned public snapshot offline"
    )
    public_run.add_argument("--plan", type=Path)
    public_run.add_argument("--output", type=Path, required=True)
    public_run.add_argument("--train-documents", type=int, default=16)
    public_run.add_argument("--heldout-documents", type=int, default=4)
    public_run.add_argument("--split-seed", type=int, default=52)
    public_verify = commands.add_parser(
        "verify-public", help="replay a pinned public-corpus report"
    )
    public_verify.add_argument("--report", type=Path, required=True)
    for command in (public_run, public_verify):
        command.add_argument("--snapshot", type=Path, required=True)
        command.add_argument("--source", choices=sorted(PUBLIC_SOURCES), required=True)
    return parser


def verify_decoder_document_report(report, train_bytes, heldout_bytes):
    if (
        type(report) is not dict
        or set(report)
        != {"schema", "plan", "runtime", "trials", "summary", "report_sha256"}
        or report["schema"] != DOCUMENT_REPORT_SCHEMA
    ):
        raise ValueError("document report fields or schema are invalid")
    plan = decoder_plan_from_mapping(report["plan"])
    if report["runtime"] != {
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "torch_version": str(torch.__version__),
    }:
        raise ValueError(
            "document report runtime does not match the exact replay contract"
        )
    if report["report_sha256"] != decoder_comparison_digest(report):
        raise ValueError("document report digest does not match its contents")
    if type(report["trials"]) is not list or len(report["trials"]) != len(plan.seeds):
        raise ValueError("document report does not contain every declared seed")
    for trial in report["trials"]:
        if (
            type(trial) is not dict
            or trial.get("train_sha256") != sha256(train_bytes).hexdigest()
            or trial.get("heldout_sha256") != sha256(heldout_bytes).hexdigest()
        ):
            raise ValueError("document report source hashes differ from supplied bytes")
    expected = run_decoder_document_comparison(plan, train_bytes, heldout_bytes)
    if report != expected:
        raise ValueError("document report does not match deterministic replay")
    return plan


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            write_new_report(args.output, _default_plan().to_mapping())
            print(f"plan={args.output} device=cpu")
        elif args.command in {"run", "run-public"}:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("output already exists; choose a new path")
            plan = (
                decoder_plan_from_mapping(_read_report(args.plan))
                if args.plan
                else _default_plan()
            )
            if args.command == "run-public":
                report = run_public_decoder_comparison(
                    plan,
                    args.snapshot,
                    PUBLIC_SOURCES[args.source],
                    train_documents=args.train_documents,
                    heldout_documents=args.heldout_documents,
                    split_seed=args.split_seed,
                )
            else:
                report = run_decoder_document_comparison(
                    plan, _read_bytes(args.train), _read_bytes(args.heldout)
                )
            write_new_report(args.output, report)
            print(
                f"report={args.output} device=cpu seeds={len(plan.seeds)} variants={len(plan.variants)}"
            )
        else:
            report = _read_report(args.report, max_bytes=2 * 1024 * 1024)
            if args.command == "verify-public":
                verify_public_decoder_report(
                    report, args.snapshot, PUBLIC_SOURCES[args.source]
                )
            else:
                verify_decoder_document_report(
                    report, _read_bytes(args.train), _read_bytes(args.heldout)
                )
            print(f"verified={args.report} device=cpu documents=complete")
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        FloatingPointError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
