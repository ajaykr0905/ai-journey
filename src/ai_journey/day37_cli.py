"""Operate bounded transformer certification and optional predeclared trials."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .ablation_protocol import (
    load_ablation_protocol,
    run_ablation,
    write_ablation_report,
)
from .transformer_certification import (
    TransformerCertificationConfig,
    certify_transformer,
)
from .transformer_evidence import build_certification_report, write_certification_report
from .transformer_lab import CharacterCodec, TokenCorpus


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--norm-placement", choices=("pre", "post"), default="pre")
    parser.add_argument("--backend", choices=("manual", "sdpa"), default="manual")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--ablation-output", type=Path)
    return parser


def _protect_input_paths(outputs, inputs):
    for index, output in enumerate(outputs):
        for protected in [*inputs, *outputs[:index]]:
            if output.resolve() == protected.resolve() or (
                output.exists() and protected.exists() and output.samefile(protected)
            ):
                raise ValueError(
                    "report paths must be distinct from inputs and other outputs"
                )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        selected = (args.protocol, args.corpus, args.ablation_output)
        if any(path is not None for path in selected) and not all(
            path is not None for path in selected
        ):
            raise ValueError(
                "--protocol, --corpus and --ablation-output must be supplied together"
            )
        outputs = [args.output] + (
            [args.ablation_output] if args.ablation_output else []
        )
        inputs = [path for path in (args.protocol, args.corpus) if path is not None]
        _protect_input_paths(outputs, inputs)
        default = TransformerCertificationConfig()
        model = replace(
            default.model,
            head_count=args.heads,
            normalization_placement=args.norm_placement,
            attention_backend=args.backend,
        )
        config = TransformerCertificationConfig(
            model=model, seed=args.seed, tolerance=args.tolerance
        )
        ablation = None
        if args.protocol is not None:
            text = args.corpus.read_text(encoding="utf-8")
            protocol = load_ablation_protocol(
                args.protocol, vocab_size=len(CharacterCodec.from_text(text).tokens)
            )
            corpus = TokenCorpus.from_path(
                args.corpus, block_size=protocol.baseline_arm.model_config.block_size
            )
            ablation = run_ablation(corpus, protocol)
        result = certify_transformer(config)
        report = build_certification_report(result)
        write_certification_report(args.output, report)
        if ablation is not None:
            write_ablation_report(args.ablation_output, ablation)
            print(
                f"ablation={ablation.protocol.name} trials={len(ablation.trials)} output={args.ablation_output}"
            )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"cache_rollover_error={result['cache_rollover_error']:.3e} padding_error={result['padding_error']:.3e} report={args.output}"
    )
    return 0
