"""Immutable public-source identity for controlled GQA comparisons."""

from dataclasses import asdict

from .decoder_variant_comparison import decoder_comparison_digest
from .decoder_variant_public import PUBLIC_SOURCES, public_decoder_partitions
from .gqa_experiment import run_gqa_comparison
from .gqa_experiment_report import (
    verify_gqa_experiment_report,
    replay_gqa_experiment_report,
)
from .moe_report import _digest, _equal, _fields, _integer, canonical_moe_json

PUBLIC_GQA_SCHEMA = "ai-journey-controlled-gqa-public-v1"


def run_public_gqa_comparison(
    plan,
    snapshot,
    source_name,
    *,
    train_documents=16,
    heldout_documents=4,
    split_seed=53,
):
    if type(source_name) is not str or source_name not in PUBLIC_SOURCES:
        raise ValueError("unsupported pinned GQA public source")
    train, heldout, evidence = public_decoder_partitions(
        snapshot,
        PUBLIC_SOURCES[source_name],
        train_documents=train_documents,
        heldout_documents=heldout_documents,
        split_seed=split_seed,
    )
    report = {
        "schema": PUBLIC_GQA_SCHEMA,
        "public_source": source_name,
        "public_corpus": evidence,
        "comparison": run_gqa_comparison(plan, train, heldout),
    }
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_public_gqa_report(report)
    return report


def verify_public_gqa_report(report):
    _fields(
        report,
        "schema public_source public_corpus comparison report_sha256",
        "public GQA report",
    )
    if (
        report["schema"] != PUBLIC_GQA_SCHEMA
        or type(report["public_source"]) is not str
        or report["public_source"] not in PUBLIC_SOURCES
    ):
        raise ValueError("public GQA schema or source differs")
    _equal(
        report["report_sha256"], decoder_comparison_digest(report), "public GQA digest"
    )
    source = PUBLIC_SOURCES[report["public_source"]]
    evidence = report["public_corpus"]
    _fields(
        evidence,
        "source source_url license_url split_seed train_documents heldout_documents train_document_sha256 heldout_document_sha256",
        "public GQA corpus",
    )
    _equal(
        [evidence["source"], evidence["source_url"], evidence["license_url"]],
        [asdict(source), source.url, source.license_url],
        "public GQA immutable pin/license",
    )
    _integer(evidence["split_seed"], 0, 2**63 - 1)
    identities = []
    for name, maximum in (("train", 128), ("heldout", 32)):
        count, digests = (
            evidence[f"{name}_documents"],
            evidence[f"{name}_document_sha256"],
        )
        _integer(count, 1, maximum)
        if type(digests) is not list or len(digests) != count:
            raise ValueError("public GQA partition exposure differs")
        for digest in digests:
            _digest(digest)
        identities.extend(digests)
    if len(set(identities)) != len(identities):
        raise ValueError("public GQA partitions duplicate or leak content")
    plan = verify_gqa_experiment_report(report["comparison"])
    _equal(
        [
            len(report["comparison"]["documents"][f"{name}_documents"])
            for name in ("train", "heldout")
        ],
        [evidence["train_documents"], evidence["heldout_documents"]],
        "public GQA measured source counts",
    )
    return plan


def replay_public_gqa_report(report, snapshot):
    plan = verify_public_gqa_report(report)
    evidence = report["public_corpus"]
    train, heldout, expected = public_decoder_partitions(
        snapshot,
        PUBLIC_SOURCES[report["public_source"]],
        train_documents=evidence["train_documents"],
        heldout_documents=evidence["heldout_documents"],
        split_seed=evidence["split_seed"],
    )
    if canonical_moe_json(expected) != canonical_moe_json(evidence):
        raise ValueError("public GQA snapshot partition evidence differs")
    replay_gqa_experiment_report(report["comparison"], train, heldout)
    return plan
