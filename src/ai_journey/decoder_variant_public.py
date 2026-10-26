"""Offline, pinned public-corpus inputs for document decoder comparisons."""

from dataclasses import asdict
from hashlib import sha256

from .decoder_variant_comparison import (
    decoder_comparison_digest,
    run_decoder_document_comparison,
)
from .decoder_variant_plan import decoder_plan_from_mapping
from .tokenizer_sources import ENGLISH, HINDI, PublicCorpus, load_public_corpus

PUBLIC_SOURCES = {"en-pud": ENGLISH, "hi-pud": HINDI}
PUBLIC_REPORT_SCHEMA = "ai-journey-decoder-public-v1"


def public_decoder_partitions(
    snapshot,
    source: PublicCorpus,
    *,
    train_documents=16,
    heldout_documents=4,
    split_seed=52,
):
    for name, value, maximum in (
        ("train_documents", train_documents, 128),
        ("heldout_documents", heldout_documents, 32),
        ("split_seed", split_seed, 2**63 - 1),
    ):
        if (
            type(value) is not int
            or not (0 if name == "split_seed" else 1) <= value <= maximum
        ):
            raise ValueError(f"invalid public decoder {name}")
    records = load_public_corpus(snapshot, source)
    documents = {
        sha256(row.text.encode()).hexdigest(): row.text.encode() for row in records
    }
    if len(documents) != len(records):
        raise ValueError("public decoder corpus contains repeated document content")
    if train_documents + heldout_documents > len(documents):
        raise ValueError("public decoder source contains too few distinct documents")
    ordered = sorted(
        documents,
        key=lambda digest: (sha256(f"{split_seed}:{digest}".encode()).digest(), digest),
    )
    heldout = ordered[:heldout_documents]
    train = ordered[heldout_documents : heldout_documents + train_documents]
    evidence = {
        "source": asdict(source),
        "source_url": source.url,
        "license_url": source.license_url,
        "split_seed": split_seed,
        "train_documents": train_documents,
        "heldout_documents": heldout_documents,
        "train_document_sha256": train,
        "heldout_document_sha256": heldout,
    }
    return (
        b"\n".join(documents[digest] for digest in train) + b"\n",
        b"\n".join(documents[digest] for digest in heldout) + b"\n",
        evidence,
    )


def run_public_decoder_comparison(
    plan, snapshot, source, *, train_documents=16, heldout_documents=4, split_seed=52
):
    train, heldout, evidence = public_decoder_partitions(
        snapshot,
        source,
        train_documents=train_documents,
        heldout_documents=heldout_documents,
        split_seed=split_seed,
    )
    report = {
        "schema": PUBLIC_REPORT_SCHEMA,
        "public_corpus": evidence,
        "comparison": run_decoder_document_comparison(plan, train, heldout),
    }
    report["report_sha256"] = decoder_comparison_digest(report)
    return report


def verify_public_decoder_report(report, snapshot, source):
    if (
        type(report) is not dict
        or set(report) != {"schema", "public_corpus", "comparison", "report_sha256"}
        or report["schema"] != PUBLIC_REPORT_SCHEMA
    ):
        raise ValueError("public decoder report schema or fields are invalid")
    evidence = report["public_corpus"]
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
        raise ValueError("public decoder source evidence fields are invalid")
    if (
        evidence["source"] != asdict(source)
        or evidence["source_url"] != source.url
        or evidence["license_url"] != source.license_url
    ):
        raise ValueError(
            "public decoder report differs from the pinned source identity"
        )
    if report["report_sha256"] != decoder_comparison_digest(report):
        raise ValueError("public decoder report digest does not match")
    plan = decoder_plan_from_mapping(report["comparison"]["plan"])
    expected = run_public_decoder_comparison(
        plan,
        snapshot,
        source,
        train_documents=evidence["train_documents"],
        heldout_documents=evidence["heldout_documents"],
        split_seed=evidence["split_seed"],
    )
    if report != expected:
        raise ValueError("public decoder report does not match deterministic replay")
