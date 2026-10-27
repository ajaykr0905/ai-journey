import copy
from hashlib import sha256
import json
import os
from pathlib import Path
from io import BytesIO
import shutil

import pytest
import torch

from ai_journey.prefix_readiness import main, run_prefix_readiness
from ai_journey.prefix_report import seal_prefix_report, verify_prefix_report


@pytest.mark.parametrize(
    "stage", ["step_permanent", "step_transient", "serving_transient"]
)
def test_every_artifact_checkpoint_reopen_is_bound_to_reported_snapshot(
    actual, tmp_path, monkeypatch, stage
):
    from ai_journey import ddp_checkpoint, ddp_pipeline
    from ai_journey.ddp_generation import SamplingConfig, generate_tokens
    from ai_journey.gpt2_training import load_training_spec
    from ai_journey.training_models import training_model_fingerprint

    original_directory, original_report = actual
    directory = tmp_path / "owned-artifacts"
    shutil.copytree(original_directory, directory)
    checkpoint = directory / "checkpoint.pt"
    checkpoint_a = checkpoint.read_bytes()
    model = ddp_pipeline.load_serving_model(
        checkpoint,
        load_training_spec(directory / "spec.json"),
        (directory / "source.txt").read_bytes(),
    )
    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1e-12)
    identity_b = training_model_fingerprint(model)
    assert identity_b != original_report["model_sha256"]
    payload = torch.load(BytesIO(checkpoint_a), weights_only=True)
    payload["model"] = copy.deepcopy(model.state_dict())
    payload["model_sha256"] = identity_b
    encoded = BytesIO()
    torch.save(payload, encoded)
    checkpoint_b = encoded.getvalue()
    report_b = copy.deepcopy(original_report)
    report_b["model_sha256"] = identity_b
    for row in report_b["rows"]:
        tokens = generate_tokens(
            model,
            row["request"]["prompt"],
            SamplingConfig(**row["request"]["sampling"]),
            use_cache=False,
        )["tokens"]
        row["control_tokens"] = tokens
        row["observation"]["tokens"] = tokens
        row["observation"]["model_sha256"] = identity_b
    report_b["cancellation"]["tokens"] = report_b["rows"][0]["control_tokens"][:1]
    report_b = seal_prefix_report(report_b)
    assert report_b["checkpoint_sha256"] == sha256(checkpoint_a).hexdigest()
    step_load = ddp_checkpoint.bounded_checkpoint_load
    serving_load = ddp_pipeline.bounded_checkpoint_load
    swaps = []

    def replace(candidate, args, kwargs, *, boundary, real):
        if not stage.startswith(boundary) or candidate != checkpoint:
            return real(candidate, *args, **kwargs)
        swaps.append(boundary)
        checkpoint.write_bytes(checkpoint_b)
        try:
            return real(candidate, *args, **kwargs)
        finally:
            if stage.endswith("transient"):
                checkpoint.write_bytes(checkpoint_a)

    monkeypatch.setattr(
        ddp_checkpoint,
        "bounded_checkpoint_load",
        lambda candidate, *args, **kwargs: replace(
            candidate, args, kwargs, boundary="step", real=step_load
        ),
    )
    monkeypatch.setattr(
        ddp_pipeline,
        "bounded_checkpoint_load",
        lambda candidate, *args, **kwargs: replace(
            candidate, args, kwargs, boundary="serving", real=serving_load
        ),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        verify_prefix_report(
            original_report if stage == "step_transient" else report_b,
            directory=directory,
        )
    assert len(swaps) == 1
    if stage.endswith("transient"):
        assert checkpoint.read_bytes() == checkpoint_a


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    directory = tmp_path_factory.mktemp("actual-prefix") / "run"
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    report = run_prefix_readiness(
        installed_root=Path(__file__).resolve().parents[1] / "src", output_dir=directory
    )
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.get_rng_state(), rng)
    return directory, report


def test_actual_operator_readiness_preserves_raw_observations_and_runtime(
    actual, capsys
):
    directory, report = actual
    assert verify_prefix_report(report, directory=directory) == report
    assert json.loads((directory / "raw-observations.json").read_text()) == report
    assert json.loads((directory / "report.json").read_text()) == report
    assert report["runtime"]["trained"] is False
    assert len({row["observation"]["request_id"] for row in report["rows"]}) == 6
    assert sum(len(row["observation"]["tokens"]) for row in report["rows"]) == 48
    assert (
        report["rows"][-1]["after"]["ai_journey_prefix_reused_prompt_tokens_total"]
        == 12
    )
    assert (
        report["rows"][-1]["after"]["ai_journey_prefix_forwarded_prefill_tokens_total"]
        == 10
    )
    assert (
        main(
            [
                "--verify-report",
                str(directory / "report.json"),
                "--artifact-dir",
                str(directory),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["completed_output_tokens"] == 48


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "bool_step",
        "trained",
        "cuda",
        "config",
        "model",
        "control",
        "time",
        "receipt",
        "reuse",
        "forwarded",
        "generated",
        "replay",
        "bool_metric",
        "bytes",
        "resident",
        "entries",
        "cancel",
        "final",
        "exit",
        "digest",
    ],
)
def test_resealed_impossible_observations_are_rejected(actual, mutation):
    _, original = actual
    report = copy.deepcopy(original)
    first = report["rows"][0]
    if mutation == "unknown":
        report["extra"] = True
    elif mutation == "bool_step":
        report["runtime"]["checkpoint_step"] = False
    elif mutation == "trained":
        report["runtime"]["trained"] = True
    elif mutation == "cuda":
        report["runtime"]["device"] = "cuda"
    elif mutation == "config":
        report["model_config"]["config"]["kv_heads"] = 4
    elif mutation == "model":
        first["observation"]["model_sha256"] = "a" * 64
    elif mutation == "control":
        first["control_tokens"][0] = 999
    elif mutation == "time":
        first["observation"]["finished_ns"] = first["observation"]["started_ns"]
    elif mutation == "receipt":
        report["rows"][1]["observation"]["token_received_ns"].clear()
    elif mutation in (
        "reuse",
        "forwarded",
        "generated",
        "replay",
        "bool_metric",
        "bytes",
        "resident",
        "entries",
    ):
        field, value = {
            "reuse": ("ai_journey_prefix_reused_prompt_tokens_total", 1),
            "forwarded": ("ai_journey_prefix_forwarded_prefill_tokens_total", 0),
            "generated": ("ai_journey_serving_generated_tokens_total", 0),
            "replay": ("ai_journey_serving_cache_hits_total", 1),
            "bool_metric": ("ai_journey_prefix_entries", True),
            "bytes": ("ai_journey_prefix_resident_bytes", 32769),
            "resident": ("ai_journey_prefix_resident_bytes", 0),
            "entries": ("ai_journey_prefix_entries", 0),
        }[mutation]
        first["after"][field] = value
    elif mutation == "cancel":
        report["cancellation"]["error_kind"] = None
    elif mutation == "final":
        report["final_metrics"]["ai_journey_prefix_forwarded_prefill_tokens_total"] += 1
    elif mutation == "exit":
        report["child_exit_code"] = False
    else:
        report["report_sha256"] = "a" * 64
    if mutation != "digest":
        report = seal_prefix_report(report)
    with pytest.raises(ValueError):
        verify_prefix_report(report)


def test_artifact_mode_recomputes_model_identity_instead_of_trusting_resealed_metadata(
    actual,
):
    directory, original = actual
    report = copy.deepcopy(original)
    report["model_sha256"] = "a" * 64
    for row in report["rows"]:
        row["observation"]["model_sha256"] = report["model_sha256"]
    report = seal_prefix_report(report)
    # Without artifacts this is consistent, not authenticated. Artifact mode
    # independently loads the bounded fixed control and hashes actual tensors.
    assert verify_prefix_report(report) == report
    with pytest.raises(ValueError, match="checkpoint tensors"):
        verify_prefix_report(report, directory=directory)


def test_artifact_mode_recomputes_uncached_output_not_just_matching_claims(actual):
    directory, original = actual
    report = copy.deepcopy(original)
    report["rows"][2]["control_tokens"][1] ^= 1
    report["rows"][2]["observation"]["tokens"][1] ^= 1
    report = seal_prefix_report(report)
    assert verify_prefix_report(report) == report
    with pytest.raises(ValueError, match="reproduce recorded uncached"):
        verify_prefix_report(report, directory=directory)


@pytest.mark.parametrize("mutation", ["earlier", "vocabulary", "contradictory"])
def test_resealed_false_cancellation_is_rejected_even_with_artifacts(actual, mutation):
    directory, original = actual
    report = copy.deepcopy(original)
    cancellation = report["cancellation"]
    if mutation == "earlier":
        offset = (
            cancellation["finished_ns"]
            - report["rows"][0]["observation"]["started_ns"]
            + 1_000_000_000
        )
        for field in ("started_ns", "headers_ns", "finished_ns"):
            cancellation[field] -= offset
        cancellation["token_received_ns"] = [
            stamp - offset for stamp in cancellation["token_received_ns"]
        ]
    else:
        cancellation["tokens"][0] = (
            256 if mutation == "vocabulary" else cancellation["tokens"][0] ^ 1
        )
    with pytest.raises(ValueError):
        verify_prefix_report(seal_prefix_report(report), directory=directory)


@pytest.mark.parametrize(
    "filename,field",
    [
        ("source.txt", "source_sha256"),
        ("spec.json", "spec_sha256"),
        ("checkpoint.pt", "checkpoint_sha256"),
    ],
)
def test_artifact_checks_reject_resealed_false_digests(actual, filename, field):
    directory, original = actual
    report = copy.deepcopy(original)
    report[field] = sha256(b"not the retained artifact").hexdigest()
    with pytest.raises(ValueError, match="artifact differs"):
        verify_prefix_report(seal_prefix_report(report), directory=directory)


def test_readiness_rejects_nonempty_and_wrong_import_root_before_launch(
    actual, tmp_path
):
    directory, _ = actual
    root = Path(__file__).resolve().parents[1] / "src"
    with pytest.raises(ValueError, match="new or empty"):
        run_prefix_readiness(installed_root=root, output_dir=directory)
    with pytest.raises(ValueError, match="installed-root"):
        run_prefix_readiness(installed_root=tmp_path, output_dir=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_verify_fifo_is_rejected_before_any_potential_blocking_read(
    tmp_path, monkeypatch
):
    path = tmp_path / "owned.fifo"
    os.mkfifo(path)
    original = Path.read_text

    def guarded(self, *args, **kwargs):
        if self == path:
            raise AssertionError("FIFO must never reach a blocking read")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(SystemExit) as result:
        main(["--verify-report", str(path)])
    assert result.value.code == 2


@pytest.mark.parametrize(
    "raw", [b"x" * (1024 * 1024 + 1), b'{"schema": 1, "schema": 2}', b'{"schema": NaN}']
)
def test_verify_rejects_oversize_duplicate_and_nonfinite_regular_json(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(SystemExit) as result:
        main(["--verify-report", str(path)])
    assert result.value.code == 2
