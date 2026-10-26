import copy

import pytest
import torch

from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.moe_experiment_comparison import run_moe_document_comparison
from ai_journey.moe_replay import replay_moe_report
from ai_journey.moe_report import verify_moe_report
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan


@pytest.fixture(scope="module")
def report():
    return run_moe_document_comparison(plan(), TRAIN, HELDOUT)


def test_real_report_replays_all_tensors_updates_losses_routes_and_summary(report):
    original = copy.deepcopy(report)
    rng = torch.get_rng_state().clone()
    assert replay_moe_report(report, TRAIN, HELDOUT) == plan()
    assert report == original
    assert torch.equal(torch.get_rng_state(), rng)


def test_replay_ignores_and_restores_ambient_default_dtype_device_and_rng(
    report, monkeypatch
):
    monkeypatch.setattr(
        torch, "manual_seed", lambda *a: pytest.fail("must not seed accelerators")
    )
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            replay_moe_report(report, TRAIN, HELDOUT)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(dtype)


def test_semantically_plausible_fabricated_model_identity_needs_actual_replay(report):
    forged = copy.deepcopy(report)
    forged["trials"][0]["arms"][1]["final_model_sha256"] = "0" * 64
    forged["report_sha256"] = decoder_comparison_digest(forged)
    verify_moe_report(forged)
    with pytest.raises(ValueError, match="exact executable replay"):
        replay_moe_report(forged, TRAIN, HELDOUT)


@pytest.mark.parametrize("case", ["source", "runtime", "semantic"])
def test_invalid_prerequisites_reject_before_any_model_execution(
    report, monkeypatch, case
):
    import ai_journey.moe_replay as module

    monkeypatch.setattr(
        module,
        "run_moe_document_comparison",
        lambda *a: pytest.fail("must reject before model execution"),
    )
    actual = copy.deepcopy(report)
    source = TRAIN
    if case == "source":
        source += b"changed"
    elif case == "runtime":
        actual["runtime"]["torch_version"] = "other-version"
        actual["report_sha256"] = decoder_comparison_digest(actual)
    else:
        actual["summary"]["heldout_nll"]["baseline"]["mean"] = 0.0
        actual["report_sha256"] = decoder_comparison_digest(actual)
    with pytest.raises(ValueError):
        replay_moe_report(actual, source, HELDOUT)
