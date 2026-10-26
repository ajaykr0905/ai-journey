"""Adversarial restart contracts for the controlled training entry point."""

import copy
import zipfile

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import (
    GPT2BatchCursor,
    GPT2ByteCorpus,
    GPT2TrainingConfig,
    build_gpt2_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
    train_optimizer_step,
)


@pytest.fixture
def restart_state(tmp_path):
    model = GPT2LanguageModel(GPT2Config(256, 4, 8, 2, 1))
    config = GPT2TrainingConfig(2, 1, 1, 4, 0.01, 0.001, 0, 0.1, 1.0, 52)
    corpus = GPT2ByteCorpus.from_bytes(b"public restart fixture" * 3)
    cursor = GPT2BatchCursor(corpus, config)
    optimizer = build_gpt2_optimizer(model, config)
    train_optimizer_step(model, cursor, optimizer, config, step=0)
    path = tmp_path / "checkpoint.pt"
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        cursor=cursor,
        config=config,
        corpus_fingerprint=corpus.fingerprint(),
    )
    save_training_checkpoint(path, step=1, **kwargs)
    return path, kwargs


@pytest.mark.parametrize("corruption", ["shape", "negative_variance", "counter", "lr"])
def test_invalid_finite_adamw_state_is_rejected_transactionally(
    restart_state, corruption
):
    path, kwargs = restart_state
    payload = torch.load(path, weights_only=True)
    state = next(iter(payload["optimizer_state"]["state"].values()))
    if corruption == "shape":
        state["exp_avg"] = torch.zeros(1)
    elif corruption == "negative_variance":
        state["exp_avg_sq"].fill_(-1)
    elif corruption == "counter":
        state["step"].fill_(0)
    else:
        payload["optimizer_state"]["param_groups"][0]["lr"] = -0.1
    torch.save(payload, path)
    before = copy.deepcopy(kwargs["model"].state_dict())
    cursor = kwargs["cursor"].state_dict()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="AdamW"):
        load_training_checkpoint(path, **kwargs)
    assert all(
        torch.equal(tensor, before[name])
        for name, tensor in kwargs["model"].state_dict().items()
    )
    assert kwargs["cursor"].state_dict() == cursor
    assert torch.equal(torch.get_rng_state(), rng)
    for state in kwargs["optimizer"].state.values():
        assert state["step"].item() == 1
        assert (state["exp_avg_sq"] >= 0).all()


def test_invalid_live_adamw_state_cannot_replace_durable_checkpoint(restart_state):
    path, kwargs = restart_state
    before = path.read_bytes()
    next(iter(kwargs["optimizer"].state.values()))["exp_avg_sq"].fill_(-1)
    with pytest.raises(ValueError, match="AdamW"):
        save_training_checkpoint(path, step=1, **kwargs)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["compressed", "records", "expanded"])
def test_archive_budget_rejects_before_deserialization(
    restart_state, monkeypatch, kind
):
    path, kwargs = restart_state
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED
        if kind == "compressed"
        else zipfile.ZIP_STORED,
    ) as archive:
        if kind == "records":
            for index in range(1025):
                archive.writestr(f"record-{index}", b"")
        else:
            archive.writestr("storage", b"x" * 100000)
    if kind == "expanded":
        # The archive itself must fit; its metadata describes storage beyond budget.
        with path.open("r+b") as stream:
            data = stream.read()
            central = data.index(b"PK\x01\x02")
            stream.seek(central + 24)
            stream.write((1000000).to_bytes(4, "little"))

    def unexpected(*args, **kwargs):
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(torch, "load", unexpected)
    with pytest.raises(ValueError, match="weights-only archive"):
        load_training_checkpoint(path, max_checkpoint_bytes=200000, **kwargs)


def test_corrupt_pickle_is_a_training_error(restart_state):
    path, kwargs = restart_state
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"invalid pickle")
        archive.writestr("archive/version", b"3\n")
    with pytest.raises(ValueError, match="weights-only archive"):
        load_training_checkpoint(path, **kwargs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("lr", 0.5),
        ("weight_decay", 0.2),
        ("betas", (0.8, 0.99)),
        ("eps", 1e-6),
        ("maximize", True),
    ],
)
def test_finite_optimizer_control_drift_cannot_override_spec(
    restart_state, name, value
):
    path, kwargs = restart_state
    payload = torch.load(path, weights_only=True)
    payload["optimizer_state"]["param_groups"][0][name] = value
    torch.save(payload, path)
    original = copy.deepcopy(kwargs["optimizer"].param_groups[0])
    with pytest.raises(ValueError, match="AdamW controls"):
        load_training_checkpoint(path, **kwargs)
    restored = kwargs["optimizer"].param_groups[0]
    assert restored[name] == original[name]


def test_live_optimizer_control_drift_cannot_publish(restart_state):
    path, kwargs = restart_state
    original = path.read_bytes()
    kwargs["optimizer"].param_groups[0]["weight_decay"] = 0.2
    with pytest.raises(ValueError, match="AdamW controls"):
        save_training_checkpoint(path, step=1, **kwargs)
    assert path.read_bytes() == original


def _same_restart_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_restart_state(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _same_restart_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


@pytest.mark.parametrize(
    "name,value",
    [
        ("capturable", True),
        ("capturable", 0),
        ("capturable", None),
        ("differentiable", True),
        ("differentiable", 0),
        ("differentiable", None),
        ("foreach", True),
        ("foreach", False),
        ("foreach", 0),
        ("foreach", "false"),
        ("fused", True),
        ("fused", False),
        ("fused", 0),
        ("fused", "false"),
    ],
)
def test_cpu_execution_control_drift_is_rejected_and_restores_every_state(
    restart_state, name, value
):
    path, kwargs = restart_state
    model, optimizer, cursor = kwargs["model"], kwargs["optimizer"], kwargs["cursor"]
    model.ln_f.eval()
    model.wte.weight.grad = torch.ones_like(model.wte.weight)
    before = copy.deepcopy(
        (
            model.state_dict(),
            optimizer.state_dict(),
            cursor.state_dict(),
            torch.get_rng_state(),
            model.wte.weight.grad,
        )
    )
    modes = [module.training for module in model.modules()]
    payload = torch.load(path, weights_only=True)
    for group in payload["optimizer_state"]["param_groups"]:
        group[name] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=f"AdamW CPU execution control {name}"):
        load_training_checkpoint(path, **kwargs)
    after = (
        model.state_dict(),
        optimizer.state_dict(),
        cursor.state_dict(),
        torch.get_rng_state(),
        model.wte.weight.grad,
    )
    assert _same_restart_state(before, after)
    assert [module.training for module in model.modules()] == modes


@pytest.mark.parametrize("name", ["capturable", "differentiable", "foreach", "fused"])
def test_unsupported_live_cpu_execution_controls_cannot_replace_checkpoint(
    restart_state, name
):
    path, kwargs = restart_state
    original = path.read_bytes()
    kwargs["optimizer"].param_groups[0][name] = True
    with pytest.raises(ValueError, match="AdamW CPU execution control"):
        save_training_checkpoint(path, step=1, **kwargs)
    assert path.read_bytes() == original


def test_default_cpu_execution_controls_restore_and_continue_training(restart_state):
    path, kwargs = restart_state
    assert load_training_checkpoint(path, **kwargs) == 1
    for group in kwargs["optimizer"].param_groups:
        assert group["capturable"] is group["differentiable"] is False
        assert group["foreach"] is group["fused"] is None
    metric = train_optimizer_step(
        kwargs["model"], kwargs["cursor"], kwargs["optimizer"], kwargs["config"], step=1
    )
    assert metric.step == 2


@pytest.mark.parametrize(
    "state",
    [{"epoch": 0, "offset": 0}, {"epoch": 0, "offset": 2}, {"epoch": 1, "offset": 1}],
)
def test_valid_but_inconsistent_cursor_boundary_is_rejected(restart_state, state):
    path, kwargs = restart_state
    payload = torch.load(path, weights_only=True)
    payload["cursor_state"] = state
    torch.save(payload, path)
    original = kwargs["cursor"].state_dict()
    with pytest.raises(ValueError, match="cursor progress"):
        load_training_checkpoint(path, **kwargs)
    assert kwargs["cursor"].state_dict() == original


def test_live_cursor_drift_cannot_replace_recoverable_checkpoint(restart_state):
    path, kwargs = restart_state
    original = path.read_bytes()
    kwargs["cursor"].next()
    with pytest.raises(ValueError, match="cursor progress"):
        save_training_checkpoint(path, step=1, **kwargs)
    assert path.read_bytes() == original


def test_restart_reproduces_sampler_across_multiple_epoch_boundaries(tmp_path):
    from ai_journey.gpt2_training import train_optimizer_steps

    model = GPT2LanguageModel(
        GPT2Config(
            256,
            4,
            8,
            2,
            1,
            embedding_dropout=0,
            attention_dropout=0,
            residual_dropout=0,
        )
    )
    config = GPT2TrainingConfig(4, 2, 3, 4, 0.01, 0.001, 0, 0.1, 1.0, 52)
    corpus = GPT2ByteCorpus.from_bytes(b"abcdefghijkl")
    cursor = GPT2BatchCursor(corpus, config)
    optimizer = build_gpt2_optimizer(model, config)
    train_optimizer_steps(model, cursor, optimizer, config, step_count=3)
    path = tmp_path / "epochs.pt"
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        cursor=cursor,
        config=config,
        corpus_fingerprint=corpus.fingerprint(),
    )
    save_training_checkpoint(path, step=3, **kwargs)
    expected = cursor.next()
    assert load_training_checkpoint(path, **kwargs) == 3
    actual = cursor.next()
    assert all(
        torch.equal(left, right) for left, right in zip(expected, actual, strict=True)
    )
