"""Publish usable trained MoE state bound to an unchanged experiment report."""

from hashlib import sha256
import os
from pathlib import Path
import tempfile

from .decoder_variant_data import document_schedule_fingerprint
from .decoder_variant_report_io import write_new_report
from .decoder_variant_runtime import isolated_decoder_runtime
from .decoder_variant_verify import _read_report
from .moe_checkpoint import read_moe_checkpoint, save_moe_checkpoint
from .moe_config import MoEConfig
from .moe_experiment import run_moe_document_trial
from .moe_experiment_comparison import assemble_moe_comparison
from .moe_experiment_plan import MoEExperimentPlan
from .moe_inference import load_moe_model_checkpoint
from .moe_report import canonical_moe_json, verify_moe_report
from .training_models import training_model_fingerprint

ARTIFACT_SCHEMA = "ai-journey-moe-trained-arm-v1"
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def _index_digest(index):
    return sha256(
        canonical_moe_json(
            {key: value for key, value in index.items() if key != "index_sha256"}
        ).encode()
    ).hexdigest()


def _checkpoint_digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact checkpoint must be a regular non-symlink file")
    digest, size = sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_CHECKPOINT_BYTES:
                raise ValueError("artifact checkpoint exceeds its byte budget")
            digest.update(chunk)
    return digest.hexdigest()


def _selection(plan, arm_name, seed):
    if not isinstance(plan, MoEExperimentPlan):
        raise TypeError("artifact publication requires a validated MoE plan")
    if type(seed) is not int or seed not in plan.seeds:
        raise ValueError("artifact seed must be declared by the plan")
    candidates = [arm for arm in plan.arms if arm.name == arm_name]
    if (
        type(arm_name) is not str
        or len(candidates) != 1
        or not isinstance(candidates[0].config, MoEConfig)
    ):
        raise ValueError("artifact arm must name one declared MoE treatment")
    return candidates[0]


def _training_contract(plan, *, seed, arm, schedule_sha256, heldout_sha256):
    return {
        "protocol": "ai-journey-moe-experiment-training-v1",
        "plan": plan.to_mapping(),
        "seed": seed,
        "arm": arm,
        "batch_schedule_sha256": schedule_sha256,
        "heldout_sha256": heldout_sha256,
    }


@isolated_decoder_runtime()
def publish_moe_experiment(plan, train_bytes, heldout_bytes, destination, *, arm, seed):
    """Run the declared comparison and publish one selected completed MoE arm.

    A new directory is claimed without replacement only after all measurements
    and checkpoint validation succeed. The index is linked last as the commit
    marker; interruption during publication may leave an explicitly incomplete
    directory, which readers reject and this function will never overwrite.
    """

    selected = _selection(plan, arm, seed)
    if not isinstance(destination, Path):
        raise TypeError("artifact destination must be pathlib.Path")
    if destination.exists() or destination.is_symlink():
        raise ValueError("artifact destination already exists")
    if not destination.parent.is_dir():
        raise ValueError("artifact parent directory must already exist")
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".moe-artifact-"
    ) as temporary:
        stage = Path(temporary)
        contracts = []

        def completed(declared, model, optimizer, evidence, schedule):
            if declared.name != arm:
                return
            contract = _training_contract(
                plan,
                seed=seed,
                arm=arm,
                schedule_sha256=document_schedule_fingerprint(schedule),
                heldout_sha256=sha256(heldout_bytes).hexdigest(),
            )
            save_moe_checkpoint(
                stage / "model.pt",
                model=model,
                optimizer=optimizer,
                step=plan.steps,
                training_contract=contract,
                source_sha256=sha256(train_bytes).hexdigest(),
            )
            if training_model_fingerprint(model) != evidence["final_model_sha256"]:
                raise RuntimeError("completed model differs from measured arm identity")
            contracts.append(contract)

        trials = [
            run_moe_document_trial(
                plan,
                train_bytes,
                heldout_bytes,
                seed=current,
                _completed_arm=completed if current == seed else None,
            )
            for current in plan.seeds
        ]
        if len(contracts) != 1:
            raise RuntimeError("selected trained arm was not published exactly once")
        report = assemble_moe_comparison(plan, trials)
        verify_moe_report(report)
        write_new_report(stage / "report.json", report)
        index = {
            "schema": ARTIFACT_SCHEMA,
            "report_sha256": report["report_sha256"],
            "checkpoint_sha256": _checkpoint_digest(stage / "model.pt"),
            "arm": selected.name,
            "seed": seed,
            "training_contract": contracts[0],
        }
        index["index_sha256"] = _index_digest(index)
        write_new_report(stage / "index.json", index)
        load_moe_experiment_artifact(stage, train_bytes)
        destination.mkdir()  # Atomic unused-path claim: no replace and no exist_ok.
        for name in ("model.pt", "report.json", "index.json"):
            os.link(stage / name, destination / name)
        for directory in (destination, destination.parent):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return index


@isolated_decoder_runtime()
def load_moe_experiment_artifact(directory, train_bytes):
    """Verify bindings and actual training bytes, then return frozen dropless eval.

    This is checkpoint/report integrity, not authentication of a training claim.
    Exact source replay remains the separate experiment replay command.
    """

    if (
        not isinstance(directory, Path)
        or directory.is_symlink()
        or not directory.is_dir()
    ):
        raise ValueError("artifact directory must be a regular non-symlink directory")
    for name in ("index.json", "report.json"):
        if (directory / name).is_symlink() or not (directory / name).is_file():
            raise ValueError("artifact requires complete regular index/report files")
    index = _read_report(directory / "index.json", max_bytes=128 * 1024)
    if (
        type(index) is not dict
        or set(index)
        != {
            "schema",
            "report_sha256",
            "checkpoint_sha256",
            "arm",
            "seed",
            "training_contract",
            "index_sha256",
        }
        or index["schema"] != ARTIFACT_SCHEMA
        or index["index_sha256"] != _index_digest(index)
    ):
        raise ValueError("artifact index fields or seal differ")
    report = _read_report(directory / "report.json", max_bytes=4 * 1024 * 1024)
    plan = verify_moe_report(report)
    selected = _selection(plan, index["arm"], index["seed"])
    trial = next(row for row in report["trials"] if row["seed"] == index["seed"])
    measured = next(row for row in trial["arms"] if row["name"] == selected.name)
    contract = _training_contract(
        plan,
        seed=index["seed"],
        arm=selected.name,
        schedule_sha256=trial["batch_schedule_sha256"],
        heldout_sha256=trial["heldout_sha256"],
    )
    if (
        index["report_sha256"] != report["report_sha256"]
        or canonical_moe_json(index["training_contract"])
        != canonical_moe_json(contract)
        or type(train_bytes) is not bytes
        or sha256(train_bytes).hexdigest() != trial["train_sha256"]
    ):
        raise ValueError("artifact report, training contract or source binding differs")
    checkpoint = directory / "model.pt"
    if _checkpoint_digest(checkpoint) != index["checkpoint_sha256"]:
        raise ValueError("artifact checkpoint file digest differs")
    payload = read_moe_checkpoint(checkpoint)
    if payload["step"] != plan.steps:
        raise ValueError("artifact checkpoint progress differs from the experiment")
    del payload
    model = load_moe_model_checkpoint(
        checkpoint,
        config=selected.config,
        training_contract=contract,
        source=train_bytes,
        max_parameters=2_000_000,
    )
    if training_model_fingerprint(model) != measured["final_model_sha256"]:
        raise ValueError(
            "artifact checkpoint model differs from the measured final arm"
        )
    return model, report
