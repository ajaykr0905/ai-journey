"""Bounded public-source provenance and operational replicated MoE admission."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlsplit

from .checkpoint_journal import _read_bytes
from .gpt2_training import GPT2TrainingSource
from .moe_ddp_config import MoEDDPConfig, moe_ddp_config_from_mapping
from .moe_ddp_resources import require_moe_ddp_budget

SCHEMA = "ai-journey-public-moe-ddp-plan-v1"
MAX_PLAN_BYTES = 64 * 1024
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_SAMPLES = 65536


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate MoE DDP JSON field")
        result[key] = value
    return result


@dataclass(frozen=True)
class PublicMoEDDPPlan:
    run: MoEDDPConfig
    source: GPT2TrainingSource

    def __post_init__(self):
        if (
            type(self.run) is not MoEDDPConfig
            or type(self.source) is not GPT2TrainingSource
            or self.run.model.gpt2.vocab_size != 256
        ):
            raise ValueError(
                "public byte MoE plan requires exact run/source and vocab256"
            )
        if any(
            type(value) is not str or not value.strip() or len(value) > 4096
            for value in asdict(self.source).values()
        ):
            raise ValueError("public source provenance must contain bounded strings")
        url = urlsplit(self.source.url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("public source URL must be credential-free HTTPS")
        # Guarantee canonical round-trip; the reader also bounds raw formatting.
        if len(canonical(self.to_mapping()).encode("utf-8")) > MAX_PLAN_BYTES:
            raise ValueError("public MoE plan exceeds its canonical wire byte budget")
        require_moe_ddp_budget(self.run)
        positions = (
            self.run.world_size
            * self.run.micro_batch_size
            * self.run.accumulation_steps
            * self.run.sequence_length
        )
        rows = (
            self.run.optimizer_steps
            * self.run.world_size
            * self.run.accumulation_steps
            * len(self.run.model.moe_layers)
        )
        estimate = (
            # Both raw rank receipts and the recomputed global view retain
            # routing evidence. Charge both representations before training.
            positions * self.run.optimizer_steps * len(self.run.model.moe_layers) * 256
            + rows * (4096 + 256 * self.run.model.expert_count)
            + MAX_PLAN_BYTES
        )
        if estimate > MAX_REPORT_BYTES:
            raise ValueError(
                "public MoE evidence would exceed its retained report budget"
            )

    def to_mapping(self):
        return {
            "schema": SCHEMA,
            "run": self.run.to_mapping(),
            "source": asdict(self.source),
        }

    def fingerprint(self):
        return sha256(canonical(self.to_mapping()).encode()).hexdigest()

    def data_contract(self):
        return {
            "schema": "ai-journey-public-byte-window-v1",
            "plan_sha256": self.fingerprint(),
            "sampler": "rank-strided-epoch-shuffle-v1",
            "tail_policy": "drop_global_incomplete_window",
            "capacity_scope": "rank_microbatch",
            "rank_rng": "cpu_generator_seed_plus_1000_plus_rank",
        }


def public_moe_ddp_plan_from_mapping(value):
    if (
        type(value) is not dict
        or set(value) != {"schema", "run", "source"}
        or value["schema"] != SCHEMA
    ):
        raise ValueError("public MoE DDP plan fields differ")
    source = value["source"]
    if type(source) is not dict or set(source) != {
        "name",
        "url",
        "revision",
        "license",
        "sha256",
    }:
        raise ValueError("public MoE DDP source fields differ")
    try:
        return PublicMoEDDPPlan(
            moe_ddp_config_from_mapping(value["run"]), GPT2TrainingSource(**source)
        )
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("public MoE DDP plan is invalid") from error


def source_admission(plan, source):
    if (
        type(source) is not bytes
        or not 1 < len(source) <= 1024 * 1024
        or sha256(source).hexdigest() != plan.source.sha256
    ):
        raise ValueError("public MoE source differs from its bounded declared bytes")
    count = (len(source) - 1) // plan.run.sequence_length
    global_batch = (
        plan.run.world_size * plan.run.micro_batch_size * plan.run.accumulation_steps
    )
    if not global_batch <= count <= MAX_SAMPLES:
        raise ValueError(
            "public MoE sample count cannot supply a bounded optimizer window"
        )
    return {
        "source_bytes": len(source),
        "sample_count": count,
        "discarded_samples": count % global_batch,
        "updates_per_epoch": count // global_batch,
    }


def read_public_moe_ddp_plan(config_path: Path, source_path: Path):
    value = json.loads(
        _read_bytes(config_path, MAX_PLAN_BYTES), object_pairs_hook=unique_fields
    )
    plan = public_moe_ddp_plan_from_mapping(value)
    source = _read_bytes(source_path, 1024 * 1024)
    source_admission(plan, source)
    return plan, source
