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
SCHEMA_V2 = "ai-journey-public-moe-ddp-plan-v2"
SAMPLER_V1 = "rank-strided-epoch-shuffle-v1"
SAMPLER_V2 = "rank-strided-full-epoch-shuffle-v2"
MAX_PLAN_BYTES = 64 * 1024
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_SAMPLES = 65536


def canonical(value):
    """Sorted, compact, NaN-free JSON used for plan digests and byte budgets."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_fields(pairs):
    """JSON object hook that rejects a duplicate key instead of keeping the last.

    A plan or report with two values for one field must not be read
    differently by different parsers. Raises ValueError.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate MoE DDP JSON field")
        result[key] = value
    return result


@dataclass(frozen=True)
class PublicMoEDDPPlan:
    """One replicated MoE run bound to its public byte source and sampler.

    Construction requires the 256-byte vocabulary, a supported sampler,
    bounded credential-free HTTPS provenance and a canonical form within
    64 KiB, and charges the run's compute and retained-evidence budgets, so
    an oversized plan fails before any worker starts. Raises ValueError.
    """

    run: MoEDDPConfig
    source: GPT2TrainingSource
    sampler: str = SAMPLER_V1

    def __post_init__(self):
        if (
            type(self.run) is not MoEDDPConfig
            or type(self.source) is not GPT2TrainingSource
            or self.run.model.gpt2.vocab_size != 256
        ):
            raise ValueError(
                "public byte MoE plan requires exact run/source and vocab256"
            )
        if type(self.sampler) is not str or self.sampler not in (
            SAMPLER_V1,
            SAMPLER_V2,
        ):
            raise ValueError("public MoE plan requires a supported sampler version")
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
        """Canonical wire form; v1 plans keep exactly schema, run and source.

        Leaving the sampler out of v1 preserves the bytes and digests of plans
        recorded before v2 existed; v2 plans carry their explicit selector.
        """
        mapping = {
            "schema": SCHEMA if self.sampler == SAMPLER_V1 else SCHEMA_V2,
            "run": self.run.to_mapping(),
            "source": asdict(self.source),
        }
        if self.sampler == SAMPLER_V2:
            mapping["sampler"] = self.sampler
        return mapping

    def fingerprint(self):
        """SHA-256 of the canonical form: the plan identity launches check."""
        return sha256(canonical(self.to_mapping()).encode()).hexdigest()

    def data_contract(self):
        """Data-order contract that every replicated checkpoint records.

        Restore and archive verification compare it with the current plan's
        contract, so a run cannot resume or verify under another plan, sampler
        or tail policy.
        """
        return {
            "schema": (
                "ai-journey-public-byte-window-v1"
                if self.sampler == SAMPLER_V1
                else "ai-journey-public-byte-window-v2"
            ),
            "plan_sha256": self.fingerprint(),
            "sampler": self.sampler,
            "tail_policy": "drop_global_incomplete_window",
            "capacity_scope": "rank_microbatch",
            "rank_rng": "cpu_generator_seed_plus_1000_plus_rank",
        }


def public_moe_ddp_plan_from_mapping(value):
    """Rebuild a plan from JSON, accepting only the exact v1 or v2 field set.

    A v2 plan must name the v2 sampler and a v1 plan may not name one, so a
    plan cannot change its data order without changing its schema. Raises
    ValueError, including for any error raised while constructing the plan.
    """
    if type(value) is not dict:
        raise ValueError("public MoE DDP plan fields differ")
    if value.get("schema") == SCHEMA:
        expected = {"schema", "run", "source"}
        sampler = SAMPLER_V1
    elif value.get("schema") == SCHEMA_V2:
        expected = {"schema", "run", "source", "sampler"}
        sampler = value.get("sampler")
        if type(sampler) is not str or sampler != SAMPLER_V2:
            raise ValueError("public MoE v2 plan requires its explicit sampler")
    else:
        raise ValueError("public MoE DDP plan fields differ")
    if set(value) != expected:
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
            moe_ddp_config_from_mapping(value["run"]),
            GPT2TrainingSource(**source),
            sampler=sampler,
        )
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("public MoE DDP plan is invalid") from error


def source_admission(plan, source):
    """Check source bytes against ``plan.source.sha256`` and size its windows.

    The bytes (at most 1 MiB) must match that declared digest and supply at
    least one global optimizer window and at most 65536 samples. Returns the
    counts the plan command reports; raises ValueError otherwise.
    """
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
    """Read and admit a plan and its source through bounded regular-file reads.

    Reads refuse a symlinked final path and a file that changes while read;
    duplicate JSON keys, a plan over 64 KiB, a source over 1 MiB or a source
    that differs from its declared digest raise ValueError. Returns both.
    """
    value = json.loads(
        _read_bytes(config_path, MAX_PLAN_BYTES), object_pairs_hook=unique_fields
    )
    plan = public_moe_ddp_plan_from_mapping(value)
    source = _read_bytes(source_path, 1024 * 1024)
    source_admission(plan, source)
    return plan, source
