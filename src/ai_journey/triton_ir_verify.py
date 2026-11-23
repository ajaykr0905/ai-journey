"""Independent verification of sealed Day 80 Triton IR reports, on the CPU.

The verifier checks the field set, the label and its evidence boundary and
the SHA-256 seal, and then rebuilds every section from the fixture folder
with other code than the lab's:

* the manifest from the raw ``index.json`` bytes with plain ``json``, and
  its digest against a digest pinned here;
* the source digest with ``hashlib`` over ``TRITON_SOURCE``;
* each committed text by its bytes, its digest and its line count;
* the PTX counts with whole-text regular expressions and a prefix table;
* the TTIR and TTGIR counts with a brace-depth scan and a fixpoint pass over
  the ``#loc`` table, where the lab resolves each location by recursion;
* the stage map, the pipeline facts and the comparison from those recounts
  and from the raw manifest mapping, not from the lab's model.

A regeneration section cannot be recompiled here, so its rows are checked
for structure: each recorded digest is the manifest digest, each match flag
is the equality of the two digests, and the summary recounts. The verifier
imports neither triton nor torch.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import canonical_json_bytes, loads_strict, read_bounded
from .kernel_launch_verify import KernelLaunchVerificationError, _compare
from .triton_ir_compare import NEEDS_FULL_DUMPS
from .triton_ir_manifest import TritonIrError
from .triton_ir_report import (
    BOUNDARIES,
    DIGEST_FIELD,
    LABELS,
    RECORDED_LABEL,
    REGENERATION_LABEL,
    REPORT_KIND,
    SCHEMA_VERSION,
)
from .triton_ir_runtime import ENVIRONMENT_FIELDS
from .triton_ir_stage_map import STATEMENTS
from .triton_matmul_kernel import TRITON_SOURCE

MAX_REPORT_BYTES = 1024 * 1024
# Pinned here independently of the modules that build the report.
_MANIFEST_SHA256 = "95c187abaae39edda3262eec53588855360b4d9b933f7cbe4e3536b3696b6473"
_TARGETS = ("sm_80", "sm_90")
_STAGES = ("ttir", "ttgir", "llir", "ptx")
_COMMITTED = ("sm_80", "m32_n64_k32_g8_w2_s5", ("ttir", "ttgir", "ptx"))
_PREFIX = "<ai_journey.triton_matmul_kernel.TRITON_SOURCE"
_WARP = 32
_FLOAT = 4
_FAMILY_ORDER = (
    "ld.global",
    "st.global",
    "cp.async",
    "cp.async.commit_group",
    "cp.async.wait_group",
    "ld.shared",
    "ldmatrix",
    "st.shared",
    "bar.sync",
    "fma.rn.f32",
    "mma",
    "setp",
    "selp",
    "branch",
    "other",
)
# First match wins: exact names, then prefixes.
_EXACT = {
    "cp.async.commit_group": "cp.async.commit_group",
    "fma.rn.f32": "fma.rn.f32",
    "bra": "branch",
}
_PREFIXES = (
    ("cp.async.wait_group", "cp.async.wait_group"),
    ("cp.async.wait_all", "cp.async.wait_group"),
    ("cp.async.", "cp.async"),
    ("ld.global", "ld.global"),
    ("st.global", "st.global"),
    ("ld.shared", "ld.shared"),
    ("st.shared", "st.shared"),
    ("ldmatrix.", "ldmatrix"),
    ("bar.sync", "bar.sync"),
    ("barrier.sync", "bar.sync"),
    ("mma.", "mma"),
    ("wgmma.", "mma"),
    ("setp.", "setp"),
    ("selp.", "selp"),
    ("bra.", "branch"),
)
_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "label",
    "evidence_boundary",
    "manifest",
    "manifest_sha256",
    "source",
    "fixtures",
    "ptx",
    "ttir",
    "ttgir",
    "stage_map",
    "pipeline",
    "comparison",
    "regeneration",
    "recorded_at",
    "host_load",
    "summary",
    DIGEST_FIELD,
}
_HEX = re.compile(r"[0-9a-f]{64}")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class TritonIrVerificationError(TritonIrError):
    """Raised when a Triton IR report does not reproduce from its fixtures."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TritonIrVerificationError(message)


def _read(path: Path, limit: int, label: str) -> bytes:
    try:
        return read_bounded(path, limit=limit, label=label)
    except KernelLaunchError as error:
        raise TritonIrError(str(error)) from error


def _family(opcode: str) -> str:
    if opcode in _EXACT:
        return _EXACT[opcode]
    for prefix, name in _PREFIXES:
        if opcode.startswith(prefix):
            return name
    return "other"


def _families(opcodes: list[str]) -> dict[str, int]:
    counts = Counter(_family(opcode) for opcode in opcodes)
    return {name: counts.get(name, 0) for name in _FAMILY_ORDER}


def _recount_ptx(text: str) -> tuple[dict[str, object], list[tuple[int, str, int]]]:
    """Recount a PTX text; return its mapping and (line, opcode, source) rows."""

    def one(pattern: str) -> str:
        found = re.findall(pattern, text, re.MULTILINE)
        _require(len(found) == 1, f"the PTX must match {pattern} once")
        return found[0]

    entry_start = text.index(".visible .entry ")
    header_end = text.index("\n)\n", entry_start)
    entry = text[entry_start:header_end]
    parameters = re.findall(r"^\s*\.param (.+?) (\w+),?$", entry, re.MULTILINE)
    body_start = text.index("\n{\n", header_end) + 3
    body_end = text.index("\n}\n", body_start)
    body = text[body_start:body_end].split("\n")
    first_line = text[:body_start].count("\n") + 1
    files = dict(re.findall(r'^\s*\.file\s+(\d+)\s+"([^"]*)"', text, re.MULTILINE))
    source = [int(key) for key, name in files.items() if name.startswith(_PREFIX)]
    source_file = source[0] if len(source) == 1 else None
    rows: list[tuple[int, str, int]] = []
    guarded = 0
    labels: dict[str, int] = {}
    branches: list[tuple[int, str]] = []
    current = 0
    for offset, raw in enumerate(body):
        number = first_line + offset
        line = raw.strip()
        if line.startswith(".loc"):
            inline = re.search(r"inlined_at (\d+) (\d+) ", line)
            fields = line.split()
            file, at = (
                (inline.group(1), inline.group(2)) if inline else (fields[1], fields[2])
            )
            current = int(at) if source_file == int(file) else 0
            continue
        code = line.split("//")[0].strip()
        if not code or code[0] in ".{}":
            continue
        if code.startswith("$") and code.endswith(":"):
            labels[code[:-1]] = number
            continue
        if code.startswith("@"):
            guarded += 1
            code = code.split(None, 1)[1]
        opcode = code.split()[0].rstrip(";")
        rows.append((number, opcode, current))
        if opcode == "bra" or opcode.startswith("bra."):
            branches.append((number, code.split()[-1].rstrip(";")))
    loops = []
    for label, start in sorted(labels.items(), key=lambda item: item[1]):
        back = [
            number for number, target in branches if target == label and number > start
        ]
        if back:
            inside = [
                opcode for number, opcode, _ in rows if start < number <= max(back)
            ]
            loops.append(
                {
                    "label": label,
                    "first_line": start,
                    "last_line": max(back),
                    "instructions": len(inside),
                    "families": _families(inside),
                }
            )
    registers = re.findall(r"^\s*\.reg \.(\w+)\s+%(\w+)<(\d+)>;", text, re.MULTILINE)
    shared = re.findall(
        r"^(\.extern )?\.shared \.align (\d+) \.(\w+) (\w+)\[(\d*)\];",
        text,
        re.MULTILINE,
    )
    reqntid = re.findall(r"^\.reqntid ([\d, ]+)$", text, re.MULTILINE)
    opcodes = Counter(opcode for _, opcode, _ in rows)
    mapping = {
        "version": one(r"^\.version (\S+)$"),
        "target": one(r"^\.target (\S+)$"),
        "address_size": int(one(r"^\.address_size (\d+)$")),
        "entry": one(r"^\.visible \.entry (\w+)\($"),
        "parameters": [{"name": name, "type": kind} for kind, name in parameters],
        "reqntid": reqntid[0] if reqntid else None,
        "registers": [
            {"class": kind, "prefix": prefix, "declared": int(count)}
            for kind, prefix, count in registers
        ],
        "shared": [
            {
                "name": name,
                "extern": bool(extern),
                "align": int(align),
                "element": element,
                "elements": int(size) if size else None,
            }
            for extern, align, element, name, size in shared
        ],
        "instructions": len(rows),
        "predicated": guarded,
        "families": _families([opcode for _, opcode, _ in rows]),
        "opcodes": dict(opcodes),
        "loops": loops,
    }
    return mapping, rows


def _locations(text: str) -> tuple[dict[str, int], int]:
    """Resolve every ``#loc`` alias to a source line (0 for none) by fixpoint."""

    table = dict(re.findall(r"^(#loc\d*) = loc\((.*)\)$", text, re.MULTILINE))
    resolved: dict[str, int] = {}

    def place(expression: str) -> int | None:
        found = re.fullmatch(r'"([^"]*)":(\d+):\d+', expression)
        if found:
            return int(found.group(2)) if found.group(1).startswith(_PREFIX) else 0
        if expression == "unknown":
            return 0
        found = re.fullmatch(r'"[^"]*"\((.*)\)', expression)
        if found:
            return place(found.group(1))
        found = re.fullmatch(r"callsite\(#loc\d* at (#loc\d*)\)", expression)
        reference = found.group(1) if found else expression
        return resolved.get(reference)

    changed = True
    while changed:
        changed = False
        for name, expression in table.items():
            if name not in resolved:
                value = place(expression)
                if value is not None:
                    resolved[name] = value
                    changed = True
    _require(len(resolved) == len(table), "an MLIR location does not resolve")
    return resolved, len(table)


def _recount_mlir(text: str) -> list[tuple[int, str, str, int]]:
    """Return (line, op name, op text, source line) for each op of the module."""

    resolved, _ = _locations(text)
    lines = text.split("\n")
    module = [i for i, line in enumerate(lines) if line.startswith("module")]
    _require(len(module) == 1, "an MLIR text must hold one module")
    rows: list[tuple[int, str, str, int]] = []
    stack: list[tuple[int, str, str]] = []
    for index in range(module[0] + 1, len(lines)):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("}"):
            loc = re.search(r"loc\((#loc\d*)\)$", line)
            if stack:
                number, name, body = stack.pop()
                rows.append((number, name, body, resolved[loc.group(1)]))
            continue
        name = re.match(r"(?:%\S+(?:, %\S+)* = )?([a-z_]+\.[a-z_.0-9]+)", line).group(1)
        if line.count("{") > line.count("}"):
            stack.append((index + 1, name, line))
            continue
        loc = re.search(r" loc\((#loc\d*)\)$", line)
        rows.append((index + 1, name, line[: loc.start()], resolved[loc.group(1)]))
    # The module region closes last and is not an op.
    return sorted(row for row in rows if row[1] != "module")


def _operand_count(name: str, body: str) -> int:
    after = body.split(name, 1)[1].split(" : ", 1)[0]
    return len([part for part in after.split(",") if part.strip()])


def _common(text: str, rows: list[tuple[int, str, str, int]]) -> dict[str, object]:
    names = Counter(name for _, name, _, _ in rows)
    loads = [body for _, name, body, _ in rows if name == "tt.load"]
    stores = [body for _, name, body, _ in rows if name == "tt.store"]
    ranges = set()
    for _, name, body, _ in rows:
        if name == "tt.make_range":
            end, start = re.search(
                r"end = (-?\d+) : i32, start = (-?\d+)", body
            ).groups()
            ranges.add((int(start), int(end)))
    return {
        "function": re.search(r"tt\.func \w* ?@(\w+)\(", text).group(1),
        "ops": len(rows),
        "histogram": dict(names),
        "locations": _locations(text)[1],
        "ops_with_source_line": sum(line > 0 for _, _, _, line in rows),
        "tt_load": len(loads),
        "tt_load_masked": sum(_operand_count("tt.load", b) >= 2 for b in loads),
        "tt_load_with_other": sum(_operand_count("tt.load", b) >= 3 for b in loads),
        "tt_store": len(stores),
        "tt_store_masked": sum(_operand_count("tt.store", b) >= 3 for b in stores),
        "tt_dot": names["tt.dot"],
        "tt_make_range": [{"start": s, "end": e} for s, e in sorted(ranges)],
        "tt_splat": names["tt.splat"],
        "scf_for": names["scf.for"],
    }


def _ttgir(text: str, rows: list[tuple[int, str, str, int]]) -> dict[str, object]:
    names = Counter(name for _, name, _, _ in rows)
    header = re.search(r"^module attributes \{(.*)\} \{$", text, re.MULTILINE).group(1)
    attributes = {}
    for key, value in re.findall(r'"?([\w.-]+)"? = ("[^"]*"|[\w-]+)', header):
        attributes[key] = value.strip('"')
    aliases = re.findall(r"^#(\w+) = #([\w.]+)(<.*>)?$", text, re.MULTILINE)
    kinds = {name: kind for name, kind, _ in aliases}
    bodies = [body for _, _, body, _ in rows]
    inline = Counter(
        kind for body in bodies for kind in re.findall(r"#ttg\.(\w+)<", body)
    )
    dots = set()
    for _, name, body, _ in rows:
        if name == "tt.dot":
            dots |= set(re.findall(r"opIdx = (\d), parent = #(\w+)\}", body))
    every = list(kinds.values()) + [f"ttg.{kind}" for kind in inline]
    return {
        **_common(text, rows),
        "attributes": attributes,
        "num_warps": int(attributes["ttg.num-warps"]),
        "threads_per_warp": int(attributes["ttg.threads-per-warp"]),
        "target": attributes["ttg.target"],
        "encodings": {
            name: {"kind": kind, "parameters": parameters}
            for name, kind, parameters in aliases
        },
        "inline_encodings": dict(inline),
        "mma_encoding": any("mma" in kind for kind in every),
        "dot_operands": [
            {
                "op_idx": int(index),
                "parent": "#" + parent,
                "parent_kind": kinds.get(parent),
            }
            for index, parent in sorted(dots)
        ],
        "local_alloc": [
            re.search(r"memdesc<([0-9x]+)x[a-z]+\d+", body).group(1)
            for _, name, body, _ in rows
            if name == "ttg.local_alloc"
        ],
        "async_copy_global_to_local": names["ttg.async_copy_global_to_local"],
        "async_commit_group": names["ttg.async_commit_group"],
        "async_wait": [
            int(re.search(r"\{num = (\d+) : i32\}", body).group(1))
            for _, name, body, _ in rows
            if name == "ttg.async_wait"
        ],
        "local_load": names["ttg.local_load"],
        "convert_layout": names["ttg.convert_layout"],
    }


def _stage_map(ttir_rows, ttgir_rows, ptx_rows, loop) -> dict[str, object]:
    source = TRITON_SOURCE.split("\n")
    first, last = loop["first_line"], loop["last_line"]

    def count(items) -> dict[str, int]:
        return dict(Counter(items))

    statements = []
    for key, statement in STATEMENTS:
        line = [i + 1 for i, text in enumerate(source) if text.strip() == statement]
        _require(len(line) == 1, f"the kernel text must hold {statement!r} once")
        number = line[0]
        mine = [(n, op) for n, op, at in ptx_rows if at == number]
        statements.append(
            {
                "key": key,
                "line": number,
                "statement": statement,
                "ttir": count(name for _, name, _, at in ttir_rows if at == number),
                "ttgir": count(name for _, name, _, at in ttgir_rows if at == number),
                "ptx": count(_family(op) for _, op in mine),
                "ptx_k_loop": count(_family(op) for n, op in mine if first < n <= last),
            }
        )
    used = sorted(
        {at for _, _, _, at in (*ttir_rows, *ttgir_rows) if at}
        | {at for _, _, at in ptx_rows if at}
    )
    return {
        "statements": statements,
        "lines": [
            {
                "line": line,
                "ttir_ops": sum(at == line for _, _, _, at in ttir_rows),
                "ttgir_ops": sum(at == line for _, _, _, at in ttgir_rows),
                "ptx_instructions": sum(at == line for _, _, at in ptx_rows),
            }
            for line in used
        ],
        "without_source_line": {
            "ttir_ops": sum(at == 0 for _, _, _, at in ttir_rows),
            "ttgir_ops": sum(at == 0 for _, _, _, at in ttgir_rows),
            "ptx_instructions": sum(at == 0 for _, _, at in ptx_rows),
        },
    }


def _config(row: Mapping[str, object]) -> Mapping[str, object]:
    return row["config"]


def _tile_bytes(config: Mapping[str, object]) -> int:
    m, n, k = config["block_m"], config["block_n"], config["block_k"]
    return (m * k + k * n) * _FLOAT


def _pipeline(row, ttgir, ptx, ptx_rows) -> dict[str, object]:
    config = _config(row)
    loop = ptx["loops"][0]
    first, last = loop["first_line"], loop["last_line"]
    threads = row["num_warps"] * _WARP
    buffers = int(ttgir["local_alloc"][0].split("x")[0])
    tile = _tile_bytes(config)
    waits = [
        (first < n <= last, n)
        for n, op, _ in ptx_rows
        if op.startswith("cp.async.wait_group")
    ]
    return {
        "config": config["name"],
        "target": _COMMITTED[0],
        "num_stages": row["num_stages"],
        "buffers_per_operand": buffers,
        "buffers_are_stages_minus_one": buffers == row["num_stages"] - 1,
        "tile_bytes": tile,
        "ring_bytes": buffers * tile,
        "shared_bytes": row["shared_bytes"],
        "ring_fills_shared": buffers * tile == row["shared_bytes"],
        "commit_groups": ptx["families"]["cp.async.commit_group"],
        "commit_groups_in_k_loop": loop["families"]["cp.async.commit_group"],
        "prologue_commit_groups": ptx["families"]["cp.async.commit_group"]
        - loop["families"]["cp.async.commit_group"],
        "_waits": waits,
        "cp_async_per_thread_per_iteration": loop["families"]["cp.async"],
        "tile_elements_per_thread": tile // _FLOAT // threads,
        "threads": threads,
        "reqntid": ptx["reqntid"],
        "input_precision": "ieee",
        "mma_encoding": ttgir["mma_encoding"],
        "dot_parent_kinds": sorted(
            {item["parent_kind"] for item in ttgir["dot_operands"]}
        ),
        "mma_instructions": ptx["families"]["mma"],
        "fma_in_k_loop": loop["families"]["fma.rn.f32"],
        "fma_per_thread_per_iteration": config["block_m"]
        * config["block_n"]
        * config["block_k"]
        // threads,
        "fma_matches_tile": loop["families"]["fma.rn.f32"] * threads
        == config["block_m"] * config["block_n"] * config["block_k"],
        "ld_global": ptx["families"]["ld.global"],
    }


def _comparison(raw, ttgir_text, ptx) -> dict[str, object]:
    compiles = []
    for target in _TARGETS:
        for row in raw["targets"][target]:
            config = _config(row)
            tile = _tile_bytes(config)
            threads = row["num_warps"] * _WARP
            exact = row["shared_bytes"] % tile == 0
            ring = row["shared_bytes"] // tile
            compiles.append(
                {
                    "target": target,
                    "config": config["name"],
                    "block": [config["block_m"], config["block_n"], config["block_k"]],
                    "num_warps": row["num_warps"],
                    "num_stages": row["num_stages"],
                    "threads": threads,
                    "tile_bytes": tile,
                    "shared_bytes": row["shared_bytes"],
                    "ring_buffers": ring if exact else None,
                    "ring_is_stages_minus_one": exact and ring == row["num_stages"] - 1,
                    "lines": {s: row["stages"][s]["lines"] for s in _STAGES},
                    "cubin_bytes": row["cubin_bytes"],
                    "derived_fma_per_thread_per_iteration": config["block_m"]
                    * config["block_n"]
                    * config["block_k"]
                    // threads,
                    "committed_texts": (target, config["name"]) == _COMMITTED[:2],
                }
            )
    retargeted = sha256(
        ttgir_text.replace('"cuda:80"', '"cuda:90"').encode("ascii")
    ).hexdigest()
    targets = []
    for old, new in zip(raw["targets"]["sm_80"], raw["targets"]["sm_90"]):
        name = _config(old)["name"]
        same = {
            s: old["stages"][s]["sha256"] == new["stages"][s]["sha256"] for s in _STAGES
        }
        targets.append(
            {
                "config": name,
                "identical": {
                    **same,
                    "cubin": old["cubin_sha256"] == new["cubin_sha256"],
                },
                "line_change": {
                    s: new["stages"][s]["lines"] - old["stages"][s]["lines"]
                    for s in _STAGES
                },
                "shared_bytes_equal": old["shared_bytes"] == new["shared_bytes"],
                "cubin_bytes_change": new["cubin_bytes"] - old["cubin_bytes"],
                "ttgir_is_retargeted_sm_80": retargeted
                == new["stages"]["ttgir"]["sha256"]
                if name == _COMMITTED[1]
                else None,
            }
        )
    small = next(row for row in compiles if row["committed_texts"])
    loop = ptx["loops"][0]["families"]
    return {
        "compiles": compiles,
        "targets": targets,
        "committed": {
            "target": _COMMITTED[0],
            "config": _COMMITTED[1],
            "ptx_families": ptx["families"],
            "ptx_k_loop_families": loop,
            "derived_fma_confirmed": loop["fma.rn.f32"]
            == small["derived_fma_per_thread_per_iteration"],
        },
        "needs_full_dumps": list(NEEDS_FULL_DUMPS),
    }


def _regeneration(label: str, raw, actual: object) -> object:
    if label == RECORDED_LABEL:
        _require(actual is None, "a recorded report holds no regeneration")
        return None
    _require(isinstance(actual, dict), "a regeneration report needs its section")
    _require(
        set(actual) == {"backend", "environment", "compiles", "summary"},
        "regeneration fields do not match the schema",
    )
    backend = "triton_compiler" if label == REGENERATION_LABEL else "injected"
    environment = actual["environment"]
    _require(
        isinstance(environment, dict)
        and set(environment) == set(ENVIRONMENT_FIELDS)
        and all(type(value) is str for value in environment.values()),
        "regeneration environment does not match the schema",
    )
    rows = actual["compiles"]
    records = [(t, row) for t in _TARGETS for row in raw["targets"][t]]
    _require(
        isinstance(rows, list) and len(rows) == len(records),
        "regeneration must hold one row for each recorded compile",
    )
    expected_rows = []
    for (target, record), row in zip(records, rows):
        _require(isinstance(row, dict), "a regeneration row must be an object")
        stages = {}
        for stage in _STAGES:
            entry = row.get("stages", {}).get(stage, {})
            digest = entry.get("regenerated_sha256")
            lines = entry.get("regenerated_lines")
            _require(
                type(digest) is str and _HEX.fullmatch(digest) is not None,
                "a regenerated digest must be hex",
            )
            _require(type(lines) is int and lines >= 0, "regenerated lines")
            recorded = record["stages"][stage]
            stages[stage] = {
                "recorded_sha256": recorded["sha256"],
                "regenerated_sha256": digest,
                "recorded_lines": recorded["lines"],
                "regenerated_lines": lines,
                "match": digest == recorded["sha256"] and lines == recorded["lines"],
            }
        metadata = row.get("metadata", {})
        values = [metadata.get(k) for k in ("shared_bytes", "num_warps", "num_stages")]
        _require(all(type(v) is int for v in values), "regenerated metadata")
        cubin = row.get("cubin", {})
        digest, size = cubin.get("regenerated_sha256"), cubin.get("regenerated_bytes")
        _require(
            type(digest) is str
            and _HEX.fullmatch(digest) is not None
            and type(size) is int,
            "a regenerated cubin needs a digest and a size",
        )
        expected_rows.append(
            {
                "target": target,
                "config": _config(record)["name"],
                "stages": stages,
                "metadata": {
                    "shared_bytes": values[0],
                    "num_warps": values[1],
                    "num_stages": values[2],
                    "match": values
                    == [
                        record["shared_bytes"],
                        record["num_warps"],
                        record["num_stages"],
                    ],
                },
                "cubin": {
                    "recorded_sha256": record["cubin_sha256"],
                    "regenerated_sha256": digest,
                    "recorded_bytes": record["cubin_bytes"],
                    "regenerated_bytes": size,
                    "match": digest == record["cubin_sha256"],
                },
            }
        )
    texts = [s["match"] for r in expected_rows for s in r["stages"].values()]
    metadata = [r["metadata"]["match"] for r in expected_rows]
    cubins = {
        t: [r["cubin"]["match"] for r in expected_rows if r["target"] == t]
        for t in _TARGETS
    }
    return {
        "backend": backend,
        "environment": environment,
        "compiles": expected_rows,
        "summary": {
            "texts_matched": sum(texts),
            "texts": len(texts),
            "metadata_matched": sum(metadata),
            "compiles": len(metadata),
            "cubins_matched": sum(sum(v) for v in cubins.values()),
            "cubins_by_target": {t: sum(v) for t, v in cubins.items()},
            "all_texts_and_metadata_match": all(texts) and all(metadata),
        },
    }


def _host(report: Mapping[str, object]) -> None:
    _require(
        type(report["recorded_at"]) is str
        and _TIME.fullmatch(report["recorded_at"]) is not None,
        "recorded_at must be a UTC time",
    )
    load = report["host_load"]
    _require(
        isinstance(load, dict) and set(load) == {"before", "after"},
        "host_load must hold before and after",
    )
    for values in load.values():
        _require(
            isinstance(values, list)
            and len(values) == 3
            and all(
                type(value) is float and math.isfinite(value) and value >= 0
                for value in values
            ),
            "a load average must be three finite values",
        )


def _verify(report: Mapping[str, object], directory: Path) -> None:
    _require(report["schema_version"] == SCHEMA_VERSION, "schema_version mismatch")
    _require(report["kind"] == REPORT_KIND, "kind mismatch")
    label = report["label"]
    _require(label in LABELS, "label is not a Day 80 label")
    _compare(dict(BOUNDARIES[label]), report["evidence_boundary"], "evidence_boundary")
    body = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    _require(
        report[DIGEST_FIELD] == sha256(canonical_json_bytes(body)).hexdigest(),
        "report digest does not match its fields",
    )
    _host(report)
    payload = _read(directory / "index.json", 64 * 1024, "manifest")
    digest = sha256(payload).hexdigest()
    _require(digest == _MANIFEST_SHA256, "the fixture manifest is not the recorded one")
    raw = json.loads(payload)
    _compare(raw, report["manifest"], "manifest")
    _compare(digest, report["manifest_sha256"], "manifest_sha256")
    source = sha256(TRITON_SOURCE.encode("ascii")).hexdigest()
    _require(raw["source_sha256"] == source, "the dump compiled another kernel text")
    _compare({"kernel": "matmul_kernel", "sha256": source}, report["source"], "source")
    target, name, stages = _COMMITTED
    record = next(row for row in raw["targets"][target] if _config(row)["name"] == name)
    texts, fixtures = {}, []
    for stage in stages:
        entry = record["stages"][stage]
        data = _read(directory / entry["file"], 1024 * 1024, stage)
        _require(sha256(data).hexdigest() == entry["sha256"], f"{stage} digest")
        texts[stage] = data.decode("ascii")
        _require(len(texts[stage].splitlines()) == entry["lines"], f"{stage} lines")
        fixtures.append(
            {
                "target": target,
                "config": name,
                "stage": stage,
                "file": entry["file"],
                "sha256": entry["sha256"],
                "lines": entry["lines"],
                "bytes": len(data),
            }
        )
    _compare(fixtures, report["fixtures"], "fixtures")
    ptx, ptx_rows = _recount_ptx(texts["ptx"])
    _compare(ptx, report["ptx"], "ptx")
    _require(len(ptx["loops"]) == 1, "the committed PTX must hold one loop")
    ttir_rows = _recount_mlir(texts["ttir"])
    ttgir_rows = _recount_mlir(texts["ttgir"])
    _compare(_common(texts["ttir"], ttir_rows), report["ttir"], "ttir")
    ttgir = _ttgir(texts["ttgir"], ttgir_rows)
    _compare(ttgir, report["ttgir"], "ttgir")
    _compare(
        _stage_map(ttir_rows, ttgir_rows, ptx_rows, ptx["loops"][0]),
        report["stage_map"],
        "stage_map",
    )
    pipeline = _pipeline(record, ttgir, ptx, ptx_rows)
    waits = pipeline.pop("_waits")
    operands = {
        n: int(m)
        for n, m in (
            (i + 1, found.group(1))
            for i, line in enumerate(texts["ptx"].split("\n"))
            for found in [re.search(r"cp\.async\.wait_group\s+(\d+);", line)]
            if found
        )
    }
    pipeline["wait_group_in_k_loop"] = [operands[n] for inside, n in waits if inside]
    pipeline["wait_group_after_k_loop"] = [
        operands[n] for inside, n in waits if not inside
    ]
    _compare(pipeline, report["pipeline"], "pipeline")
    comparison = _comparison(raw, texts["ttgir"], ptx)
    _compare(comparison, report["comparison"], "comparison")
    regeneration = _regeneration(label, raw, report["regeneration"])
    _compare(regeneration, report["regeneration"], "regeneration")
    families = ptx["families"]
    summary = {
        "source_sha256": source,
        "fixtures": len(fixtures),
        "ptx_target": ptx["target"],
        "ptx_instructions": ptx["instructions"],
        "fma_rn_f32": families["fma.rn.f32"],
        "cp_async_all": sum(v for k, v in families.items() if k.startswith("cp.async")),
        "mma": families["mma"],
        "ld_global": families["ld.global"],
        "fma_in_k_loop": pipeline["fma_in_k_loop"],
        "ring_buffers": pipeline["buffers_per_operand"],
        "shared_bytes": pipeline["shared_bytes"],
        "ttir_ops": len(ttir_rows),
        "ttgir_ops": len(ttgir_rows),
        "compiles": len(comparison["compiles"]),
        "rings_of_stages_minus_one": sum(
            row["ring_is_stages_minus_one"] for row in comparison["compiles"]
        ),
        "ttir_identical_across_targets": sum(
            row["identical"]["ttir"] for row in comparison["targets"]
        ),
        "regeneration": None if regeneration is None else regeneration["summary"],
    }
    _compare(summary, report["summary"], "summary")


def verify_report(report: object, directory: Path) -> None:
    """Return only if every field of the report reproduces from the fixtures."""

    if not isinstance(directory, Path):
        raise TypeError("fixture folder must be pathlib.Path")
    _require(
        isinstance(report, Mapping) and set(report) == _REPORT_FIELDS,
        "report fields do not match the schema",
    )
    try:
        _verify(report, directory)
    except KernelLaunchVerificationError as error:
        raise TritonIrVerificationError(str(error)) from error
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
        if isinstance(error, TritonIrError):
            raise
        raise TritonIrVerificationError(
            f"report does not have the structure of the schema: {error!r}"
        ) from error


def load_report(path: Path, directory: Path) -> dict[str, object]:
    """Load canonical report bytes and verify every field against the fixtures."""

    payload = _read(path, MAX_REPORT_BYTES, "report")
    try:
        report = loads_strict(payload, label="report")
    except KernelLaunchError as error:
        raise TritonIrVerificationError(str(error)) from error
    _require(type(report) is dict, "report must be a JSON object")
    _require(
        payload == canonical_json_bytes(report) + b"\n",
        "report bytes are not canonical JSON",
    )
    verify_report(report, directory)
    return report
