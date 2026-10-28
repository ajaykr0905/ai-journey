"""Read bounded local tensor archives without following filesystem aliases."""

from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re
import stat
import zipfile

import torch

MAX_CHECKPOINT_RECORDS = 16_384
MAX_CHECKPOINT_LOGICAL_BYTES = 128 * 1024 * 1024
MAX_CHECKPOINT_HASH_ITEMS = 100_000
MAX_CHECKPOINT_HASH_DEPTH = 64


def validate_checkpoint_tensor_work(value) -> None:
    """Bound logical tensors and container traversal before any materialization.

    Archive storage size does not bound expanded tensor views. Repeated aliases
    are charged independently because callers can process every occurrence.
    """
    items, logical_bytes = 0, 0

    def visit(item, depth):
        nonlocal items, logical_bytes
        items += 1
        if depth > MAX_CHECKPOINT_HASH_DEPTH:
            raise ValueError("checkpoint nesting exceeds its hash-work budget")
        if items > MAX_CHECKPOINT_HASH_ITEMS:
            raise ValueError("checkpoint items exceed their hash-work budget")
        if isinstance(item, torch.Tensor):
            logical_bytes += item.numel() * item.element_size()
            if logical_bytes > MAX_CHECKPOINT_LOGICAL_BYTES:
                raise ValueError("checkpoint logical tensor work exceeds its budget")
        elif isinstance(item, dict):
            for key, part in item.items():
                visit(key, depth + 1)
                visit(part, depth + 1)
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part, depth + 1)

    visit(value, 0)


def load_checkpoint_archive(
    path: Path, *, max_bytes: int, max_records: int = 1024, expected_sha256=None
):
    """Load one bounded immutable snapshot after ZIP and optional digest admission.

    Expert architectures may explicitly admit more small tensor records; the
    legacy 1024-record default and all other archive safeguards remain intact.
    """
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be pathlib.Path")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("checkpoint max_bytes must be a positive integer")
    if type(max_records) is not int or not 1 <= max_records <= MAX_CHECKPOINT_RECORDS:
        raise ValueError("checkpoint max_records must be an integer in [1, 16384]")
    if expected_sha256 is not None and (
        type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("checkpoint expected_sha256 must be a lowercase SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max_bytes:
            raise ValueError("checkpoint exceeds its regular-file byte budget")
        # Both ZIP admission and deserialization must see these same bytes, even
        # when the caller has no externally supplied archive identity.
        snapshot = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
        if (
            len(snapshot) != info.st_size
            or after.st_size != info.st_size
            or after.st_mtime_ns != info.st_mtime_ns
        ):
            raise ValueError("checkpoint snapshot changed during its bounded read")
        if (
            expected_sha256 is not None
            and sha256(snapshot).hexdigest() != expected_sha256
        ):
            raise ValueError("checkpoint snapshot differs from its SHA-256")
        admitted = BytesIO(snapshot)
        with zipfile.ZipFile(admitted) as archive:
            records = archive.infolist()
            if (
                len(records) > max_records
                or len({record.filename for record in records}) != len(records)
                or sum(record.file_size for record in records) > max_bytes
                or any(record.compress_type != zipfile.ZIP_STORED for record in records)
            ):
                raise ValueError("checkpoint exceeds its expanded storage budget")
        admitted.seek(0)
        payload = torch.load(admitted, map_location="cpu", weights_only=True)
        validate_checkpoint_tensor_work(payload)
        return payload
