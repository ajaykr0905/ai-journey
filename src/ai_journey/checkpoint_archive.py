"""Read bounded local tensor archives without following filesystem aliases."""

import os
from pathlib import Path
import stat
import zipfile

import torch


def load_checkpoint_archive(path: Path, *, max_bytes: int):
    """Bound archive expansion before invoking PyTorch weights-only loading."""
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be pathlib.Path")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("checkpoint max_bytes must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max_bytes:
            raise ValueError("checkpoint exceeds its regular-file byte budget")
        with zipfile.ZipFile(stream) as archive:
            records = archive.infolist()
            if (
                len(records) > 1024
                or len({record.filename for record in records}) != len(records)
                or sum(record.file_size for record in records) > max_bytes
                or any(record.compress_type != zipfile.ZIP_STORED for record in records)
            ):
                raise ValueError("checkpoint exceeds its expanded storage budget")
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=True)
