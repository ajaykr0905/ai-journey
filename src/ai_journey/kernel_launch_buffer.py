"""Instrumented device buffers that record misuse instead of corrupting memory.

A ``DeviceBuffer`` is a flat typed array that a simulated kernel indexes like
a CUDA pointer. During a launch an ``AccessRecorder`` attributes each access
to the thread that makes it. An access outside the buffer is recorded with
the buffer name, the index, the block and the thread. Such a read returns
zero and such a write is dropped, so no memory outside the buffer exists to
corrupt. The recorder also finds elements that two threads wrote (write
conflicts) and elements that one thread wrote and another thread read
(read-write conflicts). CUDA gives the threads of one launch no order, so
both are data races, even when the two threads store the same value.
"""

from __future__ import annotations

import itertools
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from .kernel_launch_config import Index3, KernelLaunchError, LaunchConfig, require_int

DTYPES = ("float32", "uint8")
MAX_BUFFER_ELEMENTS = 1 << 22
READ = "read"
WRITE = "write"
_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")
_ZERO = {"float32": 0.0, "uint8": 0}


def to_float32(value: float) -> float:
    """Round a finite float to the nearest float32 value, ties to even.

    A float32 buffer stores only float32 values, so a kernel rounds each
    result through this function, as a CUDA ``float`` store does.
    """

    if type(value) is not float or not math.isfinite(value):
        raise KernelLaunchError(f"float32 rounding needs a finite float, not {value!r}")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise KernelLaunchError(f"{value!r} is outside the float32 range") from error


def _check_value(dtype: str, value: object, name: str) -> int | float:
    if dtype == "uint8":
        if type(value) is not int or not 0 <= value <= 255:
            raise KernelLaunchError(f"uint8 buffer {name} cannot store {value!r}")
        return value
    exact = type(value) is float and math.isfinite(value)
    if exact:
        try:
            exact = to_float32(value) == value
        except KernelLaunchError:
            exact = False
    if not exact:
        raise KernelLaunchError(
            f"float32 buffer {name} cannot store {value!r}; round it with to_float32"
        )
    return value


def _require_dtype(dtype: object) -> str:
    if type(dtype) is not str or dtype not in DTYPES:
        raise KernelLaunchError(f"dtype must be one of {DTYPES}")
    return dtype


def encode_values(dtype: str, values: Sequence[int | float]) -> bytes:
    """Return the little-endian bytes that a device buffer of ``dtype`` holds."""

    _require_dtype(dtype)
    checked = [_check_value(dtype, value, "values") for value in values]
    if dtype == "uint8":
        return bytes(checked)
    return struct.pack(f"<{len(checked)}f", *checked)


def values_sha256(dtype: str, values: Sequence[int | float]) -> str:
    """Digest typed values as device-buffer bytes: equal digests mean equal bits."""

    return sha256(encode_values(dtype, values)).hexdigest()


class DeviceBuffer:
    """A flat typed device array that a simulated kernel indexes like a pointer.

    Host code fills the buffer when it creates it and copies it back with
    ``to_host``. Kernel code may index it only while a launch's recorder is
    attached and one of its threads runs. A pointer has no length, so the
    buffer has no ``len`` and refuses iteration.
    """

    __slots__ = ("_name", "_dtype", "_values", "_recorder")
    __iter__ = None

    def __init__(self, name: str, dtype: str, values: Iterable[int | float]) -> None:
        if type(name) is not str or _NAME.fullmatch(name) is None:
            raise KernelLaunchError(
                "buffer name must be a lowercase identifier of at most 32 characters"
            )
        _require_dtype(dtype)
        stored = [
            _check_value(dtype, value, name)
            for value in itertools.islice(values, MAX_BUFFER_ELEMENTS + 1)
        ]
        if not 1 <= len(stored) <= MAX_BUFFER_ELEMENTS:
            raise KernelLaunchError(
                f"buffer {name} must hold 1 to {MAX_BUFFER_ELEMENTS} elements"
            )
        self._name = name
        self._dtype = dtype
        self._values = stored
        self._recorder: AccessRecorder | None = None

    @classmethod
    def zeros(cls, name: str, dtype: str, size: int) -> DeviceBuffer:
        """Allocate a buffer whose elements are all zero."""

        require_int(size, "size", minimum=1, maximum=MAX_BUFFER_ELEMENTS)
        return cls(name, dtype, itertools.repeat(_ZERO[_require_dtype(dtype)], size))

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def size(self) -> int:
        return len(self._values)

    def _host_access(self) -> None:
        if self._recorder is not None:
            raise KernelLaunchError(
                f"buffer {self._name} is in a running launch; host copies wait"
            )

    def to_host(self) -> tuple[int | float, ...]:
        """Copy the contents back to the host after a launch."""

        self._host_access()
        return tuple(self._values)

    def sha256(self) -> str:
        """Digest the buffer bytes, the same way ``values_sha256`` does."""

        self._host_access()
        return values_sha256(self._dtype, self._values)

    def _recorder_for(self, index: object) -> AccessRecorder:
        recorder = self._recorder
        if recorder is None:
            raise KernelLaunchError(
                f"buffer {self._name} is indexed outside a kernel launch"
            )
        if type(index) is not int:
            raise TypeError(f"buffer {self._name} index must be an integer")
        return recorder

    def __getitem__(self, index: int) -> int | float:
        # Reads dominate a kernel's buffer traffic, so an in-bounds read is
        # recorded here with no helper calls. The checks and the records are
        # those of _recorder_for and AccessRecorder._running, in that order.
        recorder = self._recorder
        if recorder is None:
            raise KernelLaunchError(
                f"buffer {self._name} is indexed outside a kernel launch"
            )
        if type(index) is not int:
            raise TypeError(f"buffer {self._name} index must be an integer")
        thread = recorder._thread
        if thread is None:
            raise KernelLaunchError("buffers are indexed only by a running thread")
        values = self._values
        if 0 <= index < len(values):
            recorder._reads += 1
            table = recorder._readers[self._name]
            readers = table.get(index)
            if readers is None:
                table[index] = {thread}
            else:
                readers.add(thread)
            return values[index]
        recorder._out_of_bounds.append((thread, self._name, READ, index))
        return _ZERO[self._dtype]

    def __setitem__(self, index: int, value: int | float) -> None:
        recorder = self._recorder_for(index)
        stored = _check_value(self._dtype, value, self._name)
        if 0 <= index < len(self._values):
            recorder._record_write(self._name, index)
            self._values[index] = stored
        else:
            recorder._record_out_of_bounds(self._name, WRITE, index)

    def __repr__(self) -> str:
        return (
            f"DeviceBuffer(name={self._name!r}, dtype={self._dtype!r}, "
            f"size={len(self._values)})"
        )


@dataclass(frozen=True)
class ThreadRef:
    """One simulated thread, named by its ``blockIdx`` and ``threadIdx``."""

    block: Index3
    thread: Index3

    def to_mapping(self) -> dict[str, list[int]]:
        return {"block": self.block.to_list(), "thread": self.thread.to_list()}


@dataclass(frozen=True)
class OutOfBoundsAccess:
    """One read or write outside a buffer, with the thread that made it."""

    buffer: str
    access: str
    index: int
    block: Index3
    thread: Index3

    def to_mapping(self) -> dict[str, object]:
        return {
            "buffer": self.buffer,
            "access": self.access,
            "index": self.index,
            "block": self.block.to_list(),
            "thread": self.thread.to_list(),
        }


@dataclass(frozen=True)
class WriteConflict:
    """One element that two or more threads of a launch wrote."""

    buffer: str
    index: int
    writers: tuple[ThreadRef, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "buffer": self.buffer,
            "index": self.index,
            "writers": [writer.to_mapping() for writer in self.writers],
        }


@dataclass(frozen=True)
class ReadWriteConflict:
    """One element that a thread wrote while a different thread read it."""

    buffer: str
    index: int
    writers: tuple[ThreadRef, ...]
    readers: tuple[ThreadRef, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "buffer": self.buffer,
            "index": self.index,
            "writers": [writer.to_mapping() for writer in self.writers],
            "readers": [reader.to_mapping() for reader in self.readers],
        }


@dataclass(frozen=True)
class LaunchAccessLog:
    """The in-bounds traffic, out-of-bounds accesses and races of one launch.

    ``out_of_bounds`` keeps execution order. ``canonical_out_of_bounds`` sorts
    the same records by linear thread ID and keeps each thread's program
    order, so it does not depend on the launch order. Conflicts are sorted
    by buffer name and index, and list their threads by linear thread ID.
    """

    reads: int
    writes: int
    out_of_bounds: tuple[OutOfBoundsAccess, ...]
    canonical_out_of_bounds: tuple[OutOfBoundsAccess, ...]
    write_conflicts: tuple[WriteConflict, ...]
    read_write_conflicts: tuple[ReadWriteConflict, ...]
    writer_threads: Mapping[str, int]

    @property
    def out_of_bounds_reads(self) -> int:
        return sum(record.access == READ for record in self.out_of_bounds)

    @property
    def out_of_bounds_writes(self) -> int:
        return sum(record.access == WRITE for record in self.out_of_bounds)

    @property
    def memory_safe(self) -> bool:
        return not self.out_of_bounds

    @property
    def race_free(self) -> bool:
        return not self.write_conflicts and not self.read_write_conflicts

    def counts(self) -> dict[str, int]:
        """Return the access and misuse counts that a run report states."""

        return {
            "reads": self.reads,
            "writes": self.writes,
            "out_of_bounds_reads": self.out_of_bounds_reads,
            "out_of_bounds_writes": self.out_of_bounds_writes,
            "write_conflicts": len(self.write_conflicts),
            "read_write_conflicts": len(self.read_write_conflicts),
        }


class AccessRecorder:
    """Attribute each buffer access of one launch to the running thread.

    The recorder attaches itself to every buffer of the launch. The simulator
    calls ``enter`` and ``leave`` around each thread and ``finish`` once at
    the end, which detaches the buffers and returns the launch's log.
    """

    def __init__(self, config: LaunchConfig, buffers: Sequence[DeviceBuffer]) -> None:
        if type(config) is not LaunchConfig:
            raise TypeError("config must be LaunchConfig")
        unique: dict[int, DeviceBuffer] = {}
        for buffer in buffers:
            if type(buffer) is not DeviceBuffer:
                raise TypeError("buffers must be DeviceBuffer values")
            unique.setdefault(id(buffer), buffer)
        names = [buffer.name for buffer in unique.values()]
        if len(set(names)) != len(names):
            raise KernelLaunchError("the buffers of one launch need distinct names")
        for buffer in unique.values():
            if buffer._recorder is not None:
                raise KernelLaunchError(f"buffer {buffer.name} is in a running launch")
        self._config = config
        self._buffers = tuple(unique.values())
        self._thread: int | None = None
        self._closed = False
        self._reads = 0
        self._writes = 0
        self._writers: dict[str, dict[int, list[int]]] = {name: {} for name in names}
        self._readers: dict[str, dict[int, set[int]]] = {name: {} for name in names}
        self._out_of_bounds: list[tuple[int, str, str, int]] = []
        for buffer in self._buffers:
            buffer._recorder = self

    def enter(self, thread_id: int) -> None:
        """Mark the thread with this linear ID as the one that runs now."""

        if self._closed:
            raise KernelLaunchError("the launch recorder is finished")
        if self._thread is not None:
            raise KernelLaunchError("another simulated thread is still running")
        self._thread = require_int(
            thread_id,
            "thread_id",
            minimum=0,
            maximum=self._config.launched_threads - 1,
        )

    def leave(self) -> None:
        """Mark that no simulated thread runs."""

        self._thread = None

    def _running(self) -> int:
        if self._thread is None:
            raise KernelLaunchError("buffers are indexed only by a running thread")
        return self._thread

    def _record_write(self, name: str, index: int) -> None:
        thread = self._running()
        self._writes += 1
        writers = self._writers[name].get(index)
        if writers is None:
            self._writers[name][index] = [thread]
        elif thread not in writers:
            writers.append(thread)

    def _record_out_of_bounds(self, name: str, access: str, index: int) -> None:
        self._out_of_bounds.append((self._running(), name, access, index))

    def _ref(self, thread_id: int) -> ThreadRef:
        return ThreadRef(*self._config.coordinates(thread_id))

    def finish(self) -> LaunchAccessLog:
        """Detach every buffer and return what the launch did to memory."""

        if self._closed:
            raise KernelLaunchError("the launch recorder is finished")
        self._closed = True
        self._thread = None
        for buffer in self._buffers:
            buffer._recorder = None
        records = [
            (
                thread,
                OutOfBoundsAccess(
                    name, access, index, *self._config.coordinates(thread)
                ),
            )
            for thread, name, access, index in self._out_of_bounds
        ]
        write_conflicts = []
        read_write_conflicts = []
        writer_threads = {}
        for name in sorted(self._writers):
            writers = self._writers[name]
            readers = self._readers[name]
            writer_threads[name] = len(
                {thread for threads in writers.values() for thread in threads}
            )
            for index in sorted(writers):
                ordered = sorted(writers[index])
                writing = tuple(self._ref(thread) for thread in ordered)
                if len(ordered) > 1:
                    write_conflicts.append(WriteConflict(name, index, writing))
                reading = readers.get(index)
                if reading and (len(ordered) > 1 or reading != {ordered[0]}):
                    read_write_conflicts.append(
                        ReadWriteConflict(
                            name,
                            index,
                            writing,
                            tuple(self._ref(thread) for thread in sorted(reading)),
                        )
                    )
        return LaunchAccessLog(
            reads=self._reads,
            writes=self._writes,
            out_of_bounds=tuple(record for _, record in records),
            canonical_out_of_bounds=tuple(
                record for _, record in sorted(records, key=lambda item: item[0])
            ),
            write_conflicts=tuple(write_conflicts),
            read_write_conflicts=tuple(read_write_conflicts),
            writer_threads=MappingProxyType(writer_threads),
        )
