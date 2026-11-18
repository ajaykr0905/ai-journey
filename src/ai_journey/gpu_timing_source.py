"""Metal source of a near-empty kernel that measures what a launch costs alone.

The kernel does no work, but the compiler must not be able to prove that. It
reads its thread position and stores to its buffer only when that position
equals a value that no launch here reaches, so every thread runs one compare
and returns. The guard depends on ``thread_position_in_grid``, which the
compiler cannot know, so the store and the buffer binding stay in the
pipeline. A one-thread launch of this kernel gives the floor that every
launch through ``torch.mps.compile_shader`` pays on this host.

It lives in its own small library, so the Day 74 matmul source and its
digest stay unchanged, and the timing report can quote both digests.
"""

from __future__ import annotations

from hashlib import sha256

from .kernel_metal_geometry import LaunchGeometry, vector_launch

EMPTY_KERNEL = "empty_kernel"
# A thread position this lab never launches: the empty launch has 1 thread.
NEVER_POSITION = 0xFFFFFFFF
EMPTY_BUFFERS = ("sink",)
EMPTY_SOURCE = (
    "#include <metal_stdlib>\nusing namespace metal;\n\n"
    "// One compare per thread and no store in any launch of this lab: the\n"
    "// guard depends on the thread position, so the compiler keeps the store\n"
    "// and the buffer, and the launch keeps its full dispatch path.\n"
    f"kernel void {EMPTY_KERNEL}(device uint* sink [[buffer(0)]],\n"
    "                         uint position [[thread_position_in_grid]]) {\n"
    f"    if (position == 0x{NEVER_POSITION:X}u) {{\n"
    "        sink[0] = position;\n"
    "    }\n"
    "}\n"
)


def empty_launch() -> LaunchGeometry:
    """Return the one-thread, one-group launch of the empty kernel."""

    return vector_launch(1, 1)


def empty_source_sha256() -> str:
    """Hash the exact source text that ``compile_shader`` receives."""

    return sha256(EMPTY_SOURCE.encode("ascii")).hexdigest()


def empty_kernel_mapping() -> dict[str, object]:
    """Return the kernel's bindings and launch as a report records them."""

    return {
        "name": EMPTY_KERNEL,
        "buffers": list(EMPTY_BUFFERS),
        "never_position": NEVER_POSITION,
        "launch": empty_launch().to_mapping(),
    }
