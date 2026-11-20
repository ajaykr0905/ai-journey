from dataclasses import replace

import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_variants import (
    DAY_74,
    DAY_76,
    DAY_77,
    DESIGNS,
    LIBRARY_VARIANTS,
    NAIVE_VARIANTS,
    VARIANTS,
    ChecklistAdmission,
    admitted_names,
    checklist_library,
    guard_elements,
    require_known,
    variant_launch,
)
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_source import SHADER_SOURCE
from ai_journey.matmul_tiled_source import HEADER, tiled_kernel_source
from ai_journey.matmul_tiled_variants import (
    FEWER_THREADS,
    OTHER_THREADGROUP_BYTES,
    text_sha256,
)
from ai_journey.matmul_tiled_variants import VARIANTS as TILED_VARIANTS
from test_matmul_tiled_variants import DAY_74_SOURCE_SHA256, VARIANT_SHA256

# Each kernel's own digest; a later variant must leave every one alone.
DAY_77_SHA256 = {
    "matmul_s2_coarse_4x1": (
        "e2520d4f573c267360eb3833306bad7fbfe88325a4060b72b051f59bf54c144b"
    ),
    "matmul_s3_block_4x4": (
        "d4530e74d254196749d27543ccf78238c0edaf5b551c28555095f5e77a23bc9d"
    ),
    "matmul_s4_vec4": (
        "6df4d1063583f53372a700e08fb895858a42371f2e3d0f79bb48bd13fe4eac05"
    ),
    "matmul_s5_pad4": (
        "8567cddd838fcc4aa82000c0f434864b5d4bd1c494e21d5937f787400e784028"
    ),
    "matmul_s6_group_8x8": (
        "919930b29ccdd65de0b096cc22c1c5b0683e9f9c03c0791469098039511ae2bc"
    ),
    "matmul_s6_group_32x32": (
        "fc609adecff160fd1e17f2ad85ba7b85785bf48cd040b5ef0a281fa4f090dcc0"
    ),
    "matmul_s7_interior": (
        "b5053c2f0c52043d9c93ce8359a491878fd7e8ef44f2c85b6e901019841225ab"
    ),
}
LIBRARY_SHA256 = "d0a8f46a8831029a8894edca48da9b8d61646571c40c36d2f502d6f30cea027d"
RAGGED = matmul_shape(2000, 3000, 1000)


def limits(threads=1024, memory=0):
    return PipelineLimits(
        max_threads_per_threadgroup=threads,
        thread_execution_width=32,
        static_threadgroup_memory_bytes=memory,
    )


def test_the_registry_pins_every_digest_and_moves_no_earlier_one():
    assert list(VARIANTS) == [
        "matmul_x_columns",
        "matmul_x_rows",
        "matmul_tiled_32",
        *DAY_77_SHA256,
    ]
    for name in NAIVE_VARIANTS:
        assert VARIANTS[name].origin == DAY_74
        assert VARIANTS[name].source == SHADER_SOURCE
        assert VARIANTS[name].source_sha256 == DAY_74_SOURCE_SHA256
    tiled = VARIANTS["matmul_tiled_32"]
    assert tiled.origin == DAY_76 and tiled.source == tiled_kernel_source(32)
    assert tiled.source_sha256 == VARIANT_SHA256["matmul_tiled_32"]
    for name, variant in TILED_VARIANTS.items():
        assert variant.source_sha256 == VARIANT_SHA256[name]
    for design in DESIGNS:
        variant = VARIANTS[design.name]
        assert variant.origin == DAY_77 and variant.design == design
        assert variant.source_sha256 == DAY_77_SHA256[design.name]
        assert variant.source_sha256 == text_sha256(variant.source)
    for variant in VARIANTS.values():
        assert variant.correctness_mode == "k_order_fma"
        assert variant.arg_casts == {3: "int32", 4: "int32", 5: "int32"}


def test_the_library_holds_the_day_76_text_and_then_the_day_77_kernels():
    library = checklist_library()

    assert LIBRARY_VARIANTS == ("matmul_tiled_32", *DAY_77_SHA256)
    assert library == HEADER + "".join(
        "\n" + VARIANTS[name].source for name in LIBRARY_VARIANTS
    )
    assert "\n" + tiled_kernel_source(32) in library
    assert "matmul_x_columns" not in library and "matmul_x_rows" not in library
    assert text_sha256(library) == LIBRARY_SHA256


def test_every_report_mapping_names_the_launch_shape_and_memory():
    assert VARIANTS["matmul_x_rows"].to_mapping() == {
        "name": "matmul_x_rows",
        "origin": DAY_74,
        "mapping": "x_rows",
        "outputs_per_thread": [1, 1],
        "group_size": None,
        "threadgroup_bytes": 0,
        "correctness_mode": "k_order_fma",
        "design": None,
        "source_sha256": DAY_74_SOURCE_SHA256,
    }
    assert VARIANTS["matmul_s7_interior"].to_mapping()["design"] == {
        "block": [64, 64, 16],
        "outputs_per_thread": [4, 4],
        "vector_loads": True,
        "pad": 4,
        "interior_path": True,
    }
    assert [VARIANTS[design.name].group_size for design in DESIGNS] == [
        (32, 8),
        (16, 16),
        (16, 16),
        (16, 16),
        (8, 8),
        (32, 32),
        (16, 16),
    ]


@pytest.mark.parametrize("name", list(VARIANTS))
def test_a_launch_covers_whole_groups_and_the_tail_holds_every_spare_slot(name):
    variant = VARIANTS[name]
    launch = variant_launch(variant, RAGGED, (32, 8))
    rows, columns = variant.outputs_per_thread
    x, y = launch.threads
    guard = guard_elements(variant, RAGGED, launch)

    assert launch.group_size == variant.group((32, 8))
    if name == "matmul_x_rows":
        assert launch.extent == (2000, 3000)
    else:
        assert launch.extent == (-(-3000 // columns), -(-2000 // rows))
        assert x * columns >= 3000 and y * rows >= 2000
    assert guard == x * y * rows * columns - 2000 * 3000 >= 0
    # The farthest store that escapes its guard still lands in the buffer.
    if name != "matmul_x_rows":
        assert (y * rows - 1) * 3000 + x * columns - 1 < 2000 * 3000 + guard


def test_the_ragged_shape_has_78_edge_groups_of_64_x_64():
    launch = variant_launch(VARIANTS["matmul_s7_interior"], RAGGED, (32, 8))
    groups_x, groups_y = launch.groups
    interior = (3000 // 64) * (2000 // 64)

    assert (groups_x, groups_y) == (47, 32)
    assert groups_x * groups_y - interior == 78


def test_each_pipeline_verdict_follows_its_limits_and_day_74_is_not_admitted():
    s6 = VARIANTS["matmul_s6_group_32x32"]
    admitted = ChecklistAdmission(s6, limits(1024, 18688))
    fewer = ChecklistAdmission(s6, limits(896, 18688))
    other = ChecklistAdmission(s6, limits(1024, 18704))

    assert (admitted.reason, fewer.reason, other.reason) == (
        None,
        FEWER_THREADS,
        OTHER_THREADGROUP_BYTES,
    )
    assert fewer.to_mapping() == {
        "name": "matmul_s6_group_32x32",
        "admitted": False,
        "reason": FEWER_THREADS,
        "required_threads": 1024,
        "allowed_threads": 896,
        "declared_threadgroup_bytes": 18688,
        "static_threadgroup_bytes": 18688,
    }
    with pytest.raises(MatmulChecklistError, match="not admitted by tile"):
        ChecklistAdmission(VARIANTS["matmul_x_columns"], limits())
    admissions = {
        name: ChecklistAdmission(
            VARIANTS[name], limits(memory=VARIANTS[name].threadgroup_bytes)
        )
        for name in LIBRARY_VARIANTS
    }
    admissions["matmul_s6_group_32x32"] = fewer
    assert admitted_names(admissions) == tuple(
        name for name in VARIANTS if name != "matmul_s6_group_32x32"
    )


def test_a_variant_record_refuses_mixed_origins_and_unknown_names():
    tiled = VARIANTS["matmul_tiled_32"]

    with pytest.raises(MatmulChecklistError, match="origin must be one of"):
        replace(tiled, origin="day_75")
    with pytest.raises(MatmulChecklistError, match="naive group"):
        replace(tiled, group_size=None)
    with pytest.raises(MatmulChecklistError, match="have a design"):
        replace(tiled, design=DESIGNS[0])
    with pytest.raises(MatmulChecklistError, match="source that defines it"):
        replace(tiled, source=SHADER_SOURCE)
    with pytest.raises(TypeError, match="outputs_per_thread must be an"):
        replace(tiled, outputs_per_thread=[1, 1])
    assert require_known(["matmul_x_rows", "matmul_s5_pad4"]) == (
        "matmul_x_rows",
        "matmul_s5_pad4",
    )
    with pytest.raises(MatmulChecklistError, match="not a registered variant"):
        require_known(["matmul_tiled_16"])
    with pytest.raises(MatmulChecklistError, match="only once"):
        require_known(["matmul_x_rows", "matmul_x_rows"])
