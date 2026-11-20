from dataclasses import replace

import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.matmul_metal_source import source_sha256
from ai_journey.matmul_tiled_source import (
    HEADER,
    MatmulTiledError,
    tiled_kernel_source,
)
from ai_journey.matmul_tiled_variants import (
    FEWER_THREADS,
    K_ORDER_FMA,
    OTHER_THREADGROUP_BYTES,
    REORDERED_SUM,
    VARIANTS,
    TiledVariant,
    VariantAdmission,
    library_source,
    text_sha256,
    variants_named,
)

# The Day 74 naive source; recorded reports and later labs pin this digest.
DAY_74_SOURCE_SHA256 = (
    "beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc"
)
# Each Day 76 kernel's own digest. A later variant must leave all four alone.
VARIANT_SHA256 = {
    "matmul_tiled_8": "21b53bf8ce1c123d1755b12512dedf1dfa1a5cc2509d476ac1019538a924e1ba",
    "matmul_tiled_16": "9e79d70317e309e13e043d0c015574670bfe09f0d28cf23abc4300f98a2eeb13",
    "matmul_tiled_32": "f5ae7c6f03bcd1772c6eec5aed127916cb962b70bc7f99179d9f115a589ced30",
}
LIBRARY_SHA256 = "08b88577018024d54344a00f991f387225f1085047caac66f347dc5e719a5fa5"
DAY_76 = ("matmul_tiled_8", "matmul_tiled_16", "matmul_tiled_32")


def limits(threads=1024, width=32, memory=0):
    return PipelineLimits(
        max_threads_per_threadgroup=threads,
        thread_execution_width=width,
        static_threadgroup_memory_bytes=memory,
    )


def test_the_registry_pins_every_day_76_digest_and_leaves_day_74_alone():
    assert tuple(VARIANTS) == DAY_76
    for name, variant in VARIANTS.items():
        assert (
            variant.source_sha256 == VARIANT_SHA256[name] == text_sha256(variant.source)
        )
        assert variant.correctness_mode == K_ORDER_FMA
        assert variant.source == tiled_kernel_source(variant.tile)
        assert variant.group_size == (variant.tile, variant.tile)
        assert variant.arg_casts == {3: "int32", 4: "int32", 5: "int32"}
    assert text_sha256(library_source(variants_named(DAY_76))) == LIBRARY_SHA256
    assert source_sha256() == DAY_74_SOURCE_SHA256


def test_a_new_variant_moves_no_day_76_digest():
    extra = TiledVariant(
        name="matmul_tiled_4_probe",
        tile=4,
        source=tiled_kernel_source(4).replace("matmul_tiled_4", "matmul_tiled_4_probe"),
        correctness_mode=REORDERED_SUM,
    )
    joined = library_source((*variants_named(DAY_76), extra))

    assert joined.startswith(library_source(variants_named(DAY_76)))
    assert text_sha256(library_source(variants_named(DAY_76))) == LIBRARY_SHA256
    assert {name: VARIANTS[name].source_sha256 for name in DAY_76} == VARIANT_SHA256
    assert extra.source_sha256 not in VARIANT_SHA256.values()


def test_a_library_is_the_header_and_its_kernels_in_config_order():
    chosen = variants_named(("matmul_tiled_32", "matmul_tiled_8"))
    text = library_source(chosen)

    assert text == (
        HEADER + "\n" + tiled_kernel_source(32) + "\n" + tiled_kernel_source(8)
    )
    assert text.index("matmul_tiled_32(") < text.index("matmul_tiled_8(")
    assert text_sha256(text) != LIBRARY_SHA256


def test_the_report_mapping_names_the_bindings_and_digest_without_the_text():
    mapping = VARIANTS["matmul_tiled_16"].to_mapping()

    assert mapping == {
        "name": "matmul_tiled_16",
        "tile": 16,
        "group_size": [16, 16],
        "threads_per_group": 256,
        "threadgroup_bytes": 2048,
        "correctness_mode": K_ORDER_FMA,
        "buffers": ["c", "a", "b", "m", "n", "k"],
        "scalar_buffers": [3, 4, 5],
        "source_sha256": VARIANT_SHA256["matmul_tiled_16"],
    }


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ((), "1 to 8 variants"),
        (("matmul_tiled_8", "matmul_tiled_8"), "twice"),
        (("matmul_tiled_64",), "not a registered variant"),
        ((["matmul_tiled_8"],), "must be strings"),
    ],
)
def test_a_run_names_registered_variants_once_each(names, message):
    with pytest.raises(MatmulTiledError, match=message):
        variants_named(names)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "tiled_8"}, "variant name"),
        ({"name": "matmul_tiled_9"}, "source that defines it"),
        ({"source": "kernel void matmul_tiled_8(é"}, "ASCII"),
        ({"correctness_mode": "bit_exact"}, "correctness_mode"),
        ({"tile": 64}, "tile must be in"),
    ],
)
def test_a_variant_must_declare_a_known_mode_and_its_own_kernel(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(VARIANTS["matmul_tiled_8"], **changes)


def test_a_pipeline_admits_a_tile_only_with_its_threads_and_its_static_bytes():
    variant = VARIANTS["matmul_tiled_32"]

    admitted = VariantAdmission(variant, limits(memory=8192))
    too_few = VariantAdmission(variant, limits(threads=896, memory=8192))
    other_bytes = VariantAdmission(variant, limits(memory=8208))

    assert (admitted.admitted, admitted.reason) == (True, None)
    assert (too_few.admitted, too_few.reason) == (False, FEWER_THREADS)
    assert (other_bytes.admitted, other_bytes.reason) == (
        False,
        OTHER_THREADGROUP_BYTES,
    )
    assert too_few.to_mapping() == {
        "name": "matmul_tiled_32",
        "admitted": False,
        "reason": FEWER_THREADS,
        "required_threads": 1024,
        "allowed_threads": 896,
        "declared_threadgroup_bytes": 8192,
        "static_threadgroup_bytes": 8192,
    }
    with pytest.raises(TypeError, match="PipelineLimits"):
        VariantAdmission(variant, limits().to_mapping())
