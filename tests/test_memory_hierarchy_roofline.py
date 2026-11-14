import pytest

from ai_journey.memory_hierarchy_matmul import MatmulMeasurement
from ai_journey.memory_hierarchy_roofline import measured_roofline, stream_workload
from ai_journey.memory_hierarchy_stream import MemoryHierarchyError
from ai_journey.memory_hierarchy_sweep import StreamMeasurement
from ai_journey.memory_hierarchy_timing import BatchTiming

NS = 1_000_000_000


def timing(*samples):
    return BatchTiming(1, 0, len(samples), samples, "injected")


def stream(dtype="float64"):
    medians = {
        1024: {"copy": 90, "scale": 100, "add": 150, "triad": 205},
        4096: {"copy": 900, "scale": 950, "add": 1300, "triad": 1640},
    }
    return tuple(
        StreamMeasurement(
            kernel, elements, dtype, timing(median - 5, median, median + 9)
        )
        for elements, row in medians.items()
        for kernel, median in row.items()
    )


def matmul(dtype="float64", median=2000):
    return MatmulMeasurement(64, 64, 64, dtype, timing(1900, median, 2100), 0.01)


def test_ceilings_come_from_the_matmul_and_triad_medians():
    section = measured_roofline(stream(), matmul())
    peak = 2 * 64**3 * NS / 2000
    largest = 5 * 4096 * 8 * NS / 1640
    in_cache = 5 * 1024 * 8 * NS / 205

    assert section["ceiling_statistic"] == "median"
    assert section["compute_ceiling"] == {
        "source": "matmul",
        "dtype": "float64",
        "m": 64,
        "n": 64,
        "k": 64,
        "flops_per_second": peak,
    }
    assert section["bandwidth_ceilings"] == [
        {
            "name": "largest_working_set",
            "kernel": "triad",
            "elements": 4096,
            "working_set_bytes": 3 * 4096 * 8,
            "bytes_per_second": largest,
            "ridge_point_flops_per_byte": peak / largest,
        },
        {
            "name": "peak_sweep_point",
            "kernel": "triad",
            "elements": 1024,
            "working_set_bytes": 3 * 1024 * 8,
            "bytes_per_second": in_cache,
            "ridge_point_flops_per_byte": peak / in_cache,
        },
    ]
    assert section["ridge_point_flops_per_byte"] == peak / largest
    assert section["placement_elements"] == 4096


def test_kernels_with_flops_are_placed_at_their_arithmetic_intensity():
    section = measured_roofline(stream(), matmul())
    placements = {item["kind"]: item for item in section["placements"]}
    peak = 2 * 64**3 * NS / 2000
    bandwidth = 5 * 4096 * 8 * NS / 1640

    assert list(placements) == ["scale", "add", "triad", "matmul"]
    assert [item["source"] for item in section["placements"]] == [
        "stream",
        "stream",
        "stream",
        "matmul",
    ]
    scale = placements["scale"]
    assert (scale["flops"], scale["bytes_read"], scale["bytes_written"]) == (
        4096,
        4096 * 8,
        4096 * 8,
    )
    assert scale["arithmetic_intensity"] == 4096 / (2 * 4096 * 8)
    assert scale["attainable_flops_per_second"] == bandwidth * (4096 / 65536)
    assert scale["measured_flops_per_second"] == 4096 * NS / 950
    assert scale["fraction_of_attainable"] == (4096 * NS / 950) / (
        bandwidth * (4096 / 65536)
    )
    assert scale["bottleneck"] == "memory"
    assert placements["triad"]["fraction_of_attainable"] == pytest.approx(1.0)
    matmul_entry = placements["matmul"]
    assert matmul_entry["arithmetic_intensity"] == 2 * 64**3 / (3 * 64 * 64 * 8)
    assert matmul_entry["bottleneck"] == "compute"
    assert matmul_entry["attainable_flops_per_second"] == peak
    assert matmul_entry["fraction_of_attainable"] == 1.0


def test_stream_workloads_keep_reads_and_writes_apart():
    triad = stream()[-1]

    workload = stream_workload(triad)

    assert (workload.kind, workload.flops) == ("triad", 2 * 4096)
    assert workload.bytes_read == 3 * 4096 * 8
    assert workload.bytes_written == 2 * 4096 * 8
    with pytest.raises(TypeError, match="StreamMeasurement"):
        stream_workload(matmul())


def test_a_roofline_needs_triad_one_dtype_and_typed_inputs():
    without_triad = tuple(item for item in stream() if item.kernel != "triad")

    with pytest.raises(MemoryHierarchyError, match="needs triad"):
        measured_roofline(without_triad, matmul())
    with pytest.raises(MemoryHierarchyError, match="one dtype"):
        measured_roofline(stream("float32"), matmul("float64"))
    with pytest.raises(TypeError, match="tuple of StreamMeasurement"):
        measured_roofline(list(stream()), matmul())
    with pytest.raises(TypeError, match="MatmulMeasurement"):
        measured_roofline(stream(), stream()[0])
