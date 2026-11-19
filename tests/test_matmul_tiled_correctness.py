import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.kernel_metal_reference import float32_sentinels
from ai_journey.matmul_metal_mapping import X_COLUMNS, matmul_launch, matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_correctness import check_case
from ai_journey.matmul_tiled_runtime import NAIVE_KERNEL, SideCheck, TiledWorkload
from ai_journey.matmul_tiled_source import MatmulTiledError
from ai_journey.matmul_tiled_variants import K_ORDER_FMA, VARIANTS, VariantAdmission
from test_kernel_metal_runtime import requires_metal
from test_matmul_tiled_runtime import tiled_runtime

ALL_VARIANTS = tuple(VARIANTS.values())
NAIVE_GROUP = (32, 8)
STAND_IN_TORCH_DISAGREEMENT = 2.0**-30


class StandInBackend:
    """A NumPy stand-in for ``MatmulTiledRuntime``; reports built on it are synthetic.

    Every kernel returns the CPU replay of the fma chain with a sentinel guard
    tail, which is what the development host's GPU returned in every case.
    ``drift`` kernels move one element by one ulp, ``leak`` kernels write
    into their tail and ``scale`` kernels multiply the output by a factor.
    """

    backend = "stand-in"
    torch_version = "2.15.0"
    float32_matmul_precision = "highest"

    def __init__(
        self, variants=ALL_VARIANTS, *, threads=None, drift=(), leak=(), scale=None
    ):
        threads = threads or {}
        self.limits = {
            name: PipelineLimits(1024, 32, 0)
            for name in (NAIVE_KERNEL, "matmul_x_rows")
        }
        for variant in variants:
            self.limits[variant.name] = PipelineLimits(
                threads.get(variant.name, 1024), 32, variant.threadgroup_bytes
            )
        self.admissions = {
            variant.name: VariantAdmission(variant, self.limits[variant.name])
            for variant in variants
        }
        self.mps_switches = {
            "PYTORCH_MPS_FAST_MATH": None,
            "PYTORCH_MPS_PREFER_METAL": None,
        }
        self.drift, self.leak, self.scale = set(drift), set(leak), dict(scale or {})
        self.runs = []

    def synchronize(self):
        pass

    def _refuse(self, name):
        if name in self.admissions and not self.admissions[name].admitted:
            raise MatmulTiledError(f"{name} was refused by its pipeline")

    def run(self, name, left, right, launch):
        self._refuse(name)
        self.runs.append((name, launch))
        output = fma_replay(left, right).reshape(-1).copy()
        if name in self.drift:
            output[0] = numpy.nextafter(output[0], numpy.float32(numpy.inf))
        if name in self.scale:
            output *= numpy.float32(self.scale[name])
        tail = float32_sentinels(launch.geometry.padding_threads)
        if name in self.leak:
            tail[:1] = 0.0
        return numpy.concatenate([output, tail])

    def workload(self, left, right, launches):
        names = tuple(launches)
        for name in names:
            self._refuse(name)
        self.runs.extend(launches.items())

        def side():
            return None

        def check():
            results = {
                name: SideCheck(name not in self.drift, 0.0) for name in names[1:]
            }
            results[TORCH_MATMUL] = SideCheck(False, STAND_IN_TORCH_DISAGREEMENT)
            return results

        return TiledWorkload(
            sides={name: side for name in (*names, TORCH_MATMUL)},
            synchronize=self.synchronize,
            check=check,
        )


def run_case(backend, shape=matmul_shape(17, 33, 9), index=1, variants=ALL_VARIANTS):
    return check_case(
        backend,
        seed=76,
        index=index,
        shape=shape,
        naive_group_size=NAIVE_GROUP,
        variants=variants,
    )


def test_every_kernel_records_its_digest_guard_error_and_measured_bits():
    backend = StandInBackend()
    shape = matmul_shape(17, 33, 9)

    case = run_case(backend, shape)
    mapping = case.to_mapping()

    left, right = matmul_inputs(76, CORRECTNESS_STREAM, 1, shape)
    replay = fma_replay(left, right).reshape(-1)
    assert [output.kernel for output in case.outputs] == [NAIVE_KERNEL, *VARIANTS]
    assert [output.guard_elements for output in case.outputs] == [975, 399, 975, 1487]
    for output in case.outputs:
        tail = float32_sentinels(output.guard_elements)
        assert output.buffer_sha256 == matrix_sha256(numpy.concatenate([replay, tail]))
        assert output.replay_bits is True and output.differing_elements == 0
        assert output.correctness_mode == K_ORDER_FMA
        assert 0.0 <= output.max_relative_error <= error_bound(9)
    assert mapping["replay_sha256"] == matrix_sha256(replay.reshape(17, 33))
    assert list(mapping["launches"]) == [NAIVE_KERNEL, *VARIANTS]
    assert mapping["launches"]["matmul_tiled_8"]["group_size"] == [8, 8]
    assert mapping["outputs"][2]["bound_ratio"] == (
        case.outputs[2].max_relative_error / error_bound(9)
    )
    assert [name for name, _ in backend.runs] == [NAIVE_KERNEL, *VARIANTS]


def test_a_variant_with_other_bits_is_recorded_not_refused():
    case = run_case(StandInBackend(drift=("matmul_tiled_16",)))

    records = {output.kernel: output for output in case.outputs}
    assert records["matmul_tiled_16"].differing_elements == 1
    assert records["matmul_tiled_16"].replay_bits is False
    assert case.to_mapping()["outputs"][2]["replay_bits"] is False
    assert all(records[name].replay_bits for name in ("matmul_tiled_8", NAIVE_KERNEL))


def test_a_padding_thread_write_or_an_error_past_the_bound_stops_the_run():
    with pytest.raises(MatmulTiledError, match="matmul_tiled_8: a padding thread"):
        run_case(StandInBackend(leak=("matmul_tiled_8",)))
    with pytest.raises(MatmulTiledError, match="matmul_tiled_32 has a relative error"):
        run_case(StandInBackend(scale={"matmul_tiled_32": 1.001}))


def test_only_admitted_variants_run_and_outputs_must_be_whole_buffers():
    backend = StandInBackend(threads={"matmul_tiled_32": 512})
    admitted = tuple(
        variant for variant in ALL_VARIANTS if backend.admissions[variant.name].admitted
    )

    case = run_case(backend, variants=admitted)

    assert [output.kernel for output in case.outputs] == [
        NAIVE_KERNEL,
        "matmul_tiled_8",
        "matmul_tiled_16",
    ]
    with pytest.raises(MatmulTiledError, match="refused by its pipeline"):
        run_case(backend)
    short = StandInBackend()
    short.run = lambda name, left, right, launch: numpy.zeros(3, numpy.float32)
    with pytest.raises(MatmulTiledError, match="no float32 buffer"):
        run_case(short)
    with pytest.raises(TypeError, match="TiledVariant"):
        run_case(StandInBackend(), variants=list(ALL_VARIANTS))


@requires_metal
def test_the_stand_in_matches_this_gpu_bit_for_bit_on_a_padded_k_tail_case():
    runtime = tiled_runtime()
    shape = matmul_shape(31, 47, 129)
    admitted = tuple(
        variant for variant in ALL_VARIANTS if runtime.admissions[variant.name].admitted
    )

    measured = run_case(runtime, shape, index=3, variants=admitted).to_mapping()
    stand_in = run_case(StandInBackend(), shape, index=3, variants=admitted)

    assert measured == stand_in.to_mapping()
    for name, launch in measured["launches"].items():
        expected = matmul_launch(X_COLUMNS, shape, tuple(launch["group_size"]))
        assert launch["padding_threads"] == expected.padding_threads, name
