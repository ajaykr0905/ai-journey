import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.kernel_metal_reference import float32_sentinels
from ai_journey.matmul_checklist_correctness import check_case, runnable
from ai_journey.matmul_checklist_runtime import ChecklistWorkload
from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_variants import (
    LIBRARY_VARIANTS,
    VARIANTS,
    ChecklistAdmission,
    guard_elements,
)
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_runtime import SideCheck
from test_kernel_metal_runtime import requires_metal
from test_matmul_checklist_runtime import VECTOR, checklist_runtime

NAIVE_GROUP = (32, 8)
STAND_IN_TORCH_DISAGREEMENT = 2.0**-30


class StandInBackend:
    """A NumPy stand-in for ``MatmulChecklistRuntime``; its reports are synthetic.

    Every kernel returns the CPU replay of the fma chain with a sentinel guard
    tail, which is what the development host's GPU returned in every case.
    ``drift`` kernels move one element by one ulp, ``leak`` kernels write
    into their tail and ``scale`` kernels multiply the output by a factor.
    """

    backend = "stand-in"
    torch_version = "2.15.0"
    float32_matmul_precision = "highest"

    def __init__(self, *, threads=None, memory=None, drift=(), leak=(), scale=None):
        threads, memory = threads or {}, memory or {}
        self.limits = {
            name: PipelineLimits(
                threads.get(name, 1024),
                32,
                memory.get(name, variant.threadgroup_bytes),
            )
            for name, variant in VARIANTS.items()
        }
        self.admissions = {
            name: ChecklistAdmission(VARIANTS[name], self.limits[name])
            for name in LIBRARY_VARIANTS
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
        admission = self.admissions.get(name)
        if admission is not None and not admission.admitted:
            raise MatmulChecklistError(f"{name} was refused by its pipeline")

    def run(self, name, left, right, launch):
        self._refuse(name)
        self.runs.append((name, launch))
        output = fma_replay(left, right).reshape(-1).copy()
        # At K = 1 the bound is about half an ulp, so one ulp can break it.
        if name in self.drift and left.shape[1] > 1:
            output[0] = numpy.nextafter(output[0], numpy.float32(numpy.inf))
        if name in self.scale:
            output *= numpy.float32(self.scale[name])
        shape = matmul_shape(left.shape[0], right.shape[1], left.shape[1])
        tail = float32_sentinels(guard_elements(VARIANTS[name], shape, launch.geometry))
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

        return ChecklistWorkload(
            sides={name: side for name in (*names, TORCH_MATMUL)},
            synchronize=self.synchronize,
            check=check,
        )


def run_case(backend, shape=matmul_shape(17, 33, 9), index=1, names=tuple(VARIANTS)):
    return check_case(
        backend,
        seed=77,
        index=index,
        shape=shape,
        naive_group_size=NAIVE_GROUP,
        names=names,
    )


def test_every_kernel_records_its_digest_guard_error_and_measured_bits():
    backend = StandInBackend()
    shape = matmul_shape(65, 68, 40)

    case = run_case(backend, shape)
    left, right = matmul_inputs(77, CORRECTNESS_STREAM, 1, shape)
    replay = fma_replay(left, right).reshape(-1)
    mapping = case.to_mapping()

    assert [output.kernel for output in case.outputs] == list(VARIANTS)
    assert [name for name, _ in backend.runs] == list(VARIANTS)
    assert mapping["bound"] == error_bound(40)
    assert list(mapping["launches"]) == list(VARIANTS)
    for output, launch in zip(case.outputs, case.launches):
        guard = guard_elements(VARIANTS[output.kernel], shape, launch.geometry)
        assert output.guard_elements == guard > 0
        buffer = numpy.concatenate([replay, float32_sentinels(guard)])
        assert output.buffer_sha256 == matrix_sha256(buffer)
        assert output.replay_bits is True and output.differing_elements == 0
        assert output.correctness_mode == "k_order_fma"
    assert case.replay_sha256 == matrix_sha256(replay.reshape(65, 68))
    assert mapping["launches"]["matmul_s6_group_32x32"]["groups"] == [1, 1]
    assert mapping["launches"]["matmul_x_rows"]["extent"] == [65, 68]


def test_other_bits_are_counted_and_do_not_stop_the_case():
    case = run_case(StandInBackend(drift=("matmul_s4_vec4",)))

    outputs = {output.kernel: output for output in case.outputs}
    assert outputs["matmul_s4_vec4"].differing_elements == 1
    assert outputs["matmul_s4_vec4"].replay_bits is False
    assert all(
        output.replay_bits
        for name, output in outputs.items()
        if name != "matmul_s4_vec4"
    )


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (StandInBackend(leak=("matmul_s7_interior",)), "escaped into the guard tail"),
        (StandInBackend(scale={"matmul_s3_block_4x4": 1.01}), "the bound is"),
    ],
)
def test_a_broken_guard_or_bound_stops_the_case(backend, message):
    with pytest.raises(MatmulChecklistError, match=message):
        run_case(backend)


def test_a_refused_kernel_or_a_short_buffer_is_an_error():
    backend = StandInBackend(threads={"matmul_s6_group_32x32": 512})

    assert "matmul_s6_group_32x32" not in [
        output.kernel
        for output in run_case(
            backend,
            names=tuple(name for name in VARIANTS if name != "matmul_s6_group_32x32"),
        ).outputs
    ]
    with pytest.raises(MatmulChecklistError, match="refused by its pipeline"):
        run_case(backend)
    with pytest.raises(MatmulChecklistError, match="not a registered variant"):
        runnable(backend, ["matmul_tiled_16"])

    class ShortBackend(StandInBackend):
        def run(self, name, left, right, launch):
            return super().run(name, left, right, launch)[:-1]

    with pytest.raises(MatmulChecklistError, match="no float32 buffer"):
        run_case(ShortBackend())


@requires_metal
def test_a_real_case_holds_the_replay_bits_for_every_admitted_kernel():
    runtime = checklist_runtime()
    names = tuple(
        name
        for name in VARIANTS
        if name not in runtime.admissions or runtime.admissions[name].admitted
    )

    case = run_case(runtime, VECTOR, names=names)

    assert [output.kernel for output in case.outputs] == list(names)
    for output in case.outputs:
        # Observed on the development host's GPU; a run records it per case.
        assert output.replay_bits is True, output.kernel
        assert output.max_relative_error <= case.bound
