"""Observations recorded on the development host's Apple GPU, replayed on any host.

``RecordedBackend`` stands in for ``MetalRuntime`` where MPS is missing: it
computes outputs with NumPy and replays what the GPU produced on 2026-11-16
(torch 2.15.0, macOS 27.2, arm64). Reports built on it are synthetic, never
measurements. ``RECORDED_REPORT_JSON`` is one real measured report from that
host, so a host without torch or MPS can verify a genuine Apple GPU report.
"""

import json

import numpy

from ai_journey.kernel_metal_geometry import CheckedLaunch, PipelineLimits, image_launch
from ai_journey.kernel_metal_reference import (
    ALL_TRIPLES_SHAPE,
    UINT8_SENTINEL,
    all_rgb_triples,
    array_sha256,
    float32_sentinels,
    rgb_to_gray_reference,
    uint8_sentinels,
    vector_add_reference,
)
from ai_journey.kernel_metal_runtime import DeviceWorkload
from ai_journey.kernel_metal_source import KERNELS, PROBE_FUSED_RESULT
from ai_journey.kernel_metal_verify import verify_report
from ai_journey.memory_access_plan import canonical_json_bytes

# Both pipelines reported these limits.
RECORDED_LIMITS = PipelineLimits(
    max_threads_per_threadgroup=1024,
    thread_execution_width=32,
    static_threadgroup_memory_bytes=0,
)
RECORDED_TORCH_VERSION = "2.15.0"
# The contraction probe returned 2**-24: the compiler fused the multiply-add.
RECORDED_CONTRACTION = PROBE_FUSED_RESULT
# (flat pixel index, Metal level) for every pixel of the all-triples image
# where the Metal kernel differed from the float32 reference. Three runs and
# 16 x 16 and 32 x 8 threadgroups gave the same 21 pixels.
RECORDED_ALL_TRIPLES_MISMATCHES = (
    (669970, 38),
    (670691, 63),
    (671633, 56),
    (683937, 86),
    (1341132, 98),
    (1350831, 117),
    (1367624, 144),
    (2023877, 163),
    (7921339, 187),
    (11164471, 111),
    (15101710, 136),
    (15757963, 155),
    (15767662, 174),
    (15777361, 193),
    (15787060, 212),
    (15791328, 241),
    (16400249, 126),
    (16412553, 156),
    (16433614, 212),
    (16441650, 213),
    (16443313, 231),
)
ALL_TRIPLES_SHA256 = "95eeb80877c99cdcb38755b9bb5ed29066bf70e870ea6eff9ee30285bd4cd5b7"
# The canonical file that measure_report wrote on the development host for a
# config with two vector lengths, one seeded image plus every RGB triple, and
# one sweep point per kernel. Under heavy load from other jobs; the timings
# describe that moment and are here only to be verified.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"device":"mps","rgb_to_gr'
    'ay":[{"bytes_per_call":1230720,"height":480,"kernel":"rgb_to_gray","launch":'
    '{"active_threads":307680,"extent":[641,480],"group_size":[16,16],"groups":[4'
    '1,30],"idle_lanes_per_threadgroup":0,"launched_threads":314880,"padding_thre'
    'ads":7200,"simd_groups_per_threadgroup":8,"threads":[656,480],"threads_per_g'
    'roup":256},"metal_kernel":{"best_gb_per_second":5.704380069524913,"bytes_per'
    '_sample":1230720,"calls_per_sample":1,"implementation":"metal_kernel","iqr_n'
    's_per_call":247375.0,"median_gb_per_second":4.782578283475949,"median_ns_per'
    '_call":257334.0,"repeats":3,"samples_ns":[463125,257334,215750],"timing":{"c'
    'ount":3,"iqr_ns":247375,"max_ns":463125,"median_ns":257334,"min_ns":215750,"'
    'p25_ns":215750,"p75_ns":463125},"warmup":1},"torch":{"best_gb_per_second":1.'
    '4788592769364426,"bytes_per_sample":1230720,"calls_per_sample":1,"implementa'
    'tion":"torch","iqr_ns_per_call":365041.0,"median_gb_per_second":1.0956367533'
    '998284,"median_ns_per_call":1123292.0,"repeats":3,"samples_ns":[1197250,1123'
    '292,832209],"timing":{"count":3,"iqr_ns":365041,"max_ns":1197250,"median_ns"'
    ':1123292,"min_ns":832209,"p25_ns":832209,"p75_ns":1197250},"warmup":1},"torc'
    'h_to_metal_time_ratio":4.365113043748591,"width":641}],"synchronization":"to'
    "rch.mps.synchronize() before the first and before the second clock reading o"
    "f every sample; untimed warmups first; the two sides alternate and the order"
    ' flips every repeat","torch_equivalents":{"rgb_to_gray":"p = rgb.to(float32)'
    "; y = p[..., 0] * 0.2989 + p[..., 1] * 0.5870; y = y + p[..., 2] * 0.1140; f"
    'loor(y + 0.5).clamp_(0, 255).to(uint8)","vector_add":"torch.add(a, b, out=ou'
    't)"},"vector_add":[{"bytes_per_call":786444,"kernel":"vector_add","launch":{'
    '"active_threads":65537,"extent":[65537],"group_size":[256],"groups":[257],"i'
    'dle_lanes_per_threadgroup":0,"launched_threads":65792,"padding_threads":255,'
    '"simd_groups_per_threadgroup":8,"threads":[65792],"threads_per_group":256},"'
    'length":65537,"metal_kernel":{"best_gb_per_second":3.515815590947192,"bytes_'
    'per_sample":1572888,"calls_per_sample":2,"implementation":"metal_kernel","iq'
    'r_ns_per_call":51604.5,"median_gb_per_second":2.9926481780267284,"median_ns_'
    'per_call":262792.0,"repeats":3,"samples_ns":[525584,550584,447375],"timing":'
    '{"count":3,"iqr_ns":103209,"max_ns":550584,"median_ns":525584,"min_ns":44737'
    '5,"p25_ns":447375,"p75_ns":550584},"warmup":1},"torch":{"best_gb_per_second"'
    ':3.9523870549478715,"bytes_per_sample":1572888,"calls_per_sample":2,"impleme'
    'ntation":"torch","iqr_ns_per_call":63520.5,"median_gb_per_second":3.79351944'
    '5281881,"median_ns_per_call":207312.5,"repeats":3,"samples_ns":[525000,41462'
    '5,397959],"timing":{"count":3,"iqr_ns":127041,"max_ns":525000,"median_ns":41'
    '4625,"min_ns":397959,"p25_ns":397959,"p75_ns":525000},"warmup":1},"torch_to_'
    'metal_time_ratio":0.7888843648208469}]},"config":{"benchmark":{"max_runtime_'
    'seconds":60,"repeats":3,"rgb_to_gray_shapes":[[480,641]],"target_bytes_per_s'
    'ample":1048576,"vector_add_lengths":[65537],"warmup":1},"correctness":{"rgb_'
    'to_gray_all_triples":true,"rgb_to_gray_shapes":[[17,33]],"vector_add_lengths'
    '":[257,1000]},"launch":{"rgb_to_gray_group_size":[16,16],"vector_add_group_s'
    'ize":256},"name":"kernel-metal-recorded","schema_version":1,"seed":73},"conf'
    'ig_sha256":"5bd953359af1d7f57c90b99b20233b78560784b61495a28cf04717a039f3beb2'
    '","correctness":{"bound_levels":1,"reference":"NumPy float32 on the CPU, reb'
    'uilt from the seed","rgb_to_gray":[{"bound_levels":1,"height":17,"index":0,"'
    'launch":{"active_threads":561,"extent":[33,17],"group_size":[16,16],"groups"'
    ':[3,2],"idle_lanes_per_threadgroup":0,"launched_threads":1536,"padding_threa'
    'ds":975,"simd_groups_per_threadgroup":8,"threads":[48,32],"threads_per_group'
    '":256},"outputs":[{"buffer_sha256":"84a196970c543d2fa59a7749d174459f0fe458b1'
    '2c17df5ce1d3bfc610eac0b5","device":"mps","guard_elements":975,"implementatio'
    'n":"metal_kernel","max_abs_difference_levels":0,"mismatched_pixels":0,"misma'
    'tches":[]},{"buffer_sha256":"fd17eeaccc763bbf357490f3b0712eb4748da7fcc8f9c73'
    'aed9c762dd233cb9c","device":"mps","guard_elements":0,"implementation":"torch'
    '","max_abs_difference_levels":0,"mismatched_pixels":0,"mismatches":[]},{"buf'
    'fer_sha256":"fd17eeaccc763bbf357490f3b0712eb4748da7fcc8f9c73aed9c762dd233cb9'
    'c","device":"cpu","guard_elements":0,"implementation":"torch","max_abs_diffe'
    'rence_levels":0,"mismatched_pixels":0,"mismatches":[]}],"pattern":"uniform",'
    '"reference_sha256":"fd17eeaccc763bbf357490f3b0712eb4748da7fcc8f9c73aed9c762d'
    'd233cb9c","rgb_sha256":"2c2b39a98d773e94e2af75d529a1d042f3e7cf439621a4acc69b'
    '9be8d369b11d","width":33},{"bound_levels":1,"height":4096,"index":1,"launch"'
    ':{"active_threads":16777216,"extent":[4096,4096],"group_size":[16,16],"group'
    's":[256,256],"idle_lanes_per_threadgroup":0,"launched_threads":16777216,"pad'
    'ding_threads":0,"simd_groups_per_threadgroup":8,"threads":[4096,4096],"threa'
    'ds_per_group":256},"outputs":[{"buffer_sha256":"6841f0d4beb4e411d0acf1e2ac90'
    'b97f1e1e189e804fef26256b23b8802ff8aa","device":"mps","guard_elements":0,"imp'
    'lementation":"metal_kernel","max_abs_difference_levels":1,"mismatched_pixels'
    '":21,"mismatches":[[669970,38],[670691,63],[671633,56],[683937,86],[1341132,'
    "98],[1350831,117],[1367624,144],[2023877,163],[7921339,187],[11164471,111],["
    "15101710,136],[15757963,155],[15767662,174],[15777361,193],[15787060,212],[1"
    "5791328,241],[16400249,126],[16412553,156],[16433614,212],[16441650,213],[16"
    '443313,231]]},{"buffer_sha256":"73b98ca1d9177e3bcea9816a73f18bcd9f4080ad50a6'
    'af8c28408ff6726bbaa4","device":"mps","guard_elements":0,"implementation":"to'
    'rch","max_abs_difference_levels":0,"mismatched_pixels":0,"mismatches":[]},{"'
    'buffer_sha256":"73b98ca1d9177e3bcea9816a73f18bcd9f4080ad50a6af8c28408ff6726b'
    'baa4","device":"cpu","guard_elements":0,"implementation":"torch","max_abs_di'
    'fference_levels":0,"mismatched_pixels":0,"mismatches":[]}],"pattern":"all_rg'
    'b_triples","reference_sha256":"73b98ca1d9177e3bcea9816a73f18bcd9f4080ad50a6a'
    'f8c28408ff6726bbaa4","rgb_sha256":"95eeb80877c99cdcb38755b9bb5ed29066bf70e87'
    '0ea6eff9ee30285bd4cd5b7","width":4096}],"rounding":"y = 0.2989 R + 0.5870 G '
    "+ 0.1140 B in float32, left to right; gray = clamp(floor(y + 0.5), 0, 255): "
    'round half up, then clamp to 0..255","sentinels":{"float32_bits":2143289344,'
    '"uint8":165},"vector_add":[{"a_sha256":"d670226a2734ec995ee2289ca7f197d857aa'
    '65bb85d213085eb8ec3f4019609e","b_sha256":"eda4ab11100bc7190af416f868edb6eb81'
    '4d16b6b2fa19cbd35cf8a18c568a92","index":0,"launch":{"active_threads":257,"ex'
    'tent":[257],"group_size":[256],"groups":[2],"idle_lanes_per_threadgroup":0,"'
    'launched_threads":512,"padding_threads":255,"simd_groups_per_threadgroup":8,'
    '"threads":[512],"threads_per_group":256},"length":257,"outputs":[{"buffer_sh'
    'a256":"cc0d00e37bb4a191a55476c66b4e80123c305356bf7ce25d8936bedd81a16345","de'
    'vice":"mps","guard_elements":255,"implementation":"metal_kernel"},{"buffer_s'
    'ha256":"ff6bb71977006a9d8961c4165b0d3762461b7ae0eb83bea832a4a6f669f8ad17","d'
    'evice":"mps","guard_elements":0,"implementation":"torch"},{"buffer_sha256":"'
    'ff6bb71977006a9d8961c4165b0d3762461b7ae0eb83bea832a4a6f669f8ad17","device":"'
    'cpu","guard_elements":0,"implementation":"torch"}],"reference_sha256":"ff6bb'
    '71977006a9d8961c4165b0d3762461b7ae0eb83bea832a4a6f669f8ad17"},{"a_sha256":"0'
    '57854840512a38dc9b14a11c33d1e4226baeb01845c06b21a2cbc6166f16ed9","b_sha256":'
    '"665fa1eff1d9a9aec106acc4942c8a6372a4d5810d22ef41207a5bd2742af06f","index":1'
    ',"launch":{"active_threads":1000,"extent":[1000],"group_size":[256],"groups"'
    ':[4],"idle_lanes_per_threadgroup":0,"launched_threads":1024,"padding_threads'
    '":24,"simd_groups_per_threadgroup":8,"threads":[1024],"threads_per_group":25'
    '6},"length":1000,"outputs":[{"buffer_sha256":"5268a021dfa24d567632d5a424d87d'
    '99902518539e6515484c71e54970f0907d","device":"mps","guard_elements":24,"impl'
    'ementation":"metal_kernel"},{"buffer_sha256":"d85977aa2a0e8f34005070b60c60a3'
    '71cb92f6bdabcbd6663198d887c1c6c3d8","device":"mps","guard_elements":0,"imple'
    'mentation":"torch"},{"buffer_sha256":"d85977aa2a0e8f34005070b60c60a371cb92f6'
    'bdabcbd6663198d887c1c6c3d8","device":"cpu","guard_elements":0,"implementatio'
    'n":"torch"}],"reference_sha256":"d85977aa2a0e8f34005070b60c60a371cb92f6bdabc'
    'bd6663198d887c1c6c3d8"}],"weights":[0.2989,0.587,0.114]},"environment":{"mac'
    'os_version":"27.2","numpy_version":"1.26.4","platform_machine":"arm64","plat'
    'form_system":"Darwin","python_implementation":"CPython","python_version":"3.'
    '11.14","torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_'
    'host":true,"cuda_kernel_executed":false,"distributed_claim":false,"host_spec'
    'ific":true,"nvidia_gpu_used":false,"portable_performance_claim":false,"produ'
    'ction_claim":false,"statement":"These are wall-clock timings on the Apple GP'
    "U of the host that made this report. The Metal kernels came from torch.mps.c"
    "ompile_shader, and their PyTorch equivalents ran on the MPS device. The timi"
    "ngs are host-specific Apple GPU (Metal) measurements. No CUDA kernel ran, an"
    "d no NVIDIA GPU was used. Effective bandwidth counts the bytes that each ker"
    "nel must read and write, not the hardware traffic. These timings do not meas"
    'ure or predict CUDA, NVIDIA, distributed or production performance."},"kind"'
    ':"kernel_metal_correctness_and_benchmark","label":"host-specific Apple GPU ('
    'Metal) measurement","measured":true,"measurement_scope":"apple_gpu_metal_thi'
    's_host","percentile_method":"nearest rank: sorted samples[ceil(p * n / 100) '
    "- 1] for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no interpol"
    'ation","planned_work":{"accounted_bytes":22428864,"benchmark_points":2,"call'
    's":24,"correctness_cases":4,"timed_samples":12},"report_sha256":"bd63dcf93fd'
    'eb13895c3a357c08d8b40ecea12d13a3bfdd9afa929d85ce93e6e","run":{"elapsed_ns":1'
    '023263875,"max_runtime_ns":60000000000,"timed_ns":6950087},"schema_version":'
    '1,"shaders":{"contraction_probe":{"expression":"x[0] * x[1] + x[2]","fused":'
    'true,"fused_result":5.960464477539063e-08,"inputs":[1.000244140625,1.0002441'
    '40625,-1.00048828125],"observed":5.960464477539063e-08,"unfused_result":0.0}'
    ',"kernels":[{"axes":1,"buffers":["out","a","b","n"],"bytes_per_item":12,"lim'
    'its":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0'
    ',"thread_execution_width":32},"name":"vector_add","scalar_buffers":[3]},{"ax'
    'es":2,"buffers":["out","rgb","width","height"],"bytes_per_item":4,"limits":{'
    '"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thre'
    'ad_execution_width":32},"name":"rgb_to_gray","scalar_buffers":[2,3]}],"sourc'
    'e":"#include <metal_stdlib>\\nusing namespace metal;\\n\\n// One thread per ele'
    "ment. thread_position_in_grid is CUDA's\\n// blockIdx.x * blockDim.x + thread"
    "Idx.x. The grid is whole threadgroups,\\n// so the guard stops the padding th"
    "reads past n.\\nkernel void vector_add(device float* out [[buffer(0)]],\\n    "
    "                   device const float* a [[buffer(1)]],\\n                   "
    "    device const float* b [[buffer(2)]],\\n                       constant ui"
    "nt& n [[buffer(3)]],\\n                       uint i [[thread_position_in_gri"
    "d]]) {\\n    if (i < n) {\\n        out[i] = a[i] + b[i];\\n    }\\n}\\n\\n// One "
    "thread per pixel: x is the column and y the row, as in a CUDA 2D\\n// launch."
    " Pixels are row-major RGB triples; out holds one level per pixel.\\n// Round "
    "half up, then clamp to 0..255.\\nkernel void rgb_to_gray(device uchar* out [["
    "buffer(0)]],\\n                        device const uchar* rgb [[buffer(1)]],"
    "\\n                        constant uint& width [[buffer(2)]],\\n             "
    "           constant uint& height [[buffer(3)]],\\n                        uin"
    "t2 position [[thread_position_in_grid]]) {\\n    uint column = position.x;\\n "
    "   uint row = position.y;\\n    if (column < width && row < height) {\\n      "
    "  uint pixel = row * width + column;\\n        float r = float(rgb[3 * pixel]"
    ");\\n        float g = float(rgb[3 * pixel + 1]);\\n        float b = float(rg"
    "b[3 * pixel + 2]);\\n        float y = 0.2989f * r + 0.5870f * g + 0.1140f * "
    "b;\\n        out[pixel] = uchar(clamp(floor(y + 0.5f), 0.0f, 255.0f));\\n    }"
    "\\n}\\n\\n// Fused multiply-add probe: 0 if the product is rounded before the s"
    "um,\\n// 2**-24 if the compiler fuses them.\\nkernel void contraction_probe(de"
    "vice float* out [[buffer(0)]],\\n                              device const f"
    "loat* x [[buffer(1)]],\\n                              uint i [[thread_positi"
    "on_in_grid]]) {\\n    if (i == 0) {\\n        out[0] = x[0] * x[1] + x[2];\\n  "
    '  }\\n}\\n","source_sha256":"f12a5e8af97c2baf1f835af8ec166d3c4cdf02a07bcf53d32'
    'c658a7fb0938802"},"summary":{"benchmark_points":2,"contraction_fused":true,"'
    'correctness_cases":4,"label":"host-specific Apple GPU (Metal) measurement","'
    'peak_metal_kernel_gb_per_second":4.782578283475949,"rgb_to_gray_largest":{"h'
    'eight":480,"metal_kernel_gb_per_second":4.782578283475949,"torch_gb_per_seco'
    'nd":1.0956367533998284,"torch_to_metal_time_ratio":4.365113043748591,"width"'
    ':641},"rgb_to_gray_max_abs_difference_levels":1,"rgb_to_gray_metal_mismatche'
    'd_pixels":21,"timed_samples":12,"vector_add_exact_cases":2,"vector_add_large'
    'st":{"length":65537,"metal_kernel_gb_per_second":2.9926481780267284,"torch_g'
    'b_per_second":3.793519445281881,"torch_to_metal_time_ratio":0.78888436482084'
    '69}},"timer":"time.perf_counter_ns"}'
)


class RecordedBackend:
    """A NumPy stand-in for ``MetalRuntime`` that replays recorded GPU output."""

    backend = "recorded"
    torch_version = RECORDED_TORCH_VERSION

    def __init__(self, mismatches=RECORDED_ALL_TRIPLES_MISMATCHES, limits=None):
        self.mismatches = mismatches
        self.limits = {name: limits or RECORDED_LIMITS for name in KERNELS}
        self.launches = []

    def synchronize(self):
        pass

    def contraction_probe(self):
        return RECORDED_CONTRACTION

    def vector_add(self, first, second, launch):
        self.launches.append(launch)
        tail = float32_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([vector_add_reference(first, second), tail])

    def rgb_to_gray(self, rgb, launch):
        self.launches.append(launch)
        gray = rgb_to_gray_reference(rgb).reshape(-1)
        if rgb.shape == (*ALL_TRIPLES_SHAPE, 3) and array_sha256(rgb) == (
            ALL_TRIPLES_SHA256
        ):
            for index, level in self.mismatches:
                gray[index] = level
        tail = uint8_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([gray, tail])

    def torch_vector_add(self, first, second, device):
        return vector_add_reference(first, second)

    def torch_rgb_to_gray(self, rgb, device):
        return rgb_to_gray_reference(rgb)

    def vector_add_workload(self, first, second, launch):
        self.launches.append(launch)
        return DeviceWorkload(
            metal=lambda: numpy.add(first, second),
            torch=lambda: numpy.add(first, second),
            synchronize=self.synchronize,
            check=lambda: True,
        )

    def rgb_to_gray_workload(self, rgb, launch):
        self.launches.append(launch)
        return DeviceWorkload(
            metal=lambda: rgb_to_gray_reference(rgb),
            torch=lambda: rgb_to_gray_reference(rgb),
            synchronize=self.synchronize,
            check=lambda: True,
        )


def test_every_recorded_mismatch_is_a_one_level_flip_at_an_exact_half():
    image = all_rgb_triples()
    reference = rgb_to_gray_reference(image).reshape(-1)
    pixels = image.reshape(-1, 3).astype(numpy.int64)

    assert array_sha256(image) == ALL_TRIPLES_SHA256
    for index, level in RECORDED_ALL_TRIPLES_MISMATCHES:
        red, green, blue = pixels[index]
        # The exact decimal value is k + 0.5, a tie that a fused multiply-add
        # and two float32 roundings can settle on different sides.
        assert (2989 * red + 5870 * green + 1140 * blue) % 10000 == 5000
        assert abs(level - int(reference[index])) == 1


def test_the_recorded_backend_replays_its_mismatches_only_on_all_triples():
    backend = RecordedBackend()
    image = all_rgb_triples()
    launch = CheckedLaunch(image_launch(4096, 4096, (48, 16)), RECORDED_LIMITS)
    reference = rgb_to_gray_reference(image).reshape(-1)

    buffer = backend.rgb_to_gray(image, launch)

    changed = numpy.nonzero(buffer[: reference.size] != reference)[0]
    assert changed.tolist() == [index for index, _ in RECORDED_ALL_TRIPLES_MISMATCHES]
    assert launch.geometry.padding_threads == 32 * 4096
    assert set(buffer[reference.size :].tolist()) == {UINT8_SENTINEL}
    stripe = image[:16].copy()
    launch = CheckedLaunch(image_launch(16, 4096, (16, 16)), RECORDED_LIMITS)
    assert numpy.array_equal(
        backend.rgb_to_gray(stripe, launch), rgb_to_gray_reference(stripe).reshape(-1)
    )


def test_the_recorded_metal_report_verifies_on_any_host():
    report = json.loads(RECORDED_REPORT_JSON)

    summary = verify_report(report)

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert summary["rgb_to_gray_metal_mismatched_pixels"] == 21
    assert report["environment"]["torch_version"] == RECORDED_TORCH_VERSION
    shaders = report["shaders"]
    assert shaders["contraction_probe"]["observed"] == RECORDED_CONTRACTION
    for kernel in shaders["kernels"]:
        assert kernel["limits"] == RECORDED_LIMITS.to_mapping()
    metal = report["correctness"]["rgb_to_gray"][-1]["outputs"][0]
    assert metal["mismatches"] == [
        list(item) for item in RECORDED_ALL_TRIPLES_MISMATCHES
    ]
