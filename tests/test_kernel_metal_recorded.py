"""Observations recorded on the development host's Apple GPU, replayed on any host.

``RecordedBackend`` stands in for ``MetalRuntime`` where MPS is missing: it
computes outputs with NumPy, which is what the GPU produced on 2026-11-16
(torch 2.15.0, macOS 27.2, arm64), because the vector add matched the
reference bit for bit and the integer grayscale rule matched it on every
RGB triple. Reports built on it are synthetic, never measurements.
``RECORDED_REPORT_JSON`` is one real measured report from that host, so a
host without torch or MPS can verify a genuine Apple GPU report.
"""

import json

import numpy

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.kernel_metal_reference import (
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
    'roup":256},"metal_kernel":{"best_gb_per_second":9.15605285085109,"bytes_per_'
    'sample":1230720,"calls_per_sample":1,"implementation":"metal_kernel","iqr_ns'
    '_per_call":267668.0,"median_gb_per_second":7.517775551591859,"median_ns_per_'
    'call":163708.0,"repeats":3,"samples_ns":[402084,163708,134416],"timing":{"co'
    'unt":3,"iqr_ns":267668,"max_ns":402084,"median_ns":163708,"min_ns":134416,"p'
    '25_ns":134416,"p75_ns":402084},"warmup":1},"torch":{"best_gb_per_second":3.3'
    '413152193389695,"bytes_per_sample":1230720,"calls_per_sample":1,"implementat'
    'ion":"torch","iqr_ns_per_call":221333.0,"median_gb_per_second":2.16437898439'
    '2174,"median_ns_per_call":568625.0,"repeats":3,"samples_ns":[589667,568625,3'
    '68334],"timing":{"count":3,"iqr_ns":221333,"max_ns":589667,"median_ns":56862'
    '5,"min_ns":368334,"p25_ns":368334,"p75_ns":589667},"warmup":1},"torch_to_met'
    'al_time_ratio":3.47340997385589,"width":641}],"synchronization":"torch.mps.s'
    "ynchronize() before the first and before the second clock reading of every s"
    "ample; untimed warmups first; the two sides alternate and the order flips ev"
    'ery repeat","torch_equivalents":{"rgb_to_gray":"p = rgb.to(int32); y = p[...'
    ", 0] * 2989 + p[..., 1] * 5870; y = y + p[..., 2] * 1140; div(y + 5000, 1000"
    '0, rounding_mode=\'floor\').clamp_(0, 255).to(uint8)","vector_add":"torch.add('
    'a, b, out=out)"},"vector_add":[{"bytes_per_call":786444,"kernel":"vector_add'
    '","launch":{"active_threads":65537,"extent":[65537],"group_size":[256],"grou'
    'ps":[257],"idle_lanes_per_threadgroup":0,"launched_threads":65792,"padding_t'
    'hreads":255,"simd_groups_per_threadgroup":8,"threads":[65792],"threads_per_g'
    'roup":256},"length":65537,"metal_kernel":{"best_gb_per_second":12.9900564897'
    '09623,"bytes_per_sample":1572888,"calls_per_sample":2,"implementation":"meta'
    'l_kernel","iqr_ns_per_call":120666.5,"median_gb_per_second":10.5386130653266'
    '33,"median_ns_per_call":74625.0,"repeats":3,"samples_ns":[362417,149250,1210'
    '84],"timing":{"count":3,"iqr_ns":241333,"max_ns":362417,"median_ns":149250,"'
    'min_ns":121084,"p25_ns":121084,"p75_ns":362417},"warmup":1},"torch":{"best_g'
    'b_per_second":10.835099127894962,"bytes_per_sample":1572888,"calls_per_sampl'
    'e":2,"implementation":"torch","iqr_ns_per_call":11625.0,"median_gb_per_secon'
    'd":9.789799957676173,"median_ns_per_call":80333.0,"repeats":3,"samples_ns":['
    '160666,168416,145166],"timing":{"count":3,"iqr_ns":23250,"max_ns":168416,"me'
    'dian_ns":160666,"min_ns":145166,"p25_ns":145166,"p75_ns":168416},"warmup":1}'
    ',"torch_to_metal_time_ratio":1.0764891122278057}]},"config":{"benchmark":{"m'
    'ax_runtime_seconds":60,"repeats":3,"rgb_to_gray_shapes":[[480,641]],"target_'
    'bytes_per_sample":1048576,"vector_add_lengths":[65537],"warmup":1},"correctn'
    'ess":{"rgb_to_gray_all_triples":true,"rgb_to_gray_shapes":[[17,33]],"vector_'
    'add_lengths":[257,1000]},"launch":{"rgb_to_gray_group_size":[16,16],"vector_'
    'add_group_size":256},"name":"kernel-metal-recorded","schema_version":1,"seed'
    '":73},"config_sha256":"5bd953359af1d7f57c90b99b20233b78560784b61495a28cf0471'
    '7a039f3beb2","correctness":{"bound_levels":0,"reference":"NumPy integers on '
    'the CPU, rebuilt from the seed","rgb_to_gray":[{"bound_levels":0,"height":17'
    ',"index":0,"launch":{"active_threads":561,"extent":[33,17],"group_size":[16,'
    '16],"groups":[3,2],"idle_lanes_per_threadgroup":0,"launched_threads":1536,"p'
    'adding_threads":975,"simd_groups_per_threadgroup":8,"threads":[48,32],"threa'
    'ds_per_group":256},"outputs":[{"buffer_sha256":"84a196970c543d2fa59a7749d174'
    '459f0fe458b12c17df5ce1d3bfc610eac0b5","device":"mps","guard_elements":975,"i'
    'mplementation":"metal_kernel","max_abs_difference_levels":0,"mismatched_pixe'
    'ls":0,"mismatches":[]},{"buffer_sha256":"fd17eeaccc763bbf357490f3b0712eb4748'
    'da7fcc8f9c73aed9c762dd233cb9c","device":"mps","guard_elements":0,"implementa'
    'tion":"torch","max_abs_difference_levels":0,"mismatched_pixels":0,"mismatche'
    's":[]},{"buffer_sha256":"fd17eeaccc763bbf357490f3b0712eb4748da7fcc8f9c73aed9'
    'c762dd233cb9c","device":"cpu","guard_elements":0,"implementation":"torch","m'
    'ax_abs_difference_levels":0,"mismatched_pixels":0,"mismatches":[]}],"pattern'
    '":"uniform","reference_sha256":"fd17eeaccc763bbf357490f3b0712eb4748da7fcc8f9'
    'c73aed9c762dd233cb9c","rgb_sha256":"2c2b39a98d773e94e2af75d529a1d042f3e7cf43'
    '9621a4acc69b9be8d369b11d","width":33},{"bound_levels":0,"height":4096,"index'
    '":1,"launch":{"active_threads":16777216,"extent":[4096,4096],"group_size":[1'
    '6,16],"groups":[256,256],"idle_lanes_per_threadgroup":0,"launched_threads":1'
    '6777216,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[4096,'
    '4096],"threads_per_group":256},"outputs":[{"buffer_sha256":"fd765e780c79467c'
    '5a60363988d11528e0e76a30efe4d37517897ca0fa08f9c1","device":"mps","guard_elem'
    'ents":0,"implementation":"metal_kernel","max_abs_difference_levels":0,"misma'
    'tched_pixels":0,"mismatches":[]},{"buffer_sha256":"fd765e780c79467c5a6036398'
    '8d11528e0e76a30efe4d37517897ca0fa08f9c1","device":"mps","guard_elements":0,"'
    'implementation":"torch","max_abs_difference_levels":0,"mismatched_pixels":0,'
    '"mismatches":[]},{"buffer_sha256":"fd765e780c79467c5a60363988d11528e0e76a30e'
    'fe4d37517897ca0fa08f9c1","device":"cpu","guard_elements":0,"implementation":'
    '"torch","max_abs_difference_levels":0,"mismatched_pixels":0,"mismatches":[]}'
    '],"pattern":"all_rgb_triples","reference_sha256":"fd765e780c79467c5a60363988'
    'd11528e0e76a30efe4d37517897ca0fa08f9c1","rgb_sha256":"95eeb80877c99cdcb38755'
    'b9bb5ed29066bf70e870ea6eff9ee30285bd4cd5b7","width":4096}],"rounding":"gray '
    "= min((2989 R + 5870 G + 1140 B + 5000) // 10000, 255) in integers: the exac"
    "t weighted sum of the decimal weights 0.2989, 0.5870 and 0.1140, rounded hal"
    'f up, then clamped to 0..255, the rule of the CUDA half","sentinels":{"float'
    '32_bits":2143289344,"uint8":165},"vector_add":[{"a_sha256":"d670226a2734ec99'
    '5ee2289ca7f197d857aa65bb85d213085eb8ec3f4019609e","b_sha256":"eda4ab11100bc7'
    '190af416f868edb6eb814d16b6b2fa19cbd35cf8a18c568a92","index":0,"launch":{"act'
    'ive_threads":257,"extent":[257],"group_size":[256],"groups":[2],"idle_lanes_'
    'per_threadgroup":0,"launched_threads":512,"padding_threads":255,"simd_groups'
    '_per_threadgroup":8,"threads":[512],"threads_per_group":256},"length":257,"o'
    'utputs":[{"buffer_sha256":"cc0d00e37bb4a191a55476c66b4e80123c305356bf7ce25d8'
    '936bedd81a16345","device":"mps","guard_elements":255,"implementation":"metal'
    '_kernel"},{"buffer_sha256":"ff6bb71977006a9d8961c4165b0d3762461b7ae0eb83bea8'
    '32a4a6f669f8ad17","device":"mps","guard_elements":0,"implementation":"torch"'
    '},{"buffer_sha256":"ff6bb71977006a9d8961c4165b0d3762461b7ae0eb83bea832a4a6f6'
    '69f8ad17","device":"cpu","guard_elements":0,"implementation":"torch"}],"refe'
    'rence_sha256":"ff6bb71977006a9d8961c4165b0d3762461b7ae0eb83bea832a4a6f669f8a'
    'd17"},{"a_sha256":"057854840512a38dc9b14a11c33d1e4226baeb01845c06b21a2cbc616'
    '6f16ed9","b_sha256":"665fa1eff1d9a9aec106acc4942c8a6372a4d5810d22ef41207a5bd'
    '2742af06f","index":1,"launch":{"active_threads":1000,"extent":[1000],"group_'
    'size":[256],"groups":[4],"idle_lanes_per_threadgroup":0,"launched_threads":1'
    '024,"padding_threads":24,"simd_groups_per_threadgroup":8,"threads":[1024],"t'
    'hreads_per_group":256},"length":1000,"outputs":[{"buffer_sha256":"5268a021df'
    'a24d567632d5a424d87d99902518539e6515484c71e54970f0907d","device":"mps","guar'
    'd_elements":24,"implementation":"metal_kernel"},{"buffer_sha256":"d85977aa2a'
    '0e8f34005070b60c60a371cb92f6bdabcbd6663198d887c1c6c3d8","device":"mps","guar'
    'd_elements":0,"implementation":"torch"},{"buffer_sha256":"d85977aa2a0e8f3400'
    '5070b60c60a371cb92f6bdabcbd6663198d887c1c6c3d8","device":"cpu","guard_elemen'
    'ts":0,"implementation":"torch"}],"reference_sha256":"d85977aa2a0e8f34005070b'
    '60c60a371cb92f6bdabcbd6663198d887c1c6c3d8"}],"weights":[0.2989,0.587,0.114]}'
    ',"environment":{"macos_version":"27.2","numpy_version":"1.26.4","platform_ma'
    'chine":"arm64","platform_system":"Darwin","python_implementation":"CPython",'
    '"python_version":"3.11.14","torch_version":"2.15.0"},"evidence_boundary":{"a'
    'pple_gpu_metal_this_host":true,"cuda_kernel_executed":false,"distributed_cla'
    'im":false,"host_specific":true,"nvidia_gpu_used":false,"portable_performance'
    '_claim":false,"production_claim":false,"statement":"These are wall-clock tim'
    "ings on the Apple GPU of the host that made this report. The Metal kernels c"
    "ame from torch.mps.compile_shader, and their PyTorch equivalents ran on the "
    "MPS device. The timings are host-specific Apple GPU (Metal) measurements. No"
    " CUDA kernel ran, and no NVIDIA GPU was used. Effective bandwidth counts the"
    " bytes that each kernel must read and write, not the hardware traffic. These"
    " timings do not measure or predict CUDA, NVIDIA, distributed or production p"
    'erformance."},"kind":"kernel_metal_correctness_and_benchmark","label":"host-'
    'specific Apple GPU (Metal) measurement","measured":true,"measurement_scope":'
    '"apple_gpu_metal_this_host","percentile_method":"nearest rank: sorted sample'
    "s[ceil(p * n / 100) - 1] for p in 25, 50 and 75; median is p50 and IQR is p7"
    '5 - p25; no interpolation","planned_work":{"accounted_bytes":22428864,"bench'
    'mark_points":2,"calls":24,"correctness_cases":4,"timed_samples":12},"report_'
    'sha256":"bbed5f97fca4505a4eeb8b39903e152e2ece0128aa53f83b6ee5b9d3e4395231","'
    'run":{"elapsed_ns":1148263209,"max_runtime_ns":60000000000,"timed_ns":333383'
    '3},"schema_version":1,"shaders":{"contraction_probe":{"expression":"x[0] * x'
    '[1] + x[2]","fused":true,"fused_result":5.960464477539063e-08,"inputs":[1.00'
    '0244140625,1.000244140625,-1.00048828125],"observed":5.960464477539063e-08,"'
    'unfused_result":0.0},"kernels":[{"axes":1,"buffers":["out","a","b","n"],"byt'
    'es_per_item":12,"limits":{"max_threads_per_threadgroup":1024,"static_threadg'
    'roup_memory_bytes":0,"thread_execution_width":32},"name":"vector_add","scala'
    'r_buffers":[3]},{"axes":2,"buffers":["out","rgb","width","height"],"bytes_pe'
    'r_item":4,"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_m'
    'emory_bytes":0,"thread_execution_width":32},"name":"rgb_to_gray","scalar_buf'
    'fers":[2,3]}],"source":"#include <metal_stdlib>\\nusing namespace metal;\\n\\n/'
    "/ One thread per element. thread_position_in_grid is CUDA's\\n// blockIdx.x *"
    " blockDim.x + threadIdx.x. The grid is whole threadgroups,\\n// so the guard "
    "stops the padding threads past n.\\nkernel void vector_add(device float* out "
    "[[buffer(0)]],\\n                       device const float* a [[buffer(1)]],\\"
    "n                       device const float* b [[buffer(2)]],\\n              "
    "         constant uint& n [[buffer(3)]],\\n                       uint i [[th"
    "read_position_in_grid]]) {\\n    if (i < n) {\\n        out[i] = a[i] + b[i];\\"
    "n    }\\n}\\n\\n// One thread per pixel: x is the column and y the row, as in a"
    " CUDA 2D\\n// launch. Pixels are row-major RGB triples; out holds one level p"
    "er pixel.\\n// Exact rule: round the weighted sum half up, then clamp to 255."
    "\\nkernel void rgb_to_gray(device uchar* out [[buffer(0)]],\\n                "
    "        device const uchar* rgb [[buffer(1)]],\\n                        cons"
    "tant uint& width [[buffer(2)]],\\n                        constant uint& heig"
    "ht [[buffer(3)]],\\n                        uint2 position [[thread_position_"
    "in_grid]]) {\\n    uint column = position.x;\\n    uint row = position.y;\\n   "
    " if (column < width && row < height) {\\n        uint pixel = row * width + c"
    "olumn;\\n        uint r = rgb[3 * pixel];\\n        uint g = rgb[3 * pixel + 1"
    "];\\n        uint b = rgb[3 * pixel + 2];\\n        uint luma = (2989u * r + 5"
    "870u * g + 1140u * b + 5000u) / 10000u;\\n        out[pixel] = uchar(luma < 2"
    "55u ? luma : 255u);\\n    }\\n}\\n\\n// Fused multiply-add probe: 0 if the produ"
    "ct is rounded before the sum,\\n// 2**-24 if the compiler fuses them.\\nkernel"
    " void contraction_probe(device float* out [[buffer(0)]],\\n                  "
    "            device const float* x [[buffer(1)]],\\n                          "
    "    uint i [[thread_position_in_grid]]) {\\n    if (i == 0) {\\n        out[0]"
    ' = x[0] * x[1] + x[2];\\n    }\\n}\\n","source_sha256":"07892c3a5dcfb94cc8fb846'
    '52c64482aec1225f5ad68e16360e952be2e16cf76"},"summary":{"benchmark_points":2,'
    '"contraction_fused":true,"correctness_cases":4,"label":"host-specific Apple '
    'GPU (Metal) measurement","peak_metal_kernel_gb_per_second":10.53861306532663'
    '3,"rgb_to_gray_largest":{"height":480,"metal_kernel_gb_per_second":7.5177755'
    '51591859,"torch_gb_per_second":2.164378984392174,"torch_to_metal_time_ratio"'
    ':3.47340997385589,"width":641},"rgb_to_gray_max_abs_difference_levels":0,"rg'
    'b_to_gray_metal_mismatched_pixels":0,"timed_samples":12,"vector_add_exact_ca'
    'ses":2,"vector_add_largest":{"length":65537,"metal_kernel_gb_per_second":10.'
    '538613065326633,"torch_gb_per_second":9.789799957676173,"torch_to_metal_time'
    '_ratio":1.0764891122278057}},"timer":"time.perf_counter_ns"}'
)


class RecordedBackend:
    """A NumPy stand-in for ``MetalRuntime`` that returns what the GPU produced."""

    backend = "recorded"
    torch_version = RECORDED_TORCH_VERSION

    def __init__(self, limits=None):
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
        tail = uint8_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([rgb_to_gray_reference(rgb).reshape(-1), tail])

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


def test_the_recorded_metal_report_verifies_on_any_host():
    report = json.loads(RECORDED_REPORT_JSON)

    summary = verify_report(report)

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert summary["rgb_to_gray_metal_mismatched_pixels"] == 0
    assert report["environment"]["torch_version"] == RECORDED_TORCH_VERSION
    shaders = report["shaders"]
    assert shaders["contraction_probe"]["observed"] == RECORDED_CONTRACTION
    for kernel in shaders["kernels"]:
        assert kernel["limits"] == RECORDED_LIMITS.to_mapping()


def test_the_apple_gpu_matched_the_exact_rule_on_every_rgb_triple():
    report = json.loads(RECORDED_REPORT_JSON)
    triples = report["correctness"]["rgb_to_gray"][-1]
    exact = array_sha256(rgb_to_gray_reference(all_rgb_triples()))

    assert (triples["pattern"], triples["reference_sha256"]) == (
        "all_rgb_triples",
        exact,
    )
    for output in triples["outputs"]:
        assert (output["buffer_sha256"], output["mismatches"]) == (exact, [])
    assert triples["launch"]["padding_threads"] == 0
