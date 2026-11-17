"""Observations recorded on the development host's Apple GPU, replayed on any host.

``RecordedBackend`` stands in for ``MatmulMetalRuntime`` where MPS is missing.
Its naive kernels return the CPU replay of the fma chain with a sentinel
guard tail, which is what the GPU produced on 2026-11-17 (torch 2.15.0,
macOS 27.2, arm64): every naive output matched the replay bit for bit. Its
PyTorch outputs come from NumPy's float32 matmul, which the GPU did not
produce; they only need to stay within the bound. Reports built on it are
synthetic, never measurements. ``RECORDED_REPORT_JSON`` is one real measured
report from that host, so a host without torch or MPS can verify a genuine
Apple GPU report.
"""

import json

import numpy

from ai_journey.kernel_metal_geometry import CheckedLaunch, PipelineLimits
from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS, float32_sentinels
from ai_journey.matmul_metal_config import config_from_mapping
from ai_journey.matmul_metal_mapping import X_ROWS, matmul_launch, matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.matmul_metal_runtime import SIDES, MatmulWorkload, WorkloadCheck
from ai_journey.matmul_metal_source import KERNEL_FOR_MAPPING, KERNELS
from ai_journey.matmul_metal_verify import verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from test_kernel_metal_runtime import requires_metal
from test_matmul_metal_runtime import launches, matmul_runtime

# Both pipelines reported these limits.
RECORDED_LIMITS = PipelineLimits(
    max_threads_per_threadgroup=1024,
    thread_execution_width=32,
    static_threadgroup_memory_bytes=0,
)
RECORDED_TORCH_VERSION = "2.15.0"
RECORDED_MATMUL_PRECISION = "highest"
# The config of the recorded report: three correctness cases and two sweep
# points. Its target of 2**24 FLOPs gives one or two calls per sample, so the
# cost of each synchronize call sets most of its rates.
RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "matmul-metal-recorded",
    "seed": 74,
    "launch": {"group_size": [32, 8]},
    "correctness": {"shapes": [[1, 1, 1], [17, 33, 9], [64, 48, 300]]},
    "benchmark": {
        "shapes": [[256, 256, 256], [129, 65, 1000]],
        "warmup": 1,
        "repeats": 3,
        "target_flops_per_sample": 16777216,
        "max_runtime_seconds": 60,
    },
}
# The canonical file that measure_report wrote on the development host on
# 2026-11-17 under the shared GPU lock, at load averages near 8.2, 9.0 and
# 9.0 from other jobs. The timings describe that moment and are here only to
# be verified.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"agreement":"4 * gamma_K(2*'
    '*-24) * (|A| @ |B|), on the device in float32","ceiling":{"ceiling_index":1,"d'
    'efinition":"the highest median GFLOP/s of torch.matmul on the MPS device over '
    "this sweep, measured on this host in the same run; not a hardware or vendor pe"
    'ak","measured_ceiling_gflops_per_second":63.12413660710589,"naive":{"matmul_x_'
    'columns":{"best_index":0,"best_median_gflops_per_second":45.241910453073956,"f'
    'raction_of_measured_ceiling":0.7167133347845437},"matmul_x_rows":{"best_index"'
    ':0,"best_median_gflops_per_second":25.331268849674437,"fraction_of_measured_ce'
    'iling":0.40129291600992595}}},"device":"mps","flops_per_call":"2 * M * N * K: '
    'one multiply and one add per term of each output","gflops":"FLOPs of one sampl'
    'e divided by the median sample duration in ns","points":[{"access_steps":{"x_c'
    'olumns":{"a_step_bytes":{"x":0,"y":1024},"b_step_bytes":{"x":4,"y":0},"c_step_'
    'bytes":{"x":4,"y":1024},"mapping":"x_columns"},"x_rows":{"a_step_bytes":{"x":1'
    '024,"y":0},"b_step_bytes":{"x":0,"y":4},"c_step_bytes":{"x":1024,"y":4},"mappi'
    'ng":"x_rows"}},"agreement_bound":6.103608758678569e-05,"flops_per_call":335544'
    '32,"index":0,"k":256,"launches":{"x_columns":{"active_threads":65536,"extent":'
    '[256,256],"group_size":[32,8],"groups":[8,32],"idle_lanes_per_threadgroup":0,"'
    'launched_threads":65536,"padding_threads":0,"simd_groups_per_threadgroup":8,"t'
    'hreads":[256,256],"threads_per_group":256},"x_rows":{"active_threads":65536,"e'
    'xtent":[256,256],"group_size":[32,8],"groups":[8,32],"idle_lanes_per_threadgro'
    'up":0,"launched_threads":65536,"padding_threads":0,"simd_groups_per_threadgrou'
    'p":8,"threads":[256,256],"threads_per_group":256}},"m":256,"n":256,"naive_to_t'
    'orch_gflops_ratio":{"matmul_x_columns":0.7480904502964268,"matmul_x_rows":0.41'
    '886118712843257},"sides":{"matmul_x_columns":{"best_gflops_per_second":45.5103'
    '703824265,"calls_per_sample":1,"flops_per_sample":33554432,"implementation":"m'
    'atmul_x_columns","iqr_gflops_per_second":[45.16834191485781,45.5103703824265],'
    '"iqr_ns_per_call":5583.0,"median_gflops_per_second":45.241910453073956,"median'
    '_ns_per_call":741667.0,"repeats":3,"samples_ns":[742875,741667,737292],"timing'
    '":{"count":3,"iqr_ns":5583,"max_ns":742875,"median_ns":741667,"min_ns":737292,'
    '"p25_ns":737292,"p75_ns":742875},"warmup":1},"matmul_x_rows":{"best_gflops_per'
    '_second":25.66960278954624,"calls_per_sample":1,"flops_per_sample":33554432,"i'
    'mplementation":"matmul_x_rows","iqr_gflops_per_second":[25.081181283463632,25.'
    '66960278954624],"iqr_ns_per_call":30667.0,"median_gflops_per_second":25.331268'
    '849674437,"median_ns_per_call":1324625.0,"repeats":3,"samples_ns":[1307166,132'
    '4625,1337833],"timing":{"count":3,"iqr_ns":30667,"max_ns":1337833,"median_ns":'
    '1324625,"min_ns":1307166,"p25_ns":1307166,"p75_ns":1337833},"warmup":1},"torch'
    '_matmul":{"best_gflops_per_second":120.95304902006727,"calls_per_sample":1,"fl'
    'ops_per_sample":33554432,"implementation":"torch_matmul","iqr_gflops_per_secon'
    'd":[58.38518343228213,120.95304902006727],"iqr_ns_per_call":297291.0,"median_g'
    'flops_per_second":60.476524510033634,"median_ns_per_call":554834.0,"repeats":3'
    ',"samples_ns":[574708,554834,277417],"timing":{"count":3,"iqr_ns":297291,"max_'
    'ns":574708,"median_ns":554834,"min_ns":277417,"p25_ns":277417,"p75_ns":574708}'
    ',"warmup":1}},"torch_disagreement":0.0,"x_rows_to_x_columns_gflops_ratio":0.55'
    '99071435311881},{"access_steps":{"x_columns":{"a_step_bytes":{"x":0,"y":4000},'
    '"b_step_bytes":{"x":4,"y":0},"c_step_bytes":{"x":4,"y":260},"mapping":"x_colum'
    'ns"},"x_rows":{"a_step_bytes":{"x":4000,"y":0},"b_step_bytes":{"x":0,"y":4},"c'
    '_step_bytes":{"x":260,"y":4},"mapping":"x_rows"}},"agreement_bound":0.00023843'
    '279080336115,"flops_per_call":16770000,"index":1,"k":1000,"launches":{"x_colum'
    'ns":{"active_threads":8385,"extent":[65,129],"group_size":[32,8],"groups":[3,1'
    '7],"idle_lanes_per_threadgroup":0,"launched_threads":13056,"padding_threads":4'
    '671,"simd_groups_per_threadgroup":8,"threads":[96,136],"threads_per_group":256'
    '},"x_rows":{"active_threads":8385,"extent":[129,65],"group_size":[32,8],"group'
    's":[5,9],"idle_lanes_per_threadgroup":0,"launched_threads":11520,"padding_thre'
    'ads":3135,"simd_groups_per_threadgroup":8,"threads":[160,72],"threads_per_grou'
    'p":256}},"m":129,"n":65,"naive_to_torch_gflops_ratio":{"matmul_x_columns":0.38'
    '42125015456459,"matmul_x_rows":0.22736537069994356},"sides":{"matmul_x_columns'
    '":{"best_gflops_per_second":24.61049886523147,"calls_per_sample":2,"flops_per_'
    'sample":33540000,"implementation":"matmul_x_columns","iqr_gflops_per_second":['
    '22.915048299920677,24.61049886523147],"iqr_ns_per_call":50417.0,"median_gflops'
    '_per_second":24.253082433725233,"median_ns_per_call":691458.5,"repeats":3,"sam'
    'ples_ns":[1382917,1463667,1362833],"timing":{"count":3,"iqr_ns":100834,"max_ns'
    '":1463667,"median_ns":1382917,"min_ns":1362833,"p25_ns":1362833,"p75_ns":14636'
    '67},"warmup":1},"matmul_x_rows":{"best_gflops_per_second":14.763948497854077,"'
    'calls_per_sample":2,"flops_per_sample":33540000,"implementation":"matmul_x_row'
    's","iqr_gflops_per_second":[14.29109520749784,14.763948497854077],"iqr_ns_per_'
    'call":37583.0,"median_gflops_per_second":14.352242719788508,"median_ns_per_cal'
    'l":1168458.5,"repeats":3,"samples_ns":[2336917,2271750,2346916],"timing":{"cou'
    'nt":3,"iqr_ns":75166,"max_ns":2346916,"median_ns":2336917,"min_ns":2271750,"p2'
    '5_ns":2271750,"p75_ns":2346916},"warmup":1},"torch_matmul":{"best_gflops_per_s'
    'econd":84.88453021195824,"calls_per_sample":2,"flops_per_sample":33540000,"imp'
    'lementation":"torch_matmul","iqr_gflops_per_second":[49.56955477554037,84.8845'
    '3021195824],"iqr_ns_per_call":140750.0,"median_gflops_per_second":63.124136607'
    '10589,"median_ns_per_call":265667.0,"repeats":3,"samples_ns":[676625,531334,39'
    '5125],"timing":{"count":3,"iqr_ns":281500,"max_ns":676625,"median_ns":531334,"'
    'min_ns":395125,"p25_ns":395125,"p75_ns":676625},"warmup":1}},"torch_disagreeme'
    'nt":0.0,"x_rows_to_x_columns_gflops_ratio":0.5917698403494861}],"synchronizati'
    'on":"torch.mps.synchronize() before the first and before the second clock read'
    "ing of every sample; untimed warmups first; the three sides rotate and the ord"
    'er shifts by one side every repeat","torch_equivalent":"torch.matmul(a, b, out'
    '=c) on the MPS device, float32"},"config":{"benchmark":{"max_runtime_seconds":'
    '60,"repeats":3,"shapes":[[256,256,256],[129,65,1000]],"target_flops_per_sample'
    '":16777216,"warmup":1},"correctness":{"shapes":[[1,1,1],[17,33,9],[64,48,300]]'
    '},"launch":{"group_size":[32,8]},"name":"matmul-metal-recorded","schema_versio'
    'n":1,"seed":74},"config_sha256":"cdfad8333b006ac307772c9c4195549fddcf53684b9b4'
    '09dae5c0033e52a3344","correctness":{"bound":"(gamma_K(2**-24) + gamma_K(2**-53'
    ')) * (|A| @ |B|)","bound_source":"N. J. Higham, Accuracy and Stability of Nume'
    'rical Algorithms, 2nd ed., SIAM, 2002, section 3.1","cases":[{"a_sha256":"3053'
    '7cf63198fad07b5033aaa6e8e688c106b4937a7edd8559fa5e5be85af4fb","b_sha256":"987c'
    'a93d379ff1eca4d586d5bcd0ffa92f745bb06757c55a4bf19faaa1a71b4e","bound":5.960464'
    '843912682e-08,"index":0,"k":1,"launches":{"x_columns":{"active_threads":1,"ext'
    'ent":[1,1],"group_size":[32,8],"groups":[1,1],"idle_lanes_per_threadgroup":0,"'
    'launched_threads":256,"padding_threads":255,"simd_groups_per_threadgroup":8,"t'
    'hreads":[32,8],"threads_per_group":256},"x_rows":{"active_threads":1,"extent":'
    '[1,1],"group_size":[32,8],"groups":[1,1],"idle_lanes_per_threadgroup":0,"launc'
    'hed_threads":256,"padding_threads":255,"simd_groups_per_threadgroup":8,"thread'
    's":[32,8],"threads_per_group":256}},"m":1,"n":1,"outputs":[{"bound_ratio":0.51'
    '55942404037749,"buffer_sha256":"f12a4f22057a4569dc5c301c745e2f2b88818790f8e3f1'
    '0e7dc0334436383355","device":"mps","differing_elements":0,"guard_elements":255'
    ',"implementation":"matmul_x_columns","max_relative_error":3.073181343650564e-0'
    '8},{"bound_ratio":0.5155942404037749,"buffer_sha256":"f12a4f22057a4569dc5c301c'
    '745e2f2b88818790f8e3f10e7dc0334436383355","device":"mps","differing_elements":'
    '0,"guard_elements":255,"implementation":"matmul_x_rows","max_relative_error":3'
    '.073181343650564e-08},{"bound_ratio":0.5155942404037749,"buffer_sha256":"c1913'
    '79486fedbbc3670285734a47dca9ca4cf00935f00fe38571af72588e654","device":"mps","d'
    'iffering_elements":0,"guard_elements":0,"implementation":"torch_matmul","max_r'
    'elative_error":3.073181343650564e-08},{"bound_ratio":0.5155942404037749,"buffe'
    'r_sha256":"c191379486fedbbc3670285734a47dca9ca4cf00935f00fe38571af72588e654","'
    'device":"cpu","differing_elements":0,"guard_elements":0,"implementation":"torc'
    'h_matmul","max_relative_error":3.073181343650564e-08}],"reference_sha256":"b67'
    '4be6d99b3d8e95d1a3394f211418a34aebc100ea381ae5ef9db819b81d510","replay_sha256"'
    ':"c191379486fedbbc3670285734a47dca9ca4cf00935f00fe38571af72588e654"},{"a_sha25'
    '6":"94c79862a33b068c1d9ddbc97cf62badd4543db40b8b5dfdc15c1fa829350816","b_sha25'
    '6":"6c6ed4f9bfba3310e5939ce6b8a1bb4f0e284579f2aeeea20e35f740107e6d55","bound":'
    '5.364420917476787e-07,"index":1,"k":9,"launches":{"x_columns":{"active_threads'
    '":561,"extent":[33,17],"group_size":[32,8],"groups":[2,3],"idle_lanes_per_thre'
    'adgroup":0,"launched_threads":1536,"padding_threads":975,"simd_groups_per_thre'
    'adgroup":8,"threads":[64,24],"threads_per_group":256},"x_rows":{"active_thread'
    's":561,"extent":[17,33],"group_size":[32,8],"groups":[1,5],"idle_lanes_per_thr'
    'eadgroup":0,"launched_threads":1280,"padding_threads":719,"simd_groups_per_thr'
    'eadgroup":8,"threads":[32,40],"threads_per_group":256}},"m":17,"n":33,"outputs'
    '":[{"bound_ratio":0.20095336639332986,"buffer_sha256":"005e9513dff5d2e27c99201'
    '10cf5948558bb40428d171883f2924c167848b459","device":"mps","differing_elements"'
    ':0,"guard_elements":975,"implementation":"matmul_x_columns","max_relative_erro'
    'r":1.0779984421177555e-07},{"bound_ratio":0.20095336639332986,"buffer_sha256":'
    '"3c3e9d35c7d8c9a4d6d5d347ee63214a5fb67d8a4874941ad3025f33484f41ec","device":"m'
    'ps","differing_elements":0,"guard_elements":719,"implementation":"matmul_x_row'
    's","max_relative_error":1.0779984421177555e-07},{"bound_ratio":0.2009533663933'
    '2986,"buffer_sha256":"3b282a4121200a322b2e56328041b32cf50a6ffd2cefa1437e35ef6e'
    '8944795e","device":"mps","differing_elements":0,"guard_elements":0,"implementa'
    'tion":"torch_matmul","max_relative_error":1.0779984421177555e-07},{"bound_rati'
    'o":0.20095336639332986,"buffer_sha256":"3b282a4121200a322b2e56328041b32cf50a6f'
    'fd2cefa1437e35ef6e8944795e","device":"cpu","differing_elements":0,"guard_eleme'
    'nts":0,"implementation":"torch_matmul","max_relative_error":1.0779984421177555'
    'e-07}],"reference_sha256":"6f1f0815099f8204a9afb740fa126ed3c9c8dfc5ff0d5551c1b'
    'ed27eb7335fb3","replay_sha256":"3b282a4121200a322b2e56328041b32cf50a6ffd2cefa1'
    '437e35ef6e8944795e"},{"a_sha256":"28b37bd629388cfd2d68780431bb94f83aa700ee6e9b'
    '7ec098252c8f40efb688","b_sha256":"6d8bc29fd49946ac96cbc957df31ed88af0000efdfe0'
    '5153a917f46b5e2c7074","bound":1.7881713215872544e-05,"index":2,"k":300,"launch'
    'es":{"x_columns":{"active_threads":3072,"extent":[48,64],"group_size":[32,8],"'
    'groups":[2,8],"idle_lanes_per_threadgroup":0,"launched_threads":4096,"padding_'
    'threads":1024,"simd_groups_per_threadgroup":8,"threads":[64,64],"threads_per_g'
    'roup":256},"x_rows":{"active_threads":3072,"extent":[64,48],"group_size":[32,8'
    '],"groups":[2,6],"idle_lanes_per_threadgroup":0,"launched_threads":3072,"paddi'
    'ng_threads":0,"simd_groups_per_threadgroup":8,"threads":[64,48],"threads_per_g'
    'roup":256}},"m":64,"n":48,"outputs":[{"bound_ratio":0.01105381465078547,"buffe'
    'r_sha256":"942288386257433f422a962207002a247be2e9d46c56030f344949effc16bfe5","'
    'device":"mps","differing_elements":0,"guard_elements":1024,"implementation":"m'
    'atmul_x_columns","max_relative_error":1.9766114352675608e-07},{"bound_ratio":0'
    '.01105381465078547,"buffer_sha256":"5cdc6199f984e64b66f09fed1cb81dc8a0aa6d2c76'
    '124b74b722fe40d8c49a6d","device":"mps","differing_elements":0,"guard_elements"'
    ':0,"implementation":"matmul_x_rows","max_relative_error":1.9766114352675608e-0'
    '7},{"bound_ratio":0.01105381465078547,"buffer_sha256":"5cdc6199f984e64b66f09fe'
    'd1cb81dc8a0aa6d2c76124b74b722fe40d8c49a6d","device":"mps","differing_elements"'
    ':0,"guard_elements":0,"implementation":"torch_matmul","max_relative_error":1.9'
    '766114352675608e-07},{"bound_ratio":0.01105381465078547,"buffer_sha256":"5cdc6'
    '199f984e64b66f09fed1cb81dc8a0aa6d2c76124b74b722fe40d8c49a6d","device":"cpu","d'
    'iffering_elements":0,"guard_elements":0,"implementation":"torch_matmul","max_r'
    'elative_error":1.9766114352675608e-07}],"reference_sha256":"47c7c272f23f8c3fff'
    '2c41b9204280e44fe817bac381b18f68c2add84713b000","replay_sha256":"5cdc6199f984e'
    '64b66f09fed1cb81dc8a0aa6d2c76124b74b722fe40d8c49a6d"}],"gamma":"gamma_K(u) = K'
    ' * u / (1 - K * u)","reference":"float64 sum of the exact float32 products in '
    'increasing k order, on the CPU, rebuilt from the seed","relative_error":"max o'
    'ver elements of |C - R| / (|A| @ |B|), in float64","replay":"each naive output'
    " equals the CPU replay of `sum = fma(a[row * k + i], b[i * n + column], sum);`"
    ' for k = 0 to K - 1, one float32 rounding per step, bit for bit","sentinel_flo'
    'at32_bits":2143289344},"environment":{"macos_version":"27.2","numpy_version":"'
    '1.26.4","platform_machine":"arm64","platform_system":"Darwin","python_implemen'
    'tation":"CPython","python_version":"3.11.14","torch_float32_matmul_precision":'
    '"highest","torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this'
    '_host":true,"cuda_kernel_executed":false,"distributed_claim":false,"host_speci'
    'fic":true,"nvidia_gpu_used":false,"portable_performance_claim":false,"producti'
    'on_claim":false,"statement":"These are wall-clock timings on the Apple GPU of '
    "the host that made this report. The naive kernels came from torch.mps.compile_"
    "shader, and torch.matmul ran on the MPS device. The timings are host-specific "
    "Apple GPU (Metal) measurements. No CUDA kernel ran, and no NVIDIA GPU was used"
    ". The measured ceiling is the best median rate of torch.matmul in this run, no"
    "t a hardware or vendor peak. These timings do not measure or predict CUDA, NVI"
    'DIA, distributed or production performance.","vendor_peak_claimed":false},"kin'
    'd":"matmul_metal_correctness_and_gflops","label":"host-specific Apple GPU (Met'
    'al) measurement","measured":true,"measurement_scope":"apple_gpu_metal_this_hos'
    't","percentile_method":"nearest rank: sorted samples[ceil(p * n / 100) - 1] fo'
    'r p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no interpolation","p'
    'lanned_work":{"benchmark_points":2,"calls":36,"correctness_cases":3,"flops":80'
    '5133184,"replay_steps":926650,"timed_samples":18},"report_sha256":"0330edbda18'
    '69e074aa2cd62f80572d615e8e91a838230aa6abc2abbfdfad161","run":{"elapsed_ns":566'
    '337542,"max_runtime_ns":60000000000,"timed_ns":20366501},"schema_version":1,"s'
    'haders":{"kernels":[{"axes":2,"buffers":["c","a","b","m","n","k"],"limits":{"m'
    'ax_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_e'
    'xecution_width":32},"mapping":"x_columns","name":"matmul_x_columns","scalar_bu'
    'ffers":[3,4,5]},{"axes":2,"buffers":["c","a","b","m","n","k"],"limits":{"max_t'
    'hreads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execu'
    'tion_width":32},"mapping":"x_rows","name":"matmul_x_rows","scalar_buffers":[3,'
    '4,5]}],"source":"#include <metal_stdlib>\\nusing namespace metal;\\n\\n// One '
    "thread per element of C = A @ B; all three are row-major float32.\\n// x is th"
    "e column of C and y its row, so threads that are consecutive\\n// in x read co"
    "nsecutive elements of B. The grid is whole threadgroups,\\n// so the guard sto"
    "ps the padding threads past n and past m.\\nkernel void matmul_x_columns(devic"
    "e float* c [[buffer(0)]],\\n                             device const float* a"
    " [[buffer(1)]],\\n                             device const float* b [[buffer("
    "2)]],\\n                             constant uint& m [[buffer(3)]],\\n       "
    "                      constant uint& n [[buffer(4)]],\\n                      "
    "       constant uint& k [[buffer(5)]],\\n                             uint2 po"
    "sition [[thread_position_in_grid]]) {\\n    uint column = position.x;\\n    ui"
    "nt row = position.y;\\n    if (row < m && column < n) {\\n        float sum = "
    "0.0f;\\n        for (uint i = 0; i < k; ++i) {\\n            sum = fma(a[row *"
    " k + i], b[i * n + column], sum);\\n        }\\n        c[row * n + column] = "
    "sum;\\n    }\\n}\\n\\n// The same guard, loop and store with x as the row of C"
    " and y as its\\n// column, so threads that are consecutive in x read elements "
    "of A\\n// that lie k floats apart.\\nkernel void matmul_x_rows(device float* c"
    " [[buffer(0)]],\\n                          device const float* a [[buffer(1)]"
    "],\\n                          device const float* b [[buffer(2)]],\\n        "
    "                  constant uint& m [[buffer(3)]],\\n                          "
    "constant uint& n [[buffer(4)]],\\n                          constant uint& k ["
    "[buffer(5)]],\\n                          uint2 position [[thread_position_in_"
    "grid]]) {\\n    uint row = position.x;\\n    uint column = position.y;\\n    i"
    "f (row < m && column < n) {\\n        float sum = 0.0f;\\n        for (uint i "
    "= 0; i < k; ++i) {\\n            sum = fma(a[row * k + i], b[i * n + column], "
    'sum);\\n        }\\n        c[row * n + column] = sum;\\n    }\\n}\\n","source'
    '_sha256":"beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc"},"'
    'summary":{"benchmark_points":2,"correctness_cases":3,"label":"host-specific Ap'
    'ple GPU (Metal) measurement","largest_point":{"k":256,"m":256,"median_gflops_p'
    'er_second":{"matmul_x_columns":45.241910453073956,"matmul_x_rows":25.331268849'
    '674437,"torch_matmul":60.476524510033634},"n":256,"naive_to_torch_gflops_ratio'
    '":{"matmul_x_columns":0.7480904502964268,"matmul_x_rows":0.41886118712843257}}'
    ',"max_bound_ratio":0.5155942404037749,"max_relative_error":{"matmul_x_columns/'
    'mps":1.9766114352675608e-07,"matmul_x_rows/mps":1.9766114352675608e-07,"torch_'
    'matmul/cpu":1.9766114352675608e-07,"torch_matmul/mps":1.9766114352675608e-07},'
    '"measured_ceiling_gflops_per_second":63.12413660710589,"naive_fraction_of_meas'
    'ured_ceiling":{"matmul_x_columns":0.7167133347845437,"matmul_x_rows":0.4012929'
    '1600992595},"naive_outputs_replay_exactly":true,"timed_samples":18,"torch_case'
    's_with_other_bits":{"torch_matmul/cpu":0,"torch_matmul/mps":0}},"timer":"time.'
    'perf_counter_ns"}'
)


class RecordedBackend:
    """A NumPy stand-in for ``MatmulMetalRuntime``; its reports are synthetic."""

    backend = "recorded"
    torch_version = RECORDED_TORCH_VERSION
    float32_matmul_precision = RECORDED_MATMUL_PRECISION

    def __init__(self, limits=None):
        self.limits = {name: limits or RECORDED_LIMITS for name in KERNELS}
        self.launches = []

    def synchronize(self):
        pass

    def run(self, name, left, right, launch):
        self.launches.append((name, launch))
        tail = float32_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([fma_replay(left, right).reshape(-1), tail])

    def torch_matmul(self, left, right, device):
        return left @ right

    def workload(self, left, right, launches):
        self.launches.extend(launches.items())

        def side():
            return left @ right

        return MatmulWorkload(
            sides={name: side for name in SIDES},
            synchronize=self.synchronize,
            check=lambda: WorkloadCheck(True, 0.0),
        )


def test_the_stand_in_returns_the_replay_with_a_sentinel_tail():
    backend = RecordedBackend()
    shape = matmul_shape(17, 33, 9)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 0, shape)
    name = KERNEL_FOR_MAPPING[X_ROWS]
    launch = CheckedLaunch(matmul_launch(X_ROWS, shape, (32, 8)), RECORDED_LIMITS)

    buffer = backend.run(name, left, right, launch)

    replay = fma_replay(left, right).reshape(-1)
    assert numpy.array_equal(buffer[: replay.size], replay)
    tail = buffer[replay.size :].view(numpy.uint32)
    assert tail.size == launch.geometry.padding_threads == 32 * 40 - 17 * 33
    assert set(tail.tolist()) == {FLOAT32_SENTINEL_BITS}
    assert backend.launches == [(name, launch)]


@requires_metal
def test_the_stand_in_matches_this_gpu_bit_for_bit():
    runtime = matmul_runtime()
    backend = RecordedBackend()
    shape = matmul_shape(31, 47, 257)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 5, shape)

    for name, launch in launches(runtime, shape).items():
        expected = runtime.run(name, left, right, launch).view(numpy.uint32)
        recorded = backend.run(name, left, right, launch).view(numpy.uint32)
        assert numpy.array_equal(recorded, expected), name


def test_the_recorded_matmul_report_verifies_on_any_host():
    report = json.loads(RECORDED_REPORT_JSON)

    summary = verify_report(report, config=config_from_mapping(RECORDED_CONFIG))

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    environment = report["environment"]
    assert environment["torch_version"] == RECORDED_TORCH_VERSION
    assert environment["torch_float32_matmul_precision"] == RECORDED_MATMUL_PRECISION
    for kernel in report["shaders"]["kernels"]:
        assert kernel["limits"] == RECORDED_LIMITS.to_mapping()


def test_the_apple_gpu_held_the_replay_bits_in_every_recorded_case():
    report = json.loads(RECORDED_REPORT_JSON)

    for case in report["correctness"]["cases"]:
        shape = matmul_shape(case["m"], case["n"], case["k"])
        left, right = matmul_inputs(74, CORRECTNESS_STREAM, case["index"], shape)
        replay = fma_replay(left, right)
        assert case["replay_sha256"] == matrix_sha256(replay)
        for output in case["outputs"][:2]:
            tail = float32_sentinels(output["guard_elements"])
            buffer = numpy.concatenate([replay.reshape(-1), tail])
            assert output["buffer_sha256"] == matrix_sha256(buffer)
            assert output["differing_elements"] == 0
