"""One real tiled matmul report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``matmul_tiled_cli run``
wrote in a fresh process on 2026-11-19 at 11:41 IST under the shared GPU
lock, with torch 2.15.0 on macOS 27.2 (arm64) and load averages near 11.0,
12.7 and 14.7 from other jobs. The run is short (three correctness cases, two
sweep points, three repeats and one or two calls per sample), so the file
stays small and the cost of each synchronize call sets most of its rates.
The timings describe that moment and are here only to be verified. A host
without torch or MPS can still check a genuine Apple GPU (Metal) report, and
the Linux CI does.
"""

import json

import numpy

from ai_journey.kernel_metal_reference import float32_sentinels
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.matmul_tiled_config import config_from_mapping
from ai_journey.matmul_tiled_variants import VARIANTS
from ai_journey.matmul_tiled_verify import verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from test_kernel_metal_runtime import requires_metal
from test_matmul_tiled_runtime import tiled_runtime

RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "matmul-tiled-recorded",
    "seed": 76,
    "naive": {"group_size": [32, 8]},
    "variants": ["matmul_tiled_8", "matmul_tiled_16", "matmul_tiled_32"],
    "correctness": {"shapes": [[1, 1, 1], [17, 33, 9], [31, 47, 129]]},
    "benchmark": {
        "shapes": [[256, 256, 256], [200, 300, 100]],
        "warmup": 1,
        "repeats": 3,
        "target_flops_per_sample": 16777216,
        "max_runtime_seconds": 60,
    },
    "comparison": {"min_faster_ratio": 1.1},
}
# The canonical report of that run, byte for byte.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"agreement":"every side wit'
    "hin 4 * gamma_K(2**-24) * (|A| @ |B|) of the naive output, on the device in fl"
    'oat32","device":"mps","flops_per_call":"2 * M * N * K: one multiply and one ad'
    'd per term of each output","gflops":"FLOPs of one sample divided by the median'
    ' sample duration in ns","loads_per_output":"analytical global element loads pe'
    "r output element: 2 * K for the naive kernel; useful tile loads / (M * N) for "
    'a T x T tile, from memory_access_tiling, with no cache modeled","points":[{"ag'
    'reement_bound":6.103608758678569e-05,"checks":{"matmul_tiled_16":{"disagreemen'
    't":0.0,"same_bits_as_naive":true},"matmul_tiled_32":{"disagreement":0.0,"same_'
    'bits_as_naive":true},"matmul_tiled_8":{"disagreement":0.0,"same_bits_as_naive"'
    ':true},"torch_matmul":{"disagreement":0.0,"same_bits_as_naive":true}},"flops_p'
    'er_call":33554432,"global_loads_per_output":{"matmul_tiled_16":32.0,"matmul_ti'
    'led_32":16.0,"matmul_tiled_8":64.0,"matmul_x_columns":512.0},"index":0,"k":256'
    ',"launches":{"matmul_tiled_16":{"active_threads":65536,"extent":[256,256],"gro'
    'up_size":[16,16],"groups":[16,16],"idle_lanes_per_threadgroup":0,"launched_thr'
    'eads":65536,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[256'
    ',256],"threads_per_group":256},"matmul_tiled_32":{"active_threads":65536,"exte'
    'nt":[256,256],"group_size":[32,32],"groups":[8,8],"idle_lanes_per_threadgroup"'
    ':0,"launched_threads":65536,"padding_threads":0,"simd_groups_per_threadgroup":'
    '32,"threads":[256,256],"threads_per_group":1024},"matmul_tiled_8":{"active_thr'
    'eads":65536,"extent":[256,256],"group_size":[8,8],"groups":[32,32],"idle_lanes'
    '_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"simd_groups_'
    'per_threadgroup":2,"threads":[256,256],"threads_per_group":64},"matmul_x_colum'
    'ns":{"active_threads":65536,"extent":[256,256],"group_size":[32,8],"groups":[8'
    ',32],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"padding_threads"'
    ':0,"simd_groups_per_threadgroup":8,"threads":[256,256],"threads_per_group":256'
    '}},"m":256,"n":256,"side_order":["matmul_x_columns","matmul_tiled_8","matmul_t'
    'iled_16","matmul_tiled_32","torch_matmul"],"sides":{"matmul_tiled_16":{"best_g'
    'flops_per_second":52.56571726669309,"calls_per_sample":1,"flops_per_sample":33'
    '554432,"implementation":"matmul_tiled_16","iqr_gflops_per_second":[51.64532429'
    '1336586,52.56571726669309],"iqr_ns_per_call":11376.0,"median_gflops_per_second'
    '":51.90167362722351,"median_ns_per_call":646500.0,"repeats":3,"samples_ns":[64'
    '6500,638333,649709],"timing":{"count":3,"iqr_ns":11376,"max_ns":649709,"median'
    '_ns":646500,"min_ns":638333,"p25_ns":638333,"p75_ns":649709},"warmup":1},"matm'
    'ul_tiled_32":{"best_gflops_per_second":55.86195346173504,"calls_per_sample":1,'
    '"flops_per_sample":33554432,"implementation":"matmul_tiled_32","iqr_gflops_per'
    '_second":[48.219050835279326,55.86195346173504],"iqr_ns_per_call":95208.0,"med'
    'ian_gflops_per_second":54.756731087760755,"median_ns_per_call":612791.0,"repea'
    'ts":3,"samples_ns":[612791,695875,600667],"timing":{"count":3,"iqr_ns":95208,"'
    'max_ns":695875,"median_ns":612791,"min_ns":600667,"p25_ns":600667,"p75_ns":695'
    '875},"warmup":1},"matmul_tiled_8":{"best_gflops_per_second":47.61466025644664,'
    '"calls_per_sample":1,"flops_per_sample":33554432,"implementation":"matmul_tile'
    'd_8","iqr_gflops_per_second":[34.957085037114204,47.61466025644664],"iqr_ns_pe'
    'r_call":255167.0,"median_gflops_per_second":46.98677682478558,"median_ns_per_c'
    'all":714125.0,"repeats":3,"samples_ns":[714125,704708,959875],"timing":{"count'
    '":3,"iqr_ns":255167,"max_ns":959875,"median_ns":714125,"min_ns":704708,"p25_ns'
    '":704708,"p75_ns":959875},"warmup":1},"matmul_x_columns":{"best_gflops_per_sec'
    'ond":38.80619012868538,"calls_per_sample":1,"flops_per_sample":33554432,"imple'
    'mentation":"matmul_x_columns","iqr_gflops_per_second":[37.260263263678986,38.8'
    '0619012868538],"iqr_ns_per_call":35875.0,"median_gflops_per_second":37.6963145'
    '6256144,"median_ns_per_call":890125.0,"repeats":3,"samples_ns":[890125,900542,'
    '864667],"timing":{"count":3,"iqr_ns":35875,"max_ns":900542,"median_ns":890125,'
    '"min_ns":864667,"p25_ns":864667,"p75_ns":900542},"warmup":1},"torch_matmul":{"'
    'best_gflops_per_second":94.17677635198096,"calls_per_sample":1,"flops_per_samp'
    'le":33554432,"implementation":"torch_matmul","iqr_gflops_per_second":[35.68671'
    '31082159,94.17677635198096],"iqr_ns_per_call":583958.0,"median_gflops_per_seco'
    'nd":54.932251531924436,"median_ns_per_call":610833.0,"repeats":3,"samples_ns":'
    '[356292,940250,610833],"timing":{"count":3,"iqr_ns":583958,"max_ns":940250,"me'
    'dian_ns":610833,"min_ns":356292,"p25_ns":356292,"p75_ns":940250},"warmup":1}}}'
    ',{"agreement_bound":2.384200001955044e-05,"checks":{"matmul_tiled_16":{"disagr'
    'eement":0.0,"same_bits_as_naive":true},"matmul_tiled_32":{"disagreement":0.0,"'
    'same_bits_as_naive":true},"matmul_tiled_8":{"disagreement":0.0,"same_bits_as_n'
    'aive":true},"torch_matmul":{"disagreement":0.0,"same_bits_as_naive":true}},"fl'
    'ops_per_call":12000000,"global_loads_per_output":{"matmul_tiled_16":12.8333333'
    '33333334,"matmul_tiled_32":6.833333333333333,"matmul_tiled_8":25.1666666666666'
    '68,"matmul_x_columns":200.0},"index":1,"k":100,"launches":{"matmul_tiled_16":{'
    '"active_threads":60000,"extent":[300,200],"group_size":[16,16],"groups":[19,13'
    '],"idle_lanes_per_threadgroup":0,"launched_threads":63232,"padding_threads":32'
    '32,"simd_groups_per_threadgroup":8,"threads":[304,208],"threads_per_group":256'
    '},"matmul_tiled_32":{"active_threads":60000,"extent":[300,200],"group_size":[3'
    '2,32],"groups":[10,7],"idle_lanes_per_threadgroup":0,"launched_threads":71680,'
    '"padding_threads":11680,"simd_groups_per_threadgroup":32,"threads":[320,224],"'
    'threads_per_group":1024},"matmul_tiled_8":{"active_threads":60000,"extent":[30'
    '0,200],"group_size":[8,8],"groups":[38,25],"idle_lanes_per_threadgroup":0,"lau'
    'nched_threads":60800,"padding_threads":800,"simd_groups_per_threadgroup":2,"th'
    'reads":[304,200],"threads_per_group":64},"matmul_x_columns":{"active_threads":'
    '60000,"extent":[300,200],"group_size":[32,8],"groups":[10,25],"idle_lanes_per_'
    'threadgroup":0,"launched_threads":64000,"padding_threads":4000,"simd_groups_pe'
    'r_threadgroup":8,"threads":[320,200],"threads_per_group":256}},"m":200,"n":300'
    ',"side_order":["matmul_x_columns","matmul_tiled_8","matmul_tiled_16","matmul_t'
    'iled_32","torch_matmul"],"sides":{"matmul_tiled_16":{"best_gflops_per_second":'
    '34.623690972016846,"calls_per_sample":2,"flops_per_sample":24000000,"implement'
    'ation":"matmul_tiled_16","iqr_gflops_per_second":[32.2328068193875,34.62369097'
    '2016846],"iqr_ns_per_call":25708.0,"median_gflops_per_second":33.5918976342906'
    '1,"median_ns_per_call":357229.0,"repeats":3,"samples_ns":[693167,744583,714458'
    '],"timing":{"count":3,"iqr_ns":51416,"max_ns":744583,"median_ns":714458,"min_n'
    's":693167,"p25_ns":693167,"p75_ns":744583},"warmup":1},"matmul_tiled_32":{"bes'
    't_gflops_per_second":41.39415514529348,"calls_per_sample":2,"flops_per_sample"'
    ':24000000,"implementation":"matmul_tiled_32","iqr_gflops_per_second":[30.95775'
    '55627217,41.39415514529348],"iqr_ns_per_call":97729.0,"median_gflops_per_secon'
    'd":32.241813602015114,"median_ns_per_call":372187.5,"repeats":3,"samples_ns":['
    '579792,775250,744375],"timing":{"count":3,"iqr_ns":195458,"max_ns":775250,"med'
    'ian_ns":744375,"min_ns":579792,"p25_ns":579792,"p75_ns":775250},"warmup":1},"m'
    'atmul_tiled_8":{"best_gflops_per_second":44.427249453452276,"calls_per_sample"'
    ':2,"flops_per_sample":24000000,"implementation":"matmul_tiled_8","iqr_gflops_p'
    'er_second":[28.016926893331387,44.427249453452276],"iqr_ns_per_call":158208.0,'
    '"median_gflops_per_second":28.137343061038102,"median_ns_per_call":426479.5,"r'
    'epeats":3,"samples_ns":[856625,852959,540209],"timing":{"count":3,"iqr_ns":316'
    '416,"max_ns":856625,"median_ns":852959,"min_ns":540209,"p25_ns":540209,"p75_ns'
    '":856625},"warmup":1},"matmul_x_columns":{"best_gflops_per_second":48.70218814'
    '872837,"calls_per_sample":2,"flops_per_sample":24000000,"implementation":"matm'
    'ul_x_columns","iqr_gflops_per_second":[31.398201144726084,48.70218814872837],"'
    'iqr_ns_per_call":135792.0,"median_gflops_per_second":36.63188164239041,"median'
    '_ns_per_call":327583.5,"repeats":3,"samples_ns":[764375,655167,492791],"timing'
    '":{"count":3,"iqr_ns":271584,"max_ns":764375,"median_ns":655167,"min_ns":49279'
    '1,"p25_ns":492791,"p75_ns":764375},"warmup":1},"torch_matmul":{"best_gflops_pe'
    'r_second":68.73501791406404,"calls_per_sample":2,"flops_per_sample":24000000,"'
    'implementation":"torch_matmul","iqr_gflops_per_second":[41.409010600706715,68.'
    '73501791406404],"iqr_ns_per_call":115208.5,"median_gflops_per_second":64.53781'
    '512605042,"median_ns_per_call":185937.5,"repeats":3,"samples_ns":[349167,37187'
    '5,579584],"timing":{"count":3,"iqr_ns":230417,"max_ns":579584,"median_ns":3718'
    '75,"min_ns":349167,"p25_ns":349167,"p75_ns":579584},"warmup":1}}}],"synchroniz'
    'ation":"torch.mps.synchronize() before the first and before the second clock r'
    "eading of every sample; untimed warmups of each side first; repeat r starts wi"
    'th side r mod S, so the order shifts by one side every repeat","torch_equivale'
    'nt":"torch.matmul(a, b, out=c) on the MPS device, float32"},"comparison":{"cub'
    'las":"not measured, no NVIDIA GPU or cuBLAS on this host","min_faster_ratio":1'
    '.1,"naive":"naive GPU matmul: matmul_x_columns, the Day 74 kernel with x over '
    'the columns of C","points":[{"cublas":"not measured, no NVIDIA GPU or cuBLAS o'
    'n this host","index":0,"k":256,"m":256,"n":256,"naive":{"iqr_gflops_per_second'
    '":[37.260263263678986,38.80619012868538],"median_gflops_per_second":37.6963145'
    '6256144,"side":"matmul_x_columns"},"tiled":{"iqr_gflops_per_second":[48.219050'
    '835279326,55.86195346173504],"median_gflops_per_second":54.756731087760755,"si'
    'de":"matmul_tiled_32","tile":32},"tiled_over_naive":{"first":"matmul_tiled_32"'
    ',"iqr_separated":true,"ratio":1.4525751846877646,"second":"matmul_x_columns","'
    'verdict":"faster"},"vendor":{"iqr_gflops_per_second":[35.6867131082159,94.1767'
    '7635198096],"median_gflops_per_second":54.932251531924436,"side":"torch_matmul'
    '"},"vendor_over_tiled":{"first":"torch_matmul","iqr_separated":false,"ratio":1'
    '.003205458775148,"second":"matmul_tiled_32","verdict":"no_clear_difference"}},'
    '{"cublas":"not measured, no NVIDIA GPU or cuBLAS on this host","index":1,"k":1'
    '00,"m":200,"n":300,"naive":{"iqr_gflops_per_second":[31.398201144726084,48.702'
    '18814872837],"median_gflops_per_second":36.63188164239041,"side":"matmul_x_col'
    'umns"},"tiled":{"iqr_gflops_per_second":[32.2328068193875,34.623690972016846],'
    '"median_gflops_per_second":33.59189763429061,"side":"matmul_tiled_16","tile":1'
    '6},"tiled_over_naive":{"first":"matmul_tiled_16","iqr_separated":false,"ratio"'
    ':0.9170126165568866,"second":"matmul_x_columns","verdict":"no_clear_difference'
    '"},"vendor":{"iqr_gflops_per_second":[41.409010600706715,68.73501791406404],"m'
    'edian_gflops_per_second":64.53781512605042,"side":"torch_matmul"},"vendor_over'
    '_tiled":{"first":"torch_matmul","iqr_separated":true,"ratio":1.921231596638655'
    '3,"second":"matmul_tiled_16","verdict":"faster"}}],"ratio_rule":"ratio = media'
    "n GFLOP/s of the first side / median GFLOP/s of the second, from the same poin"
    "t of the same run; faster when ratio >= min_faster_ratio and the first side's "
    "p75 sample time is below the second side's p25; slower when 1 / ratio >= min_f"
    "aster_ratio and the second side's p75 is below the first side's p25; otherwise"
    ' no_clear_difference","tiled_rule":"the admitted tiled variant with the highes'
    "t median GFLOP/s at this point in this run; a tie goes to the variant listed f"
    'irst in the config","vendor":"vendor GPU matmul: torch.matmul on MPS, PyTorch '
    '2.15.0"},"config":{"benchmark":{"max_runtime_seconds":60,"repeats":3,"shapes":'
    '[[256,256,256],[200,300,100]],"target_flops_per_sample":16777216,"warmup":1},"'
    'comparison":{"min_faster_ratio":1.1},"correctness":{"shapes":[[1,1,1],[17,33,9'
    '],[31,47,129]]},"naive":{"group_size":[32,8]},"name":"matmul-tiled-recorded","'
    'schema_version":1,"seed":76,"variants":["matmul_tiled_8","matmul_tiled_16","ma'
    'tmul_tiled_32"]},"config_sha256":"6a55c69ef00c1dd51f600b90b0c827b40bf4df0492ff'
    '58faf9206c581a6b23bc","correctness":{"bound":"(gamma_K(2**-24) + gamma_K(2**-5'
    '3)) * (|A| @ |B|)","bound_source":"N. J. Higham, Accuracy and Stability of Num'
    'erical Algorithms, 2nd ed., SIAM, 2002, section 3.1","cases":[{"a_sha256":"ff3'
    '709807ab673262ebc83544cbfbf144f51ba321489ee72208ecbf278ea1ff4","b_sha256":"924'
    'd4d11ca19d9d89d963af3d9cf2bb6895e4e2ba823b649213bd28bb0129732","bound":5.96046'
    '4843912682e-08,"index":0,"k":1,"launches":{"matmul_tiled_16":{"active_threads"'
    ':1,"extent":[1,1],"group_size":[16,16],"groups":[1,1],"idle_lanes_per_threadgr'
    'oup":0,"launched_threads":256,"padding_threads":255,"simd_groups_per_threadgro'
    'up":8,"threads":[16,16],"threads_per_group":256},"matmul_tiled_32":{"active_th'
    'reads":1,"extent":[1,1],"group_size":[32,32],"groups":[1,1],"idle_lanes_per_th'
    'readgroup":0,"launched_threads":1024,"padding_threads":1023,"simd_groups_per_t'
    'hreadgroup":32,"threads":[32,32],"threads_per_group":1024},"matmul_tiled_8":{"'
    'active_threads":1,"extent":[1,1],"group_size":[8,8],"groups":[1,1],"idle_lanes'
    '_per_threadgroup":0,"launched_threads":64,"padding_threads":63,"simd_groups_pe'
    'r_threadgroup":2,"threads":[8,8],"threads_per_group":64},"matmul_x_columns":{"'
    'active_threads":1,"extent":[1,1],"group_size":[32,8],"groups":[1,1],"idle_lane'
    's_per_threadgroup":0,"launched_threads":256,"padding_threads":255,"simd_groups'
    '_per_threadgroup":8,"threads":[32,8],"threads_per_group":256}},"m":1,"n":1,"ou'
    'tputs":[{"bound_ratio":0.45706416418859963,"buffer_sha256":"ba0c7af42121363c79'
    '59ffa662973ea4cbcdeb46bdb5cb5d8742a7b4ed1ae05f","correctness_mode":"k_order_fm'
    'a","differing_elements":0,"guard_elements":255,"kernel":"matmul_x_columns","ma'
    'x_relative_error":2.724314882058482e-08,"replay_bits":true},{"bound_ratio":0.4'
    '5706416418859963,"buffer_sha256":"235f1aa44c8080f765560f71e3e755b66c2c784c3b06'
    'f51b316d946f1b7090a7","correctness_mode":"k_order_fma","differing_elements":0,'
    '"guard_elements":63,"kernel":"matmul_tiled_8","max_relative_error":2.724314882'
    '058482e-08,"replay_bits":true},{"bound_ratio":0.45706416418859963,"buffer_sha2'
    '56":"ba0c7af42121363c7959ffa662973ea4cbcdeb46bdb5cb5d8742a7b4ed1ae05f","correc'
    'tness_mode":"k_order_fma","differing_elements":0,"guard_elements":255,"kernel"'
    ':"matmul_tiled_16","max_relative_error":2.724314882058482e-08,"replay_bits":tr'
    'ue},{"bound_ratio":0.45706416418859963,"buffer_sha256":"1e5ce0853b0011b39ce1a1'
    'b95ed868728714015d607c41c1ed580691d293884b","correctness_mode":"k_order_fma","'
    'differing_elements":0,"guard_elements":1023,"kernel":"matmul_tiled_32","max_re'
    'lative_error":2.724314882058482e-08,"replay_bits":true}],"reference_sha256":"b'
    'c2dbfbf85137bd9cd522701974eb534160b5b3701756c53c403025f790208cb","replay_sha25'
    '6":"dc38f12978f4bd1b1c7007df1e1db1aa9eee300fa6abe4014ad8263e01816949"},{"a_sha'
    '256":"4b99da21b682417039f00553291abb6e59c081217061c6e74bd7bc7ca96f1e02","b_sha'
    '256":"7f35f7922c92f1b2124131565a66e4e68521ba023c40e4e5c3ef066bbef20934","bound'
    '":5.364420917476787e-07,"index":1,"k":9,"launches":{"matmul_tiled_16":{"active'
    '_threads":561,"extent":[33,17],"group_size":[16,16],"groups":[3,2],"idle_lanes'
    '_per_threadgroup":0,"launched_threads":1536,"padding_threads":975,"simd_groups'
    '_per_threadgroup":8,"threads":[48,32],"threads_per_group":256},"matmul_tiled_3'
    '2":{"active_threads":561,"extent":[33,17],"group_size":[32,32],"groups":[2,1],'
    '"idle_lanes_per_threadgroup":0,"launched_threads":2048,"padding_threads":1487,'
    '"simd_groups_per_threadgroup":32,"threads":[64,32],"threads_per_group":1024},"'
    'matmul_tiled_8":{"active_threads":561,"extent":[33,17],"group_size":[8,8],"gro'
    'ups":[5,3],"idle_lanes_per_threadgroup":0,"launched_threads":960,"padding_thre'
    'ads":399,"simd_groups_per_threadgroup":2,"threads":[40,24],"threads_per_group"'
    ':64},"matmul_x_columns":{"active_threads":561,"extent":[33,17],"group_size":[3'
    '2,8],"groups":[2,3],"idle_lanes_per_threadgroup":0,"launched_threads":1536,"pa'
    'dding_threads":975,"simd_groups_per_threadgroup":8,"threads":[64,24],"threads_'
    'per_group":256}},"m":17,"n":33,"outputs":[{"bound_ratio":0.16777678089432368,"'
    'buffer_sha256":"b4c2db2955059f13f499d0d4acba39d13034c30b60fba4635345a272a274c8'
    'a3","correctness_mode":"k_order_fma","differing_elements":0,"guard_elements":9'
    '75,"kernel":"matmul_x_columns","max_relative_error":9.000252728964298e-08,"rep'
    'lay_bits":true},{"bound_ratio":0.16777678089432368,"buffer_sha256":"c2cde3295e'
    '846ec25cd67c584c343f6f18a5ce1ab959e416b17d1b64cdacaf9c","correctness_mode":"k_'
    'order_fma","differing_elements":0,"guard_elements":399,"kernel":"matmul_tiled_'
    '8","max_relative_error":9.000252728964298e-08,"replay_bits":true},{"bound_rati'
    'o":0.16777678089432368,"buffer_sha256":"b4c2db2955059f13f499d0d4acba39d13034c3'
    '0b60fba4635345a272a274c8a3","correctness_mode":"k_order_fma","differing_elemen'
    'ts":0,"guard_elements":975,"kernel":"matmul_tiled_16","max_relative_error":9.0'
    '00252728964298e-08,"replay_bits":true},{"bound_ratio":0.16777678089432368,"buf'
    'fer_sha256":"3df109900d9a1ea50ad45a5e30e7c0348ace96eaec7ac11cb204b6ad6752984a"'
    ',"correctness_mode":"k_order_fma","differing_elements":0,"guard_elements":1487'
    ',"kernel":"matmul_tiled_32","max_relative_error":9.000252728964298e-08,"replay'
    '_bits":true}],"reference_sha256":"f0b7e8fb39bec17530e36e8b61ac10c1e4e4f34d7b16'
    '67a32715370cb287615f","replay_sha256":"5dad842fa42f31911fc7cfd7726ca7d0fe6f096'
    '9d3d08dc845c7c93ab383ca62"},{"a_sha256":"7eb0d278614b973064151e3af1e77d8314c26'
    '76037f9d8fa022129649e3b784c","b_sha256":"6ce517ae773bca5acf6d70e36fa412d99dd15'
    'b543d95a704c42edab993ac0821","bound":7.689058311510179e-06,"index":2,"k":129,"'
    'launches":{"matmul_tiled_16":{"active_threads":1457,"extent":[47,31],"group_si'
    'ze":[16,16],"groups":[3,2],"idle_lanes_per_threadgroup":0,"launched_threads":1'
    '536,"padding_threads":79,"simd_groups_per_threadgroup":8,"threads":[48,32],"th'
    'reads_per_group":256},"matmul_tiled_32":{"active_threads":1457,"extent":[47,31'
    '],"group_size":[32,32],"groups":[2,1],"idle_lanes_per_threadgroup":0,"launched'
    '_threads":2048,"padding_threads":591,"simd_groups_per_threadgroup":32,"threads'
    '":[64,32],"threads_per_group":1024},"matmul_tiled_8":{"active_threads":1457,"e'
    'xtent":[47,31],"group_size":[8,8],"groups":[6,4],"idle_lanes_per_threadgroup":'
    '0,"launched_threads":1536,"padding_threads":79,"simd_groups_per_threadgroup":2'
    ',"threads":[48,32],"threads_per_group":64},"matmul_x_columns":{"active_threads'
    '":1457,"extent":[47,31],"group_size":[32,8],"groups":[2,4],"idle_lanes_per_thr'
    'eadgroup":0,"launched_threads":2048,"padding_threads":591,"simd_groups_per_thr'
    'eadgroup":8,"threads":[64,32],"threads_per_group":256}},"m":31,"n":47,"outputs'
    '":[{"bound_ratio":0.019657861130341278,"buffer_sha256":"46237ca8d9b3ca1db447f0'
    '77bbeacd73ec7930d52287bc126ee49ed26047c40f","correctness_mode":"k_order_fma","'
    'differing_elements":0,"guard_elements":591,"kernel":"matmul_x_columns","max_re'
    'lative_error":1.511504405107635e-07,"replay_bits":true},{"bound_ratio":0.01965'
    '7861130341278,"buffer_sha256":"7b211097cb2086a0067b32ec9b4640c777a3dcd25756e33'
    '83faf564b536a8228","correctness_mode":"k_order_fma","differing_elements":0,"gu'
    'ard_elements":79,"kernel":"matmul_tiled_8","max_relative_error":1.511504405107'
    '635e-07,"replay_bits":true},{"bound_ratio":0.019657861130341278,"buffer_sha256'
    '":"7b211097cb2086a0067b32ec9b4640c777a3dcd25756e3383faf564b536a8228","correctn'
    'ess_mode":"k_order_fma","differing_elements":0,"guard_elements":79,"kernel":"m'
    'atmul_tiled_16","max_relative_error":1.511504405107635e-07,"replay_bits":true}'
    ',{"bound_ratio":0.019657861130341278,"buffer_sha256":"46237ca8d9b3ca1db447f077'
    'bbeacd73ec7930d52287bc126ee49ed26047c40f","correctness_mode":"k_order_fma","di'
    'ffering_elements":0,"guard_elements":591,"kernel":"matmul_tiled_32","max_relat'
    'ive_error":1.511504405107635e-07,"replay_bits":true}],"reference_sha256":"3b30'
    '9836ece126681e2470d82b0a9845c0c65d05fe1f71df0215563051b8934d","replay_sha256":'
    '"20d0760b682f206383574949c9a4725029b4439471b3183635cc7996a0f1cbe3"}],"gamma":"'
    'gamma_K(u) = K * u / (1 - K * u)","reference":"float64 sum of the exact float3'
    '2 products in increasing k order, on the CPU, rebuilt from the seed","relative'
    '_error":"max over elements of |C - R| / (|A| @ |B|), in float64","replay":"rep'
    "lay_bits compares an output with the CPU replay of sum = fma(a[i, k], b[k, j],"
    " sum) for k = 0 to K - 1, one float32 rounding per step; the tiled kernels run"
    ' that chain as `sum = fma(a_tile[local.y][i], b_tile[i][local.x], sum);`","rul'
    'es":"every output keeps its sentinel guard tail and stays within the bound; re'
    'play_bits is measured: true only when no element differs from the replay","sen'
    'tinel_float32_bits":2143289344},"environment":{"macos_version":"27.2","numpy_v'
    'ersion":"1.26.4","platform_machine":"arm64","platform_system":"Darwin","python'
    '_implementation":"CPython","python_version":"3.11.14","torch_float32_matmul_pr'
    'ecision":"highest","torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu_m'
    'etal_this_host":true,"cublas_measured":false,"cuda_kernel_executed":false,"dis'
    'tributed_claim":false,"host_specific":true,"nvidia_gpu_used":false,"portable_p'
    'erformance_claim":false,"production_claim":false,"statement":"These are wall-c'
    "lock timings on the Apple GPU of the host that made this report, at the record"
    "ed load averages. The naive and tiled kernels came from torch.mps.compile_shad"
    "er, and torch.matmul ran on the MPS device. PyTorch does not report which kern"
    "el torch.matmul ran, so the vendor number names the call, not a kernel. No CUD"
    "A kernel ran, no NVIDIA GPU was used, and cuBLAS was not measured. Each ratio "
    "compares two sides of one run on this host. These timings do not measure or pr"
    'edict CUDA, NVIDIA, cuBLAS, distributed or production performance.","vendor_ke'
    'rnel_known":false,"vendor_peak_claimed":false},"host_load":{"after":[10.961914'
    '0625,12.70703125,14.70458984375],"before":[10.9619140625,12.70703125,14.704589'
    '84375],"source":"os.getloadavg(): the 1, 5 and 15 minute load averages that up'
    'time prints, read before the checks and after the last sample"},"kind":"matmul'
    '_tiled_correctness_and_gflops","label":"host-specific Apple GPU (Metal) measur'
    'ement","measured":true,"measurement_scope":"apple_gpu_metal_this_host","percen'
    'tile_method":"nearest rank: sorted samples[ceil(p * n / 100) - 1] for p in 25,'
    ' 50 and 75; median is p50 and IQR is p75 - p25; no interpolation","planned_wor'
    'k":{"benchmark_points":2,"calls":60,"correctness_cases":3,"correctness_launche'
    's":12,"flops":1151088640,"replay_steps":193003,"sides_per_point":5,"timed_samp'
    'les":30},"refusals":[],"report_sha256":"a6aa39a9faa825598a11c26d710c955910d9d1'
    'ae1f2c9ad8734d6d2c506a1308","run":{"elapsed_ns":872349208,"max_runtime_ns":600'
    '00000000,"timed_ns":20499669},"schema_version":1,"shaders":{"naive":{"kernels"'
    ':[{"axes":2,"buffers":["c","a","b","m","n","k"],"limits":{"max_threads_per_thr'
    'eadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_width":32'
    '},"mapping":"x_columns","name":"matmul_x_columns","scalar_buffers":[3,4,5]},{"'
    'axes":2,"buffers":["c","a","b","m","n","k"],"limits":{"max_threads_per_threadg'
    'roup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_width":32},"m'
    'apping":"x_rows","name":"matmul_x_rows","scalar_buffers":[3,4,5]}],"source_sha'
    '256":"beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc"},"tile'
    'd":{"source":"#include <metal_stdlib>\\nusing namespace metal;\\n\\n// One 8 x'
    " 8 threadgroup per 8 x 8 tile of C = A @ B; all three\\n// are row-major float"
    "32. Each phase stages one tile of A and one of B,\\n// zero-filled past the ma"
    "trices, and the inner loop stops at the end\\n// of K, so every output takes K"
    " fma steps in increasing k.\\nkernel void matmul_tiled_8(device float* c [[buf"
    "fer(0)]],\\n                           device const float* a [[buffer(1)]],\\n"
    "                           device const float* b [[buffer(2)]],\\n            "
    "               constant uint& m [[buffer(3)]],\\n                           co"
    "nstant uint& n [[buffer(4)]],\\n                           constant uint& k [["
    "buffer(5)]],\\n                           uint2 position [[thread_position_in_"
    "grid]],\\n                           uint2 local [[thread_position_in_threadgr"
    "oup]]) {\\n    threadgroup float a_tile[8][8];\\n    threadgroup float b_tile["
    "8][8];\\n    uint column = position.x;\\n    uint row = position.y;\\n    floa"
    "t sum = 0.0f;\\n    for (uint start = 0; start < k; start += 8) {\\n        ui"
    "nt a_column = start + local.x;\\n        uint b_row = start + local.y;\\n     "
    "   a_tile[local.y][local.x] =\\n            row < m && a_column < k ? a[row * "
    "k + a_column] : 0.0f;\\n        b_tile[local.y][local.x] =\\n            b_row"
    " < k && column < n ? b[b_row * n + column] : 0.0f;\\n        threadgroup_barri"
    "er(mem_flags::mem_threadgroup);\\n        uint steps = min(8u, k - start);\\n "
    "       for (uint i = 0; i < steps; ++i) {\\n            sum = fma(a_tile[local"
    ".y][i], b_tile[i][local.x], sum);\\n        }\\n        threadgroup_barrier(me"
    "m_flags::mem_threadgroup);\\n    }\\n    if (row < m && column < n) {\\n      "
    "  c[row * n + column] = sum;\\n    }\\n}\\n\\n// One 16 x 16 threadgroup per 1"
    "6 x 16 tile of C = A @ B; all three\\n// are row-major float32. Each phase sta"
    "ges one tile of A and one of B,\\n// zero-filled past the matrices, and the in"
    "ner loop stops at the end\\n// of K, so every output takes K fma steps in incr"
    "easing k.\\nkernel void matmul_tiled_16(device float* c [[buffer(0)]],\\n     "
    "                       device const float* a [[buffer(1)]],\\n                "
    "            device const float* b [[buffer(2)]],\\n                           "
    " constant uint& m [[buffer(3)]],\\n                            constant uint& "
    "n [[buffer(4)]],\\n                            constant uint& k [[buffer(5)]],"
    "\\n                            uint2 position [[thread_position_in_grid]],\\n "
    "                           uint2 local [[thread_position_in_threadgroup]]) {"
    "\\n    threadgroup float a_tile[16][16];\\n    threadgroup float b_tile[16][16"
    "];\\n    uint column = position.x;\\n    uint row = position.y;\\n    float su"
    "m = 0.0f;\\n    for (uint start = 0; start < k; start += 16) {\\n        uint "
    "a_column = start + local.x;\\n        uint b_row = start + local.y;\\n        "
    "a_tile[local.y][local.x] =\\n            row < m && a_column < k ? a[row * k +"
    " a_column] : 0.0f;\\n        b_tile[local.y][local.x] =\\n            b_row < "
    "k && column < n ? b[b_row * n + column] : 0.0f;\\n        threadgroup_barrier("
    "mem_flags::mem_threadgroup);\\n        uint steps = min(16u, k - start);\\n   "
    "     for (uint i = 0; i < steps; ++i) {\\n            sum = fma(a_tile[local.y"
    "][i], b_tile[i][local.x], sum);\\n        }\\n        threadgroup_barrier(mem_"
    "flags::mem_threadgroup);\\n    }\\n    if (row < m && column < n) {\\n        "
    "c[row * n + column] = sum;\\n    }\\n}\\n\\n// One 32 x 32 threadgroup per 32 "
    "x 32 tile of C = A @ B; all three\\n// are row-major float32. Each phase stage"
    "s one tile of A and one of B,\\n// zero-filled past the matrices, and the inne"
    "r loop stops at the end\\n// of K, so every output takes K fma steps in increa"
    "sing k.\\nkernel void matmul_tiled_32(device float* c [[buffer(0)]],\\n       "
    "                     device const float* a [[buffer(1)]],\\n                  "
    "          device const float* b [[buffer(2)]],\\n                            c"
    "onstant uint& m [[buffer(3)]],\\n                            constant uint& n "
    "[[buffer(4)]],\\n                            constant uint& k [[buffer(5)]],"
    "\\n                            uint2 position [[thread_position_in_grid]],\\n "
    "                           uint2 local [[thread_position_in_threadgroup]]) {"
    "\\n    threadgroup float a_tile[32][32];\\n    threadgroup float b_tile[32][32"
    "];\\n    uint column = position.x;\\n    uint row = position.y;\\n    float su"
    "m = 0.0f;\\n    for (uint start = 0; start < k; start += 32) {\\n        uint "
    "a_column = start + local.x;\\n        uint b_row = start + local.y;\\n        "
    "a_tile[local.y][local.x] =\\n            row < m && a_column < k ? a[row * k +"
    " a_column] : 0.0f;\\n        b_tile[local.y][local.x] =\\n            b_row < "
    "k && column < n ? b[b_row * n + column] : 0.0f;\\n        threadgroup_barrier("
    "mem_flags::mem_threadgroup);\\n        uint steps = min(32u, k - start);\\n   "
    "     for (uint i = 0; i < steps; ++i) {\\n            sum = fma(a_tile[local.y"
    "][i], b_tile[i][local.x], sum);\\n        }\\n        threadgroup_barrier(mem_"
    "flags::mem_threadgroup);\\n    }\\n    if (row < m && column < n) {\\n        "
    'c[row * n + column] = sum;\\n    }\\n}\\n","source_sha256":"08b88577018024d543'
    '44a00f991f387225f1085047caac66f347dc5e719a5fa5","variants":[{"admission":{"adm'
    'itted":true,"allowed_threads":1024,"declared_threadgroup_bytes":512,"name":"ma'
    'tmul_tiled_8","reason":null,"required_threads":64,"static_threadgroup_bytes":5'
    '12},"buffers":["c","a","b","m","n","k"],"correctness_mode":"k_order_fma","grou'
    'p_size":[8,8],"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup'
    '_memory_bytes":512,"thread_execution_width":32},"name":"matmul_tiled_8","scala'
    'r_buffers":[3,4,5],"source_sha256":"21b53bf8ce1c123d1755b12512dedf1dfa1a5cc250'
    '9d476ac1019538a924e1ba","threadgroup_bytes":512,"threads_per_group":64,"tile":'
    '8},{"admission":{"admitted":true,"allowed_threads":1024,"declared_threadgroup_'
    'bytes":2048,"name":"matmul_tiled_16","reason":null,"required_threads":256,"sta'
    'tic_threadgroup_bytes":2048},"buffers":["c","a","b","m","n","k"],"correctness_'
    'mode":"k_order_fma","group_size":[16,16],"limits":{"max_threads_per_threadgrou'
    'p":1024,"static_threadgroup_memory_bytes":2048,"thread_execution_width":32},"n'
    'ame":"matmul_tiled_16","scalar_buffers":[3,4,5],"source_sha256":"9e79d70317e30'
    '9e13e043d0c015574670bfe09f0d28cf23abc4300f98a2eeb13","threadgroup_bytes":2048,'
    '"threads_per_group":256,"tile":16},{"admission":{"admitted":true,"allowed_thre'
    'ads":1024,"declared_threadgroup_bytes":8192,"name":"matmul_tiled_32","reason":'
    'null,"required_threads":1024,"static_threadgroup_bytes":8192},"buffers":["c","'
    'a","b","m","n","k"],"correctness_mode":"k_order_fma","group_size":[32,32],"lim'
    'its":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":819'
    '2,"thread_execution_width":32},"name":"matmul_tiled_32","scalar_buffers":[3,4,'
    '5],"source_sha256":"f5ae7c6f03bcd1772c6eec5aed127916cb962b70bc7f99179d9f115a58'
    '9ced30","threadgroup_bytes":8192,"threads_per_group":1024,"tile":32}]}},"summa'
    'ry":{"benchmark_points":2,"cases_with_other_bits":{"matmul_tiled_16":[],"matmu'
    'l_tiled_32":[],"matmul_tiled_8":[],"matmul_x_columns":[]},"correctness_cases":'
    '3,"kernels_checked":["matmul_x_columns","matmul_tiled_8","matmul_tiled_16","ma'
    'tmul_tiled_32"],"label":"host-specific Apple GPU (Metal) measurement","load_av'
    'erage_after":[10.9619140625,12.70703125,14.70458984375],"load_average_before":'
    '[10.9619140625,12.70703125,14.70458984375],"max_bound_ratio":0.457064164188599'
    '63,"refused_variants":[],"three_numbers":[{"k":256,"m":256,"n":256,"naive_gflo'
    'ps_per_second":37.69631456256144,"tiled_gflops_per_second":54.756731087760755,'
    '"tiled_over_naive":"faster","tiled_side":"matmul_tiled_32","vendor_gflops_per_'
    'second":54.932251531924436,"vendor_over_tiled":"no_clear_difference"},{"k":100'
    ',"m":200,"n":300,"naive_gflops_per_second":36.63188164239041,"tiled_gflops_per'
    '_second":33.59189763429061,"tiled_over_naive":"no_clear_difference","tiled_sid'
    'e":"matmul_tiled_16","vendor_gflops_per_second":64.53781512605042,"vendor_over'
    '_tiled":"faster"}],"timed_samples":30,"warnings":0},"timer":"time.perf_counter'
    '_ns","torch_mps_switches":{"source":"os.environ, read before the runtime impor'
    "ts torch; null means unset, and a value that is not a short token is 'unknown'"
    '","values":{"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}},"wa'
    'rnings":{"capture":"warnings.catch_warnings(record=True) with simplefilter(\'a'
    "lways'), from before the runtime imports torch to the end of the run; equal wa"
    'rnings are kept once with a count","count":0,"distinct":0,"records":[],"torch_'
    'imported_before_capture":false}}'
)


def recorded():
    return json.loads(RECORDED_REPORT_JSON)


def test_the_recorded_report_verifies_as_a_measurement_on_any_host():
    report = recorded()

    summary = verify_report(report, config=config_from_mapping(RECORDED_CONFIG))

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert [round(value, 2) for value in summary["load_average_before"]] == [
        10.96,
        12.71,
        14.7,
    ]
    assert report["environment"]["torch_version"] == "2.15.0"
    assert report["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert report["evidence_boundary"]["cublas_measured"] is False


def test_the_fresh_process_raised_no_warning_from_the_torch_import_on():
    captured = recorded()["warnings"]

    assert captured["torch_imported_before_capture"] is False
    assert (captured["count"], captured["records"]) == (0, [])


def test_every_recorded_pipeline_admitted_its_tile_with_the_declared_bytes():
    report = recorded()
    variants = report["shaders"]["tiled"]["variants"]

    assert report["refusals"] == [] and report["summary"]["refused_variants"] == []
    for entry in variants:
        variant = VARIANTS[entry["name"]]
        assert entry["source_sha256"] == variant.source_sha256
        assert entry["limits"] == {
            "max_threads_per_threadgroup": 1024,
            "thread_execution_width": 32,
            "static_threadgroup_memory_bytes": variant.threadgroup_bytes,
        }
        assert entry["admission"]["admitted"] is True


def test_the_apple_gpu_held_the_replay_bits_for_every_kernel_and_case():
    report = recorded()

    assert set(map(tuple, report["summary"]["cases_with_other_bits"].values())) == {()}
    for case in report["correctness"]["cases"]:
        shape = matmul_shape(case["m"], case["n"], case["k"])
        left, right = matmul_inputs(76, CORRECTNESS_STREAM, case["index"], shape)
        replay = fma_replay(left, right)
        assert case["replay_sha256"] == matrix_sha256(replay)
        for output in case["outputs"]:
            tail = float32_sentinels(output["guard_elements"])
            buffer = numpy.concatenate([replay.reshape(-1), tail])
            assert output["buffer_sha256"] == matrix_sha256(buffer), output["kernel"]
            assert output["replay_bits"] is True


def test_the_recorded_numbers_name_torch_on_mps_and_measure_no_cublas():
    comparison = recorded()["comparison"]

    assert comparison["vendor"] == (
        "vendor GPU matmul: torch.matmul on MPS, PyTorch 2.15.0"
    )
    for entry in comparison["points"]:
        assert entry["cublas"] == "not measured, no NVIDIA GPU or cuBLAS on this host"
        assert entry["tiled"]["side"] in VARIANTS


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    shaders = recorded()["shaders"]
    runtime = tiled_runtime()

    for kernel in shaders["naive"]["kernels"]:
        assert runtime.limits[kernel["name"]].to_mapping() == kernel["limits"]
    for entry in shaders["tiled"]["variants"]:
        assert runtime.limits[entry["name"]].to_mapping() == entry["limits"]
        assert runtime.admissions[entry["name"]].to_mapping() == entry["admission"]
