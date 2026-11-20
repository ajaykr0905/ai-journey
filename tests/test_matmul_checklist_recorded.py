"""One real checklist report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``matmul_checklist_cli
run`` wrote in a fresh process on 2026-11-20 at 11:26 IST under the shared
GPU lock, with torch 2.15.0 on macOS 27.2 (arm64) and load averages near
21.3, 23.9 and 24.2 from other jobs. The run is short: two correctness
cases, two sweep points, three repeats and one or two calls for each sample.
The cost of each synchronize call thus sets most of its rates. The timings
describe that moment and are here only to be verified. A host without torch
or MPS can still check a genuine Apple GPU (Metal) report, and the Linux CI
does.
"""

import json

import numpy

from ai_journey.kernel_metal_reference import float32_sentinels
from ai_journey.matmul_checklist_config import config_from_mapping
from ai_journey.matmul_checklist_steps import checklist_mapping, checklist_sha256
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_checklist_verify import verify_report
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.memory_access_plan import canonical_json_bytes
from test_kernel_metal_runtime import requires_metal
from test_matmul_checklist_runtime import checklist_runtime

RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "matmul-checklist-recorded",
    "seed": 77,
    "checklist_sha256": (
        "cb5e3164e0d37716699dc9381b587faeeb40c2ca4f1a52d75b8ec586379323dc"
    ),
    "naive": {"group_size": [32, 8]},
    "correctness": {"shapes": [[17, 33, 9], [65, 68, 40]]},
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
    '{"backend":"torch.mps.compile_shader","benchmark":{"agreement":"every side withi'
    "n 4 * gamma_K(2**-24) * (|A| @ |B|) of the naive output, on the device in float3"
    '2","device":"mps","flops_per_call":"2 * M * N * K: one multiply and one add per '
    'term of each output","gflops":"FLOPs of one sample divided by the median sample '
    'duration in ns","points":[{"agreement_bound":6.103608758678569e-05,"checks":{"ma'
    'tmul_s2_coarse_4x1":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_s3_bl'
    'ock_4x4":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_s4_vec4":{"disag'
    'reement":0.0,"same_bits_as_naive":true},"matmul_s5_pad4":{"disagreement":0.0,"sa'
    'me_bits_as_naive":true},"matmul_s6_group_32x32":{"disagreement":0.0,"same_bits_a'
    's_naive":true},"matmul_s6_group_8x8":{"disagreement":0.0,"same_bits_as_naive":tr'
    'ue},"matmul_s7_interior":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_'
    'tiled_32":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_x_rows":{"disag'
    'reement":0.0,"same_bits_as_naive":true},"torch_matmul":{"disagreement":0.0,"same'
    '_bits_as_naive":true}},"flops_per_call":33554432,"index":0,"k":256,"launches":{"'
    'matmul_s2_coarse_4x1":{"active_threads":16384,"extent":[256,64],"group_size":[32'
    ',8],"groups":[8,8],"idle_lanes_per_threadgroup":0,"launched_threads":16384,"padd'
    'ing_threads":0,"simd_groups_per_threadgroup":8,"threads":[256,64],"threads_per_g'
    'roup":256},"matmul_s3_block_4x4":{"active_threads":4096,"extent":[64,64],"group_'
    'size":[16,16],"groups":[4,4],"idle_lanes_per_threadgroup":0,"launched_threads":4'
    '096,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[64,64],"threa'
    'ds_per_group":256},"matmul_s4_vec4":{"active_threads":4096,"extent":[64,64],"gro'
    'up_size":[16,16],"groups":[4,4],"idle_lanes_per_threadgroup":0,"launched_threads'
    '":4096,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[64,64],"th'
    'reads_per_group":256},"matmul_s5_pad4":{"active_threads":4096,"extent":[64,64],"'
    'group_size":[16,16],"groups":[4,4],"idle_lanes_per_threadgroup":0,"launched_thre'
    'ads":4096,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[64,64],'
    '"threads_per_group":256},"matmul_s6_group_32x32":{"active_threads":4096,"extent"'
    ':[64,64],"group_size":[32,32],"groups":[2,2],"idle_lanes_per_threadgroup":0,"lau'
    'nched_threads":4096,"padding_threads":0,"simd_groups_per_threadgroup":32,"thread'
    's":[64,64],"threads_per_group":1024},"matmul_s6_group_8x8":{"active_threads":409'
    '6,"extent":[64,64],"group_size":[8,8],"groups":[8,8],"idle_lanes_per_threadgroup'
    '":0,"launched_threads":4096,"padding_threads":0,"simd_groups_per_threadgroup":2,'
    '"threads":[64,64],"threads_per_group":64},"matmul_s7_interior":{"active_threads"'
    ':4096,"extent":[64,64],"group_size":[16,16],"groups":[4,4],"idle_lanes_per_threa'
    'dgroup":0,"launched_threads":4096,"padding_threads":0,"simd_groups_per_threadgro'
    'up":8,"threads":[64,64],"threads_per_group":256},"matmul_tiled_32":{"active_thre'
    'ads":65536,"extent":[256,256],"group_size":[32,32],"groups":[8,8],"idle_lanes_pe'
    'r_threadgroup":0,"launched_threads":65536,"padding_threads":0,"simd_groups_per_t'
    'hreadgroup":32,"threads":[256,256],"threads_per_group":1024},"matmul_x_columns":'
    '{"active_threads":65536,"extent":[256,256],"group_size":[32,8],"groups":[8,32],"'
    'idle_lanes_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"simd'
    '_groups_per_threadgroup":8,"threads":[256,256],"threads_per_group":256},"matmul_'
    'x_rows":{"active_threads":65536,"extent":[256,256],"group_size":[32,8],"groups":'
    '[8,32],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"padding_threads"'
    ':0,"simd_groups_per_threadgroup":8,"threads":[256,256],"threads_per_group":256}}'
    ',"m":256,"model":{"matmul_s2_coarse_4x1":{"flops_per_byte":7.529411764705882,"lo'
    'ads_per_output":16.0},"matmul_s3_block_4x4":{"flops_per_byte":14.222222222222221'
    ',"loads_per_output":8.0},"matmul_s4_vec4":{"flops_per_byte":14.222222222222221,"'
    'loads_per_output":8.0},"matmul_s5_pad4":{"flops_per_byte":14.222222222222221,"lo'
    'ads_per_output":8.0},"matmul_s6_group_32x32":{"flops_per_byte":25.6,"loads_per_o'
    'utput":4.0},"matmul_s6_group_8x8":{"flops_per_byte":7.529411764705882,"loads_per'
    '_output":16.0},"matmul_s7_interior":{"flops_per_byte":14.222222222222221,"loads_'
    'per_output":8.0},"matmul_tiled_32":{"flops_per_byte":7.529411764705882,"loads_pe'
    'r_output":16.0},"matmul_x_columns":{"flops_per_byte":0.24951267056530213,"loads_'
    'per_output":512.0},"matmul_x_rows":{"flops_per_byte":0.24951267056530213,"loads_'
    'per_output":512.0}},"n":256,"side_order":["matmul_x_columns","matmul_x_rows","ma'
    'tmul_tiled_32","matmul_s2_coarse_4x1","matmul_s3_block_4x4","matmul_s4_vec4","ma'
    'tmul_s5_pad4","matmul_s6_group_8x8","matmul_s6_group_32x32","matmul_s7_interior"'
    ',"torch_matmul"],"sides":{"matmul_s2_coarse_4x1":{"best_gflops_per_second":137.5'
    '8864992311635,"calls_per_sample":1,"flops_per_sample":33554432,"implementation":'
    '"matmul_s2_coarse_4x1","iqr_gflops_per_second":[25.774739329284685,137.588649923'
    '11635],"iqr_ns_per_call":1057959.0,"median_gflops_per_second":94.76431232225212,'
    '"median_ns_per_call":354083.0,"repeats":3,"samples_ns":[1301834,354083,243875],"'
    'timing":{"count":3,"iqr_ns":1057959,"max_ns":1301834,"median_ns":354083,"min_ns"'
    ':243875,"p25_ns":243875,"p75_ns":1301834},"warmup":1},"matmul_s3_block_4x4":{"be'
    'st_gflops_per_second":138.46357506581825,"calls_per_sample":1,"flops_per_sample"'
    ':33554432,"implementation":"matmul_s3_block_4x4","iqr_gflops_per_second":[75.382'
    '04324627914,138.46357506581825],"iqr_ns_per_call":202791.0,"median_gflops_per_se'
    'cond":105.72487435998424,"median_ns_per_call":317375.0,"repeats":3,"samples_ns":'
    '[445125,317375,242334],"timing":{"count":3,"iqr_ns":202791,"max_ns":445125,"medi'
    'an_ns":317375,"min_ns":242334,"p25_ns":242334,"p75_ns":445125},"warmup":1},"matm'
    'ul_s4_vec4":{"best_gflops_per_second":142.86080681213411,"calls_per_sample":1,"f'
    'lops_per_sample":33554432,"implementation":"matmul_s4_vec4","iqr_gflops_per_seco'
    'nd":[61.21675165336374,142.86080681213411],"iqr_ns_per_call":313250.0,"median_gf'
    'lops_per_second":114.76505173150919,"median_ns_per_call":292375.0,"repeats":3,"s'
    'amples_ns":[548125,292375,234875],"timing":{"count":3,"iqr_ns":313250,"max_ns":5'
    '48125,"median_ns":292375,"min_ns":234875,"p25_ns":234875,"p75_ns":548125},"warmu'
    'p":1},"matmul_s5_pad4":{"best_gflops_per_second":145.67791849192255,"calls_per_s'
    'ample":1,"flops_per_sample":33554432,"implementation":"matmul_s5_pad4","iqr_gflo'
    'ps_per_second":[102.88825450364124,145.67791849192255],"iqr_ns_per_call":95792.0'
    ',"median_gflops_per_second":110.4672658436214,"median_ns_per_call":303750.0,"rep'
    'eats":3,"samples_ns":[303750,326125,230333],"timing":{"count":3,"iqr_ns":95792,"'
    'max_ns":326125,"median_ns":303750,"min_ns":230333,"p25_ns":230333,"p75_ns":32612'
    '5},"warmup":1},"matmul_s6_group_32x32":{"best_gflops_per_second":108.41496607431'
    '341,"calls_per_sample":1,"flops_per_sample":33554432,"implementation":"matmul_s6'
    '_group_32x32","iqr_gflops_per_second":[34.642805004836966,108.41496607431341],"i'
    'qr_ns_per_call":659083.0,"median_gflops_per_second":74.73147438752784,"median_ns'
    '_per_call":449000.0,"repeats":3,"samples_ns":[449000,968583,309500],"timing":{"c'
    'ount":3,"iqr_ns":659083,"max_ns":968583,"median_ns":449000,"min_ns":309500,"p25_'
    'ns":309500,"p75_ns":968583},"warmup":1},"matmul_s6_group_8x8":{"best_gflops_per_'
    'second":155.25474609600926,"calls_per_sample":1,"flops_per_sample":33554432,"imp'
    'lementation":"matmul_s6_group_8x8","iqr_gflops_per_second":[20.565039148088562,1'
    '55.25474609600926],"iqr_ns_per_call":1415500.0,"median_gflops_per_second":116.49'
    '186053374346,"median_ns_per_call":288041.0,"repeats":3,"samples_ns":[288041,1631'
    '625,216125],"timing":{"count":3,"iqr_ns":1415500,"max_ns":1631625,"median_ns":28'
    '8041,"min_ns":216125,"p25_ns":216125,"p75_ns":1631625},"warmup":1},"matmul_s7_in'
    'terior":{"best_gflops_per_second":149.79657142857144,"calls_per_sample":1,"flops'
    '_per_sample":33554432,"implementation":"matmul_s7_interior","iqr_gflops_per_seco'
    'nd":[42.48289134172497,149.79657142857144],"iqr_ns_per_call":565834.0,"median_gf'
    'lops_per_second":126.62049811320755,"median_ns_per_call":265000.0,"repeats":3,"s'
    'amples_ns":[265000,789834,224000],"timing":{"count":3,"iqr_ns":565834,"max_ns":7'
    '89834,"median_ns":265000,"min_ns":224000,"p25_ns":224000,"p75_ns":789834},"warmu'
    'p":1},"matmul_tiled_32":{"best_gflops_per_second":86.10147084483768,"calls_per_s'
    'ample":1,"flops_per_sample":33554432,"implementation":"matmul_tiled_32","iqr_gfl'
    'ops_per_second":[44.79395101216953,86.10147084483768],"iqr_ns_per_call":359376.0'
    ',"median_gflops_per_second":46.64386724587315,"median_ns_per_call":719375.0,"rep'
    'eats":3,"samples_ns":[749084,719375,389708],"timing":{"count":3,"iqr_ns":359376,'
    '"max_ns":749084,"median_ns":719375,"min_ns":389708,"p25_ns":389708,"p75_ns":7490'
    '84},"warmup":1},"matmul_x_columns":{"best_gflops_per_second":79.16899344789103,"'
    'calls_per_sample":1,"flops_per_sample":33554432,"implementation":"matmul_x_colum'
    'ns","iqr_gflops_per_second":[54.02202777218756,79.16899344789103],"iqr_ns_per_ca'
    'll":197292.0,"median_gflops_per_second":75.80781022310082,"median_ns_per_call":4'
    '42625.0,"repeats":3,"samples_ns":[621125,442625,423833],"timing":{"count":3,"iqr'
    '_ns":197292,"max_ns":621125,"median_ns":442625,"min_ns":423833,"p25_ns":423833,"'
    'p75_ns":621125},"warmup":1},"matmul_x_rows":{"best_gflops_per_second":49.0352539'
    '5591356,"calls_per_sample":1,"flops_per_sample":33554432,"implementation":"matmu'
    'l_x_rows","iqr_gflops_per_second":[26.774830674556817,49.03525395591356],"iqr_ns'
    '_per_call":568916.0,"median_gflops_per_second":29.66575574029492,"median_ns_per_'
    'call":1131083.0,"repeats":3,"samples_ns":[1253208,1131083,684292],"timing":{"cou'
    'nt":3,"iqr_ns":568916,"max_ns":1253208,"median_ns":1131083,"min_ns":684292,"p25_'
    'ns":684292,"p75_ns":1253208},"warmup":1},"torch_matmul":{"best_gflops_per_second'
    '":112.31609037656904,"calls_per_sample":1,"flops_per_sample":33554432,"implement'
    'ation":"torch_matmul","iqr_gflops_per_second":[89.17149759096019,112.31609037656'
    '904],"iqr_ns_per_call":77541.0,"median_gflops_per_second":92.73432568525898,"med'
    'ian_ns_per_call":361834.0,"repeats":3,"samples_ns":[361834,376291,298750],"timin'
    'g":{"count":3,"iqr_ns":77541,"max_ns":376291,"median_ns":361834,"min_ns":298750,'
    '"p25_ns":298750,"p75_ns":376291},"warmup":1}}},{"agreement_bound":2.384200001955'
    '044e-05,"checks":{"matmul_s2_coarse_4x1":{"disagreement":0.0,"same_bits_as_naive'
    '":true},"matmul_s3_block_4x4":{"disagreement":0.0,"same_bits_as_naive":true},"ma'
    'tmul_s4_vec4":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_s5_pad4":{"'
    'disagreement":0.0,"same_bits_as_naive":true},"matmul_s6_group_32x32":{"disagreem'
    'ent":0.0,"same_bits_as_naive":true},"matmul_s6_group_8x8":{"disagreement":0.0,"s'
    'ame_bits_as_naive":true},"matmul_s7_interior":{"disagreement":0.0,"same_bits_as_'
    'naive":true},"matmul_tiled_32":{"disagreement":0.0,"same_bits_as_naive":true},"m'
    'atmul_x_rows":{"disagreement":0.0,"same_bits_as_naive":true},"torch_matmul":{"di'
    'sagreement":0.0,"same_bits_as_naive":true}},"flops_per_call":12000000,"index":1,'
    '"k":100,"launches":{"matmul_s2_coarse_4x1":{"active_threads":15000,"extent":[300'
    ',50],"group_size":[32,8],"groups":[10,7],"idle_lanes_per_threadgroup":0,"launche'
    'd_threads":17920,"padding_threads":2920,"simd_groups_per_threadgroup":8,"threads'
    '":[320,56],"threads_per_group":256},"matmul_s3_block_4x4":{"active_threads":3750'
    ',"extent":[75,50],"group_size":[16,16],"groups":[5,4],"idle_lanes_per_threadgrou'
    'p":0,"launched_threads":5120,"padding_threads":1370,"simd_groups_per_threadgroup'
    '":8,"threads":[80,64],"threads_per_group":256},"matmul_s4_vec4":{"active_threads'
    '":3750,"extent":[75,50],"group_size":[16,16],"groups":[5,4],"idle_lanes_per_thre'
    'adgroup":0,"launched_threads":5120,"padding_threads":1370,"simd_groups_per_threa'
    'dgroup":8,"threads":[80,64],"threads_per_group":256},"matmul_s5_pad4":{"active_t'
    'hreads":3750,"extent":[75,50],"group_size":[16,16],"groups":[5,4],"idle_lanes_pe'
    'r_threadgroup":0,"launched_threads":5120,"padding_threads":1370,"simd_groups_per'
    '_threadgroup":8,"threads":[80,64],"threads_per_group":256},"matmul_s6_group_32x3'
    '2":{"active_threads":3750,"extent":[75,50],"group_size":[32,32],"groups":[3,2],"'
    'idle_lanes_per_threadgroup":0,"launched_threads":6144,"padding_threads":2394,"si'
    'md_groups_per_threadgroup":32,"threads":[96,64],"threads_per_group":1024},"matmu'
    'l_s6_group_8x8":{"active_threads":3750,"extent":[75,50],"group_size":[8,8],"grou'
    'ps":[10,7],"idle_lanes_per_threadgroup":0,"launched_threads":4480,"padding_threa'
    'ds":730,"simd_groups_per_threadgroup":2,"threads":[80,56],"threads_per_group":64'
    '},"matmul_s7_interior":{"active_threads":3750,"extent":[75,50],"group_size":[16,'
    '16],"groups":[5,4],"idle_lanes_per_threadgroup":0,"launched_threads":5120,"paddi'
    'ng_threads":1370,"simd_groups_per_threadgroup":8,"threads":[80,64],"threads_per_'
    'group":256},"matmul_tiled_32":{"active_threads":60000,"extent":[300,200],"group_'
    'size":[32,32],"groups":[10,7],"idle_lanes_per_threadgroup":0,"launched_threads":'
    '71680,"padding_threads":11680,"simd_groups_per_threadgroup":32,"threads":[320,22'
    '4],"threads_per_group":1024},"matmul_x_columns":{"active_threads":60000,"extent"'
    ':[300,200],"group_size":[32,8],"groups":[10,25],"idle_lanes_per_threadgroup":0,"'
    'launched_threads":64000,"padding_threads":4000,"simd_groups_per_threadgroup":8,"'
    'threads":[320,200],"threads_per_group":256},"matmul_x_rows":{"active_threads":60'
    '000,"extent":[200,300],"group_size":[32,8],"groups":[7,38],"idle_lanes_per_threa'
    'dgroup":0,"launched_threads":68096,"padding_threads":8096,"simd_groups_per_threa'
    'dgroup":8,"threads":[224,304],"threads_per_group":256}},"m":200,"model":{"matmul'
    '_s2_coarse_4x1":{"flops_per_byte":6.382978723404255,"loads_per_output":6.8333333'
    '33333333},"matmul_s3_block_4x4":{"flops_per_byte":10.714285714285715,"loads_per_'
    'output":3.6666666666666665},"matmul_s4_vec4":{"flops_per_byte":10.71428571428571'
    '5,"loads_per_output":3.6666666666666665},"matmul_s5_pad4":{"flops_per_byte":10.7'
    '14285714285715,"loads_per_output":3.6666666666666665},"matmul_s6_group_32x32":{"'
    'flops_per_byte":16.666666666666668,"loads_per_output":2.0},"matmul_s6_group_8x8"'
    ':{"flops_per_byte":6.382978723404255,"loads_per_output":6.833333333333333},"matm'
    'ul_s7_interior":{"flops_per_byte":10.714285714285715,"loads_per_output":3.666666'
    '6666666665},"matmul_tiled_32":{"flops_per_byte":6.382978723404255,"loads_per_out'
    'put":6.833333333333333},"matmul_x_columns":{"flops_per_byte":0.24875621890547264'
    ',"loads_per_output":200.0},"matmul_x_rows":{"flops_per_byte":0.24875621890547264'
    ',"loads_per_output":200.0}},"n":300,"side_order":["matmul_x_columns","matmul_x_r'
    'ows","matmul_tiled_32","matmul_s2_coarse_4x1","matmul_s3_block_4x4","matmul_s4_v'
    'ec4","matmul_s5_pad4","matmul_s6_group_8x8","matmul_s6_group_32x32","matmul_s7_i'
    'nterior","torch_matmul"],"sides":{"matmul_s2_coarse_4x1":{"best_gflops_per_secon'
    'd":55.35272369996056,"calls_per_sample":2,"flops_per_sample":24000000,"implement'
    'ation":"matmul_s2_coarse_4x1","iqr_gflops_per_second":[17.58348676146608,55.3527'
    '2369996056],"iqr_ns_per_call":465667.0,"median_gflops_per_second":39.20767817030'
    '8355,"median_ns_per_call":306062.5,"repeats":3,"samples_ns":[433583,1364917,6121'
    '25],"timing":{"count":3,"iqr_ns":931334,"max_ns":1364917,"median_ns":612125,"min'
    '_ns":433583,"p25_ns":433583,"p75_ns":1364917},"warmup":1},"matmul_s3_block_4x4":'
    '{"best_gflops_per_second":57.03544039677654,"calls_per_sample":2,"flops_per_samp'
    'le":24000000,"implementation":"matmul_s3_block_4x4","iqr_gflops_per_second":[39.'
    '36039360393604,57.03544039677654],"iqr_ns_per_call":94479.5,"median_gflops_per_s'
    'econd":41.48065186844411,"median_ns_per_call":289291.5,"repeats":3,"samples_ns":'
    '[420791,609750,578583],"timing":{"count":3,"iqr_ns":188959,"max_ns":609750,"medi'
    'an_ns":578583,"min_ns":420791,"p25_ns":420791,"p75_ns":609750},"warmup":1},"matm'
    'ul_s4_vec4":{"best_gflops_per_second":59.68293441094187,"calls_per_sample":2,"fl'
    'ops_per_sample":24000000,"implementation":"matmul_s4_vec4","iqr_gflops_per_secon'
    'd":[37.371069809158406,59.68293441094187],"iqr_ns_per_call":120041.5,"median_gfl'
    'ops_per_second":37.59298389610552,"median_ns_per_call":319208.5,"repeats":3,"sam'
    'ples_ns":[642208,402125,638417],"timing":{"count":3,"iqr_ns":240083,"max_ns":642'
    '208,"median_ns":638417,"min_ns":402125,"p25_ns":402125,"p75_ns":642208},"warmup"'
    ':1},"matmul_s5_pad4":{"best_gflops_per_second":80.17826300474722,"calls_per_samp'
    'le":2,"flops_per_sample":24000000,"implementation":"matmul_s5_pad4","iqr_gflops_'
    'per_second":[12.2360517931679,80.17826300474722],"iqr_ns_per_call":831042.0,"med'
    'ian_gflops_per_second":67.18774495532016,"median_ns_per_call":178604.0,"repeats"'
    ':3,"samples_ns":[299333,357208,1961417],"timing":{"count":3,"iqr_ns":1662084,"ma'
    'x_ns":1961417,"median_ns":357208,"min_ns":299333,"p25_ns":299333,"p75_ns":196141'
    '7},"warmup":1},"matmul_s6_group_32x32":{"best_gflops_per_second":58.217101273499'
    '09,"calls_per_sample":2,"flops_per_sample":24000000,"implementation":"matmul_s6_'
    'group_32x32","iqr_gflops_per_second":[48.77216085058649,58.21710127349909],"iqr_'
    'ns_per_call":39917.0,"median_gflops_per_second":50.946436190650054,"median_ns_pe'
    'r_call":235541.5,"repeats":3,"samples_ns":[412250,471083,492084],"timing":{"coun'
    't":3,"iqr_ns":79834,"max_ns":492084,"median_ns":471083,"min_ns":412250,"p25_ns":'
    '412250,"p75_ns":492084},"warmup":1},"matmul_s6_group_8x8":{"best_gflops_per_seco'
    'nd":87.35213830755232,"calls_per_sample":2,"flops_per_sample":24000000,"implemen'
    'tation":"matmul_s6_group_8x8","iqr_gflops_per_second":[51.622117496241266,87.352'
    '13830755232],"iqr_ns_per_call":95083.5,"median_gflops_per_second":81.25264494286'
    '924,"median_ns_per_call":147687.5,"repeats":3,"samples_ns":[274750,295375,464917'
    '],"timing":{"count":3,"iqr_ns":190167,"max_ns":464917,"median_ns":295375,"min_ns'
    '":274750,"p25_ns":274750,"p75_ns":464917},"warmup":1},"matmul_s7_interior":{"bes'
    't_gflops_per_second":78.18659230252999,"calls_per_sample":2,"flops_per_sample":2'
    '4000000,"implementation":"matmul_s7_interior","iqr_gflops_per_second":[62.909567'
    '49672346,78.18659230252999],"iqr_ns_per_call":37271.0,"median_gflops_per_second"'
    ':64.21404682274247,"median_ns_per_call":186875.0,"repeats":3,"samples_ns":[37375'
    '0,381500,306958],"timing":{"count":3,"iqr_ns":74542,"max_ns":381500,"median_ns":'
    '373750,"min_ns":306958,"p25_ns":306958,"p75_ns":381500},"warmup":1},"matmul_tile'
    'd_32":{"best_gflops_per_second":34.53237410071942,"calls_per_sample":2,"flops_pe'
    'r_sample":24000000,"implementation":"matmul_tiled_32","iqr_gflops_per_second":[1'
    '0.761125434368344,34.53237410071942],"iqr_ns_per_call":767625.0,"median_gflops_p'
    'er_second":15.39735768510992,"median_ns_per_call":779354.5,"repeats":3,"samples_'
    'ns":[1558709,2230250,695000],"timing":{"count":3,"iqr_ns":1535250,"max_ns":22302'
    '50,"median_ns":1558709,"min_ns":695000,"p25_ns":695000,"p75_ns":2230250},"warmup'
    '":1},"matmul_x_columns":{"best_gflops_per_second":48.05199225562058,"calls_per_s'
    'ample":2,"flops_per_sample":24000000,"implementation":"matmul_x_columns","iqr_gf'
    'lops_per_second":[36.144578313253014,48.05199225562058],"iqr_ns_per_call":82270.'
    '5,"median_gflops_per_second":44.63040446304045,"median_ns_per_call":268875.0,"re'
    'peats":3,"samples_ns":[537750,664000,499459],"timing":{"count":3,"iqr_ns":164541'
    ',"max_ns":664000,"median_ns":537750,"min_ns":499459,"p25_ns":499459,"p75_ns":664'
    '000},"warmup":1},"matmul_x_rows":{"best_gflops_per_second":24.347981349446286,"c'
    'alls_per_sample":2,"flops_per_sample":24000000,"implementation":"matmul_x_rows",'
    '"iqr_gflops_per_second":[19.350282030360592,24.347981349446286],"iqr_ns_per_call'
    '":127292.0,"median_gflops_per_second":20.37351443123939,"median_ns_per_call":589'
    '000.0,"repeats":3,"samples_ns":[1240292,985708,1178000],"timing":{"count":3,"iqr'
    '_ns":254584,"max_ns":1240292,"median_ns":1178000,"min_ns":985708,"p25_ns":985708'
    ',"p75_ns":1240292},"warmup":1},"torch_matmul":{"best_gflops_per_second":63.66047'
    '74535809,"calls_per_sample":2,"flops_per_sample":24000000,"implementation":"torc'
    'h_matmul","iqr_gflops_per_second":[25.40354591161683,63.6604774535809],"iqr_ns_p'
    'er_call":283875.0,"median_gflops_per_second":37.89224393132031,"median_ns_per_ca'
    'll":316687.5,"repeats":3,"samples_ns":[633375,944750,377000],"timing":{"count":3'
    ',"iqr_ns":567750,"max_ns":944750,"median_ns":633375,"min_ns":377000,"p25_ns":377'
    '000,"p75_ns":944750},"warmup":1}}}],"synchronization":"torch.mps.synchronize() b'
    "efore the first and before the second clock reading of every sample; untimed war"
    "mups of each side first; repeat r starts with side r mod S, so the order shifts "
    'by one side every repeat","torch_equivalent":"torch.matmul(a, b, out=c) on the M'
    'PS device, float32","traffic_model":"analytical global element loads for each ou'
    "tput element, with no cache modeled: 2 * K for a Day 74 kernel; (ceil(N / BN) * "
    "M * K + ceil(M / BM) * K * N) / (M * N) for a kernel with BM x BN tiles; FLOPs f"
    'or each byte divide 2 * K by 4 bytes times those loads plus one store"},"checkli'
    'st":{"item_source":"the Day 77 tracker notes list coalescing, occupancy, tiling,'
    " arithmetic intensity and memory vs compute bound; this lab adds control diverge"
    'nce for the ragged shape","items":["coalescing","occupancy","tiling","arithmetic'
    '_intensity","memory_vs_compute_bound","control_divergence"],"reference":"matmul_'
    'x_columns","sha256":"cb5e3164e0d37716699dc9381b587faeeb40c2ca4f1a52d75b8ec586379'
    '323dc","steps":[{"base":"matmul_x_rows","chain":"matmul_x_columns","change":"Map'
    " x to the columns of C, so that the lanes of a SIMD group read consecutive eleme"
    'nts of B (Day 74).","hypothesis":"matmul_x_columns is faster than matmul_x_rows '
    "at every point. On Day 74, matmul_x_rows ran at 0.30 to 0.63 of the matmul_x_col"
    'umns rate. Its lanes read elements of A that are K floats apart.","item":"coales'
    'cing","predictions":[{"variant":"matmul_x_columns","verdicts":["faster"]}],"stat'
    'us":"already_applied","step":"S0"},{"base":"matmul_x_columns","chain":"matmul_ti'
    'led_32","change":"Stage 32 x 32 tiles of A and B in threadgroup memory, with one'
    ' output for each thread (Day 76).","hypothesis":"Tiling alone does not make the '
    "kernel faster. On Day 76, tile 32 ran at 0.87 to 0.96 of the naive rate. The bar"
    "riers and the threadgroup reads probably cost more time than the device loads th"
    'at the tiles save.","item":"tiling","predictions":[{"variant":"matmul_tiled_32",'
    '"verdicts":["slower","no_clear_difference"]}],"status":"already_applied","step":'
    '"S1"},{"base":"matmul_tiled_32","chain":"matmul_s2_coarse_4x1","change":"Give ea'
    "ch thread 4 outputs in one column of the 32 x 32 tile, with 4 register accumulat"
    'ors, in groups of 32 x 8 threads.","hypothesis":"Each B value from threadgroup m'
    "emory now serves 4 fma steps, so threadgroup reads fall from 2 to 1.25 for each "
    "fma. Each barrier also covers 4 times the work of a thread. S2 is faster than S1"
    ' by 1.1 or more.","item":"arithmetic_intensity","predictions":[{"variant":"matmu'
    'l_s2_coarse_4x1","verdicts":["faster"]}],"status":"new_kernel","step":"S2"},{"ba'
    'se":"matmul_s2_coarse_4x1","chain":"matmul_s3_block_4x4","change":"Give each thr'
    'ead a 4 x 4 block of outputs from 64 x 64 tiles, with K phases of 16.","hypothes'
    'is":"Each value from threadgroup memory now serves 4 fma steps, so threadgroup r'
    "eads fall from 1.25 to 0.5 for each fma. The model also halves the device loads "
    'for each output. S3 is faster than S2 by 1.1 or more.","item":"arithmetic_intens'
    'ity","predictions":[{"variant":"matmul_s3_block_4x4","verdicts":["faster"]}],"st'
    'atus":"new_kernel","step":"S3"},{"base":"matmul_s3_block_4x4","chain":"matmul_s4'
    '_vec4","change":"Stage A and B with float4 loads and stores when K and N are mul'
    'tiples of 4, and with scalars otherwise.","hypothesis":"The no-cache model gives'
    " S3 15.9 FLOPs for each byte at 4096 cubed. That is below the ridge point of 38."
    "8 from the Day 76 gap run, so the model calls S3 memory bound. On Day 76, the na"
    "ive kernel ran 12 times above its no-cache roof, so caches probably serve most r"
    "epeated loads. Each thread of S3 also stages only 8 values for the 256 fma steps"
    ' of a phase. S4 shows no clear difference from S3.","item":"memory_vs_compute_bo'
    'und","predictions":[{"variant":"matmul_s4_vec4","verdicts":["no_clear_difference'
    '"]}],"status":"new_kernel","step":"S4"},{"base":"matmul_s4_vec4","chain":"matmul'
    '_s5_pad4","change":"Add 4 floats of padding to each tile row in threadgroup memo'
    'ry, as a control for bank conflicts.","hypothesis":"At each k step, the lanes of'
    " a SIMD group of S4 read two A words and 16 consecutive B words. The 32-bank mod"
    "el of the Day 74 memory-access lab finds no conflict there, so padding cannot re"
    "move one. Padding moves the B rows off their 64-byte boundaries, and the gpu-cei"
    "lings lab measured slower consecutive reads off 128-byte boundaries. S5 is not f"
    'aster than S4.","item":"tiling","predictions":[{"variant":"matmul_s5_pad4","verd'
    'icts":["no_clear_difference","slower"]}],"status":"new_kernel","step":"S5"},{"ba'
    'se":"matmul_s5_pad4","chain":"matmul_s5_pad4","change":"Run the S5 kernel in 8 x'
    ' 8 and 32 x 32 groups, with tiles of 32 x 32 and 128 x 128.","hypothesis":"No oc'
    "cupancy counter exists here, so this step records the time and the pipeline limi"
    "ts only. The 8 x 8 group doubles the modeled device loads for each output and ho"
    "lds only 2 SIMD groups, so it is slower than S5. The 32 x 32 group halves those "
    "loads, but each group holds 1024 threads with 16 accumulators, so it is not fast"
    'er than S5.","item":"occupancy","predictions":[{"variant":"matmul_s6_group_8x8",'
    '"verdicts":["slower"]},{"variant":"matmul_s6_group_32x32","verdicts":["no_clear_'
    'difference","slower"]}],"status":"new_kernel","step":"S6"},{"base":"matmul_s5_pa'
    'd4","chain":"matmul_s7_interior","change":"Give interior groups a path with no g'
    'uards: no zero-fill selects, a constant k-loop bound and no store guards.","hypo'
    'thesis":"At the ragged shape, only 78 of the 1504 groups lie at an edge, so dive'
    "rgence costs little there. In the interior path, the constant loop bound lets th"
    'e compiler unroll the k loop. S7 is faster than S5 by 1.1 or more.","item":"cont'
    'rol_divergence","predictions":[{"variant":"matmul_s7_interior","verdicts":["fast'
    'er"]}],"status":"new_kernel","step":"S7"}]},"config":{"benchmark":{"max_runtime_'
    'seconds":60,"repeats":3,"shapes":[[256,256,256],[200,300,100]],"target_flops_per'
    '_sample":16777216,"warmup":1},"checklist_sha256":"cb5e3164e0d37716699dc9381b587f'
    'aeeb40c2ca4f1a52d75b8ec586379323dc","comparison":{"min_faster_ratio":1.1},"corre'
    'ctness":{"shapes":[[17,33,9],[65,68,40]]},"naive":{"group_size":[32,8]},"name":"'
    'matmul-checklist-recorded","schema_version":1,"seed":77},"config_sha256":"2a6304'
    'ef3ca81ffedd1d9f2887d6d14e584068a855eae5eef0ece16c562987de","correctness":{"boun'
    'd":"(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)","bound_source":"N. J. Hig'
    "ham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM, 2002, sectio"
    'n 3.1","cases":[{"a_sha256":"54a0cbbbffdbccab06c3d4d71f41d76a85e2085e7e2e4a280c9'
    '47943a069f7dd","b_sha256":"71fa1066aa0668aedab1c99aa0f758a8d9577d12614da1919ff16'
    '9c14188c3a1","bound":5.364420917476787e-07,"index":0,"k":9,"launches":{"matmul_s'
    '2_coarse_4x1":{"active_threads":165,"extent":[33,5],"group_size":[32,8],"groups"'
    ':[2,1],"idle_lanes_per_threadgroup":0,"launched_threads":512,"padding_threads":3'
    '47,"simd_groups_per_threadgroup":8,"threads":[64,8],"threads_per_group":256},"ma'
    'tmul_s3_block_4x4":{"active_threads":45,"extent":[9,5],"group_size":[16,16],"gro'
    'ups":[1,1],"idle_lanes_per_threadgroup":0,"launched_threads":256,"padding_thread'
    's":211,"simd_groups_per_threadgroup":8,"threads":[16,16],"threads_per_group":256'
    '},"matmul_s4_vec4":{"active_threads":45,"extent":[9,5],"group_size":[16,16],"gro'
    'ups":[1,1],"idle_lanes_per_threadgroup":0,"launched_threads":256,"padding_thread'
    's":211,"simd_groups_per_threadgroup":8,"threads":[16,16],"threads_per_group":256'
    '},"matmul_s5_pad4":{"active_threads":45,"extent":[9,5],"group_size":[16,16],"gro'
    'ups":[1,1],"idle_lanes_per_threadgroup":0,"launched_threads":256,"padding_thread'
    's":211,"simd_groups_per_threadgroup":8,"threads":[16,16],"threads_per_group":256'
    '},"matmul_s6_group_32x32":{"active_threads":45,"extent":[9,5],"group_size":[32,3'
    '2],"groups":[1,1],"idle_lanes_per_threadgroup":0,"launched_threads":1024,"paddin'
    'g_threads":979,"simd_groups_per_threadgroup":32,"threads":[32,32],"threads_per_g'
    'roup":1024},"matmul_s6_group_8x8":{"active_threads":45,"extent":[9,5],"group_siz'
    'e":[8,8],"groups":[2,1],"idle_lanes_per_threadgroup":0,"launched_threads":128,"p'
    'adding_threads":83,"simd_groups_per_threadgroup":2,"threads":[16,8],"threads_per'
    '_group":64},"matmul_s7_interior":{"active_threads":45,"extent":[9,5],"group_size'
    '":[16,16],"groups":[1,1],"idle_lanes_per_threadgroup":0,"launched_threads":256,"'
    'padding_threads":211,"simd_groups_per_threadgroup":8,"threads":[16,16],"threads_'
    'per_group":256},"matmul_tiled_32":{"active_threads":561,"extent":[33,17],"group_'
    'size":[32,32],"groups":[2,1],"idle_lanes_per_threadgroup":0,"launched_threads":2'
    '048,"padding_threads":1487,"simd_groups_per_threadgroup":32,"threads":[64,32],"t'
    'hreads_per_group":1024},"matmul_x_columns":{"active_threads":561,"extent":[33,17'
    '],"group_size":[32,8],"groups":[2,3],"idle_lanes_per_threadgroup":0,"launched_th'
    'reads":1536,"padding_threads":975,"simd_groups_per_threadgroup":8,"threads":[64,'
    '24],"threads_per_group":256},"matmul_x_rows":{"active_threads":561,"extent":[17,'
    '33],"group_size":[32,8],"groups":[1,5],"idle_lanes_per_threadgroup":0,"launched_'
    'threads":1280,"padding_threads":719,"simd_groups_per_threadgroup":8,"threads":[3'
    '2,40],"threads_per_group":256}},"m":17,"n":33,"outputs":[{"bound_ratio":0.226690'
    '85662113317,"buffer_sha256":"35154ff85ca6a4303ea20803a6cdfa7cfd41fb9a84f4a264219'
    'c0347d64c3260","correctness_mode":"k_order_fma","differing_elements":0,"guard_el'
    'ements":975,"kernel":"matmul_x_columns","max_relative_error":1.216065173059138e-'
    '07,"replay_bits":true},{"bound_ratio":0.22669085662113317,"buffer_sha256":"87da6'
    '64db79b96d413eaaf7b56d8dca71422909d9f159d6381328acb0ce4b68d","correctness_mode":'
    '"k_order_fma","differing_elements":0,"guard_elements":719,"kernel":"matmul_x_row'
    's","max_relative_error":1.216065173059138e-07,"replay_bits":true},{"bound_ratio"'
    ':0.22669085662113317,"buffer_sha256":"7efd66cea11960219a46f560221ce6bd2db414b466'
    '927b64b4d98604fa5a7030","correctness_mode":"k_order_fma","differing_elements":0,'
    '"guard_elements":1487,"kernel":"matmul_tiled_32","max_relative_error":1.21606517'
    '3059138e-07,"replay_bits":true},{"bound_ratio":0.22669085662113317,"buffer_sha25'
    '6":"7efd66cea11960219a46f560221ce6bd2db414b466927b64b4d98604fa5a7030","correctne'
    'ss_mode":"k_order_fma","differing_elements":0,"guard_elements":1487,"kernel":"ma'
    'tmul_s2_coarse_4x1","max_relative_error":1.216065173059138e-07,"replay_bits":tru'
    'e},{"bound_ratio":0.22669085662113317,"buffer_sha256":"a4cc65cd7f043d149072f160b'
    '89fa4839940903d3615951f077aff835e1243e3","correctness_mode":"k_order_fma","diffe'
    'ring_elements":0,"guard_elements":3535,"kernel":"matmul_s3_block_4x4","max_relat'
    'ive_error":1.216065173059138e-07,"replay_bits":true},{"bound_ratio":0.2266908566'
    '2113317,"buffer_sha256":"a4cc65cd7f043d149072f160b89fa4839940903d3615951f077aff8'
    '35e1243e3","correctness_mode":"k_order_fma","differing_elements":0,"guard_elemen'
    'ts":3535,"kernel":"matmul_s4_vec4","max_relative_error":1.216065173059138e-07,"r'
    'eplay_bits":true},{"bound_ratio":0.22669085662113317,"buffer_sha256":"a4cc65cd7f'
    '043d149072f160b89fa4839940903d3615951f077aff835e1243e3","correctness_mode":"k_or'
    'der_fma","differing_elements":0,"guard_elements":3535,"kernel":"matmul_s5_pad4",'
    '"max_relative_error":1.216065173059138e-07,"replay_bits":true},{"bound_ratio":0.'
    '22669085662113317,"buffer_sha256":"7efd66cea11960219a46f560221ce6bd2db414b466927'
    'b64b4d98604fa5a7030","correctness_mode":"k_order_fma","differing_elements":0,"gu'
    'ard_elements":1487,"kernel":"matmul_s6_group_8x8","max_relative_error":1.2160651'
    '73059138e-07,"replay_bits":true},{"bound_ratio":0.22669085662113317,"buffer_sha2'
    '56":"e8a7abafb3778e4a642a45ad90706ae3a7ddb8dd788429e63132c5d976413568","correctn'
    'ess_mode":"k_order_fma","differing_elements":0,"guard_elements":15823,"kernel":"'
    'matmul_s6_group_32x32","max_relative_error":1.216065173059138e-07,"replay_bits":'
    'true},{"bound_ratio":0.22669085662113317,"buffer_sha256":"a4cc65cd7f043d149072f1'
    '60b89fa4839940903d3615951f077aff835e1243e3","correctness_mode":"k_order_fma","di'
    'ffering_elements":0,"guard_elements":3535,"kernel":"matmul_s7_interior","max_rel'
    'ative_error":1.216065173059138e-07,"replay_bits":true}],"reference_sha256":"2179'
    '899ecb3233d9f7bfdf8a19e42a60252d71a8f03a2954415586c48d58da63","replay_sha256":"e'
    'c39f39a219565e7cab32a7202cb44b521c5cecd9d9af4e0041de357a79b82de"},{"a_sha256":"d'
    'fb5396a2485e946d13c54a82e41b5db2ed2682d0e20e417a1074b613fb6d312","b_sha256":"8e8'
    '8f9782352a6d565f58a3be1e5880edff4b20043341e8f3dfe58bdeea9a8df","bound":2.3841914'
    '798119557e-06,"index":1,"k":40,"launches":{"matmul_s2_coarse_4x1":{"active_threa'
    'ds":1156,"extent":[68,17],"group_size":[32,8],"groups":[3,3],"idle_lanes_per_thr'
    'eadgroup":0,"launched_threads":2304,"padding_threads":1148,"simd_groups_per_thre'
    'adgroup":8,"threads":[96,24],"threads_per_group":256},"matmul_s3_block_4x4":{"ac'
    'tive_threads":289,"extent":[17,17],"group_size":[16,16],"groups":[2,2],"idle_lan'
    'es_per_threadgroup":0,"launched_threads":1024,"padding_threads":735,"simd_groups'
    '_per_threadgroup":8,"threads":[32,32],"threads_per_group":256},"matmul_s4_vec4":'
    '{"active_threads":289,"extent":[17,17],"group_size":[16,16],"groups":[2,2],"idle'
    '_lanes_per_threadgroup":0,"launched_threads":1024,"padding_threads":735,"simd_gr'
    'oups_per_threadgroup":8,"threads":[32,32],"threads_per_group":256},"matmul_s5_pa'
    'd4":{"active_threads":289,"extent":[17,17],"group_size":[16,16],"groups":[2,2],"'
    'idle_lanes_per_threadgroup":0,"launched_threads":1024,"padding_threads":735,"sim'
    'd_groups_per_threadgroup":8,"threads":[32,32],"threads_per_group":256},"matmul_s'
    '6_group_32x32":{"active_threads":289,"extent":[17,17],"group_size":[32,32],"grou'
    'ps":[1,1],"idle_lanes_per_threadgroup":0,"launched_threads":1024,"padding_thread'
    's":735,"simd_groups_per_threadgroup":32,"threads":[32,32],"threads_per_group":10'
    '24},"matmul_s6_group_8x8":{"active_threads":289,"extent":[17,17],"group_size":[8'
    ',8],"groups":[3,3],"idle_lanes_per_threadgroup":0,"launched_threads":576,"paddin'
    'g_threads":287,"simd_groups_per_threadgroup":2,"threads":[24,24],"threads_per_gr'
    'oup":64},"matmul_s7_interior":{"active_threads":289,"extent":[17,17],"group_size'
    '":[16,16],"groups":[2,2],"idle_lanes_per_threadgroup":0,"launched_threads":1024,'
    '"padding_threads":735,"simd_groups_per_threadgroup":8,"threads":[32,32],"threads'
    '_per_group":256},"matmul_tiled_32":{"active_threads":4420,"extent":[68,65],"grou'
    'p_size":[32,32],"groups":[3,3],"idle_lanes_per_threadgroup":0,"launched_threads"'
    ':9216,"padding_threads":4796,"simd_groups_per_threadgroup":32,"threads":[96,96],'
    '"threads_per_group":1024},"matmul_x_columns":{"active_threads":4420,"extent":[68'
    ',65],"group_size":[32,8],"groups":[3,9],"idle_lanes_per_threadgroup":0,"launched'
    '_threads":6912,"padding_threads":2492,"simd_groups_per_threadgroup":8,"threads":'
    '[96,72],"threads_per_group":256},"matmul_x_rows":{"active_threads":4420,"extent"'
    ':[65,68],"group_size":[32,8],"groups":[3,9],"idle_lanes_per_threadgroup":0,"laun'
    'ched_threads":6912,"padding_threads":2492,"simd_groups_per_threadgroup":8,"threa'
    'ds":[96,72],"threads_per_group":256}},"m":65,"n":68,"outputs":[{"bound_ratio":0.'
    '05974978844197654,"buffer_sha256":"ba648fa8341cbf722bc1566b9495cf615d0d5ed9a88e2'
    'c26e1c036bbe21b5ea8","correctness_mode":"k_order_fma","differing_elements":0,"gu'
    'ard_elements":2492,"kernel":"matmul_x_columns","max_relative_error":1.4245493652'
    '392733e-07,"replay_bits":true},{"bound_ratio":0.05974978844197654,"buffer_sha256'
    '":"ba648fa8341cbf722bc1566b9495cf615d0d5ed9a88e2c26e1c036bbe21b5ea8","correctnes'
    's_mode":"k_order_fma","differing_elements":0,"guard_elements":2492,"kernel":"mat'
    'mul_x_rows","max_relative_error":1.4245493652392733e-07,"replay_bits":true},{"bo'
    'und_ratio":0.05974978844197654,"buffer_sha256":"c096058b4b701d388f3f705f3bb4aa50'
    '6b6623f982c7697710fd5a0ec9d0b545","correctness_mode":"k_order_fma","differing_el'
    'ements":0,"guard_elements":4796,"kernel":"matmul_tiled_32","max_relative_error":'
    '1.4245493652392733e-07,"replay_bits":true},{"bound_ratio":0.05974978844197654,"b'
    'uffer_sha256":"c096058b4b701d388f3f705f3bb4aa506b6623f982c7697710fd5a0ec9d0b545"'
    ',"correctness_mode":"k_order_fma","differing_elements":0,"guard_elements":4796,"'
    'kernel":"matmul_s2_coarse_4x1","max_relative_error":1.4245493652392733e-07,"repl'
    'ay_bits":true},{"bound_ratio":0.05974978844197654,"buffer_sha256":"aca6410106e4e'
    '9c72790f1d34c1701a12123980bfb595054d7a921d26480bf31","correctness_mode":"k_order'
    '_fma","differing_elements":0,"guard_elements":11964,"kernel":"matmul_s3_block_4x'
    '4","max_relative_error":1.4245493652392733e-07,"replay_bits":true},{"bound_ratio'
    '":0.05974978844197654,"buffer_sha256":"aca6410106e4e9c72790f1d34c1701a12123980bf'
    'b595054d7a921d26480bf31","correctness_mode":"k_order_fma","differing_elements":0'
    ',"guard_elements":11964,"kernel":"matmul_s4_vec4","max_relative_error":1.4245493'
    '652392733e-07,"replay_bits":true},{"bound_ratio":0.05974978844197654,"buffer_sha'
    '256":"aca6410106e4e9c72790f1d34c1701a12123980bfb595054d7a921d26480bf31","correct'
    'ness_mode":"k_order_fma","differing_elements":0,"guard_elements":11964,"kernel":'
    '"matmul_s5_pad4","max_relative_error":1.4245493652392733e-07,"replay_bits":true}'
    ',{"bound_ratio":0.05974978844197654,"buffer_sha256":"c096058b4b701d388f3f705f3bb'
    '4aa506b6623f982c7697710fd5a0ec9d0b545","correctness_mode":"k_order_fma","differi'
    'ng_elements":0,"guard_elements":4796,"kernel":"matmul_s6_group_8x8","max_relativ'
    'e_error":1.4245493652392733e-07,"replay_bits":true},{"bound_ratio":0.05974978844'
    '197654,"buffer_sha256":"aca6410106e4e9c72790f1d34c1701a12123980bfb595054d7a921d2'
    '6480bf31","correctness_mode":"k_order_fma","differing_elements":0,"guard_element'
    's":11964,"kernel":"matmul_s6_group_32x32","max_relative_error":1.424549365239273'
    '3e-07,"replay_bits":true},{"bound_ratio":0.05974978844197654,"buffer_sha256":"ac'
    'a6410106e4e9c72790f1d34c1701a12123980bfb595054d7a921d26480bf31","correctness_mod'
    'e":"k_order_fma","differing_elements":0,"guard_elements":11964,"kernel":"matmul_'
    's7_interior","max_relative_error":1.4245493652392733e-07,"replay_bits":true}],"r'
    'eference_sha256":"0fa7ddef8b0bb11870a44c172640f0cd426d49ea61d19406753c06a4740623'
    '22","replay_sha256":"c7e5a390c20108e5eabc39e23463c47ed0323b7c92355bbde3a2a16b67b'
    '31aac"}],"gamma":"gamma_K(u) = K * u / (1 - K * u)","reference":"float64 sum of '
    "the exact float32 products in increasing k order, on the CPU, rebuilt from the s"
    'eed","relative_error":"max over elements of |C - R| / (|A| @ |B|), in float64","'
    'replay":"replay_bits compares an output with the CPU replay of sum = fma(a[i, k]'
    ", b[k, j], sum) for k = 0 to K - 1, one float32 rounding per step; every checkli"
    'st kernel runs that chain, one fma per k in increasing order","rules":"every out'
    "put keeps its sentinel guard tail and stays within the bound; replay_bits is mea"
    'sured: true only when no element differs from the replay","sentinel_float32_bits'
    '":2143289344},"environment":{"macos_version":"27.2","numpy_version":"1.26.4","pl'
    'atform_machine":"arm64","platform_system":"Darwin","python_implementation":"CPyt'
    'hon","python_version":"3.11.14","torch_float32_matmul_precision":"highest","torc'
    'h_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_host":true,"cubl'
    'as_measured":false,"cuda_kernel_executed":false,"distributed_claim":false,"hardw'
    'are_counters_used":false,"host_specific":true,"nvidia_gpu_used":false,"occupancy'
    '_measured":false,"portable_performance_claim":false,"production_claim":false,"st'
    'atement":"These are wall-clock timings on the Apple GPU of the host that made th'
    "is report, at the recorded load averages. The checklist kernels came from torch."
    "mps.compile_shader, and torch.matmul ran on the MPS device. PyTorch does not rep"
    "ort which kernel torch.matmul ran, so the vendor number names the call, not a ke"
    "rnel. No hardware counter was read, so the occupancy step records times and pipe"
    "line limits, not occupancy. No CUDA kernel ran, no NVIDIA GPU was used, and cuBL"
    "AS was not measured. Each ratio compares two sides of one run on this host. Thes"
    "e timings do not measure or predict CUDA, NVIDIA, cuBLAS, distributed or product"
    'ion performance.","vendor_kernel_known":false,"vendor_peak_claimed":false},"host'
    '_load":{"after":[21.29736328125,23.84912109375,24.2353515625],"before":[21.29736'
    '328125,23.84912109375,24.2353515625],"source":"os.getloadavg(): the 1, 5 and 15 '
    "minute load averages that uptime prints, read before the checks and after the la"
    'st sample"},"kind":"matmul_checklist_correctness_and_speedup","label":"host-spec'
    'ific Apple GPU (Metal) measurement","measured":true,"measurement_scope":"apple_g'
    'pu_metal_this_host","percentile_method":"nearest rank: sorted samples[ceil(p * n'
    " / 100) - 1] for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no inte"
    'rpolation","planned_work":{"benchmark_points":2,"calls":132,"correctness_cases":'
    '2,"correctness_launches":20,"flops":2532395008,"replay_steps":181849,"sides_per_'
    'point":11,"timed_samples":66},"refusals":[],"report_sha256":"2da27f7cfe023cea5c7'
    '7395eeab67c6eee5825cf28467c44c1407020616e5fb5","run":{"elapsed_ns":726795750,"ma'
    'x_runtime_ns":60000000000,"timed_ns":40770542},"schema_version":1,"shaders":{"li'
    'brary_sha256":"d0a8f46a8831029a8894edca48da9b8d61646571c40c36d2f502d6f30cea027d"'
    ',"naive_source_sha256":"beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc'
    '928df9cc","variants":[{"admission":null,"correctness_mode":"k_order_fma","design'
    '":null,"group_size":null,"limits":{"max_threads_per_threadgroup":1024,"static_th'
    'readgroup_memory_bytes":0,"thread_execution_width":32},"mapping":"x_columns","na'
    'me":"matmul_x_columns","origin":"day_74_naive","outputs_per_thread":[1,1],"sourc'
    'e_sha256":"beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc","th'
    'readgroup_bytes":0},{"admission":null,"correctness_mode":"k_order_fma","design":'
    'null,"group_size":null,"limits":{"max_threads_per_threadgroup":1024,"static_thre'
    'adgroup_memory_bytes":0,"thread_execution_width":32},"mapping":"x_rows","name":"'
    'matmul_x_rows","origin":"day_74_naive","outputs_per_thread":[1,1],"source_sha256'
    '":"beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc","threadgrou'
    'p_bytes":0},{"admission":{"admitted":true,"allowed_threads":1024,"declared_threa'
    'dgroup_bytes":8192,"name":"matmul_tiled_32","reason":null,"required_threads":102'
    '4,"static_threadgroup_bytes":8192},"correctness_mode":"k_order_fma","design":nul'
    'l,"group_size":[32,32],"limits":{"max_threads_per_threadgroup":1024,"static_thre'
    'adgroup_memory_bytes":8192,"thread_execution_width":32},"mapping":"x_columns","n'
    'ame":"matmul_tiled_32","origin":"day_76_tiled","outputs_per_thread":[1,1],"sourc'
    'e_sha256":"f5ae7c6f03bcd1772c6eec5aed127916cb962b70bc7f99179d9f115a589ced30","th'
    'readgroup_bytes":8192},{"admission":{"admitted":true,"allowed_threads":1024,"dec'
    'lared_threadgroup_bytes":8192,"name":"matmul_s2_coarse_4x1","reason":null,"requi'
    'red_threads":256,"static_threadgroup_bytes":8192},"correctness_mode":"k_order_fm'
    'a","design":{"block":[32,32,32],"interior_path":false,"outputs_per_thread":[4,1]'
    ',"pad":0,"vector_loads":false},"group_size":[32,8],"limits":{"max_threads_per_th'
    'readgroup":1024,"static_threadgroup_memory_bytes":8192,"thread_execution_width":'
    '32},"mapping":"x_columns","name":"matmul_s2_coarse_4x1","origin":"day_77_checkli'
    'st","outputs_per_thread":[4,1],"source_sha256":"e2520d4f573c267360eb3833306bad7f'
    'bfe88325a4060b72b051f59bf54c144b","threadgroup_bytes":8192},{"admission":{"admit'
    'ted":true,"allowed_threads":1024,"declared_threadgroup_bytes":8192,"name":"matmu'
    'l_s3_block_4x4","reason":null,"required_threads":256,"static_threadgroup_bytes":'
    '8192},"correctness_mode":"k_order_fma","design":{"block":[64,64,16],"interior_pa'
    'th":false,"outputs_per_thread":[4,4],"pad":0,"vector_loads":false},"group_size":'
    '[16,16],"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_'
    'bytes":8192,"thread_execution_width":32},"mapping":"x_columns","name":"matmul_s3'
    '_block_4x4","origin":"day_77_checklist","outputs_per_thread":[4,4],"source_sha25'
    '6":"d4530e74d254196749d27543ccf78238c0edaf5b551c28555095f5e77a23bc9d","threadgro'
    'up_bytes":8192},{"admission":{"admitted":true,"allowed_threads":1024,"declared_t'
    'hreadgroup_bytes":8192,"name":"matmul_s4_vec4","reason":null,"required_threads":'
    '256,"static_threadgroup_bytes":8192},"correctness_mode":"k_order_fma","design":{'
    '"block":[64,64,16],"interior_path":false,"outputs_per_thread":[4,4],"pad":0,"vec'
    'tor_loads":true},"group_size":[16,16],"limits":{"max_threads_per_threadgroup":10'
    '24,"static_threadgroup_memory_bytes":8192,"thread_execution_width":32},"mapping"'
    ':"x_columns","name":"matmul_s4_vec4","origin":"day_77_checklist","outputs_per_th'
    'read":[4,4],"source_sha256":"6df4d1063583f53372a700e08fb895858a42371f2e3d0f79bb4'
    '8bd13fe4eac05","threadgroup_bytes":8192},{"admission":{"admitted":true,"allowed_'
    'threads":1024,"declared_threadgroup_bytes":9472,"name":"matmul_s5_pad4","reason"'
    ':null,"required_threads":256,"static_threadgroup_bytes":9472},"correctness_mode"'
    ':"k_order_fma","design":{"block":[64,64,16],"interior_path":false,"outputs_per_t'
    'hread":[4,4],"pad":4,"vector_loads":true},"group_size":[16,16],"limits":{"max_th'
    'reads_per_threadgroup":1024,"static_threadgroup_memory_bytes":9472,"thread_execu'
    'tion_width":32},"mapping":"x_columns","name":"matmul_s5_pad4","origin":"day_77_c'
    'hecklist","outputs_per_thread":[4,4],"source_sha256":"8567cddd838fcc4aa82000c0f4'
    '34864b5d4bd1c494e21d5937f787400e784028","threadgroup_bytes":9472},{"admission":{'
    '"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":4864,"name":'
    '"matmul_s6_group_8x8","reason":null,"required_threads":64,"static_threadgroup_by'
    'tes":4864},"correctness_mode":"k_order_fma","design":{"block":[32,32,16],"interi'
    'or_path":false,"outputs_per_thread":[4,4],"pad":4,"vector_loads":true},"group_si'
    'ze":[8,8],"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memor'
    'y_bytes":4864,"thread_execution_width":32},"mapping":"x_columns","name":"matmul_'
    's6_group_8x8","origin":"day_77_checklist","outputs_per_thread":[4,4],"source_sha'
    '256":"919930b29ccdd65de0b096cc22c1c5b0683e9f9c03c0791469098039511ae2bc","threadg'
    'roup_bytes":4864},{"admission":{"admitted":true,"allowed_threads":1024,"declared'
    '_threadgroup_bytes":18688,"name":"matmul_s6_group_32x32","reason":null,"required'
    '_threads":1024,"static_threadgroup_bytes":18688},"correctness_mode":"k_order_fma'
    '","design":{"block":[128,128,16],"interior_path":false,"outputs_per_thread":[4,4'
    '],"pad":4,"vector_loads":true},"group_size":[32,32],"limits":{"max_threads_per_t'
    'hreadgroup":1024,"static_threadgroup_memory_bytes":18688,"thread_execution_width'
    '":32},"mapping":"x_columns","name":"matmul_s6_group_32x32","origin":"day_77_chec'
    'klist","outputs_per_thread":[4,4],"source_sha256":"fc609adecff160fd1e17f2ad85ba7'
    'b85785bf48cd040b5ef0a281fa4f090dcc0","threadgroup_bytes":18688},{"admission":{"a'
    'dmitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":9472,"name":"m'
    'atmul_s7_interior","reason":null,"required_threads":256,"static_threadgroup_byte'
    's":9472},"correctness_mode":"k_order_fma","design":{"block":[64,64,16],"interior'
    '_path":true,"outputs_per_thread":[4,4],"pad":4,"vector_loads":true},"group_size"'
    ':[16,16],"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory'
    '_bytes":9472,"thread_execution_width":32},"mapping":"x_columns","name":"matmul_s'
    '7_interior","origin":"day_77_checklist","outputs_per_thread":[4,4],"source_sha25'
    '6":"b5053c2f0c52043d9c93ce8359a491878fd7e8ef44f2c85b6e901019841225ab","threadgro'
    'up_bytes":9472}]},"speedup":{"best_rule":"the timed kernel with the highest medi'
    "an GFLOP/s at this point; a tie goes to the kernel that the registry lists first"
    '","cublas":"not measured, no NVIDIA GPU or cuBLAS on this host","fraction_rule":'
    '"median GFLOP/s of the kernel / median GFLOP/s of torch.matmul at the same point'
    ' of the same run","hypotheses":[{"held_at":[0,1],"missed_at":[],"no_verdict_at":'
    '[],"predicted":["faster"],"step":"S0","variant":"matmul_x_columns"},{"held_at":['
    '0,1],"missed_at":[],"no_verdict_at":[],"predicted":["slower","no_clear_differenc'
    'e"],"step":"S1","variant":"matmul_tiled_32"},{"held_at":[],"missed_at":[0,1],"no'
    '_verdict_at":[],"predicted":["faster"],"step":"S2","variant":"matmul_s2_coarse_4'
    'x1"},{"held_at":[],"missed_at":[0,1],"no_verdict_at":[],"predicted":["faster"],"'
    'step":"S3","variant":"matmul_s3_block_4x4"},{"held_at":[0,1],"missed_at":[],"no_'
    'verdict_at":[],"predicted":["no_clear_difference"],"step":"S4","variant":"matmul'
    '_s4_vec4"},{"held_at":[0,1],"missed_at":[],"no_verdict_at":[],"predicted":["no_c'
    'lear_difference","slower"],"step":"S5","variant":"matmul_s5_pad4"},{"held_at":[]'
    ',"missed_at":[0,1],"no_verdict_at":[],"predicted":["slower"],"step":"S6","varian'
    't":"matmul_s6_group_8x8"},{"held_at":[0,1],"missed_at":[],"no_verdict_at":[],"pr'
    'edicted":["no_clear_difference","slower"],"step":"S6","variant":"matmul_s6_group'
    '_32x32"},{"held_at":[],"missed_at":[0,1],"no_verdict_at":[],"predicted":["faster'
    '"],"step":"S7","variant":"matmul_s7_interior"}],"hypothesis_rule":"a prediction '
    "holds at a point when the verdict of the kernel over its base is one of the verd"
    "icts that its step predicted; a kernel or base that did not run gives no verdict"
    '","min_faster_ratio":1.1,"points":[{"best":{"fraction_of_torch":1.36541132075471'
    '72,"to_s0":{"first":"matmul_s7_interior","iqr_separated":false,"ratio":1.6702830'
    '188679245,"second":"matmul_x_columns","verdict":"no_clear_difference"},"variant"'
    ':"matmul_s7_interior"},"index":0,"k":256,"m":256,"n":256,"rows":[{"base":"matmul'
    '_x_rows","base_rate":{"iqr_gflops_per_second":[26.774830674556817,49.03525395591'
    '356],"median_gflops_per_second":29.66575574029492},"fraction_of_torch":0.8174730'
    '302174528,"held":true,"item":"coalescing","predicted":["faster"],"ran":true,"rat'
    'e":{"iqr_gflops_per_second":[54.02202777218756,79.16899344789103],"median_gflops'
    '_per_second":75.80781022310082},"step":"S0","to_base":{"first":"matmul_x_columns'
    '","iqr_separated":true,"ratio":2.55539791019486,"second":"matmul_x_rows","verdic'
    't":"faster"},"to_s0":{"first":"matmul_x_columns","iqr_separated":false,"ratio":1'
    '.0,"second":"matmul_x_columns","verdict":"no_clear_difference"},"variant":"matmu'
    'l_x_columns"},{"base":"matmul_x_columns","base_rate":{"iqr_gflops_per_second":[5'
    '4.02202777218756,79.16899344789103],"median_gflops_per_second":75.80781022310082'
    '},"fraction_of_torch":0.5029838401390095,"held":true,"item":"tiling","predicted"'
    ':["slower","no_clear_difference"],"ran":true,"rate":{"iqr_gflops_per_second":[44'
    '.79395101216953,86.10147084483768],"median_gflops_per_second":46.64386724587315}'
    ',"step":"S1","to_base":{"first":"matmul_tiled_32","iqr_separated":false,"ratio":'
    '0.6152910512597741,"second":"matmul_x_columns","verdict":"no_clear_difference"},'
    '"to_s0":{"first":"matmul_tiled_32","iqr_separated":false,"ratio":0.6152910512597'
    '741,"second":"matmul_x_columns","verdict":"no_clear_difference"},"variant":"matm'
    'ul_tiled_32"},{"base":"matmul_tiled_32","base_rate":{"iqr_gflops_per_second":[44'
    '.79395101216953,86.10147084483768],"median_gflops_per_second":46.64386724587315}'
    ',"fraction_of_torch":1.0218903477433257,"held":false,"item":"arithmetic_intensit'
    'y","predicted":["faster"],"ran":true,"rate":{"iqr_gflops_per_second":[25.7747393'
    '29284685,137.58864992311635],"median_gflops_per_second":94.76431232225212},"step'
    '":"S2","to_base":{"first":"matmul_s2_coarse_4x1","iqr_separated":false,"ratio":2'
    '.0316564195400515,"second":"matmul_tiled_32","verdict":"no_clear_difference"},"t'
    'o_s0":{"first":"matmul_s2_coarse_4x1","iqr_separated":false,"ratio":1.2500600141'
    '77467,"second":"matmul_x_columns","verdict":"no_clear_difference"},"variant":"ma'
    'tmul_s2_coarse_4x1"},{"base":"matmul_s2_coarse_4x1","base_rate":{"iqr_gflops_per'
    '_second":[25.774739329284685,137.58864992311635],"median_gflops_per_second":94.7'
    '6431232225212},"fraction_of_torch":1.140083497439937,"held":false,"item":"arithm'
    'etic_intensity","predicted":["faster"],"ran":true,"rate":{"iqr_gflops_per_second'
    '":[75.38204324627914,138.46357506581825],"median_gflops_per_second":105.72487435'
    '998424},"step":"S3","to_base":{"first":"matmul_s3_block_4x4","iqr_separated":fal'
    'se,"ratio":1.115661283970067,"second":"matmul_s2_coarse_4x1","verdict":"no_clear'
    '_difference"},"to_s0":{"first":"matmul_s3_block_4x4","iqr_separated":false,"rati'
    'o":1.3946435604568728,"second":"matmul_x_columns","verdict":"no_clear_difference'
    '"},"variant":"matmul_s3_block_4x4"},{"base":"matmul_s3_block_4x4","base_rate":{"'
    'iqr_gflops_per_second":[75.38204324627914,138.46357506581825],"median_gflops_per'
    '_second":105.72487435998424},"fraction_of_torch":1.237568191534844,"held":true,"'
    'item":"memory_vs_compute_bound","predicted":["no_clear_difference"],"ran":true,"'
    'rate":{"iqr_gflops_per_second":[61.21675165336374,142.86080681213411],"median_gf'
    'lops_per_second":114.76505173150919},"step":"S4","to_base":{"first":"matmul_s4_v'
    'ec4","iqr_separated":false,"ratio":1.0855066267635742,"second":"matmul_s3_block_'
    '4x4","verdict":"no_clear_difference"},"to_s0":{"first":"matmul_s4_vec4","iqr_sep'
    'arated":false,"ratio":1.5138948268490808,"second":"matmul_x_columns","verdict":"'
    'no_clear_difference"},"variant":"matmul_s4_vec4"},{"base":"matmul_s4_vec4","base'
    '_rate":{"iqr_gflops_per_second":[61.21675165336374,142.86080681213411],"median_g'
    'flops_per_second":114.76505173150919},"fraction_of_torch":1.19122304526749,"held'
    '":true,"item":"tiling","predicted":["no_clear_difference","slower"],"ran":true,"'
    'rate":{"iqr_gflops_per_second":[102.88825450364124,145.67791849192255],"median_g'
    'flops_per_second":110.4672658436214},"step":"S5","to_base":{"first":"matmul_s5_p'
    'ad4","iqr_separated":false,"ratio":0.9625514403292181,"second":"matmul_s4_vec4",'
    '"verdict":"no_clear_difference"},"to_s0":{"first":"matmul_s5_pad4","iqr_separate'
    'd":true,"ratio":1.457201646090535,"second":"matmul_x_columns","verdict":"faster"'
    '},"variant":"matmul_s5_pad4"},{"base":"matmul_s5_pad4","base_rate":{"iqr_gflops_'
    'per_second":[102.88825450364124,145.67791849192255],"median_gflops_per_second":1'
    '10.4672658436214},"fraction_of_torch":1.2561892230619947,"held":false,"item":"oc'
    'cupancy","predicted":["slower"],"ran":true,"rate":{"iqr_gflops_per_second":[20.5'
    '65039148088562,155.25474609600926],"median_gflops_per_second":116.49186053374346'
    '},"step":"S6","to_base":{"first":"matmul_s6_group_8x8","iqr_separated":false,"ra'
    'tio":1.0545373748876028,"second":"matmul_s5_pad4","verdict":"no_clear_difference'
    '"},"to_s0":{"first":"matmul_s6_group_8x8","iqr_separated":false,"ratio":1.536673'
    '5985502065,"second":"matmul_x_columns","verdict":"no_clear_difference"},"variant'
    '":"matmul_s6_group_8x8"},{"base":"matmul_s5_pad4","base_rate":{"iqr_gflops_per_s'
    'econd":[102.88825450364124,145.67791849192255],"median_gflops_per_second":110.46'
    '72658436214},"fraction_of_torch":0.8058663697104678,"held":true,"item":"occupanc'
    'y","predicted":["no_clear_difference","slower"],"ran":true,"rate":{"iqr_gflops_p'
    'er_second":[34.642805004836966,108.41496607431341],"median_gflops_per_second":74'
    '.73147438752784},"step":"S6","to_base":{"first":"matmul_s6_group_32x32","iqr_sep'
    'arated":false,"ratio":0.6765033407572383,"second":"matmul_s5_pad4","verdict":"no'
    '_clear_difference"},"to_s0":{"first":"matmul_s6_group_32x32","iqr_separated":fal'
    'se,"ratio":0.9858017817371938,"second":"matmul_x_columns","verdict":"no_clear_di'
    'fference"},"variant":"matmul_s6_group_32x32"},{"base":"matmul_s5_pad4","base_rat'
    'e":{"iqr_gflops_per_second":[102.88825450364124,145.67791849192255],"median_gflo'
    'ps_per_second":110.4672658436214},"fraction_of_torch":1.3654113207547172,"held":'
    'false,"item":"control_divergence","predicted":["faster"],"ran":true,"rate":{"iqr'
    '_gflops_per_second":[42.48289134172497,149.79657142857144],"median_gflops_per_se'
    'cond":126.62049811320755},"step":"S7","to_base":{"first":"matmul_s7_interior","i'
    'qr_separated":false,"ratio":1.1462264150943395,"second":"matmul_s5_pad4","verdic'
    't":"no_clear_difference"},"to_s0":{"first":"matmul_s7_interior","iqr_separated":'
    'false,"ratio":1.6702830188679245,"second":"matmul_x_columns","verdict":"no_clear'
    '_difference"},"variant":"matmul_s7_interior"}],"torch":{"iqr_gflops_per_second":'
    '[89.17149759096019,112.31609037656904],"median_gflops_per_second":92.73432568525'
    '898}},{"best":{"fraction_of_torch":2.1443080829454084,"to_s0":{"first":"matmul_s'
    '6_group_8x8","iqr_separated":true,"ratio":1.8205670757511638,"second":"matmul_x_'
    'columns","verdict":"faster"},"variant":"matmul_s6_group_8x8"},"index":1,"k":100,'
    '"m":200,"n":300,"rows":[{"base":"matmul_x_rows","base_rate":{"iqr_gflops_per_sec'
    'ond":[19.350282030360592,24.347981349446286],"median_gflops_per_second":20.37351'
    '443123939},"fraction_of_torch":1.1778242677824269,"held":true,"item":"coalescing'
    '","predicted":["faster"],"ran":true,"rate":{"iqr_gflops_per_second":[36.14457831'
    '3253014,48.05199225562058],"median_gflops_per_second":44.63040446304045},"step":'
    '"S0","to_base":{"first":"matmul_x_columns","iqr_separated":true,"ratio":2.190609'
    '019060902,"second":"matmul_x_rows","verdict":"faster"},"to_s0":{"first":"matmul_'
    'x_columns","iqr_separated":false,"ratio":1.0,"second":"matmul_x_columns","verdic'
    't":"no_clear_difference"},"variant":"matmul_x_columns"},{"base":"matmul_x_column'
    's","base_rate":{"iqr_gflops_per_second":[36.144578313253014,48.05199225562058],"'
    'median_gflops_per_second":44.63040446304045},"fraction_of_torch":0.4063458926586'
    '04,"held":true,"item":"tiling","predicted":["slower","no_clear_difference"],"ran'
    '":true,"rate":{"iqr_gflops_per_second":[10.761125434368344,34.53237410071942],"m'
    'edian_gflops_per_second":15.39735768510992},"step":"S1","to_base":{"first":"matm'
    'ul_tiled_32","iqr_separated":true,"ratio":0.3449970456319941,"second":"matmul_x_'
    'columns","verdict":"slower"},"to_s0":{"first":"matmul_tiled_32","iqr_separated":'
    'true,"ratio":0.3449970456319941,"second":"matmul_x_columns","verdict":"slower"},'
    '"variant":"matmul_tiled_32"},{"base":"matmul_tiled_32","base_rate":{"iqr_gflops_'
    'per_second":[10.761125434368344,34.53237410071942],"median_gflops_per_second":15'
    '.39735768510992},"fraction_of_torch":1.0347151317132939,"held":false,"item":"ari'
    'thmetic_intensity","predicted":["faster"],"ran":true,"rate":{"iqr_gflops_per_sec'
    'ond":[17.58348676146608,55.35272369996056],"median_gflops_per_second":39.2076781'
    '70308355},"step":"S2","to_base":{"first":"matmul_s2_coarse_4x1","iqr_separated":'
    'false,"ratio":2.5463900347151323,"second":"matmul_tiled_32","verdict":"no_clear_'
    'difference"},"to_s0":{"first":"matmul_s2_coarse_4x1","iqr_separated":false,"rati'
    'o":0.8784970390034715,"second":"matmul_x_columns","verdict":"no_clear_difference'
    '"},"variant":"matmul_s2_coarse_4x1"},{"base":"matmul_s2_coarse_4x1","base_rate":'
    '{"iqr_gflops_per_second":[17.58348676146608,55.35272369996056],"median_gflops_pe'
    'r_second":39.207678170308355},"fraction_of_torch":1.094700328215658,"held":false'
    ',"item":"arithmetic_intensity","predicted":["faster"],"ran":true,"rate":{"iqr_gf'
    'lops_per_second":[39.36039360393604,57.03544039677654],"median_gflops_per_second'
    '":41.48065186844411},"step":"S3","to_base":{"first":"matmul_s3_block_4x4","iqr_s'
    'eparated":false,"ratio":1.0579726677071397,"second":"matmul_s2_coarse_4x1","verd'
    'ict":"no_clear_difference"},"to_s0":{"first":"matmul_s3_block_4x4","iqr_separate'
    'd":false,"ratio":0.9294258559273259,"second":"matmul_x_columns","verdict":"no_cl'
    'ear_difference"},"variant":"matmul_s3_block_4x4"},{"base":"matmul_s3_block_4x4",'
    '"base_rate":{"iqr_gflops_per_second":[39.36039360393604,57.03544039677654],"medi'
    'an_gflops_per_second":41.48065186844411},"fraction_of_torch":0.9921023406331598,'
    '"held":true,"item":"memory_vs_compute_bound","predicted":["no_clear_difference"]'
    ',"ran":true,"rate":{"iqr_gflops_per_second":[37.371069809158406,59.6829344109418'
    '7],"median_gflops_per_second":37.59298389610552},"step":"S4","to_base":{"first":'
    '"matmul_s4_vec4","iqr_separated":false,"ratio":0.9062775583983509,"second":"matm'
    'ul_s3_block_4x4","verdict":"no_clear_difference"},"to_s0":{"first":"matmul_s4_ve'
    'c4","iqr_separated":false,"ratio":0.8423177954221143,"second":"matmul_x_columns"'
    ',"verdict":"no_clear_difference"},"variant":"matmul_s4_vec4"},{"base":"matmul_s4'
    '_vec4","base_rate":{"iqr_gflops_per_second":[37.371069809158406,59.6829344109418'
    '7],"median_gflops_per_second":37.59298389610552},"fraction_of_torch":1.773126581'
    '711496,"held":true,"item":"tiling","predicted":["no_clear_difference","slower"],'
    '"ran":true,"rate":{"iqr_gflops_per_second":[12.2360517931679,80.17826300474722],'
    '"median_gflops_per_second":67.18774495532016},"step":"S5","to_base":{"first":"ma'
    'tmul_s5_pad4","iqr_separated":false,"ratio":1.7872416071308594,"second":"matmul_'
    's4_vec4","verdict":"no_clear_difference"},"to_s0":{"first":"matmul_s5_pad4","iqr'
    '_separated":false,"ratio":1.505425410405142,"second":"matmul_x_columns","verdict'
    '":"no_clear_difference"},"variant":"matmul_s5_pad4"},{"base":"matmul_s5_pad4","b'
    'ase_rate":{"iqr_gflops_per_second":[12.2360517931679,80.17826300474722],"median_'
    'gflops_per_second":67.18774495532016},"fraction_of_torch":2.1443080829454084,"he'
    'ld":false,"item":"occupancy","predicted":["slower"],"ran":true,"rate":{"iqr_gflo'
    'ps_per_second":[51.622117496241266,87.35213830755232],"median_gflops_per_second"'
    ':81.25264494286924},"step":"S6","to_base":{"first":"matmul_s6_group_8x8","iqr_se'
    'parated":false,"ratio":1.2093372831146847,"second":"matmul_s5_pad4","verdict":"n'
    'o_clear_difference"},"to_s0":{"first":"matmul_s6_group_8x8","iqr_separated":true'
    ',"ratio":1.8205670757511638,"second":"matmul_x_columns","verdict":"faster"},"var'
    'iant":"matmul_s6_group_8x8"},{"base":"matmul_s5_pad4","base_rate":{"iqr_gflops_p'
    'er_second":[12.2360517931679,80.17826300474722],"median_gflops_per_second":67.18'
    '774495532016},"fraction_of_torch":1.344508292593874,"held":true,"item":"occupanc'
    'y","predicted":["no_clear_difference","slower"],"ran":true,"rate":{"iqr_gflops_p'
    'er_second":[48.77216085058649,58.21710127349909],"median_gflops_per_second":50.9'
    '46436190650054},"step":"S6","to_base":{"first":"matmul_s6_group_32x32","iqr_sepa'
    'rated":false,"ratio":0.7582697741162384,"second":"matmul_s5_pad4","verdict":"no_'
    'clear_difference"},"to_s0":{"first":"matmul_s6_group_32x32","iqr_separated":true'
    ',"ratio":1.1415185858967527,"second":"matmul_x_columns","verdict":"faster"},"var'
    'iant":"matmul_s6_group_32x32"},{"base":"matmul_s5_pad4","base_rate":{"iqr_gflops'
    '_per_second":[12.2360517931679,80.17826300474722],"median_gflops_per_second":67.'
    '18774495532016},"fraction_of_torch":1.6946488294314381,"held":false,"item":"cont'
    'rol_divergence","predicted":["faster"],"ran":true,"rate":{"iqr_gflops_per_second'
    '":[62.90956749672346,78.18659230252999],"median_gflops_per_second":64.2140468227'
    '4247},"step":"S7","to_base":{"first":"matmul_s7_interior","iqr_separated":false,'
    '"ratio":0.9557404682274246,"second":"matmul_s5_pad4","verdict":"no_clear_differe'
    'nce"},"to_s0":{"first":"matmul_s7_interior","iqr_separated":true,"ratio":1.43879'
    '59866220734,"second":"matmul_x_columns","verdict":"faster"},"variant":"matmul_s7'
    '_interior"}],"torch":{"iqr_gflops_per_second":[25.40354591161683,63.660477453580'
    '9],"median_gflops_per_second":37.89224393132031}}],"ratio_rule":"ratio = median '
    "GFLOP/s of the first side / median GFLOP/s of the second, from the same point of"
    " the same run; faster when ratio >= min_faster_ratio and the first side's p75 sa"
    "mple time is below the second side's p25; slower when 1 / ratio >= min_faster_ra"
    "tio and the second side's p75 is below the first side's p25; otherwise no_clear_"
    'difference","reference":"matmul_x_columns","vendor":"vendor GPU matmul: torch.ma'
    'tmul on MPS, PyTorch 2.15.0"},"summary":{"benchmark_points":2,"best":[{"fraction'
    '_of_torch":1.3654113207547172,"k":256,"m":256,"n":256,"over_s0":1.67028301886792'
    '45,"torch_gflops_per_second":92.73432568525898,"variant":"matmul_s7_interior"},{'
    '"fraction_of_torch":2.1443080829454084,"k":100,"m":200,"n":300,"over_s0":1.82056'
    '70757511638,"torch_gflops_per_second":37.89224393132031,"variant":"matmul_s6_gro'
    'up_8x8"}],"cases_with_other_bits":{"matmul_s2_coarse_4x1":[],"matmul_s3_block_4x'
    '4":[],"matmul_s4_vec4":[],"matmul_s5_pad4":[],"matmul_s6_group_32x32":[],"matmul'
    '_s6_group_8x8":[],"matmul_s7_interior":[],"matmul_tiled_32":[],"matmul_x_columns'
    '":[],"matmul_x_rows":[]},"correctness_cases":2,"hypotheses":{"matmul_s2_coarse_4'
    'x1":{"held":0,"missed":2,"no_verdict":0},"matmul_s3_block_4x4":{"held":0,"missed'
    '":2,"no_verdict":0},"matmul_s4_vec4":{"held":2,"missed":0,"no_verdict":0},"matmu'
    'l_s5_pad4":{"held":2,"missed":0,"no_verdict":0},"matmul_s6_group_32x32":{"held":'
    '2,"missed":0,"no_verdict":0},"matmul_s6_group_8x8":{"held":0,"missed":2,"no_verd'
    'ict":0},"matmul_s7_interior":{"held":0,"missed":2,"no_verdict":0},"matmul_tiled_'
    '32":{"held":2,"missed":0,"no_verdict":0},"matmul_x_columns":{"held":2,"missed":0'
    ',"no_verdict":0}},"kernels_checked":["matmul_x_columns","matmul_x_rows","matmul_'
    'tiled_32","matmul_s2_coarse_4x1","matmul_s3_block_4x4","matmul_s4_vec4","matmul_'
    's5_pad4","matmul_s6_group_8x8","matmul_s6_group_32x32","matmul_s7_interior"],"la'
    'bel":"host-specific Apple GPU (Metal) measurement","load_average_after":[21.2973'
    '6328125,23.84912109375,24.2353515625],"load_average_before":[21.29736328125,23.8'
    '4912109375,24.2353515625],"max_bound_ratio":0.22669085662113317,"refused_variant'
    's":[],"timed_samples":66,"warnings":0},"timer":"time.perf_counter_ns","torch_mps'
    '_switches":{"source":"os.environ, read before the runtime imports torch; null me'
    'ans unset, and a value that is not a short token is \'unknown\'","values":{"PYTORC'
    'H_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}},"warnings":{"capture":"w'
    "arnings.catch_warnings(record=True) with simplefilter('always'), from before the"
    " runtime imports torch to the end of the run; equal warnings are kept once with "
    'a count","count":0,"distinct":0,"records":[],"torch_imported_before_capture":fal'
    "se}}"
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
        21.3,
        23.85,
        24.24,
    ]
    assert report["environment"]["torch_version"] == "2.15.0"
    assert report["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert report["evidence_boundary"]["cublas_measured"] is False
    assert report["evidence_boundary"]["occupancy_measured"] is False


def test_the_report_seals_the_plan_that_was_declared_before_the_run():
    report = recorded()

    assert report["checklist"] == {**checklist_mapping(), "sha256": checklist_sha256()}
    assert report["config"]["checklist_sha256"] == checklist_sha256()


def test_the_fresh_process_captured_the_warnings_from_before_the_torch_import():
    captured = recorded()["warnings"]

    assert captured["torch_imported_before_capture"] is False
    assert (captured["count"], captured["records"]) == (0, [])


def test_every_recorded_pipeline_admitted_its_kernel_with_the_declared_bytes():
    report = recorded()

    assert report["refusals"] == [] and report["summary"]["refused_variants"] == []
    for entry in report["shaders"]["variants"]:
        variant = VARIANTS[entry["name"]]
        assert entry["source_sha256"] == variant.source_sha256
        assert entry["limits"] == {
            "max_threads_per_threadgroup": 1024,
            "thread_execution_width": 32,
            "static_threadgroup_memory_bytes": variant.threadgroup_bytes,
        }
        assert entry["admission"] is None or entry["admission"]["admitted"] is True


def test_the_apple_gpu_held_the_replay_bits_for_every_kernel_and_case():
    report = recorded()

    assert set(map(tuple, report["summary"]["cases_with_other_bits"].values())) == {()}
    for case in report["correctness"]["cases"]:
        shape = matmul_shape(case["m"], case["n"], case["k"])
        left, right = matmul_inputs(77, CORRECTNESS_STREAM, case["index"], shape)
        replay = fma_replay(left, right)
        assert case["replay_sha256"] == matrix_sha256(replay)
        for output in case["outputs"]:
            tail = float32_sentinels(output["guard_elements"])
            buffer = numpy.concatenate([replay.reshape(-1), tail])
            assert output["buffer_sha256"] == matrix_sha256(buffer), output["kernel"]
            assert output["replay_bits"] is True


def test_the_recorded_speedup_names_torch_on_mps_and_measures_no_cublas():
    speedup = recorded()["speedup"]

    assert speedup["vendor"] == "vendor GPU matmul: torch.matmul on MPS, PyTorch 2.15.0"
    assert speedup["cublas"] == "not measured, no NVIDIA GPU or cuBLAS on this host"
    for entry in speedup["points"]:
        assert entry["best"]["variant"] in VARIANTS
        rows = [row["variant"] for row in entry["rows"]]
        assert rows == ["matmul_x_columns", *list(VARIANTS)[2:]]


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    runtime = checklist_runtime()

    for entry in recorded()["shaders"]["variants"]:
        assert runtime.limits[entry["name"]].to_mapping() == entry["limits"]
        if entry["admission"] is not None:
            admission = runtime.admissions[entry["name"]].to_mapping()
            assert admission == entry["admission"]
