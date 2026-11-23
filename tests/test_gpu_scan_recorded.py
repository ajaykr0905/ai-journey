"""One real scan report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``gpu_scan_cli run`` wrote
in a fresh process on 2026-11-21 at 02:16 IST under the shared GPU lock, with
torch 2.15.0 on macOS 27.2 (arm64) and load averages of 8.62, 10.81 and 13.00
from other jobs. The run is short: three correctness cases, one sweep point at
2^16, three repeats and one call for each sample. The cost of each synchronize
call thus sets its rates. The timings describe that moment and are here only
to be verified. A host without torch or MPS can still check a genuine Apple
GPU (Metal) report, and the Linux CI does.
"""

import json

import numpy

from ai_journey import gpu_scan_cli
from ai_journey.gpu_scan_config import config_from_mapping
from ai_journey.gpu_scan_plan import VENDOR, plan_mapping, plan_sha256
from ai_journey.gpu_scan_reference import (
    CORRECTNESS_STREAM,
    REPLAYS,
    exact_prefix,
    scan_inputs,
)
from ai_journey.gpu_scan_source import KERNELS, S0_HILLIS_STEELE, guard_elements
from ai_journey.gpu_scan_verify import verify_report
from ai_journey.kernel_metal_reference import array_sha256, float32_sentinels
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_scan_runtime import scan_runtime
from test_gpu_scan_verify import hide_the_first_run_error, reseal
from test_kernel_metal_runtime import requires_metal

RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "gpu-scan-recorded",
    "seed": 78,
    "plan_sha256": "13f7eead88f966d08046b82eaa2693e6956feac22558321f5d65d256ffafec66",
    "correctness": {
        "cases": [
            [31, "standard_normal"],
            [1025, "small_integers"],
            [1048577, "standard_normal"],
        ],
        "runs": 2,
    },
    "benchmark": {
        "lengths": [65536],
        "warmup": 1,
        "repeats": 3,
        "target_bytes_per_sample": 65536,
        "max_runtime_seconds": 60,
    },
    "comparison": {"min_faster_ratio": 1.1},
}
# The canonical report of that run, byte for byte.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"bytes_per_element":8,"bytes_'
    'rule":"GB/s = bytes of one sample / median sample time in ns; 8 bytes for each e'
    "lement and call for the scans and torch.cumsum, 32 bytes for each float4 and cal"
    'l for the copy","check_rule":"every kernel output within its bound of the float6'
    "4 reference, or the run stops; the copy equal to its input bit for bit; torch.cu"
    "msum measured; same_bits_as_s0 compares each output with the S0 output bit for b"
    'it","device":"mps","points":[{"bytes_per_call":524288,"checks":{"scan_s0_hillis_'
    'steele":{"bound":1.251706382998618e-06,"bound_ratio":0.05636555878849591,"max_re'
    'lative_error":7.055312971684418e-08,"same_bits_as_s0":true},"scan_s1_blelloch":{'
    '"bound":2.324593825904829e-06,"bound_ratio":0.030350734365123748,"max_relative_e'
    'rror":7.055312971684418e-08,"same_bits_as_s0":false},"scan_s2_simd":{"bound":7.4'
    '50643384500478e-06,"bound_ratio":0.009469400973292503,"max_relative_error":7.055'
    '312971684418e-08,"same_bits_as_s0":false},"stream_copy":{"same_bits_as_input":tr'
    'ue},"torch_cumsum":{"bound":null,"bound_ratio":null,"max_relative_error":7.05531'
    '2971684418e-08,"same_bits_as_s0":false}},"copy_bytes_per_call":524288,"copy_laun'
    'ch":{"active_threads":4096,"extent":[4096],"group_size":[256],"groups":[16],"idl'
    'e_lanes_per_threadgroup":0,"launched_threads":4096,"padding_threads":0,"simd_gro'
    'ups_per_threadgroup":8,"threads":[4096],"threads_per_group":256},"dispatches":3,'
    '"index":0,"launches":{"scan_s0_hillis_steele":[{"add":{"active_threads":65536,"e'
    'xtent":[65536],"group_size":[256],"groups":[256],"idle_lanes_per_threadgroup":0,'
    '"launched_threads":65536,"padding_threads":0,"simd_groups_per_threadgroup":8,"th'
    'reads":[65536],"threads_per_group":256},"blocks":64,"length":65536,"scan":{"acti'
    've_threads":65536,"extent":[65536],"group_size":[1024],"groups":[64],"idle_lanes'
    '_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"simd_groups_pe'
    'r_threadgroup":32,"threads":[65536],"threads_per_group":1024}},{"add":null,"bloc'
    'ks":1,"length":64,"scan":{"active_threads":1024,"extent":[1024],"group_size":[10'
    '24],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_threads":1024,"padding'
    '_threads":0,"simd_groups_per_threadgroup":32,"threads":[1024],"threads_per_group'
    '":1024}}],"scan_s1_blelloch":[{"add":{"active_threads":65536,"extent":[65536],"g'
    'roup_size":[256],"groups":[256],"idle_lanes_per_threadgroup":0,"launched_threads'
    '":65536,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[65536],"t'
    'hreads_per_group":256},"blocks":64,"length":65536,"scan":{"active_threads":32768'
    ',"extent":[32768],"group_size":[512],"groups":[64],"idle_lanes_per_threadgroup":'
    '0,"launched_threads":32768,"padding_threads":0,"simd_groups_per_threadgroup":16,'
    '"threads":[32768],"threads_per_group":512}},{"add":null,"blocks":1,"length":64,"'
    'scan":{"active_threads":512,"extent":[512],"group_size":[512],"groups":[1],"idle'
    '_lanes_per_threadgroup":0,"launched_threads":512,"padding_threads":0,"simd_group'
    's_per_threadgroup":16,"threads":[512],"threads_per_group":512}}],"scan_s2_simd":'
    '[{"add":{"active_threads":65536,"extent":[65536],"group_size":[256],"groups":[25'
    '6],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"'
    'simd_groups_per_threadgroup":8,"threads":[65536],"threads_per_group":256},"block'
    's":64,"length":65536,"scan":{"active_threads":65536,"extent":[65536],"group_size'
    '":[1024],"groups":[64],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"'
    'padding_threads":0,"simd_groups_per_threadgroup":32,"threads":[65536],"threads_p'
    'er_group":1024}},{"add":null,"blocks":1,"length":64,"scan":{"active_threads":102'
    '4,"extent":[1024],"group_size":[1024],"groups":[1],"idle_lanes_per_threadgroup":'
    '0,"launched_threads":1024,"padding_threads":0,"simd_groups_per_threadgroup":32,"'
    'threads":[1024],"threads_per_group":1024}}]},"length":65536,"levels":[65536,64],'
    '"modeled_bytes_per_call":1041408,"side_order":["scan_s0_hillis_steele","scan_s1_'
    'blelloch","scan_s2_simd","torch_cumsum","stream_copy"],"sides":{"scan_s0_hillis_'
    'steele":{"best_gb_per_second":3.2066544342507646,"bytes_per_sample":524288,"call'
    's_per_sample":1,"implementation":"scan_s0_hillis_steele","iqr_gb_per_second":[2.'
    '8365497503151493,3.2066544342507646],"iqr_ns_per_call":21333.0,"median_gb_per_se'
    'cond":3.1480827659089,"median_ns_per_call":166542.0,"repeats":3,"samples_ns":[18'
    '4833,166542,163500],"timing":{"count":3,"iqr_ns":21333,"max_ns":184833,"median_n'
    's":166542,"min_ns":163500,"p25_ns":163500,"p75_ns":184833},"warmup":1},"scan_s1_'
    'blelloch":{"best_gb_per_second":2.9620790960451977,"bytes_per_sample":524288,"ca'
    'lls_per_sample":1,"implementation":"scan_s1_blelloch","iqr_gb_per_second":[1.390'
    '8356081165325,2.9620790960451977],"iqr_ns_per_call":199959.0,"median_gb_per_seco'
    'nd":2.604083781595665,"median_ns_per_call":201333.0,"repeats":3,"samples_ns":[37'
    '6959,201333,177000],"timing":{"count":3,"iqr_ns":199959,"max_ns":376959,"median_'
    'ns":201333,"min_ns":177000,"p25_ns":177000,"p75_ns":376959},"warmup":1},"scan_s2'
    '_simd":{"best_gb_per_second":3.8586053357865686,"bytes_per_sample":524288,"calls'
    '_per_sample":1,"implementation":"scan_s2_simd","iqr_gb_per_second":[1.5480929644'
    '754287,3.8586053357865686],"iqr_ns_per_call":202792.0,"median_gb_per_second":3.2'
    '870721003134795,"median_ns_per_call":159500.0,"repeats":3,"samples_ns":[338667,1'
    '59500,135875],"timing":{"count":3,"iqr_ns":202792,"max_ns":338667,"median_ns":15'
    '9500,"min_ns":135875,"p25_ns":135875,"p75_ns":338667},"warmup":1},"stream_copy":'
    '{"best_gb_per_second":4.229561625711935,"bytes_per_sample":524288,"calls_per_sam'
    'ple":1,"implementation":"stream_copy","iqr_gb_per_second":[3.2939491226196385,4.'
    '229561625711935],"iqr_ns_per_call":35209.0,"median_gb_per_second":3.723847945906'
    '017,"median_ns_per_call":140792.0,"repeats":3,"samples_ns":[159167,140792,123958'
    '],"timing":{"count":3,"iqr_ns":35209,"max_ns":159167,"median_ns":140792,"min_ns"'
    ':123958,"p25_ns":123958,"p75_ns":159167},"warmup":1},"torch_cumsum":{"best_gb_pe'
    'r_second":3.344740031897927,"bytes_per_sample":524288,"calls_per_sample":1,"impl'
    'ementation":"torch_cumsum","iqr_gb_per_second":[2.6390291291457566,3.34474003189'
    '7927],"iqr_ns_per_call":41917.0,"median_gb_per_second":2.7395845852384064,"media'
    'n_ns_per_call":191375.0,"repeats":3,"samples_ns":[198667,191375,156750],"timing"'
    ':{"count":3,"iqr_ns":41917,"max_ns":198667,"median_ns":191375,"min_ns":156750,"p'
    '25_ns":156750,"p75_ns":198667},"warmup":1}}}],"synchronization":"torch.mps.synch'
    "ronize() before the first and before the second clock reading of every sample; u"
    "ntimed warmups of each side first; repeat r starts with side r mod S, so the ord"
    'er shifts by one side every repeat","traffic_model":"modeled device bytes of one'
    " scan with no cache: 4 * (2 L + B) for each level of L elements in B blocks, and"
    " 4 * (2 (L - min(L, 1024)) + B - 1) for the add below each level but the top; on"
    "e float32 read and one float32 write of each element, 8 N bytes, is the compulso"
    'ry part"},"config":{"benchmark":{"lengths":[65536],"max_runtime_seconds":60,"rep'
    'eats":3,"target_bytes_per_sample":65536,"warmup":1},"comparison":{"min_faster_ra'
    'tio":1.1},"correctness":{"cases":[[31,"standard_normal"],[1025,"small_integers"]'
    ',[1048577,"standard_normal"]],"runs":2},"name":"gpu-scan-recorded","plan_sha256"'
    ':"13f7eead88f966d08046b82eaa2693e6956feac22558321f5d65d256ffafec66","schema_vers'
    'ion":1,"seed":78},"config_sha256":"e667165772b456b02d990e4d9c0a4548f1a2abbc5f597'
    '86a0de1a0382c4ee32c","correctness":{"bound_rule":"(gamma_D(2**-24) + gamma_N(2**'
    "-53)) / (1 - gamma_N(2**-53)), gamma_k(u) = k u / (1 - k u); D = levels * block_"
    'depth + levels - 1 of the kernel at length N","bound_source":"N. J. Higham, Accu'
    "racy and Stability of Numerical Algorithms, 2nd ed., SIAM, 2002, section 4.2: a "
    "sum whose terms pass at most D additions has error at most gamma_D times the sum"
    ' of their absolute values","cases":[{"exact_sha256":null,"index":0,"input_sha256'
    '":"f56d3f274665b9fe1a3fe68f2736756560b13287aa092181916b26f1283ffb83","launches":'
    '{"scan_s0_hillis_steele":[{"add":null,"blocks":1,"length":31,"scan":{"active_thr'
    'eads":1024,"extent":[1024],"group_size":[1024],"groups":[1],"idle_lanes_per_thre'
    'adgroup":0,"launched_threads":1024,"padding_threads":0,"simd_groups_per_threadgr'
    'oup":32,"threads":[1024],"threads_per_group":1024}}],"scan_s1_blelloch":[{"add":'
    'null,"blocks":1,"length":31,"scan":{"active_threads":512,"extent":[512],"group_s'
    'ize":[512],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_threads":512,"p'
    'adding_threads":0,"simd_groups_per_threadgroup":16,"threads":[512],"threads_per_'
    'group":512}}],"scan_s2_simd":[{"add":null,"blocks":1,"length":31,"scan":{"active'
    '_threads":1024,"extent":[1024],"group_size":[1024],"groups":[1],"idle_lanes_per_'
    'threadgroup":0,"launched_threads":1024,"padding_threads":0,"simd_groups_per_thre'
    'adgroup":32,"threads":[1024],"threads_per_group":1024}}]},"length":31,"levels":['
    '31],"outputs":[{"bound":5.960468064671793e-07,"bound_ratio":0.07360778073319106,'
    '"buffer_sha256":"7dd210f4c193a300e396ee26d204fdd21e92ab11f2f15ced1187f9a1842529e'
    'f","deterministic":true,"exact_bits":null,"guard_elements":2017,"max_relative_er'
    'ror":4.3873682637154896e-08,"replay_bits":true,"replay_differences":0,"run_sha25'
    '6":["7dd210f4c193a300e396ee26d204fdd21e92ab11f2f15ced1187f9a1842529ef","7dd210f4'
    'c193a300e396ee26d204fdd21e92ab11f2f15ced1187f9a1842529ef"],"side":"scan_s0_hilli'
    's_steele"},{"bound":1.1324895367052076e-06,"bound_ratio":0.03019947077233526,"bu'
    'ffer_sha256":"119a9433deeaaeb4b3edf9f0738b73ce229764d27cc9571f74823f2a7601bd8d",'
    '"deterministic":true,"exact_bits":null,"guard_elements":2017,"max_relative_error'
    '":3.4200584663704415e-08,"replay_bits":true,"replay_differences":0,"run_sha256":'
    '["119a9433deeaaeb4b3edf9f0738b73ce229764d27cc9571f74823f2a7601bd8d","119a9433dee'
    'aaeb4b3edf9f0738b73ce229764d27cc9571f74823f2a7601bd8d"],"side":"scan_s1_blelloch'
    '"},{"bound":3.695501636197772e-06,"bound_ratio":0.009254652826752016,"buffer_sha'
    '256":"72ea2adfa701905ee85e8197648193084729c24c3e415e5c07a592b094aecb40","determi'
    'nistic":true,"exact_bits":null,"guard_elements":2017,"max_relative_error":3.4200'
    '584663704415e-08,"replay_bits":null,"replay_differences":null,"run_sha256":["72e'
    'a2adfa701905ee85e8197648193084729c24c3e415e5c07a592b094aecb40","72ea2adfa701905e'
    'e85e8197648193084729c24c3e415e5c07a592b094aecb40"],"side":"scan_s2_simd"},{"boun'
    'd":null,"bound_ratio":null,"buffer_sha256":"59a635238ae67d488474b4691cafba253985'
    '6cac2459b6a094c67dbd3c0d60d2","deterministic":true,"exact_bits":null,"guard_elem'
    'ents":null,"max_relative_error":3.4200584663704415e-08,"replay_bits":null,"repla'
    'y_differences":null,"run_sha256":["59a635238ae67d488474b4691cafba2539856cac2459b'
    '6a094c67dbd3c0d60d2","59a635238ae67d488474b4691cafba2539856cac2459b6a094c67dbd3c'
    '0d60d2"],"side":"torch_cumsum"}],"pattern":"standard_normal","reference_sha256":'
    '"7745f9f721f4818173c864479bf58887a09602369a6fe610b88fe17f83bb9a76","replay_sha25'
    '6":{"scan_s0_hillis_steele":"f970f4ba880ec8cbf1c00d96d6e841c2d6c61bc696d7cdd019c'
    '22a5146b320b2","scan_s1_blelloch":"9b68b70bc0d66cc7c38611bb39372b8d5e419f5c0922b'
    'f4d076aa8d2d5f91729"}},{"exact_sha256":"dec5d0ac61c9c3608f86fa65b6d780c81ad8e912'
    '659034045bc46029e822192a","index":1,"input_sha256":"72a0e4077020b98905d5e2cf620b'
    'd4fbff3ea2d59401c9f66d7b7efa0a999fd7","launches":{"scan_s0_hillis_steele":[{"add'
    '":{"active_threads":1025,"extent":[1025],"group_size":[256],"groups":[5],"idle_l'
    'anes_per_threadgroup":0,"launched_threads":1280,"padding_threads":255,"simd_grou'
    'ps_per_threadgroup":8,"threads":[1280],"threads_per_group":256},"blocks":2,"leng'
    'th":1025,"scan":{"active_threads":2048,"extent":[2048],"group_size":[1024],"grou'
    'ps":[2],"idle_lanes_per_threadgroup":0,"launched_threads":2048,"padding_threads"'
    ':0,"simd_groups_per_threadgroup":32,"threads":[2048],"threads_per_group":1024}},'
    '{"add":null,"blocks":1,"length":2,"scan":{"active_threads":1024,"extent":[1024],'
    '"group_size":[1024],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_thread'
    's":1024,"padding_threads":0,"simd_groups_per_threadgroup":32,"threads":[1024],"t'
    'hreads_per_group":1024}}],"scan_s1_blelloch":[{"add":{"active_threads":1025,"ext'
    'ent":[1025],"group_size":[256],"groups":[5],"idle_lanes_per_threadgroup":0,"laun'
    'ched_threads":1280,"padding_threads":255,"simd_groups_per_threadgroup":8,"thread'
    's":[1280],"threads_per_group":256},"blocks":2,"length":1025,"scan":{"active_thre'
    'ads":1024,"extent":[1024],"group_size":[512],"groups":[2],"idle_lanes_per_thread'
    'group":0,"launched_threads":1024,"padding_threads":0,"simd_groups_per_threadgrou'
    'p":16,"threads":[1024],"threads_per_group":512}},{"add":null,"blocks":1,"length"'
    ':2,"scan":{"active_threads":512,"extent":[512],"group_size":[512],"groups":[1],"'
    'idle_lanes_per_threadgroup":0,"launched_threads":512,"padding_threads":0,"simd_g'
    'roups_per_threadgroup":16,"threads":[512],"threads_per_group":512}}],"scan_s2_si'
    'md":[{"add":{"active_threads":1025,"extent":[1025],"group_size":[256],"groups":['
    '5],"idle_lanes_per_threadgroup":0,"launched_threads":1280,"padding_threads":255,'
    '"simd_groups_per_threadgroup":8,"threads":[1280],"threads_per_group":256},"block'
    's":2,"length":1025,"scan":{"active_threads":2048,"extent":[2048],"group_size":[1'
    '024],"groups":[2],"idle_lanes_per_threadgroup":0,"launched_threads":2048,"paddin'
    'g_threads":0,"simd_groups_per_threadgroup":32,"threads":[2048],"threads_per_grou'
    'p":1024}},{"add":null,"blocks":1,"length":2,"scan":{"active_threads":1024,"exten'
    't":[1024],"group_size":[1024],"groups":[1],"idle_lanes_per_threadgroup":0,"launc'
    'hed_threads":1024,"padding_threads":0,"simd_groups_per_threadgroup":32,"threads"'
    ':[1024],"threads_per_group":1024}}]},"length":1025,"levels":[1025,2],"outputs":['
    '{"bound":1.2516992208298991e-06,"bound_ratio":0.0,"buffer_sha256":"08eeafd1210a2'
    'bac9ecec917c829ffd3c29dbd00f78c6bcdc85f577e8c7e472a","deterministic":true,"exact'
    '_bits":true,"guard_elements":2047,"max_relative_error":0.0,"replay_bits":true,"r'
    'eplay_differences":0,"run_sha256":["08eeafd1210a2bac9ecec917c829ffd3c29dbd00f78c'
    '6bcdc85f577e8c7e472a","08eeafd1210a2bac9ecec917c829ffd3c29dbd00f78c6bcdc85f577e8'
    'c7e472a"],"side":"scan_s0_hillis_steele"},{"bound":2.324586663728426e-06,"bound_'
    'ratio":0.0,"buffer_sha256":"08eeafd1210a2bac9ecec917c829ffd3c29dbd00f78c6bcdc85f'
    '577e8c7e472a","deterministic":true,"exact_bits":true,"guard_elements":2047,"max_'
    'relative_error":0.0,"replay_bits":true,"replay_differences":0,"run_sha256":["08e'
    'eafd1210a2bac9ecec917c829ffd3c29dbd00f78c6bcdc85f577e8c7e472a","08eeafd1210a2bac'
    '9ecec917c829ffd3c29dbd00f78c6bcdc85f577e8c7e472a"],"side":"scan_s1_blelloch"},{"'
    'bound":7.450636222287361e-06,"bound_ratio":0.0,"buffer_sha256":"08eeafd1210a2bac'
    '9ecec917c829ffd3c29dbd00f78c6bcdc85f577e8c7e472a","deterministic":true,"exact_bi'
    'ts":true,"guard_elements":2047,"max_relative_error":0.0,"replay_bits":null,"repl'
    'ay_differences":null,"run_sha256":["08eeafd1210a2bac9ecec917c829ffd3c29dbd00f78c'
    '6bcdc85f577e8c7e472a","08eeafd1210a2bac9ecec917c829ffd3c29dbd00f78c6bcdc85f577e8'
    'c7e472a"],"side":"scan_s2_simd"},{"bound":null,"bound_ratio":null,"buffer_sha256'
    '":"dec5d0ac61c9c3608f86fa65b6d780c81ad8e912659034045bc46029e822192a","determinis'
    'tic":true,"exact_bits":true,"guard_elements":null,"max_relative_error":0.0,"repl'
    'ay_bits":null,"replay_differences":null,"run_sha256":["dec5d0ac61c9c3608f86fa65b'
    '6d780c81ad8e912659034045bc46029e822192a","dec5d0ac61c9c3608f86fa65b6d780c81ad8e9'
    '12659034045bc46029e822192a"],"side":"torch_cumsum"}],"pattern":"small_integers",'
    '"reference_sha256":"e421f1496482cdf7fe18bcef522496fc56a291971c0d6783ececa4543a05'
    'f5d2","replay_sha256":{"scan_s0_hillis_steele":"dec5d0ac61c9c3608f86fa65b6d780c8'
    '1ad8e912659034045bc46029e822192a","scan_s1_blelloch":"dec5d0ac61c9c3608f86fa65b6'
    'd780c81ad8e912659034045bc46029e822192a"}},{"exact_sha256":null,"index":2,"input_'
    'sha256":"32a9c73347a48207ba36ef6896a2349bada940eb0af7972528351cf759d80f5e","laun'
    'ches":{"scan_s0_hillis_steele":[{"add":{"active_threads":1048577,"extent":[10485'
    '77],"group_size":[256],"groups":[4097],"idle_lanes_per_threadgroup":0,"launched_'
    'threads":1048832,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads"'
    ':[1048832],"threads_per_group":256},"blocks":1025,"length":1048577,"scan":{"acti'
    've_threads":1049600,"extent":[1049600],"group_size":[1024],"groups":[1025],"idle'
    '_lanes_per_threadgroup":0,"launched_threads":1049600,"padding_threads":0,"simd_g'
    'roups_per_threadgroup":32,"threads":[1049600],"threads_per_group":1024}},{"add":'
    '{"active_threads":1025,"extent":[1025],"group_size":[256],"groups":[5],"idle_lan'
    'es_per_threadgroup":0,"launched_threads":1280,"padding_threads":255,"simd_groups'
    '_per_threadgroup":8,"threads":[1280],"threads_per_group":256},"blocks":2,"length'
    '":1025,"scan":{"active_threads":2048,"extent":[2048],"group_size":[1024],"groups'
    '":[2],"idle_lanes_per_threadgroup":0,"launched_threads":2048,"padding_threads":0'
    ',"simd_groups_per_threadgroup":32,"threads":[2048],"threads_per_group":1024}},{"'
    'add":null,"blocks":1,"length":2,"scan":{"active_threads":1024,"extent":[1024],"g'
    'roup_size":[1024],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_threads"'
    ':1024,"padding_threads":0,"simd_groups_per_threadgroup":32,"threads":[1024],"thr'
    'eads_per_group":1024}}],"scan_s1_blelloch":[{"add":{"active_threads":1048577,"ex'
    'tent":[1048577],"group_size":[256],"groups":[4097],"idle_lanes_per_threadgroup":'
    '0,"launched_threads":1048832,"padding_threads":255,"simd_groups_per_threadgroup"'
    ':8,"threads":[1048832],"threads_per_group":256},"blocks":1025,"length":1048577,"'
    'scan":{"active_threads":524800,"extent":[524800],"group_size":[512],"groups":[10'
    '25],"idle_lanes_per_threadgroup":0,"launched_threads":524800,"padding_threads":0'
    ',"simd_groups_per_threadgroup":16,"threads":[524800],"threads_per_group":512}},{'
    '"add":{"active_threads":1025,"extent":[1025],"group_size":[256],"groups":[5],"id'
    'le_lanes_per_threadgroup":0,"launched_threads":1280,"padding_threads":255,"simd_'
    'groups_per_threadgroup":8,"threads":[1280],"threads_per_group":256},"blocks":2,"'
    'length":1025,"scan":{"active_threads":1024,"extent":[1024],"group_size":[512],"g'
    'roups":[2],"idle_lanes_per_threadgroup":0,"launched_threads":1024,"padding_threa'
    'ds":0,"simd_groups_per_threadgroup":16,"threads":[1024],"threads_per_group":512}'
    '},{"add":null,"blocks":1,"length":2,"scan":{"active_threads":512,"extent":[512],'
    '"group_size":[512],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_threads'
    '":512,"padding_threads":0,"simd_groups_per_threadgroup":16,"threads":[512],"thre'
    'ads_per_group":512}}],"scan_s2_simd":[{"add":{"active_threads":1048577,"extent":'
    '[1048577],"group_size":[256],"groups":[4097],"idle_lanes_per_threadgroup":0,"lau'
    'nched_threads":1048832,"padding_threads":255,"simd_groups_per_threadgroup":8,"th'
    'reads":[1048832],"threads_per_group":256},"blocks":1025,"length":1048577,"scan":'
    '{"active_threads":1049600,"extent":[1049600],"group_size":[1024],"groups":[1025]'
    ',"idle_lanes_per_threadgroup":0,"launched_threads":1049600,"padding_threads":0,"'
    'simd_groups_per_threadgroup":32,"threads":[1049600],"threads_per_group":1024}},{'
    '"add":{"active_threads":1025,"extent":[1025],"group_size":[256],"groups":[5],"id'
    'le_lanes_per_threadgroup":0,"launched_threads":1280,"padding_threads":255,"simd_'
    'groups_per_threadgroup":8,"threads":[1280],"threads_per_group":256},"blocks":2,"'
    'length":1025,"scan":{"active_threads":2048,"extent":[2048],"group_size":[1024],"'
    'groups":[2],"idle_lanes_per_threadgroup":0,"launched_threads":2048,"padding_thre'
    'ads":0,"simd_groups_per_threadgroup":32,"threads":[2048],"threads_per_group":102'
    '4}},{"add":null,"blocks":1,"length":2,"scan":{"active_threads":1024,"extent":[10'
    '24],"group_size":[1024],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_th'
    'reads":1024,"padding_threads":0,"simd_groups_per_threadgroup":32,"threads":[1024'
    '],"threads_per_group":1024}}]},"length":1048577,"levels":[1048577,1025,2],"outpu'
    'ts":[{"bound":1.9074686864531678e-06,"bound_ratio":0.0192425308034411,"buffer_sh'
    'a256":"d12cd790f3807f6267d2ad025d87933b8335a994e5345229b2a23e4b5fc6790e","determ'
    'inistic":true,"exact_bits":null,"guard_elements":2047,"max_relative_error":3.670'
    '452495567441e-08,"replay_bits":true,"replay_differences":0,"run_sha256":["d12cd7'
    '90f3807f6267d2ad025d87933b8335a994e5345229b2a23e4b5fc6790e","d12cd790f3807f6267d'
    '2ad025d87933b8335a994e5345229b2a23e4b5fc6790e"],"side":"scan_s0_hillis_steele"},'
    '{"bound":3.5168028246301265e-06,"bound_ratio":0.007695880008784749,"buffer_sha25'
    '6":"252819cf948087600b635ea205ada9b04e58849a1fa9e3cd4d6987166e0ee04c","determini'
    'stic":true,"exact_bits":null,"guard_elements":2047,"max_relative_error":2.706489'
    '255290873e-08,"replay_bits":true,"replay_differences":0,"run_sha256":["252819cf9'
    '48087600b635ea205ada9b04e58849a1fa9e3cd4d6987166e0ee04c","252819cf948087600b635e'
    'a205ada9b04e58849a1fa9e3cd4d6987166e0ee04c"],"side":"scan_s1_blelloch"},{"bound"'
    ':1.1205915203030184e-05,"bound_ratio":0.00241523267511342,"buffer_sha256":"223a9'
    'b2bf3753e18481d910e247f4aa29e66e87b5402ce4246f25f36a8b86744","deterministic":tru'
    'e,"exact_bits":null,"guard_elements":2047,"max_relative_error":2.706489255290873'
    'e-08,"replay_bits":null,"replay_differences":null,"run_sha256":["223a9b2bf3753e1'
    '8481d910e247f4aa29e66e87b5402ce4246f25f36a8b86744","223a9b2bf3753e18481d910e247f'
    '4aa29e66e87b5402ce4246f25f36a8b86744"],"side":"scan_s2_simd"},{"bound":null,"bou'
    'nd_ratio":null,"buffer_sha256":"2dc125882f00c41a7f267782c01efdd6ee51ff29ad5e167b'
    'cbefe42860818861","deterministic":false,"exact_bits":null,"guard_elements":null,'
    '"max_relative_error":2.706489255290873e-08,"replay_bits":null,"replay_difference'
    's":null,"run_sha256":["2dc125882f00c41a7f267782c01efdd6ee51ff29ad5e167bcbefe4286'
    '0818861","ffa0c6aeeb7ee2e96251134778f9f672d95f3568b7e68290d356320fcdcad911"],"si'
    'de":"torch_cumsum"}],"pattern":"standard_normal","reference_sha256":"303a38646ad'
    '70f8eb67b8524c9a6da6bd9fcf67d0b2170f35823066d5760cec1","replay_sha256":{"scan_s0'
    '_hillis_steele":"20dea81eec2b31e0934308ba902f1bc4ba549c91769b4686fe2e144852880ad'
    '9","scan_s1_blelloch":"8912242b67d0e48859636337936284f5b81675f702f93b634646a328a'
    '2fa8c76"}}],"error_rule":"max over i of |output[i] - reference[i]| / sum(|x[j]|,'
    " j <= i), with the float64 reference and sum from numpy.cumsum; an equal element"
    ' counts 0","patterns":["standard_normal","small_integers"],"reference":"numpy.cu'
    "msum of the float64 inputs in order, rebuilt from the seed; S0 and S1 also again"
    'st their float32 CPU replay, S2 and torch.cumsum against no replay","rules":"eve'
    "ry kernel run keeps its sentinel guard tail and stays within its bound, and on s"
    "mall_integers equals the exact prefix sums, or the run stops; deterministic, rep"
    'lay_bits and every torch.cumsum value are measured","runs":2,"sentinel_float32_b'
    'its":2143289344},"environment":{"macos_version":"27.2","numpy_version":"1.26.4",'
    '"platform_machine":"arm64","platform_system":"Darwin","python_implementation":"C'
    'Python","python_version":"3.11.14","torch_version":"2.15.0"},"evidence_boundary"'
    ':{"apple_gpu_metal_this_host":true,"cuda_kernel_executed":false,"distributed_cla'
    'im":false,"hardware_counters_used":false,"host_specific":true,"nvidia_gpu_used":'
    'false,"occupancy_measured":false,"portable_performance_claim":false,"production_'
    'claim":false,"statement":"These are wall-clock timings on the Apple GPU of the h'
    "ost that made this report, at the recorded load averages. The scan kernels and t"
    "he Day 74 copy came from torch.mps.compile_shader, and torch.cumsum ran on the M"
    "PS device. PyTorch does not report which kernel torch.cumsum ran, so the vendor "
    "number names the call, not a kernel. The SIMD-group functions of S2 are the Meta"
    "l counterpart of CUDA warp shuffles, not the same instruction. No hardware count"
    "er was read, so a bound name is an inference from the timings and the same-run c"
    "opy rate. No CUDA kernel ran, and no NVIDIA GPU was used. Each ratio compares tw"
    "o sides of one run on this host. These timings do not measure or predict CUDA, N"
    'VIDIA, distributed or production performance.","vendor_kernel_known":false,"vend'
    'or_peak_claimed":false,"warp_shuffle_executed":false},"host_load":{"after":[8.62'
    '060546875,10.81396484375,12.99658203125],"before":[8.62060546875,10.81396484375,'
    '12.99658203125],"source":"os.getloadavg(): the 1, 5 and 15 minute load averages '
    'that uptime prints, read before the checks and after the last sample"},"kind":"g'
    'pu_scan_correctness_and_bandwidth","label":"host-specific Apple GPU (Metal) meas'
    'urement","measured":true,"measurement_scope":"apple_gpu_metal_this_host","percen'
    'tile_method":"nearest rank: sorted samples[ceil(p * n / 100) - 1] for p in 25, 5'
    '0 and 75; median is p50 and IQR is p75 - p25; no interpolation","plan":{"add_ker'
    'nel_sha256":"1151ad028c5dce620771ed8a5b31d4a728e0e2d751809a3b72c8adcd1d55b131","'
    'copy":"stream_copy","hypotheses":[{"base":"torch_cumsum","change":"Scan each blo'
    "ck of 1024 elements with the 10 Hillis-Steele steps in two threadgroup buffers, "
    "with one barrier at each step. Then scan the block sums with the same kernel and"
    ' add the sum of the earlier blocks to each element.","hypothesis":"PyTorch does '
    "not report which kernel torch.cumsum runs. An untimed check before this plan ran"
    " each side three times. torch.cumsum gave other bits from run to run at 65537 an"
    "d 1048577 elements, and the three scans of this lab did not. Sums in an order th"
    "at changes from run to run can come from a single-pass scan. Such a scan reads a"
    "nd writes each element once, in one dispatch. S0 moves about 16 bytes for each e"
    "lement in 3 or 5 dispatches. On Day 73, the median time of torch.add was 0.85 to"
    " 1.24 times that of a Metal vector add. Thus S0 is slower than torch.cumsum at e"
    'very length.","kernel":"scan_s0_hillis_steele","predictions":[{"length":65536,"v'
    'erdicts":["slower"]},{"length":1048576,"verdicts":["slower"]},{"length":16777216'
    ',"verdicts":["slower"]},{"length":5000003,"verdicts":["slower"]}],"step":"S0"},{'
    '"base":"scan_s0_hillis_steele","change":"Scan each block with 512 threads in one'
    " threadgroup buffer: an up-sweep and a down-sweep of 10 levels each, with a barr"
    "ier at each level. Element j takes the exclusive sum of element j + 1, and the l"
    'ast element takes the block total.","hypothesis":"S1 makes about 2 adds for each'
    " element, and S0 makes about 9. But S1 waits at 22 barriers in each block, and S"
    "0 waits at 11. At each level, the lanes of S1 read words 2 to 512 words apart. O"
    "n Day 74, threadgroup reads at strides of 2 to 32 words took 2.0 to 21.5 times a"
    "s long as reads at stride 1. Both kernels make the same dispatches and move the "
    "same modeled bytes. Thus S1 is not faster than S0 at any length, and at 2^16 the"
    ' run shows no clear difference.","kernel":"scan_s1_blelloch","predictions":[{"le'
    'ngth":65536,"verdicts":["no_clear_difference"]},{"length":1048576,"verdicts":["n'
    'o_clear_difference","slower"]},{"length":16777216,"verdicts":["no_clear_differen'
    'ce","slower"]},{"length":5000003,"verdicts":["no_clear_difference","slower"]}],"'
    'step":"S1"},{"base":"scan_s0_hillis_steele","change":"Scan each run of 32 elemen'
    "ts with simd_prefix_inclusive_sum. SIMD group 0 scans the 32 run totals and move"
    "s them up one lane with simd_shuffle_up. Each thread adds the total of the runs "
    'before its own, with 2 barriers in each block.","hypothesis":"SIMD-group functio'
    "ns are the Metal counterpart of CUDA warp shuffles, not the same instruction. S2"
    " replaces the 10 threadgroup steps and 11 barriers of S0 with two SIMD-group sca"
    "ns and 2 barriers. At 2^16, the dispatches probably set the time, so the run sho"
    "ws no clear difference. S0 makes about 31 threadgroup accesses for each element."
    " If each access takes the Day 74 time of 1.3 ps for a stride-1 threadgroup read,"
    " S0 spends about 42 us on them at 2^20. Its 16.8 MB of modeled traffic take 168 "
    "us at the Day 74 copy ceiling of about 100 GB/s. Thus S2 is faster than S0 by 1."
    '1 or more at 2^20, 2^24 and the ragged length.","kernel":"scan_s2_simd","predict'
    'ions":[{"length":65536,"verdicts":["no_clear_difference"]},{"length":1048576,"ve'
    'rdicts":["faster"]},{"length":16777216,"verdicts":["faster"]},{"length":5000003,'
    '"verdicts":["faster"]}],"step":"S2"}],"kernels":{"scan_s0_hillis_steele":"a8db04'
    '86becd559294f413bc7f32b71d62502bb874e1fd1da4d6d3cb102249fe","scan_s1_blelloch":"'
    'd2087a1a14eb30c7b2b4da1e3646fdabe6ab5b948ca8c70b57a80c76b7840f85","scan_s2_simd"'
    ':"ba41e335c10dd9dc5910678ffe6fa64449baae4856228f2c2250ad87be729dab"},"lengths":['
    '65536,1048576,16777216,5000003],"library_sha256":"39df7e35f24dcf5fd41ef7553563cc'
    '75eb5fab2ec9a501e192457d11ff5be1c5","sha256":"13f7eead88f966d08046b82eaa2693e695'
    '6feac22558321f5d65d256ffafec66","sides":["scan_s0_hillis_steele","scan_s1_blello'
    'ch","scan_s2_simd","torch_cumsum","stream_copy"],"vendor":"torch_cumsum"},"plann'
    'ed_work":{"benchmark_points":1,"bytes":10485760,"calls":20,"case_elements":10496'
    '33,"correctness_cases":3,"correctness_runs":24,"sides_per_point":5,"timed_sample'
    's":15},"refusals":[],"report_sha256":"2517c7825d88ce1994d4944004dbd5df9f74c45f49'
    'ed1bda236b668851c1b3bb","run":{"elapsed_ns":555599875,"max_runtime_ns":600000000'
    '00,"timed_ns":2874918},"schema_version":1,"shaders":{"add":{"group_size":256,"li'
    'mits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"t'
    'hread_execution_width":32},"name":"scan_add_offsets","source_sha256":"1151ad028c'
    '5dce620771ed8a5b31d4a728e0e2d751809a3b72c8adcd1d55b131","threadgroup_bytes":0},"'
    'copy":{"float4_per_thread":4,"group_size":256,"kernel":"stream_copy","limits":{"'
    'max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_ex'
    'ecution_width":32},"source_sha256":"735583b6a215ae98d3092c68ab7f3e5f6eeaf7ca4bcc'
    '5bae32827447cecac008","text":"stream_copy from the Day 74 ceiling source, unchan'
    "ged: ceil(N / 4) float4 values from a zero-padded copy of the input, 4 for each "
    'thread in groups of 256, as the pinned Day 74 config runs it"},"kernels":[{"admi'
    'ssion":{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":8192'
    ',"name":"scan_s0_hillis_steele","reason":null,"required_simd_width":null,"requir'
    'ed_threads":1024,"simd_width":32,"static_threadgroup_bytes":8192},"algorithm":"h'
    'illis_steele","barriers_per_block":11,"block":1024,"block_depth":10,"elements_pe'
    'r_thread":1,"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_mem'
    'ory_bytes":8192,"thread_execution_width":32},"name":"scan_s0_hillis_steele","rep'
    'lay":true,"simd_width":null,"source_sha256":"a8db0486becd559294f413bc7f32b71d625'
    '02bb874e1fd1da4d6d3cb102249fe","threadgroup_bytes":8192,"threads_per_group":1024'
    '},{"admission":{"admitted":true,"allowed_threads":1024,"declared_threadgroup_byt'
    'es":4096,"name":"scan_s1_blelloch","reason":null,"required_simd_width":null,"req'
    'uired_threads":512,"simd_width":32,"static_threadgroup_bytes":4096},"algorithm":'
    '"blelloch","barriers_per_block":22,"block":1024,"block_depth":19,"elements_per_t'
    'hread":2,"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory'
    '_bytes":4096,"thread_execution_width":32},"name":"scan_s1_blelloch","replay":tru'
    'e,"simd_width":null,"source_sha256":"d2087a1a14eb30c7b2b4da1e3646fdabe6ab5b948ca'
    '8c70b57a80c76b7840f85","threadgroup_bytes":4096,"threads_per_group":512},{"admis'
    'sion":{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":128,"'
    'name":"scan_s2_simd","reason":null,"required_simd_width":32,"required_threads":1'
    '024,"simd_width":32,"static_threadgroup_bytes":128},"algorithm":"simd_prefix","b'
    'arriers_per_block":2,"block":1024,"block_depth":62,"elements_per_thread":1,"limi'
    'ts":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":128,"t'
    'hread_execution_width":32},"name":"scan_s2_simd","replay":false,"simd_width":32,'
    '"source_sha256":"ba41e335c10dd9dc5910678ffe6fa64449baae4856228f2c2250ad87be729da'
    'b","threadgroup_bytes":128,"threads_per_group":1024}],"library_sha256":"39df7e35'
    'f24dcf5fd41ef7553563cc75eb5fab2ec9a501e192457d11ff5be1c5"},"speedup":{"best_rule'
    '":"the timed kernel with the highest median GB/s at this point; a tie goes to th'
    'e kernel that the registry lists first","copy":"stream_copy","copy_rule":"fracti'
    "on_of_copy = median GB/s of the side / median GB/s of stream_copy at the same po"
    "int; modeled_fraction_of_copy uses the modeled bytes of one scan instead of its "
    '8 N bytes","hypotheses":[{"base":"torch_cumsum","held_at":[],"kernel":"scan_s0_h'
    'illis_steele","missed_at":[0],"no_prediction_at":[],"no_verdict_at":[],"step":"S'
    '0"},{"base":"scan_s0_hillis_steele","held_at":[0],"kernel":"scan_s1_blelloch","m'
    'issed_at":[],"no_prediction_at":[],"no_verdict_at":[],"step":"S1"},{"base":"scan'
    '_s0_hillis_steele","held_at":[0],"kernel":"scan_s2_simd","missed_at":[],"no_pred'
    'iction_at":[],"no_verdict_at":[],"step":"S2"}],"hypothesis_rule":"a prediction h'
    "olds at a point when the verdict of the kernel over its base is one that the pla"
    "n predicted at that length; a kernel or base that did not run gives no verdict, "
    'and a length that the plan does not pin gives no prediction","min_faster_ratio":'
    '1.1,"points":[{"best":{"fraction_of_copy":0.8827084639498433,"kernel":"scan_s2_s'
    'imd","to_vendor":{"first":"scan_s2_simd","iqr_separated":false,"ratio":1.1998432'
    '601880877,"second":"torch_cumsum","verdict":"no_clear_difference"}},"copy":{"iqr'
    '_gb_per_second":[3.2939491226196385,4.229561625711935],"median_gb_per_second":3.'
    '723847945906017},"index":0,"length":65536,"rows":[{"base":"torch_cumsum","fracti'
    'on_of_copy":0.845384347491924,"held":false,"kernel":"scan_s0_hillis_steele","mod'
    'eled_fraction_of_copy":1.679210705857982,"modeled_gb_per_second":6.2531253377526'
    '39,"predicted":["slower"],"ran":true,"rate":{"iqr_gb_per_second":[2.836549750315'
    '1493,3.2066544342507646],"median_gb_per_second":3.1480827659089},"step":"S0","to'
    '_base":{"first":"scan_s0_hillis_steele","iqr_separated":false,"ratio":1.14910953'
    '39313806,"second":"torch_cumsum","verdict":"no_clear_difference"},"to_s0":null,"'
    'to_vendor":{"first":"scan_s0_hillis_steele","iqr_separated":false,"ratio":1.1491'
    '095339313806,"second":"torch_cumsum","verdict":"no_clear_difference"}},{"base":"'
    'scan_s0_hillis_steele","fraction_of_copy":0.6992991710251176,"held":true,"kernel'
    '":"scan_s1_blelloch","modeled_fraction_of_copy":1.3890376111963763,"modeled_gb_p'
    'er_second":5.172564855239827,"predicted":["no_clear_difference"],"ran":true,"rat'
    'e":{"iqr_gb_per_second":[1.3908356081165325,2.9620790960451977],"median_gb_per_s'
    'econd":2.604083781595665},"step":"S1","to_base":{"first":"scan_s1_blelloch","iqr'
    '_separated":false,"ratio":0.8271967337694267,"second":"scan_s0_hillis_steele","v'
    'erdict":"no_clear_difference"},"to_s0":{"first":"scan_s1_blelloch","iqr_separate'
    'd":false,"ratio":0.8271967337694267,"second":"scan_s0_hillis_steele","verdict":"'
    'no_clear_difference"},"to_vendor":{"first":"scan_s1_blelloch","iqr_separated":fa'
    'lse,"ratio":0.9505396532113463,"second":"torch_cumsum","verdict":"no_clear_diffe'
    'rence"}},{"base":"scan_s0_hillis_steele","fraction_of_copy":0.8827084639498433,"'
    'held":true,"kernel":"scan_s2_simd","modeled_fraction_of_copy":1.7533486481191225'
    ',"modeled_gb_per_second":6.529203761755486,"predicted":["no_clear_difference"],"'
    'ran":true,"rate":{"iqr_gb_per_second":[1.5480929644754287,3.8586053357865686],"m'
    'edian_gb_per_second":3.2870721003134795},"step":"S2","to_base":{"first":"scan_s2'
    '_simd","iqr_separated":false,"ratio":1.0441504702194357,"second":"scan_s0_hillis'
    '_steele","verdict":"no_clear_difference"},"to_s0":{"first":"scan_s2_simd","iqr_s'
    'eparated":false,"ratio":1.0441504702194357,"second":"scan_s0_hillis_steele","ver'
    'dict":"no_clear_difference"},"to_vendor":{"first":"scan_s2_simd","iqr_separated"'
    ':false,"ratio":1.1998432601880877,"second":"torch_cumsum","verdict":"no_clear_di'
    'fference"}}],"vendor":{"fraction_of_copy":0.7356864794252124,"iqr_gb_per_second"'
    ':[2.6390291291457566,3.344740031897927],"median_gb_per_second":2.739584585238406'
    '4}}],"ratio_rule":"ratio = median GB/s of the first side / median GB/s of the se'
    "cond, from the same point of the same run; faster when ratio >= min_faster_ratio"
    " and the first side's p75 sample time is below the second side's p25; slower whe"
    "n 1 / ratio >= min_faster_ratio and the second side's p75 is below the first sid"
    'e\'s p25; otherwise no_clear_difference","vendor":"vendor GPU scan: torch.cumsum '
    'on MPS, PyTorch 2.15.0"},"summary":{"benchmark_points":1,"cases_with_other_bits"'
    ':{"scan_s0_hillis_steele":[],"scan_s1_blelloch":[]},"correctness_cases":3,"hypot'
    'heses":{"scan_s0_hillis_steele":{"held":0,"missed":1,"no_prediction":0,"no_verdi'
    'ct":0},"scan_s1_blelloch":{"held":1,"missed":0,"no_prediction":0,"no_verdict":0}'
    ',"scan_s2_simd":{"held":1,"missed":0,"no_prediction":0,"no_verdict":0}},"kernels'
    '_checked":["scan_s0_hillis_steele","scan_s1_blelloch","scan_s2_simd"],"label":"h'
    'ost-specific Apple GPU (Metal) measurement","load_average_after":[8.62060546875,'
    '10.81396484375,12.99658203125],"load_average_before":[8.62060546875,10.813964843'
    '75,12.99658203125],"max_bound_ratio":0.07360778073319106,"nondeterministic_cases'
    '":{"scan_s0_hillis_steele":[],"scan_s1_blelloch":[],"scan_s2_simd":[],"torch_cum'
    'sum":[2]},"points":[{"best":"scan_s2_simd","best_fraction_of_copy":0.88270846394'
    '98433,"best_to_vendor":1.1998432601880877,"copy_gb_per_second":3.723847945906017'
    ',"length":65536,"vendor_gb_per_second":2.7395845852384064}],"refused_kernels":[]'
    ',"timed_samples":15,"vendor_max_relative_error":3.4200584663704415e-08,"vendor_m'
    'issed_exact_at":[],"warnings":0},"timer":"time.perf_counter_ns","torch_mps_switc'
    'hes":{"source":"os.environ, read before the runtime imports torch; null means un'
    'set, and a value that is not a short token is \'unknown\'","values":{"PYTORCH_MPS_'
    'FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}},"warnings":{"capture":"warning'
    "s.catch_warnings(record=True) with simplefilter('always'), from before the runti"
    "me imports torch to the end of the run; equal warnings are kept once with a coun"
    't","count":0,"distinct":0,"records":[],"torch_imported_before_capture":false}}'
)


def recorded():
    return json.loads(RECORDED_REPORT_JSON)


def test_the_recorded_report_verifies_as_a_measurement_on_any_host():
    report = recorded()

    summary = verify_report(report, config=config_from_mapping(RECORDED_CONFIG))

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["report_sha256"] == (
        "2517c7825d88ce1994d4944004dbd5df9f74c45f49ed1bda236b668851c1b3bb"
    )
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert [round(value, 2) for value in summary["load_average_before"]] == [
        8.62,
        10.81,
        13.0,
    ]
    assert report["environment"]["torch_version"] == "2.15.0"
    assert report["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    boundary = report["evidence_boundary"]
    assert boundary["cuda_kernel_executed"] is False
    assert boundary["warp_shuffle_executed"] is False
    assert boundary["hardware_counters_used"] is False


def test_the_report_seals_the_plan_that_was_declared_before_the_run():
    report = recorded()

    assert report["plan"] == {**plan_mapping(), "sha256": plan_sha256()}
    assert report["config"]["plan_sha256"] == plan_sha256()
    assert [entry["kernel"] for entry in report["plan"]["hypotheses"]] == list(KERNELS)


def test_the_fresh_process_captured_the_warnings_from_before_the_torch_import():
    captured = recorded()["warnings"]

    assert captured["torch_imported_before_capture"] is False
    assert (captured["count"], captured["records"]) == (0, [])


def test_every_recorded_pipeline_admitted_its_kernel_with_the_declared_bytes():
    report = recorded()

    assert report["refusals"] == [] and report["summary"]["refused_kernels"] == []
    for entry in report["shaders"]["kernels"]:
        kernel = KERNELS[entry["name"]]
        assert entry["source_sha256"] == kernel.source_sha256
        assert entry["limits"] == {
            "max_threads_per_threadgroup": 1024,
            "thread_execution_width": 32,
            "static_threadgroup_memory_bytes": kernel.threadgroup_bytes,
        }
        assert entry["admission"]["admitted"] is True


def test_the_apple_gpu_held_the_replay_bits_and_the_exact_integer_sums():
    report = recorded()

    for case in report["correctness"]["cases"]:
        values = scan_inputs(
            78, CORRECTNESS_STREAM, case["index"], case["length"], case["pattern"]
        )
        tail = float32_sentinels(guard_elements(case["length"]))
        outputs = {output["side"]: output for output in case["outputs"]}
        for name, replay in REPLAYS.items():
            expected = replay(values)
            if case["pattern"] == "small_integers":
                expected = exact_prefix(values)
            buffer = numpy.concatenate([expected, tail])
            assert outputs[name]["run_sha256"] == [array_sha256(buffer)] * 2, name
        for output in case["outputs"][:3]:
            assert output["deterministic"] is True, output["side"]
            assert output["bound_ratio"] < 0.1


def test_torch_cumsum_changed_its_bits_from_run_to_run_on_the_largest_case():
    summary = recorded()["summary"]

    assert summary["nondeterministic_cases"] == {
        "scan_s0_hillis_steele": [],
        "scan_s1_blelloch": [],
        "scan_s2_simd": [],
        VENDOR: [2],
    }
    assert summary["vendor_missed_exact_at"] == []
    speedup = recorded()["speedup"]
    assert speedup["vendor"] == "vendor GPU scan: torch.cumsum on MPS, PyTorch 2.15.0"
    assert speedup["copy"] == "stream_copy"


def test_verify_refuses_a_recorded_report_that_hides_the_error_of_run_1(
    tmp_path, capsys
):
    report = recorded()
    hide_the_first_run_error(report, 0, S0_HILLIS_STEELE)
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(canonical_json_bytes(reseal(report)) + b"\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(RECORDED_CONFIG), encoding="utf-8")

    argv = ["verify", "--report", str(tampered), "--config", str(config)]
    assert gpu_scan_cli.main(argv) == gpu_scan_cli.EXIT_REJECTED
    assert "records less error than its first run" in capsys.readouterr().err


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    runtime = scan_runtime()

    for entry in recorded()["shaders"]["kernels"]:
        assert runtime.limits[entry["name"]].to_mapping() == entry["limits"]
        assert runtime.admissions[entry["name"]].to_mapping() == entry["admission"]
