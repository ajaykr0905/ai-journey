"""One real fused softmax report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``metal_fused_cli run``
wrote in a fresh process on 2026-11-22 at 11:37 IST under the shared GPU
lock, with torch 2.15.0 on macOS 27.2 (arm64). The ``uptime`` command printed
load averages of 8.77, 8.20 and 8.21 before and after the run, from other
jobs. The run is short: two correctness cases, two small timing points,
five repeats and a few calls for each sample, so the cost of each
synchronize call sets most of its rates. Its ceiling is the recorded Day 74
gpu-ceilings report of 2026-11-17, 82.28 GB/s at 4 MiB. The timings describe
that moment and are here only to be verified. A host without torch or MPS
can still check a genuine Apple GPU (Metal) report. The reference uses
``exp64``, not the libm of the host, so another libm gives the same digests.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.metal_fused_cli import EXIT_OK, main
from ai_journey.metal_fused_config import config_from_mapping
from ai_journey.metal_fused_plan import plan_mapping, plan_sha256
from ai_journey.metal_fused_source import KERNELS, STEP_VARIANTS
from ai_journey.metal_fused_verify import verify_report
from test_kernel_metal_runtime import requires_metal
from test_metal_fused_ceiling import recorded_ceiling, write_recorded
from test_metal_fused_runtime import fused_runtime

ROOT = Path(__file__).resolve().parents[1]
RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "metal-fused-recorded",
    "seed": 79,
    "plan_sha256": "198870239a4a52adb106ed2396e7f9a99d1c2c52ba9770e9fea820e28b032990",
    "correctness": {
        "cases": [
            {"lengths": [1, 31, 33], "inputs": "uniform_80"},
            {"rows": 2, "cols": 64, "inputs": "offset_plus_1000"},
        ],
        "repeats": 2,
    },
    "benchmark": {
        "inputs": "normal",
        "shapes": [{"rows": 64, "cols": 128}, {"rows": 16, "max_length": 300}],
        "warmup": 1,
        "repeats": 5,
        "target_bytes_per_sample": 1048576,
        "max_runtime_seconds": 60,
    },
    "comparison": {"min_faster_ratio": 1.1},
}
# The canonical report of that run, byte for byte.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"compulsory":"8 n bytes f'
    "or every side: each of the n elements read once and written once; a ragged p"
    "oint credits torch.softmax with the n elements, not with the padding it also"
    ' processes","device":"mps","inputs":"normal","points":[{"checks":{"softmax_f'
    '1_threadgroup":{"differing_from_rounded_reference":5265,"max_abs_error":1.55'
    '6794844426168e-08,"max_bound_ratio":0.22786905895972429},"softmax_f2_simd":{'
    '"differing_from_rounded_reference":5307,"max_abs_error":1.556794844426168e-0'
    '8,"max_bound_ratio":0.12021195329831333},"softmax_f3_online":{"differing_fro'
    'm_rounded_reference":5266,"max_abs_error":1.2717068578749213e-08,"max_bound_'
    'ratio":0.0625105601439616},"softmax_u0_unfused":{"differing_from_rounded_ref'
    'erence":5265,"max_abs_error":1.556794844426168e-08,"max_bound_ratio":0.22786'
    '905895972429},"torch_softmax":{"differing_from_rounded_reference":5318,"max_'
    'abs_error":1.556794844426168e-08,"max_bound_ratio":0.030314339904192338}},"c'
    'ompulsory_bytes_per_call":65536,"index":0,"input_sha256":"4a947f9a63fa7592d5'
    '0c7bd0dd12544efeffc032c014f87b120a8bd80c8078fe","launch":{"active_threads":2'
    '048,"extent":[2048],"group_size":[32],"groups":[64],"launched_threads":2048,'
    '"padding_threads":0,"threads":[2048],"threads_per_group":32},"layout":{"buff'
    'er_elements":8192,"dense":true,"elements":8192,"gap":0,"group_size":32,"leng'
    'ths_sha256":"349e41450f522adda995a39826876934e7a5a48bfb28cca7a48af3516dd90eb'
    '8","longest":128,"rows":64,"shortest":128},"reference_sha256":"2f58bfa078933'
    '113527bb3a27ec89d37031d6faf0ebd8d6319cbab0a0016a6b7","shape":{"cols":128,"ro'
    'ws":64},"side_order":["softmax_u0_unfused","softmax_f1_threadgroup","softmax'
    '_f2_simd","softmax_f3_online","torch_softmax"],"sides":{"softmax_f1_threadgr'
    'oup":{"calls_per_sample":16,"compulsory_bytes_per_call":65536,"effective_gb_'
    'per_second":2.685213828425096,"implementation":"softmax_f1_threadgroup","iqr'
    '_effective_gb_per_second":[2.0893170610211707,3.04745076042699],"iqr_ns_per_'
    'call":9862.0,"median_ns_per_call":24406.25,"modeled_bytes_per_call":66048,"m'
    'odeled_gb_per_second":2.7061920614596673,"repeats":5,"samples_ns":[710084,39'
    '0500,344083,334792,501875],"timing":{"count":5,"iqr_ns":157792,"max_ns":7100'
    '84,"median_ns":390500,"min_ns":334792,"p25_ns":344083,"p75_ns":501875},"warm'
    'up":1},"softmax_f2_simd":{"calls_per_sample":16,"compulsory_bytes_per_call":'
    '65536,"effective_gb_per_second":3.5305589225589227,"implementation":"softmax'
    '_f2_simd","iqr_effective_gb_per_second":[2.8574823276778267,3.65198432738354'
    '4],"iqr_ns_per_call":4989.5625,"median_ns_per_call":18562.5,"modeled_bytes_p'
    'er_call":66048,"modeled_gb_per_second":3.558141414141414,"repeats":5,"sample'
    's_ns":[366958,287125,275125,297000,488417],"timing":{"count":5,"iqr_ns":7983'
    '3,"max_ns":488417,"median_ns":297000,"min_ns":275125,"p25_ns":287125,"p75_ns'
    '":366958},"warmup":1},"softmax_f3_online":{"calls_per_sample":16,"compulsory'
    '_bytes_per_call":65536,"effective_gb_per_second":3.4388899274228724,"impleme'
    'ntation":"softmax_f3_online","iqr_effective_gb_per_second":[3.09086514369933'
    '7,3.4812240006108714],"iqr_ns_per_call":2377.5625,"median_ns_per_call":19057'
    '.3125,"modeled_bytes_per_call":66048,"modeled_gb_per_second":3.4657562549808'
    '64,"repeats":5,"samples_ns":[508500,301209,282458,304917,339250],"timing":{"'
    'count":5,"iqr_ns":38041,"max_ns":508500,"median_ns":304917,"min_ns":282458,"'
    'p25_ns":301209,"p75_ns":339250},"warmup":1},"softmax_u0_unfused":{"calls_per'
    '_sample":16,"compulsory_bytes_per_call":65536,"effective_gb_per_second":1.72'
    '60510288065845,"implementation":"softmax_u0_unfused","iqr_effective_gb_per_s'
    'econd":[1.7005084127305898,1.8970167345092719],"iqr_ns_per_call":3992.1875,"'
    'median_ns_per_call":37968.75,"modeled_bytes_per_call":133888,"modeled_gb_per'
    '_second":3.5262683127572014,"repeats":5,"samples_ns":[607500,543792,616625,5'
    '52750,714459],"timing":{"count":5,"iqr_ns":63875,"max_ns":714459,"median_ns"'
    ':607500,"min_ns":543792,"p25_ns":552750,"p75_ns":616625},"warmup":1},"torch_'
    'softmax":{"calls_per_sample":16,"compulsory_bytes_per_call":65536,"effective'
    '_gb_per_second":3.4160243420924035,"implementation":"torch_softmax","iqr_eff'
    'ective_gb_per_second":[3.4035392944781293,3.417404850831394],"iqr_ns_per_cal'
    'l":78.125,"median_ns_per_call":19184.875,"modeled_bytes_per_call":null,"mode'
    'led_gb_per_second":null,"repeats":5,"samples_ns":[308084,273708,517625,30683'
    '4,306958],"timing":{"count":5,"iqr_ns":1250,"max_ns":517625,"median_ns":3069'
    '58,"min_ns":273708,"p25_ns":306834,"p75_ns":308084},"warmup":1}},"torch_inpu'
    't":{"cols":128,"padded":false,"rows":64}},{"checks":{"softmax_f1_threadgroup'
    '":{"differing_from_rounded_reference":1352,"max_abs_error":1.540520241927190'
    '2e-08,"max_bound_ratio":0.21071182140598751},"softmax_f2_simd":{"differing_f'
    'rom_rounded_reference":1346,"max_abs_error":1.8185893041477996e-08,"max_boun'
    'd_ratio":0.09452667092907471},"softmax_f3_online":{"differing_from_rounded_r'
    'eference":1320,"max_abs_error":1.8185893041477996e-08,"max_bound_ratio":0.05'
    '056777853831246},"softmax_u0_unfused":{"differing_from_rounded_reference":13'
    '52,"max_abs_error":1.5405202419271902e-08,"max_bound_ratio":0.21071182140598'
    '751},"torch_softmax":{"differing_from_rounded_reference":1392,"max_abs_error'
    '":1.8185893041477996e-08,"max_bound_ratio":0.026708221991821803}},"compulsor'
    'y_bytes_per_call":17096,"index":1,"input_sha256":"6ac3e0902ecbcfe0ac5e0340c1'
    '7852c29fccbec946338a1b78374b4cd98b4d9b","launch":{"active_threads":512,"exte'
    'nt":[512],"group_size":[32],"groups":[16],"launched_threads":512,"padding_th'
    'reads":0,"threads":[512],"threads_per_group":32},"layout":{"buffer_elements"'
    ':2137,"dense":false,"elements":2137,"gap":0,"group_size":32,"lengths_sha256"'
    ':"e6e8663a89239a10457540ee3d6601bd5ae4c647be7760b1c0b100250cde0599","longest'
    '":298,"rows":16,"shortest":3},"reference_sha256":"92bb4f91ed371117ef9b108c92'
    'd21b0d509511d187d449889633dc3c7a6b1ca4","shape":{"max_length":300,"rows":16}'
    ',"side_order":["softmax_u0_unfused","softmax_f1_threadgroup","softmax_f2_sim'
    'd","softmax_f3_online","torch_softmax"],"sides":{"softmax_f1_threadgroup":{"'
    'calls_per_sample":62,"compulsory_bytes_per_call":17096,"effective_gb_per_sec'
    'ond":2.0656799025578563,"implementation":"softmax_f1_threadgroup","iqr_effec'
    'tive_gb_per_second":[1.9497852379857439,2.130912292905148],"iqr_ns_per_call"'
    ':745.2903225806451,"median_ns_per_call":8276.209677419354,"modeled_bytes_per'
    '_call":17224,"modeled_gb_per_second":2.0811459196102313,"repeats":5,"samples'
    '_ns":[513125,479833,497417,683000,543625],"timing":{"count":5,"iqr_ns":46208'
    ',"max_ns":683000,"median_ns":513125,"min_ns":479833,"p25_ns":497417,"p75_ns"'
    ':543625},"warmup":1},"softmax_f2_simd":{"calls_per_sample":62,"compulsory_by'
    'tes_per_call":17096,"effective_gb_per_second":2.577127685440028,"implementat'
    'ion":"softmax_f2_simd","iqr_effective_gb_per_second":[2.487181051515137,2.59'
    '39479963291525],"iqr_ns_per_call":282.9193548387097,"median_ns_per_call":663'
    '3.741935483871,"modeled_bytes_per_call":17224,"modeled_gb_per_second":2.5964'
    '229792945157,"repeats":5,"samples_ns":[408625,403834,426166,411292,774375],"'
    'timing":{"count":5,"iqr_ns":17541,"max_ns":774375,"median_ns":411292,"min_ns'
    '":403834,"p25_ns":408625,"p75_ns":426166},"warmup":1},"softmax_f3_online":{"'
    'calls_per_sample":62,"compulsory_bytes_per_call":17096,"effective_gb_per_sec'
    'ond":2.3922199678164136,"implementation":"softmax_f3_online","iqr_effective_'
    'gb_per_second":[2.3452444696188586,2.5070294612953887],"iqr_ns_per_call":470'
    '.4193548387097,"median_ns_per_call":7146.5,"modeled_bytes_per_call":17224,"m'
    'odeled_gb_per_second":2.4101308332750295,"repeats":5,"samples_ns":[422792,41'
    '8417,451958,443083,644167],"timing":{"count":5,"iqr_ns":29166,"max_ns":64416'
    '7,"median_ns":443083,"min_ns":418417,"p25_ns":422792,"p75_ns":451958},"warmu'
    'p":1},"softmax_u0_unfused":{"calls_per_sample":62,"compulsory_bytes_per_call'
    '":17096,"effective_gb_per_second":0.88832048702281,"implementation":"softmax'
    '_u0_unfused","iqr_effective_gb_per_second":[0.8669775385393057,0.89826440677'
    '9661],"iqr_ns_per_call":686.8225806451613,"median_ns_per_call":19245.3064516'
    '129,"modeled_bytes_per_call":34896,"modeled_gb_per_second":1.813221321662843'
    '7,"repeats":5,"samples_ns":[1666208,1193209,1222583,1180000,1167417],"timing'
    '":{"count":5,"iqr_ns":42583,"max_ns":1666208,"median_ns":1193209,"min_ns":11'
    '67417,"p25_ns":1180000,"p75_ns":1222583},"warmup":1},"torch_softmax":{"calls'
    '_per_sample":62,"compulsory_bytes_per_call":17096,"effective_gb_per_second":'
    '1.7565858379625332,"implementation":"torch_softmax","iqr_effective_gb_per_se'
    'cond":[1.496403527719307,1.790081486172683],"iqr_ns_per_call":1874.322580645'
    '1612,"median_ns_per_call":9732.516129032258,"modeled_bytes_per_call":null,"m'
    'odeled_gb_per_second":null,"repeats":5,"samples_ns":[590833,708333,1223958,6'
    '03416,592125],"timing":{"count":5,"iqr_ns":116208,"max_ns":1223958,"median_n'
    's":603416,"min_ns":590833,"p25_ns":592125,"p75_ns":708333},"warmup":1}},"tor'
    'ch_input":{"cols":298,"padded":true,"rows":16}}],"synchronization":"torch.mp'
    "s.synchronize() before the first and before the second clock reading of ever"
    "y sample; untimed warmups of each side first; repeat r starts with side r mo"
    'd S, so the order shifts by one side every repeat","torch_equivalent":"torch'
    ".softmax(x, dim=-1, out=y) on the MPS device: on the rows x cols matrix of a"
    " dense point, and on the rows x longest matrix of a ragged point, with the m"
    'ost negative finite float32 in each slot past a row","traffic_model":"device'
    " bytes of one call, with no cache, for n elements in r rows: U0 reads the ro"
    "w in each of its three kernels and writes it once, 16 n bytes, and moves 44 "
    "r bytes of row data (8 r of starts and lengths in each kernel, the max writt"
    "en once and read twice, the sum written once and read once); a fused step re"
    "ads the row once and writes it once, 8 n bytes, plus 8 r of starts and lengt"
    'hs; torch.softmax has no model, because PyTorch does not name its kernel","v'
    'endor_bound":"|y - y_ref| <= 2**-16 * y_ref + 2**-125 for every torch.softma'
    "x output: a sanity bound only, because PyTorch names no kernel and no derive"
    'd bound applies"},"ceiling":{"buffer_bytes":4194304,"config_sha256":"9e1db26'
    '39799674c80116c53f0ab2a97583f5400def85d536483da68aff6d173","definition":"the'
    " bandwidth ceiling of a verified Day 74 gpu-ceilings report: the better medi"
    "an GB/s of stream_copy and stream_read_reduce at its largest buffer, on the "
    'host that made it; not a vendor peak and not a hardware limit","environment"'
    ':{"macos_version":"27.2","numpy_version":"1.26.4","platform_machine":"arm64"'
    ',"platform_system":"Darwin","python_implementation":"CPython","python_versio'
    'n":"3.11.14","torch_version":"2.15.0"},"gb_per_second":82.28275618954719,"ke'
    'rnel":"stream_read_reduce","label":"host-specific Apple GPU (Metal) measurem'
    'ent","load_average_after":[9.4765625,9.40478515625,8.75341796875],"load_aver'
    'age_before":[9.4765625,9.40478515625,8.75341796875],"measured":true,"report_'
    'sha256":"6b784c28e36b2890eadb65c742858b29835b90f8e259b4befba92a499c919cb6","'
    'source":"gpu_ceilings"},"config":{"benchmark":{"inputs":"normal","max_runtim'
    'e_seconds":60,"repeats":5,"shapes":[{"cols":128,"rows":64},{"max_length":300'
    ',"rows":16}],"target_bytes_per_sample":1048576,"warmup":1},"comparison":{"mi'
    'n_faster_ratio":1.1},"correctness":{"cases":[{"inputs":"uniform_80","lengths'
    '":[1,31,33]},{"cols":64,"inputs":"offset_plus_1000","rows":2}],"repeats":2},'
    '"name":"metal-fused-recorded","plan_sha256":"198870239a4a52adb106ed2396e7f9a'
    '99d1c2c52ba9770e9fea820e28b032990","schema_version":1,"seed":79},"config_sha'
    '256":"f5356d711d410732341692d900e6841b8615c83ceebadf9c2b0036897ba7ca60","cor'
    'rectness":{"bound":"|y - y_ref| <= (rel + ref) / (1 - ref) * y_ref + 2**-125'
    " for each element, with rel = alpha (1 + u) / (1 - rho) - 1, alpha = exp(d u"
    ") (1 + EXP_ERROR), rho = (sum_j y_ref_j (beta_j - 1)) / (1 - ref) + (n + 2 G"
    ") 2**-126, beta = exp(d u) (1 + EXP_ERROR)**N (1 + gamma_A), d = max - x, u "
    "= 2**-24, EXP_ERROR = 4 * 2**-23, gamma_A = A u / (1 - A u) and ref = gamma_"
    '(n+8) for the unit 2**-53","cases":[{"case":{"inputs":"uniform_80","lengths"'
    ':[1,31,33]},"index":0,"input_sha256":"4994f5444fc0d8d4fa7d53978a9a5a838a1e51'
    '3897a45a01a80389020b34e12d","launch":{"active_threads":96,"extent":[96],"gro'
    'up_size":[32],"groups":[3],"launched_threads":96,"padding_threads":0,"thread'
    's":[96],"threads_per_group":32},"layout":{"buffer_elements":161,"dense":fals'
    'e,"elements":65,"gap":32,"group_size":32,"lengths_sha256":"72e8f7ca85b07b7a5'
    '31470b61a3cc9ae6d349f144413996ddaded1b5c430f6ad","longest":33,"rows":3,"shor'
    'test":1},"naive_float32_finite":true,"outputs":[{"bound_sha256":"f54c032321f'
    'd1c3cd0426543828c1861dde8196f854dfc0ea8a641711265e3f4","buffer_sha256":"23d2'
    '685078ed024181edd4b9baace5f4f236dd3d6d24f653fe13adb14e29feaf","differing_fro'
    'm_rounded_reference":30,"identical_repeats":true,"kernel":"softmax_u0_unfuse'
    'd","max_abs_error":3.6409037562634694e-08,"max_bound_ratio":0.63430912082611'
    '53,"max_row_sum_error":5.1672818912607e-08,"max_row_sum_ratio":0.03704329415'
    '53379,"row_sum_bound_sha256":"047becdedd9e9420835d00f2c2339f6a47c41c32785e15'
    '5db60066b78b8080e6"},{"bound_sha256":"f54c032321fd1c3cd0426543828c1861dde819'
    '6f854dfc0ea8a641711265e3f4","buffer_sha256":"23d2685078ed024181edd4b9baace5f'
    '4f236dd3d6d24f653fe13adb14e29feaf","differing_from_rounded_reference":30,"id'
    'entical_repeats":true,"kernel":"softmax_f1_threadgroup","max_abs_error":3.64'
    '09037562634694e-08,"max_bound_ratio":0.6343091208261153,"max_row_sum_error":'
    '5.1672818912607e-08,"max_row_sum_ratio":0.0370432941553379,"row_sum_bound_sh'
    'a256":"047becdedd9e9420835d00f2c2339f6a47c41c32785e155db60066b78b8080e6"},{"'
    'bound_sha256":"774298709a748830cc29b6e648eee30923f2c7857b5bdbd39c66cd6def1d6'
    '81f","buffer_sha256":"23d2685078ed024181edd4b9baace5f4f236dd3d6d24f653fe13ad'
    'b14e29feaf","differing_from_rounded_reference":30,"identical_repeats":true,"'
    'kernel":"softmax_f2_simd","max_abs_error":3.6409037562634694e-08,"max_bound_'
    'ratio":0.5083186886895924,"max_row_sum_error":5.1672818912607e-08,"max_row_s'
    'um_ratio":0.01754796653361059,"row_sum_bound_sha256":"fbd3e65ccf22a2e58e2540'
    'b449f31484c4124432d1674ce1d4665fd5974177c5"},{"bound_sha256":"96ff3a91f0d720'
    '13fc6ea4a059ec1f40499eb2ffa8861ff343e5ad4d1e92a669","buffer_sha256":"23d2685'
    '078ed024181edd4b9baace5f4f236dd3d6d24f653fe13adb14e29feaf","differing_from_r'
    'ounded_reference":30,"identical_repeats":true,"kernel":"softmax_f3_online","'
    'max_abs_error":3.6409037562634694e-08,"max_bound_ratio":0.4240841587300845,"'
    'max_row_sum_error":5.1672818912607e-08,"max_row_sum_ratio":0.011497162344799'
    '925,"row_sum_bound_sha256":"34299eb3a4820e512dbf935835f0a35133a287d67ca9f4d0'
    '9459ab17d5ce053b"}],"reference_sha256":"6396bcc2433fc42929e2e80c0100d1310051'
    '3d085964e2aed11b1884555271ef","repeats":2,"same_bits":[{"as":"softmax_u0_unf'
    'used","same":true,"variant":"softmax_f1_threadgroup"}]},{"case":{"cols":64,"'
    'inputs":"offset_plus_1000","rows":2},"index":1,"input_sha256":"11e3791e9aaae'
    '4aa253c8d8a3d0b79a9812369ca279a07c9539fd287cd12e015","launch":{"active_threa'
    'ds":64,"extent":[64],"group_size":[32],"groups":[2],"launched_threads":64,"p'
    'adding_threads":0,"threads":[64],"threads_per_group":32},"layout":{"buffer_e'
    'lements":192,"dense":true,"elements":128,"gap":32,"group_size":32,"lengths_s'
    'ha256":"d8a162db4b52873aa8bee8c7bda17ad1e10b8233d0640427722f394ec172caf3","l'
    'ongest":64,"rows":2,"shortest":64},"naive_float32_finite":false,"outputs":[{'
    '"bound_sha256":"7a2460a5717f7291521a77edfe1b63ec302e9015a89bd7b6e66aaa7ba31b'
    '0c8e","buffer_sha256":"6a19f287433a0b60746677ba1cd6c609f5ba98b3ccd4463aba873'
    '776b30da656","differing_from_rounded_reference":55,"identical_repeats":true,'
    '"kernel":"softmax_u0_unfused","max_abs_error":5.059845709087796e-09,"max_bou'
    'nd_ratio":0.08736721050653032,"max_row_sum_error":2.7241185307502747e-08,"ma'
    'x_row_sum_ratio":0.017561609524159815,"row_sum_bound_sha256":"a4b07252c01495'
    '1851e776dfd55f2de2c12fd233c2368d0893b92f375732a331"},{"bound_sha256":"7a2460'
    'a5717f7291521a77edfe1b63ec302e9015a89bd7b6e66aaa7ba31b0c8e","buffer_sha256":'
    '"6a19f287433a0b60746677ba1cd6c609f5ba98b3ccd4463aba873776b30da656","differin'
    'g_from_rounded_reference":55,"identical_repeats":true,"kernel":"softmax_f1_t'
    'hreadgroup","max_abs_error":5.059845709087796e-09,"max_bound_ratio":0.087367'
    '21050653032,"max_row_sum_error":2.7241185307502747e-08,"max_row_sum_ratio":0'
    '.017561609524159815,"row_sum_bound_sha256":"a4b07252c014951851e776dfd55f2de2'
    'c12fd233c2368d0893b92f375732a331"},{"bound_sha256":"2264fface3403967049d847b'
    '72083af62541324ebaa42ec3ebb3019dbc3ed763","buffer_sha256":"6a19f287433a0b607'
    '46677ba1cd6c609f5ba98b3ccd4463aba873776b30da656","differing_from_rounded_ref'
    'erence":55,"identical_repeats":true,"kernel":"softmax_f2_simd","max_abs_erro'
    'r":5.059845709087796e-09,"max_bound_ratio":0.04467182532349286,"max_row_sum_'
    'error":2.7241185307502747e-08,"max_row_sum_ratio":0.008784901583625923,"row_'
    'sum_bound_sha256":"a20188b0762c78072330c9ceac0aa1a574b19fe3e72891caf412d18fe'
    'dcd490b"},{"bound_sha256":"512bfbf4667f3d9d9f063719cec1a41c73a789cf10f5fe645'
    '43e6a2bffa98ff2","buffer_sha256":"2ebb017dd396f26cd70e545ad9ce58c3e1dfd6cb7d'
    'a32909a10d2a6e7fb51d62","differing_from_rounded_reference":87,"identical_rep'
    'eats":true,"kernel":"softmax_f3_online","max_abs_error":3.2335373462144545e-'
    '08,"max_bound_ratio":0.04329807811253955,"max_row_sum_error":1.2107193470001'
    '22e-07,"max_row_sum_ratio":0.02582797996179625,"row_sum_bound_sha256":"481fa'
    '42ff346952a076c01b0d0f526bc361073fe98ff11926992d3430e362764"}],"reference_sh'
    'a256":"9316936ac5d54a06704fc0cc53804f785621b1c0b4117ba508eccc217a889a56","re'
    'peats":2,"same_bits":[{"as":"softmax_u0_unfused","same":true,"variant":"soft'
    'max_f1_threadgroup"}]}],"distributions":{"normal":"a standard normal, drawn '
    'as float32","offset_minus_1000":"-1000 plus a standard normal, rounded to fl'
    'oat32","offset_plus_1000":"1000 plus a standard normal, rounded to float32",'
    '"uniform_80":"uniform from -80 to 80, drawn in float64 and rounded to float3'
    '2"},"gap":32,"gap_input":1.2676506002282294e+30,"order_terms":{"softmax_f1_t'
    'hreadgroup":"A = (k - 1) + log2(G) and N = 1: the additions of U0 in its ord'
    'er","softmax_f2_simd":"A = (k - 1) + 31 + (G / 32 - 1) and N = 1: k slots in'
    ' order, simd_sum over 32 lanes, then simd_sum over the G / 32 partials","sof'
    'tmax_f3_online":"A = (k - 1) + 1 + 31 + 1 + (G / 32 - 1) and N = k + 2: k on'
    "line updates, a rescale and simd_sum in each SIMD group, a rescale and simd_"
    'sum of the partials; each update or rescale rounds once","softmax_u0_unfused'
    '":"A = (k - 1) + log2(G) and N = 1: k slots added in order, then a tree"},"r'
    'eference":"float64 softmax of the float32 inputs, row by row: exp(x - max) d'
    "ivided by the sum of exp(x - max) over the row, with numpy.add.reduceat and "
    "exp64, an exp of IEEE additions, products and ldexp only, so each host compu"
    'tes the same bits","row_sum_bound":"|sum(y) - 1| <= the sum of the element b'
    "ounds of the row + ref + 2 gamma_n for the unit 2**-53, where sum adds the o"
    'utputs of the row in float64 in order","rules":"every gap slot keeps its sen'
    "tinel in every repeat; every output is finite; every element is within its b"
    "ound of the reference; every row sum is within its bound of 1; identical_rep"
    'eats, same_bits and differing_from_rounded_reference are measured","sentinel'
    '_float32_bits":2143289344},"environment":{"macos_version":"27.2","numpy_vers'
    'ion":"1.26.4","platform_machine":"arm64","platform_system":"Darwin","python_'
    'implementation":"CPython","python_version":"3.11.14","torch_version":"2.15.0'
    '"},"evidence_boundary":{"apple_gpu_metal_this_host":true,"cuda_kernel_execut'
    'ed":false,"distributed_claim":false,"hardware_counters_used":false,"host_spe'
    'cific":true,"nvidia_gpu_used":false,"occupancy_measured":false,"portable_per'
    'formance_claim":false,"production_claim":false,"statement":"These are wall-c'
    "lock timings on the Apple GPU of the host that made this report, at the reco"
    "rded load averages. The softmax kernels came from torch.mps.compile_shader, "
    "and torch.softmax ran on the MPS device. PyTorch does not report which kerne"
    "l torch.softmax ran, so the vendor number names the call, not a kernel. The "
    "bandwidth ceiling comes from a separate gpu-ceilings run with its own load a"
    "verages. No hardware counter was read, so no cache hit, stall or occupancy w"
    "as observed, and each byte count is a model. No CUDA kernel ran, no NVIDIA G"
    "PU was used, and no Triton kernel ran on a GPU. Each ratio compares two side"
    "s of one run on this host. These timings do not measure or predict CUDA, NVI"
    'DIA, Triton, distributed or production performance.","triton_gpu_kernel_exec'
    'uted":false,"vendor_kernel_known":false,"vendor_peak_claimed":false},"exp_pr'
    'obe":{"arguments":"precise::exp at 2**18 float32 arguments: 262136 evenly sp'
    "aced from -104 to 0, then the extremes -3.4028234663852886e+38, -1e+30, -100"
    '0.0, -200.0, -150.0, -105.0, -87.5, -0.0","arguments_sha256":"ef74306503b0bf'
    '88378fecc220238ffe4510b9ce6992098cb47e1e39804aab8d","count":262144,"exp_erro'
    'r":4.76837158203125e-07,"exp_ulps":4,"holds":true,"max_relative_error":1.614'
    '39084446152e-07,"max_ulps":1.4061690587550402,"normal_results":220136,"resul'
    'ts_sha256":"918b6099a06c5fc92292905d92d8968e8f6b44a6cc1e883b02864e0cc4467498'
    '","rule":"each result whose float64 exp is at least 2**-126 has a relative e'
    "rror of at most EXP_ERROR = 4 * 2**-23; each smaller result is within 2**-12"
    '6 of it","tiny_max_abs_error":1.1751589259924035e-38,"tiny_nonzero":0,"tiny_'
    'results":42008},"host_load":{"after":[8.7666015625,8.19970703125,8.213867187'
    '5],"before":[8.7666015625,8.19970703125,8.2138671875],"source":"os.getloadav'
    "g(): the 1, 5 and 15 minute load averages that uptime prints, read before th"
    'e checks and after the last sample"},"kind":"metal_fused_softmax_correctness'
    '_and_bandwidth","label":"host-specific Apple GPU (Metal) measurement","measu'
    'red":true,"measurement_scope":"apple_gpu_metal_this_host","percentile_method'
    '":"nearest rank: sorted samples[ceil(p * n / 100) - 1] for p in 25, 50 and 7'
    '5; median is p50 and IQR is p75 - p25; no interpolation","plan":{"baseline":'
    '"softmax_u0_unfused","sha256":"198870239a4a52adb106ed2396e7f9a99d1c2c52ba977'
    '0e9fea820e28b032990","steps":[{"base":"torch_softmax","change":"Compute the '
    "softmax in three kernels: the max of each row, the sum of exp(x - max) and t"
    "he normalize pass. The max and the sum go through two buffers in device memo"
    'ry.","hypothesis":"U0 reads each row three times and writes it once, so the '
    "model gives it about twice the bytes of one read and one write. It also laun"
    "ches three kernels for each call. U0 is slower than torch.softmax at every p"
    'oint.","predicted":["slower"],"same_bits_as":null,"source_sha256":"c38c9faf7'
    '3e14e5695bd758ad3f4d3547b17d571877f091d3264d729bb71997c","step":"U0","varian'
    't":"softmax_u0_unfused"},{"base":"softmax_u0_unfused","change":"Fuse the thr'
    "ee passes into one kernel. Each thread keeps its 16 slots of the row in regi"
    'sters, and two trees in threadgroup memory give the max and the sum.","hypot'
    'hesis":"F1 reads each element once and writes it once, so the model gives it'
    " about half the bytes of U0, with one launch in place of three. F1 is faster"
    " than U0 by 1.1 or more at every point. F1 adds the same values in the same "
    'order as U0, so its outputs can keep the U0 bits.","predicted":["faster"],"s'
    'ame_bits_as":"softmax_u0_unfused","source_sha256":"4a41f2e138c5a3e499293b771'
    'f2e105f4c13920ba291b466b75cab657ebe665a","step":"F1","variant":"softmax_f1_t'
    'hreadgroup"},{"base":"softmax_f1_threadgroup","change":"Replace each threadg'
    "roup tree with simd_max or simd_sum in each SIMD group and one threadgroup s"
    'tep.","hypothesis":"A tree over G threads needs log2(G) + 1 barriers for eac'
    "h value, and F2 needs one. F2 moves the same modeled bytes as F1, and the lo"
    "ads of the row probably set most of the time. F2 is faster than F1 or shows "
    'no clear difference.","predicted":["faster","no_clear_difference"],"same_bit'
    's_as":null,"source_sha256":"a8380a8b2ea58e8a84ed8c8a1025c7cb5ea7f4f1111d1abd'
    'e374135e0c0af4d8","step":"F2","variant":"softmax_f2_simd"},{"base":"softmax_'
    'f2_simd","change":"Keep a running max and a running sum in one pass over the'
    " slots, rescale the sum when the max grows, and merge the (max, sum) pairs i"
    'n one threadgroup step.","hypothesis":"F3 moves the same modeled bytes as F2'
    " and needs one barrier in place of two. It calls precise::exp twice for each"
    " element, and F2 calls it once. F3 is not faster than F2: it shows no clear "
    'difference or it is slower.","predicted":["no_clear_difference","slower"],"s'
    'ame_bits_as":null,"source_sha256":"169845113557a1bf67e8fff9320b28ca920547ecc'
    '54f175b435039b7fb7124c5","step":"F3","variant":"softmax_f3_online"}],"vendor'
    '":"torch_softmax"},"planned_work":{"benchmark_points":2,"calls":2340,"compul'
    'sory_bytes":63255840,"correctness_cases":2,"correctness_elements":193,"corre'
    'ctness_launches":25,"exp_probe_arguments":262144,"kernel_launches":3276,"sid'
    'es_per_point":5,"timed_samples":50},"refusals":[],"report_sha256":"9c40082b1'
    'd9e900606c9cd1cda67246734d86503e01a13f296bbf695293e1e6c","run":{"elapsed_ns"'
    ':139939208,"max_runtime_ns":60000000000,"timed_ns":28150419},"schema_version'
    '":1,"shaders":{"admissions":[{"admitted":true,"kernels":[{"allowed_threads":'
    '1024,"declared_threadgroup_bytes":4096,"name":"softmax_u0_row_max","static_t'
    'hreadgroup_bytes":4096,"thread_execution_width":32},{"allowed_threads":1024,'
    '"declared_threadgroup_bytes":4096,"name":"softmax_u0_exp_sum","static_thread'
    'group_bytes":4096,"thread_execution_width":32},{"allowed_threads":1024,"decl'
    'ared_threadgroup_bytes":0,"name":"softmax_u0_normalize","static_threadgroup_'
    'bytes":0,"thread_execution_width":32}],"reason":null,"required_threads":32,"'
    'variant":"softmax_u0_unfused"},{"admitted":true,"kernels":[{"allowed_threads'
    '":1024,"declared_threadgroup_bytes":8192,"name":"softmax_f1_threadgroup","st'
    'atic_threadgroup_bytes":8192,"thread_execution_width":32}],"reason":null,"re'
    'quired_threads":32,"variant":"softmax_f1_threadgroup"},{"admitted":true,"ker'
    'nels":[{"allowed_threads":1024,"declared_threadgroup_bytes":256,"name":"soft'
    'max_f2_simd","static_threadgroup_bytes":256,"thread_execution_width":32}],"r'
    'eason":null,"required_threads":32,"variant":"softmax_f2_simd"},{"admitted":t'
    'rue,"kernels":[{"allowed_threads":1024,"declared_threadgroup_bytes":256,"nam'
    'e":"softmax_f3_online","static_threadgroup_bytes":256,"thread_execution_widt'
    'h":32}],"reason":null,"required_threads":32,"variant":"softmax_f3_online"}],'
    '"kernels":[{"buffers":["row_max","x","starts","lengths"],"limits":{"max_thre'
    'ads_per_threadgroup":1024,"static_threadgroup_memory_bytes":4096,"thread_exe'
    'cution_width":32},"name":"softmax_u0_row_max","simd_functions":false,"source'
    '_sha256":"d7a01ba402c8b179dd88ae3b9c73af5407df42ea574228538b6e9c414c13c47b",'
    '"threadgroup_bytes":4096},{"buffers":["row_sum","x","row_max","starts","leng'
    'ths"],"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memor'
    'y_bytes":4096,"thread_execution_width":32},"name":"softmax_u0_exp_sum","simd'
    '_functions":false,"source_sha256":"286b2d0374b70cd5077c5e0a52e74311c57d406dc'
    'f8a295319a0857a48a0bc64","threadgroup_bytes":4096},{"buffers":["y","x","row_'
    'max","row_sum","starts","lengths"],"limits":{"max_threads_per_threadgroup":1'
    '024,"static_threadgroup_memory_bytes":0,"thread_execution_width":32},"name":'
    '"softmax_u0_normalize","simd_functions":false,"source_sha256":"e6fd4b2679dac'
    'd8ff2671a26f6074baab5a1e4d0259758b0197f4f7930e60b8c","threadgroup_bytes":0},'
    '{"buffers":["y","x","starts","lengths"],"limits":{"max_threads_per_threadgro'
    'up":1024,"static_threadgroup_memory_bytes":8192,"thread_execution_width":32}'
    ',"name":"softmax_f1_threadgroup","simd_functions":false,"source_sha256":"4a4'
    '1f2e138c5a3e499293b771f2e105f4c13920ba291b466b75cab657ebe665a","threadgroup_'
    'bytes":8192},{"buffers":["y","x","starts","lengths"],"limits":{"max_threads_'
    'per_threadgroup":1024,"static_threadgroup_memory_bytes":256,"thread_executio'
    'n_width":32},"name":"softmax_f2_simd","simd_functions":true,"source_sha256":'
    '"a8380a8b2ea58e8a84ed8c8a1025c7cb5ea7f4f1111d1abde374135e0c0af4d8","threadgr'
    'oup_bytes":256},{"buffers":["y","x","starts","lengths"],"limits":{"max_threa'
    'ds_per_threadgroup":1024,"static_threadgroup_memory_bytes":256,"thread_execu'
    'tion_width":32},"name":"softmax_f3_online","simd_functions":true,"source_sha'
    '256":"169845113557a1bf67e8fff9320b28ca920547ecc54f175b435039b7fb7124c5","thr'
    'eadgroup_bytes":256},{"buffers":["y","x"],"limits":{"max_threads_per_threadg'
    'roup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_width":32},'
    '"name":"softmax_exp_probe","simd_functions":false,"source_sha256":"1d62fece5'
    'aebee7e36cc9803962fdc9bd8c2aece802b5fa4c04f1b18d85170f6","threadgroup_bytes"'
    ':0}],"library_sha256":"c8c1b7ff06c16966a8a0b9e646868e62055faa6ca247e4c086459'
    '5a7e9cd7521","steps":[{"kernels":["softmax_u0_row_max","softmax_u0_exp_sum",'
    '"softmax_u0_normalize"],"launches_per_call":3,"name":"softmax_u0_unfused","r'
    'eduction":"threadgroup_tree","row_reads":3,"source_sha256":"c38c9faf73e14e56'
    '95bd758ad3f4d3547b17d571877f091d3264d729bb71997c","step":"U0","summary":"thr'
    'ee kernels; the max and the sum go through device memory"},{"kernels":["soft'
    'max_f1_threadgroup"],"launches_per_call":1,"name":"softmax_f1_threadgroup","'
    'reduction":"threadgroup_tree","row_reads":1,"source_sha256":"4a41f2e138c5a3e'
    '499293b771f2e105f4c13920ba291b466b75cab657ebe665a","step":"F1","summary":"on'
    'e kernel; the row stays in registers; two threadgroup trees"},{"kernels":["s'
    'oftmax_f2_simd"],"launches_per_call":1,"name":"softmax_f2_simd","reduction":'
    '"simd_then_one_threadgroup_step","row_reads":1,"source_sha256":"a8380a8b2ea5'
    '8e8a84ed8c8a1025c7cb5ea7f4f1111d1abde374135e0c0af4d8","step":"F2","summary":'
    '"one kernel; simd_max and simd_sum, one threadgroup step for each"},{"kernel'
    's":["softmax_f3_online"],"launches_per_call":1,"name":"softmax_f3_online","r'
    'eduction":"online_simd_then_one_threadgroup_step","row_reads":1,"source_sha2'
    '56":"169845113557a1bf67e8fff9320b28ca920547ecc54f175b435039b7fb7124c5","step'
    '":"F3","summary":"one kernel; one online pass for max and sum, one merge ste'
    'p"}]},"speedup":{"baseline":"softmax_u0_unfused","best_rule":"the step with '
    "the highest effective GB/s at this point; a tie goes to the earlier step of "
    'the plan","ceiling_gb_per_second":82.28275618954719,"fraction_rule":"effecti'
    "ve GB/s of a side / the bandwidth ceiling GB/s, and for a step also its mode"
    'led GB/s / the ceiling; the ceiling comes from a separate run","hypotheses":'
    '[{"held_at":[0,1],"missed_at":[],"no_verdict_at":[],"predicted":["slower"],"'
    'step":"U0","variant":"softmax_u0_unfused"},{"held_at":[0,1],"missed_at":[],"'
    'no_verdict_at":[],"predicted":["faster"],"step":"F1","variant":"softmax_f1_t'
    'hreadgroup"},{"held_at":[0,1],"missed_at":[],"no_verdict_at":[],"predicted":'
    '["faster","no_clear_difference"],"step":"F2","variant":"softmax_f2_simd"},{"'
    'held_at":[0,1],"missed_at":[],"no_verdict_at":[],"predicted":["no_clear_diff'
    'erence","slower"],"step":"F3","variant":"softmax_f3_online"}],"hypothesis_ru'
    'le":"a prediction holds at a point when the verdict of the step over its bas'
    "e is one of the verdicts that the step predicted; a step or base that did no"
    't run gives no verdict","min_faster_ratio":1.1,"points":[{"best":{"to_torch"'
    ':{"first":"softmax_f2_simd","iqr_separated":false,"ratio":1.0335286195286195'
    ',"second":"torch_softmax","verdict":"no_clear_difference"},"to_u0":{"first":'
    '"softmax_f2_simd","iqr_separated":true,"ratio":2.0454545454545454,"second":"'
    'softmax_u0_unfused","verdict":"faster"},"variant":"softmax_f2_simd"},"index"'
    ':0,"rows":[{"base":"torch_softmax","held":true,"predicted":["slower"],"ran":'
    'true,"rate":{"effective_fraction_of_ceiling":0.02097706869262425,"effective_'
    'gb_per_second":1.7260510288065845,"iqr_effective_gb_per_second":[1.700508412'
    '7305898,1.8970167345092719],"modeled_fraction_of_ceiling":0.0428554958056346'
    '9,"modeled_gb_per_second":3.5262683127572014},"step":"U0","to_base":{"first"'
    ':"softmax_u0_unfused","iqr_separated":true,"ratio":0.505280658436214,"second'
    '":"torch_softmax","verdict":"slower"},"to_torch":{"first":"softmax_u0_unfuse'
    'd","iqr_separated":true,"ratio":0.505280658436214,"second":"torch_softmax","'
    'verdict":"slower"},"to_u0":null,"variant":"softmax_u0_unfused"},{"base":"sof'
    'tmax_u0_unfused","held":true,"predicted":["faster"],"ran":true,"rate":{"effe'
    'ctive_fraction_of_ceiling":0.03263398010440264,"effective_gb_per_second":2.6'
    '85213828425096,"iqr_effective_gb_per_second":[2.0893170610211707,3.047450760'
    '42699],"modeled_fraction_of_ceiling":0.032888933073968285,"modeled_gb_per_se'
    'cond":2.7061920614596673},"step":"F1","to_base":{"first":"softmax_f1_threadg'
    'roup","iqr_separated":true,"ratio":1.555697823303457,"second":"softmax_u0_un'
    'fused","verdict":"faster"},"to_torch":{"first":"softmax_f1_threadgroup","iqr'
    '_separated":true,"ratio":0.7860640204865557,"second":"torch_softmax","verdic'
    't":"slower"},"to_u0":{"first":"softmax_f1_threadgroup","iqr_separated":true,'
    '"ratio":1.555697823303457,"second":"softmax_u0_unfused","verdict":"faster"},'
    '"variant":"softmax_f1_threadgroup"},{"base":"softmax_f1_threadgroup","held":'
    'true,"predicted":["faster","no_clear_difference"],"ran":true,"rate":{"effect'
    'ive_fraction_of_ceiling":0.04290764050764051,"effective_gb_per_second":3.530'
    '5589225589227,"iqr_effective_gb_per_second":[2.8574823276778267,3.6519843273'
    '83544],"modeled_fraction_of_ceiling":0.04324285644910645,"modeled_gb_per_sec'
    'ond":3.558141414141414},"step":"F2","to_base":{"first":"softmax_f2_simd","iq'
    'r_separated":false,"ratio":1.3148148148148149,"second":"softmax_f1_threadgro'
    'up","verdict":"no_clear_difference"},"to_torch":{"first":"softmax_f2_simd","'
    'iqr_separated":false,"ratio":1.0335286195286195,"second":"torch_softmax","ve'
    'rdict":"no_clear_difference"},"to_u0":{"first":"softmax_f2_simd","iqr_separa'
    'ted":true,"ratio":2.0454545454545454,"second":"softmax_u0_unfused","verdict"'
    ':"faster"},"variant":"softmax_f2_simd"},{"base":"softmax_f2_simd","held":tru'
    'e,"predicted":["no_clear_difference","slower"],"ran":true,"rate":{"effective'
    '_fraction_of_ceiling":0.04179356753073535,"effective_gb_per_second":3.438889'
    '9274228724,"iqr_effective_gb_per_second":[3.090865143699337,3.48122400061087'
    '14],"modeled_fraction_of_ceiling":0.04212007977706922,"modeled_gb_per_second'
    '":3.465756254980864},"step":"F3","to_base":{"first":"softmax_f3_online","iqr'
    '_separated":false,"ratio":0.9740355572172099,"second":"softmax_f2_simd","ver'
    'dict":"no_clear_difference"},"to_torch":{"first":"softmax_f3_online","iqr_se'
    'parated":false,"ratio":1.0066936248224927,"second":"torch_softmax","verdict"'
    ':"no_clear_difference"},"to_u0":{"first":"softmax_f3_online","iqr_separated"'
    ':true,"ratio":1.992345457944293,"second":"softmax_u0_unfused","verdict":"fas'
    'ter"},"variant":"softmax_f3_online"}],"shape":{"cols":128,"rows":64},"torch"'
    ':{"rate":{"effective_fraction_of_ceiling":0.04151567716355081,"effective_gb_'
    'per_second":3.4160243420924035,"iqr_effective_gb_per_second":[3.403539294478'
    '1293,3.417404850831394],"modeled_fraction_of_ceiling":null,"modeled_gb_per_s'
    'econd":null},"to_u0":{"first":"torch_softmax","iqr_separated":true,"ratio":1'
    '.9790981176577904,"second":"softmax_u0_unfused","verdict":"faster"}}},{"best'
    '":{"to_torch":{"first":"softmax_f2_simd","iqr_separated":true,"ratio":1.4671'
    '231144782781,"second":"torch_softmax","verdict":"faster"},"to_u0":{"first":"'
    'softmax_f2_simd","iqr_separated":true,"ratio":2.9011237758089146,"second":"s'
    'oftmax_u0_unfused","verdict":"faster"},"variant":"softmax_f2_simd"},"index":'
    '1,"rows":[{"base":"torch_softmax","held":true,"predicted":["slower"],"ran":t'
    'rue,"rate":{"effective_fraction_of_ceiling":0.01079594957874853,"effective_g'
    'b_per_second":0.88832048702281,"iqr_effective_gb_per_second":[0.866977538539'
    '3057,0.898264406779661],"modeled_fraction_of_ceiling":0.022036467974965415,"'
    'modeled_gb_per_second":1.8132213216628437},"step":"U0","to_base":{"first":"s'
    'oftmax_u0_unfused","iqr_separated":true,"ratio":0.5057085556679509,"second":'
    '"torch_softmax","verdict":"slower"},"to_torch":{"first":"softmax_u0_unfused"'
    ',"iqr_separated":true,"ratio":0.5057085556679509,"second":"torch_softmax","v'
    'erdict":"slower"},"to_u0":null,"variant":"softmax_u0_unfused"},{"base":"soft'
    'max_u0_unfused","held":true,"predicted":["faster"],"ran":true,"rate":{"effec'
    'tive_fraction_of_ceiling":0.02510465130506008,"effective_gb_per_second":2.06'
    '56799025578563,"iqr_effective_gb_per_second":[1.9497852379857439,2.130912292'
    '905148],"modeled_fraction_of_ceiling":0.025292613130460623,"modeled_gb_per_s'
    'econd":2.0811459196102313},"step":"F1","to_base":{"first":"softmax_f1_thread'
    'group","iqr_separated":true,"ratio":2.325376857490865,"second":"softmax_u0_u'
    'nfused","verdict":"faster"},"to_torch":{"first":"softmax_f1_threadgroup","iq'
    'r_separated":true,"ratio":1.1759629719853837,"second":"torch_softmax","verdi'
    'ct":"faster"},"to_u0":{"first":"softmax_f1_threadgroup","iqr_separated":true'
    ',"ratio":2.325376857490865,"second":"softmax_u0_unfused","verdict":"faster"}'
    ',"variant":"softmax_f1_threadgroup"},{"base":"softmax_f1_threadgroup","held"'
    ':true,"predicted":["faster","no_clear_difference"],"ran":true,"rate":{"effec'
    'tive_fraction_of_ceiling":0.03132038600534159,"effective_gb_per_second":2.57'
    '7127685440028,"iqr_effective_gb_per_second":[2.487181051515137,2.59394799632'
    '91525],"modeled_fraction_of_ceiling":0.03155488585376717,"modeled_gb_per_sec'
    'ond":2.5964229792945157},"step":"F2","to_base":{"first":"softmax_f2_simd","i'
    'qr_separated":true,"ratio":1.2475929509934547,"second":"softmax_f1_threadgro'
    'up","verdict":"faster"},"to_torch":{"first":"softmax_f2_simd","iqr_separated'
    '":true,"ratio":1.4671231144782781,"second":"torch_softmax","verdict":"faster'
    '"},"to_u0":{"first":"softmax_f2_simd","iqr_separated":true,"ratio":2.9011237'
    '758089146,"second":"softmax_u0_unfused","verdict":"faster"},"variant":"softm'
    'ax_f2_simd"},{"base":"softmax_f2_simd","held":true,"predicted":["no_clear_di'
    'fference","slower"],"ran":true,"rate":{"effective_fraction_of_ceiling":0.029'
    '073162818047532,"effective_gb_per_second":2.3922199678164136,"iqr_effective_'
    'gb_per_second":[2.3452444696188586,2.5070294612953887],"modeled_fraction_of_'
    'ceiling":0.029290837410976288,"modeled_gb_per_second":2.4101308332750295},"s'
    'tep":"F3","to_base":{"first":"softmax_f3_online","iqr_separated":false,"rati'
    'o":0.9282504632314938,"second":"softmax_f2_simd","verdict":"no_clear_differe'
    'nce"},"to_torch":{"first":"softmax_f3_online","iqr_separated":true,"ratio":1'
    '.3618577106320937,"second":"torch_softmax","verdict":"faster"},"to_u0":{"fir'
    'st":"softmax_f3_online","iqr_separated":true,"ratio":2.692969488786525,"seco'
    'nd":"softmax_u0_unfused","verdict":"faster"},"variant":"softmax_f3_online"}]'
    ',"shape":{"max_length":300,"rows":16},"torch":{"rate":{"effective_fraction_o'
    'f_ceiling":0.021348164783348392,"effective_gb_per_second":1.7565858379625332'
    ',"iqr_effective_gb_per_second":[1.496403527719307,1.790081486172683],"modele'
    'd_fraction_of_ceiling":null,"modeled_gb_per_second":null},"to_u0":{"first":"'
    'torch_softmax","iqr_separated":true,"ratio":1.9774235353388043,"second":"sof'
    'tmax_u0_unfused","verdict":"faster"}}}],"ratio_rule":"ratio = median sample '
    "time of the second side / median sample time of the first side, from the sam"
    "e point of the same run with the same calls per sample; faster when ratio >="
    " min_faster_ratio and the first side's p75 sample time is below the second s"
    "ide's p25; slower when 1 / ratio >= min_faster_ratio and the second side's p"
    '75 is below the first side\'s p25; otherwise no_clear_difference","vendor":"v'
    'endor GPU softmax: torch.softmax(dim=-1) on MPS, PyTorch 2.15.0"},"summary":'
    '{"benchmark_points":2,"ceiling_gb_per_second":82.28275618954719,"correctness'
    '_cases":2,"exp_probe_max_ulps":1.4061690587550402,"hypotheses":{"softmax_f1_'
    'threadgroup":{"held":2,"missed":0,"no_verdict":0},"softmax_f2_simd":{"held":'
    '2,"missed":0,"no_verdict":0},"softmax_f3_online":{"held":2,"missed":0,"no_ve'
    'rdict":0},"softmax_u0_unfused":{"held":2,"missed":0,"no_verdict":0}},"identi'
    'cal_repeats":{"softmax_f1_threadgroup":true,"softmax_f2_simd":true,"softmax_'
    'f3_online":true,"softmax_u0_unfused":true},"label":"host-specific Apple GPU '
    '(Metal) measurement","load_average_after":[8.7666015625,8.19970703125,8.2138'
    '671875],"load_average_before":[8.7666015625,8.19970703125,8.2138671875],"max'
    '_bound_ratio":0.6343091208261153,"max_row_sum_ratio":0.0370432941553379,"nai'
    've_float32_failed_cases":[1],"points":[{"best":"softmax_f2_simd","effective_'
    'gb_per_second":{"softmax_f1_threadgroup":2.685213828425096,"softmax_f2_simd"'
    ':3.5305589225589227,"softmax_f3_online":3.4388899274228724,"softmax_u0_unfus'
    'ed":1.7260510288065845,"torch_softmax":3.4160243420924035},"index":0,"shape"'
    ':{"cols":128,"rows":64},"to_base":{"softmax_f1_threadgroup":"faster","softma'
    'x_f2_simd":"no_clear_difference","softmax_f3_online":"no_clear_difference","'
    'softmax_u0_unfused":"slower"}},{"best":"softmax_f2_simd","effective_gb_per_s'
    'econd":{"softmax_f1_threadgroup":2.0656799025578563,"softmax_f2_simd":2.5771'
    '27685440028,"softmax_f3_online":2.3922199678164136,"softmax_u0_unfused":0.88'
    '832048702281,"torch_softmax":1.7565858379625332},"index":1,"shape":{"max_len'
    'gth":300,"rows":16},"to_base":{"softmax_f1_threadgroup":"faster","softmax_f2'
    '_simd":"faster","softmax_f3_online":"no_clear_difference","softmax_u0_unfuse'
    'd":"slower"}}],"refused_steps":[],"same_bits":[{"as":"softmax_u0_unfused","d'
    'iffer_at":[],"same_at":[0,1],"variant":"softmax_f1_threadgroup"}],"steps_che'
    'cked":["softmax_u0_unfused","softmax_f1_threadgroup","softmax_f2_simd","soft'
    'max_f3_online"],"timed_samples":50,"warnings":0},"timer":"time.perf_counter_'
    'ns","torch_mps_switches":{"source":"os.environ, read before the runtime impo'
    "rts torch; null means unset, and a value that is not a short token is 'unkno"
    'wn\'","values":{"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}'
    '},"warnings":{"capture":"warnings.catch_warnings(record=True) with simplefil'
    "ter('always'), from before the runtime imports torch to the end of the run; "
    'equal warnings are kept once with a count","count":0,"distinct":0,"records":'
    '[],"torch_imported_before_capture":false}}'
)


def recorded():
    return json.loads(RECORDED_REPORT_JSON)


def test_the_recorded_report_verifies_as_a_measurement_on_any_host(tmp_path):
    config = config_from_mapping(RECORDED_CONFIG)

    summary = verify_report(
        recorded(), config=config, ceiling=recorded_ceiling(tmp_path)
    )

    assert canonical_json_bytes(recorded()).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert summary["config_sha256"] == config.fingerprint()
    assert summary["load_average_before"] == [8.7666015625, 8.19970703125, 8.2138671875]
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (2, 2)


def test_the_recorded_report_verifies_where_libm_rounds_exp_otherwise(
    monkeypatch, tmp_path
):
    # A libm on another host can round a float64 exp to the other neighbor.
    config = config_from_mapping(RECORDED_CONFIG)
    ceiling = recorded_ceiling(tmp_path)
    real = numpy.exp

    def other_libm(values, *args, **kwargs):
        result = real(values, *args, **kwargs)
        if isinstance(result, numpy.ndarray) and result.dtype == numpy.float64:
            up = numpy.nextafter(result, numpy.inf)
            result = numpy.where(result.view(numpy.int64) % 3 == 0, up, result)
        return result

    monkeypatch.setattr(numpy, "exp", other_libm)
    summary = verify_report(recorded(), config=config, ceiling=ceiling)

    assert summary["measured"] is True


def test_the_report_seals_the_plan_that_was_declared_before_the_run():
    report = recorded()

    assert report["plan"] == {**plan_mapping(), "sha256": plan_sha256()}
    assert report["config"]["plan_sha256"] == plan_sha256()


def test_the_fresh_process_captured_the_warnings_from_before_the_torch_import():
    captured = recorded()["warnings"]

    assert captured["torch_imported_before_capture"] is False
    assert captured["count"] == 0 and captured["records"] == []
    assert recorded()["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }


def test_every_recorded_pipeline_admitted_its_step_with_the_declared_bytes():
    shaders = recorded()["shaders"]

    assert all(entry["admitted"] for entry in shaders["admissions"])
    for entry in shaders["kernels"]:
        limits = entry["limits"]
        assert limits["max_threads_per_threadgroup"] == 1024
        assert limits["thread_execution_width"] == 32
        assert limits["static_threadgroup_memory_bytes"] == (
            KERNELS[entry["name"]].threadgroup_bytes
        )


def test_the_apple_gpu_kept_the_declared_bits_and_the_bounds():
    report = recorded()
    summary = report["summary"]

    assert summary["identical_repeats"] == dict.fromkeys(STEP_VARIANTS, True)
    assert summary["same_bits"][0]["same_at"] == [0, 1]
    assert summary["naive_float32_failed_cases"] == [1]
    assert summary["max_bound_ratio"] <= 1
    assert report["exp_probe"]["holds"] is True
    assert report["exp_probe"]["max_ulps"] <= 4


def test_the_command_verifies_the_recorded_file_without_torch(tmp_path, capsys):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")
    ceilings = write_recorded(tmp_path)

    assert main(["verify", "--report", str(path), "--ceilings", str(ceilings)]) == (
        EXIT_OK
    )
    expected = capsys.readouterr().out
    assert expected.startswith(
        f"verified_metal_fused_measurement={recorded()['report_sha256']} "
    )
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_fused_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "verify",
            "--report",
            str(path),
            "--ceilings",
            str(ceilings),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout == expected


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    runtime = fused_runtime()

    for entry in recorded()["shaders"]["kernels"]:
        assert runtime.limits[entry["name"]].to_mapping() == entry["limits"]
