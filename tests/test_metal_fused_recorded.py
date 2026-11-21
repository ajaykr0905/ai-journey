"""One real fused softmax report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``metal_fused_cli run``
wrote in a fresh process on 2026-11-22 at 05:00 IST under the shared GPU
lock, with torch 2.15.0 on macOS 27.2 (arm64). The ``uptime`` command printed
load averages of 6.25, 6.46 and 6.68 before the run and 6.63, 6.54 and 6.70
after it, from other jobs. The run is short: two correctness cases, two
small timing points, five repeats and a few calls for each sample, so the
cost of each synchronize call sets most of its rates. Its ceiling is the
recorded Day 74 gpu-ceilings report of 2026-11-17, 82.28 GB/s at 4 MiB.
The timings describe that moment and are here only to be verified. A host
without torch or MPS can still check a genuine Apple GPU (Metal) report.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

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
    '8","longest":128,"rows":64,"shortest":128},"reference_sha256":"78f339a670cdc'
    '05ceeb945d19aaa97d0158fbf31194134ae8007ff22f54852af","shape":{"cols":128,"ro'
    'ws":64},"side_order":["softmax_u0_unfused","softmax_f1_threadgroup","softmax'
    '_f2_simd","softmax_f3_online","torch_softmax"],"sides":{"softmax_f1_threadgr'
    'oup":{"calls_per_sample":16,"compulsory_bytes_per_call":65536,"effective_gb_'
    'per_second":2.956158428458257,"implementation":"softmax_f1_threadgroup","iqr'
    '_effective_gb_per_second":[2.9009622141438287,2.9610335333568654],"iqr_ns_pe'
    'r_call":458.3125,"median_ns_per_call":22169.3125,"modeled_bytes_per_call":66'
    '048,"modeled_gb_per_second":2.9792534161805873,"repeats":5,"samples_ns":[361'
    '458,385500,354125,354709,348375],"timing":{"count":5,"iqr_ns":7333,"max_ns":'
    '385500,"median_ns":354709,"min_ns":348375,"p25_ns":354125,"p75_ns":361458},"'
    'warmup":1},"softmax_f2_simd":{"calls_per_sample":16,"compulsory_bytes_per_ca'
    'll":65536,"effective_gb_per_second":3.4506820238584943,"implementation":"sof'
    'tmax_f2_simd","iqr_effective_gb_per_second":[2.6970102265478713,3.4870123607'
    '87339],"iqr_ns_per_call":5505.1875,"median_ns_per_call":18992.1875,"modeled_'
    'bytes_per_call":66048,"modeled_gb_per_second":3.477640477169889,"repeats":5,'
    '"samples_ns":[300709,1952500,388792,296250,303875],"timing":{"count":5,"iqr_'
    'ns":88083,"max_ns":1952500,"median_ns":303875,"min_ns":296250,"p25_ns":30070'
    '9,"p75_ns":388792},"warmup":1},"softmax_f3_online":{"calls_per_sample":16,"c'
    'ompulsory_bytes_per_call":65536,"effective_gb_per_second":3.35947456948338,"'
    'implementation":"softmax_f3_online","iqr_effective_gb_per_second":[3.1230856'
    '29188384,3.3639580506177236],"iqr_ns_per_call":1502.5625,"median_ns_per_call'
    '":19507.8125,"modeled_bytes_per_call":66048,"modeled_gb_per_second":3.385720'
    '464557469,"repeats":5,"samples_ns":[312125,335750,626417,310125,311709],"tim'
    'ing":{"count":5,"iqr_ns":24041,"max_ns":626417,"median_ns":312125,"min_ns":3'
    '10125,"p25_ns":311709,"p75_ns":335750},"warmup":1},"softmax_u0_unfused":{"ca'
    'lls_per_sample":16,"compulsory_bytes_per_call":65536,"effective_gb_per_secon'
    'd":1.8051663438777705,"implementation":"softmax_u0_unfused","iqr_effective_g'
    'b_per_second":[1.6364822473663676,1.8056823641445199],"iqr_ns_per_call":3752'
    '.5625,"median_ns_per_call":36304.6875,"modeled_bytes_per_call":133888,"model'
    'ed_gb_per_second":3.687898429094039,"repeats":5,"samples_ns":[786416,580709,'
    '640750,580875,571334],"timing":{"count":5,"iqr_ns":60041,"max_ns":786416,"me'
    'dian_ns":580875,"min_ns":571334,"p25_ns":580709,"p75_ns":640750},"warmup":1}'
    ',"torch_softmax":{"calls_per_sample":16,"compulsory_bytes_per_call":65536,"e'
    'ffective_gb_per_second":3.428590112904755,"implementation":"torch_softmax","'
    'iqr_effective_gb_per_second":[3.2978443693821196,3.5118409286530046],"iqr_ns'
    '_per_call":1210.9375,"median_ns_per_call":19114.5625,"modeled_bytes_per_call'
    '":null,"modeled_gb_per_second":null,"repeats":5,"samples_ns":[298583,317958,'
    '448500,305833,297375],"timing":{"count":5,"iqr_ns":19375,"max_ns":448500,"me'
    'dian_ns":305833,"min_ns":297375,"p25_ns":298583,"p75_ns":317958},"warmup":1}'
    '},"torch_input":{"cols":128,"padded":false,"rows":64}},{"checks":{"softmax_f'
    '1_threadgroup":{"differing_from_rounded_reference":1352,"max_abs_error":1.54'
    '05202419271902e-08,"max_bound_ratio":0.21071182140598751},"softmax_f2_simd":'
    '{"differing_from_rounded_reference":1346,"max_abs_error":1.8185893041477996e'
    '-08,"max_bound_ratio":0.09452667092907471},"softmax_f3_online":{"differing_f'
    'rom_rounded_reference":1320,"max_abs_error":1.8185893041477996e-08,"max_boun'
    'd_ratio":0.05056777853831246},"softmax_u0_unfused":{"differing_from_rounded_'
    'reference":1352,"max_abs_error":1.5405202419271902e-08,"max_bound_ratio":0.2'
    '1071182140598751},"torch_softmax":{"differing_from_rounded_reference":1392,"'
    'max_abs_error":1.8185893041477996e-08,"max_bound_ratio":0.026708221991821803'
    '}},"compulsory_bytes_per_call":17096,"index":1,"input_sha256":"6ac3e0902ecbc'
    'fe0ac5e0340c17852c29fccbec946338a1b78374b4cd98b4d9b","launch":{"active_threa'
    'ds":512,"extent":[512],"group_size":[32],"groups":[16],"launched_threads":51'
    '2,"padding_threads":0,"threads":[512],"threads_per_group":32},"layout":{"buf'
    'fer_elements":2137,"dense":false,"elements":2137,"gap":0,"group_size":32,"le'
    'ngths_sha256":"e6e8663a89239a10457540ee3d6601bd5ae4c647be7760b1c0b100250cde0'
    '599","longest":298,"rows":16,"shortest":3},"reference_sha256":"7b82ee887c215'
    '67dfad7a7b8a8142b04661af636f7c21b38dc874bd2207f4024","shape":{"max_length":3'
    '00,"rows":16},"side_order":["softmax_u0_unfused","softmax_f1_threadgroup","s'
    'oftmax_f2_simd","softmax_f3_online","torch_softmax"],"sides":{"softmax_f1_th'
    'readgroup":{"calls_per_sample":62,"compulsory_bytes_per_call":17096,"effecti'
    've_gb_per_second":2.03055938697318,"implementation":"softmax_f1_threadgroup"'
    ',"iqr_effective_gb_per_second":[1.9689523027568439,2.2757960279119698],"iqr_'
    'ns_per_call":1170.6935483870968,"median_ns_per_call":8419.354838709678,"mode'
    'led_bytes_per_call":17224,"modeled_gb_per_second":2.0457624521072795,"repeat'
    's":5,"samples_ns":[522000,465208,1200708,465750,538333],"timing":{"count":5,'
    '"iqr_ns":72583,"max_ns":1200708,"median_ns":522000,"min_ns":465208,"p25_ns":'
    '465750,"p75_ns":538333},"warmup":1},"softmax_f2_simd":{"calls_per_sample":62'
    ',"compulsory_bytes_per_call":17096,"effective_gb_per_second":2.3420006009934'
    '067,"implementation":"softmax_f2_simd","iqr_effective_gb_per_second":[2.2072'
    '768466997634,2.6101856024507306],"iqr_ns_per_call":1195.5645161290322,"media'
    'n_ns_per_call":7299.741935483871,"modeled_bytes_per_call":17224,"modeled_gb_'
    'per_second":2.3595354674491364,"repeats":5,"samples_ns":[1078584,400125,4525'
    '84,406083,480208],"timing":{"count":5,"iqr_ns":74125,"max_ns":1078584,"media'
    'n_ns":452584,"min_ns":400125,"p25_ns":406083,"p75_ns":480208},"warmup":1},"s'
    'oftmax_f3_online":{"calls_per_sample":62,"compulsory_bytes_per_call":17096,"'
    'effective_gb_per_second":2.4898510484861727,"implementation":"softmax_f3_onl'
    'ine","iqr_effective_gb_per_second":[2.486445910314995,2.49571941192537],"iqr'
    '_ns_per_call":25.548387096774192,"median_ns_per_call":6866.274193548387,"mod'
    'eled_bytes_per_call":17224,"modeled_gb_per_second":2.508492890683542,"repeat'
    's":5,"samples_ns":[425709,410958,426292,424708,467500],"timing":{"count":5,"'
    'iqr_ns":1584,"max_ns":467500,"median_ns":425709,"min_ns":410958,"p25_ns":424'
    '708,"p75_ns":426292},"warmup":1},"softmax_u0_unfused":{"calls_per_sample":62'
    ',"compulsory_bytes_per_call":17096,"effective_gb_per_second":0.8749687349607'
    '939,"implementation":"softmax_u0_unfused","iqr_effective_gb_per_second":[0.7'
    '680811594202899,0.9072663341014061],"iqr_ns_per_call":3414.6451612903224,"me'
    'dian_ns_per_call":19538.983870967742,"modeled_bytes_per_call":34896,"modeled'
    '_gb_per_second":1.7859680027604037,"repeats":5,"samples_ns":[1211417,1608375'
    ',1168292,1047042,1380000],"timing":{"count":5,"iqr_ns":211708,"max_ns":16083'
    '75,"median_ns":1211417,"min_ns":1047042,"p25_ns":1168292,"p75_ns":1380000},"'
    'warmup":1},"torch_softmax":{"calls_per_sample":62,"compulsory_bytes_per_call'
    '":17096,"effective_gb_per_second":1.8662506734652915,"implementation":"torch'
    '_softmax","iqr_effective_gb_per_second":[1.6967159056711296,1.88310370863868'
    '53],"iqr_ns_per_call":997.3064516129032,"median_ns_per_call":9160.6129032258'
    '07,"modeled_bytes_per_call":null,"modeled_gb_per_second":null,"repeats":5,"s'
    'amples_ns":[562875,561583,624708,567958,885292],"timing":{"count":5,"iqr_ns"'
    ':61833,"max_ns":885292,"median_ns":567958,"min_ns":561583,"p25_ns":562875,"p'
    '75_ns":624708},"warmup":1}},"torch_input":{"cols":298,"padded":true,"rows":1'
    '6}}],"synchronization":"torch.mps.synchronize() before the first and before '
    "the second clock reading of every sample; untimed warmups of each side first"
    "; repeat r starts with side r mod S, so the order shifts by one side every r"
    'epeat","torch_equivalent":"torch.softmax(x, dim=-1, out=y) on the MPS device'
    ": on the rows x cols matrix of a dense point, and on the rows x longest matr"
    "ix of a ragged point, with the most negative finite float32 in each slot pas"
    't a row","traffic_model":"device bytes of one call, with no cache, for n ele'
    "ments in r rows: U0 reads the row in each of its three kernels and writes it"
    " once, 16 n bytes, and moves 44 r bytes of row data (8 r of starts and lengt"
    "hs in each kernel, the max written once and read twice, the sum written once"
    " and read once); a fused step reads the row once and writes it once, 8 n byt"
    "es, plus 8 r of starts and lengths; torch.softmax has no model, because PyTo"
    'rch does not name its kernel","vendor_bound":"|y - y_ref| <= 2**-16 * y_ref '
    "+ 2**-125 for every torch.softmax output: a sanity bound only, because PyTor"
    'ch names no kernel and no derived bound applies"},"ceiling":{"buffer_bytes":'
    '4194304,"config_sha256":"9e1db2639799674c80116c53f0ab2a97583f5400def85d53648'
    '3da68aff6d173","definition":"the bandwidth ceiling of a verified Day 74 gpu-'
    "ceilings report: the better median GB/s of stream_copy and stream_read_reduc"
    "e at its largest buffer, on the host that made it; not a vendor peak and not"
    ' a hardware limit","environment":{"macos_version":"27.2","numpy_version":"1.'
    '26.4","platform_machine":"arm64","platform_system":"Darwin","python_implemen'
    'tation":"CPython","python_version":"3.11.14","torch_version":"2.15.0"},"gb_p'
    'er_second":82.28275618954719,"kernel":"stream_read_reduce","label":"host-spe'
    'cific Apple GPU (Metal) measurement","load_average_after":[9.4765625,9.40478'
    '515625,8.75341796875],"load_average_before":[9.4765625,9.40478515625,8.75341'
    '796875],"measured":true,"report_sha256":"6b784c28e36b2890eadb65c742858b29835'
    'b90f8e259b4befba92a499c919cb6","source":"gpu_ceilings"},"config":{"benchmark'
    '":{"inputs":"normal","max_runtime_seconds":60,"repeats":5,"shapes":[{"cols":'
    '128,"rows":64},{"max_length":300,"rows":16}],"target_bytes_per_sample":10485'
    '76,"warmup":1},"comparison":{"min_faster_ratio":1.1},"correctness":{"cases":'
    '[{"inputs":"uniform_80","lengths":[1,31,33]},{"cols":64,"inputs":"offset_plu'
    's_1000","rows":2}],"repeats":2},"name":"metal-fused-recorded","plan_sha256":'
    '"198870239a4a52adb106ed2396e7f9a99d1c2c52ba9770e9fea820e28b032990","schema_v'
    'ersion":1,"seed":79},"config_sha256":"f5356d711d410732341692d900e6841b8615c8'
    '3ceebadf9c2b0036897ba7ca60","correctness":{"bound":"|y - y_ref| <= (rel + re'
    "f) / (1 - ref) * y_ref + 2**-125 for each element, with rel = alpha (1 + u) "
    "/ (1 - rho) - 1, alpha = exp(d u) (1 + EXP_ERROR), rho = (sum_j y_ref_j (bet"
    "a_j - 1)) / (1 - ref) + (n + 2 G) 2**-126, beta = exp(d u) (1 + EXP_ERROR)**"
    "N (1 + gamma_A), d = max - x, u = 2**-24, EXP_ERROR = 4 * 2**-23, gamma_A = "
    'A u / (1 - A u) and ref = gamma_(n+8) for the unit 2**-53","cases":[{"case":'
    '{"inputs":"uniform_80","lengths":[1,31,33]},"index":0,"input_sha256":"4994f5'
    '444fc0d8d4fa7d53978a9a5a838a1e513897a45a01a80389020b34e12d","launch":{"activ'
    'e_threads":96,"extent":[96],"group_size":[32],"groups":[3],"launched_threads'
    '":96,"padding_threads":0,"threads":[96],"threads_per_group":32},"layout":{"b'
    'uffer_elements":161,"dense":false,"elements":65,"gap":32,"group_size":32,"le'
    'ngths_sha256":"72e8f7ca85b07b7a531470b61a3cc9ae6d349f144413996ddaded1b5c430f'
    '6ad","longest":33,"rows":3,"shortest":1},"naive_float32_finite":true,"output'
    's":[{"bound_sha256":"f54c032321fd1c3cd0426543828c1861dde8196f854dfc0ea8a6417'
    '11265e3f4","buffer_sha256":"23d2685078ed024181edd4b9baace5f4f236dd3d6d24f653'
    'fe13adb14e29feaf","differing_from_rounded_reference":30,"identical_repeats":'
    'true,"kernel":"softmax_u0_unfused","max_abs_error":3.6409037562634694e-08,"m'
    'ax_bound_ratio":0.6343091208261153,"max_row_sum_error":5.1672818912607e-08,"'
    'max_row_sum_ratio":0.0370432941553379,"row_sum_bound_sha256":"047becdedd9e94'
    '20835d00f2c2339f6a47c41c32785e155db60066b78b8080e6"},{"bound_sha256":"f54c03'
    '2321fd1c3cd0426543828c1861dde8196f854dfc0ea8a641711265e3f4","buffer_sha256":'
    '"23d2685078ed024181edd4b9baace5f4f236dd3d6d24f653fe13adb14e29feaf","differin'
    'g_from_rounded_reference":30,"identical_repeats":true,"kernel":"softmax_f1_t'
    'hreadgroup","max_abs_error":3.6409037562634694e-08,"max_bound_ratio":0.63430'
    '91208261153,"max_row_sum_error":5.1672818912607e-08,"max_row_sum_ratio":0.03'
    '70432941553379,"row_sum_bound_sha256":"047becdedd9e9420835d00f2c2339f6a47c41'
    'c32785e155db60066b78b8080e6"},{"bound_sha256":"774298709a748830cc29b6e648eee'
    '30923f2c7857b5bdbd39c66cd6def1d681f","buffer_sha256":"23d2685078ed024181edd4'
    'b9baace5f4f236dd3d6d24f653fe13adb14e29feaf","differing_from_rounded_referenc'
    'e":30,"identical_repeats":true,"kernel":"softmax_f2_simd","max_abs_error":3.'
    '6409037562634694e-08,"max_bound_ratio":0.5083186886895924,"max_row_sum_error'
    '":5.1672818912607e-08,"max_row_sum_ratio":0.01754796653361059,"row_sum_bound'
    '_sha256":"fbd3e65ccf22a2e58e2540b449f31484c4124432d1674ce1d4665fd5974177c5"}'
    ',{"bound_sha256":"96ff3a91f0d72013fc6ea4a059ec1f40499eb2ffa8861ff343e5ad4d1e'
    '92a669","buffer_sha256":"23d2685078ed024181edd4b9baace5f4f236dd3d6d24f653fe1'
    '3adb14e29feaf","differing_from_rounded_reference":30,"identical_repeats":tru'
    'e,"kernel":"softmax_f3_online","max_abs_error":3.6409037562634694e-08,"max_b'
    'ound_ratio":0.4240841587300845,"max_row_sum_error":5.1672818912607e-08,"max_'
    'row_sum_ratio":0.011497162344799925,"row_sum_bound_sha256":"34299eb3a4820e51'
    '2dbf935835f0a35133a287d67ca9f4d09459ab17d5ce053b"}],"reference_sha256":"6396'
    'bcc2433fc42929e2e80c0100d13100513d085964e2aed11b1884555271ef","repeats":2,"s'
    'ame_bits":[{"as":"softmax_u0_unfused","same":true,"variant":"softmax_f1_thre'
    'adgroup"}]},{"case":{"cols":64,"inputs":"offset_plus_1000","rows":2},"index"'
    ':1,"input_sha256":"11e3791e9aaae4aa253c8d8a3d0b79a9812369ca279a07c9539fd287c'
    'd12e015","launch":{"active_threads":64,"extent":[64],"group_size":[32],"grou'
    'ps":[2],"launched_threads":64,"padding_threads":0,"threads":[64],"threads_pe'
    'r_group":32},"layout":{"buffer_elements":192,"dense":true,"elements":128,"ga'
    'p":32,"group_size":32,"lengths_sha256":"d8a162db4b52873aa8bee8c7bda17ad1e10b'
    '8233d0640427722f394ec172caf3","longest":64,"rows":2,"shortest":64},"naive_fl'
    'oat32_finite":false,"outputs":[{"bound_sha256":"2faaacd31df7e1eae43af162a9ff'
    'd51b44bd150206dec25173beddfd5d6848c8","buffer_sha256":"6a19f287433a0b6074667'
    '7ba1cd6c609f5ba98b3ccd4463aba873776b30da656","differing_from_rounded_referen'
    'ce":55,"identical_repeats":true,"kernel":"softmax_u0_unfused","max_abs_error'
    '":5.059845709087796e-09,"max_bound_ratio":0.08736721050653032,"max_row_sum_e'
    'rror":2.7241185307502747e-08,"max_row_sum_ratio":0.017561609524159815,"row_s'
    'um_bound_sha256":"ac3714f9881e1b93e46f4d1c8b6caae6edc55a5e92e91e84608a4626cc'
    'f2723c"},{"bound_sha256":"2faaacd31df7e1eae43af162a9ffd51b44bd150206dec25173'
    'beddfd5d6848c8","buffer_sha256":"6a19f287433a0b60746677ba1cd6c609f5ba98b3ccd'
    '4463aba873776b30da656","differing_from_rounded_reference":55,"identical_repe'
    'ats":true,"kernel":"softmax_f1_threadgroup","max_abs_error":5.05984570908779'
    '6e-09,"max_bound_ratio":0.08736721050653032,"max_row_sum_error":2.7241185307'
    '502747e-08,"max_row_sum_ratio":0.017561609524159815,"row_sum_bound_sha256":"'
    'ac3714f9881e1b93e46f4d1c8b6caae6edc55a5e92e91e84608a4626ccf2723c"},{"bound_s'
    'ha256":"95383d2623b18c0b2e07eca8b686fa5bb76ea8c26f4748d688b1e664357938ef","b'
    'uffer_sha256":"6a19f287433a0b60746677ba1cd6c609f5ba98b3ccd4463aba873776b30da'
    '656","differing_from_rounded_reference":55,"identical_repeats":true,"kernel"'
    ':"softmax_f2_simd","max_abs_error":5.059845709087796e-09,"max_bound_ratio":0'
    '.04467182532349286,"max_row_sum_error":2.7241185307502747e-08,"max_row_sum_r'
    'atio":0.008784901583625923,"row_sum_bound_sha256":"0686144995939c2569b2d8902'
    '38448b25b272a2a01bdd363434974bb198750b2"},{"bound_sha256":"471bf64733ebb9569'
    'cbad1d8fd6440b7e3ccd9e83f9b0fe2bc8ad06ef8c3a6e4","buffer_sha256":"2ebb017dd3'
    '96f26cd70e545ad9ce58c3e1dfd6cb7da32909a10d2a6e7fb51d62","differing_from_roun'
    'ded_reference":87,"identical_repeats":true,"kernel":"softmax_f3_online","max'
    '_abs_error":3.233537348990012e-08,"max_bound_ratio":0.04329807816237615,"max'
    '_row_sum_error":1.210719347000122e-07,"max_row_sum_ratio":0.0258279799617962'
    '48,"row_sum_bound_sha256":"ab5835d5fb0440cec06ac77d8683d32f26eb09e364ad579f7'
    '6a1cde25320424f"}],"reference_sha256":"65710e0bb1856a97599ef0602b04e193f5ec0'
    '90b02e55fa247ec8237686a7202","repeats":2,"same_bits":[{"as":"softmax_u0_unfu'
    'sed","same":true,"variant":"softmax_f1_threadgroup"}]}],"distributions":{"no'
    'rmal":"a standard normal, drawn as float32","offset_minus_1000":"-1000 plus '
    'a standard normal, rounded to float32","offset_plus_1000":"1000 plus a stand'
    'ard normal, rounded to float32","uniform_80":"uniform from -80 to 80, drawn '
    'in float64 and rounded to float32"},"gap":32,"gap_input":1.2676506002282294e'
    '+30,"order_terms":{"softmax_f1_threadgroup":"A = (k - 1) + log2(G) and N = 1'
    ': the additions of U0 in its order","softmax_f2_simd":"A = (k - 1) + 31 + (G'
    " / 32 - 1) and N = 1: k slots in order, simd_sum over 32 lanes, then simd_su"
    'm over the G / 32 partials","softmax_f3_online":"A = (k - 1) + 1 + 31 + 1 + '
    "(G / 32 - 1) and N = k + 2: k online updates, a rescale and simd_sum in each"
    " SIMD group, a rescale and simd_sum of the partials; each update or rescale "
    'rounds once","softmax_u0_unfused":"A = (k - 1) + log2(G) and N = 1: k slots '
    'added in order, then a tree"},"reference":"float64 softmax of the float32 in'
    "puts, row by row: exp(x - max) divided by the sum of exp(x - max) over the r"
    'ow, with numpy.exp and numpy.add.reduceat","row_sum_bound":"|sum(y) - 1| <= '
    "the sum of the element bounds of the row + ref + 2 gamma_n for the unit 2**-"
    '53, where sum adds the outputs of the row in float64 in order","rules":"ever'
    "y gap slot keeps its sentinel in every repeat; every output is finite; every"
    " element is within its bound of the reference; every row sum is within its b"
    "ound of 1; identical_repeats, same_bits and differing_from_rounded_reference"
    ' are measured","sentinel_float32_bits":2143289344},"environment":{"macos_ver'
    'sion":"27.2","numpy_version":"1.26.4","platform_machine":"arm64","platform_s'
    'ystem":"Darwin","python_implementation":"CPython","python_version":"3.11.14"'
    ',"torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_host":'
    'true,"cuda_kernel_executed":false,"distributed_claim":false,"hardware_counte'
    'rs_used":false,"host_specific":true,"nvidia_gpu_used":false,"occupancy_measu'
    'red":false,"portable_performance_claim":false,"production_claim":false,"stat'
    'ement":"These are wall-clock timings on the Apple GPU of the host that made '
    "this report, at the recorded load averages. The softmax kernels came from to"
    "rch.mps.compile_shader, and torch.softmax ran on the MPS device. PyTorch doe"
    "s not report which kernel torch.softmax ran, so the vendor number names the "
    "call, not a kernel. The bandwidth ceiling comes from a separate gpu-ceilings"
    " run with its own load averages. No hardware counter was read, so no cache h"
    "it, stall or occupancy was observed, and each byte count is a model. No CUDA"
    " kernel ran, no NVIDIA GPU was used, and no Triton kernel ran on a GPU. Each"
    " ratio compares two sides of one run on this host. These timings do not meas"
    'ure or predict CUDA, NVIDIA, Triton, distributed or production performance."'
    ',"triton_gpu_kernel_executed":false,"vendor_kernel_known":false,"vendor_peak'
    '_claimed":false},"exp_probe":{"arguments":"precise::exp at 2**18 float32 arg'
    "uments: 262136 evenly spaced from -104 to 0, then the extremes -3.4028234663"
    '852886e+38, -1e+30, -1000.0, -200.0, -150.0, -105.0, -87.5, -0.0","arguments'
    '_sha256":"ef74306503b0bf88378fecc220238ffe4510b9ce6992098cb47e1e39804aab8d",'
    '"count":262144,"exp_error":4.76837158203125e-07,"exp_ulps":4,"holds":true,"m'
    'ax_relative_error":1.61439084446152e-07,"max_ulps":1.4061690587550402,"norma'
    'l_results":220136,"results_sha256":"918b6099a06c5fc92292905d92d8968e8f6b44a6'
    'cc1e883b02864e0cc4467498","rule":"each result whose float64 exp is at least '
    "2**-126 has a relative error of at most EXP_ERROR = 4 * 2**-23; each smaller"
    ' result is within 2**-126 of it","tiny_max_abs_error":1.1751589259924035e-38'
    ',"tiny_nonzero":0,"tiny_results":42008},"host_load":{"after":[6.6337890625,6'
    '.53662109375,6.7041015625],"before":[6.25390625,6.46142578125,6.67919921875]'
    ',"source":"os.getloadavg(): the 1, 5 and 15 minute load averages that uptime'
    ' prints, read before the checks and after the last sample"},"kind":"metal_fu'
    'sed_softmax_correctness_and_bandwidth","label":"host-specific Apple GPU (Met'
    'al) measurement","measured":true,"measurement_scope":"apple_gpu_metal_this_h'
    'ost","percentile_method":"nearest rank: sorted samples[ceil(p * n / 100) - 1'
    "] for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no interpolati"
    'on","plan":{"baseline":"softmax_u0_unfused","sha256":"198870239a4a52adb106ed'
    '2396e7f9a99d1c2c52ba9770e9fea820e28b032990","steps":[{"base":"torch_softmax"'
    ',"change":"Compute the softmax in three kernels: the max of each row, the su'
    "m of exp(x - max) and the normalize pass. The max and the sum go through two"
    ' buffers in device memory.","hypothesis":"U0 reads each row three times and '
    "writes it once, so the model gives it about twice the bytes of one read and "
    "one write. It also launches three kernels for each call. U0 is slower than t"
    'orch.softmax at every point.","predicted":["slower"],"same_bits_as":null,"so'
    'urce_sha256":"c38c9faf73e14e5695bd758ad3f4d3547b17d571877f091d3264d729bb7199'
    '7c","step":"U0","variant":"softmax_u0_unfused"},{"base":"softmax_u0_unfused"'
    ',"change":"Fuse the three passes into one kernel. Each thread keeps its 16 s'
    "lots of the row in registers, and two trees in threadgroup memory give the m"
    'ax and the sum.","hypothesis":"F1 reads each element once and writes it once'
    ", so the model gives it about half the bytes of U0, with one launch in place"
    " of three. F1 is faster than U0 by 1.1 or more at every point. F1 adds the s"
    'ame values in the same order as U0, so its outputs can keep the U0 bits.","p'
    'redicted":["faster"],"same_bits_as":"softmax_u0_unfused","source_sha256":"4a'
    '41f2e138c5a3e499293b771f2e105f4c13920ba291b466b75cab657ebe665a","step":"F1",'
    '"variant":"softmax_f1_threadgroup"},{"base":"softmax_f1_threadgroup","change'
    '":"Replace each threadgroup tree with simd_max or simd_sum in each SIMD grou'
    'p and one threadgroup step.","hypothesis":"A tree over G threads needs log2('
    "G) + 1 barriers for each value, and F2 needs one. F2 moves the same modeled "
    "bytes as F1, and the loads of the row probably set most of the time. F2 is f"
    'aster than F1 or shows no clear difference.","predicted":["faster","no_clear'
    '_difference"],"same_bits_as":null,"source_sha256":"a8380a8b2ea58e8a84ed8c8a1'
    '025c7cb5ea7f4f1111d1abde374135e0c0af4d8","step":"F2","variant":"softmax_f2_s'
    'imd"},{"base":"softmax_f2_simd","change":"Keep a running max and a running s'
    "um in one pass over the slots, rescale the sum when the max grows, and merge"
    ' the (max, sum) pairs in one threadgroup step.","hypothesis":"F3 moves the s'
    "ame modeled bytes as F2 and needs one barrier in place of two. It calls prec"
    "ise::exp twice for each element, and F2 calls it once. F3 is not faster than"
    ' F2: it shows no clear difference or it is slower.","predicted":["no_clear_d'
    'ifference","slower"],"same_bits_as":null,"source_sha256":"169845113557a1bf67'
    'e8fff9320b28ca920547ecc54f175b435039b7fb7124c5","step":"F3","variant":"softm'
    'ax_f3_online"}],"vendor":"torch_softmax"},"planned_work":{"benchmark_points"'
    ':2,"calls":2340,"compulsory_bytes":63255840,"correctness_cases":2,"correctne'
    'ss_elements":193,"correctness_launches":25,"exp_probe_arguments":262144,"ker'
    'nel_launches":3276,"sides_per_point":5,"timed_samples":50},"refusals":[],"re'
    'port_sha256":"5fc3ec4f0d777048d488da34bebbb286224eef11a0387369f6736c2927f65e'
    '68","run":{"elapsed_ns":486203208,"max_runtime_ns":60000000000,"timed_ns":29'
    '553044},"schema_version":1,"shaders":{"admissions":[{"admitted":true,"kernel'
    's":[{"allowed_threads":1024,"declared_threadgroup_bytes":4096,"name":"softma'
    'x_u0_row_max","static_threadgroup_bytes":4096,"thread_execution_width":32},{'
    '"allowed_threads":1024,"declared_threadgroup_bytes":4096,"name":"softmax_u0_'
    'exp_sum","static_threadgroup_bytes":4096,"thread_execution_width":32},{"allo'
    'wed_threads":1024,"declared_threadgroup_bytes":0,"name":"softmax_u0_normaliz'
    'e","static_threadgroup_bytes":0,"thread_execution_width":32}],"reason":null,'
    '"required_threads":32,"variant":"softmax_u0_unfused"},{"admitted":true,"kern'
    'els":[{"allowed_threads":1024,"declared_threadgroup_bytes":8192,"name":"soft'
    'max_f1_threadgroup","static_threadgroup_bytes":8192,"thread_execution_width"'
    ':32}],"reason":null,"required_threads":32,"variant":"softmax_f1_threadgroup"'
    '},{"admitted":true,"kernels":[{"allowed_threads":1024,"declared_threadgroup_'
    'bytes":256,"name":"softmax_f2_simd","static_threadgroup_bytes":256,"thread_e'
    'xecution_width":32}],"reason":null,"required_threads":32,"variant":"softmax_'
    'f2_simd"},{"admitted":true,"kernels":[{"allowed_threads":1024,"declared_thre'
    'adgroup_bytes":256,"name":"softmax_f3_online","static_threadgroup_bytes":256'
    ',"thread_execution_width":32}],"reason":null,"required_threads":32,"variant"'
    ':"softmax_f3_online"}],"kernels":[{"buffers":["row_max","x","starts","length'
    's"],"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_'
    'bytes":4096,"thread_execution_width":32},"name":"softmax_u0_row_max","simd_f'
    'unctions":false,"source_sha256":"d7a01ba402c8b179dd88ae3b9c73af5407df42ea574'
    '228538b6e9c414c13c47b","threadgroup_bytes":4096},{"buffers":["row_sum","x","'
    'row_max","starts","lengths"],"limits":{"max_threads_per_threadgroup":1024,"s'
    'tatic_threadgroup_memory_bytes":4096,"thread_execution_width":32},"name":"so'
    'ftmax_u0_exp_sum","simd_functions":false,"source_sha256":"286b2d0374b70cd507'
    '7c5e0a52e74311c57d406dcf8a295319a0857a48a0bc64","threadgroup_bytes":4096},{"'
    'buffers":["y","x","row_max","row_sum","starts","lengths"],"limits":{"max_thr'
    'eads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execu'
    'tion_width":32},"name":"softmax_u0_normalize","simd_functions":false,"source'
    '_sha256":"e6fd4b2679dacd8ff2671a26f6074baab5a1e4d0259758b0197f4f7930e60b8c",'
    '"threadgroup_bytes":0},{"buffers":["y","x","starts","lengths"],"limits":{"ma'
    'x_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":8192,"thre'
    'ad_execution_width":32},"name":"softmax_f1_threadgroup","simd_functions":fal'
    'se,"source_sha256":"4a41f2e138c5a3e499293b771f2e105f4c13920ba291b466b75cab65'
    '7ebe665a","threadgroup_bytes":8192},{"buffers":["y","x","starts","lengths"],'
    '"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_byte'
    's":256,"thread_execution_width":32},"name":"softmax_f2_simd","simd_functions'
    '":true,"source_sha256":"a8380a8b2ea58e8a84ed8c8a1025c7cb5ea7f4f1111d1abde374'
    '135e0c0af4d8","threadgroup_bytes":256},{"buffers":["y","x","starts","lengths'
    '"],"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_b'
    'ytes":256,"thread_execution_width":32},"name":"softmax_f3_online","simd_func'
    'tions":true,"source_sha256":"169845113557a1bf67e8fff9320b28ca920547ecc54f175'
    'b435039b7fb7124c5","threadgroup_bytes":256},{"buffers":["y","x"],"limits":{"'
    'max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"threa'
    'd_execution_width":32},"name":"softmax_exp_probe","simd_functions":false,"so'
    'urce_sha256":"1d62fece5aebee7e36cc9803962fdc9bd8c2aece802b5fa4c04f1b18d85170'
    'f6","threadgroup_bytes":0}],"library_sha256":"c8c1b7ff06c16966a8a0b9e646868e'
    '62055faa6ca247e4c0864595a7e9cd7521","steps":[{"kernels":["softmax_u0_row_max'
    '","softmax_u0_exp_sum","softmax_u0_normalize"],"launches_per_call":3,"name":'
    '"softmax_u0_unfused","reduction":"threadgroup_tree","row_reads":3,"source_sh'
    'a256":"c38c9faf73e14e5695bd758ad3f4d3547b17d571877f091d3264d729bb71997c","st'
    'ep":"U0","summary":"three kernels; the max and the sum go through device mem'
    'ory"},{"kernels":["softmax_f1_threadgroup"],"launches_per_call":1,"name":"so'
    'ftmax_f1_threadgroup","reduction":"threadgroup_tree","row_reads":1,"source_s'
    'ha256":"4a41f2e138c5a3e499293b771f2e105f4c13920ba291b466b75cab657ebe665a","s'
    'tep":"F1","summary":"one kernel; the row stays in registers; two threadgroup'
    ' trees"},{"kernels":["softmax_f2_simd"],"launches_per_call":1,"name":"softma'
    'x_f2_simd","reduction":"simd_then_one_threadgroup_step","row_reads":1,"sourc'
    'e_sha256":"a8380a8b2ea58e8a84ed8c8a1025c7cb5ea7f4f1111d1abde374135e0c0af4d8"'
    ',"step":"F2","summary":"one kernel; simd_max and simd_sum, one threadgroup s'
    'tep for each"},{"kernels":["softmax_f3_online"],"launches_per_call":1,"name"'
    ':"softmax_f3_online","reduction":"online_simd_then_one_threadgroup_step","ro'
    'w_reads":1,"source_sha256":"169845113557a1bf67e8fff9320b28ca920547ecc54f175b'
    '435039b7fb7124c5","step":"F3","summary":"one kernel; one online pass for max'
    ' and sum, one merge step"}]},"speedup":{"baseline":"softmax_u0_unfused","bes'
    't_rule":"the step with the highest effective GB/s at this point; a tie goes '
    'to the earlier step of the plan","ceiling_gb_per_second":82.28275618954719,"'
    'fraction_rule":"effective GB/s of a side / the bandwidth ceiling GB/s, and f'
    "or a step also its modeled GB/s / the ceiling; the ceiling comes from a sepa"
    'rate run","hypotheses":[{"held_at":[0,1],"missed_at":[],"no_verdict_at":[],"'
    'predicted":["slower"],"step":"U0","variant":"softmax_u0_unfused"},{"held_at"'
    ':[0,1],"missed_at":[],"no_verdict_at":[],"predicted":["faster"],"step":"F1",'
    '"variant":"softmax_f1_threadgroup"},{"held_at":[0,1],"missed_at":[],"no_verd'
    'ict_at":[],"predicted":["faster","no_clear_difference"],"step":"F2","variant'
    '":"softmax_f2_simd"},{"held_at":[0,1],"missed_at":[],"no_verdict_at":[],"pre'
    'dicted":["no_clear_difference","slower"],"step":"F3","variant":"softmax_f3_o'
    'nline"}],"hypothesis_rule":"a prediction holds at a point when the verdict o'
    "f the step over its base is one of the verdicts that the step predicted; a s"
    'tep or base that did not run gives no verdict","min_faster_ratio":1.1,"point'
    's":[{"best":{"to_torch":{"first":"softmax_f2_simd","iqr_separated":false,"ra'
    'tio":1.006443438914027,"second":"torch_softmax","verdict":"no_clear_differen'
    'ce"},"to_u0":{"first":"softmax_f2_simd","iqr_separated":true,"ratio":1.91155'
    '9029206088,"second":"softmax_u0_unfused","verdict":"faster"},"variant":"soft'
    'max_f2_simd"},"index":0,"rows":[{"base":"torch_softmax","held":true,"predict'
    'ed":["slower"],"ran":true,"rate":{"effective_fraction_of_ceiling":0.02193857'
    '4100743242,"effective_gb_per_second":1.8051663438777705,"iqr_effective_gb_pe'
    'r_second":[1.6364822473663676,1.8056823641445199],"modeled_fraction_of_ceili'
    'ng":0.04481982130737779,"modeled_gb_per_second":3.687898429094039},"step":"U'
    '0","to_base":{"first":"softmax_u0_unfused","iqr_separated":true,"ratio":0.52'
    '65039810630514,"second":"torch_softmax","verdict":"slower"},"to_torch":{"fir'
    'st":"softmax_u0_unfused","iqr_separated":true,"ratio":0.5265039810630514,"se'
    'cond":"torch_softmax","verdict":"slower"},"to_u0":null,"variant":"softmax_u0'
    '_unfused"},{"base":"softmax_u0_unfused","held":true,"predicted":["faster"],"'
    'ran":true,"rate":{"effective_fraction_of_ceiling":0.035926827993564385,"effe'
    'ctive_gb_per_second":2.956158428458257,"iqr_effective_gb_per_second":[2.9009'
    '622141438287,2.9610335333568654],"modeled_fraction_of_ceiling":0.03620750633'
    '7264106,"modeled_gb_per_second":2.9792534161805873},"step":"F1","to_base":{"'
    'first":"softmax_f1_threadgroup","iqr_separated":true,"ratio":1.6376099845225'
    '24,"second":"softmax_u0_unfused","verdict":"faster"},"to_torch":{"first":"so'
    'ftmax_f1_threadgroup","iqr_separated":true,"ratio":0.862208176279711,"second'
    '":"torch_softmax","verdict":"slower"},"to_u0":{"first":"softmax_f1_threadgro'
    'up","iqr_separated":true,"ratio":1.637609984522524,"second":"softmax_u0_unfu'
    'sed","verdict":"faster"},"variant":"softmax_f1_threadgroup"},{"base":"softma'
    'x_f1_threadgroup","held":true,"predicted":["faster","no_clear_difference"],"'
    'ran":true,"rate":{"effective_fraction_of_ceiling":0.04193687941018258,"effec'
    'tive_gb_per_second":3.4506820238584943,"iqr_effective_gb_per_second":[2.6970'
    '102265478713,3.487012360787339],"modeled_fraction_of_ceiling":0.042264511280'
    '57463,"modeled_gb_per_second":3.477640477169889},"step":"F2","to_base":{"fir'
    'st":"softmax_f2_simd","iqr_separated":false,"ratio":1.1672858905800083,"seco'
    'nd":"softmax_f1_threadgroup","verdict":"no_clear_difference"},"to_torch":{"f'
    'irst":"softmax_f2_simd","iqr_separated":false,"ratio":1.006443438914027,"sec'
    'ond":"torch_softmax","verdict":"no_clear_difference"},"to_u0":{"first":"soft'
    'max_f2_simd","iqr_separated":true,"ratio":1.911559029206088,"second":"softma'
    'x_u0_unfused","verdict":"faster"},"variant":"softmax_f2_simd"},{"base":"soft'
    'max_f2_simd","held":true,"predicted":["no_clear_difference","slower"],"ran":'
    'true,"rate":{"effective_fraction_of_ceiling":0.04082841563722621,"effective_'
    'gb_per_second":3.35947456948338,"iqr_effective_gb_per_second":[3.12308562918'
    '8384,3.3639580506177236],"modeled_fraction_of_ceiling":0.04114738763439204,"'
    'modeled_gb_per_second":3.385720464557469},"step":"F3","to_base":{"first":"so'
    'ftmax_f3_online","iqr_separated":false,"ratio":0.973568281938326,"second":"s'
    'oftmax_f2_simd","verdict":"no_clear_difference"},"to_torch":{"first":"softma'
    'x_f3_online","iqr_separated":false,"ratio":0.9798414096916299,"second":"torc'
    'h_softmax","verdict":"no_clear_difference"},"to_u0":{"first":"softmax_f3_onl'
    'ine","iqr_separated":true,"ratio":1.8610332398878655,"second":"softmax_u0_un'
    'fused","verdict":"faster"},"variant":"softmax_f3_online"}],"shape":{"cols":1'
    '28,"rows":64},"torch":{"rate":{"effective_fraction_of_ceiling":0.04166839167'
    '3786775,"effective_gb_per_second":3.428590112904755,"iqr_effective_gb_per_se'
    'cond":[3.2978443693821196,3.5118409286530046],"modeled_fraction_of_ceiling":'
    'null,"modeled_gb_per_second":null},"to_u0":{"first":"torch_softmax","iqr_sep'
    'arated":true,"ratio":1.89932087119441,"second":"softmax_u0_unfused","verdict'
    '":"faster"}}},{"best":{"to_torch":{"first":"softmax_f3_online","iqr_separate'
    'd":true,"ratio":1.334146095102523,"second":"torch_softmax","verdict":"faster'
    '"},"to_u0":{"first":"softmax_f3_online","iqr_separated":true,"ratio":2.84564'
    '57345275763,"second":"softmax_u0_unfused","verdict":"faster"},"variant":"sof'
    'tmax_f3_online"},"index":1,"rows":[{"base":"torch_softmax","held":true,"pred'
    'icted":["slower"],"ran":true,"rate":{"effective_fraction_of_ceiling":0.01063'
    '3682869655086,"effective_gb_per_second":0.8749687349607939,"iqr_effective_gb'
    '_per_second":[0.7680811594202899,0.9072663341014061],"modeled_fraction_of_ce'
    'iling":0.021705252539745194,"modeled_gb_per_second":1.7859680027604037},"ste'
    'p":"U0","to_base":{"first":"softmax_u0_unfused","iqr_separated":true,"ratio"'
    ':0.4688377330019308,"second":"torch_softmax","verdict":"slower"},"to_torch":'
    '{"first":"softmax_u0_unfused","iqr_separated":true,"ratio":0.468837733001930'
    '8,"second":"torch_softmax","verdict":"slower"},"to_u0":null,"variant":"softm'
    'ax_u0_unfused"},{"base":"softmax_u0_unfused","held":true,"predicted":["faste'
    'r"],"ran":true,"rate":{"effective_fraction_of_ceiling":0.024677824139672323,'
    '"effective_gb_per_second":2.03055938697318,"iqr_effective_gb_per_second":[1.'
    '9689523027568439,2.2757960279119698],"modeled_fraction_of_ceiling":0.0248625'
    '90253960933,"modeled_gb_per_second":2.0457624521072795},"step":"F1","to_base'
    '":{"first":"softmax_f1_threadgroup","iqr_separated":true,"ratio":2.320722222'
    '2222224,"second":"softmax_u0_unfused","verdict":"faster"},"to_torch":{"first'
    '":"softmax_f1_threadgroup","iqr_separated":true,"ratio":1.0880421455938698,"'
    'second":"torch_softmax","verdict":"no_clear_difference"},"to_u0":{"first":"s'
    'oftmax_f1_threadgroup","iqr_separated":true,"ratio":2.3207222222222224,"seco'
    'nd":"softmax_u0_unfused","verdict":"faster"},"variant":"softmax_f1_threadgro'
    'up"},{"base":"softmax_f1_threadgroup","held":true,"predicted":["faster","no_'
    'clear_difference"],"ran":true,"rate":{"effective_fraction_of_ceiling":0.0284'
    '62836072218535,"effective_gb_per_second":2.3420006009934067,"iqr_effective_g'
    'b_per_second":[2.2072768466997634,2.6101856024507306],"modeled_fraction_of_c'
    'eiling":0.028675941068547733,"modeled_gb_per_second":2.3595354674491364},"st'
    'ep":"F2","to_base":{"first":"softmax_f2_simd","iqr_separated":false,"ratio":'
    '1.1533770526576281,"second":"softmax_f1_threadgroup","verdict":"no_clear_dif'
    'ference"},"to_torch":{"first":"softmax_f2_simd","iqr_separated":true,"ratio"'
    ':1.2549228430523394,"second":"torch_softmax","verdict":"faster"},"to_u0":{"f'
    'irst":"softmax_f2_simd","iqr_separated":true,"ratio":2.676667756703728,"seco'
    'nd":"softmax_u0_unfused","verdict":"faster"},"variant":"softmax_f2_simd"},{"'
    'base":"softmax_f2_simd","held":true,"predicted":["no_clear_difference","slow'
    'er"],"ran":true,"rate":{"effective_fraction_of_ceiling":0.030259694300352954'
    ',"effective_gb_per_second":2.4898510484861727,"iqr_effective_gb_per_second":'
    '[2.486445910314995,2.49571941192537],"modeled_fraction_of_ceiling":0.0304862'
    '52610510016,"modeled_gb_per_second":2.508492890683542},"step":"F3","to_base"'
    ':{"first":"softmax_f3_online","iqr_separated":false,"ratio":1.06312997845946'
    '4,"second":"softmax_f2_simd","verdict":"no_clear_difference"},"to_torch":{"f'
    'irst":"softmax_f3_online","iqr_separated":true,"ratio":1.334146095102523,"se'
    'cond":"torch_softmax","verdict":"faster"},"to_u0":{"first":"softmax_f3_onlin'
    'e","iqr_separated":true,"ratio":2.8456457345275763,"second":"softmax_u0_unfu'
    'sed","verdict":"faster"},"variant":"softmax_f3_online"}],"shape":{"max_lengt'
    'h":300,"rows":16},"torch":{"rate":{"effective_fraction_of_ceiling":0.0226809'
    '45071482318,"effective_gb_per_second":1.8662506734652915,"iqr_effective_gb_p'
    'er_second":[1.6967159056711296,1.8831037086386853],"modeled_fraction_of_ceil'
    'ing":null,"modeled_gb_per_second":null},"to_u0":{"first":"torch_softmax","iq'
    'r_separated":true,"ratio":2.132934125410682,"second":"softmax_u0_unfused","v'
    'erdict":"faster"}}}],"ratio_rule":"ratio = median sample time of the second '
    "side / median sample time of the first side, from the same point of the same"
    " run with the same calls per sample; faster when ratio >= min_faster_ratio a"
    "nd the first side's p75 sample time is below the second side's p25; slower w"
    "hen 1 / ratio >= min_faster_ratio and the second side's p75 is below the fir"
    'st side\'s p25; otherwise no_clear_difference","vendor":"vendor GPU softmax: '
    'torch.softmax(dim=-1) on MPS, PyTorch 2.15.0"},"summary":{"benchmark_points"'
    ':2,"ceiling_gb_per_second":82.28275618954719,"correctness_cases":2,"exp_prob'
    'e_max_ulps":1.4061690587550402,"hypotheses":{"softmax_f1_threadgroup":{"held'
    '":2,"missed":0,"no_verdict":0},"softmax_f2_simd":{"held":2,"missed":0,"no_ve'
    'rdict":0},"softmax_f3_online":{"held":2,"missed":0,"no_verdict":0},"softmax_'
    'u0_unfused":{"held":2,"missed":0,"no_verdict":0}},"identical_repeats":{"soft'
    'max_f1_threadgroup":true,"softmax_f2_simd":true,"softmax_f3_online":true,"so'
    'ftmax_u0_unfused":true},"label":"host-specific Apple GPU (Metal) measurement'
    '","load_average_after":[6.6337890625,6.53662109375,6.7041015625],"load_avera'
    'ge_before":[6.25390625,6.46142578125,6.67919921875],"max_bound_ratio":0.6343'
    '091208261153,"max_row_sum_ratio":0.0370432941553379,"naive_float32_failed_ca'
    'ses":[1],"points":[{"best":"softmax_f2_simd","effective_gb_per_second":{"sof'
    'tmax_f1_threadgroup":2.956158428458257,"softmax_f2_simd":3.4506820238584943,'
    '"softmax_f3_online":3.35947456948338,"softmax_u0_unfused":1.8051663438777705'
    ',"torch_softmax":3.428590112904755},"index":0,"shape":{"cols":128,"rows":64}'
    ',"to_base":{"softmax_f1_threadgroup":"faster","softmax_f2_simd":"no_clear_di'
    'fference","softmax_f3_online":"no_clear_difference","softmax_u0_unfused":"sl'
    'ower"}},{"best":"softmax_f3_online","effective_gb_per_second":{"softmax_f1_t'
    'hreadgroup":2.03055938697318,"softmax_f2_simd":2.3420006009934067,"softmax_f'
    '3_online":2.4898510484861727,"softmax_u0_unfused":0.8749687349607939,"torch_'
    'softmax":1.8662506734652915},"index":1,"shape":{"max_length":300,"rows":16},'
    '"to_base":{"softmax_f1_threadgroup":"faster","softmax_f2_simd":"no_clear_dif'
    'ference","softmax_f3_online":"no_clear_difference","softmax_u0_unfused":"slo'
    'wer"}}],"refused_steps":[],"same_bits":[{"as":"softmax_u0_unfused","differ_a'
    't":[],"same_at":[0,1],"variant":"softmax_f1_threadgroup"}],"steps_checked":['
    '"softmax_u0_unfused","softmax_f1_threadgroup","softmax_f2_simd","softmax_f3_'
    'online"],"timed_samples":50,"warnings":0},"timer":"time.perf_counter_ns","to'
    'rch_mps_switches":{"source":"os.environ, read before the runtime imports tor'
    "ch; null means unset, and a value that is not a short token is 'unknown'\",\"v"
    'alues":{"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}},"warn'
    'ings":{"capture":"warnings.catch_warnings(record=True) with simplefilter(\'al'
    "ways'), from before the runtime imports torch to the end of the run; equal w"
    'arnings are kept once with a count","count":0,"distinct":0,"records":[],"tor'
    'ch_imported_before_capture":false}}'
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
    assert summary["load_average_before"] == [6.25390625, 6.46142578125, 6.67919921875]
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (2, 2)


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
