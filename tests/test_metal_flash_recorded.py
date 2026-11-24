"""One real Day 81 report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``metal_flash_cli run``
wrote in a fresh process on 2026-11-24 at 08:47 IST under the shared GPU
lock, with torch 2.15.0 on macOS 27.2.1 (arm64). The ``uptime`` command
printed load averages of 9.80, 13.49 and 16.05 before the run, from other
jobs. The run is short: two correctness cases, two timing points of 4
sequences of 256 tokens, three repeats and one call for each sample, so the
cost of each synchronize call sets most of its times, and three of its six
predictions missed. Its ceilings come from a fresh Day 74 gpu-ceilings
report of the same morning, 107.71 GB/s and
4665.5 GFLOP/s. The numbers
describe that moment and are here only to be verified. A host without torch
or MPS can still check a genuine Apple GPU (Metal) report. The reference
uses ``exp64`` and sums the keys in order, not the libm of the host, so
another libm gives the same digests.
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import seal_report
from ai_journey.metal_flash_cli import EXIT_OK, EXIT_REJECTED, main
from ai_journey.metal_flash_config import config_from_mapping
from ai_journey.metal_flash_plan import SIDES, plan_mapping, plan_sha256
from ai_journey.metal_flash_source import KERNELS
from ai_journey.metal_flash_verify import MetalFlashVerificationError, verify_report
from test_kernel_metal_runtime import requires_metal
from test_metal_flash_runtime import flash_runtime

ROOT = Path(__file__).resolve().parents[1]
CEILING_SHA256 = "cd75564918285983fe22b4106b5084b44ef42ed9313ca2a8ef19fa164e68fd86"
RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "metal-flash-recorded",
    "seed": 81,
    "plan_sha256": "23a502f6a118c208a02fbc5ed7b36496edabe8788c9afc624e3b9b0bdfa0cbb5",
    "correctness": {
        "cases": [
            {"lengths": [1, 31, 33, 100], "inputs": "normal"},
            {"lengths": [1000, 40], "inputs": "wide_scores"},
        ],
        "repeats": 2,
    },
    "benchmark": {
        "inputs": "normal",
        "shapes": [{"sequences": 4, "length": 256}],
        "modes": ["full", "causal"],
        "check_rows": 8,
        "warmup": 1,
        "repeats": 3,
        "target_bytes_per_sample": 1048576,
        "max_runtime_seconds": 120,
    },
    "comparison": {"min_faster_ratio": 1.1},
}
# The canonical report of that run, byte for byte.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"arithmetic":"4 * 64 floa'
    "ting-point operations for each visible (query, key) pair: one multiply and o"
    "ne add for each of the 64 products of q k and of p v; the exp, max and sum a"
    're not counted","check_rule":"a timing point checks check_rows query rows of'
    " each sequence, spread evenly from the first row to the last; a correctness "
    'case checks every row","compulsory":"16 * 64 bytes for each token: read its '
    'rows of Q, K and V once and write its row of O once, as float32","device":"m'
    'ps","inputs":"normal","points":[{"check_rows":8,"checks":{"f1":{"differing_f'
    'rom_rounded_reference":1908,"max_abs_error":2.8002292645767923e-07,"max_boun'
    'd_ratio":0.0016196665494902255},"n0":{"differing_from_rounded_reference":190'
    '7,"max_abs_error":2.2144290939829503e-07,"max_bound_ratio":0.001210744142244'
    '3159},"torch_sdpa":{"differing_from_rounded_reference":1898,"max_abs_error":'
    '2.3564651263630054e-07,"max_bound_ratio":0.004351798445805665}},"compulsory_'
    'bytes_per_call":1048576,"index":0,"k_sha256":"6010bc814f5d578c3ade49a2e7e0cb'
    '1d89279a7ba98f7dc84744b7ef90ed2099","launches":{"f1":[{"active_threads":4096'
    ',"extent":[4096],"group_size":[128],"groups":[32],"kernel":"flash_f1","launc'
    'hed_threads":4096,"padding_threads":0,"threads":[4096],"threads_per_group":1'
    '28}],"n0":[{"active_threads":262144,"extent":[262144],"group_size":[256],"gr'
    'oups":[1024],"kernel":"flash_n0_scores","launched_threads":262144,"padding_t'
    'hreads":0,"threads":[262144],"threads_per_group":256},{"active_threads":2621'
    '44,"extent":[262144],"group_size":[256],"groups":[1024],"kernel":"flash_n0_s'
    'oftmax","launched_threads":262144,"padding_threads":0,"threads":[262144],"th'
    'reads_per_group":256},{"active_threads":65536,"extent":[65536],"group_size":'
    '[64],"groups":[1024],"kernel":"flash_n0_pv","launched_threads":65536,"paddin'
    'g_threads":0,"threads":[65536],"threads_per_group":64}]},"layout":{"block_gr'
    'oups":32,"buffer_rows":1024,"dense":true,"gap":0,"lengths_sha256":"33266c334'
    '860b0308e5587ed3dcb95344edbc942f6c26d921feeb92dc68bbfcf","longest":256,"quer'
    'y_groups":1024,"score_elements":262144,"sequences":4,"shortest":256,"tokens"'
    ':1024},"length":256,"memory":{"f1":{"allocated_current_bytes":1049600,"const'
    'ruction":"Q, K, V and O, 16 * 64 bytes for each buffer row; starts and lengt'
    "hs, 8 bytes for each sequence; the sequence and block of each group, 8 bytes"
    ' for each block of 32 queries; the K and V tiles are threadgroup memory","co'
    'nstruction_bytes":{"index":288,"scratch":0,"tensors":1048576},"known_constru'
    'ction_bytes":1048864,"peak_current_over_allocation_bytes":0,"peak_driver_ove'
    'r_before_bytes":8388608,"readings":{"after_allocation":{"current_bytes":1049'
    '600,"driver_bytes":8880128},"after_call":{"current_bytes":1049600,"driver_by'
    'tes":8880128},"before":{"current_bytes":0,"driver_bytes":491520},"peak_durin'
    'g_call":{"current_bytes":1049600,"driver_bytes":8880128},"polls":19018}},"n0'
    '":{"allocated_current_bytes":2106112,"construction":"Q, K, V and O, 16 * 64 '
    "bytes for each buffer row; the score buffer, 4 L * L bytes for each sequence"
    "; starts, lengths and score starts, 12 bytes for each sequence; the sequence"
    ' and row of each query group, 8 bytes for each token","construction_bytes":{'
    '"index":8240,"scratch":1048576,"tensors":1048576},"known_construction_bytes"'
    ':2105392,"peak_current_over_allocation_bytes":0,"peak_driver_over_before_byt'
    'es":8388608,"readings":{"after_allocation":{"current_bytes":2106112,"driver_'
    'bytes":8880128},"after_call":{"current_bytes":2106112,"driver_bytes":8880128'
    '},"before":{"current_bytes":0,"driver_bytes":491520},"peak_during_call":{"cu'
    'rrent_bytes":2106112,"driver_bytes":8880128},"polls":18476}},"torch_sdpa":{"'
    'allocated_current_bytes":786432,"construction":"Q, K and V, and the O that t'
    "he call returns, 16 * 64 bytes for each buffer row; any scratch of the call "
    'is not known by construction","construction_bytes":{"index":0,"scratch":null'
    ',"tensors":1048576},"known_construction_bytes":1048576,"peak_current_over_al'
    'location_bytes":262144,"peak_driver_over_before_bytes":8388608,"readings":{"'
    'after_allocation":{"current_bytes":786432,"driver_bytes":8880128},"after_cal'
    'l":{"current_bytes":1048576,"driver_bytes":8863744},"before":{"current_bytes'
    '":0,"driver_bytes":491520},"peak_during_call":{"current_bytes":1048576,"driv'
    'er_bytes":8880128},"polls":625981}}},"mode":"full","pairs":262144,"q_sha256"'
    ':"89339ee610aa49f05f46444c9c0a2c4047a2a7a79e43a0243cb51999ff692d4f","referen'
    'ce_sha256":"10b0aa55cbf016631bcfb3cefd036cc7210b7948dac780289b411a43c87aff9c'
    '","sequences":4,"side_order":["n0","f1","torch_sdpa"],"sides":{"f1":{"calls_'
    'per_sample":1,"compulsory_bytes_per_call":1048576,"effective_gb_per_second":'
    '1.114666825412349,"flops_per_call":67108864,"gflops_per_second":71.338676826'
    '39034,"implementation":"f1","iqr_effective_gb_per_second":[0.879677852348993'
    '3,1.2991494502090755],"iqr_gflops_per_second":[56.29938255033557,83.14556481'
    '338083],"iqr_ns_per_call":384875.0,"median_ns_per_call":940708.0,"modeled_by'
    'tes_per_call":4718592,"modeled_gb_per_second":5.01600071435557,"repeats":3,"'
    'samples_ns":[940708,807125,1192000],"timing":{"count":3,"iqr_ns":384875,"max'
    '_ns":1192000,"median_ns":940708,"min_ns":807125,"p25_ns":807125,"p75_ns":119'
    '2000},"warmup":1},"n0":{"calls_per_sample":1,"compulsory_bytes_per_call":104'
    '8576,"effective_gb_per_second":0.9296578823245703,"flops_per_call":67108864,'
    '"gflops_per_second":59.498104468772496,"implementation":"n0","iqr_effective_'
    'gb_per_second":[0.9235845342291645,1.0077616530514175],"iqr_gflops_per_secon'
    'd":[59.10941019066653,64.49674579529072],"iqr_ns_per_call":94833.0,"median_n'
    's_per_call":1127916.0,"modeled_bytes_per_call":5242880,"modeled_gb_per_secon'
    'd":4.6482894116228515,"repeats":3,"samples_ns":[1127916,1135333,1040500],"ti'
    'ming":{"count":3,"iqr_ns":94833,"max_ns":1135333,"median_ns":1127916,"min_ns'
    '":1040500,"p25_ns":1040500,"p75_ns":1135333},"warmup":1},"torch_sdpa":{"call'
    's_per_sample":1,"compulsory_bytes_per_call":1048576,"effective_gb_per_second'
    '":2.2753876667086197,"flops_per_call":67108864,"gflops_per_second":145.62481'
    '066935166,"implementation":"torch_sdpa","iqr_effective_gb_per_second":[1.866'
    '2086763070078,3.739345332130363],"iqr_gflops_per_second":[119.4373552836485,'
    '239.31810125634323],"iqr_ns_per_call":281458.0,"median_ns_per_call":460834.0'
    ',"modeled_bytes_per_call":null,"modeled_gb_per_second":null,"repeats":3,"sam'
    'ples_ns":[561875,460834,280417],"timing":{"count":3,"iqr_ns":281458,"max_ns"'
    ':561875,"median_ns":460834,"min_ns":280417,"p25_ns":280417,"p75_ns":561875},'
    '"warmup":1}},"v_sha256":"cdb7fdf46fd59f720497cf25b279faaa28621a546b28539c574'
    'abd1feb0cdc89"},{"check_rows":8,"checks":{"f1":{"differing_from_rounded_refe'
    'rence":1620,"max_abs_error":2.3961911499448263e-07,"max_bound_ratio":0.00135'
    '72903466319548},"n0":{"differing_from_rounded_reference":1630,"max_abs_error'
    '":4.151777421856906e-07,"max_bound_ratio":0.002061526202846968},"torch_sdpa"'
    ':{"differing_from_rounded_reference":1636,"max_abs_error":3.5557309741029997'
    'e-07,"max_bound_ratio":0.005972054064697406}},"compulsory_bytes_per_call":10'
    '48576,"index":1,"k_sha256":"81c33ff39143128e081864bd794470ec35568bc5c502e0da'
    '55b0648b55f0345d","launches":{"f1":[{"active_threads":4096,"extent":[4096],"'
    'group_size":[128],"groups":[32],"kernel":"flash_f1_causal","launched_threads'
    '":4096,"padding_threads":0,"threads":[4096],"threads_per_group":128}],"n0":['
    '{"active_threads":262144,"extent":[262144],"group_size":[256],"groups":[1024'
    '],"kernel":"flash_n0_scores_causal","launched_threads":262144,"padding_threa'
    'ds":0,"threads":[262144],"threads_per_group":256},{"active_threads":262144,"'
    'extent":[262144],"group_size":[256],"groups":[1024],"kernel":"flash_n0_softm'
    'ax_causal","launched_threads":262144,"padding_threads":0,"threads":[262144],'
    '"threads_per_group":256},{"active_threads":65536,"extent":[65536],"group_siz'
    'e":[64],"groups":[1024],"kernel":"flash_n0_pv_causal","launched_threads":655'
    '36,"padding_threads":0,"threads":[65536],"threads_per_group":64}]},"layout":'
    '{"block_groups":32,"buffer_rows":1024,"dense":true,"gap":0,"lengths_sha256":'
    '"33266c334860b0308e5587ed3dcb95344edbc942f6c26d921feeb92dc68bbfcf","longest"'
    ':256,"query_groups":1024,"score_elements":262144,"sequences":4,"shortest":25'
    '6,"tokens":1024},"length":256,"memory":{"f1":{"allocated_current_bytes":1049'
    '600,"construction":"Q, K, V and O, 16 * 64 bytes for each buffer row; starts'
    " and lengths, 8 bytes for each sequence; the sequence and block of each grou"
    "p, 8 bytes for each block of 32 queries; the K and V tiles are threadgroup m"
    'emory","construction_bytes":{"index":288,"scratch":0,"tensors":1048576},"kno'
    'wn_construction_bytes":1048864,"peak_current_over_allocation_bytes":0,"peak_'
    'driver_over_before_bytes":8388608,"readings":{"after_allocation":{"current_b'
    'ytes":1049600,"driver_bytes":8863744},"after_call":{"current_bytes":1049600,'
    '"driver_bytes":8863744},"before":{"current_bytes":0,"driver_bytes":475136},"'
    'peak_during_call":{"current_bytes":1049600,"driver_bytes":8863744},"polls":1'
    '8974}},"n0":{"allocated_current_bytes":2106112,"construction":"Q, K, V and O'
    ", 16 * 64 bytes for each buffer row; the score buffer, 4 L * L bytes for eac"
    "h sequence; starts, lengths and score starts, 12 bytes for each sequence; th"
    'e sequence and row of each query group, 8 bytes for each token","constructio'
    'n_bytes":{"index":8240,"scratch":1048576,"tensors":1048576},"known_construct'
    'ion_bytes":2105392,"peak_current_over_allocation_bytes":0,"peak_driver_over_'
    'before_bytes":8388608,"readings":{"after_allocation":{"current_bytes":210611'
    '2,"driver_bytes":8863744},"after_call":{"current_bytes":2106112,"driver_byte'
    's":8863744},"before":{"current_bytes":0,"driver_bytes":475136},"peak_during_'
    'call":{"current_bytes":2106112,"driver_bytes":8863744},"polls":20249}},"torc'
    'h_sdpa":{"allocated_current_bytes":786432,"construction":"Q, K and V, and th'
    "e O that the call returns, 16 * 64 bytes for each buffer row; any scratch of"
    ' the call is not known by construction","construction_bytes":{"index":0,"scr'
    'atch":null,"tensors":1048576},"known_construction_bytes":1048576,"peak_curre'
    'nt_over_allocation_bytes":262144,"peak_driver_over_before_bytes":8650752,"re'
    'adings":{"after_allocation":{"current_bytes":786432,"driver_bytes":8863744},'
    '"after_call":{"current_bytes":1048576,"driver_bytes":9125888},"before":{"cur'
    'rent_bytes":0,"driver_bytes":475136},"peak_during_call":{"current_bytes":104'
    '8576,"driver_bytes":9125888},"polls":29266}}},"mode":"causal","pairs":131584'
    ',"q_sha256":"f9270a3133d88711ca549919539bb0040a8acc0d6338119d059130ca418c4fc'
    '2","reference_sha256":"67b8e154bfe64e0ab8cbe746465935d59dee3186e747edbbb8cbf'
    'e8fd2c05001","sequences":4,"side_order":["n0","f1","torch_sdpa"],"sides":{"f'
    '1":{"calls_per_sample":1,"compulsory_bytes_per_call":1048576,"effective_gb_p'
    'er_second":1.294605399292309,"flops_per_call":33685504,"gflops_per_second":4'
    '1.58919845226543,"implementation":"f1","iqr_effective_gb_per_second":[0.8725'
    '103844924912,1.7061584846961273],"iqr_gflops_per_second":[28.02939610182128,'
    '54.81034132086309],"iqr_ns_per_call":587209.0,"median_ns_per_call":809958.0,'
    '"modeled_bytes_per_call":2883584,"modeled_gb_per_second":3.5601648480538497,'
    '"repeats":3,"samples_ns":[614583,1201792,809958],"timing":{"count":3,"iqr_ns'
    '":587209,"max_ns":1201792,"median_ns":809958,"min_ns":614583,"p25_ns":614583'
    ',"p75_ns":1201792},"warmup":1},"n0":{"calls_per_sample":1,"compulsory_bytes_'
    'per_call":1048576,"effective_gb_per_second":1.374581330382928,"flops_per_cal'
    'l":33685504,"gflops_per_second":44.15842523855156,"implementation":"n0","iqr'
    '_effective_gb_per_second":[1.0049847849526776,1.40567699521957],"iqr_gflops_'
    'per_second":[32.28513621660477,45.157373471428684],"iqr_ns_per_call":297417.'
    '0,"median_ns_per_call":762833.0,"modeled_bytes_per_call":3153920,"modeled_gb'
    '_per_second":4.134482907792401,"repeats":3,"samples_ns":[745958,762833,10433'
    '75],"timing":{"count":3,"iqr_ns":297417,"max_ns":1043375,"median_ns":762833,'
    '"min_ns":745958,"p25_ns":745958,"p75_ns":1043375},"warmup":1},"torch_sdpa":{'
    '"calls_per_sample":1,"compulsory_bytes_per_call":1048576,"effective_gb_per_s'
    'econd":1.868286859688196,"flops_per_call":33685504,"gflops_per_second":60.01'
    '8715367483296,"implementation":"torch_sdpa","iqr_effective_gb_per_second":[1'
    '.5610569457905135,2.93889997533577],"iqr_gflops_per_second":[50.148954383520'
    '24,94.41216170766161],"iqr_ns_per_call":314917.0,"median_ns_per_call":561250'
    '.0,"modeled_bytes_per_call":null,"modeled_gb_per_second":null,"repeats":3,"s'
    'amples_ns":[356792,671709,561250],"timing":{"count":3,"iqr_ns":314917,"max_n'
    's":671709,"median_ns":561250,"min_ns":356792,"p25_ns":356792,"p75_ns":671709'
    '},"warmup":1}},"v_sha256":"225d4426faa5ece6e3747823a5905779c03ca78afccd16ea9'
    'dfc0be4e7518615"}],"sides":["n0","f1","torch_sdpa"],"synchronization":"torch'
    ".mps.synchronize() before the first and before the second clock reading of e"
    "very sample; untimed warmups of each side first; repeat r starts with side r"
    ' mod S, so the order shifts by one side every repeat","torch_call":"torch.nn'
    ".functional.scaled_dot_product_attention(q, k, v, is_causal=causal) on 1 x H"
    " x N x 64 float32 views of the device rows, with the default scale 1 / sqrt("
    '64), on the MPS device","traffic_model":{"f1":"8 * 64 bytes for each token ('
    "read Q once, write O once) plus 2 * 32 * 64 * 4 bytes for each K/V tile that"
    ' a query block loads","n0":"16 * 64 bytes for each token (read Q, K and V on'
    "ce, write O once) plus 16 bytes for each visible pair: S written once, read "
    "and written by the softmax, and read once by P V; the model ignores the rere"
    'ads of K and V rows, which the cache can serve","torch_sdpa":"no model, beca'
    'use PyTorch does not name its kernel"},"vendor_bound":"|o - o_ref| <= 2**-14'
    " a + 2**-100 for each output value of torch.nn.functional.scaled_dot_product"
    "_attention: a sanity bound only, because PyTorch names no kernel and no deri"
    'ved bound applies"},"ceiling":{"buffer_bytes":268435456,"config_sha256":"b3f'
    '1096b3691a46d5a77068639ba57389194a3d07a06bfc16f5693cf52215199","definition":'
    '"the bandwidth ceiling of a verified Day 74 gpu-ceilings report: the better '
    "median GB/s of stream_copy and stream_read_reduce at its largest buffer, on "
    'the host that made it; not a vendor peak and not a hardware limit","environm'
    'ent":{"macos_version":"27.2.1","numpy_version":"1.26.4","platform_machine":"'
    'arm64","platform_system":"Darwin","python_implementation":"CPython","python_'
    'version":"3.11.14","torch_version":"2.15.0"},"gb_per_second":107.71090325448'
    '17,"kernel":"stream_read_reduce","label":"host-specific Apple GPU (Metal) me'
    'asurement","load_average_after":[10.21728515625,13.6328125,16.11767578125],"'
    'load_average_before":[11.72607421875,14.0302734375,16.28515625],"measured":t'
    'rue,"report_sha256":"cd75564918285983fe22b4106b5084b44ef42ed9313ca2a8ef19fa1'
    '64e68fd86","source":"gpu_ceilings"},"compute_ceiling":{"definition":"the FMA'
    " ceiling of the same verified Day 74 gpu-ceilings report: the best median GF"
    "LOP/s of its FMA chains, 2 operations for each fma, on the host that made it"
    '; not a vendor peak and not a hardware limit","fma_gflops_per_second":4665.4'
    '72950926219,"report_sha256":"cd75564918285983fe22b4106b5084b44ef42ed9313ca2a'
    '8ef19fa164e68fd86","source":"gpu_ceilings"},"config":{"benchmark":{"check_ro'
    'ws":8,"inputs":"normal","max_runtime_seconds":120,"modes":["full","causal"],'
    '"repeats":3,"shapes":[{"length":256,"sequences":4}],"target_bytes_per_sample'
    '":1048576,"warmup":1},"comparison":{"min_faster_ratio":1.1},"correctness":{"'
    'cases":[{"inputs":"normal","lengths":[1,31,33,100]},{"inputs":"wide_scores",'
    '"lengths":[1000,40]}],"repeats":2},"name":"metal-flash-recorded","plan_sha25'
    '6":"23a502f6a118c208a02fbc5ed7b36496edabe8788c9afc624e3b9b0bdfa0cbb5","schem'
    'a_version":1,"seed":81},"config_sha256":"bd8683ad2f7eed1dc98cc8db20acc85075c'
    'ee40177375e82446ac5e02d25cab5","correctness":{"bound":"|o - o_ref| <= a (2 E'
    " / (1 - E) (1 + u) + u + R) + 4 (L + 2 T + 2) 2**-126 vmax + 2**-125 for eac"
    "h output value, where a = sum_j P_j |v_jc|, vmax = max_j |v_jc| over the vis"
    "ible keys, E = (1 + A)(1 + gamma_(L + 2 T + 64)) - 1, A = exp(D) (1 + EXP_ER"
    "ROR)**(T + 1) (1 + u)**(2 T + 2) - 1, D = 2 (g + r) + 2 u (S + 2 g), g = gam"
    "ma_66 * 0.125 |q| max_j |k_j|, r = the same with the unit 2**-53, S = max - "
    "min of the visible float64 scores, T = ceil(L / 32), u = 2**-24, EXP_ERROR ="
    " 4 * 2**-23 and R = 4 gamma_(L + 70) for the unit 2**-53; one bound covers N"
    '0 and F1, because F1 has the more exp calls and roundings","cases":[{"case":'
    '{"inputs":"normal","lengths":[1,31,33,100]},"index":0,"k_sha256":"504ccb0e08'
    '6911e35daad396687c2d3a91f90faf5f210ff2089b14cfcd7f7806","layout":{"block_gro'
    'ups":8,"buffer_rows":177,"dense":false,"gap":3,"lengths_sha256":"f25090b8170'
    'e2d58e2d9db239a31d8bc38a452de97bebf7749c07564b264657c","longest":100,"query_'
    'groups":165,"score_elements":12051,"sequences":4,"shortest":1,"tokens":165},'
    '"modes":{"causal":{"bound_sha256":"8de60a129f6a9db2b268599bc445bb938a6c3315d'
    'b9019aaa87ce598a7085b87","max_score":3.5529218709306907,"outputs":{"f1":{"bu'
    'ffer_sha256":"50cd6b1bc8e9852f98daba5fadc85a16bed11a85866fbb715ae2d1a6134598'
    'bf","differing_from_rounded_reference":8444,"identical_repeats":true,"launch'
    'es":[{"active_threads":1024,"extent":[1024],"group_size":[128],"groups":[8],'
    '"kernel":"flash_f1_causal","launched_threads":1024,"padding_threads":0,"thre'
    'ads":[1024],"threads_per_group":128}],"max_abs_error":3.388921322455829e-07,'
    '"max_bound_ratio":0.0017079514153271601},"n0":{"buffer_sha256":"dc4f2450f226'
    '75f19aacde6fad1125fee9747dccf9721f37ca5348928511f839","differing_from_rounde'
    'd_reference":8699,"identical_repeats":true,"launches":[{"active_threads":422'
    '40,"extent":[42240],"group_size":[256],"groups":[165],"kernel":"flash_n0_sco'
    'res_causal","launched_threads":42240,"padding_threads":0,"threads":[42240],"'
    'threads_per_group":256},{"active_threads":42240,"extent":[42240],"group_size'
    '":[256],"groups":[165],"kernel":"flash_n0_softmax_causal","launched_threads"'
    ':42240,"padding_threads":0,"threads":[42240],"threads_per_group":256},{"acti'
    've_threads":10560,"extent":[10560],"group_size":[64],"groups":[165],"kernel"'
    ':"flash_n0_pv_causal","launched_threads":10560,"padding_threads":0,"threads"'
    ':[10560],"threads_per_group":64}],"max_abs_error":5.167251837523423e-07,"max'
    '_bound_ratio":0.00243267127427785}},"reference_sha256":"937c078f6544d94ed632'
    '85ea4fa91f05faaefc8f8b8ef945388b6904e244a7ab","unscaled_exp_overflows":false'
    '},"full":{"bound_sha256":"3665d50ef844b96997afb35875799362b52112023e2ea42999'
    'f2698c598b5764","max_score":3.925690800614327,"outputs":{"f1":{"buffer_sha25'
    '6":"b119764072d204adbbc97b44e07941be2d158fcf7e72e5d6dc69808ad971aee1","diffe'
    'ring_from_rounded_reference":9223,"identical_repeats":true,"launches":[{"act'
    'ive_threads":1024,"extent":[1024],"group_size":[128],"groups":[8],"kernel":"'
    'flash_f1","launched_threads":1024,"padding_threads":0,"threads":[1024],"thre'
    'ads_per_group":128}],"max_abs_error":2.7175217021113696e-07,"max_bound_ratio'
    '":0.0017491475716906026},"n0":{"buffer_sha256":"389e17532769bf488baf05670d60'
    'c505efe4d10a2442627ad21205fef9d7121f","differing_from_rounded_reference":944'
    '6,"identical_repeats":true,"launches":[{"active_threads":42240,"extent":[422'
    '40],"group_size":[256],"groups":[165],"kernel":"flash_n0_scores","launched_t'
    'hreads":42240,"padding_threads":0,"threads":[42240],"threads_per_group":256}'
    ',{"active_threads":42240,"extent":[42240],"group_size":[256],"groups":[165],'
    '"kernel":"flash_n0_softmax","launched_threads":42240,"padding_threads":0,"th'
    'reads":[42240],"threads_per_group":256},{"active_threads":10560,"extent":[10'
    '560],"group_size":[64],"groups":[165],"kernel":"flash_n0_pv","launched_threa'
    'ds":10560,"padding_threads":0,"threads":[10560],"threads_per_group":64}],"ma'
    'x_abs_error":7.199866381424336e-07,"max_bound_ratio":0.004055841862126337}},'
    '"reference_sha256":"187f2d0d939cdf582f962ee68ea551c7806363c0c54c9c9e631fa47e'
    '96fcfd38","unscaled_exp_overflows":false}},"q_sha256":"823f099293037b3d49c61'
    '42dc7c82d673d23cc9a7b038deb33889010ce20213a","repeats":2,"v_sha256":"55e2d3d'
    '43dd9b09680da15e5a608f2654f815e18fd2a8805a879b1c3a3fd358c"},{"case":{"inputs'
    '":"wide_scores","lengths":[1000,40]},"index":1,"k_sha256":"c33ebd72f36cb6f56'
    '284085b461014bf61b4abb0fb54ba6dae6421a795f5c93d","layout":{"block_groups":34'
    ',"buffer_rows":1046,"dense":false,"gap":3,"lengths_sha256":"2faf4874566cff5b'
    '11e265c2f5c7de98307b299feed9d734f54f90b479248e99","longest":1000,"query_grou'
    'ps":1040,"score_elements":1001600,"sequences":2,"shortest":40,"tokens":1040}'
    ',"modes":{"causal":{"bound_sha256":"9eb9869aff0a21989f23c1187f600078d8a9fc62'
    'f4b3f492f282d0845ca74e69","max_score":141.03400706124827,"outputs":{"f1":{"b'
    'uffer_sha256":"342ac517770a1d30edb9540cd65eb35f95db2e50bdf59e7803cd3a8e8a032'
    'b78","differing_from_rounded_reference":43188,"identical_repeats":true,"laun'
    'ches":[{"active_threads":4352,"extent":[4352],"group_size":[128],"groups":[3'
    '4],"kernel":"flash_f1_causal","launched_threads":4352,"padding_threads":0,"t'
    'hreads":[4352],"threads_per_group":128}],"max_abs_error":1.7794819305638754e'
    '-05,"max_bound_ratio":0.0026859725775266324},"n0":{"buffer_sha256":"86f30cba'
    '2018f1ade4bd2954bc4d5e9c1ea3de61b69dd4e494151afece6f1ad1","differing_from_ro'
    'unded_reference":44541,"identical_repeats":true,"launches":[{"active_threads'
    '":266240,"extent":[266240],"group_size":[256],"groups":[1040],"kernel":"flas'
    'h_n0_scores_causal","launched_threads":266240,"padding_threads":0,"threads":'
    '[266240],"threads_per_group":256},{"active_threads":266240,"extent":[266240]'
    ',"group_size":[256],"groups":[1040],"kernel":"flash_n0_softmax_causal","laun'
    'ched_threads":266240,"padding_threads":0,"threads":[266240],"threads_per_gro'
    'up":256},{"active_threads":66560,"extent":[66560],"group_size":[64],"groups"'
    ':[1040],"kernel":"flash_n0_pv_causal","launched_threads":66560,"padding_thre'
    'ads":0,"threads":[66560],"threads_per_group":64}],"max_abs_error":3.15780488'
    '37355155e-05,"max_bound_ratio":0.006668579607580067}},"reference_sha256":"e1'
    '658fc6dc1c8ba158f9b4d29dc1d7c421e3f86b33ce79257e567efb0e497188","unscaled_ex'
    'p_overflows":true},"full":{"bound_sha256":"af73d55b4e5acaedbd920eb8a94224a6c'
    '98fc4e0a357511c3d4a051240ba673a","max_score":141.03400706124827,"outputs":{"'
    'f1":{"buffer_sha256":"ed81d4614d720e02a928e2d8f13178f74b5d1fee96f79f6bceddd2'
    '2a0bc3810a","differing_from_rounded_reference":45694,"identical_repeats":tru'
    'e,"launches":[{"active_threads":4352,"extent":[4352],"group_size":[128],"gro'
    'ups":[34],"kernel":"flash_f1","launched_threads":4352,"padding_threads":0,"t'
    'hreads":[4352],"threads_per_group":128}],"max_abs_error":2.1445801517860907e'
    '-05,"max_bound_ratio":0.002833720472196918},"n0":{"buffer_sha256":"aaf942dd7'
    '53ed619d7dde2e5c4f76cb18b41ab130f2e216480171b797cedcca1","differing_from_rou'
    'nded_reference":46887,"identical_repeats":true,"launches":[{"active_threads"'
    ':266240,"extent":[266240],"group_size":[256],"groups":[1040],"kernel":"flash'
    '_n0_scores","launched_threads":266240,"padding_threads":0,"threads":[266240]'
    ',"threads_per_group":256},{"active_threads":266240,"extent":[266240],"group_'
    'size":[256],"groups":[1040],"kernel":"flash_n0_softmax","launched_threads":2'
    '66240,"padding_threads":0,"threads":[266240],"threads_per_group":256},{"acti'
    've_threads":66560,"extent":[66560],"group_size":[64],"groups":[1040],"kernel'
    '":"flash_n0_pv","launched_threads":66560,"padding_threads":0,"threads":[6656'
    '0],"threads_per_group":64}],"max_abs_error":3.206074999484443e-05,"max_bound'
    '_ratio":0.007619722286007599}},"reference_sha256":"524e9551f610f043632cefb1a'
    'edd974a4910eaaa14e702d842fcb4240d976bde","unscaled_exp_overflows":true}},"q_'
    'sha256":"8ea85f9ab397487ef4578b93ad4e1e6637f0b2e4f904dab2215c82a5ac2e95f6","'
    'repeats":2,"v_sha256":"0aac3ce4a425565b29d8783905816602524ccc27745812bd2eead'
    '7b6a632b658"}],"distributions":{"normal":"q, k and v from a standard normal,'
    ' drawn as float32","offset_values":"q and k from a standard normal, drawn as'
    ' float32, and v from 1000 plus a standard normal, rounded to float32","wide_'
    'scores":"q and k from 5 times a standard normal and v from a standard normal'
    ", drawn in float64 and rounded to float32; the scores reach past 88.7, where"
    ' a float32 exp with no max subtraction overflows"},"gap":3,"gap_input":1.267'
    '6506002282294e+30,"reference":"float64 attention of the float32 inputs: s = '
    "0.125 * sum_c q_c k_c in column order, the max over the visible keys, w = ex"
    "p64(s - max), and o = sum_j w_j v_j / sum_j w_j with the keys added in order"
    '; a causal query i sees the keys j <= i","rules":"every gap row keeps its se'
    "ntinel in every repeat; every output is finite; every value is within its bo"
    "und of the reference; identical_repeats and differing_from_rounded_reference"
    ' are measured","sentinel_float32_bits":2143289344},"environment":{"macos_ver'
    'sion":"27.2.1","numpy_version":"1.26.4","platform_machine":"arm64","platform'
    '_system":"Darwin","python_implementation":"CPython","python_version":"3.11.1'
    '4","torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_host'
    '":true,"backward_pass_measured":false,"cuda_kernel_executed":false,"distribu'
    'ted_claim":false,"hardware_counters_used":false,"host_specific":true,"nvidia'
    '_gpu_used":false,"occupancy_measured":false,"portable_performance_claim":fal'
    'se,"production_claim":false,"ptx_executed":false,"statement":"These are wall'
    "-clock timings and allocator readings on the Apple GPU of the host that made"
    " this report, at the recorded load averages. N0 and F1 came from torch.mps.c"
    "ompile_shader, and the PyTorch call ran on the MPS device. PyTorch does not "
    "report which kernels it ran, so its numbers name the call, not a kernel. Bot"
    "h ceilings come from a separate gpu-ceilings run with its own load averages."
    " No hardware counter was read, and each byte and operation count is a model."
    " No CUDA kernel ran, no NVIDIA GPU was used, no PTX ran, and no Triton kerne"
    "l ran on a GPU. Only forward passes were timed. These numbers do not measure"
    ' or predict CUDA, NVIDIA, Triton, distributed or production performance.","t'
    'riton_gpu_kernel_executed":false,"vendor_kernel_known":false,"vendor_peak_cl'
    'aimed":false},"exp_probe":{"arguments":"precise::exp at 2**18 float32 argume'
    "nts: 262136 evenly spaced from -104 to 0, then the extremes -3.4028234663852"
    '886e+38, -1e+30, -1000.0, -200.0, -150.0, -105.0, -87.5, -0.0","arguments_sh'
    'a256":"ef74306503b0bf88378fecc220238ffe4510b9ce6992098cb47e1e39804aab8d","co'
    'unt":262144,"exp_error":4.76837158203125e-07,"exp_ulps":4,"holds":true,"max_'
    'relative_error":1.61439084446152e-07,"max_ulps":1.4061690587550402,"normal_r'
    'esults":220136,"results_sha256":"918b6099a06c5fc92292905d92d8968e8f6b44a6cc1'
    'e883b02864e0cc4467498","rule":"each result whose float64 exp is at least 2**'
    "-126 has a relative error of at most EXP_ERROR = 4 * 2**-23; each smaller re"
    'sult is within 2**-126 of it","tiny_max_abs_error":1.1751589259924035e-38,"t'
    'iny_nonzero":0,"tiny_results":42008},"host_load":{"after":[9.89501953125,13.'
    '44775390625,16.0224609375],"before":[9.89501953125,13.44775390625,16.0224609'
    '375],"source":"os.getloadavg(): the 1, 5 and 15 minute load averages that up'
    'time prints, read before the checks and after the last sample"},"kind":"meta'
    'l_flash_attention","label":"host-specific Apple GPU (Metal) measurement","me'
    'asured":true,"measurement_scope":"apple_gpu_metal_this_host","memory_method"'
    ':{"cannot_show":"a buffer that lives and dies between two polls; threadgroup'
    " memory, registers and other on-chip storage; memory that Metal or the opera"
    "ting system holds outside these two counters; which kernel PyTorch ran; the "
    'driver counter grows in large heaps, so it is coarse","construction":{"f1":"'
    "Q, K, V and O, 16 * 64 bytes for each buffer row; starts and lengths, 8 byte"
    "s for each sequence; the sequence and block of each group, 8 bytes for each "
    'block of 32 queries; the K and V tiles are threadgroup memory","n0":"Q, K, V'
    " and O, 16 * 64 bytes for each buffer row; the score buffer, 4 L * L bytes f"
    "or each sequence; starts, lengths and score starts, 12 bytes for each sequen"
    'ce; the sequence and row of each query group, 8 bytes for each token","torch'
    '_sdpa":"Q, K and V, and the O that the call returns, 16 * 64 bytes for each '
    'buffer row; any scratch of the call is not known by construction"},"measures'
    '":"torch.mps.current_allocated_memory() is the bytes of live tensors in the '
    "PyTorch MPS allocator, rounded up to its blocks; torch.mps.driver_allocated_"
    "memory() is the bytes that the Metal driver holds for this process, with the"
    " allocator cache and the buffers of the MPS frameworks; each side runs alone"
    " after torch.mps.empty_cache(), and a second thread polls both counters whil"
    'e its call runs"},"percentile_method":"nearest rank: sorted samples[ceil(p *'
    " n / 100) - 1] for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; n"
    'o interpolation","plan":{"kernels":{"flash_f1":"bd4c160e839abc38c5469b97c11e'
    'f44f5233917c3b0051f32638e4451fb86876","flash_f1_causal":"2ebbcb676b46632e910'
    'ce44c754d73d8baca1e1ed0b092429e37dd644def7b65","flash_n0_pv":"4064d4e534c132'
    '4ab559ff4de1c3155f9763d8a334060e99fc191eb143bdd822","flash_n0_pv_causal":"13'
    'd7a9f80f34c66ce735800f6327c8526a50514eacb950fcf4118ddfcf916c6c","flash_n0_sc'
    'ores":"70d5a1b87e7d12d44fca9c154d6b9d2203fa076cdde717a7a4cb842012c6a68c","fl'
    'ash_n0_scores_causal":"e29642489cc8b6dcad4dc3d774de4af23fa800712a104207a73e2'
    'c43c6880cf8","flash_n0_softmax":"6b4312452989d65df14605d9785b3eb05261c21613c'
    '1ce08349b1bbeff3a3d0d","flash_n0_softmax_causal":"62dca87a332165243c3513eaaf'
    '0e2bd535167c895080a087a630d963f774e2a4"},"sha256":"23a502f6a118c208a02fbc5ed'
    '7b36496edabe8788c9afc624e3b9b0bdfa0cbb5","sides":["n0","f1","torch_sdpa"],"s'
    'teps":[{"base":"torch_sdpa","change":"Compare the naive three-kernel attenti'
    "on with the PyTorch call. N0 writes the N x N scores, reads and writes them "
    "in the softmax and reads them again for P V, and it uses no SIMD-group matri"
    'x unit.","first":"n0","hypothesis":"N0 moves 16 bytes for each visible pair '
    "through device memory and computes each score with scalar fma. N0 is slower "
    'than the PyTorch call at every point.","predicted":["slower"],"step":"n0_ove'
    'r_torch"},{"base":"n0","change":"Fuse the three steps into one kernel. Load '
    "K and V in tiles of 32 rows into threadgroup memory and keep a running max a"
    'nd sum, so S and P never go to device memory.","first":"f1","hypothesis":"F1'
    " does the same arithmetic as N0 but removes the N x N traffic and two launch"
    'es. F1 is faster than N0 by 1.1 or more at every point.","predicted":["faste'
    'r"],"step":"f1_over_n0"},{"base":"torch_sdpa","change":"Compare the flash ke'
    "rnel with the PyTorch call. F1 uses scalar fma for both products, and four t"
    'hreads share one query row.","first":"f1","hypothesis":"A scratch prototype '
    "of F1 ran before this plan was written, at 512, 2048 and 4096 tokens, and to"
    "ok about 2.5 to 3 times the time of the PyTorch call. Scalar fma probably li"
    'mits F1. F1 is slower than the PyTorch call at every point.","predicted":["s'
    'lower"],"step":"f1_over_torch"}],"torch_call":"torch.nn.functional.scaled_do'
    "t_product_attention(q, k, v, is_causal=causal) on 1 x H x N x 64 float32 vie"
    "ws of the device rows, with the default scale 1 / sqrt(64), on the MPS devic"
    'e"},"planned_work":{"benchmark_points":2,"calls":24,"compulsory_bytes":25165'
    '824,"correctness_cases":2,"correctness_launches":16,"correctness_pairs":1013'
    '651,"sides_per_point":3,"timed_samples":18},"report_sha256":"9a454eaab41e0c6'
    'f86706d103a87c5e7d992486621140f5b27cfd1af804befe9","run":{"elapsed_ns":14475'
    '04042,"max_runtime_ns":120000000000,"timed_ns":14314958},"schema_version":1,'
    '"shaders":{"admissions":[{"admitted":true,"allowed_threads":1024,"declared_t'
    'hreadgroup_bytes":256,"kernel":"flash_n0_scores","reason":null,"required_thr'
    'eads":256,"static_threadgroup_bytes":256,"thread_execution_width":32},{"admi'
    'tted":true,"allowed_threads":1024,"declared_threadgroup_bytes":64,"kernel":"'
    'flash_n0_softmax","reason":null,"required_threads":256,"static_threadgroup_b'
    'ytes":64,"thread_execution_width":32},{"admitted":true,"allowed_threads":102'
    '4,"declared_threadgroup_bytes":0,"kernel":"flash_n0_pv","reason":null,"requi'
    'red_threads":64,"static_threadgroup_bytes":0,"thread_execution_width":32},{"'
    'admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":256,"kern'
    'el":"flash_n0_scores_causal","reason":null,"required_threads":256,"static_th'
    'readgroup_bytes":256,"thread_execution_width":32},{"admitted":true,"allowed_'
    'threads":1024,"declared_threadgroup_bytes":64,"kernel":"flash_n0_softmax_cau'
    'sal","reason":null,"required_threads":256,"static_threadgroup_bytes":64,"thr'
    'ead_execution_width":32},{"admitted":true,"allowed_threads":1024,"declared_t'
    'hreadgroup_bytes":0,"kernel":"flash_n0_pv_causal","reason":null,"required_th'
    'reads":64,"static_threadgroup_bytes":0,"thread_execution_width":32},{"admitt'
    'ed":true,"allowed_threads":1024,"declared_threadgroup_bytes":16384,"kernel":'
    '"flash_f1","reason":null,"required_threads":128,"static_threadgroup_bytes":1'
    '6384,"thread_execution_width":32},{"admitted":true,"allowed_threads":1024,"d'
    'eclared_threadgroup_bytes":16384,"kernel":"flash_f1_causal","reason":null,"r'
    'equired_threads":128,"static_threadgroup_bytes":16384,"thread_execution_widt'
    'h":32},{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":'
    '0,"kernel":"flash_probe_exp","reason":null,"required_threads":256,"static_th'
    'readgroup_bytes":0,"thread_execution_width":32}],"kernels":[{"barriers":1,"b'
    'uffers":["s","q","k","starts","lengths","score_starts","seqs","rows"],"group'
    '_size":256,"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_'
    'memory_bytes":256,"thread_execution_width":32},"mode":"full","name":"flash_n'
    '0_scores","source_sha256":"70d5a1b87e7d12d44fca9c154d6b9d2203fa076cdde717a7a'
    '4cb842012c6a68c","summary":"one query row per group; writes its row of the N'
    ' x N scores","threadgroup_bytes":256,"variant":"n0"},{"barriers":2,"buffers"'
    ':["s","starts","lengths","score_starts","seqs","rows"],"group_size":256,"lim'
    'its":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":6'
    '4,"thread_execution_width":32},"mode":"full","name":"flash_n0_softmax","sour'
    'ce_sha256":"6b4312452989d65df14605d9785b3eb05261c21613c1ce08349b1bbeff3a3d0d'
    '","summary":"reads and writes one row of the N x N buffer twice in place","t'
    'hreadgroup_bytes":64,"variant":"n0"},{"barriers":0,"buffers":["o","s","v","s'
    'tarts","lengths","score_starts","seqs","rows"],"group_size":64,"limits":{"ma'
    'x_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_'
    'execution_width":32},"mode":"full","name":"flash_n0_pv","source_sha256":"406'
    '4d4e534c1324ab559ff4de1c3155f9763d8a334060e99fc191eb143bdd822","summary":"re'
    'ads one row of P and the V rows; writes one row of O","threadgroup_bytes":0,'
    '"variant":"n0"},{"barriers":1,"buffers":["s","q","k","starts","lengths","sco'
    're_starts","seqs","rows"],"group_size":256,"limits":{"max_threads_per_thread'
    'group":1024,"static_threadgroup_memory_bytes":256,"thread_execution_width":3'
    '2},"mode":"causal","name":"flash_n0_scores_causal","source_sha256":"e2964248'
    '9cc8b6dcad4dc3d774de4af23fa800712a104207a73e2c43c6880cf8","summary":"one que'
    'ry row per group; writes its row of the N x N scores","threadgroup_bytes":25'
    '6,"variant":"n0"},{"barriers":2,"buffers":["s","starts","lengths","score_sta'
    'rts","seqs","rows"],"group_size":256,"limits":{"max_threads_per_threadgroup"'
    ':1024,"static_threadgroup_memory_bytes":64,"thread_execution_width":32},"mod'
    'e":"causal","name":"flash_n0_softmax_causal","source_sha256":"62dca87a332165'
    '243c3513eaaf0e2bd535167c895080a087a630d963f774e2a4","summary":"reads and wri'
    'tes one row of the N x N buffer twice in place","threadgroup_bytes":64,"vari'
    'ant":"n0"},{"barriers":0,"buffers":["o","s","v","starts","lengths","score_st'
    'arts","seqs","rows"],"group_size":64,"limits":{"max_threads_per_threadgroup"'
    ':1024,"static_threadgroup_memory_bytes":0,"thread_execution_width":32},"mode'
    '":"causal","name":"flash_n0_pv_causal","source_sha256":"13d7a9f80f34c66ce735'
    '800f6327c8526a50514eacb950fcf4118ddfcf916c6c","summary":"reads one row of P '
    'and the V rows; writes one row of O","threadgroup_bytes":0,"variant":"n0"},{'
    '"barriers":2,"buffers":["o","q","k","v","starts","lengths","seqs","blocks"],'
    '"group_size":128,"limits":{"max_threads_per_threadgroup":1024,"static_thread'
    'group_memory_bytes":16384,"thread_execution_width":32},"mode":"full","name":'
    '"flash_f1","source_sha256":"bd4c160e839abc38c5469b97c11ef44f5233917c3b0051f3'
    '2638e4451fb86876","summary":"one block of 32 queries per group; K/V tiles in'
    ' threadgroup memory; online softmax; no N x N buffer","threadgroup_bytes":16'
    '384,"variant":"f1"},{"barriers":2,"buffers":["o","q","k","v","starts","lengt'
    'hs","seqs","blocks"],"group_size":128,"limits":{"max_threads_per_threadgroup'
    '":1024,"static_threadgroup_memory_bytes":16384,"thread_execution_width":32},'
    '"mode":"causal","name":"flash_f1_causal","source_sha256":"2ebbcb676b46632e91'
    '0ce44c754d73d8baca1e1ed0b092429e37dd644def7b65","summary":"one block of 32 q'
    "ueries per group; K/V tiles in threadgroup memory; online softmax; no N x N "
    'buffer","threadgroup_bytes":16384,"variant":"f1"},{"barriers":0,"buffers":["'
    'y","x"],"group_size":256,"limits":{"max_threads_per_threadgroup":1024,"stati'
    'c_threadgroup_memory_bytes":0,"thread_execution_width":32},"mode":null,"name'
    '":"flash_probe_exp","source_sha256":"b3824d5577bbfaec7ad150fa794a2d4011250ec'
    '0cf8115c5974100063a57051d","summary":"precise::exp at each argument","thread'
    'group_bytes":0,"variant":null}],"library_sha256":"2787a67f6dc7cbabddea66fad2'
    '81db77a73292e856094ad9fc9ab75a77229bc4"},"speedup":{"bandwidth_gb_per_second'
    '":107.7109032544817,"fma_gflops_per_second":4665.472950926219,"fraction_rule'
    '":"effective GB/s of a side / the bandwidth ceiling, modeled GB/s / the band'
    "width ceiling, and GFLOP/s / the FMA ceiling; both ceilings come from one se"
    'parate Day 74 run on this host","hypotheses":[{"held_at":[0,1],"missed_at":['
    '],"predicted":["slower"],"step":"n0_over_torch"},{"held_at":[],"missed_at":['
    '0,1],"predicted":["faster"],"step":"f1_over_n0"},{"held_at":[0],"missed_at":'
    '[1],"predicted":["slower"],"step":"f1_over_torch"}],"hypothesis_rule":"a pre'
    "diction holds at a point when the verdict of the first side of a step over i"
    'ts base is one of the verdicts that the step predicted","min_faster_ratio":1'
    '.1,"points":[{"index":0,"length":256,"mode":"full","rates":{"f1":{"effective'
    '_fraction_of_bandwidth":0.010348690724269544,"effective_gb_per_second":1.114'
    '666825412349,"fraction_of_fma":0.015290770641426126,"gflops_per_second":71.3'
    '3867682639034,"iqr_effective_gb_per_second":[0.8796778523489933,1.2991494502'
    '090755],"iqr_gflops_per_second":[56.29938255033557,83.14556481338083],"media'
    'n_ns_per_call":940708.0,"modeled_fraction_of_bandwidth":0.04656910825921294,'
    '"modeled_gb_per_second":5.01600071435557},"n0":{"effective_fraction_of_bandw'
    'idth":0.008631047129259762,"effective_gb_per_second":0.9296578823245703,"fra'
    'ction_of_fma":0.012752855947211218,"gflops_per_second":59.498104468772496,"i'
    'qr_effective_gb_per_second":[0.9235845342291645,1.0077616530514175],"iqr_gfl'
    'ops_per_second":[59.10941019066653,64.49674579529072],"median_ns_per_call":1'
    '127916.0,"modeled_fraction_of_bandwidth":0.04315523564629881,"modeled_gb_per'
    '_second":4.6482894116228515},"torch_sdpa":{"effective_fraction_of_bandwidth"'
    ':0.021124952051815086,"effective_gb_per_second":2.2753876667086197,"fraction'
    '_of_fma":0.031213300816681687,"gflops_per_second":145.62481066935166,"iqr_ef'
    'fective_gb_per_second":[1.8662086763070078,3.739345332130363],"iqr_gflops_pe'
    'r_second":[119.4373552836485,239.31810125634323],"median_ns_per_call":460834'
    '.0,"modeled_fraction_of_bandwidth":null,"modeled_gb_per_second":null}},"sequ'
    'ences":4,"steps":[{"first":"n0","held":true,"iqr_separated":true,"predicted"'
    ':["slower"],"ratio":0.4085712056571589,"second":"torch_sdpa","step":"n0_over'
    '_torch","verdict":"slower"},{"first":"f1","held":false,"iqr_separated":false'
    ',"predicted":["faster"],"ratio":1.1990075560110045,"second":"n0","step":"f1_'
    'over_n0","verdict":"no_clear_difference"},{"first":"f1","held":true,"iqr_sep'
    'arated":true,"predicted":["slower"],"ratio":0.48987996275145956,"second":"to'
    'rch_sdpa","step":"f1_over_torch","verdict":"slower"}]},{"index":1,"length":2'
    '56,"mode":"causal","rates":{"f1":{"effective_fraction_of_bandwidth":0.012019'
    '260447882672,"effective_gb_per_second":1.294605399292309,"fraction_of_fma":0'
    '.008914251328798055,"gflops_per_second":41.58919845226543,"iqr_effective_gb_'
    'per_second":[0.8725103844924912,1.7061584846961273],"iqr_gflops_per_second":'
    '[28.02939610182128,54.81034132086309],"median_ns_per_call":809958.0,"modeled'
    '_fraction_of_bandwidth":0.03305296623167735,"modeled_gb_per_second":3.560164'
    '8480538497},"n0":{"effective_fraction_of_bandwidth":0.012761765883025713,"ef'
    'fective_gb_per_second":1.374581330382928,"fraction_of_fma":0.009464940790147'
    '536,"gflops_per_second":44.15842523855156,"iqr_effective_gb_per_second":[1.0'
    '049847849526776,1.40567699521957],"iqr_gflops_per_second":[32.28513621660477'
    ',45.157373471428684],"median_ns_per_call":762833.0,"modeled_fraction_of_band'
    'width":0.03838499894503828,"modeled_gb_per_second":4.134482907792401},"torch'
    '_sdpa":{"effective_fraction_of_bandwidth":0.017345382902175775,"effective_gb'
    '_per_second":1.868286859688196,"fraction_of_fma":0.012864443969301764,"gflop'
    's_per_second":60.018715367483296,"iqr_effective_gb_per_second":[1.5610569457'
    '905135,2.93889997533577],"iqr_gflops_per_second":[50.14895438352024,94.41216'
    '170766161],"median_ns_per_call":561250.0,"modeled_fraction_of_bandwidth":nul'
    'l,"modeled_gb_per_second":null}},"sequences":4,"steps":[{"first":"n0","held"'
    ':true,"iqr_separated":true,"predicted":["slower"],"ratio":0.7357442585729773'
    ',"second":"torch_sdpa","step":"n0_over_torch","verdict":"slower"},{"first":"'
    'f1","held":false,"iqr_separated":false,"predicted":["faster"],"ratio":0.9418'
    '179708083629,"second":"n0","step":"f1_over_n0","verdict":"no_clear_differenc'
    'e"},{"first":"f1","held":false,"iqr_separated":false,"predicted":["slower"],'
    '"ratio":0.692937164643105,"second":"torch_sdpa","step":"f1_over_torch","verd'
    'ict":"no_clear_difference"}]}],"ratio_rule":"ratio = median sample time of t'
    "he second side / median sample time of the first side, from the same point o"
    "f the same run with the same calls per sample; faster when ratio >= min_fast"
    "er_ratio and the first side's p75 sample time is below the second side's p25"
    "; slower when 1 / ratio >= min_faster_ratio and the second side's p75 is bel"
    'ow the first side\'s p25; otherwise no_clear_difference","vendor":"torch.nn.f'
    "unctional.scaled_dot_product_attention(q, k, v, is_causal=causal) on 1 x H x"
    " N x 64 float32 views of the device rows, with the default scale 1 / sqrt(64"
    '), on the MPS device, PyTorch 2.15.0"},"summary":{"bandwidth_gb_per_second":'
    '107.7109032544817,"benchmark_points":2,"correctness_cases":2,"fma_gflops_per'
    '_second":4665.472950926219,"hypotheses":{"f1_over_n0":{"held":0,"missed":2},'
    '"f1_over_torch":{"held":1,"missed":1},"n0_over_torch":{"held":2,"missed":0}}'
    ',"identical_repeats":{"f1":true,"n0":true},"label":"host-specific Apple GPU '
    '(Metal) measurement","load_average_after":[9.89501953125,13.44775390625,16.0'
    '224609375],"load_average_before":[9.89501953125,13.44775390625,16.0224609375'
    '],"max_bound_ratio":{"f1":0.002833720472196918,"n0":0.007619722286007599},"p'
    'oints":[{"gflops_per_second":{"f1":71.33867682639034,"n0":59.498104468772496'
    ',"torch_sdpa":145.62481066935166},"index":0,"length":256,"median_ms_per_call'
    '":{"f1":0.940708,"n0":1.127916,"torch_sdpa":0.460834},"mode":"full","peak_dr'
    'iver_over_before_bytes":{"f1":8388608,"n0":8388608,"torch_sdpa":8388608},"se'
    'quences":4,"verdicts":{"f1_over_n0":"no_clear_difference","f1_over_torch":"s'
    'lower","n0_over_torch":"slower"}},{"gflops_per_second":{"f1":41.589198452265'
    '43,"n0":44.15842523855156,"torch_sdpa":60.018715367483296},"index":1,"length'
    '":256,"median_ms_per_call":{"f1":0.809958,"n0":0.762833,"torch_sdpa":0.56125'
    '},"mode":"causal","peak_driver_over_before_bytes":{"f1":8388608,"n0":8388608'
    ',"torch_sdpa":8650752},"sequences":4,"verdicts":{"f1_over_n0":"no_clear_diff'
    'erence","f1_over_torch":"no_clear_difference","n0_over_torch":"slower"}}],"p'
    'robe_max_ulps":1.4061690587550402,"timed_samples":18,"unscaled_exp_overflow_'
    'cases":[1],"warnings":0},"timer":"time.perf_counter_ns","torch_mps_switches"'
    ':{"source":"os.environ, read before the runtime imports torch; null means un'
    'set, and a value that is not a short token is \'unknown\'","values":{"PYTORCH_'
    'MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}},"verification":{"gpu_o'
    'nly":["the samples_ns of each side: checked for form only, never observed ag'
    'ain","each kernel output digest, max_abs_error and differing count, unless e'
    "very value equals the rounded reference: max_bound_ratio is checked to be at"
    ' most 1, not recomputed","identical_repeats: the CPU cannot rerun the GPU","'
    'the exp probe results digest and measured error: checked against the rule","'
    'the pipeline limits: checked for form, then used to rebuild the admissions",'
    '"every allocator reading and poll count: checked for form only","the PyTorch'
    " side checks, the warnings, the MPS switches and the load averages: checked "
    'for form only"],"recomputed":["the config, its digest and the planned work",'
    '"the declared plan, every kernel digest and the library digest","every input'
    ' digest, reference digest and bound digest, from the seed","each layout and '
    'launch, and each admission from the recorded limits","the exp probe argument'
    's and the declared probe assumption","each sample statistic, GB/s and GFLOP/'
    "s rate, model byte count, ratio, verdict, ceiling fraction and held hypothes"
    'is, from the raw samples","the bytes of each side by construction and each m'
    'emory difference, from the recorded readings","the run totals, the summary a'
    'nd the SHA-256 seal"]},"warnings":{"capture":"warnings.catch_warnings(record'
    "=True) with simplefilter('always'), from before the runtime imports torch to"
    ' the end of the run; equal warnings are kept once with a count","count":0,"d'
    'istinct":0,"records":[],"torch_imported_before_capture":false}}'
)


def recorded():
    return json.loads(RECORDED_REPORT_JSON)


def test_the_recorded_report_verifies_as_a_measurement_on_any_host():
    config = config_from_mapping(RECORDED_CONFIG)

    summary = verify_report(recorded(), config=config)

    assert canonical_json_bytes(recorded()).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert summary["config_sha256"] == config.fingerprint()
    assert summary["ceiling_report_sha256"] == CEILING_SHA256
    assert summary["load_average_before"] == [
        9.89501953125,
        13.44775390625,
        16.0224609375,
    ]
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (2, 2)


def test_the_recorded_report_verifies_where_libm_rounds_otherwise(monkeypatch):
    real_exp, real_log = numpy.exp, numpy.log

    def nudge(function):
        def other(values, *args, **kwargs):
            result = function(values, *args, **kwargs)
            if isinstance(result, numpy.ndarray) and result.dtype == numpy.float64:
                up = numpy.nextafter(result, numpy.inf)
                result = numpy.where(result.view(numpy.int64) % 3 == 0, up, result)
            return result

        return other

    monkeypatch.setattr(numpy, "exp", nudge(real_exp))
    monkeypatch.setattr(numpy, "log", nudge(real_log))
    assert verify_report(recorded())["measured"] is True


def test_the_report_seals_the_plan_that_was_declared_before_the_run():
    report = recorded()

    assert report["plan"] == {**plan_mapping(), "sha256": plan_sha256()}
    assert report["config"]["plan_sha256"] == plan_sha256()
    assert report["ceiling"]["environment"] == report["environment"]
    assert report["compute_ceiling"]["report_sha256"] == CEILING_SHA256


def test_the_fresh_process_captured_the_warnings_from_before_the_torch_import():
    report = recorded()

    assert report["warnings"]["torch_imported_before_capture"] is False
    assert report["warnings"]["count"] == 0
    assert report["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }


def test_the_apple_gpu_kept_its_bounds_bits_and_probe_assumption():
    report = recorded()
    summary = report["summary"]

    assert summary["identical_repeats"] == {"n0": True, "f1": True}
    assert max(summary["max_bound_ratio"].values()) <= 1
    assert summary["unscaled_exp_overflow_cases"] == [1]
    assert report["exp_probe"]["holds"] is True
    for entry in report["shaders"]["kernels"]:
        assert entry["limits"]["thread_execution_width"] == 32
        assert entry["limits"]["static_threadgroup_memory_bytes"] == (
            KERNELS[entry["name"]].threadgroup_bytes
        )


def test_the_memory_record_sets_each_reading_beside_the_design():
    for point in recorded()["benchmark"]["points"]:
        memory = point["memory"]
        assert set(memory) == set(SIDES)
        scores = 4 * point["sequences"] * point["length"] ** 2
        assert memory["n0"]["construction_bytes"]["scratch"] == scores
        assert memory["f1"]["construction_bytes"]["scratch"] == 0
        assert memory["torch_sdpa"]["construction_bytes"]["scratch"] is None
        for entry in memory.values():
            assert entry["readings"]["polls"] >= 1


@pytest.mark.parametrize(
    "path",
    [
        ("summary", "points", 0, "verdicts", "f1_over_n0"),
        ("speedup", "points", 1, "steps", 0, "held"),
        ("environment", "macos_version"),
    ],
)
def test_a_recorded_field_changed_and_resealed_is_rejected(path):
    report = copy.deepcopy(recorded())
    target = report
    for key in path[:-1]:
        target = target[key]
    value = target[path[-1]]
    target[path[-1]] = "27.3" if isinstance(value, str) else not value
    report.pop("report_sha256")

    with pytest.raises(MetalFlashVerificationError):
        verify_report(seal_report(report))


def test_the_command_verifies_the_recorded_file_without_torch(tmp_path, capsys):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")

    assert main(["verify", "--report", str(path)]) == EXIT_OK
    expected = capsys.readouterr().out
    assert expected.startswith(
        f"verified_metal_flash_measurement={recorded()['report_sha256']} "
    )
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_flash_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, "verify", "--report", str(path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout == expected
    path.write_text(RECORDED_REPORT_JSON.replace('"polls":', '"polls": ') + "\n")
    assert main(["verify", "--report", str(path)]) == EXIT_REJECTED


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    runtime = flash_runtime()

    for entry in recorded()["shaders"]["kernels"]:
        assert runtime.limits[entry["name"]].to_mapping() == entry["limits"]
