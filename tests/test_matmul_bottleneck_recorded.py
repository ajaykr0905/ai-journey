"""The bottleneck report of the development host's Apple GPU, replayed anywhere.

``RECORDED_TIMING_JSON`` is a one-call timing input in the declared schema. A
scratch capture wrote it on 2026-11-18 at 08:15 IST under the GPU lock. It
timed one call of each kernel with a ``torch.mps.Event`` pair and
``time.perf_counter_ns``, 11 times, at the nine pinned Day 74 shapes and at
129 x 65 x 1000. ``RECORDED_REPORT_JSON`` is the canonical file that
``matmul_bottleneck_cli classify`` wrote from that input and from three Day 74
reports that the Day 74 commands made on the same host minutes before. Like
the Day 74 labs, the repository keeps none of those source reports, so any
host can check the arithmetic of this report but cannot rebuild it from its
sources. The guide quotes its values.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from ai_journey.matmul_bottleneck_cli import EXIT_OK
from ai_journey.matmul_bottleneck_config import load_config
from ai_journey.matmul_bottleneck_io import load_report
from ai_journey.matmul_bottleneck_report import build_report
from ai_journey.matmul_bottleneck_timing import timing_from_mapping
from ai_journey.matmul_bottleneck_verify import verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD
from test_matmul_bottleneck_sources import recorded_sources

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config/day-75-matmul-bottleneck.json")
REPORT_SHA256 = "bad8d9707745344c26a16676a757e1bb6c60c880afd3a10557cbdc34aa0ec2f5"
TIMING_SHA256 = "59c6abd676437622cc6335046e62a947d4000484f94c1164f9b51c6c09fbf9d4"
GAP_SHA256 = "94ca519c6033bb5dedbfca46da9111f4bdfaf5e1816d1453629aced833598341"
MATMUL_SHA256 = "c9ce115a7ffed725e5ea0a0d7898709ba5b77e697467728949c64989a5416637"
CEILING_SHA256 = "9697e2556bcb54cf7d4a842fc99f3601d7d434f50ffc35c2c95b0534ef7ca57a"
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_bottleneck_cli import main
sys.exit(main(sys.argv[1:]))
"""
RECORDED_TIMING_JSON = (
    '{"environment":{"macos_version":"27.2","numpy_version":"1.26.4","platform_ma'
    'chine":"arm64","platform_system":"Darwin","python_implementation":"CPython",'
    '"python_version":"3.11.14","torch_version":"2.15.0"},"kind":"matmul_gpu_timi'
    'ng","label":"host-specific Apple GPU (Metal) measurement","load_average_afte'
    'r":[17.61181640625,11.85595703125,9.853515625],"load_average_before":[10.508'
    '30078125,10.19384765625,9.22900390625],"measured":true,"points":[{"k":128,"k'
    'ernel":"matmul_x_columns","m":128,"median_dispatch_ns":89833,"median_gpu_ns"'
    ':249125,"median_host_ns":460125,"n":128},{"k":128,"kernel":"matmul_x_rows","'
    'm":128,"median_dispatch_ns":79584,"median_gpu_ns":247458,"median_host_ns":53'
    '3833,"n":128},{"k":128,"kernel":"torch_matmul","m":128,"median_dispatch_ns":'
    '312333,"median_gpu_ns":638208,"median_host_ns":972625,"n":128},{"k":256,"ker'
    'nel":"matmul_x_columns","m":256,"median_dispatch_ns":113458,"median_gpu_ns":'
    '552708,"median_host_ns":789583,"n":256},{"k":256,"kernel":"matmul_x_rows","m'
    '":256,"median_dispatch_ns":90417,"median_gpu_ns":1132875,"median_host_ns":16'
    '16458,"n":256},{"k":256,"kernel":"torch_matmul","m":256,"median_dispatch_ns"'
    ':296541,"median_gpu_ns":473333,"median_host_ns":811250,"n":256},{"k":512,"ke'
    'rnel":"matmul_x_columns","m":512,"median_dispatch_ns":112000,"median_gpu_ns"'
    ':1315750,"median_host_ns":1550750,"n":512},{"k":512,"kernel":"matmul_x_rows"'
    ',"m":512,"median_dispatch_ns":122458,"median_gpu_ns":3215583,"median_host_ns'
    '":3647916,"n":512},{"k":512,"kernel":"torch_matmul","m":512,"median_dispatch'
    '_ns":329333,"median_gpu_ns":519792,"median_host_ns":793958,"n":512},{"k":102'
    '4,"kernel":"matmul_x_columns","m":1024,"median_dispatch_ns":122167,"median_g'
    'pu_ns":5648875,"median_host_ns":5937709,"n":1024},{"k":1024,"kernel":"matmul'
    '_x_rows","m":1024,"median_dispatch_ns":141084,"median_gpu_ns":14069875,"medi'
    'an_host_ns":14437958,"n":1024},{"k":1024,"kernel":"torch_matmul","m":1024,"m'
    'edian_dispatch_ns":319833,"median_gpu_ns":919000,"median_host_ns":1176208,"n'
    '":1024},{"k":2048,"kernel":"matmul_x_columns","m":2048,"median_dispatch_ns":'
    '76208,"median_gpu_ns":42522333,"median_host_ns":42741916,"n":2048},{"k":2048'
    ',"kernel":"matmul_x_rows","m":2048,"median_dispatch_ns":70833,"median_gpu_ns'
    '":112765000,"median_host_ns":112956667,"n":2048},{"k":2048,"kernel":"torch_m'
    'atmul","m":2048,"median_dispatch_ns":199750,"median_gpu_ns":4346792,"median_'
    'host_ns":4594083,"n":2048},{"k":4096,"kernel":"matmul_x_columns","m":4096,"m'
    'edian_dispatch_ns":150125,"median_gpu_ns":467651000,"median_host_ns":4678778'
    '33,"n":4096},{"k":4096,"kernel":"matmul_x_rows","m":4096,"median_dispatch_ns'
    '":141417,"median_gpu_ns":888684375,"median_host_ns":889160750,"n":4096},{"k"'
    ':4096,"kernel":"torch_matmul","m":4096,"median_dispatch_ns":328416,"median_g'
    'pu_ns":38197542,"median_host_ns":38540875,"n":4096},{"k":1024,"kernel":"matm'
    'ul_x_columns","m":4096,"median_dispatch_ns":98833,"median_gpu_ns":21492750,"'
    'median_host_ns":21972542,"n":1024},{"k":1024,"kernel":"matmul_x_rows","m":40'
    '96,"median_dispatch_ns":125084,"median_gpu_ns":56914250,"median_host_ns":571'
    '57834,"n":1024},{"k":1024,"kernel":"torch_matmul","m":4096,"median_dispatch_'
    'ns":383333,"median_gpu_ns":2525250,"median_host_ns":2981500,"n":1024},{"k":2'
    '048,"kernel":"matmul_x_columns","m":1024,"median_dispatch_ns":116375,"median'
    '_gpu_ns":44983708,"median_host_ns":45316958,"n":4096},{"k":2048,"kernel":"ma'
    'tmul_x_rows","m":1024,"median_dispatch_ns":123708,"median_gpu_ns":109513375,'
    '"median_host_ns":109726333,"n":4096},{"k":2048,"kernel":"torch_matmul","m":1'
    '024,"median_dispatch_ns":338584,"median_gpu_ns":4651167,"median_host_ns":504'
    '6667,"n":4096},{"k":1000,"kernel":"matmul_x_columns","m":2000,"median_dispat'
    'ch_ns":76708,"median_gpu_ns":29945458,"median_host_ns":30188167,"n":3000},{"'
    'k":1000,"kernel":"matmul_x_rows","m":2000,"median_dispatch_ns":74625,"median'
    '_gpu_ns":100014500,"median_host_ns":100211333,"n":3000},{"k":1000,"kernel":"'
    'torch_matmul","m":2000,"median_dispatch_ns":206791,"median_gpu_ns":3226542,"'
    'median_host_ns":3543333,"n":3000},{"k":1000,"kernel":"matmul_x_columns","m":'
    '129,"median_dispatch_ns":43333,"median_gpu_ns":309125,"median_host_ns":53483'
    '3,"n":65},{"k":1000,"kernel":"matmul_x_rows","m":129,"median_dispatch_ns":92'
    '667,"median_gpu_ns":380125,"median_host_ns":571125,"n":65},{"k":1000,"kernel'
    '":"torch_matmul","m":129,"median_dispatch_ns":208542,"median_gpu_ns":296417,'
    '"median_host_ns":479208,"n":65}],"producer":"scratch capture with torch.mps.'
    'Event pairs and time.perf_counter_ns, one call per sample","repeats":11,"sch'
    'ema_version":1}'
)
RECORDED_REPORT_JSON = (
    '{"ceilings":{"bandwidth_gb_per_second":102.60967650780991,"fma_gflops_per_se'
    'cond":4757.315844849298,"fma_scales_with_work":true},"config":{"mapping":{"s'
    'lower_ratio_min":1.1},"name":"day-75-matmul-bottleneck","rule":{"dispatch_ov'
    'erhead_share_min":0.5,"gpu_agreement_max_ratio":1.5,"intensity_model":"ideal'
    '_reuse","near_roof_share_min":0.7,"require_fma_scales_with_work":true,"roof_'
    'excess_tolerance":0.05,"timing_tolerance":0.05},"schema_version":1},"config_'
    'sha256":"efb4a04127d0c4aff4137f086bf9e4b4b8c8f2b131682621dbee3ac2ced199ee","'
    'definitions":{"bandwidth_roof_gflops_per_second":"the measured bandwidth cei'
    "ling in GB/s times the intensity of the config's traffic model in FLOPs per "
    'byte","dispatch_share":"median_dispatch_ns / median_host_ns","fixed_share_bo'
    'und":"the smallest median_host_ns of the kernel over the whole timing input,'
    " divided by median_host_ns: the largest share that a fixed one-call cost can"
    ' take if it does not depend on the shape","gpu_excess_share":"(median_gpu_ns'
    ' - steady_ns_per_call) / median_host_ns","gpu_to_steady_ratio":"median_gpu_n'
    's / steady_ns_per_call","host_outside_gpu_share":"1 - median_gpu_ns / median'
    '_host_ns","lower_roof_share":"the rate divided by the lower of the FMA ceili'
    'ng and the bandwidth roof","model_sector_ratio":"32-byte sectors of the x-ov'
    "er-rows probe warp divided by those of the x-over-columns probe warp, in the"
    ' Day 74 per-warp model","overhead_share":"1 - steady_ns_per_call / median_ho'
    'st_ns","rate_gflops_per_second":"flops_per_sample divided by median_sample_n'
    's","steady_ns_per_call":"the Day 74 median sample time divided by its calls '
    "per sample. With many calls per sample it approaches the time per call of ba"
    'ck-to-back calls; with one call it also holds that call\'s synchronization","'
    'threadgroups":"the threadgroups of each kernel\'s launch, x by y","top_bottle'
    'neck":"the name that most points of a kernel get; a tie between names gives '
    'indeterminate"},"environment":{"macos_version":"27.2","numpy_version":"1.26.'
    '4","platform_machine":"arm64","platform_system":"Darwin","python_implementat'
    'ion":"CPython","python_version":"3.11.14","torch_float32_matmul_precision":"'
    'highest","torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_thi'
    's_host":true,"cuda_kernel_executed":false,"distributed_claim":false,"hardwar'
    'e_counters_used":false,"host_specific":true,"inference_from_timings":true,"i'
    'nstruments_used":false,"nsight_used":false,"nvidia_gpu_used":false,"portable'
    '_performance_claim":false,"production_claim":false,"statement":"Each bottlen'
    "eck name is an inference from host-specific Apple GPU (Metal) timings and fr"
    "om the ceilings that the Day 74 ceiling lab measured on the same host. No ha"
    "rdware counters, Nsight or Instruments were used, so no stall, cache hit or "
    "occupancy was observed. The mapping explanation comes from the Day 74 per-wa"
    "rp access model, not from measured traffic. No CUDA kernel ran, and no NVIDI"
    "A GPU was used. These names do not describe CUDA, NVIDIA, distributed or pro"
    'duction performance.","traffic_measured":false,"vendor_peak_used":false},"ge'
    'ometry":{"group_size":[32,8],"simd_widths":{"matmul_x_columns":32,"matmul_x_'
    'rows":32}},"kind":"matmul_bottleneck","label":"host-specific Apple GPU (Meta'
    'l) measurement","measured":true,"measurement_scope":"apple_gpu_metal_this_ho'
    'st","model_scope":"The Day 74 per-warp model of the first warp of threadgrou'
    "p (0, 0): lane addresses from the kernels' index formulas, 32-byte sectors p"
    "er request, no cache. It assumes that a SIMD group holds 32 consecutive x-th"
    'reads. It predicts the direction of a difference, not its size.","points":[{'
    '"calls_per_sample":256,"flops_per_call":4194304,"index":0,"intensity":{"flop'
    's_per_byte":{"decimal":"21.333333","denominator":3,"numerator":64},"model":"'
    'ideal_reuse"},"k":128,"kernels":{"matmul_x_columns":{"decision":{"bottleneck'
    '":"dispatch-bound","deciding":{"limit":0.5,"quantity":"overhead_share","valu'
    'e":0.9563754329665851},"explanation":"most of the one-call host time lies ou'
    'tside the steady-state time per call","numbers":{"bandwidth_roof_gflops_per_'
    'second":2189.0064321666114,"bandwidth_roof_share":0.09545658797343648,"dispa'
    'tch_ns":89833.0,"dispatch_share":0.19523607715294758,"fixed_share_bound":1.0'
    ',"fma_ceiling_gflops_per_second":4757.315844849298,"fma_ceiling_share":0.043'
    '92289515373763,"gpu_excess_share":0.4978043924884542,"gpu_ns":249125.0,"gpu_'
    'to_steady_ratio":12.41110219173417,"host_ns":460125.0,"host_outside_gpu_shar'
    'e":0.4585710404781309,"lower_roof_share":0.09545658797343648,"overhead_share'
    '":0.9563754329665851,"rate_gflops_per_second":208.95508506653044,"steady_ns_'
    'per_call":20072.75390625},"reason":"one_call_overhead_dominates"},"inputs":{'
    '"median_dispatch_ns":89833,"median_gpu_ns":249125,"median_host_ns":460125},"'
    'mapping":"x_columns"},"matmul_x_rows":{"decision":{"bottleneck":"dispatch-bo'
    'und","deciding":{"limit":0.5,"quantity":"overhead_share","value":0.941061082'
    '6559992},"explanation":"most of the one-call host time lies outside the stea'
    'dy-state time per call","numbers":{"bandwidth_roof_gflops_per_second":2189.0'
    '064321666114,"bandwidth_roof_share":0.06089831774216336,"dispatch_ns":79584.'
    '0,"dispatch_share":0.14908033036548884,"fixed_share_bound":1.0,"fma_ceiling_'
    'gflops_per_second":4757.315844849298,"fma_ceiling_share":0.02802143342869524'
    '4,"gpu_excess_share":0.4046105447536964,"gpu_ns":247458.0,"gpu_to_steady_rat'
    'io":7.864913082677792,"host_ns":533833.0,"host_outside_gpu_share":0.53645053'
    '79023028,"lower_roof_share":0.06089831774216336,"overhead_share":0.941061082'
    '6559992,"rate_gflops_per_second":133.30680924572167,"steady_ns_per_call":314'
    '63.5390625},"reason":"one_call_overhead_dominates"},"inputs":{"median_dispat'
    'ch_ns":79584,"median_gpu_ns":247458,"median_host_ns":533833},"mapping":"x_ro'
    'ws"}},"m":128,"mapping":{"model_line_ratio":16.560311284046694,"model_predic'
    'ts_x_rows_worse":true,"model_sector_ratio":6.608695652173913,"probe_warp":{"'
    'block":[0,0,0],"warp":0},"timing_shows_x_rows_slower":true,"verdict":"x_rows'
    '_worse_as_modeled","warps":{"x_columns":{"active_lanes":32,"lines":257,"mapp'
    'ing":"x_columns","operands":{"a":{"classes":{"broadcast":128,"coalesced":0,"'
    'single":0,"strided":0},"first_request":{"active_lanes":32,"class":"broadcast'
    '","distinct_addresses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines'
    '":128,"requested_bytes":16384,"requests":128,"sectors":128,"transferred_byte'
    's":4096,"unique_bytes":512},"b":{"classes":{"broadcast":0,"coalesced":128,"s'
    'ingle":0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesced"'
    ',"distinct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"lines'
    '":128,"requested_bytes":16384,"requests":128,"sectors":512,"transferred_byte'
    's":16384,"unique_bytes":16384},"c":{"classes":{"broadcast":0,"coalesced":1,"'
    'single":0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesced'
    '","distinct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"line'
    's":1,"requested_bytes":128,"requests":1,"sectors":4,"transferred_bytes":128,'
    '"unique_bytes":128}},"sectors":644},"x_rows":{"active_lanes":32,"lines":4256'
    ',"mapping":"x_rows","operands":{"a":{"classes":{"broadcast":0,"coalesced":0,'
    '"single":0,"strided":128},"first_request":{"active_lanes":32,"class":"stride'
    'd","distinct_addresses":32,"lane_stride_bytes":512,"lines":32,"sectors":32},'
    '"lines":4096,"requested_bytes":16384,"requests":128,"sectors":4096,"transfer'
    'red_bytes":131072,"unique_bytes":16384},"b":{"classes":{"broadcast":128,"coa'
    'lesced":0,"single":0,"strided":0},"first_request":{"active_lanes":32,"class"'
    ':"broadcast","distinct_addresses":1,"lane_stride_bytes":0,"lines":1,"sectors'
    '":1},"lines":128,"requested_bytes":16384,"requests":128,"sectors":128,"trans'
    'ferred_bytes":4096,"unique_bytes":512},"c":{"classes":{"broadcast":0,"coales'
    'ced":0,"single":0,"strided":1},"first_request":{"active_lanes":32,"class":"s'
    'trided","distinct_addresses":32,"lane_stride_bytes":512,"lines":32,"sectors"'
    ':32},"lines":32,"requested_bytes":128,"requests":1,"sectors":32,"transferred'
    '_bytes":1024,"unique_bytes":128}},"sectors":4256}},"x_columns_to_x_rows_rate'
    '_ratio":1.5674749568221071},"median_sample_ns":{"matmul_x_columns":5138625,"'
    'matmul_x_rows":8054666},"n":128,"threadgroups":{"matmul_x_columns":[4,16],"m'
    'atmul_x_rows":[4,16]}},{"calls_per_sample":32,"flops_per_call":33554432,"ind'
    'ex":1,"intensity":{"flops_per_byte":{"decimal":"42.666667","denominator":3,"'
    'numerator":128},"model":"ideal_reuse"},"k":256,"kernels":{"matmul_x_columns"'
    ':{"decision":{"bottleneck":"dispatch-bound","deciding":{"limit":0.5,"quantit'
    'y":"overhead_share","value":0.857620335670854},"explanation":"most of the on'
    'e-call host time lies outside the steady-state time per call","numbers":{"ba'
    'ndwidth_roof_gflops_per_second":4378.012864333223,"bandwidth_roof_share":0.0'
    '6817530731074553,"dispatch_ns":113458.0,"dispatch_share":0.1436935698970216,'
    '"fixed_share_bound":0.582744309337967,"fma_ceiling_gflops_per_second":4757.3'
    '15844849298,"fma_ceiling_share":0.06273965870049769,"gpu_excess_share":0.557'
    '6202090217242,"gpu_ns":552708.0,"gpu_to_steady_ratio":4.916431546942313,"hos'
    't_ns":789583.0,"host_outside_gpu_share":0.30000012664913,"lower_roof_share":'
    '0.06817530731074553,"overhead_share":0.857620335670854,"rate_gflops_per_seco'
    'nd":298.47237243631474,"steady_ns_per_call":112420.5625},"reason":"one_call_'
    'overhead_dominates"},"inputs":{"median_dispatch_ns":113458,"median_gpu_ns":5'
    '52708,"median_host_ns":789583},"mapping":"x_columns"},"matmul_x_rows":{"deci'
    'sion":{"bottleneck":"dispatch-bound","deciding":{"limit":0.5,"quantity":"ove'
    'rhead_share","value":0.8300690846901064},"explanation":"most of the one-call'
    ' host time lies outside the steady-state time per call","numbers":{"bandwidt'
    'h_roof_gflops_per_second":4378.012864333223,"bandwidth_roof_share":0.0279020'
    '45116427177,"dispatch_ns":90417.0,"dispatch_share":0.0559352609223376,"fixed'
    '_share_bound":0.3302486052838985,"fma_ceiling_gflops_per_second":4757.315844'
    '849298,"fma_ceiling_share":0.025677402225286517,"gpu_excess_share":0.5309069'
    '660331416,"gpu_ns":1132875.0,"gpu_to_steady_ratio":4.1242517882338,"host_ns"'
    ':1616458.0,"host_outside_gpu_share":0.29916211865696485,"lower_roof_share":0'
    '.027902045116427177,"overhead_share":0.8300690846901064,"rate_gflops_per_sec'
    'ond":122.15551246092416,"steady_ns_per_call":274686.1875},"reason":"one_call'
    '_overhead_dominates"},"inputs":{"median_dispatch_ns":90417,"median_gpu_ns":1'
    '132875,"median_host_ns":1616458},"mapping":"x_rows"}},"m":256,"mapping":{"mo'
    'del_line_ratio":16.530214424951268,"model_predicts_x_rows_worse":true,"model'
    '_sector_ratio":6.604361370716511,"probe_warp":{"block":[0,0,0],"warp":0},"ti'
    'ming_shows_x_rows_slower":true,"verdict":"x_rows_worse_as_modeled","warps":{'
    '"x_columns":{"active_lanes":32,"lines":513,"mapping":"x_columns","operands":'
    '{"a":{"classes":{"broadcast":256,"coalesced":0,"single":0,"strided":0},"firs'
    't_request":{"active_lanes":32,"class":"broadcast","distinct_addresses":1,"la'
    'ne_stride_bytes":0,"lines":1,"sectors":1},"lines":256,"requested_bytes":3276'
    '8,"requests":256,"sectors":256,"transferred_bytes":8192,"unique_bytes":1024}'
    ',"b":{"classes":{"broadcast":0,"coalesced":256,"single":0,"strided":0},"firs'
    't_request":{"active_lanes":32,"class":"coalesced","distinct_addresses":32,"l'
    'ane_stride_bytes":4,"lines":1,"sectors":4},"lines":256,"requested_bytes":327'
    '68,"requests":256,"sectors":1024,"transferred_bytes":32768,"unique_bytes":32'
    '768},"c":{"classes":{"broadcast":0,"coalesced":1,"single":0,"strided":0},"fi'
    'rst_request":{"active_lanes":32,"class":"coalesced","distinct_addresses":32,'
    '"lane_stride_bytes":4,"lines":1,"sectors":4},"lines":1,"requested_bytes":128'
    ',"requests":1,"sectors":4,"transferred_bytes":128,"unique_bytes":128}},"sect'
    'ors":1284},"x_rows":{"active_lanes":32,"lines":8480,"mapping":"x_rows","oper'
    'ands":{"a":{"classes":{"broadcast":0,"coalesced":0,"single":0,"strided":256}'
    ',"first_request":{"active_lanes":32,"class":"strided","distinct_addresses":3'
    '2,"lane_stride_bytes":1024,"lines":32,"sectors":32},"lines":8192,"requested_'
    'bytes":32768,"requests":256,"sectors":8192,"transferred_bytes":262144,"uniqu'
    'e_bytes":32768},"b":{"classes":{"broadcast":256,"coalesced":0,"single":0,"st'
    'rided":0},"first_request":{"active_lanes":32,"class":"broadcast","distinct_a'
    'ddresses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":256,"reques'
    'ted_bytes":32768,"requests":256,"sectors":256,"transferred_bytes":8192,"uniq'
    'ue_bytes":1024},"c":{"classes":{"broadcast":0,"coalesced":0,"single":0,"stri'
    'ded":1},"first_request":{"active_lanes":32,"class":"strided","distinct_addre'
    'sses":32,"lane_stride_bytes":1024,"lines":32,"sectors":32},"lines":32,"reque'
    'sted_bytes":128,"requests":1,"sectors":32,"transferred_bytes":1024,"unique_b'
    'ytes":128}},"sectors":8480}},"x_columns_to_x_rows_rate_ratio":2.443380297977'
    '0713},"median_sample_ns":{"matmul_x_columns":3597458,"matmul_x_rows":8789958'
    '},"n":256,"threadgroups":{"matmul_x_columns":[8,32],"matmul_x_rows":[8,32]}}'
    ',{"calls_per_sample":4,"flops_per_call":268435456,"index":2,"intensity":{"fl'
    'ops_per_byte":{"decimal":"85.333333","denominator":3,"numerator":256},"model'
    '":"ideal_reuse"},"k":512,"kernels":{"matmul_x_columns":{"decision":{"bottlen'
    'eck":"dispatch-bound","deciding":{"limit":0.5,"quantity":"overhead_share","v'
    'alue":0.5154630017733355},"explanation":"most of the one-call host time lies'
    ' outside the steady-state time per call","numbers":{"bandwidth_roof_gflops_p'
    'er_second":8756.025728666446,"bandwidth_roof_share":0.04080037129027879,"dis'
    'patch_ns":112000.0,"dispatch_share":0.07222311784620344,"fixed_share_bound":'
    '0.2967112687409318,"fma_ceiling_gflops_per_second":4757.315844849298,"fma_ce'
    'iling_share":0.07509467784099624,"gpu_excess_share":0.36392342414960505,"gpu'
    '_ns":1315750.0,"gpu_to_steady_ratio":1.7510745835333776,"host_ns":1550750.0,'
    '"host_outside_gpu_share":0.1515395776237305,"lower_roof_share":0.07509467784'
    '099624,"overhead_share":0.5154630017733355,"rate_gflops_per_second":357.2491'
    '007568249,"steady_ns_per_call":751395.75},"reason":"one_call_overhead_domina'
    'tes"},"inputs":{"median_dispatch_ns":112000,"median_gpu_ns":1315750,"median_'
    'host_ns":1550750},"mapping":"x_columns"},"matmul_x_rows":{"decision":{"bottl'
    'eneck":"dispatch-bound","deciding":{"limit":0.5,"quantity":"overhead_share",'
    '"value":0.5462677868678993},"explanation":"most of the one-call host time li'
    'es outside the steady-state time per call","numbers":{"bandwidth_roof_gflops'
    '_per_second":8756.025728666446,"bandwidth_roof_share":0.018522022470066648,"'
    'dispatch_ns":122458.0,"dispatch_share":0.03356930367914174,"fixed_share_boun'
    'd":0.14633917009053937,"fma_ceiling_gflops_per_second":4757.315844849298,"fm'
    'a_ceiling_share":0.03409050619803426,"gpu_excess_share":0.4277527223762828,"'
    'gpu_ns":3215583.0,"gpu_to_steady_ratio":1.9427426794838256,"host_ns":3647916'
    '.0,"host_outside_gpu_share":0.11851506449161653,"lower_roof_share":0.0340905'
    '0619803426,"overhead_share":0.5462677868678993,"rate_gflops_per_second":162.'
    '17930529484158,"steady_ns_per_call":1655177.0},"reason":"one_call_overhead_d'
    'ominates"},"inputs":{"median_dispatch_ns":122458,"median_gpu_ns":3215583,"me'
    'dian_host_ns":3647916},"mapping":"x_rows"}},"m":512,"mapping":{"model_line_r'
    'atio":16.515121951219513,"model_predicts_x_rows_worse":true,"model_sector_ra'
    'tio":6.6021840873634945,"probe_warp":{"block":[0,0,0],"warp":0},"timing_show'
    's_x_rows_slower":true,"verdict":"x_rows_worse_as_modeled","warps":{"x_column'
    's":{"active_lanes":32,"lines":1025,"mapping":"x_columns","operands":{"a":{"c'
    'lasses":{"broadcast":512,"coalesced":0,"single":0,"strided":0},"first_reques'
    't":{"active_lanes":32,"class":"broadcast","distinct_addresses":1,"lane_strid'
    'e_bytes":0,"lines":1,"sectors":1},"lines":512,"requested_bytes":65536,"reque'
    'sts":512,"sectors":512,"transferred_bytes":16384,"unique_bytes":2048},"b":{"'
    'classes":{"broadcast":0,"coalesced":512,"single":0,"strided":0},"first_reque'
    'st":{"active_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_str'
    'ide_bytes":4,"lines":1,"sectors":4},"lines":512,"requested_bytes":65536,"req'
    'uests":512,"sectors":2048,"transferred_bytes":65536,"unique_bytes":65536},"c'
    '":{"classes":{"broadcast":0,"coalesced":1,"single":0,"strided":0},"first_req'
    'uest":{"active_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_s'
    'tride_bytes":4,"lines":1,"sectors":4},"lines":1,"requested_bytes":128,"reque'
    'sts":1,"sectors":4,"transferred_bytes":128,"unique_bytes":128}},"sectors":25'
    '64},"x_rows":{"active_lanes":32,"lines":16928,"mapping":"x_rows","operands":'
    '{"a":{"classes":{"broadcast":0,"coalesced":0,"single":0,"strided":512},"firs'
    't_request":{"active_lanes":32,"class":"strided","distinct_addresses":32,"lan'
    'e_stride_bytes":2048,"lines":32,"sectors":32},"lines":16384,"requested_bytes'
    '":65536,"requests":512,"sectors":16384,"transferred_bytes":524288,"unique_by'
    'tes":65536},"b":{"classes":{"broadcast":512,"coalesced":0,"single":0,"stride'
    'd":0},"first_request":{"active_lanes":32,"class":"broadcast","distinct_addre'
    'sses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":512,"requested_'
    'bytes":65536,"requests":512,"sectors":512,"transferred_bytes":16384,"unique_'
    'bytes":2048},"c":{"classes":{"broadcast":0,"coalesced":0,"single":0,"strided'
    '":1},"first_request":{"active_lanes":32,"class":"strided","distinct_addresse'
    's":32,"lane_stride_bytes":2048,"lines":32,"sectors":32},"lines":32,"requeste'
    'd_bytes":128,"requests":1,"sectors":32,"transferred_bytes":1024,"unique_byte'
    's":128}},"sectors":16928}},"x_columns_to_x_rows_rate_ratio":2.20280324981875'
    '4},"median_sample_ns":{"matmul_x_columns":3005583,"matmul_x_rows":6620708},"'
    'n":512,"threadgroups":{"matmul_x_columns":[16,64],"matmul_x_rows":[16,64]}},'
    '{"calls_per_sample":1,"flops_per_call":2147483648,"index":3,"intensity":{"fl'
    'ops_per_byte":{"decimal":"170.666667","denominator":3,"numerator":512},"mode'
    'l":"ideal_reuse"},"k":1024,"kernels":{"matmul_x_columns":{"decision":{"bottl'
    'eneck":"latency or occupancy-bound","deciding":{"limit":0.7,"quantity":"lowe'
    'r_roof_share","value":0.07901509040545145},"explanation":"the rate is below '
    'the near-roof share of the lower roof, so neither measured roof explains it"'
    ',"numbers":{"bandwidth_roof_gflops_per_second":17512.05145733289,"bandwidth_'
    'roof_share":0.021465203119343957,"dispatch_ns":122167.0,"dispatch_share":0.0'
    '2057477050492033,"fixed_share_bound":0.07749200912338412,"fma_ceiling_gflops'
    '_per_second":4757.315844849298,"fma_ceiling_share":0.07901509040545145,"gpu_'
    'excess_share":-0.010785472982929948,"gpu_ns":5648875.0,"gpu_to_steady_ratio"'
    ':0.9887901379960776,"host_ns":5937709.0,"host_outside_gpu_share":0.048644014'
    '04649499,"lower_roof_share":0.07901509040545145,"overhead_share":0.037858541'
    '06356512,"rate_gflops_per_second":375.89974156805386,"steady_ns_per_call":57'
    '12916.0},"reason":"below_both_roofs"},"inputs":{"median_dispatch_ns":122167,'
    '"median_gpu_ns":5648875,"median_host_ns":5937709},"mapping":"x_columns"},"ma'
    'tmul_x_rows":{"decision":{"bottleneck":"latency or occupancy-bound","decidin'
    'g":{"limit":0.7,"quantity":"lower_roof_share","value":0.03186249481244988},"'
    'explanation":"the rate is below the near-roof share of the lower roof, so ne'
    'ither measured roof explains it","numbers":{"bandwidth_roof_gflops_per_secon'
    'd":17512.05145733289,"bandwidth_roof_share":0.008655750686720642,"dispatch_n'
    's":141084.0,"dispatch_share":0.009771741959631687,"fixed_share_bound":0.0369'
    '7427295466575,"fma_ceiling_gflops_per_second":4757.315844849298,"fma_ceiling'
    '_share":0.03186249481244988,"gpu_excess_share":-0.006750123528548843,"gpu_ns'
    '":14069875.0,"gpu_to_steady_ratio":0.9931209353235362,"host_ns":14437958.0,"'
    'host_outside_gpu_share":0.025494117658466653,"lower_roof_share":0.0318624948'
    '1244988,"overhead_share":0.018743994129917785,"rate_gflops_per_second":151.5'
    '799514276964,"steady_ns_per_call":14167333.0},"reason":"below_both_roofs"},"'
    'inputs":{"median_dispatch_ns":141084,"median_gpu_ns":14069875,"median_host_n'
    's":14437958},"mapping":"x_rows"}},"m":1024,"mapping":{"model_line_ratio":16.'
    '50756466569058,"model_predicts_x_rows_worse":true,"model_sector_ratio":6.601'
    '092896174864,"probe_warp":{"block":[0,0,0],"warp":0},"timing_shows_x_rows_sl'
    'ower":true,"verdict":"x_rows_worse_as_modeled","warps":{"x_columns":{"active'
    '_lanes":32,"lines":2049,"mapping":"x_columns","operands":{"a":{"classes":{"b'
    'roadcast":1024,"coalesced":0,"single":0,"strided":0},"first_request":{"activ'
    'e_lanes":32,"class":"broadcast","distinct_addresses":1,"lane_stride_bytes":0'
    ',"lines":1,"sectors":1},"lines":1024,"requested_bytes":131072,"requests":102'
    '4,"sectors":1024,"transferred_bytes":32768,"unique_bytes":4096},"b":{"classe'
    's":{"broadcast":0,"coalesced":1024,"single":0,"strided":0},"first_request":{'
    '"active_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_stride_b'
    'ytes":4,"lines":1,"sectors":4},"lines":1024,"requested_bytes":131072,"reques'
    'ts":1024,"sectors":4096,"transferred_bytes":131072,"unique_bytes":131072},"c'
    '":{"classes":{"broadcast":0,"coalesced":1,"single":0,"strided":0},"first_req'
    'uest":{"active_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_s'
    'tride_bytes":4,"lines":1,"sectors":4},"lines":1,"requested_bytes":128,"reque'
    'sts":1,"sectors":4,"transferred_bytes":128,"unique_bytes":128}},"sectors":51'
    '24},"x_rows":{"active_lanes":32,"lines":33824,"mapping":"x_rows","operands":'
    '{"a":{"classes":{"broadcast":0,"coalesced":0,"single":0,"strided":1024},"fir'
    'st_request":{"active_lanes":32,"class":"strided","distinct_addresses":32,"la'
    'ne_stride_bytes":4096,"lines":32,"sectors":32},"lines":32768,"requested_byte'
    's":131072,"requests":1024,"sectors":32768,"transferred_bytes":1048576,"uniqu'
    'e_bytes":131072},"b":{"classes":{"broadcast":1024,"coalesced":0,"single":0,"'
    'strided":0},"first_request":{"active_lanes":32,"class":"broadcast","distinct'
    '_addresses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":1024,"req'
    'uested_bytes":131072,"requests":1024,"sectors":1024,"transferred_bytes":3276'
    '8,"unique_bytes":4096},"c":{"classes":{"broadcast":0,"coalesced":0,"single":'
    '0,"strided":1},"first_request":{"active_lanes":32,"class":"strided","distinc'
    't_addresses":32,"lane_stride_bytes":4096,"lines":32,"sectors":32},"lines":32'
    ',"requested_bytes":128,"requests":1,"sectors":32,"transferred_bytes":1024,"u'
    'nique_bytes":128}},"sectors":33824}},"x_columns_to_x_rows_rate_ratio":2.4798'
    '777016850937},"median_sample_ns":{"matmul_x_columns":5712916,"matmul_x_rows"'
    ':14167333},"n":1024,"threadgroups":{"matmul_x_columns":[32,128],"matmul_x_ro'
    'ws":[32,128]}},{"calls_per_sample":1,"flops_per_call":17179869184,"index":4,'
    '"intensity":{"flops_per_byte":{"decimal":"341.333333","denominator":3,"numer'
    'ator":1024},"model":"ideal_reuse"},"k":2048,"kernels":{"matmul_x_columns":{"'
    'decision":{"bottleneck":"latency or occupancy-bound","deciding":{"limit":0.7'
    ',"quantity":"lower_roof_share","value":0.08381677235599153},"explanation":"t'
    "he rate is below the near-roof share of the lower roof, so neither measured "
    'roof explains it","numbers":{"bandwidth_roof_gflops_per_second":35024.102914'
    '66578,"bandwidth_roof_share":0.0113848129148318,"dispatch_ns":76208.0,"dispa'
    'tch_share":0.0017829804354114588,"fixed_share_bound":0.010765193586548623,"f'
    'ma_ceiling_gflops_per_second":4757.315844849298,"fma_ceiling_share":0.083816'
    '77235599153,"gpu_excess_share":-0.013166232416908966,"gpu_ns":42522333.0,"gp'
    'u_to_steady_ratio":0.9869386348866962,"host_ns":42741916.0,"host_outside_gpu'
    '_share":0.005137415926791844,"lower_roof_share":0.08381677235599153,"overhea'
    'd_share":-0.008028816490116997,"rate_gflops_per_second":398.7428591932851,"s'
    'teady_ns_per_call":43085083.0},"reason":"below_both_roofs"},"inputs":{"media'
    'n_dispatch_ns":76208,"median_gpu_ns":42522333,"median_host_ns":42741916},"ma'
    'pping":"x_columns"},"matmul_x_rows":{"decision":{"bottleneck":"latency or oc'
    'cupancy-bound","deciding":{"limit":0.7,"quantity":"lower_roof_share","value"'
    ':0.033342205442773554},"explanation":"the rate is below the near-roof share '
    'of the lower roof, so neither measured roof explains it","numbers":{"bandwid'
    'th_roof_gflops_per_second":35024.10291466578,"bandwidth_roof_share":0.004528'
    '864098006855,"dispatch_ns":70833.0,"dispatch_share":0.0006270811797235484,"f'
    'ixed_share_bound":0.004725998156443479,"fma_ceiling_gflops_per_second":4757.'
    '315844849298,"fma_ceiling_share":0.033342205442773554,"gpu_excess_share":0.0'
    '39450969281875144,"gpu_ns":112765000.0,"gpu_to_steady_ratio":1.0411439518967'
    '767,"host_ns":112956667.0,"host_outside_gpu_share":0.0016968188340755752,"lo'
    'wer_roof_share":0.033342205442773554,"overhead_share":0.041147788115950656,"'
    'rate_gflops_per_second":158.6194022551271,"steady_ns_per_call":108308750.0},'
    '"reason":"below_both_roofs"},"inputs":{"median_dispatch_ns":70833,"median_gp'
    'u_ns":112765000,"median_host_ns":112956667},"mapping":"x_rows"}},"m":2048,"m'
    'apping":{"model_line_ratio":16.503783256041007,"model_predicts_x_rows_worse"'
    ':true,"model_sector_ratio":6.600546661460367,"probe_warp":{"block":[0,0,0],"'
    'warp":0},"timing_shows_x_rows_slower":true,"verdict":"x_rows_worse_as_modele'
    'd","warps":{"x_columns":{"active_lanes":32,"lines":4097,"mapping":"x_columns'
    '","operands":{"a":{"classes":{"broadcast":2048,"coalesced":0,"single":0,"str'
    'ided":0},"first_request":{"active_lanes":32,"class":"broadcast","distinct_ad'
    'dresses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":2048,"reques'
    'ted_bytes":262144,"requests":2048,"sectors":2048,"transferred_bytes":65536,"'
    'unique_bytes":8192},"b":{"classes":{"broadcast":0,"coalesced":2048,"single":'
    '0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesced","disti'
    'nct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"lines":2048,'
    '"requested_bytes":262144,"requests":2048,"sectors":8192,"transferred_bytes":'
    '262144,"unique_bytes":262144},"c":{"classes":{"broadcast":0,"coalesced":1,"s'
    'ingle":0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesced"'
    ',"distinct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"lines'
    '":1,"requested_bytes":128,"requests":1,"sectors":4,"transferred_bytes":128,"'
    'unique_bytes":128}},"sectors":10244},"x_rows":{"active_lanes":32,"lines":676'
    '16,"mapping":"x_rows","operands":{"a":{"classes":{"broadcast":0,"coalesced":'
    '0,"single":0,"strided":2048},"first_request":{"active_lanes":32,"class":"str'
    'ided","distinct_addresses":32,"lane_stride_bytes":8192,"lines":32,"sectors":'
    '32},"lines":65536,"requested_bytes":262144,"requests":2048,"sectors":65536,"'
    'transferred_bytes":2097152,"unique_bytes":262144},"b":{"classes":{"broadcast'
    '":2048,"coalesced":0,"single":0,"strided":0},"first_request":{"active_lanes"'
    ':32,"class":"broadcast","distinct_addresses":1,"lane_stride_bytes":0,"lines"'
    ':1,"sectors":1},"lines":2048,"requested_bytes":262144,"requests":2048,"secto'
    'rs":2048,"transferred_bytes":65536,"unique_bytes":8192},"c":{"classes":{"bro'
    'adcast":0,"coalesced":0,"single":0,"strided":1},"first_request":{"active_lan'
    'es":32,"class":"strided","distinct_addresses":32,"lane_stride_bytes":8192,"l'
    'ines":32,"sectors":32},"lines":32,"requested_bytes":128,"requests":1,"sector'
    's":32,"transferred_bytes":1024,"unique_bytes":128}},"sectors":67616}},"x_col'
    'umns_to_x_rows_rate_ratio":2.513834080347484},"median_sample_ns":{"matmul_x_'
    'columns":43085083,"matmul_x_rows":108308750},"n":2048,"threadgroups":{"matmu'
    'l_x_columns":[64,256],"matmul_x_rows":[64,256]}},{"calls_per_sample":1,"flop'
    's_per_call":137438953472,"index":5,"intensity":{"flops_per_byte":{"decimal":'
    '"682.666667","denominator":3,"numerator":2048},"model":"ideal_reuse"},"k":40'
    '96,"kernels":{"matmul_x_columns":{"decision":{"bottleneck":"latency or occup'
    'ancy-bound","deciding":{"limit":0.7,"quantity":"lower_roof_share","value":0.'
    '07013960417523357},"explanation":"the rate is below the near-roof share of t'
    'he lower roof, so neither measured roof explains it","numbers":{"bandwidth_r'
    'oof_gflops_per_second":70048.20582933156,"bandwidth_roof_share":0.0047635231'
    '53002372,"dispatch_ns":150125.0,"dispatch_share":0.00032086367297507764,"fix'
    'ed_share_bound":0.0009834297920243637,"fma_ceiling_gflops_per_second":4757.3'
    '15844849298,"fma_ceiling_share":0.07013960417523357,"gpu_excess_share":0.119'
    '17186724253294,"gpu_ns":467651000.0,"gpu_to_steady_ratio":1.135369763697804,'
    '"host_ns":467877833.0,"host_outside_gpu_share":0.0004848124531687814,"lower_'
    'roof_share":0.07013960417523357,"overhead_share":0.1196566796957017,"rate_gf'
    'lops_per_second":333.6762502942966,"steady_ns_per_call":411893125.0},"reason'
    '":"below_both_roofs"},"inputs":{"median_dispatch_ns":150125,"median_gpu_ns":'
    '467651000,"median_host_ns":467877833},"mapping":"x_columns"},"matmul_x_rows"'
    ':{"decision":{"bottleneck":"latency or occupancy-bound","deciding":{"limit":'
    '0.7,"quantity":"lower_roof_share","value":0.038969764731511466},"explanation'
    '":"the rate is below the near-roof share of the lower roof, so neither measu'
    'red roof explains it","numbers":{"bandwidth_roof_gflops_per_second":70048.20'
    '582933156,"bandwidth_roof_share":0.0026466270910487635,"dispatch_ns":141417.'
    '0,"dispatch_share":0.00015904548193338493,"fixed_share_bound":0.000600378502'
    '987227,"fma_ceiling_gflops_per_second":4757.315844849298,"fma_ceiling_share"'
    ':0.038969764731511466,"gpu_excess_share":0.1657066790228876,"gpu_ns":8886843'
    '75.0,"gpu_to_steady_ratio":1.198746837671285,"host_ns":889160750.0,"host_out'
    'side_gpu_share":0.0005357580167590914,"lower_roof_share":0.03896976473151146'
    '6,"overhead_share":0.1662424370396467,"rate_gflops_per_second":185.391479227'
    '26883,"steady_ns_per_call":741344500.0},"reason":"below_both_roofs"},"inputs'
    '":{"median_dispatch_ns":141417,"median_gpu_ns":888684375,"median_host_ns":88'
    '9160750},"mapping":"x_rows"}},"m":4096,"mapping":{"model_line_ratio":16.5018'
    '91858903942,"model_predicts_x_rows_worse":true,"model_sector_ratio":6.600273'
    '384104667,"probe_warp":{"block":[0,0,0],"warp":0},"timing_shows_x_rows_slowe'
    'r":true,"verdict":"x_rows_worse_as_modeled","warps":{"x_columns":{"active_la'
    'nes":32,"lines":8193,"mapping":"x_columns","operands":{"a":{"classes":{"broa'
    'dcast":4096,"coalesced":0,"single":0,"strided":0},"first_request":{"active_l'
    'anes":32,"class":"broadcast","distinct_addresses":1,"lane_stride_bytes":0,"l'
    'ines":1,"sectors":1},"lines":4096,"requested_bytes":524288,"requests":4096,"'
    'sectors":4096,"transferred_bytes":131072,"unique_bytes":16384},"b":{"classes'
    '":{"broadcast":0,"coalesced":4096,"single":0,"strided":0},"first_request":{"'
    'active_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_stride_by'
    'tes":4,"lines":1,"sectors":4},"lines":4096,"requested_bytes":524288,"request'
    's":4096,"sectors":16384,"transferred_bytes":524288,"unique_bytes":524288},"c'
    '":{"classes":{"broadcast":0,"coalesced":1,"single":0,"strided":0},"first_req'
    'uest":{"active_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_s'
    'tride_bytes":4,"lines":1,"sectors":4},"lines":1,"requested_bytes":128,"reque'
    'sts":1,"sectors":4,"transferred_bytes":128,"unique_bytes":128}},"sectors":20'
    '484},"x_rows":{"active_lanes":32,"lines":135200,"mapping":"x_rows","operands'
    '":{"a":{"classes":{"broadcast":0,"coalesced":0,"single":0,"strided":4096},"f'
    'irst_request":{"active_lanes":32,"class":"strided","distinct_addresses":32,"'
    'lane_stride_bytes":16384,"lines":32,"sectors":32},"lines":131072,"requested_'
    'bytes":524288,"requests":4096,"sectors":131072,"transferred_bytes":4194304,"'
    'unique_bytes":524288},"b":{"classes":{"broadcast":4096,"coalesced":0,"single'
    '":0,"strided":0},"first_request":{"active_lanes":32,"class":"broadcast","dis'
    'tinct_addresses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":4096'
    ',"requested_bytes":524288,"requests":4096,"sectors":4096,"transferred_bytes"'
    ':131072,"unique_bytes":16384},"c":{"classes":{"broadcast":0,"coalesced":0,"s'
    'ingle":0,"strided":1},"first_request":{"active_lanes":32,"class":"strided","'
    'distinct_addresses":32,"lane_stride_bytes":16384,"lines":32,"sectors":32},"l'
    'ines":32,"requested_bytes":128,"requests":1,"sectors":32,"transferred_bytes"'
    ':1024,"unique_bytes":128}},"sectors":135200}},"x_columns_to_x_rows_rate_rati'
    'o":1.7998467442252162},"median_sample_ns":{"matmul_x_columns":411893125,"mat'
    'mul_x_rows":741344500},"n":4096,"threadgroups":{"matmul_x_columns":[128,512]'
    ',"matmul_x_rows":[128,512]}},{"calls_per_sample":1,"flops_per_call":85899345'
    '92,"index":6,"intensity":{"flops_per_byte":{"decimal":"227.555556","denomina'
    'tor":9,"numerator":2048},"model":"ideal_reuse"},"k":1024,"kernels":{"matmul_'
    'x_columns":{"decision":{"bottleneck":"latency or occupancy-bound","deciding"'
    ':{"limit":0.7,"quantity":"lower_roof_share","value":0.08409556075310447},"ex'
    'planation":"the rate is below the near-roof share of the lower roof, so neit'
    'her measured roof explains it","numbers":{"bandwidth_roof_gflops_per_second"'
    ':23349.40194311052,"bandwidth_roof_share":0.01713402101805332,"dispatch_ns":'
    '98833.0,"dispatch_share":0.004498023032564916,"fixed_share_bound":0.02094090'
    '8885280548,"fma_ceiling_gflops_per_second":4757.315844849298,"fma_ceiling_sh'
    'are":0.08409556075310447,"gpu_excess_share":0.000984182895178901,"gpu_ns":21'
    '492750.0,"gpu_to_steady_ratio":1.001007166601657,"host_ns":21972542.0,"host_'
    'outside_gpu_share":0.021835980561557222,"lower_roof_share":0.084095560753104'
    '47,"overhead_share":0.022820163456736098,"rate_gflops_per_second":400.069143'
    '6522306,"steady_ns_per_call":21471125.0},"reason":"below_both_roofs"},"input'
    's":{"median_dispatch_ns":98833,"median_gpu_ns":21492750,"median_host_ns":219'
    '72542},"mapping":"x_columns"},"matmul_x_rows":{"decision":{"bottleneck":"lat'
    'ency or occupancy-bound","deciding":{"limit":0.7,"quantity":"lower_roof_shar'
    'e","value":0.03120461211958354},"explanation":"the rate is below the near-ro'
    'of share of the lower roof, so neither measured roof explains it","numbers":'
    '{"bandwidth_roof_gflops_per_second":23349.40194311052,"bandwidth_roof_share"'
    ':0.006357772932709866,"dispatch_ns":125084.0,"dispatch_share":0.002188396432'
    '237093,"fixed_share_bound":0.009339629629772184,"fma_ceiling_gflops_per_seco'
    'nd":4757.315844849298,"fma_ceiling_share":0.03120461211958354,"gpu_excess_sh'
    'are":-0.01661773957354647,"gpu_ns":56914250.0,"gpu_to_steady_ratio":0.983585'
    '0853527726,"host_ns":57157834.0,"host_outside_gpu_share":0.00426160305514722'
    '3,"lower_roof_share":0.03120461211958354,"overhead_share":-0.012356136518399'
    '286,"rate_gflops_per_second":148.4501956688712,"steady_ns_per_call":57864084'
    '.0},"reason":"below_both_roofs"},"inputs":{"median_dispatch_ns":125084,"medi'
    'an_gpu_ns":56914250,"median_host_ns":57157834},"mapping":"x_rows"}},"m":4096'
    ',"mapping":{"model_line_ratio":16.50756466569058,"model_predicts_x_rows_wors'
    'e":true,"model_sector_ratio":6.601092896174864,"probe_warp":{"block":[0,0,0]'
    ',"warp":0},"timing_shows_x_rows_slower":true,"verdict":"x_rows_worse_as_mode'
    'led","warps":{"x_columns":{"active_lanes":32,"lines":2049,"mapping":"x_colum'
    'ns","operands":{"a":{"classes":{"broadcast":1024,"coalesced":0,"single":0,"s'
    'trided":0},"first_request":{"active_lanes":32,"class":"broadcast","distinct_'
    'addresses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":1024,"requ'
    'ested_bytes":131072,"requests":1024,"sectors":1024,"transferred_bytes":32768'
    ',"unique_bytes":4096},"b":{"classes":{"broadcast":0,"coalesced":1024,"single'
    '":0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesced","dis'
    'tinct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"lines":102'
    '4,"requested_bytes":131072,"requests":1024,"sectors":4096,"transferred_bytes'
    '":131072,"unique_bytes":131072},"c":{"classes":{"broadcast":0,"coalesced":1,'
    '"single":0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesce'
    'd","distinct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"lin'
    'es":1,"requested_bytes":128,"requests":1,"sectors":4,"transferred_bytes":128'
    ',"unique_bytes":128}},"sectors":5124},"x_rows":{"active_lanes":32,"lines":33'
    '824,"mapping":"x_rows","operands":{"a":{"classes":{"broadcast":0,"coalesced"'
    ':0,"single":0,"strided":1024},"first_request":{"active_lanes":32,"class":"st'
    'rided","distinct_addresses":32,"lane_stride_bytes":4096,"lines":32,"sectors"'
    ':32},"lines":32768,"requested_bytes":131072,"requests":1024,"sectors":32768,'
    '"transferred_bytes":1048576,"unique_bytes":131072},"b":{"classes":{"broadcas'
    't":1024,"coalesced":0,"single":0,"strided":0},"first_request":{"active_lanes'
    '":32,"class":"broadcast","distinct_addresses":1,"lane_stride_bytes":0,"lines'
    '":1,"sectors":1},"lines":1024,"requested_bytes":131072,"requests":1024,"sect'
    'ors":1024,"transferred_bytes":32768,"unique_bytes":4096},"c":{"classes":{"br'
    'oadcast":0,"coalesced":0,"single":0,"strided":1},"first_request":{"active_la'
    'nes":32,"class":"strided","distinct_addresses":32,"lane_stride_bytes":4096,"'
    'lines":32,"sectors":32},"lines":32,"requested_bytes":128,"requests":1,"secto'
    'rs":32,"transferred_bytes":1024,"unique_bytes":128}},"sectors":33824}},"x_co'
    'lumns_to_x_rows_rate_ratio":2.6949721544632617},"median_sample_ns":{"matmul_'
    'x_columns":21471125,"matmul_x_rows":57864084},"n":1024,"threadgroups":{"matm'
    'ul_x_columns":[32,512],"matmul_x_rows":[128,128]}},{"calls_per_sample":1,"fl'
    'ops_per_call":17179869184,"index":7,"intensity":{"flops_per_byte":{"decimal"'
    ':"292.571429","denominator":7,"numerator":2048},"model":"ideal_reuse"},"k":2'
    '048,"kernels":{"matmul_x_columns":{"decision":{"bottleneck":"latency or occu'
    'pancy-bound","deciding":{"limit":0.7,"quantity":"lower_roof_share","value":0'
    '.08261057968599927},"explanation":"the rate is below the near-roof share of '
    'the lower roof, so neither measured roof explains it","numbers":{"bandwidth_'
    'roof_gflops_per_second":30020.659641142098,"bandwidth_roof_share":0.01309113'
    '8715479556,"dispatch_ns":116375.0,"dispatch_share":0.002568023211090206,"fix'
    'ed_share_bound":0.010153483823870085,"fma_ceiling_gflops_per_second":4757.31'
    '5844849298,"fma_ceiling_share":0.08261057968599927,"gpu_excess_share":0.0280'
    '1472243569394,"gpu_ns":44983708.0,"gpu_to_steady_ratio":1.0290418899905354,"'
    'host_ns":45316958.0,"host_outside_gpu_share":0.007353759270425919,"lower_roo'
    'f_share":0.08261057968599927,"overhead_share":0.035368481706119836,"rate_gfl'
    'ops_per_second":393.0046196923899,"steady_ns_per_call":43714166.0},"reason":'
    '"below_both_roofs"},"inputs":{"median_dispatch_ns":116375,"median_gpu_ns":44'
    '983708,"median_host_ns":45316958},"mapping":"x_columns"},"matmul_x_rows":{"d'
    'ecision":{"bottleneck":"latency or occupancy-bound","deciding":{"limit":0.7,'
    '"quantity":"lower_roof_share","value":0.03306227000681552},"explanation":"th'
    "e rate is below the near-roof share of the lower roof, so neither measured r"
    'oof explains it","numbers":{"bandwidth_roof_gflops_per_second":30020.6596411'
    '42098,"bandwidth_roof_share":0.005239313954132866,"dispatch_ns":123708.0,"di'
    'spatch_share":0.0011274230771933296,"fixed_share_bound":0.004865131144043609'
    ',"fma_ceiling_gflops_per_second":4757.315844849298,"fma_ceiling_share":0.033'
    '06227000681552,"gpu_excess_share":0.002620920540559758,"gpu_ns":109513375.0,'
    '"gpu_to_steady_ratio":1.0026329312643751,"host_ns":109726333.0,"host_outside'
    '_gpu_share":0.0019408103249016762,"lower_roof_share":0.03306227000681552,"ov'
    'erhead_share":0.004561730865461411,"rate_gflops_per_second":157.287660970109'
    '16,"steady_ns_per_call":109225791.0},"reason":"below_both_roofs"},"inputs":{'
    '"median_dispatch_ns":123708,"median_gpu_ns":109513375,"median_host_ns":10972'
    '6333},"mapping":"x_rows"}},"m":1024,"mapping":{"model_line_ratio":16.5037832'
    '56041007,"model_predicts_x_rows_worse":true,"model_sector_ratio":6.600546661'
    '460367,"probe_warp":{"block":[0,0,0],"warp":0},"timing_shows_x_rows_slower":'
    'true,"verdict":"x_rows_worse_as_modeled","warps":{"x_columns":{"active_lanes'
    '":32,"lines":4097,"mapping":"x_columns","operands":{"a":{"classes":{"broadca'
    'st":2048,"coalesced":0,"single":0,"strided":0},"first_request":{"active_lane'
    's":32,"class":"broadcast","distinct_addresses":1,"lane_stride_bytes":0,"line'
    's":1,"sectors":1},"lines":2048,"requested_bytes":262144,"requests":2048,"sec'
    'tors":2048,"transferred_bytes":65536,"unique_bytes":8192},"b":{"classes":{"b'
    'roadcast":0,"coalesced":2048,"single":0,"strided":0},"first_request":{"activ'
    'e_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_stride_bytes":'
    '4,"lines":1,"sectors":4},"lines":2048,"requested_bytes":262144,"requests":20'
    '48,"sectors":8192,"transferred_bytes":262144,"unique_bytes":262144},"c":{"cl'
    'asses":{"broadcast":0,"coalesced":1,"single":0,"strided":0},"first_request":'
    '{"active_lanes":32,"class":"coalesced","distinct_addresses":32,"lane_stride_'
    'bytes":4,"lines":1,"sectors":4},"lines":1,"requested_bytes":128,"requests":1'
    ',"sectors":4,"transferred_bytes":128,"unique_bytes":128}},"sectors":10244},"'
    'x_rows":{"active_lanes":32,"lines":67616,"mapping":"x_rows","operands":{"a":'
    '{"classes":{"broadcast":0,"coalesced":0,"single":0,"strided":2048},"first_re'
    'quest":{"active_lanes":32,"class":"strided","distinct_addresses":32,"lane_st'
    'ride_bytes":8192,"lines":32,"sectors":32},"lines":65536,"requested_bytes":26'
    '2144,"requests":2048,"sectors":65536,"transferred_bytes":2097152,"unique_byt'
    'es":262144},"b":{"classes":{"broadcast":2048,"coalesced":0,"single":0,"strid'
    'ed":0},"first_request":{"active_lanes":32,"class":"broadcast","distinct_addr'
    'esses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":2048,"requeste'
    'd_bytes":262144,"requests":2048,"sectors":2048,"transferred_bytes":65536,"un'
    'ique_bytes":8192},"c":{"classes":{"broadcast":0,"coalesced":0,"single":0,"st'
    'rided":1},"first_request":{"active_lanes":32,"class":"strided","distinct_add'
    'resses":32,"lane_stride_bytes":16384,"lines":32,"sectors":32},"lines":32,"re'
    'quested_bytes":128,"requests":1,"sectors":32,"transferred_bytes":1024,"uniqu'
    'e_bytes":128}},"sectors":67616}},"x_columns_to_x_rows_rate_ratio":2.49863604'
    '85523162},"median_sample_ns":{"matmul_x_columns":43714166,"matmul_x_rows":10'
    '9225791},"n":4096,"threadgroups":{"matmul_x_columns":[128,128],"matmul_x_row'
    's":[32,512]}},{"calls_per_sample":1,"flops_per_call":12000000000,"index":8,"'
    'intensity":{"flops_per_byte":{"decimal":"272.727273","denominator":11,"numer'
    'ator":3000},"model":"ideal_reuse"},"k":1000,"kernels":{"matmul_x_columns":{"'
    'decision":{"bottleneck":"latency or occupancy-bound","deciding":{"limit":0.7'
    ',"quantity":"lower_roof_share","value":0.0811084035884357},"explanation":"th'
    "e rate is below the near-roof share of the lower roof, so neither measured r"
    'oof explains it","numbers":{"bandwidth_roof_gflops_per_second":27984.4572294'
    '02706,"bandwidth_roof_share":0.013788307215631228,"dispatch_ns":76708.0,"dis'
    'patch_share":0.0025409956159312356,"fixed_share_bound":0.015241899251451736,'
    '"fma_ceiling_gflops_per_second":4757.315844849298,"fma_ceiling_share":0.0811'
    '084035884357,"gpu_excess_share":-0.03822828991240177,"gpu_ns":29945458.0,"gp'
    'u_to_steady_ratio":0.9628919436003794,"host_ns":30188167.0,"host_outside_gpu'
    '_share":0.008039872046553875,"lower_roof_share":0.0811084035884357,"overhead'
    '_share":-0.030188417865847805,"rate_gflops_per_second":385.8582935416968,"st'
    'eady_ns_per_call":31099500.0},"reason":"below_both_roofs"},"inputs":{"median'
    '_dispatch_ns":76708,"median_gpu_ns":29945458,"median_host_ns":30188167},"map'
    'ping":"x_columns"},"matmul_x_rows":{"decision":{"bottleneck":"latency or occ'
    'upancy-bound","deciding":{"limit":0.7,"quantity":"lower_roof_share","value":'
    '0.025025118540949392},"explanation":"the rate is below the near-roof share o'
    'f the lower roof, so neither measured roof explains it","numbers":{"bandwidt'
    'h_roof_gflops_per_second":27984.457229402706,"bandwidth_roof_share":0.004254'
    '232697034572,"dispatch_ns":74625.0,"dispatch_share":0.0007446762533335426,"f'
    'ixed_share_bound":0.005327072138637254,"fma_ceiling_gflops_per_second":4757.'
    '315844849298,"fma_ceiling_share":0.025025118540949392,"gpu_excess_share":-0.'
    '007798100041239846,"gpu_ns":100014500.0,"gpu_to_steady_ratio":0.992247129592'
    '2402,"host_ns":100211333.0,"host_outside_gpu_share":0.0019641790415062532,"l'
    'ower_roof_share":0.025025118540949392,"overhead_share":-0.005833920999733566'
    ',"rate_gflops_per_second":119.05239295409048,"steady_ns_per_call":100795958.'
    '0},"reason":"below_both_roofs"},"inputs":{"median_dispatch_ns":74625,"median'
    '_gpu_ns":100014500,"median_host_ns":100211333},"mapping":"x_rows"}},"m":2000'
    ',"mapping":{"model_line_ratio":12.007270083605961,"model_predicts_x_rows_wor'
    'se":true,"model_sector_ratio":6.601119104716227,"probe_warp":{"block":[0,0,0'
    '],"warp":0},"timing_shows_x_rows_slower":true,"verdict":"x_rows_worse_as_mod'
    'eled","warps":{"x_columns":{"active_lanes":32,"lines":2751,"mapping":"x_colu'
    'mns","operands":{"a":{"classes":{"broadcast":1000,"coalesced":0,"single":0,"'
    'strided":0},"first_request":{"active_lanes":32,"class":"broadcast","distinct'
    '_addresses":1,"lane_stride_bytes":0,"lines":1,"sectors":1},"lines":1000,"req'
    'uested_bytes":128000,"requests":1000,"sectors":1000,"transferred_bytes":3200'
    '0,"unique_bytes":4000},"b":{"classes":{"broadcast":0,"coalesced":1000,"singl'
    'e":0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesced","di'
    'stinct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"lines":17'
    '50,"requested_bytes":128000,"requests":1000,"sectors":4000,"transferred_byte'
    's":128000,"unique_bytes":128000},"c":{"classes":{"broadcast":0,"coalesced":1'
    ',"single":0,"strided":0},"first_request":{"active_lanes":32,"class":"coalesc'
    'ed","distinct_addresses":32,"lane_stride_bytes":4,"lines":1,"sectors":4},"li'
    'nes":1,"requested_bytes":128,"requests":1,"sectors":4,"transferred_bytes":12'
    '8,"unique_bytes":128}},"sectors":5004},"x_rows":{"active_lanes":32,"lines":3'
    '3032,"mapping":"x_rows","operands":{"a":{"classes":{"broadcast":0,"coalesced'
    '":0,"single":0,"strided":1000},"first_request":{"active_lanes":32,"class":"s'
    'trided","distinct_addresses":32,"lane_stride_bytes":4000,"lines":32,"sectors'
    '":32},"lines":32000,"requested_bytes":128000,"requests":1000,"sectors":32000'
    ',"transferred_bytes":1024000,"unique_bytes":128000},"b":{"classes":{"broadca'
    'st":1000,"coalesced":0,"single":0,"strided":0},"first_request":{"active_lane'
    's":32,"class":"broadcast","distinct_addresses":1,"lane_stride_bytes":0,"line'
    's":1,"sectors":1},"lines":1000,"requested_bytes":128000,"requests":1000,"sec'
    'tors":1000,"transferred_bytes":32000,"unique_bytes":4000},"c":{"classes":{"b'
    'roadcast":0,"coalesced":0,"single":0,"strided":1},"first_request":{"active_l'
    'anes":32,"class":"strided","distinct_addresses":32,"lane_stride_bytes":12000'
    ',"lines":32,"sectors":32},"lines":32,"requested_bytes":128,"requests":1,"sec'
    'tors":32,"transferred_bytes":1024,"unique_bytes":128}},"sectors":33032}},"x_'
    'columns_to_x_rows_rate_ratio":3.241079695815045},"median_sample_ns":{"matmul'
    '_x_columns":31099500,"matmul_x_rows":100795958},"n":3000,"threadgroups":{"ma'
    'tmul_x_columns":[94,250],"matmul_x_rows":[63,375]}}],"reasons":{"below_both_'
    'roofs":{"bottleneck":"latency or occupancy-bound","explanation":"the rate is'
    " below the near-roof share of the lower roof, so neither measured roof expla"
    'ins it"},"dispatch_time_above_host_time":{"bottleneck":"indeterminate","expl'
    'anation":"the dispatch time of one call exceeds its host time"},"fixed_cost_'
    'not_ruled_out":{"bottleneck":"indeterminate","explanation":"the smallest one'
    "-call host time of this kernel does not rule out a fixed cost that fills the"
    ' pinned share of this call"},"fma_ceiling_does_not_scale_with_work":{"bottle'
    'neck":"indeterminate","explanation":"the rate is near the FMA ceiling, but t'
    'he ceiling lab flags that ceiling as fixed-cost bound"},"gpu_time_above_host'
    '_time":{"bottleneck":"indeterminate","explanation":"the GPU time of one call'
    ' exceeds its host time by more than the tolerance"},"gpu_time_disagrees_with'
    '_steady_time":{"bottleneck":"indeterminate","explanation":"the one-call GPU '
    "time and the steady-state time per call differ by more than the agreement ra"
    'tio"},"near_bandwidth_roof":{"bottleneck":"memory-bound","explanation":"the '
    'rate reaches the near-roof share of the bandwidth roof, the lower roof"},"ne'
    'ar_fma_ceiling":{"bottleneck":"compute-bound","explanation":"the rate reache'
    's the near-roof share of the FMA ceiling, the lower roof"},"one_call_overhea'
    'd_dominates":{"bottleneck":"dispatch-bound","explanation":"most of the one-c'
    'all host time lies outside the steady-state time per call"},"rate_above_band'
    'width_roof":{"bottleneck":"indeterminate","explanation":"the rate exceeds th'
    "e bandwidth roof by more than the tolerance, so the traffic model overcounts"
    ' the bytes or the ceiling is low"},"rate_above_fma_ceiling":{"bottleneck":"i'
    'ndeterminate","explanation":"the rate exceeds the measured FMA ceiling by mo'
    're than the tolerance"},"steady_time_above_host_time":{"bottleneck":"indeter'
    'minate","explanation":"the steady-state time per call exceeds the one-call h'
    'ost time by more than the tolerance"}},"report_sha256":"bad8d9707745344c26a1'
    '6676a757e1bb6c60c880afd3a10557cbdc34aa0ec2f5","schema_version":1,"sources":{'
    '"gpu_ceilings":{"kind":"gpu_ceiling_measurement","load_average_after":[10.64'
    '306640625,10.07080078125,9.13720703125],"load_average_before":[11.4077148437'
    '5,10.193359375,9.1689453125],"report_sha256":"9697e2556bcb54cf7d4a842fc99f36'
    '01d7d434f50ffc35c2c95b0534ef7ca57a"},"matmul_metal":{"kind":"matmul_metal_co'
    'rrectness_and_gflops","report_sha256":"c9ce115a7ffed725e5ea0a0d7898709ba5b77'
    'e697467728949c64989a5416637"},"matmul_roofline_gap":{"kind":"matmul_roofline'
    '_gap","label":"host-specific Apple GPU (Metal) measurement","measured":true,'
    '"report_sha256":"94ca519c6033bb5dedbfca46da9111f4bdfaf5e1816d1453629aced8335'
    '98341"},"timing":{"entries":30,"entries_used":18,"fastest_calls":{"matmul_x_'
    'columns":{"k":128,"kernel":"matmul_x_columns","m":128,"median_dispatch_ns":8'
    '9833,"median_gpu_ns":249125,"median_host_ns":460125,"n":128},"matmul_x_rows"'
    ':{"k":128,"kernel":"matmul_x_rows","m":128,"median_dispatch_ns":79584,"media'
    'n_gpu_ns":247458,"median_host_ns":533833,"n":128}},"input_sha256":"59c6abd67'
    '6437622cc6335046e62a947d4000484f94c1164f9b51c6c09fbf9d4","kind":"matmul_gpu_'
    'timing","label":"host-specific Apple GPU (Metal) measurement","load_average_'
    'after":[17.61181640625,11.85595703125,9.853515625],"load_average_before":[10'
    '.50830078125,10.19384765625,9.22900390625],"measured":true,"producer":"scrat'
    "ch capture with torch.mps.Event pairs and time.perf_counter_ns, one call per"
    ' sample","repeats":11}},"summary":{"label":"host-specific Apple GPU (Metal) '
    'measurement","mapping_verdicts":{"model_predicts_x_rows_worse_but_timing_doe'
    's_not_show_it":0,"no_difference_in_model_or_timing":0,"x_rows_slower_without'
    '_a_model_reason":0,"x_rows_worse_as_modeled":9},"points":9,"top_bottleneck":'
    '{"matmul_x_columns":{"counts":{"compute-bound":0,"dispatch-bound":3,"indeter'
    'minate":0,"latency or occupancy-bound":6,"memory-bound":0},"reason":"most_po'
    'ints","top":"latency or occupancy-bound","top_points":6},"matmul_x_rows":{"c'
    'ounts":{"compute-bound":0,"dispatch-bound":3,"indeterminate":0,"latency or o'
    'ccupancy-bound":6,"memory-bound":0},"reason":"most_points","top":"latency or'
    ' occupancy-bound","top_points":6}}}}'
)
RECORDED_TIMING = json.loads(RECORDED_TIMING_JSON)
RECORDED_REPORT = json.loads(RECORDED_REPORT_JSON)
NAIVE = ("matmul_x_columns", "matmul_x_rows")
SMALL = ((128, 128, 128), (256, 256, 256), (512, 512, 512))


def test_the_recorded_timing_is_a_measured_input_with_its_load_averages():
    timing = timing_from_mapping(RECORDED_TIMING)

    assert canonical_json_bytes(RECORDED_TIMING).decode() == RECORDED_TIMING_JSON
    assert timing.digest() == TIMING_SHA256
    assert (timing.measured, timing.repeats, len(timing.entries)) == (True, 11, 30)
    assert timing.load_average_before == (10.50830078125, 10.19384765625, 9.22900390625)
    assert timing.load_average_after == (17.61181640625, 11.85595703125, 9.853515625)
    assert dict(timing.environment)["platform_system"] == "Darwin"


def test_the_recorded_report_verifies_on_any_host_and_names_its_sources():
    summary = verify_report(RECORDED_REPORT, config=CONFIG)
    sources = RECORDED_REPORT["sources"]

    assert canonical_json_bytes(RECORDED_REPORT).decode() == RECORDED_REPORT_JSON
    assert RECORDED_REPORT[DIGEST_FIELD] == REPORT_SHA256
    assert summary["measured"] is True
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert sources["matmul_roofline_gap"]["report_sha256"] == GAP_SHA256
    assert sources["matmul_metal"]["report_sha256"] == MATMUL_SHA256
    assert sources["gpu_ceilings"]["report_sha256"] == CEILING_SHA256
    assert sources["timing"]["input_sha256"] == TIMING_SHA256
    assert sources["gpu_ceilings"]["load_average_before"] == [
        11.40771484375,
        10.193359375,
        9.1689453125,
    ]
    assert RECORDED_REPORT["ceilings"]["fma_scales_with_work"] is True


def test_the_small_shapes_are_dispatch_bound_and_the_rest_sit_below_both_roofs():
    for point in RECORDED_REPORT["points"]:
        shape = (point["m"], point["n"], point["k"])
        for kernel in NAIVE:
            decision = point["kernels"][kernel]["decision"]
            if shape in SMALL:
                assert decision["reason"] == "one_call_overhead_dominates"
            else:
                assert decision["reason"] == "below_both_roofs"
                assert decision["numbers"]["lower_roof_share"] < 0.1
        assert point["mapping"]["verdict"] == "x_rows_worse_as_modeled"
        assert 6.6 <= point["mapping"]["model_sector_ratio"] < 6.62
    for kernel in NAIVE:
        top = RECORDED_REPORT["summary"]["top_bottleneck"][kernel]
        assert (top["top"], top["top_points"]) == ("latency or occupancy-bound", 6)
        assert top["counts"]["dispatch-bound"] == 3


def test_x_rows_runs_slower_at_every_shape_as_the_warp_model_predicts():
    ratios = [
        point["mapping"]["x_columns_to_x_rows_rate_ratio"]
        for point in RECORDED_REPORT["points"]
    ]

    assert min(ratios) > 1.5
    assert max(ratios) < 3.3
    for point in RECORDED_REPORT["points"]:
        warps = point["mapping"]["warps"]
        assert warps["x_rows"]["operands"]["a"]["classes"]["strided"] == point["k"]
        assert (
            warps["x_columns"]["operands"]["a"]["classes"]["broadcast"] == (point["k"])
        )


def test_the_recorded_timing_joins_the_recorded_day_74_fixture_and_flags_the_mix():
    # The Day 74 fixture is a small run from 2026-11-17 with one call per sample
    # at 256 x 256 x 256, so its times do not pair cleanly with this input.
    report = build_report(
        recorded_sources(), timing_from_mapping(RECORDED_TIMING), CONFIG
    )
    reasons = {
        (point["m"], kernel): point["kernels"][kernel]["decision"]["reason"]
        for point in report["points"]
        for kernel in NAIVE
    }

    assert verify_report(report)["measured"] is True
    assert reasons == {
        (256, "matmul_x_columns"): "fixed_cost_not_ruled_out",
        (256, "matmul_x_rows"): "below_both_roofs",
        (129, "matmul_x_columns"): "steady_time_above_host_time",
        (129, "matmul_x_rows"): "steady_time_above_host_time",
    }


def test_the_installed_verify_checks_the_recorded_report_without_torch(tmp_path):
    path = tmp_path / "bottleneck.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="utf-8")
    assert load_report(path)[1]["report_sha256"] == REPORT_SHA256

    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, "verify", "--report", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout.startswith(
        f"verified_matmul_bottleneck_measurement={REPORT_SHA256} points=9 "
    )
