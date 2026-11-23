"""One real measured report from the development host's Apple GPU, replayed anywhere.

``RECORDED_REPORT_JSON`` is the canonical file that ``gpu_reduce_cli run``
wrote on 2026-11-21 at 02:17 IST in a fresh process (torch 2.15.0, macOS
27.2, arm64) for ``RECORDED_CONFIG``: six correctness cases with two runs of
each kernel, and timing at 2^16 and 2^20 elements with seven repeats. Its
ceiling is the recorded Day 74 gpu-ceilings fixture of 2026-11-17, 82.28
GB/s at a 4 MiB buffer. Other jobs kept the host's load averages at 11.59,
11.22 and 12.97. The timings describe that moment and are here only to be
verified, on hosts without torch or MPS as well. The last test pins the CI
job whose globs run these files on Linux.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_reduce_ceiling import load_ceiling
from ai_journey.gpu_reduce_cli import EXIT_OK, main
from ai_journey.gpu_reduce_config import config_from_mapping
from ai_journey.gpu_reduce_io import load_report
from ai_journey.gpu_reduce_source import STEP_KERNELS
from ai_journey.gpu_reduce_verify import GpuReduceVerificationError, verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import seal_report
from test_gpu_reduce_speedup import write_ceiling
from test_matmul_roofline_ci_contract import ci_job

ROOT = Path(__file__).resolve().parents[1]
RECORDED_SHA256 = "54666a185b431d201d2c3a7719077f401df8a142957974fdbd21153f42c96be9"
RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "gpu-reduce-recorded",
    "seed": 78,
    "plan_sha256": "b2134229a87d220e40772e732ca3d3e5d6c6a6ae82a90a475eb0405d87a7217f",
    "correctness": {
        "cases": [
            ["normal", 1],
            ["normal", 33],
            ["positive", 300007],
            ["mixed", 1600001],
            ["ones", 1025],
            ["halves", 2049],
        ],
        "runs": 2,
    },
    "benchmark": {
        "sizes": [65536, 1048576],
        "warmup": 1,
        "repeats": 7,
        "target_bytes_per_sample": 16777216,
        "max_runtime_seconds": 120,
    },
    "comparison": {"min_faster_ratio": 1.1},
}
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"device":"mps","gb_per_se'
    'cond":"decimal GB/s: 4 * n bytes for each call, each element read once, time'
    "s the calls of a sample, divided by the median sample duration in ns; the pa"
    'rtials of pass 2 are not counted","points":[{"bytes_per_call":262144,"checks'
    '":{"reduce_r0_interleaved":{"bound":0.056220427053109784,"bound_ratio":0.000'
    '8569176192564273,"depth":18,"error":4.817627450393047e-05,"exact_result":fal'
    'se,"output":-92.49168395996094,"rule":"order_bound","same_bits_on_repeat":tr'
    'ue},"reduce_r1_sequential":{"bound":0.056220427053109784,"bound_ratio":0.000'
    '5001326436387185,"depth":18,"error":2.8117670808569528e-05,"exact_result":fa'
    'lse,"output":-92.49160766601562,"rule":"order_bound","same_bits_on_repeat":t'
    'rue},"reduce_r2_first_add":{"bound":0.05934378764878506,"bound_ratio":0.0004'
    '738098446795918,"depth":19,"error":2.8117670808569528e-05,"exact_result":fal'
    'se,"output":-92.49160766601562,"rule":"order_bound","same_bits_on_repeat":tr'
    'ue},"reduce_r3_simd_shuffle":{"bound":0.05934378764878506,"bound_ratio":0.00'
    '04738098446795918,"depth":19,"error":2.8117670808569528e-05,"exact_result":f'
    'alse,"output":-92.49160766601562,"rule":"order_bound","same_bits_on_repeat":'
    'true},"reduce_r4_simd_sum":{"bound":0.15304477865467192,"bound_ratio":0.0001'
    '1538254708300311,"depth":49,"error":1.7658696378930472e-05,"exact_result":fa'
    'lse,"output":-92.49165344238281,"rule":"order_bound","same_bits_on_repeat":t'
    'rue},"reduce_r5_grid_stride":{"bound":0.14992140688894642,"bound_ratio":0.00'
    '03213435326125029,"depth":48,"error":4.817627450393047e-05,"exact_result":fa'
    'lse,"output":-92.49168395996094,"rule":"order_bound","same_bits_on_repeat":t'
    'rue},"torch_sum":{"bound":12.79638090238203,"bound_ratio":1.3799758317324963'
    'e-06,"depth":4096,"error":1.7658696378930472e-05,"exact_result":false,"outpu'
    't":-92.49165344238281,"rule":"declared_tolerance","same_bits_on_repeat":true'
    '}},"exact_in_any_order":false,"grid_exponent":-39,"index":0,"launches":{"red'
    'uce_r0_interleaved":{"active_threads":65536,"extent":[65536],"group_size":[2'
    '56],"groups":[256],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"'
    'padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[65536],"thread'
    's_per_group":256},"reduce_r1_sequential":{"active_threads":65536,"extent":[6'
    '5536],"group_size":[256],"groups":[256],"idle_lanes_per_threadgroup":0,"laun'
    'ched_threads":65536,"padding_threads":0,"simd_groups_per_threadgroup":8,"thr'
    'eads":[65536],"threads_per_group":256},"reduce_r2_first_add":{"active_thread'
    's":32768,"extent":[32768],"group_size":[256],"groups":[128],"idle_lanes_per_'
    'threadgroup":0,"launched_threads":32768,"padding_threads":0,"simd_groups_per'
    '_threadgroup":8,"threads":[32768],"threads_per_group":256},"reduce_r3_simd_s'
    'huffle":{"active_threads":32768,"extent":[32768],"group_size":[256],"groups"'
    ':[128],"idle_lanes_per_threadgroup":0,"launched_threads":32768,"padding_thre'
    'ads":0,"simd_groups_per_threadgroup":8,"threads":[32768],"threads_per_group"'
    ':256},"reduce_r4_simd_sum":{"active_threads":32768,"extent":[32768],"group_s'
    'ize":[256],"groups":[128],"idle_lanes_per_threadgroup":0,"launched_threads":'
    '32768,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[32768],'
    '"threads_per_group":256},"reduce_r5_grid_stride":{"active_threads":262144,"e'
    'xtent":[262144],"group_size":[256],"groups":[1024],"idle_lanes_per_threadgro'
    'up":0,"launched_threads":262144,"padding_threads":0,"simd_groups_per_threadg'
    'roup":8,"threads":[262144],"threads_per_group":256}},"n":65536,"reference":-'
    '92.49163578368643,"side_order":["reduce_r0_interleaved","reduce_r1_sequentia'
    'l","reduce_r2_first_add","reduce_r3_simd_shuffle","reduce_r4_simd_sum","redu'
    'ce_r5_grid_stride","torch_sum"],"sides":{"reduce_r0_interleaved":{"best_gb_p'
    'er_second":10.093835301195758,"bytes_per_sample":16777216,"calls_per_sample"'
    ':64,"implementation":"reduce_r0_interleaved","iqr_gb_per_second":[5.77910112'
    '1428109,9.100740981828045],"iqr_ns_per_call":16556.0,"median_gb_per_second":'
    '9.04046630250928,"median_ns_per_call":28996.734375,"repeats":7,"samples_ns":'
    '[2903084,2488000,2921291,1662125,1850500,1843500,1855791],"timing":{"count":'
    '7,"iqr_ns":1059584,"max_ns":2921291,"median_ns":1855791,"min_ns":1662125,"p2'
    '5_ns":1843500,"p75_ns":2903084},"warmup":1},"reduce_r1_sequential":{"best_gb'
    '_per_second":19.739828712686062,"bytes_per_sample":16777216,"calls_per_sampl'
    'e":64,"implementation":"reduce_r1_sequential","iqr_gb_per_second":[8.0751896'
    '9977739,19.572867364235908],"iqr_ns_per_call":19069.65625,"median_gb_per_sec'
    'ond":14.183422593257951,"median_ns_per_call":18482.421875,"repeats":7,"sampl'
    'es_ns":[1550833,1182875,2077625,2142375,958125,849917,857167],"timing":{"cou'
    'nt":7,"iqr_ns":1220458,"max_ns":2142375,"median_ns":1182875,"min_ns":849917,'
    '"p25_ns":857167,"p75_ns":2077625},"warmup":1},"reduce_r2_first_add":{"best_g'
    'b_per_second":25.925773227738073,"bytes_per_sample":16777216,"calls_per_samp'
    'le":64,"implementation":"reduce_r2_first_add","iqr_gb_per_second":[10.270452'
    '795214325,23.386953824708137],"iqr_ns_per_call":14315.109375,"median_gb_per_'
    'second":11.596822579262712,"median_ns_per_call":22604.8125,"repeats":7,"samp'
    'les_ns":[2118875,1633542,1554500,1044750,1446708,717375,647125],"timing":{"c'
    'ount":7,"iqr_ns":916167,"max_ns":2118875,"median_ns":1446708,"min_ns":647125'
    ',"p25_ns":717375,"p75_ns":1633542},"warmup":1},"reduce_r3_simd_shuffle":{"be'
    'st_gb_per_second":26.933308932008597,"bytes_per_sample":16777216,"calls_per_'
    'sample":64,"implementation":"reduce_r3_simd_shuffle","iqr_gb_per_second":[13'
    '.874067397146993,17.77168388694571],"iqr_ns_per_call":4143.875,"median_gb_pe'
    'r_second":15.357886228427423,"median_ns_per_call":17069.015625,"repeats":7,"'
    'samples_ns":[1155250,1209250,1092417,1066125,1215000,622917,944042],"timing"'
    ':{"count":7,"iqr_ns":265208,"max_ns":1215000,"median_ns":1092417,"min_ns":62'
    '2917,"p25_ns":944042,"p75_ns":1209250},"warmup":1},"reduce_r4_simd_sum":{"be'
    'st_gb_per_second":30.116187354487572,"bytes_per_sample":16777216,"calls_per_'
    'sample":64,"implementation":"reduce_r4_simd_sum","iqr_gb_per_second":[13.395'
    '4213271652,28.024287291599503],"iqr_ns_per_call":10215.5,"median_gb_per_seco'
    'nd":24.907421950969667,"median_ns_per_call":10524.734375,"repeats":7,"sample'
    's_ns":[1334334,916250,673583,621042,598667,1252459,557083],"timing":{"count"'
    ':7,"iqr_ns":653792,"max_ns":1334334,"median_ns":673583,"min_ns":557083,"p25_'
    'ns":598667,"p75_ns":1252459},"warmup":1},"reduce_r5_grid_stride":{"best_gb_p'
    'er_second":21.49547213324792,"bytes_per_sample":16777216,"calls_per_sample":'
    '64,"implementation":"reduce_r5_grid_stride","iqr_gb_per_second":[11.61689728'
    '9033158,18.793621206486332],"iqr_ns_per_call":8617.1875,"median_gb_per_secon'
    'd":15.155008712414446,"median_ns_per_call":17297.515625,"repeats":7,"samples'
    '_ns":[1107041,1492791,1444208,1248042,892708,1086208,780500],"timing":{"coun'
    't":7,"iqr_ns":551500,"max_ns":1492791,"median_ns":1107041,"min_ns":780500,"p'
    '25_ns":892708,"p75_ns":1444208},"warmup":1},"torch_sum":{"best_gb_per_second'
    '":26.86323097565896,"bytes_per_sample":16777216,"calls_per_sample":64,"imple'
    'mentation":"torch_sum","iqr_gb_per_second":[22.109233520418854,25.7582772560'
    '27256],"iqr_ns_per_call":1679.6875,"median_gb_per_second":23.727364397107554'
    ',"median_ns_per_call":11048.171875,"repeats":7,"samples_ns":[758833,707083,7'
    '55958,667375,651333,624542,1000959],"timing":{"count":7,"iqr_ns":107500,"max'
    '_ns":1000959,"median_ns":707083,"min_ns":624542,"p25_ns":651333,"p75_ns":758'
    '833},"warmup":1}},"sum_abs":52401.17979525441},{"bytes_per_call":4194304,"ch'
    'ecks":{"reduce_r0_interleaved":{"bound":1.0486026958789276,"bound_ratio":6.5'
    '43557714501535e-05,"depth":21,"error":6.861592260065663e-05,"exact_result":f'
    'alse,"output":1615.739501953125,"rule":"order_bound","same_bits_on_repeat":t'
    'rue},"reduce_r1_sequential":{"bound":1.0486026958789276,"bound_ratio":0.0002'
    '982602932739053,"depth":21,"error":0.00031275654760065663,"exact_result":fal'
    'se,"output":1615.73974609375,"rule":"order_bound","same_bits_on_repeat":true'
    '},"reduce_r2_first_add":{"bound":0.9986691746450113,"bound_ratio":6.87073600'
    '9754879e-05,"depth":20,"error":6.861592260065663e-05,"exact_result":false,"o'
    'utput":1615.739501953125,"rule":"order_bound","same_bits_on_repeat":true},"r'
    'educe_r3_simd_shuffle":{"bound":0.9986691746450113,"bound_ratio":6.870736009'
    '754879e-05,"depth":20,"error":6.861592260065663e-05,"exact_result":false,"ou'
    'tput":1615.739501953125,"rule":"order_bound","same_bits_on_repeat":true},"re'
    'duce_r4_simd_sum":{"bound":2.4966774010249386,"bound_ratio":7.03033168511425'
    '7e-05,"depth":50,"error":0.00017552470239934337,"exact_result":false,"output'
    '":1615.7392578125,"rule":"order_bound","same_bits_on_repeat":true},"reduce_r'
    '5_grid_stride":{"bound":2.5466111008357397,"bound_ratio":2.6944012997563097e'
    '-05,"depth":51,"error":6.861592260065663e-05,"exact_result":false,"output":1'
    '615.739501953125,"rule":"order_bound","same_bits_on_repeat":true},"torch_sum'
    '":{"bound":204.57714874453706,"bound_ratio":8.579878225721459e-07,"depth":40'
    '96,"error":0.00017552470239934337,"exact_result":false,"output":1615.7392578'
    '125,"rule":"declared_tolerance","same_bits_on_repeat":true}},"exact_in_any_o'
    'rder":false,"grid_exponent":-44,"index":1,"launches":{"reduce_r0_interleaved'
    '":{"active_threads":1048576,"extent":[1048576],"group_size":[256],"groups":['
    '4096],"idle_lanes_per_threadgroup":0,"launched_threads":1048576,"padding_thr'
    'eads":0,"simd_groups_per_threadgroup":8,"threads":[1048576],"threads_per_gro'
    'up":256},"reduce_r1_sequential":{"active_threads":1048576,"extent":[1048576]'
    ',"group_size":[256],"groups":[4096],"idle_lanes_per_threadgroup":0,"launched'
    '_threads":1048576,"padding_threads":0,"simd_groups_per_threadgroup":8,"threa'
    'ds":[1048576],"threads_per_group":256},"reduce_r2_first_add":{"active_thread'
    's":524288,"extent":[524288],"group_size":[256],"groups":[2048],"idle_lanes_p'
    'er_threadgroup":0,"launched_threads":524288,"padding_threads":0,"simd_groups'
    '_per_threadgroup":8,"threads":[524288],"threads_per_group":256},"reduce_r3_s'
    'imd_shuffle":{"active_threads":524288,"extent":[524288],"group_size":[256],"'
    'groups":[2048],"idle_lanes_per_threadgroup":0,"launched_threads":524288,"pad'
    'ding_threads":0,"simd_groups_per_threadgroup":8,"threads":[524288],"threads_'
    'per_group":256},"reduce_r4_simd_sum":{"active_threads":524288,"extent":[5242'
    '88],"group_size":[256],"groups":[2048],"idle_lanes_per_threadgroup":0,"launc'
    'hed_threads":524288,"padding_threads":0,"simd_groups_per_threadgroup":8,"thr'
    'eads":[524288],"threads_per_group":256},"reduce_r5_grid_stride":{"active_thr'
    'eads":262144,"extent":[262144],"group_size":[256],"groups":[1024],"idle_lane'
    's_per_threadgroup":0,"launched_threads":262144,"padding_threads":0,"simd_gro'
    'ups_per_threadgroup":8,"threads":[262144],"threads_per_group":256}},"n":1048'
    '576,"reference":1615.7394333372024,"side_order":["reduce_r0_interleaved","re'
    'duce_r1_sequential","reduce_r2_first_add","reduce_r3_simd_shuffle","reduce_r'
    '4_simd_sum","reduce_r5_grid_stride","torch_sum"],"sides":{"reduce_r0_interle'
    'aved":{"best_gb_per_second":18.51534390950476,"bytes_per_sample":16777216,"c'
    'alls_per_sample":4,"implementation":"reduce_r0_interleaved","iqr_gb_per_seco'
    'nd":[13.435659898070647,18.421318693384574],"iqr_ns_per_call":84489.5,"media'
    'n_gb_per_second":17.19491487207238,"median_ns_per_call":243927.0,"repeats":7'
    ',"samples_ns":[1082000,906125,910750,1365333,1248708,975708,919833],"timing"'
    ':{"count":7,"iqr_ns":337958,"max_ns":1365333,"median_ns":975708,"min_ns":906'
    '125,"p25_ns":910750,"p75_ns":1248708},"warmup":1},"reduce_r1_sequential":{"b'
    'est_gb_per_second":40.39052904002408,"bytes_per_sample":16777216,"calls_per_'
    'sample":4,"implementation":"reduce_r1_sequential","iqr_gb_per_second":[26.50'
    '257408287931,40.1889914674051],"iqr_ns_per_call":53895.75,"median_gb_per_sec'
    'ond":33.38749452736318,"median_ns_per_call":125625.0,"repeats":7,"samples_ns'
    '":[633041,502500,454959,615667,417458,778459,415375],"timing":{"count":7,"iq'
    'r_ns":215583,"max_ns":778459,"median_ns":502500,"min_ns":415375,"p25_ns":417'
    '458,"p75_ns":633041},"warmup":1},"reduce_r2_first_add":{"best_gb_per_second"'
    ':57.563065690425375,"bytes_per_sample":16777216,"calls_per_sample":4,"implem'
    'entation":"reduce_r2_first_add","iqr_gb_per_second":[34.03376352549304,57.02'
    '501631498804],"iqr_ns_per_call":49687.5,"median_gb_per_second":46.7277254471'
    '6217,"median_ns_per_call":89760.5,"repeats":7,"samples_ns":[359042,294208,30'
    '8125,440584,291458,492958,725459],"timing":{"count":7,"iqr_ns":198750,"max_n'
    's":725459,"median_ns":359042,"min_ns":291458,"p25_ns":294208,"p75_ns":492958'
    '},"warmup":1},"reduce_r3_simd_shuffle":{"best_gb_per_second":77.225390103567'
    '32,"bytes_per_sample":16777216,"calls_per_sample":4,"implementation":"reduce'
    '_r3_simd_shuffle","iqr_gb_per_second":[42.88106325878594,72.66796318354088],'
    '"iqr_ns_per_call":40093.75,"median_gb_per_second":68.85304944863276,"median_'
    'ns_per_call":60916.75,"repeats":7,"samples_ns":[378000,230875,243667,941625,'
    '217250,391250,242417],"timing":{"count":7,"iqr_ns":160375,"max_ns":941625,"m'
    'edian_ns":243667,"min_ns":217250,"p25_ns":230875,"p75_ns":391250},"warmup":1'
    '},"reduce_r4_simd_sum":{"best_gb_per_second":86.85387697627947,"bytes_per_sa'
    'mple":16777216,"calls_per_sample":4,"implementation":"reduce_r4_simd_sum","i'
    'qr_gb_per_second":[30.624690370970715,86.70395865633076],"iqr_ns_per_call":8'
    '8583.25,"median_gb_per_second":47.53870173440214,"median_ns_per_call":88229.'
    '25,"repeats":7,"samples_ns":[520542,193500,547833,560917,323625,352917,19316'
    '6],"timing":{"count":7,"iqr_ns":354333,"max_ns":560917,"median_ns":352917,"m'
    'in_ns":193166,"p25_ns":193500,"p75_ns":547833},"warmup":1},"reduce_r5_grid_s'
    'tride":{"best_gb_per_second":92.0560548696845,"bytes_per_sample":16777216,"c'
    'alls_per_sample":4,"implementation":"reduce_r5_grid_stride","iqr_gb_per_seco'
    'nd":[66.04111131230269,89.77774448160535],"iqr_ns_per_call":16791.75,"median'
    '_gb_per_second":82.5110336246453,"median_ns_per_call":50833.25,"repeats":7,"'
    'samples_ns":[451708,191375,203333,254042,217666,182250,186875],"timing":{"co'
    'unt":7,"iqr_ns":67167,"max_ns":451708,"median_ns":203333,"min_ns":182250,"p2'
    '5_ns":186875,"p75_ns":254042},"warmup":1},"torch_sum":{"best_gb_per_second":'
    '95.68772564206395,"bytes_per_sample":16777216,"calls_per_sample":4,"implemen'
    'tation":"torch_sum","iqr_gb_per_second":[39.313921499707085,88.6707538793285'
    '7],"iqr_ns_per_call":59385.5,"median_gb_per_second":77.94293147502904,"media'
    'n_ns_per_call":53812.5,"repeats":7,"samples_ns":[300916,189208,191917,426750'
    ',215250,175333,515042],"timing":{"count":7,"iqr_ns":237542,"max_ns":515042,"'
    'median_ns":215250,"min_ns":175333,"p25_ns":189208,"p75_ns":426750},"warmup":'
    '1}},"sum_abs":837743.4241088793}],"synchronization":"torch.mps.synchronize()'
    " before the first and before the second clock reading of every sample; untim"
    "ed warmups of each side first; repeat r starts with side r mod S, so the ord"
    'er shifts by one side every repeat","torch_equivalent":"torch.sum(x[:n], 0, '
    'out=out) on the MPS device, float32","vendor_rule":"torch.sum states no orde'
    "r, so its result gets the declared tolerance gamma_4096(u) * sum |x|; it is "
    'a tolerance, not a derived bound"},"ceiling":{"buffer_bytes":4194304,"config'
    '_sha256":"9e1db2639799674c80116c53f0ab2a97583f5400def85d536483da68aff6d173",'
    '"definition":"the bandwidth ceiling of a verified Day 74 gpu-ceilings report'
    ": the better median GB/s of stream_copy and stream_read_reduce at its larges"
    't buffer, on this host; not a vendor peak and not a hardware limit","environ'
    'ment":{"macos_version":"27.2","numpy_version":"1.26.4","platform_machine":"a'
    'rm64","platform_system":"Darwin","python_implementation":"CPython","python_v'
    'ersion":"3.11.14","torch_version":"2.15.0"},"fraction_rule":"median GB/s of '
    "a side / the bandwidth ceiling GB/s; a measurement only when the run and the"
    ' ceiling report are measurements from the same environment","gb_per_second":'
    '82.28275618954719,"kernel":"stream_read_reduce","label":"host-specific Apple'
    ' GPU (Metal) measurement","load_average_after":[9.4765625,9.40478515625,8.75'
    '341796875],"load_average_before":[9.4765625,9.40478515625,8.75341796875],"me'
    'asured":true,"report_sha256":"6b784c28e36b2890eadb65c742858b29835b90f8e259b4'
    'befba92a499c919cb6","same_environment":true,"source":"gpu_ceilings"},"config'
    '":{"benchmark":{"max_runtime_seconds":120,"repeats":7,"sizes":[65536,1048576'
    '],"target_bytes_per_sample":16777216,"warmup":1},"comparison":{"min_faster_r'
    'atio":1.1},"correctness":{"cases":[["normal",1],["normal",33],["positive",30'
    '0007],["mixed",1600001],["ones",1025],["halves",2049]],"runs":2},"name":"gpu'
    '-reduce-recorded","plan_sha256":"b2134229a87d220e40772e732ca3d3e5d6c6a6ae82a'
    '90a475eb0405d87a7217f","schema_version":1,"seed":78},"config_sha256":"650d70'
    'f00832de5372a7169d27f66d7559a48cd2e6e12a280eee1c2c7bcfd25b","correctness":{"'
    'bound_source":"N. J. Higham, Accuracy and Stability of Numerical Algorithms,'
    " 2nd ed., SIAM, 2002, section 4.2: an order whose values pass through at mos"
    't D roundings errs by at most gamma_D(u) * sum |x|, u = 2^-24","cases":[{"ex'
    'act_in_any_order":true,"grid_exponent":-25,"index":0,"input_guard":1024,"inp'
    'ut_sha256":"2638118e134ea710e92e6ab1ab457b9554cb7b13d5fd57c664353f139437b4fc'
    '","n":1,"outputs":[{"bound":4.0522607979972653e-07,"bound_ratio":0.0,"depth"'
    ':18,"error":0.0,"exact_result":true,"kernel":"reduce_r0_interleaved","launch'
    '":{"active_threads":1,"extent":[1],"group_size":[256],"groups":[1],"idle_lan'
    'es_per_threadgroup":0,"launched_threads":256,"padding_threads":255,"simd_gro'
    'ups_per_threadgroup":8,"threads":[256],"threads_per_group":256},"output":-0.'
    '3776976764202118,"output_guard":31,"partials":1,"partials_guard":64,"runs":2'
    ',"same_bits_every_run":true},{"bound":4.0522607979972653e-07,"bound_ratio":0'
    '.0,"depth":18,"error":0.0,"exact_result":true,"kernel":"reduce_r1_sequential'
    '","launch":{"active_threads":1,"extent":[1],"group_size":[256],"groups":[1],'
    '"idle_lanes_per_threadgroup":0,"launched_threads":256,"padding_threads":255,'
    '"simd_groups_per_threadgroup":8,"threads":[256],"threads_per_group":256},"ou'
    'tput":-0.3776976764202118,"output_guard":31,"partials":1,"partials_guard":64'
    ',"runs":2,"same_bits_every_run":true},{"bound":4.277386652838388e-07,"bound_'
    'ratio":0.0,"depth":19,"error":0.0,"exact_result":true,"kernel":"reduce_r2_fi'
    'rst_add","launch":{"active_threads":1,"extent":[1],"group_size":[256],"group'
    's":[1],"idle_lanes_per_threadgroup":0,"launched_threads":256,"padding_thread'
    's":255,"simd_groups_per_threadgroup":8,"threads":[256],"threads_per_group":2'
    '56},"output":-0.3776976764202118,"output_guard":31,"partials":1,"partials_gu'
    'ard":64,"runs":2,"same_bits_every_run":true},{"bound":4.277386652838388e-07,'
    '"bound_ratio":0.0,"depth":19,"error":0.0,"exact_result":true,"kernel":"reduc'
    'e_r3_simd_shuffle","launch":{"active_threads":1,"extent":[1],"group_size":[2'
    '56],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_threads":256,"padd'
    'ing_threads":255,"simd_groups_per_threadgroup":8,"threads":[256],"threads_pe'
    'r_group":256},"output":-0.3776976764202118,"output_guard":31,"partials":1,"p'
    'artials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":1.1031174777'
    '356854e-06,"bound_ratio":0.0,"depth":49,"error":0.0,"exact_result":true,"ker'
    'nel":"reduce_r4_simd_sum","launch":{"active_threads":1,"extent":[1],"group_s'
    'ize":[256],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_threads":25'
    '6,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads":[256],"thr'
    'eads_per_group":256},"output":-0.3776976764202118,"output_guard":31,"partial'
    's":1,"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":1.080'
    '6048117399888e-06,"bound_ratio":0.0,"depth":48,"error":0.0,"exact_result":tr'
    'ue,"kernel":"reduce_r5_grid_stride","launch":{"active_threads":262144,"exten'
    't":[262144],"group_size":[256],"groups":[1024],"idle_lanes_per_threadgroup":'
    '0,"launched_threads":262144,"padding_threads":0,"simd_groups_per_threadgroup'
    '":8,"threads":[262144],"threads_per_group":256},"output":-0.3776976764202118'
    ',"output_guard":31,"partials":1024,"partials_guard":64,"runs":2,"same_bits_e'
    'very_run":true}],"pattern":"normal","reference":-0.3776976764202118,"same_or'
    'der":[{"kernel":"reduce_r3_simd_shuffle","same_bits":true,"same_order_as":"r'
    'educe_r2_first_add"}],"sum_abs":0.3776976764202118},{"exact_in_any_order":fa'
    'lse,"grid_exponent":-26,"index":1,"input_guard":1024,"input_sha256":"1c38bd4'
    'c96d2fc7392a1507babf61fd037e93206956bc888b6dc187239a1d7d6","n":33,"outputs":'
    '[{"bound":3.256893002752126e-05,"bound_ratio":0.00869301086783824,"depth":18'
    ',"error":2.8312206268310547e-07,"exact_result":false,"kernel":"reduce_r0_int'
    'erleaved","launch":{"active_threads":33,"extent":[33],"group_size":[256],"gr'
    'oups":[1],"idle_lanes_per_threadgroup":0,"launched_threads":256,"padding_thr'
    'eads":223,"simd_groups_per_threadgroup":8,"threads":[256],"threads_per_group'
    '":256},"output":4.084628582000732,"output_guard":31,"partials":1,"partials_g'
    'uard":64,"runs":2,"same_bits_every_run":true},{"bound":3.256893002752126e-05'
    ',"bound_ratio":0.00869301086783824,"depth":18,"error":2.8312206268310547e-07'
    ',"exact_result":false,"kernel":"reduce_r1_sequential","launch":{"active_thre'
    'ads":33,"extent":[33],"group_size":[256],"groups":[1],"idle_lanes_per_thread'
    'group":0,"launched_threads":256,"padding_threads":223,"simd_groups_per_threa'
    'dgroup":8,"threads":[256],"threads_per_group":256},"output":4.08462858200073'
    '2,"output_guard":31,"partials":1,"partials_guard":64,"runs":2,"same_bits_eve'
    'ry_run":true},{"bound":3.437831707815979e-05,"bound_ratio":0.008235483489183'
    '655,"depth":19,"error":2.8312206268310547e-07,"exact_result":false,"kernel":'
    '"reduce_r2_first_add","launch":{"active_threads":17,"extent":[17],"group_siz'
    'e":[256],"groups":[1],"idle_lanes_per_threadgroup":0,"launched_threads":256,'
    '"padding_threads":239,"simd_groups_per_threadgroup":8,"threads":[256],"threa'
    'ds_per_group":256},"output":4.084628582000732,"output_guard":31,"partials":1'
    ',"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":3.4378317'
    '07815979e-05,"bound_ratio":0.008235483489183655,"depth":19,"error":2.8312206'
    '268310547e-07,"exact_result":false,"kernel":"reduce_r3_simd_shuffle","launch'
    '":{"active_threads":17,"extent":[17],"group_size":[256],"groups":[1],"idle_l'
    'anes_per_threadgroup":0,"launched_threads":256,"padding_threads":239,"simd_g'
    'roups_per_threadgroup":8,"threads":[256],"threads_per_group":256},"output":4'
    '.084628582000732,"output_guard":31,"partials":1,"partials_guard":64,"runs":2'
    ',"same_bits_every_run":true},{"bound":8.866002889613011e-05,"bound_ratio":0.'
    '003193345030541303,"depth":49,"error":2.8312206268310547e-07,"exact_result":'
    'false,"kernel":"reduce_r4_simd_sum","launch":{"active_threads":17,"extent":['
    '17],"group_size":[256],"groups":[1],"idle_lanes_per_threadgroup":0,"launched'
    '_threads":256,"padding_threads":239,"simd_groups_per_threadgroup":8,"threads'
    '":[256],"threads_per_group":256},"output":4.084628582000732,"output_guard":3'
    '1,"partials":1,"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bo'
    'und":8.685063537459474e-05,"bound_ratio":0.003259873246315057,"depth":48,"er'
    'ror":2.8312206268310547e-07,"exact_result":false,"kernel":"reduce_r5_grid_st'
    'ride","launch":{"active_threads":262144,"extent":[262144],"group_size":[256]'
    ',"groups":[1024],"idle_lanes_per_threadgroup":0,"launched_threads":262144,"p'
    'adding_threads":0,"simd_groups_per_threadgroup":8,"threads":[262144],"thread'
    's_per_group":256},"output":4.084628582000732,"output_guard":31,"partials":10'
    '24,"partials_guard":64,"runs":2,"same_bits_every_run":true}],"pattern":"norm'
    'al","reference":4.084628865122795,"same_order":[{"kernel":"reduce_r3_simd_sh'
    'uffle","same_bits":true,"same_order_as":"reduce_r2_first_add"}],"sum_abs":30'
    '.356410428881645},{"exact_in_any_order":false,"grid_exponent":-24,"index":2,'
    '"input_guard":1024,"input_sha256":"dd40cbf1d384f767722255476f25471090f193fb5'
    '64eab3e5dcd240dd4510d7b","n":300007,"outputs":[{"bound":0.1696590104660325,"'
    'bound_ratio":0.05356368952095581,"depth":19,"error":0.009087562561035156,"ex'
    'act_result":false,"kernel":"reduce_r0_interleaved","launch":{"active_threads'
    '":300007,"extent":[300007],"group_size":[256],"groups":[1172],"idle_lanes_pe'
    'r_threadgroup":0,"launched_threads":300032,"padding_threads":25,"simd_groups'
    '_per_threadgroup":8,"threads":[300032],"threads_per_group":256},"output":149'
    '810.65625,"output_guard":31,"partials":1172,"partials_guard":64,"runs":2,"sa'
    'me_bits_every_run":true},{"bound":0.1696590104660325,"bound_ratio":0.0535636'
    '8952095581,"depth":19,"error":0.009087562561035156,"exact_result":false,"ker'
    'nel":"reduce_r1_sequential","launch":{"active_threads":300007,"extent":[3000'
    '07],"group_size":[256],"groups":[1172],"idle_lanes_per_threadgroup":0,"launc'
    'hed_threads":300032,"padding_threads":25,"simd_groups_per_threadgroup":8,"th'
    'reads":[300032],"threads_per_group":256},"output":149810.65625,"output_guard'
    '":31,"partials":1172,"partials_guard":64,"runs":2,"same_bits_every_run":true'
    '},{"bound":0.1696590104660325,"bound_ratio":0.05356368952095581,"depth":19,"'
    'error":0.009087562561035156,"exact_result":false,"kernel":"reduce_r2_first_a'
    'dd","launch":{"active_threads":150004,"extent":[150004],"group_size":[256],"'
    'groups":[586],"idle_lanes_per_threadgroup":0,"launched_threads":150016,"padd'
    'ing_threads":12,"simd_groups_per_threadgroup":8,"threads":[150016],"threads_'
    'per_group":256},"output":149810.65625,"output_guard":31,"partials":586,"part'
    'ials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":0.1696590104660'
    '325,"bound_ratio":0.05356368952095581,"depth":19,"error":0.00908756256103515'
    '6,"exact_result":false,"kernel":"reduce_r3_simd_shuffle","launch":{"active_t'
    'hreads":150004,"extent":[150004],"group_size":[256],"groups":[586],"idle_lan'
    'es_per_threadgroup":0,"launched_threads":150016,"padding_threads":12,"simd_g'
    'roups_per_threadgroup":8,"threads":[150016],"threads_per_group":256},"output'
    '":149810.65625,"output_guard":31,"partials":586,"partials_guard":64,"runs":2'
    ',"same_bits_every_run":true},{"bound":0.43754244095803335,"bound_ratio":0.02'
    '076955675691078,"depth":49,"error":0.009087562561035156,"exact_result":false'
    ',"kernel":"reduce_r4_simd_sum","launch":{"active_threads":150004,"extent":[1'
    '50004],"group_size":[256],"groups":[586],"idle_lanes_per_threadgroup":0,"lau'
    'nched_threads":150016,"padding_threads":12,"simd_groups_per_threadgroup":8,"'
    'threads":[150016],"threads_per_group":256},"output":149810.65625,"output_gua'
    'rd":31,"partials":586,"partials_guard":64,"runs":2,"same_bits_every_run":tru'
    'e},{"bound":0.43754244095803335,"bound_ratio":0.02076955675691078,"depth":49'
    ',"error":0.009087562561035156,"exact_result":false,"kernel":"reduce_r5_grid_'
    'stride","launch":{"active_threads":262144,"extent":[262144],"group_size":[25'
    '6],"groups":[1024],"idle_lanes_per_threadgroup":0,"launched_threads":262144,'
    '"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[262144],"thre'
    'ads_per_group":256},"output":149810.65625,"output_guard":31,"partials":1024,'
    '"partials_guard":64,"runs":2,"same_bits_every_run":true}],"pattern":"positiv'
    'e","reference":149810.66533756256,"same_order":[{"kernel":"reduce_r3_simd_sh'
    'uffle","same_bits":true,"same_order_as":"reduce_r2_first_add"}],"sum_abs":14'
    '9810.66533756256},{"exact_in_any_order":false,"grid_exponent":-47,"index":3,'
    '"input_guard":1024,"input_sha256":"0536d36d5967c5bdcb62a7cd5f13422724045b869'
    'e8503f060f6df1b5d245d9d","n":1600001,"outputs":[{"bound":2355489.0849266183,'
    '"bound_ratio":0.0002893894775674426,"depth":24,"error":681.6537557027275,"ex'
    'act_result":false,"kernel":"reduce_r0_interleaved","launch":{"active_threads'
    '":1600001,"extent":[1600001],"group_size":[256],"groups":[6251],"idle_lanes_'
    'per_threadgroup":0,"launched_threads":1600256,"padding_threads":255,"simd_gr'
    'oups_per_threadgroup":8,"threads":[1600256],"threads_per_group":256},"output'
    '":-6252103680.0,"output_guard":31,"partials":6251,"partials_guard":64,"runs"'
    ':2,"same_bits_every_run":true},{"bound":2355489.0849266183,"bound_ratio":7.2'
    '02485326227389e-05,"depth":24,"error":169.6537557027275,"exact_result":false'
    ',"kernel":"reduce_r1_sequential","launch":{"active_threads":1600001,"extent"'
    ':[1600001],"group_size":[256],"groups":[6251],"idle_lanes_per_threadgroup":0'
    ',"launched_threads":1600256,"padding_threads":255,"simd_groups_per_threadgro'
    'up":8,"threads":[1600256],"threads_per_group":256},"output":-6252104192.0,"o'
    'utput_guard":31,"partials":6251,"partials_guard":64,"runs":2,"same_bits_ever'
    'y_run":true},{"bound":2159198.070452564,"bound_ratio":0.00031569764952589746'
    ',"depth":22,"error":681.6537557027275,"exact_result":false,"kernel":"reduce_'
    'r2_first_add","launch":{"active_threads":800001,"extent":[800001],"group_siz'
    'e":[256],"groups":[3126],"idle_lanes_per_threadgroup":0,"launched_threads":8'
    '00256,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads":[80025'
    '6],"threads_per_group":256},"output":-6252103680.0,"output_guard":31,"partia'
    'ls":3126,"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":2'
    '159198.070452564,"bound_ratio":0.00031569764952589746,"depth":22,"error":681'
    '.6537557027275,"exact_result":false,"kernel":"reduce_r3_simd_shuffle","launc'
    'h":{"active_threads":800001,"extent":[800001],"group_size":[256],"groups":[3'
    '126],"idle_lanes_per_threadgroup":0,"launched_threads":800256,"padding_threa'
    'ds":255,"simd_groups_per_threadgroup":8,"threads":[800256],"threads_per_grou'
    'p":256},"output":-6252103680.0,"output_guard":31,"partials":3126,"partials_g'
    'uard":64,"runs":2,"same_bits_every_run":true},{"bound":5103568.20151821,"bou'
    'nd_ratio":6.707978237567812e-05,"depth":52,"error":342.3462442972725,"exact_'
    'result":false,"kernel":"reduce_r4_simd_sum","launch":{"active_threads":80000'
    '1,"extent":[800001],"group_size":[256],"groups":[3126],"idle_lanes_per_threa'
    'dgroup":0,"launched_threads":800256,"padding_threads":255,"simd_groups_per_t'
    'hreadgroup":8,"threads":[800256],"threads_per_group":256},"output":-62521047'
    '04.0,"output_guard":31,"partials":3126,"partials_guard":64,"runs":2,"same_bi'
    'ts_every_run":true},{"bound":5299859.9179864805,"bound_ratio":0.000128617315'
    '59910756,"depth":54,"error":681.6537557027275,"exact_result":false,"kernel":'
    '"reduce_r5_grid_stride","launch":{"active_threads":262144,"extent":[262144],'
    '"group_size":[256],"groups":[1024],"idle_lanes_per_threadgroup":0,"launched_'
    'threads":262144,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads'
    '":[262144],"threads_per_group":256},"output":-6252103680.0,"output_guard":31'
    ',"partials":1024,"partials_guard":64,"runs":2,"same_bits_every_run":true}],"'
    'pattern":"mixed","reference":-6252104361.653756,"same_order":[{"kernel":"red'
    'uce_r3_simd_shuffle","same_bits":true,"same_order_as":"reduce_r2_first_add"}'
    '],"sum_abs":1646603859654.924},{"exact_in_any_order":true,"grid_exponent":0,'
    '"index":4,"input_guard":1024,"input_sha256":"9c09f7316fd2110a5a51fbe1fea2b98'
    'b16b20a7e63a27e0e3480a74c2586d4e2","n":1025,"outputs":[{"bound":0.0010997068'
    '759634357,"bound_ratio":0.0,"depth":18,"error":0.0,"exact_result":true,"kern'
    'el":"reduce_r0_interleaved","launch":{"active_threads":1025,"extent":[1025],'
    '"group_size":[256],"groups":[5],"idle_lanes_per_threadgroup":0,"launched_thr'
    'eads":1280,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads":['
    '1280],"threads_per_group":256},"output":1025.0,"output_guard":31,"partials":'
    '5,"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":0.001099'
    '7068759634357,"bound_ratio":0.0,"depth":18,"error":0.0,"exact_result":true,"'
    'kernel":"reduce_r1_sequential","launch":{"active_threads":1025,"extent":[102'
    '5],"group_size":[256],"groups":[5],"idle_lanes_per_threadgroup":0,"launched_'
    'threads":1280,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads'
    '":[1280],"threads_per_group":256},"output":1025.0,"output_guard":31,"partial'
    's":5,"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":0.001'
    '1608017715951002,"bound_ratio":0.0,"depth":19,"error":0.0,"exact_result":tru'
    'e,"kernel":"reduce_r2_first_add","launch":{"active_threads":513,"extent":[51'
    '3],"group_size":[256],"groups":[3],"idle_lanes_per_threadgroup":0,"launched_'
    'threads":768,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads"'
    ':[768],"threads_per_group":256},"output":1025.0,"output_guard":31,"partials"'
    ':3,"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":0.00116'
    '08017715951002,"bound_ratio":0.0,"depth":19,"error":0.0,"exact_result":true,'
    '"kernel":"reduce_r3_simd_shuffle","launch":{"active_threads":513,"extent":[5'
    '13],"group_size":[256],"groups":[3],"idle_lanes_per_threadgroup":0,"launched'
    '_threads":768,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads'
    '":[768],"threads_per_group":256},"output":1025.0,"output_guard":31,"partials'
    '":3,"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":0.0029'
    '936520271867115,"bound_ratio":0.0,"depth":49,"error":0.0,"exact_result":true'
    ',"kernel":"reduce_r4_simd_sum","launch":{"active_threads":513,"extent":[513]'
    ',"group_size":[256],"groups":[3],"idle_lanes_per_threadgroup":0,"launched_th'
    'reads":768,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads":['
    '768],"threads_per_group":256},"output":1025.0,"output_guard":31,"partials":3'
    ',"partials_guard":64,"runs":2,"same_bits_every_run":true},{"bound":0.0029325'
    '56913061847,"bound_ratio":0.0,"depth":48,"error":0.0,"exact_result":true,"ke'
    'rnel":"reduce_r5_grid_stride","launch":{"active_threads":262144,"extent":[26'
    '2144],"group_size":[256],"groups":[1024],"idle_lanes_per_threadgroup":0,"lau'
    'nched_threads":262144,"padding_threads":0,"simd_groups_per_threadgroup":8,"t'
    'hreads":[262144],"threads_per_group":256},"output":1025.0,"output_guard":31,'
    '"partials":1024,"partials_guard":64,"runs":2,"same_bits_every_run":true}],"p'
    'attern":"ones","reference":1025.0,"same_order":[{"kernel":"reduce_r3_simd_sh'
    'uffle","same_bits":true,"same_order_as":"reduce_r2_first_add"}],"sum_abs":10'
    '25.0},{"exact_in_any_order":true,"grid_exponent":-1,"index":5,"input_guard":'
    '1024,"input_sha256":"98365ec7324a24502b806f8d6317187ec5b6f4495383416756839e2'
    '9070d13b3","n":2049,"outputs":[{"bound":0.004624133302831617,"bound_ratio":0'
    '.0,"depth":18,"error":0.0,"exact_result":true,"kernel":"reduce_r0_interleave'
    'd","launch":{"active_threads":2049,"extent":[2049],"group_size":[256],"group'
    's":[9],"idle_lanes_per_threadgroup":0,"launched_threads":2304,"padding_threa'
    'ds":255,"simd_groups_per_threadgroup":8,"threads":[2304],"threads_per_group"'
    ':256},"output":-110.0,"output_guard":31,"partials":9,"partials_guard":64,"ru'
    'ns":2,"same_bits_every_run":true},{"bound":0.004624133302831617,"bound_ratio'
    '":0.0,"depth":18,"error":0.0,"exact_result":true,"kernel":"reduce_r1_sequent'
    'ial","launch":{"active_threads":2049,"extent":[2049],"group_size":[256],"gro'
    'ups":[9],"idle_lanes_per_threadgroup":0,"launched_threads":2304,"padding_thr'
    'eads":255,"simd_groups_per_threadgroup":8,"threads":[2304],"threads_per_grou'
    'p":256},"output":-110.0,"output_guard":31,"partials":9,"partials_guard":64,"'
    'runs":2,"same_bits_every_run":true},{"bound":0.004881029888365738,"bound_rat'
    'io":0.0,"depth":19,"error":0.0,"exact_result":true,"kernel":"reduce_r2_first'
    '_add","launch":{"active_threads":1025,"extent":[1025],"group_size":[256],"gr'
    'oups":[5],"idle_lanes_per_threadgroup":0,"launched_threads":1280,"padding_th'
    'reads":255,"simd_groups_per_threadgroup":8,"threads":[1280],"threads_per_gro'
    'up":256},"output":-110.0,"output_guard":31,"partials":5,"partials_guard":64,'
    '"runs":2,"same_bits_every_run":true},{"bound":0.004881029888365738,"bound_ra'
    'tio":0.0,"depth":19,"error":0.0,"exact_result":true,"kernel":"reduce_r3_simd'
    '_shuffle","launch":{"active_threads":1025,"extent":[1025],"group_size":[256]'
    ',"groups":[5],"idle_lanes_per_threadgroup":0,"launched_threads":1280,"paddin'
    'g_threads":255,"simd_groups_per_threadgroup":8,"threads":[1280],"threads_per'
    '_group":256},"output":-110.0,"output_guard":31,"partials":5,"partials_guard"'
    ':64,"runs":2,"same_bits_every_run":true},{"bound":0.012587941694804611,"boun'
    'd_ratio":0.0,"depth":49,"error":0.0,"exact_result":true,"kernel":"reduce_r4_'
    'simd_sum","launch":{"active_threads":1025,"extent":[1025],"group_size":[256]'
    ',"groups":[5],"idle_lanes_per_threadgroup":0,"launched_threads":1280,"paddin'
    'g_threads":255,"simd_groups_per_threadgroup":8,"threads":[1280],"threads_per'
    '_group":256},"output":-110.0,"output_guard":31,"partials":5,"partials_guard"'
    ':64,"runs":2,"same_bits_every_run":true},{"bound":0.01233104419053323,"bound'
    '_ratio":0.0,"depth":48,"error":0.0,"exact_result":true,"kernel":"reduce_r5_g'
    'rid_stride","launch":{"active_threads":262144,"extent":[262144],"group_size"'
    ':[256],"groups":[1024],"idle_lanes_per_threadgroup":0,"launched_threads":262'
    '144,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[262144],"'
    'threads_per_group":256},"output":-110.0,"output_guard":31,"partials":1024,"p'
    'artials_guard":64,"runs":2,"same_bits_every_run":true}],"pattern":"halves","'
    'reference":-110.0,"same_order":[{"kernel":"reduce_r3_simd_shuffle","same_bit'
    's":true,"same_order_as":"reduce_r2_first_add"}],"sum_abs":4310.0}],"combine_'
    'launch":{"active_threads":1024,"extent":[1024],"group_size":[1024],"groups":'
    '[1],"idle_lanes_per_threadgroup":0,"launched_threads":1024,"padding_threads"'
    ':0,"simd_groups_per_threadgroup":32,"threads":[1024],"threads_per_group":102'
    '4},"depth_rule":"D = thread chain + group tree + combine chain + combine tre'
    "e: chain 0 for one element, 1 for two, ceil(n / 262144) - 1 for the R5 loop;"
    " tree 8 for the threadgroup trees of R0 to R3, 31 + 7 for the two simd_sum s"
    "teps of R4 and R5; combine chain ceil(G / 1024) - 1 for G partials; combine "
    'tree 10","exact_rule":"when sum |x| <= 2^24 * 2^g, with 2^g the finest power'
    " of two that divides every value, every partial sum of every order is exact "
    'in float32, so the result must equal the exact sum","reference":"the exact s'
    "um of the float32 inputs, from their integer significands, rounded once to f"
    'loat64","rules":"every partial slot is written, the partial and output tails'
    " keep their sentinels, and the result is finite, within the bound of its ord"
    "er and equal to the exact sum when every order is exact; same_bits_every_run"
    ' and same_order are measured, not required","sentinel_float32_bits":21432893'
    '44,"unit_roundoff":"2^-24"},"environment":{"macos_version":"27.2","numpy_ver'
    'sion":"1.26.4","platform_machine":"arm64","platform_system":"Darwin","python'
    '_implementation":"CPython","python_version":"3.11.14","torch_version":"2.15.'
    '0"},"evidence_boundary":{"apple_gpu_metal_this_host":true,"cuda_kernel_execu'
    'ted":false,"distributed_claim":false,"hardware_counters_used":false,"host_sp'
    'ecific":true,"nvidia_gpu_used":false,"occupancy_measured":false,"portable_pe'
    'rformance_claim":false,"production_claim":false,"statement":"These are wall-'
    "clock timings on the Apple GPU of the host that made this report, at the rec"
    "orded load averages. The reduction kernels came from torch.mps.compile_shade"
    "r, and torch.sum ran on the MPS device. PyTorch does not report which kernel"
    " torch.sum ran, so the vendor number names the call, not a kernel. simd_shuf"
    "fle_down and simd_sum are the Metal counterpart of CUDA warp shuffles, not t"
    "he same instructions. No hardware counter was read, so a limit named from th"
    "ese timings is an inference. The bandwidth ceiling comes from a separate Day"
    " 74 run on this host, and it is not a vendor peak. No CUDA kernel ran, and n"
    "o NVIDIA GPU was used. These timings do not measure or predict CUDA, NVIDIA,"
    ' warp-level, distributed or production performance.","vendor_kernel_known":f'
    'alse,"vendor_peak_claimed":false,"warp_level_claim":false},"host_load":{"aft'
    'er":[11.58984375,11.216796875,12.97265625],"before":[11.58984375,11.21679687'
    '5,12.97265625],"source":"os.getloadavg(): the 1, 5 and 15 minute load averag'
    'es that uptime prints, read before the checks and after the last sample"},"k'
    'ind":"gpu_reduce_correctness_and_bandwidth","label":"host-specific Apple GPU'
    ' (Metal) measurement","measured":true,"measurement_scope":"apple_gpu_metal_t'
    'his_host","percentile_method":"nearest rank: sorted samples[ceil(p * n / 100'
    ") - 1] for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no interp"
    'olation","plan":{"reference_kernel":"reduce_r0_interleaved","sha256":"b21342'
    '29a87d220e40772e732ca3d3e5d6c6a6ae82a90a475eb0405d87a7217f","size_classes":['
    '{"first_n":1,"last_n":262143,"name":"small"},{"first_n":262144,"last_n":4194'
    '303,"name":"medium"},{"first_n":4194304,"last_n":67108864,"name":"large"}],"'
    'steps":[{"base":"torch_sum","change":"Load one element per thread. Then add '
    "pairs in threadgroup memory with interleaved addressing. At stride s, only t"
    'he threads with t % (2 s) == 0 add, and s doubles from 1 to 128.","hypothesi'
    's":"R0 is slower than torch.sum from 2^18 elements. At each level, the modul'
    "o test keeps most lanes of every SIMD group idle. Every SIMD group still tak"
    "es the branch, and a group waits at 9 barriers for each 256 elements. Below "
    "2^18 elements, the host dispatch of a call probably sets the time. R0 dispat"
    "ches two short kernels, so it is faster than torch.sum or shows no clear dif"
    'ference.","kernel":"reduce_r0_interleaved","predictions":[{"size_class":"sma'
    'll","verdicts":["faster","no_clear_difference"]},{"size_class":"medium","ver'
    'dicts":["slower"]},{"size_class":"large","verdicts":["slower"]}],"same_order'
    '_as":null,"source_sha256":"40303012a3f4b3bcdd1909023d2ee8af860d79dbbde9cbf07'
    '14af7ece8f8f6c2","step":"R0"},{"base":"reduce_r0_interleaved","change":"Add '
    "pairs with sequential addressing: at stride s, the threads t < s add shared["
    't + s], and s halves from 128 to 1.","hypothesis":"R1 is faster than R0 by 1'
    ".1 or more from 2^18 elements. While s is 32 or more, a whole SIMD group tak"
    "es the branch or skips it. Thus only the last 5 levels diverge, and no threa"
    "d calculates a modulo. Below 2^18 elements, the dispatch probably sets the t"
    'ime, so R1 shows no clear difference from R0 or is faster.","kernel":"reduce'
    '_r1_sequential","predictions":[{"size_class":"small","verdicts":["no_clear_d'
    'ifference","faster"]},{"size_class":"medium","verdicts":["faster"]},{"size_c'
    'lass":"large","verdicts":["faster"]}],"same_order_as":null,"source_sha256":"'
    'b230db3f18d0db381928728e77c376adc34532c49d421d4cd36c4adc6e244173","step":"R1'
    '"},{"base":"reduce_r1_sequential","change":"Give each thread two elements 25'
    "6 apart, and add them during the global load. One group then covers 512 elem"
    'ents, and the grid has half the groups.","hypothesis":"R2 is faster than R1 '
    "by 1.1 or more from 2^18 elements. Each thread has two loads in flight befor"
    "e its first barrier, so more loads are in flight. Also, the tree work for ea"
    "ch element halves. Below 2^18 elements, R2 shows no clear difference from R1"
    '.","kernel":"reduce_r2_first_add","predictions":[{"size_class":"small","verd'
    'icts":["no_clear_difference"]},{"size_class":"medium","verdicts":["faster"]}'
    ',{"size_class":"large","verdicts":["faster"]}],"same_order_as":null,"source_'
    'sha256":"d52e657352c87e5cd8ae8baf6ce45cc92c08cb5ed2f887d7f7f58f1c48f414a1","'
    'step":"R2"},{"base":"reduce_r2_first_add","change":"Stop the threadgroup tre'
    "e at 32 values. Then add them in SIMD group 0 with simd_shuffle_down at delt"
    'as 16, 8, 4, 2 and 1, with no barrier.","hypothesis":"R3 removes 5 of the 9 '
    "barriers of a group. From 2^22 elements, the device loads probably set the t"
    "ime of R2, so R3 shows no clear difference from R2. From 2^18 to 2^22 elemen"
    "ts, a cache can serve the input, so the barriers can matter, and R3 is faste"
    "r or shows no clear difference. Below 2^18 elements, R3 shows no clear diffe"
    "rence. R3 adds the same pairs in the same order as R2, so each result keeps "
    'the bits of R2.","kernel":"reduce_r3_simd_shuffle","predictions":[{"size_cla'
    'ss":"small","verdicts":["no_clear_difference"]},{"size_class":"medium","verd'
    'icts":["no_clear_difference","faster"]},{"size_class":"large","verdicts":["n'
    'o_clear_difference"]}],"same_order_as":"reduce_r2_first_add","source_sha256"'
    ':"9ddceaf22450689e0761dbfc5a81ffca455866fc2772e14eff229bebb875f31a","step":"'
    'R3"},{"base":"reduce_r3_simd_shuffle","change":"Add the two loaded values, a'
    "nd add each SIMD group with simd_sum. Write the 8 group sums to threadgroup "
    "memory, and wait at one barrier. Then SIMD group 0 adds the 8 sums with simd"
    '_sum.","hypothesis":"R4 waits at 1 barrier for each group instead of 4. From'
    " 2^22 elements, the loads probably set the time, so R4 shows no clear differ"
    "ence from R3. From 2^18 to 2^22 elements, R4 is faster or shows no clear dif"
    "ference. Below 2^18 elements, R4 shows no clear difference. simd_sum does no"
    "t state its order of additions. Thus the bound of R4 allows any order in a S"
    'IMD group, and its bits can differ from R3.","kernel":"reduce_r4_simd_sum","'
    'predictions":[{"size_class":"small","verdicts":["no_clear_difference"]},{"si'
    'ze_class":"medium","verdicts":["no_clear_difference","faster"]},{"size_class'
    '":"large","verdicts":["no_clear_difference"]}],"same_order_as":null,"source_'
    'sha256":"8ee154263d19714e61dd6849d78ff74afdbb5e702194248f5c7c65b69dd53db0","'
    'step":"R4"},{"base":"reduce_r4_simd_sum","change":"Launch a fixed grid of 10'
    "24 groups. Each thread adds the elements gid, gid + 262144, gid + 2 * 262144"
    " and so on in one loop. Then it reduces as R4 does. The combine pass thus al"
    'ways adds 1024 partials.","hypothesis":"R5 shows no clear difference from R4'
    " from 2^18 elements. Both read each element once, and the combine pass of R4"
    " reads at most 131072 partials, 1/512 of the input. Below 2^18 elements, mos"
    "t R5 threads have no element, so R5 is slower or shows no clear difference. "
    "Each R5 thread adds up to n / 262144 elements in one chain, so the bound of "
    'R5 grows with n.","kernel":"reduce_r5_grid_stride","predictions":[{"size_cla'
    'ss":"small","verdicts":["no_clear_difference","slower"]},{"size_class":"medi'
    'um","verdicts":["no_clear_difference"]},{"size_class":"large","verdicts":["n'
    'o_clear_difference"]}],"same_order_as":null,"source_sha256":"fb97a05461e22da'
    'fc3a0c345051a73bcba9088d9e9e14dcefb29114917713460","step":"R5"}],"vendor":"t'
    'orch_sum"},"planned_work":{"benchmark_points":2,"bytes_read":1879048192,"cal'
    'ls":3808,"correctness_cases":6,"correctness_elements":1903116,"correctness_l'
    'aunches":144,"sides_per_point":7,"timed_samples":98},"report_sha256":"54666a'
    '185b431d201d2c3a7719077f401df8a142957974fdbd21153f42c96be9","run":{"elapsed_'
    'ns":786126125,"max_runtime_ns":120000000000,"timed_ns":83959082},"schema_ver'
    'sion":1,"shaders":{"kernels":[{"admission":{"admitted":true,"allowed_threads'
    '":1024,"declared_threadgroup_bytes":1024,"name":"reduce_r0_interleaved","rea'
    'son":null,"required_simd_width":null,"required_threads":256,"simd_width":32,'
    '"static_threadgroup_bytes":1024},"fixed_groups":null,"group_size":256,"limit'
    's":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":102'
    '4,"thread_execution_width":32},"mapping":"one_element_per_thread","name":"re'
    'duce_r0_interleaved","simd_lanes":null,"source_sha256":"40303012a3f4b3bcdd19'
    '09023d2ee8af860d79dbbde9cbf0714af7ece8f8f6c2","step":"R0","threadgroup_bytes'
    '":1024,"tree":"interleaved_tree"},{"admission":{"admitted":true,"allowed_thr'
    'eads":1024,"declared_threadgroup_bytes":1024,"name":"reduce_r1_sequential","'
    'reason":null,"required_simd_width":null,"required_threads":256,"simd_width":'
    '32,"static_threadgroup_bytes":1024},"fixed_groups":null,"group_size":256,"li'
    'mits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":'
    '1024,"thread_execution_width":32},"mapping":"one_element_per_thread","name":'
    '"reduce_r1_sequential","simd_lanes":null,"source_sha256":"b230db3f18d0db3819'
    '28728e77c376adc34532c49d421d4cd36c4adc6e244173","step":"R1","threadgroup_byt'
    'es":1024,"tree":"sequential_tree"},{"admission":{"admitted":true,"allowed_th'
    'reads":1024,"declared_threadgroup_bytes":1024,"name":"reduce_r2_first_add","'
    'reason":null,"required_simd_width":null,"required_threads":256,"simd_width":'
    '32,"static_threadgroup_bytes":1024},"fixed_groups":null,"group_size":256,"li'
    'mits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":'
    '1024,"thread_execution_width":32},"mapping":"two_elements_256_apart","name":'
    '"reduce_r2_first_add","simd_lanes":null,"source_sha256":"d52e657352c87e5cd8a'
    'e8baf6ce45cc92c08cb5ed2f887d7f7f58f1c48f414a1","step":"R2","threadgroup_byte'
    's":1024,"tree":"sequential_tree"},{"admission":{"admitted":true,"allowed_thr'
    'eads":1024,"declared_threadgroup_bytes":1024,"name":"reduce_r3_simd_shuffle"'
    ',"reason":null,"required_simd_width":32,"required_threads":256,"simd_width":'
    '32,"static_threadgroup_bytes":1024},"fixed_groups":null,"group_size":256,"li'
    'mits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":'
    '1024,"thread_execution_width":32},"mapping":"two_elements_256_apart","name":'
    '"reduce_r3_simd_shuffle","simd_lanes":32,"source_sha256":"9ddceaf22450689e07'
    '61dbfc5a81ffca455866fc2772e14eff229bebb875f31a","step":"R3","threadgroup_byt'
    'es":1024,"tree":"sequential_tree_then_simd_shuffle_down"},{"admission":{"adm'
    'itted":true,"allowed_threads":1024,"declared_threadgroup_bytes":32,"name":"r'
    'educe_r4_simd_sum","reason":null,"required_simd_width":32,"required_threads"'
    ':256,"simd_width":32,"static_threadgroup_bytes":32},"fixed_groups":null,"gro'
    'up_size":256,"limits":{"max_threads_per_threadgroup":1024,"static_threadgrou'
    'p_memory_bytes":32,"thread_execution_width":32},"mapping":"two_elements_256_'
    'apart","name":"reduce_r4_simd_sum","simd_lanes":32,"source_sha256":"8ee15426'
    '3d19714e61dd6849d78ff74afdbb5e702194248f5c7c65b69dd53db0","step":"R4","threa'
    'dgroup_bytes":32,"tree":"simd_sum_then_simd_sum"},{"admission":{"admitted":t'
    'rue,"allowed_threads":1024,"declared_threadgroup_bytes":32,"name":"reduce_r5'
    '_grid_stride","reason":null,"required_simd_width":32,"required_threads":256,'
    '"simd_width":32,"static_threadgroup_bytes":32},"fixed_groups":1024,"group_si'
    'ze":256,"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_mem'
    'ory_bytes":32,"thread_execution_width":32},"mapping":"grid_stride_loop","nam'
    'e":"reduce_r5_grid_stride","simd_lanes":32,"source_sha256":"fb97a05461e22daf'
    'c3a0c345051a73bcba9088d9e9e14dcefb29114917713460","step":"R5","threadgroup_b'
    'ytes":32,"tree":"simd_sum_then_simd_sum"},{"admission":{"admitted":true,"all'
    'owed_threads":1024,"declared_threadgroup_bytes":4096,"name":"reduce_combine"'
    ',"reason":null,"required_simd_width":null,"required_threads":1024,"simd_widt'
    'h":32,"static_threadgroup_bytes":4096},"fixed_groups":null,"group_size":1024'
    ',"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_byt'
    'es":4096,"thread_execution_width":32},"mapping":"grid_stride_in_one_group","'
    'name":"reduce_combine","simd_lanes":null,"source_sha256":"551bea995a2f88e2d9'
    '1426e40862f8bf65e58c2de9e76a1a7d537b006233d770","step":null,"threadgroup_byt'
    'es":4096,"tree":"sequential_tree"}],"library_sha256":"753b102d673402b6ce260b'
    '644639d1f09a8c81f16647e831b2e35c8ac54f7edc"},"speedup":{"best_rule":"the ste'
    "p kernel with the highest median GB/s at this size; a tie goes to the earlie"
    'r step","ceiling_gb_per_second":82.28275618954719,"fraction_rule":"median GB'
    "/s of a side / the measured bandwidth ceiling, and median GB/s of a kernel /"
    ' median GB/s of torch.sum at the same size of the same run","hypotheses":[{"'
    'base":"torch_sum","held_at":[1],"kernel":"reduce_r0_interleaved","missed_at"'
    ':[0],"step":"R0"},{"base":"reduce_r0_interleaved","held_at":[0,1],"kernel":"'
    'reduce_r1_sequential","missed_at":[],"step":"R1"},{"base":"reduce_r1_sequent'
    'ial","held_at":[0],"kernel":"reduce_r2_first_add","missed_at":[1],"step":"R2'
    '"},{"base":"reduce_r2_first_add","held_at":[0,1],"kernel":"reduce_r3_simd_sh'
    'uffle","missed_at":[],"step":"R3"},{"base":"reduce_r3_simd_shuffle","held_at'
    '":[0,1],"kernel":"reduce_r4_simd_sum","missed_at":[],"step":"R4"},{"base":"r'
    'educe_r4_simd_sum","held_at":[0,1],"kernel":"reduce_r5_grid_stride","missed_'
    'at":[],"step":"R5"}],"hypothesis_rule":"a prediction holds at a size when th'
    "e verdict of the step's kernel over its base is one of the verdicts that the"
    ' plan states for the size class of n","min_faster_ratio":1.1,"points":[{"bes'
    't":{"fraction_of_ceiling":0.3027052459642059,"fraction_of_torch_sum":1.04973'
    '40342615535,"iqr_gb_per_second":[13.3954213271652,28.024287291599503],"kerne'
    'l":"reduce_r4_simd_sum","median_gb_per_second":24.907421950969667,"to_r0":{"'
    'first":"reduce_r4_simd_sum","iqr_separated":true,"ratio":2.75510367690396,"s'
    'econd":"reduce_r0_interleaved","verdict":"faster"}},"index":0,"n":65536,"row'
    's":[{"base":"torch_sum","fraction_of_ceiling":0.10987072773405393,"fraction_'
    'of_torch_sum":0.3810143491373759,"held":false,"kernel":"reduce_r0_interleave'
    'd","predicted":["faster","no_clear_difference"],"rate":{"iqr_gb_per_second":'
    '[5.779101121428109,9.100740981828045],"median_gb_per_second":9.0404663025092'
    '8},"step":"R0","to_base":{"first":"reduce_r0_interleaved","iqr_separated":tr'
    'ue,"ratio":0.3810143491373759,"second":"torch_sum","verdict":"slower"},"to_r'
    '0":null},{"base":"reduce_r0_interleaved","fraction_of_ceiling":0.17237417959'
    '827342,"fraction_of_torch_sum":0.5977664588396914,"held":true,"kernel":"redu'
    'ce_r1_sequential","predicted":["no_clear_difference","faster"],"rate":{"iqr_'
    'gb_per_second":[8.07518969977739,19.572867364235908],"median_gb_per_second":'
    '14.183422593257951},"step":"R1","to_base":{"first":"reduce_r1_sequential","i'
    'qr_separated":false,"ratio":1.5688817499735812,"second":"reduce_r0_interleav'
    'ed","verdict":"no_clear_difference"},"to_r0":{"first":"reduce_r1_sequential"'
    ',"iqr_separated":false,"ratio":1.5688817499735812,"second":"reduce_r0_interl'
    'eaved","verdict":"no_clear_difference"}},{"base":"reduce_r1_sequential","fra'
    'ction_of_ceiling":0.14093867435053078,"fraction_of_torch_sum":0.488753086317'
    '34943,"held":true,"kernel":"reduce_r2_first_add","predicted":["no_clear_diff'
    'erence"],"rate":{"iqr_gb_per_second":[10.270452795214325,23.386953824708137]'
    ',"median_gb_per_second":11.596822579262712},"step":"R2","to_base":{"first":"'
    'reduce_r2_first_add","iqr_separated":false,"ratio":0.8176321690348017,"secon'
    'd":"reduce_r1_sequential","verdict":"no_clear_difference"},"to_r0":{"first":'
    '"reduce_r2_first_add","iqr_separated":true,"ratio":1.2827681881900148,"secon'
    'd":"reduce_r0_interleaved","verdict":"faster"}},{"base":"reduce_r2_first_add'
    '","fraction_of_ceiling":0.1866476882841513,"fraction_of_torch_sum":0.6472647'
    '349867313,"held":true,"kernel":"reduce_r3_simd_shuffle","predicted":["no_cle'
    'ar_difference"],"rate":{"iqr_gb_per_second":[13.874067397146993,17.771683886'
    '94571],"median_gb_per_second":15.357886228427423},"step":"R3","to_base":{"fi'
    'rst":"reduce_r3_simd_shuffle","iqr_separated":false,"ratio":1.32431846080754'
    '9,"second":"reduce_r2_first_add","verdict":"no_clear_difference"},"to_r0":{"'
    'first":"reduce_r3_simd_shuffle","iqr_separated":true,"ratio":1.6987935925566'
    '886,"second":"reduce_r0_interleaved","verdict":"faster"}},{"base":"reduce_r3'
    '_simd_shuffle","fraction_of_ceiling":0.3027052459642059,"fraction_of_torch_s'
    'um":1.0497340342615535,"held":true,"kernel":"reduce_r4_simd_sum","predicted"'
    ':["no_clear_difference"],"rate":{"iqr_gb_per_second":[13.3954213271652,28.02'
    '4287291599503],"median_gb_per_second":24.907421950969667},"step":"R4","to_ba'
    'se":{"first":"reduce_r4_simd_sum","iqr_separated":false,"ratio":1.6218001345'
    '045823,"second":"reduce_r3_simd_shuffle","verdict":"no_clear_difference"},"t'
    'o_r0":{"first":"reduce_r4_simd_sum","iqr_separated":true,"ratio":2.755103676'
    '90396,"second":"reduce_r0_interleaved","verdict":"faster"}},{"base":"reduce_'
    'r4_simd_sum","fraction_of_ceiling":0.1841820742793697,"fraction_of_torch_sum'
    '":0.6387143746256914,"held":true,"kernel":"reduce_r5_grid_stride","predicted'
    '":["no_clear_difference","slower"],"rate":{"iqr_gb_per_second":[11.616897289'
    '033158,18.793621206486332],"median_gb_per_second":15.155008712414446},"step"'
    ':"R5","to_base":{"first":"reduce_r5_grid_stride","iqr_separated":false,"rati'
    'o":0.608453526111499,"second":"reduce_r4_simd_sum","verdict":"no_clear_diffe'
    'rence"},"to_r0":{"first":"reduce_r5_grid_stride","iqr_separated":true,"ratio'
    '":1.6763525470149707,"second":"reduce_r0_interleaved","verdict":"faster"}}],'
    '"size_class":"small","torch_sum":{"fraction_of_ceiling":0.28836375318358337,'
    '"iqr_gb_per_second":[22.109233520418854,25.758277256027256],"median_gb_per_s'
    'econd":23.727364397107554}},{"best":{"fraction_of_ceiling":1.002774304674144'
    ',"fraction_of_torch_sum":1.05860829280048,"iqr_gb_per_second":[66.0411113123'
    '0269,89.77774448160535],"kernel":"reduce_r5_grid_stride","median_gb_per_seco'
    'nd":82.5110336246453,"to_r0":{"first":"reduce_r5_grid_stride","iqr_separated'
    '":true,"ratio":4.7985718009373795,"second":"reduce_r0_interleaved","verdict"'
    ':"faster"}},"index":1,"n":1048576,"rows":[{"base":"torch_sum","fraction_of_c'
    'eiling":0.20897349175399574,"fraction_of_torch_sum":0.22060903467020868,"hel'
    'd":true,"kernel":"reduce_r0_interleaved","predicted":["slower"],"rate":{"iqr'
    '_gb_per_second":[13.435659898070647,18.421318693384574],"median_gb_per_secon'
    'd":17.19491487207238},"step":"R0","to_base":{"first":"reduce_r0_interleaved"'
    ',"iqr_separated":true,"ratio":0.22060903467020868,"second":"torch_sum","verd'
    'ict":"slower"},"to_r0":null},{"base":"reduce_r0_interleaved","fraction_of_ce'
    'iling":0.40576538844240334,"fraction_of_torch_sum":0.4283582089552238,"held"'
    ':true,"kernel":"reduce_r1_sequential","predicted":["faster"],"rate":{"iqr_gb'
    '_per_second":[26.50257408287931,40.1889914674051],"median_gb_per_second":33.'
    '38749452736318},"step":"R1","to_base":{"first":"reduce_r1_sequential","iqr_s'
    'eparated":true,"ratio":1.9417074626865671,"second":"reduce_r0_interleaved","'
    'verdict":"faster"},"to_r0":{"first":"reduce_r1_sequential","iqr_separated":t'
    'rue,"ratio":1.9417074626865671,"second":"reduce_r0_interleaved","verdict":"f'
    'aster"}},{"base":"reduce_r1_sequential","fraction_of_ceiling":0.567892078621'
    '1855,"fraction_of_torch_sum":0.5995120348037277,"held":false,"kernel":"reduc'
    'e_r2_first_add","predicted":["faster"],"rate":{"iqr_gb_per_second":[34.03376'
    '352549304,57.02501631498804],"median_gb_per_second":46.72772544716217},"step'
    '":"R2","to_base":{"first":"reduce_r2_first_add","iqr_separated":false,"ratio'
    '":1.3995577119111415,"second":"reduce_r1_sequential","verdict":"no_clear_dif'
    'ference"},"to_r0":{"first":"reduce_r2_first_add","iqr_separated":true,"ratio'
    '":2.7175316536784,"second":"reduce_r0_interleaved","verdict":"faster"}},{"ba'
    'se":"reduce_r2_first_add","fraction_of_ceiling":0.8367858909590042,"fraction'
    '_of_torch_sum":0.8833777245174765,"held":true,"kernel":"reduce_r3_simd_shuff'
    'le","predicted":["no_clear_difference","faster"],"rate":{"iqr_gb_per_second"'
    ':[42.88106325878594,72.66796318354088],"median_gb_per_second":68.85304944863'
    '276},"step":"R3","to_base":{"first":"reduce_r3_simd_shuffle","iqr_separated"'
    ':false,"ratio":1.4734945643029214,"second":"reduce_r2_first_add","verdict":"'
    'no_clear_difference"},"to_r0":{"first":"reduce_r3_simd_shuffle","iqr_separat'
    'ed":true,"ratio":4.004268120016252,"second":"reduce_r0_interleaved","verdict'
    '":"faster"}},{"base":"reduce_r3_simd_shuffle","fraction_of_ceiling":0.577748'
    '047536128,"fraction_of_torch_sum":0.6099167792993819,"held":true,"kernel":"r'
    'educe_r4_simd_sum","predicted":["no_clear_difference","faster"],"rate":{"iqr'
    '_gb_per_second":[30.624690370970715,86.70395865633076],"median_gb_per_second'
    '":47.53870173440214},"step":"R4","to_base":{"first":"reduce_r4_simd_sum","iq'
    'r_separated":false,"ratio":0.6904371282766203,"second":"reduce_r3_simd_shuff'
    'le","verdict":"no_clear_difference"},"to_r0":{"first":"reduce_r4_simd_sum","'
    'iqr_separated":true,"ratio":2.764695381633642,"second":"reduce_r0_interleave'
    'd","verdict":"faster"}},{"base":"reduce_r4_simd_sum","fraction_of_ceiling":1'
    '.002774304674144,"fraction_of_torch_sum":1.05860829280048,"held":true,"kerne'
    'l":"reduce_r5_grid_stride","predicted":["no_clear_difference"],"rate":{"iqr_'
    'gb_per_second":[66.04111131230269,89.77774448160535],"median_gb_per_second":'
    '82.5110336246453},"step":"R5","to_base":{"first":"reduce_r5_grid_stride","iq'
    'r_separated":false,"ratio":1.7356602223938074,"second":"reduce_r4_simd_sum",'
    '"verdict":"no_clear_difference"},"to_r0":{"first":"reduce_r5_grid_stride","i'
    'qr_separated":true,"ratio":4.7985718009373795,"second":"reduce_r0_interleave'
    'd","verdict":"faster"}}],"size_class":"medium","torch_sum":{"fraction_of_cei'
    'ling":0.947257178593764,"iqr_gb_per_second":[39.313921499707085,88.670753879'
    '32857],"median_gb_per_second":77.94293147502904}}],"ratio_rule":"ratio = med'
    "ian GB/s of the first side / median GB/s of the second, from the same size o"
    "f the same run; faster when ratio >= min_faster_ratio and the first side's p"
    "75 sample time is below the second side's p25; slower when 1 / ratio >= min_"
    "faster_ratio and the second side's p75 is below the first side's p25; otherw"
    'ise no_clear_difference","reference_kernel":"reduce_r0_interleaved","vendor"'
    ':"vendor GPU sum: torch.sum on MPS, PyTorch 2.15.0"},"summary":{"benchmark_p'
    'oints":2,"cases_with_changed_bits":{"reduce_r0_interleaved":[],"reduce_r1_se'
    'quential":[],"reduce_r2_first_add":[],"reduce_r3_simd_shuffle":[],"reduce_r4'
    '_simd_sum":[],"reduce_r5_grid_stride":[]},"ceiling_gb_per_second":82.2827561'
    '8954719,"ceiling_report_sha256":"6b784c28e36b2890eadb65c742858b29835b90f8e25'
    '9b4befba92a499c919cb6","correctness_cases":6,"exact_cases":[0,4,5],"fraction'
    '_measured":true,"hypotheses":{"R0":{"held":1,"missed":1},"R1":{"held":2,"mis'
    'sed":0},"R2":{"held":1,"missed":1},"R3":{"held":2,"missed":0},"R4":{"held":2'
    ',"missed":0},"R5":{"held":2,"missed":0}},"kernels_checked":["reduce_r0_inter'
    'leaved","reduce_r1_sequential","reduce_r2_first_add","reduce_r3_simd_shuffle'
    '","reduce_r4_simd_sum","reduce_r5_grid_stride"],"label":"host-specific Apple'
    ' GPU (Metal) measurement","load_average_after":[11.58984375,11.216796875,12.'
    '97265625],"load_average_before":[11.58984375,11.216796875,12.97265625],"max_'
    'bound_ratio":{"reduce_r0_interleaved":0.05356368952095581,"reduce_r1_sequent'
    'ial":0.05356368952095581,"reduce_r2_first_add":0.05356368952095581,"reduce_r'
    '3_simd_shuffle":0.05356368952095581,"reduce_r4_simd_sum":0.02076955675691078'
    ',"reduce_r5_grid_stride":0.02076955675691078},"points":[{"best_fraction_of_c'
    'eiling":0.3027052459642059,"best_gb_per_second":24.907421950969667,"best_ker'
    'nel":"reduce_r4_simd_sum","best_over_r0":2.75510367690396,"n":65536,"torch_s'
    'um_fraction_of_ceiling":0.28836375318358337,"torch_sum_gb_per_second":23.727'
    '364397107554},{"best_fraction_of_ceiling":1.002774304674144,"best_gb_per_sec'
    'ond":82.5110336246453,"best_kernel":"reduce_r5_grid_stride","best_over_r0":4'
    '.7985718009373795,"n":1048576,"torch_sum_fraction_of_ceiling":0.947257178593'
    '764,"torch_sum_gb_per_second":77.94293147502904}],"points_with_changed_bits"'
    ':{"reduce_r0_interleaved":[],"reduce_r1_sequential":[],"reduce_r2_first_add"'
    ':[],"reduce_r3_simd_shuffle":[],"reduce_r4_simd_sum":[],"reduce_r5_grid_stri'
    'de":[],"torch_sum":[]},"same_order":[{"cases_with_other_bits":[],"kernel":"r'
    'educe_r3_simd_shuffle","points_with_other_bits":[],"same_order_as":"reduce_r'
    '2_first_add"}],"timed_samples":98,"warnings":0},"timer":"time.perf_counter_n'
    's","torch_mps_switches":{"source":"os.environ, read before the runtime impor'
    "ts torch; null means unset, and a value that is not a short token is 'unknow"
    'n\'","values":{"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}}'
    ',"warnings":{"capture":"warnings.catch_warnings(record=True) with simplefilt'
    "er('always'), from before the runtime imports torch to the end of the run; e"
    'qual warnings are kept once with a count","count":0,"distinct":0,"records":['
    '],"torch_imported_before_capture":false}}'
)

# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.gpu_reduce_cli import main
sys.exit(main(sys.argv[1:]))
"""


def recorded_files(directory):
    """Write the config, the Day 74 ceiling and the report as the run read them."""

    config = directory / "config.json"
    config.write_text(json.dumps(RECORDED_CONFIG), encoding="utf-8")
    report = directory / "report.json"
    report.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")
    return config, write_ceiling(directory), report


def verify_argv(config, ceilings, report):
    return [
        "verify",
        "--report",
        str(report),
        "--config",
        str(config),
        "--ceilings",
        str(ceilings),
    ]


def test_the_recorded_report_verifies_as_a_measurement_of_its_inputs(tmp_path):
    report = json.loads(RECORDED_REPORT_JSON)
    _, ceilings, _ = recorded_files(tmp_path)

    summary = verify_report(
        report,
        config=config_from_mapping(RECORDED_CONFIG),
        ceiling=load_ceiling(ceilings),
    )

    assert summary["report_sha256"] == RECORDED_SHA256
    assert (summary["measured"], summary["fraction_measured"]) == (True, True)
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert summary["ceiling_checked"] is True
    assert summary["exact_cases"] == [0, 4, 5]
    assert summary["cases_with_changed_bits"] == {name: [] for name in STEP_KERNELS}
    assert summary["same_order"] == [
        {
            "kernel": "reduce_r3_simd_shuffle",
            "same_order_as": "reduce_r2_first_add",
            "cases_with_other_bits": [],
            "points_with_other_bits": [],
        }
    ]
    held = sum(record["held"] for record in summary["hypotheses"].values())
    assert held == 10 and summary["hypotheses"]["R0"] == {"held": 1, "missed": 1}
    assert report["warnings"]["torch_imported_before_capture"] is False
    assert report["warnings"]["count"] == 0
    assert report["environment"]["torch_version"] == "2.15.0"


def test_the_recorded_file_is_canonical_and_the_cli_verifies_it(tmp_path, capsys):
    config, ceilings, path = recorded_files(tmp_path)

    report, _ = load_report(path)
    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert main(verify_argv(config, ceilings, path)) == EXIT_OK
    line = capsys.readouterr().out
    assert line.startswith(f"verified_gpu_reduce_measurement={RECORDED_SHA256} ")
    assert " fraction_measured=true scope=apple_gpu_metal_this_host " in line
    assert line.endswith(
        "cuda=false nvidia=false warp_level=false hardware_counters=false\n"
    )


def test_a_host_without_torch_verifies_the_recorded_report(tmp_path):
    config, ceilings, path = recorded_files(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, *verify_argv(config, ceilings, path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout.startswith(
        f"verified_gpu_reduce_measurement={RECORDED_SHA256} "
    )


def test_a_tampered_recorded_report_is_rejected_even_when_sealed_again():
    report = json.loads(RECORDED_REPORT_JSON)
    point = report["benchmark"]["points"][1]
    # The new sample is the largest, so the statistics of the side move.
    point["sides"]["reduce_r4_simd_sum"]["samples_ns"][0] = 10**12
    report.pop("report_sha256")

    with pytest.raises(GpuReduceVerificationError, match="does not recompute"):
        verify_report(seal_report(report))


def test_the_nvidia_preflight_job_gates_the_day_78_files():
    job = ci_job("nvidia-preflight-contracts")
    sources = sorted(ROOT.glob("src/ai_journey/gpu_reduce_*.py"))
    tests = sorted(ROOT.glob("tests/test_gpu_reduce_*.py"))

    assert sources and Path(__file__).resolve() in tests
    assert set(sources) <= set(ROOT.glob("src/ai_journey/gpu*.py"))
    assert set(tests) <= set(ROOT.glob("tests/test_gpu*.py"))
    for command in (
        "ruff check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "ruff format --check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m compileall -q src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m pytest -q tests/test_gpu*.py",
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
    ):
        assert command in job, command
