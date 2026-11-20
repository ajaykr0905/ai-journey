"""One real one-call timing of the checklist sides, and the bound that it makes.

``RECORDED_TIMING_JSON`` is the canonical file that ``matmul_checklist_bound_cli
timing`` wrote in a fresh process on 2026-11-20 at 12:17 IST under the shared
GPU lock, with torch 2.15.0 on macOS 27.2 (arm64) and load averages near
14.3, 13.4 and 14.1 from other jobs. It times the recorded checklist config
of ``test_matmul_checklist_recorded``: a 1 x 1 x 1 floor and two sweep
points, with one warmup and three samples on each clock. Joined with the
recorded checklist report and the recorded gpu-ceilings report, it makes one
measured bound report, so a host without torch or MPS, the Linux CI
included, can rebuild and verify it. The three runs are separate, and the
bound keeps the load averages of each. The timings describe that moment and
are here only to be verified.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from ai_journey.gpu_timing_clocks import (
    DEVICE_DOMAIN,
    DISPATCH,
    DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION,
    EVENT,
    SYNCHRONIZED,
)
from ai_journey.matmul_bottleneck_config import load_config as load_rule_config
from ai_journey.matmul_checklist_bound import build_bound_report
from ai_journey.matmul_checklist_bound_timing import method_mapping
from ai_journey.matmul_checklist_bound_timing_verify import verify_timing_report
from ai_journey.matmul_checklist_bound_verify import verify_bound_report
from ai_journey.matmul_checklist_config import config_from_mapping
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_matmul_checklist_recorded import RECORDED_CONFIG
from test_matmul_checklist_recorded import RECORDED_REPORT_JSON as CHECKLIST_JSON

ROOT = Path(__file__).resolve().parents[1]
RULE = load_rule_config(ROOT / "config/day-75-matmul-bottleneck.json")
RECORDED_BOUND_SHA256 = (
    "10c4012f0db032f37e5c3edd504c6f719ebff534578f91e77205b99eb3fbb9e2"
)
SIDES = [*VARIANTS, TORCH_MATMUL]
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_checklist_bound_cli import main
sys.exit(main(sys.argv[1:]))
"""
RECORDED_TIMING_JSON = (
    '{"backend":"torch.mps.compile_shader","config":{"benchmark":{"max_runtime_se'
    'conds":60,"repeats":3,"shapes":[[256,256,256],[200,300,100]],"target_flops_p'
    'er_sample":16777216,"warmup":1},"checklist_sha256":"cb5e3164e0d37716699dc938'
    '1b587faeeb40c2ca4f1a52d75b8ec586379323dc","comparison":{"min_faster_ratio":1'
    '.1},"correctness":{"shapes":[[17,33,9],[65,68,40]]},"naive":{"group_size":[3'
    '2,8]},"name":"matmul-checklist-recorded","schema_version":1,"seed":77},"conf'
    'ig_sha256":"2a6304ef3ca81ffedd1d9f2887d6d14e584068a855eae5eef0ece16c562987de'
    '","environment":{"macos_version":"27.2","numpy_version":"1.26.4","platform_m'
    'achine":"arm64","platform_system":"Darwin","python_implementation":"CPython"'
    ',"python_version":"3.11.14","torch_float32_matmul_precision":"highest","torc'
    'h_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_host":true,"'
    'cublas_measured":false,"cuda_kernel_executed":false,"distributed_claim":fals'
    'e,"hardware_counters_used":false,"host_specific":true,"nvidia_gpu_used":fals'
    'e,"occupancy_measured":false,"portable_performance_claim":false,"production_'
    'claim":false,"statement":"These are one-call timings of the checklist kernel'
    "s and torch.matmul on the Apple GPU of the host that made this report, at th"
    "e recorded load averages, on two clocks: MPS events in the command stream an"
    "d perf_counter_ns on the host. An MPS event interval holds the signal latenc"
    "y of its events and can hold host time between the two records, so it is lon"
    "ger than the kernel. No CUDA kernel ran, no NVIDIA GPU was used, cuBLAS was "
    "not measured, and no hardware counter was read. These timings do not measure"
    ' or predict CUDA, NVIDIA, cuBLAS, distributed or production performance.","v'
    'endor_peak_claimed":false},"floor":{"agreement_bound":2.3841859331241806e-07'
    ',"checks":{"matmul_s2_coarse_4x1":{"disagreement":0.0,"same_bits_as_naive":t'
    'rue},"matmul_s3_block_4x4":{"disagreement":0.0,"same_bits_as_naive":true},"m'
    'atmul_s4_vec4":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_s5_pad'
    '4":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_s6_group_32x32":{"'
    'disagreement":0.0,"same_bits_as_naive":true},"matmul_s6_group_8x8":{"disagre'
    'ement":0.0,"same_bits_as_naive":true},"matmul_s7_interior":{"disagreement":0'
    '.0,"same_bits_as_naive":true},"matmul_tiled_32":{"disagreement":0.0,"same_bi'
    'ts_as_naive":true},"matmul_x_rows":{"disagreement":0.0,"same_bits_as_naive":'
    'true},"torch_matmul":{"disagreement":0.0,"same_bits_as_naive":true}},"flops_'
    'per_call":2,"index":16,"k":1,"m":1,"n":1,"side_order":["matmul_x_columns","m'
    'atmul_x_rows","matmul_tiled_32","matmul_s2_coarse_4x1","matmul_s3_block_4x4"'
    ',"matmul_s4_vec4","matmul_s5_pad4","matmul_s6_group_8x8","matmul_s6_group_32'
    'x32","matmul_s7_interior","torch_matmul"],"sides":{"matmul_s2_coarse_4x1":{"'
    'clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counte'
    'r_ns","samples_ns":[10167,6542,41125],"timing":{"count":3,"iqr_ns":34583,"ma'
    'x_ns":41125,"median_ns":10167,"min_ns":6542,"p25_ns":6542,"p75_ns":41125}},"'
    'host_synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_'
    'ns","samples_ns":[321041,335500,168959],"timing":{"count":3,"iqr_ns":166541,'
    '"max_ns":335500,"median_ns":321041,"min_ns":168959,"p25_ns":168959,"p75_ns":'
    '335500}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elaps'
    'ed_ms":[0.3404583258088678,0.35099999513477087,0.1298749994020909],"samples_'
    'ns":[340458,351000,129875],"timing":{"count":3,"iqr_ns":221125,"max_ns":3510'
    '00,"median_ns":340458,"min_ns":129875,"p25_ns":129875,"p75_ns":351000}}},"im'
    'plementation":"matmul_s2_coarse_4x1","repeats":3,"warmup":1},"matmul_s3_bloc'
    'k_4x4":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_pe'
    'rf_counter_ns","samples_ns":[5417,6125,16125],"timing":{"count":3,"iqr_ns":1'
    '0708,"max_ns":16125,"median_ns":6125,"min_ns":5417,"p25_ns":5417,"p75_ns":16'
    '125}},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_c'
    'ounter_ns","samples_ns":[146042,166792,158959],"timing":{"count":3,"iqr_ns":'
    '20750,"max_ns":166792,"median_ns":158959,"min_ns":146042,"p25_ns":146042,"p7'
    '5_ns":166792}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream",'
    '"elapsed_ms":[0.2313333097845316,0.08625001646578312,0.08412500028498471],"s'
    'amples_ns":[231333,86250,84125],"timing":{"count":3,"iqr_ns":147208,"max_ns"'
    ':231333,"median_ns":86250,"min_ns":84125,"p25_ns":84125,"p75_ns":231333}}},"'
    'implementation":"matmul_s3_block_4x4","repeats":3,"warmup":1},"matmul_s4_vec'
    '4":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_c'
    'ounter_ns","samples_ns":[5208,6916,38500],"timing":{"count":3,"iqr_ns":33292'
    ',"max_ns":38500,"median_ns":6916,"min_ns":5208,"p25_ns":5208,"p75_ns":38500}'
    '},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_count'
    'er_ns","samples_ns":[145416,161791,181958],"timing":{"count":3,"iqr_ns":3654'
    '2,"max_ns":181958,"median_ns":161791,"min_ns":145416,"p25_ns":145416,"p75_ns'
    '":181958}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","ela'
    'psed_ms":[0.08470832835882902,0.08545833406969905,0.18745832494460046],"samp'
    'les_ns":[84708,85458,187458],"timing":{"count":3,"iqr_ns":102750,"max_ns":18'
    '7458,"median_ns":85458,"min_ns":84708,"p25_ns":84708,"p75_ns":187458}}},"imp'
    'lementation":"matmul_s4_vec4","repeats":3,"warmup":1},"matmul_s5_pad4":{"clo'
    'cks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_n'
    's","samples_ns":[5208,6292,22625],"timing":{"count":3,"iqr_ns":17417,"max_ns'
    '":22625,"median_ns":6292,"min_ns":5208,"p25_ns":5208,"p75_ns":22625}},"host_'
    'synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","'
    'samples_ns":[141791,162958,408000],"timing":{"count":3,"iqr_ns":266209,"max_'
    'ns":408000,"median_ns":162958,"min_ns":141791,"p25_ns":141791,"p75_ns":40800'
    '0}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms'
    '":[0.08004167466424406,0.2346250112168491,0.09954164852388203],"samples_ns":'
    '[80042,234625,99542],"timing":{"count":3,"iqr_ns":154583,"max_ns":234625,"me'
    'dian_ns":99542,"min_ns":80042,"p25_ns":80042,"p75_ns":234625}}},"implementat'
    'ion":"matmul_s5_pad4","repeats":3,"warmup":1},"matmul_s6_group_32x32":{"cloc'
    'ks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns'
    '","samples_ns":[8833,8125,6459],"timing":{"count":3,"iqr_ns":2374,"max_ns":8'
    '833,"median_ns":8125,"min_ns":6459,"p25_ns":6459,"p75_ns":8833}},"host_synch'
    'ronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","sampl'
    'es_ns":[201375,247958,142459],"timing":{"count":3,"iqr_ns":105499,"max_ns":2'
    '47958,"median_ns":201375,"min_ns":142459,"p25_ns":142459,"p75_ns":247958}},"'
    'mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.'
    '10833333362825215,0.12287500430829823,0.07124998955987394],"samples_ns":[108'
    '333,122875,71250],"timing":{"count":3,"iqr_ns":51625,"max_ns":122875,"median'
    '_ns":108333,"min_ns":71250,"p25_ns":71250,"p75_ns":122875}}},"implementation'
    '":"matmul_s6_group_32x32","repeats":3,"warmup":1},"matmul_s6_group_8x8":{"cl'
    'ocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_'
    'ns","samples_ns":[20000,13000,45167],"timing":{"count":3,"iqr_ns":32167,"max'
    '_ns":45167,"median_ns":20000,"min_ns":13000,"p25_ns":13000,"p75_ns":45167}},'
    '"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_counter'
    '_ns","samples_ns":[896125,316792,142666],"timing":{"count":3,"iqr_ns":753459'
    ',"max_ns":896125,"median_ns":316792,"min_ns":142666,"p25_ns":142666,"p75_ns"'
    ':896125}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elap'
    'sed_ms":[0.10429168469272554,0.08599998545832932,0.1022916694637388],"sample'
    's_ns":[104292,86000,102292],"timing":{"count":3,"iqr_ns":18292,"max_ns":1042'
    '92,"median_ns":102292,"min_ns":86000,"p25_ns":86000,"p75_ns":104292}}},"impl'
    'ementation":"matmul_s6_group_8x8","repeats":3,"warmup":1},"matmul_s7_interio'
    'r":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_c'
    'ounter_ns","samples_ns":[9458,11791,6666],"timing":{"count":3,"iqr_ns":5125,'
    '"max_ns":11791,"median_ns":9458,"min_ns":6666,"p25_ns":6666,"p75_ns":11791}}'
    ',"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_counte'
    'r_ns","samples_ns":[380417,389958,140791],"timing":{"count":3,"iqr_ns":24916'
    '7,"max_ns":389958,"median_ns":380417,"min_ns":140791,"p25_ns":140791,"p75_ns'
    '":389958}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","ela'
    'psed_ms":[0.16850000247359276,0.14054166967980564,0.10691667557694018],"samp'
    'les_ns":[168500,140542,106917],"timing":{"count":3,"iqr_ns":61583,"max_ns":1'
    '68500,"median_ns":140542,"min_ns":106917,"p25_ns":106917,"p75_ns":168500}}},'
    '"implementation":"matmul_s7_interior","repeats":3,"warmup":1},"matmul_tiled_'
    '32":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_'
    'counter_ns","samples_ns":[16583,5959,51083],"timing":{"count":3,"iqr_ns":451'
    '24,"max_ns":51083,"median_ns":16583,"min_ns":5959,"p25_ns":5959,"p75_ns":510'
    '83}},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_co'
    'unter_ns","samples_ns":[322250,314625,545417],"timing":{"count":3,"iqr_ns":2'
    '30792,"max_ns":545417,"median_ns":322250,"min_ns":314625,"p25_ns":314625,"p7'
    '5_ns":545417}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream",'
    '"elapsed_ms":[0.23370832786895335,0.23237502318806946,0.4282916779629886],"s'
    'amples_ns":[233708,232375,428292],"timing":{"count":3,"iqr_ns":195917,"max_n'
    's":428292,"median_ns":233708,"min_ns":232375,"p25_ns":232375,"p75_ns":428292'
    '}}},"implementation":"matmul_tiled_32","repeats":3,"warmup":1},"matmul_x_col'
    'umns":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_per'
    'f_counter_ns","samples_ns":[15417,26500,9833],"timing":{"count":3,"iqr_ns":1'
    '6667,"max_ns":26500,"median_ns":15417,"min_ns":9833,"p25_ns":9833,"p75_ns":2'
    '6500}},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_'
    'counter_ns","samples_ns":[362333,220000,114500],"timing":{"count":3,"iqr_ns"'
    ':247833,"max_ns":362333,"median_ns":220000,"min_ns":114500,"p25_ns":114500,"'
    'p75_ns":362333}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream'
    '","elapsed_ms":[0.07862498750910163,0.3976666775997728,0.06429164204746485],'
    '"samples_ns":[78625,397667,64292],"timing":{"count":3,"iqr_ns":333375,"max_n'
    's":397667,"median_ns":78625,"min_ns":64292,"p25_ns":64292,"p75_ns":397667}}}'
    ',"implementation":"matmul_x_columns","repeats":3,"warmup":1},"matmul_x_rows"'
    ':{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_cou'
    'nter_ns","samples_ns":[6042,12417,5917],"timing":{"count":3,"iqr_ns":6500,"m'
    'ax_ns":12417,"median_ns":6042,"min_ns":5917,"p25_ns":5917,"p75_ns":12417}},"'
    'host_synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_'
    'ns","samples_ns":[306000,298917,108208],"timing":{"count":3,"iqr_ns":197792,'
    '"max_ns":306000,"median_ns":298917,"min_ns":108208,"p25_ns":108208,"p75_ns":'
    '306000}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elaps'
    'ed_ms":[0.23620834690518677,0.22641668329015374,0.06695833872072399],"sample'
    's_ns":[236208,226417,66958],"timing":{"count":3,"iqr_ns":169250,"max_ns":236'
    '208,"median_ns":226417,"min_ns":66958,"p25_ns":66958,"p75_ns":236208}}},"imp'
    'lementation":"matmul_x_rows","repeats":3,"warmup":1},"torch_matmul":{"clocks'
    '":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns",'
    '"samples_ns":[8666,6958,17666],"timing":{"count":3,"iqr_ns":10708,"max_ns":1'
    '7666,"median_ns":8666,"min_ns":6958,"p25_ns":6958,"p75_ns":17666}},"host_syn'
    'chronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","sam'
    'ples_ns":[305833,174959,126458],"timing":{"count":3,"iqr_ns":179375,"max_ns"'
    ':305833,"median_ns":174959,"min_ns":126458,"p25_ns":126458,"p75_ns":305833}}'
    ',"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":['
    '0.10199999087490141,0.08270831312984228,0.10087501141242683],"samples_ns":[1'
    '02000,82708,100875],"timing":{"count":3,"iqr_ns":19292,"max_ns":102000,"medi'
    'an_ns":100875,"min_ns":82708,"p25_ns":82708,"p75_ns":102000}}},"implementati'
    'on":"torch_matmul","repeats":3,"warmup":1}}},"host_load":{"after":[14.314453'
    '125,13.43701171875,14.12109375],"before":[14.314453125,13.43701171875,14.121'
    '09375],"source":"os.getloadavg(): the 1, 5 and 15 minute load averages that '
    'uptime prints, read before the checks and after the last sample"},"kernels":'
    '{"admissions":[{"admitted":true,"allowed_threads":1024,"declared_threadgroup'
    '_bytes":8192,"name":"matmul_tiled_32","reason":null,"required_threads":1024,'
    '"static_threadgroup_bytes":8192},{"admitted":true,"allowed_threads":1024,"de'
    'clared_threadgroup_bytes":8192,"name":"matmul_s2_coarse_4x1","reason":null,"'
    'required_threads":256,"static_threadgroup_bytes":8192},{"admitted":true,"all'
    'owed_threads":1024,"declared_threadgroup_bytes":8192,"name":"matmul_s3_block'
    '_4x4","reason":null,"required_threads":256,"static_threadgroup_bytes":8192},'
    '{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":8192,"n'
    'ame":"matmul_s4_vec4","reason":null,"required_threads":256,"static_threadgro'
    'up_bytes":8192},{"admitted":true,"allowed_threads":1024,"declared_threadgrou'
    'p_bytes":9472,"name":"matmul_s5_pad4","reason":null,"required_threads":256,"'
    'static_threadgroup_bytes":9472},{"admitted":true,"allowed_threads":1024,"dec'
    'lared_threadgroup_bytes":4864,"name":"matmul_s6_group_8x8","reason":null,"re'
    'quired_threads":64,"static_threadgroup_bytes":4864},{"admitted":true,"allowe'
    'd_threads":1024,"declared_threadgroup_bytes":18688,"name":"matmul_s6_group_3'
    '2x32","reason":null,"required_threads":1024,"static_threadgroup_bytes":18688'
    '},{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":9472,'
    '"name":"matmul_s7_interior","reason":null,"required_threads":256,"static_thr'
    'eadgroup_bytes":9472}],"library_sha256":"d0a8f46a8831029a8894edca48da9b8d616'
    '46571c40c36d2f502d6f30cea027d","limits":{"matmul_s2_coarse_4x1":{"max_thread'
    's_per_threadgroup":1024,"static_threadgroup_memory_bytes":8192,"thread_execu'
    'tion_width":32},"matmul_s3_block_4x4":{"max_threads_per_threadgroup":1024,"s'
    'tatic_threadgroup_memory_bytes":8192,"thread_execution_width":32},"matmul_s4'
    '_vec4":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes"'
    ':8192,"thread_execution_width":32},"matmul_s5_pad4":{"max_threads_per_thread'
    'group":1024,"static_threadgroup_memory_bytes":9472,"thread_execution_width":'
    '32},"matmul_s6_group_32x32":{"max_threads_per_threadgroup":1024,"static_thre'
    'adgroup_memory_bytes":18688,"thread_execution_width":32},"matmul_s6_group_8x'
    '8":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":486'
    '4,"thread_execution_width":32},"matmul_s7_interior":{"max_threads_per_thread'
    'group":1024,"static_threadgroup_memory_bytes":9472,"thread_execution_width":'
    '32},"matmul_tiled_32":{"max_threads_per_threadgroup":1024,"static_threadgrou'
    'p_memory_bytes":8192,"thread_execution_width":32},"matmul_x_columns":{"max_t'
    'hreads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_exe'
    'cution_width":32},"matmul_x_rows":{"max_threads_per_threadgroup":1024,"stati'
    'c_threadgroup_memory_bytes":0,"thread_execution_width":32}},"naive_source_sh'
    'a256":"beef26392d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc"},"k'
    'ind":"matmul_checklist_one_call_timing","label":"host-specific Apple GPU (Me'
    'tal) measurement","measured":true,"measurement_scope":"apple_gpu_metal_this_'
    'host","method":{"agreement":"every side within 4 * gamma_K(2**-24) * (|A| @ '
    '|B|) of the naive output, on the device in float32","clocks":{"host_dispatch'
    '":"time.perf_counter_ns() just before and just after the launch call alone, '
    "with no synchronize between the readings; the queue is drained before the fi"
    'rst reading and after the second","host_synchronized":"time.perf_counter_ns('
    ") just before the launch and just after the torch.mps.synchronize() that fol"
    'lows it; the queue is drained before the first reading","mps_event":"torch.m'
    "ps.Event(enable_timing=True): start.record() just before the launch and end."
    "record() just after it, then torch.mps.synchronize() and start.elapsed_time("
    'end) in ms; samples_ns = round(ms * 1e6)"},"domains":{"host_perf_counter_ns"'
    ':"host-side: time.perf_counter_ns on the CPU thread that issued the launch",'
    '"mps_event_stream":"the interval between two points in the MPS command strea'
    "m, the start and end events; it can also hold host time between the two reco"
    "rd() calls, including the launch call, so it is not GPU time alone; the PyTo"
    "rch MPS header says a command-buffer completion handler stores each timed ev"
    "ent's completion time and does not name the clock it reads, so these values "
    'are never subtracted from host values without a cross_domain record"},"floor'
    '":"every side first runs one 1 x 1 x 1 call on operands from case index 16; '
    'its host time bounds a fixed cost that every call of the side pays","operand'
    's":"point i uses the operands of the checklist run\'s point i: seed, benchmar'
    'k stream and case index i of the Day 74 generator","schedule":"one launch an'
    "d one clock per sample; per side, warmup untimed samples on each clock first"
    "; then repeat r visits the sides shifted by r and, for each side, the clocks"
    ' shifted by r"},"percentile_method":"nearest rank: sorted samples[ceil(p * n'
    " / 100) - 1] for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no "
    'interpolation","points":[{"agreement_bound":6.103608758678569e-05,"checks":{'
    '"matmul_s2_coarse_4x1":{"disagreement":0.0,"same_bits_as_naive":true},"matmu'
    'l_s3_block_4x4":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_s4_ve'
    'c4":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_s5_pad4":{"disagr'
    'eement":0.0,"same_bits_as_naive":true},"matmul_s6_group_32x32":{"disagreemen'
    't":0.0,"same_bits_as_naive":true},"matmul_s6_group_8x8":{"disagreement":0.0,'
    '"same_bits_as_naive":true},"matmul_s7_interior":{"disagreement":0.0,"same_bi'
    'ts_as_naive":true},"matmul_tiled_32":{"disagreement":0.0,"same_bits_as_naive'
    '":true},"matmul_x_rows":{"disagreement":0.0,"same_bits_as_naive":true},"torc'
    'h_matmul":{"disagreement":0.0,"same_bits_as_naive":true}},"flops_per_call":3'
    '3554432,"index":0,"k":256,"m":256,"n":256,"side_order":["matmul_x_columns","'
    'matmul_x_rows","matmul_tiled_32","matmul_s2_coarse_4x1","matmul_s3_block_4x4'
    '","matmul_s4_vec4","matmul_s5_pad4","matmul_s6_group_8x8","matmul_s6_group_3'
    '2x32","matmul_s7_interior","torch_matmul"],"sides":{"matmul_s2_coarse_4x1":{'
    '"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_count'
    'er_ns","samples_ns":[5667,9375,9875],"timing":{"count":3,"iqr_ns":4208,"max_'
    'ns":9875,"median_ns":9375,"min_ns":5667,"p25_ns":5667,"p75_ns":9875}},"host_'
    'synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","'
    'samples_ns":[356667,399208,383625],"timing":{"count":3,"iqr_ns":42541,"max_n'
    's":399208,"median_ns":383625,"min_ns":356667,"p25_ns":356667,"p75_ns":399208'
    '}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms"'
    ':[0.29399999766610563,0.2851249882951379,0.2843750116880983],"samples_ns":[2'
    '94000,285125,284375],"timing":{"count":3,"iqr_ns":9625,"max_ns":294000,"medi'
    'an_ns":285125,"min_ns":284375,"p25_ns":284375,"p75_ns":294000}}},"implementa'
    'tion":"matmul_s2_coarse_4x1","repeats":3,"warmup":1},"matmul_s3_block_4x4":{'
    '"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_count'
    'er_ns","samples_ns":[5750,5250,8542],"timing":{"count":3,"iqr_ns":3292,"max_'
    'ns":8542,"median_ns":5750,"min_ns":5250,"p25_ns":5250,"p75_ns":8542}},"host_'
    'synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","'
    'samples_ns":[340375,357416,356375],"timing":{"count":3,"iqr_ns":17041,"max_n'
    's":357416,"median_ns":356375,"min_ns":340375,"p25_ns":340375,"p75_ns":357416'
    '}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms"'
    ':[0.27116667479276657,0.26804168010130525,0.2643750049173832],"samples_ns":['
    '271167,268042,264375],"timing":{"count":3,"iqr_ns":6792,"max_ns":271167,"med'
    'ian_ns":268042,"min_ns":264375,"p25_ns":264375,"p75_ns":271167}}},"implement'
    'ation":"matmul_s3_block_4x4","repeats":3,"warmup":1},"matmul_s4_vec4":{"cloc'
    'ks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns'
    '","samples_ns":[5250,5208,7709],"timing":{"count":3,"iqr_ns":2501,"max_ns":7'
    '709,"median_ns":5250,"min_ns":5208,"p25_ns":5208,"p75_ns":7709}},"host_synch'
    'ronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","sampl'
    'es_ns":[296375,317584,324625],"timing":{"count":3,"iqr_ns":28250,"max_ns":32'
    '4625,"median_ns":317584,"min_ns":296375,"p25_ns":296375,"p75_ns":324625}},"m'
    'ps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.2'
    '510000194888562,0.23958334350027144,0.25025001377798617],"samples_ns":[25100'
    '0,239583,250250],"timing":{"count":3,"iqr_ns":11417,"max_ns":251000,"median_'
    'ns":250250,"min_ns":239583,"p25_ns":239583,"p75_ns":251000}}},"implementatio'
    'n":"matmul_s4_vec4","repeats":3,"warmup":1},"matmul_s5_pad4":{"clocks":{"hos'
    't_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","sample'
    's_ns":[6417,5000,6500],"timing":{"count":3,"iqr_ns":1500,"max_ns":6500,"medi'
    'an_ns":6417,"min_ns":5000,"p25_ns":5000,"p75_ns":6500}},"host_synchronized":'
    '{"clock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[3'
    '05041,311750,326708],"timing":{"count":3,"iqr_ns":21667,"max_ns":326708,"med'
    'ian_ns":311750,"min_ns":305041,"p25_ns":305041,"p75_ns":326708}},"mps_event"'
    ':{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.2325833484'
    '0647876,0.23920834064483643,0.23787497775629163],"samples_ns":[232583,239208'
    ',237875],"timing":{"count":3,"iqr_ns":6625,"max_ns":239208,"median_ns":23787'
    '5,"min_ns":232583,"p25_ns":232583,"p75_ns":239208}}},"implementation":"matmu'
    'l_s5_pad4","repeats":3,"warmup":1},"matmul_s6_group_32x32":{"clocks":{"host_'
    'dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_'
    'ns":[13834,7959,6375],"timing":{"count":3,"iqr_ns":7459,"max_ns":13834,"medi'
    'an_ns":7959,"min_ns":6375,"p25_ns":6375,"p75_ns":13834}},"host_synchronized"'
    ':{"clock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":['
    '692500,493875,667750],"timing":{"count":3,"iqr_ns":198625,"max_ns":692500,"m'
    'edian_ns":667750,"min_ns":493875,"p25_ns":493875,"p75_ns":692500}},"mps_even'
    't":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.45091667'
    '561791837,0.4222083371132612,0.5106666649226099],"samples_ns":[450917,422208'
    ',510667],"timing":{"count":3,"iqr_ns":88459,"max_ns":510667,"median_ns":4509'
    '17,"min_ns":422208,"p25_ns":422208,"p75_ns":510667}}},"implementation":"matm'
    'ul_s6_group_32x32","repeats":3,"warmup":1},"matmul_s6_group_8x8":{"clocks":{'
    '"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","sa'
    'mples_ns":[568667,32708,6708],"timing":{"count":3,"iqr_ns":561959,"max_ns":5'
    '68667,"median_ns":32708,"min_ns":6708,"p25_ns":6708,"p75_ns":568667}},"host_'
    'synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","'
    'samples_ns":[411334,597750,310083],"timing":{"count":3,"iqr_ns":287667,"max_'
    'ns":597750,"median_ns":411334,"min_ns":310083,"p25_ns":310083,"p75_ns":59775'
    '0}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms'
    '":[0.2322500222362578,0.23224999313242733,0.22412498947232962],"samples_ns":'
    '[232250,232250,224125],"timing":{"count":3,"iqr_ns":8125,"max_ns":232250,"me'
    'dian_ns":232250,"min_ns":224125,"p25_ns":224125,"p75_ns":232250}}},"implemen'
    'tation":"matmul_s6_group_8x8","repeats":3,"warmup":1},"matmul_s7_interior":{'
    '"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_count'
    'er_ns","samples_ns":[21000,10750,12083],"timing":{"count":3,"iqr_ns":10250,"'
    'max_ns":21000,"median_ns":12083,"min_ns":10750,"p25_ns":10750,"p75_ns":21000'
    '}},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_coun'
    'ter_ns","samples_ns":[354500,335750,291250],"timing":{"count":3,"iqr_ns":632'
    '50,"max_ns":354500,"median_ns":335750,"min_ns":291250,"p25_ns":291250,"p75_n'
    's":354500}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","el'
    'apsed_ms":[0.26041665114462376,0.21170833497308195,0.2605416520964354],"samp'
    'les_ns":[260417,211708,260542],"timing":{"count":3,"iqr_ns":48834,"max_ns":2'
    '60542,"median_ns":260417,"min_ns":211708,"p25_ns":211708,"p75_ns":260542}}},'
    '"implementation":"matmul_s7_interior","repeats":3,"warmup":1},"matmul_tiled_'
    '32":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_'
    'counter_ns","samples_ns":[13083,53125,7166],"timing":{"count":3,"iqr_ns":459'
    '59,"max_ns":53125,"median_ns":13083,"min_ns":7166,"p25_ns":7166,"p75_ns":531'
    '25}},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_co'
    'unter_ns","samples_ns":[770875,801458,780875],"timing":{"count":3,"iqr_ns":3'
    '0583,"max_ns":801458,"median_ns":780875,"min_ns":770875,"p25_ns":770875,"p75'
    '_ns":801458}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","'
    'elapsed_ms":[0.6495000270660967,0.7973749889060855,0.6544583302456886],"samp'
    'les_ns":[649500,797375,654458],"timing":{"count":3,"iqr_ns":147875,"max_ns":'
    '797375,"median_ns":654458,"min_ns":649500,"p25_ns":649500,"p75_ns":797375}}}'
    ',"implementation":"matmul_tiled_32","repeats":3,"warmup":1},"matmul_x_column'
    's":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_c'
    'ounter_ns","samples_ns":[37541,8375,8666],"timing":{"count":3,"iqr_ns":29166'
    ',"max_ns":37541,"median_ns":8666,"min_ns":8375,"p25_ns":8375,"p75_ns":37541}'
    '},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_count'
    'er_ns","samples_ns":[603542,580500,587042],"timing":{"count":3,"iqr_ns":2304'
    '2,"max_ns":603542,"median_ns":587042,"min_ns":580500,"p25_ns":580500,"p75_ns'
    '":603542}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","ela'
    'psed_ms":[0.4904583329334855,0.49000000581145287,0.4816250002477318],"sample'
    's_ns":[490458,490000,481625],"timing":{"count":3,"iqr_ns":8833,"max_ns":4904'
    '58,"median_ns":490000,"min_ns":481625,"p25_ns":481625,"p75_ns":490458}}},"im'
    'plementation":"matmul_x_columns","repeats":3,"warmup":1},"matmul_x_rows":{"c'
    'locks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter'
    '_ns","samples_ns":[27416,40417,11709],"timing":{"count":3,"iqr_ns":28708,"ma'
    'x_ns":40417,"median_ns":27416,"min_ns":11709,"p25_ns":11709,"p75_ns":40417}}'
    ',"host_synchronized":{"clock":"host_synchronized","domain":"host_perf_counte'
    'r_ns","samples_ns":[1284834,1174000,1353917],"timing":{"count":3,"iqr_ns":17'
    '9917,"max_ns":1353917,"median_ns":1284834,"min_ns":1174000,"p25_ns":1174000,'
    '"p75_ns":1353917}},"mps_event":{"clock":"mps_event","domain":"mps_event_stre'
    'am","elapsed_ms":[1.097124972147867,1.1159583227708936,1.0750000074040145],"'
    'samples_ns":[1097125,1115958,1075000],"timing":{"count":3,"iqr_ns":40958,"ma'
    'x_ns":1115958,"median_ns":1097125,"min_ns":1075000,"p25_ns":1075000,"p75_ns"'
    ':1115958}}},"implementation":"matmul_x_rows","repeats":3,"warmup":1},"torch_'
    'matmul":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_p'
    'erf_counter_ns","samples_ns":[36250,53208,117959],"timing":{"count":3,"iqr_n'
    's":81709,"max_ns":117959,"median_ns":53208,"min_ns":36250,"p25_ns":36250,"p7'
    '5_ns":117959}},"host_synchronized":{"clock":"host_synchronized","domain":"ho'
    'st_perf_counter_ns","samples_ns":[297375,368334,233500],"timing":{"count":3,'
    '"iqr_ns":134834,"max_ns":368334,"median_ns":297375,"min_ns":233500,"p25_ns":'
    '233500,"p75_ns":368334}},"mps_event":{"clock":"mps_event","domain":"mps_even'
    't_stream","elapsed_ms":[0.2462499833200127,0.12395833618938923,0.20437501370'
    '90683],"samples_ns":[246250,123958,204375],"timing":{"count":3,"iqr_ns":1222'
    '92,"max_ns":246250,"median_ns":204375,"min_ns":123958,"p25_ns":123958,"p75_n'
    's":246250}}},"implementation":"torch_matmul","repeats":3,"warmup":1}}},{"agr'
    'eement_bound":2.384200001955044e-05,"checks":{"matmul_s2_coarse_4x1":{"disag'
    'reement":0.0,"same_bits_as_naive":true},"matmul_s3_block_4x4":{"disagreement'
    '":0.0,"same_bits_as_naive":true},"matmul_s4_vec4":{"disagreement":0.0,"same_'
    'bits_as_naive":true},"matmul_s5_pad4":{"disagreement":0.0,"same_bits_as_naiv'
    'e":true},"matmul_s6_group_32x32":{"disagreement":0.0,"same_bits_as_naive":tr'
    'ue},"matmul_s6_group_8x8":{"disagreement":0.0,"same_bits_as_naive":true},"ma'
    'tmul_s7_interior":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_til'
    'ed_32":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_x_rows":{"disa'
    'greement":0.0,"same_bits_as_naive":true},"torch_matmul":{"disagreement":0.0,'
    '"same_bits_as_naive":true}},"flops_per_call":12000000,"index":1,"k":100,"m":'
    '200,"n":300,"side_order":["matmul_x_columns","matmul_x_rows","matmul_tiled_3'
    '2","matmul_s2_coarse_4x1","matmul_s3_block_4x4","matmul_s4_vec4","matmul_s5_'
    'pad4","matmul_s6_group_8x8","matmul_s6_group_32x32","matmul_s7_interior","to'
    'rch_matmul"],"sides":{"matmul_s2_coarse_4x1":{"clocks":{"host_dispatch":{"cl'
    'ock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[43583,222'
    '91,7625],"timing":{"count":3,"iqr_ns":35958,"max_ns":43583,"median_ns":22291'
    ',"min_ns":7625,"p25_ns":7625,"p75_ns":43583}},"host_synchronized":{"clock":"'
    'host_synchronized","domain":"host_perf_counter_ns","samples_ns":[333875,3383'
    '75,262666],"timing":{"count":3,"iqr_ns":75709,"max_ns":338375,"median_ns":33'
    '3875,"min_ns":262666,"p25_ns":262666,"p75_ns":338375}},"mps_event":{"clock":'
    '"mps_event","domain":"mps_event_stream","elapsed_ms":[0.18087500939145684,0.'
    '16991666052490473,0.16783335013315082],"samples_ns":[180875,169917,167833],"'
    'timing":{"count":3,"iqr_ns":13042,"max_ns":180875,"median_ns":169917,"min_ns'
    '":167833,"p25_ns":167833,"p75_ns":180875}}},"implementation":"matmul_s2_coar'
    'se_4x1","repeats":3,"warmup":1},"matmul_s3_block_4x4":{"clocks":{"host_dispa'
    'tch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":['
    '11208,7792,6083],"timing":{"count":3,"iqr_ns":5125,"max_ns":11208,"median_ns'
    '":7792,"min_ns":6083,"p25_ns":6083,"p75_ns":11208}},"host_synchronized":{"cl'
    'ock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[26866'
    '7,222709,226875],"timing":{"count":3,"iqr_ns":45958,"max_ns":268667,"median_'
    'ns":226875,"min_ns":222709,"p25_ns":222709,"p75_ns":268667}},"mps_event":{"c'
    'lock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.18545833881944'
    '418,0.14424999244511127,0.16200001118704677],"samples_ns":[185458,144250,162'
    '000],"timing":{"count":3,"iqr_ns":41208,"max_ns":185458,"median_ns":162000,"'
    'min_ns":144250,"p25_ns":144250,"p75_ns":185458}}},"implementation":"matmul_s'
    '3_block_4x4","repeats":3,"warmup":1},"matmul_s4_vec4":{"clocks":{"host_dispa'
    'tch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":['
    '5084,7208,5875],"timing":{"count":3,"iqr_ns":2124,"max_ns":7208,"median_ns":'
    '5875,"min_ns":5084,"p25_ns":5084,"p75_ns":7208}},"host_synchronized":{"clock'
    '":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[215625,2'
    '39834,210958],"timing":{"count":3,"iqr_ns":28876,"max_ns":239834,"median_ns"'
    ':215625,"min_ns":210958,"p25_ns":210958,"p75_ns":239834}},"mps_event":{"cloc'
    'k":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.12466666521504521'
    ',0.13412497355602682,0.1506666885688901],"samples_ns":[124667,134125,150667]'
    ',"timing":{"count":3,"iqr_ns":26000,"max_ns":150667,"median_ns":134125,"min_'
    'ns":124667,"p25_ns":124667,"p75_ns":150667}}},"implementation":"matmul_s4_ve'
    'c4","repeats":3,"warmup":1},"matmul_s5_pad4":{"clocks":{"host_dispatch":{"cl'
    'ock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[14625,604'
    '2,5500],"timing":{"count":3,"iqr_ns":9125,"max_ns":14625,"median_ns":6042,"m'
    'in_ns":5500,"p25_ns":5500,"p75_ns":14625}},"host_synchronized":{"clock":"hos'
    't_synchronized","domain":"host_perf_counter_ns","samples_ns":[197375,222916,'
    '188375],"timing":{"count":3,"iqr_ns":34541,"max_ns":222916,"median_ns":19737'
    '5,"min_ns":188375,"p25_ns":188375,"p75_ns":222916}},"mps_event":{"clock":"mp'
    's_event","domain":"mps_event_stream","elapsed_ms":[0.15449998318217695,0.141'
    '33332297205925,0.1371666439808905],"samples_ns":[154500,141333,137167],"timi'
    'ng":{"count":3,"iqr_ns":17333,"max_ns":154500,"median_ns":141333,"min_ns":13'
    '7167,"p25_ns":137167,"p75_ns":154500}}},"implementation":"matmul_s5_pad4","r'
    'epeats":3,"warmup":1},"matmul_s6_group_32x32":{"clocks":{"host_dispatch":{"c'
    'lock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[5209,745'
    '9,44125],"timing":{"count":3,"iqr_ns":38916,"max_ns":44125,"median_ns":7459,'
    '"min_ns":5209,"p25_ns":5209,"p75_ns":44125}},"host_synchronized":{"clock":"h'
    'ost_synchronized","domain":"host_perf_counter_ns","samples_ns":[252125,27329'
    '2,342166],"timing":{"count":3,"iqr_ns":90041,"max_ns":342166,"median_ns":273'
    '292,"min_ns":252125,"p25_ns":252125,"p75_ns":342166}},"mps_event":{"clock":"'
    'mps_event","domain":"mps_event_stream","elapsed_ms":[0.20345833036117256,0.2'
    '1016664686612785,0.35866667167283595],"samples_ns":[203458,210167,358667],"t'
    'iming":{"count":3,"iqr_ns":155209,"max_ns":358667,"median_ns":210167,"min_ns'
    '":203458,"p25_ns":203458,"p75_ns":358667}}},"implementation":"matmul_s6_grou'
    'p_32x32","repeats":3,"warmup":1},"matmul_s6_group_8x8":{"clocks":{"host_disp'
    'atch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":'
    '[5333,5166,5833],"timing":{"count":3,"iqr_ns":667,"max_ns":5833,"median_ns":'
    '5333,"min_ns":5166,"p25_ns":5166,"p75_ns":5833}},"host_synchronized":{"clock'
    '":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[195875,2'
    '51042,232542],"timing":{"count":3,"iqr_ns":55167,"max_ns":251042,"median_ns"'
    ':232542,"min_ns":195875,"p25_ns":195875,"p75_ns":251042}},"mps_event":{"cloc'
    'k":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.14287501107901335'
    ',0.1391666883137077,0.14558332622982562],"samples_ns":[142875,139167,145583]'
    ',"timing":{"count":3,"iqr_ns":6416,"max_ns":145583,"median_ns":142875,"min_n'
    's":139167,"p25_ns":139167,"p75_ns":145583}}},"implementation":"matmul_s6_gro'
    'up_8x8","repeats":3,"warmup":1},"matmul_s7_interior":{"clocks":{"host_dispat'
    'ch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[5'
    '500,5167,20250],"timing":{"count":3,"iqr_ns":15083,"max_ns":20250,"median_ns'
    '":5500,"min_ns":5167,"p25_ns":5167,"p75_ns":20250}},"host_synchronized":{"cl'
    'ock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[21354'
    '2,216959,192000],"timing":{"count":3,"iqr_ns":24959,"max_ns":216959,"median_'
    'ns":213542,"min_ns":192000,"p25_ns":192000,"p75_ns":216959}},"mps_event":{"c'
    'lock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.15412500943057'
    '24,0.137208349769935,0.47179165994748473],"samples_ns":[154125,137208,471792'
    '],"timing":{"count":3,"iqr_ns":334584,"max_ns":471792,"median_ns":154125,"mi'
    'n_ns":137208,"p25_ns":137208,"p75_ns":471792}}},"implementation":"matmul_s7_'
    'interior","repeats":3,"warmup":1},"matmul_tiled_32":{"clocks":{"host_dispatc'
    'h":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[15'
    '833,17667,8750],"timing":{"count":3,"iqr_ns":8917,"max_ns":17667,"median_ns"'
    ':15833,"min_ns":8750,"p25_ns":8750,"p75_ns":17667}},"host_synchronized":{"cl'
    'ock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[36845'
    '8,509125,353792],"timing":{"count":3,"iqr_ns":155333,"max_ns":509125,"median'
    '_ns":368458,"min_ns":353792,"p25_ns":353792,"p75_ns":509125}},"mps_event":{"'
    'clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.3094583516940'
    '4745,0.5772500007878989,0.280875014141202],"samples_ns":[309458,577250,28087'
    '5],"timing":{"count":3,"iqr_ns":296375,"max_ns":577250,"median_ns":309458,"m'
    'in_ns":280875,"p25_ns":280875,"p75_ns":577250}}},"implementation":"matmul_ti'
    'led_32","repeats":3,"warmup":1},"matmul_x_columns":{"clocks":{"host_dispatch'
    '":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[700'
    '0,16416,41375],"timing":{"count":3,"iqr_ns":34375,"max_ns":41375,"median_ns"'
    ':16416,"min_ns":7000,"p25_ns":7000,"p75_ns":41375}},"host_synchronized":{"cl'
    'ock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[30716'
    '6,809667,279875],"timing":{"count":3,"iqr_ns":529792,"max_ns":809667,"median'
    '_ns":307166,"min_ns":279875,"p25_ns":279875,"p75_ns":809667}},"mps_event":{"'
    'clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.2751249994616'
    '9555,0.2276250161230564,0.26695834822021425],"samples_ns":[275125,227625,266'
    '958],"timing":{"count":3,"iqr_ns":47500,"max_ns":275125,"median_ns":266958,"'
    'min_ns":227625,"p25_ns":227625,"p75_ns":275125}}},"implementation":"matmul_x'
    '_columns","repeats":3,"warmup":1},"matmul_x_rows":{"clocks":{"host_dispatch"'
    ':{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[2700'
    '0,5875,7416],"timing":{"count":3,"iqr_ns":21125,"max_ns":27000,"median_ns":7'
    '416,"min_ns":5875,"p25_ns":5875,"p75_ns":27000}},"host_synchronized":{"clock'
    '":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[584167,5'
    '49375,578333],"timing":{"count":3,"iqr_ns":34792,"max_ns":584167,"median_ns"'
    ':578333,"min_ns":549375,"p25_ns":549375,"p75_ns":584167}},"mps_event":{"cloc'
    'k":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.5461666733026505,'
    '0.49687499995343387,0.4919166676700115],"samples_ns":[546167,496875,491917],'
    '"timing":{"count":3,"iqr_ns":54250,"max_ns":546167,"median_ns":496875,"min_n'
    's":491917,"p25_ns":491917,"p75_ns":546167}}},"implementation":"matmul_x_rows'
    '","repeats":3,"warmup":1},"torch_matmul":{"clocks":{"host_dispatch":{"clock"'
    ':"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[34542,275291,'
    '135166],"timing":{"count":3,"iqr_ns":240749,"max_ns":275291,"median_ns":1351'
    '66,"min_ns":34542,"p25_ns":34542,"p75_ns":275291}},"host_synchronized":{"clo'
    'ck":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[221500'
    ',1223916,252542],"timing":{"count":3,"iqr_ns":1002416,"max_ns":1223916,"medi'
    'an_ns":252542,"min_ns":221500,"p25_ns":221500,"p75_ns":1223916}},"mps_event"'
    ':{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.2273333084'
    '3038857,0.20529166795313358,0.1568750012665987],"samples_ns":[227333,205292,'
    '156875],"timing":{"count":3,"iqr_ns":70458,"max_ns":227333,"median_ns":20529'
    '2,"min_ns":156875,"p25_ns":156875,"p75_ns":227333}}},"implementation":"torch'
    '_matmul","repeats":3,"warmup":1}}}],"report_sha256":"f4d25ead6ecd6a00c1433aa'
    '9440a4fca723d4cb565abf9e56fe2c281bdd95a94","run":{"elapsed_ns":691052791,"ho'
    'st_timed_ns":39257371,"max_runtime_ns":60000000000},"schema_version":1,"summ'
    'ary":{"fastest_host":{"matmul_s2_coarse_4x1":{"k":1,"m":1,"median_host_ns":3'
    '21041,"n":1},"matmul_s3_block_4x4":{"k":1,"m":1,"median_host_ns":158959,"n":'
    '1},"matmul_s4_vec4":{"k":1,"m":1,"median_host_ns":161791,"n":1},"matmul_s5_p'
    'ad4":{"k":1,"m":1,"median_host_ns":162958,"n":1},"matmul_s6_group_32x32":{"k'
    '":1,"m":1,"median_host_ns":201375,"n":1},"matmul_s6_group_8x8":{"k":100,"m":'
    '200,"median_host_ns":232542,"n":300},"matmul_s7_interior":{"k":100,"m":200,"'
    'median_host_ns":213542,"n":300},"matmul_tiled_32":{"k":1,"m":1,"median_host_'
    'ns":322250,"n":1},"matmul_x_columns":{"k":1,"m":1,"median_host_ns":220000,"n'
    '":1},"matmul_x_rows":{"k":1,"m":1,"median_host_ns":298917,"n":1},"torch_matm'
    'ul":{"k":1,"m":1,"median_host_ns":174959,"n":1}},"label":"host-specific Appl'
    'e GPU (Metal) measurement","load_average_after":[14.314453125,13.43701171875'
    ',14.12109375],"load_average_before":[14.314453125,13.43701171875,14.12109375'
    '],"points":2,"refused_variants":[],"sides":["matmul_x_columns","matmul_x_row'
    's","matmul_tiled_32","matmul_s2_coarse_4x1","matmul_s3_block_4x4","matmul_s4'
    '_vec4","matmul_s5_pad4","matmul_s6_group_8x8","matmul_s6_group_32x32","matmu'
    'l_s7_interior","torch_matmul"],"timed_samples":297,"warnings":0},"timer":"ti'
    'me.perf_counter_ns","torch_mps_switches":{"source":"os.environ, read before '
    "the runtime imports torch; null means unset, and a value that is not a short"
    ' token is \'unknown\'","values":{"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PRE'
    'FER_METAL":null}},"warnings":{"capture":"warnings.catch_warnings(record=True'
    ") with simplefilter('always'), from before the runtime imports torch to the "
    'end of the run; equal warnings are kept once with a count","count":0,"distin'
    'ct":0,"records":[],"torch_imported_before_capture":false}}'
)


def recorded():
    return json.loads(RECORDED_TIMING_JSON)


def recorded_bound():
    return build_bound_report(
        json.loads(CHECKLIST_JSON), json.loads(CEILING_JSON), recorded(), RULE
    )


def test_the_recorded_timing_verifies_as_a_measurement_on_any_host():
    report = recorded()

    summary = verify_timing_report(report, config=config_from_mapping(RECORDED_CONFIG))

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_TIMING_JSON
    assert (summary["measured"], summary["points"], summary["sides"]) == (
        True,
        2,
        SIDES,
    )
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert (summary["refused_variants"], summary["warnings"]) == ([], 0)
    assert report["backend"] == "torch.mps.compile_shader"
    assert report["timer"] == "time.perf_counter_ns"
    assert report["environment"] == json.loads(CHECKLIST_JSON)["environment"]
    assert report["evidence_boundary"]["hardware_counters_used"] is False
    assert report["warnings"]["torch_imported_before_capture"] is False
    assert summary["load_average_before"] == [
        14.314453125,
        13.43701171875,
        14.12109375,
    ]


def test_the_recorded_timing_seals_the_version_2_event_text():
    texts = recorded()["method"]["domains"]

    assert recorded()["method"] == method_mapping()
    assert texts == DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[2]
    assert "host time between the two record() calls" in texts[DEVICE_DOMAIN]


def test_each_recorded_kernel_call_was_issued_in_microseconds_and_waited_longer():
    report = recorded()

    for point in (report["floor"], *report["points"]):
        for side in SIDES:
            clocks = point["sides"][side]["clocks"]
            dispatch = clocks[DISPATCH]["timing"]["median_ns"]
            host = clocks[SYNCHRONIZED]["timing"]["median_ns"]
            assert dispatch < host and host > 150_000, (point["m"], side)
            # torch.matmul does more host work to issue a call than a kernel.
            assert side == TORCH_MATMUL or dispatch < 50_000, (point["m"], side)
            assert all(sample > 0 for sample in clocks[EVENT]["elapsed_ms"])


def test_the_three_recorded_reports_make_one_measured_bound_on_any_host():
    bound = recorded_bound()

    summary = verify_bound_report(bound, config=RULE, rebuilt=recorded_bound())

    assert summary["report_sha256"] == RECORDED_BOUND_SHA256
    assert (summary["measured"], summary["sources_checked"]) == (True, True)
    assert bound["evidence_boundary"]["inference_from_timings"] is True
    assert summary["timing_report_sha256"] == recorded()["report_sha256"]
    assert (
        summary["checklist_report_sha256"]
        == (json.loads(CHECKLIST_JSON)["report_sha256"])
    )
    reasons = {
        side: [point["sides"][side]["decision"]["reason"] for point in bound["points"]]
        for side in SIDES
    }
    # The checklist run was slower than this one-call run for torch.matmul, so
    # its steady times sit above the one-call host times.
    assert reasons[TORCH_MATMUL] == ["steady_time_above_host_time"] * 2
    # The 1 x 1 x 1 call of S7 took about as long as these small calls, so the
    # rule cannot rule out a fixed cost that fills them.
    assert reasons["matmul_s7_interior"] == ["fixed_cost_not_ruled_out"] * 2
    for point in bound["points"]:
        sides = point["sides"]
        assert sides["matmul_x_columns"]["traffic"]["roof"]["attainable_share"] > 1
        assert all(sides[side]["fma_ceiling_share"] < 0.08 for side in SIDES)
        assert point["ideal_reuse"]["bound"] == "compute"


def test_the_bound_command_rebuilds_the_recorded_bound_without_torch(tmp_path):
    paths = {
        "checklist-report": CHECKLIST_JSON,
        "ceiling-report": CEILING_JSON,
        "timing-report": RECORDED_TIMING_JSON,
    }
    arguments = ["--rule-config", str(ROOT / "config/day-75-matmul-bottleneck.json")]
    for option, text in paths.items():
        path = tmp_path / f"{option}.json"
        path.write_text(text + "\n", encoding="ascii")
        arguments += [f"--{option}", str(path)]
    output = tmp_path / "bound.json"

    def run(*argv):
        return subprocess.run(
            [sys.executable, "-c", WITHOUT_TORCH, *argv],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

    made = run("bound", *arguments, "--output", str(output))
    checked = run("verify", "--report", str(output), *arguments)
    timing = run("verify-timing", "--report", str(tmp_path / "timing-report.json"))

    assert made.returncode == 0, made.stderr
    assert made.stdout.splitlines()[-1].startswith(
        f"matmul_checklist_bound_measurement={RECORDED_BOUND_SHA256} points=2 "
        "sources_checked=true"
    )
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.startswith(
        f"verified_matmul_checklist_bound_measurement={RECORDED_BOUND_SHA256} "
    )
    assert timing.returncode == 0, timing.stderr
    assert timing.stdout.startswith(
        f"verified_matmul_checklist_one_call_measurement={recorded()['report_sha256']} "
    )
