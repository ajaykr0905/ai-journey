"""One real one-call timing from the development host, and the gap it makes.

``RECORDED_TIMING_JSON`` is the canonical file that ``matmul_tiled_gap_cli
timing`` wrote in a fresh process on 2026-11-19 at 13:04 IST under the
shared GPU lock, with torch 2.15.0 on macOS 27.2 (arm64) and load averages
near 10.4, 10.7 and 13.2 from other jobs. It times the recorded tiled config
of ``test_matmul_tiled_recorded``: a 1 x 1 x 1 floor and two sweep points,
with one warmup and three samples on each clock. Joined with the recorded
tiled report and the recorded gpu-ceilings report, it makes one measured gap
report, so a host without torch or MPS, the Linux CI included, can rebuild
and verify it. The three runs are separate, and the gap keeps the load
averages of each. The timings describe that moment and are here only to be
verified. The report is a schema_version 1 record, so it keeps the version 1
event-domain text, which calls the event interval device-side.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_timing_clocks import (
    DEVICE_DOMAIN,
    DISPATCH,
    DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION,
    EVENT,
    SYNCHRONIZED,
)
from ai_journey.matmul_bottleneck_config import load_config as load_rule_config
from ai_journey.matmul_tiled_config import config_from_mapping
from ai_journey.matmul_tiled_gap import build_gap_report
from ai_journey.matmul_tiled_gap_timing import (
    REPORT_SCHEMA_VERSION,
    MatmulTiledGapVerificationError,
    method_mapping,
)
from ai_journey.matmul_tiled_gap_timing_verify import verify_timing_report
from ai_journey.matmul_tiled_gap_verify import verify_gap_report
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD, seal_report
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_matmul_tiled_recorded import RECORDED_CONFIG
from test_matmul_tiled_recorded import RECORDED_REPORT_JSON as TILED_JSON

ROOT = Path(__file__).resolve().parents[1]
RULE = load_rule_config(ROOT / "config/day-75-matmul-bottleneck.json")
RECORDED_GAP_SHA256 = "5d70138982e33407fe0f58437834091dce3e1b7b17aa3759544af9af18cbf7ca"
SIDES = [
    "matmul_x_columns",
    "matmul_tiled_8",
    "matmul_tiled_16",
    "matmul_tiled_32",
    "torch_matmul",
]
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_tiled_gap_cli import main
sys.exit(main(sys.argv[1:]))
"""
RECORDED_TIMING_JSON = (
    '{"backend":"torch.mps.compile_shader","config":{"benchmark":{"max_runtime_se'
    'conds":60,"repeats":3,"shapes":[[256,256,256],[200,300,100]],"target_flops_p'
    'er_sample":16777216,"warmup":1},"comparison":{"min_faster_ratio":1.1},"corre'
    'ctness":{"shapes":[[1,1,1],[17,33,9],[31,47,129]]},"naive":{"group_size":[32'
    ',8]},"name":"matmul-tiled-recorded","schema_version":1,"seed":76,"variants":'
    '["matmul_tiled_8","matmul_tiled_16","matmul_tiled_32"]},"config_sha256":"6a5'
    '5c69ef00c1dd51f600b90b0c827b40bf4df0492ff58faf9206c581a6b23bc","environment"'
    ':{"macos_version":"27.2","numpy_version":"1.26.4","platform_machine":"arm64"'
    ',"platform_system":"Darwin","python_implementation":"CPython","python_versio'
    'n":"3.11.14","torch_float32_matmul_precision":"highest","torch_version":"2.1'
    '5.0"},"evidence_boundary":{"apple_gpu_metal_this_host":true,"cublas_measured'
    '":false,"cuda_kernel_executed":false,"distributed_claim":false,"hardware_cou'
    'nters_used":false,"host_specific":true,"nvidia_gpu_used":false,"portable_per'
    'formance_claim":false,"production_claim":false,"statement":"These are one-ca'
    "ll timings on the Apple GPU of the host that made this report, at the record"
    "ed load averages, on two clocks: MPS events in the command stream and perf_c"
    "ounter_ns on the host. An MPS event interval holds the signal latency of its"
    " events, so it is longer than the kernel. No CUDA kernel ran, no NVIDIA GPU "
    "was used, cuBLAS was not measured, and no hardware counter was read. These t"
    "imings do not measure or predict CUDA, NVIDIA, cuBLAS, distributed or produc"
    'tion performance.","vendor_peak_claimed":false},"floor":{"agreement_bound":2'
    '.3841859331241806e-07,"checks":{"matmul_tiled_16":{"disagreement":0.0,"same_'
    'bits_as_naive":true},"matmul_tiled_32":{"disagreement":0.0,"same_bits_as_nai'
    've":true},"matmul_tiled_8":{"disagreement":0.0,"same_bits_as_naive":true},"t'
    'orch_matmul":{"disagreement":0.0,"same_bits_as_naive":true}},"flops_per_call'
    '":2,"index":16,"k":1,"m":1,"n":1,"side_order":["matmul_x_columns","matmul_ti'
    'led_8","matmul_tiled_16","matmul_tiled_32","torch_matmul"],"sides":{"matmul_'
    'tiled_16":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host'
    '_perf_counter_ns","samples_ns":[8209,126375,7500],"timing":{"count":3,"iqr_n'
    's":118875,"max_ns":126375,"median_ns":8209,"min_ns":7500,"p25_ns":7500,"p75_'
    'ns":126375}},"host_synchronized":{"clock":"host_synchronized","domain":"host'
    '_perf_counter_ns","samples_ns":[319042,290292,319875],"timing":{"count":3,"i'
    'qr_ns":29583,"max_ns":319875,"median_ns":319042,"min_ns":290292,"p25_ns":290'
    '292,"p75_ns":319875}},"mps_event":{"clock":"mps_event","domain":"mps_event_s'
    'tream","elapsed_ms":[0.2381249942118302,0.28287500026635826,0.37062499905005'
    '1],"samples_ns":[238125,282875,370625],"timing":{"count":3,"iqr_ns":132500,"'
    'max_ns":370625,"median_ns":282875,"min_ns":238125,"p25_ns":238125,"p75_ns":3'
    '70625}}},"implementation":"matmul_tiled_16","repeats":3,"warmup":1},"matmul_'
    'tiled_32":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host'
    '_perf_counter_ns","samples_ns":[10750,8333,7417],"timing":{"count":3,"iqr_ns'
    '":3333,"max_ns":10750,"median_ns":8333,"min_ns":7417,"p25_ns":7417,"p75_ns":'
    '10750}},"host_synchronized":{"clock":"host_synchronized","domain":"host_perf'
    '_counter_ns","samples_ns":[328417,122625,316958],"timing":{"count":3,"iqr_ns'
    '":205792,"max_ns":328417,"median_ns":316958,"min_ns":122625,"p25_ns":122625,'
    '"p75_ns":328417}},"mps_event":{"clock":"mps_event","domain":"mps_event_strea'
    'm","elapsed_ms":[0.23587500618305057,0.07216667290776968,0.22650000755675137'
    '],"samples_ns":[235875,72167,226500],"timing":{"count":3,"iqr_ns":163708,"ma'
    'x_ns":235875,"median_ns":226500,"min_ns":72167,"p25_ns":72167,"p75_ns":23587'
    '5}}},"implementation":"matmul_tiled_32","repeats":3,"warmup":1},"matmul_tile'
    'd_8":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf'
    '_counter_ns","samples_ns":[8709,7791,8750],"timing":{"count":3,"iqr_ns":959,'
    '"max_ns":8750,"median_ns":8709,"min_ns":7791,"p25_ns":7791,"p75_ns":8750}},"'
    'host_synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_'
    'ns","samples_ns":[302500,136292,293917],"timing":{"count":3,"iqr_ns":166208,'
    '"max_ns":302500,"median_ns":293917,"min_ns":136292,"p25_ns":136292,"p75_ns":'
    '302500}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elaps'
    'ed_ms":[0.2384999970672652,0.06900000153109431,0.22412500402424484],"samples'
    '_ns":[238500,69000,224125],"timing":{"count":3,"iqr_ns":169500,"max_ns":2385'
    '00,"median_ns":224125,"min_ns":69000,"p25_ns":69000,"p75_ns":238500}}},"impl'
    'ementation":"matmul_tiled_8","repeats":3,"warmup":1},"matmul_x_columns":{"cl'
    'ocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_'
    'ns","samples_ns":[20042,7375,8000],"timing":{"count":3,"iqr_ns":12667,"max_n'
    's":20042,"median_ns":8000,"min_ns":7375,"p25_ns":7375,"p75_ns":20042}},"host'
    '_synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns",'
    '"samples_ns":[296042,117167,305292],"timing":{"count":3,"iqr_ns":188125,"max'
    '_ns":305292,"median_ns":296042,"min_ns":117167,"p25_ns":117167,"p75_ns":3052'
    '92}},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_m'
    's":[0.36583334440365434,0.07962500967551023,0.33075000101234764],"samples_ns'
    '":[365833,79625,330750],"timing":{"count":3,"iqr_ns":286208,"max_ns":365833,'
    '"median_ns":330750,"min_ns":79625,"p25_ns":79625,"p75_ns":365833}}},"impleme'
    'ntation":"matmul_x_columns","repeats":3,"warmup":1},"torch_matmul":{"clocks"'
    ':{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","'
    'samples_ns":[8125,8250,7959],"timing":{"count":3,"iqr_ns":291,"max_ns":8250,'
    '"median_ns":8125,"min_ns":7959,"p25_ns":7959,"p75_ns":8250}},"host_synchroni'
    'zed":{"clock":"host_synchronized","domain":"host_perf_counter_ns","samples_n'
    's":[140917,135625,300625],"timing":{"count":3,"iqr_ns":165000,"max_ns":30062'
    '5,"median_ns":140917,"min_ns":135625,"p25_ns":135625,"p75_ns":300625}},"mps_'
    'event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.0942'
    '5000462215394,0.0821249996079132,0.22587500279769301],"samples_ns":[94250,82'
    '125,225875],"timing":{"count":3,"iqr_ns":143750,"max_ns":225875,"median_ns":'
    '94250,"min_ns":82125,"p25_ns":82125,"p75_ns":225875}}},"implementation":"tor'
    'ch_matmul","repeats":3,"warmup":1}}},"host_load":{"after":[10.39892578125,10'
    '.6953125,13.1787109375],"before":[10.39892578125,10.6953125,13.1787109375],"'
    'source":"os.getloadavg(): the 1, 5 and 15 minute load averages that uptime p'
    'rints, read before the checks and after the last sample"},"kernels":{"admiss'
    'ions":[{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":'
    '512,"name":"matmul_tiled_8","reason":null,"required_threads":64,"static_thre'
    'adgroup_bytes":512},{"admitted":true,"allowed_threads":1024,"declared_thread'
    'group_bytes":2048,"name":"matmul_tiled_16","reason":null,"required_threads":'
    '256,"static_threadgroup_bytes":2048},{"admitted":true,"allowed_threads":1024'
    ',"declared_threadgroup_bytes":8192,"name":"matmul_tiled_32","reason":null,"r'
    'equired_threads":1024,"static_threadgroup_bytes":8192}],"limits":{"matmul_ti'
    'led_16":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes'
    '":2048,"thread_execution_width":32},"matmul_tiled_32":{"max_threads_per_thre'
    'adgroup":1024,"static_threadgroup_memory_bytes":8192,"thread_execution_width'
    '":32},"matmul_tiled_8":{"max_threads_per_threadgroup":1024,"static_threadgro'
    'up_memory_bytes":512,"thread_execution_width":32},"matmul_x_columns":{"max_t'
    'hreads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_exe'
    'cution_width":32}},"naive_source_sha256":"beef26392d0034be037dbd0740d2b5eca9'
    '429f76cc8bda48ba4d37fc928df9cc","tiled_source_sha256":"08b88577018024d54344a'
    '00f991f387225f1085047caac66f347dc5e719a5fa5"},"kind":"matmul_tiled_one_call_'
    'timing","label":"host-specific Apple GPU (Metal) measurement","measured":tru'
    'e,"measurement_scope":"apple_gpu_metal_this_host","method":{"agreement":"eve'
    "ry side within 4 * gamma_K(2**-24) * (|A| @ |B|) of the naive output, on the"
    ' device in float32","clocks":{"host_dispatch":"time.perf_counter_ns() just b'
    "efore and just after the launch call alone, with no synchronize between the "
    "readings; the queue is drained before the first reading and after the second"
    '","host_synchronized":"time.perf_counter_ns() just before the launch and jus'
    "t after the torch.mps.synchronize() that follows it; the queue is drained be"
    'fore the first reading","mps_event":"torch.mps.Event(enable_timing=True): st'
    "art.record() just before the launch and end.record() just after it, then tor"
    "ch.mps.synchronize() and start.elapsed_time(end) in ms; samples_ns = round(m"
    's * 1e6)"},"domains":{"host_perf_counter_ns":"host-side: time.perf_counter_n'
    's on the CPU thread that issued the launch","mps_event_stream":"device-side:'
    " the interval between two points in the MPS command stream; the PyTorch MPS "
    "header says a command-buffer completion handler stores each timed event's co"
    "mpletion time and does not name the clock it reads, so these values are neve"
    'r subtracted from host values without a cross_domain record"},"floor":"every'
    " side first runs one 1 x 1 x 1 call on operands from case index 16; its host"
    ' time bounds a fixed cost that every call of the side pays","operands":"poin'
    "t i uses the operands of the tiled run's point i: seed, benchmark stream and"
    ' case index i of the Day 74 generator","schedule":"one launch and one clock '
    "per sample; per side, warmup untimed samples on each clock first; then repea"
    "t r visits the sides shifted by r and, for each side, the clocks shifted by "
    'r"},"percentile_method":"nearest rank: sorted samples[ceil(p * n / 100) - 1]'
    " for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no interpolatio"
    'n","points":[{"agreement_bound":6.103608758678569e-05,"checks":{"matmul_tile'
    'd_16":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_tiled_32":{"dis'
    'agreement":0.0,"same_bits_as_naive":true},"matmul_tiled_8":{"disagreement":0'
    '.0,"same_bits_as_naive":true},"torch_matmul":{"disagreement":0.0,"same_bits_'
    'as_naive":true}},"flops_per_call":33554432,"index":0,"k":256,"m":256,"n":256'
    ',"side_order":["matmul_x_columns","matmul_tiled_8","matmul_tiled_16","matmul'
    '_tiled_32","torch_matmul"],"sides":{"matmul_tiled_16":{"clocks":{"host_dispa'
    'tch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":['
    '9250,7417,11083],"timing":{"count":3,"iqr_ns":3666,"max_ns":11083,"median_ns'
    '":9250,"min_ns":7417,"p25_ns":7417,"p75_ns":11083}},"host_synchronized":{"cl'
    'ock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[63845'
    '8,631667,663833],"timing":{"count":3,"iqr_ns":32166,"max_ns":663833,"median_'
    'ns":638458,"min_ns":631667,"p25_ns":631667,"p75_ns":663833}},"mps_event":{"c'
    'lock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.54587499471381'
    '31,0.5492916679941118,0.55104166676756],"samples_ns":[545875,549292,551042],'
    '"timing":{"count":3,"iqr_ns":5167,"max_ns":551042,"median_ns":549292,"min_ns'
    '":545875,"p25_ns":545875,"p75_ns":551042}}},"implementation":"matmul_tiled_1'
    '6","repeats":3,"warmup":1},"matmul_tiled_32":{"clocks":{"host_dispatch":{"cl'
    'ock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[8959,1262'
    '5,19791],"timing":{"count":3,"iqr_ns":10832,"max_ns":19791,"median_ns":12625'
    ',"min_ns":8959,"p25_ns":8959,"p75_ns":19791}},"host_synchronized":{"clock":"'
    'host_synchronized","domain":"host_perf_counter_ns","samples_ns":[578709,8077'
    '08,388375],"timing":{"count":3,"iqr_ns":419333,"max_ns":807708,"median_ns":5'
    '78709,"min_ns":388375,"p25_ns":388375,"p75_ns":807708}},"mps_event":{"clock"'
    ':"mps_event","domain":"mps_event_stream","elapsed_ms":[0.5404999974416569,0.'
    '520500005222857,0.5025833379477262],"samples_ns":[540500,520500,502583],"tim'
    'ing":{"count":3,"iqr_ns":37917,"max_ns":540500,"median_ns":520500,"min_ns":5'
    '02583,"p25_ns":502583,"p75_ns":540500}}},"implementation":"matmul_tiled_32",'
    '"repeats":3,"warmup":1},"matmul_tiled_8":{"clocks":{"host_dispatch":{"clock"'
    ':"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[17666,8541,77'
    '50],"timing":{"count":3,"iqr_ns":9916,"max_ns":17666,"median_ns":8541,"min_n'
    's":7750,"p25_ns":7750,"p75_ns":17666}},"host_synchronized":{"clock":"host_sy'
    'nchronized","domain":"host_perf_counter_ns","samples_ns":[968500,976458,2814'
    '59],"timing":{"count":3,"iqr_ns":694999,"max_ns":976458,"median_ns":968500,"'
    'min_ns":281459,"p25_ns":281459,"p75_ns":976458}},"mps_event":{"clock":"mps_e'
    'vent","domain":"mps_event_stream","elapsed_ms":[0.890750001417473,0.86749999'
    '90034848,0.37650000012945384],"samples_ns":[890750,867500,376500],"timing":{'
    '"count":3,"iqr_ns":514250,"max_ns":890750,"median_ns":867500,"min_ns":376500'
    ',"p25_ns":376500,"p75_ns":890750}}},"implementation":"matmul_tiled_8","repea'
    'ts":3,"warmup":1},"matmul_x_columns":{"clocks":{"host_dispatch":{"clock":"ho'
    'st_dispatch","domain":"host_perf_counter_ns","samples_ns":[19708,8584,11709]'
    ',"timing":{"count":3,"iqr_ns":11124,"max_ns":19708,"median_ns":11709,"min_ns'
    '":8584,"p25_ns":8584,"p75_ns":19708}},"host_synchronized":{"clock":"host_syn'
    'chronized","domain":"host_perf_counter_ns","samples_ns":[665667,732667,46370'
    '8],"timing":{"count":3,"iqr_ns":268959,"max_ns":732667,"median_ns":665667,"m'
    'in_ns":463708,"p25_ns":463708,"p75_ns":732667}},"mps_event":{"clock":"mps_ev'
    'ent","domain":"mps_event_stream","elapsed_ms":[0.5630000086966902,0.65724999'
    '87669289,0.3982500056736171],"samples_ns":[563000,657250,398250],"timing":{"'
    'count":3,"iqr_ns":259000,"max_ns":657250,"median_ns":563000,"min_ns":398250,'
    '"p25_ns":398250,"p75_ns":657250}}},"implementation":"matmul_x_columns","repe'
    'ats":3,"warmup":1},"torch_matmul":{"clocks":{"host_dispatch":{"clock":"host_'
    'dispatch","domain":"host_perf_counter_ns","samples_ns":[48166,78166,125542],'
    '"timing":{"count":3,"iqr_ns":77376,"max_ns":125542,"median_ns":78166,"min_ns'
    '":48166,"p25_ns":48166,"p75_ns":125542}},"host_synchronized":{"clock":"host_'
    'synchronized","domain":"host_perf_counter_ns","samples_ns":[572833,651458,42'
    '7916],"timing":{"count":3,"iqr_ns":223542,"max_ns":651458,"median_ns":572833'
    ',"min_ns":427916,"p25_ns":427916,"p75_ns":651458}},"mps_event":{"clock":"mps'
    '_event","domain":"mps_event_stream","elapsed_ms":[0.567750001209788,0.446999'
    '998530373,0.4887500108452514],"samples_ns":[567750,447000,488750],"timing":{'
    '"count":3,"iqr_ns":120750,"max_ns":567750,"median_ns":488750,"min_ns":447000'
    ',"p25_ns":447000,"p75_ns":567750}}},"implementation":"torch_matmul","repeats'
    '":3,"warmup":1}}},{"agreement_bound":2.384200001955044e-05,"checks":{"matmul'
    '_tiled_16":{"disagreement":0.0,"same_bits_as_naive":true},"matmul_tiled_32":'
    '{"disagreement":0.0,"same_bits_as_naive":true},"matmul_tiled_8":{"disagreeme'
    'nt":0.0,"same_bits_as_naive":true},"torch_matmul":{"disagreement":0.0,"same_'
    'bits_as_naive":true}},"flops_per_call":12000000,"index":1,"k":100,"m":200,"n'
    '":300,"side_order":["matmul_x_columns","matmul_tiled_8","matmul_tiled_16","m'
    'atmul_tiled_32","torch_matmul"],"sides":{"matmul_tiled_16":{"clocks":{"host_'
    'dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_'
    'ns":[7250,7375,8208],"timing":{"count":3,"iqr_ns":958,"max_ns":8208,"median_'
    'ns":7375,"min_ns":7250,"p25_ns":7250,"p75_ns":8208}},"host_synchronized":{"c'
    'lock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[1968'
    '33,193708,160334],"timing":{"count":3,"iqr_ns":36499,"max_ns":196833,"median'
    '_ns":193708,"min_ns":160334,"p25_ns":160334,"p75_ns":196833}},"mps_event":{"'
    'clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.1156249927589'
    '6698,0.12062500172760338,0.11129166523460299],"samples_ns":[115625,120625,11'
    '1292],"timing":{"count":3,"iqr_ns":9333,"max_ns":120625,"median_ns":115625,"'
    'min_ns":111292,"p25_ns":111292,"p75_ns":120625}}},"implementation":"matmul_t'
    'iled_16","repeats":3,"warmup":1},"matmul_tiled_32":{"clocks":{"host_dispatch'
    '":{"clock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[766'
    '7,16583,8333],"timing":{"count":3,"iqr_ns":8916,"max_ns":16583,"median_ns":8'
    '333,"min_ns":7667,"p25_ns":7667,"p75_ns":16583}},"host_synchronized":{"clock'
    '":"host_synchronized","domain":"host_perf_counter_ns","samples_ns":[206875,4'
    '90125,168667],"timing":{"count":3,"iqr_ns":321458,"max_ns":490125,"median_ns'
    '":206875,"min_ns":168667,"p25_ns":168667,"p75_ns":490125}},"mps_event":{"clo'
    'ck":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.1225000014528632'
    '2,0.1097083295462653,0.1165416615549475],"samples_ns":[122500,109708,116542]'
    ',"timing":{"count":3,"iqr_ns":12792,"max_ns":122500,"median_ns":116542,"min_'
    'ns":109708,"p25_ns":109708,"p75_ns":122500}}},"implementation":"matmul_tiled'
    '_32","repeats":3,"warmup":1},"matmul_tiled_8":{"clocks":{"host_dispatch":{"c'
    'lock":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[10333,76'
    '25,7958],"timing":{"count":3,"iqr_ns":2708,"max_ns":10333,"median_ns":7958,"'
    'min_ns":7625,"p25_ns":7625,"p75_ns":10333}},"host_synchronized":{"clock":"ho'
    'st_synchronized","domain":"host_perf_counter_ns","samples_ns":[184250,205000'
    ',161750],"timing":{"count":3,"iqr_ns":43250,"max_ns":205000,"median_ns":1842'
    '50,"min_ns":161750,"p25_ns":161750,"p75_ns":205000}},"mps_event":{"clock":"m'
    'ps_event","domain":"mps_event_stream","elapsed_ms":[0.11295833974145353,0.12'
    '162499479018152,0.11324998922646046],"samples_ns":[112958,121625,113250],"ti'
    'ming":{"count":3,"iqr_ns":8667,"max_ns":121625,"median_ns":113250,"min_ns":1'
    '12958,"p25_ns":112958,"p75_ns":121625}}},"implementation":"matmul_tiled_8","'
    'repeats":3,"warmup":1},"matmul_x_columns":{"clocks":{"host_dispatch":{"clock'
    '":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[8833,9417,92'
    '08],"timing":{"count":3,"iqr_ns":584,"max_ns":9417,"median_ns":9208,"min_ns"'
    ':8833,"p25_ns":8833,"p75_ns":9417}},"host_synchronized":{"clock":"host_synch'
    'ronized","domain":"host_perf_counter_ns","samples_ns":[167000,161875,155042]'
    ',"timing":{"count":3,"iqr_ns":11958,"max_ns":167000,"median_ns":161875,"min_'
    'ns":155042,"p25_ns":155042,"p75_ns":167000}},"mps_event":{"clock":"mps_event'
    '","domain":"mps_event_stream","elapsed_ms":[0.11249999806750566,0.1039583294'
    '1867411,0.1049999991664663],"samples_ns":[112500,103958,105000],"timing":{"c'
    'ount":3,"iqr_ns":8542,"max_ns":112500,"median_ns":105000,"min_ns":103958,"p2'
    '5_ns":103958,"p75_ns":112500}}},"implementation":"matmul_x_columns","repeats'
    '":3,"warmup":1},"torch_matmul":{"clocks":{"host_dispatch":{"clock":"host_dis'
    'patch","domain":"host_perf_counter_ns","samples_ns":[36000,49875,94583],"tim'
    'ing":{"count":3,"iqr_ns":58583,"max_ns":94583,"median_ns":49875,"min_ns":360'
    '00,"p25_ns":36000,"p75_ns":94583}},"host_synchronized":{"clock":"host_synchr'
    'onized","domain":"host_perf_counter_ns","samples_ns":[194792,242833,155125],'
    '"timing":{"count":3,"iqr_ns":87708,"max_ns":242833,"median_ns":194792,"min_n'
    's":155125,"p25_ns":155125,"p75_ns":242833}},"mps_event":{"clock":"mps_event"'
    ',"domain":"mps_event_stream","elapsed_ms":[0.1348750083707273,0.104125007055'
    '69983,0.1279166608583182],"samples_ns":[134875,104125,127917],"timing":{"cou'
    'nt":3,"iqr_ns":30750,"max_ns":134875,"median_ns":127917,"min_ns":104125,"p25'
    '_ns":104125,"p75_ns":134875}}},"implementation":"torch_matmul","repeats":3,"'
    'warmup":1}}}],"report_sha256":"ec4525e7fd121ee74774e2dda992a8b9f48ab5f8dccd8'
    '6d887a338af496750d6","run":{"elapsed_ns":727886333,"host_timed_ns":17157001,'
    '"max_runtime_ns":60000000000},"schema_version":1,"summary":{"fastest_host":{'
    '"matmul_tiled_16":{"k":100,"m":200,"median_host_ns":193708,"n":300},"matmul_'
    'tiled_32":{"k":100,"m":200,"median_host_ns":206875,"n":300},"matmul_tiled_8"'
    ':{"k":100,"m":200,"median_host_ns":184250,"n":300},"matmul_x_columns":{"k":1'
    '00,"m":200,"median_host_ns":161875,"n":300},"torch_matmul":{"k":1,"m":1,"med'
    'ian_host_ns":140917,"n":1}},"label":"host-specific Apple GPU (Metal) measure'
    'ment","load_average_after":[10.39892578125,10.6953125,13.1787109375],"load_a'
    'verage_before":[10.39892578125,10.6953125,13.1787109375],"points":2,"sides":'
    '["matmul_x_columns","matmul_tiled_8","matmul_tiled_16","matmul_tiled_32","to'
    'rch_matmul"],"timed_samples":135},"timer":"time.perf_counter_ns","torch_mps_'
    'switches":{"source":"os.environ, read before the runtime imports torch; null'
    ' means unset, and a value that is not a short token is \'unknown\'","values":{'
    '"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL":null}}}'
)


def recorded():
    return json.loads(RECORDED_TIMING_JSON)


def recorded_gap():
    return build_gap_report(
        json.loads(TILED_JSON), json.loads(CEILING_JSON), recorded(), RULE
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
    assert report["backend"] == "torch.mps.compile_shader"
    assert report["timer"] == "time.perf_counter_ns"
    assert report["environment"] == json.loads(TILED_JSON)["environment"]
    assert report["evidence_boundary"]["hardware_counters_used"] is False
    assert summary["load_average_before"] == [
        10.39892578125,
        10.6953125,
        13.1787109375,
    ]


def test_the_recorded_timing_is_a_schema_1_record_with_the_device_side_text():
    report = recorded()
    texts = report["method"]["domains"]

    summary = verify_timing_report(report)

    assert summary["measured"] is True
    assert report["schema_version"] == 1 < REPORT_SCHEMA_VERSION
    assert report["method"] == method_mapping(1)
    assert texts == DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[1]
    assert texts[DEVICE_DOMAIN].startswith("device-side: ")


@pytest.mark.parametrize(
    ("version", "domains"), [(1, 2), (2, 1)], ids=["schema-1", "schema-2"]
)
def test_the_recorded_timing_refuses_the_domain_texts_of_the_other_version(
    version, domains
):
    report = recorded()
    del report[DIGEST_FIELD]
    report["schema_version"] = version
    report["method"]["domains"] = dict(DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[domains])

    with pytest.raises(
        MatmulTiledGapVerificationError,
        match=f"method texts are not those of schema_version {version}$",
    ):
        verify_timing_report(seal_report(report))


def test_each_recorded_call_was_issued_in_microseconds_and_waited_for_longer():
    report = recorded()

    for point in (report["floor"], *report["points"]):
        for side in SIDES:
            clocks = point["sides"][side]["clocks"]
            dispatch = clocks[DISPATCH]["timing"]["median_ns"]
            host = clocks[SYNCHRONIZED]["timing"]["median_ns"]
            assert dispatch < 100_000 < host, (point["m"], side)
            assert all(sample > 0 for sample in clocks[EVENT]["elapsed_ms"])


def test_the_three_recorded_reports_make_one_measured_gap_on_any_host():
    gap = recorded_gap()

    summary = verify_gap_report(gap, config=RULE, rebuilt=recorded_gap())

    assert summary["report_sha256"] == RECORDED_GAP_SHA256
    assert (summary["measured"], summary["sources_checked"]) == (True, True)
    assert gap["evidence_boundary"]["inference_from_timings"] is True
    assert summary["timing_report_sha256"] == recorded()["report_sha256"]
    assert summary["tiled_report_sha256"] == json.loads(TILED_JSON)["report_sha256"]
    first, second = gap["points"]
    names = {
        side: [point["sides"][side]["decision"]["reason"] for point in gap["points"]]
        for side in SIDES
    }
    # The tiled run was slower than this one-call run, so most steady times
    # sit above the one-call host times, and the rule claims no name there.
    assert names["matmul_x_columns"] == ["steady_time_above_host_time"] * 2
    assert names["matmul_tiled_8"] == [
        "below_both_roofs",
        "steady_time_above_host_time",
    ]
    assert names["torch_matmul"][1] == "fixed_cost_not_ruled_out"
    # The naive kernel beat the bandwidth roof of its no-reuse traffic.
    for point in (first, second):
        assert point["sides"]["matmul_x_columns"]["traffic"]["attainable_share"] > 1
        assert all(point["sides"][side]["fma_ceiling_share"] < 0.04 for side in SIDES)
        assert point["ideal_reuse"]["bound"] == "compute"


def test_the_gap_command_rebuilds_the_recorded_gap_without_torch(tmp_path):
    paths = {
        "tiled-report": TILED_JSON,
        "ceiling-report": CEILING_JSON,
        "timing-report": RECORDED_TIMING_JSON,
    }
    arguments = ["--rule-config", str(ROOT / "config/day-75-matmul-bottleneck.json")]
    for option, text in paths.items():
        path = tmp_path / f"{option}.json"
        path.write_text(text + "\n", encoding="ascii")
        arguments += [f"--{option}", str(path)]
    output = tmp_path / "gap.json"

    def run(*argv):
        return subprocess.run(
            [sys.executable, "-c", WITHOUT_TORCH, *argv],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

    made = run("gap", *arguments, "--output", str(output))
    checked = run("verify", "--report", str(output), *arguments)

    assert made.returncode == 0, made.stderr
    assert made.stdout.splitlines()[-1].startswith(
        f"matmul_tiled_gap_measurement={RECORDED_GAP_SHA256} points=2 "
        "sources_checked=true"
    )
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.startswith(
        f"verified_matmul_tiled_gap_measurement={RECORDED_GAP_SHA256} "
    )
