"""One real measured report from the development host's Apple GPU, replayed anywhere.

``RECORDED_REPORT_JSON`` is the canonical file that ``gpu_ceiling_cli run``
wrote on 2026-11-17 (torch 2.15.0, macOS 27.2, arm64) for a small config:
two chain counts at three step counts on 65536 threads, one 4 MiB buffer,
and strides 1 and 32 with 16 trips. Other jobs kept the host's 1-minute
load average near 9.5. Launches this small are dominated by the fixed cost
of a call, and the report's own fit says so: its FMA "ceiling" does not
scale with work. The timings describe that moment and are here only to be
verified, on hosts without torch or MPS as well.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from ai_journey.gpu_ceiling_cli import EXIT_OK, main
from ai_journey.gpu_ceiling_config import config_from_mapping
from ai_journey.gpu_ceiling_io import load_report
from ai_journey.gpu_ceiling_verify import verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_ceiling_backend import RECORDED_LIMITS, RECORDED_TORCH_VERSION

ROOT = Path(__file__).resolve().parents[1]
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"bank":[{"accesses_per_'
    'call":16777216,"calls_per_sample":1,"iqr_ns_per_call":793500.0,"kernel":"b'
    'ank_rows_32","launch":{"active_threads":65536,"extent":[65536],"group_size'
    '":[256],"groups":[256],"idle_lanes_per_threadgroup":0,"launched_threads":6'
    '5536,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[65536]'
    ',"threads_per_group":256},"median_ns_per_call":329625.0,"median_ps_per_acc'
    'ess":19.647181034088135,"output_sha256":"10a72619a67874889bdd19c71db14a310'
    '5fdde8aa4281a180432393b13af5d35","repeats":3,"row_words":32,"samples_ns":['
    '975000,181500,329625],"stride":1,"threads":65536,"timing":{"count":3,"iqr_'
    'ns":793500,"max_ns":975000,"median_ns":329625,"min_ns":181500,"p25_ns":181'
    '500,"p75_ns":975000},"trips":16,"warmup":1},{"accesses_per_call":16777216,'
    '"calls_per_sample":1,"iqr_ns_per_call":449042.0,"kernel":"bank_rows_32","l'
    'aunch":{"active_threads":65536,"extent":[65536],"group_size":[256],"groups'
    '":[256],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"padding_t'
    'hreads":0,"simd_groups_per_threadgroup":8,"threads":[65536],"threads_per_g'
    'roup":256},"median_ns_per_call":600750.0,"median_ps_per_access":35.8074903'
    '4881592,"output_sha256":"aa230530a15f973ee53ebbd41eadfd2a22a3ac01e766761d0'
    'e726037848b1b03","repeats":3,"row_words":32,"samples_ns":[600750,591541,10'
    '40583],"stride":32,"threads":65536,"timing":{"count":3,"iqr_ns":449042,"ma'
    'x_ns":1040583,"median_ns":600750,"min_ns":591541,"p25_ns":591541,"p75_ns":'
    '1040583},"trips":16,"warmup":1},{"accesses_per_call":16777216,"calls_per_s'
    'ample":1,"iqr_ns_per_call":7833.0,"kernel":"bank_rows_33","launch":{"activ'
    'e_threads":65536,"extent":[65536],"group_size":[256],"groups":[256],"idle_'
    'lanes_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"sim'
    'd_groups_per_threadgroup":8,"threads":[65536],"threads_per_group":256},"me'
    'dian_ns_per_call":174125.0,"median_ps_per_access":10.378658771514893,"outp'
    'ut_sha256":"b59daf540eba2f157c3f01033f4aec872242c3e2c4f9bd20e325aa04f0cdae'
    '0c","repeats":3,"row_words":33,"samples_ns":[176000,168167,174125],"stride'
    '":1,"threads":65536,"timing":{"count":3,"iqr_ns":7833,"max_ns":176000,"med'
    'ian_ns":174125,"min_ns":168167,"p25_ns":168167,"p75_ns":176000},"trips":16'
    ',"warmup":1},{"accesses_per_call":16777216,"calls_per_sample":1,"iqr_ns_pe'
    'r_call":5167.0,"kernel":"bank_rows_33","launch":{"active_threads":65536,"e'
    'xtent":[65536],"group_size":[256],"groups":[256],"idle_lanes_per_threadgro'
    'up":0,"launched_threads":65536,"padding_threads":0,"simd_groups_per_thread'
    'group":8,"threads":[65536],"threads_per_group":256},"median_ns_per_call":2'
    '39959.0,"median_ps_per_access":14.302670955657959,"output_sha256":"dc43b54'
    '233b09d9cc47d2a2f531a2e86969267b73bd2d81ab0910723e1634468","repeats":3,"ro'
    'w_words":33,"samples_ns":[239959,237375,242542],"stride":32,"threads":6553'
    '6,"timing":{"count":3,"iqr_ns":5167,"max_ns":242542,"median_ns":239959,"mi'
    'n_ns":237375,"p25_ns":237375,"p75_ns":242542},"trips":16,"warmup":1}],"dev'
    'ice":"mps","fma":[{"best_gflops_per_second":62.8949053420806,"calls_per_sa'
    'mple":1,"chains":1,"flops_per_call":8388608,"fma_steps":64,"iqr_ns_per_cal'
    'l":711958.0,"kernel":"fma_chains_1","launch":{"active_threads":65536,"exte'
    'nt":[65536],"group_size":[256],"groups":[256],"idle_lanes_per_threadgroup"'
    ':0,"launched_threads":65536,"padding_threads":0,"simd_groups_per_threadgro'
    'up":8,"threads":[65536],"threads_per_group":256},"median_gflops_per_second'
    '":62.8949053420806,"median_ns_per_call":133375.0,"output_sha256":"62911b2f'
    '5cd1a439f7745cfae56258cf159abe8edd3b354084e172d3e735bf6d","repeats":3,"sam'
    'ples_ns":[845333,133375,133375],"threads":65536,"timing":{"count":3,"iqr_n'
    's":711958,"max_ns":845333,"median_ns":133375,"min_ns":133375,"p25_ns":1333'
    '75,"p75_ns":845333},"warmup":1},{"best_gflops_per_second":229.040491467576'
    '8,"calls_per_sample":1,"chains":1,"flops_per_call":33554432,"fma_steps":25'
    '6,"iqr_ns_per_call":163125.0,"kernel":"fma_chains_1","launch":{"active_thr'
    'eads":65536,"extent":[65536],"group_size":[256],"groups":[256],"idle_lanes'
    '_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"simd_gro'
    'ups_per_threadgroup":8,"threads":[65536],"threads_per_group":256},"median_'
    'gflops_per_second":214.46287182502653,"median_ns_per_call":156458.0,"outpu'
    't_sha256":"738f0d5b455803957e7726deaedd2c080a6858d131b9d28005931e6d2d55596'
    'a","repeats":3,"samples_ns":[309625,146500,156458],"threads":65536,"timing'
    '":{"count":3,"iqr_ns":163125,"max_ns":309625,"median_ns":156458,"min_ns":1'
    '46500,"p25_ns":146500,"p75_ns":309625},"warmup":1},{"best_gflops_per_secon'
    'd":778.0737855072464,"calls_per_sample":1,"chains":1,"flops_per_call":1342'
    '17728,"fma_steps":1024,"iqr_ns_per_call":85417.0,"kernel":"fma_chains_1","'
    'launch":{"active_threads":65536,"extent":[65536],"group_size":[256],"group'
    's":[256],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"padding_'
    'threads":0,"simd_groups_per_threadgroup":8,"threads":[65536],"threads_per_'
    'group":256},"median_gflops_per_second":768.9709525501025,"median_ns_per_ca'
    'll":174542.0,"output_sha256":"d8717b76e7860636a5cafa0a9f198e8e32c93d58ae04'
    '631ce797462aa9b9920f","repeats":3,"samples_ns":[257917,174542,172500],"thr'
    'eads":65536,"timing":{"count":3,"iqr_ns":85417,"max_ns":257917,"median_ns"'
    ':174542,"min_ns":172500,"p25_ns":172500,"p75_ns":257917},"warmup":1},{"bes'
    't_gflops_per_second":453.9504981972902,"calls_per_sample":1,"chains":8,"fl'
    'ops_per_call":67108864,"fma_steps":64,"iqr_ns_per_call":50708.0,"kernel":"'
    'fma_chains_8","launch":{"active_threads":65536,"extent":[65536],"group_siz'
    'e":[256],"groups":[256],"idle_lanes_per_threadgroup":0,"launched_threads":'
    '65536,"padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[65536'
    '],"threads_per_group":256},"median_gflops_per_second":441.0211412461309,"m'
    'edian_ns_per_call":152167.0,"output_sha256":"1a0225dd1df9ccf384e3d07c61ffe'
    '2f26adf505460db692166c9d1b63f3e7857","repeats":3,"samples_ns":[198541,1521'
    '67,147833],"threads":65536,"timing":{"count":3,"iqr_ns":50708,"max_ns":198'
    '541,"median_ns":152167,"min_ns":147833,"p25_ns":147833,"p75_ns":198541},"w'
    'armup":1},{"best_gflops_per_second":1360.3171086437646,"calls_per_sample":'
    '1,"chains":8,"flops_per_call":268435456,"fma_steps":256,"iqr_ns_per_call":'
    '180917.0,"kernel":"fma_chains_8","launch":{"active_threads":65536,"extent"'
    ':[65536],"group_size":[256],"groups":[256],"idle_lanes_per_threadgroup":0,'
    '"launched_threads":65536,"padding_threads":0,"simd_groups_per_threadgroup"'
    ':8,"threads":[65536],"threads_per_group":256},"median_gflops_per_second":8'
    '63.6002483648776,"median_ns_per_call":310833.0,"output_sha256":"536c23870b'
    '1ea9eaf69284efc4279b09bfa146d595e8219f629e8cc3ad729166","repeats":3,"sampl'
    'es_ns":[378250,310833,197333],"threads":65536,"timing":{"count":3,"iqr_ns"'
    ':180917,"max_ns":378250,"median_ns":310833,"min_ns":197333,"p25_ns":197333'
    ',"p75_ns":378250},"warmup":1},{"best_gflops_per_second":2624.214132615778,'
    '"calls_per_sample":1,"chains":8,"flops_per_call":1073741824,"fma_steps":10'
    '24,"iqr_ns_per_call":691416.0,"kernel":"fma_chains_8","launch":{"active_th'
    'reads":65536,"extent":[65536],"group_size":[256],"groups":[256],"idle_lane'
    's_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"simd_gr'
    'oups_per_threadgroup":8,"threads":[65536],"threads_per_group":256},"median'
    '_gflops_per_second":1802.7144998950682,"median_ns_per_call":595625.0,"outp'
    'ut_sha256":"a28514957e4ea9e9b07d070be9abee68473bedaf9898e77677eff86b157025'
    '27","repeats":3,"samples_ns":[1100583,595625,409167],"threads":65536,"timi'
    'ng":{"count":3,"iqr_ns":691416,"max_ns":1100583,"median_ns":595625,"min_ns'
    '":409167,"p25_ns":409167,"p75_ns":1100583},"warmup":1}],"stream":[{"best_g'
    'b_per_second":75.88616091621284,"buffer_bytes":4194304,"bytes_per_call":83'
    '88608,"bytes_per_sample":16777216,"calls_per_sample":2,"float4_count":2621'
    '44,"float4_per_thread":4,"iqr_ns_per_call":4666.5,"kernel":"stream_copy","'
    'launch":{"active_threads":65536,"extent":[65536],"group_size":[256],"group'
    's":[256],"idle_lanes_per_threadgroup":0,"launched_threads":65536,"padding_'
    'threads":0,"simd_groups_per_threadgroup":8,"threads":[65536],"threads_per_'
    'group":256},"median_gb_per_second":74.38621631447802,"median_ns_per_call":'
    '112771.0,"output_sha256":"38943157abc3f7398aa83075a0dc02f538244edf14b60392'
    'af621411de32e59a","repeats":3,"samples_ns":[221084,225542,230417],"threads'
    '":65536,"timing":{"count":3,"iqr_ns":9333,"max_ns":230417,"median_ns":2255'
    '42,"min_ns":221084,"p25_ns":221084,"p75_ns":230417},"warmup":1},{"best_gb_'
    'per_second":83.80026852503038,"buffer_bytes":4194304,"bytes_per_call":4259'
    '840,"bytes_per_sample":17039360,"calls_per_sample":4,"float4_count":262144'
    ',"float4_per_thread":16,"iqr_ns_per_call":139979.25,"kernel":"stream_read_'
    'reduce","launch":{"active_threads":16384,"extent":[16384],"group_size":[25'
    '6],"groups":[64],"idle_lanes_per_threadgroup":0,"launched_threads":16384,"'
    'padding_threads":0,"simd_groups_per_threadgroup":8,"threads":[16384],"thre'
    'ads_per_group":256},"median_gb_per_second":82.28275618954719,"median_ns_pe'
    'r_call":51770.75,"output_sha256":"c5f4488fcacc2965c3904fe668eb1289e19e06b3'
    '517c3c7a58c58b68374d58ef","repeats":3,"samples_ns":[207083,763250,203333],'
    '"threads":16384,"timing":{"count":3,"iqr_ns":559917,"max_ns":763250,"media'
    'n_ns":207083,"min_ns":203333,"p25_ns":203333,"p75_ns":763250},"warmup":1}]'
    ',"synchronization":"torch.mps.synchronize() before the first and before th'
    "e second clock reading of every sample; untimed warmups first; samples rot"
    'ate over the points of a sweep and the order reverses every repeat"},"ceil'
    'ings":{"bandwidth":{"best_sample_gb_per_second":83.80026852503038,"buffer_'
    'bytes":4194304,"copy_gb_per_second":74.38621631447802,"gb_per_second":82.2'
    '8275618954719,"kernel":"stream_read_reduce","read_reduce_gb_per_second":82'
    '.28275618954719},"bank":{"finding":"stride_slowdown_removed_by_padding","l'
    'argest_padded_ratio":0.7279757299962079,"largest_unpadded_ratio":1.8225255'
    '972696246,"model":"CUDA shared memory as the Day 74 memory-access lab mode'
    "ls it: 32 banks of 4-byte words, conflict degree = most distinct words one"
    ' bank serves","padded_within_limit":true,"padding_speedup_at_largest_strid'
    'e":2.5035526902512513,"rows":[{"model_padded_conflict_degree":1,"model_unp'
    'added_conflict_degree":1,"padded_ps_per_access":10.378658771514893,"padded'
    '_relative_to_own_stride_1":1.0,"padded_relative_to_unpadded_stride_1":0.52'
    '82518012893439,"padding_speedup":1.8930366116295765,"stride":1,"unpadded_p'
    's_per_access":19.647181034088135,"unpadded_relative_to_stride_1":1.0},{"mo'
    'del_padded_conflict_degree":1,"model_unpadded_conflict_degree":32,"padded_'
    'ps_per_access":14.302670955657959,"padded_relative_to_own_stride_1":1.3780'
    '847092605886,"padded_relative_to_unpadded_stride_1":0.7279757299962079,"pa'
    'dding_speedup":2.5035526902512513,"stride":32,"unpadded_ps_per_access":35.'
    '80749034881592,"unpadded_relative_to_stride_1":1.8225255972696246}],"slowd'
    'own_limit":1.25,"unpadded_rises_at_every_stride":true},"definition":"Each '
    "ceiling is the best median rate that these kernels reached on this host in"
    ' this run. It is not a vendor peak and not a hardware limit.","fma":{"best'
    '_sample_gflops_per_second":2624.214132615778,"chains":8,"fixed_share_limit'
    '":0.05,"fixed_share_of_ceiling_call":0.26457978314095837,"flops_per_call":'
    '1073741824,"fma_steps":1024,"gflops_per_second":1802.7144998950682,"kernel'
    '":"fma_chains_8","scales_with_work":false},"fma_scaling":[{"chains":1,"fix'
    'ed_ns_per_call":138055.33333333334,"fixed_share_at_largest":0.790957668259'
    '4065,"marginal_gflops_per_second":3508.5496225776255,"points":[{"flops_per'
    '_call":8388608,"fma_steps":64,"median_ns_per_call":133375.0,"time_ratio_to'
    '_previous":null,"work_ratio_to_previous":null},{"flops_per_call":33554432,'
    '"fma_steps":256,"median_ns_per_call":156458.0,"time_ratio_to_previous":1.1'
    '730684161199625,"work_ratio_to_previous":4.0},{"flops_per_call":134217728,'
    '"fma_steps":1024,"median_ns_per_call":174542.0,"time_ratio_to_previous":1.'
    '1155837349320585,"work_ratio_to_previous":4.0}]},{"chains":8,"fixed_ns_per'
    '_call":157590.33333333334,"fixed_share_at_largest":0.26457978314095837,"ma'
    'rginal_gflops_per_second":2405.5244890365175,"points":[{"flops_per_call":6'
    '7108864,"fma_steps":64,"median_ns_per_call":152167.0,"time_ratio_to_previo'
    'us":null,"work_ratio_to_previous":null},{"flops_per_call":268435456,"fma_s'
    'teps":256,"median_ns_per_call":310833.0,"time_ratio_to_previous":2.0427096'
    '545243058,"work_ratio_to_previous":4.0},{"flops_per_call":1073741824,"fma_'
    'steps":1024,"median_ns_per_call":595625.0,"time_ratio_to_previous":1.91622'
    '18940717364,"work_ratio_to_previous":4.0}]}],"ridge_point_flops_per_byte":'
    '21.908776314472515},"config":{"bandwidth":{"buffer_bytes":[4194304],"copy_'
    'float4_per_thread":4,"group_size":256,"reduce_float4_per_thread":16,"targe'
    't_bytes_per_sample":16777216},"bank":{"group_size":256,"strides":[1,32],"t'
    'hreads":65536,"trips":16},"correctness":{"bank_trips":9,"fma_steps":64,"st'
    'ream_float4":4099,"threads":1000},"fma":{"chain_counts":[1,8],"fma_steps":'
    '[64,256,1024],"group_size":256,"threads":65536},"name":"gpu-ceiling-record'
    'ed","schema_version":1,"seed":74,"timing":{"max_runtime_seconds":60,"repea'
    'ts":3,"warmup":1}},"config_sha256":"9e1db2639799674c80116c53f0ab2a97583f54'
    '00def85d536483da68aff6d173","correctness":{"bank":[{"buffer_sha256":"67267'
    '65e5a22484abc78e6bfd3d7835d240d057c849750fe8b2662ce52e46e43","expected_sha'
    '256":"bd357bdcfc8b9163d88ef597ac74d1bc53bf044b27f96cb080af4599eda852f5","g'
    'uard_words":24,"kernel":"bank_rows_32","launch":{"active_threads":1000,"ex'
    'tent":[1000],"group_size":[256],"groups":[4],"idle_lanes_per_threadgroup":'
    '0,"launched_threads":1024,"padding_threads":24,"simd_groups_per_threadgrou'
    'p":8,"threads":[1024],"threads_per_group":256},"row_words":32,"stride":1,"'
    'threads":1000,"trips":9},{"buffer_sha256":"43deefdfa346bf05d4f879ee0fb1fb5'
    '766f5e2cff29e67398040d2b1005dc32b","expected_sha256":"83ef1567ffd471c89a35'
    'dd6f057da833573eb630bd6c8409f97c8796d392ca2d","guard_words":24,"kernel":"b'
    'ank_rows_32","launch":{"active_threads":1000,"extent":[1000],"group_size":'
    '[256],"groups":[4],"idle_lanes_per_threadgroup":0,"launched_threads":1024,'
    '"padding_threads":24,"simd_groups_per_threadgroup":8,"threads":[1024],"thr'
    'eads_per_group":256},"row_words":32,"stride":32,"threads":1000,"trips":9},'
    '{"buffer_sha256":"cb9b6d57877dbeacc6f19b2a65e1b757b62c5714c33604db05a88db8'
    '302ae0d6","expected_sha256":"5aaa8afa393453c5ead700712916cbe9d23976fe0957c'
    '5242fcbecef6aaa2d25","guard_words":24,"kernel":"bank_rows_33","launch":{"a'
    'ctive_threads":1000,"extent":[1000],"group_size":[256],"groups":[4],"idle_'
    'lanes_per_threadgroup":0,"launched_threads":1024,"padding_threads":24,"sim'
    'd_groups_per_threadgroup":8,"threads":[1024],"threads_per_group":256},"row'
    '_words":33,"stride":1,"threads":1000,"trips":9},{"buffer_sha256":"db07ba0c'
    'a0c9beacc3b2b4b96b598265294cab23ae10f32365e1f2e3e05040d2","expected_sha256'
    '":"ddbe1aaf2e5bd9287e1e07d6e5e6d20e50c7ffdace44ca3d9eb202193fcf78d5","guar'
    'd_words":24,"kernel":"bank_rows_33","launch":{"active_threads":1000,"exten'
    't":[1000],"group_size":[256],"groups":[4],"idle_lanes_per_threadgroup":0,"'
    'launched_threads":1024,"padding_threads":24,"simd_groups_per_threadgroup":'
    '8,"threads":[1024],"threads_per_group":256},"row_words":33,"stride":32,"th'
    'reads":1000,"trips":9}],"fma":[{"buffer_sha256":"0ccd7c2bb147e0b6e960229ff'
    'c96b08eb983f0941a1129f29d75baa7f5b7a470","chains":1,"expected_sha256":"088'
    'bee2dcfc05026c6537124718132067e9250effd9eaabe8f9b44b3a48b0341","fma_steps"'
    ':64,"guard_words":24,"kernel":"fma_chains_1","launch":{"active_threads":10'
    '00,"extent":[1000],"group_size":[256],"groups":[4],"idle_lanes_per_threadg'
    'roup":0,"launched_threads":1024,"padding_threads":24,"simd_groups_per_thre'
    'adgroup":8,"threads":[1024],"threads_per_group":256},"start_sha256":"c035b'
    '1a65e09c776d902e1dcdc64fd729b69463ebccae1193ee08ecb4141648d","threads":100'
    '0},{"buffer_sha256":"e9567707877415f56ce1f474f5371223e0a510de398fc64b433ba'
    '9f155e21c6f","chains":8,"expected_sha256":"38f69523c7da4eb7279513899e3d743'
    '182956364aa35f41bee82fa963fba9e76","fma_steps":64,"guard_words":24,"kernel'
    '":"fma_chains_8","launch":{"active_threads":1000,"extent":[1000],"group_si'
    'ze":[256],"groups":[4],"idle_lanes_per_threadgroup":0,"launched_threads":1'
    '024,"padding_threads":24,"simd_groups_per_threadgroup":8,"threads":[1024],'
    '"threads_per_group":256},"start_sha256":"c035b1a65e09c776d902e1dcdc64fd729'
    'b69463ebccae1193ee08ecb4141648d","threads":1000}],"fma_coefficients":[0.99'
    '97512698173523,0.0003656000772025436],"fma_rule":"x = fma(x, a, b) with a '
    "= (2^24 - 4173) * 2^-24 and b = 12561923 * 2^-35, from starts m * 2^-23 in"
    " [1, 2); float64 holds each exact x * a + b, so one rounding to float32 eq"
    'uals the fused result, and x stays in [1, 2)","reduce_max_value":1023,"ref'
    'erence":"NumPy on the CPU, rebuilt from the seed: exact fused FMA chains, '
    'copies, exact integer partials and wrapping lane sums","sentinel_bits":214'
    '3289344,"stream":[{"buffer_sha256":"81e578e5ced2f60d221543d5a57b2ad2598eba'
    '9e57c49b1e5f67697ed0d235a9","expected_sha256":"30c36d2bb8c394618f284eaffd6'
    '73f00e4bdd65f12ae3a930302e128e135e71b","float4_count":4099,"float4_per_thr'
    'ead":4,"guard_words":128,"kernel":"stream_copy","launch":{"active_threads"'
    ':1025,"extent":[1025],"group_size":[256],"groups":[5],"idle_lanes_per_thre'
    'adgroup":0,"launched_threads":1280,"padding_threads":255,"simd_groups_per_'
    'threadgroup":8,"threads":[1280],"threads_per_group":256},"source_sha256":"'
    '30c36d2bb8c394618f284eaffd673f00e4bdd65f12ae3a930302e128e135e71b","threads'
    '":1025},{"buffer_sha256":"29c60b86e06e842a64d6a024af586d3ee723e2cf52459349'
    '4e5e99f957b7f029","expected_sha256":"c6789439f727c5dd769238ea6b4152a191244'
    'b1d5ba2b645f07ac25936e42dbe","float4_count":4099,"float4_per_thread":16,"g'
    'uard_words":255,"kernel":"stream_read_reduce","launch":{"active_threads":2'
    '57,"extent":[257],"group_size":[256],"groups":[2],"idle_lanes_per_threadgr'
    'oup":0,"launched_threads":512,"padding_threads":255,"simd_groups_per_threa'
    'dgroup":8,"threads":[512],"threads_per_group":256},"source_sha256":"f54e57'
    'd367f118f30fae2263cbdc0f65a885a4f672288c38bcc65ae44dc2a7ae","threads":257}'
    '],"stream_guard_float4":32},"environment":{"macos_version":"27.2","numpy_v'
    'ersion":"1.26.4","platform_machine":"arm64","platform_system":"Darwin","py'
    'thon_implementation":"CPython","python_version":"3.11.14","torch_version":'
    '"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_host":true,"cuda_kern'
    'el_executed":false,"distributed_claim":false,"host_specific":true,"nvidia_'
    'gpu_used":false,"portable_performance_claim":false,"production_claim":fals'
    'e,"statement":"These are wall-clock timings on the Apple GPU of the host t'
    "hat made this report. The kernels came from torch.mps.compile_shader. Each"
    " ceiling is the best median rate that these kernels reached in this run. I"
    "t is not a vendor peak and not a hardware limit. The timings are host-spec"
    "ific Apple GPU (Metal) measurements. No CUDA kernel ran, and no NVIDIA GPU"
    " was used. Bandwidth counts the bytes that each kernel must read and write"
    ", not the hardware traffic. These timings do not measure or predict CUDA, "
    'NVIDIA, distributed or production performance.","vendor_peak_used":false},'
    '"host_load":{"after":[9.4765625,9.40478515625,8.75341796875],"before":[9.4'
    '765625,9.40478515625,8.75341796875],"source":"os.getloadavg(): the 1, 5 an'
    "d 15 minute load averages that uptime prints, read before the checks and a"
    'fter the last sample"},"kind":"gpu_ceiling_measurement","label":"host-spec'
    'ific Apple GPU (Metal) measurement","measured":true,"measurement_scope":"a'
    'pple_gpu_metal_this_host","percentile_method":"nearest rank: sorted sample'
    "s[ceil(p * n / 100) - 1] for p in 25, 50 and 75; median is p50 and IQR is "
    'p75 - p25; no interpolation","planned_work":{"bank_accesses":268435456,"ba'
    'nk_points":4,"calls":64,"correctness_cases":8,"flops":6341787648,"fma_poin'
    'ts":6,"stream_bytes":135266304,"stream_points":2,"timed_samples":36},"repo'
    'rt_sha256":"6b784c28e36b2890eadb65c742858b29835b90f8e259b4befba92a499c919c'
    'b6","run":{"elapsed_ns":64894041,"max_runtime_ns":60000000000,"timed_ns":1'
    '2627833},"schema_version":1,"shaders":{"kernels":[{"buffers":["out","start'
    '","coefficients","n","trips"],"family":"fma","limits":{"max_threads_per_th'
    'readgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_widt'
    'h":32},"name":"fma_chains_1","scalar_buffers":[3,4]},{"buffers":["out","st'
    'art","coefficients","n","trips"],"family":"fma","limits":{"max_threads_per'
    '_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_w'
    'idth":32},"name":"fma_chains_2","scalar_buffers":[3,4]},{"buffers":["out",'
    '"start","coefficients","n","trips"],"family":"fma","limits":{"max_threads_'
    'per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_executio'
    'n_width":32},"name":"fma_chains_4","scalar_buffers":[3,4]},{"buffers":["ou'
    't","start","coefficients","n","trips"],"family":"fma","limits":{"max_threa'
    'ds_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execu'
    'tion_width":32},"name":"fma_chains_8","scalar_buffers":[3,4]},{"buffers":['
    '"out","start","coefficients","n","trips"],"family":"fma","limits":{"max_th'
    'reads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_ex'
    'ecution_width":32},"name":"fma_chains_16","scalar_buffers":[3,4]},{"buffer'
    's":["out","start","coefficients","n","trips"],"family":"fma","limits":{"ma'
    'x_threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"threa'
    'd_execution_width":32},"name":"fma_chains_32","scalar_buffers":[3,4]},{"bu'
    'ffers":["out","in","n","threads"],"family":"bandwidth","limits":{"max_thre'
    'ads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_exec'
    'ution_width":32},"name":"stream_copy","scalar_buffers":[2,3]},{"buffers":['
    '"out","in","n","threads"],"family":"bandwidth","limits":{"max_threads_per_'
    'threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_wi'
    'dth":32},"name":"stream_read_reduce","scalar_buffers":[2,3]},{"buffers":["'
    'out","n","stride","trips"],"family":"bank","limits":{"max_threads_per_thre'
    'adgroup":1024,"static_threadgroup_memory_bytes":14336,"thread_execution_wi'
    'dth":32},"name":"bank_rows_32","scalar_buffers":[1,2,3]},{"buffers":["out"'
    ',"n","stride","trips"],"family":"bank","limits":{"max_threads_per_threadgr'
    'oup":1024,"static_threadgroup_memory_bytes":14784,"thread_execution_width"'
    ':32},"name":"bank_rows_33","scalar_buffers":[1,2,3]}],"source":"#include <'
    "metal_stdlib>\\nusing namespace metal;\\n\\n// One fused multiply-add on chai"
    "n x. Eight steps of every chain per loop trip\\n// keep the loop branch a s"
    "mall share of the work.\\n#define FMA_STEP(x) x = fma(x, a, b);\\n#define EI"
    "GHT_TIMES(steps) steps steps steps steps steps steps steps steps\\n\\n// 1 i"
    "ndependent chain in registers; the loop touches no memory.\\nkernel void fm"
    "a_chains_1(device uint* out [[buffer(0)]],\\n                         devic"
    "e const float* start [[buffer(1)]],\\n                         device const"
    " float* coefficients [[buffer(2)]],\\n                         constant uin"
    "t& n [[buffer(3)]],\\n                         constant uint& trips [[buffe"
    "r(4)]],\\n                         uint i [[thread_position_in_grid]]) {\\n "
    "   if (i < n) {\\n        const uint row = (i % 32u) * 32u;\\n        const "
    "float a = coefficients[0];\\n        const float b = coefficients[1];\\n    "
    "    float x0 = start[row + 0u];\\n        for (uint t = 0u; t < trips; ++t)"
    " {\\n            EIGHT_TIMES(\\n                FMA_STEP(x0)\\n            )\\"
    "n        }\\n        uint bits = as_type<uint>(x0);\\n        out[i] = bits;"
    "\\n    }\\n}\\n\\n// 2 independent chains in registers; the loop touches no me"
    "mory.\\nkernel void fma_chains_2(device uint* out [[buffer(0)]],\\n         "
    "                device const float* start [[buffer(1)]],\\n                "
    "         device const float* coefficients [[buffer(2)]],\\n                "
    "         constant uint& n [[buffer(3)]],\\n                         constan"
    "t uint& trips [[buffer(4)]],\\n                         uint i [[thread_pos"
    "ition_in_grid]]) {\\n    if (i < n) {\\n        const uint row = (i % 32u) *"
    " 32u;\\n        const float a = coefficients[0];\\n        const float b = c"
    "oefficients[1];\\n        float x0 = start[row + 0u];\\n        float x1 = s"
    "tart[row + 1u];\\n        for (uint t = 0u; t < trips; ++t) {\\n            "
    "EIGHT_TIMES(\\n                FMA_STEP(x0) FMA_STEP(x1)\\n            )\\n  "
    "      }\\n        uint bits = as_type<uint>(x0);\\n        bits ^= as_type<u"
    "int>(x1);\\n        out[i] = bits;\\n    }\\n}\\n\\n// 4 independent chains in "
    "registers; the loop touches no memory.\\nkernel void fma_chains_4(device ui"
    "nt* out [[buffer(0)]],\\n                         device const float* start"
    " [[buffer(1)]],\\n                         device const float* coefficients"
    " [[buffer(2)]],\\n                         constant uint& n [[buffer(3)]],\\"
    "n                         constant uint& trips [[buffer(4)]],\\n           "
    "              uint i [[thread_position_in_grid]]) {\\n    if (i < n) {\\n   "
    "     const uint row = (i % 32u) * 32u;\\n        const float a = coefficien"
    "ts[0];\\n        const float b = coefficients[1];\\n        float x0 = start"
    "[row + 0u];\\n        float x1 = start[row + 1u];\\n        float x2 = start"
    "[row + 2u];\\n        float x3 = start[row + 3u];\\n        for (uint t = 0u"
    "; t < trips; ++t) {\\n            EIGHT_TIMES(\\n                FMA_STEP(x0"
    ") FMA_STEP(x1) FMA_STEP(x2) FMA_STEP(x3)\\n            )\\n        }\\n      "
    "  uint bits = as_type<uint>(x0);\\n        bits ^= as_type<uint>(x1);\\n    "
    "    bits ^= as_type<uint>(x2);\\n        bits ^= as_type<uint>(x3);\\n      "
    "  out[i] = bits;\\n    }\\n}\\n\\n// 8 independent chains in registers; the lo"
    "op touches no memory.\\nkernel void fma_chains_8(device uint* out [[buffer("
    "0)]],\\n                         device const float* start [[buffer(1)]],\\n"
    "                         device const float* coefficients [[buffer(2)]],\\n"
    "                         constant uint& n [[buffer(3)]],\\n                "
    "         constant uint& trips [[buffer(4)]],\\n                         uin"
    "t i [[thread_position_in_grid]]) {\\n    if (i < n) {\\n        const uint r"
    "ow = (i % 32u) * 32u;\\n        const float a = coefficients[0];\\n        c"
    "onst float b = coefficients[1];\\n        float x0 = start[row + 0u];\\n    "
    "    float x1 = start[row + 1u];\\n        float x2 = start[row + 2u];\\n    "
    "    float x3 = start[row + 3u];\\n        float x4 = start[row + 4u];\\n    "
    "    float x5 = start[row + 5u];\\n        float x6 = start[row + 6u];\\n    "
    "    float x7 = start[row + 7u];\\n        for (uint t = 0u; t < trips; ++t)"
    " {\\n            EIGHT_TIMES(\\n                FMA_STEP(x0) FMA_STEP(x1) FM"
    "A_STEP(x2) FMA_STEP(x3)\\n                FMA_STEP(x4) FMA_STEP(x5) FMA_STE"
    "P(x6) FMA_STEP(x7)\\n            )\\n        }\\n        uint bits = as_type<"
    "uint>(x0);\\n        bits ^= as_type<uint>(x1);\\n        bits ^= as_type<ui"
    "nt>(x2);\\n        bits ^= as_type<uint>(x3);\\n        bits ^= as_type<uint"
    ">(x4);\\n        bits ^= as_type<uint>(x5);\\n        bits ^= as_type<uint>("
    "x6);\\n        bits ^= as_type<uint>(x7);\\n        out[i] = bits;\\n    }\\n}"
    "\\n\\n// 16 independent chains in registers; the loop touches no memory.\\nke"
    "rnel void fma_chains_16(device uint* out [[buffer(0)]],\\n                 "
    "         device const float* start [[buffer(1)]],\\n                       "
    "   device const float* coefficients [[buffer(2)]],\\n                      "
    "    constant uint& n [[buffer(3)]],\\n                          constant ui"
    "nt& trips [[buffer(4)]],\\n                          uint i [[thread_positi"
    "on_in_grid]]) {\\n    if (i < n) {\\n        const uint row = (i % 32u) * 32"
    "u;\\n        const float a = coefficients[0];\\n        const float b = coef"
    "ficients[1];\\n        float x0 = start[row + 0u];\\n        float x1 = star"
    "t[row + 1u];\\n        float x2 = start[row + 2u];\\n        float x3 = star"
    "t[row + 3u];\\n        float x4 = start[row + 4u];\\n        float x5 = star"
    "t[row + 5u];\\n        float x6 = start[row + 6u];\\n        float x7 = star"
    "t[row + 7u];\\n        float x8 = start[row + 8u];\\n        float x9 = star"
    "t[row + 9u];\\n        float x10 = start[row + 10u];\\n        float x11 = s"
    "tart[row + 11u];\\n        float x12 = start[row + 12u];\\n        float x13"
    " = start[row + 13u];\\n        float x14 = start[row + 14u];\\n        float"
    " x15 = start[row + 15u];\\n        for (uint t = 0u; t < trips; ++t) {\\n   "
    "         EIGHT_TIMES(\\n                FMA_STEP(x0) FMA_STEP(x1) FMA_STEP("
    "x2) FMA_STEP(x3)\\n                FMA_STEP(x4) FMA_STEP(x5) FMA_STEP(x6) F"
    "MA_STEP(x7)\\n                FMA_STEP(x8) FMA_STEP(x9) FMA_STEP(x10) FMA_S"
    "TEP(x11)\\n                FMA_STEP(x12) FMA_STEP(x13) FMA_STEP(x14) FMA_ST"
    "EP(x15)\\n            )\\n        }\\n        uint bits = as_type<uint>(x0);\\"
    "n        bits ^= as_type<uint>(x1);\\n        bits ^= as_type<uint>(x2);\\n "
    "       bits ^= as_type<uint>(x3);\\n        bits ^= as_type<uint>(x4);\\n   "
    "     bits ^= as_type<uint>(x5);\\n        bits ^= as_type<uint>(x6);\\n     "
    "   bits ^= as_type<uint>(x7);\\n        bits ^= as_type<uint>(x8);\\n       "
    " bits ^= as_type<uint>(x9);\\n        bits ^= as_type<uint>(x10);\\n        "
    "bits ^= as_type<uint>(x11);\\n        bits ^= as_type<uint>(x12);\\n        "
    "bits ^= as_type<uint>(x13);\\n        bits ^= as_type<uint>(x14);\\n        "
    "bits ^= as_type<uint>(x15);\\n        out[i] = bits;\\n    }\\n}\\n\\n// 32 ind"
    "ependent chains in registers; the loop touches no memory.\\nkernel void fma"
    "_chains_32(device uint* out [[buffer(0)]],\\n                          devi"
    "ce const float* start [[buffer(1)]],\\n                          device con"
    "st float* coefficients [[buffer(2)]],\\n                          constant "
    "uint& n [[buffer(3)]],\\n                          constant uint& trips [[b"
    "uffer(4)]],\\n                          uint i [[thread_position_in_grid]])"
    " {\\n    if (i < n) {\\n        const uint row = (i % 32u) * 32u;\\n        c"
    "onst float a = coefficients[0];\\n        const float b = coefficients[1];\\"
    "n        float x0 = start[row + 0u];\\n        float x1 = start[row + 1u];\\"
    "n        float x2 = start[row + 2u];\\n        float x3 = start[row + 3u];\\"
    "n        float x4 = start[row + 4u];\\n        float x5 = start[row + 5u];\\"
    "n        float x6 = start[row + 6u];\\n        float x7 = start[row + 7u];\\"
    "n        float x8 = start[row + 8u];\\n        float x9 = start[row + 9u];\\"
    "n        float x10 = start[row + 10u];\\n        float x11 = start[row + 11"
    "u];\\n        float x12 = start[row + 12u];\\n        float x13 = start[row "
    "+ 13u];\\n        float x14 = start[row + 14u];\\n        float x15 = start["
    "row + 15u];\\n        float x16 = start[row + 16u];\\n        float x17 = st"
    "art[row + 17u];\\n        float x18 = start[row + 18u];\\n        float x19 "
    "= start[row + 19u];\\n        float x20 = start[row + 20u];\\n        float "
    "x21 = start[row + 21u];\\n        float x22 = start[row + 22u];\\n        fl"
    "oat x23 = start[row + 23u];\\n        float x24 = start[row + 24u];\\n      "
    "  float x25 = start[row + 25u];\\n        float x26 = start[row + 26u];\\n  "
    "      float x27 = start[row + 27u];\\n        float x28 = start[row + 28u];"
    "\\n        float x29 = start[row + 29u];\\n        float x30 = start[row + 3"
    "0u];\\n        float x31 = start[row + 31u];\\n        for (uint t = 0u; t <"
    " trips; ++t) {\\n            EIGHT_TIMES(\\n                FMA_STEP(x0) FMA"
    "_STEP(x1) FMA_STEP(x2) FMA_STEP(x3)\\n                FMA_STEP(x4) FMA_STEP"
    "(x5) FMA_STEP(x6) FMA_STEP(x7)\\n                FMA_STEP(x8) FMA_STEP(x9) "
    "FMA_STEP(x10) FMA_STEP(x11)\\n                FMA_STEP(x12) FMA_STEP(x13) F"
    "MA_STEP(x14) FMA_STEP(x15)\\n                FMA_STEP(x16) FMA_STEP(x17) FM"
    "A_STEP(x18) FMA_STEP(x19)\\n                FMA_STEP(x20) FMA_STEP(x21) FMA"
    "_STEP(x22) FMA_STEP(x23)\\n                FMA_STEP(x24) FMA_STEP(x25) FMA_"
    "STEP(x26) FMA_STEP(x27)\\n                FMA_STEP(x28) FMA_STEP(x29) FMA_S"
    "TEP(x30) FMA_STEP(x31)\\n            )\\n        }\\n        uint bits = as_t"
    "ype<uint>(x0);\\n        bits ^= as_type<uint>(x1);\\n        bits ^= as_typ"
    "e<uint>(x2);\\n        bits ^= as_type<uint>(x3);\\n        bits ^= as_type<"
    "uint>(x4);\\n        bits ^= as_type<uint>(x5);\\n        bits ^= as_type<ui"
    "nt>(x6);\\n        bits ^= as_type<uint>(x7);\\n        bits ^= as_type<uint"
    ">(x8);\\n        bits ^= as_type<uint>(x9);\\n        bits ^= as_type<uint>("
    "x10);\\n        bits ^= as_type<uint>(x11);\\n        bits ^= as_type<uint>("
    "x12);\\n        bits ^= as_type<uint>(x13);\\n        bits ^= as_type<uint>("
    "x14);\\n        bits ^= as_type<uint>(x15);\\n        bits ^= as_type<uint>("
    "x16);\\n        bits ^= as_type<uint>(x17);\\n        bits ^= as_type<uint>("
    "x18);\\n        bits ^= as_type<uint>(x19);\\n        bits ^= as_type<uint>("
    "x20);\\n        bits ^= as_type<uint>(x21);\\n        bits ^= as_type<uint>("
    "x22);\\n        bits ^= as_type<uint>(x23);\\n        bits ^= as_type<uint>("
    "x24);\\n        bits ^= as_type<uint>(x25);\\n        bits ^= as_type<uint>("
    "x26);\\n        bits ^= as_type<uint>(x27);\\n        bits ^= as_type<uint>("
    "x28);\\n        bits ^= as_type<uint>(x29);\\n        bits ^= as_type<uint>("
    "x30);\\n        bits ^= as_type<uint>(x31);\\n        out[i] = bits;\\n    }\\"
    "n}\\n\\n// Each thread copies every threads-th float4 from its own index, so"
    " one SIMD\\n// group reads and writes 32 consecutive float4 values per step"
    ".\\nkernel void stream_copy(device float4* out [[buffer(0)]],\\n            "
    "            device const float4* in [[buffer(1)]],\\n                      "
    "  constant uint& n [[buffer(2)]],\\n                        constant uint& "
    "threads [[buffer(3)]],\\n                        uint i [[thread_position_i"
    "n_grid]]) {\\n    if (i < threads) {\\n        for (uint e = i; e < n; e += "
    "threads) {\\n            out[e] = in[e];\\n        }\\n    }\\n}\\n\\n// The sam"
    "e walk, summed into one partial per thread. The inputs are whole\\n// numbe"
    "rs below 1024, so every float32 sum is exact in any order.\\nkernel void st"
    "ream_read_reduce(device float* out [[buffer(0)]],\\n                       "
    "        device const float4* in [[buffer(1)]],\\n                          "
    "     constant uint& n [[buffer(2)]],\\n                               const"
    "ant uint& threads [[buffer(3)]],\\n                               uint i [["
    "thread_position_in_grid]]) {\\n    if (i < threads) {\\n        float4 sum ="
    " float4(0.0f);\\n        for (uint e = i; e < n; e += threads) {\\n         "
    "   sum += in[e];\\n        }\\n        out[i] = (sum.x + sum.y) + (sum.z + s"
    "um.w);\\n    }\\n}\\n\\n// Lane l reads logical word l * stride of a 32-column"
    " tile\\n// with 32-word rows. Trip k moves every lane down k % 64 rows\\n// "
    "and reads 16 rows from there, so every load has the same bank pattern.\\nke"
    "rnel void bank_rows_32(device uint* out [[buffer(0)]],\\n                  "
    "       constant uint& n [[buffer(1)]],\\n                         constant "
    "uint& stride [[buffer(2)]],\\n                         constant uint& trips"
    " [[buffer(3)]],\\n                         uint i [[thread_position_in_grid"
    "]],\\n                         uint t [[thread_index_in_threadgroup]],\\n   "
    "                      uint group [[threads_per_threadgroup]]) {\\n    threa"
    "dgroup uint tile[3584];\\n    for (uint w = t; w < 3584u; w += group) {\\n  "
    "      tile[w] = w * 2654435761u;\\n    }\\n    threadgroup_barrier(mem_flags"
    "::mem_threadgroup);\\n    const uint logical = (t % 32u) * stride;\\n    con"
    "st uint first = (logical / 32u) * 32u + logical % 32u;\\n    uint s0 = 0u, "
    "s1 = 0u, s2 = 0u, s3 = 0u;\\n    for (uint k = 0u; k < trips; ++k) {\\n     "
    "   const uint q = first + (k % 64u) * 32u;\\n        s0 += tile[q + 0u];\\n "
    "       s1 += tile[q + 32u];\\n        s2 += tile[q + 64u];\\n        s3 += t"
    "ile[q + 96u];\\n        s0 += tile[q + 128u];\\n        s1 += tile[q + 160u]"
    ";\\n        s2 += tile[q + 192u];\\n        s3 += tile[q + 224u];\\n        s"
    "0 += tile[q + 256u];\\n        s1 += tile[q + 288u];\\n        s2 += tile[q "
    "+ 320u];\\n        s3 += tile[q + 352u];\\n        s0 += tile[q + 384u];\\n  "
    "      s1 += tile[q + 416u];\\n        s2 += tile[q + 448u];\\n        s3 += "
    "tile[q + 480u];\\n    }\\n    if (i < n) {\\n        out[i] = (s0 + s1) + (s2"
    " + s3);\\n    }\\n}\\n\\n// Lane l reads logical word l * stride of a 32-colum"
    "n tile\\n// with 33-word rows. Trip k moves every lane down k % 64 rows\\n//"
    " and reads 16 rows from there, so every load has the same bank pattern.\\nk"
    "ernel void bank_rows_33(device uint* out [[buffer(0)]],\\n                 "
    "        constant uint& n [[buffer(1)]],\\n                         constant"
    " uint& stride [[buffer(2)]],\\n                         constant uint& trip"
    "s [[buffer(3)]],\\n                         uint i [[thread_position_in_gri"
    "d]],\\n                         uint t [[thread_index_in_threadgroup]],\\n  "
    "                       uint group [[threads_per_threadgroup]]) {\\n    thre"
    "adgroup uint tile[3696];\\n    for (uint w = t; w < 3696u; w += group) {\\n "
    "       tile[w] = w * 2654435761u;\\n    }\\n    threadgroup_barrier(mem_flag"
    "s::mem_threadgroup);\\n    const uint logical = (t % 32u) * stride;\\n    co"
    "nst uint first = (logical / 32u) * 33u + logical % 32u;\\n    uint s0 = 0u,"
    " s1 = 0u, s2 = 0u, s3 = 0u;\\n    for (uint k = 0u; k < trips; ++k) {\\n    "
    "    const uint q = first + (k % 64u) * 33u;\\n        s0 += tile[q + 0u];\\n"
    "        s1 += tile[q + 33u];\\n        s2 += tile[q + 66u];\\n        s3 += "
    "tile[q + 99u];\\n        s0 += tile[q + 132u];\\n        s1 += tile[q + 165u"
    "];\\n        s2 += tile[q + 198u];\\n        s3 += tile[q + 231u];\\n        "
    "s0 += tile[q + 264u];\\n        s1 += tile[q + 297u];\\n        s2 += tile[q"
    " + 330u];\\n        s3 += tile[q + 363u];\\n        s0 += tile[q + 396u];\\n "
    "       s1 += tile[q + 429u];\\n        s2 += tile[q + 462u];\\n        s3 +="
    " tile[q + 495u];\\n    }\\n    if (i < n) {\\n        out[i] = (s0 + s1) + (s"
    '2 + s3);\\n    }\\n}\\n","source_sha256":"735583b6a215ae98d3092c68ab7f3e5f6ee'
    'af7ca4bcc5bae32827447cecac008"},"summary":{"bandwidth_ceiling_buffer_bytes'
    '":4194304,"bandwidth_ceiling_gb_per_second":82.28275618954719,"bandwidth_c'
    'eiling_kernel":"stream_read_reduce","bank_finding":"stride_slowdown_remove'
    'd_by_padding","bank_largest_padded_ratio":0.7279757299962079,"bank_largest'
    '_unpadded_ratio":1.8225255972696246,"bank_points":4,"correctness_cases":8,'
    '"fma_ceiling_chains":8,"fma_ceiling_fma_steps":1024,"fma_ceiling_gflops_pe'
    'r_second":1802.7144998950682,"fma_fixed_share_of_ceiling_call":0.264579783'
    '14095837,"fma_points":6,"fma_scales_with_work":false,"label":"host-specifi'
    'c Apple GPU (Metal) measurement","load_average_after":[9.4765625,9.4047851'
    '5625,8.75341796875],"load_average_before":[9.4765625,9.40478515625,8.75341'
    '796875],"ridge_point_flops_per_byte":21.908776314472515,"stream_points":2,'
    '"timed_samples":36},"timer":"time.perf_counter_ns"}'
)
RECORDED = json.loads(RECORDED_REPORT_JSON)


def test_the_recorded_report_verifies_on_any_host_as_a_measurement():
    summary = verify_report(json.loads(RECORDED_REPORT_JSON))

    assert canonical_json_bytes(RECORDED).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert RECORDED["backend"] == "torch.mps.compile_shader"
    assert RECORDED["timer"] == "time.perf_counter_ns"
    assert RECORDED["environment"]["torch_version"] == RECORDED_TORCH_VERSION
    assert RECORDED["evidence_boundary"]["vendor_peak_used"] is False
    for kernel in RECORDED["shaders"]["kernels"]:
        assert kernel["limits"] == RECORDED_LIMITS[kernel["name"]].to_mapping()


def test_the_recorded_report_binds_to_its_own_config_only():
    config = config_from_mapping(RECORDED["config"])
    summary = verify_report(json.loads(RECORDED_REPORT_JSON), config=config)

    assert summary["config_sha256"] == config.fingerprint()
    assert config.name == "gpu-ceiling-recorded"
    assert (summary["correctness_cases"], summary["timed_samples"]) == (8, 36)


def test_its_own_fit_flags_these_small_launches_as_fixed_cost_bound():
    ceiling = RECORDED["ceilings"]["fma"]

    assert ceiling["fixed_share_of_ceiling_call"] > ceiling["fixed_share_limit"]
    assert ceiling["scales_with_work"] is False
    assert RECORDED["summary"]["fma_scales_with_work"] is False
    loads = RECORDED["host_load"]
    assert all(value > 0 for value in loads["before"] + loads["after"])


def test_the_command_verifies_the_recorded_file_without_torch(tmp_path, capsys):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")
    expected = (
        f"verified_gpu_ceiling_measurement={RECORDED['report_sha256']} "
        "correctness_cases=8 fma_points=6 stream_points=2 bank_points=4 "
        "scope=apple_gpu_metal_this_host vendor_peak=false cuda=false nvidia=false\n"
    )

    assert load_report(path)[0] == RECORDED
    assert main(["verify", "--report", str(path)]) == EXIT_OK
    assert capsys.readouterr().out == expected
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.gpu_ceiling_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, "verify", "--report", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout == expected
