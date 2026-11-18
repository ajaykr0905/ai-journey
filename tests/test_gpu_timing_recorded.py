"""One real three-clock report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``gpu_timing_cli run``
wrote on 2026-11-18 at 08:32 IST under the shared GPU lock, with torch
2.15.0 on macOS 27.2 (arm64) and load averages near 9.9, 13.6 and 12.7 from
other jobs. The sweep is small (four shapes, three repeats), so the file
stays short; the timings describe that moment and are here only to be
verified. A host without torch or MPS can still check a genuine Apple GPU
(Metal) report, and the Linux CI does.
"""

import json

from ai_journey.gpu_timing_config import config_from_mapping
from ai_journey.gpu_timing_verify import verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_timing_runtime import timing_runtime
from test_kernel_metal_runtime import requires_metal

RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "gpu-timing-recorded",
    "seed": 75,
    "launch": {"group_size": [32, 8]},
    "sweep": {
        "shapes": [[1, 1, 1], [16, 16, 16], [64, 64, 64], [256, 256, 256]],
        "warmup": 1,
        "repeats": 3,
        "max_runtime_seconds": 60,
    },
    "reference": {
        "fma_ceiling_gflops_per_second": 4618.0,
        "source": "docs/gpu-ceilings-lab.md run 1 FMA ceiling",
    },
}
RECORDED_REPORT_JSON = (
    '{"analysis":{"cross_domain_meaning":"cross-domain: the host_synchronized med'
    "ian minus the mps_event median of the same side, from separate samples on tw"
    "o clocks; an estimate of the host cost around the GPU work (submit, wait and"
    ' wake), not a single-clock interval","fit_method":"weighted least squares of'
    " median_ns = intercept_ns + slope_ns_per_flop * flops over the sweep points "
    "of one side and one clock, with weight 1 / median_ns**2 (relative residuals)"
    '; every sum is math.fsum","fits":{"matmul_x_columns":{"host_dispatch":{"inte'
    'rcept_ns":6833.333439798532,"intercept_to_empty_kernel_ratio":1.031446556950'
    '7218,"max_relative_residual":0.05736183683374425,"points":4,"slope_ns_per_fl'
    'op":9.683144004178206e-05},"host_synchronized":{"intercept_ns":136368.842686'
    '15942,"intercept_to_empty_kernel_ratio":1.131691640549041,"max_relative_resi'
    'dual":0.24242991319509344,"points":4,"slope_ns_per_flop":0.01347528863708640'
    '1},"mps_event":{"intercept_ns":86907.28881138389,"intercept_to_empty_kernel_'
    'ratio":0.3414824707716459,"max_relative_residual":0.24170976956623855,"point'
    's":4,"slope_ns_per_flop":0.012557659681807401}},"matmul_x_rows":{"host_dispa'
    'tch":{"intercept_ns":6619.911862102061,"intercept_to_empty_kernel_ratio":0.9'
    '992319791852167,"max_relative_residual":0.19449872171285135,"points":4,"slop'
    'e_ns_per_flop":2.6815280863602256e-05},"host_synchronized":{"intercept_ns":1'
    '25839.98753673444,"intercept_to_empty_kernel_ratio":1.0443152492675056,"max_'
    'relative_residual":0.1635390222593298,"points":4,"slope_ns_per_flop":0.03179'
    '941472212264},"mps_event":{"intercept_ns":74458.82188934032,"intercept_to_em'
    'pty_kernel_ratio":0.2925690447518284,"max_relative_residual":0.1991278835283'
    '8884,"points":4,"slope_ns_per_flop":0.032884665387457185}},"torch_matmul":{"'
    'host_dispatch":{"intercept_ns":11655.984378535759,"intercept_to_empty_kernel'
    '_ratio":1.7593938684582278,"max_relative_residual":0.6982047975738092,"point'
    's":4,"slope_ns_per_flop":0.0011528887177052391},"host_synchronized":{"interc'
    'ept_ns":145222.8241109161,"intercept_to_empty_kernel_ratio":1.20516866482088'
    '04,"max_relative_residual":0.1830780495531305,"points":4,"slope_ns_per_flop"'
    ':0.003235865335676328},"mps_event":{"intercept_ns":94642.00129028098,"interc'
    'ept_to_empty_kernel_ratio":0.371874268331163,"max_relative_residual":0.31598'
    '93551527745,"points":4,"slope_ns_per_flop":0.002582120549988543}}},"fma_ceil'
    'ing_gflops_per_second":4618.0,"fraction_rule":"event_gflops_per_second = flo'
    "ps_per_call / event median_ns; event_fraction_of_fma_ceiling = that / the Da"
    'y 74 FMA ceiling","inference_basis":"an inference from measured timings on t'
    "his host and the Day 74 measured FMA ceiling; no hardware counter, profiler "
    'or vendor figure is used","overhead_estimate":{"empty_kernel_dispatch_to_syn'
    'chronized_ratio":0.054979253112033194,"empty_kernel_host_minus_event":{"cros'
    's_domain":true,"minuend":"host_synchronized","subtrahend":"mps_event","value'
    '_ns":-134000},"empty_kernel_median_ns":{"host_dispatch":6625,"host_synchroni'
    'zed":120500,"mps_event":254500},"smallest_shape":{"index":0,"k":1,"m":1,"med'
    'ian_ns":{"matmul_x_columns":{"host_dispatch":6541,"host_synchronized":118167'
    ',"mps_event":72125},"matmul_x_rows":{"host_dispatch":5542,"host_synchronized'
    '":112583,"mps_event":64417},"torch_matmul":{"host_dispatch":8875,"host_synch'
    'ronized":122750,"mps_event":71917}},"n":1}},"points":[{"flops_per_call":2,"i'
    'ndex":0,"k":1,"m":1,"n":1,"sides":{"matmul_x_columns":{"clocks":{"host_dispa'
    'tch":{"fixed_ns":6833.333439798532,"fixed_share":1.04469246901063,"iqr_ns":4'
    '000,"median_ns":6541,"regime":"fixed_overhead_dominant","size_ns":-292.33343'
    '97985318},"host_synchronized":{"fixed_ns":136368.84268615942,"fixed_share":1'
    '.1540349055672008,"iqr_ns":2874,"median_ns":118167,"regime":"fixed_overhead_'
    'dominant","size_ns":-18201.84268615942},"mps_event":{"fixed_ns":86907.288811'
    '38389,"fixed_share":1.204953744351943,"iqr_ns":101167,"median_ns":72125,"reg'
    'ime":"fixed_overhead_dominant","size_ns":-14782.288811383885}},"dispatch_to_'
    'synchronized_ratio":0.05535386359982059,"event_fraction_of_fma_ceiling":6.00'
    '4685155592651e-09,"event_gflops_per_second":2.7729636048526865e-05,"host_min'
    'us_event":{"cross_domain":true,"minuend":"host_synchronized","subtrahend":"m'
    'ps_event","value_ns":46042}},"matmul_x_rows":{"clocks":{"host_dispatch":{"fi'
    'xed_ns":6619.911862102061,"fixed_share":1.1944987120357382,"iqr_ns":6416,"me'
    'dian_ns":5542,"regime":"fixed_overhead_dominant","size_ns":-1077.91186210206'
    '07},"host_synchronized":{"fixed_ns":125839.98753673444,"fixed_share":1.11775'
    '30136586735,"iqr_ns":1750,"median_ns":112583,"regime":"fixed_overhead_domina'
    'nt","size_ns":-13256.987536734436},"mps_event":{"fixed_ns":74458.82188934032'
    ',"fixed_share":1.1558877608292892,"iqr_ns":2000,"median_ns":64417,"regime":"'
    'fixed_overhead_dominant","size_ns":-10041.821889340325}},"dispatch_to_synchr'
    'onized_ratio":0.04922590444383255,"event_fraction_of_fma_ceiling":6.72319289'
    '7016626e-09,"event_gflops_per_second":3.104770479842278e-05,"host_minus_even'
    't":{"cross_domain":true,"minuend":"host_synchronized","subtrahend":"mps_even'
    't","value_ns":48166}},"torch_matmul":{"clocks":{"host_dispatch":{"fixed_ns":'
    '11655.984378535759,"fixed_share":1.3133503525110715,"iqr_ns":3500,"median_ns'
    '":8875,"regime":"fixed_overhead_dominant","size_ns":-2780.984378535759},"hos'
    't_synchronized":{"fixed_ns":145222.8241109161,"fixed_share":1.18307799683027'
    '36,"iqr_ns":26208,"median_ns":122750,"regime":"fixed_overhead_dominant","siz'
    'e_ns":-22472.824110916088},"mps_event":{"fixed_ns":94642.00129028098,"fixed_'
    'share":1.3159892833444247,"iqr_ns":131292,"median_ns":71917,"regime":"fixed_'
    'overhead_dominant","size_ns":-22725.001290280983}},"dispatch_to_synchronized'
    '_ratio":0.07230142566191446,"event_fraction_of_fma_ceiling":6.02205204398292'
    '46e-09,"event_gflops_per_second":2.7809836339113145e-05,"host_minus_event":{'
    '"cross_domain":true,"minuend":"host_synchronized","subtrahend":"mps_event","'
    'value_ns":50833}}}},{"flops_per_call":8192,"index":1,"k":16,"m":16,"n":16,"s'
    'ides":{"matmul_x_columns":{"clocks":{"host_dispatch":{"fixed_ns":6833.333439'
    '798532,"fixed_share":0.9425287503170389,"iqr_ns":4083,"median_ns":7250,"regi'
    'me":"fixed_overhead_dominant","size_ns":416.6665602014682},"host_synchronize'
    'd":{"fixed_ns":136368.84268615942,"fixed_share":0.9932759569833597,"iqr_ns":'
    '171542,"median_ns":137292,"regime":"fixed_overhead_dominant","size_ns":923.1'
    '573138405802},"mps_event":{"fixed_ns":86907.28881138389,"fixed_share":0.9120'
    '103346701075,"iqr_ns":192125,"median_ns":95292,"regime":"fixed_overhead_domi'
    'nant","size_ns":8384.711188616115}},"dispatch_to_synchronized_ratio":0.05280'
    '715555167089,"event_fraction_of_fma_ceiling":1.861570863667258e-05,"event_gf'
    'lops_per_second":0.08596734248415397,"host_minus_event":{"cross_domain":true'
    ',"minuend":"host_synchronized","subtrahend":"mps_event","value_ns":42000}},"'
    'matmul_x_rows":{"clocks":{"host_dispatch":{"fixed_ns":6619.911862102061,"fix'
    'ed_share":0.9027562882997492,"iqr_ns":1917,"median_ns":7333,"regime":"fixed_'
    'overhead_dominant","size_ns":713.0881378979393},"host_synchronized":{"fixed_'
    'ns":125839.98753673444,"fixed_share":0.9850488261192519,"iqr_ns":21291,"medi'
    'an_ns":127750,"regime":"fixed_overhead_dominant","size_ns":1910.012463265564'
    '3},"mps_event":{"fixed_ns":74458.82188934032,"fixed_share":0.940528526903132'
    '9,"iqr_ns":78125,"median_ns":79167,"regime":"fixed_overhead_dominant","size_'
    'ns":4708.178110659675}},"dispatch_to_synchronized_ratio":0.05740117416829746'
    ',"event_fraction_of_fma_ceiling":2.2407418588626618e-05,"event_gflops_per_se'
    'cond":0.10347745904227772,"host_minus_event":{"cross_domain":true,"minuend":'
    '"host_synchronized","subtrahend":"mps_event","value_ns":48583}},"torch_matmu'
    'l":{"clocks":{"host_dispatch":{"fixed_ns":11655.984378535759,"fixed_share":0'
    '.3115157382616393,"iqr_ns":29583,"median_ns":37417,"regime":"size_dependent_'
    'dominant","size_ns":25761.01562146424},"host_synchronized":{"fixed_ns":14522'
    '2.8241109161,"fixed_share":0.9200984839192828,"iqr_ns":120833,"median_ns":15'
    '7834,"regime":"fixed_overhead_dominant","size_ns":12611.175889083912},"mps_e'
    'vent":{"fixed_ns":94642.00129028098,"fixed_share":0.709151952601425,"iqr_ns"'
    ':252625,"median_ns":133458,"regime":"fixed_overhead_dominant","size_ns":3881'
    '5.99870971902}},"dispatch_to_synchronized_ratio":0.23706552453843913,"event_'
    'fraction_of_fma_ceiling":1.3292032754917677e-05,"event_gflops_per_second":0.'
    '061382607262209836,"host_minus_event":{"cross_domain":true,"minuend":"host_s'
    'ynchronized","subtrahend":"mps_event","value_ns":24376}}}},{"flops_per_call"'
    ':524288,"index":2,"k":64,"m":64,"n":64,"sides":{"matmul_x_columns":{"clocks"'
    ':{"host_dispatch":{"fixed_ns":6833.333439798532,"fixed_share":1.000048798448'
    '49,"iqr_ns":7666,"median_ns":6833,"regime":"fixed_overhead_dominant","size_n'
    's":-0.33343979853179917},"host_synchronized":{"fixed_ns":136368.84268615942,'
    '"fixed_share":0.7202554358232511,"iqr_ns":43750,"median_ns":189334,"regime":'
    '"fixed_overhead_dominant","size_ns":52965.15731384058},"mps_event":{"fixed_n'
    's":86907.28881138389,"fixed_share":0.7048899264460297,"iqr_ns":23417,"median'
    '_ns":123292,"regime":"fixed_overhead_dominant","size_ns":36384.711188616115}'
    '},"dispatch_to_synchronized_ratio":0.03608966165612093,"event_fraction_of_fm'
    'a_ceiling":0.0009208334593807499,"event_gflops_per_second":4.252408915420303'
    ',"host_minus_event":{"cross_domain":true,"minuend":"host_synchronized","subt'
    'rahend":"mps_event","value_ns":66042}},"matmul_x_rows":{"clocks":{"host_disp'
    'atch":{"fixed_ns":6619.911862102061,"fixed_share":0.8188906311358314,"iqr_ns'
    '":2000,"median_ns":8084,"regime":"fixed_overhead_dominant","size_ns":1464.08'
    '81378979393},"host_synchronized":{"fixed_ns":125839.98753673444,"fixed_share'
    '":0.7386059429889035,"iqr_ns":19292,"median_ns":170375,"regime":"fixed_overh'
    'ead_dominant","size_ns":44535.012463265564},"mps_event":{"fixed_ns":74458.82'
    '188934032,"fixed_share":0.6502953876798282,"iqr_ns":41083,"median_ns":114500'
    ',"regime":"fixed_overhead_dominant","size_ns":40041.178110659675}},"dispatch'
    '_to_synchronized_ratio":0.04744827586206896,"event_fraction_of_fma_ceiling":'
    '0.0009915406015194009,"event_gflops_per_second":4.578934497816594,"host_minu'
    's_event":{"cross_domain":true,"minuend":"host_synchronized","subtrahend":"mp'
    's_event","value_ns":55875}},"torch_matmul":{"clocks":{"host_dispatch":{"fixe'
    'd_ns":11655.984378535759,"fixed_share":0.2869165385485725,"iqr_ns":20749,"me'
    'dian_ns":40625,"regime":"size_dependent_dominant","size_ns":28969.0156214642'
    '4},"host_synchronized":{"fixed_ns":145222.8241109161,"fixed_share":0.8111103'
    '769557763,"iqr_ns":141542,"median_ns":179042,"regime":"fixed_overhead_domina'
    'nt","size_ns":33819.17588908391},"mps_event":{"fixed_ns":94642.00129028098,"'
    'fixed_share":0.6744101621876607,"iqr_ns":395833,"median_ns":140333,"regime":'
    '"fixed_overhead_dominant","size_ns":45690.99870971902}},"dispatch_to_synchro'
    'nized_ratio":0.22690206767127266,"event_fraction_of_fma_ceiling":0.000809014'
    '2651690722,"event_gflops_per_second":3.7360278765507755,"host_minus_event":{'
    '"cross_domain":true,"minuend":"host_synchronized","subtrahend":"mps_event","'
    'value_ns":38709}}}},{"flops_per_call":33554432,"index":3,"k":256,"m":256,"n"'
    ':256,"sides":{"matmul_x_columns":{"clocks":{"host_dispatch":{"fixed_ns":6833'
    '.333439798532,"fixed_share":0.6776411582505486,"iqr_ns":3083,"median_ns":100'
    '84,"regime":"fixed_overhead_dominant","size_ns":3250.666560201468},"host_syn'
    'chronized":{"fixed_ns":136368.84268615942,"fixed_share":0.23441141845493668,'
    '"iqr_ns":22000,"median_ns":581750,"regime":"size_dependent_dominant","size_n'
    's":445381.1573138406},"mps_event":{"fixed_ns":86907.28881138389,"fixed_share'
    '":0.17362636664132858,"iqr_ns":15708,"median_ns":500542,"regime":"size_depen'
    'dent_dominant","size_ns":413634.7111886161}},"dispatch_to_synchronized_ratio'
    '":0.01733390631714654,"event_fraction_of_fma_ceiling":0.014516283404657692,"'
    'event_gflops_per_second":67.03619676270922,"host_minus_event":{"cross_domain'
    '":true,"minuend":"host_synchronized","subtrahend":"mps_event","value_ns":812'
    '08}},"matmul_x_rows":{"clocks":{"host_dispatch":{"fixed_ns":6619.91186210206'
    '1,"fixed_share":0.8826549149469414,"iqr_ns":8167,"median_ns":7500,"regime":"'
    'fixed_overhead_dominant","size_ns":880.0881378979393},"host_synchronized":{"'
    'fixed_ns":125839.98753673444,"fixed_share":0.10735285903515025,"iqr_ns":5187'
    '92,"median_ns":1172209,"regime":"size_dependent_dominant","size_ns":1046369.'
    '0124632656},"mps_event":{"fixed_ns":74458.82188934032,"fixed_share":0.065188'
    '49621902519,"iqr_ns":86292,"median_ns":1142208,"regime":"size_dependent_domi'
    'nant","size_ns":1067749.1781106596}},"dispatch_to_synchronized_ratio":0.0063'
    '981764344071745,"event_fraction_of_fma_ceiling":0.0063613715960089325,"event'
    '_gflops_per_second":29.37681403036925,"host_minus_event":{"cross_domain":tru'
    'e,"minuend":"host_synchronized","subtrahend":"mps_event","value_ns":30001}},'
    '"torch_matmul":{"clocks":{"host_dispatch":{"fixed_ns":11655.984378535759,"fi'
    'xed_share":0.23468267418075903,"iqr_ns":92875,"median_ns":49667,"regime":"si'
    'ze_dependent_dominant","size_ns":38011.01562146424},"host_synchronized":{"fi'
    'xed_ns":145222.8241109161,"fixed_share":0.574475553462594,"iqr_ns":354459,"m'
    'edian_ns":252792,"regime":"fixed_overhead_dominant","size_ns":107569.1758890'
    '8391},"mps_event":{"fixed_ns":94642.00129028098,"fixed_share":0.525424018266'
    '6536,"iqr_ns":73708,"median_ns":180125,"regime":"fixed_overhead_dominant","s'
    'ize_ns":85482.99870971902}},"dispatch_to_synchronized_ratio":0.1964737808158'
    '486,"event_fraction_of_fma_ceiling":0.04033870660893363,"event_gflops_per_se'
    'cond":186.2841471200555,"host_minus_event":{"cross_domain":true,"minuend":"h'
    'ost_synchronized","subtrahend":"mps_event","value_ns":72667}}}}],"regime_rul'
    'e":"fixed_overhead_dominant when fixed_share >= 0.5, otherwise size_dependen'
    't_dominant; an inference from the fitted line, not a hardware counter","spli'
    't_rule":"fixed_ns = the fitted intercept_ns; size_ns = median_ns - fixed_ns;'
    ' fixed_share = fixed_ns / median_ns"},"backend":"torch.mps.compile_shader","'
    'benchmark":{"agreement":"4 * gamma_K(2**-24) * (|A| @ |B|), on the device in'
    ' float32","device":"mps","empty":{"kernel":{"buffers":["sink"],"launch":{"ac'
    'tive_threads":1,"extent":[1],"group_size":[1],"groups":[1],"launched_threads'
    '":1,"padding_threads":0,"threads":[1],"threads_per_group":1},"name":"empty_k'
    'ernel","never_position":4294967295},"launch":{"active_threads":1,"extent":[1'
    '],"group_size":[1],"groups":[1],"idle_lanes_per_threadgroup":31,"launched_th'
    'reads":1,"padding_threads":0,"simd_groups_per_threadgroup":1,"threads":[1],"'
    'threads_per_group":1},"timing":{"clocks":{"host_dispatch":{"clock":"host_dis'
    'patch","domain":"host_perf_counter_ns","samples_ns":[6625,12167,5458],"timin'
    'g":{"count":3,"iqr_ns":6709,"max_ns":12167,"median_ns":6625,"min_ns":5458,"p'
    '25_ns":5458,"p75_ns":12167}},"host_synchronized":{"clock":"host_synchronized'
    '","domain":"host_perf_counter_ns","samples_ns":[120500,125542,101333],"timin'
    'g":{"count":3,"iqr_ns":24209,"max_ns":125542,"median_ns":120500,"min_ns":101'
    '333,"p25_ns":101333,"p75_ns":125542}},"mps_event":{"clock":"mps_event","doma'
    'in":"mps_event_stream","elapsed_ms":[0.2544999879319221,0.23874998441897333,'
    '0.2604166802484542],"samples_ns":[254500,238750,260417],"timing":{"count":3,'
    '"iqr_ns":21667,"max_ns":260417,"median_ns":254500,"min_ns":238750,"p25_ns":2'
    '38750,"p75_ns":260417}}},"implementation":"empty_kernel","repeats":3,"warmup'
    '":1}},"points":[{"agreement_bound":2.3841859331241806e-07,"flops_per_call":2'
    ',"index":0,"k":1,"launches":{"x_columns":{"active_threads":1,"extent":[1,1],'
    '"group_size":[32,8],"groups":[1,1],"idle_lanes_per_threadgroup":0,"launched_'
    'threads":256,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads"'
    ':[32,8],"threads_per_group":256},"x_rows":{"active_threads":1,"extent":[1,1]'
    ',"group_size":[32,8],"groups":[1,1],"idle_lanes_per_threadgroup":0,"launched'
    '_threads":256,"padding_threads":255,"simd_groups_per_threadgroup":8,"threads'
    '":[32,8],"threads_per_group":256}},"m":1,"n":1,"sides":{"matmul_x_columns":{'
    '"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_perf_count'
    'er_ns","samples_ns":[6541,5416,9416],"timing":{"count":3,"iqr_ns":4000,"max_'
    'ns":9416,"median_ns":6541,"min_ns":5416,"p25_ns":5416,"p75_ns":9416}},"host_'
    'synchronized":{"clock":"host_synchronized","domain":"host_perf_counter_ns","'
    'samples_ns":[119333,116459,118167],"timing":{"count":3,"iqr_ns":2874,"max_ns'
    '":119333,"median_ns":118167,"min_ns":116459,"p25_ns":116459,"p75_ns":119333}'
    '},"mps_event":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":'
    '[0.0721250253263861,0.1681666763033718,0.06700001540593803],"samples_ns":[72'
    '125,168167,67000],"timing":{"count":3,"iqr_ns":101167,"max_ns":168167,"media'
    'n_ns":72125,"min_ns":67000,"p25_ns":67000,"p75_ns":168167}}},"implementation'
    '":"matmul_x_columns","repeats":3,"warmup":1},"matmul_x_rows":{"clocks":{"hos'
    't_dispatch":{"clock":"host_dispatch","domain":"host_perf_counter_ns","sample'
    's_ns":[5500,5542,11916],"timing":{"count":3,"iqr_ns":6416,"max_ns":11916,"me'
    'dian_ns":5542,"min_ns":5500,"p25_ns":5500,"p75_ns":11916}},"host_synchronize'
    'd":{"clock":"host_synchronized","domain":"host_perf_counter_ns","samples_ns"'
    ':[112583,111708,113458],"timing":{"count":3,"iqr_ns":1750,"max_ns":113458,"m'
    'edian_ns":112583,"min_ns":111708,"p25_ns":111708,"p75_ns":113458}},"mps_even'
    't":{"clock":"mps_event","domain":"mps_event_stream","elapsed_ms":[0.06487499'
    '922513962,0.06287501309998333,0.06441664299927652],"samples_ns":[64875,62875'
    ',64417],"timing":{"count":3,"iqr_ns":2000,"max_ns":64875,"median_ns":64417,"'
    'min_ns":62875,"p25_ns":62875,"p75_ns":64875}}},"implementation":"matmul_x_ro'
    'ws","repeats":3,"warmup":1},"torch_matmul":{"clocks":{"host_dispatch":{"cloc'
    'k":"host_dispatch","domain":"host_perf_counter_ns","samples_ns":[9458,5958,8'
    '875],"timing":{"count":3,"iqr_ns":3500,"max_ns":9458,"median_ns":8875,"min_n'
    's":5958,"p25_ns":5958,"p75_ns":9458}},"host_synchronized":{"clock":"host_syn'
    'chronized","domain":"host_perf_counter_ns","samples_ns":[120500,122750,14670'
    '8],"timing":{"count":3,"iqr_ns":26208,"max_ns":146708,"median_ns":122750,"mi'
    'n_ns":120500,"p25_ns":120500,"p75_ns":146708}},"mps_event":{"clock":"mps_eve'
    'nt","domain":"mps_event_stream","elapsed_ms":[0.07191664190031588,0.06745834'
    '252797067,0.19874999998137355],"samples_ns":[71917,67458,198750],"timing":{"'
    'count":3,"iqr_ns":131292,"max_ns":198750,"median_ns":71917,"min_ns":67458,"p'
    '25_ns":67458,"p75_ns":198750}}},"implementation":"torch_matmul","repeats":3,'
    '"warmup":1}},"torch_disagreement":0.0},{"agreement_bound":3.8147009036072765'
    'e-06,"flops_per_call":8192,"index":1,"k":16,"launches":{"x_columns":{"active'
    '_threads":256,"extent":[16,16],"group_size":[32,8],"groups":[1,2],"idle_lane'
    's_per_threadgroup":0,"launched_threads":512,"padding_threads":256,"simd_grou'
    'ps_per_threadgroup":8,"threads":[32,16],"threads_per_group":256},"x_rows":{"'
    'active_threads":256,"extent":[16,16],"group_size":[32,8],"groups":[1,2],"idl'
    'e_lanes_per_threadgroup":0,"launched_threads":512,"padding_threads":256,"sim'
    'd_groups_per_threadgroup":8,"threads":[32,16],"threads_per_group":256}},"m":'
    '16,"n":16,"sides":{"matmul_x_columns":{"clocks":{"host_dispatch":{"clock":"h'
    'ost_dispatch","domain":"host_perf_counter_ns","samples_ns":[7250,7084,11167]'
    ',"timing":{"count":3,"iqr_ns":4083,"max_ns":11167,"median_ns":7250,"min_ns":'
    '7084,"p25_ns":7084,"p75_ns":11167}},"host_synchronized":{"clock":"host_synch'
    'ronized","domain":"host_perf_counter_ns","samples_ns":[137292,305084,133542]'
    ',"timing":{"count":3,"iqr_ns":171542,"max_ns":305084,"median_ns":137292,"min'
    '_ns":133542,"p25_ns":133542,"p75_ns":305084}},"mps_event":{"clock":"mps_even'
    't","domain":"mps_event_stream","elapsed_ms":[0.09529167436994612,0.269250012'
    '9342079,0.07712497608736157],"samples_ns":[95292,269250,77125],"timing":{"co'
    'unt":3,"iqr_ns":192125,"max_ns":269250,"median_ns":95292,"min_ns":77125,"p25'
    '_ns":77125,"p75_ns":269250}}},"implementation":"matmul_x_columns","repeats":'
    '3,"warmup":1},"matmul_x_rows":{"clocks":{"host_dispatch":{"clock":"host_disp'
    'atch","domain":"host_perf_counter_ns","samples_ns":[5958,7875,7333],"timing"'
    ':{"count":3,"iqr_ns":1917,"max_ns":7875,"median_ns":7333,"min_ns":5958,"p25_'
    'ns":5958,"p75_ns":7875}},"host_synchronized":{"clock":"host_synchronized","d'
    'omain":"host_perf_counter_ns","samples_ns":[127750,125792,147083],"timing":{'
    '"count":3,"iqr_ns":21291,"max_ns":147083,"median_ns":127750,"min_ns":125792,'
    '"p25_ns":125792,"p75_ns":147083}},"mps_event":{"clock":"mps_event","domain":'
    '"mps_event_stream","elapsed_ms":[0.0791666388977319,0.15354165225289762,0.07'
    '541666855104268],"samples_ns":[79167,153542,75417],"timing":{"count":3,"iqr_'
    'ns":78125,"max_ns":153542,"median_ns":79167,"min_ns":75417,"p25_ns":75417,"p'
    '75_ns":153542}}},"implementation":"matmul_x_rows","repeats":3,"warmup":1},"t'
    'orch_matmul":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"h'
    'ost_perf_counter_ns","samples_ns":[27792,37417,57375],"timing":{"count":3,"i'
    'qr_ns":29583,"max_ns":57375,"median_ns":37417,"min_ns":27792,"p25_ns":27792,'
    '"p75_ns":57375}},"host_synchronized":{"clock":"host_synchronized","domain":"'
    'host_perf_counter_ns","samples_ns":[157834,275541,154708],"timing":{"count":'
    '3,"iqr_ns":120833,"max_ns":275541,"median_ns":157834,"min_ns":154708,"p25_ns'
    '":154708,"p75_ns":275541}},"mps_event":{"clock":"mps_event","domain":"mps_ev'
    'ent_stream","elapsed_ms":[0.13345832121558487,0.1078750065062195,0.360499980'
    '1609665],"samples_ns":[133458,107875,360500],"timing":{"count":3,"iqr_ns":25'
    '2625,"max_ns":360500,"median_ns":133458,"min_ns":107875,"p25_ns":107875,"p75'
    '_ns":360500}}},"implementation":"torch_matmul","repeats":3,"warmup":1}},"tor'
    'ch_disagreement":0.0},{"agreement_bound":1.5258847270382958e-05,"flops_per_c'
    'all":524288,"index":2,"k":64,"launches":{"x_columns":{"active_threads":4096,'
    '"extent":[64,64],"group_size":[32,8],"groups":[2,8],"idle_lanes_per_threadgr'
    'oup":0,"launched_threads":4096,"padding_threads":0,"simd_groups_per_threadgr'
    'oup":8,"threads":[64,64],"threads_per_group":256},"x_rows":{"active_threads"'
    ':4096,"extent":[64,64],"group_size":[32,8],"groups":[2,8],"idle_lanes_per_th'
    'readgroup":0,"launched_threads":4096,"padding_threads":0,"simd_groups_per_th'
    'readgroup":8,"threads":[64,64],"threads_per_group":256}},"m":64,"n":64,"side'
    's":{"matmul_x_columns":{"clocks":{"host_dispatch":{"clock":"host_dispatch","'
    'domain":"host_perf_counter_ns","samples_ns":[6833,5792,13458],"timing":{"cou'
    'nt":3,"iqr_ns":7666,"max_ns":13458,"median_ns":6833,"min_ns":5792,"p25_ns":5'
    '792,"p75_ns":13458}},"host_synchronized":{"clock":"host_synchronized","domai'
    'n":"host_perf_counter_ns","samples_ns":[189334,160583,204333],"timing":{"cou'
    'nt":3,"iqr_ns":43750,"max_ns":204333,"median_ns":189334,"min_ns":160583,"p25'
    '_ns":160583,"p75_ns":204333}},"mps_event":{"clock":"mps_event","domain":"mps'
    '_event_stream","elapsed_ms":[0.13124998076818883,0.12329168384894729,0.10783'
    '332982100546],"samples_ns":[131250,123292,107833],"timing":{"count":3,"iqr_n'
    's":23417,"max_ns":131250,"median_ns":123292,"min_ns":107833,"p25_ns":107833,'
    '"p75_ns":131250}}},"implementation":"matmul_x_columns","repeats":3,"warmup":'
    '1},"matmul_x_rows":{"clocks":{"host_dispatch":{"clock":"host_dispatch","doma'
    'in":"host_perf_counter_ns","samples_ns":[6125,8125,8084],"timing":{"count":3'
    ',"iqr_ns":2000,"max_ns":8125,"median_ns":8084,"min_ns":6125,"p25_ns":6125,"p'
    '75_ns":8125}},"host_synchronized":{"clock":"host_synchronized","domain":"hos'
    't_perf_counter_ns","samples_ns":[170375,163083,182375],"timing":{"count":3,"'
    'iqr_ns":19292,"max_ns":182375,"median_ns":170375,"min_ns":163083,"p25_ns":16'
    '3083,"p75_ns":182375}},"mps_event":{"clock":"mps_event","domain":"mps_event_'
    'stream","elapsed_ms":[0.1514999894425273,0.11041667312383652,0.1144999987445'
    '7717],"samples_ns":[151500,110417,114500],"timing":{"count":3,"iqr_ns":41083'
    ',"max_ns":151500,"median_ns":114500,"min_ns":110417,"p25_ns":110417,"p75_ns"'
    ':151500}}},"implementation":"matmul_x_rows","repeats":3,"warmup":1},"torch_m'
    'atmul":{"clocks":{"host_dispatch":{"clock":"host_dispatch","domain":"host_pe'
    'rf_counter_ns","samples_ns":[31209,51958,40625],"timing":{"count":3,"iqr_ns"'
    ':20749,"max_ns":51958,"median_ns":40625,"min_ns":31209,"p25_ns":31209,"p75_n'
    's":51958}},"host_synchronized":{"clock":"host_synchronized","domain":"host_p'
    'erf_counter_ns","samples_ns":[179042,173166,314708],"timing":{"count":3,"iqr'
    '_ns":141542,"max_ns":314708,"median_ns":179042,"min_ns":173166,"p25_ns":1731'
    '66,"p75_ns":314708}},"mps_event":{"clock":"mps_event","domain":"mps_event_st'
    'ream","elapsed_ms":[0.14033331535756588,0.10462498175911605,0.50045835087075'
    '83],"samples_ns":[140333,104625,500458],"timing":{"count":3,"iqr_ns":395833,'
    '"max_ns":500458,"median_ns":140333,"min_ns":104625,"p25_ns":104625,"p75_ns":'
    '500458}}},"implementation":"torch_matmul","repeats":3,"warmup":1}},"torch_di'
    'sagreement":0.0},{"agreement_bound":6.103608758678569e-05,"flops_per_call":3'
    '3554432,"index":3,"k":256,"launches":{"x_columns":{"active_threads":65536,"e'
    'xtent":[256,256],"group_size":[32,8],"groups":[8,32],"idle_lanes_per_threadg'
    'roup":0,"launched_threads":65536,"padding_threads":0,"simd_groups_per_thread'
    'group":8,"threads":[256,256],"threads_per_group":256},"x_rows":{"active_thre'
    'ads":65536,"extent":[256,256],"group_size":[32,8],"groups":[8,32],"idle_lane'
    's_per_threadgroup":0,"launched_threads":65536,"padding_threads":0,"simd_grou'
    'ps_per_threadgroup":8,"threads":[256,256],"threads_per_group":256}},"m":256,'
    '"n":256,"sides":{"matmul_x_columns":{"clocks":{"host_dispatch":{"clock":"hos'
    't_dispatch","domain":"host_perf_counter_ns","samples_ns":[10875,10084,7792],'
    '"timing":{"count":3,"iqr_ns":3083,"max_ns":10875,"median_ns":10084,"min_ns":'
    '7792,"p25_ns":7792,"p75_ns":10875}},"host_synchronized":{"clock":"host_synch'
    'ronized","domain":"host_perf_counter_ns","samples_ns":[591708,581750,569708]'
    ',"timing":{"count":3,"iqr_ns":22000,"max_ns":591708,"median_ns":581750,"min_'
    'ns":569708,"p25_ns":569708,"p75_ns":591708}},"mps_event":{"clock":"mps_event'
    '","domain":"mps_event_stream","elapsed_ms":[0.5005416460335255,0.49099998432'
    '21158,0.5067083402536809],"samples_ns":[500542,491000,506708],"timing":{"cou'
    'nt":3,"iqr_ns":15708,"max_ns":506708,"median_ns":500542,"min_ns":491000,"p25'
    '_ns":491000,"p75_ns":506708}}},"implementation":"matmul_x_columns","repeats"'
    ':3,"warmup":1},"matmul_x_rows":{"clocks":{"host_dispatch":{"clock":"host_dis'
    'patch","domain":"host_perf_counter_ns","samples_ns":[14917,7500,6750],"timin'
    'g":{"count":3,"iqr_ns":8167,"max_ns":14917,"median_ns":7500,"min_ns":6750,"p'
    '25_ns":6750,"p75_ns":14917}},"host_synchronized":{"clock":"host_synchronized'
    '","domain":"host_perf_counter_ns","samples_ns":[1368875,1172209,850083],"tim'
    'ing":{"count":3,"iqr_ns":518792,"max_ns":1368875,"median_ns":1172209,"min_ns'
    '":850083,"p25_ns":850083,"p75_ns":1368875}},"mps_event":{"clock":"mps_event"'
    ',"domain":"mps_event_stream","elapsed_ms":[1.1288333334960043,1.215124997543'
    '171,1.1422083480283618],"samples_ns":[1128833,1215125,1142208],"timing":{"co'
    'unt":3,"iqr_ns":86292,"max_ns":1215125,"median_ns":1142208,"min_ns":1128833,'
    '"p25_ns":1128833,"p75_ns":1215125}}},"implementation":"matmul_x_rows","repea'
    'ts":3,"warmup":1},"torch_matmul":{"clocks":{"host_dispatch":{"clock":"host_d'
    'ispatch","domain":"host_perf_counter_ns","samples_ns":[30000,122875,49667],"'
    'timing":{"count":3,"iqr_ns":92875,"max_ns":122875,"median_ns":49667,"min_ns"'
    ':30000,"p25_ns":30000,"p75_ns":122875}},"host_synchronized":{"clock":"host_s'
    'ynchronized","domain":"host_perf_counter_ns","samples_ns":[252792,605792,251'
    '333],"timing":{"count":3,"iqr_ns":354459,"max_ns":605792,"median_ns":252792,'
    '"min_ns":251333,"p25_ns":251333,"p75_ns":605792}},"mps_event":{"clock":"mps_'
    'event","domain":"mps_event_stream","elapsed_ms":[0.2227500081062317,0.180125'
    '00368058681,0.14904167619533837],"samples_ns":[222750,180125,149042],"timing'
    '":{"count":3,"iqr_ns":73708,"max_ns":222750,"median_ns":180125,"min_ns":1490'
    '42,"p25_ns":149042,"p75_ns":222750}}},"implementation":"torch_matmul","repea'
    'ts":3,"warmup":1}},"torch_disagreement":0.0}],"torch_equivalent":"torch.matm'
    'ul(a, b, out=c) on the MPS device, float32"},"clocks":{"definitions":{"host_'
    'dispatch":"time.perf_counter_ns() just before and just after the launch call'
    " alone, with no synchronize between the readings; the queue is drained befor"
    'e the first reading and after the second","host_synchronized":"time.perf_cou'
    "nter_ns() just before the launch and just after the torch.mps.synchronize() "
    'that follows it; the queue is drained before the first reading","mps_event":'
    '"torch.mps.Event(enable_timing=True): start.record() just before the launch '
    "and end.record() just after it, then torch.mps.synchronize() and start.elaps"
    'ed_time(end) in ms; samples_ns = round(ms * 1e6)"},"domain_definitions":{"ho'
    'st_perf_counter_ns":"host-side: time.perf_counter_ns on the CPU thread that '
    'issued the launch","mps_event_stream":"device-side: the interval between two'
    " points in the MPS command stream; the PyTorch MPS header says a command-buf"
    "fer completion handler stores each timed event's completion time and does no"
    "t name the clock it reads, so these values are never subtracted from host va"
    'lues without a cross_domain record"},"domains":{"host_dispatch":"host_perf_c'
    'ounter_ns","host_synchronized":"host_perf_counter_ns","mps_event":"mps_event'
    '_stream"},"order":["mps_event","host_synchronized","host_dispatch"],"schedul'
    'e":"one launch and one clock per sample; per side, warmup untimed samples on'
    " each clock first; then repeat r visits the sides shifted by r and, for each"
    ' side, the clocks shifted by r"},"config":{"launch":{"group_size":[32,8]},"n'
    'ame":"gpu-timing-recorded","reference":{"fma_ceiling_gflops_per_second":4618'
    '.0,"source":"docs/gpu-ceilings-lab.md run 1 FMA ceiling"},"schema_version":1'
    ',"seed":75,"sweep":{"max_runtime_seconds":60,"repeats":3,"shapes":[[1,1,1],['
    '16,16,16],[64,64,64],[256,256,256]],"warmup":1}},"config_sha256":"a82eae9b4d'
    'd762b7f4013b9367bd01b29b7ad73dea149fd4ff36a4f6fbd7ee3e","environment":{"maco'
    's_version":"27.2","numpy_version":"1.26.4","platform_machine":"arm64","platf'
    'orm_system":"Darwin","python_implementation":"CPython","python_version":"3.1'
    '1.14","torch_float32_matmul_precision":"highest","torch_version":"2.15.0"},"'
    'evidence_boundary":{"apple_gpu_metal_this_host":true,"cuda_kernel_executed":'
    'false,"distributed_claim":false,"hardware_counters_used":false,"host_specifi'
    'c":true,"nvidia_gpu_used":false,"portable_performance_claim":false,"producti'
    'on_claim":false,"statement":"These are timings on the Apple GPU of the host '
    "that made this report, on two clocks: MPS events in the command stream and p"
    "erf_counter_ns on the host. They are host-specific Apple GPU (Metal) measure"
    "ments at the recorded load averages. No CUDA kernel ran, no NVIDIA GPU was u"
    "sed, and no hardware counter was read. A fixed overhead or a regime is an in"
    "ference from a fitted line, and a fraction of the FMA ceiling compares with "
    'an earlier measurement on this host, not a vendor peak.","vendor_peak_claime'
    'd":false},"host_load":{"after":[9.87109375,13.619140625,12.64697265625],"bef'
    'ore":[9.87109375,13.619140625,12.64697265625],"source":"os.getloadavg(): the'
    " 1, 5 and 15 minute load averages that uptime prints, read before the checks"
    ' and after the last sample"},"kind":"gpu_timing_three_clocks","label":"host-'
    'specific Apple GPU (Metal) measurement","measured":true,"measurement_scope":'
    '"apple_gpu_metal_this_host","percentile_method":"nearest rank: sorted sample'
    "s[ceil(p * n / 100) - 1] for p in 25, 50 and 75; median is p50 and IQR is p7"
    '5 - p25; no interpolation","planned_work":{"flops":1227128904,"launches":156'
    ',"sides":13,"sweep_points":4,"timed_samples":117},"report_sha256":"9428099af'
    'c19010ba059f29f9ddd9e2f3d10eb57b66247412ceed1edd623133a","run":{"elapsed_ns"'
    ':859320750,"max_runtime_ns":60000000000,"timed_host_ns":11669708},"schema_ve'
    'rsion":1,"shaders":{"empty_source":"#include <metal_stdlib>\\nusing namespace'
    " metal;\\n\\n// One compare per thread and no store in any launch of this lab:"
    " the\\n// guard depends on the thread position, so the compiler keeps the sto"
    "re\\n// and the buffer, and the launch keeps its full dispatch path.\\nkernel "
    "void empty_kernel(device uint* sink [[buffer(0)]],\\n                        "
    " uint position [[thread_position_in_grid]]) {\\n    if (position == 0xFFFFFFF"
    'Fu) {\\n        sink[0] = position;\\n    }\\n}\\n","empty_source_sha256":"ee3b2'
    'fd4da599a7bd74ceea19b0d3ea1f4ff492df4c3e23124e5c372b6aa7520","limits":{"empt'
    'y_kernel":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_byt'
    'es":0,"thread_execution_width":32},"matmul_x_columns":{"max_threads_per_thre'
    'adgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_width":3'
    '2},"matmul_x_rows":{"max_threads_per_threadgroup":1024,"static_threadgroup_m'
    'emory_bytes":0,"thread_execution_width":32}},"matmul_source_sha256":"beef263'
    '92d0034be037dbd0740d2b5eca9429f76cc8bda48ba4d37fc928df9cc"},"summary":{"empt'
    'y_kernel_median_ns":{"host_dispatch":6625,"host_synchronized":120500,"mps_ev'
    'ent":254500},"label":"host-specific Apple GPU (Metal) measurement","largest_'
    'point":{"index":3,"k":256,"m":256,"n":256},"load_average_after":[9.87109375,'
    '13.619140625,12.64697265625],"load_average_before":[9.87109375,13.619140625,'
    '12.64697265625],"sides":{"matmul_x_columns":{"first_size_dominant_index":3,"'
    'intercept_ns":{"host_dispatch":6833.333439798532,"host_synchronized":136368.'
    '84268615942,"mps_event":86907.28881138389},"largest_dispatch_to_synchronized'
    '_ratio":0.01733390631714654,"largest_event_fraction_of_fma_ceiling":0.014516'
    '283404657692,"largest_event_gflops_per_second":67.03619676270922,"largest_ev'
    'ent_regime":"size_dependent_dominant","smallest_shape_median_ns":{"host_disp'
    'atch":6541,"host_synchronized":118167,"mps_event":72125}},"matmul_x_rows":{"'
    'first_size_dominant_index":3,"intercept_ns":{"host_dispatch":6619.9118621020'
    '61,"host_synchronized":125839.98753673444,"mps_event":74458.82188934032},"la'
    'rgest_dispatch_to_synchronized_ratio":0.0063981764344071745,"largest_event_f'
    'raction_of_fma_ceiling":0.0063613715960089325,"largest_event_gflops_per_seco'
    'nd":29.37681403036925,"largest_event_regime":"size_dependent_dominant","smal'
    'lest_shape_median_ns":{"host_dispatch":5542,"host_synchronized":112583,"mps_'
    'event":64417}},"torch_matmul":{"first_size_dominant_index":null,"intercept_n'
    's":{"host_dispatch":11655.984378535759,"host_synchronized":145222.8241109161'
    ',"mps_event":94642.00129028098},"largest_dispatch_to_synchronized_ratio":0.1'
    '964737808158486,"largest_event_fraction_of_fma_ceiling":0.04033870660893363,'
    '"largest_event_gflops_per_second":186.2841471200555,"largest_event_regime":"'
    'fixed_overhead_dominant","smallest_shape_median_ns":{"host_dispatch":8875,"h'
    'ost_synchronized":122750,"mps_event":71917}}},"sweep_points":4,"timed_sample'
    's":117},"timer":"time.perf_counter_ns"}'
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
        9.87,
        13.62,
        12.65,
    ]
    assert report["environment"]["torch_version"] == "2.15.0"
    assert report["evidence_boundary"]["hardware_counters_used"] is False


def test_every_recorded_event_sample_is_its_raw_ms_rounded_to_ns():
    report = recorded()
    benchmark = report["benchmark"]
    timings = [benchmark["empty"]["timing"]] + [
        timing for point in benchmark["points"] for timing in point["sides"].values()
    ]

    for timing in timings:
        event = timing["clocks"]["mps_event"]
        assert event["domain"] == "mps_event_stream"
        assert event["samples_ns"] == [round(ms * 1e6) for ms in event["elapsed_ms"]]
        for clock in ("host_synchronized", "host_dispatch"):
            assert timing["clocks"][clock]["domain"] == "host_perf_counter_ns"


def test_the_recorded_dispatch_cost_is_a_small_part_of_a_synchronized_launch():
    report = recorded()

    for point in report["analysis"]["points"]:
        for side in point["sides"].values():
            assert side["dispatch_to_synchronized_ratio"] < 0.5
            assert side["host_minus_event"]["cross_domain"] is True


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    limits = recorded()["shaders"]["limits"]

    runtime = timing_runtime()

    for name, entry in limits.items():
        assert runtime.limits[name].to_mapping() == entry, name
