"""One measured run of the pinned sweep on the development host's Apple GPU.

``RECORDED_REPORT_JSON`` is the canonical report that ``run`` wrote for
``config/day-74-metal-launch-sweep.json`` on 2026-11-17 at 11:18 IST, under
the GPU lock that the lab's timed runs share. The host was a shared Darwin
arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14, NumPy 1.26.4 and
PyTorch 2.15.0; ``uptime`` gave load averages of 6.58, 9.01 and 8.39 before
the run and 7.23, 9.03 and 8.41 after it. The timings describe that moment on
that host. They are here so that a host without torch or MPS can verify a
genuine Apple GPU (Metal) measurement, and so that the guide's figures are
tied to the report they come from.
"""

import json
from pathlib import Path

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.metal_launch_sweep_config import load_config
from ai_journey.metal_launch_sweep_verify import verify_report

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-74-metal-launch-sweep.json"
RECORDED_LOAD_AVERAGES = ([6.58, 9.01, 8.39], [7.23, 9.03, 8.41])
RECORDED_REPORT_JSON = (
    '{"admission":{"matmul":[{"admitted":true,"columns_per_simd_group":8,"group_s'
    'ize":[8,8],"idle_lanes":0,"reason":null,"rows_per_simd_group":4,"simd_groups'
    '":2,"threads_per_group":64},{"admitted":true,"columns_per_simd_group":16,"gr'
    'oup_size":[16,16],"idle_lanes":0,"reason":null,"rows_per_simd_group":2,"simd'
    '_groups":8,"threads_per_group":256},{"admitted":true,"columns_per_simd_group'
    '":32,"group_size":[32,32],"idle_lanes":0,"reason":null,"rows_per_simd_group"'
    ':1,"simd_groups":32,"threads_per_group":1024},{"admitted":true,"columns_per_'
    'simd_group":32,"group_size":[32,8],"idle_lanes":0,"reason":null,"rows_per_si'
    'md_group":1,"simd_groups":8,"threads_per_group":256},{"admitted":true,"colum'
    'ns_per_simd_group":8,"group_size":[8,32],"idle_lanes":0,"reason":null,"rows_'
    'per_simd_group":4,"simd_groups":8,"threads_per_group":256},{"admitted":true,'
    '"columns_per_simd_group":32,"group_size":[64,4],"idle_lanes":0,"reason":null'
    ',"rows_per_simd_group":1,"simd_groups":8,"threads_per_group":256},{"admitted'
    '":true,"columns_per_simd_group":4,"group_size":[4,64],"idle_lanes":0,"reason'
    '":null,"rows_per_simd_group":8,"simd_groups":8,"threads_per_group":256}],"ve'
    'ctor":[{"admitted":true,"columns_per_simd_group":8,"group_size":[8],"idle_la'
    'nes":24,"reason":null,"rows_per_simd_group":1,"simd_groups":1,"threads_per_g'
    'roup":8},{"admitted":true,"columns_per_simd_group":16,"group_size":[16],"idl'
    'e_lanes":16,"reason":null,"rows_per_simd_group":1,"simd_groups":1,"threads_p'
    'er_group":16},{"admitted":true,"columns_per_simd_group":32,"group_size":[32]'
    ',"idle_lanes":0,"reason":null,"rows_per_simd_group":1,"simd_groups":1,"threa'
    'ds_per_group":32},{"admitted":true,"columns_per_simd_group":32,"group_size":'
    '[48],"idle_lanes":16,"reason":null,"rows_per_simd_group":1,"simd_groups":2,"'
    'threads_per_group":48},{"admitted":true,"columns_per_simd_group":32,"group_s'
    'ize":[64],"idle_lanes":0,"reason":null,"rows_per_simd_group":1,"simd_groups"'
    ':2,"threads_per_group":64},{"admitted":true,"columns_per_simd_group":32,"gro'
    'up_size":[128],"idle_lanes":0,"reason":null,"rows_per_simd_group":1,"simd_gr'
    'oups":4,"threads_per_group":128},{"admitted":true,"columns_per_simd_group":3'
    '2,"group_size":[256],"idle_lanes":0,"reason":null,"rows_per_simd_group":1,"s'
    'imd_groups":8,"threads_per_group":256},{"admitted":true,"columns_per_simd_gr'
    'oup":32,"group_size":[512],"idle_lanes":0,"reason":null,"rows_per_simd_group'
    '":1,"simd_groups":16,"threads_per_group":512},{"admitted":true,"columns_per_'
    'simd_group":32,"group_size":[1024],"idle_lanes":0,"reason":null,"rows_per_si'
    'md_group":1,"simd_groups":32,"threads_per_group":1024}]},"backend":"torch.mp'
    's.compile_shader","benchmark":{"device":"mps","matmul":[{"accuracy":{"max_ab'
    's_error":3.956804206950437e-05,"max_bound_ratio":0.016386094614486563,"outpu'
    't_sha256":"6286034c0a65c24c5136b2d578e58791d4d9bf25ecd68ebdf1adc446ea2e5009"'
    ',"samples":[[0,0,4.636427402496338],[0,63,11.148138999938965],[0,127,-0.1289'
    "5090878009796],[0,191,-0.11529389023780823],[0,255,2.784524440765381],[63,0,"
    "13.87157917022705],[63,63,-6.395634651184082],[63,127,13.000932693481445],[6"
    "3,191,-27.595739364624023],[63,255,-6.0813069343566895],[127,0,1.01394164562"
    "22534],[127,63,-0.8993123173713684],[127,127,-4.004425525665283],[127,191,-1"
    ".4134793281555176],[127,255,-32.126949310302734],[191,0,16.460243225097656],"
    "[191,63,37.227073669433594],[191,127,24.577476501464844],[191,191,-8.1214275"
    "36010742],[191,255,20.47744369506836],[255,0,18.318180084228516],[255,63,41."
    "67854690551758],[255,127,5.481174468994141],[255,191,4.672799110412598],[255"
    ',255,16.85531234741211]]},"calls_per_sample":32,"k":256,"kernel":"matmul_nai'
    've","m":256,"n":256,"output_sha256":"6286034c0a65c24c5136b2d578e58791d4d9bf2'
    '5ecd68ebdf1adc446ea2e5009","ranking":[[32,32],[8,8],[16,16],[8,32],[32,8],[6'
    '4,4],[4,64]],"repeats":21,"separated":true,"shapes":[{"best_gflops_per_secon'
    'd":314.8726362378863,"columns_per_simd_group":8,"group_size":[8,8],"idle_lan'
    'es":0,"iqr_ns_per_call":1877.625,"median_gflops_per_second":304.309084591626'
    '1,"median_ns_per_call":110264.3125,"rank":2,"relative_to_fastest":1.03188876'
    '9342844,"rows_per_simd_group":4,"samples_ns":[3517167,3567458,3552750,349266'
    "6,3546625,3488833,3569583,3540792,3436375,3548417,3410083,3471209,3524209,35"
    '14416,3721292,3556084,3546750,3517292,3528458,3553584,3481375],"simd_groups"'
    ':2,"threads_per_group":64,"timing":{"count":21,"iqr_ns":60084,"max_ns":37212'
    '92,"median_ns":3528458,"min_ns":3410083,"p25_ns":3492666,"p75_ns":3552750}},'
    '{"best_gflops_per_second":308.26224438441676,"columns_per_simd_group":16,"gr'
    'oup_size":[16,16],"idle_lanes":0,"iqr_ns_per_call":2282.5625,"median_gflops_'
    'per_second":299.82662949858553,"median_ns_per_call":111912.78125,"rank":3,"r'
    'elative_to_fastest":1.0473156681387499,"rows_per_simd_group":2,"samples_ns":'
    "[3570750,3566459,3612459,3584583,3585500,3647125,3577375,3534875,3483209,348"
    "7542,3540708,3523167,3581209,3621708,3640708,3645375,3625958,3551625,3542833"
    ',3613500,3615875],"simd_groups":8,"threads_per_group":256,"timing":{"count":'
    '21,"iqr_ns":73042,"max_ns":3647125,"median_ns":3581209,"min_ns":3483209,"p25'
    '_ns":3542833,"p75_ns":3615875}},{"best_gflops_per_second":323.9732117749432,'
    '"columns_per_simd_group":32,"group_size":[32,32],"idle_lanes":0,"iqr_ns_per_'
    'call":2915.34375,"median_gflops_per_second":314.0131267991006,"median_ns_per'
    '_call":106856.78125,"rank":1,"relative_to_fastest":1.0,"rows_per_simd_group"'
    ':1,"samples_ns":[3333542,3363166,3419417,3420167,3400875,3389667,3462375,334'
    "4209,3335334,3320833,3339583,3314292,3434209,3426625,3540083,3469250,3410709"
    ',3437500,3434834,3442000,3463333],"simd_groups":32,"threads_per_group":1024,'
    '"timing":{"count":21,"iqr_ns":93291,"max_ns":3540083,"median_ns":3419417,"mi'
    'n_ns":3314292,"p25_ns":3344209,"p75_ns":3437500}},{"best_gflops_per_second":'
    '305.1559380742588,"columns_per_simd_group":32,"group_size":[32,8],"idle_lane'
    's":0,"iqr_ns_per_call":2667.96875,"median_gflops_per_second":294.60297615062'
    '717,"median_ns_per_call":113897.125,"rank":5,"relative_to_fastest":1.0658857'
    '928120495,"rows_per_simd_group":1,"samples_ns":[3518666,3600791,3654875,3645'
    "833,3665000,3655750,3568500,3566792,3536167,3563000,3545250,3527625,3631875,"
    '3648084,3644708,4078459,3648375,3674417,3560500,3648167,3646042],"simd_group'
    's":8,"threads_per_group":256,"timing":{"count":21,"iqr_ns":85375,"max_ns":40'
    '78459,"median_ns":3644708,"min_ns":3518666,"p25_ns":3563000,"p75_ns":3648375'
    '}},{"best_gflops_per_second":306.3057163999683,"columns_per_simd_group":8,"g'
    'roup_size":[8,32],"idle_lanes":0,"iqr_ns_per_call":3256.5,"median_gflops_per'
    '_second":297.5969020307696,"median_ns_per_call":112751.28125,"rank":4,"relat'
    'ive_to_fastest":1.0551626198267132,"rows_per_simd_group":4,"samples_ns":[351'
    "8250,3512708,3598292,3659917,3667584,3686625,3535792,3586166,3505458,3536667"
    ",3546458,3507375,3641417,3626792,3608041,3614459,3640875,3631041,3656792,362"
    '1500,3540958],"simd_groups":8,"threads_per_group":256,"timing":{"count":21,"'
    'iqr_ns":104208,"max_ns":3686625,"median_ns":3608041,"min_ns":3505458,"p25_ns'
    '":3536667,"p75_ns":3640875}},{"best_gflops_per_second":298.1028779952453,"co'
    'lumns_per_simd_group":32,"group_size":[64,4],"idle_lanes":0,"iqr_ns_per_call'
    '":4169.25,"median_gflops_per_second":288.1014017329951,"median_ns_per_call":'
    '116467.4375,"rank":6,"relative_to_fastest":1.0899396008149926,"rows_per_simd'
    '_group":1,"samples_ns":[3723333,3731167,3778042,3762167,3791750,3803625,3679'
    "583,3640875,3640459,3620250,3601917,3617708,3636833,3740416,3709458,3989125,"
    '3782375,3689750,3774291,3732250,3726958],"simd_groups":8,"threads_per_group"'
    ':256,"timing":{"count":21,"iqr_ns":133416,"max_ns":3989125,"median_ns":37269'
    '58,"min_ns":3601917,"p25_ns":3640875,"p75_ns":3774291}},{"best_gflops_per_se'
    'cond":282.3190604350126,"columns_per_simd_group":4,"group_size":[4,64],"idle'
    '_lanes":0,"iqr_ns_per_call":2339.84375,"median_gflops_per_second":273.002584'
    '48652113,"median_ns_per_call":122908.84375,"rank":7,"relative_to_fastest":1.'
    '1502203445792074,"rows_per_simd_group":8,"samples_ns":[3883875,3870917,39510'
    "84,3945792,3959000,3963000,3970666,3824166,3877167,3803292,3840583,3890541,3"
    '945417,3856541,3916750,3933083,3952500,3825125,3938125,3944458,3937750],"sim'
    'd_groups":8,"threads_per_group":256,"timing":{"count":21,"iqr_ns":74875,"max'
    '_ns":3970666,"median_ns":3933083,"min_ns":3803292,"p25_ns":3870917,"p75_ns":'
    '3945792}}],"spread":1.1502203445792074,"warmup":2,"work_per_call":33554432,"'
    'work_per_sample":1073741824},{"accuracy":{"max_abs_error":0.0001980125097418'
    '295,"max_bound_ratio":0.004916602824688285,"output_sha256":"a2a2520b226aea68'
    '5a61ba5714684e5609b5e85db09890cd3c3f716f7a47b767","samples":[[0,0,3.26426815'
    "9866333],[0,255,-53.85161590576172],[0,511,-42.28669738769531],[0,767,-13.87"
    "7323150634766],[0,1023,40.75189208984375],[255,0,9.422743797302246],[255,255"
    ",25.113054275512695],[255,511,-34.37384033203125],[255,767,-73.6190032958984"
    "4],[255,1023,9.831048965454102],[511,0,14.732046127319336],[511,255,54.63386"
    "535644531],[511,511,-4.527405261993408],[511,767,64.62089538574219],[511,102"
    "3,14.95177173614502],[767,0,25.841215133666992],[767,255,29.513904571533203]"
    ",[767,511,2.056400775909424],[767,767,-34.39276885986328],[767,1023,-16.7402"
    "34375],[1023,0,56.98127365112305],[1023,255,65.2945556640625],[1023,511,-7.3"
    "566412925720215],[1023,767,10.29438591003418],[1023,1023,-5.402690887451172]"
    ']},"calls_per_sample":1,"k":1024,"kernel":"matmul_naive","m":1024,"n":1024,"'
    'output_sha256":"a2a2520b226aea685a61ba5714684e5609b5e85db09890cd3c3f716f7a47'
    'b767","ranking":[[8,8],[16,16],[8,32],[64,4],[4,64],[32,8],[32,32]],"repeats'
    '":21,"separated":true,"shapes":[{"best_gflops_per_second":394.60989973984033'
    ',"columns_per_simd_group":8,"group_size":[8,8],"idle_lanes":0,"iqr_ns_per_ca'
    'll":46042.0,"median_gflops_per_second":389.82254053686097,"median_ns_per_cal'
    'l":5508875.0,"rank":1,"relative_to_fastest":1.0,"rows_per_simd_group":4,"sam'
    'ples_ns":[6205000,11049292,7410250,5534667,5491375,5442042,5745042,5531334,5'
    "528334,5479583,5450375,5522167,5488625,5587500,5484083,5489917,5472541,54986"
    '25,5490541,5511209,5508875],"simd_groups":2,"threads_per_group":64,"timing":'
    '{"count":21,"iqr_ns":46042,"max_ns":11049292,"median_ns":5508875,"min_ns":54'
    '42042,"p25_ns":5488625,"p75_ns":5534667}},{"best_gflops_per_second":392.7305'
    '5205443075,"columns_per_simd_group":16,"group_size":[16,16],"idle_lanes":0,"'
    'iqr_ns_per_call":31000.0,"median_gflops_per_second":386.9252760973852,"media'
    'n_ns_per_call":5550125.0,"rank":2,"relative_to_fastest":1.0074879172244786,"'
    'rows_per_simd_group":2,"samples_ns":[5650834,9038000,5572083,5542542,5550125'
    ",5468084,5485041,6051458,5544834,5529291,5541083,5570916,5504791,5547292,554"
    '6875,5583125,5575625,5571208,5539459,5557084,5559792],"simd_groups":8,"threa'
    'ds_per_group":256,"timing":{"count":21,"iqr_ns":31000,"max_ns":9038000,"medi'
    'an_ns":5550125,"min_ns":5468084,"p25_ns":5541083,"p75_ns":5572083}},{"best_g'
    'flops_per_second":379.7075739639739,"columns_per_simd_group":32,"group_size"'
    ':[32,32],"idle_lanes":0,"iqr_ns_per_call":62958.0,"median_gflops_per_second"'
    ':374.6128066745957,"median_ns_per_call":5732542.0,"rank":7,"relative_to_fast'
    'est":1.040601211681151,"rows_per_simd_group":1,"samples_ns":[7607959,6148375'
    ",9308042,5726750,5717625,5692208,5655625,6032667,5700500,5735875,5735000,593"
    "3750,5663000,5705250,5709125,5772083,5750417,5709500,5732542,5732292,5744875"
    '],"simd_groups":32,"threads_per_group":1024,"timing":{"count":21,"iqr_ns":62'
    '958,"max_ns":9308042,"median_ns":5732542,"min_ns":5655625,"p25_ns":5709125,"'
    'p75_ns":5772083}},{"best_gflops_per_second":391.89746022963266,"columns_per_'
    'simd_group":32,"group_size":[32,8],"idle_lanes":0,"iqr_ns_per_call":56917.0,'
    '"median_gflops_per_second":383.8018905894456,"median_ns_per_call":5595292.0,'
    '"rank":6,"relative_to_fastest":1.0156868689160672,"rows_per_simd_group":1,"s'
    'amples_ns":[7082209,5761417,6268167,5894875,5582833,5595292,5538000,5479708,'
    "5593042,5606334,5565417,5535292,5563458,5545000,5552625,5649709,5602250,5602"
    '375,5613167,5602667,5556250],"simd_groups":8,"threads_per_group":256,"timing'
    '":{"count":21,"iqr_ns":56917,"max_ns":7082209,"median_ns":5595292,"min_ns":5'
    '479708,"p25_ns":5556250,"p75_ns":5613167}},{"best_gflops_per_second":391.269'
    '681698096,"columns_per_simd_group":8,"group_size":[8,32],"idle_lanes":0,"iqr'
    '_ns_per_call":55584.0,"median_gflops_per_second":385.99799764822524,"median_'
    'ns_per_call":5563458.0,"rank":3,"relative_to_fastest":1.0099081935966963,"ro'
    'ws_per_simd_group":4,"samples_ns":[5824708,6011333,9588750,5589458,5535375,5'
    "547541,5488500,5542666,5503667,5588625,5603583,5563458,5590959,5518875,55269"
    '59,5588916,5728625,5523417,5568792,5539500,5552291],"simd_groups":8,"threads'
    '_per_group":256,"timing":{"count":21,"iqr_ns":55584,"max_ns":9588750,"median'
    '_ns":5563458,"min_ns":5488500,"p25_ns":5535375,"p75_ns":5590959}},{"best_gfl'
    'ops_per_second":391.31124760086567,"columns_per_simd_group":32,"group_size":'
    '[64,4],"idle_lanes":0,"iqr_ns_per_call":50333.0,"median_gflops_per_second":3'
    '85.339933058901,"median_ns_per_call":5572959.0,"rank":4,"relative_to_fastest'
    '":1.011632865149418,"rows_per_simd_group":1,"samples_ns":[7545959,10059084,7'
    "228833,5586792,5572959,5499417,5487917,5593833,5535458,5503083,5596708,57202"
    "08,5551667,5507583,5565541,5546375,5700208,5554500,5562542,5574375,5590708],"
    '"simd_groups":8,"threads_per_group":256,"timing":{"count":21,"iqr_ns":50333,'
    '"max_ns":10059084,"median_ns":5572959,"min_ns":5487917,"p25_ns":5546375,"p75'
    '_ns":5596708}},{"best_gflops_per_second":389.9906742940162,"columns_per_simd'
    '_group":4,"group_size":[4,64],"idle_lanes":0,"iqr_ns_per_call":195542.0,"med'
    'ian_gflops_per_second":384.07076041224207,"median_ns_per_call":5591375.0,"ra'
    'nk":5,"relative_to_fastest":1.0149758344489574,"rows_per_simd_group":8,"samp'
    'les_ns":[7431125,11876833,8305833,5621041,5591375,5506500,5804625,5536917,58'
    "88250,5518208,5560792,5746542,5551000,5574875,5595416,5583792,5526708,557058"
    '4,5598167,5595708,5545917],"simd_groups":8,"threads_per_group":256,"timing":'
    '{"count":21,"iqr_ns":195542,"max_ns":11876833,"median_ns":5591375,"min_ns":5'
    '506500,"p25_ns":5551000,"p75_ns":5746542}}],"spread":1.040601211681151,"warm'
    'up":2,"work_per_call":2147483648,"work_per_sample":2147483648},{"accuracy":{'
    '"max_abs_error":0.0005203014048760224,"max_bound_ratio":0.003147573335406229'
    '6,"output_sha256":"ccc104e6696e8b2e6e66a2c5c51ca68a6d1a2c1d3ab690940f4aaae7c'
    '49f4f51","samples":[[0,0,82.02619171142578],[0,511,-9.010750770568848],[0,10'
    "23,49.07962417602539],[0,1535,-78.34883880615234],[0,2047,-33.64238739013672"
    "],[511,0,28.992815017700195],[511,511,23.828046798706055],[511,1023,32.13307"
    "571411133],[511,1535,-16.254108428955078],[511,2047,-91.67420959472656],[102"
    "3,0,-45.06179428100586],[1023,511,40.896358489990234],[1023,1023,8.208472251"
    "89209],[1023,1535,54.076255798339844],[1023,2047,2.8755314350128174],[1535,0"
    ",20.282732009887695],[1535,511,39.25163650512695],[1535,1023,-13.51064109802"
    "2461],[1535,1535,-25.679685592651367],[1535,2047,-0.9204555153846741],[2047,"
    "0,62.925777435302734],[2047,511,43.834266662597656],[2047,1023,9.72481155395"
    '5078],[2047,1535,-5.50368595123291],[2047,2047,-62.44763946533203]]},"calls_'
    'per_sample":1,"k":2048,"kernel":"matmul_naive","m":2048,"n":2048,"output_sha'
    '256":"ccc104e6696e8b2e6e66a2c5c51ca68a6d1a2c1d3ab690940f4aaae7c49f4f51","ran'
    'king":[[8,8],[8,32],[16,16],[64,4],[32,8],[4,64],[32,32]],"repeats":21,"sepa'
    'rated":true,"shapes":[{"best_gflops_per_second":413.5301545813003,"columns_p'
    'er_simd_group":8,"group_size":[8,8],"idle_lanes":0,"iqr_ns_per_call":320082.'
    '0,"median_gflops_per_second":412.19443537113926,"median_ns_per_call":4167904'
    '2.0,"rank":1,"relative_to_fastest":1.0,"rows_per_simd_group":4,"samples_ns":'
    "[41620125,41731666,41679042,41851667,42409917,41654583,42145083,41581125,420"
    "19375,41651000,41550667,41727250,41966416,41652000,41544417,41987042,4164633"
    '4,41624083,42081000,41652209,41723959],"simd_groups":2,"threads_per_group":6'
    '4,"timing":{"count":21,"iqr_ns":320082,"max_ns":42409917,"median_ns":4167904'
    '2,"min_ns":41544417,"p25_ns":41646334,"p75_ns":41966416}},{"best_gflops_per_'
    'second":412.4141934649713,"columns_per_simd_group":16,"group_size":[16,16],"'
    'idle_lanes":0,"iqr_ns_per_call":314708.0,"median_gflops_per_second":411.0102'
    '5880326443,"median_ns_per_call":41799125.0,"rank":3,"relative_to_fastest":1.'
    '0028811362794758,"rows_per_simd_group":2,"samples_ns":[41982500,41790709,420'
    "36125,41707250,41781750,42270458,41716000,41687959,41724958,41656833,4169579"
    "2,42009292,42030708,41723125,41804959,42141667,41704916,42129916,42550417,41"
    '799709,41799125],"simd_groups":8,"threads_per_group":256,"timing":{"count":2'
    '1,"iqr_ns":314708,"max_ns":42550417,"median_ns":41799125,"min_ns":41656833,"'
    'p25_ns":41716000,"p75_ns":42030708}},{"best_gflops_per_second":405.493214120'
    '8824,"columns_per_simd_group":32,"group_size":[32,32],"idle_lanes":0,"iqr_ns'
    '_per_call":399584.0,"median_gflops_per_second":403.25767039837774,"median_ns'
    '_per_call":42602709.0,"rank":7,"relative_to_fastest":1.0221614258792224,"row'
    's_per_simd_group":1,"samples_ns":[42367834,42559667,42602709,42383583,427613'
    "75,43956542,42591750,42403041,42543375,43277375,42902667,43048125,42667250,4"
    '2475917,43032542,44085375,42565833,42488583,42503083,42895625,42609625],"sim'
    'd_groups":32,"threads_per_group":1024,"timing":{"count":21,"iqr_ns":399584,"'
    'max_ns":44085375,"median_ns":42602709,"min_ns":42367834,"p25_ns":42503083,"p'
    '75_ns":42902667}},{"best_gflops_per_second":410.772772137898,"columns_per_si'
    'md_group":32,"group_size":[32,8],"idle_lanes":0,"iqr_ns_per_call":392375.0,"'
    'median_gflops_per_second":409.2726836466731,"median_ns_per_call":41976584.0,'
    '"rank":5,"relative_to_fastest":1.0071388876932441,"rows_per_simd_group":1,"s'
    'amples_ns":[42384500,41940083,42180292,41832167,41948000,41887250,41971250,4'
    "2814625,42660875,41935167,41860292,42614083,41925250,42317625,41823291,42824"
    '209,41873208,42214750,42142333,42220500,41976584],"simd_groups":8,"threads_p'
    'er_group":256,"timing":{"count":21,"iqr_ns":392375,"max_ns":42824209,"median'
    '_ns":41976584,"min_ns":41823291,"p25_ns":41925250,"p75_ns":42317625}},{"best'
    '_gflops_per_second":412.76719272540095,"columns_per_simd_group":8,"group_siz'
    'e":[8,32],"idle_lanes":0,"iqr_ns_per_call":117459.0,"median_gflops_per_secon'
    'd":411.442548292419,"median_ns_per_call":41755208.0,"rank":2,"relative_to_fa'
    'stest":1.0018274412353336,"rows_per_simd_group":4,"samples_ns":[41810875,417'
    "55208,41910625,41714166,41752250,41716584,41684208,41804458,42510083,4181175"
    "0,42073500,41682625,42068000,41682917,41694291,41770000,42104125,41662125,41"
    '727250,41787584,41621208],"simd_groups":8,"threads_per_group":256,"timing":{'
    '"count":21,"iqr_ns":117459,"max_ns":42510083,"median_ns":41755208,"min_ns":4'
    '1621208,"p25_ns":41694291,"p75_ns":41811750}},{"best_gflops_per_second":412.'
    '73661024672805,"columns_per_simd_group":32,"group_size":[64,4],"idle_lanes":'
    '0,"iqr_ns_per_call":115458.0,"median_gflops_per_second":410.91072391875,"med'
    'ian_ns_per_call":41809250.0,"rank":4,"relative_to_fastest":1.003124064127961'
    '5,"rows_per_simd_group":1,"samples_ns":[41756709,43115541,41782166,41809250,'
    "42222292,42953708,42258834,41784959,42130542,41863041,41624292,41754500,4187"
    "0292,41666292,41842458,41779375,41831583,41730291,41872167,41787333,41722375"
    '],"simd_groups":8,"threads_per_group":256,"timing":{"count":21,"iqr_ns":1154'
    '58,"max_ns":43115541,"median_ns":41809250,"min_ns":41624292,"p25_ns":4175670'
    '9,"p75_ns":41872167}},{"best_gflops_per_second":409.88867987641214,"columns_'
    'per_simd_group":4,"group_size":[4,64],"idle_lanes":0,"iqr_ns_per_call":11691'
    '7.0,"median_gflops_per_second":408.52614306627885,"median_ns_per_call":42053'
    '292.0,"rank":6,"relative_to_fastest":1.0089793330662447,"rows_per_simd_group'
    '":8,"samples_ns":[42507375,42064667,41971334,41993250,42167833,41973834,4202'
    "1000,42051250,42102375,42564542,41980041,42110167,42008666,42246708,42079417"
    ',42027375,41991792,41913500,42380709,42053292,42054917],"simd_groups":8,"thr'
    'eads_per_group":256,"timing":{"count":21,"iqr_ns":116917,"max_ns":42564542,"'
    'median_ns":42053292,"min_ns":41913500,"p25_ns":41993250,"p75_ns":42110167}}]'
    ',"spread":1.0221614258792224,"warmup":2,"work_per_call":17179869184,"work_pe'
    'r_sample":17179869184}],"synchronization":"torch.mps.synchronize() before th'
    "e first and before the second clock reading of every sample; untimed warmups"
    ' for each shape first","turn_order":"repeat r runs every admitted shape once'
    ", starting with shape r mod S; one batch size and one set of input buffers p"
    'er point","vector":[{"calls_per_sample":11,"kernel":"vector_add","length":10'
    '48576,"output_sha256":"11ff3cdd940c9af5820806e2d80390fa774c99f108ba6ed955839'
    'a2cbf4dda21","ranking":[[64],[512],[128],[1024],[256],[48],[32],[16],[8]],"r'
    'epeats":21,"separated":false,"shapes":[{"best_gb_per_second":124.55525939257'
    '593,"columns_per_simd_group":8,"group_size":[8],"idle_lanes":24,"iqr_ns_per_'
    'call":56946.90909090909,"median_gb_per_second":82.48424771251256,"median_ns_'
    'per_call":152549.27272727274,"rank":9,"relative_to_fastest":1.40627865074376'
    '7,"rows_per_simd_group":1,"samples_ns":[1865167,1494375,1583834,1270709,1688'
    "167,1931041,1296917,1304625,2605500,2702000,3187125,1883166,1541125,1701875,"
    '1273084,1511625,2169875,1272458,1939250,1111250,1678042],"simd_groups":1,"th'
    'reads_per_group":8,"timing":{"count":21,"iqr_ns":626416,"max_ns":3187125,"me'
    'dian_ns":1678042,"min_ns":1111250,"p25_ns":1304625,"p75_ns":1931041}},{"best'
    '_gb_per_second":115.15144093178037,"columns_per_simd_group":16,"group_size":'
    '[16],"idle_lanes":16,"iqr_ns_per_call":48109.818181818184,"median_gb_per_sec'
    'ond":98.03419707835326,"median_ns_per_call":128352.27272727272,"rank":8,"rel'
    'ative_to_fastest":1.183218101822753,"rows_per_simd_group":1,"samples_ns":[18'
    "45875,1316667,1286375,1296709,1426917,1379042,1321250,1202000,2419166,299462"
    "5,3098375,1264458,1962708,1363416,1790042,1411875,1951542,1265792,1334375,14"
    '32500,1423750],"simd_groups":1,"threads_per_group":16,"timing":{"count":21,"'
    'iqr_ns":529208,"max_ns":3098375,"median_ns":1411875,"min_ns":1202000,"p25_ns'
    '":1316667,"p75_ns":1845875}},{"best_gb_per_second":118.27136475341197,"colum'
    'ns_per_simd_group":32,"group_size":[32],"idle_lanes":0,"iqr_ns_per_call":441'
    '55.27272727273,"median_gb_per_second":101.00612916714404,"median_ns_per_call'
    '":124575.72727272728,"rank":7,"relative_to_fastest":1.1484039388225435,"rows'
    '_per_simd_group":1,"samples_ns":[1927334,1266333,1830084,1370333,1313667,136'
    "4750,1296416,1310834,4495375,1389166,1322125,1665584,1895584,1353000,1799375"
    ',1360000,1658083,1929584,1265792,1395833,1170292],"simd_groups":1,"threads_p'
    'er_group":32,"timing":{"count":21,"iqr_ns":485708,"max_ns":4495375,"median_n'
    's":1370333,"min_ns":1170292,"p25_ns":1313667,"p75_ns":1799375}},{"best_gb_pe'
    'r_second":124.1641910742319,"columns_per_simd_group":32,"group_size":[48],"i'
    'dle_lanes":16,"iqr_ns_per_call":33056.818181818184,"median_gb_per_second":10'
    '6.24604260218769,"median_ns_per_call":118431.81818181818,"rank":6,"relative_'
    'to_fastest":1.0917661847894407,"rows_per_simd_group":1,"samples_ns":[1114750'
    ",1504209,1210583,1128709,1225250,1197792,1165250,2141375,8218250,1701333,116"
    "0333,1242541,2020917,1574208,1361583,1281625,1549625,1855125,1364292,1270708"
    ',1302750],"simd_groups":2,"threads_per_group":48,"timing":{"count":21,"iqr_n'
    's":363625,"max_ns":8218250,"median_ns":1302750,"min_ns":1114750,"p25_ns":121'
    '0583,"p75_ns":1574208}},{"best_gb_per_second":127.36326846100759,"columns_pe'
    'r_simd_group":32,"group_size":[64],"idle_lanes":0,"iqr_ns_per_call":35102.27'
    '272727273,"median_gb_per_second":115.99583658076682,"median_ns_per_call":108'
    '477.27272727272,"rank":1,"relative_to_fastest":1.0,"rows_per_simd_group":1,"'
    'samples_ns":[1090000,1558708,1086750,1095042,1686500,1143917,1130042,1149333'
    ",1761917,1129375,1095916,1516167,1580500,1132208,1306416,1567167,1156375,133"
    '9084,1325500,1193250,1353583],"simd_groups":2,"threads_per_group":64,"timing'
    '":{"count":21,"iqr_ns":386125,"max_ns":1761917,"median_ns":1193250,"min_ns":'
    '1086750,"p25_ns":1130042,"p75_ns":1516167}},{"best_gb_per_second":131.648585'
    '89941744,"columns_per_simd_group":32,"group_size":[128],"idle_lanes":0,"iqr_'
    'ns_per_call":42079.63636363636,"median_gb_per_second":109.73829374958376,"me'
    'dian_ns_per_call":114662.90909090909,"rank":3,"relative_to_fastest":1.057022'
    '4177666039,"rows_per_simd_group":1,"samples_ns":[1541042,1238333,1161416,105'
    "1375,1814792,1272334,1137041,1197708,1115750,1638417,1208000,2140167,1665917"
    ',1231667,1630917,1860917,1076459,1434208,1261292,1168041,1419375],"simd_grou'
    'ps":4,"threads_per_group":128,"timing":{"count":21,"iqr_ns":462876,"max_ns":'
    '2140167,"median_ns":1261292,"min_ns":1051375,"p25_ns":1168041,"p75_ns":16309'
    '17}},{"best_gb_per_second":135.8701728071243,"columns_per_simd_group":32,"gr'
    'oup_size":[256],"idle_lanes":0,"iqr_ns_per_call":20640.090909090908,"median_'
    'gb_per_second":106.62457929706308,"median_ns_per_call":118011.36363636363,"r'
    'ank":5,"relative_to_fastest":1.0878902157971926,"rows_per_simd_group":1,"sam'
    'ples_ns":[1500875,1018708,1111959,1289333,3037042,1184583,1094959,2412542,20'
    "23791,1714333,1177500,1312833,1297375,1290375,1378333,1378708,1176750,130954"
    '2,1298125,1095792,1404541],"simd_groups":8,"threads_per_group":256,"timing":'
    '{"count":21,"iqr_ns":227041,"max_ns":3037042,"median_ns":1298125,"min_ns":10'
    '18708,"p25_ns":1177500,"p75_ns":1404541}},{"best_gb_per_second":131.55474111'
    '91636,"columns_per_simd_group":32,"group_size":[512],"idle_lanes":0,"iqr_ns_'
    'per_call":30825.727272727272,"median_gb_per_second":114.35079509854464,"medi'
    'an_ns_per_call":110037.81818181818,"rank":2,"relative_to_fastest":1.01438592'
    '08045254,"rows_per_simd_group":1,"samples_ns":[1575666,1052125,1055041,14465'
    "83,1243708,1303000,1107500,1099667,1154916,1134333,1061292,6079542,1088584,1"
    '496041,1210416,1386458,1536292,1191083,1313208,1196125,1500750],"simd_groups'
    '":16,"threads_per_group":512,"timing":{"count":21,"iqr_ns":339083,"max_ns":6'
    '079542,"median_ns":1210416,"min_ns":1052125,"p25_ns":1107500,"p75_ns":144658'
    '3}},{"best_gb_per_second":132.83304414587332,"columns_per_simd_group":32,"gr'
    'oup_size":[1024],"idle_lanes":0,"iqr_ns_per_call":37708.27272727273,"median_'
    'gb_per_second":109.29419930402071,"median_ns_per_call":115128.81818181818,"r'
    'ank":4,"relative_to_fastest":1.0613174104336895,"rows_per_simd_group":1,"sam'
    'ples_ns":[1178750,1095542,1549667,1115958,1266417,1258042,1157167,1149375,15'
    "93541,2386083,1856792,7440041,1042000,1525792,1186250,1614083,1413917,119604"
    '2,1320334,1178791,1994042],"simd_groups":32,"threads_per_group":1024,"timing'
    '":{"count":21,"iqr_ns":414791,"max_ns":7440041,"median_ns":1266417,"min_ns":'
    '1042000,"p25_ns":1178750,"p75_ns":1593541}}],"spread":1.406278650743767,"war'
    'mup":2,"work_per_call":12582912,"work_per_sample":138412032},{"calls_per_sam'
    'ple":1,"kernel":"vector_add","length":16777216,"output_sha256":"673bf914feff'
    '1adab1be74d8e1a3b84ee7fe504cd3dd6adf029c606634e5eaf4","ranking":[[256],[128]'
    ',[8],[64],[32],[16],[48],[1024],[512]],"repeats":21,"separated":false,"shape'
    's":[{"best_gb_per_second":118.7650491072561,"columns_per_simd_group":8,"grou'
    'p_size":[8],"idle_lanes":24,"iqr_ns_per_call":133208.0,"median_gb_per_second'
    '":112.59612022536383,"median_ns_per_call":1788042.0,"rank":3,"relative_to_fa'
    'stest":1.007181917115607,"rows_per_simd_group":1,"samples_ns":[1925292,18758'
    "75,1771083,1841666,1788042,1759375,1904291,1757084,1784541,1782833,1715542,1"
    "695167,2006917,1832666,1742458,1783459,1943292,1790958,2126125,2004041,17827"
    '08],"simd_groups":1,"threads_per_group":8,"timing":{"count":21,"iqr_ns":1332'
    '08,"max_ns":2126125,"median_ns":1788042,"min_ns":1695167,"p25_ns":1771083,"p'
    '75_ns":1904291}},{"best_gb_per_second":120.66922837635914,"columns_per_simd_'
    'group":16,"group_size":[16],"idle_lanes":16,"iqr_ns_per_call":149459.0,"medi'
    'an_gb_per_second":111.67234680084067,"median_ns_per_call":1802833.0,"rank":6'
    ',"relative_to_fastest":1.0155135042573278,"rows_per_simd_group":1,"samples_n'
    's":[2143417,1830958,1686541,1809750,1778291,1766833,1911417,1745625,1699417,'
    "1719250,1668417,1735333,1895084,1873292,1802833,1751583,1936167,1820792,1926"
    '583,1923833,1762916],"simd_groups":1,"threads_per_group":16,"timing":{"count'
    '":21,"iqr_ns":149459,"max_ns":2143417,"median_ns":1802833,"min_ns":1668417,"'
    'p25_ns":1745625,"p75_ns":1895084}},{"best_gb_per_second":119.77786728366233,'
    '"columns_per_simd_group":32,"group_size":[32],"idle_lanes":0,"iqr_ns_per_cal'
    'l":130999.0,"median_gb_per_second":111.71882316186324,"median_ns_per_call":1'
    '802083.0,"rank":5,"relative_to_fastest":1.015091038544645,"rows_per_simd_gro'
    'up":1,"samples_ns":[1751917,1879083,2153500,1869375,1825042,1682625,1798166,'
    "1730459,1701041,1691708,1680833,1691917,1802083,1825917,1752750,1802041,1861"
    '458,1832250,1855000,1890542,1985875],"simd_groups":1,"threads_per_group":32,'
    '"timing":{"count":21,"iqr_ns":130999,"max_ns":2153500,"median_ns":1802083,"m'
    'in_ns":1680833,"p25_ns":1730459,"p75_ns":1861458}},{"best_gb_per_second":119'
    '.8640125027908,"columns_per_simd_group":32,"group_size":[48],"idle_lanes":16'
    ',"iqr_ns_per_call":123167.0,"median_gb_per_second":111.60783978934239,"media'
    'n_ns_per_call":1803875.0,"rank":7,"relative_to_fastest":1.0161004499541484,"'
    'rows_per_simd_group":1,"samples_ns":[1801750,1854750,1900041,1685666,1875542'
    ",1767167,1829208,1705084,1679625,1705209,1820500,1709917,1697458,1822500,173"
    '9291,1706041,1803875,1820250,1852416,1817916,1863083],"simd_groups":2,"threa'
    'ds_per_group":48,"timing":{"count":21,"iqr_ns":123167,"max_ns":1900041,"medi'
    'an_ns":1803875,"min_ns":1679625,"p25_ns":1706041,"p75_ns":1829208}},{"best_g'
    'b_per_second":120.81412564449693,"columns_per_simd_group":32,"group_size":[6'
    '4],"idle_lanes":0,"iqr_ns_per_call":79459.0,"median_gb_per_second":112.21955'
    '338838221,"median_ns_per_call":1794042.0,"rank":4,"relative_to_fastest":1.01'
    '0561642817069,"rows_per_simd_group":1,"samples_ns":[1846584,1794000,1782917,'
    "1768292,1911209,1825750,1868625,1765083,1736959,1666416,1856458,1707042,1680"
    '791,2215917,1794042,1938417,1779375,1767125,1843792,1811250,1840708],"simd_g'
    'roups":2,"threads_per_group":64,"timing":{"count":21,"iqr_ns":79459,"max_ns"'
    ':2215917,"median_ns":1794042,"min_ns":1666416,"p25_ns":1767125,"p75_ns":1846'
    '584}},{"best_gb_per_second":119.69775226254163,"columns_per_simd_group":32,"'
    'group_size":[128],"idle_lanes":0,"iqr_ns_per_call":104250.0,"median_gb_per_s'
    'econd":112.75377000372995,"median_ns_per_call":1785542.0,"rank":2,"relative_'
    'to_fastest":1.005773698073331,"rows_per_simd_group":1,"samples_ns":[2177666,'
    "1793541,1689958,1785542,1836667,1814125,1757875,1697250,1729583,1712958,1681"
    "958,1700208,1721125,1754666,1813833,1847209,1910209,1856292,1823791,1825375,"
    '1760708],"simd_groups":4,"threads_per_group":128,"timing":{"count":21,"iqr_n'
    's":104250,"max_ns":2177666,"median_ns":1785542,"min_ns":1681958,"p25_ns":172'
    '1125,"p75_ns":1825375}},{"best_gb_per_second":120.16807699768708,"columns_pe'
    'r_simd_group":32,"group_size":[256],"idle_lanes":0,"iqr_ns_per_call":101666.'
    '0,"median_gb_per_second":113.4047762283613,"median_ns_per_call":1775292.0,"r'
    'ank":1,"relative_to_fastest":1.0,"rows_per_simd_group":1,"samples_ns":[17972'
    "09,1800958,1704417,1812250,1793166,1680125,1760583,1680417,1810333,1724125,1"
    "708667,1675375,1759708,1704333,1806750,1775292,1865250,1976417,1951209,19235"
    '00,1758250],"simd_groups":8,"threads_per_group":256,"timing":{"count":21,"iq'
    'r_ns":101666,"max_ns":1976417,"median_ns":1775292,"min_ns":1675375,"p25_ns":'
    '1708667,"p75_ns":1810333}},{"best_gb_per_second":121.3511122684055,"columns_'
    'per_simd_group":32,"group_size":[512],"idle_lanes":0,"iqr_ns_per_call":12208'
    '2.0,"median_gb_per_second":109.50345518538884,"median_ns_per_call":1838541.0'
    ',"rank":9,"relative_to_fastest":1.0356273784819625,"rows_per_simd_group":1,"'
    'samples_ns":[1838541,1770167,1763209,1842875,1872917,1794625,1896833,1852250'
    ",1895083,1659042,1760208,1755000,1749625,1778500,1830208,1766334,1840958,212"
    '0875,1888416,1902917,1907500],"simd_groups":16,"threads_per_group":512,"timi'
    'ng":{"count":21,"iqr_ns":122082,"max_ns":2120875,"median_ns":1838541,"min_ns'
    '":1659042,"p25_ns":1766334,"p75_ns":1888416}},{"best_gb_per_second":118.6629'
    '8799086421,"columns_per_simd_group":32,"group_size":[1024],"idle_lanes":0,"i'
    'qr_ns_per_call":117458.0,"median_gb_per_second":110.60630510012861,"median_n'
    's_per_call":1820209.0,"rank":8,"relative_to_fastest":1.025301189888762,"rows'
    '_per_simd_group":1,"samples_ns":[1904250,1893167,1710000,1827417,1820209,174'
    "4875,1971750,1798583,1742000,1713750,1696625,1783333,1771000,1848500,1797500"
    ',1797917,1888458,1837250,1897417,1928334,1824875],"simd_groups":32,"threads_'
    'per_group":1024,"timing":{"count":21,"iqr_ns":117458,"max_ns":1971750,"media'
    'n_ns":1820209,"min_ns":1696625,"p25_ns":1771000,"p75_ns":1888458}}],"spread"'
    ':1.0356273784819625,"warmup":2,"work_per_call":201326592,"work_per_sample":2'
    '01326592}]},"config":{"matmul":{"correctness_cases":[[1,1,1],[37,53,29],[64,'
    '64,16]],"group_shapes":[[8,8],[16,16],[32,32],[32,8],[8,32],[64,4],[4,64]],"'
    'sizes":[[256,256,256],[1024,1024,1024],[2048,2048,2048]],"target_flops_per_s'
    'ample":1073741824},"name":"day-74-metal-launch-sweep","schema_version":1,"se'
    'ed":74,"timing":{"max_runtime_seconds":300,"repeats":21,"warmup":2},"vector"'
    ':{"correctness_length":65537,"group_sizes":[8,16,32,48,64,128,256,512,1024],'
    '"lengths":[1048576,16777216],"target_bytes_per_sample":134217728}},"config_s'
    'ha256":"48f0bf4286df929a0f0f8891e816840895bcc7fcecc84fe8a200d9c8fc39effa","c'
    'orrectness":{"bound":"(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)","ma'
    'tmul":[{"a_sha256":"38d3fad09c9e8b0f722cf78a399dc136ddd5898b0e0fdea2d4f3b03b'
    'c1114431","b_sha256":"f5369643905c52296f253da35f30eafca14c508c3ca545301c4059'
    '083a0f0977","index":0,"k":1,"m":1,"max_abs_error":5.525784274595935e-10,"max'
    '_bound_ratio":0.20455428435808637,"n":1,"output_sha256":"510a2a81df417fc39c5'
    'bda8ee498fdb1275a4f7d788b84c82b270a29f23cac11","samples":[[0,0,-0.0453215986'
    '4902496]],"shapes":[{"group_size":[8,8],"guard_elements":63},{"group_size":['
    '16,16],"guard_elements":255},{"group_size":[32,32],"guard_elements":1023},{"'
    'group_size":[32,8],"guard_elements":255},{"group_size":[8,32],"guard_element'
    's":255},{"group_size":[64,4],"guard_elements":255},{"group_size":[4,64],"gua'
    'rd_elements":255}]},{"a_sha256":"75740505e70fcc8c57e5411bf1e6ba6820c7f329e9b'
    '765eeb79b37262d19b875","b_sha256":"8f6ba64f914b4664ffa0d6c19247de8b68db423d6'
    '86eb65ab79c1220b06a40cf","index":1,"k":29,"m":37,"max_abs_error":3.419684961'
    '869507e-06,"max_bound_ratio":0.09917879341335349,"n":53,"output_sha256":"67d'
    'eb2e180be89dabbd2644d34f819cd3096b0c530f9f515e3f955cc18b2df80","samples":[[0'
    ",0,3.450803756713867],[0,13,-6.009391784667969],[0,26,0.5910936594009399],[0"
    ",39,-6.014091491699219],[0,52,-5.844898223876953],[9,0,-6.429825782775879],["
    "9,13,4.9415130615234375],[9,26,-9.623539924621582],[9,39,3.1546566486358643]"
    ",[9,52,0.5323816537857056],[18,0,5.728922367095947],[18,13,-0.24113707244396"
    "21],[18,26,-0.8759801387786865],[18,39,-5.611300945281982],[18,52,-11.443386"
    "07788086],[27,0,4.45838737487793],[27,13,-5.332376003265381],[27,26,-3.71408"
    "1287384033],[27,39,6.85807991027832],[27,52,1.6774375438690186],[36,0,0.3103"
    "0702590942383],[36,13,1.5998064279556274],[36,26,13.608980178833008],[36,39,"
    '2.8321471214294434],[36,52,10.708967208862305]],"shapes":[{"group_size":[8,8'
    '],"guard_elements":279},{"group_size":[16,16],"guard_elements":1111},{"group'
    '_size":[32,32],"guard_elements":2135},{"group_size":[32,8],"guard_elements":'
    '599},{"group_size":[8,32],"guard_elements":1623},{"group_size":[64,4],"guard'
    '_elements":599},{"group_size":[4,64],"guard_elements":1623}]},{"a_sha256":"9'
    '506a649b0fd60eb7264bd52f89796716b703ad37a322ec555bdc50d9bdd2e9a","b_sha256":'
    '"3ebace29fac57fcb945add70d001f8f4d3d29f0365f497f180c67618a8436182","index":2'
    ',"k":16,"m":64,"max_abs_error":2.5923081388157243e-06,"max_bound_ratio":0.22'
    '059864890504105,"n":64,"output_sha256":"cb7f1b4bf65caac2f52567a7d664cc098cba'
    'a37ea2b299720aad1f04baeb9fcf","samples":[[0,0,-1.7367225885391235],[0,15,-1.'
    "0695340633392334],[0,31,-1.4025617837905884],[0,47,-6.215281009674072],[0,63"
    ",-1.1875684261322021],[15,0,7.669331073760986],[15,15,0.47956153750419617],["
    "15,31,-2.6291286945343018],[15,47,-4.101755619049072],[15,63,2.1354300975799"
    "56],[31,0,7.841847896575928],[31,15,-6.971167087554932],[31,31,-0.2184890508"
    "6517334],[31,47,-1.354910135269165],[31,63,7.705273151397705],[47,0,-4.72947"
    "0252990723],[47,15,2.1963140964508057],[47,31,-3.2975664138793945],[47,47,3."
    "2282700538635254],[47,63,-9.736368179321289],[63,0,0.3971271216869354],[63,1"
    "5,-3.2987871170043945],[63,31,-1.1310657262802124],[63,47,5.078048229217529]"
    ',[63,63,1.3328386545181274]],"shapes":[{"group_size":[8,8],"guard_elements":'
    '0},{"group_size":[16,16],"guard_elements":0},{"group_size":[32,32],"guard_el'
    'ements":0},{"group_size":[32,8],"guard_elements":0},{"group_size":[8,32],"gu'
    'ard_elements":0},{"group_size":[64,4],"guard_elements":0},{"group_size":[4,6'
    '4],"guard_elements":0}]}],"reference":"float64 matmul on the CPU, rebuilt fr'
    'om the seed","vector_add":[{"a_sha256":"a7cec0111634ab4ef9b057787900ad5b5901'
    '1b8f0a84f61c6f5bcc83e5e5fc6d","b_sha256":"cd1ada83346fda740710defe9edaaa5899'
    'f4b55a4ecedff227da88eb9ae328b9","index":0,"launch":{"active_threads":65537,"'
    'extent":[65537],"group_size":[8],"groups":[8193],"idle_lanes_per_threadgroup'
    '":24,"launched_threads":65544,"padding_threads":7,"simd_groups_per_threadgro'
    'up":1,"threads":[65544],"threads_per_group":8},"length":65537,"outputs":[{"b'
    'uffer_sha256":"e9239d30205505be2111270433f8da35342596017effd661867b4f42ac29f'
    '2fb","device":"mps","guard_elements":7,"implementation":"metal_kernel"},{"bu'
    'ffer_sha256":"8df656c8c8e1213909af30c2dc87f8db090ba03bbf7a3bbd90801044a2b37c'
    '52","device":"mps","guard_elements":0,"implementation":"torch"},{"buffer_sha'
    '256":"8df656c8c8e1213909af30c2dc87f8db090ba03bbf7a3bbd90801044a2b37c52","dev'
    'ice":"cpu","guard_elements":0,"implementation":"torch"}],"reference_sha256":'
    '"8df656c8c8e1213909af30c2dc87f8db090ba03bbf7a3bbd90801044a2b37c52"},{"a_sha2'
    '56":"6c2cf292bc3ba3cf581598c708f57692fa283ca1a3d28088dff9e7745bac69dd","b_sh'
    'a256":"93bedfddcc125c773577bfeee14e5e03f8f34f0ba606c072c2b6666e49a88058","in'
    'dex":1,"launch":{"active_threads":65537,"extent":[65537],"group_size":[16],"'
    'groups":[4097],"idle_lanes_per_threadgroup":16,"launched_threads":65552,"pad'
    'ding_threads":15,"simd_groups_per_threadgroup":1,"threads":[65552],"threads_'
    'per_group":16},"length":65537,"outputs":[{"buffer_sha256":"f386d1d018e4e7b7b'
    'f5aabcfb064a0ee5d774e954b56d95087664c58a4dd0aab","device":"mps","guard_eleme'
    'nts":15,"implementation":"metal_kernel"},{"buffer_sha256":"2f1ea67cc22e8fb96'
    '22bcfa183224b450ccf0ec9013e416fc7b5bdf00cda369e","device":"mps","guard_eleme'
    'nts":0,"implementation":"torch"},{"buffer_sha256":"2f1ea67cc22e8fb9622bcfa18'
    '3224b450ccf0ec9013e416fc7b5bdf00cda369e","device":"cpu","guard_elements":0,"'
    'implementation":"torch"}],"reference_sha256":"2f1ea67cc22e8fb9622bcfa183224b'
    '450ccf0ec9013e416fc7b5bdf00cda369e"},{"a_sha256":"a12560db510cac354c746f6670'
    'e69d673618463d7ac8b41ae6e47ce64e44b1b8","b_sha256":"d5ea6ec732a7166eb4c2b06b'
    '5614f567e0fe51c425a9f0761101fb349561194a","index":2,"launch":{"active_thread'
    's":65537,"extent":[65537],"group_size":[32],"groups":[2049],"idle_lanes_per_'
    'threadgroup":0,"launched_threads":65568,"padding_threads":31,"simd_groups_pe'
    'r_threadgroup":1,"threads":[65568],"threads_per_group":32},"length":65537,"o'
    'utputs":[{"buffer_sha256":"723548e4534a10062c8bc7bd9f6a5ad536065a6b14a69e498'
    '0a90da7add616b2","device":"mps","guard_elements":31,"implementation":"metal_'
    'kernel"},{"buffer_sha256":"04fdb64ab3b106bdcbc5de80f2d88807eeef2e2700f5ac83a'
    '01bbd97eba03cc8","device":"mps","guard_elements":0,"implementation":"torch"}'
    ',{"buffer_sha256":"04fdb64ab3b106bdcbc5de80f2d88807eeef2e2700f5ac83a01bbd97e'
    'ba03cc8","device":"cpu","guard_elements":0,"implementation":"torch"}],"refer'
    'ence_sha256":"04fdb64ab3b106bdcbc5de80f2d88807eeef2e2700f5ac83a01bbd97eba03c'
    'c8"},{"a_sha256":"65e30f3b3a4bafea427962b4044b803fb883e864cf3b0eb30f3dc4f13b'
    '28b907","b_sha256":"d2ee04ae8f2e7e2d532903ac7f528af27a61f67fcdbf7094bf3f1dd1'
    '52665c41","index":3,"launch":{"active_threads":65537,"extent":[65537],"group'
    '_size":[48],"groups":[1366],"idle_lanes_per_threadgroup":16,"launched_thread'
    's":65568,"padding_threads":31,"simd_groups_per_threadgroup":2,"threads":[655'
    '68],"threads_per_group":48},"length":65537,"outputs":[{"buffer_sha256":"22cc'
    '3ceebb1be00cf04f19e29482cfc556ee05167d3f2702277d2532509d711e","device":"mps"'
    ',"guard_elements":31,"implementation":"metal_kernel"},{"buffer_sha256":"2098'
    '417214ea1a3761a94238aa156ee7b0ad4de2c02d1e389cf8715b86389526","device":"mps"'
    ',"guard_elements":0,"implementation":"torch"},{"buffer_sha256":"2098417214ea'
    '1a3761a94238aa156ee7b0ad4de2c02d1e389cf8715b86389526","device":"cpu","guard_'
    'elements":0,"implementation":"torch"}],"reference_sha256":"2098417214ea1a376'
    '1a94238aa156ee7b0ad4de2c02d1e389cf8715b86389526"},{"a_sha256":"aef9fc80e7a8f'
    '3ccc814bd4b6ea345367d1a14e549cd9a6dbd1d2a9f1e7eed79","b_sha256":"9dceefd07e5'
    '45c9c2a5a53a2c903450f78c9fb4b99eda0d34dc1000da940e475","index":4,"launch":{"'
    'active_threads":65537,"extent":[65537],"group_size":[64],"groups":[1025],"id'
    'le_lanes_per_threadgroup":0,"launched_threads":65600,"padding_threads":63,"s'
    'imd_groups_per_threadgroup":2,"threads":[65600],"threads_per_group":64},"len'
    'gth":65537,"outputs":[{"buffer_sha256":"9cf2caaea586b3461192ba15a4aa7a7d247e'
    'c8ef144a8fb0122a87f41ee7e48a","device":"mps","guard_elements":63,"implementa'
    'tion":"metal_kernel"},{"buffer_sha256":"bead1dd85f640469f1dfbd04f62bda65a2d0'
    '36604a1c209b2a8d687bf5120880","device":"mps","guard_elements":0,"implementat'
    'ion":"torch"},{"buffer_sha256":"bead1dd85f640469f1dfbd04f62bda65a2d036604a1c'
    '209b2a8d687bf5120880","device":"cpu","guard_elements":0,"implementation":"to'
    'rch"}],"reference_sha256":"bead1dd85f640469f1dfbd04f62bda65a2d036604a1c209b2'
    'a8d687bf5120880"},{"a_sha256":"9ce8be1678646fb2e4404448bb76fb49131396f8740a8'
    'b9d4e91822ac2492634","b_sha256":"6388a47e5088850bf7edfeabf4729fa4f6c24afe778'
    '7978b59fb6f18c3724c79","index":5,"launch":{"active_threads":65537,"extent":['
    '65537],"group_size":[128],"groups":[513],"idle_lanes_per_threadgroup":0,"lau'
    'nched_threads":65664,"padding_threads":127,"simd_groups_per_threadgroup":4,"'
    'threads":[65664],"threads_per_group":128},"length":65537,"outputs":[{"buffer'
    '_sha256":"c32a1ca512f06bc43c9a3a205446e191fb36f3bdb39e3d8c1a858ef585119bd7",'
    '"device":"mps","guard_elements":127,"implementation":"metal_kernel"},{"buffe'
    'r_sha256":"d382482ba66e427a0c662a0e48ddb4109569ff94bc7dc2d44041a3b104efa927"'
    ',"device":"mps","guard_elements":0,"implementation":"torch"},{"buffer_sha256'
    '":"d382482ba66e427a0c662a0e48ddb4109569ff94bc7dc2d44041a3b104efa927","device'
    '":"cpu","guard_elements":0,"implementation":"torch"}],"reference_sha256":"d3'
    '82482ba66e427a0c662a0e48ddb4109569ff94bc7dc2d44041a3b104efa927"},{"a_sha256"'
    ':"1b2e3f8665d604062bed945f6baf5feb48e7c5c6644fba584704922ea2da17f8","b_sha25'
    '6":"a026bd2c8c7b5a52a25010a435c59a07a00ffbf0df8ace9b25b85f31f4cc6a9a","index'
    '":6,"launch":{"active_threads":65537,"extent":[65537],"group_size":[256],"gr'
    'oups":[257],"idle_lanes_per_threadgroup":0,"launched_threads":65792,"padding'
    '_threads":255,"simd_groups_per_threadgroup":8,"threads":[65792],"threads_per'
    '_group":256},"length":65537,"outputs":[{"buffer_sha256":"ea2b09366a9e45745b3'
    'dfca56600e4a9cda11119fb7156cdbc5c188d632e4f82","device":"mps","guard_element'
    's":255,"implementation":"metal_kernel"},{"buffer_sha256":"8ddec500c971a05590'
    'ce973375b07d8c13b58428c495dbe8ec3c885bbcabbc9c","device":"mps","guard_elemen'
    'ts":0,"implementation":"torch"},{"buffer_sha256":"8ddec500c971a05590ce973375'
    'b07d8c13b58428c495dbe8ec3c885bbcabbc9c","device":"cpu","guard_elements":0,"i'
    'mplementation":"torch"}],"reference_sha256":"8ddec500c971a05590ce973375b07d8'
    'c13b58428c495dbe8ec3c885bbcabbc9c"},{"a_sha256":"8ad607a8c79c7e21594921f684a'
    '16dc69d28d96c879816bdca9800d590a5e1f7","b_sha256":"11a432e09036c0f2fe518ca97'
    'a552a56db8fdf95cee3b8c5d76648ffb5976340","index":7,"launch":{"active_threads'
    '":65537,"extent":[65537],"group_size":[512],"groups":[129],"idle_lanes_per_t'
    'hreadgroup":0,"launched_threads":66048,"padding_threads":511,"simd_groups_pe'
    'r_threadgroup":16,"threads":[66048],"threads_per_group":512},"length":65537,'
    '"outputs":[{"buffer_sha256":"754bf701fc69b23b8ceec9e3c700f869ae7590fb1eeaa8a'
    '39ae4f36c755341f4","device":"mps","guard_elements":511,"implementation":"met'
    'al_kernel"},{"buffer_sha256":"30b5f9e895ea11e0927c0f17c63f55fd5ac29dcf25c219'
    'db8ec81e54b7ffa1da","device":"mps","guard_elements":0,"implementation":"torc'
    'h"},{"buffer_sha256":"30b5f9e895ea11e0927c0f17c63f55fd5ac29dcf25c219db8ec81e'
    '54b7ffa1da","device":"cpu","guard_elements":0,"implementation":"torch"}],"re'
    'ference_sha256":"30b5f9e895ea11e0927c0f17c63f55fd5ac29dcf25c219db8ec81e54b7f'
    'fa1da"},{"a_sha256":"78f171867a68fee1fc4ea119a2148a9d2076497cdb1d047c6c614e3'
    '8cdd277a1","b_sha256":"fca20b0fc430adb980d28b38d8dd1e2a28d1de9dbf666167216d8'
    '578fd27ebb1","index":8,"launch":{"active_threads":65537,"extent":[65537],"gr'
    'oup_size":[1024],"groups":[65],"idle_lanes_per_threadgroup":0,"launched_thre'
    'ads":66560,"padding_threads":1023,"simd_groups_per_threadgroup":32,"threads"'
    ':[66560],"threads_per_group":1024},"length":65537,"outputs":[{"buffer_sha256'
    '":"128a52bf367518c46f23ec7eb2054929b4f5cb45c67fd9a0900f000f985e3147","device'
    '":"mps","guard_elements":1023,"implementation":"metal_kernel"},{"buffer_sha2'
    '56":"c2322d209b72dfc45766bcb524b3027969124529b5f1d364c33947629db6e890","devi'
    'ce":"mps","guard_elements":0,"implementation":"torch"},{"buffer_sha256":"c23'
    '22d209b72dfc45766bcb524b3027969124529b5f1d364c33947629db6e890","device":"cpu'
    '","guard_elements":0,"implementation":"torch"}],"reference_sha256":"c2322d20'
    '9b72dfc45766bcb524b3027969124529b5f1d364c33947629db6e890"}]},"dispatch":{"gr'
    'id":"whole threadgroups on every axis: ceil(extent / group) groups per axis"'
    ',"requires_whole_simd_groups":false,"rules":["exceeds_max_threads_per_thread'
    'group","not_whole_simd_groups"]},"environment":{"macos_version":"27.2","nump'
    'y_version":"1.26.4","platform_machine":"arm64","platform_system":"Darwin","p'
    'ython_implementation":"CPython","python_version":"3.11.14","torch_version":"'
    '2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_host":true,"cuda_kernel_'
    'executed":false,"distributed_claim":false,"host_specific":true,"nvidia_gpu_u'
    'sed":false,"portable_performance_claim":false,"production_claim":false,"sche'
    'duler_internals_claim":false,"statement":"These are wall-clock timings on th'
    "e Apple GPU of the host that made this report. The kernels came from torch.m"
    "ps.compile_shader. The timings are host-specific Apple GPU (Metal) measureme"
    "nts. No CUDA kernel ran, and no NVIDIA GPU was used. A best rate is the high"
    "est median rate that this run measured, not a vendor peak. The layout probe "
    "shows which SIMD group and lane ran each thread, not how the GPU schedules S"
    "IMD groups. These timings do not measure or predict CUDA, NVIDIA, distribute"
    'd or production performance.","vendor_peak_claim":false},"kind":"metal_launc'
    'h_sweep","label":"host-specific Apple GPU (Metal) measurement","mapping":{"c'
    'ase":[13,19,3],"checks":[{"exact":true,"group_size":[8,8],"launched_threads"'
    ':384,"order_seed":74,"out_of_bounds":0,"output_sha256":"3138724e38bb14b526a4'
    'ab6609258ea9c38b7ab971597ee77a4e72e566172b9b","read_write_conflicts":0,"read'
    's":1482,"write_conflicts":0,"writer_threads":247,"writes":247},{"exact":true'
    ',"group_size":[16,16],"launched_threads":512,"order_seed":74,"out_of_bounds"'
    ':0,"output_sha256":"3138724e38bb14b526a4ab6609258ea9c38b7ab971597ee77a4e72e5'
    '66172b9b","read_write_conflicts":0,"reads":1482,"write_conflicts":0,"writer_'
    'threads":247,"writes":247},{"exact":true,"group_size":[32,32],"launched_thre'
    'ads":1024,"order_seed":74,"out_of_bounds":0,"output_sha256":"3138724e38bb14b'
    '526a4ab6609258ea9c38b7ab971597ee77a4e72e566172b9b","read_write_conflicts":0,'
    '"reads":1482,"write_conflicts":0,"writer_threads":247,"writes":247},{"exact"'
    ':true,"group_size":[32,8],"launched_threads":512,"order_seed":74,"out_of_bou'
    'nds":0,"output_sha256":"3138724e38bb14b526a4ab6609258ea9c38b7ab971597ee77a4e'
    '72e566172b9b","read_write_conflicts":0,"reads":1482,"write_conflicts":0,"wri'
    'ter_threads":247,"writes":247},{"exact":true,"group_size":[8,32],"launched_t'
    'hreads":768,"order_seed":74,"out_of_bounds":0,"output_sha256":"3138724e38bb1'
    '4b526a4ab6609258ea9c38b7ab971597ee77a4e72e566172b9b","read_write_conflicts":'
    '0,"reads":1482,"write_conflicts":0,"writer_threads":247,"writes":247},{"exac'
    't":true,"group_size":[64,4],"launched_threads":1024,"order_seed":74,"out_of_'
    'bounds":0,"output_sha256":"3138724e38bb14b526a4ab6609258ea9c38b7ab971597ee77'
    'a4e72e566172b9b","read_write_conflicts":0,"reads":1482,"write_conflicts":0,"'
    'writer_threads":247,"writes":247},{"exact":true,"group_size":[4,64],"launche'
    'd_threads":1280,"order_seed":74,"out_of_bounds":0,"output_sha256":"3138724e3'
    '8bb14b526a4ab6609258ea9c38b7ab971597ee77a4e72e566172b9b","read_write_conflic'
    'ts":0,"reads":1482,"write_conflicts":0,"writer_threads":247,"writes":247}]},'
    '"measured":true,"measurement_scope":"apple_gpu_metal_this_host","percentile_'
    'method":"nearest rank: sorted samples[ceil(p * n / 100) - 1] for p in 25, 50'
    ' and 75; median is p50 and IQR is p75 - p25; no interpolation","pipelines":{'
    '"matmul_naive":{"max_threads_per_threadgroup":1024,"static_threadgroup_memor'
    'y_bytes":0,"thread_execution_width":32},"simd_layout_probe":{"max_threads_pe'
    'r_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_wi'
    'dth":32},"vector_add":{"max_threads_per_threadgroup":1024,"static_threadgrou'
    'p_memory_bytes":0,"thread_execution_width":32}},"planned_work":{"benchmark_p'
    'oints":5,"calls":7958,"correctness_launches":30,"layout_probes":16,"mapping_'
    'checks":7,"matmul_flops":3284576239616,"timed_samples":819,"vector_bytes":70'
    '325895168},"report_sha256":"0d5ea2c1065145ceef6c82cd5191cc71767b9496151984b2'
    '1559055d391e73b7","run":{"elapsed_ns":10806261084,"load_average_after":[7.23'
    ',9.03,8.41],"load_average_before":[6.58,9.01,8.39],"max_runtime_ns":30000000'
    '0000,"timed_ns":8222642637},"schema_version":1,"shaders":{"day_73_kernel":{"'
    'axes":1,"buffers":["out","a","b","n"],"bytes_per_item":12,"name":"vector_add'
    '","scalar_buffers":[3]},"day_73_source_sha256":"07892c3a5dcfb94cc8fb84652c64'
    '482aec1225f5ad68e16360e952be2e16cf76","kernels":[{"axes":2,"buffers":["c","a'
    '","b","m","n","k"],"name":"matmul_naive","scalar_buffers":[3,4,5]},{"axes":2'
    ',"buffers":["out"],"name":"simd_layout_probe","scalar_buffers":[]}],"source"'
    ':"#include <metal_stdlib>\\nusing namespace metal;\\n\\n// One thread per el'
    "ement of C = A B; every matrix is row-major float32.\\n// x runs over the n "
    "columns and y over the m rows, as in the naive CUDA\\n// matmul: col = block"
    "Idx.x * blockDim.x + threadIdx.x. The grid is whole\\n// threadgroups, so th"
    "e guard stops the padding threads on both axes.\\nkernel void matmul_naive(d"
    "evice float* c [[buffer(0)]],\\n                         device const float*"
    " a [[buffer(1)]],\\n                         device const float* b [[buffer("
    "2)]],\\n                         constant uint& m [[buffer(3)]],\\n         "
    "                constant uint& n [[buffer(4)]],\\n                         c"
    "onstant uint& k [[buffer(5)]],\\n                         uint2 position [[t"
    "hread_position_in_grid]]) {\\n    uint column = position.x;\\n    uint row ="
    " position.y;\\n    if (row < m && column < n) {\\n        float sum = 0.0f;"
    "\\n        for (uint i = 0; i < k; ++i) {\\n            sum += a[row * k + i"
    "] * b[i * n + column];\\n        }\\n        c[row * n + column] = sum;\\n  "
    "  }\\n}\\n\\n// Run as one threadgroup. At its linear index, each thread rec"
    "ords its SIMD\\n// group, its lane, its x and y position and the SIMD width "
    "it sees.\\nkernel void simd_layout_probe(device uint* out [[buffer(0)]],\\n "
    "                             uint linear [[thread_index_in_threadgroup]],\\n"
    "                              uint2 position [[thread_position_in_threadgrou"
    "p]],\\n                              uint simd_group [[simdgroup_index_in_th"
    "readgroup]],\\n                              uint lane [[thread_index_in_sim"
    "dgroup]],\\n                              uint width [[threads_per_simdgroup"
    "]]) {\\n    out[5 * linear] = simd_group;\\n    out[5 * linear + 1] = lane;"
    "\\n    out[5 * linear + 2] = position.x;\\n    out[5 * linear + 3] = positio"
    'n.y;\\n    out[5 * linear + 4] = width;\\n}\\n","source_sha256":"016bda2dedd'
    '2ed17ac8fb2d480558d087b5da3411adf9025a5ef53786305b6ef"},"simd_layout":[{"col'
    'umns_per_simd_group":8,"group_size":[8,8],"idle_lanes":0,"layout":"linear_x_'
    'fastest","observed_sha256":"4ea1ff3ae89a19a0770ed8a9b7cc07b7b9bfb53b00393baa'
    'd700abc6604414c9","rows_per_simd_group":4,"simd_groups":2,"simd_width":32},{'
    '"columns_per_simd_group":16,"group_size":[16,16],"idle_lanes":0,"layout":"li'
    'near_x_fastest","observed_sha256":"0d546f85f600db2fb6dacd98f9846f59e732aa1d3'
    '710ee1dd16e19238c0e8383","rows_per_simd_group":2,"simd_groups":8,"simd_width'
    '":32},{"columns_per_simd_group":32,"group_size":[32,32],"idle_lanes":0,"layo'
    'ut":"linear_x_fastest","observed_sha256":"50baf2b050825244caddc791d7719ff07b'
    'e61ec41bb3d72f9bd7ad83be8f1e74","rows_per_simd_group":1,"simd_groups":32,"si'
    'md_width":32},{"columns_per_simd_group":32,"group_size":[32,8],"idle_lanes":'
    '0,"layout":"linear_x_fastest","observed_sha256":"c10f4b843b616d1df2e5b6cc34f'
    '6f71484a60161efca22cc61956bb1b47d0d99","rows_per_simd_group":1,"simd_groups"'
    ':8,"simd_width":32},{"columns_per_simd_group":8,"group_size":[8,32],"idle_la'
    'nes":0,"layout":"linear_x_fastest","observed_sha256":"e59c6b660ff616d36f5f4a'
    '026d552af2a17c8495b569c3ed919156c00cb1df79","rows_per_simd_group":4,"simd_gr'
    'oups":8,"simd_width":32},{"columns_per_simd_group":32,"group_size":[64,4],"i'
    'dle_lanes":0,"layout":"linear_x_fastest","observed_sha256":"8190e48398aea40d'
    'b132e2463b9df4a04071e57f1a016001cac7be1f2eec7408","rows_per_simd_group":1,"s'
    'imd_groups":8,"simd_width":32},{"columns_per_simd_group":4,"group_size":[4,6'
    '4],"idle_lanes":0,"layout":"linear_x_fastest","observed_sha256":"b42a9779a62'
    '24ff9a7ea978cd3615beaf6f51a13f5cc718dabd26a0d0709fd36","rows_per_simd_group"'
    ':8,"simd_groups":8,"simd_width":32},{"columns_per_simd_group":8,"group_size"'
    ':[8],"idle_lanes":24,"layout":"linear_x_fastest","observed_sha256":"2ec21440'
    '9ee5887a1682f8887727a11bcae39839f06cb467a2a7b42ce67835ff","rows_per_simd_gro'
    'up":1,"simd_groups":1,"simd_width":32},{"columns_per_simd_group":16,"group_s'
    'ize":[16],"idle_lanes":16,"layout":"linear_x_fastest","observed_sha256":"2eb'
    'bb059a8cb09f520f303eb285130580c64a25f4cdcdf3dbd351cad1280a3e1","rows_per_sim'
    'd_group":1,"simd_groups":1,"simd_width":32},{"columns_per_simd_group":32,"gr'
    'oup_size":[32],"idle_lanes":0,"layout":"linear_x_fastest","observed_sha256":'
    '"9cee3855aefd64dcb41611415d4546bb0fba28121cde966e9c393693aa10beed","rows_per'
    '_simd_group":1,"simd_groups":1,"simd_width":32},{"columns_per_simd_group":32'
    ',"group_size":[48],"idle_lanes":16,"layout":"linear_x_fastest","observed_sha'
    '256":"b029a1db5fc5bace9f0138741c96bfe86857d72fe02ca46e16fc44bc569b6167","row'
    's_per_simd_group":1,"simd_groups":2,"simd_width":32},{"columns_per_simd_grou'
    'p":32,"group_size":[64],"idle_lanes":0,"layout":"linear_x_fastest","observed'
    '_sha256":"395a7d51280022000b90c0779555a5f67240fde1957924de931e3b69e10db36b",'
    '"rows_per_simd_group":1,"simd_groups":2,"simd_width":32},{"columns_per_simd_'
    'group":32,"group_size":[128],"idle_lanes":0,"layout":"linear_x_fastest","obs'
    'erved_sha256":"a3c3e07bfeb881cc0f5cace566f6ad3e20170a5e80a092e5750a134904a8b'
    '107","rows_per_simd_group":1,"simd_groups":4,"simd_width":32},{"columns_per_'
    'simd_group":32,"group_size":[256],"idle_lanes":0,"layout":"linear_x_fastest"'
    ',"observed_sha256":"13a4b3ea4a638a8e7cfcd7e5cd505df1f2508b5046bee553163d1792'
    'bc187997","rows_per_simd_group":1,"simd_groups":8,"simd_width":32},{"columns'
    '_per_simd_group":32,"group_size":[512],"idle_lanes":0,"layout":"linear_x_fas'
    'test","observed_sha256":"7bfea7290b0a263f98d7e4b09a3547acf8a8085a405ca75e809'
    '449d125504680","rows_per_simd_group":1,"simd_groups":16,"simd_width":32},{"c'
    'olumns_per_simd_group":32,"group_size":[1024],"idle_lanes":0,"layout":"linea'
    'r_x_fastest","observed_sha256":"286f502604a11ef36fa4149109915c557239b1db423c'
    '0948ef41f71232f6f37a","rows_per_simd_group":1,"simd_groups":32,"simd_width":'
    '32}],"summary":{"admitted_shapes":16,"benchmark_points":5,"best_matmul_gflop'
    's_per_second":412.19443537113926,"best_vector_gb_per_second":115.99583658076'
    '682,"correctness_cases":12,"label":"host-specific Apple GPU (Metal) measurem'
    'ent","layout_probes":16,"mapping_checks":7,"matmul":[{"best_median_gflops_pe'
    'r_second":314.0131267991006,"fastest":[32,32],"k":256,"m":256,"n":256,"separ'
    'ated":true,"slowest":[4,64],"spread":1.1502203445792074},{"best_median_gflop'
    's_per_second":389.82254053686097,"fastest":[8,8],"k":1024,"m":1024,"n":1024,'
    '"separated":true,"slowest":[32,32],"spread":1.040601211681151},{"best_median'
    '_gflops_per_second":412.19443537113926,"fastest":[8,8],"k":2048,"m":2048,"n"'
    ':2048,"separated":true,"slowest":[32,32],"spread":1.0221614258792224}],"matm'
    'ul_max_bound_ratio":0.22059864890504105,"rejected_shapes":0,"timed_samples":'
    '819,"vector":[{"best_median_gb_per_second":115.99583658076682,"fastest":[64]'
    ',"length":1048576,"separated":false,"slowest":[8],"spread":1.406278650743767'
    '},{"best_median_gb_per_second":113.4047762283613,"fastest":[256],"length":16'
    '777216,"separated":false,"slowest":[512],"spread":1.0356273784819625}]},"tim'
    'er":"time.perf_counter_ns"}'
)


def recorded_report():
    return json.loads(RECORDED_REPORT_JSON)


def test_the_recorded_run_verifies_on_any_host_as_a_measurement():
    report = recorded_report()

    summary = verify_report(report, config=load_config(PINNED))

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert (summary["admitted_shapes"], summary["rejected_shapes"]) == (16, 0)
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (12, 5)
    assert summary["timed_samples"] == 819


def test_the_recorded_run_states_its_host_load_and_pipeline_limits():
    report = recorded_report()

    run = report["run"]
    assert (run["load_average_before"], run["load_average_after"]) == (
        RECORDED_LOAD_AVERAGES
    )
    assert report["environment"] == {
        "macos_version": "27.2",
        "numpy_version": "1.26.4",
        "platform_machine": "arm64",
        "platform_system": "Darwin",
        "python_implementation": "CPython",
        "python_version": "3.11.14",
        "torch_version": "2.15.0",
    }
    for limits in report["pipelines"].values():
        assert limits == {
            "max_threads_per_threadgroup": 1024,
            "thread_execution_width": 32,
            "static_threadgroup_memory_bytes": 0,
        }
    assert report["dispatch"]["requires_whole_simd_groups"] is False
    assert {item["layout"] for item in report["simd_layout"]} == {"linear_x_fastest"}
    assert run["elapsed_ns"] < 60 * 10**9


def test_every_matmul_shape_gave_one_result_within_the_bound():
    report = recorded_report()

    for case in report["correctness"]["matmul"]:
        assert len(case["shapes"]) == 7
        assert case["max_bound_ratio"] < 1
    for point in report["benchmark"]["matmul"]:
        assert point["accuracy"]["max_bound_ratio"] < 1
        assert point["output_sha256"] == point["accuracy"]["output_sha256"]
    for check in report["mapping"]["checks"]:
        assert (check["out_of_bounds"], check["write_conflicts"]) == (0, 0)
