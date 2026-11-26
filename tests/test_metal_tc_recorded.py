"""One real Day 82 report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``metal_tc_cli run``
wrote in a fresh process on 2026-11-26 at 18:15 IST under the shared GPU
lock, with torch 2.15.0 on macOS 27.2.1 (arm64). The ``uptime`` command
printed load averages of 4.80, 4.87 and 5.22 before and after the run, from
other jobs. The run is short: five correctness cases, two sizes of 64 and
256, seven repeats and few calls for each sample, so the cost of each
synchronize call sets most of its times, and 8 of its 18 predictions missed.
Its ceilings come from the fresh Day 74 gpu-ceilings report of 2026-11-24,
107.71 GB/s and 4665.5 GFLOP/s, made on this host with the same
interpreter, libraries and platform. The numbers describe that moment and
are here only to be verified. A host without torch or MPS can still check a
genuine Apple GPU (Metal) report. The reference adds the products in k
order, not through the BLAS of the host, so another host gives the same
digests.
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
from ai_journey.metal_tc_cli import EXIT_OK, EXIT_REJECTED, main
from ai_journey.metal_tc_config import config_from_mapping
from ai_journey.metal_tc_plan import plan_mapping, plan_sha256
from ai_journey.metal_tc_source import KERNELS
from ai_journey.metal_tc_verify import MetalTcVerificationError, verify_report
from test_kernel_metal_runtime import requires_metal
from test_metal_tc_runtime import tc_runtime

ROOT = Path(__file__).resolve().parents[1]
CEILING_SHA256 = "cd75564918285983fe22b4106b5084b44ef42ed9313ca2a8ef19fa164e68fd86"
RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "metal-tc-recorded",
    "seed": 82,
    "plan_sha256": "69a30fcd7855fadccf99d7028dccf726b7d3812cbaac6ff87573bdd61521ec1a",
    "correctness": {
        "cases": [
            {"m": 1, "n": 1, "k": 1},
            {"m": 7, "n": 9, "k": 5},
            {"m": 33, "n": 65, "k": 17},
            {"m": 64, "n": 32, "k": 32},
            {"m": 34, "n": 41, "k": 1000},
        ],
        "repeats": 2,
    },
    "benchmark": {
        "sizes": [64, 256],
        "check_rows": 4,
        "warmup": 1,
        "repeats": 7,
        "target_flops_per_sample": 4194304,
        "max_runtime_seconds": 120,
    },
    "comparison": {"min_faster_ratio": 1.1},
}
# The canonical report of that run, byte for byte.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"arithmetic":"2 n**3 floa'
    "ting-point operations for each call of size n: one multiply and one add for "
    'each of the n**3 products","check_rule":"a timing point checks check_rows ro'
    "ws of C, spread evenly from the first row to the last, at every column; a co"
    'rrectness case checks every output","compulsory":"(2 e + c) n**2 bytes for e'
    "ach call: read A and B once at e bytes for each input value and write C once"
    " at c bytes, with c = 4 for a kernel and c = e for a PyTorch call, whose out"
    'put has the input type","device":"mps","points":[{"check_rows":4,"checks":{"'
    'm1":{"max_abs_error":5.238630656378973e-06,"max_bound_ratio":0.0281607272393'
    '7254,"max_relative_error":1.0742505919221867e-07},"m2":{"max_abs_error":0.00'
    '7965810911853666,"max_bound_ratio":0.14699838238803048,"max_relative_error":'
    '0.00017078045138608513},"m3":{"max_abs_error":0.052192096406818766,"max_boun'
    'd_ratio":0.14767137262091531,"max_relative_error":0.0011565036182198806},"s1'
    '":{"max_abs_error":5.238630656378973e-06,"max_bound_ratio":0.028160727239372'
    '54,"max_relative_error":1.0742505919221867e-07},"torch_bfloat16":{"max_abs_e'
    'rror":0.0718550929834123,"max_bound_ratio":0.019904348273462893,"max_relativ'
    'e_error":0.0012440217670914308},"torch_float16":{"max_abs_error":0.009404905'
    '455310697,"max_bound_ratio":0.010694639494734665,"max_relative_error":0.0001'
    '6710374210522914},"torch_float32":{"max_abs_error":5.238630656378973e-06,"ma'
    'x_bound_ratio":0.00044001304245132767,"max_relative_error":1.074250591922186'
    '7e-07}},"index":0,"inputs":{"bfloat16":{"a_sha256":"cb3a011d7b9f4ab086f5c326'
    '7a70879bc675d5feebd32a01ee76805626b1ce64","b_sha256":"5ad3c690b8985c47435053'
    'bac259c451e674b8a7e56e421de9c36b73ed6515a0"},"float16":{"a_sha256":"062aa219'
    'e8a41219ee3aa1f2d816783272aae6fab3b9c7d2c3aeab3d47de6f3e","b_sha256":"61838e'
    '5bdad9559aadafd69161af8380f33514de805d0566d212d23a7eb4204b"},"float32":{"a_s'
    'ha256":"64b10dab3c2bc005efaefc5374d62a7d2bbece95df2a1feb4ac03cdfa41c1afc","b'
    '_sha256":"3fa7b9d0aa70dca54deffd2aa6b9afd885434a5683b846c8a5220a948c230798"}'
    '},"launch":{"active_threads":512,"extent":[512],"group_size":[128],"groups":'
    '[4],"launched_threads":512,"padding_threads":0,"threads":[512],"threads_per_'
    'group":128},"m1_same_bits_as_s1":true,"magnitude_sha256":"4b96b3624fd5247ea2'
    '7fd9a6ee3108ce2a447c9f25c5e7b490fa0009020aeed7","reference_sha256":"b2c7767b'
    '8033748a6ba69b0193a59e6b0a6f9a882627a7314bbf7f76b5b13561","side_order":["s1"'
    ',"m1","m2","m3","torch_float32","torch_float16","torch_bfloat16"],"sides":{"'
    'm1":{"calls_per_sample":8,"compulsory_bytes_per_call":49152,"effective_gb_pe'
    'r_second":1.4529921478060046,"flops_per_call":524288,"implementation":"m1","'
    'intensity_flops_per_byte":10.666666666666666,"iqr_ns_per_call":3395.75,"iqr_'
    'tflops_per_second":[0.01503787520256995,0.016660591857000993],"median_ns_per'
    '_call":33828.125,"repeats":7,"samples_ns":[242000,278916,270625,275875,26837'
    '5,281875,251750],"tflops_per_second":0.015498582909930716,"timing":{"count":'
    '7,"iqr_ns":27166,"max_ns":281875,"median_ns":270625,"min_ns":242000,"p25_ns"'
    ':251750,"p75_ns":278916},"warmup":1},"m2":{"calls_per_sample":8,"compulsory_'
    'bytes_per_call":32768,"effective_gb_per_second":1.016062015503876,"flops_per'
    '_call":524288,"implementation":"m2","intensity_flops_per_byte":16.0,"iqr_ns_'
    'per_call":2463.5,"iqr_tflops_per_second":[0.015870022059018277,0.01714879611'
    '420254],"median_ns_per_call":32250.0,"repeats":7,"samples_ns":[224917,261042'
    ',264291,256833,427333,258000,244583],"tflops_per_second":0.01625699224806201'
    '7,"timing":{"count":7,"iqr_ns":19708,"max_ns":427333,"median_ns":258000,"min'
    '_ns":224917,"p25_ns":244583,"p75_ns":264291},"warmup":1},"m3":{"calls_per_sa'
    'mple":8,"compulsory_bytes_per_call":32768,"effective_gb_per_second":1.017540'
    '9995147986,"flops_per_call":524288,"implementation":"m3","intensity_flops_pe'
    'r_byte":16.0,"iqr_ns_per_call":4239.625,"iqr_tflops_per_second":[0.015153270'
    '325731959,0.017269393721049922],"median_ns_per_call":32203.125,"repeats":7,"'
    'samples_ns":[219625,257292,276792,267333,371333,257625,242875],"tflops_per_s'
    'econd":0.016280655992236777,"timing":{"count":7,"iqr_ns":33917,"max_ns":3713'
    '33,"median_ns":257625,"min_ns":219625,"p25_ns":242875,"p75_ns":276792},"warm'
    'up":1},"s1":{"calls_per_sample":8,"compulsory_bytes_per_call":49152,"effecti'
    've_gb_per_second":1.3839542454905411,"flops_per_call":524288,"implementation'
    '":"s1","intensity_flops_per_byte":10.666666666666666,"iqr_ns_per_call":1739.'
    '625,"iqr_tflops_per_second":[0.014327255337318531,0.015042351515064572],"med'
    'ian_ns_per_call":35515.625,"repeats":7,"samples_ns":[278833,284125,292750,29'
    '1625,281750,405292,269833],"tflops_per_second":0.014762178618565772,"timing"'
    ':{"count":7,"iqr_ns":13917,"max_ns":405292,"median_ns":284125,"min_ns":26983'
    '3,"p25_ns":278833,"p75_ns":292750},"warmup":1},"torch_bfloat16":{"calls_per_'
    'sample":8,"compulsory_bytes_per_call":24576,"effective_gb_per_second":0.6198'
    '879454672145,"flops_per_call":524288,"implementation":"torch_bfloat16","inte'
    'nsity_flops_per_byte":21.333333333333332,"iqr_ns_per_call":6281.25,"iqr_tflo'
    'ps_per_second":[0.01169414526566425,0.013599457876835582],"median_ns_per_cal'
    'l":39645.875,"repeats":7,"samples_ns":[358667,317167,326958,308417,313583,37'
    '0458,296917],"tflops_per_second":0.013224276169967242,"timing":{"count":7,"i'
    'qr_ns":50250,"max_ns":370458,"median_ns":317167,"min_ns":296917,"p25_ns":308'
    '417,"p75_ns":358667},"warmup":1},"torch_float16":{"calls_per_sample":8,"comp'
    'ulsory_bytes_per_call":24576,"effective_gb_per_second":0.6090083727499976,"f'
    'lops_per_call":524288,"implementation":"torch_float16","intensity_flops_per_'
    'byte":21.333333333333332,"iqr_ns_per_call":9036.5,"iqr_tflops_per_second":[0'
    '.01090372711982052,0.013427143657462984],"median_ns_per_call":40354.125,"rep'
    'eats":7,"samples_ns":[308166,322833,384667,312375,315250,393250,327500],"tfl'
    'ops_per_second":0.012992178618666618,"timing":{"count":7,"iqr_ns":72292,"max'
    '_ns":393250,"median_ns":322833,"min_ns":308166,"p25_ns":312375,"p75_ns":3846'
    '67},"warmup":1},"torch_float32":{"calls_per_sample":8,"compulsory_bytes_per_'
    'call":49152,"effective_gb_per_second":1.2037249055610317,"flops_per_call":52'
    '4288,"implementation":"torch_float32","intensity_flops_per_byte":10.66666666'
    '6666666,"iqr_ns_per_call":8994.875,"iqr_tflops_per_second":[0.01071796470534'
    '1216,0.013132850097847358],"median_ns_per_call":40833.25,"repeats":7,"sample'
    's_ns":[326666,325166,391334,353708,393375,319375,316750],"tflops_per_second"'
    ':0.012839732325984338,"timing":{"count":7,"iqr_ns":71959,"max_ns":393375,"me'
    'dian_ns":326666,"min_ns":316750,"p25_ns":319375,"p75_ns":391334},"warmup":1}'
    '},"size":64,"supported":["s1","m1","m2","m3","torch_float32","torch_float16"'
    ',"torch_bfloat16"],"torch_float32_same_bits_as_s1":true},{"check_rows":4,"ch'
    'ecks":{"m1":{"max_abs_error":2.17060032525751e-05,"max_bound_ratio":0.009624'
    '960166279433,"max_relative_error":1.4686747820564652e-07},"m2":{"max_abs_err'
    'or":0.018164658543604162,"max_bound_ratio":0.09491369474111114,"max_relative'
    '_error":0.00010843049807868107},"m3":{"max_abs_error":0.17761611274227462,"m'
    'ax_bound_ratio":0.1214783836544931,"max_relative_error":0.000952771636509280'
    '9},"s1":{"max_abs_error":2.17060032525751e-05,"max_bound_ratio":0.0096249601'
    '66279433,"max_relative_error":1.4686747820564652e-07},"torch_bfloat16":{"max'
    '_abs_error":0.22886703744497083,"max_bound_ratio":0.01856706542624247,"max_r'
    'elative_error":0.0011604415891401543},"torch_float16":{"max_abs_error":0.023'
    '18612499843198,"max_bound_ratio":0.008923083418894308,"max_relative_error":0'
    '.00013942317842022357},"torch_float32":{"max_abs_error":2.17060032525751e-05'
    ',"max_bound_ratio":0.0006015691907303282,"max_relative_error":1.468674782056'
    '4652e-07}},"index":1,"inputs":{"bfloat16":{"a_sha256":"355d0f0042025428a9bbd'
    '8251a15234c0356bb88eb51be4de7720ccdce4ced82","b_sha256":"7179ee38544c82acb71'
    'c8388c4694bd7d9dc92e3e69c4d8a85fc3dd702ba7e4e"},"float16":{"a_sha256":"0546d'
    'c466cc4a0ca5101bbbb80662d0cfe185ed8ba7e866bf9039d664e3268cd","b_sha256":"c3d'
    '6d7fe59f9f244f9f4dc5b94de0d962875fe4d25a27546fd294590eb584479"},"float32":{"'
    'a_sha256":"2c7771e8943ada171464f5e4c1bc8fab60f64ba306a007a133f61d71e4e5d024"'
    ',"b_sha256":"e1cc488f092171f8053d734bdf6cbfc124448fe16bfb58d25b4463be4c97987'
    '5"}},"launch":{"active_threads":8192,"extent":[8192],"group_size":[128],"gro'
    'ups":[64],"launched_threads":8192,"padding_threads":0,"threads":[8192],"thre'
    'ads_per_group":128},"m1_same_bits_as_s1":true,"magnitude_sha256":"b4543ac4bb'
    '702ba6f5b8ee9625ef0e219b424ceba21c76f0aaa4803f0812007c","reference_sha256":"'
    '737b7f6e3c10332b736b5806fccb4c3e62e3237b6e0152e8cf91fe00c3d67d1a","side_orde'
    'r":["s1","m1","m2","m3","torch_float32","torch_float16","torch_bfloat16"],"s'
    'ides":{"m1":{"calls_per_sample":1,"compulsory_bytes_per_call":786432,"effect'
    'ive_gb_per_second":2.9685977117361286,"flops_per_call":33554432,"implementat'
    'ion":"m1","intensity_flops_per_byte":42.666666666666664,"iqr_ns_per_call":39'
    '042.0,"iqr_tflops_per_second":[0.1127565485931656,0.1297837944465288],"media'
    'n_ns_per_call":264917.0,"repeats":7,"samples_ns":[312375,258541,264333,27929'
    '2,297583,258250,264917],"tflops_per_second":0.12666016903407482,"timing":{"c'
    'ount":7,"iqr_ns":39042,"max_ns":312375,"median_ns":264917,"min_ns":258250,"p'
    '25_ns":258541,"p75_ns":297583},"warmup":1},"m2":{"calls_per_sample":1,"compu'
    'lsory_bytes_per_call":524288,"effective_gb_per_second":2.009405289804804,"fl'
    'ops_per_call":33554432,"implementation":"m2","intensity_flops_per_byte":64.0'
    ',"iqr_ns_per_call":37916.0,"iqr_tflops_per_second":[0.11469992924068763,0.13'
    '177980166912126],"median_ns_per_call":260917.0,"repeats":7,"samples_ns":[260'
    '167,254625,199542,292541,290541,353000,260917],"tflops_per_second":0.1286019'
    '3854750746,"timing":{"count":7,"iqr_ns":37916,"max_ns":353000,"median_ns":26'
    '0917,"min_ns":199542,"p25_ns":254625,"p75_ns":292541},"warmup":1},"m3":{"cal'
    'ls_per_sample":1,"compulsory_bytes_per_call":524288,"effective_gb_per_second'
    '":1.9620969432052933,"flops_per_call":33554432,"implementation":"m3","intens'
    'ity_flops_per_byte":64.0,"iqr_ns_per_call":78166.0,"iqr_tflops_per_second":['
    '0.11562200904179072,0.158244272361136],"median_ns_per_call":267208.0,"repeat'
    's":7,"samples_ns":[276791,206750,233750,267208,290208,416375,212042],"tflops'
    '_per_second":0.12557420436513878,"timing":{"count":7,"iqr_ns":78166,"max_ns"'
    ':416375,"median_ns":267208,"min_ns":206750,"p25_ns":212042,"p75_ns":290208},'
    '"warmup":1},"s1":{"calls_per_sample":1,"compulsory_bytes_per_call":786432,"e'
    'ffective_gb_per_second":2.5834036863907075,"flops_per_call":33554432,"implem'
    'entation":"s1","intensity_flops_per_byte":42.666666666666664,"iqr_ns_per_cal'
    'l":101667.0,"iqr_tflops_per_second":[0.10473486383144753,0.1534211460028897]'
    ',"median_ns_per_call":304417.0,"repeats":7,"samples_ns":[347333,217333,21870'
    '8,302958,318625,320375,304417],"tflops_per_second":0.11022522395267019,"timi'
    'ng":{"count":7,"iqr_ns":101667,"max_ns":347333,"median_ns":304417,"min_ns":2'
    '17333,"p25_ns":218708,"p75_ns":320375},"warmup":1},"torch_bfloat16":{"calls_'
    'per_sample":1,"compulsory_bytes_per_call":393216,"effective_gb_per_second":1'
    '.8761827056583789,"flops_per_call":33554432,"implementation":"torch_bfloat16'
    '","intensity_flops_per_byte":85.33333333333333,"iqr_ns_per_call":41833.0,"iq'
    'r_tflops_per_second":[0.13596239748452138,0.16371289867729644],"median_ns_pe'
    'r_call":209583.0,"repeats":7,"samples_ns":[209583,200667,204959,237125,24679'
    '2,208250,266708],"tflops_per_second":0.16010092421618166,"timing":{"count":7'
    ',"iqr_ns":41833,"max_ns":266708,"median_ns":209583,"min_ns":200667,"p25_ns":'
    '204959,"p75_ns":246792},"warmup":1},"torch_float16":{"calls_per_sample":1,"c'
    'ompulsory_bytes_per_call":393216,"effective_gb_per_second":1.774240270727580'
    '4,"flops_per_call":33554432,"implementation":"torch_float16","intensity_flop'
    's_per_byte":85.33333333333333,"iqr_ns_per_call":24750.0,"iqr_tflops_per_seco'
    'nd":[0.14465736038420748,0.1619359870275279],"median_ns_per_call":221625.0,"'
    'repeats":7,"samples_ns":[231958,203000,218667,207208,244167,229500,221625],"'
    'tflops_per_second":0.15140183643542018,"timing":{"count":7,"iqr_ns":24750,"m'
    'ax_ns":244167,"median_ns":221625,"min_ns":203000,"p25_ns":207208,"p75_ns":23'
    '1958},"warmup":1},"torch_float32":{"calls_per_sample":1,"compulsory_bytes_pe'
    'r_call":786432,"effective_gb_per_second":3.2357864072876295,"flops_per_call"'
    ':33554432,"implementation":"torch_float32","intensity_flops_per_byte":42.666'
    '666666666664,"iqr_ns_per_call":25917.0,"iqr_tflops_per_second":[0.1314355013'
    '0830578,0.14628635204359675],"median_ns_per_call":243042.0,"repeats":7,"samp'
    'les_ns":[253917,205958,234875,243042,255292,283125,229375],"tflops_per_secon'
    'd":0.13806022004427218,"timing":{"count":7,"iqr_ns":25917,"max_ns":283125,"m'
    'edian_ns":243042,"min_ns":205958,"p25_ns":229375,"p75_ns":255292},"warmup":1'
    '}},"size":256,"supported":["s1","m1","m2","m3","torch_float32","torch_float1'
    '6","torch_bfloat16"],"torch_float32_same_bits_as_s1":true}],"sides":["s1","m'
    '1","m2","m3","torch_float32","torch_float16","torch_bfloat16"],"synchronizat'
    'ion":"torch.mps.synchronize() before the first and before the second clock r'
    "eading of every sample; untimed warmups of each side first; repeat r starts "
    'with side r mod S, so the order shifts by one side every repeat","torch_call'
    '":"torch.matmul(a, b) on m x k and k x n tensors of the side\'s dtype on the '
    'MPS device; the output has the dtype of the inputs","vendor_bound":"|C - R| '
    "<= v S + 2**-100 for each output of torch.matmul, with v = 2**-12 for float3"
    "2, 2**-6 for float16 and 2**-4 for bfloat16: a sanity bound only, because Py"
    'Torch names no kernel and no derived bound applies"},"ceiling":{"buffer_byte'
    's":268435456,"config_sha256":"b3f1096b3691a46d5a77068639ba57389194a3d07a06bf'
    'c16f5693cf52215199","definition":"the bandwidth ceiling of a verified Day 74'
    " gpu-ceilings report: the better median GB/s of stream_copy and stream_read_"
    "reduce at its largest buffer, on the host that made it; not a vendor peak an"
    'd not a hardware limit","environment":{"macos_version":"27.2.1","numpy_versi'
    'on":"1.26.4","platform_machine":"arm64","platform_system":"Darwin","python_i'
    'mplementation":"CPython","python_version":"3.11.14","torch_version":"2.15.0"'
    '},"gb_per_second":107.7109032544817,"kernel":"stream_read_reduce","label":"h'
    'ost-specific Apple GPU (Metal) measurement","load_average_after":[10.2172851'
    '5625,13.6328125,16.11767578125],"load_average_before":[11.72607421875,14.030'
    '2734375,16.28515625],"measured":true,"report_sha256":"cd75564918285983fe22b4'
    '106b5084b44ef42ed9313ca2a8ef19fa164e68fd86","source":"gpu_ceilings"},"comput'
    'e_ceiling":{"definition":"the FMA ceiling of the same verified Day 74 gpu-ce'
    "ilings report: the best median GFLOP/s of its FMA chains, 2 operations for e"
    "ach fma, on the host that made it; not a vendor peak and not a hardware limi"
    't","fma_gflops_per_second":4665.472950926219,"report_sha256":"cd755649182859'
    '83fe22b4106b5084b44ef42ed9313ca2a8ef19fa164e68fd86","source":"gpu_ceilings"}'
    ',"config":{"benchmark":{"check_rows":4,"max_runtime_seconds":120,"repeats":7'
    ',"sizes":[64,256],"target_flops_per_sample":4194304,"warmup":1},"comparison"'
    ':{"min_faster_ratio":1.1},"correctness":{"cases":[{"k":1,"m":1,"n":1},{"k":5'
    ',"m":7,"n":9},{"k":17,"m":33,"n":65},{"k":32,"m":64,"n":32},{"k":1000,"m":34'
    ',"n":41}],"repeats":2},"name":"metal-tc-recorded","plan_sha256":"69a30fcd785'
    '5fadccf99d7028dccf726b7d3812cbaac6ff87573bdd61521ec1a","schema_version":1,"s'
    'eed":82},"config_sha256":"9880c20c39750756cc5256ce386be0199e85dbc25383658434'
    'fc3d9a71b7c3d3","correctness":{"bound":"|C - R| <= (gamma_K(2**-24) (1 + u)*'
    "*2 + 2 u + u**2 + gamma_K(2**-53)) S + e (1 + u) (1 + gamma_K(2**-24)) (R_i "
    "+ C_j) + K e**2 (1 + gamma_K(2**-24)) + 2 (K + 1) 2**-126, with gamma_K(v) ="
    " K v / (1 - K v); u = 0 and e = 2**-126 for float32 inputs, u = 2**-11 and e"
    " = 2**-14 for half, u = 2**-8 and e = 2**-126 for bfloat; it assumes that th"
    "e float32 accumulators round each product and each sum at most once, in any "
    'order","cases":[{"bound_sha256":{"bfloat16":"48d2fe50fa0a8a40395cb02f12a86a6'
    'dc2583d1f855b6329132789b75e16c432","float16":"3dff51b3055f967d80daa6854f29b3'
    '92db5d9993da7f2164a2a4e63c857bb281","float32":"af9bb2f757f683d4f935a48d50d97'
    'fbc75060f021572861db1bd1012a5d60ed6"},"case":{"k":1,"m":1,"n":1},"index":0,"'
    'inputs":{"bfloat16":{"a_sha256":"64ed923c16a36aa4ed1138b12cd852e29b491569c1a'
    'afe52bd660ed1c9e14d7f","b_sha256":"0cd65a74756f9fd49c3a50100f647a91d9446331e'
    '7cc93bc83c0b9ca7e9856ab"},"float16":{"a_sha256":"e6a498282a7af2761f13c9a51d0'
    '1ccca208bcaefcabfe83566049299c3ba6833","b_sha256":"fe0793d41fafc1241a503cae3'
    '9989d065e20fba4784e64f9ad2092601f313535"},"float32":{"a_sha256":"9ae477edbab'
    'a6f5ca4592f0edfa2afdca2edec272c2a1cb0eb7c16017e7e6ea2","b_sha256":"1fa4047eb'
    'd10f06c6475bb478fe4f63bdb08c2aa31204b60a0981b0987f11605"}},"launch":{"active'
    '_threads":128,"extent":[128],"group_size":[128],"groups":[1],"launched_threa'
    'ds":128,"padding_threads":0,"threads":[128],"threads_per_group":128},"m1_sam'
    'e_bits_as_s1":true,"magnitude_sha256":"4a1d6d4e608992d5cc22d4ab0f042ca4048ad'
    '9c08fd79dcac7c53fefbd7d86af","outputs":{"m1":{"buffer_sha256":"777ef398695be'
    'c28b14fc3ff4d119e776102428f80098a436e18c1c6ed6e98e8","identical_repeats":tru'
    'e,"max_abs_error":9.606466733202979e-10,"max_bound_ratio":0.0255384731815474'
    '84,"max_relative_error":1.5222117156582066e-09},"m2":{"buffer_sha256":"fd8bd'
    '5daa6b2efe1fa97f7b2300b4b8363ca39d7275a428bcd236fb0f90292bd","identical_repe'
    'ats":true,"max_abs_error":4.673100215057957e-05,"max_bound_ratio":0.06453492'
    '012080414,"max_relative_error":7.404853515205323e-05},"m3":{"buffer_sha256":'
    '"07b3b0bc4010af49d41a4674c9ce744fbab47bcf61b7a7ce554b4a85913474dd","identica'
    'l_repeats":true,"max_abs_error":0.0017263879706277407,"max_bound_ratio":0.34'
    '94692845736694,"max_relative_error":0.002735582256874939},"s1":{"buffer_sha2'
    '56":"777ef398695bec28b14fc3ff4d119e776102428f80098a436e18c1c6ed6e98e8","iden'
    'tical_repeats":true,"max_abs_error":9.606466733202979e-10,"max_bound_ratio":'
    '0.025538473181547484,"max_relative_error":1.5222117156582066e-09}},"referenc'
    'e_sha256":"4a1d6d4e608992d5cc22d4ab0f042ca4048ad9c08fd79dcac7c53fefbd7d86af"'
    ',"repeats":2},{"bound_sha256":{"bfloat16":"d5e1fb7674753eb2ac37810a228c56975'
    '39c741e4873c5e264d168a21f53a977","float16":"40bb75dd1591e2c47010ffcf30bbd7ff'
    '9e6759b71c81e6e592e0c3d64aa23012","float32":"0a93e69b2ed8c8aea675eca9309d2c3'
    '708c7988d5f6663186246d485199f7768"},"case":{"k":5,"m":7,"n":9},"index":1,"in'
    'puts":{"bfloat16":{"a_sha256":"1b716ccfd9c64c952fd33b9ac40ac0cfe847491df5f14'
    '217f9d7e935cffc408c","b_sha256":"a58a0a513ba604d97fdbe40c2cd2f4f8dcf3918a99e'
    '3dfdeb6630a7ea3d2213c"},"float16":{"a_sha256":"4411a59363994722ff56290bb56b1'
    '090df851d506d52c0f3a33e12f5ea9429eb","b_sha256":"961f19392bde2e2f6290be9366c'
    'c2797326282333dee577a71954c98db0ceb70"},"float32":{"a_sha256":"8f88cef4469ef'
    '3039bada9c3eae166a974a47344ec0ef153c3149db7f8540722","b_sha256":"99662205610'
    '39c7e3d2ef197cae363fd6d0aa4fe7ed2ef086f0eea3a9ab8aed9"}},"launch":{"active_t'
    'hreads":128,"extent":[128],"group_size":[128],"groups":[1],"launched_threads'
    '":128,"padding_threads":0,"threads":[128],"threads_per_group":128},"m1_same_'
    'bits_as_s1":true,"magnitude_sha256":"19a1f9e0575d9f88fcc3bae2455ea48cf6acf7d'
    'f72b43f5b513108c6ab7548e1","outputs":{"m1":{"buffer_sha256":"dcd0db99d4fcc86'
    '7a190bec4e1cf4338e35034ea47992d986603635aa2ceca1b","identical_repeats":true,'
    '"max_abs_error":3.193193558459484e-07,"max_bound_ratio":0.30476404457829903,'
    '"max_relative_error":9.082679032467498e-08},"m2":{"buffer_sha256":"b4ff236bc'
    '2de7de017dbecd618fc044d6873ce2a0cc4d9d14e1bd6552fbc8b71","identical_repeats"'
    ':true,"max_abs_error":0.0017285087300837887,"max_bound_ratio":0.303418833838'
    '9854,"max_relative_error":0.0003325659168697289},"m3":{"buffer_sha256":"0d31'
    'bf07a25b160676d2419af883530309dcd28b533087e0bdd09e7371056c66","identical_rep'
    'eats":true,"max_abs_error":0.016531624955546498,"max_bound_ratio":0.48078726'
    '07474941,"max_relative_error":0.003763631113404673},"s1":{"buffer_sha256":"d'
    'cd0db99d4fcc867a190bec4e1cf4338e35034ea47992d986603635aa2ceca1b","identical_'
    'repeats":true,"max_abs_error":3.193193558459484e-07,"max_bound_ratio":0.3047'
    '6404457829903,"max_relative_error":9.082679032467498e-08}},"reference_sha256'
    '":"7c0411756afae91157f763dd14425e79a2e55352e3fc600584c761aa48691faa","repeat'
    's":2},{"bound_sha256":{"bfloat16":"c24d0f7559d9f806a07b95535ccc541da9b197711'
    'ed7c97b0bb9f8f4d648239e","float16":"07c6b55a988348efde6bdf2757e322430f1d16e4'
    '479817691e84475154c6f366","float32":"bb159a71cfd9e93e003f10b78fec263a3e89172'
    'cee43f4ce727ce7916ef3bd19"},"case":{"k":17,"m":33,"n":65},"index":2,"inputs"'
    ':{"bfloat16":{"a_sha256":"bf7bb1de2f2295902782458da83e2357e8f3e8e47dae8014e7'
    '5b7c525757cd56","b_sha256":"43202b1c9b87e4bf3378a2144fa6db05d485323a157e8b32'
    '0cf3e20b59114ec2"},"float16":{"a_sha256":"b4d0d2532df14d5fd9a22c5e7538348dc1'
    'cc4791965a08b9afc2d226f2eb0d52","b_sha256":"9bb8099f406311f59de76df064c3ff9f'
    '74ae7a2bfc67e509d005c99581cc413c"},"float32":{"a_sha256":"a04077076b6812d25c'
    '8d5b408090c52572e53ddac982977a3a5feb1c4615347d","b_sha256":"5df6d21f074cd8b4'
    '62ff92f802c56651514d031770d8975d7f4915e6fdbd3bae"}},"launch":{"active_thread'
    's":768,"extent":[768],"group_size":[128],"groups":[6],"launched_threads":768'
    ',"padding_threads":0,"threads":[768],"threads_per_group":128},"m1_same_bits_'
    'as_s1":true,"magnitude_sha256":"803db0d6c1393b1468a38f51b8743cfae2a41de96ae4'
    'd1c6a09def5ae0b139cd","outputs":{"m1":{"buffer_sha256":"d62a7e8d99edcb1d6ef1'
    '5eef7e069626b3a5a9db9f3f69de26d59b00f889dac8","identical_repeats":true,"max_'
    'abs_error":1.8267626007428817e-06,"max_bound_ratio":0.14137045380158708,"max'
    '_relative_error":1.4324785198670334e-07},"m2":{"buffer_sha256":"d6d1da5ba271'
    'd8c47ae1f6745b2f86b98cfe08c6d6f60f883fea2dc0d8d44c02","identical_repeats":tr'
    'ue,"max_abs_error":0.005890985635124579,"max_bound_ratio":0.3664536263750756'
    ',"max_relative_error":0.000413331606162518},"m3":{"buffer_sha256":"fc1546d89'
    'db5497303c584134710b407b2df205d518f8510ba14b4efa37dcd75","identical_repeats"'
    ':true,"max_abs_error":0.03996518880317623,"max_bound_ratio":0.36296189968796'
    '14,"max_relative_error":0.0028415488613162893},"s1":{"buffer_sha256":"d62a7e'
    '8d99edcb1d6ef15eef7e069626b3a5a9db9f3f69de26d59b00f889dac8","identical_repea'
    'ts":true,"max_abs_error":1.8267626007428817e-06,"max_bound_ratio":0.14137045'
    '380158708,"max_relative_error":1.4324785198670334e-07}},"reference_sha256":"'
    'c9f4a6e0bcbc82c8a5881333c1e6bff217e33637718989e4183a5158c53c9563","repeats":'
    '2},{"bound_sha256":{"bfloat16":"0562ba1814d24fce7de8626aa9a35b94d98145a800fa'
    '83bc3b9a35f8eea80d22","float16":"36e81b33ac635ea4c3f66f2d34e052255b80e860ec8'
    '94e609e837003f0e14a74","float32":"18d3dfcfdf7e91a16f6ceb22c7034c59513c56aaa3'
    '588a80934328c27dbc4e8c"},"case":{"k":32,"m":64,"n":32},"index":3,"inputs":{"'
    'bfloat16":{"a_sha256":"12a6ee4a4e1dcb206de81f18e752ecef8c3a132b12208600324f7'
    'e4dbbe068ca","b_sha256":"213422ea5f38772a0521e3af60692ae93bd7a7bbe935de8cc0e'
    'b4cedfa750c73"},"float16":{"a_sha256":"568a173216b195d99675451305744c44a4e40'
    '6dad58f19a77b32c4ce88ba9724","b_sha256":"235dcd5652786c2fba99a172d5e23026f1f'
    '8a7831631b4f3665b199b8ed7e445"},"float32":{"a_sha256":"0a8b4d41d58583976a7c2'
    '50e05a536cb0f47d20acaabcccd9e449d91582a39d4","b_sha256":"61183a173be3112e8e0'
    '318fe347938ac423dbc3dd1153308508123d3a6124b98"}},"launch":{"active_threads":'
    '256,"extent":[256],"group_size":[128],"groups":[2],"launched_threads":256,"p'
    'adding_threads":0,"threads":[256],"threads_per_group":128},"m1_same_bits_as_'
    's1":true,"magnitude_sha256":"85892bffe2a5c1800e309943f0b4712c921a41c599a1448'
    '5d2a999df30940809","outputs":{"m1":{"buffer_sha256":"3c821c02e96e51c9a3f82f3'
    '61905d71f6970da1534ba17ac9bd9c691421e58a8","identical_repeats":true,"max_abs'
    '_error":5.449559171921692e-06,"max_bound_ratio":0.13089700820638553,"max_rel'
    'ative_error":2.4966670630818563e-07},"m2":{"buffer_sha256":"06f509a6437be168'
    '4941e63b4bc42f26672ada57bd9a95009356d7996089b280","identical_repeats":true,"'
    'max_abs_error":0.005970339961840931,"max_bound_ratio":0.25501783808504425,"m'
    'ax_relative_error":0.00029647577838607655},"m3":{"buffer_sha256":"061e454f30'
    '5df0540c8d9160c63dfe74f9f269b45af0500f0092104ce1ecce9b","identical_repeats":'
    'true,"max_abs_error":0.06535758056140839,"max_bound_ratio":0.316870043506108'
    '4,"max_relative_error":0.0024809913818064633},"s1":{"buffer_sha256":"3c821c0'
    '2e96e51c9a3f82f361905d71f6970da1534ba17ac9bd9c691421e58a8","identical_repeat'
    's":true,"max_abs_error":5.449559171921692e-06,"max_bound_ratio":0.1308970082'
    '0638553,"max_relative_error":2.4966670630818563e-07}},"reference_sha256":"e8'
    '41a600b3a871afee6c147565b1c8d984d761a372a06ab4d0ee25a193fb3fb5","repeats":2}'
    ',{"bound_sha256":{"bfloat16":"dfef231d94786c65d6343fcc3e831b95db244b3e45b404'
    'e12b7cbbec0858abf0","float16":"a52be7769a84c31bcc2b451d84c4079916764587f7d74'
    'db399b571bc0c1b5598","float32":"42efd6fba9d53900dad98ffb37fc17f1c07703e800f6'
    '81723f905c526b8aa0e4"},"case":{"k":1000,"m":34,"n":41},"index":4,"inputs":{"'
    'bfloat16":{"a_sha256":"6c9beadf07f7722653d1fe9c7df3c93da5a3e37781b61f39c8917'
    '52a69003943","b_sha256":"2c28d23ac0f4bf4ddc9685056d06e486fde3367ce1099cc1c09'
    '933cade66bf5b"},"float16":{"a_sha256":"cbd37049750f11bafdf291bfc967675181bfb'
    '23b59e4e251304ff4ad1fca3469","b_sha256":"a116fcd7ba6fbc62db80bc2f832bd8a1a3a'
    '3e282422cb5747505acc28424b26b"},"float32":{"a_sha256":"8bfdf60a0bd11c8311b2b'
    'de1432ef133cfe2383ac5af086dcc8fc74a6816b80e","b_sha256":"878128128f905f3065a'
    'd97a2b1d39aa6dd89cf99e7312b0287d3ff3b03d85c0f"}},"launch":{"active_threads":'
    '512,"extent":[512],"group_size":[128],"groups":[4],"launched_threads":512,"p'
    'adding_threads":0,"threads":[512],"threads_per_group":128},"m1_same_bits_as_'
    's1":true,"magnitude_sha256":"84eaef465891135fdaa3938f97e069db82d8ba9b98b8a1d'
    '6af643f44ad264120","outputs":{"m1":{"buffer_sha256":"c3a2438849dad768c34d7a5'
    '5b41bc08e5b8a49b61ef0741801f05fddc075b2af","identical_repeats":true,"max_abs'
    '_error":0.00011806837666483716,"max_bound_ratio":0.0031076965555344193,"max_'
    'relative_error":1.852441910215397e-07},"m2":{"buffer_sha256":"d0558545d7f5da'
    'a2c86d023a303d7dbc778666d9c82810002b429265dc646873","identical_repeats":true'
    ',"max_abs_error":0.03996888483527883,"max_bound_ratio":0.05136197174425405,"'
    'max_relative_error":6.101283670770219e-05},"m3":{"buffer_sha256":"5a2feb7b72'
    '396474f213dd676453d971ac22dd2472f492d9ae51c109fe62eec7","identical_repeats":'
    'true,"max_abs_error":0.22995811625556684,"max_bound_ratio":0.048557218106227'
    '91,"max_relative_error":0.0003830112557951904},"s1":{"buffer_sha256":"c3a243'
    '8849dad768c34d7a55b41bc08e5b8a49b61ef0741801f05fddc075b2af","identical_repea'
    'ts":true,"max_abs_error":0.00011806837666483716,"max_bound_ratio":0.00310769'
    '65555344193,"max_relative_error":1.852441910215397e-07}},"reference_sha256":'
    '"c9279d9c828f3dd33541ae1eeb6982ba9fa685713a07f2d73b087a43b98712f5","repeats"'
    ':2}],"distribution":"A and then B from a standard normal, drawn as float32 w'
    "ith the Day 74 matmul_inputs; half and bfloat operands are the float32 draws"
    ' rounded to nearest, ties to even","guard_float32_bits":2143289344,"referenc'
    'e":"float64 product of the float32 draws, with the exact products of each ou'
    "tput added in increasing k order; S = |A| @ |B| by the same loop, and the ro"
    'w sums of |A| and the column sums of |B| in k order","relative_error":"max o'
    'ver the checked outputs of |C - R| / S, in float64","rules":"every C buffer '
    "keeps its tail sentinels in every repeat; every output is finite; every outp"
    "ut is within the bound of its input type; identical_repeats and m1_same_bits"
    '_as_s1 are measured","sentinel_float32_bits":2143289344,"tail":64},"environm'
    'ent":{"macos_version":"27.2.1","numpy_version":"1.26.4","platform_machine":"'
    'arm64","platform_system":"Darwin","python_implementation":"CPython","python_'
    'version":"3.11.14","torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu'
    '_metal_this_host":true,"cuda_kernel_executed":false,"distributed_claim":fals'
    'e,"hardware_counters_used":false,"host_specific":true,"nvidia_gpu_used":fals'
    'e,"occupancy_measured":false,"portable_performance_claim":false,"production_'
    'claim":false,"ptx_executed":false,"statement":"These are wall-clock timings '
    "on the Apple GPU of the host that made this report, at the recorded load ave"
    "rages. Apple GPUs have no NVIDIA Tensor Cores. The simdgroup_matrix path is "
    "the matrix path of the Apple GPU, and this report times it against scalar fm"
    "a on the same tiles. S1, M1, M2 and M3 came from torch.mps.compile_shader, a"
    "nd torch.matmul ran on the MPS device. PyTorch does not report which kernels"
    " it ran, so its numbers name the call, not a kernel. Both ceilings come from"
    " a separate gpu-ceilings run with its own load averages. No hardware counter"
    " was read, and each byte and operation count is a model. No CUDA kernel ran,"
    " no NVIDIA GPU was used, no PTX ran, and no Triton kernel ran on a GPU. Thes"
    "e numbers do not measure or predict CUDA, NVIDIA, Tensor Core, Triton, distr"
    'ibuted or production performance.","tensor_cores_used":false,"triton_gpu_ker'
    'nel_executed":false,"vendor_kernel_known":false,"vendor_peak_claimed":false}'
    ',"host_load":{"after":[4.79638671875,4.86669921875,5.21923828125],"before":['
    '4.79638671875,4.86669921875,5.21923828125],"source":"os.getloadavg(): the 1,'
    " 5 and 15 minute load averages that uptime prints, read before the checks an"
    'd after the last sample"},"kind":"metal_matrix_precision","label":"host-spec'
    'ific Apple GPU (Metal) measurement","measured":true,"measurement_scope":"app'
    'le_gpu_metal_this_host","percentile_method":"nearest rank: sorted samples[ce'
    "il(p * n / 100) - 1] for p in 25, 50 and 75; median is p50 and IQR is p75 - "
    'p25; no interpolation","plan":{"kernels":{"tc_m1_float":"3442d3672b6025910de'
    'd3ed1a31705f89b13f5aa34447dd0e5bfb37f824862dd","tc_m2_half":"65cff25e1ceb603'
    '0b1cafd5a69f393a8f4a2e7391f96d337145d728087311ea7","tc_m3_bfloat":"6a6a35477'
    'de0356297fff5a72eb2370f05a8a95c170f8ed8786a4a685ba6bf50","tc_s1_float":"ba1c'
    '7a853364ef75986c21b539ef07e5768331a998e9aec26c7c9b2a2b502d4f"},"libraries":{'
    '"bfloat":"2ba3e64ba708a41c22f6f6f11ff71309e93a0cab059b8405d7a078766947ef68",'
    '"main":"93c29857581d44f58be7b5e8cf6357734e2b6e90ed1fced726e7021da67cc5cc"},"'
    'precisions":{"m1":"float32","m2":"float16","m3":"bfloat16","s1":"float32","t'
    'orch_bfloat16":"bfloat16","torch_float16":"float16","torch_float32":"float32'
    '"},"sha256":"69a30fcd7855fadccf99d7028dccf726b7d3812cbaac6ff87573bdd61521ec1'
    'a","sides":["s1","m1","m2","m3","torch_float32","torch_float16","torch_bfloa'
    't16"],"steps":[{"base":"s1","change":"Use the simdgroup matrix path for floa'
    "t32 inputs. M1 and S1 stage the same tiles and hold 8 float accumulators per"
    " thread, but M1 multiplies 8 x 8 fragments with simdgroup_multiply_accumulat"
    'e in place of scalar fma.","first":"m1","hypothesis":"In the scratch prototy'
    "pe, an MMA chain with its operands in registers ran at about 4.6 TFLOP/s, ne"
    "ar the Day 74 FMA ceiling, so the matrix path adds no arithmetic rate. M1 ne"
    "eds fewer loads from threadgroup memory for each fma. In the prototype, M1 t"
    "ook about 0.65 to 0.75 of the time of S1. M1 is faster than S1 by 1.1 or mor"
    'e at every size.","predicted":["faster"],"step":"m1_over_s1"},{"base":"m1","'
    'change":"Read half inputs in place of float32 inputs on the matrix path. The'
    " tiles and fragments hold half the bytes, and the accumulators stay float32."
    '","first":"m2","hypothesis":"The MMA chain of the prototype ran at the same '
    "rate for half and float, so the arithmetic rate does not change. Half the by"
    "tes can help only where loads limit M1. In the prototype, M2 was faster than"
    " M1 at 1024 and a little slower at 2048. M2 is faster than M1 or shows no cl"
    'ear difference.","predicted":["faster","no_clear_difference"],"step":"m2_ove'
    'r_m1"},{"base":"m1","change":"Read bfloat inputs in place of float32 inputs '
    "on the matrix path. Bfloat has the bytes of half and the exponent range of f"
    'loat32.","first":"m3","hypothesis":"Bfloat moves the same bytes as half, and'
    " the MMA chain of the prototype ran at the same rate for bfloat and half. M3"
    ' has the speed of M2: faster than M1 or no clear difference.","predicted":["'
    'faster","no_clear_difference"],"step":"m3_over_m1"},{"base":"s1","change":"C'
    "ompare bfloat inputs on the matrix path with float32 inputs without it: the "
    'pair of the hands-on task, fp32 without the matrix path and bf16 with it.","'
    'first":"m3","hypothesis":"This step adds the two changes before it. In the p'
    "rototype, M3 took about 0.55 to 0.8 of the time of S1. M3 is faster than S1 "
    'by 1.1 or more at every size.","predicted":["faster"],"step":"m3_over_s1"},{'
    '"base":"torch_float32","change":"Call torch.matmul on float16 tensors in pla'
    'ce of float32 tensors.","first":"torch_float16","hypothesis":"In the prototy'
    "pe at 2048, the float16 call was about 1.1 times as fast as the float32 call"
    '. The float16 call is faster or shows no clear difference.","predicted":["fa'
    'ster","no_clear_difference"],"step":"torch_float16_over_torch_float32"},{"ba'
    'se":"torch_float32","change":"Call torch.matmul on bfloat16 tensors in place'
    ' of float32 tensors.","first":"torch_bfloat16","hypothesis":"In the prototyp'
    "e at 2048, the bfloat16 call had about the speed of the float16 call. The bf"
    'loat16 call is faster or shows no clear difference.","predicted":["faster","'
    'no_clear_difference"],"step":"torch_bfloat16_over_torch_float32"},{"base":"t'
    'orch_float32","change":"Compare M1 with the float32 PyTorch call on the same'
    ' tensors.","first":"m1","hypothesis":"In the prototype at 2048, the float32 '
    "call ran at about 3.8 to 4.0 TFLOP/s and M1 at about 1.8. M1 is slower than "
    'the PyTorch call at every size.","predicted":["slower"],"step":"m1_over_torc'
    'h_float32"},{"base":"torch_float16","change":"Compare M2 with the float16 Py'
    'Torch call on the same tensors.","first":"m2","hypothesis":"The PyTorch call'
    " writes float16 outputs and M2 writes float32, but both do the same 2 n**3 o"
    'perations. M2 is slower than the PyTorch call at every size.","predicted":["'
    'slower"],"step":"m2_over_torch_float16"},{"base":"torch_bfloat16","change":"'
    'Compare M3 with the bfloat16 PyTorch call on the same tensors.","first":"m3"'
    ',"hypothesis":"M3 is slower than the bfloat16 PyTorch call at every size.","'
    'predicted":["slower"],"step":"m3_over_torch_bfloat16"}],"torch_call":"torch.'
    "matmul(a, b) on m x k and k x n tensors of the side's dtype on the MPS devic"
    'e; the output has the dtype of the inputs"},"planned_work":{"benchmark_point'
    's":2,"calls":504,"correctness_cases":5,"correctness_launches":40,"correctnes'
    's_products":1496317,"flops":2113929216,"sides_per_point":7,"timed_samples":9'
    '8},"report_sha256":"6e6caa94dc97560ca030390dee94e2dc78707650bdb9995786767129'
    '9bfdb6c9","run":{"elapsed_ns":103743042,"max_runtime_ns":120000000000,"timed'
    '_ns":27501370},"schema_version":1,"shaders":{"admissions":[{"admitted":true,'
    '"allowed_threads":1024,"declared_threadgroup_bytes":4864,"kernel":"tc_s1_flo'
    'at","reason":null,"required_threads":128,"static_threadgroup_bytes":4864,"th'
    'read_execution_width":32},{"admitted":true,"allowed_threads":1024,"declared_'
    'threadgroup_bytes":5888,"kernel":"tc_m1_float","reason":null,"required_threa'
    'ds":128,"static_threadgroup_bytes":5888,"thread_execution_width":32},{"admit'
    'ted":true,"allowed_threads":1024,"declared_threadgroup_bytes":3456,"kernel":'
    '"tc_m2_half","reason":null,"required_threads":128,"static_threadgroup_bytes"'
    ':3456,"thread_execution_width":32},{"admitted":true,"allowed_threads":1024,"'
    'declared_threadgroup_bytes":3456,"kernel":"tc_m3_bfloat","reason":null,"requ'
    'ired_threads":128,"static_threadgroup_bytes":3456,"thread_execution_width":3'
    '2},{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":0,"k'
    'ernel":"tc_probe_version","reason":null,"required_threads":32,"static_thread'
    'group_bytes":0,"thread_execution_width":32}],"kernels":[{"group_size":128,"l'
    'ibrary":"main","limits":{"max_threads_per_threadgroup":1024,"static_threadgr'
    'oup_memory_bytes":4864,"thread_execution_width":32},"name":"tc_s1_float","pr'
    'ecision":"float32","simdgroup_barriers":0,"source_sha256":"ba1c7a853364ef759'
    '86c21b539ef07e5768331a998e9aec26c7c9b2a2b502d4f","summary":"scalar fma from '
    'the padded tiles; 8 float accumulators per thread","threadgroup_barriers":2,'
    '"threadgroup_bytes":4864,"variant":"s1"},{"group_size":128,"library":"main",'
    '"limits":{"max_threads_per_threadgroup":1024,"static_threadgroup_memory_byte'
    's":5888,"thread_execution_width":32},"name":"tc_m1_float","precision":"float'
    '32","simdgroup_barriers":2,"source_sha256":"3442d3672b6025910ded3ed1a31705f8'
    '9b13f5aa34447dd0e5bfb37f824862dd","summary":"simdgroup_multiply_accumulate o'
    'n float 8 x 8 fragments into 4 float accumulators per SIMD group","threadgro'
    'up_barriers":2,"threadgroup_bytes":5888,"variant":"m1"},{"group_size":128,"l'
    'ibrary":"main","limits":{"max_threads_per_threadgroup":1024,"static_threadgr'
    'oup_memory_bytes":3456,"thread_execution_width":32},"name":"tc_m2_half","pre'
    'cision":"float16","simdgroup_barriers":2,"source_sha256":"65cff25e1ceb6030b1'
    'cafd5a69f393a8f4a2e7391f96d337145d728087311ea7","summary":"simdgroup_multipl'
    "y_accumulate on half 8 x 8 fragments into 4 float accumulators per SIMD grou"
    'p","threadgroup_barriers":2,"threadgroup_bytes":3456,"variant":"m2"},{"group'
    '_size":128,"library":"bfloat","limits":{"max_threads_per_threadgroup":1024,"'
    'static_threadgroup_memory_bytes":3456,"thread_execution_width":32},"name":"t'
    'c_m3_bfloat","precision":"bfloat16","simdgroup_barriers":2,"source_sha256":"'
    '6a6a35477de0356297fff5a72eb2370f05a8a95c170f8ed8786a4a685ba6bf50","summary":'
    '"simdgroup_multiply_accumulate on bfloat 8 x 8 fragments into 4 float accumu'
    'lators per SIMD group","threadgroup_barriers":2,"threadgroup_bytes":3456,"va'
    'riant":"m3"},{"group_size":32,"library":"main","limits":{"max_threads_per_th'
    'readgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execution_width"'
    ':32},"name":"tc_probe_version","precision":null,"simdgroup_barriers":0,"sour'
    'ce_sha256":"8ed1ed739efb7b05186907686a212861e6b977f85f8c75744fcff8fea690b4d8'
    '","summary":"writes __METAL_VERSION__ once","threadgroup_barriers":0,"thread'
    'group_bytes":0,"variant":null}],"libraries":{"bfloat":"2ba3e64ba708a41c22f6f'
    '6f11ff71309e93a0cab059b8405d7a078766947ef68","main":"93c29857581d44f58be7b5e'
    '8cf6357734e2b6e90ed1fced726e7021da67cc5cc"}},"speedup":{"bandwidth_gb_per_se'
    'cond":107.7109032544817,"fma_gflops_per_second":4665.472950926219,"fraction_'
    'rule":"TFLOP/s of a side * 1000 / the FMA ceiling in GFLOP/s, and the same o'
    "ver the roofline of the side; both ceilings come from one separate Day 74 ru"
    'n on this host","hypotheses":[{"held_at":[],"missed_at":[0,1],"not_run_at":['
    '],"predicted":["faster"],"step":"m1_over_s1"},{"held_at":[0,1],"missed_at":['
    '],"not_run_at":[],"predicted":["faster","no_clear_difference"],"step":"m2_ov'
    'er_m1"},{"held_at":[0,1],"missed_at":[],"not_run_at":[],"predicted":["faster'
    '","no_clear_difference"],"step":"m3_over_m1"},{"held_at":[0],"missed_at":[1]'
    ',"not_run_at":[],"predicted":["faster"],"step":"m3_over_s1"},{"held_at":[0,1'
    '],"missed_at":[],"not_run_at":[],"predicted":["faster","no_clear_difference"'
    '],"step":"torch_float16_over_torch_float32"},{"held_at":[0,1],"missed_at":[]'
    ',"not_run_at":[],"predicted":["faster","no_clear_difference"],"step":"torch_'
    'bfloat16_over_torch_float32"},{"held_at":[],"missed_at":[0,1],"not_run_at":['
    '],"predicted":["slower"],"step":"m1_over_torch_float32"},{"held_at":[1],"mis'
    'sed_at":[0],"not_run_at":[],"predicted":["slower"],"step":"m2_over_torch_flo'
    'at16"},{"held_at":[],"missed_at":[0,1],"not_run_at":[],"predicted":["slower"'
    '],"step":"m3_over_torch_bfloat16"}],"hypothesis_rule":"a prediction holds at'
    " a point when the verdict of the first side of a step over its base is one o"
    "f the verdicts that the step predicted; a step with an unsupported side give"
    's no verdict","min_faster_ratio":1.1,"points":[{"index":0,"rates":{"m1":{"co'
    'mpute_bound":false,"fraction_of_fma":0.003321974657864824,"fraction_of_roofl'
    'ine":0.013489740628886127,"intensity_flops_per_byte":10.666666666666666,"iqr'
    '_tflops_per_second":[0.01503787520256995,0.016660591857000993],"median_ns_pe'
    'r_call":33828.125,"roofline_gflops_per_second":1148.916301381138,"tflops_per'
    '_second":0.015498582909930716},"m2":{"compute_bound":false,"fraction_of_fma"'
    ':0.003484532526297163,"fraction_of_roofline":0.009433232707215265,"intensity'
    '_flops_per_byte":16.0,"iqr_tflops_per_second":[0.015870022059018277,0.017148'
    '79611420254],"median_ns_per_call":32250.0,"roofline_gflops_per_second":1723.'
    '3744520717073,"tflops_per_second":0.016256992248062017},"m3":{"compute_bound'
    '":false,"fraction_of_fma":0.00348960462604432,"fraction_of_roofline":0.00944'
    '6963759190833,"intensity_flops_per_byte":16.0,"iqr_tflops_per_second":[0.015'
    '153270325731959,0.017269393721049922],"median_ns_per_call":32203.125,"roofli'
    'ne_gflops_per_second":1723.3744520717073,"tflops_per_second":0.0162806559922'
    '36777},"s1":{"compute_bound":false,"fraction_of_fma":0.003164133363078462,"f'
    'raction_of_roofline":0.012848785068868659,"intensity_flops_per_byte":10.6666'
    '66666666666,"iqr_tflops_per_second":[0.014327255337318531,0.0150423515150645'
    '72],"median_ns_per_call":35515.625,"roofline_gflops_per_second":1148.9163013'
    '81138,"tflops_per_second":0.014762178618565772},"torch_bfloat16":{"compute_b'
    'ound":false,"fraction_of_fma":0.0028344985190283606,"fraction_of_roofline":0'
    '.005755108598454928,"intensity_flops_per_byte":21.333333333333332,"iqr_tflop'
    's_per_second":[0.01169414526566425,0.013599457876835582],"median_ns_per_call'
    '":39645.875,"roofline_gflops_per_second":2297.832602762276,"tflops_per_secon'
    'd":0.013224276169967242},"torch_float16":{"compute_bound":false,"fraction_of'
    '_fma":0.002784750604134856,"fraction_of_roofline":0.005654101435869797,"inte'
    'nsity_flops_per_byte":21.333333333333332,"iqr_tflops_per_second":[0.01090372'
    '711982052,0.013427143657462984],"median_ns_per_call":40354.125,"roofline_gfl'
    'ops_per_second":2297.832602762276,"tflops_per_second":0.012992178618666618},'
    '"torch_float32":{"compute_bound":false,"fraction_of_fma":0.00275207518316772'
    '46,"fraction_of_roofline":0.011175515840927148,"intensity_flops_per_byte":10'
    '.666666666666666,"iqr_tflops_per_second":[0.010717964705341216,0.01313285009'
    '7847358],"median_ns_per_call":40833.25,"roofline_gflops_per_second":1148.916'
    '301381138,"tflops_per_second":0.012839732325984338}},"size":64,"steps":[{"fi'
    'rst":"m1","held":false,"iqr_separated":false,"predicted":["faster"],"ran":tr'
    'ue,"ratio":1.0498845265588914,"second":"s1","step":"m1_over_s1","verdict":"n'
    'o_clear_difference"},{"first":"m2","held":true,"iqr_separated":false,"predic'
    'ted":["faster","no_clear_difference"],"ran":true,"ratio":1.0489341085271318,'
    '"second":"m1","step":"m2_over_m1","verdict":"no_clear_difference"},{"first":'
    '"m3","held":true,"iqr_separated":false,"predicted":["faster","no_clear_diffe'
    'rence"],"ran":true,"ratio":1.0504609412906356,"second":"m1","step":"m3_over_'
    'm1","verdict":"no_clear_difference"},{"first":"m3","held":true,"iqr_separate'
    'd":true,"predicted":["faster"],"ran":true,"ratio":1.1028626880155263,"second'
    '":"s1","step":"m3_over_s1","verdict":"faster"},{"first":"torch_float16","hel'
    'd":true,"iqr_separated":false,"predicted":["faster","no_clear_difference"],"'
    'ran":true,"ratio":1.0118730117429136,"second":"torch_float32","step":"torch_'
    'float16_over_torch_float32","verdict":"no_clear_difference"},{"first":"torch'
    '_bfloat16","held":true,"iqr_separated":false,"predicted":["faster","no_clear'
    '_difference"],"ran":true,"ratio":1.0299495218607233,"second":"torch_float32"'
    ',"step":"torch_bfloat16_over_torch_float32","verdict":"no_clear_difference"}'
    ',{"first":"m1","held":false,"iqr_separated":true,"predicted":["slower"],"ran'
    '":true,"ratio":1.207079907621247,"second":"torch_float32","step":"m1_over_to'
    'rch_float32","verdict":"faster"},{"first":"m2","held":false,"iqr_separated":'
    'true,"predicted":["slower"],"ran":true,"ratio":1.2512906976744187,"second":"'
    'torch_float16","step":"m2_over_torch_float16","verdict":"faster"},{"first":"'
    'm3","held":false,"iqr_separated":true,"predicted":["slower"],"ran":true,"rat'
    'io":1.231118874332848,"second":"torch_bfloat16","step":"m3_over_torch_bfloat'
    '16","verdict":"faster"}]},{"index":1,"rates":{"m1":{"compute_bound":false,"f'
    'raction_of_fma":0.027148409253756244,"fraction_of_roofline":0.02756079117378'
    '128,"intensity_flops_per_byte":42.666666666666664,"iqr_tflops_per_second":[0'
    '.1127565485931656,0.1297837944465288],"median_ns_per_call":264917.0,"rooflin'
    'e_gflops_per_second":4595.665205524552,"tflops_per_second":0.126660169034074'
    '82},"m2":{"compute_bound":true,"fraction_of_fma":0.027564609183293322,"fract'
    'ion_of_roofline":0.027564609183293322,"intensity_flops_per_byte":64.0,"iqr_t'
    'flops_per_second":[0.11469992924068763,0.13177980166912126],"median_ns_per_c'
    'all":260917.0,"roofline_gflops_per_second":4665.472950926219,"tflops_per_sec'
    'ond":0.12860193854750746},"m3":{"compute_bound":true,"fraction_of_fma":0.026'
    '915642998253584,"fraction_of_roofline":0.026915642998253584,"intensity_flops'
    '_per_byte":64.0,"iqr_tflops_per_second":[0.11562200904179072,0.1582442723611'
    '36],"median_ns_per_call":267208.0,"roofline_gflops_per_second":4665.47295092'
    '6219,"tflops_per_second":0.12557420436513878},"s1":{"compute_bound":false,"f'
    'raction_of_fma":0.023625734220747673,"fraction_of_roofline":0.02398460702058'
    '2346,"intensity_flops_per_byte":42.666666666666664,"iqr_tflops_per_second":['
    '0.10473486383144753,0.1534211460028897],"median_ns_per_call":304417.0,"roofl'
    'ine_gflops_per_second":4595.665205524552,"tflops_per_second":0.1102252239526'
    '7019},"torch_bfloat16":{"compute_bound":true,"fraction_of_fma":0.03431611883'
    '7297606,"fraction_of_roofline":0.034316118837297606,"intensity_flops_per_byt'
    'e":85.33333333333333,"iqr_tflops_per_second":[0.13596239748452138,0.16371289'
    '867729644],"median_ns_per_call":209583.0,"roofline_gflops_per_second":4665.4'
    '72950926219,"tflops_per_second":0.16010092421618166},"torch_float16":{"compu'
    'te_bound":true,"fraction_of_fma":0.032451551649305556,"fraction_of_roofline"'
    ':0.032451551649305556,"intensity_flops_per_byte":85.33333333333333,"iqr_tflo'
    'ps_per_second":[0.14465736038420748,0.1619359870275279],"median_ns_per_call"'
    ':221625.0,"roofline_gflops_per_second":4665.472950926219,"tflops_per_second"'
    ':0.15140183643542018},"torch_float32":{"compute_bound":false,"fraction_of_fm'
    'a":0.029591902363695754,"fraction_of_roofline":0.030041400726560082,"intensi'
    'ty_flops_per_byte":42.666666666666664,"iqr_tflops_per_second":[0.13143550130'
    '830578,0.14628635204359675],"median_ns_per_call":243042.0,"roofline_gflops_p'
    'er_second":4595.665205524552,"tflops_per_second":0.13806022004427218}},"size'
    '":256,"steps":[{"first":"m1","held":false,"iqr_separated":false,"predicted":'
    '["faster"],"ran":true,"ratio":1.1491033040537224,"second":"s1","step":"m1_ov'
    'er_s1","verdict":"no_clear_difference"},{"first":"m2","held":true,"iqr_separ'
    'ated":false,"predicted":["faster","no_clear_difference"],"ran":true,"ratio":'
    '1.0153305457291015,"second":"m1","step":"m2_over_m1","verdict":"no_clear_dif'
    'ference"},{"first":"m3","held":true,"iqr_separated":false,"predicted":["fast'
    'er","no_clear_difference"],"ran":true,"ratio":0.9914261549055418,"second":"m'
    '1","step":"m3_over_m1","verdict":"no_clear_difference"},{"first":"m3","held"'
    ':false,"iqr_separated":false,"predicted":["faster"],"ran":true,"ratio":1.139'
    '2510703272356,"second":"s1","step":"m3_over_s1","verdict":"no_clear_differen'
    'ce"},{"first":"torch_float16","held":true,"iqr_separated":false,"predicted":'
    '["faster","no_clear_difference"],"ran":true,"ratio":1.0966362098138749,"seco'
    'nd":"torch_float32","step":"torch_float16_over_torch_float32","verdict":"no_'
    'clear_difference"},{"first":"torch_bfloat16","held":true,"iqr_separated":fal'
    'se,"predicted":["faster","no_clear_difference"],"ran":true,"ratio":1.1596455'
    '819412834,"second":"torch_float32","step":"torch_bfloat16_over_torch_float32'
    '","verdict":"no_clear_difference"},{"first":"m1","held":false,"iqr_separated'
    '":true,"predicted":["slower"],"ran":true,"ratio":0.9174269676917676,"second"'
    ':"torch_float32","step":"m1_over_torch_float32","verdict":"no_clear_differen'
    'ce"},{"first":"m2","held":true,"iqr_separated":true,"predicted":["slower"],"'
    'ran":true,"ratio":0.849408049303035,"second":"torch_float16","step":"m2_over'
    '_torch_float16","verdict":"slower"},{"first":"m3","held":false,"iqr_separate'
    'd":false,"predicted":["slower"],"ran":true,"ratio":0.7843440316158199,"secon'
    'd":"torch_bfloat16","step":"m3_over_torch_bfloat16","verdict":"no_clear_diff'
    'erence"}]}],"ratio_rule":"ratio = median sample time of the second side / me'
    "dian sample time of the first side, from the same point of the same run with"
    " the same calls per sample; faster when ratio >= min_faster_ratio and the fi"
    "rst side's p75 sample time is below the second side's p25; slower when 1 / r"
    "atio >= min_faster_ratio and the second side's p75 is below the first side's"
    ' p25; otherwise no_clear_difference","ridge_flops_per_byte":43.3147695354796'
    '4,"roofline_rule":"roofline GFLOP/s = min(F, I B), with F the FMA ceiling in'
    " GFLOP/s, I the compulsory intensity of the side in FLOP/byte and B the band"
    'width ceiling in GB/s; a side is compute-bound by this model when I B >= F",'
    '"vendor":"torch.matmul(a, b) on m x k and k x n tensors of the side\'s dtype '
    'on the MPS device; the output has the dtype of the inputs, PyTorch 2.15.0"},'
    '"summary":{"bandwidth_gb_per_second":107.7109032544817,"benchmark_points":2,'
    '"correctness_cases":5,"fma_gflops_per_second":4665.472950926219,"hypotheses"'
    ':{"m1_over_s1":{"held":0,"missed":2,"not_run":0},"m1_over_torch_float32":{"h'
    'eld":0,"missed":2,"not_run":0},"m2_over_m1":{"held":2,"missed":0,"not_run":0'
    '},"m2_over_torch_float16":{"held":1,"missed":1,"not_run":0},"m3_over_m1":{"h'
    'eld":2,"missed":0,"not_run":0},"m3_over_s1":{"held":1,"missed":1,"not_run":0'
    '},"m3_over_torch_bfloat16":{"held":0,"missed":2,"not_run":0},"torch_bfloat16'
    '_over_torch_float32":{"held":2,"missed":0,"not_run":0},"torch_float16_over_t'
    'orch_float32":{"held":2,"missed":0,"not_run":0}},"identical_repeats":{"m1":t'
    'rue,"m2":true,"m3":true,"s1":true},"label":"host-specific Apple GPU (Metal) '
    'measurement","load_average_after":[4.79638671875,4.86669921875,5.21923828125'
    '],"load_average_before":[4.79638671875,4.86669921875,5.21923828125],"m1_same'
    '_bits_as_s1_cases":5,"max_bound_ratio":{"m1":0.30476404457829903,"m2":0.3664'
    '536263750756,"m3":0.4807872607474941,"s1":0.30476404457829903},"max_relative'
    '_error":{"m1":2.4966670630818563e-07,"m2":0.000413331606162518,"m3":0.003763'
    '631113404673,"s1":2.4966670630818563e-07},"metal_language_version":400,"poin'
    'ts":[{"fraction_of_fma":{"m1":0.003321974657864824,"m2":0.003484532526297163'
    ',"m3":0.00348960462604432,"s1":0.003164133363078462,"torch_bfloat16":0.00283'
    '44985190283606,"torch_float16":0.002784750604134856,"torch_float32":0.002752'
    '0751831677246},"index":0,"m1_same_bits_as_s1":true,"size":64,"tflops_per_sec'
    'ond":{"m1":0.015498582909930716,"m2":0.016256992248062017,"m3":0.01628065599'
    '2236777,"s1":0.014762178618565772,"torch_bfloat16":0.013224276169967242,"tor'
    'ch_float16":0.012992178618666618,"torch_float32":0.012839732325984338},"torc'
    'h_float32_same_bits_as_s1":true,"verdicts":{"m1_over_s1":"no_clear_differenc'
    'e","m1_over_torch_float32":"faster","m2_over_m1":"no_clear_difference","m2_o'
    'ver_torch_float16":"faster","m3_over_m1":"no_clear_difference","m3_over_s1":'
    '"faster","m3_over_torch_bfloat16":"faster","torch_bfloat16_over_torch_float3'
    '2":"no_clear_difference","torch_float16_over_torch_float32":"no_clear_differ'
    'ence"}},{"fraction_of_fma":{"m1":0.027148409253756244,"m2":0.027564609183293'
    '322,"m3":0.026915642998253584,"s1":0.023625734220747673,"torch_bfloat16":0.0'
    '34316118837297606,"torch_float16":0.032451551649305556,"torch_float32":0.029'
    '591902363695754},"index":1,"m1_same_bits_as_s1":true,"size":256,"tflops_per_'
    'second":{"m1":0.12666016903407482,"m2":0.12860193854750746,"m3":0.1255742043'
    '6513878,"s1":0.11022522395267019,"torch_bfloat16":0.16010092421618166,"torch'
    '_float16":0.15140183643542018,"torch_float32":0.13806022004427218},"torch_fl'
    'oat32_same_bits_as_s1":true,"verdicts":{"m1_over_s1":"no_clear_difference","'
    'm1_over_torch_float32":"no_clear_difference","m2_over_m1":"no_clear_differen'
    'ce","m2_over_torch_float16":"slower","m3_over_m1":"no_clear_difference","m3_'
    'over_s1":"no_clear_difference","m3_over_torch_bfloat16":"no_clear_difference'
    '","torch_bfloat16_over_torch_float32":"no_clear_difference","torch_float16_o'
    'ver_torch_float32":"no_clear_difference"}}],"supported":{"m3":true,"torch_bf'
    'loat16":true},"timed_samples":98,"warnings":0},"support":{"metal_language_ve'
    'rsion":400,"rule":"M3 is supported when the bfloat library compiles and C = '
    "A @ B for 8 x 8 integers A[i][j] = (3 i + j) mod 5 - 2 and B[i][j] = (i + 2 "
    "j) mod 5 - 2 is exact; the bfloat16 PyTorch call is supported when the same "
    'product on bfloat16 tensors is exact","sides":{"m3":{"reason":null,"supporte'
    'd":true},"torch_bfloat16":{"reason":null,"supported":true}}},"timer":"time.p'
    'erf_counter_ns","torch_mps_switches":{"source":"os.environ, read before the '
    "runtime imports torch; null means unset, and a value that is not a short tok"
    'en is \'unknown\'","values":{"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_'
    'METAL":null}},"verification":{"gpu_only":["the samples_ns of each side: chec'
    'ked for form only, never observed again","each kernel output digest, max_abs'
    "_error, max_bound_ratio and max_relative_error: max_bound_ratio is checked t"
    "o be at most 1, and the errors are checked against the recomputed bounds and"
    ' scale, not recomputed","identical_repeats, m1_same_bits_as_s1 and torch_flo'
    "at32_same_bits_as_s1: checked for form and for agreement with the digests an"
    'd errors only","the Metal language version and the support of M3 and of the '
    'bfloat16 PyTorch call: checked for form only","the pipeline limits: checked '
    'for form, then used to rebuild the admissions","the PyTorch side checks, the'
    ' warnings, the MPS switches and the load averages: checked for form only"],"'
    'recomputed":["the config, its digest and the planned work","the declared pla'
    'n, every kernel digest and both library digests","every operand digest in ea'
    "ch input type, reference digest, scale digest and bound digest, from the see"
    'd","each launch, and each admission from the recorded limits","each sample s'
    "tatistic, TFLOP/s, GB/s, intensity, roofline, ceiling fraction, ratio, verdi"
    'ct and held hypothesis, from the raw samples","the run totals, the summary a'
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
        4.79638671875,
        4.86669921875,
        5.21923828125,
    ]
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (5, 2)


def test_the_verifier_needs_no_blas_product_of_the_host(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the verifier must not use a BLAS product")

    monkeypatch.setattr(numpy, "matmul", refuse)
    monkeypatch.setattr(numpy, "dot", refuse)
    monkeypatch.setattr(numpy, "einsum", refuse)

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


def test_the_apple_gpu_kept_its_bounds_bits_and_bfloat_support():
    report = recorded()
    summary = report["summary"]

    assert summary["supported"] == {"m3": True, "torch_bfloat16": True}
    assert summary["metal_language_version"] == 400
    assert summary["identical_repeats"] == dict.fromkeys(("s1", "m1", "m2", "m3"), True)
    assert max(summary["max_bound_ratio"].values()) <= 1
    assert summary["m1_same_bits_as_s1_cases"] == 5
    errors = summary["max_relative_error"]
    assert errors["m3"] > errors["m2"] > errors["m1"] == errors["s1"]
    for entry in report["shaders"]["kernels"]:
        assert entry["limits"]["thread_execution_width"] == 32
        assert entry["limits"]["static_threadgroup_memory_bytes"] == (
            KERNELS[entry["name"]].threadgroup_bytes
        )


@pytest.mark.parametrize(
    "path",
    [
        ("summary", "points", 0, "verdicts", "m1_over_s1"),
        ("speedup", "points", 1, "steps", 0, "held"),
        ("environment", "macos_version"),
        ("support", "sides", "m3", "supported"),
        ("correctness", "cases", 4, "m1_same_bits_as_s1"),
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

    with pytest.raises(MetalTcVerificationError):
        verify_report(seal_report(report))


def test_the_command_verifies_the_recorded_file_without_torch(tmp_path, capsys):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")

    assert main(["verify", "--report", str(path)]) == EXIT_OK
    expected = capsys.readouterr().out
    assert expected.startswith(
        f"verified_metal_tc_measurement={recorded()['report_sha256']} "
    )
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_tc_cli import main\n"
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
    path.write_text(RECORDED_REPORT_JSON.replace('"size":64', '"size": 64') + "\n")
    assert main(["verify", "--report", str(path)]) == EXIT_REJECTED


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    runtime = tc_runtime()

    for entry in recorded()["shaders"]["kernels"]:
        assert runtime.limits[entry["name"]].to_mapping() == entry["limits"]
