"""One real Day 80 report from the development host, verified on any host.

``RECORDED_REPORT_JSON`` is the canonical file that ``metal_liger_cli run``
wrote in a fresh process on 2026-11-23 at 16:02 IST under the shared GPU
lock, with torch 2.15.0 on macOS 27.2.1 (arm64). The ``uptime`` command
printed load averages of 6.94, 7.75 and 7.60 before and after the run, from
other jobs. The run is short: four correctness cases, three small timing
points, four repeats and two calls for each sample, so the cost of each
synchronize call sets most of its rates, and two of its three predictions
missed. Its ceiling is a fresh Day 74 gpu-ceilings report of the same day,
124.61 GB/s at 256 MiB. The timings describe that moment and are here only
to be verified. A host without torch or MPS can still check a genuine Apple
GPU (Metal) report. The reference uses ``exp64`` and ``log64``, not the
libm of the host, so another libm gives the same digests.
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
from ai_journey.metal_liger_cli import EXIT_OK, EXIT_REJECTED, main
from ai_journey.metal_liger_config import config_from_mapping
from ai_journey.metal_liger_plan import plan_mapping, plan_sha256
from ai_journey.metal_liger_source import FAMILIES, KERNELS
from ai_journey.metal_liger_verify import MetalLigerVerificationError, verify_report
from test_kernel_metal_runtime import requires_metal
from test_metal_liger_runtime import liger_runtime

ROOT = Path(__file__).resolve().parents[1]
CEILING_SHA256 = "93fefcf83d3561e07fa135281b02d071b2073ef98c2e718c8317a8585acf0f81"
RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "metal-liger-recorded",
    "seed": 80,
    "plan_sha256": "43d5f63eb9d93415d0b95244f7150d0652aa73a7845d98da7658993ff3f13165",
    "correctness": {
        "cases": {
            "rmsnorm": [{"lengths": [1, 31, 33, 300], "inputs": "uniform_80"}],
            "swiglu": [{"lengths": [1, 31, 33, 300], "inputs": "offset_minus_1000"}],
            "cross_entropy": [
                {"lengths": [1, 31, 33, 16400], "inputs": "uniform_80"},
                {"rows": 2, "cols": 64, "inputs": "offset_plus_1000"},
            ],
        },
        "repeats": 2,
    },
    "benchmark": {
        "inputs": "normal",
        "shapes": {
            "rmsnorm": [{"rows": 64, "cols": 300}],
            "swiglu": [{"rows": 64, "cols": 300}],
            "cross_entropy": [{"rows": 16, "cols": 2049}],
        },
        "warmup": 1,
        "repeats": 4,
        "target_bytes_per_sample": 262144,
        "max_runtime_seconds": 60,
    },
    "comparison": {"min_faster_ratio": 1.1},
}
# The canonical report of that run, byte for byte.
RECORDED_REPORT_JSON = (
    '{"backend":"torch.mps.compile_shader","benchmark":{"compulsory":{"cross_en'
    'tropy":"4 n + 8 r bytes: read each logit once, read one 4-byte target and '
    'write one loss for each row","rmsnorm":"8 n + 4 H bytes: read x once, writ'
    'e y once, read the H weights once","swiglu":"12 n bytes: read a and b once'
    ', write y once"},"device":"mps","inputs":"normal","kernels":{"cross_entrop'
    'y":"liger_cross_entropy","rmsnorm":"liger_rmsnorm","swiglu":"liger_swiglu"'
    '},"points":[{"checks":{"liger_rmsnorm":{"differing_from_rounded_reference"'
    ':8067,"max_abs_error":9.238182263615613e-07,"max_bound_ratio":0.1174676336'
    '3374532},"torch_rmsnorm":{"differing_from_rounded_reference":8598,"max_abs'
    '_error":9.238182263615613e-07,"max_bound_ratio":0.01261864174328973}},"com'
    'pulsory_bytes_per_call":154800,"family":"rmsnorm","index":0,"launch":{"act'
    'ive_threads":2048,"extent":[2048],"group_size":[32],"groups":[64],"launche'
    'd_threads":2048,"padding_threads":0,"threads":[2048],"threads_per_group":3'
    '2},"layout":{"buffer_elements":19200,"dense":true,"elements":19200,"gap":0'
    ',"group_size":32,"lengths_sha256":"58eea0764bb07ddbf3c6d3778646ee9aa2cb357'
    '8137523cda833a29396711615","longest":300,"output_elements":19200,"rows":64'
    ',"shortest":300},"main_sha256":"0c3b4f91aeff7f561120a0ebca098a64846c70380b'
    '11bfb709629b5b30c53793","reference_sha256":"5adaae832b6b768443eb0cb2533aa9'
    'f3e7286f24f0bcc3b5b69a5498a4c99e5c","second_sha256":"1b15be70bb14333dd803d'
    'c4cad6f894b585905b76e46db44a441224e19c6719e","shape":{"cols":300,"rows":64'
    '},"side_order":["liger_rmsnorm","torch_rmsnorm"],"sides":{"liger_rmsnorm":'
    '{"calls_per_sample":2,"compulsory_bytes_per_call":154800,"effective_gb_per'
    '_second":1.3829145728643215,"implementation":"liger_rmsnorm","iqr_effectiv'
    'e_gb_per_second":[0.5632931544234705,1.51919604302426],"iqr_ns_per_call":1'
    '72916.5,"median_ns_per_call":111937.5,"modeled_bytes_per_call":230912,"mod'
    'eled_gb_per_second":2.0628654383026244,"repeats":4,"samples_ns":[549625,22'
    '3875,203792,1333708],"timing":{"count":4,"iqr_ns":345833,"max_ns":1333708,'
    '"median_ns":223875,"min_ns":203792,"p25_ns":203792,"p75_ns":549625},"warmu'
    'p":1},"torch_rmsnorm":{"calls_per_sample":2,"compulsory_bytes_per_call":15'
    '4800,"effective_gb_per_second":0.8644012429955859,"implementation":"torch_'
    'rmsnorm","iqr_effective_gb_per_second":[0.7982817346641466,0.8682410546773'
    '122],"iqr_ns_per_call":15625.0,"median_ns_per_call":179083.5,"modeled_byte'
    's_per_call":769280,"modeled_gb_per_second":4.295649794648865,"repeats":4,"'
    'samples_ns":[636208,387833,356583,358167],"timing":{"count":4,"iqr_ns":312'
    '50,"max_ns":636208,"median_ns":358167,"min_ns":356583,"p25_ns":356583,"p75'
    '_ns":387833},"warmup":1}},"unwritten_softmax_bytes":null},{"checks":{"lige'
    'r_swiglu":{"differing_from_rounded_reference":7883,"max_abs_error":5.47281'
    '0071793788e-07,"max_bound_ratio":0.29264143269905407},"torch_swiglu":{"dif'
    'fering_from_rounded_reference":7883,"max_abs_error":5.472810071793788e-07,'
    '"max_bound_ratio":0.012574443954031905}},"compulsory_bytes_per_call":23040'
    '0,"family":"swiglu","index":0,"launch":{"active_threads":2048,"extent":[20'
    '48],"group_size":[32],"groups":[64],"launched_threads":2048,"padding_threa'
    'ds":0,"threads":[2048],"threads_per_group":32},"layout":{"buffer_elements"'
    ':19200,"dense":true,"elements":19200,"gap":0,"group_size":32,"lengths_sha2'
    '56":"58eea0764bb07ddbf3c6d3778646ee9aa2cb3578137523cda833a29396711615","lo'
    'ngest":300,"output_elements":19200,"rows":64,"shortest":300},"main_sha256"'
    ':"39ce2dfadbe784e0bd9f5443c26baec026f8c24b6fad3e99c8bbac304ea4e534","refer'
    'ence_sha256":"004b9c970585f11875d5f71b3f44cfe502420d5d6afa1f980d50786a00d8'
    '1e6d","second_sha256":"f72eb00989c831418698dc36233f30abc8365a135263266410d'
    '30583399b035e","shape":{"cols":300,"rows":64},"side_order":["liger_swiglu"'
    ',"torch_swiglu"],"sides":{"liger_swiglu":{"calls_per_sample":2,"compulsory'
    '_bytes_per_call":230400,"effective_gb_per_second":2.1403489214648013,"impl'
    'ementation":"liger_swiglu","iqr_effective_gb_per_second":[1.86432654844700'
    '15,2.204784688995215],"iqr_ns_per_call":19083.5,"median_ns_per_call":10764'
    '6.0,"modeled_bytes_per_call":230912,"modeled_gb_per_second":2.145105252401'
    '3896,"repeats":4,"samples_ns":[247167,215292,209000,393541],"timing":{"cou'
    'nt":4,"iqr_ns":38167,"max_ns":393541,"median_ns":215292,"min_ns":209000,"p'
    '25_ns":209000,"p75_ns":247167},"warmup":1},"torch_swiglu":{"calls_per_samp'
    'le":2,"compulsory_bytes_per_call":230400,"effective_gb_per_second":1.92334'
    '18063886,"implementation":"torch_swiglu","iqr_effective_gb_per_second":[1.'
    '7805874238284949,2.0288387451843697],"iqr_ns_per_call":15833.0,"median_ns_'
    'per_call":119791.5,"modeled_bytes_per_call":384000,"modeled_gb_per_second"'
    ':3.2055696773143336,"repeats":4,"samples_ns":[239583,227125,265708,258791]'
    ',"timing":{"count":4,"iqr_ns":31666,"max_ns":265708,"median_ns":239583,"mi'
    'n_ns":227125,"p25_ns":227125,"p75_ns":258791},"warmup":1}},"unwritten_soft'
    'max_bytes":null},{"checks":{"liger_cross_entropy":{"differing_from_rounded'
    '_reference":8,"max_abs_error":6.296722290244361e-07,"max_bound_ratio":0.04'
    '094727432007379},"torch_cross_entropy":{"differing_from_rounded_reference"'
    ':7,"max_abs_error":7.026280393063189e-07,"max_bound_ratio":0.0057323142517'
    '75714}},"compulsory_bytes_per_call":131264,"family":"cross_entropy","index'
    '":0,"launch":{"active_threads":4096,"extent":[4096],"group_size":[256],"gr'
    'oups":[16],"launched_threads":4096,"padding_threads":0,"threads":[4096],"t'
    'hreads_per_group":256},"layout":{"buffer_elements":32784,"dense":true,"ele'
    'ments":32784,"gap":0,"group_size":256,"lengths_sha256":"3f4b7ee126a668ea7c'
    'c5b7998a60b2b057d7d0e053c3483f7d12d1f19ee374c8","longest":2049,"output_ele'
    'ments":16,"rows":16,"shortest":2049},"main_sha256":"f011b0c9ad6f3a0f8b16b7'
    'c63e6d5504ee63be46b425f8716e15698fff2d611f","reference_sha256":"7fa48943ee'
    'd9b9f987d351b296d4c4e23e32cf7539e24f40e9150abe02ea2a5e","second_sha256":"2'
    '344abe9e3bd92cb53cd9b3774c5771abe701d8eb688e26dd4b6af9520097833","shape":{'
    '"cols":2049,"rows":16},"side_order":["liger_cross_entropy","torch_cross_en'
    'tropy"],"sides":{"liger_cross_entropy":{"calls_per_sample":2,"compulsory_b'
    'ytes_per_call":131264,"effective_gb_per_second":1.3532371134020618,"implem'
    'entation":"liger_cross_entropy","iqr_effective_gb_per_second":[0.927388787'
    '0341914,1.4171551956815114],"iqr_ns_per_call":48916.5,"median_ns_per_call"'
    ':97000.0,"modeled_bytes_per_call":131392,"modeled_gb_per_second":1.3545567'
    '010309278,"repeats":4,"samples_ns":[194000,283083,351333,185250],"timing":'
    '{"count":4,"iqr_ns":97833,"max_ns":351333,"median_ns":194000,"min_ns":1852'
    '50,"p25_ns":185250,"p75_ns":283083},"warmup":1},"torch_cross_entropy":{"ca'
    'lls_per_sample":2,"compulsory_bytes_per_call":131264,"effective_gb_per_sec'
    'ond":0.9140692459820061,"implementation":"torch_cross_entropy","iqr_effect'
    'ive_gb_per_second":[0.6643486534772728,1.0063363679923334],"iqr_ns_per_cal'
    'l":67145.5,"median_ns_per_call":143604.0,"modeled_bytes_per_call":262528,"'
    'modeled_gb_per_second":1.8281384919640122,"repeats":4,"samples_ns":[287208'
    ',395166,404666,260875],"timing":{"count":4,"iqr_ns":134291,"max_ns":404666'
    ',"median_ns":287208,"min_ns":260875,"p25_ns":260875,"p75_ns":395166},"warm'
    'up":1}},"unwritten_softmax_bytes":131136}],"synchronization":"torch.mps.sy'
    "nchronize() before the first and before the second clock reading of every "
    "sample; untimed warmups of each side first; repeat r starts with side r mo"
    'd S, so the order shifts by one side every repeat","torch_sequences":{"cro'
    'ss_entropy":"torch.nn.functional.cross_entropy(logits, targets, reduction='
    '\'none\') with int64 targets, on the MPS device","rmsnorm":"ms = x.pow(2).me'
    "an(-1, keepdim=True); inv = torch.rsqrt(ms + eps); torch.mul(x * inv, w, o"
    'ut=y), on the MPS device","swiglu":"torch.mul(torch.nn.functional.silu(a),'
    ' b, out=y), on the MPS device"},"traffic_model":{"cross_entropy":"device b'
    "ytes of one call with no cache, for n logits in r rows: the kernel reads e"
    "ach logit once, 4 n, and reads a 4-byte target, starts and lengths and wri"
    "tes a loss for each row, 16 r; the PyTorch call is modeled as log_softmax,"
    " which reads n logits and writes n log-probabilities, then nll_loss, which"
    " reads r of them, r 8-byte targets and writes r losses, 8 n + 16 r; PyTorc"
    "h does not name its MPS kernels, so this side of the model is an assumptio"
    'n","rmsnorm":"device bytes of one call with no cache, for n elements in r '
    "rows, where a broadcast operand is read once for each output element: the "
    "kernel reads x and w and writes y, 12 n, plus 8 r of starts and lengths; t"
    "he PyTorch sequence runs pow (8 n), mean (4 n + 4 r), the add of eps (8 r)"
    ', rsqrt (8 r), x * inv (12 n) and * w (12 n), 40 n + 20 r","swiglu":"devic'
    "e bytes of one call with no cache, for n elements in r rows: the kernel re"
    "ads a and b and writes y, 12 n, plus 8 r of starts and lengths; the PyTorc"
    'h sequence runs silu (8 n) and the multiply (12 n), 20 n"},"unwritten_soft'
    'max":"4 n bytes for n logits: the float32 softmax or log-probabilities tha'
    't the fused cross-entropy kernel never writes and never reads again","vend'
    'or_bounds":{"cross_entropy":"|loss - loss_ref| <= 2**-16 (loss_ref + 1): a'
    ' sanity bound only","rmsnorm":"|y - y_ref| <= 2**-16 |y_ref| + 2**-100: a '
    'sanity bound only","swiglu":"|y - y_ref| <= 2**-16 |y_ref| + 2**-100: a sa'
    'nity bound only"}},"ceiling":{"buffer_bytes":268435456,"config_sha256":"b3'
    'f1096b3691a46d5a77068639ba57389194a3d07a06bfc16f5693cf52215199","definitio'
    'n":"the bandwidth ceiling of a verified Day 74 gpu-ceilings report: the be'
    "tter median GB/s of stream_copy and stream_read_reduce at its largest buff"
    'er, on the host that made it; not a vendor peak and not a hardware limit",'
    '"environment":{"macos_version":"27.2.1","numpy_version":"1.26.4","platform'
    '_machine":"arm64","platform_system":"Darwin","python_implementation":"CPyt'
    'hon","python_version":"3.11.14","torch_version":"2.15.0"},"gb_per_second":'
    '124.60522827782013,"kernel":"stream_read_reduce","label":"host-specific Ap'
    'ple GPU (Metal) measurement","load_average_after":[5.7998046875,6.65478515'
    '625,6.744140625],"load_average_before":[5.61181640625,6.6708984375,6.75244'
    '140625],"measured":true,"report_sha256":"93fefcf83d3561e07fa135281b02d071b'
    '2073ef98c2e718c8317a8585acf0f81","source":"gpu_ceilings"},"config":{"bench'
    'mark":{"inputs":"normal","max_runtime_seconds":60,"repeats":4,"shapes":{"c'
    'ross_entropy":[{"cols":2049,"rows":16}],"rmsnorm":[{"cols":300,"rows":64}]'
    ',"swiglu":[{"cols":300,"rows":64}]},"target_bytes_per_sample":262144,"warm'
    'up":1},"comparison":{"min_faster_ratio":1.1},"correctness":{"cases":{"cros'
    's_entropy":[{"inputs":"uniform_80","lengths":[1,31,33,16400]},{"cols":64,"'
    'inputs":"offset_plus_1000","rows":2}],"rmsnorm":[{"inputs":"uniform_80","l'
    'engths":[1,31,33,300]}],"swiglu":[{"inputs":"offset_minus_1000","lengths":'
    '[1,31,33,300]}]},"repeats":2},"name":"metal-liger-recorded","plan_sha256":'
    '"43d5f63eb9d93415d0b95244f7150d0652aa73a7845d98da7658993ff3f13165","schema'
    '_version":1,"seed":80},"config_sha256":"a1a2afcc54c5c5c4acadc0cc3ed3480074'
    '89243d7286d17acf762fd3fb897186","correctness":{"bounds":{"cross_entropy":"'
    "|loss - loss_ref| <= i + u (loss_ref + i) + R + 2**-125 with i = u m + t +"
    " LOG_ERROR (log S + t), m = max - x[t], t = rho / (1 - rho), rho = exp(N u"
    " D) (1 + EXP_ERROR)**N (1 + gamma_A) - 1 + (n + 2 G) 2**-126, D = max - mi"
    "n of the row, k = ceil(n / G), N = k + 2, A = k + 1 + 31 + 1 + (G / 32 - 1"
    "), LOG_ERROR = 4 * 2**-23 and R = gamma_(n + 8) (1 + 2 loss_ref) for the u"
    'nit 2**-53","rmsnorm":"|y - y_ref| <= (rel + ref) / (1 - ref) * |y_ref| + '
    "2**-125 with rel = (1 + c) (1 + RSQRT_ERROR) (1 + u)**2 - 1, c = 1 / sqrt("
    "1 - g) - 1, g = gamma_(A + 2), A = k + 31 + (G / 32 - 1) fma and add round"
    "ings of the sum of squares, k = ceil(n / G), RSQRT_ERROR = 2 * 2**-23 and "
    'ref = gamma_(n + 8) for the unit 2**-53","swiglu":"|y - y_ref| <= (rel + r'
    "ef) / (1 - ref) * |y_ref| + 2**-120 |b| + 2**-125 with rel = (1 + e / (1 -"
    " e)) (1 + u)**2 - 1, e = (1 + EXP_ERROR) (1 + u) - 1, EXP_ERROR = 4 * 2**-"
    "23 and ref = 16 * 2**-53; the 2**-120 |b| term covers exp(-a) overflow for"
    ' a below -88.7, where the kernel returns 0"},"cases":[{"bound_sha256":"283'
    'f3cc7dfaac29d345df564174d5bf991fa39284928e44ec1d1f51dcbefc576","case":{"in'
    'puts":"uniform_80","lengths":[1,31,33,300]},"family":"rmsnorm","index":0,"'
    'launch":{"active_threads":128,"extent":[128],"group_size":[32],"groups":[4'
    '],"launched_threads":128,"padding_threads":0,"threads":[128],"threads_per_'
    'group":32},"layout":{"buffer_elements":493,"dense":false,"elements":365,"g'
    'ap":32,"group_size":32,"lengths_sha256":"6affeba37d01ef9b7f904d76414054b40'
    '4bfb57bbe3b9d8d476320f49ad3f648","longest":300,"output_elements":493,"rows'
    '":4,"shortest":1},"main_sha256":"124645262eb59d5d4d1e47df30d4085afbd8aaf73'
    '20a9b8fcfffacd711e7306a","naive_float32_finite":null,"output":{"buffer_sha'
    '256":"df258439d8d35bbbb6998dc15befdc7d5353aa2933b88791b15c531dbd772e16","d'
    'iffering_from_rounded_reference":181,"identical_repeats":true,"max_abs_err'
    'or":2.4266339337941645e-07,"max_bound_ratio":0.08200084972074735},"referen'
    'ce_sha256":"c7ba67cfee1065b1c4b5fd49dacc33eff96edb8946df63b18047b7c002b929'
    '87","repeats":2,"second_sha256":"3df4bd41a662b8340c8595d7df33a249f29952e9c'
    'ef237aac560047db69d8fa3"},{"bound_sha256":"3b6457c6a07e63b3a8171daf0c80f2e'
    '34a9cba9aedb686d7091fafe037cc9607","case":{"inputs":"offset_minus_1000","l'
    'engths":[1,31,33,300]},"family":"swiglu","index":0,"launch":{"active_threa'
    'ds":128,"extent":[128],"group_size":[32],"groups":[4],"launched_threads":1'
    '28,"padding_threads":0,"threads":[128],"threads_per_group":32},"layout":{"'
    'buffer_elements":493,"dense":false,"elements":365,"gap":32,"group_size":32'
    ',"lengths_sha256":"6affeba37d01ef9b7f904d76414054b404bfb57bbe3b9d8d476320f'
    '49ad3f648","longest":300,"output_elements":493,"rows":4,"shortest":1},"mai'
    'n_sha256":"c9bcf5692b9869df15c2d961f3e184c4de6155b3cc2524ad9b71c387da76171'
    'd","naive_float32_finite":null,"output":{"buffer_sha256":"119d839f2efd7eaa'
    'c9a5556fcfa3b907c3250e2ba10677bfd8aab7c90a9644e0","differing_from_rounded_'
    'reference":0,"identical_repeats":true,"max_abs_error":0.0,"max_bound_ratio'
    '":0.0},"reference_sha256":"9a02e7d73b0c20593f7fb7cb507f7231a1aa8d85788f4ad'
    '48d173be6ff36afd7","repeats":2,"second_sha256":"b0ec09fff25bc977d97e378b48'
    '6ff523627a6f79abd07c6c0e54a6d0f1912c8e"},{"bound_sha256":"5be94c3332684ba6'
    '0c2e0cc60da29224610ce2a1136d1034e332195ba8ce75ba","case":{"inputs":"unifor'
    'm_80","lengths":[1,31,33,16400]},"family":"cross_entropy","index":0,"launc'
    'h":{"active_threads":4096,"extent":[4096],"group_size":[1024],"groups":[4]'
    ',"launched_threads":4096,"padding_threads":0,"threads":[4096],"threads_per'
    '_group":1024},"layout":{"buffer_elements":16593,"dense":false,"elements":1'
    '6465,"gap":32,"group_size":1024,"lengths_sha256":"f42ce29fc6b49473d95e8cea'
    '8bce55b4393598d8dd96d13fe3fd49b712c9a9aa","longest":16400,"output_elements'
    '":36,"rows":4,"shortest":1},"main_sha256":"05a0c037908edd9b80330b0c48f85f8'
    '821b966f92f1be3b897ac3601de7722eb","naive_float32_finite":true,"output":{"'
    'buffer_sha256":"4e1afe72c5d2041be3ae59bd2be82ff53df323aa6148de2255b4c70113'
    '8836fd","differing_from_rounded_reference":0,"identical_repeats":true,"max'
    '_abs_error":7.866399052147699e-07,"max_bound_ratio":0.014965148345659537},'
    '"reference_sha256":"265b932027f9ba6327329aa5ea5eb99faced9b1e36d11cd9f11916'
    '5da66443b4","repeats":2,"second_sha256":"4ceefaceb9ce379dc77af770f19dbc515'
    '25de8c22ff6a07e7d5f3fbea54ab18f"},{"bound_sha256":"f58e637fc8f5917fd8ad5de'
    '18d7b57fbc2e47bd255235c6b5c9eaf0252c66bb6","case":{"cols":64,"inputs":"off'
    'set_plus_1000","rows":2},"family":"cross_entropy","index":1,"launch":{"act'
    'ive_threads":64,"extent":[64],"group_size":[32],"groups":[2],"launched_thr'
    'eads":64,"padding_threads":0,"threads":[64],"threads_per_group":32},"layou'
    't":{"buffer_elements":192,"dense":true,"elements":128,"gap":32,"group_size'
    '":32,"lengths_sha256":"d8a162db4b52873aa8bee8c7bda17ad1e10b8233d0640427722'
    'f394ec172caf3","longest":64,"output_elements":34,"rows":2,"shortest":64},"'
    'main_sha256":"a85daec79f6f4b273449e8c5be9d9cf2f2feb507fbfd10e7207be7a077fe'
    'cf92","naive_float32_finite":false,"output":{"buffer_sha256":"91a8bbc8f7b6'
    '9e922d562281664306feba15f9abbc03438ddade755339d0c7e5","differing_from_roun'
    'ded_reference":1,"identical_repeats":true,"max_abs_error":2.40937308326749'
    '64e-07,"max_bound_ratio":0.03698740090526622},"reference_sha256":"c0db3262'
    'c19bccede7531fb6849c83f702fc889147dfca2c9f97b70058ccf970","repeats":2,"sec'
    'ond_sha256":"17311bb8b87c0eadca7ec033093dba7deab3e557731a270128d3061152628'
    '6d6"}],"distributions":{"normal":"a standard normal, drawn as float32","of'
    'fset_minus_1000":"-1000 plus a standard normal, rounded to float32","offse'
    't_plus_1000":"1000 plus a standard normal, rounded to float32","uniform_80'
    '":"uniform from -80 to 80, drawn in float64 and rounded to float32"},"gap"'
    ':32,"gap_input":1.2676506002282294e+30,"reference":{"cross_entropy":"float'
    "64 loss = (max - x[t]) + log64(sum exp64(x - max)) for each row, with nump"
    "y.add.reduceat; exp64 and log64 use IEEE operations, frexp and ldexp only,"
    ' so each host computes the same bits","rmsnorm":"float64 y = x * w / sqrt('
    "sum(x^2) / n + eps) with numpy.add.reduceat over each row and eps = float3"
    '2(1e-6)","swiglu":"float64 y = silu(a) * b with silu(a) = a / (1 + exp64(-'
    'a)) for a >= 0 and a * exp64(a) / (1 + exp64(a)) for a < 0"},"rules":"ever'
    "y gap slot keeps its sentinel in every repeat; every output is finite; eve"
    "ry value is within its bound of the reference; identical_repeats and diffe"
    'ring_from_rounded_reference are measured","sentinel_float32_bits":21432893'
    '44},"environment":{"macos_version":"27.2.1","numpy_version":"1.26.4","plat'
    'form_machine":"arm64","platform_system":"Darwin","python_implementation":"'
    'CPython","python_version":"3.11.14","torch_version":"2.15.0"},"evidence_bo'
    'undary":{"apple_gpu_metal_this_host":true,"backward_pass_measured":false,"'
    'cuda_kernel_executed":false,"distributed_claim":false,"hardware_counters_u'
    'sed":false,"host_specific":true,"liger_kernel_code_executed":false,"nvidia'
    '_gpu_used":false,"occupancy_measured":false,"portable_performance_claim":f'
    'alse,"production_claim":false,"statement":"These are wall-clock timings on'
    " the Apple GPU of the host that made this report, at the recorded load ave"
    "rages. The three forward kernels came from torch.mps.compile_shader, and e"
    "ach PyTorch sequence ran on the MPS device. PyTorch does not report which "
    "kernels it ran, so each vendor number names the sequence, not a kernel. Th"
    "e bandwidth ceiling comes from a separate gpu-ceilings run with its own lo"
    "ad averages. No hardware counter was read, and each byte count is a model."
    " No CUDA kernel ran, no NVIDIA GPU was used, no Triton kernel ran on a GPU"
    ", and no code of the Liger Kernel project ran. Only forward passes were ti"
    "med. These timings do not measure or predict CUDA, NVIDIA, Triton, Liger K"
    'ernel, distributed or production performance.","triton_gpu_kernel_executed'
    '":false,"vendor_kernel_known":false,"vendor_peak_claimed":false},"host_loa'
    'd":{"after":[6.939453125,7.7509765625,7.6025390625],"before":[6.939453125,'
    '7.7509765625,7.6025390625],"source":"os.getloadavg(): the 1, 5 and 15 minu'
    "te load averages that uptime prints, read before the checks and after the "
    'last sample"},"kind":"metal_liger_correctness_and_bandwidth","label":"host'
    '-specific Apple GPU (Metal) measurement","math_probes":{"exp":{"arguments"'
    ':"precise::exp at 2**18 float32 arguments: 262136 evenly spaced from -104 '
    "to 0, then the extremes -3.4028234663852886e+38, -1e+30, -1000.0, -200.0, "
    '-150.0, -105.0, -87.5, -0.0","arguments_sha256":"ef74306503b0bf88378fecc22'
    '0238ffe4510b9ce6992098cb47e1e39804aab8d","count":262144,"exp_error":4.7683'
    '7158203125e-07,"exp_ulps":4,"holds":true,"max_relative_error":1.6143908444'
    '6152e-07,"max_ulps":1.4061690587550402,"normal_results":220136,"results_sh'
    'a256":"918b6099a06c5fc92292905d92d8968e8f6b44a6cc1e883b02864e0cc4467498","'
    'rule":"each result whose float64 exp is at least 2**-126 has a relative er'
    "ror of at most EXP_ERROR = 4 * 2**-23; each smaller result is within 2**-1"
    '26 of it","tiny_max_abs_error":1.1751589259924035e-38,"tiny_nonzero":0,"ti'
    'ny_results":42008},"log":{"arguments":"2**15 float32 arguments evenly spac'
    'ed from 1 to 2**17, then 1 + i 2**-23 for i below 2**15","assumed_error":4'
    '.76837158203125e-07,"count":65536,"exact_zero":true,"function":"log","hold'
    's":true,"max_relative_error":1.561947983469494e-07,"max_ulps":1.8407074026'
    '76344,"results_sha256":"430951ae06ff4287f9270b593ed6c42df5eb6252c8ae6561b8'
    '35c84f62356732","rule":"each result has an error of at most LOG_ERROR |log'
    ' x|, 4 * 2**-23 |log x|"},"rsqrt":{"arguments":"2**16 float32 arguments, 2'
    '**(-20 + 60 i / (2**16 - 1)) for each i","assumed_error":2.384185791015625'
    'e-07,"count":65536,"exact_zero":null,"function":"rsqrt","holds":true,"max_'
    'relative_error":5.764373496122864e-08,"max_ulps":0.49996383860707283,"resu'
    'lts_sha256":"7d698bd7afd0912df75781f5a15d7fe48228f9d849f5f6fe1d28f7b81fcbe'
    'b96","rule":"each result has a relative error of at most RSQRT_ERROR = 2 *'
    ' 2**-23"}},"measured":true,"measurement_scope":"apple_gpu_metal_this_host"'
    ',"percentile_method":"nearest rank: sorted samples[ceil(p * n / 100) - 1] '
    "for p in 25, 50 and 75; median is p50 and IQR is p75 - p25; no interpolati"
    'on","plan":{"sha256":"43d5f63eb9d93415d0b95244f7150d0652aa73a7845d98da7658'
    '993ff3f13165","steps":[{"base":"torch_rmsnorm","base_sequence":"ms = x.pow'
    "(2).mean(-1, keepdim=True); inv = torch.rsqrt(ms + eps); torch.mul(x * inv"
    ', w, out=y), on the MPS device","change":"Fuse the square, the mean, the r'
    "sqrt and both products into one kernel. Each thread keeps its slots of the"
    ' row in registers, so the kernel reads x once and writes y once.","family"'
    ':"rmsnorm","hypothesis":"The PyTorch sequence runs five or more kernels, a'
    "nd the model gives it about 40 bytes per element. The fused kernel moves a"
    "bout 12 bytes per element. The fused kernel is faster than the PyTorch seq"
    'uence by 1.1 or more at every point.","kernel":"liger_rmsnorm","predicted"'
    ':["faster"],"source_sha256":"181f7549d9245c85b8db746115a8db3483aa660234bf2'
    'de65dbb687c02bfd42f"},{"base":"torch_swiglu","base_sequence":"torch.mul(to'
    'rch.nn.functional.silu(a), b, out=y), on the MPS device","change":"Compute'
    " silu(a) * b in one pass. The value silu(a) stays in a register and never "
    'goes to device memory.","family":"swiglu","hypothesis":"The PyTorch sequen'
    "ce writes silu(a) and reads it again, about 20 bytes per element in the mo"
    "del, and the fused pass moves 12. The fused kernel is faster than the PyTo"
    'rch sequence by 1.1 or more at every point.","kernel":"liger_swiglu","pred'
    'icted":["faster"],"source_sha256":"8e85f3c621f926057d21e416eb46ef6d04b7e66'
    '1546734e5dfa3037d5bbbb509"},{"base":"torch_cross_entropy","base_sequence":'
    "\"torch.nn.functional.cross_entropy(logits, targets, reduction='none') with"
    ' int64 targets, on the MPS device","change":"Compute the loss of each row '
    "in one online pass over the vocabulary, with the Day 79 rescaling update. "
    "Write one float for each row and never write the softmax or the log-probab"
    'ilities.","family":"cross_entropy","hypothesis":"If PyTorch writes the log'
    "-probabilities, it moves at least 8 bytes per logit, and the fused kernel "
    "moves 4. The online pass calls precise::exp once for each logit, and the l"
    "oads probably set most of the time. The fused kernel is faster than the Py"
    'Torch call by 1.1 or more, or it shows no clear difference.","kernel":"lig'
    'er_cross_entropy","predicted":["faster","no_clear_difference"],"source_sha'
    '256":"d5aaccbb567d50ebbbc089ac81e8d5a1abd0d0316e4e655cef8b1dd56fbffa3f"}]}'
    ',"planned_work":{"benchmark_points":3,"calls":60,"compulsory_bytes":103292'
    '80,"correctness_cases":4,"correctness_elements":17323,"correctness_launche'
    's":8,"sides_per_point":2,"timed_samples":24},"report_sha256":"374520ad4a36'
    'd301ed8da69590e3e623bfce3fc242f95c13ec6b88aff7ad388c","run":{"elapsed_ns":'
    '198345667,"max_runtime_ns":60000000000,"timed_ns":8467579},"schema_version'
    '":1,"shaders":{"admissions":[{"admitted":true,"allowed_threads":1024,"decl'
    'ared_threadgroup_bytes":128,"kernel":"liger_rmsnorm","reason":null,"requir'
    'ed_threads":1024,"static_threadgroup_bytes":128,"thread_execution_width":3'
    '2},{"admitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":0,'
    '"kernel":"liger_swiglu","reason":null,"required_threads":1024,"static_thre'
    'adgroup_bytes":0,"thread_execution_width":32},{"admitted":true,"allowed_th'
    'reads":1024,"declared_threadgroup_bytes":256,"kernel":"liger_cross_entropy'
    '","reason":null,"required_threads":1024,"static_threadgroup_bytes":256,"th'
    'read_execution_width":32},{"admitted":true,"allowed_threads":1024,"declare'
    'd_threadgroup_bytes":0,"kernel":"liger_probe_exp","reason":null,"required_'
    'threads":256,"static_threadgroup_bytes":0,"thread_execution_width":32},{"a'
    'dmitted":true,"allowed_threads":1024,"declared_threadgroup_bytes":0,"kerne'
    'l":"liger_probe_rsqrt","reason":null,"required_threads":256,"static_thread'
    'group_bytes":0,"thread_execution_width":32},{"admitted":true,"allowed_thre'
    'ads":1024,"declared_threadgroup_bytes":0,"kernel":"liger_probe_log","reaso'
    'n":null,"required_threads":256,"static_threadgroup_bytes":0,"thread_execut'
    'ion_width":32}],"kernels":[{"barriers":1,"buffers":["y","x","w","starts","'
    'lengths"],"family":"rmsnorm","limits":{"max_threads_per_threadgroup":1024,'
    '"static_threadgroup_memory_bytes":128,"thread_execution_width":32},"name":'
    '"liger_rmsnorm","source_sha256":"181f7549d9245c85b8db746115a8db3483aa66023'
    '4bf2de65dbb687c02bfd42f","summary":"one read of x into registers; simd_sum'
    ' and one threadgroup step","threadgroup_bytes":128},{"barriers":0,"buffers'
    '":["y","a","b","starts","lengths"],"family":"swiglu","limits":{"max_thread'
    's_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_execut'
    'ion_width":32},"name":"liger_swiglu","source_sha256":"8e85f3c621f926057d21'
    'e416eb46ef6d04b7e661546734e5dfa3037d5bbbb509","summary":"one pass: read a '
    'and b, write y","threadgroup_bytes":0},{"barriers":1,"buffers":["loss","x"'
    ',"targets","starts","lengths"],"family":"cross_entropy","limits":{"max_thr'
    'eads_per_threadgroup":1024,"static_threadgroup_memory_bytes":256,"thread_e'
    'xecution_width":32},"name":"liger_cross_entropy","source_sha256":"d5aaccbb'
    '567d50ebbbc089ac81e8d5a1abd0d0316e4e655cef8b1dd56fbffa3f","summary":"one o'
    'nline pass for max and sum; one float written for each row","threadgroup_b'
    'ytes":256},{"barriers":0,"buffers":["y","x"],"family":null,"limits":{"max_'
    'threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_'
    'execution_width":32},"name":"liger_probe_exp","source_sha256":"e3d5fc7285d'
    '761c1c9186c84dd1fde55522c58e1b7eaedba7e85eefe81c8801a","summary":"precise:'
    ':exp at each argument","threadgroup_bytes":0},{"barriers":0,"buffers":["y"'
    ',"x"],"family":null,"limits":{"max_threads_per_threadgroup":1024,"static_t'
    'hreadgroup_memory_bytes":0,"thread_execution_width":32},"name":"liger_prob'
    'e_rsqrt","source_sha256":"1af21f63fa51020a23b30eddd94d4f005a4b256470707ce2'
    'cb68ef9e6bce4ffb","summary":"precise::rsqrt at each argument","threadgroup'
    '_bytes":0},{"barriers":0,"buffers":["y","x"],"family":null,"limits":{"max_'
    'threads_per_threadgroup":1024,"static_threadgroup_memory_bytes":0,"thread_'
    'execution_width":32},"name":"liger_probe_log","source_sha256":"93e368706ed'
    '6f303617e7f43dca508a735ac3c7e55b299974fb4f833fdd625cc","summary":"precise:'
    ':log at each argument","threadgroup_bytes":0}],"library_sha256":"7ef7eeb1a'
    'ad59640b2eb8c07cfc3768db098d6ca7af3afe2b322706dc74e62ca"},"speedup":{"ceil'
    'ing_gb_per_second":124.60522827782013,"fraction_rule":"effective GB/s of a'
    " side / the bandwidth ceiling GB/s, and its modeled GB/s / the ceiling; th"
    'e ceiling comes from a separate Day 74 run on this host","hypotheses":[{"f'
    'amily":"rmsnorm","held_at":[],"missed_at":[0],"predicted":["faster"]},{"fa'
    'mily":"swiglu","held_at":[],"missed_at":[0],"predicted":["faster"]},{"fami'
    'ly":"cross_entropy","held_at":[0],"missed_at":[],"predicted":["faster","no'
    '_clear_difference"]}],"hypothesis_rule":"a prediction holds at a point whe'
    "n the verdict of the fused kernel over its PyTorch sequence is one of the "
    'verdicts that the plan predicted","min_faster_ratio":1.1,"points":[{"famil'
    'y":"rmsnorm","held":false,"index":0,"kernel":{"effective_fraction_of_ceili'
    'ng":0.011098367155035996,"effective_gb_per_second":1.3829145728643215,"iqr'
    '_effective_gb_per_second":[0.5632931544234705,1.51919604302426],"modeled_f'
    'raction_of_ceiling":0.01655520772935189,"modeled_gb_per_second":2.06286543'
    '83026244},"predicted":["faster"],"shape":{"cols":300,"rows":64},"to_torch"'
    ':{"first":"liger_rmsnorm","iqr_separated":false,"ratio":1.599852596314908,'
    '"second":"torch_rmsnorm","verdict":"no_clear_difference"},"torch":{"effect'
    'ive_fraction_of_ceiling":0.006937118569923203,"effective_gb_per_second":0.'
    '8644012429955859,"iqr_effective_gb_per_second":[0.7982817346641466,0.86824'
    '10546773122],"modeled_fraction_of_ceiling":0.034474073472031797,"modeled_g'
    'b_per_second":4.295649794648865}},{"family":"swiglu","held":false,"index":'
    '0,"kernel":{"effective_fraction_of_ceiling":0.017177039447274826,"effectiv'
    'e_gb_per_second":2.1403489214648013,"iqr_effective_gb_per_second":[1.86432'
    '65484470015,2.204784688995215],"modeled_fraction_of_ceiling":0.01721521064'
    '6046548,"modeled_gb_per_second":2.1451052524013896},"predicted":["faster"]'
    ',"shape":{"cols":300,"rows":64},"to_torch":{"first":"liger_swiglu","iqr_se'
    'parated":false,"ratio":1.112828158965498,"second":"torch_swiglu","verdict"'
    ':"no_clear_difference"},"torch":{"effective_fraction_of_ceiling":0.0154354'
    '82386824993,"effective_gb_per_second":1.9233418063886,"iqr_effective_gb_pe'
    'r_second":[1.7805874238284949,2.0288387451843697],"modeled_fraction_of_cei'
    'ling":0.025725803978041653,"modeled_gb_per_second":3.2055696773143336}},{"'
    'family":"cross_entropy","held":true,"index":0,"kernel":{"effective_fractio'
    'n_of_ceiling":0.010860195291203037,"effective_gb_per_second":1.35323711340'
    '20618,"iqr_effective_gb_per_second":[0.9273887870341914,1.4171551956815114'
    '],"modeled_fraction_of_ceiling":0.010870785437757113,"modeled_gb_per_secon'
    'd":1.3545567010309278},"predicted":["faster","no_clear_difference"],"shape'
    '":{"cols":2049,"rows":16},"to_torch":{"first":"liger_cross_entropy","iqr_s'
    'eparated":false,"ratio":1.4804536082474227,"second":"torch_cross_entropy",'
    '"verdict":"no_clear_difference"},"torch":{"effective_fraction_of_ceiling":'
    '0.007335721450981134,"effective_gb_per_second":0.9140692459820061,"iqr_eff'
    'ective_gb_per_second":[0.6643486534772728,1.0063363679923334],"modeled_fra'
    'ction_of_ceiling":0.014671442901962269,"modeled_gb_per_second":1.828138491'
    '9640122}}],"ratio_rule":"ratio = median sample time of the second side / m'
    "edian sample time of the first side, from the same point of the same run w"
    "ith the same calls per sample; faster when ratio >= min_faster_ratio and t"
    "he first side's p75 sample time is below the second side's p25; slower whe"
    "n 1 / ratio >= min_faster_ratio and the second side's p75 is below the fir"
    'st side\'s p25; otherwise no_clear_difference","vendors":{"cross_entropy":"'
    "torch.nn.functional.cross_entropy(logits, targets, reduction='none') with "
    'int64 targets, on the MPS device, PyTorch 2.15.0","rmsnorm":"ms = x.pow(2)'
    ".mean(-1, keepdim=True); inv = torch.rsqrt(ms + eps); torch.mul(x * inv, w"
    ', out=y), on the MPS device, PyTorch 2.15.0","swiglu":"torch.mul(torch.nn.'
    'functional.silu(a), b, out=y), on the MPS device, PyTorch 2.15.0"}},"summa'
    'ry":{"benchmark_points":3,"ceiling_gb_per_second":124.60522827782013,"corr'
    'ectness_cases":4,"hypotheses":{"cross_entropy":{"held":1,"missed":0},"rmsn'
    'orm":{"held":0,"missed":1},"swiglu":{"held":0,"missed":1}},"identical_repe'
    'ats":{"cross_entropy":true,"rmsnorm":true,"swiglu":true},"label":"host-spe'
    'cific Apple GPU (Metal) measurement","load_average_after":[6.939453125,7.7'
    '509765625,7.6025390625],"load_average_before":[6.939453125,7.7509765625,7.'
    '6025390625],"max_bound_ratio":{"cross_entropy":0.03698740090526622,"rmsnor'
    'm":0.08200084972074735,"swiglu":0.0},"naive_float32_failed_cases":[1],"poi'
    'nts":[{"family":"rmsnorm","held":false,"index":0,"kernel_gb_per_second":1.'
    '3829145728643215,"ratio":1.599852596314908,"shape":{"cols":300,"rows":64},'
    '"torch_gb_per_second":0.8644012429955859,"verdict":"no_clear_difference"},'
    '{"family":"swiglu","held":false,"index":0,"kernel_gb_per_second":2.1403489'
    '214648013,"ratio":1.112828158965498,"shape":{"cols":300,"rows":64},"torch_'
    'gb_per_second":1.9233418063886,"verdict":"no_clear_difference"},{"family":'
    '"cross_entropy","held":true,"index":0,"kernel_gb_per_second":1.35323711340'
    '20618,"ratio":1.4804536082474227,"shape":{"cols":2049,"rows":16},"torch_gb'
    '_per_second":0.9140692459820061,"verdict":"no_clear_difference"}],"probe_m'
    'ax_ulps":{"exp":1.4061690587550402,"log":1.840707402676344,"rsqrt":0.49996'
    '383860707283},"timed_samples":24,"warnings":0},"timer":"time.perf_counter_'
    'ns","torch_mps_switches":{"source":"os.environ, read before the runtime im'
    "ports torch; null means unset, and a value that is not a short token is 'u"
    'nknown\'","values":{"PYTORCH_MPS_FAST_MATH":null,"PYTORCH_MPS_PREFER_METAL"'
    ':null}},"verification":{"gpu_only":["the samples_ns of each side: checked '
    'for form only, never observed again","each kernel output digest, max_abs_e'
    "rror and differing count, unless every value equals the rounded reference:"
    ' max_bound_ratio is checked to be at most 1, not recomputed","identical_re'
    'peats: the CPU cannot rerun the GPU","each probe results digest and measur'
    'ed error: checked against the rule","the pipeline limits: checked for form'
    ', then used to rebuild the admissions","the PyTorch side checks, the warni'
    'ngs, the MPS switches and the load averages: checked for form only"],"reco'
    'mputed":["the config, its digest and the planned work","the declared plan '
    'and every kernel digest","every input digest, reference digest and bound d'
    'igest, from the seed","each layout and launch, and each admission from the'
    ' recorded limits","the probe argument rules and the declared probe assumpt'
    'ions","each sample statistic, GB/s rate, model byte count, ratio, verdict,'
    ' ceiling fraction and held hypothesis, from the raw samples","the run tota'
    'ls, the summary and the SHA-256 seal"]},"warnings":{"capture":"warnings.ca'
    "tch_warnings(record=True) with simplefilter('always'), from before the run"
    "time imports torch to the end of the run; equal warnings are kept once wit"
    'h a count","count":0,"distinct":0,"records":[],"torch_imported_before_capt'
    'ure":false}}'
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
    assert summary["load_average_before"] == [6.939453125, 7.7509765625, 7.6025390625]
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (4, 3)


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


def test_the_fresh_process_captured_the_warnings_from_before_the_torch_import():
    report = recorded()

    assert report["warnings"]["torch_imported_before_capture"] is False
    assert report["warnings"]["count"] == 0
    assert report["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }


def test_the_apple_gpu_kept_its_bounds_bits_and_probe_assumptions():
    report = recorded()
    summary = report["summary"]

    assert summary["identical_repeats"] == dict.fromkeys(FAMILIES, True)
    assert max(summary["max_bound_ratio"].values()) <= 1
    assert summary["naive_float32_failed_cases"] == [1]
    assert all(probe["holds"] for probe in report["math_probes"].values())
    for entry in report["shaders"]["kernels"]:
        assert entry["limits"]["thread_execution_width"] == 32
        assert entry["limits"]["static_threadgroup_memory_bytes"] == (
            KERNELS[entry["name"]].threadgroup_bytes
        )


@pytest.mark.parametrize(
    "path",
    [
        ("summary", "points", 0, "verdict"),
        ("speedup", "points", 1, "held"),
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

    with pytest.raises(MetalLigerVerificationError):
        verify_report(seal_report(report))


def test_the_command_verifies_the_recorded_file_without_torch(tmp_path, capsys):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")

    assert main(["verify", "--report", str(path)]) == EXIT_OK
    expected = capsys.readouterr().out
    assert expected.startswith(
        f"verified_metal_liger_measurement={recorded()['report_sha256']} "
    )
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_liger_cli import main\n"
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
    path.write_text(RECORDED_REPORT_JSON.replace("6.939453125", "6.94") + "\n")
    assert main(["verify", "--report", str(path)]) == EXIT_REJECTED


@requires_metal
def test_this_gpu_reports_the_recorded_pipeline_limits():
    runtime = liger_runtime()

    for entry in recorded()["shaders"]["kernels"]:
        assert runtime.limits[entry["name"]].to_mapping() == entry["limits"]
