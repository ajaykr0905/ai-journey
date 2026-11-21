"""One real Triton interpreter report from a linux/aarch64 container, verified anywhere.

``RECORDED_REPORT_JSON`` is the canonical file that ``triton_matmul_cli check``
wrote in a fresh process on 2026-11-22 at 04:47 IST (2026-11-21T23:17Z) for
the pinned plan. The process ran in a python:3.11-slim container on Docker
Desktop on the development Mac: linux/aarch64, CPython 3.11.18, triton 3.9.0,
CPU torch 2.15.0+cpu, NumPy 1.26.4 and TRITON_INTERPRET=1. The interpreter ran
each program on the CPU, so the report holds no GPU result and no time. The
verifier never imports triton, so a host without it, such as macOS, checks the
file, and each exact buffer of it, again.
"""

import json
from pathlib import Path

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_matmul_cli import EXIT_OK, main
from ai_journey.triton_matmul_config import PINNED_PLAN
from ai_journey.triton_matmul_loc import loc_mapping
from ai_journey.triton_matmul_report import (
    DIGEST_FIELD,
    INTERPRETER_BOUNDARY,
    INTERPRETER_LABEL,
)
from ai_journey.triton_matmul_runtime import PYPI_VERSION
from ai_journey.triton_matmul_verify import verify_report

ROOT = Path(__file__).resolve().parents[1]
RECORDED_SHA256 = "7d9b3c3005859ba6672b98c73c44869befbbbcec2e3af48761546f4d579de3af"
RECORDED_LINE = (
    "verified_triton_matmul_interpreter="
    "7d9b3c3005859ba6672b98c73c44869befbbbcec2e3af48761546f4d579de3af "
    "plan=day-79-triton-matmul cases=9 configs=6 within_bound=54/54 "
    "exact_bits=54/54 max_bound_ratio=0.516 triton=3.9.0 torch=2.15.0+cpu "
    "machine=aarch64 load_before=0.74,0.40,0.55 load_after=2.03,0.80,0.68 "
    "kernel_lines=33 host_lines=9 gpu=false autotuned=false timing=false"
)
# The canonical report of that run, byte for byte, without its final newline.
RECORDED_REPORT_JSON = (
    '{"backend":"triton_interpreter","ci_install":{"core_metadata_sha256":"5dfdd395'
    '50d7c33bb319d2427993a7cd259ea1d9209213938e5cab6dfc7373ae","read_on":"2026-11-2'
    '2","requires_dist":["importlib-metadata; python_version < \\"3.10\\""],"requires'
    '_python":">=3.11,<3.16","resolver_check":"uv 0.9.7: uv pip compile with --pyth'
    "on-version 3.11, --python-platform x86_64-manylinux_2_28 and --only-binary :al"
    'l: resolved triton==3.9.0 and no other package","source":"https://pypi.org/pyp'
    'i/triton/3.9.0/json","version":"3.9.0","wheels":[{"filename":"triton-3.9.0-cp3'
    '11-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl","sha256":"1622118a03'
    '8b0c6e16fa4a0305da62c8b3c15904ee1cc4d4e65846a25969e41e","size":285777556,"tags'
    '":{"abi":"cp311","platforms":["manylinux_2_27_x86_64","manylinux_2_28_x86_64"]'
    ',"python":"cp311","version":"3.9.0"},"uploaded":"2026-10-27T18:28:00.919896Z"}'
    ',{"filename":"triton-3.9.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_a'
    'arch64.whl","sha256":"feaaec13578ca84301d561d034cb08b3ae293d7f0b7ab6b8ddbb2d67'
    '96363716","size":263373771,"tags":{"abi":"cp311","platforms":["manylinux_2_27_'
    'aarch64","manylinux_2_28_aarch64"],"python":"cp311","version":"3.9.0"},"upload'
    'ed":"2026-10-27T18:38:49.077989Z"}]},"correctness":{"cases":[{"error_bound":5.'
    '960464843912682e-08,"index":0,"operands":[{"a_sha256":"30537cf63198fad07b5033a'
    'aa6e8e688c106b4937a7edd8559fa5e5be85af4fb","b_sha256":"987ca93d379ff1eca4d586d'
    '5bcd0ffa92f745bb06757c55a4bf19faaa1a71b4e","family":"normal","reference_sha256'
    '":"b674be6d99b3d8e95d1a3394f211418a34aebc100ea381ae5ef9db819b81d510"},{"a_sha2'
    '56":"71426d210d52fa91812d0a39251aa75ded92519c3d746b8ced4e5a02ec97960d","b_sha2'
    '56":"4bb8b6f7c4656ab2458282989317052b159802a1b162c13c61d7fcd1d96a226e","family'
    '":"exact","reference_sha256":"950b4cbf64175991383c55306a9f2f1ff88eb265e5bf3762'
    'f380a7da00088927"}],"runs":[{"bound_ratio":0.5155942404037749,"buffer_sha256":'
    '"48618c60114e2f872d9b0915989a7cd04d17af226509fab101be8310e6f9c13c","config":"m'
    '128_n128_k32_g8_w4_s4","family":"normal","guard_elements":254,"k_blocks":1,"ma'
    'x_relative_error":3.073181343650564e-08,"programs":1,"rounded_reference_differ'
    'ences":0},{"bound_ratio":0.0,"buffer_sha256":"e86e437824519eca0d4ddb65fa417573'
    'fb8468753769c9a7f00d365dfcf2305d","config":"m128_n128_k32_g8_w4_s4","family":"'
    'exact","guard_elements":254,"k_blocks":1,"max_relative_error":0.0,"programs":1'
    ',"rounded_reference_differences":0},{"bound_ratio":0.5155942404037749,"buffer_'
    'sha256":"f11b188a7074d8ec98576c6749ca8dbe066e870246b9f9ce26a6aa0e24ff1fa3","co'
    'nfig":"m128_n64_k32_g8_w4_s4","family":"normal","guard_elements":190,"k_blocks'
    '":1,"max_relative_error":3.073181343650564e-08,"programs":1,"rounded_reference'
    '_differences":0},{"bound_ratio":0.0,"buffer_sha256":"18b8b448f88426c29af072b84'
    '31aac29366ef79343629311483b395860f4b68c","config":"m128_n64_k32_g8_w4_s4","fam'
    'ily":"exact","guard_elements":190,"k_blocks":1,"max_relative_error":0.0,"progr'
    'ams":1,"rounded_reference_differences":0},{"bound_ratio":0.5155942404037749,"b'
    'uffer_sha256":"f11b188a7074d8ec98576c6749ca8dbe066e870246b9f9ce26a6aa0e24ff1fa'
    '3","config":"m64_n128_k32_g8_w4_s4","family":"normal","guard_elements":190,"k_'
    'blocks":1,"max_relative_error":3.073181343650564e-08,"programs":1,"rounded_ref'
    'erence_differences":0},{"bound_ratio":0.0,"buffer_sha256":"18b8b448f88426c29af'
    '072b8431aac29366ef79343629311483b395860f4b68c","config":"m64_n128_k32_g8_w4_s4'
    '","family":"exact","guard_elements":190,"k_blocks":1,"max_relative_error":0.0,'
    '"programs":1,"rounded_reference_differences":0},{"bound_ratio":0.5155942404037'
    '749,"buffer_sha256":"9a9414eae63dd6cd33ad50380833f9b2e83c3c8f04ba7378c641acbaa'
    'ad429b6","config":"m128_n32_k32_g8_w4_s4","family":"normal","guard_elements":1'
    '58,"k_blocks":1,"max_relative_error":3.073181343650564e-08,"programs":1,"round'
    'ed_reference_differences":0},{"bound_ratio":0.0,"buffer_sha256":"ca86e60065cce'
    '2fb09c9620c7aac8582a7343b9052c95e4d52e1d28680b7697f","config":"m128_n32_k32_g8'
    '_w4_s4","family":"exact","guard_elements":158,"k_blocks":1,"max_relative_error'
    '":0.0,"programs":1,"rounded_reference_differences":0},{"bound_ratio":0.5155942'
    '404037749,"buffer_sha256":"32f1b454be1661eda58e998a67eb6862cdee020937ee78f0571'
    '9d271865910b6","config":"m64_n32_k32_g8_w2_s5","family":"normal","guard_elemen'
    'ts":94,"k_blocks":1,"max_relative_error":3.073181343650564e-08,"programs":1,"r'
    'ounded_reference_differences":0},{"bound_ratio":0.0,"buffer_sha256":"f3f61e33e'
    '35dc7151f676ac6df8b0e8f89ae63a3380c8e912784a38920e821b9","config":"m64_n32_k32'
    '_g8_w2_s5","family":"exact","guard_elements":94,"k_blocks":1,"max_relative_err'
    'or":0.0,"programs":1,"rounded_reference_differences":0},{"bound_ratio":0.51559'
    '42404037749,"buffer_sha256":"32f1b454be1661eda58e998a67eb6862cdee020937ee78f05'
    '719d271865910b6","config":"m32_n64_k32_g8_w2_s5","family":"normal","guard_elem'
    'ents":94,"k_blocks":1,"max_relative_error":3.073181343650564e-08,"programs":1,'
    '"rounded_reference_differences":0},{"bound_ratio":0.0,"buffer_sha256":"f3f61e3'
    '3e35dc7151f676ac6df8b0e8f89ae63a3380c8e912784a38920e821b9","config":"m32_n64_k'
    '32_g8_w2_s5","family":"exact","guard_elements":94,"k_blocks":1,"max_relative_e'
    'rror":0.0,"programs":1,"rounded_reference_differences":0}],"shape":{"k":1,"m":'
    '1,"n":1}},{"error_bound":5.364420917476787e-07,"index":1,"operands":[{"a_sha25'
    '6":"94c79862a33b068c1d9ddbc97cf62badd4543db40b8b5dfdc15c1fa829350816","b_sha25'
    '6":"6c6ed4f9bfba3310e5939ce6b8a1bb4f0e284579f2aeeea20e35f740107e6d55","family"'
    ':"normal","reference_sha256":"6f1f0815099f8204a9afb740fa126ed3c9c8dfc5ff0d5551'
    'c1bed27eb7335fb3"},{"a_sha256":"88ee02037ddbd60c5c5bf935c78120f157243884cc0298'
    '0ba4e75c0390aef22f","b_sha256":"5b6687d1b6d5343d81f8f0447f954af9c869356d5a1371'
    'c160112c4d6e7f8364","family":"exact","reference_sha256":"17cfe2731aa888ba399c2'
    'c7a1309568fc99364fd6705a11e9a5d36806dd71f7c"}],"runs":[{"bound_ratio":0.200953'
    '36639332986,"buffer_sha256":"0555af253530dc584756439baa17ee5d77a57b84084c6f167'
    '699c7b520dcac00","config":"m128_n128_k32_g8_w4_s4","family":"normal","guard_el'
    'ements":3758,"k_blocks":1,"max_relative_error":1.0779984421177555e-07,"program'
    's":1,"rounded_reference_differences":288},{"bound_ratio":0.0,"buffer_sha256":"'
    'fe4b0514682466d2347a32cb147748c09c4692d1ad49e6f1b0d630a94bcae8ca","config":"m1'
    '28_n128_k32_g8_w4_s4","family":"exact","guard_elements":3758,"k_blocks":1,"max'
    '_relative_error":0.0,"programs":1,"rounded_reference_differences":0},{"bound_r'
    'atio":0.20095336639332986,"buffer_sha256":"fd51a9bd409714bcc52a47e41d98bad5917'
    'ce4b5b5413410485d09c01a18f6cc","config":"m128_n64_k32_g8_w4_s4","family":"norm'
    'al","guard_elements":3694,"k_blocks":1,"max_relative_error":1.0779984421177555'
    'e-07,"programs":1,"rounded_reference_differences":288},{"bound_ratio":0.0,"buf'
    'fer_sha256":"6c27dc6ebc60738f607e6c3173713bfd9505ee98dfc79753d329c2bc1a4ab6ee"'
    ',"config":"m128_n64_k32_g8_w4_s4","family":"exact","guard_elements":3694,"k_bl'
    'ocks":1,"max_relative_error":0.0,"programs":1,"rounded_reference_differences":'
    '0},{"bound_ratio":0.20095336639332986,"buffer_sha256":"a22c43b6be5148699e6abab'
    '05d69f64fbd8921dd2dd9685e996319410af11428","config":"m64_n128_k32_g8_w4_s4","f'
    'amily":"normal","guard_elements":1646,"k_blocks":1,"max_relative_error":1.0779'
    '984421177555e-07,"programs":1,"rounded_reference_differences":288},{"bound_rat'
    'io":0.0,"buffer_sha256":"f847ed3978ed388bc9e451e0d537dbf27a9de4ee6dc610212a8a9'
    '972b6db0202","config":"m64_n128_k32_g8_w4_s4","family":"exact","guard_elements'
    '":1646,"k_blocks":1,"max_relative_error":0.0,"programs":1,"rounded_reference_d'
    'ifferences":0},{"bound_ratio":0.20095336639332986,"buffer_sha256":"fd51a9bd409'
    '714bcc52a47e41d98bad5917ce4b5b5413410485d09c01a18f6cc","config":"m128_n32_k32_'
    'g8_w4_s4","family":"normal","guard_elements":3694,"k_blocks":1,"max_relative_e'
    'rror":1.0779984421177555e-07,"programs":2,"rounded_reference_differences":288}'
    ',{"bound_ratio":0.0,"buffer_sha256":"6c27dc6ebc60738f607e6c3173713bfd9505ee98d'
    'fc79753d329c2bc1a4ab6ee","config":"m128_n32_k32_g8_w4_s4","family":"exact","gu'
    'ard_elements":3694,"k_blocks":1,"max_relative_error":0.0,"programs":2,"rounded'
    '_reference_differences":0},{"bound_ratio":0.20095336639332986,"buffer_sha256":'
    '"edd70fbaf59c95458306e087e54be46e0a253ba8081cf0942f5d9f98f8dcc9e2","config":"m'
    '64_n32_k32_g8_w2_s5","family":"normal","guard_elements":1582,"k_blocks":1,"max'
    '_relative_error":1.0779984421177555e-07,"programs":2,"rounded_reference_differ'
    'ences":288},{"bound_ratio":0.0,"buffer_sha256":"058f9cfc437c5e7e6a10a61c2e8c30'
    'e22a9d643295051f4d46b93b851f21ac0f","config":"m64_n32_k32_g8_w2_s5","family":"'
    'exact","guard_elements":1582,"k_blocks":1,"max_relative_error":0.0,"programs":'
    '2,"rounded_reference_differences":0},{"bound_ratio":0.20095336639332986,"buffe'
    'r_sha256":"9a9d6ac713c0c88ff9f2ea07c536c8b7b50c754a0779387089d6b1b7cb3d851d","'
    'config":"m32_n64_k32_g8_w2_s5","family":"normal","guard_elements":526,"k_block'
    's":1,"max_relative_error":1.0779984421177555e-07,"programs":1,"rounded_referen'
    'ce_differences":288},{"bound_ratio":0.0,"buffer_sha256":"80cf5a6cea3bf81678e43'
    'f157df1764d6237d44d1c64c5b299fefec889e3b6e3","config":"m32_n64_k32_g8_w2_s5","'
    'family":"exact","guard_elements":526,"k_blocks":1,"max_relative_error":0.0,"pr'
    'ograms":1,"rounded_reference_differences":0}],"shape":{"k":9,"m":17,"n":33}},{'
    '"error_bound":9.53675227678176e-07,"index":2,"operands":[{"a_sha256":"ad2e8b4e'
    '4db61b7a0564dd5752c65a51d696bece9a9dc32157a61f3796c900e3","b_sha256":"317ef66e'
    '9b9f4c2cfbefd1e7657b475d9b6f88eb3025ea22fa53031a84acbc27","family":"normal","r'
    'eference_sha256":"21f657af3b2d5b7b7cb1e7e11939c1b6bc3f493f6145d06ce5fceb848945'
    '9c68"},{"a_sha256":"8ef354f3c820007782f94dae10f4c35523a287655b04ee3fdc43cb4947'
    '60d4ea","b_sha256":"a926270fb9a99aafd966957aebd8652ca5a6fe7885c0e4aa55d7dd6d4b'
    '2680b4","family":"exact","reference_sha256":"31ca315c2ce5354553ef3f7c262955e44'
    'f717512b97dbb0658ab5206ba65d43c"}],"runs":[{"bound_ratio":0.17144255483889112,'
    '"buffer_sha256":"bf34eb9b9c24ee2b9028c7c33b799b4a45fb5bc8694c130423743bf645e92'
    '7c7","config":"m128_n128_k32_g8_w4_s4","family":"normal","guard_elements":3168'
    ',"k_blocks":1,"max_relative_error":1.6350051751970765e-07,"programs":1,"rounde'
    'd_reference_differences":634},{"bound_ratio":0.0,"buffer_sha256":"564d2e1f6ae8'
    '345405d87a6236d860af5ee3910337432a438bbdc55b23c9ff97","config":"m128_n128_k32_'
    'g8_w4_s4","family":"exact","guard_elements":3168,"k_blocks":1,"max_relative_er'
    'ror":0.0,"programs":1,"rounded_reference_differences":0},{"bound_ratio":0.1714'
    '4255483889112,"buffer_sha256":"01d4e2350cc357b9e1bfa8f2beea741a74f6ad6d387c48c'
    'e0c9895b45ddfb9cf","config":"m128_n64_k32_g8_w4_s4","family":"normal","guard_e'
    'lements":3104,"k_blocks":1,"max_relative_error":1.6350051751970765e-07,"progra'
    'ms":1,"rounded_reference_differences":634},{"bound_ratio":0.0,"buffer_sha256":'
    '"a2171251df2c6c45b40e2809fd21e3f62aba0e540603e5935b7434eeac784488","config":"m'
    '128_n64_k32_g8_w4_s4","family":"exact","guard_elements":3104,"k_blocks":1,"max'
    '_relative_error":0.0,"programs":1,"rounded_reference_differences":0},{"bound_r'
    'atio":0.17144255483889112,"buffer_sha256":"46fa585b30717822c7204de6efe5cc534c3'
    '7dd6d7c651b4685d3fb7096ba886e","config":"m64_n128_k32_g8_w4_s4","family":"norm'
    'al","guard_elements":1120,"k_blocks":1,"max_relative_error":1.6350051751970765'
    'e-07,"programs":1,"rounded_reference_differences":634},{"bound_ratio":0.0,"buf'
    'fer_sha256":"5705392874d9eed30fb6bd1042737e4236e9d1de6617890f23b08a071d6195f9"'
    ',"config":"m64_n128_k32_g8_w4_s4","family":"exact","guard_elements":1120,"k_bl'
    'ocks":1,"max_relative_error":0.0,"programs":1,"rounded_reference_differences":'
    '0},{"bound_ratio":0.17144255483889112,"buffer_sha256":"53b4a1720b34a56014af4ad'
    'a77768cf35288b2ea01a520bd25fa5e541f67f76f","config":"m128_n32_k32_g8_w4_s4","f'
    'amily":"normal","guard_elements":3072,"k_blocks":1,"max_relative_error":1.6350'
    '051751970765e-07,"programs":1,"rounded_reference_differences":634},{"bound_rat'
    'io":0.0,"buffer_sha256":"a71d7052d3ae21488fc68b8c22f6bcd5c29ead925daf96ed09a53'
    'ebb29a73ec0","config":"m128_n32_k32_g8_w4_s4","family":"exact","guard_elements'
    '":3072,"k_blocks":1,"max_relative_error":0.0,"programs":1,"rounded_reference_d'
    'ifferences":0},{"bound_ratio":0.17144255483889112,"buffer_sha256":"27b894137ce'
    'f1e4ed159b49f82b48b991e4bface2378735b0dc8c59c65c6243b","config":"m64_n32_k32_g'
    '8_w2_s5","family":"normal","guard_elements":1024,"k_blocks":1,"max_relative_er'
    'ror":1.6350051751970765e-07,"programs":1,"rounded_reference_differences":634},'
    '{"bound_ratio":0.0,"buffer_sha256":"0bdfdb0dd666a0eb642926d0335ba5992726ddea5b'
    '1ba9af4cb14e449ddaa955","config":"m64_n32_k32_g8_w2_s5","family":"exact","guar'
    'd_elements":1024,"k_blocks":1,"max_relative_error":0.0,"programs":1,"rounded_r'
    'eference_differences":0},{"bound_ratio":0.17144255483889112,"buffer_sha256":"9'
    '50fd18e42bd76f3cdc3c971c18ae1b1bff88763791e7f94953b6a715e75ad92","config":"m32'
    '_n64_k32_g8_w2_s5","family":"normal","guard_elements":32,"k_blocks":1,"max_rel'
    'ative_error":1.6350051751970765e-07,"programs":1,"rounded_reference_difference'
    's":634},{"bound_ratio":0.0,"buffer_sha256":"fd17fe520d53b80a5a9fc7eca037316421'
    '45e67f553c4089c7d69a28f21ecf0c","config":"m32_n64_k32_g8_w2_s5","family":"exac'
    't","guard_elements":32,"k_blocks":1,"max_relative_error":0.0,"programs":1,"rou'
    'nded_reference_differences":0}],"shape":{"k":16,"m":32,"n":32}},{"error_bound"'
    ':7.689058311510179e-06,"index":3,"operands":[{"a_sha256":"f7f6938dcf21bb28a58f'
    'd294bab536d08ab99264801e010da5e377df87d31a9b","b_sha256":"e942503a6a042d234084'
    '3d9eddf8cd55624221318bfb76af540c512f0c043517","family":"normal","reference_sha'
    '256":"9e7cdcc31e5e060d7d1adc052e0b6003fa37330c9e5dca326c9c728613435efc"},{"a_s'
    'ha256":"f98b21d6aeee3cf41346d5bce109668655284607828fb6bc62c9feaf55b3dfdd","b_s'
    'ha256":"43cc693dcf6ef6257e5030c6e535005ac044e515b91ef9c09abfb9408b46da8b","fam'
    'ily":"exact","reference_sha256":"cab06f913919014fa5b17c1a1880c3b1b844161f21d02'
    '2e81c9252c299003655"}],"runs":[{"bound_ratio":0.008642277023130874,"buffer_sha'
    '256":"848530d1f1e74b945b9f279f4bbb633f23c12564e4a2753593034c609bbc7a9d","confi'
    'g":"m128_n128_k32_g8_w4_s4","family":"normal","guard_elements":4640,"k_blocks"'
    ':5,"max_relative_error":6.645097197507789e-08,"programs":1,"rounded_reference_'
    'differences":1115},{"bound_ratio":0.0,"buffer_sha256":"e9b316db2a34055daef75d5'
    '83eef8e2b7ff3cabd5a607648b54fecb6fd826752","config":"m128_n128_k32_g8_w4_s4","'
    'family":"exact","guard_elements":4640,"k_blocks":5,"max_relative_error":0.0,"p'
    'rograms":1,"rounded_reference_differences":0},{"bound_ratio":0.008642277023130'
    '874,"buffer_sha256":"3573ce7bebc873b2cca10e6cae29465bf37d6e2c744f8155fb4026bb9'
    '8ab378f","config":"m128_n64_k32_g8_w4_s4","family":"normal","guard_elements":4'
    '576,"k_blocks":5,"max_relative_error":6.645097197507789e-08,"programs":1,"roun'
    'ded_reference_differences":1115},{"bound_ratio":0.0,"buffer_sha256":"3fca77903'
    'f0ef9a27e89e9f4304365bbb373727fb4adaeb7ccc6caf30edc9de8","config":"m128_n64_k3'
    '2_g8_w4_s4","family":"exact","guard_elements":4576,"k_blocks":5,"max_relative_'
    'error":0.0,"programs":1,"rounded_reference_differences":0},{"bound_ratio":0.00'
    '8642277023130874,"buffer_sha256":"47c622adc078dd39a8cbb1dac06a5c55a41832c96ada'
    '74b948621f0216b89c5d","config":"m64_n128_k32_g8_w4_s4","family":"normal","guar'
    'd_elements":1632,"k_blocks":5,"max_relative_error":6.645097197507789e-08,"prog'
    'rams":1,"rounded_reference_differences":1115},{"bound_ratio":0.0,"buffer_sha25'
    '6":"b965c9d33d1cc8166ecb3a53ba550eb788ea4aa38d6ce350feb776e4e3ab3213","config"'
    ':"m64_n128_k32_g8_w4_s4","family":"exact","guard_elements":1632,"k_blocks":5,"'
    'max_relative_error":0.0,"programs":1,"rounded_reference_differences":0},{"boun'
    'd_ratio":0.008642277023130874,"buffer_sha256":"3573ce7bebc873b2cca10e6cae29465'
    'bf37d6e2c744f8155fb4026bb98ab378f","config":"m128_n32_k32_g8_w4_s4","family":"'
    'normal","guard_elements":4576,"k_blocks":5,"max_relative_error":6.645097197507'
    '789e-08,"programs":2,"rounded_reference_differences":1115},{"bound_ratio":0.0,'
    '"buffer_sha256":"3fca77903f0ef9a27e89e9f4304365bbb373727fb4adaeb7ccc6caf30edc9'
    'de8","config":"m128_n32_k32_g8_w4_s4","family":"exact","guard_elements":4576,"'
    'k_blocks":5,"max_relative_error":0.0,"programs":2,"rounded_reference_differenc'
    'es":0},{"bound_ratio":0.008642277023130874,"buffer_sha256":"c94face316931b870a'
    '4500264cc6c4d9924d417b4c75ab9f3e4507babd894f7f","config":"m64_n32_k32_g8_w2_s5'
    '","family":"normal","guard_elements":1568,"k_blocks":5,"max_relative_error":6.'
    '645097197507789e-08,"programs":2,"rounded_reference_differences":1115},{"bound'
    '_ratio":0.0,"buffer_sha256":"490b45dad92402e359971255d0673c6a0361c2a3a9ab4842d'
    '2ac99f4066c7a87","config":"m64_n32_k32_g8_w2_s5","family":"exact","guard_eleme'
    'nts":1568,"k_blocks":5,"max_relative_error":0.0,"programs":2,"rounded_referenc'
    'e_differences":0},{"bound_ratio":0.008642277023130874,"buffer_sha256":"dbb0cd9'
    'e585b8ce2f07bdad079d224294997c9755b71450f79efd36a2b02418d","config":"m32_n64_k'
    '32_g8_w2_s5","family":"normal","guard_elements":64,"k_blocks":5,"max_relative_'
    'error":6.645097197507789e-08,"programs":1,"rounded_reference_differences":1115'
    '},{"bound_ratio":0.0,"buffer_sha256":"f0627c4c662360e56c66180b5b53d7f16e3907bf'
    'feec7a87ba3d9c9cbed03ce9","config":"m32_n64_k32_g8_w2_s5","family":"exact","gu'
    'ard_elements":64,"k_blocks":5,"max_relative_error":0.0,"programs":1,"rounded_r'
    'eference_differences":0}],"shape":{"k":129,"m":31,"n":47}},{"error_bound":5.96'
    '0819781186259e-05,"index":4,"operands":[{"a_sha256":"70289e54b41b99a478f5372fc'
    'e09e991254f719022b77cbe566fd83187c22c9e","b_sha256":"b609025ef74c5618c97a6add3'
    '2cb0a26d1382994e2b6e5c9d7d8ac62dd97e86c","family":"normal","reference_sha256":'
    '"6f784618b8cefe75de4a6940fb4bc3143c09665b013aa8cb292269983cb640e9"},{"a_sha256'
    '":"af57516baef0ac99ad5072c54196572585272c5815d4f01f816a8a969d6b12c4","b_sha256'
    '":"568ea3152ef2143616b811ed83c9696759f08eb3ecec0806c95ddaef30f437b9","family":'
    '"exact","reference_sha256":"9e6cb00556b9ca2429a4a1cb79757cab1f5cc2aa6949ea2f8c'
    'ac0ff57a97dd7a"}],"runs":[{"bound_ratio":0.0006326941252029826,"buffer_sha256"'
    ':"6e2c00bf61754f59532c83b97b1bfd30725b4f5b11aea67f2244f1e5b5ca1427","config":"'
    'm128_n128_k32_g8_w4_s4","family":"normal","guard_elements":8318,"k_blocks":32,'
    '"max_relative_error":3.771375656950274e-08,"programs":2,"rounded_reference_dif'
    'ferences":6966},{"bound_ratio":0.0,"buffer_sha256":"9f09280f19967692327f5c1c00'
    'e9c60352525e2650a4cd9971057177fbc27d3d","config":"m128_n128_k32_g8_w4_s4","fam'
    'ily":"exact","guard_elements":8318,"k_blocks":32,"max_relative_error":0.0,"pro'
    'grams":2,"rounded_reference_differences":0},{"bound_ratio":0.00063269412520298'
    '26,"buffer_sha256":"6e2c00bf61754f59532c83b97b1bfd30725b4f5b11aea67f2244f1e5b5'
    'ca1427","config":"m128_n64_k32_g8_w4_s4","family":"normal","guard_elements":83'
    '18,"k_blocks":32,"max_relative_error":3.771375656950274e-08,"programs":4,"roun'
    'ded_reference_differences":6966},{"bound_ratio":0.0,"buffer_sha256":"9f09280f1'
    '9967692327f5c1c00e9c60352525e2650a4cd9971057177fbc27d3d","config":"m128_n64_k3'
    '2_g8_w4_s4","family":"exact","guard_elements":8318,"k_blocks":32,"max_relative'
    '_error":0.0,"programs":4,"rounded_reference_differences":0},{"bound_ratio":0.0'
    '006326941252029826,"buffer_sha256":"2c9d235c4eb08b0feccd5ae9c51cdbd197c0713e72'
    '4dc0483b0d870e267073f9","config":"m64_n128_k32_g8_w4_s4","family":"normal","gu'
    'ard_elements":4158,"k_blocks":32,"max_relative_error":3.771375656950274e-08,"p'
    'rograms":3,"rounded_reference_differences":6966},{"bound_ratio":0.0,"buffer_sh'
    'a256":"3a5cfbd3823bd8cc54ab3ee9da365baf8a8b250c914b8167f024fd79aacea680","conf'
    'ig":"m64_n128_k32_g8_w4_s4","family":"exact","guard_elements":4158,"k_blocks":'
    '32,"max_relative_error":0.0,"programs":3,"rounded_reference_differences":0},{"'
    'bound_ratio":0.0006326941252029826,"buffer_sha256":"9b6a9dc2bc1c0434159afc47fa'
    'ab7ea52b681b5994f301b0f7a2613a1aa48876","config":"m128_n32_k32_g8_w4_s4","fami'
    'ly":"normal","guard_elements":8286,"k_blocks":32,"max_relative_error":3.771375'
    '656950274e-08,"programs":6,"rounded_reference_differences":6966},{"bound_ratio'
    '":0.0,"buffer_sha256":"f35b92983236abfeca148fdf709160f61c3871365e18a654caf6115'
    'c666e2faa","config":"m128_n32_k32_g8_w4_s4","family":"exact","guard_elements":'
    '8286,"k_blocks":32,"max_relative_error":0.0,"programs":6,"rounded_reference_di'
    'fferences":0},{"bound_ratio":0.0006326941252029826,"buffer_sha256":"26ddf69797'
    '9a2bde043edb00ba93d95bc79fbaa389fe8d17d63c50f7ebb6966d","config":"m64_n32_k32_'
    'g8_w2_s5","family":"normal","guard_elements":4126,"k_blocks":32,"max_relative_'
    'error":3.771375656950274e-08,"programs":9,"rounded_reference_differences":6966'
    '},{"bound_ratio":0.0,"buffer_sha256":"ac641ef9922647e4e76823ee9cefbb31cabc0323'
    '5ba03557c18fdffac3c0d018","config":"m64_n32_k32_g8_w2_s5","family":"exact","gu'
    'ard_elements":4126,"k_blocks":32,"max_relative_error":0.0,"programs":9,"rounde'
    'd_reference_differences":0},{"bound_ratio":0.0006326941252029826,"buffer_sha25'
    '6":"f620ccf53075319878882fdb2e9cb3563690fef3a32355eb581a0e8b50038908","config"'
    ':"m32_n64_k32_g8_w2_s5","family":"normal","guard_elements":2078,"k_blocks":32,'
    '"max_relative_error":3.771375656950274e-08,"programs":10,"rounded_reference_di'
    'fferences":6966},{"bound_ratio":0.0,"buffer_sha256":"21082bd92f19e3d1f13f325ab'
    '2a9d50332220dd11fa28c077d3f634ecbfb43ec","config":"m32_n64_k32_g8_w2_s5","fami'
    'ly":"exact","guard_elements":2078,"k_blocks":32,"max_relative_error":0.0,"prog'
    'rams":10,"rounded_reference_differences":0}],"shape":{"k":1000,"m":129,"n":65}'
    '},{"error_bound":3.045890125336189e-05,"index":5,"operands":[{"a_sha256":"f0bc'
    '87bd065025e4e11bee064727c075ade406eb288409b7b9006b09d87febfa","b_sha256":"a450'
    'd0825c45d652e0966335cf1e28677356b1256cb08af99e1dc8b4d10a094d","family":"normal'
    '","reference_sha256":"294ac2b0e693dbbd613f74aca2ec9d4b84a91dc608cd94034daa7e3d'
    'c7f3a95a"},{"a_sha256":"82cea34f06ded0bb71ee3364d3b78ceec4c7a131ff6c0d7032e4a2'
    'c3f5a70aad","b_sha256":"641dd3b549ee2836631322b19035f1867f8b2acf663d4d41b071f8'
    '7634efacc1","family":"exact","reference_sha256":"59052d0395ba77545af02754e1914'
    'c46d9c72401f8cbc30409c30615f5d8e5a1"}],"runs":[{"bound_ratio":0.00163956916968'
    '05585,"buffer_sha256":"f28bbe65211fbc806c4442de96c8563cf2c09517cee63b4b476587b'
    '8f4580625","config":"m128_n128_k32_g8_w4_s4","family":"normal","guard_elements'
    '":384,"k_blocks":16,"max_relative_error":4.993947543735668e-08,"programs":6,"r'
    'ounded_reference_differences":52498},{"bound_ratio":0.0,"buffer_sha256":"63f22'
    '03a05d580b955c3c996d33dcb98b6c2117c75484a6ea4c5ab63343480ec","config":"m128_n1'
    '28_k32_g8_w4_s4","family":"exact","guard_elements":384,"k_blocks":16,"max_rela'
    'tive_error":0.0,"programs":6,"rounded_reference_differences":0},{"bound_ratio"'
    ':0.0016395691696805585,"buffer_sha256":"1db4b7451e8a969c9525028ea99d9fad2d78b7'
    'c98de1632076adbcbeb0e286d4","config":"m128_n64_k32_g8_w4_s4","family":"normal"'
    ',"guard_elements":320,"k_blocks":16,"max_relative_error":4.993947543735668e-08'
    ',"programs":10,"rounded_reference_differences":52498},{"bound_ratio":0.0,"buff'
    'er_sha256":"7ad8734ea5a8e01cf28a3b4deb0f84a299046e4fc60450bc4c3410f9d20418e0",'
    '"config":"m128_n64_k32_g8_w4_s4","family":"exact","guard_elements":320,"k_bloc'
    'ks":16,"max_relative_error":0.0,"programs":10,"rounded_reference_differences":'
    '0},{"bound_ratio":0.0016395691696805585,"buffer_sha256":"f28bbe65211fbc806c444'
    '2de96c8563cf2c09517cee63b4b476587b8f4580625","config":"m64_n128_k32_g8_w4_s4",'
    '"family":"normal","guard_elements":384,"k_blocks":16,"max_relative_error":4.99'
    '3947543735668e-08,"programs":12,"rounded_reference_differences":52498},{"bound'
    '_ratio":0.0,"buffer_sha256":"63f2203a05d580b955c3c996d33dcb98b6c2117c75484a6ea'
    '4c5ab63343480ec","config":"m64_n128_k32_g8_w4_s4","family":"exact","guard_elem'
    'ents":384,"k_blocks":16,"max_relative_error":0.0,"programs":12,"rounded_refere'
    'nce_differences":0},{"bound_ratio":0.0016395691696805585,"buffer_sha256":"cf0e'
    '3d5493cce1332a73a012f2862871646a859c69a3c53bf4c7fdb81180c4e8","config":"m128_n'
    '32_k32_g8_w4_s4","family":"normal","guard_elements":288,"k_blocks":16,"max_rel'
    'ative_error":4.993947543735668e-08,"programs":18,"rounded_reference_difference'
    's":52498},{"bound_ratio":0.0,"buffer_sha256":"57fd1cf93b91eb1f4eeb9bacfb8ed68c'
    '682243b69b63584118e51ec5a7e2f0d8","config":"m128_n32_k32_g8_w4_s4","family":"e'
    'xact","guard_elements":288,"k_blocks":16,"max_relative_error":0.0,"programs":1'
    '8,"rounded_reference_differences":0},{"bound_ratio":0.0016395691696805585,"buf'
    'fer_sha256":"cf0e3d5493cce1332a73a012f2862871646a859c69a3c53bf4c7fdb81180c4e8"'
    ',"config":"m64_n32_k32_g8_w2_s5","family":"normal","guard_elements":288,"k_blo'
    'cks":16,"max_relative_error":4.993947543735668e-08,"programs":36,"rounded_refe'
    'rence_differences":52498},{"bound_ratio":0.0,"buffer_sha256":"57fd1cf93b91eb1f'
    '4eeb9bacfb8ed68c682243b69b63584118e51ec5a7e2f0d8","config":"m64_n32_k32_g8_w2_'
    's5","family":"exact","guard_elements":288,"k_blocks":16,"max_relative_error":0'
    '.0,"programs":36,"rounded_reference_differences":0},{"bound_ratio":0.001639569'
    '1696805585,"buffer_sha256":"1db4b7451e8a969c9525028ea99d9fad2d78b7c98de1632076'
    'adbcbeb0e286d4","config":"m32_n64_k32_g8_w2_s5","family":"normal","guard_eleme'
    'nts":320,"k_blocks":16,"max_relative_error":4.993947543735668e-08,"programs":4'
    '0,"rounded_reference_differences":52498},{"bound_ratio":0.0,"buffer_sha256":"7'
    'ad8734ea5a8e01cf28a3b4deb0f84a299046e4fc60450bc4c3410f9d20418e0","config":"m32'
    '_n64_k32_g8_w2_s5","family":"exact","guard_elements":320,"k_blocks":16,"max_re'
    'lative_error":0.0,"programs":40,"rounded_reference_differences":0}],"shape":{"'
    'k":511,"m":255,"n":257}},{"error_bound":0.00024420024465499155,"index":6,"oper'
    'ands":[{"a_sha256":"f8e1af9514c2d6020bf65589f92c81ecc4db38faad014fb95987ffd23e'
    'd62104","b_sha256":"e7fe2033d9f222a74ed681ac04b5b3196a5359e1a6e6723be5b06bf562'
    'e4246f","family":"normal","reference_sha256":"794d19530a9a5d192633de1a6384aba2'
    '3a8ebdc5ca94576c75f6822dbb684d19"},{"a_sha256":"0aae26ebb8d59ec8c9d1f52adc205d'
    '10b8e9434596e1626346ac32d609a404e7","b_sha256":"2351f42c98e8230b0bd5d8a8950f5c'
    '37987981e884ed1bdefe597148feef9390","family":"exact","reference_sha256":"b5d54'
    '22910e3c7b3f181afd7cd18f6a140eb0d338f1c34ab10dcfcf4ec7777d8"}],"runs":[{"bound'
    '_ratio":0.00013629105559182094,"buffer_sha256":"6106e6cfe550db3119a5bbebc316a7'
    '9adabf57a1fa5d4826db39a7e4907f8070","config":"m128_n128_k32_g8_w4_s4","family"'
    ':"normal","guard_elements":3152,"k_blocks":128,"max_relative_error":3.32823091'
    '1980973e-08,"programs":1,"rounded_reference_differences":2698},{"bound_ratio":'
    '0.0,"buffer_sha256":"77c176773f6930f43af814ae7f7198c63b122ebf8177cc717bfdbcb7e'
    'c33fde0","config":"m128_n128_k32_g8_w4_s4","family":"exact","guard_elements":3'
    '152,"k_blocks":128,"max_relative_error":0.0,"programs":1,"rounded_reference_di'
    'fferences":0},{"bound_ratio":0.00013629105559182094,"buffer_sha256":"1ec11afd5'
    '1e19d2a1699c92a52033ad33f95cf63487ab60309cd1817c191428b","config":"m128_n64_k3'
    '2_g8_w4_s4","family":"normal","guard_elements":3088,"k_blocks":128,"max_relati'
    've_error":3.328230911980973e-08,"programs":1,"rounded_reference_differences":2'
    '698},{"bound_ratio":0.0,"buffer_sha256":"7bd29383bbec61a6166748a0151c022ce0253'
    'c3a1be2e8f79d9959ac4293be80","config":"m128_n64_k32_g8_w4_s4","family":"exact"'
    ',"guard_elements":3088,"k_blocks":128,"max_relative_error":0.0,"programs":1,"r'
    'ounded_reference_differences":0},{"bound_ratio":0.00013629105559182094,"buffer'
    '_sha256":"c245d7c68ece53b6d8ed1467e585af79e44278525f9a077bf1fa12d6fd2ef538","c'
    'onfig":"m64_n128_k32_g8_w4_s4","family":"normal","guard_elements":80,"k_blocks'
    '":128,"max_relative_error":3.328230911980973e-08,"programs":1,"rounded_referen'
    'ce_differences":2698},{"bound_ratio":0.0,"buffer_sha256":"d8d114a571bce1f5436d'
    'edffcda5dace9da8c5d5e0879ff10cf92482d002dac9","config":"m64_n128_k32_g8_w4_s4"'
    ',"family":"exact","guard_elements":80,"k_blocks":128,"max_relative_error":0.0,'
    '"programs":1,"rounded_reference_differences":0},{"bound_ratio":0.0001362910555'
    '9182094,"buffer_sha256":"1ec11afd51e19d2a1699c92a52033ad33f95cf63487ab60309cd1'
    '817c191428b","config":"m128_n32_k32_g8_w4_s4","family":"normal","guard_element'
    's":3088,"k_blocks":128,"max_relative_error":3.328230911980973e-08,"programs":2'
    ',"rounded_reference_differences":2698},{"bound_ratio":0.0,"buffer_sha256":"7bd'
    '29383bbec61a6166748a0151c022ce0253c3a1be2e8f79d9959ac4293be80","config":"m128_'
    'n32_k32_g8_w4_s4","family":"exact","guard_elements":3088,"k_blocks":128,"max_r'
    'elative_error":0.0,"programs":2,"rounded_reference_differences":0},{"bound_rat'
    'io":0.00013629105559182094,"buffer_sha256":"61ef31fa0f4aaf764854c70ab7308a2ea4'
    'a3fd6c079e3bd38b56fef5360ae686","config":"m64_n32_k32_g8_w2_s5","family":"norm'
    'al","guard_elements":16,"k_blocks":128,"max_relative_error":3.328230911980973e'
    '-08,"programs":2,"rounded_reference_differences":2698},{"bound_ratio":0.0,"buf'
    'fer_sha256":"98fe44a33fff2e8ffee7343a6a29f30c880d12e46587a5c4b4e6210d8889731a"'
    ',"config":"m64_n32_k32_g8_w2_s5","family":"exact","guard_elements":16,"k_block'
    's":128,"max_relative_error":0.0,"programs":2,"rounded_reference_differences":0'
    '},{"bound_ratio":0.00013629105559182094,"buffer_sha256":"61ef31fa0f4aaf764854c'
    '70ab7308a2ea4a3fd6c079e3bd38b56fef5360ae686","config":"m32_n64_k32_g8_w2_s5","'
    'family":"normal","guard_elements":16,"k_blocks":128,"max_relative_error":3.328'
    '230911980973e-08,"programs":2,"rounded_reference_differences":2698},{"bound_ra'
    'tio":0.0,"buffer_sha256":"98fe44a33fff2e8ffee7343a6a29f30c880d12e46587a5c4b4e6'
    '210d8889731a","config":"m32_n64_k32_g8_w2_s5","family":"exact","guard_elements'
    '":16,"k_blocks":128,"max_relative_error":0.0,"programs":2,"rounded_reference_d'
    'ifferences":0}],"shape":{"k":4096,"m":64,"n":48}},{"error_bound":0.00012208521'
    '5707779,"index":7,"operands":[{"a_sha256":"8a218096a8a333c1606e76b073dbbd3c5ed'
    '2ee98cdbc899dd0cf3a1182baaa27","b_sha256":"7eb02cc04639b8d2c944c3d8e15c1449cd3'
    '8d63be664606dde99fe82bf11b7d3","family":"normal","reference_sha256":"f17c53418'
    '8097437f8acee1b14c9afe62b34734bcc16c983cae254b62d0a27a9"},{"a_sha256":"1f86bb3'
    '8615bf0dba768e7f130c78c68953b07302add1ef9c10e7462ed530f4c","b_sha256":"40c2639'
    '8d645a8f6148bdb5dfa0ea1412a1d4da45aa4acb0e839b69d98b2fb44","family":"exact","r'
    'eference_sha256":"a2078d53230e878703bb7a88d3aa1ab2ce9677fad3d7229113b3e94e5668'
    '8a54"}],"runs":[{"bound_ratio":0.00024843898717842513,"buffer_sha256":"a5a5b1e'
    'b52b90294c838f97f4f16a162a2025b6ca37d7bbb4fb4461fca6b6a3d","config":"m128_n128'
    '_k32_g8_w4_s4","family":"normal","guard_elements":125024,"k_blocks":64,"max_re'
    'lative_error":3.033072733990017e-08,"programs":8,"rounded_reference_difference'
    's":2545},{"bound_ratio":0.0,"buffer_sha256":"54fb52e9e0b3e2146db37edcc28f5e6d5'
    '00679ea9b02ee9da7af2579cc46a1e1","config":"m128_n128_k32_g8_w4_s4","family":"e'
    'xact","guard_elements":125024,"k_blocks":64,"max_relative_error":0.0,"programs'
    '":8,"rounded_reference_differences":0},{"bound_ratio":0.00024843898717842513,"'
    'buffer_sha256":"a5a5b1eb52b90294c838f97f4f16a162a2025b6ca37d7bbb4fb4461fca6b6a'
    '3d","config":"m128_n64_k32_g8_w4_s4","family":"normal","guard_elements":125024'
    ',"k_blocks":64,"max_relative_error":3.033072733990017e-08,"programs":16,"round'
    'ed_reference_differences":2545},{"bound_ratio":0.0,"buffer_sha256":"54fb52e9e0'
    'b3e2146db37edcc28f5e6d500679ea9b02ee9da7af2579cc46a1e1","config":"m128_n64_k32'
    '_g8_w4_s4","family":"exact","guard_elements":125024,"k_blocks":64,"max_relativ'
    'e_error":0.0,"programs":16,"rounded_reference_differences":0},{"bound_ratio":0'
    '.00024843898717842513,"buffer_sha256":"e3de72e3935441ac5cec6b8d87ba507e332a3b2'
    'cfe842497b86c5476b603612c","config":"m64_n128_k32_g8_w4_s4","family":"normal",'
    '"guard_elements":61024,"k_blocks":64,"max_relative_error":3.033072733990017e-0'
    '8,"programs":8,"rounded_reference_differences":2545},{"bound_ratio":0.0,"buffe'
    'r_sha256":"3e08203a5b4aa53bddf73710bc5b2a1ef4bfb2a84d6f2d29c2bac9e2dec46a4c","'
    'config":"m64_n128_k32_g8_w4_s4","family":"exact","guard_elements":61024,"k_blo'
    'cks":64,"max_relative_error":0.0,"programs":8,"rounded_reference_differences":'
    '0},{"bound_ratio":0.00024843898717842513,"buffer_sha256":"a5a5b1eb52b90294c838'
    'f97f4f16a162a2025b6ca37d7bbb4fb4461fca6b6a3d","config":"m128_n32_k32_g8_w4_s4"'
    ',"family":"normal","guard_elements":125024,"k_blocks":64,"max_relative_error":'
    '3.033072733990017e-08,"programs":32,"rounded_reference_differences":2545},{"bo'
    'und_ratio":0.0,"buffer_sha256":"54fb52e9e0b3e2146db37edcc28f5e6d500679ea9b02ee'
    '9da7af2579cc46a1e1","config":"m128_n32_k32_g8_w4_s4","family":"exact","guard_e'
    'lements":125024,"k_blocks":64,"max_relative_error":0.0,"programs":32,"rounded_'
    'reference_differences":0},{"bound_ratio":0.00024843898717842513,"buffer_sha256'
    '":"e3de72e3935441ac5cec6b8d87ba507e332a3b2cfe842497b86c5476b603612c","config":'
    '"m64_n32_k32_g8_w2_s5","family":"normal","guard_elements":61024,"k_blocks":64,'
    '"max_relative_error":3.033072733990017e-08,"programs":32,"rounded_reference_di'
    'fferences":2545},{"bound_ratio":0.0,"buffer_sha256":"3e08203a5b4aa53bddf73710b'
    'c5b2a1ef4bfb2a84d6f2d29c2bac9e2dec46a4c","config":"m64_n32_k32_g8_w2_s5","fami'
    'ly":"exact","guard_elements":61024,"k_blocks":64,"max_relative_error":0.0,"pro'
    'grams":32,"rounded_reference_differences":0},{"bound_ratio":0.0002484389871784'
    '2513,"buffer_sha256":"8c4bdd20f99b1d2c1770beccf815c561226e792ca410782da966a9a3'
    'a061611a","config":"m32_n64_k32_g8_w2_s5","family":"normal","guard_elements":2'
    '9024,"k_blocks":64,"max_relative_error":3.033072733990017e-08,"programs":16,"r'
    'ounded_reference_differences":2545},{"bound_ratio":0.0,"buffer_sha256":"e1ddb5'
    '9ce58727b28eafd8765190d71e228faeebd8d42cedb2892375e6baa2d5","config":"m32_n64_'
    'k32_g8_w2_s5","family":"exact","guard_elements":29024,"k_blocks":64,"max_relat'
    'ive_error":0.0,"programs":16,"rounded_reference_differences":0}],"shape":{"k":'
    '2048,"m":3,"n":1000}},{"error_bound":5.96050001598984e-06,"index":8,"operands"'
    ':[{"a_sha256":"ca6c6374d367a5cc5bab09c8ab7ed536feddbfeaee7950e9a23d08f1d81efd2'
    'c","b_sha256":"0b0e2a0751dc8ad36cc6f751a92f6f74acf18c7e71e76285b0ecbf599aa6bc8'
    '9","family":"normal","reference_sha256":"96e068fb6e964f5329bf8b8bc01fb4c8c677e'
    '905bc4532a5fccf780d1ee9ea04"},{"a_sha256":"146d19d9e6d16ff0b1974d18bbfb9341970'
    '54f8b1a4773d31554f7ac98ee01d7","b_sha256":"e08192b4f3e18ab8f1bb21b4c4f594fd46e'
    '4a52051b1e4246113ca4bc61a5428","family":"exact","reference_sha256":"dc307012e7'
    '9be97369491b98c85c7637ac308487eb121cc861e990e996885433"}],"runs":[{"bound_rati'
    'o":0.028574912763016998,"buffer_sha256":"b37236b300803167f863ce17eb833c1f79771'
    'f05d92b3899f0ba79e1a3755d56","config":"m128_n128_k32_g8_w4_s4","family":"norma'
    'l","guard_elements":24001,"k_blocks":4,"max_relative_error":1.703207679808711e'
    '-07,"programs":64,"rounded_reference_differences":756529},{"bound_ratio":0.0,"'
    'buffer_sha256":"356046dfbea6c75403f0ea8261629c91d96bbd3f5b805085b2ff7ea105007e'
    '2c","config":"m128_n128_k32_g8_w4_s4","family":"exact","guard_elements":24001,'
    '"k_blocks":4,"max_relative_error":0.0,"programs":64,"rounded_reference_differe'
    'nces":0},{"bound_ratio":0.028574912763016998,"buffer_sha256":"b37236b300803167'
    'f863ce17eb833c1f79771f05d92b3899f0ba79e1a3755d56","config":"m128_n64_k32_g8_w4'
    '_s4","family":"normal","guard_elements":24001,"k_blocks":4,"max_relative_error'
    '":1.703207679808711e-07,"programs":128,"rounded_reference_differences":756529}'
    ',{"bound_ratio":0.0,"buffer_sha256":"356046dfbea6c75403f0ea8261629c91d96bbd3f5'
    'b805085b2ff7ea105007e2c","config":"m128_n64_k32_g8_w4_s4","family":"exact","gu'
    'ard_elements":24001,"k_blocks":4,"max_relative_error":0.0,"programs":128,"roun'
    'ded_reference_differences":0},{"bound_ratio":0.028574912763016998,"buffer_sha2'
    '56":"b37236b300803167f863ce17eb833c1f79771f05d92b3899f0ba79e1a3755d56","config'
    '":"m64_n128_k32_g8_w4_s4","family":"normal","guard_elements":24001,"k_blocks":'
    '4,"max_relative_error":1.703207679808711e-07,"programs":128,"rounded_reference'
    '_differences":756529},{"bound_ratio":0.0,"buffer_sha256":"356046dfbea6c75403f0'
    'ea8261629c91d96bbd3f5b805085b2ff7ea105007e2c","config":"m64_n128_k32_g8_w4_s4"'
    ',"family":"exact","guard_elements":24001,"k_blocks":4,"max_relative_error":0.0'
    ',"programs":128,"rounded_reference_differences":0},{"bound_ratio":0.0285749127'
    '63016998,"buffer_sha256":"b37236b300803167f863ce17eb833c1f79771f05d92b3899f0ba'
    '79e1a3755d56","config":"m128_n32_k32_g8_w4_s4","family":"normal","guard_elemen'
    'ts":24001,"k_blocks":4,"max_relative_error":1.703207679808711e-07,"programs":2'
    '56,"rounded_reference_differences":756529},{"bound_ratio":0.0,"buffer_sha256":'
    '"356046dfbea6c75403f0ea8261629c91d96bbd3f5b805085b2ff7ea105007e2c","config":"m'
    '128_n32_k32_g8_w4_s4","family":"exact","guard_elements":24001,"k_blocks":4,"ma'
    'x_relative_error":0.0,"programs":256,"rounded_reference_differences":0},{"boun'
    'd_ratio":0.028574912763016998,"buffer_sha256":"b37236b300803167f863ce17eb833c1'
    'f79771f05d92b3899f0ba79e1a3755d56","config":"m64_n32_k32_g8_w2_s5","family":"n'
    'ormal","guard_elements":24001,"k_blocks":4,"max_relative_error":1.703207679808'
    '711e-07,"programs":512,"rounded_reference_differences":756529},{"bound_ratio":'
    '0.0,"buffer_sha256":"356046dfbea6c75403f0ea8261629c91d96bbd3f5b805085b2ff7ea10'
    '5007e2c","config":"m64_n32_k32_g8_w2_s5","family":"exact","guard_elements":240'
    '01,"k_blocks":4,"max_relative_error":0.0,"programs":512,"rounded_reference_dif'
    'ferences":0},{"bound_ratio":0.028574912763016998,"buffer_sha256":"b37236b30080'
    '3167f863ce17eb833c1f79771f05d92b3899f0ba79e1a3755d56","config":"m32_n64_k32_g8'
    '_w2_s5","family":"normal","guard_elements":24001,"k_blocks":4,"max_relative_er'
    'ror":1.703207679808711e-07,"programs":512,"rounded_reference_differences":7565'
    '29},{"bound_ratio":0.0,"buffer_sha256":"356046dfbea6c75403f0ea8261629c91d96bbd'
    '3f5b805085b2ff7ea105007e2c","config":"m32_n64_k32_g8_w2_s5","family":"exact","'
    'guard_elements":24001,"k_blocks":4,"max_relative_error":0.0,"programs":512,"ro'
    'unded_reference_differences":0}],"shape":{"k":100,"m":1000,"n":999}}]},"enviro'
    'nment":{"implementation":"CPython","machine":"aarch64","numpy":"1.26.4","pytho'
    'n":"3.11.18","system":"Linux","torch":"2.15.0+cpu","triton":"3.9.0","triton_in'
    'terpret":"1"},"evidence_boundary":{"autotuned":false,"cublas_called":false,"cu'
    'da_compiled":false,"gpu_executed":false,"interpreter_time_reported":false,"met'
    'al_rates_copied_from_recorded_reports":true,"nvidia_or_amd_gpu_used":false,"st'
    'atement":"The pinned Triton text ran in the Triton CPU interpreter with TRITON'
    "_INTERPRET=1, one program after another, as Python and NumPy code. No GPU comp"
    "iled or ran it, nothing was tuned, and no time was reported. The CUDA texts an"
    "d cuBLAS were not compiled or called. Each Metal rate is a copy of a recorded "
    'host-specific Apple GPU (Metal) measurement.","triton_compiled_for_gpu":false,'
    '"triton_interpreter_on_cpu":true},"host_load":{"after":[2.0263671875,0.8017578'
    '125,0.67822265625],"before":[0.7373046875,0.4013671875,0.54736328125]},"kernel'
    '":{"accumulator":"float32","autotune":{"configs":[{"block_k":32,"block_m":128,'
    '"block_n":128,"group_m":8,"name":"m128_n128_k32_g8_w4_s4","num_stages":4,"num_'
    'warps":4,"tile_bytes":32768},{"block_k":32,"block_m":128,"block_n":64,"group_m'
    '":8,"name":"m128_n64_k32_g8_w4_s4","num_stages":4,"num_warps":4,"tile_bytes":2'
    '4576},{"block_k":32,"block_m":64,"block_n":128,"group_m":8,"name":"m64_n128_k3'
    '2_g8_w4_s4","num_stages":4,"num_warps":4,"tile_bytes":24576},{"block_k":32,"bl'
    'ock_m":128,"block_n":32,"group_m":8,"name":"m128_n32_k32_g8_w4_s4","num_stages'
    '":4,"num_warps":4,"tile_bytes":20480},{"block_k":32,"block_m":64,"block_n":32,'
    '"group_m":8,"name":"m64_n32_k32_g8_w2_s5","num_stages":5,"num_warps":2,"tile_b'
    'ytes":12288},{"block_k":32,"block_m":32,"block_n":64,"group_m":8,"name":"m32_n'
    '64_k32_g8_w2_s5","num_stages":5,"num_warps":2,"tile_bytes":12288}],"key":["M",'
    '"N","K"],"source":"python/tutorials/03-matrix-multiplication.py of triton-lang'
    '/triton at tag v3.9.0, get_cuda_autotune_config","tuned":false},"input_precisi'
    'on":"ieee","kernel":"matmul_kernel","language":"Triton (Python)","launcher":"l'
    'aunch_matmul","sha256":"9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf1392'
    '0851a31d56","text":"# Triton source generated by ai_journey.triton_matmul_kern'
    "el.\\n# The lab ran this text only in the Triton CPU interpreter, with\\n# TRITO"
    "N_INTERPRET=1. No GPU compiled or ran it. It is source for a\\n# later run on a"
    " GPU. One program computes one BLOCK_M x BLOCK_N tile of\\n# C = A @ B, and all"
    " three matrices are row-major float32. Masks fill\\n# each slot outside A or B "
    "with 0 and keep each store inside C. tl.dot\\n# adds each tile product to a flo"
    "at32 accumulator in IEEE float32\\n# arithmetic, with no TF32.\\nimport triton\\n"
    "import triton.language as tl\\n\\n\\n@triton.jit\\ndef matmul_kernel(\\n    a_ptr, "
    "b_ptr, c_ptr, M, N, K,\\n    stride_am, stride_ak, stride_bk, stride_bn, stride"
    "_cm, stride_cn,\\n    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl"
    ".constexpr,\\n    GROUP_M: tl.constexpr,\\n):\\n    # Grouped order: GROUP_M rows"
    " of tiles take consecutive program ids.\\n    pid = tl.program_id(axis=0)\\n    "
    "num_pid_m = tl.cdiv(M, BLOCK_M)\\n    num_pid_n = tl.cdiv(N, BLOCK_N)\\n    num_"
    "pid_in_group = GROUP_M * num_pid_n\\n    group_id = pid // num_pid_in_group\\n  "
    "  first_pid_m = group_id * GROUP_M\\n    group_size_m = min(num_pid_m - first_p"
    "id_m, GROUP_M)\\n    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_si"
    "ze_m)\\n    pid_n = (pid % num_pid_in_group) // group_size_m\\n    offs_m = pid_"
    "m * BLOCK_M + tl.arange(0, BLOCK_M)\\n    offs_n = pid_n * BLOCK_N + tl.arange("
    "0, BLOCK_N)\\n    offs_k = tl.arange(0, BLOCK_K)\\n    a_ptrs = a_ptr + offs_m[:"
    ", None] * stride_am + offs_k[None, :] * stride_ak\\n    b_ptrs = b_ptr + offs_k"
    "[:, None] * stride_bk + offs_n[None, :] * stride_bn\\n    acc = tl.zeros((BLOCK"
    "_M, BLOCK_N), dtype=tl.float32)\\n    for k0 in range(0, K, BLOCK_K):\\n        "
    "# A slot past M, past N or past the end of K loads 0.\\n        a_mask = (offs_"
    "m[:, None] < M) & (offs_k[None, :] < K - k0)\\n        b_mask = (offs_k[:, None"
    "] < K - k0) & (offs_n[None, :] < N)\\n        a = tl.load(a_ptrs, mask=a_mask, "
    "other=0.0)\\n        b = tl.load(b_ptrs, mask=b_mask, other=0.0)\\n        acc ="
    ' tl.dot(a, b, acc, input_precision=\\"ieee\\")\\n        a_ptrs += BLOCK_K * stri'
    "de_ak\\n        b_ptrs += BLOCK_K * stride_bk\\n    c_ptrs = c_ptr + offs_m[:, N"
    "one] * stride_cm + offs_n[None, :] * stride_cn\\n    c_mask = (offs_m[:, None] "
    "< M) & (offs_n[None, :] < N)\\n    tl.store(c_ptrs, acc, mask=c_mask)\\n\\n\\ndef "
    "launch_matmul(\\n    a, b, c, m, n, k, *, block_m, block_n, block_k, group_m, n"
    "um_warps, num_stages\\n):\\n    grid = (triton.cdiv(m, block_m) * triton.cdiv(n,"
    " block_n),)\\n    matmul_kernel[grid](\\n        a, b, c, m, n, k, k, 1, n, 1, n"
    ", 1,\\n        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group"
    '_m,\\n        num_warps=num_warps, num_stages=num_stages,\\n    )\\n"},"kind":"tr'
    'iton_matmul_comparison","label":"triton_cpu_interpreter_check","loc":{"rule":"'
    "A line is code when it holds code after its comments are removed; blank lines "
    "and comment-only lines are not code. Kernel lines are the code lines of a __gl"
    "obal__, kernel void or @triton.jit function with its decorator, and of each he"
    "lper that a kernel calls. Host lines are the code lines of the launcher and of"
    " each helper that only host code calls; a Metal text holds no host code. Other"
    " lines are includes, imports and other top-level statements. Physical lines, a"
    's each text is formatted.","texts":[{"lab":"day-79","language":"Triton (Python'
    ')","lines":{"blank":4,"comment":10,"host":9,"kernel":33,"other":2,"total":58},'
    '"name":"matmul_kernel","origin":"ai_journey.triton_matmul_kernel.TRITON_SOURCE'
    '","sha256":"9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56"}'
    ',{"lab":"day-76","language":"CUDA C++","lines":{"blank":3,"comment":10,"host":'
    '9,"kernel":24,"other":0,"total":46},"name":"tiled_matmul_t8_kernel","origin":"'
    'ai_journey.gpu_tiled_cuda.tiled_cuda_source(tile=8)","sha256":"3b72d60eb486fb5'
    'd1671efcf5428d3b6d93f4c61d0e6ec1b4c97815be31884fa"},{"lab":"day-76","language"'
    ':"CUDA C++","lines":{"blank":3,"comment":10,"host":9,"kernel":24,"other":0,"to'
    'tal":46},"name":"tiled_matmul_t16_kernel","origin":"ai_journey.gpu_tiled_cuda.'
    'tiled_cuda_source(tile=16)","sha256":"a789a301b41dde5d8449860297a0c5340fabfbc9'
    'c8c39209fa15791b038fc867"},{"lab":"day-76","language":"CUDA C++","lines":{"bla'
    'nk":3,"comment":10,"host":9,"kernel":24,"other":0,"total":46},"name":"tiled_ma'
    'tmul_t32_kernel","origin":"ai_journey.gpu_tiled_cuda.tiled_cuda_source(tile=32'
    ')","sha256":"178e6d6ec2976d70ef02e0f9cc38eedbdacdd863588b3e441fecc393826b7d6e"'
    '},{"lab":"day-77","language":"CUDA C++","lines":{"blank":3,"comment":11,"host"'
    ':9,"kernel":33,"other":0,"total":56},"name":"coarsened_matmul_t16_c2_kernel","'
    'origin":"ai_journey.gpu_checklist_cuda.cuda_source(coarsened_matmul_t16_c2)","'
    'sha256":"5a57f3c70b5e10cd958d6b5b23e804cee6fe91bf5294ff87ae3cbc9b6b1d6ab7"},{"'
    'lab":"day-77","language":"CUDA C++","lines":{"blank":3,"comment":11,"host":9,"'
    'kernel":33,"other":0,"total":56},"name":"coarsened_matmul_t16_c4_kernel","orig'
    'in":"ai_journey.gpu_checklist_cuda.cuda_source(coarsened_matmul_t16_c4)","sha2'
    '56":"a4a6110ff1be048cfa79490bf185a0a725a05249bc68a6aad482cc662c8bf74c"},{"lab"'
    ':"day-77","language":"CUDA C++","lines":{"blank":3,"comment":11,"host":9,"kern'
    'el":33,"other":0,"total":56},"name":"coarsened_matmul_t32_c2_kernel","origin":'
    '"ai_journey.gpu_checklist_cuda.cuda_source(coarsened_matmul_t32_c2)","sha256":'
    '"41e1bbc8274250492ab9316d6721822a982baa5a6cebac126c17cfd1e7494550"},{"lab":"da'
    'y-77","language":"CUDA C++","lines":{"blank":3,"comment":15,"host":9,"kernel":'
    '84,"other":0,"total":111},"name":"blocked_matmul_m64_n64_k16_t4x4_kernel","ori'
    'gin":"ai_journey.gpu_checklist_cuda.cuda_source(blocked_matmul_m64_n64_k16_t4x'
    '4)","sha256":"71fb4757e0e4f52fc4d59dfebb42a6863fc3fb2dc4f6215cd058060b795d2081'
    '"},{"lab":"day-77","language":"CUDA C++","lines":{"blank":3,"comment":15,"host'
    '":9,"kernel":84,"other":0,"total":111},"name":"blocked_matmul_m128_n128_k8_t8x'
    '8_kernel","origin":"ai_journey.gpu_checklist_cuda.cuda_source(blocked_matmul_m'
    '128_n128_k8_t8x8)","sha256":"b8438783e768956b556e375380ebd97bd0123ce32cec6e773'
    'cee41dfbc8eee0c"},{"lab":"day-76","language":"Metal Shading Language","lines":'
    '{"blank":0,"comment":4,"host":null,"kernel":31,"other":0,"total":35},"name":"m'
    'atmul_tiled_8","origin":"ai_journey.matmul_tiled_source.tiled_kernel_source(8)'
    '","sha256":"21b53bf8ce1c123d1755b12512dedf1dfa1a5cc2509d476ac1019538a924e1ba"}'
    ',{"lab":"day-76","language":"Metal Shading Language","lines":{"blank":0,"comme'
    'nt":4,"host":null,"kernel":31,"other":0,"total":35},"name":"matmul_tiled_16","'
    'origin":"ai_journey.matmul_tiled_source.tiled_kernel_source(16)","sha256":"9e7'
    '9d70317e309e13e043d0c015574670bfe09f0d28cf23abc4300f98a2eeb13"},{"lab":"day-76'
    '","language":"Metal Shading Language","lines":{"blank":0,"comment":4,"host":nu'
    'll,"kernel":31,"other":0,"total":35},"name":"matmul_tiled_32","origin":"ai_jou'
    'rney.matmul_tiled_source.tiled_kernel_source(32)","sha256":"f5ae7c6f03bcd1772c'
    '6eec5aed127916cb962b70bc7f99179d9f115a589ced30"},{"lab":"day-77","language":"M'
    'etal Shading Language","lines":{"blank":0,"comment":3,"host":null,"kernel":126'
    ',"other":0,"total":129},"name":"matmul_s2_coarse_4x1","origin":"ai_journey.mat'
    'mul_checklist_source.block_kernel_source(matmul_s2_coarse_4x1)","sha256":"e252'
    '0d4f573c267360eb3833306bad7fbfe88325a4060b72b051f59bf54c144b"},{"lab":"day-77"'
    ',"language":"Metal Shading Language","lines":{"blank":0,"comment":3,"host":nul'
    'l,"kernel":192,"other":0,"total":195},"name":"matmul_s3_block_4x4","origin":"a'
    'i_journey.matmul_checklist_source.block_kernel_source(matmul_s3_block_4x4)","s'
    'ha256":"d4530e74d254196749d27543ccf78238c0edaf5b551c28555095f5e77a23bc9d"},{"l'
    'ab":"day-77","language":"Metal Shading Language","lines":{"blank":0,"comment":'
    '3,"host":null,"kernel":222,"other":0,"total":225},"name":"matmul_s4_vec4","ori'
    'gin":"ai_journey.matmul_checklist_source.block_kernel_source(matmul_s4_vec4)",'
    '"sha256":"6df4d1063583f53372a700e08fb895858a42371f2e3d0f79bb48bd13fe4eac05"},{'
    '"lab":"day-77","language":"Metal Shading Language","lines":{"blank":0,"comment'
    '":3,"host":null,"kernel":222,"other":0,"total":225},"name":"matmul_s5_pad4","o'
    'rigin":"ai_journey.matmul_checklist_source.block_kernel_source(matmul_s5_pad4)'
    '","sha256":"8567cddd838fcc4aa82000c0f434864b5d4bd1c494e21d5937f787400e784028"}'
    ',{"lab":"day-77","language":"Metal Shading Language","lines":{"blank":0,"comme'
    'nt":3,"host":null,"kernel":316,"other":0,"total":319},"name":"matmul_s6_group_'
    '8x8","origin":"ai_journey.matmul_checklist_source.block_kernel_source(matmul_s'
    '6_group_8x8)","sha256":"919930b29ccdd65de0b096cc22c1c5b0683e9f9c03c07914690980'
    '39511ae2bc"},{"lab":"day-77","language":"Metal Shading Language","lines":{"bla'
    'nk":0,"comment":3,"host":null,"kernel":186,"other":0,"total":189},"name":"matm'
    'ul_s6_group_32x32","origin":"ai_journey.matmul_checklist_source.block_kernel_s'
    'ource(matmul_s6_group_32x32)","sha256":"fc609adecff160fd1e17f2ad85ba7b85785bf4'
    '8cd040b5ef0a281fa4f090dcc0"},{"lab":"day-77","language":"Metal Shading Languag'
    'e","lines":{"blank":0,"comment":3,"host":null,"kernel":420,"other":0,"total":4'
    '23},"name":"matmul_s7_interior","origin":"ai_journey.matmul_checklist_source.b'
    'lock_kernel_source(matmul_s7_interior)","sha256":"b5053c2f0c52043d9c93ce8359a4'
    '91878fd7e8ef44f2c85b6e901019841225ab"}]},"performance":{"boundary":"Each Metal'
    " rate is a host-specific Apple GPU (Metal) measurement from one recorded run a"
    "t the load averages of its report. The lab measured no rate: it copies the rec"
    "orded rates, and no rate here measures or predicts Triton, CUDA, cuBLAS, NVIDI"
    'A or AMD performance.","interpreter_time":"not reported: the Triton CPU interp'
    "reter runs each program as Python and NumPy code on the CPU, so its time is no"
    't a performance number","later_gpu_run":["Use a Linux host with an NVIDIA or A'
    "MD GPU. Install triton 3.9.0 and a torch build for that GPU, leave TRITON_INTE"
    "RPRET unset, and record the GPU name, the driver and the CUDA or ROCm version."
    '","Check that TRITON_SOURCE still has its pinned SHA-256 digest. Run the corre'
    "ctness cases of config/day-79-triton-matmul.json on the GPU: each normal outpu"
    't must keep the Day 74 bound, and each exact output must hold its exact bits."'
    ',"Wrap matmul_kernel in triton.autotune with AUTOTUNE_CONFIGS and the key M, N'
    ", K. Record the config that the tuner picks at each shape and each config that"
    ' fails to compile. The tuning times are not results.","Time the four shapes of'
    " this table with the Day 74 benchmark operands: 1024 x 1024 x 1024, 2048 x 204"
    '8 x 2048, 4096 x 4096 x 4096 and 2000 x 3000 x 1000.","Give each side 2 calls '
    "that are not timed and then 11 timed samples, with CUDA events or triton.testi"
    "ng.do_bench. Move the first side by one at each repeat, as the Day 76 Metal la"
    'b does.","On an NVIDIA GPU, time the tuned Triton kernel, the Day 76 CUDA text'
    " at tile 32, the five Day 77 CUDA texts and sgemm_row_major with cuBLAS on the"
    " same device operands, and check the Day 76 TF32 assumption. On an AMD GPU, re"
    'cord that the CUDA texts and cuBLAS do not run there.","Record the load averag'
    "es before and after the timing. Divide 2 * M * N * K by each median time to ge"
    "t GFLOP/s, and apply the Day 76 ratio rule: a ratio is faster only when it is "
    '1.1 or more and the IQR ranges do not overlap.","Seal the times, the host reco'
    "rds and the correctness results in a report that a verifier checks. Compare a "
    'GPU rate with a Metal rate of this table only as a rate of other hardware."],"'
    'metal_reports":[{"guide":"docs/matmul-tiled-lab.md","lab":"day-76","load_avera'
    'ge_after":[9.7,11.4,13.9],"load_average_before":[9.93,11.62,14.06],"measuremen'
    't":"host-specific Apple GPU (Metal) measurement","report_sha256":"371d65363ce5'
    'c53813c34a6edfc9db4725445bf4ffea605b7c81e0076fc03427","run":"2026-11-19 11:43 '
    'IST"},{"guide":"docs/matmul-checklist-lab.md","lab":"day-77","load_average_aft'
    'er":[16.63,20.39,22.64],"load_average_before":[19.21,21.17,23.01],"measurement'
    '":"host-specific Apple GPU (Metal) measurement","report_sha256":"130322a7b46db'
    '6bc7e9e9bf135a39225e2c139097cd167b74ceb22fd54b2e062","run":"2026-11-20 11:29 I'
    'ST"}],"shapes":[{"cublas":"not measured: no NVIDIA GPU or cuBLAS on this host"'
    ',"cuda_text":"not measured: no NVIDIA GPU, CUDA toolkit or nvcc on this host, '
    'and the Day 76 and Day 77 CUDA texts were never compiled","metal":[{"iqr_gflop'
    's_per_second":[332.6,342.8],"kernel":"matmul_tiled_32","lab":"day-76","median_'
    'gflops_per_second":338.5,"stands_for":"the tiling of the Day 76 CUDA text at t'
    'ile 32: one output for each thread, 32 x 32 tiles"},{"iqr_gflops_per_second":['
    '2002.0,2380.0],"kernel":"torch_matmul","lab":"day-76","median_gflops_per_secon'
    'd":2294.1,"stands_for":"torch.matmul on the MPS device in the Day 76 run"},{"i'
    'qr_gflops_per_second":[571.7,1049.5],"kernel":"matmul_s3_block_4x4","lab":"day'
    '-77","median_gflops_per_second":920.3,"stands_for":"the blocking of the Day 77'
    " CUDA text blocked_matmul_m64_n64_k16_t4x4: 64 x 64 x 16 tiles, 4 x 4 outputs "
    'for each thread"},{"iqr_gflops_per_second":[1145.2,1469.4],"kernel":"matmul_s7'
    '_interior","lab":"day-77","median_gflops_per_second":1312.0,"stands_for":"the '
    'fastest Metal kernel of the Day 77 run"},{"iqr_gflops_per_second":[1840.6,2534'
    '.7],"kernel":"torch_matmul","lab":"day-77","median_gflops_per_second":2246.2,"'
    'stands_for":"torch.matmul on the MPS device in the Day 77 run"}],"shape":{"k":'
    '1024,"m":1024,"n":1024},"triton_gpu":"not measured: no NVIDIA or AMD GPU on th'
    'is host"},{"cublas":"not measured: no NVIDIA GPU or cuBLAS on this host","cuda'
    '_text":"not measured: no NVIDIA GPU, CUDA toolkit or nvcc on this host, and th'
    'e Day 76 and Day 77 CUDA texts were never compiled","metal":[{"iqr_gflops_per_'
    'second":[337.1,348.3],"kernel":"matmul_tiled_32","lab":"day-76","median_gflops'
    '_per_second":339.6,"stands_for":"the tiling of the Day 76 CUDA text at tile 32'
    ': one output for each thread, 32 x 32 tiles"},{"iqr_gflops_per_second":[3464.3'
    ',3656.4],"kernel":"torch_matmul","lab":"day-76","median_gflops_per_second":360'
    '8.8,"stands_for":"torch.matmul on the MPS device in the Day 76 run"},{"iqr_gfl'
    'ops_per_second":[1116.9,1222.9],"kernel":"matmul_s3_block_4x4","lab":"day-77",'
    '"median_gflops_per_second":1177.9,"stands_for":"the blocking of the Day 77 CUD'
    "A text blocked_matmul_m64_n64_k16_t4x4: 64 x 64 x 16 tiles, 4 x 4 outputs for "
    'each thread"},{"iqr_gflops_per_second":[1668.4,1819.4],"kernel":"matmul_s7_int'
    'erior","lab":"day-77","median_gflops_per_second":1774.7,"stands_for":"the fast'
    'est Metal kernel of the Day 77 run"},{"iqr_gflops_per_second":[2710.8,3505.9],'
    '"kernel":"torch_matmul","lab":"day-77","median_gflops_per_second":3390.3,"stan'
    'ds_for":"torch.matmul on the MPS device in the Day 77 run"}],"shape":{"k":2048'
    ',"m":2048,"n":2048},"triton_gpu":"not measured: no NVIDIA or AMD GPU on this h'
    'ost"},{"cublas":"not measured: no NVIDIA GPU or cuBLAS on this host","cuda_tex'
    't":"not measured: no NVIDIA GPU, CUDA toolkit or nvcc on this host, and the Da'
    'y 76 and Day 77 CUDA texts were never compiled","metal":[{"iqr_gflops_per_seco'
    'nd":[333.1,359.1],"kernel":"matmul_tiled_32","lab":"day-76","median_gflops_per'
    '_second":353.5,"stands_for":"the tiling of the Day 76 CUDA text at tile 32: on'
    'e output for each thread, 32 x 32 tiles"},{"iqr_gflops_per_second":[3703.5,401'
    '0.8],"kernel":"torch_matmul","lab":"day-76","median_gflops_per_second":3940.1,'
    '"stands_for":"torch.matmul on the MPS device in the Day 76 run"},{"iqr_gflops_'
    'per_second":[1153.2,1195.8],"kernel":"matmul_s3_block_4x4","lab":"day-77","med'
    'ian_gflops_per_second":1160.2,"stands_for":"the blocking of the Day 77 CUDA te'
    "xt blocked_matmul_m64_n64_k16_t4x4: 64 x 64 x 16 tiles, 4 x 4 outputs for each"
    ' thread"},{"iqr_gflops_per_second":[1815.1,1832.2],"kernel":"matmul_s7_interio'
    'r","lab":"day-77","median_gflops_per_second":1825.8,"stands_for":"the fastest '
    'Metal kernel of the Day 77 run"},{"iqr_gflops_per_second":[3250.9,3698.8],"ker'
    'nel":"torch_matmul","lab":"day-77","median_gflops_per_second":3521.4,"stands_f'
    'or":"torch.matmul on the MPS device in the Day 77 run"}],"shape":{"k":4096,"m"'
    ':4096,"n":4096},"triton_gpu":"not measured: no NVIDIA or AMD GPU on this host"'
    '},{"cublas":"not measured: no NVIDIA GPU or cuBLAS on this host","cuda_text":"'
    "not measured: no NVIDIA GPU, CUDA toolkit or nvcc on this host, and the Day 76"
    ' and Day 77 CUDA texts were never compiled","metal":[{"iqr_gflops_per_second":'
    '[339.8,345.4],"kernel":"matmul_tiled_32","lab":"day-76","median_gflops_per_sec'
    'ond":342.3,"stands_for":"the tiling of the Day 76 CUDA text at tile 32: one ou'
    'tput for each thread, 32 x 32 tiles"},{"iqr_gflops_per_second":[3269.8,3483.6]'
    ',"kernel":"torch_matmul","lab":"day-76","median_gflops_per_second":3448.9,"sta'
    'nds_for":"torch.matmul on the MPS device in the Day 76 run"},{"iqr_gflops_per_'
    'second":[1293.8,1368.5],"kernel":"matmul_s3_block_4x4","lab":"day-77","median_'
    'gflops_per_second":1334.0,"stands_for":"the blocking of the Day 77 CUDA text b'
    "locked_matmul_m64_n64_k16_t4x4: 64 x 64 x 16 tiles, 4 x 4 outputs for each thr"
    'ead"},{"iqr_gflops_per_second":[1505.8,1737.9],"kernel":"matmul_s7_interior","'
    'lab":"day-77","median_gflops_per_second":1665.8,"stands_for":"the fastest Meta'
    'l kernel of the Day 77 run"},{"iqr_gflops_per_second":[2966.5,3291.3],"kernel"'
    ':"torch_matmul","lab":"day-77","median_gflops_per_second":3042.8,"stands_for":'
    '"torch.matmul on the MPS device in the Day 77 run"}],"shape":{"k":1000,"m":200'
    '0,"n":3000},"triton_gpu":"not measured: no NVIDIA or AMD GPU on this host"}]},'
    '"plan":{"configs":["m128_n128_k32_g8_w4_s4","m128_n64_k32_g8_w4_s4","m64_n128_'
    'k32_g8_w4_s4","m128_n32_k32_g8_w4_s4","m64_n32_k32_g8_w2_s5","m32_n64_k32_g8_w'
    '2_s5"],"correctness":{"shapes":[[1,1,1],[17,33,9],[32,32,16],[31,47,129],[129,'
    '65,1000],[255,257,511],[64,48,4096],[3,1000,2048],[1000,999,100]]},"name":"day'
    '-79-triton-matmul","schema_version":1,"seed":74},"plan_sha256":"fb7d92a324177d'
    '79cb88988b394440cfac4ffa7258663110a61aaec151abf49f","recorded_at":"2026-11-21T'
    '23:17:07Z","report_sha256":"7d9b3c3005859ba6672b98c73c44869befbbbcec2e3af48761'
    '546f4d579de3af","rule":{"bound_source":"N. J. Higham, Accuracy and Stability o'
    'f Numerical Algorithms, 2nd ed., SIAM, 2002, section 3.1","error_bound":"(gamm'
    'a_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)","exact":"each exact operand is a'
    "n integer from -8 to 8, so each partial sum is an integer of at most 64 * K <="
    " 2**19 in size and exact in float32 in any order: an exact output must hold th"
    'e bits of the float64 reference","exact_inputs":"numpy.random.default_rng([see'
    "d, 2, index]).integers(-8, 9) draws A[m, k] and then B[k, n] as int64, convert"
    'ed to float32","gamma":"gamma_K(u) = K * u / (1 - K * u)","guard":"the buffer '
    "is C and then (Mp - 1) * N + Np - M * N sentinel slots, where Mp and Np are M "
    "and N padded to whole tiles; the tail must keep its sentinels, and each elemen"
    't of C must be finite","normal_inputs":"numpy.random.default_rng([seed, 0, ind'
    "ex]) draws standard-normal float32 A[m, k] and then B[k, n], as the Day 74 Met"
    'al lab draws case index","order":"tl.dot adds the products of one A tile and o'
    "ne B tile in an order that the compiler, or numpy.matmul in the CPU interprete"
    "r, chooses, and then adds them to a float32 accumulator, so no CPU replay name"
    's the output bits","reference":"float64 sum of the exact float32 products in i'
    'ncreasing k order, on the CPU, rebuilt from the seed","relative_error":"max ov'
    'er elements of |C - R| / (|A| @ |B|), in float64","sentinel_float32_bits":2143'
    '289344},"schema_version":1,"summary":{"cases":9,"configs":6,"exact_runs_with_e'
    'xact_bits":54,"interpreter_steps":35640,"loc_texts":19,"max_bound_ratio":0.515'
    '5942404037749,"normal_elements_off_the_rounded_reference":4939638,"normal_runs'
    '_within_bound":54,"runs":108,"triton_host_lines":9,"triton_kernel_lines":33}}'
)


def recorded():
    return json.loads(RECORDED_REPORT_JSON)


def test_the_recorded_report_is_canonical_and_verifies_against_the_plan():
    report = recorded()

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert report[DIGEST_FIELD] == RECORDED_SHA256
    assert verify_report(report, plan=PINNED_PLAN) == PINNED_PLAN


def test_the_recorded_run_names_the_interpreter_and_its_container():
    report = recorded()

    assert report["label"] == INTERPRETER_LABEL
    assert report["backend"] == "triton_interpreter"
    assert report["evidence_boundary"] == dict(INTERPRETER_BOUNDARY)
    assert report["environment"] == {
        "python": "3.11.18",
        "implementation": "CPython",
        "system": "Linux",
        "machine": "aarch64",
        "numpy": "1.26.4",
        "torch": "2.15.0+cpu",
        "triton": PYPI_VERSION,
        "triton_interpret": "1",
    }
    assert report["recorded_at"] == "2026-11-21T23:17:07Z"
    assert [round(value, 2) for value in report["host_load"]["before"]] == [
        0.74,
        0.4,
        0.55,
    ]


def test_every_recorded_run_held_its_rule():
    report = recorded()
    summary = report["summary"]
    runs = [run for case in report["correctness"]["cases"] for run in case["runs"]]

    assert (summary["cases"], summary["configs"], summary["runs"]) == (9, 6, 108)
    assert summary["normal_runs_within_bound"] == summary["exact_runs_with_exact_bits"]
    assert summary["exact_runs_with_exact_bits"] == 54
    assert summary["interpreter_steps"] == 35640
    assert round(summary["max_bound_ratio"], 4) == 0.5156
    for run in runs:
        if run["family"] == "exact":
            assert run["max_relative_error"] == 0.0
            assert run["rounded_reference_differences"] == 0
        else:
            assert 0 < run["bound_ratio"] <= summary["max_bound_ratio"]


def test_each_case_gave_one_error_and_one_count_for_all_six_configs():
    # Each config has BLOCK_K = 32, and the error and the count of elements off
    # the rounded reference were the same for each config of a case.
    for case in recorded()["correctness"]["cases"]:
        normal = [run for run in case["runs"] if run["family"] == "normal"]
        assert len(normal) == 6
        assert len({run["max_relative_error"] for run in normal}) == 1
        assert len({run["rounded_reference_differences"] for run in normal}) == 1


def test_the_recorded_sections_without_a_run_are_the_sections_of_today():
    assert recorded()["loc"] == loc_mapping()


def test_the_cli_verifies_the_recorded_file(tmp_path, capsys):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")

    config = ROOT / "config/day-79-triton-matmul.json"
    argv = ["verify", "--report", str(path), "--config", str(config)]
    assert main(argv) == EXIT_OK
    assert capsys.readouterr().out == f"{RECORDED_LINE}\n"
