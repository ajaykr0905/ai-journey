"""The gap of the pinned Day 74 reports from the development host, replayed anywhere.

``RECORDED_GAP_JSON`` is the canonical file that ``matmul_roofline_cli gap``
wrote on 2026-11-17 from two measured reports of the development host's
Apple GPU: the pinned matmul-metal run and run 1 of the pinned ceiling
config. Like the Day 73 labs, the repository keeps neither pinned source
report, so any host can check the arithmetic of this gap but cannot rebuild
it from its sources. The guide quotes its values.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from ai_journey.matmul_roofline_cli import EXIT_OK
from ai_journey.matmul_roofline_io import load_report
from ai_journey.matmul_roofline_verify import verify_gap_report
from ai_journey.memory_access_report import DIGEST_FIELD

ROOT = Path(__file__).resolve().parents[1]
GAP_SHA256 = "28181026bef6f0029e769ef3bff25572ab94a7bc179d4a00f1f11372a180c3ca"
MATMUL_SHA256 = "a4c843ceaa5f018af364a8cea50503f7a180b522da9ecf09b31cb43d8e94aeaf"
CEILING_SHA256 = "0a7563f44a0cbe6956e5716b97b515beed9e68bec545ede6f3c29b350366b305"
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_roofline_cli import main
sys.exit(main(sys.argv[1:]))
"""
RECORDED_GAP_JSON = (
    '{"ceilings":{"bandwidth_gb_per_second":100.73845948308181,"definition":"Th'
    "e FMA ceiling is the highest median GFLOP/s of the ceiling lab's FMA point"
    "s. The bandwidth ceiling is the higher median GB/s of its two stream kerne"
    "ls at the largest buffer. Both come from one run on this host, and neither"
    ' is a vendor peak or a hardware limit.","fma_gflops_per_second":4618.04391'
    '6641791,"fma_scales_with_work":true,"ridge_flops_per_byte":45.841915196423'
    '6},"definitions":{"bandwidth_roof":"the measured bandwidth ceiling in GB/s'
    ' times the intensity in FLOPs per byte, in GFLOP/s","bound":"bandwidth whe'
    're the bandwidth roof is below the FMA ceiling, else compute","fma_ceiling'
    '_share":"median GFLOP/s of a side divided by the measured FMA ceiling","id'
    'eal_reuse":"the model lab\'s ideal_reuse traffic: 4 * (M * K + K * N + M * '
    'N) bytes, each element moved once","median_gflops_per_second":"flops_per_s'
    'ample divided by median_sample_ns","no_reuse":"the model lab\'s no_reuse tr'
    'affic: 4 * (2 * M * N * K + M * N) bytes, K / (4 * K + 2) FLOPs per byte",'
    '"roof_share":"median GFLOP/s of a side divided by the bandwidth roof at on'
    'e intensity","torch_ratio":"median sample time of torch.matmul divided by '
    "that of the naive kernel; all sides of a point ran the same calls per samp"
    'le"},"environment":{"macos_version":"27.2","numpy_version":"1.26.4","platf'
    'orm_machine":"arm64","platform_system":"Darwin","python_implementation":"C'
    'Python","python_version":"3.11.14","torch_float32_matmul_precision":"highe'
    'st","torch_version":"2.15.0"},"evidence_boundary":{"apple_gpu_metal_this_h'
    'ost":true,"cuda_kernel_executed":false,"distributed_claim":false,"host_spe'
    'cific":true,"nvidia_gpu_used":false,"portable_performance_claim":false,"pr'
    'oduction_claim":false,"statement":"These ratios compare two verified repor'
    "ts from the Apple GPU of one host. The matmul rates and the ceilings are h"
    "ost-specific Apple GPU (Metal) measurements from two separate runs. Each c"
    "eiling is a rate that the ceiling lab's own kernels reached, not a vendor "
    "peak and not a hardware limit. The intensities come from the byte counts o"
    "f the model lab, not from measured traffic. No CUDA kernel ran, and no NVI"
    "DIA GPU was used. These ratios do not measure or predict CUDA, NVIDIA, dis"
    'tributed or production performance.","traffic_measured":false,"vendor_peak'
    '_used":false},"kind":"matmul_roofline_gap","label":"host-specific Apple GP'
    'U (Metal) measurement","measured":true,"measurement_scope":"apple_gpu_meta'
    'l_this_host","points":[{"calls_per_sample":256,"flops_per_call":4194304,"f'
    'lops_per_sample":1073741824,"index":0,"intensities":{"ideal_reuse":{"bandw'
    'idth_roof_gflops_per_second":2149.0871356390785,"bound":"bandwidth","bytes'
    '":196608,"flops_per_byte":{"decimal":"21.333333","denominator":3,"numerato'
    'r":64}},"no_reuse":{"bandwidth_roof_gflops_per_second":25.08662026037835,"'
    'bound":"bandwidth","bytes":16842752,"flops_per_byte":{"decimal":"0.249027"'
    ',"denominator":257,"numerator":64}}},"k":128,"m":128,"n":128,"sides":{"mat'
    'mul_x_columns":{"fma_ceiling_share":0.046137916766049045,"ideal_reuse_roof'
    '_share":0.09914299067479083,"mapping":"x_columns","median_gflops_per_secon'
    'd":213.0669258479781,"median_sample_ns":5039458,"no_reuse_roof_share":8.49'
    '3249534473748,"torch_ratio":1.4161576502869953},"matmul_x_rows":{"fma_ceil'
    'ing_share":0.02902868418250097,"ideal_reuse_roof_share":0.0623779911824049'
    '74,"mapping":"x_rows","median_gflops_per_second":134.0557383971144,"median'
    '_sample_ns":8009667,"no_reuse_roof_share":5.343714577959359,"torch_ratio":'
    '0.8910067047731198},"torch_matmul":{"fma_ceiling_share":0.0325796472989422'
    '1,"ideal_reuse_roof_share":0.07000844196597655,"median_gflops_per_second":'
    '150.45424201521521,"median_sample_ns":7136667,"no_reuse_roof_share":5.9973'
    '89861751992}}},{"calls_per_sample":32,"flops_per_call":33554432,"flops_per'
    '_sample":1073741824,"index":1,"intensities":{"ideal_reuse":{"bandwidth_roo'
    'f_gflops_per_second":4298.174271278157,"bound":"bandwidth","bytes":786432,'
    '"flops_per_byte":{"decimal":"42.666667","denominator":3,"numerator":128}},'
    '"no_reuse":{"bandwidth_roof_gflops_per_second":25.135522054258228,"bound":'
    '"bandwidth","bytes":134479872,"flops_per_byte":{"decimal":"0.249513","deno'
    'minator":513,"numerator":128}}},"k":256,"m":256,"n":256,"sides":{"matmul_x'
    '_columns":{"fma_ceiling_share":0.06533322255266753,"ideal_reuse_roof_share'
    '":0.07019531361957317,"mapping":"x_columns","median_gflops_per_second":301'
    '.71169096395056,"median_sample_ns":3558834,"no_reuse_roof_share":12.003398'
    '62894701,"torch_ratio":0.3938438825750232},"matmul_x_rows":{"fma_ceiling_s'
    'hare":0.02652496150054719,"ideal_reuse_roof_share":0.02849894614913637,"ma'
    'pping":"x_rows","median_gflops_per_second":122.49343709675966,"median_samp'
    'le_ns":8765709,"no_reuse_roof_share":4.873319791502319,"torch_ratio":0.159'
    '8986459623517},"torch_matmul":{"fma_ceiling_share":0.1658860920360296,"ide'
    'al_reuse_roof_share":0.1782313163292607,"median_gflops_per_second":766.069'
    '2581824668,"median_sample_ns":1401625,"no_reuse_roof_share":30.47755509230'
    '358}}},{"calls_per_sample":4,"flops_per_call":268435456,"flops_per_sample"'
    ':1073741824,"index":2,"intensities":{"ideal_reuse":{"bandwidth_roof_gflops'
    '_per_second":8596.348542556314,"bound":"compute","bytes":3145728,"flops_pe'
    'r_byte":{"decimal":"85.333333","denominator":3,"numerator":256}},"no_reuse'
    '":{"bandwidth_roof_gflops_per_second":25.16004451479897,"bound":"bandwidth'
    '","bytes":1074790400,"flops_per_byte":{"decimal":"0.249756","denominator":'
    '1025,"numerator":256}}},"k":512,"m":512,"n":512,"sides":{"matmul_x_columns'
    '":{"fma_ceiling_share":0.07856177500343799,"ideal_reuse_roof_share":0.0422'
    '0416672720448,"mapping":"x_columns","median_gflops_per_second":362.8017271'
    '352079,"median_sample_ns":2959583,"no_reuse_roof_share":14.419756965128196'
    ',"torch_ratio":0.2413907634960736},"matmul_x_rows":{"fma_ceiling_share":0.'
    '03575978064441711,"ideal_reuse_roof_share":0.019210509747000927,"mapping":'
    '"x_rows","median_gflops_per_second":165.14023746539527,"median_sample_ns":'
    '6502000,"no_reuse_roof_share":6.563590830225316,"torch_ratio":0.1098763457'
    '3977238},"torch_matmul":{"fma_ceiling_share":0.32545476830026204,"ideal_re'
    'use_roof_share":0.17483753775811292,"median_gflops_per_second":1502.964412'
    '8910886,"median_sample_ns":714416,"no_reuse_roof_share":59.73615873402191}'
    '}},{"calls_per_sample":1,"flops_per_call":2147483648,"flops_per_sample":21'
    '47483648,"index":3,"intensities":{"ideal_reuse":{"bandwidth_roof_gflops_pe'
    'r_second":17192.697085112628,"bound":"compute","bytes":12582912,"flops_per'
    '_byte":{"decimal":"170.666667","denominator":3,"numerator":512}},"no_reuse'
    '":{"bandwidth_roof_gflops_per_second":25.172323697090235,"bound":"bandwidt'
    'h","bytes":8594128896,"flops_per_byte":{"decimal":"0.249878","denominator"'
    ':2049,"numerator":512}}},"k":1024,"m":1024,"n":1024,"sides":{"matmul_x_col'
    'umns":{"fma_ceiling_share":0.08257423456563666,"ideal_reuse_roof_share":0.'
    '022179849951372104,"mapping":"x_columns","median_gflops_per_second":381.33'
    '144160719064,"median_sample_ns":5631541,"no_reuse_roof_share":15.148837516'
    '787145,"torch_ratio":0.15900763219161504},"matmul_x_rows":{"fma_ceiling_sh'
    'are":0.03303853963063664,"ideal_reuse_roof_share":0.008874316007585889,"ma'
    'pping":"x_rows","median_gflops_per_second":152.57342695599024,"median_samp'
    'le_ns":14075083,"no_reuse_roof_share":6.061157833181161,"torch_ratio":0.06'
    '362008664531499},"torch_matmul":{"fma_ceiling_share":0.5193098810887836,"i'
    'deal_reuse_roof_share":0.13948921599337993,"median_gflops_per_second":2398'
    '.195837214029,"median_sample_ns":895458,"no_reuse_roof_share":95.271134523'
    '47848}}},{"calls_per_sample":1,"flops_per_call":17179869184,"flops_per_sam'
    'ple":17179869184,"index":4,"intensities":{"ideal_reuse":{"bandwidth_roof_g'
    'flops_per_second":34385.394170225256,"bound":"compute","bytes":50331648,"f'
    'lops_per_byte":{"decimal":"341.333333","denominator":3,"numerator":1024}},'
    '"no_reuse":{"bandwidth_roof_gflops_per_second":25.17846778390915,"bound":"'
    'bandwidth","bytes":68736253952,"flops_per_byte":{"decimal":"0.249939","den'
    'ominator":4097,"numerator":1024}}},"k":2048,"m":2048,"n":2048,"sides":{"ma'
    'tmul_x_columns":{"fma_ceiling_share":0.0879988290986721,"ideal_reuse_roof_'
    'share":0.011818461506618846,"mapping":"x_columns","median_gflops_per_secon'
    'd":406.3824573907233,"median_sample_ns":42275125,"no_reuse_roof_share":16.'
    '14007893087247,"torch_ratio":0.1100597810177971},"matmul_x_rows":{"fma_cei'
    'ling_share":0.03481211165593651,"ideal_reuse_roof_share":0.004675353135761'
    '3416,"mapping":"x_rows","median_gflops_per_second":160.7638604581524,"medi'
    'an_sample_ns":106864000,"no_reuse_roof_share":6.384973932404738,"torch_rat'
    'io":0.04353936779458003},"torch_matmul":{"fma_ceiling_share":0.79955482633'
    '97174,"ideal_reuse_roof_share":0.10738220081237262,"median_gflops_per_seco'
    'nd":3692.3793017997154,"median_sample_ns":4652791,"no_reuse_roof_share":14'
    '6.64829224276352}}},{"calls_per_sample":1,"flops_per_call":137438953472,"f'
    'lops_per_sample":137438953472,"index":5,"intensities":{"ideal_reuse":{"ban'
    'dwidth_roof_gflops_per_second":68770.78834045051,"bound":"compute","bytes"'
    ':201326592,"flops_per_byte":{"decimal":"682.666667","denominator":3,"numer'
    'ator":2048}},"no_reuse":{"bandwidth_roof_gflops_per_second":25.18154095219'
    '7185,"bound":"bandwidth","bytes":549822922752,"flops_per_byte":{"decimal":'
    '"0.249969","denominator":8193,"numerator":2048}}},"k":4096,"m":4096,"n":40'
    '96,"sides":{"matmul_x_columns":{"fma_ceiling_share":0.0833379819216682,"id'
    'eal_reuse_roof_share":0.00559624907211064,"mapping":"x_columns","median_gf'
    'lops_per_second":384.8584604385634,"median_sample_ns":357115583,"no_reuse_'
    'roof_share":15.283356215934157,"torch_ratio":0.09710173302630706},"matmul_'
    'x_rows":{"fma_ceiling_share":0.04065315234510567,"ideal_reuse_roof_share":'
    '0.002729909710358823,"mapping":"x_rows","median_gflops_per_second":187.738'
    '0428796272,"median_sample_ns":732078333,"no_reuse_roof_share":7.4553834189'
    '89946,"torch_ratio":0.04736725625780814},"torch_matmul":{"fma_ceiling_shar'
    'e":0.8582543207451309,"ideal_reuse_roof_share":0.057632844416839496,"media'
    'n_gflops_per_second":3963.456144848584,"median_sample_ns":34676542,"no_reu'
    'se_roof_share":157.39529810238867}}},{"calls_per_sample":1,"flops_per_call'
    '":8589934592,"flops_per_sample":8589934592,"index":6,"intensities":{"ideal'
    '_reuse":{"bandwidth_roof_gflops_per_second":22923.596113483505,"bound":"co'
    'mpute","bytes":37748736,"flops_per_byte":{"decimal":"227.555556","denomina'
    'tor":9,"numerator":2048}},"no_reuse":{"bandwidth_roof_gflops_per_second":2'
    '5.172323697090235,"bound":"bandwidth","bytes":34376515584,"flops_per_byte"'
    ':{"decimal":"0.249878","denominator":2049,"numerator":512}}},"k":1024,"m":'
    '4096,"n":1024,"sides":{"matmul_x_columns":{"fma_ceiling_share":0.088007936'
    '93038331,"ideal_reuse_roof_share":0.0177295270665886,"mapping":"x_columns"'
    ',"median_gflops_per_second":406.42451775755103,"median_sample_ns":21135375'
    ',"no_reuse_roof_share":16.145689315306683,"torch_ratio":0.1163373254555455'
    '},"matmul_x_rows":{"fma_ceiling_share":0.03318020234692291,"ideal_reuse_ro'
    'of_share":0.0066842754881300485,"mapping":"x_rows","median_gflops_per_seco'
    'nd":153.22763160115105,"median_sample_ns":56059958,"no_reuse_roof_share":6'
    '.087146877857097,"torch_ratio":0.043860771354841184},"torch_matmul":{"fma_'
    'ceiling_share":0.7564892573021429,"ideal_reuse_roof_share":0.1523975817491'
    '4686,"median_gflops_per_second":3493.5006126890275,"median_sample_ns":2458'
    '833,"no_reuse_roof_share":138.7833977795564}}},{"calls_per_sample":1,"flop'
    's_per_call":17179869184,"flops_per_sample":17179869184,"index":7,"intensit'
    'ies":{"ideal_reuse":{"bandwidth_roof_gflops_per_second":29473.19500305022,'
    '"bound":"compute","bytes":58720256,"flops_per_byte":{"decimal":"292.571429'
    '","denominator":7,"numerator":2048}},"no_reuse":{"bandwidth_roof_gflops_pe'
    'r_second":25.17846778390915,"bound":"bandwidth","bytes":68736253952,"flops'
    '_per_byte":{"decimal":"0.249939","denominator":4097,"numerator":1024}}},"k'
    '":2048,"m":1024,"n":4096,"sides":{"matmul_x_columns":{"fma_ceiling_share":'
    '0.08692219983060252,"ideal_reuse_roof_share":0.013619512106077866,"mapping'
    '":"x_columns","median_gflops_per_second":401.4105361488361,"median_sample_'
    'ns":42798750,"no_reuse_roof_share":15.94261174245743,"torch_ratio":0.10981'
    '629136364963},"matmul_x_rows":{"fma_ceiling_share":0.03435870433704766,"id'
    'eal_reuse_roof_share":0.005383535973313238,"mapping":"x_rows","median_gflo'
    'ps_per_second":158.67000554739684,"median_sample_ns":108274208,"no_reuse_r'
    'oof_share":6.301813395046953,"torch_ratio":0.04340830643619208},"torch_mat'
    'mul":{"fma_ceiling_share":0.7915237234042554,"ideal_reuse_roof_share":0.12'
    '402087101063831,"median_gflops_per_second":3655.291315744681,"median_sampl'
    'e_ns":4700000,"no_reuse_roof_share":145.17528815159574}}},{"calls_per_samp'
    'le":1,"flops_per_call":12000000000,"flops_per_sample":12000000000,"index":'
    '8,"intensities":{"ideal_reuse":{"bandwidth_roof_gflops_per_second":27474.1'
    '2531356777,"bound":"compute","bytes":44000000,"flops_per_byte":{"decimal":'
    '"272.727273","denominator":11,"numerator":3000}},"no_reuse":{"bandwidth_ro'
    'of_gflops_per_second":25.172028856342283,"bound":"bandwidth","bytes":48024'
    '000000,"flops_per_byte":{"decimal":"0.249875","denominator":2001,"numerato'
    'r":500}}},"k":1000,"m":2000,"n":3000,"sides":{"matmul_x_columns":{"fma_cei'
    'ling_share":0.08683154616673254,"ideal_reuse_roof_share":0.014595256044415'
    '542,"mapping":"x_columns","median_gflops_per_second":400.99189354788,"medi'
    'an_sample_ns":29925792,"no_reuse_roof_share":15.930058551750273,"torch_rat'
    'io":0.11275948853751305},"matmul_x_rows":{"fma_ceiling_share":0.0260906077'
    '14855753,"ideal_reuse_roof_share":0.0043854925630544325,"mapping":"x_rows"'
    ',"median_gflops_per_second":120.48757223907698,"median_sample_ns":99595334'
    ',"no_reuse_roof_share":4.7865657920028655,"torch_ratio":0.0338812760043557'
    '9},"torch_matmul":{"fma_ceiling_share":0.7700597731768289,"ideal_reuse_roo'
    'f_share":0.1294370543332144,"median_gflops_per_second":3556.169850969812,"'
    'median_sample_ns":3374417,"no_reuse_roof_share":141.27466130223382}}}],"re'
    'port_sha256":"28181026bef6f0029e769ef3bff25572ab94a7bc179d4a00f1f11372a180'
    'c3ca","schema_version":1,"sources":{"gpu_ceilings":{"config_sha256":"b3f10'
    '96b3691a46d5a77068639ba57389194a3d07a06bfc16f5693cf52215199","kind":"gpu_c'
    'eiling_measurement","label":"host-specific Apple GPU (Metal) measurement",'
    '"load_average_after":[9.95361328125,9.49609375,8.78173828125],"load_averag'
    'e_before":[10.66455078125,9.615234375,8.81494140625],"measured":true,"repo'
    'rt_sha256":"0a7563f44a0cbe6956e5716b97b515beed9e68bec545ede6f3c29b350366b3'
    '05"},"matmul_metal":{"config_sha256":"8c867e92ffb162c7b337757fa9558df29b5e'
    '8c63329028d15b5ab6517756cfd8","kind":"matmul_metal_correctness_and_gflops"'
    ',"label":"host-specific Apple GPU (Metal) measurement","measured":true,"re'
    'port_sha256":"a4c843ceaa5f018af364a8cea50503f7a180b522da9ecf09b31cb43d8e94'
    'aeaf"}},"summary":{"best_fma_ceiling_share":{"matmul_x_columns":{"fma_ceil'
    'ing_share":0.08800793693038331,"index":6},"matmul_x_rows":{"fma_ceiling_sh'
    'are":0.04065315234510567,"index":5},"torch_matmul":{"fma_ceiling_share":0.'
    '8582543207451309,"index":5}},"ideal_reuse_roof_share_range":{"matmul_x_col'
    'umns":[0.00559624907211064,0.09914299067479083],"matmul_x_rows":[0.0027299'
    '09710358823,0.062377991182404974]},"label":"host-specific Apple GPU (Metal'
    ') measurement","no_reuse_roof_share_range":{"matmul_x_columns":[8.49324953'
    '4473748,16.145689315306683],"matmul_x_rows":[4.7865657920028655,7.45538341'
    '8989946]},"points":9,"points_above_no_reuse_roof":{"matmul_x_columns":9,"m'
    'atmul_x_rows":9},"torch_ratio_range":{"matmul_x_columns":[0.09710173302630'
    '706,1.4161576502869953],"matmul_x_rows":[0.03388127600435579,0.89100670477'
    "31198]}}}"
)
RECORDED_GAP = json.loads(RECORDED_GAP_JSON)


def test_the_recorded_gap_verifies_on_any_host_as_a_measurement():
    summary = verify_gap_report(json.loads(RECORDED_GAP_JSON))

    assert summary["report_sha256"] == RECORDED_GAP[DIGEST_FIELD] == GAP_SHA256
    assert (summary["measured"], summary["scope"]) == (
        True,
        "apple_gpu_metal_this_host",
    )
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert (summary["matmul_report_sha256"], summary["ceiling_report_sha256"]) == (
        MATMUL_SHA256,
        CEILING_SHA256,
    )
    assert summary["points"] == 9
    assert summary["sources_checked"] is False


def test_the_recorded_gap_keeps_the_load_and_flags_of_the_ceiling_run():
    source = RECORDED_GAP["sources"]["gpu_ceilings"]

    assert [f"{value:.2f}" for value in source["load_average_before"]] == [
        "10.66",
        "9.62",
        "8.81",
    ]
    assert [f"{value:.2f}" for value in source["load_average_after"]] == [
        "9.95",
        "9.50",
        "8.78",
    ]
    assert RECORDED_GAP["ceilings"]["fma_scales_with_work"] is True
    assert RECORDED_GAP["environment"]["torch_float32_matmul_precision"] == "highest"
    assert RECORDED_GAP["environment"]["platform_system"] == "Darwin"


def test_every_naive_point_ran_above_the_no_reuse_roof_and_far_below_the_ceiling():
    summary = RECORDED_GAP["summary"]

    assert summary["points_above_no_reuse_roof"] == {
        "matmul_x_columns": 9,
        "matmul_x_rows": 9,
    }
    for point in RECORDED_GAP["points"]:
        for side in ("matmul_x_columns", "matmul_x_rows"):
            assert point["sides"][side]["fma_ceiling_share"] < 0.1
            assert point["sides"][side]["ideal_reuse_roof_share"] < 0.1


def test_the_command_verifies_the_recorded_file_without_torch(tmp_path):
    path = tmp_path / "gap.json"
    path.write_text(RECORDED_GAP_JSON + "\n", encoding="ascii")

    report, summary = load_report(path)
    assert report == RECORDED_GAP
    assert summary["report_sha256"] == GAP_SHA256
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, "verify", "--report", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout == (
        f"verified_matmul_roofline_measurement={GAP_SHA256} points=9 "
        "sources_checked=false scope=apple_gpu_metal_this_host cuda=false "
        "nvidia=false\n"
    )
