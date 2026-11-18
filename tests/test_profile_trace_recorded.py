"""One real capture from the development host's Apple GPU, replayed on any host.

The capture ran on 2026-11-18 under the shared GPU lock (torch 2.15.0,
macOS 27.2, arm64, one-minute load averages 10.80 before and 11.69 after) with
``RECORDED_CONFIG``: two small shapes, one warmup and two profiled
iterations. ``RECORDED_TRACES`` holds the exact trace files that the
capture stored, so their SHA-256 digests match the report, and
``RECORDED_REPORT_JSON`` is the canonical report. A host without torch or
MPS can therefore parse genuine ``torch.profiler`` exports and verify a
genuine Apple GPU (Metal) report.
"""

import json

import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.profile_trace_config import config_from_mapping
from ai_journey.profile_trace_events import (
    ParsedTrace,
    parse_trace_bytes,
)
from ai_journey.profile_trace_io import REPORT_FILE, load_report
from ai_journey.profile_trace_layout import check_layout
from ai_journey.profile_trace_verify import (
    ProfileTraceVerificationError,
    verify_report,
)

RECORDED_CONFIG = {
    "schema_version": 1,
    "name": "profile-trace-recorded",
    "seed": 75,
    "group_size": [32, 8],
    "shapes": [[128, 128, 128], [256, 192, 320]],
    "warmup": 1,
    "iterations": 2,
    "analysis": {"top_n": 4, "gap_count": 2, "timeline_width": 48, "timeline_depth": 3},
    "day_74_ceilings": {
        "fma_gflops_per_second": 4618.0,
        "bandwidth_gb_per_second": 100.7,
        "torch_matmul_gflops_per_second": 3963.0,
    },
}
RECORDED_TRACES = {
    "trace-0-128x128x128.json": (
        ' {\n    "schemaVersion": 1,\n    "deviceProperties": [],\n    \n      "trace_id"'
        ': "C847101DDF96479790F9F990DBFFEE05",\n      "host_name": "redacted-host",\n  '
        '  "displayTimeUnit": "ms",\n    "baseTimeNanoseconds": 1790857026000000000,\n '
        '   "traceEvents": [\n    \n  {\n    "ph": "M",\n    "name": "thread_name",\n    "'
        'ts": 4114144777630.000,\n    "pid": 50515,\n    "tid": 44672914,\n    "args": {'
        '\n      "name": "thread 44672914 ()"\n    }\n  },\n  {\n    "ph": "M",\n    "name"'
        ': "thread_sort_index",\n    "ts": 4114144777630.000,\n    "pid": 50515,\n    "t'
        'id": 44672914,\n    "args": {\n      "sort_index": 44672914\n    }\n  },\n  {\n   '
        ' "ph": "M",\n    "name": "thread_name",\n    "ts": 4114144777630.000,\n    "pid'
        '": 50515,\n    "tid": 44672914,\n    "args": {\n      "name": "thread 44672914 '
        '()"\n    }\n  },\n  {\n    "ph": "M",\n    "name": "thread_sort_index",\n    "ts":'
        ' 4114144777630.000,\n    "pid": 50515,\n    "tid": 44672914,\n    "args": {\n   '
        '   "sort_index": 44672914\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_ann'
        'otation",\n    "name": "profile_trace/iteration",\n    "pid": 50515,\n    "tid"'
        ': 44672914,\n    "ts": 4114144778716.609,\n    "dur": 2738.118\n    ,\n    "args'
        '": {\n      "External id": 1,"Record function id": 0, "Ev Idx": 0\n    }\n  },\n'
        '  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "profile_trace/'
        'matmul_x_columns",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 41141447'
        '79056.952,\n    "dur": 925.530\n    ,\n    "args": {\n      "External id": 2,"Re'
        'cord function id": 0, "Ev Idx": 1\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "'
        'user_annotation",\n    "name": "matmul_x_columns/dispatch",\n    "pid": 50515,'
        '\n    "tid": 44672914,\n    "ts": 4114144779122.658,\n    "dur": 78.393\n    ,\n '
        '   "args": {\n      "External id": 3,"Record function id": 0, "Ev Idx": 2\n   '
        ' }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "matmul'
        '_x_columns/synchronize",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 41'
        '14144779221.653,\n    "dur": 754.602\n    ,\n    "args": {\n      "External id":'
        ' 4,"Record function id": 0, "Ev Idx": 3\n    }\n  },\n  {\n    "ph": "X",\n    "c'
        'at": "user_annotation",\n    "name": "profile_trace/matmul_x_rows",\n    "pid"'
        ': 50515,\n    "tid": 44672914,\n    "ts": 4114144780058.564,\n    "dur": 649.16'
        '8\n    ,\n    "args": {\n      "External id": 5,"Record function id": 0, "Ev Id'
        'x": 4\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name"'
        ': "matmul_x_rows/dispatch",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts":'
        ' 4114144780100.377,\n    "dur": 37.329\n    ,\n    "args": {\n      "External id'
        '": 6,"Record function id": 0, "Ev Idx": 5\n    }\n  },\n  {\n    "ph": "X",\n    '
        '"cat": "user_annotation",\n    "name": "matmul_x_rows/synchronize",\n    "pid"'
        ': 50515,\n    "tid": 44672914,\n    "ts": 4114144780285.597,\n    "dur": 417.25'
        '9\n    ,\n    "args": {\n      "External id": 7,"Record function id": 0, "Ev Id'
        'x": 6\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name"'
        ': "profile_trace/torch_matmul",\n    "pid": 50515,\n    "tid": 44672914,\n    "'
        'ts": 4114144780724.584,\n    "dur": 724.101\n    ,\n    "args": {\n      "Extern'
        'al id": 8,"Record function id": 0, "Ev Idx": 7\n    }\n  },\n  {\n    "ph": "X",'
        '\n    "cat": "user_annotation",\n    "name": "torch_matmul/dispatch",\n    "pid'
        '": 50515,\n    "tid": 44672914,\n    "ts": 4114144780751.520,\n    "dur": 463.5'
        '48\n    ,\n    "args": {\n      "External id": 9,"Record function id": 0, "Ev I'
        'dx": 8\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "cpu_op",\n    "name": "aten:'
        ':matmul",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114144781014.805'
        ',\n    "dur": 191.221\n    ,\n    "args": {\n      "External id": 10,"Record fun'
        'ction id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 9\n    }\n  '
        '},\n  {\n    "ph": "X",\n    "cat": "cpu_op",\n    "name": "aten::mm",\n    "pid"'
        ': 50515,\n    "tid": 44672914,\n    "ts": 4114144781018.657,\n    "dur": 186.99'
        '4\n    ,\n    "args": {\n      "External id": 11,"Record function id": 0, "Sequ'
        'ence number": 0, "Fwd thread id": 0, "Ev Idx": 10\n    }\n  },\n  {\n    "ph": "'
        'X",\n    "cat": "user_annotation",\n    "name": "torch_matmul/synchronize",\n  '
        '  "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114144781234.171,\n    "dur":'
        ' 209.912\n    ,\n    "args": {\n      "External id": 12,"Record function id": 0'
        ', "Ev Idx": 11\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n '
        '   "name": "profile_trace/iteration",\n    "pid": 50515,\n    "tid": 44672914,'
        '\n    "ts": 4114144781458.852,\n    "dur": 1462.620\n    ,\n    "args": {\n      '
        '"External id": 13,"Record function id": 0, "Ev Idx": 12\n    }\n  },\n  {\n    "'
        'ph": "X",\n    "cat": "user_annotation",\n    "name": "profile_trace/matmul_x_'
        'rows",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114144781465.770,\n '
        '   "dur": 292.156\n    ,\n    "args": {\n      "External id": 14,"Record functi'
        'on id": 0, "Ev Idx": 13\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annot'
        'ation",\n    "name": "matmul_x_rows/dispatch",\n    "pid": 50515,\n    "tid": 4'
        '4672914,\n    "ts": 4114144781481.246,\n    "dur": 19.209\n    ,\n    "args": {\n'
        '      "External id": 15,"Record function id": 0, "Ev Idx": 14\n    }\n  },\n  {'
        '\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "matmul_x_rows/syn'
        'chronize",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114144781511.59'
        '9,\n    "dur": 244.326\n    ,\n    "args": {\n      "External id": 16,"Record fu'
        'nction id": 0, "Ev Idx": 15\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_a'
        'nnotation",\n    "name": "profile_trace/torch_matmul",\n    "pid": 50515,\n    '
        '"tid": 44672914,\n    "ts": 4114144781762.009,\n    "dur": 573.043\n    ,\n    "'
        'args": {\n      "External id": 17,"Record function id": 0, "Ev Idx": 16\n    }'
        '\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "torch_ma'
        'tmul/dispatch",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 41141447817'
        '72.219,\n    "dur": 83.433\n    ,\n    "args": {\n      "External id": 18,"Recor'
        'd function id": 0, "Ev Idx": 17\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "cp'
        'u_op",\n    "name": "aten::matmul",\n    "pid": 50515,\n    "tid": 44672914,\n  '
        '  "ts": 4114144781775.635,\n    "dur": 76.725\n    ,\n    "args": {\n      "Exte'
        'rnal id": 19,"Record function id": 0, "Sequence number": 0, "Fwd thread id":'
        ' 0, "Ev Idx": 18\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "cpu_op",\n    "nam'
        'e": "aten::mm",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 41141447817'
        '76.177,\n    "dur": 75.808\n    ,\n    "args": {\n      "External id": 20,"Recor'
        'd function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 19\n  '
        '  }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "torch'
        '_matmul/synchronize",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 41141'
        '44781865.528,\n    "dur": 466.816\n    ,\n    "args": {\n      "External id": 21'
        ',"Record function id": 0, "Ev Idx": 20\n    }\n  },\n  {\n    "ph": "X",\n    "ca'
        't": "user_annotation",\n    "name": "profile_trace/matmul_x_columns",\n    "pi'
        'd": 50515,\n    "tid": 44672914,\n    "ts": 4114144782340.178,\n    "dur": 576.'
        '294\n    ,\n    "args": {\n      "External id": 22,"Record function id": 0, "Ev'
        ' Idx": 21\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "n'
        'ame": "matmul_x_columns/dispatch",\n    "pid": 50515,\n    "tid": 44672914,\n  '
        '  "ts": 4114144782352.346,\n    "dur": 12.684\n    ,\n    "args": {\n      "Exte'
        'rnal id": 23,"Record function id": 0, "Ev Idx": 22\n    }\n  },\n  {\n    "ph": '
        '"X",\n    "cat": "user_annotation",\n    "name": "matmul_x_columns/synchronize'
        '",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114144782373.364,\n    "'
        'dur": 537.482\n    ,\n    "args": {\n      "External id": 24,"Record function i'
        'd": 0, "Ev Idx": 23\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "Trace",\n    "n'
        'ame": "PyTorch Profiler (0)",\n    "pid": "Spans",\n    "tid": "PyTorch Profil'
        'er",\n    "ts": 4114144777164.000,\n    "dur": 5776.000\n    ,\n    "args": {\n  '
        '    "Op count": 0\n    }\n  },\n  {\n    "ph": "M",\n    "name": "process_sort_in'
        'dex",\n    "ts": 4114144777164.000,\n    "pid": "Spans",\n    "tid": 0,\n    "ar'
        'gs": {\n      "sort_index": 536870912\n    }\n  },\n  {\n    "ph": "i",\n    \n    '
        '"s": "g",\n    "name": "Iteration Start: PyTorch Profiler",\n    "pid": "Trace'
        's",\n    "tid": "Trace PyTorch Profiler",\n    "ts": 4114144777164.000\n    \n  '
        '},\n  {\n    "ph": "i",\n    \n    "s": "g",\n    "name": "Record Window End",\n  '
        '  "pid": "",\n    "tid": "",\n    "ts": 4114144783651.000\n    \n  }\n  ],"traceN'
        'ame": "trace-0-128x128x128.json" }'
    ),
    "trace-1-256x192x320.json": (
        ' {\n    "schemaVersion": 1,\n    "deviceProperties": [],\n    \n      "trace_id"'
        ': "6CA7E967B5D24105B0C7CC9FEC7A8C36",\n      "host_name": "redacted-host",\n  '
        '  "displayTimeUnit": "ms",\n    "baseTimeNanoseconds": 1790857026000000000,\n '
        '   "traceEvents": [\n    \n  {\n    "ph": "M",\n    "name": "thread_name",\n    "'
        'ts": 4114145763767.000,\n    "pid": 50515,\n    "tid": 44672914,\n    "args": {'
        '\n      "name": "thread 44672914 ()"\n    }\n  },\n  {\n    "ph": "M",\n    "name"'
        ': "thread_sort_index",\n    "ts": 4114145763767.000,\n    "pid": 50515,\n    "t'
        'id": 44672914,\n    "args": {\n      "sort_index": 44672914\n    }\n  },\n  {\n   '
        ' "ph": "M",\n    "name": "thread_name",\n    "ts": 4114145763767.000,\n    "pid'
        '": 50515,\n    "tid": 44672914,\n    "args": {\n      "name": "thread 44672914 '
        '()"\n    }\n  },\n  {\n    "ph": "M",\n    "name": "thread_sort_index",\n    "ts":'
        ' 4114145763767.000,\n    "pid": 50515,\n    "tid": 44672914,\n    "args": {\n   '
        '   "sort_index": 44672914\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_ann'
        'otation",\n    "name": "profile_trace/iteration",\n    "pid": 50515,\n    "tid"'
        ': 44672914,\n    "ts": 4114145763841.082,\n    "dur": 4999.007\n    ,\n    "args'
        '": {\n      "External id": 513,"Record function id": 0, "Ev Idx": 0\n    }\n  }'
        ',\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "profile_trac'
        'e/matmul_x_columns",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 411414'
        '5763857.207,\n    "dur": 1465.457\n    ,\n    "args": {\n      "External id": 51'
        '4,"Record function id": 0, "Ev Idx": 1\n    }\n  },\n  {\n    "ph": "X",\n    "ca'
        't": "user_annotation",\n    "name": "matmul_x_columns/dispatch",\n    "pid": 5'
        '0515,\n    "tid": 44672914,\n    "ts": 4114145763889.809,\n    "dur": 38.268\n  '
        '  ,\n    "args": {\n      "External id": 515,"Record function id": 0, "Ev Idx"'
        ': 2\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": '
        '"matmul_x_columns/synchronize",\n    "pid": 50515,\n    "tid": 44672914,\n    "'
        'ts": 4114145763981.363,\n    "dur": 1331.718\n    ,\n    "args": {\n      "Exter'
        'nal id": 516,"Record function id": 0, "Ev Idx": 3\n    }\n  },\n  {\n    "ph": "'
        'X",\n    "cat": "user_annotation",\n    "name": "profile_trace/matmul_x_rows",'
        '\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114145765343.265,\n    "du'
        'r": 2537.677\n    ,\n    "args": {\n      "External id": 517,"Record function i'
        'd": 0, "Ev Idx": 4\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation'
        '",\n    "name": "matmul_x_rows/dispatch",\n    "pid": 50515,\n    "tid": 446729'
        '14,\n    "ts": 4114145765400.426,\n    "dur": 41.268\n    ,\n    "args": {\n     '
        ' "External id": 518,"Record function id": 0, "Ev Idx": 5\n    }\n  },\n  {\n    '
        '"ph": "X",\n    "cat": "user_annotation",\n    "name": "matmul_x_rows/synchron'
        'ize",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114145765458.005,\n  '
        '  "dur": 2412.186\n    ,\n    "args": {\n      "External id": 519,"Record funct'
        'ion id": 0, "Ev Idx": 6\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annot'
        'ation",\n    "name": "profile_trace/torch_matmul",\n    "pid": 50515,\n    "tid'
        '": 44672914,\n    "ts": 4114145767908.168,\n    "dur": 922.713\n    ,\n    "args'
        '": {\n      "External id": 520,"Record function id": 0, "Ev Idx": 7\n    }\n  }'
        ',\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "torch_matmul'
        '/dispatch",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114145767982.0'
        '80,\n    "dur": 242.980\n    ,\n    "args": {\n      "External id": 521,"Record '
        'function id": 0, "Ev Idx": 8\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "cpu_o'
        'p",\n    "name": "aten::matmul",\n    "pid": 50515,\n    "tid": 44672914,\n    "'
        'ts": 4114145767993.348,\n    "dur": 222.169\n    ,\n    "args": {\n      "Extern'
        'al id": 522,"Record function id": 0, "Sequence number": 0, "Fwd thread id": '
        '0, "Ev Idx": 9\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "cpu_op",\n    "name"'
        ': "aten::mm",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114145767995'
        '.098,\n    "dur": 219.836\n    ,\n    "args": {\n      "External id": 523,"Recor'
        'd function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 10\n  '
        '  }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "torch'
        '_matmul/synchronize",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 41141'
        '45768251.993,\n    "dur": 570.805\n    ,\n    "args": {\n      "External id": 52'
        '4,"Record function id": 0, "Ev Idx": 11\n    }\n  },\n  {\n    "ph": "X",\n    "c'
        'at": "user_annotation",\n    "name": "profile_trace/iteration",\n    "pid": 50'
        '515,\n    "tid": 44672914,\n    "ts": 4114145768849.649,\n    "dur": 4020.509\n '
        '   ,\n    "args": {\n      "External id": 525,"Record function id": 0, "Ev Idx'
        '": 12\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name"'
        ': "profile_trace/matmul_x_rows",\n    "pid": 50515,\n    "tid": 44672914,\n    '
        '"ts": 4114145768862.524,\n    "dur": 1776.181\n    ,\n    "args": {\n      "Exte'
        'rnal id": 526,"Record function id": 0, "Ev Idx": 13\n    }\n  },\n  {\n    "ph":'
        ' "X",\n    "cat": "user_annotation",\n    "name": "matmul_x_rows/dispatch",\n  '
        '  "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114145768893.668,\n    "dur":'
        ' 49.994\n    ,\n    "args": {\n      "External id": 527,"Record function id": 0'
        ', "Ev Idx": 14\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n '
        '   "name": "matmul_x_rows/synchronize",\n    "pid": 50515,\n    "tid": 4467291'
        '4,\n    "ts": 4114145768971.597,\n    "dur": 1659.608\n    ,\n    "args": {\n    '
        '  "External id": 528,"Record function id": 0, "Ev Idx": 15\n    }\n  },\n  {\n  '
        '  "ph": "X",\n    "cat": "user_annotation",\n    "name": "profile_trace/torch_'
        'matmul",\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114145770653.681,'
        '\n    "dur": 1147.132\n    ,\n    "args": {\n      "External id": 529,"Record fu'
        'nction id": 0, "Ev Idx": 16\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_a'
        'nnotation",\n    "name": "torch_matmul/dispatch",\n    "pid": 50515,\n    "tid"'
        ': 44672914,\n    "ts": 4114145770688.658,\n    "dur": 152.257\n    ,\n    "args"'
        ': {\n      "External id": 530,"Record function id": 0, "Ev Idx": 17\n    }\n  }'
        ',\n  {\n    "ph": "X",\n    "cat": "cpu_op",\n    "name": "aten::matmul",\n    "p'
        'id": 50515,\n    "tid": 44672914,\n    "ts": 4114145770696.991,\n    "dur": 136'
        '.841\n    ,\n    "args": {\n      "External id": 531,"Record function id": 0, "'
        'Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 18\n    }\n  },\n  {\n    "ph'
        '": "X",\n    "cat": "cpu_op",\n    "name": "aten::mm",\n    "pid": 50515,\n    "'
        'tid": 44672914,\n    "ts": 4114145770698.284,\n    "dur": 135.173\n    ,\n    "a'
        'rgs": {\n      "External id": 532,"Record function id": 0, "Sequence number":'
        ' 0, "Fwd thread id": 0, "Ev Idx": 19\n    }\n  },\n  {\n    "ph": "X",\n    "cat"'
        ': "user_annotation",\n    "name": "torch_matmul/synchronize",\n    "pid": 5051'
        '5,\n    "tid": 44672914,\n    "ts": 4114145770863.517,\n    "dur": 928.879\n    '
        ',\n    "args": {\n      "External id": 533,"Record function id": 0, "Ev Idx": '
        '20\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation",\n    "name": "'
        'profile_trace/matmul_x_columns",\n    "pid": 50515,\n    "tid": 44672914,\n    '
        '"ts": 4114145771818.956,\n    "dur": 1038.160\n    ,\n    "args": {\n      "Exte'
        'rnal id": 534,"Record function id": 0, "Ev Idx": 21\n    }\n  },\n  {\n    "ph":'
        ' "X",\n    "cat": "user_annotation",\n    "name": "matmul_x_columns/dispatch",'
        '\n    "pid": 50515,\n    "tid": 44672914,\n    "ts": 4114145771855.766,\n    "du'
        'r": 27.684\n    ,\n    "args": {\n      "External id": 535,"Record function id"'
        ': 0, "Ev Idx": 22\n    }\n  },\n  {\n    "ph": "X",\n    "cat": "user_annotation"'
        ',\n    "name": "matmul_x_columns/synchronize",\n    "pid": 50515,\n    "tid": 4'
        '4672914,\n    "ts": 4114145771893.284,\n    "dur": 951.249\n    ,\n    "args": {'
        '\n      "External id": 536,"Record function id": 0, "Ev Idx": 23\n    }\n  },\n '
        ' {\n    "ph": "X",\n    "cat": "Trace",\n    "name": "PyTorch Profiler (0)",\n  '
        '  "pid": "Spans",\n    "tid": "PyTorch Profiler",\n    "ts": 4114145763425.000'
        ',\n    "dur": 9474.000\n    ,\n    "args": {\n      "Op count": 0\n    }\n  },\n  {'
        '\n    "ph": "M",\n    "name": "process_sort_index",\n    "ts": 4114145763425.00'
        '0,\n    "pid": "Spans",\n    "tid": 0,\n    "args": {\n      "sort_index": 53687'
        '0912\n    }\n  },\n  {\n    "ph": "i",\n    \n    "s": "g",\n    "name": "Iteration'
        ' Start: PyTorch Profiler",\n    "pid": "Traces",\n    "tid": "Trace PyTorch Pr'
        'ofiler",\n    "ts": 4114145763425.000\n    \n  },\n  {\n    "ph": "i",\n    \n    "'
        's": "g",\n    "name": "Record Window End",\n    "pid": "",\n    "tid": "",\n    '
        '"ts": 4114145774382.000\n    \n  }\n  ],"traceName": "trace-1-256x192x320.json"'
        " }"
    ),
}
RECORDED_REPORT_JSON = (
    '{"boundary":{"apple_gpu_metal_this_host":true,"cuda_kernel_executed":false,"'
    'hardware_counters_read":false,"nvidia_gpu_used":false,"portable_performance_'
    'claim":false,"statement":"Host-side torch.profiler spans and MPS event times'
    " taken on the Apple GPU (Metal) of the host that built this report, under it"
    "s load. The bottleneck is an inference from these timings and the measured D"
    "ay 74 ceilings, not a reading of hardware counters. Nothing here measures CU"
    'DA, NVIDIA, distributed or production performance."},"config":{"analysis":{"'
    'gap_count":2,"timeline_depth":3,"timeline_width":48,"top_n":4},"day_74_ceili'
    'ngs":{"bandwidth_gb_per_second":100.7,"fma_gflops_per_second":4618.0,"torch_'
    'matmul_gflops_per_second":3963.0},"group_size":[32,8],"iterations":2,"name":'
    '"profile-trace-recorded","schema_version":1,"seed":75,"shapes":[[128,128,128'
    '],[256,192,320]],"warmup":1},"config_sha256":"82c3092461f72f3ac1e75197ed44b1'
    '7a33f571949f5a9e13464914f43463bdda","environment":{"macos_version":"27.2","n'
    'umpy_version":"1.26.4","platform_machine":"arm64","platform_system":"Darwin"'
    ',"python_implementation":"CPython","python_version":"3.11.14","torch_float32'
    '_matmul_precision":"highest","torch_version":"2.15.0"},"kind":"profile_trace'
    '_report","label":"host-specific Apple GPU (Metal) measurement","load_average'
    '":{"after":[11.69,12.73,12.48],"before":[11.69,12.73,12.48]},"measured":true'
    ',"profiler":{"activity":"torch.profiler.ProfilerActivity.CPU","device_timer"'
    ':"torch.mps.Event(enable_timing=True)","records_gpu_kernels":false,"statemen'
    't":"ProfilerActivity.CPU records only host-side spans: the dispatch calls an'
    "d the host's wait inside torch.mps.synchronize(). It records no GPU kernel a"
    "nd no GPU kernel time. The GPU work of a side shows only as host wait time i"
    'n its synchronize span."},"report_sha256":"09b85cddb8858fd0aba0433a77a9bac92'
    'f945ac77b6d766a7ca3486abf944f46","schema_version":1,"scope":"apple_gpu_metal'
    '_host_trace_this_host","shapes":[{"analysis":{"iteration_host_ns":{"count":2'
    ',"max":2738118,"median_low":1462620,"min":1462620,"total":4200738},"iteratio'
    'n_name":"profile_trace/iteration","iterations":[{"children_ns":{"profile_tra'
    'ce/matmul_x_columns":925530,"profile_trace/matmul_x_rows":649168,"profile_tr'
    'ace/torch_matmul":724101},"host_ns":2738118,"index":0,"offset_ns":0,"uncover'
    'ed_ns":439319},{"children_ns":{"profile_trace/matmul_x_columns":576294,"prof'
    'ile_trace/matmul_x_rows":292156,"profile_trace/torch_matmul":573043},"host_n'
    's":1462620,"index":1,"offset_ns":2742243,"uncovered_ns":21127}],"ops":[{"cal'
    'ls":2,"max_ns":191221,"name":"aten::matmul","self_ns":5144,"total_ns":267946'
    '},{"calls":2,"max_ns":186994,"name":"aten::mm","self_ns":262802,"total_ns":2'
    '62802},{"calls":2,"max_ns":78393,"name":"matmul_x_columns/dispatch","self_ns'
    '":91077,"total_ns":91077},{"calls":2,"max_ns":754602,"name":"matmul_x_column'
    's/synchronize","self_ns":1292084,"total_ns":1292084},{"calls":2,"max_ns":373'
    '29,"name":"matmul_x_rows/dispatch","self_ns":56538,"total_ns":56538},{"calls'
    '":2,"max_ns":417259,"name":"matmul_x_rows/synchronize","self_ns":661585,"tot'
    'al_ns":661585},{"calls":2,"max_ns":2738118,"name":"profile_trace/iteration",'
    '"self_ns":460446,"total_ns":4200738},{"calls":2,"max_ns":925530,"name":"prof'
    'ile_trace/matmul_x_columns","self_ns":118663,"total_ns":1501824},{"calls":2,'
    '"max_ns":649168,"name":"profile_trace/matmul_x_rows","self_ns":223201,"total'
    '_ns":941324},{"calls":2,"max_ns":724101,"name":"profile_trace/torch_matmul",'
    '"self_ns":73435,"total_ns":1297144},{"calls":2,"max_ns":463548,"name":"torch'
    '_matmul/dispatch","self_ns":279035,"total_ns":546981},{"calls":2,"max_ns":46'
    '6816,"name":"torch_matmul/synchronize","self_ns":676728,"total_ns":676728}],'
    '"threads":[{"busy_ns":4200738,"extent_ns":4204863,"gaps":1,"largest_gaps":[{'
    '"after":"profile_trace/iteration","before":"profile_trace/iteration","gap_ns'
    '":4125,"offset_ns":2738118}],"max_gap_ns":4125,"pid":50515,"tid":44672914,"t'
    'op_level_calls":2,"total_gap_ns":4125}],"top_ops_by_self_time":[{"calls":2,"'
    'max_ns":754602,"name":"matmul_x_columns/synchronize","self_ns":1292084,"tota'
    'l_ns":1292084},{"calls":2,"max_ns":466816,"name":"torch_matmul/synchronize",'
    '"self_ns":676728,"total_ns":676728},{"calls":2,"max_ns":417259,"name":"matmu'
    'l_x_rows/synchronize","self_ns":661585,"total_ns":661585},{"calls":2,"max_ns'
    '":2738118,"name":"profile_trace/iteration","self_ns":460446,"total_ns":42007'
    '38}]},"bottleneck":{"basis":"inference from host-side torch.profiler spans a'
    "nd the measured Day 74 ceilings of this host; no hardware counter, GPU kerne"
    'l profiler or CUDA tool was used","kind":"gpu_wait","self_ns":1292084,"share'
    '_of_iteration_host_time":0.30758500054038124,"sides":[{"compulsory_traffic_f'
    'loor_ns":1952.413108242304,"fma_floor_ns":908.2511909917713,"fraction_of_fma'
    '_ceiling":0.0015760205572013092,"fraction_of_torch_matmul_ceiling":0.0018365'
    '033896431103,"host_observed_gflops_per_second":7.278062933155646,"host_to_fl'
    'oor_ratio":{"compulsory_traffic_floor_ns":295.1701141357422,"fma_floor_ns":6'
    '34.5094900131226,"no_reuse_traffic_floor_ns":3.4455655346584697},"median_dev'
    'ice_event_ns":457000,"median_dispatch_ns":12684,"median_host_ns":576294,"med'
    'ian_synchronize_ns":537482,"no_reuse_traffic_floor_ns":167256.722939424,"sid'
    'e":"matmul_x_columns","synchronize_share_of_host":0.9326524308772953},{"comp'
    'ulsory_traffic_floor_ns":1952.413108242304,"fma_floor_ns":908.2511909917713,'
    '"fraction_of_fma_ceiling":0.003108788424649062,"fraction_of_torch_matmul_cei'
    'ling":0.0036226053356117503,"host_observed_gflops_per_second":14.35638494502'
    '9367,"host_to_floor_ratio":{"compulsory_traffic_floor_ns":149.63841349283854'
    ',"fma_floor_ns":321.66872215270996,"no_reuse_traffic_floor_ns":1.74675190847'
    '67147},"median_device_event_ns":198125,"median_dispatch_ns":19209,"median_ho'
    'st_ns":292156,"median_synchronize_ns":244326,"no_reuse_traffic_floor_ns":167'
    '256.722939424,"side":"matmul_x_rows","synchronize_share_of_host":0.836286093'
    '7307466},{"compulsory_traffic_floor_ns":1952.413108242304,"fma_floor_ns":908'
    '.2511909917713,"fraction_of_fma_ceiling":0.0015849616712738334,"fraction_of_'
    'torch_matmul_ceiling":0.001846922280581015,"host_observed_gflops_per_second"'
    ':7.319352997942563,"host_to_floor_ratio":{"compulsory_traffic_floor_ns":293.'
    '50499521891277,"fma_floor_ns":630.9300837516785,"no_reuse_traffic_floor_ns":'
    '3.4261283488589043},"median_device_event_ns":458792,"median_dispatch_ns":834'
    '33,"median_host_ns":573043,"median_synchronize_ns":209912,"no_reuse_traffic_'
    'floor_ns":167256.722939424,"side":"torch_matmul","synchronize_share_of_host"'
    ':0.3663110796222971}],"top_op":"matmul_x_columns/synchronize"},"check":{"agr'
    'eement_bound":3.0517810957420024e-05,"naive_same_bits":true,"torch_disagreem'
    'ent":0.0},"device_event_ns":{"matmul_x_columns":[557333,457000],"matmul_x_ro'
    'ws":[408625,198125],"torch_matmul":[571583,458792]},"flops":4194304,"index":'
    '0,"k":128,"m":128,"n":128,"timeline":"timeline extent_ns=4204863 columns=48 '
    "ns_per_column=87601.3\\nthread pid=50515 tid=44672914\\n  d0 |AAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|\\n  d1 |...BBBBBBBBBBBBEEEEEEEEHHHHHHHHEEEE"
    "HHHHHHBBBBBBB|\\n  d2 |....CCDDDDDDDDDFFGGGGGGIIIIIIJJFGGGIJJJJJJDDDDDD|\\nleg"
    "end\\n  A profile_trace/iteration\\n  B profile_trace/matmul_x_columns\\n  C ma"
    "tmul_x_columns/dispatch\\n  D matmul_x_columns/synchronize\\n  E profile_trace"
    "/matmul_x_rows\\n  F matmul_x_rows/dispatch\\n  G matmul_x_rows/synchronize\\n "
    " H profile_trace/torch_matmul\\n  I torch_matmul/dispatch\\n  J torch_matmul/s"
    'ynchronize\\n","trace":{"bytes":8242,"events":32,"file":"trace-0-128x128x128.'
    'json","sha256":"2785c119b5bdf7371bb3c2c30e87f266b8121d441f6cf40a441c4732f652'
    '7924"}},{"analysis":{"iteration_host_ns":{"count":2,"max":4999007,"median_lo'
    'w":4020509,"min":4020509,"total":9019516},"iteration_name":"profile_trace/it'
    'eration","iterations":[{"children_ns":{"profile_trace/matmul_x_columns":1465'
    '457,"profile_trace/matmul_x_rows":2537677,"profile_trace/torch_matmul":92271'
    '3},"host_ns":4999007,"index":0,"offset_ns":0,"uncovered_ns":73160},{"childre'
    'n_ns":{"profile_trace/matmul_x_columns":1038160,"profile_trace/matmul_x_rows'
    '":1776181,"profile_trace/torch_matmul":1147132},"host_ns":4020509,"index":1,'
    '"offset_ns":5008567,"uncovered_ns":59036}],"ops":[{"calls":2,"max_ns":222169'
    ',"name":"aten::matmul","self_ns":4001,"total_ns":359010},{"calls":2,"max_ns"'
    ':219836,"name":"aten::mm","self_ns":355009,"total_ns":355009},{"calls":2,"ma'
    'x_ns":38268,"name":"matmul_x_columns/dispatch","self_ns":65952,"total_ns":65'
    '952},{"calls":2,"max_ns":1331718,"name":"matmul_x_columns/synchronize","self'
    '_ns":2282967,"total_ns":2282967},{"calls":2,"max_ns":49994,"name":"matmul_x_'
    'rows/dispatch","self_ns":91262,"total_ns":91262},{"calls":2,"max_ns":2412186'
    ',"name":"matmul_x_rows/synchronize","self_ns":4071794,"total_ns":4071794},{"'
    'calls":2,"max_ns":4999007,"name":"profile_trace/iteration","self_ns":132196,'
    '"total_ns":9019516},{"calls":2,"max_ns":1465457,"name":"profile_trace/matmul'
    '_x_columns","self_ns":154698,"total_ns":2503617},{"calls":2,"max_ns":2537677'
    ',"name":"profile_trace/matmul_x_rows","self_ns":150802,"total_ns":4313858},{'
    '"calls":2,"max_ns":1147132,"name":"profile_trace/torch_matmul","self_ns":174'
    '924,"total_ns":2069845},{"calls":2,"max_ns":242980,"name":"torch_matmul/disp'
    'atch","self_ns":36227,"total_ns":395237},{"calls":2,"max_ns":928879,"name":"'
    'torch_matmul/synchronize","self_ns":1499684,"total_ns":1499684}],"threads":['
    '{"busy_ns":9019516,"extent_ns":9029076,"gaps":1,"largest_gaps":[{"after":"pr'
    'ofile_trace/iteration","before":"profile_trace/iteration","gap_ns":9560,"off'
    'set_ns":4999007}],"max_gap_ns":9560,"pid":50515,"tid":44672914,"top_level_ca'
    'lls":2,"total_gap_ns":9560}],"top_ops_by_self_time":[{"calls":2,"max_ns":241'
    '2186,"name":"matmul_x_rows/synchronize","self_ns":4071794,"total_ns":4071794'
    '},{"calls":2,"max_ns":1331718,"name":"matmul_x_columns/synchronize","self_ns'
    '":2282967,"total_ns":2282967},{"calls":2,"max_ns":928879,"name":"torch_matmu'
    'l/synchronize","self_ns":1499684,"total_ns":1499684},{"calls":2,"max_ns":219'
    '836,"name":"aten::mm","self_ns":355009,"total_ns":355009}]},"bottleneck":{"b'
    'asis":"inference from host-side torch.profiler spans and the measured Day 74'
    " ceilings of this host; no hardware counter, GPU kernel profiler or CUDA too"
    'l was used","kind":"gpu_wait","self_ns":4071794,"share_of_iteration_host_tim'
    'e":0.45144262729840495,"sides":[{"compulsory_traffic_floor_ns":7646.95134061'
    '569,"fma_floor_ns":6811.883932438285,"fraction_of_fma_ceiling":0.00656149719'
    '9312519,"fraction_of_torch_matmul_ceiling":0.0076459737740159515,"host_obser'
    'ved_gflops_per_second":30.300994066425215,"host_to_floor_ratio":{"compulsory'
    '_traffic_floor_ns":135.76129280252658,"fma_floor_ns":152.40424092610675,"no_'
    'reuse_traffic_floor_ns":0.829534680410654},"median_device_event_ns":519750,"'
    'median_dispatch_ns":27684,"median_host_ns":1038160,"median_synchronize_ns":9'
    '51249,"no_reuse_traffic_floor_ns":1251496.8023833167,"side":"matmul_x_column'
    's","synchronize_share_of_host":0.9162836171688372},{"compulsory_traffic_floo'
    'r_ns":7646.95134061569,"fma_floor_ns":6811.883932438285,"fraction_of_fma_cei'
    'ling":0.0038351293772640764,"fraction_of_torch_matmul_ceiling":0.00446899507'
    '0453067,"host_observed_gflops_per_second":17.710627464205505,"host_to_floor_'
    'ratio":{"compulsory_traffic_floor_ns":232.27308778154088,"fma_floor_ns":260.'
    '7473964055379,"no_reuse_traffic_floor_ns":1.4192453361586612},"median_device'
    '_event_ns":1565250,"median_dispatch_ns":41268,"median_host_ns":1776181,"medi'
    'an_synchronize_ns":1659608,"no_reuse_traffic_floor_ns":1251496.8023833167,"s'
    'ide":"matmul_x_rows","synchronize_share_of_host":0.9343687383211509},{"compu'
    'lsory_traffic_floor_ns":7646.95134061569,"fma_floor_ns":6811.883932438285,"f'
    'raction_of_fma_ceiling":0.007382451458295575,"fraction_of_torch_matmul_ceili'
    'ng":0.008602614391725704,"host_observed_gflops_per_second":34.09216083440896'
    '5,"host_to_floor_ratio":{"compulsory_traffic_floor_ns":120.66416522086935,"f'
    'ma_floor_ns":135.45635967254637,"no_reuse_traffic_floor_ns":0.73728754100115'
    '18},"median_device_event_ns":289458,"median_dispatch_ns":152257,"median_host'
    '_ns":922713,"median_synchronize_ns":570805,"no_reuse_traffic_floor_ns":12514'
    '96.8023833167,"side":"torch_matmul","synchronize_share_of_host":0.6186159726'
    '805627}],"top_op":"matmul_x_rows/synchronize"},"check":{"agreement_bound":7.'
    '629540053177894e-05,"naive_same_bits":true,"torch_disagreement":0.0},"device'
    '_event_ns":{"matmul_x_columns":[519750,555417],"matmul_x_rows":[1565250,1598'
    '500],"torch_matmul":[712750,289458]},"flops":31457280,"index":1,"k":320,"m":'
    '256,"n":192,"timeline":"timeline extent_ns=9029076 columns=48 ns_per_column='
    "188105.8\\nthread pid=50515 tid=44672914\\n  d0 |AAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAA|\\n  d1 |BBBBBBBBEEEEEEEEEEEEEEHHHHHEEEEEEEEEHHHHHHBBBBBB"
    "|\\n  d2 |DDDDDDDDGGGGGGGGGGGGGGIJJJJGGGGGGGGGIJJJJJJDDDDD|\\nlegend\\n  A prof"
    "ile_trace/iteration\\n  B profile_trace/matmul_x_columns\\n  C matmul_x_column"
    "s/dispatch\\n  D matmul_x_columns/synchronize\\n  E profile_trace/matmul_x_row"
    "s\\n  F matmul_x_rows/dispatch\\n  G matmul_x_rows/synchronize\\n  H profile_tr"
    'ace/torch_matmul\\n  I torch_matmul/dispatch\\n  J torch_matmul/synchronize\\n"'
    ',"trace":{"bytes":8286,"events":32,"file":"trace-1-256x192x320.json","sha256'
    '":"302347d4ad6a8473adb3bbb35c536fc6338f3e68776669c1175ae5ab86b40cc6"}}],"sum'
    'mary":{"bottlenecks":[{"k":128,"kind":"gpu_wait","m":128,"n":128,"share_of_i'
    'teration_host_time":0.30758500054038124,"top_op":"matmul_x_columns/synchroni'
    'ze"},{"k":320,"kind":"gpu_wait","m":256,"n":192,"share_of_iteration_host_tim'
    'e":0.45144262729840495,"top_op":"matmul_x_rows/synchronize"}]}}'
)


def write_recorded(directory):
    for name, text in RECORDED_TRACES.items():
        (directory / name).write_bytes(text.encode("ascii"))
    (directory / REPORT_FILE).write_bytes(RECORDED_REPORT_JSON.encode("ascii") + b"\n")


def test_the_recorded_report_verifies_as_a_measurement(tmp_path):
    write_recorded(tmp_path)
    config = config_from_mapping(RECORDED_CONFIG)
    report, summary = load_report(tmp_path / REPORT_FILE, config=config)

    assert canonical_json_bytes(report).decode("ascii") == RECORDED_REPORT_JSON
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_host_trace_this_host"
    assert report["label"] == "host-specific Apple GPU (Metal) measurement"
    assert report["environment"]["torch_version"] == "2.15.0"
    assert report["boundary"]["hardware_counters_read"] is False
    assert report["profiler"]["records_gpu_kernels"] is False
    assert summary["bottlenecks"] == [
        "128x128x128:matmul_x_columns/synchronize",
        "256x192x320:matmul_x_rows/synchronize",
    ]
    for entry in report["shapes"]:
        assert entry["bottleneck"]["kind"] == "gpu_wait"
        assert all(
            value > 0
            for values in entry["device_event_ns"].values()
            for value in values
        )


@pytest.mark.parametrize("name", sorted(RECORDED_TRACES))
def test_the_recorded_torch_exports_parse_strictly(name):
    trace = parse_trace_bytes(RECORDED_TRACES[name].encode("ascii"))

    assert isinstance(trace, ParsedTrace)
    check_layout(trace, RECORDED_CONFIG["iterations"])
    categories = {span.category for span in trace.spans}
    assert categories == {"user_annotation", "cpu_op", "Trace"}
    assert {"aten::matmul", "aten::mm"} <= {span.name for span in trace.spans}
    assert {item.name for item in trace.instants} == {
        "Iteration Start: PyTorch Profiler",
        "Record Window End",
    }
    assert any(item.name == "thread_name" for item in trace.metadata)
    document = json.loads(RECORDED_TRACES[name])
    assert document["host_name"] == "redacted-host"
    assert document["traceName"] == name


def test_a_recorded_trace_with_one_changed_byte_is_rejected(tmp_path):
    write_recorded(tmp_path)
    name = sorted(RECORDED_TRACES)[0]
    text = RECORDED_TRACES[name].replace('"ph": "X"', '"ph":  "X"', 1)
    (tmp_path / name).write_bytes(text.encode("ascii"))
    with pytest.raises(ProfileTraceVerificationError, match="trace size"):
        verify_report(json.loads(RECORDED_REPORT_JSON), trace_dir=tmp_path)
