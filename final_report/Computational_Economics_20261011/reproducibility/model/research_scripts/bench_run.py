#!/usr/bin/env python
"""批量吞吐基准 —— 测 6 国情 x 3 策略在给定 T 下的墙钟时间与单次耗时。

用途
----
在开跑大规模批次（census / 参数扫描）之前，用本脚本估算总时长与并行度是否合理；
也用于换机后回归：同一 T 下单次耗时应保持同量级。

运行方式:
    python research_scripts/bench_run.py          # 默认 T=50
    python research_scripts/bench_run.py 80       # 指定 T
"""

import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from concurrent.futures import ProcessPoolExecutor


def one(args):
    from src.core.engine import SimulationEngine
    from src.core.config import SimulationConfig
    from src.params.schema import get_schema
    prof, reg, T = args
    s = get_schema()
    p = s.build_from_profile(profile_name=prof, regime=reg,
                             scenario='benign_globalization')
    e = SimulationEngine(params=p,
                         config=SimulationConfig(T_max=T, verbose=False),
                         regime=reg)
    r = e.run()
    sm = r['summary']
    return (prof, reg, sm['survived'], round(sm['survival_years'], 1),
            len(r['dataframe']))


if __name__ == '__main__':
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    profs = ['manufacturing_giant', 'resource_rich', 'digital_leapfrogger',
             'post_soviet', 'small_open', 'populous_weak_industry']
    tasks = [(p, r, T) for p in profs for r in 'ABC']
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for res in ex.map(one, tasks):
            print(res, flush=True)
    wall = time.time() - t0
    print('WALL %.1fs for %d runs (T=%g) -> %.1fs/run'
          % (wall, len(tasks), T, wall / len(tasks)))
