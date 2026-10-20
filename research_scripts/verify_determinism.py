#!/usr/bin/env python
"""确定性核验 —— 同种子三次逐值一致，异种子给出不同轨迹。

背景
----
v3.8 的可复现性约定要求每次 run 的伪随机流由其配置标识确定性地播种
（``SimulationEngine.run()`` 入口执行 ``np.random.seed(config.random_seed % 2**32)``，
批量运行器再用 ``SeedSequence`` 派生每任务子种子）。本脚本是该约定的验收器：
对同一个配置（manufacturing_giant x B x benign_globalization, T=50）跑三次同种子
（42）与两次异种子（43/44），核对

* ``SAME`` 三次的 存活/终止年/GDP/福利/CDI/复合依附/分组代价/五维评价 必须逐值相同；
* ``DIFF`` 两次应与 ``SAME`` 不同。

任一条不满足，说明播种链在某一层断裂（入口未播种、子种子串扰、或迭代顺序不稳定）。
脚本只读，不写入任何实验产物。

运行方式:
    python research_scripts/verify_determinism.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from concurrent.futures import ProcessPoolExecutor


def one(args):
    from src.core.engine import SimulationEngine
    from src.core.config import SimulationConfig
    from src.params.schema import get_schema
    tag, seed = args
    s = get_schema()
    p = s.build_from_profile(profile_name='manufacturing_giant', regime='B',
                             scenario='benign_globalization')
    e = SimulationEngine(params=p,
                         config=SimulationConfig(T_max=50.0, verbose=False,
                                                 random_seed=seed),
                         regime='B')
    r = e.run()
    sm = r['summary']
    cb = sm.get('cost_by_group', {})
    return (tag, seed, sm['survived'], round(sm['survival_years'], 3),
            round(sm['final_gdp'], 3), round(sm.get('final_welfare', 0), 6),
            sm.get('cdi'), sm.get('composite_dependency'),
            {k: round(v, 3) for k, v in cb.items()},
            'five_dims' in (sm.get('evaluation') or {}))


if __name__ == '__main__':
    tasks = [('SAME', 42), ('SAME', 42), ('SAME', 42),
             ('DIFF', 43), ('DIFF', 44)]
    with ProcessPoolExecutor(max_workers=5) as ex:
        for res in ex.map(one, tasks):
            print(res)
