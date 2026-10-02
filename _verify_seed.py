import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
