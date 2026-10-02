import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
