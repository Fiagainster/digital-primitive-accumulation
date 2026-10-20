"""v2.2 并行压力测试 — 验证没有策略是100%安全的."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

def run_one(args):
    regime, sc_id, seed = args
    from src.core.engine import SimulationEngine
    from src.core.config import SimulationConfig
    from src.params.schema import get_schema
    from src.params.presets import get_scenario

    schema = get_schema()
    config = SimulationConfig(T_max=60.0, verbose=False)
    params = schema.build_defaults(regime)
    scene = get_scenario(sc_id)
    params.update(scene)
    config.random_seed = seed
    engine = SimulationEngine(params=params, config=config, regime=regime)
    result = engine.run()
    s = result['summary']
    return {
        'regime': regime, 'scenario': sc_id, 'seed': seed,
        'survived': s['survived'], 'years': s['survival_years'],
        'gdp': s['final_gdp'], 'fracture': s.get('terminal_fracture', ''),
        'total_fractures': s['total_fractures'],
    }

def main():
    tasks = []
    for regime in ['A', 'B', 'C']:
        for sc_id in ['benign_globalization', 'tech_cold_war', 'resource_crash',
                       'tech_boom', 'global_recession', 'full_containment',
                       'tech_explosion', 'resource_war']:
            for i in range(8):
                tasks.append((regime, sc_id, 400 + i))

    print(f'总任务: {len(tasks)}, 并行: 16进程')
    results = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(run_one, t): t for t in tasks}
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            results.append(r)
            if (i + 1) % 32 == 0:
                print(f'  进度: {i+1}/{len(tasks)}')

    # === 汇总 ===
    by_key = defaultdict(list)
    for r in results:
        by_key[(r['regime'], r['scenario'])].append(r)

    scenarios = ['benign_globalization', 'tech_cold_war', 'resource_crash',
                 'tech_boom', 'global_recession', 'full_containment',
                 'tech_explosion', 'resource_war']

    print('\n' + '=' * 75)
    print('v2.2 全场景存活率矩阵 (每格8次模拟, T=60年)')
    print('=' * 75)
    header = f"{'场景':<22} {'A存活率':<10} {'B存活率':<10} {'C存活率':<10} {'崩溃类型'}"
    print(header)
    print('-' * 75)

    for sc_id in scenarios:
        row = f'{sc_id:<22}'
        crash_info = []
        for regime in ['A', 'B', 'C']:
            rs = by_key.get((regime, sc_id), [])
            if not rs:
                row += 'N/A       '
                continue
            surv = sum(1 for r_ in rs if r_['survived'])
            rate = surv / len(rs) * 100
            row += f'{rate:<10.0f}%'
            for r_ in rs:
                if not r_['survived']:
                    crash_info.append(f'{regime}:{r_["fracture"]}')
        if crash_info:
            row += ' ' + ', '.join(set(crash_info))
        print(row)

    # === 全局统计 ===
    print('\n' + '=' * 75)
    print('全局汇总')
    print('=' * 75)
    for regime in ['A', 'B', 'C']:
        rs = [r for r in results if r['regime'] == regime]
        surv = sum(1 for r in rs if r['survived'])
        crashes = [r for r in rs if not r['survived']]
        print(f'\n策略{regime}: 总存活率 = {surv}/{len(rs)} = {surv/len(rs)*100:.0f}%')
        if crashes:
            types = {}
            for c in crashes:
                t = c['fracture'] or 'unknown'
                types[t] = types.get(t, 0) + 1
            print(f'  崩溃类型分布: {types}')
        else:
            print(f'  0次崩溃 — 策略过于完美，需要进一步调整!')

    print('\n完成!')

if __name__ == '__main__':
    main()
