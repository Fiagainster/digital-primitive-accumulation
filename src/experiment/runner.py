"""批量实验运行器 —— 参数空间探索与并行调度."""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import time
from ..core.engine import SimulationEngine
from ..core.config import SimulationConfig
from ..params.schema import get_schema
from ..params.presets import get_scenario
from ..io.exporter import ResultExporter


class ExperimentRunner:
    """批量实验管理器."""

    def __init__(self, output_dir: str = "experiments/results"):
        self.schema = get_schema()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_single(self, params: dict, regime: str = "A",
                   config: SimulationConfig = None) -> dict:
        """运行单次模拟."""
        if config is None:
            config = SimulationConfig()
        engine = SimulationEngine(params=params, config=config, regime=regime)
        result = engine.run()
        result["params"] = params
        return result

    def run_comparison(self, regimes: list = None, scenarios: list = None,
                       T_max: float = 50.0, n_repeats: int = 1) -> list:
        """运行策略-场景交叉比较."""
        if regimes is None:
            regimes = ["A", "B", "C"]
        if scenarios is None:
            scenarios = ["benign_globalization", "tech_cold_war", "resource_crash"]

        results = []
        config = SimulationConfig(T_max=T_max, verbose=False)

        for regime in regimes:
            for scenario_name in scenarios:
                for rep in range(n_repeats):
                    params = self.schema.build_defaults(regime)
                    scene_params = get_scenario(scenario_name)
                    params.update(scene_params)
                    config.random_seed = 42 + rep

                    result = self.run_single(params, regime, config)
                    result["summary"]["scenario"] = scenario_name
                    result["summary"]["repeat"] = rep
                    results.append(result)

                    surv = "存活" if result["summary"]["survived"] else "崩溃"
                    s = result["summary"]
                    extra = ""
                    if not s["survived"]:
                        extra = f" [{s.get('terminal_fracture', '?')}]"
                    if s.get("total_fractures", 0) >= 2:
                        extra += f" 级联x{s['total_fractures']}"
                    print(f"  策略{regime} × {scenario_name} [{rep+1}/{n_repeats}]: "
                          f"{surv}{extra} ({s['survival_years']:.1f}年) "
                          f"G={s['final_tech_gap']:.1f}")

        return results

    def run_batch(self, param_sets: list, regime: str = "A",
                  config: SimulationConfig = None, n_workers: int = None) -> list:
        """并行运行一批模拟."""
        if config is None:
            config = SimulationConfig()

        results = []
        n = len(param_sets)

        if n_workers is None:
            n_workers = min(os.cpu_count() or 4, n)

        if n_workers <= 1 or n <= 3:
            for i, params in enumerate(param_sets):
                result = self.run_single(params, regime, config)
                results.append(result)
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{n}")
            return results

        # 并行模式
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_isolated, params, regime, config): i
                for i, params in enumerate(param_sets)
            }
            results = [None] * n
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"  Error in simulation {idx}: {e}")
                    results[idx] = {"summary": {"survived": False, "error": str(e)}}

        return results

    def run_experiment(self, name: str, regimes: list, n_samples: int = 200,
                       scenario: str = "benign_globalization",
                       T_max: float = 60.0) -> list:
        """运行一个完整的实验（LHS采样 + 批量模拟）."""
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"Regimes: {regimes}, Samples per regime: {n_samples}")
        print(f"Scenario: {scenario}, T_max: {T_max}年")
        print(f"{'='*60}\n")

        scene_params = get_scenario(scenario)
        config = SimulationConfig(T_max=T_max)

        all_results = []

        for regime in regimes:
            print(f"\n--- 策略 {regime} ---")

            # 采样策略参数
            strategy_ids = [pid for pid in self.schema.all_ids()
                          if self.schema.get(pid).layer == "strategy"]
            strategy_ids = strategy_ids[:15]  # 限制维度
            rng = np.random.default_rng(42)
            samples = self.schema.sample_lhs(n_samples, strategy_ids, rng)

            param_sets = []
            for i in range(n_samples):
                params = self.schema.build_defaults(regime)
                params.update(scene_params)
                for j, pid in enumerate(strategy_ids):
                    params[pid] = samples[i, j]
                param_sets.append(params)

            config.random_seed = 42
            batch_results = self.run_batch(param_sets, regime, config, n_workers=6)

            for r in batch_results:
                r["summary"]["scenario"] = scenario
                r["summary"]["regime"] = regime
                if "params" not in r:
                    r["params"] = param_sets[len(all_results) % len(param_sets)]

            all_results.extend(batch_results)

            survived = sum(1 for r in batch_results if r["summary"]["survived"])
            print(f"  策略{regime}: {survived}/{n_samples} 存活 ({100*survived/n_samples:.1f}%)")

        # 导出
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"{name}_{timestamp}.h5")
        ResultExporter.to_hdf5(all_results, filepath)

        csv_path = os.path.join(self.output_dir, f"{name}_{timestamp}_summary.csv")
        ResultExporter.to_csv_summary(all_results, csv_path)

        return all_results


def _run_single_isolated(params: dict, regime: str, config: SimulationConfig) -> dict:
    """隔离运行单次模拟（用于并行进程）."""
    engine = SimulationEngine(params=params, config=config, regime=regime)
    result = engine.run()
    result["params"] = params
    return result
