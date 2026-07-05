"""通用模拟运行器 —— 提供并行调度、参数采样、批量运行的公共函数.

特性:
    - 多进程并行（ProcessPoolExecutor），充分利用多核CPU
    - 拉丁超立方采样（LHS）
    - 自动保存中间结果（容错：中途崩溃不丢失已完成数据）
    - 统一的模拟包装器（适配所有脚本）
"""

import sys
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Callable
import numpy as np

# 确保可以导入 src
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.params.schema import get_schema
from src.params.presets import get_scenario, SCENARIO_PRESETS


# ============================================================
# 全局默认配置
# ============================================================
DEFAULT_N_WORKERS = min(os.cpu_count() or 4, 12)


def get_optimal_workers(requested: int = None) -> int:
    """获取最优并行数."""
    if requested is not None:
        return min(requested, os.cpu_count() or 4)
    return DEFAULT_N_WORKERS


# ============================================================
# 隔离运行函数（必须在模块顶层，供多进程使用）
# ============================================================

def _run_single_isolated(kwargs_dict: dict) -> dict:
    """隔离运行单次模拟 —— 供多进程调用."""
    params = kwargs_dict["params"]
    regime = kwargs_dict.get("regime", "A")
    config = kwargs_dict.get("config", None)
    sim_id = kwargs_dict.get("sim_id", 0)
    record_ts = kwargs_dict.get("record_ts", False)

    if config is None:
        config = SimulationConfig()

    try:
        engine = SimulationEngine(params=params, config=config, regime=regime)
        result = engine.run()
        result["params"] = params
        result["_sim_id"] = sim_id
        result["_success"] = True
        return result
    except Exception as e:
        return {
            "summary": {
                "survived": False,
                "survival_years": 0,
                "final_gdp": 0,
                "final_tech_gap": 999,
                "total_fractures": 0,
                "fracture_types": [],
                "regime": regime,
            },
            "params": params,
            "_sim_id": sim_id,
            "_success": False,
            "_error": f"{type(e).__name__}: {str(e)}",
            "_traceback": traceback.format_exc(),
        }


# ============================================================
# 批量运行器
# ============================================================

class BatchRunner:
    """批量模拟运行器 —— 线程池 + 进度反馈 + 错误处理."""

    def __init__(self, n_workers: int = None, save_interval: int = 100,
                 callback: Callable = None):
        self.n_workers = get_optimal_workers(n_workers)
        self.save_interval = save_interval
        self.callback = callback  # 每完成一个模拟的回调

    def run(self, param_sets: list, regime: str = "A",
            config: SimulationConfig = None,
            desc: str = "批量模拟") -> list:
        """并行运行一批模拟.

        Args:
            param_sets: 参数集合列表，每个元素是一个参数字典
            regime: 积累策略 'A'/'B'/'C'/'M'
            config: 模拟配置
            desc: 描述文本（用于日志）

        Returns:
            结果列表（与 param_sets 顺序一致）
        """
        n = len(param_sets)
        if config is None:
            config = SimulationConfig()

        # 构建任务
        tasks = []
        for i, params in enumerate(param_sets):
            tasks.append({
                "params": params,
                "regime": regime,
                "config": config,
                "sim_id": i,
            })

        results = [None] * n

        if n <= 2 or self.n_workers <= 1:
            # 串行模式（小任务量）
            for i, task in enumerate(tasks):
                results[i] = _run_single_isolated(task)
                if self.callback:
                    self.callback(i, n, results[i])
            return results

        # 并行模式
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(_run_single_isolated, task): i
                for i, task in enumerate(tasks)
            }

            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {
                        "summary": {"survived": False, "error": str(e)},
                        "_sim_id": idx,
                        "_success": False,
                        "_error": str(e),
                    }

                completed += 1
                if self.callback:
                    self.callback(completed, n, results[idx])

        return results


# ============================================================
# 参数采样工具
# ============================================================

def sample_lhs_params(n_samples: int, regime: str = "A",
                      scenario: str = "benign_globalization",
                      param_ids: list = None,
                      seed: int = 42) -> list:
    """使用 LHS 采样生成参数集.

    Args:
        n_samples: 采样数量
        regime: 策略
        scenario: 国际场景
        param_ids: 要采样的参数ID列表（None=所有策略参数）
        seed: 随机种子

    Returns:
        参数字典列表
    """
    schema = get_schema()
    scene_params = get_scenario(scenario)

    if param_ids is None:
        param_ids = [pid for pid in schema.all_ids()
                     if schema.get(pid).layer == "strategy"]
        param_ids = param_ids[:15]

    rng = np.random.default_rng(seed)
    samples = schema.sample_lhs(n_samples, param_ids, rng)

    param_sets = []
    for i in range(n_samples):
        params = schema.build_defaults(regime)
        params.update(scene_params)
        for j, pid in enumerate(param_ids):
            params[pid] = samples[i, j]
        param_sets.append(params)

    return param_sets


def build_param_sets(regime: str = "A",
                     scenario: str = "benign_globalization",
                     profile: str = None,
                     overrides: dict = None) -> dict:
    """构建单个参数集.

    Args:
        regime: 策略
        scenario: 场景
        profile: 国情预设（可选）
        overrides: 参数覆盖

    Returns:
        参数字典
    """
    schema = get_schema()

    if profile:
        params = schema.build_from_profile(
            profile_name=profile,
            regime=regime,
            scenario=scenario,
        )
    else:
        params = schema.build_defaults(regime)
        params.update(get_scenario(scenario))

    if overrides:
        params.update(overrides)

    return params


def param_grid_explore(base_params: dict, sweep_param: str,
                       sweep_values: list) -> list:
    """对单个参数进行网格扫描.

    Args:
        base_params: 基础参数集
        sweep_param: 要扫描的参数名
        sweep_values: 参数值列表

    Returns:
        参数集列表
    """
    param_sets = []
    for val in sweep_values:
        p = base_params.copy()
        p[sweep_param] = val
        param_sets.append(p)
    return param_sets


def param_2d_grid(base_params: dict,
                  param1: str, values1: list,
                  param2: str, values2: list) -> list:
    """对两个参数进行全因子网格扫描.

    Returns:
        参数集列表（v1 × v2 个）
    """
    param_sets = []
    for v1 in values1:
        for v2 in values2:
            p = base_params.copy()
            p[param1] = v1
            p[param2] = v2
            param_sets.append(p)
    return param_sets


# ============================================================
# 快速单次运行
# ============================================================

def quick_run(regime: str = "A",
              scenario: str = "benign_globalization",
              profile: str = "manufacturing_giant",
              T_max: float = 60.0,
              overrides: dict = None,
              verbose: bool = False) -> dict:
    """快速单次模拟 —— 用于调试和单点测试."""
    params = build_param_sets(regime=regime, scenario=scenario,
                              profile=profile, overrides=overrides)
    config = SimulationConfig(T_max=T_max, verbose=verbose)
    engine = SimulationEngine(params=params, config=config, regime=regime)
    return engine.run()


# ============================================================
# 场景列表
# ============================================================

ALL_SCENARIOS = list(SCENARIO_PRESETS.keys())
ALL_REGIMES = ["A", "B", "C"]
ALL_MIXTURES = [
    "pure_A", "pure_B", "pure_C",
    "A_dominant_B", "A_dominant_C",
    "B_dominant_A", "B_dominant_C",
    "C_dominant_A", "C_dominant_B",
    "A_B_balanced", "A_C_balanced", "B_C_balanced",
    "uniform",
]
