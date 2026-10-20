"""结果存储系统 —— 为每个实验提供结构化的结果存储与检索.

特性:
    - 自动创建实验专属子目录
    - 支持 HDF5、CSV、JSON、Pickle 多种格式
    - 元数据自动记录（参数、时间戳、版本）
    - 结果查询与加载接口
"""

import json
import os
import pickle
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import numpy as np


# ============================================================
# 全局输出目录
# ============================================================
RESULT_ROOT = Path(__file__).resolve().parent.parent / "output" / "results"
FIGURE_ROOT = Path(__file__).resolve().parent.parent / "output" / "figures"


class ResultStore:
    """实验结果存储器 —— 每个实验一个实例."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"

        # 创建输出目录
        self.result_dir = RESULT_ROOT / self.run_id
        self.fig_dir = FIGURE_ROOT / self.run_id
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # 元数据
        self.metadata = {
            "experiment": experiment_name,
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "parameters": {},
            "summary": {},
        }

        # 结果缓存（内存中）
        self._results: list = []
        self._summaries: list = []

    # ================================================================
    # 元数据
    # ================================================================

    def set_parameters(self, params: dict):
        """记录实验参数."""
        self.metadata["parameters"] = self._make_serializable(params)

    def set_config(self, config: dict):
        """记录配置信息."""
        self.metadata["config"] = self._make_serializable(config)

    # ================================================================
    # 结果存储
    # ================================================================

    def add_result(self, result: dict):
        """添加一次模拟的完整结果（内存中缓存）."""
        summary = result.get("summary", {})
        summary["sim_index"] = len(self._results)
        self._results.append(result)
        self._summaries.append(summary)

    def add_batch(self, results: list):
        """批量添加结果."""
        for r in results:
            self.add_result(r)

    # ================================================================
    # 持久化
    # ================================================================

    def save_all(self):
        """保存所有结果到磁盘."""
        self.metadata["completed_at"] = datetime.now().isoformat()
        self.metadata["total_simulations"] = len(self._results)
        self.metadata["summary"]["total"] = len(self._results)
        self.metadata["summary"]["survived"] = sum(
            1 for s in self._summaries if s.get("survived", False)
        )

        # 保存摘要 CSV
        self._save_summary_csv()

        # 保存完整结果 JSON（仅摘要，完整数据用 HDF5）
        self._save_summary_json()

        # 保存元数据
        self._save_metadata()

        # 保存完整结果 Pickle（Python 对象）
        self._save_full_pickle()

        # 如果有时间序列数据，保存 HDF5
        self._save_hdf5_if_available()

    def _save_summary_csv(self):
        """保存摘要 CSV —— 每行一次模拟."""
        if not self._summaries:
            return

        import csv
        path = self.result_dir / "summary.csv"

        # 收集所有列名
        all_keys = set()
        for s in self._summaries:
            # 展平一层
            for k, v in s.items():
                if isinstance(v, dict):
                    for sub_k in v:
                        all_keys.add(f"{k}_{sub_k}")
                elif not isinstance(v, (list, tuple)):
                    all_keys.add(k)

        sorted_keys = sorted(all_keys)

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(sorted_keys)
            for s in self._summaries:
                row = []
                for k in sorted_keys:
                    if "_" in k and k.split("_")[0] in [kk for kk in all_keys if isinstance(s.get(kk), dict)]:
                        # 嵌套键
                        parent = k.split("_", 1)[0]
                        child = k.split("_", 1)[1]
                        val = s.get(parent, {}).get(child, "")
                    else:
                        val = s.get(k, "")
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val, ensure_ascii=False)
                    row.append(val)
                writer.writerow(row)

    def _save_summary_json(self):
        """保存摘要 JSON."""
        path = self.result_dir / "summary.json"
        data = {
            "metadata": self.metadata,
            "summaries": self._make_serializable(self._summaries),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_metadata(self):
        """保存元数据."""
        path = self.result_dir / "metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _save_full_pickle(self):
        """保存完整结果到 Pickle（供 Python 后续分析）."""
        path = self.result_dir / "full_results.pkl"
        data = {
            "metadata": self.metadata,
            "results": self._results,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_hdf5_if_available(self):
        """如果结果包含时间序列，保存为 HDF5."""
        try:
            import h5py
        except ImportError:
            return

        # 检查是否有时间序列
        has_ts = False
        for r in self._results[:1]:
            if "time_series" in r and r["time_series"]:
                has_ts = True
                break

        if not has_ts:
            return

        path = self.result_dir / "time_series.h5"
        with h5py.File(path, "w") as f:
            f.attrs["experiment"] = self.experiment_name
            f.attrs["run_id"] = self.run_id
            f.attrs["timestamp"] = self.timestamp

            for i, result in enumerate(self._results):
                ts = result.get("time_series", {})
                if not ts:
                    continue

                grp = f.create_group(f"sim_{i}")
                grp.attrs["regime"] = result.get("summary", {}).get("regime", "?")
                grp.attrs["survived"] = result.get("summary", {}).get("survived", False)

                for key, arr in ts.items():
                    if isinstance(arr, (np.ndarray, list)):
                        grp.create_dataset(key, data=np.array(arr), compression="gzip")

    # ================================================================
    # 查询接口
    # ================================================================

    def get_summaries(self) -> list:
        """获取所有模拟摘要."""
        return self._summaries

    def get_survival_rate(self) -> float:
        """获取存活率."""
        if not self._summaries:
            return 0.0
        return sum(1 for s in self._summaries if s.get("survived", False)) / len(self._summaries)

    def get_summary_stats(self) -> dict:
        """计算摘要统计."""
        if not self._summaries:
            return {}

        surv = [s.get("survived", False) for s in self._summaries]
        yrs = [s.get("survival_years", 0) for s in self._summaries]
        gdps = [s.get("final_gdp", 0) for s in self._summaries]
        gaps = [s.get("final_tech_gap", 0) for s in self._summaries]
        cdis = [s.get("cdi", float("nan")) for s in self._summaries]
        fractures = [s.get("total_fractures", 0) for s in self._summaries]

        cdi_valid = [c for c in cdis if not np.isnan(c)]

        return {
            "n": len(self._summaries),
            "survival_rate": np.mean(surv),
            "mean_years": np.mean(yrs),
            "std_years": np.std(yrs),
            "mean_gdp": np.mean(gdps),
            "median_gdp": np.median(gdps),
            "mean_tech_gap": np.mean(gaps),
            "mean_cdi": np.mean(cdi_valid) if cdi_valid else float("nan"),
            "mean_fractures": np.mean(fractures),
        }

    # ================================================================
    # 图保存路径
    # ================================================================

    def figure_path(self, name: str) -> Path:
        """获取图表保存路径."""
        return self.fig_dir / name

    # ================================================================
    # 静态：加载历史结果
    # ================================================================

    @staticmethod
    def load_run(run_dir: str) -> dict:
        """加载一次历史运行."""
        run_dir = Path(run_dir)
        pkl_path = run_dir / "full_results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        json_path = run_dir / "summary.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"No results found in {run_dir}")

    @staticmethod
    def list_runs() -> list:
        """列出所有历史运行."""
        runs = []
        if RESULT_ROOT.exists():
            for d in sorted(RESULT_ROOT.iterdir(), reverse=True):
                if d.is_dir():
                    meta_path = d / "metadata.json"
                    if meta_path.exists():
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        runs.append({
                            "run_id": meta.get("run_id", d.name),
                            "experiment": meta.get("experiment", "?"),
                            "started": meta.get("started_at", "?"),
                            "simulations": meta.get("total_simulations", "?"),
                        })
        return runs

    # ================================================================
    # 工具
    # ================================================================

    @staticmethod
    def _make_serializable(obj):
        """将 numpy 类型转换为可 JSON 序列化的类型."""
        if isinstance(obj, dict):
            return {k: ResultStore._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ResultStore._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
