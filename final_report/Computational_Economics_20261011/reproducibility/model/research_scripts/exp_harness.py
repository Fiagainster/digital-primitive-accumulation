"""统一实验运行器 (v3.8) —— 面向论文的可复现批量模拟。

与 `research_scripts/common/runner.py` 的 BatchRunner 相比，本模块聚焦"可复核"：

  * 每个任务由 (params, regime, T_max, seed) 完全决定；种子在引擎 ``run()``
    内通过 ``np.random.seed`` 播种，因此同种子必然同结果、异种子给出不同轨迹。
  * 结果记录**完整 summary**，包含 ``cost_by_group``（五群组累计代价）、
    ``evaluation``（CDI + 五维分解）与 ``dependency_info``，使论文的成本表、
    五维得分表、断裂表都能逐格复算，不必再依赖任何"未落盘"的中间量。
  * 分批落盘为 JSON，支持 ``skip_existing`` 断点续跑。

用法::

    from research_scripts.exp_harness import run_batch, make_task
    run_batch("census", tasks, workers=12)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 实验产物根目录（与论文归档分离，便于比对）
RUN_ROOT = _PROJECT_ROOT / "experiments_runs" / "v38"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_WORKERS = min(os.cpu_count() or 4, 12)


# --------------------------------------------------------------------------
# 种子派生
# --------------------------------------------------------------------------

def derive_seed(*parts) -> int:
    """由稳定的标签派生一个确定的 32 位种子.

    同 (batch, 配置键, 重复号) 永远得到同一种子 —— 这既是复现的前提，
    也让"同配置的多次重复"彼此独立（重复号不同 → 种子不同）。
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# --------------------------------------------------------------------------
# JSON 序列化
# --------------------------------------------------------------------------

def _jsonable(obj):
    """把 numpy 标量/数组等转换成本 JSON 可序列化的结构."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


# --------------------------------------------------------------------------
# 单任务执行（模块级，供多进程 pickling）
# --------------------------------------------------------------------------

def _run_task(task: dict) -> dict:
    from src.core.engine import SimulationEngine
    from src.core.config import SimulationConfig

    cfg_kwargs = {"T_max": float(task["T_max"]), "verbose": False,
                  "random_seed": int(task["seed"])}
    cfg_kwargs.update(task.get("config_overrides") or {})

    try:
        engine = SimulationEngine(
            params=task["params"],
            config=SimulationConfig(**cfg_kwargs),
            regime=task["regime"],
            config_path=task.get("config_path"),
        )
        res = engine.run()
        out = {
            "meta": task.get("meta", {}),
            "seed": int(task["seed"]),
            "ok": True,
            "summary": _jsonable(res["summary"]),
        }
        if task.get("save_ts"):
            df = res["dataframe"]
            out["timeseries"] = {c: _jsonable(df[c].tolist()) for c in df.columns}
        return out
    except Exception as exc:  # noqa: BLE001 - 单次失败不应中断整批
        import traceback
        return {
            "meta": task.get("meta", {}),
            "seed": int(task["seed"]),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# --------------------------------------------------------------------------
# 批量执行
# --------------------------------------------------------------------------

def run_batch(tag: str, tasks: list, workers: int = None,
              skip_existing: bool = False, verbose: bool = True) -> list:
    """并行执行一批任务并落盘为 ``RUN_ROOT/<tag>.json``.

    Returns: 结果列表（顺序与 tasks 一致；已跳过的批次从磁盘读回）。
    """
    out_path = RUN_ROOT / f"{tag}.json"

    if skip_existing and out_path.exists():
        if verbose:
            print(f"[skip] {tag}: 已存在 {out_path.name}")
        with open(out_path, "r", encoding="utf-8") as fh:
            return json.load(fh)["results"]

    if not tasks:
        if verbose:
            print(f"[warn] {tag}: 空任务集，跳过")
        return []

    n = len(tasks)
    workers = workers or DEFAULT_WORKERS
    t0 = time.time()
    if verbose:
        print(f"[run ] {tag}: {n} 次 × 最多 {workers} 进程 ...", flush=True)

    results = [None] * n
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_task, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[idx] = {"meta": tasks[idx].get("meta", {}),
                                "seed": tasks[idx].get("seed"),
                                "ok": False, "error": str(exc)}
            done += 1
            if verbose and (done % 50 == 0 or done == n):
                el = time.time() - t0
                print(f"       {tag}: {done}/{n}  ({el:.0f}s, {el/max(done,1):.2f}s/次)",
                      flush=True)

    n_ok = sum(1 for r in results if r and r.get("ok"))
    payload = {
        "tag": tag,
        "n_tasks": n,
        "n_ok": n_ok,
        "workers": workers,
        "elapsed_s": round(time.time() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    if verbose:
        print(f"[done] {tag}: {n_ok}/{n} 成功, {payload['elapsed_s']}s -> {out_path.name}",
              flush=True)
    return results


# --------------------------------------------------------------------------
# 便捷构造
# --------------------------------------------------------------------------

def make_task(regime: str, params: dict, T_max: float, seed: int,
              meta: dict = None, config_overrides: dict = None,
              save_ts: bool = False) -> dict:
    return {
        "regime": regime,
        "params": params,
        "T_max": T_max,
        "seed": seed,
        "meta": meta or {},
        "config_overrides": config_overrides or {},
        "save_ts": save_ts,
    }


def load_batch(tag: str) -> dict:
    """读回已落盘的批次."""
    with open(RUN_ROOT / f"{tag}.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m}m{s}s" if h else (f"{m}m{s}s" if m else f"{s}s")
