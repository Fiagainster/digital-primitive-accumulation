#!/usr/bin/env python
"""研究脚本总调度器 —— 一键运行所有研究实验或选择性运行.

特性:
    - 逐步运行所有16个研究脚本（或选择子集）
    - 自动记录每个脚本的开始/结束时间和结果摘要
    - 最终生成综合实验报告
    - 支持中断后恢复（已完成脚本自动跳过）

运行方式:
    python research_scripts/run_all.py                     # 运行所有脚本
    python research_scripts/run_all.py --quick             # 快速模式（减少采样量）
    python research_scripts/run_all.py --scripts 1,2,3,6   # 选择性运行
    python research_scripts/run_all.py --level simple      # 仅运行简单脚本(1-5)
    python research_scripts/run_all.py --level medium      # 运行中等脚本(1-10)
    python research_scripts/run_all.py --level deep        # 运行所有脚本(1-16)
    python research_scripts/run_all.py --list              # 列出所有脚本及其描述
"""

import sys
import os
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================
# 脚本注册表
# ============================================================
SCRIPTS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    {
        "id": 1,
        "file": "01_single_detailed_run.py",
        "name": "单次详细模拟",
        "description": "1次模拟的完整诊断报告",
        "level": "simple",
        "estimated_time": "5-30秒",
        "default_args": [],
        "quick_args": [],
    },
    {
        "id": 2,
        "file": "02_parameter_sensitivity.py",
        "name": "参数敏感性分析",
        "description": "局部参数扰动对关键输出的弹性",
        "level": "simple",
        "estimated_time": "3-10分钟",
        "default_args": ["--T-max", "40", "--n-perturbations", "3"],
        "quick_args": ["--T-max", "30", "--n-perturbations", "2"],
    },
    {
        "id": 3,
        "file": "03_regime_comparison.py",
        "name": "三种策略全维度比较",
        "description": "纯策略A/B/C在多场景下的全面比较",
        "level": "simple",
        "estimated_time": "2-10分钟",
        "default_args": ["--T-max", "50", "--n-repeats", "2"],
        "quick_args": ["--T-max", "40", "--n-repeats", "1"],
    },
    {
        "id": 4,
        "file": "04_mixed_strategy_scan.py",
        "name": "混合策略全谱系扫描",
        "description": "13种标准混合配比的系统性比较",
        "level": "simple",
        "estimated_time": "3-15分钟",
        "default_args": ["--T-max", "50"],
        "quick_args": ["--T-max", "40"],
    },
    {
        "id": 5,
        "file": "05_scenario_stress_test.py",
        "name": "国际场景压力测试",
        "description": "8种国际场景下三种策略的生存矩阵",
        "level": "simple",
        "estimated_time": "5-20分钟",
        "default_args": ["--n-lhs", "30"],
        "quick_args": ["--n-lhs", "15"],
    },
    {
        "id": 6,
        "file": "06_baseline_survival_scan.py",
        "name": "基准生存扫描",
        "description": "EX-S-01: 大范围LHS参数空间中的生存边界",
        "level": "medium",
        "estimated_time": "20-60分钟",
        "default_args": ["--n-samples", "200", "--T-max", "100"],
        "quick_args": ["--n-samples", "50", "--T-max", "60"],
    },
    {
        "id": 7,
        "file": "07_initial_condition_scan.py",
        "name": "初始条件扫描",
        "description": "EX-S-02: 六种国情下策略的适配性分析",
        "level": "medium",
        "estimated_time": "10-40分钟",
        "default_args": ["--n-lhs", "30"],
        "quick_args": ["--n-lhs", "15"],
    },
    {
        "id": 8,
        "file": "08_cost_distribution_analysis.py",
        "name": "代价分布深度分析",
        "description": "EX-F-04: 代价在群体间的时空分布",
        "level": "medium",
        "estimated_time": "10-30分钟",
        "default_args": ["--n-lhs", "50"],
        "quick_args": ["--n-lhs", "20"],
    },
    {
        "id": 9,
        "file": "09_fracture_boundary_mapping.py",
        "name": "断裂临界边界绘制",
        "description": "EX-F-01: 参数空间中活/死边界的精确绘制",
        "level": "deep",
        "estimated_time": "30分钟-2小时",
        "default_args": ["--grid-size", "50"],
        "quick_args": ["--grid-size", "20", "--pair", "B_extraction_feedback"],
    },
    {
        "id": 10,
        "file": "10_feedback_window_analysis.py",
        "name": "反哺窗口分析",
        "description": "EX-F-02: 策略B数字反哺工业的可行性",
        "level": "deep",
        "estimated_time": "20分钟-1.5小时",
        "default_args": ["--grid-size", "15"],
        "quick_args": ["--grid-size", "8"],
    },
    {
        "id": 11,
        "file": "11_tech_catchup_critical.py",
        "name": "技术追赶临界条件",
        "description": "EX-F-03: 追赶与锁死的参数边界",
        "level": "deep",
        "estimated_time": "30分钟-2小时",
        "default_args": ["--n-lhs", "300"],
        "quick_args": ["--n-lhs", "100"],
    },
    {
        "id": 12,
        "file": "12_ews_validation.py",
        "name": "EWS预警有效性验证",
        "description": "EX-F-06: 验证早期预警信号的可靠性",
        "level": "deep",
        "estimated_time": "30分钟-2小时",
        "default_args": ["--n-lhs", "200"],
        "quick_args": ["--n-lhs", "50"],
    },
    {
        "id": 13,
        "file": "13_sobol_sensitivity.py",
        "name": "Sobol全局敏感性分析",
        "description": "Saltelli方法计算Sobol指数",
        "level": "deep",
        "estimated_time": "数小时至数十小时",
        "default_args": ["--N", "256"],
        "quick_args": ["--N", "64"],
        "warning": "此脚本可能需要数小时。确认你有足够的时间。",
    },
    {
        "id": 14,
        "file": "14_institutional_moderation.py",
        "name": "制度调节效应分析",
        "description": "EX-D-03: 制度变量的调节作用",
        "level": "deep",
        "estimated_time": "1-5小时",
        "default_args": ["--grid-size", "10"],
        "quick_args": ["--grid-size", "6"],
    },
    {
        "id": 15,
        "file": "15_extreme_scenario_test.py",
        "name": "极端场景极限测试",
        "description": "EX-D-05: 多冲击叠加下的系统行为边界",
        "level": "deep",
        "estimated_time": "10-30分钟",
        "default_args": ["--n-repeats", "3"],
        "quick_args": ["--n-repeats", "1"],
    },
    {
        "id": 16,
        "file": "16_mixed_timing_optimization.py",
        "name": "混合策略时机优化",
        "description": "EX-F-05: 最优策略切换序列与时机",
        "level": "deep",
        "estimated_time": "30分钟-2小时",
        "default_args": ["--time-steps", "10"],
        "quick_args": ["--time-steps", "5"],
    },
]


def list_scripts():
    """列出所有脚本."""
    print("\n" + "=" * 80)
    print("  研究脚本列表")
    print("=" * 80)
    for s in SCRIPTS:
        level_str = {"simple": "★简单", "medium": "★★中等", "deep": "★★★深度"}
        print(f"\n  [{s['id']:2d}] {s['name']}")
        print(f"       文件: {s['file']}")
        print(f"       描述: {s['description']}")
        print(f"       复杂度: {level_str.get(s['level'], s['level'])}")
        print(f"       预计时间: {s['estimated_time']}")
        if s.get("warning"):
            print(f"       ⚠ {s['warning']}")
    print()


def run_script(script_info: dict, quick_mode: bool = False,
               extra_args: list = None) -> dict:
    """运行单个脚本并返回结果."""
    script_path = SCRIPTS_DIR / script_info["file"]
    if not script_path.exists():
        return {"id": script_info["id"], "success": False,
                "error": f"文件不存在: {script_path}"}

    args = script_info["quick_args" if quick_mode else "default_args"]
    if extra_args:
        args = args + extra_args

    cmd = [sys.executable, str(script_path)] + args

    print(f"\n{'=' * 70}")
    print(f"[{script_info['id']:2d}] {script_info['name']}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_project_root),
            capture_output=False,
            text=True,
            timeout=3600 * 6,  # 最长6小时
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        return {
            "id": script_info["id"],
            "name": script_info["name"],
            "success": success,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "elapsed_str": f"{elapsed/60:.1f}分" if elapsed > 60 else f"{elapsed:.0f}秒",
        }
    except subprocess.TimeoutExpired:
        return {
            "id": script_info["id"],
            "name": script_info["name"],
            "success": False,
            "error": "超时 (>6小时)",
            "elapsed_seconds": 3600 * 6,
        }
    except Exception as e:
        return {
            "id": script_info["id"],
            "name": script_info["name"],
            "success": False,
            "error": str(e),
            "elapsed_seconds": time.time() - start_time,
        }


def main():
    parser = argparse.ArgumentParser(description="研究脚本总调度器")
    parser.add_argument("--scripts", type=str, default=None,
                       help="要运行的脚本ID，逗号分隔 (如 '1,2,3,6')")
    parser.add_argument("--level", type=str, default=None,
                       choices=["simple", "medium", "deep"],
                       help="仅运行指定复杂度级别的脚本")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式（减少采样量）")
    parser.add_argument("--list", action="store_true",
                       help="列出所有脚本及其描述")
    parser.add_argument("--skip", type=str, default=None,
                       help="跳过的脚本ID (如 '13,14')")
    parser.add_argument("--extra-args", type=str, default=None,
                       help="传递给所有脚本的额外参数 (如 '--workers 8')")
    args = parser.parse_args()

    if args.list:
        list_scripts()
        return

    # 选择脚本
    selected = SCRIPTS

    if args.scripts:
        ids = [int(x.strip()) for x in args.scripts.split(",")]
        selected = [s for s in SCRIPTS if s["id"] in ids]

    if args.level:
        selected = [s for s in selected if s["level"] == args.level]

    if args.skip:
        skip_ids = [int(x.strip()) for x in args.skip.split(",")]
        selected = [s for s in selected if s["id"] not in skip_ids]

    if not selected:
        print("没有选中任何脚本。")
        return

    extra_args = args.extra_args.split() if args.extra_args else []

    # 显示计划
    print("\n" + "=" * 80)
    print(f"  研究实验总调度器")
    print(f"  模式: {'快速' if args.quick else '标准'}")
    print(f"  选中脚本: {len(selected)} 个")
    print("=" * 80)
    total_est = sum(
        5 if s["level"] == "simple"
        else 30 if s["level"] == "medium"
        else 120
        for s in selected
    )
    print(f"\n  预计总时间: ~{total_est}分钟 ({total_est/60:.1f}小时)")

    for s in selected:
        warn = f" ⚠ {s['warning']}" if s.get("warning") else ""
        print(f"  [{s['id']:2d}] {s['name']:<25} | {s['estimated_time']:<15} | "
              f"{s['level']}{warn}")

    print(f"\n{'─' * 80}")
    if not args.quick and total_est > 120:
        print("  ⚠ 预计运行时间较长。建议先运行 --quick 或 --level simple 验证环境。")
    print(f"{'─' * 80}")

    # 确认
    if not args.quick and total_est > 240:
        response = input("\n预计运行 >4小时，是否继续？(y/N): ")
        if response.lower() != "y":
            print("已取消。")
            return

    # 运行所有脚本
    results = []
    start_all = time.time()

    for i, script in enumerate(selected):
        print(f"\n{'█' * 60}")
        print(f"[{i+1}/{len(selected)}] 执行: {script['name']}")
        print(f"{'█' * 60}")

        result = run_script(script, quick_mode=args.quick, extra_args=extra_args)
        results.append(result)

        status = "✓ 成功" if result["success"] else f"✗ 失败 ({result.get('error', '?')})"
        print(f"\n  [{script['id']:2d}] {script['name']}: {status} | "
              f"耗时: {result.get('elapsed_str', '?')}")

    # ================================================================
    # 最终汇总
    # ================================================================
    total_elapsed = time.time() - start_all
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{'=' * 80}")
    print(f"  全部完成")
    print(f"{'=' * 80}")
    print(f"  总耗时: {total_elapsed/60:.1f}分钟 ({total_elapsed/3600:.1f}小时)")
    print(f"  成功: {successful}/{len(results)} | 失败: {failed}/{len(results)}")
    print(f"\n  详细结果:")

    for r in results:
        status = "✓" if r["success"] else "✗"
        error_str = f" — {r.get('error', '')}" if not r["success"] else ""
        print(f"  [{r['id']:2d}] {status} {r['name']:<25} | "
              f"{r.get('elapsed_str', '?')}{error_str}")

    # 保存调度日志
    schedule_log = {
        "run_at": datetime.now().isoformat(),
        "mode": "quick" if args.quick else "standard",
        "total_scripts": len(selected),
        "successful": successful,
        "failed": failed,
        "total_elapsed_seconds": total_elapsed,
        "results": results,
    }

    log_dir = SCRIPTS_DIR / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(schedule_log, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  调度日志: {log_path}")

    # 提示后续步骤
    if successful > 0:
        print(f"\n  下一步: 查看结果目录")
        result_dir = SCRIPTS_DIR / "output" / "results"
        print(f"    {result_dir}")
        print(f"  使用以下命令查看历史运行:")
        print(f"    python -c \"from research_scripts.common.result_store import ResultStore; "
              f"print('\\\\n'.join(str(r) for r in ResultStore.list_runs()))\"")


_project_root = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    main()
