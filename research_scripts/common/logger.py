"""统一日志系统 —— 为所有研究脚本提供详细的运行日志.

特性:
    - 同时输出到控制台和文件
    - 时间戳精确到毫秒
    - 每个脚本独立日志文件
    - 自动记录系统信息（CPU、内存等）
    - 进度追踪与ETA估算
"""

import logging
import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path


# ============================================================
# 全局日志目录
# ============================================================
LOG_DIR = Path(__file__).resolve().parent.parent / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ResearchLogger:
    """研究脚本专用日志器 —— 每个实验一个实例."""

    def __init__(self, experiment_name: str, log_to_console: bool = True):
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self._progress_lock = threading.Lock()

        # 日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"{experiment_name}_{timestamp}.log"

        # 配置标准 logging
        self.logger = logging.getLogger(f"research.{experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # 文件 handler —— 详细
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        # 控制台 handler —— 简洁
        if log_to_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                "[%(levelname)-5s] %(message)s"
            ))
            self.logger.addHandler(ch)

        # 写入头部信息
        self._write_header()

    def _write_header(self):
        """写入实验日志头部 —— 包含系统信息和实验标识."""
        import platform
        import multiprocessing

        header = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  实验: {self.experiment_name:<58}║
╠══════════════════════════════════════════════════════════════════════╣
║  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<52}║
║  系统: {platform.system()} {platform.release()} | CPU: {multiprocessing.cpu_count()}核{' ' * (42 - len(str(multiprocessing.cpu_count())))}║
║  Python: {sys.version.split()[0]:<51}║
║  日志文件: {str(self.log_file):<52}║
╚══════════════════════════════════════════════════════════════════════╝
"""
        self.logger.info(header)

    # ================================================================
    # 实验级别日志方法
    # ================================================================

    def experiment_start(self, description: str, **params):
        """记录实验开始."""
        msg = f"实验开始: {description}"
        if params:
            msg += f"\n  参数: {params}"
        self.logger.info(msg)

    def experiment_end(self, success: bool = True, summary: str = ""):
        """记录实验结束."""
        elapsed = time.time() - self.start_time
        status = "✓ 成功" if success else "✗ 失败"
        msg = f"实验结束: {status} | 总耗时: {self._format_time(elapsed)}"
        if summary:
            msg += f"\n  摘要: {summary}"
        self.logger.info(msg)

    def phase_start(self, phase_name: str, phase_num: int, total_phases: int):
        """记录实验阶段开始."""
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"阶段 [{phase_num}/{total_phases}]: {phase_name}")
        self.logger.info(f"{'=' * 60}")

    def phase_end(self, phase_name: str, elapsed: float, results: dict = None):
        """记录实验阶段结束."""
        msg = f"阶段完成: {phase_name} | 耗时: {self._format_time(elapsed)}"
        if results:
            msg += f"\n  结果: {results}"
        self.logger.info(msg)

    # ================================================================
    # 模拟级别日志方法
    # ================================================================

    def sim_start(self, sim_id: int, total: int, params_info: str = ""):
        """记录单次模拟开始."""
        self.logger.debug(f"  [{sim_id}/{total}] 模拟开始 {params_info}")

    def sim_end(self, sim_id: int, result_summary: dict):
        """记录单次模拟结束."""
        survived = "存活" if result_summary.get("survived", False) else "崩溃"
        yrs = result_summary.get("survival_years", 0)
        gdp = result_summary.get("final_gdp", 0)
        gap = result_summary.get("final_tech_gap", 0)
        fractures = result_summary.get("total_fractures", 0)
        cdi = result_summary.get("cdi", float("nan"))

        msg = (f"  [{sim_id}] {survived} | {yrs:.1f}年 | "
               f"GDP={gdp:.0f} | 技术差距={gap:.2f} | "
               f"断裂={fractures} | CDI={cdi:.3f}")
        self.logger.debug(msg)

    def sim_error(self, sim_id: int, error: Exception):
        """记录单次模拟错误."""
        self.logger.error(f"  [{sim_id}] 模拟异常: {type(error).__name__}: {error}")

    # ================================================================
    # 进度追踪
    # ================================================================

    def progress(self, completed: int, total: int, extra: str = ""):
        """记录进度并估算ETA."""
        with self._progress_lock:
            elapsed = time.time() - self.start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total - completed) / rate if rate > 0 else 0
            pct = 100.0 * completed / total
            msg = (f"[进度] {completed}/{total} ({pct:.1f}%) | "
                   f"已用: {self._format_time(elapsed)} | "
                   f"预计剩余: {self._format_time(remaining)}")
            if extra:
                msg += f" | {extra}"
            self.logger.info(msg)

    # ================================================================
    # 数据分析日志
    # ================================================================

    def stat(self, label: str, value, unit: str = ""):
        """记录统计量."""
        if isinstance(value, float):
            self.logger.info(f"  [{label}] {value:.4f} {unit}".strip())
        else:
            self.logger.info(f"  [{label}] {value} {unit}".strip())

    def table(self, title: str, headers: list, rows: list):
        """记录表格."""
        self.logger.info(f"\n  {title}")
        header_line = "  " + " | ".join(f"{h:<15}" for h in headers)
        sep_line = "  " + "-" * len(header_line)
        self.logger.info(header_line)
        self.logger.info(sep_line)
        for row in rows:
            line = "  " + " | ".join(f"{str(v):<15}" for v in row)
            self.logger.info(line)

    def finding(self, text: str, level: str = "INFO"):
        """记录研究发现 —— 用于最终报告."""
        if level == "KEY":
            self.logger.warning(f"[核心发现] {text}")
        elif level == "WARN":
            self.logger.warning(f"[警示] {text}")
        else:
            self.logger.info(f"[发现] {text}")

    # ================================================================
    # 工具方法
    # ================================================================

    def get_elapsed(self) -> float:
        return time.time() - self.start_time

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间."""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}时{m}分{s}秒"
