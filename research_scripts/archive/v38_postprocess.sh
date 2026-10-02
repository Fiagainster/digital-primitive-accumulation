#!/usr/bin/env bash
# 等待主实验全部批次完成 -> 时间序列 -> 重算全表 -> 出图
set -u
cd "$(dirname "$0")/.." || exit 1

PY=D:/System/Python/python.exe
RUN=experiments_runs/v38
FIGOUT="D:/Project/latex论文协作/output"

echo "[wait] 等待主实验批次完成 ..."
while [ ! -f "$RUN/calibration_t80.json" ]; do
    sleep 20
done
# 再确认没有 python 子进程在写
sleep 5
echo "[wait] 主实验完成，开始后处理 $(date '+%H:%M:%S')"

echo "[1/3] 时间序列批次 ..."
$PY research_scripts/run_timeseries_batch.py 2>&1 | tail -3

echo "[2/3] 重算全表 ..."
$PY research_scripts/analyze_v38.py 2>&1 | tail -5

echo "[3/3] 生成图件 -> $FIGOUT ..."
$PY research_scripts/make_figures_v38.py --out "$FIGOUT" 2>&1 | tail -12

echo "[done] 后处理完成 $(date '+%H:%M:%S')"
