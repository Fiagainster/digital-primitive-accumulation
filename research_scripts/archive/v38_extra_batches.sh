#!/usr/bin/env bash
# 等主实验 + 时间序列批次都完成后，补跑两个后加批次：extraction / gspeed
set -u
cd "$(dirname "$0")/.." || exit 1
PY=D:/System/Python/python.exe
RUN=experiments_runs/v38

echo "[wait] 等待主实验与时间序列批次 ..."
while [ ! -f "$RUN/calibration_t80.json" ] || [ ! -f "$RUN/ts_calibration.json" ]; do
    sleep 20
done
echo "[wait] 就绪，补跑后加批次 $(date '+%H:%M:%S')"

$PY research_scripts/run_paper_experiments.py --only extraction gspeed --skip-existing \
    --workers 12 2>&1 | tail -12

echo "[done] 补跑完成 $(date '+%H:%M:%S')"
