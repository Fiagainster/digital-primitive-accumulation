#!/usr/bin/env python
"""诊断脚本：追踪 B×digital_leapfrogger 的 GDP 爆炸轨迹."""

import sys
from pathlib import Path
import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from research_scripts.common.runner import build_param_sets


def diagnose():
    params = build_param_sets(
        regime="B", profile="digital_leapfrogger",
        scenario="benign_globalization",
    )
    config = SimulationConfig(T_max=75.0, verbose=False)

    engine = SimulationEngine(params=params, config=config, regime="B")
    result = engine.run()

    # 获取时间序列
    df = result.get("time_series")
    if df is None:
        print("No time series in result")
        # Try to get it from recorder
        df = engine.recorder.to_dataframe()

    if df is not None:
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nKey columns every 20 steps:")
        key_cols = ['t', 'gdp', 'unemployment_rate', 'tech_gap']
        available = [c for c in key_cols if c in df.columns]
        # Also find sector output columns
        y_cols = [c for c in df.columns if 'Y_' in c or '_Y' in c or 'output' in c.lower()]
        k_cols = [c for c in df.columns if 'K_' in c or '_K' in c]
        r_cols = [c for c in df.columns if 'profit' in c.lower() or '_r' in c]

        print(f"  Y columns: {y_cols}")
        print(f"  K columns: {k_cols}")
        print(f"  Profit columns: {r_cols}")

        # Show trajectory every N rows
        n_rows = len(df)
        step = max(1, n_rows // 30)
        cols_to_show = available + y_cols[:4] + k_cols[:4]
        print(f"\n  {'idx':>5s} " + " ".join(f"{c:>15s}" for c in cols_to_show[:8]))
        for i in range(0, n_rows, step):
            row = df.iloc[i]
            vals = " ".join(f"{row.get(c, 0):15.4g}" if c in df.columns else f"{'N/A':>15s}" for c in cols_to_show[:8])
            print(f"  {i:5d} {vals}")

        # Find where GDP explodes
        gdp_col = 'gdp' if 'gdp' in df.columns else None
        if gdp_col is None:
            # Try to find a GDP-like column
            for c in df.columns:
                if 'gdp' in c.lower():
                    gdp_col = c
                    break

        if gdp_col:
            gdp_vals = df[gdp_col].values
            # 检测GDP增长异常加速
            growth_rates = []
            for i in range(2, len(gdp_vals)):
                if gdp_vals[i-2] > 0 and gdp_vals[i-1] > 0:
                    g1 = (gdp_vals[i-1] - gdp_vals[i-2]) / gdp_vals[i-2]
                    g2 = (gdp_vals[i] - gdp_vals[i-1]) / gdp_vals[i-1]
                    if g1 > 0.01:
                        accel = g2 / g1
                        growth_rates.append((i, df.iloc[i].get('t', i), g2, accel, gdp_vals[i]))
                        if accel > 3:
                            print(f"\n  *** GDP growth acceleration at step {i} (t≈{df.iloc[i].get('t', i):.1f}): "
                                  f"GDP={gdp_vals[i]:.4g}, growth_rate={g2:.4f}, accel={accel:.2f}x ***")

            # Print GDP doubling times
            print(f"\n  GDP doubling analysis:")
            milestones = [1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e15]
            for m in milestones:
                if gdp_vals[-1] > m:
                    idx = np.argmax(gdp_vals >= m)
                    t_m = df.iloc[idx].get('t', idx)
                    print(f"    GDP reaches {m:.0e} at t≈{t_m:.1f}")

            # Also show sector breakdown at key points
            print(f"\n  Sector GDP breakdown at key times:")
            y_cols_all = ['Y_b', 'Y_i', 'Y_d_infra', 'Y_d_app', 'Y_s_high', 'Y_s_low']
            for target_t in [0, 20, 30, 40, 45]:
                idx = np.argmin(np.abs(df['t'].values - target_t))
                t_actual = df.iloc[idx]['t']
                gdp = df.iloc[idx]['gdp']
                sector_ys = {c: df.iloc[idx][c] for c in y_cols_all if c in df.columns}
                shares = {c: v/gdp for c, v in sector_ys.items()}
                print(f"    t≈{t_actual:.1f} GDP={gdp:.4g}: " + " | ".join(
                    f"{c.split('_')[1]}:{shares.get(c,0)*100:.0f}%" for c in y_cols_all if c in df.columns
                ))

        # Summary
        summ = result.get("summary", {})
        print(f"\nFinal summary:")
        for k, v in summ.items():
            if isinstance(v, (int, float, bool, str)):
                print(f"  {k}: {v}")
    else:
        print("ERROR: No time series data available")
        print(f"Result keys: {list(result.keys())}")


if __name__ == "__main__":
    diagnose()
