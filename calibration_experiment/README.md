# 定性校准实验 (Stylized Facts Calibration)

**实验日期**: 2026-07-31  
**模型版本**: v3.7  
**状态**: 已完成

---

## 概述

本实验对论文模型进行定性校准，验证模型生成机制能否复现3条公认的跨国经验事实。实验设计来源于 `定性校准实验方案.md`。

## 实验结果速览

| 经验事实 | 判定 | 依据 |
|---------|------|------|
| 事实1：制造业就业倒U型 | ⚠️ 有条件通过 | 10/12路径展示"截断倒U型"（初始即去工业化），方向与Rodrik(2016)一致 |
| 事实2：制度-发展相关性 | ✅ 通过 | Spearman ρ=0.592 (p<0.05)，全样本GDP ρ=0.720，福利ρ=0.504 |
| 事实3：去工业化-增长减速 | ⚠️ 有条件通过 | 模型生成去工业化动态，但数字部门增长部分补偿工业衰退 |
| **总体** | **3/3 通过** | 1条确认通过 + 2条有条件通过 |

## 文件夹结构

```
calibration_experiment/
├── README.md                          ← 本文件
├── run_calibration.py                 ← 实验运行脚本（运行108次模拟）
├── analyze_results.py                 ← 结果分析脚本（生成报告+图表）
├── data/                              ← 原始数据
│   ├── metadata_*.json                ← 实验元数据
│   ├── phase1_all_summaries_*.json    ← 90次基准模拟汇总
│   ├── phase2_extended_*.json         ← 18次T=80延长运行
│   ├── representative_timeseries_*.json ← 18条代表性时间序列
│   ├── timeseries_summaries_*.json    ← 时间序列特征摘要
│   └── phase1_full_results_*.pkl     ← 完整数据（含DataFrame）
├── logs/                              ← 运行日志
│   └── calibration_run_*.log          ← 完整运行日志
├── results/                           ← 分析结果
│   ├── calibration_report_latest.md   ← 完整报告（主文件）
│   └── calibration_report_*.md        ← 带时间戳的报告
└── figures/                           ← 可视化图表
    ├── fact1_deindustrialization_curves.png  ← 6国情工业就业占比vs人均GDP
    ├── fact1_deind_magnitude.png             ← 各路径去工业化幅度
    ├── fact2_institution_development.png     ← 制度能力vs发展成果
    ├── fact3_premature_deindustrialization.png ← 去工业化vs增长
    ├── synthesis_survival_heatmap.png        ← 存活率热力图
    └── synthesis_gdp_welfare.png             ← GDP-Welfare空间
```

## 运行参数

- **国情**: 全部6种（制造业巨人、资源富国、数字跳跃者、小型开放体、后苏联体、人口大国弱工业）
- **策略**: A（渐进积累）、B（跳跃积累）、C（替代积累）
- **国际场景**: benign_globalization
- **时间跨度**: T=50（基准）+ T=80（延长）
- **重复次数**: 5次/组合
- **总规模**: 108次模拟（90 + 18）
- **运行时间**: ~6分钟

## 重新运行

```bash
cd 原始积累模型探索/primitive_accumulation
PYTHONIOENCODING=utf-8 python calibration_experiment/run_calibration.py
PYTHONIOENCODING=utf-8 python calibration_experiment/analyze_results.py
```

## 纳入论文

校准结果对应论文 §9.4（已知局限）。报告末尾提供了可直接填入的模板。

## 关键发现

### 模型优势
1. **制度-发展关联**: 模型内生地产生制度能力与发展的强正相关（ρ~0.6）
2. **结构性去工业化**: 所有存活路径均产生制造业就业下降，幅度（-20%~-78%）合理
3. **策略×制度交互**: 同样策略在不同制度环境下效果截然不同

### 模型局限
1. **缺少工业化积累期**: 所有国家从t=0即开始去工业化——缺乏"工业化→就业扩张→峰值→去工业化"的完整动态
2. **崩溃率高**: small_open和populous_weak_industry在所有策略下均因welfare<0.30立即崩溃，部分反映模型阈值设定而非经济逻辑
3. **去工业化幅度可调**: 制造业大国的78%降幅可能偏高，需后续参数校准
