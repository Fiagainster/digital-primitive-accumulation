# 研究脚本集 —— 原始积累模型 (Primitive Accumulation Model)

**版本**: v1.2 (2026-07-05)
**匹配模型版本**: v3.6 (六部门 + 平台垄断 + 内生人口 + 双区域空间 + 政策执行折扣)

## v3.6 主要变更

- **平台垄断模块**: 数字部门HHI市场集中度内生演化、垄断租金、数据网络护城河、反垄断政策参数
- **内生人口动态**: 队列分量人口模型，生育率/死亡率/城镇化内生调节，人口红利检测
- **双区域空间模块**: 沿海/内陆分化追踪，劳动力空间流动，集聚效应与拥挤成本，区域不平等
- **政策执行折扣**: 官僚体系内部摩擦建模，策略复杂度惩罚，制度能力约束
- **参数校准框架**: 中国1978-2023试点校准（方法论演示：World Bank WDI 6变量 + 王鹏等2023 TFP增长率，差分进化轨迹匹配 + SFC残差惩罚，蒙特卡洛不确定性传播）。定位为"证明模型可被校准"，不追求精确参数确定

## v3.5 主要变更

- **六部门结构**: 数字部门拆分为 d_infra(基础设施层) + d_app(应用层)，服务业拆分为 s_high(高技能) + s_low(低端)
- **分层次技术差距**: 新增 `tech_gap_infra` 和 `tech_gap_app` 指标
- **就业质量指数**: `employment_quality_index` 区分正式就业/零工/非正式
- **六维数字依附**: 芯片/云/OS/数据/标准 + 资本/人才/生态依附
- **策略不兼容矩阵**: AB/BC/AC 策略对冲突惩罚分析
- **混合策略预设名称**: 支持 `--mixture A_dominant_B` 等便捷参数

## 概述

本目录包含 **20 个独立的研究脚本**，覆盖从快速诊断到深度分析的全谱系实验。每个脚本都代表一个独立的研究方向，配备了：

- **详细的命令行参数**：可通过 `--help` 查看
- **完整的日志系统**：每次运行自动记录到 `output/logs/`
- **结构化的结果存储**：CSV + JSON + Pickle + HDF5，保存到 `output/results/`
- **多进程并行**：默认利用所有可用 CPU 核心
- **进度追踪**：实时显示进度和 ETA 估算

## 快速开始

```bash
# 确保在正确的目录
cd D:/Project/原始积累模型探索/primitive_accumulation

# 激活环境
conda activate pa_model

# 运行最简单的脚本（验证环境）
python research_scripts/01_single_detailed_run.py

# v3.5: 使用混合策略预设名称运行
python research_scripts/01_single_detailed_run.py --regime M --mixture A_dominant_B

# 运行所有简单脚本（快速验证）
python research_scripts/run_all.py --level simple --quick

# 运行所有脚本（完整实验，可能需要数小时）
python research_scripts/run_all.py --level deep

# v3.5 新增脚本
python research_scripts/17_digital_layer_analysis.py --sweep xi_infra_to_app
python research_scripts/18_dependency_dynamics.py --sweep institution_capacity
python research_scripts/19_financial_stability_scan.py --sweep capital_flight_sensitivity

# v3.6: 参数校准
python research_scripts/20_china_calibration.py --method de --maxiter 300
python research_scripts/20_china_calibration.py --uncertainty --n-samples 200
```

## 脚本分类

### 简单脚本 (Level 1) — 快速诊断，秒至分钟级

| ID | 脚本 | 研究问题 | 预计时间 |
|----|------|---------|---------|
| 01 | `01_single_detailed_run.py` | 给定国情×策略×场景，产生什么轨迹？(v3.5: 六部门指标) | 5-30秒 |
| 02 | `02_parameter_sensitivity.py` | 哪些参数对输出影响最大？(v3.5: 含新参数) | 3-10分钟 |
| 03 | `03_regime_comparison.py` | 三种纯策略在各维度上孰优孰劣？(v3.5: infra/app分层) | 2-10分钟 |
| 04 | `04_mixed_strategy_scan.py` | 混合策略能否克服纯策略缺陷？(v3.5: 不兼容惩罚) | 3-15分钟 |
| 05 | `05_scenario_stress_test.py` | 不同策略对国际环境的敏感度？(v3.5: 封锁分层影响) | 5-20分钟 |

### 中等脚本 (Level 2) — 聚焦探索，分钟至小时级

| ID | 脚本 | 研究问题 | 预计时间 |
|----|------|---------|---------|
| 06 | `06_baseline_survival_scan.py` | 大范围参数空间中策略生存边界？(v3.5: 六部门指标) | 20-60分钟 |
| 07 | `07_initial_condition_scan.py` | 不同国情最适合什么策略？(v3.5: 六部门初始条件) | 10-40分钟 |
| 08 | `08_cost_distribution_analysis.py` | 代价在不同群体间如何分布？(v3.5: 就业质量) | 10-30分钟 |

### 深度脚本 (Level 3) — 深度分析，小时至天级

| ID | 脚本 | 研究问题 | 预计时间 |
|----|------|---------|---------|
| 09 | `09_fracture_boundary_mapping.py` | 断裂的临界参数值是多少？(v3.5) | 0.5-2小时 |
| 10 | `10_feedback_window_analysis.py` | 策略B的反哺在什么条件下兑现？(v3.5) | 0.3-1.5小时 |
| 11 | `11_tech_catchup_critical.py` | 追赶与锁死的参数边界在哪？(v3.5: infra/app分层) | 0.5-2小时 |
| 12 | `12_ews_validation.py` | EWS预警信号可靠吗？(v3.5) | 0.5-2小时 |
| 13 | `13_sobol_sensitivity.py` | 参数的全局敏感性排名？(v3.5: 含新参数) | 数小时-数十小时 |
| 14 | `14_institutional_moderation.py` | 好制度能否弥补差策略？(v3.5) | 1-5小时 |
| 15 | `15_extreme_scenario_test.py` | 系统的绝对崩溃条件？(v3.5) | 10-30分钟 |
| 16 | `16_mixed_timing_optimization.py` | 最优策略切换时机是什么？(v3.5) | 0.5-2小时 |

### v3.5 专用脚本 (Level 1-2) — 六部门新机制探索

| ID | 脚本 | 研究问题 | 预计时间 |
|----|------|---------|---------|
| 17 | `17_digital_layer_analysis.py` | infra/app追赶路径差异？技术约束瓶颈？ | 10-60秒 |
| 18 | `18_dependency_dynamics.py` | 依附自我强化→逆转路径？隐性天花板？ | 10-60秒 |
| 19 | `19_financial_stability_scan.py` | 资产泡沫/资本外逃/信贷错配/就业质量联动？ | 1-10分钟 |

### v3.6 专用脚本 (Level 2-3) — 新机制验证与校准

| ID | 脚本 | 研究问题 | 预计时间 |
|----|------|---------|---------|
| 20 | `20_china_calibration.py` | 中国1978-2023历史数据校准（方法论演示：证明模型可被校准，非精确参数确定） | 10-60分钟 |

**校准数据源** (`calibration/china_historical_data.csv`, 方法论演示用途):
- 人均GDP: World Bank WDI (1960-2023, 64点)
- 工业/服务业就业: World Bank WDI (1991-2023, 各33点)
- 基尼系数: World Bank WDI (1981-2023, 23点)
- 城镇化率: 国家统计局 (1960-2017, 58点)
- TFP增长率: 王鹏,郭凯明,颜色(2023)「中国宏观层面全要素生产率测算及国际比较」《经济管理学刊》第2卷 (1997-2021, 25点)

## 运行方式

### 总调度器

```bash
# 列出所有脚本
python research_scripts/run_all.py --list

# 选择性运行
python research_scripts/run_all.py --scripts 1,2,3,6

# 按复杂度运行
python research_scripts/run_all.py --level simple    # 1-5 + 17-18
python research_scripts/run_all.py --level medium    # 1-10 + 19
python research_scripts/run_all.py --level deep      # 全部1-19

# 快速模式（减少采样量，加速运行）
python research_scripts/run_all.py --quick --level simple

# 跳过某些特别耗时的脚本
python research_scripts/run_all.py --skip 13,14

# 传递额外参数给所有脚本
python research_scripts/run_all.py --level medium --extra-args "--workers 16"
```

### 单独运行

每个脚本支持独立的命令行参数：

```bash
# 查看脚本的帮助
python research_scripts/01_single_detailed_run.py --help

# 运行特定配置
python research_scripts/01_single_detailed_run.py \
    --regime B \
    --profile digital_leapfrogger \
    --scenario tech_cold_war \
    --T-max 80

# 快速版（减少计算量）
python research_scripts/06_baseline_survival_scan.py \
    --n-samples 50 \
    --T-max 60 \
    --workers 8
```

## 目录结构

```
research_scripts/
├── common/                          # 公共基础设施
│   ├── __init__.py
│   ├── logger.py                    #   统一日志系统
│   ├── result_store.py              #   结果存储与检索
│   └── runner.py                    #   并行运行器 + 采样工具
├── output/                          # 输出目录（自动创建）
│   ├── logs/                        #   日志文件
│   ├── results/                     #   结果文件（CSV/JSON/HDF5/PKL）
│   └── figures/                     #   图表文件
├── 01_single_detailed_run.py        # 简单脚本 1-5
├── 02_parameter_sensitivity.py
├── 03_regime_comparison.py
├── 04_mixed_strategy_scan.py
├── 05_scenario_stress_test.py
├── 06_baseline_survival_scan.py     # 中等脚本 6-8
├── 07_initial_condition_scan.py
├── 08_cost_distribution_analysis.py
├── 09_fracture_boundary_mapping.py  # 深度脚本 9-16
├── 10_feedback_window_analysis.py
├── 11_tech_catchup_critical.py
├── 12_ews_validation.py
├── 13_sobol_sensitivity.py
├── 14_institutional_moderation.py
├── 15_extreme_scenario_test.py
├── 16_mixed_timing_optimization.py
├── 17_digital_layer_analysis.py     # v3.5 专用脚本 17-19
├── 18_dependency_dynamics.py
├── 19_financial_stability_scan.py
├── 20_china_calibration.py           # 🆕 v3.6: 中国试点校准 (15参数 DE优化)
├── calibration/                     # 🆕 v3.6: 校准工具包
│   ├── __init__.py
│   ├── calibration_report.py        #   通用校准报告 + 不确定性传播
│   └── china_historical_data.csv    #   中国1960-2023历史数据(6变量)
├── run_all.py                       # 总调度器
└── README.md                        # 本文件
```

## 公共基础设施

### 日志系统 (`common/logger.py`)

每个实验实例化一个 `ResearchLogger`，提供：
- 自动创建带时间戳的日志文件
- 同时输出到控制台和文件（控制台简洁、文件详细）
- 实验级别的结构化日志（phase_start/end, sim_start/end）
- 进度追踪与 ETA 估算
- 统计量、表格、发现（finding）等专用输出格式

### 结果存储 (`common/result_store.py`)

每个实验实例化一个 `ResultStore`，提供：
- 自动创建实验专属输出子目录
- 多格式保存：CSV（摘要表）、JSON（元数据+摘要）、Pickle（完整Python对象）、HDF5（时间序列）
- 元数据自动记录（参数、时间戳、配置）
- 统计查询接口（存活率、汇总统计）
- 历史运行加载接口

### 运行器 (`common/runner.py`)

提供统一的批量模拟工具：
- `BatchRunner` — 多进程并行调度，回调通知
- `sample_lhs_params()` — 拉丁超立方采样
- `build_param_sets()` — 参数集构建（支持国情预设和场景）
- `param_grid_explore()` / `param_2d_grid()` — 参数网格扫描
- `quick_run()` — 快速单次模拟 (v3.5: 支持 `mixture_name` 参数)
- `get_optimal_workers()` — 自动最优并行数
- `ALL_PROFILES` — v3.5: 所有国情预设名称列表

## 输出说明

每次运行在 `output/results/` 下创建以 `{实验名}_{时间戳}` 命名的子目录：

```
output/results/06_baseline_survival_scan_20260704_143022/
├── metadata.json          # 实验元数据（参数、配置、时间戳）
├── summary.csv            # 所有模拟的摘要表（每行一次）
├── summary.json           # 摘要 + 统计
├── full_results.pkl       # 完整 Python 对象（供后续分析）
└── time_series.h5         # 时间序列（如果启用）
```

## 实验设计对应

脚本与设计文档 `doc/04_模拟实验设计.md` 中实验编号的对应关系：

| 脚本 | 实验编号 | 设计文档节 |
|------|---------|-----------|
| 06 | EX-S-01 | 基准生存扫描 |
| 07 | EX-S-02 | 初始条件扫描 |
| 05 | EX-S-03 | 场景冲击扫描 |
| 09 | EX-F-01 | 断裂临界边界 |
| 10 | EX-F-02 | 反哺窗口分析 |
| 11 | EX-F-03 | 技术追赶临界 |
| 08 | EX-F-04 | 代价分布 |
| 16 | EX-F-05 | 混合策略时机 |
| 12 | EX-F-06 | EWS预警验证 |
| 13 | EX-F-07 | 数字依附路径（整合至13）|
| 14 | EX-D-03 | 制度调节效应 |
| 15 | EX-D-05 | 极端场景测试 |
