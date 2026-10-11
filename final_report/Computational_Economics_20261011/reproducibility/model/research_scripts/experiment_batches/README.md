# 分阶段实验框架

**论文**: 计算探索视角下的数字原始积累——基于六部门模型的策略空间分析

## 设计理念

五阶段实验设计，**每个阶段的结果为下一阶段提供方向**。不要一次性全部运行——先跑 Phase 1，根据结果决定 Phase 2 的焦点，依此类推。

```
Phase 1      Phase 2           Phase 3         Phase 4         Phase 5
策略空间普查 → 关键参数窗口     → 生存边界测绘   → 机制深挖       → 前沿边界
(普查)        (定位敏感区)       (量化边界)      (因果机制)      (精确标定)

10-20分钟     30-60分钟          30-60分钟       40-90分钟       1-4小时
```

## 快速开始

```bash
cd D:/Project/原始积累模型探索/primitive_accumulation

# ============================
# 第一步：策略空间普查
# ============================
python research_scripts/experiment_batches/run_phase1_survey.py

# 快速版（先验证流程）
python research_scripts/experiment_batches/run_phase1_survey.py --quick

# ============================
# 第二步：看完 Phase1 结果后，确定关键参数再跑
# ============================
# 全线扫描（默认）
python research_scripts/experiment_batches/run_phase2_parameters.py

# 仅聚焦数字分层
python research_scripts/experiment_batches/run_phase2_parameters.py --focus digital

# 仅聚焦制度能力
python research_scripts/experiment_batches/run_phase2_parameters.py --focus institution

# ============================
# 第三步：生存边界测绘
# ============================
python research_scripts/experiment_batches/run_phase3_boundaries.py

# 快速版
python research_scripts/experiment_batches/run_phase3_boundaries.py --quick

# 仅测特定策略和国情
python research_scripts/experiment_batches/run_phase3_boundaries.py --regimes A,B --profiles digital_leapfrogger

# ============================
# 第四步：机制深挖
# ============================
python research_scripts/experiment_batches/run_phase4_mechanisms.py

# 跳过代价分布（如果不需要）
python research_scripts/experiment_batches/run_phase4_mechanisms.py --skip cost

# ============================
# 第五步：前沿边界（最耗时，基于前四阶段发现选择焦点）
# ============================
# 仅断裂边界
python research_scripts/experiment_batches/run_phase5_frontiers.py --focus fracture --grid-size 20

# 仅技术追赶
python research_scripts/experiment_batches/run_phase5_frontiers.py --focus catchup

# 全线（数小时）
python research_scripts/experiment_batches/run_phase5_frontiers.py --focus all --grid-size 30
```

## 各阶段说明

### Phase 1: 策略空间普查

**研究问题**: 在国情×策略×场景的三维空间中，哪些组合存活/断裂？不同策略产生怎样的多维后果？

**默认规模**: 3国情 × 3策略 × 3场景 = 27组合，T=80

**关注输出**:
- `interaction_profile_regime` — 国情×策略存活率矩阵（核心）
- `by_regime` — 各策略的CDI、技术差距、GDP分布
- `fracture_types` — 不同组合下最常见的断裂类型

### Phase 2: 关键参数窗口

**研究问题**: 在第一阶段识别的关键区域中，哪些参数的变动能引起质变（存活↔断裂）？

**扫描焦点**:
- `digital` — infra→app技术约束 × 封锁强度
- `institution` — 制度能力的跨策略调节效应
- `world_growth` — 全球增长率的差异化影响
- `strategy` — 策略内生参数（提取率、反哺阈值、进口偏好等）

**关注输出**:
- `thresholds_detected` — 存活率跳变>30%的参数区间
- `survival_by_value1` — 存活率随参数变化的曲线

### Phase 3: 生存边界测绘

**研究问题**: 大范围参数空间中系统的存活概率曲面是什么形状？

**方法**: LHS采样 (200个参数组合) × 重复运行 (2次) = 稳健存活概率

**关注输出**:
- `survival_matrix` — 各策略×国情×场景的存活率
- `top_params` — 全局参数重要性排名
- `survival_by_sample` — 存活概率分布

### Phase 4: 机制深挖

**研究问题**: 在前三阶段发现的模式背后，因果机制是什么？

**四个子分析**:
1. **代价分布** — 哪个群体承担转型代价？
2. **金融稳定性** — 资产泡沫/资本外逃/信贷错配的联动
3. **数字依附** — 制度能力和封锁强度如何影响依附
4. **混合策略** — 混合配比能否超越纯策略？

### Phase 5: 前沿边界

**研究问题** (整合性): 精确标定关键边界——系统在什么条件下崩溃/逆转/成功？

**四个子分析**:
1. **断裂边界** — 2D高精度网格（基于前序阶段识别的关键参数对）
2. **技术追赶** — LHS大样本扫描区分追赶/锁死的参数条件
3. **反哺窗口** — 策略B的数字反哺在什么参数组合下触发
4. **EWS验证** — 预警信号在断裂前的表现

## 输出结构

```
experiment_batches/
├── shared_config.py                 # 共享配置和工具函数
├── run_phase1_survey.py
├── run_phase2_parameters.py
├── run_phase3_boundaries.py
├── run_phase4_mechanisms.py
├── run_phase5_frontiers.py
├── README.md
└── output/
    ├── phase_record.json            # 阶段执行记录（自动更新）
    ├── logs/
    └── results/
        ├── phase1_YYYYMMDD_HHMMSS_results.json
        ├── phase2_YYYYMMDD_HHMMSS_results.json
        └── ...
```

## 研究方向定制

根据你的论文研究方向，建议重点关注：

1. **"反事实轨迹"** → Phase 1 + Phase 4 混合策略：找出与"自然预期"不同的策略×国情组合
2. **"多维后果"** → Phase 1 + Phase 3：跨维度（CDI五维、技术差距、断裂模式）的系统比较
3. **"策略空间结构"** → Phase 2 + Phase 5：参数空间中的吸引子、边界和相变
4. **"数字依附"** → Phase 2 + Phase 4：依附自我强化路径和逆转条件

## 现有脚本对照

| 阶段 | 调用或等价于原有脚本 |
|------|-------------------|
| Phase 1 | 01 × 27组合 (single_detailed_run 批量) |
| Phase 2 | 17 + 18 + 02 (聚焦关键参数) |
| Phase 3 | 06 + 07 (LHS生存扫描 + 初始条件扫描) |
| Phase 4 | 08 + 19 + 04 + 18 (代价+金融+混合+依附) |
| Phase 5 | 09 + 11 + 10 + 12 (断裂+追赶+反哺+EWS) |

## 提示

- 每个阶段结束后，查看 `output/results/` 中的 JSON 结果
- `phase_record.json` 记录各阶段的完成状态，后续阶段会自动读取前序结果
- 如果某个阶段因故中断，重新运行时会重新开始（不会断点续传）
- 建议每次运行后用 `--quick` 先验证流程，确认无误后再用标准模式
