# Digital Primitive Accumulation Model (v3.8)

六部门政治经济学模拟实验室：数字时代原始积累的可行路径、多维评价与代价分布。

本仓库是论文 *Digital Primitive Accumulation and Development Trade-offs: A Computational Comparison of Three Strategies*（Computational Economics 投稿中）的模型代码。

## 分支说明

| 分支 | 内容 |
|---|---|
| `main`（当前） | 只含源码、配置、测试与文档；**不含任何模拟输出**。用于第三方 clone 后自行复刻。 |
| `with-results` | 完整 v3.8 归档：23 个实验批次、5,901 次执行记录、2026-10 投稿包、复现与验收报告。 |

获取带结果的版本：

```bash
git clone -b with-results https://github.com/Fiagainster/digital-primitive-accumulation.git
```

## 模型概要

- **六个部门**：基础/资源（b）、工业（i）、数字基础设施（d_infra）、数字应用（d_app）、高技能服务（s_high）、低技能服务（s_low）。
- **五类家庭群组**：数字、工业、基础、服务业、失业群体；聚合建模，不模拟个体。
- **三种积累规则**：渐进（A）、跳跃（B）、替代（C），以及受限混合规则。
- **参数**：`config/default.yaml` 注册 255 个参数，附定义域与来源标注；国情预设 6 个，国际场景 8 个。
- **评价**：五维评价与 CDI、福利指数、五群组累计调整成本、九类断裂模式监测。

详细设计与公式见 `doc/` 与 `模型公式参考.md`。

## 环境

Python 3.10+。

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 单国情演示（三种纯策略 + 五维评价）
python main.py demo --profile manufacturing_giant

# 多国情对比
python main.py compare --profiles manufacturing_giant,digital_leapfrogger,resource_rich
```

可用国情键：`manufacturing_giant`、`resource_rich`、`digital_leapfrogger`、`small_open`、`post_soviet`、`populous_weak_industry`。

## 复现论文实验

```bash
# 列出 23 个实验批次
PYTHONIOENCODING=utf-8 python research_scripts/run_paper_experiments.py --list

# 先跑普查与混合批（t80 依据普查结果挑选配置，需在 census 之后运行）
PYTHONIOENCODING=utf-8 python research_scripts/run_paper_experiments.py --only census mixture --workers 8
PYTHONIOENCODING=utf-8 python research_scripts/run_paper_experiments.py --only t80 --workers 8

# 重算全部汇总表 → experiments_runs/v38/tables.json
PYTHONIOENCODING=utf-8 python research_scripts/analyze_v38.py

# 重出出版级图件
PYTHONIOENCODING=utf-8 python research_scripts/make_figures_v38.py --out figures
```

输出写入 `experiments_runs/`，已被 `.gitignore` 排除。核验已发表数值时，建议直接使用 `with-results` 分支中的冻结归档，而不是重跑全部实验。

## 随机性与可复现性

每次运行在 `SimulationEngine.run()` 入口按 `config.random_seed` 播种 NumPy；批量运行器用 `SeedSequence` 为每个任务派生子种子。同配置同种子结果逐值一致，不同种子给出不同轨迹。

## 文档

| 文件 | 内容 |
|---|---|
| `doc/00_设计总纲.md` … `doc/12_Computational_Economics投稿复现_20261011.md` | 设计、数学框架、参数体系、实验设计、投稿复现 |
| `模型公式参考.md` | 公式级参考 |
| `正式实验方案.md`、`定性校准实验方案.md` | 实验与校准方案 |

## 引用

若使用本代码，请引用上述论文（审稿中）。正式发表信息确定后在此补充。

## 许可证

木兰宽松许可证，第 2 版（Mulan PSL v2），见 `LICENSE`。
