# archive —— 已完成的一次性编排脚本

存放 v3.8 实验期间用过、但**不再需要重跑**的一次性编排脚本。保留它们是为了让
`experiments_runs/v38/` 的产出过程可追溯：哪天、以什么顺序、跑了哪些批次。

| 脚本 | 用途 | 状态 |
|---|---|---|
| `v38_postprocess.sh` | 轮询等待主实验批次结束 → 跑时间序列批次 → `analyze_v38.py` 重算全表 → `make_figures_v38.py` 出图到论文仓 `output/` | 已完成，勿重跑 |
| `v38_extra_batches.sh` | 轮询等待主实验与时间序列就绪 → 补跑 `extraction` / `gspeed` 两个后加批次 | 已完成，勿重跑 |

> **两个脚本都靠"某个文件是否存在"来轮询等待，且硬编码了 `RUN=experiments_runs/v38`。**
> 日常复现请改用 `research_scripts/run_paper_experiments.py`——它支持 `--list` 列批次、
> `--only` 选批次、`--skip-existing` 断点续跑，不依赖轮询。
