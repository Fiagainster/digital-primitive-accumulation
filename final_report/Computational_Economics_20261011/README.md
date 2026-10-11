# Computational Economics：2026-10-11 投稿复现材料

论文题目：**Digital Primitive Accumulation and Development Trade-offs: A Computational Comparison of Three Strategies**。

本目录保存最新“作者基金留空版”投稿包的代码部分。论文、附录、投稿信、标题页和排版源文件见[论文仓库](https://gitee.com/fiagainster/strategy-space-digital-primitive-accumulation)。本次整理没有运行新的模型实验。

## 文件分类

| 内容 | 位置 |
|---|---|
| 期刊 Online Resource 2，保持原始 ZIP 字节 | `ESM_2.zip` |
| 可独立运行的完整数据与代码包 | `reproducibility/` |
| 原始模型、配置、测试及实验脚本快照 | `reproducibility/model/` |
| 23 个批次、5,901 次执行的 26 个接收文件 | `reproducibility/model/experiments_runs/v38/` |
| 第一、三、四阶段接受的清单与结果 | `reproducibility/accepted/` |
| 复算工具及原分析程序 | `reproducibility/analysis/` |
| 15 幅图的再生成程序 | `reproducibility/artwork/` |
| 环境、重放和复算证据 | `reproducibility/audit/` |
| 两仓库文件对应关系及原始 SHA-256 | `repository_split.json` |

根目录的 `src/`、`config/`、`tests/` 和 `experiments_runs/` 延续原有工程。本目录中的快照固定了投稿使用的版本，避免后续开发改变投稿证据；最新论文的复算请使用下面的专用入口。

## 校验与复算

在代码仓库根目录运行。仅校验文件使用 Python 标准库：

```text
python research_scripts/reproduce_submission_20261011.py --verify-only
```

在独立 Python 环境中按版本清单安装复算依赖，然后复算：

```text
python -m pip install -r final_report/Computational_Economics_20261011/reproducibility/requirements-verification.txt
python research_scripts/reproduce_submission_20261011.py
```

复算比较 31 项数值检查，输出位于 `reproducibility/output/recomputed/`，不会覆盖接收数据和接受结果。需要重新生成图件时增加 `--figures`；生成文件位于 `reproducibility/output/artwork/`。这些输出已在本目录的 `.gitignore` 中排除。

该依赖清单记录修订验证环境；原始实验环境未记录，不能据此宣称已重建原实验环境。450 条缺少实现参数向量的 LHS 记录保留为探索归档，9 条已知 census 重放不增加独立样本量；T80 使用选定配置的新种子，并非配对延续。完整解释见 `reproducibility/README.md`。

## 恢复完整原投稿包

将两个仓库检出到本地后，运行：

```text
python research_scripts/restore_submission_package_20261011.py --paper-root PATH_TO_PAPER_REPO --code-root PATH_TO_CODE_REPO --output NEW_EMPTY_DIRECTORY --zip-output NEW_ARCHIVE.zip
```

该工具先核对原投稿包 254 个成员、253 个原始文件指纹、3 个内层 ZIP 的 CRC 和解压代码包的 231 个指纹，再恢复全部原始目录。它还验证重新生成的 ZIP 与原交付 ZIP 字节完全一致。原交付 SHA-256：

```text
09d72b67ab34030eccd64df227e7d6886cd87445c4454954b3dbe3a634fd2667
```

作者和基金继续留空。本次仓库更新不代表向期刊提交，也不替代作者信息填写和最终投稿确认。
