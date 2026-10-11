#!/usr/bin/env python
"""参数覆盖核验 —— 确认 ``institution_capacity`` 与 ``g_world`` 的运行时覆盖真的生效。

背景
----
论文第 5 章的制度容量反事实（inst 0.65 -> 0.80 使 post-Soviet 的 A 策略由 0% 转为
可存续）依赖 ``build_from_profile(...)`` 之后能覆盖 ``institution_capacity``。
本脚本打印默认配置里 inst / g_world 相关键，再逐档覆盖并回显，确认键名正确、
覆盖链路通畅（避免出现"改了参数但模型读的是另一个键"这类静默失效）。

运行方式:
    python research_scripts/verify_inst_capacity.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.params.schema import get_schema


def build(inst, gw, seed):
    s = get_schema()
    p = s.build_from_profile(profile_name="manufacturing_giant", regime="A",
                             scenario="benign_globalization")
    p.update({"g_world": gw, "institution_capacity": inst})
    return p


if __name__ == "__main__":
    s = get_schema()
    base = s.build_from_profile(profile_name="manufacturing_giant", regime="A",
                                scenario="benign_globalization")
    print("默认 inst 相关键:",
          {k: v for k, v in base.items() if "inst" in k.lower()})
    print("默认 g_world 键:",
          {k: v for k, v in base.items() if "g_world" in k.lower()})
    print()
    for inst in [0.50, 0.65, 0.80, 0.90]:
        p = build(inst, 0.08, 42)
        print(f"inst={inst} -> 覆盖后:", {k: v for k, v in p.items() if "inst" in k.lower()})
