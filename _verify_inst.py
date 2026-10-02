"""交叉验证：institution_capacity 覆盖是否真的改变结果。"""
import sys
sys.path.insert(0, '.')
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
