"""预设场景参数 (v2.1 升级).

新增场景:
- full_containment: 全方位技术封锁 + 外资撤离 + 出口市场关闭
- tech_explosion: 国际技术前沿加速爆发 (AI奇点场景)
- resource_war: 资源战争 — 价格剧烈波动 + 出口通道中断
"""

from typing import Dict


def preset_benign_globalization() -> Dict[str, float]:
    """良性全球化场景."""
    return {
        "g_world": 0.03,
        "block_max": 0.0,
        "mu_p_resource": 1.2,
        "sigma_p": 0.10,
        "world_digital_demand": 1.5,
        "fdi_availability": 1.5,
    }


def preset_tech_cold_war() -> Dict[str, float]:
    """技术冷战场景."""
    return {
        "g_world": 0.03,
        "block_max": 0.8,
        "mu_p_resource": 0.8,
        "sigma_p": 0.20,
        "world_digital_demand": 0.6,
        "fdi_availability": 0.2,
    }


def preset_resource_crash() -> Dict[str, float]:
    """资源价格崩溃场景."""
    return {
        "g_world": 0.02,
        "block_max": 0.2,
        "mu_p_resource": 0.3,
        "sigma_p": 0.40,
        "world_digital_demand": 0.8,
        "fdi_availability": 0.5,
    }


def preset_tech_boom() -> Dict[str, float]:
    """技术爆发场景."""
    return {
        "g_world": 0.08,
        "block_max": 0.4,
        "mu_p_resource": 1.0,
        "sigma_p": 0.15,
        "world_digital_demand": 2.0,
        "fdi_availability": 1.5,
    }


def preset_global_recession() -> Dict[str, float]:
    """全球衰退场景."""
    return {
        "g_world": 0.01,
        "block_max": 0.3,
        "mu_p_resource": 0.5,
        "sigma_p": 0.30,
        "world_digital_demand": 0.3,
        "fdi_availability": 0.1,
    }


# ===== v2.1 新增恶意场景 =====

def preset_full_containment() -> Dict[str, float]:
    """全方位封锁场景 —— 极端技术封锁 + 资本撤离 + 市场关闭.

    这是对策略B和C的真正考验:
    - B的高投资依赖数字出口和技术吸收 → 封锁下两者都断绝
    - C的外资依赖和技术进口偏好 → 封锁下变成致命弱点
    """
    return {
        "g_world": 0.05,          # 前沿仍在增长（封锁方在加速）
        "block_max": 0.95,         # 近乎完全的技术封锁
        "mu_p_resource": 0.4,      # 资源价格低迷
        "sigma_p": 0.35,           # 高波动
        "world_digital_demand": 0.1,  # 数字出口市场几乎关闭
        "fdi_availability": 0.05,  # 外资几乎断流
    }


def preset_tech_explosion() -> Dict[str, float]:
    """技术爆发场景 —— AI奇点型前沿加速.

    国际技术前沿以12%年增速爆发，任何保守策略都会被瞬间甩开。
    这是对策略A的终极考验: "渐进"在奇点面前等于"等死"。
    """
    return {
        "g_world": 0.12,           # 前沿爆发式增长
        "block_max": 0.7,          # 技术领先方加强封锁
        "mu_p_resource": 1.0,
        "sigma_p": 0.20,
        "world_digital_demand": 2.5,  # 数字需求暴涨
        "fdi_availability": 1.0,   # 但外资只投向已有数字基础的国家
    }


def preset_resource_war() -> Dict[str, float]:
    """资源战争场景 —— 资源价格剧烈波动 + 出口通道被切断.

    对策略C的双重打击:
    - 资源出口收入极不稳定
    - 外资因地缘风险撤离
    """
    return {
        "g_world": 0.03,
        "block_max": 0.6,          # 次要技术封锁
        "mu_p_resource": 0.5,      # 资源均价偏低
        "sigma_p": 0.60,           # 极高波动 → 价格可能跌到0.1
        "world_digital_demand": 0.5,
        "fdi_availability": 0.15,  # 地缘风险驱离外资
    }


SCENARIO_PRESETS = {
    "benign_globalization": preset_benign_globalization,
    "tech_cold_war": preset_tech_cold_war,
    "resource_crash": preset_resource_crash,
    "tech_boom": preset_tech_boom,
    "global_recession": preset_global_recession,
    # v2.1 新增
    "full_containment": preset_full_containment,
    "tech_explosion": preset_tech_explosion,
    "resource_war": preset_resource_war,
}


def get_scenario(name: str) -> Dict[str, float]:
    if name not in SCENARIO_PRESETS:
        raise ValueError(
            f"Unknown scenario: {name}. Available: {list(SCENARIO_PRESETS.keys())}"
        )
    return SCENARIO_PRESETS[name]()
