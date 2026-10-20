"""YAML配置文件加载器 —— 统一管理模型所有参数、权重和初始条件.

设计原则:
- YAML配置文件是参数的真实来源，代码中不再硬编码默认值
- 分层扁平化: YAML的嵌套结构自动展开为扁平参数字典
- 向后兼容: 如果YAML文件不存在或参数缺失，回退到schema.py中的硬编码默认值
- 优先级链: YAML默认 → 国情预设 → 国际场景 → 用户覆盖

使用方式:
    from src.params.yaml_loader import load_config

    # 加载默认配置
    params = load_config("config/default.yaml")

    # 加载自定义配置
    params = load_config("config/custom.yaml", profile="digital_leapfrogger",
                         scenario="tech_cold_war", overrides={"B_g_target": 0.30})
"""

import os
import warnings
from typing import Dict, Optional, Any, List, Tuple

import yaml


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """递归展平嵌套字典到最底层标量值.

    所有嵌套结构都被展开，每个标量值获得完整的点分隔路径。
    例如:
        {"structural": {"production_base": {"sigma_b": 0.8}}}
        → {"structural.production_base.sigma_b": 0.8}

    Args:
        d: 嵌套字典
        prefix: 当前键前缀

    Returns:
        展平后的字典，每个键都是完整的点分隔路径
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            # 递归展平所有嵌套字典（包括叶子字典）
            items.update(_flatten_dict(v, new_key))
        elif isinstance(v, list):
            # 列表保留原样（如混合策略配比）
            items[new_key] = v
        else:
            # 标量值：直接记录
            items[new_key] = v
    return items


def _extract_scalar_params(flat: Dict[str, Any]) -> Dict[str, float]:
    """从展平字典中提取所有标量参数.

    将形如 "structural.production_base.sigma_b" 的键,
    提取最后一个部分 "sigma_b" 作为参数ID。

    Args:
        flat: 展平后的字典(键如 "structural.production_base.sigma_b")

    Returns:
        扁平参数字典 {param_id: value}
    """
    params: Dict[str, float] = {}
    for key, value in flat.items():
        if isinstance(value, (int, float)):
            leaf_key = key.split(".")[-1]
            if leaf_key not in params:
                params[leaf_key] = float(value)
        elif isinstance(value, bool):
            leaf_key = key.split(".")[-1]
            if leaf_key not in params:
                params[leaf_key] = 1.0 if value else 0.0
    return params


class YamlConfigLoader:
    """YAML配置加载器.

    属性:
        raw: 原始YAML数据(嵌套字典)
        flat: 展平后的参数字典
        params: 可直接传入SimulationEngine的参数字典
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML配置文件路径，为None时仅使用硬编码默认值
        """
        self.raw: Dict[str, Any] = {}
        self.flat: Dict[str, Any] = {}
        self._config_path = config_path

        if config_path and os.path.exists(config_path):
            self._load(config_path)
        elif config_path:
            warnings.warn(
                f"配置文件不存在: {config_path}，将使用硬编码默认值。"
                f"请运行 'python main.py dump-config' 生成默认配置文件。"
            )

    def _load(self, path: str):
        """加载并解析YAML文件."""
        with open(path, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.flat = _flatten_dict(self.raw)

    # ─────────────────────────────────────────────
    # 参数提取
    # ─────────────────────────────────────────────

    def get_params(self, regime: str = "A") -> Dict[str, float]:
        """获取完整的模拟参数字典.

        从YAML中提取所有标量参数，展平为 {param_id: value} 格式。

        Args:
            regime: 积累策略 (A/B/C/M)，暂未使用(保留兼容)

        Returns:
            扁平参数字典
        """
        params: Dict[str, float] = {}

        # 遍历展平后的所有键值对
        for key, value in self.flat.items():
            if isinstance(value, (int, float)):
                leaf = key.split(".")[-1]
                if leaf not in params:
                    params[leaf] = float(value)
            elif isinstance(value, bool):
                leaf = key.split(".")[-1]
                if leaf not in params:
                    params[leaf] = 1.0 if value else 0.0

        return params

    def get_structural_params(self) -> Dict[str, float]:
        """获取L1: 结构参数."""
        struct = self.raw.get("structural", {})
        flat = _flatten_dict(struct)
        result: Dict[str, float] = {}
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                result[k.split(".")[-1]] = float(v)
        return result

    def get_strategy_params(self, strategy: str = "common") -> Dict[str, float]:
        """获取L2: 策略参数."""
        strat = self.raw.get("strategy", {})
        if strategy in strat:
            flat = _flatten_dict(strat[strategy])
        else:
            flat = _flatten_dict(strat)
        result: Dict[str, float] = {}
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                result[k.split(".")[-1]] = float(v)
        return result

    def get_scenario_params(self, scenario_name: Optional[str] = None) -> Dict[str, float]:
        """获取L3: 场景参数.

        Args:
            scenario_name: 场景名称(如 "benign_globalization")，
                          为None时返回默认场景参数
        """
        if scenario_name:
            scenarios = self.raw.get("scenarios", {})
            if scenario_name in scenarios:
                flat = _flatten_dict(scenarios[scenario_name])
                result: Dict[str, float] = {}
                for k, v in flat.items():
                    if isinstance(v, (int, float)):
                        result[k.split(".")[-1]] = float(v)
                return result

        # 默认场景参数
        defaults = self.raw.get("scenario_defaults", {})
        result: Dict[str, float] = {}
        for k, v in defaults.items():
            if isinstance(v, (int, float)):
                result[k] = float(v)
        return result

    def get_meta_params(self) -> Dict[str, float]:
        """获取L4: 元参数."""
        meta = self.raw.get("meta", {})
        flat = _flatten_dict(meta)
        result: Dict[str, float] = {}
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                result[k.split(".")[-1]] = float(v)
        return result

    def get_cdi_weights(self, preset: str = "default") -> Dict[str, float]:
        """获取CDI权重方案."""
        weights = self.raw.get("cdi_weights", {})
        if preset in weights:
            return {k: float(v) for k, v in weights[preset].items()
                    if isinstance(v, (int, float))}
        return {}

    def get_simulation_config(self) -> Dict[str, Any]:
        """获取模拟运行配置."""
        return self.raw.get("simulation", {})

    def get_country_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """获取国情预设的原始数据.

        Args:
            profile_name: 国情预设名称

        Returns:
            国情预设字典，如果不存在则返回None
        """
        profiles = self.raw.get("country_profiles", {})
        return profiles.get(profile_name)

    def get_all_country_profile_names(self) -> List[str]:
        """获取所有国情预设名称."""
        profiles = self.raw.get("country_profiles", {})
        return list(profiles.keys())

    def get_all_scenario_names(self) -> List[str]:
        """获取所有国际场景名称."""
        scenarios = self.raw.get("scenarios", {})
        return list(scenarios.keys())

    def get_mixed_strategies(self) -> Dict[str, List[float]]:
        """获取混合策略标准配比."""
        mixed = self.raw.get("mixed_strategies", {})
        result = {}
        for k, v in mixed.items():
            if isinstance(v, list) and len(v) == 3:
                result[k] = [float(x) for x in v]
        return result

    def get_mixed_constraints(self) -> Dict[str, Any]:
        """获取混合策略约束规则."""
        mixed = self.raw.get("mixed_strategies", {})
        return mixed.get("constraints", {})

    def get_param_domains(self) -> Dict[str, Dict[str, float]]:
        """获取参数定义域参考 (机器可读的 min/max)."""
        return self.raw.get("param_domains", {})

    def get_parameter_sources(self) -> Dict[str, str]:
        """获取参数理论来源对照表."""
        return self.raw.get("parameter_sources", {})

    def get_profile_strategy_guide(self, profile_name: str = None) -> dict:
        """获取国情预设的策略推荐指南.

        Args:
            profile_name: 国情名称，为 None 时返回全部

        Returns:
            策略指南字典，含 recommended_regime/viable_regimes/contraindicated/note
        """
        guide = self.raw.get("profile_strategy_guide", {})
        if profile_name:
            return guide.get(profile_name, {})
        return guide

    def country_profile_to_params(self, profile_name: str) -> Dict[str, float]:
        """将国情预设转换为扁平参数字典(IC_前缀).

        模拟 country_profiles.py 中 CountryProfile.to_params_dict() 的逻辑。

        Args:
            profile_name: 国情预设名称

        Returns:
            扁平参数字典，键如 "IC_K_i", "IC_L_b" 等
        """
        profile = self.get_country_profile(profile_name)
        if profile is None:
            return {}

        p = profile
        tech_gap = p.get("tech_gap_initial", 3.0)

        result = {
            # 资本
            "IC_K_b": p.get("K_b", 200.0),
            "IC_K_i": p.get("K_i", 300.0),
            "IC_K_d": p.get("K_d", 20.0),
            "IC_K_d_infra": p.get("K_d_infra", 8.0),
            "IC_K_d_app": p.get("K_d_app", 12.0),
            "IC_K_s_high": p.get("K_s_high", 30.0),
            "IC_K_s_low": p.get("K_s_low", 20.0),
            # 劳动力
            "IC_L_b": p.get("L_b", 25.0),
            "IC_L_i": p.get("L_i", 30.0),
            "IC_L_d": p.get("L_d", 5.0),
            "IC_L_d_infra": p.get("L_d_infra", 1.5),
            "IC_L_d_app": p.get("L_d_app", 3.5),
            "IC_L_s_high": p.get("L_s_high", 8.0),
            "IC_L_s_low": p.get("L_s_low", 7.0),
            # 人力资本
            "IC_H_b": p.get("H_b", 0.8),
            "IC_H_i": p.get("H_i", 1.0),
            "IC_H_d": p.get("H_d", 1.5),
            "IC_H_d_infra": p.get("H_d_infra", 1.6),
            "IC_H_d_app": p.get("H_d_app", 1.4),
            "IC_H_s_high": p.get("H_s_high", 1.1),
            "IC_H_s_low": p.get("H_s_low", 0.5),
            # 数字化
            "IC_DL_b": p.get("DL_b", 0.05),
            "IC_DL_i": p.get("DL_i", 0.10),
            "IC_DL_d": p.get("DL_d", 0.15),
            "IC_DL_d_infra": p.get("DL_d_infra", 0.08),
            "IC_DL_d_app": p.get("DL_d_app", 0.20),
            "IC_DL_s_high": p.get("DL_s_high", 0.08),
            "IC_DL_s_low": p.get("DL_s_low", 0.03),
            # 技术水平
            "IC_A_b": p.get("A_b", 1.0),
            "IC_A_i": p.get("A_i", 1.2),
            "IC_A_d": p.get("A_d", 1.5),
            "IC_A_d_infra": p.get("A_d_infra", 1.2),
            "IC_A_d_app": p.get("A_d_app", 1.8),
            "IC_A_s_high": p.get("A_s_high", 1.0),
            "IC_A_s_low": p.get("A_s_low", 0.7),
            # 工资
            "IC_W_b": p.get("W_b", 2.0),
            "IC_W_i": p.get("W_i", 3.0),
            "IC_W_d": p.get("W_d", 5.0),
            "IC_W_d_infra": p.get("W_d_infra", 6.0),
            "IC_W_d_app": p.get("W_d_app", 5.0),
            "IC_W_s_high": p.get("W_s_high", 3.5),
            "IC_W_s_low": p.get("W_s_low", 1.2),
            # 数字密度
            "IC_DD_b": p.get("DD_b", 0.01),
            "IC_DD_i": p.get("DD_i", 0.03),
            "IC_DD_d": p.get("DD_d", 0.05),
            "IC_DD_d_infra": p.get("DD_d_infra", 0.03),
            "IC_DD_d_app": p.get("DD_d_app", 0.06),
            "IC_DD_s_high": p.get("DD_s_high", 0.04),
            "IC_DD_s_low": p.get("DD_s_low", 0.01),
            # 兼容旧服务业参数
            "IC_K_s": p.get("K_s_low", 20.0),
            "IC_L_s": p.get("L_s_low", 7.0),
            "IC_H_s": p.get("H_s_low", 0.5),
            "IC_DL_s": p.get("DL_s_low", 0.03),
            "IC_A_s": p.get("A_s_low", 0.7),
            "IC_W_s": p.get("W_s_low", 1.2),
            "IC_DD_s": p.get("DD_s_low", 0.01),
            # 进口依赖度
            "IC_import_dep_d_infra": p.get("import_dep_d_infra", 0.75),
            "IC_import_dep_d_app": p.get("import_dep_d_app", 0.30),
            # 资源/制度/不平等
            "IC_R_initial": p.get("R_initial", 0.60),
            "R_initial_ratio": p.get("R_initial", 0.60),
            "IC_institution_capacity": p.get("institution_capacity", 0.60),
            "institution_capacity": p.get("institution_capacity", 0.60),
            "IC_gini_initial": p.get("gini_initial", 0.38),
            "IC_tech_gap_initial": tech_gap,
            "IC_tech_gap_infra_initial": tech_gap * 1.3,
            "IC_tech_gap_app_initial": tech_gap * 0.8,
            # 数据资产
            "IC_data_producer": p.get("data_producer_initial", 1.0),
            "IC_data_consumer": p.get("data_consumer_initial", 0.5),
            "IC_data_public": p.get("data_public_initial", 0.3),
            # 数字服务
            "IC_DS_b": 0.5,
            "IC_DS_i": 1.0,
            "IC_DS_d": 2.0,
            "IC_DS_d_infra": 2.0 * 0.6,
            "IC_DS_d_app": 2.0 * 0.8,
            "IC_DS_s_high": 0.8,
            "IC_DS_s_low": 0.3,
        }
        return result

    def build_params(self,
                     profile_name: Optional[str] = None,
                     regime: str = "A",
                     scenario: Optional[str] = None,
                     user_overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """一站式构建完整参数集.

        参数加载优先级（从低到高）：
        1. YAML中的结构/策略/场景默认参数
        2. 国情预设覆盖
        3. 国际场景覆盖
        4. user_overrides（最高优先级）

        如果YAML未加载(没有config_path)，回退到schema.py的build_from_profile。

        Args:
            profile_name: 国情预设名称
            regime: 积累策略 A/B/C/M
            scenario: 国际场景名称
            user_overrides: 用户自定义参数覆盖

        Returns:
            完整的参数字典
        """
        # 如果YAML未加载，回退到schema
        if not self._config_path or not self.raw:
            from .schema import get_schema
            schema = get_schema()
            return schema.build_from_profile(
                profile_name=profile_name or "manufacturing_giant",
                regime=regime,
                scenario=scenario,
                user_overrides=user_overrides,
            )

        # 1. 基础参数: YAML中的结构参数 + 策略参数 + 场景默认 + 元参数
        params: Dict[str, float] = {}
        params.update(self.get_structural_params())
        params.update(self.get_strategy_params("common"))
        params.update(self.get_strategy_params(regime if regime in "ABC" else "A"))
        params.update(self.get_scenario_params())
        params.update(self.get_meta_params())

        # 2. 国情预设覆盖
        if profile_name:
            profile_params = self.country_profile_to_params(profile_name)
            params.update(profile_params)

        # 3. 国际场景覆盖
        if scenario:
            scene_params = self.get_scenario_params(scenario)
            params.update(scene_params)

        # 4. 用户自定义覆盖（最高优先级）
        if user_overrides:
            params.update(user_overrides)

        return params


# ─────────────────────────────────────────────
# 便捷函数
# ─────────────────────────────────────────────

def load_config(config_path: str) -> YamlConfigLoader:
    """加载YAML配置文件.

    Args:
        config_path: YAML文件路径

    Returns:
        YamlConfigLoader 实例
    """
    return YamlConfigLoader(config_path)


def build_params_from_yaml(config_path: str,
                           profile_name: Optional[str] = None,
                           regime: str = "A",
                           scenario: Optional[str] = None,
                           user_overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """从YAML配置文件一站式构建参数字典.

    Args:
        config_path: YAML配置文件路径
        profile_name: 国情预设名称
        regime: 积累策略
        scenario: 国际场景名称
        user_overrides: 用户自定义覆盖

    Returns:
        完整参数字典
    """
    loader = YamlConfigLoader(config_path)
    return loader.build_params(
        profile_name=profile_name,
        regime=regime,
        scenario=scenario,
        user_overrides=user_overrides,
    )
