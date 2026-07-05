"""数字化进程 —— S曲线吸收 + 数字服务分配 + 异质数据资产."""

import numpy as np
from ..core.state import SystemState


class DigitalizationModel:
    """数字化进程与数据资产模型."""

    def __init__(self, params: dict):
        self.p = params

    def step(self, state: SystemState, dt: float):
        """一个时间步的数字化更新."""
        # 1. 分配数字服务
        self._allocate_digital_services(state)
        # 2. 更新数字化水平
        self._update_digitalization(state, dt)
        # 3. 更新数据资产
        self._update_data_assets(state, dt)

    def _allocate_digital_services(self, state: SystemState):
        """将数字部门产出分配给各部门."""
        Y_d = state.sectors["d"].Y
        kappa = self.p["kappa"]

        # 市场分配：按边际回报
        market_shares = {}
        for key in ["b", "i", "d"]:
            s = state.sectors[key]
            if key == "i":
                marginal = self.p["gamma_i"] * s.Y / max(s.DS, 0.01)
            else:
                marginal = s.Y / max(Y_d, 0.01)
            market_shares[key] = marginal

        total_m = sum(market_shares.values())
        market_shares = {k: v / max(total_m, 1e-10) for k, v in market_shares.items()}

        # 计划分配：倾向于基础部门和工业
        plan_shares = {"b": 0.30, "i": 0.45, "d": 0.25}

        for key in ["b", "i", "d"]:
            share = (1 - kappa) * market_shares[key] + kappa * plan_shares[key]
            state.sectors[key].DS = share * Y_d

    def _update_digitalization(self, state: SystemState, dt: float):
        """更新各部门数字化水平."""
        for key in ["b", "i", "d"]:
            s = state.sectors[key]
            ds_ratio = s.DS / max(s.Y, 0.01)

            # 吸收因子（S曲线）
            absorb = self._absorption_factor(s.DL)

            # 数字化水平变化
            dDL = 0.3 * ds_ratio * (1 - s.DL) ** 1.0 * absorb
            s.DL += dDL * dt
            s.DL = np.clip(s.DL, 0.0, 0.99)

    def _absorption_factor(self, DL: float) -> float:
        """三阶段吸收因子."""
        if DL < 0.2:
            return 0.2 * (DL / 0.2) ** 0.3 if DL > 0 else 0.02
        elif DL < 0.6:
            return 0.2 + (DL - 0.2) / 0.4 * 0.8
        else:
            return 1.0 - 0.2 * ((DL - 0.6) / 0.4) ** 2

    def _update_data_assets(self, state: SystemState, dt: float):
        """更新异质性数据资产."""
        p = self.p

        # 生产者数据：来自工业数字化
        d_prod = p["phi_prod_data"] * state.sectors["i"].Y * state.sectors["i"].DL
        d_prod -= 0.10 * state.data_assets.producer_data

        # 消费者数据：来自数字消费
        digital_penetration = state.sectors["d"].DL
        d_cons = p["phi_cons_data"] * state.gdp * digital_penetration * 0.1
        d_cons -= 0.25 * state.data_assets.consumer_data

        # 公共数据
        d_pub = 0.02 * state.gdp * self.p["pub_infra_invest"]
        d_pub -= 0.05 * state.data_assets.public_data

        state.data_assets.producer_data += d_prod * dt
        state.data_assets.consumer_data += d_cons * dt
        state.data_assets.public_data += d_pub * dt

        # 非负约束
        for attr in ["producer_data", "consumer_data", "public_data"]:
            setattr(state.data_assets, attr,
                    max(0.0, getattr(state.data_assets, attr)))
