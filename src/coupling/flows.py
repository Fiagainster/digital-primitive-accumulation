"""部门间耦合 —— 物质流、资本流、数据流."""

from ..core.state import SystemState


class CouplingLayer:
    """部门间耦合计算."""

    def __init__(self, params: dict):
        self.p = params

    def compute_flows(self, state: SystemState):
        """计算所有跨部门流."""
        pass  # 流计算已内嵌在各模块中，此处作为协调层保留
