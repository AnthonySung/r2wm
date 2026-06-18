"""
两层 Residual 网络(对齐 ReDRAW 设计)

PhysicalResidual:
- 加在 latent stochastic mean 上
- 补偿物理参数误差(电机扭矩、PD 增益、质量、摩擦)
- 输入: prev_stoch + action

DynamicsResidual:
- 加在 deter 上(需要历史)
- 补偿未建模动力学(齿轮间隙、皮带柔性、接触动力学)
- 输入: deter_history(K 步) + action

ReDRAW 关键设计(对齐):
1. 严格零初始化
2. Stage 2 重新创建(更小结构: 1 层)
3. Stage 1 时 stop_gradients
4. 想象训练时 Residual 不参与
"""

import torch
import torch.nn as nn


def _zero_init_layer(layer: nn.Linear):
    """严格零初始化(对齐 ReDRAW)"""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class PhysicalResidual(nn.Module):
    """
    Layer 1: 物理参数误差补偿

    输入: prev_stoch + action
    输出: delta_mean(加在 latent mean 上)
    """

    def __init__(
        self,
        stoch_dim: int,
        action_dim: int,
        hidden: int = 64,
        n_layers: int = 3,
        init: str = 'zero',
    ):
        """
        Args:
            stoch_dim: latent stochastic 维度(实际是 stoch * discrete)
            action_dim: 动作维度
            hidden: MLP 隐藏层维度
            n_layers: MLP 层数
            init: 'zero'(ReDRAW 默认)或 'small_normal'
        """
        super().__init__()
        self._stoch_dim = stoch_dim
        self._action_dim = action_dim

        in_dim = stoch_dim + action_dim
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, stoch_dim))

        self.net = nn.Sequential(*layers)

        # 零初始化(对齐 ReDRAW)
        if init == 'zero':
            for m in self.net:
                if isinstance(m, nn.Linear):
                    _zero_init_layer(m)
        elif init == 'small_normal':
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        prev_stoch: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 latent mean 的修正量。

        Args:
            prev_stoch: [..., stoch_dim]
            action: [..., action_dim]
        Returns:
            delta_mean: [..., stoch_dim]
        """
        inp = torch.cat([prev_stoch, action], dim=-1)
        return self.net(inp)


class DynamicsResidual(nn.Module):
    """
    Layer 2: 未建模动力学补偿

    输入: deter_history(K 步) + action
    输出: delta_deter(加在 deter 上)
    """

    def __init__(
        self,
        deter_dim: int,
        action_dim: int,
        history_len: int = 4,
        hidden: int = 128,
        n_layers: int = 3,
        init: str = 'zero',
    ):
        """
        Args:
            deter_dim: deter 维度
            action_dim: 动作维度
            history_len: history 步数 K
            hidden: MLP 隐藏层维度
            n_layers: MLP 层数
            init: 初始化方式
        """
        super().__init__()
        self._deter_dim = deter_dim
        self._action_dim = action_dim
        self._history_len = history_len

        in_dim = deter_dim * history_len + action_dim
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, deter_dim))

        self.net = nn.Sequential(*layers)

        if init == 'zero':
            for m in self.net:
                if isinstance(m, nn.Linear):
                    _zero_init_layer(m)
        elif init == 'small_normal':
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        deter_history: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 deter 的修正量。

        Args:
            deter_history: [..., history_len, deter_dim]
            action: [..., action_dim]
        Returns:
            delta_deter: [..., deter_dim]
        """
        # 展平 history
        inp = torch.cat([deter_history.flatten(-2), action], dim=-1)
        return self.net(inp)


# ============================================================
# 工厂函数:用于 Stage 1 / Stage 2 不同配置
# ============================================================

def make_physical_residual_for_stage1(
    stoch_dim: int,
    action_dim: int,
    config: dict,
) -> PhysicalResidual:
    """Stage 1 用 3 层 MLP"""
    return PhysicalResidual(
        stoch_dim=stoch_dim,
        action_dim=action_dim,
        hidden=config['stage1_hidden'],
        n_layers=config['stage1_n_layers'],
        init=config['init'],
    )


def make_physical_residual_for_stage2(
    stoch_dim: int,
    action_dim: int,
    config: dict,
) -> PhysicalResidual:
    """Stage 2 用 1 层 MLP(对齐 ReDRAW ensemble_residual_extra_small_1_member)"""
    return PhysicalResidual(
        stoch_dim=stoch_dim,
        action_dim=action_dim,
        hidden=config['stage2_hidden'],
        n_layers=config['stage2_n_layers'],
        init=config['init'],
    )


def make_dynamics_residual_for_stage1(
    deter_dim: int,
    action_dim: int,
    config: dict,
) -> DynamicsResidual:
    """Stage 1 用 3 层 MLP"""
    return DynamicsResidual(
        deter_dim=deter_dim,
        action_dim=action_dim,
        history_len=config['history_len'],
        hidden=config['dynamics_hidden'],
        n_layers=config['stage1_n_layers'],
        init=config['init'],
    )


def make_dynamics_residual_for_stage2(
    deter_dim: int,
    action_dim: int,
    config: dict,
) -> DynamicsResidual:
    """Stage 2 用 1 层 MLP"""
    return DynamicsResidual(
        deter_dim=deter_dim,
        action_dim=action_dim,
        history_len=config['history_len'],
        hidden=config['dynamics_hidden'],
        n_layers=config['stage2_n_layers'],
        init=config['init'],
    )