"""
Actor-Critic Loss(Dreamer 风格)
"""

import torch
import torch.nn.functional as F


def compute_lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    lambda_: float = 0.95,
    gamma: float = 0.997,
) -> torch.Tensor:
    """
    计算 λ-return。

    Args:
        rewards: [T, B]
        values: [T+1, B]  (values[T] 是 bootstrap)
        continues: [T, B]
        lambda_: λ 系数
        gamma: γ 系数
    Returns:
        lambda_returns: [T, B]
    """
    T = rewards.shape[0]
    lambda_returns = torch.zeros_like(rewards)

    # 从后往前计算
    next_return = values[-1]  # bootstrap value
    for t in reversed(range(T)):
        # λ-return: r_t + γ * c_t * ((1-λ) * V(s_{t+1}) + λ * G_{t+1})
        next_values = values[t + 1] if t + 1 < T else values[t]
        lambda_returns[t] = rewards[t] + gamma * continues[t] * (
            (1 - lambda_) * next_values + lambda_ * next_return
        )
        next_return = lambda_returns[t]

    return lambda_returns


def compute_ac_loss(
    actor,
    critic,
    target_critic,
    world_model,
    init_obs: torch.Tensor,
    init_is_first: torch.Tensor,
    init_action: torch.Tensor = None,
    horizon: int = 15,
    gamma: float = 0.997,
    lambda_: float = 0.95,
    entropy_scale: float = 1e-3,
    imag_cont: float = 1.0,
) -> tuple:
    """
    计算 Actor-Critic 损失(Dreamer 风格)。

    流程:
    1. 从真实数据 init_obs 编码得到初始 state
    2. 在 WM 中想象 horizon 步
    3. 计算 λ-return
    4. Actor loss: 最大化 λ-return + entropy bonus
    5. Critic loss: 预测 λ-return

    Args:
        actor: A1Actor
        critic: A1Critic
        target_critic: SlowCritic(EMA target)
        world_model: WorldModel
        init_obs: [B, obs_dim]
        init_is_first: [B] bool
        horizon: 想象步数
    Returns:
        total_loss
        metrics
    """
    # 1. 编码初始状态
    embed = world_model.encode(init_obs)
    state = world_model.rssm.initial_state(
        batch_size=init_obs.shape[0],
        device=init_obs.device,
    )
    state, _ = world_model.rssm.obs_step(
        state, init_action if init_action is not None else torch.zeros_like(init_action[:, :world_model._action_dim]) if init_action is not None else torch.zeros(init_obs.shape[0], world_model._action_dim, device=init_obs.device),
        embed, init_is_first, sample=True,
    )

    # 2. 想象 horizon 步(Residual 不参与,干净 latent)
    states, actions, log_probs, entropies = world_model.imagine(
        actor, state, horizon,
        with_residual=False,  # ← 关键: Actor 训练在干净 latent
    )

    # 3. 计算 rewards 和 values
    # 简化: 这里假设有 reward_head(可以加一个)
    # 为了简化,使用 heuristic reward(基于动作幅度)
    imag_rewards = -0.01 * (actions ** 2).sum(dim=-1)  # [horizon, B]

    # continues(假设 1.0 = 不终止)
    imag_continues = torch.ones_like(imag_rewards) * imag_cont

    # values
    feat_seq = world_model.rssm.get_feat(states)  # [B, horizon, feat_dim]
    feat_seq_t = feat_seq.transpose(0, 1)  # [horizon, B, feat_dim]
    values_seq = critic(feat_seq.reshape(-1, feat_seq.shape[-1])).reshape(horizon, -1)

    # bootstrap value
    bootstrap_feat = world_model.rssm.get_feat(states)[:, -1]  # [B, feat_dim]
    bootstrap_value = target_critic(bootstrap_feat)  # [B]

    # 拼接 bootstrap value
    values_with_bootstrap = torch.cat([values_seq, bootstrap_value.unsqueeze(0)], dim=0)

    # 4. λ-return
    lambda_returns = compute_lambda_return(
        imag_rewards.transpose(0, 1),  # [B, horizon] -> [horizon, B] by transpose
        values_with_bootstrap,  # [horizon+1, B]
        imag_continues.transpose(0, 1),
        lambda_=lambda_,
        gamma=gamma,
    )  # [horizon, B]

    lambda_returns = lambda_returns.transpose(0, 1)  # [B, horizon]

    # 5. Actor loss
    discount = torch.cumprod(
        torch.cat([
            torch.ones_like(imag_continues[:, :1].transpose(0, 1)),  # [1, B]
            gamma * imag_continues.transpose(0, 1)[:-1],  # [horizon-1, B]
        ], dim=0),
        dim=0,
    )  # [horizon, B]

    actor_obj = (lambda_returns * discount).mean(dim=0)  # [B]
    actor_loss = -actor_obj.mean()

    # entropy bonus
    entropy_loss = -entropies.mean() * entropy_scale

    # 6. Critic loss
    critic_loss = F.mse_loss(
        values_seq,  # [horizon, B]
        lambda_returns.detach(),  # [horizon, B]
    )

    # Total
    total_loss = actor_loss + critic_loss + entropy_loss

    metrics = {
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'imag_reward_mean': imag_rewards.mean().item(),
        'lambda_return_mean': lambda_returns.mean().item(),
    }

    return total_loss, metrics