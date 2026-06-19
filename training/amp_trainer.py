"""
AMP Discriminator 训练器 (B2)

对齐 WMP AMPPPO 的 discriminator 训练:
- GAN 风格: expert_d → 1, policy_d → -1
- Gradient penalty (WGAN-GP, λ=10)
- AMPNormalizer 随训练同步更新
"""
import torch
import torch.nn as nn
import torch.optim as optim


def compute_amp_loss(
    discriminator,
    amp_loader,           # WMP 的 AMPLoader (提供 expert batch)
    policy_amp_obs,       # 当前 policy 收集的 amp_obs: [B, amp_obs_dim]
    policy_next_amp_obs,  # 下一个 amp_obs: [B, amp_obs_dim]
    amp_normalizer=None,
    expert_target: float = 1.0,
    policy_target: float = -1.0,
    grad_pen_lambda: float = 10.0,
    expert_batch_size: int = 4096,
    device: str = 'cuda',
):
    """
    算 AMP discriminator loss

    Args:
        discriminator: WMP 的 AMPDiscriminator
        amp_loader: WMP 的 AMPLoader (生成 expert 数据)
        policy_amp_obs: policy 的 amp_obs [B, amp_obs_dim]
        policy_next_amp_obs: policy 下一个 amp_obs [B, amp_obs_dim]
        amp_normalizer: WMP 的 Normalizer (RunningMeanStd)
        expert_target: 1.0 (expert 应该是真)
        policy_target: -1.0 (policy 应该是假)
        grad_pen_lambda: 10.0 (WGAN-GP 系数)
        expert_batch_size: 从 amp_loader 采多少 expert

    Returns:
        total_loss, metrics_dict
    """
    B = policy_amp_obs.shape[0]
    amp_obs_dim = policy_amp_obs.shape[-1]

    # 1. 采 expert batch
    # AMPLoader.feed_forward_generator 返回 (state, next_state) 都是 [mini_batch, obs_dim]
    # 我们采 expert_batch_size 个
    # 注意: WMP 接口是位置参数 (num_mini_batch, mini_batch_size),不是关键字
    expert_gen = amp_loader.feed_forward_generator(
        1,  # num_mini_batch
        expert_batch_size,  # mini_batch_size
    )
    expert_state, expert_next_state = next(expert_gen)
    expert_state = expert_state.to(device)
    expert_next_state = expert_next_state.to(device)

    # 2. 归一化(如果用 Normalizer)
    if amp_normalizer is not None:
        with torch.no_grad():
            policy_amp_obs_n = amp_normalizer.normalize_torch(policy_amp_obs, device)
            policy_next_amp_obs_n = amp_normalizer.normalize_torch(policy_next_amp_obs, device)
            expert_state_n = amp_normalizer.normalize_torch(expert_state, device)
            expert_next_state_n = amp_normalizer.normalize_torch(expert_next_state, device)
    else:
        policy_amp_obs_n = policy_amp_obs
        policy_next_amp_obs_n = policy_next_amp_obs
        expert_state_n = expert_state
        expert_next_state_n = expert_next_state

    # 3. Discriminator forward
    policy_d = discriminator(torch.cat([policy_amp_obs_n, policy_next_amp_obs_n], dim=-1))
    expert_d = discriminator(torch.cat([expert_state_n, expert_next_state_n], dim=-1))

    # 4. GAN loss (WMP 风格:MSE 到 target)
    expert_loss = nn.MSELoss()(
        expert_d, torch.full_like(expert_d, expert_target)
    )
    policy_loss = nn.MSELoss()(
        policy_d, torch.full_like(policy_d, policy_target)
    )
    amp_loss = 0.5 * (expert_loss + policy_loss)

    # 5. Gradient penalty (WGAN-GP)
    grad_pen_loss = discriminator.compute_grad_pen(
        expert_state, expert_next_state, lambda_=grad_pen_lambda
    )

    # 6. 总损失
    total_loss = amp_loss + grad_pen_loss

    metrics = {
        'amp_loss': amp_loss.item(),
        'expert_loss': expert_loss.item(),
        'policy_loss': policy_loss.item(),
        'grad_pen_loss': grad_pen_loss.item(),
        'expert_d_mean': expert_d.mean().item(),
        'policy_d_mean': policy_d.mean().item(),
    }

    # 7. 更新 Normalizer (用 policy 和 expert 数据)
    if amp_normalizer is not None:
        amp_normalizer.update(policy_amp_obs.cpu().numpy())
        amp_normalizer.update(expert_state.cpu().numpy())

    return total_loss, metrics
