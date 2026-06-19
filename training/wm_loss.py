"""
World Model Loss
"""

import torch
import torch.nn.functional as F


def compute_wm_loss(
    world_model,
    batch: dict,
    free_bits: float = 1.0,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    recon_scale: float = 1.0,
    reward_scale: float = 1.0,
) -> tuple:
    """
    计算 World Model 的损失。

    包括:
    - KL loss(dyn + rep)
    - Reconstruction loss
    - Reward prediction loss

    Args:
        world_model: WorldModel 实例
        batch: dict with 'obs', 'action', 'reward', 'is_first'
        free_bits: free bits 下界
        dyn_scale: dyn loss 权重
        rep_scale: rep loss 权重
        recon_scale: recon loss 权重
        reward_scale: reward loss 权重

    Returns:
        total_loss: 标量
        metrics: dict
    """
    obs = batch['obs']          # [B, T, obs_dim]
    action = batch['action']    # [B, T, action_dim]
    is_first = batch['is_first']
    reward = batch['reward'] if 'reward' in batch else None

    # 观察(Residual 参与 + stop_grad)
    posts, priors = world_model.observe(
        obs, action, is_first,
        with_residual=True,
        stop_residual_grad=True,  # ← Stage 1: Residual stop gradient
    )

    # KL loss
    kl_loss, dyn_loss, rep_loss = world_model.rssm.kl_loss(
        posts, priors, free=free_bits, dyn_scale=dyn_scale, rep_scale=rep_scale
    )

    # Reconstruction loss
    feat = world_model.rssm.get_feat(posts)
    recon = world_model.decode(posts)
    # 把 obs flatten 到 [B*T, obs_dim]
    obs_flat = obs.reshape(-1, obs.shape[-1])
    recon_flat = recon.reshape(-1, recon.shape[-1])
    recon_loss = F.mse_loss(recon_flat, obs_flat)

    # Reward prediction loss(Dreamer 风格: WM 学预测真实 env reward)
    if reward is not None and reward_scale > 0:
        # feat: [B, T, feat_dim] -> 预测 reward: [B, T]
        reward_pred = world_model.predict_reward(feat)
        reward_loss = F.mse_loss(reward_pred, reward)
        # Debug: 看 reward 实际范围
        if not hasattr(compute_wm_loss, '_debug_counter'):
            compute_wm_loss._debug_counter = 0
        compute_wm_loss._debug_counter += 1
        if compute_wm_loss._debug_counter % 5 == 0:
            print(f'[WM debug] reward_batch: min={reward.min().item():.4f} max={reward.max().item():.4f} mean={reward.mean().item():.4f} std={reward.std().item():.4f} | reward_pred: min={reward_pred.min().item():.4f} max={reward_pred.max().item():.4f}', flush=True)
    else:
        reward_loss = torch.tensor(0.0, device=obs.device)

    total = kl_loss + recon_scale * recon_loss + reward_scale * reward_loss

    metrics = {
        'wm_loss': total.item(),
        'kl_loss': kl_loss.item(),
        'dyn_loss': dyn_loss.item(),
        'rep_loss': rep_loss.item(),
        'recon_loss': recon_loss.item(),
        'reward_loss': reward_loss.item() if reward is not None else 0.0,
    }

    return total, metrics


def compute_residual_loss(
    world_model,
    batch: dict,
    free_bits: float = 1.0,
) -> tuple:
    """
    Stage 2: 只计算 Residual 的 KL loss。

    主 RSSM 冻结,Residual 从头学习。
    """
    obs = batch['obs']
    action = batch['action']
    is_first = batch['is_first']

    # 观察(Residual 参与,接收梯度)
    posts, priors = world_model.observe(
        obs, action, is_first,
        with_residual=True,
        stop_residual_grad=False,  # ← Stage 2: Residual 接收梯度
    )

    # KL loss(让 sim+residual 接近 real)
    kl_loss, dyn_loss, rep_loss = world_model.rssm.kl_loss(
        posts, priors, free=free_bits, dyn_scale=1.0, rep_scale=0.0
    )

    metrics = {
        'residual_kl_loss': kl_loss.item(),
        'dyn_loss': dyn_loss.item(),
    }

    return kl_loss, metrics