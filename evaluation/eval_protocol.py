"""
评估协议:三策略对比
"""

import os
import json
import numpy as np
import torch
from typing import Optional


def evaluate_policy(
    actor,
    world_model,
    env,
    num_episodes: int = 50,
    max_episode_steps: int = 1000,
    use_residual: bool = False,
    device: str = 'cuda:0',
) -> dict:
    """
    评估单个策略在环境上的表现。

    Args:
        actor: A1Actor
        world_model: WorldModel
        env: 环境
        num_episodes: episode 数量
        max_episode_steps: 每个 episode 最大步数
        use_residual: 是否使用 Residual
    Returns:
        metrics: dict with mean_return, success_rate, fall_rate
    """
    actor.eval()
    world_model.eval()

    episode_returns = []
    episode_lengths = []
    success_count = 0
    fall_count = 0

    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)

        # 处理 num_envs > 1 的情况:只用第一个
        if obs.dim() > 1:
            obs = obs[0]

        ep_return = 0.0
        ep_length = 0
        fallen = False

        with torch.no_grad():
            state = world_model.rssm.initial_state(batch_size=1, device=device)

            for t in range(max_episode_steps):
                ep_length += 1

                # 编码
                embed = world_model.encode(obs.unsqueeze(0))
                state, _ = world_model.rssm.obs_step(
                    state,
                    torch.zeros(1, env.action_dim, device=device),
                    embed,
                    torch.zeros(1, dtype=torch.bool, device=device),
                    sample=True,
                )

                # 应用 Residual(可选)
                if use_residual and world_model._use_residual:
                    prev_stoch = state['stoch']
                    if world_model.rssm._is_discrete:
                        prev_stoch_flat = prev_stoch.reshape(*prev_stoch.shape[:-2], world_model._stoch_flat_dim)
                    else:
                        prev_stoch_flat = prev_stoch
                    action_tensor = torch.zeros(1, env.action_dim, device=device)  # 上一步 action
                    delta = world_model.physical_residual(prev_stoch_flat, action_tensor)
                    state['mean'] = state['mean'] + delta

                # Actor
                feat = world_model.rssm.get_feat(state)
                action = actor.sample(feat)

                # Env step
                next_obs, reward, done, info = env.step(action.squeeze(0))

                if isinstance(next_obs, np.ndarray):
                    next_obs = torch.from_numpy(next_obs).float().to(device)

                if next_obs.dim() > 1:
                    next_obs = next_obs[0]
                    reward = reward[0] if torch.is_tensor(reward) else reward
                    done = done[0] if torch.is_tensor(done) else done

                ep_return += float(reward) if torch.is_tensor(reward) else reward

                obs = next_obs

                if bool(done) if torch.is_tensor(done) else done:
                    if t < max_episode_steps - 1:
                        fallen = True
                    break

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        if ep_return > 200:  # success threshold(可调)
            success_count += 1
        if fallen:
            fall_count += 1

    metrics = {
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'success_rate': success_count / num_episodes,
        'fall_rate': fall_count / num_episodes,
        'mean_episode_length': float(np.mean(episode_lengths)),
        'num_episodes': num_episodes,
    }

    return metrics


def three_policy_comparison(
    actor,
    world_model,
    pseudo_real_env,
    inaccurate_sim_env,
    num_episodes: int = 50,
    device: str = 'cuda:0',
) -> dict:
    """
    三策略对比:
    - A: Zero-shot(Stage 1,Residual≈0)
    - B: Stage 1 + Stage 2 Residual
    - C: Upper bound(stage 2 全量微调,可选)

    Returns:
        summary: dict with all metrics
    """
    print("=" * 60)
    print("三策略对比评估")
    print("=" * 60)

    # 策略 A: Zero-shot(Residual 不启用,因为 Stage 1 训的 Residual≈0)
    print("\n[策略 A] Zero-shot(Stage 1 模型,Residual≈0)")
    metrics_A = evaluate_policy(
        actor, world_model, pseudo_real_env,
        num_episodes=num_episodes, use_residual=False, device=device,
    )
    print(f"  mean_return: {metrics_A['mean_return']:.1f} ± {metrics_A['std_return']:.1f}")
    print(f"  success_rate: {metrics_A['success_rate']:.2f}")
    print(f"  fall_rate: {metrics_A['fall_rate']:.2f}")

    # 策略 B: Stage 1 + Stage 2 Residual
    print("\n[策略 B] Stage 1 + Stage 2 Residual")
    metrics_B = evaluate_policy(
        actor, world_model, pseudo_real_env,
        num_episodes=num_episodes, use_residual=True, device=device,
    )
    print(f"  mean_return: {metrics_B['mean_return']:.1f} ± {metrics_B['std_return']:.1f}")
    print(f"  success_rate: {metrics_B['success_rate']:.2f}")
    print(f"  fall_rate: {metrics_B['fall_rate']:.2f}")

    # Sanity check: 在 inaccurate sim 上评估 A
    print("\n[Sanity] Sim 内部表现(策略 A 在 InaccurateSim 上)")
    metrics_A_sim = evaluate_policy(
        actor, world_model, inaccurate_sim_env,
        num_episodes=20, use_residual=False, device=device,
    )
    print(f"  mean_return: {metrics_A_sim['mean_return']:.1f}")

    # 计算 sim-to-real gap 闭合率
    sim_return = metrics_A_sim['mean_return']
    A_real_return = metrics_A['mean_return']
    B_real_return = metrics_B['mean_return']

    gap_before = sim_return - A_real_return
    gap_after = sim_return - B_real_return
    gap_closed_pct = (gap_before - gap_after) / max(abs(gap_before), 1) * 100

    summary = {
        'A_zeroshot_pseudo_real': metrics_A,
        'B_residual_pseudo_real': metrics_B,
        'A_zeroshot_sim': metrics_A_sim,
        'sim_performance': sim_return,
        'zeroshot_real_performance': A_real_return,
        'finetuned_real_performance': B_real_return,
        'gap_before': gap_before,
        'gap_after': gap_after,
        'gap_closed_pct': gap_closed_pct,
    }

    print("\n" + "=" * 60)
    print("📊 最终结果汇总")
    print("=" * 60)
    print(f"Sim 内部表现:               {sim_return:.1f}")
    print(f"A. Zero-shot → 伪 real:    {A_real_return:.1f} (gap = {gap_before:.1f})")
    print(f"B. Residual → 伪 real:     {B_real_return:.1f} (gap = {gap_after:.1f})")
    print(f"Gap 闭合率:                 {gap_closed_pct:.1f}%")
    print()
    if gap_closed_pct > 70:
        print("✅ ReDRAW 算法有效:成功闭合了 70% 以上的 sim-to-real gap")
    elif gap_closed_pct > 30:
        print("⚠️ 部分有效:Residual 学到了一些 gap,但还有改进空间")
    else:
        print("❌ 效果不佳:Residual 未学到有效的 sim-to-real gap 补偿")

    return summary


def save_results(summary: dict, output_dir: str = 'results'):
    """保存结果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'comparison.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Results] Saved to {path}")
    return path