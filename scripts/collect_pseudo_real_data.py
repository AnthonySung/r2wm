"""
采集伪 real 数据:用 Stage 1 训练的 Actor 在 PseudoRealEnv 上跑,收集 transitions
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import PseudoRealEnv
from training.replay_buffer import ReplayBuffer
from models.world_model import WorldModel
from models.actor import A1Actor


def main():
    parser = argparse.ArgumentParser(description='Collect pseudo real data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stage1_final.ckpt')
    parser.add_argument('--output', type=str, default='datasets/pseudo_real_data.npz')
    parser.add_argument('--num_episodes', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print(f"[Collect] Loading checkpoint from {args.checkpoint}")

    # 加载模型
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    world_model = WorldModel(action_dim=12).to(args.device)
    world_model.load_state_dict(ckpt['world_model'])
    actor = A1Actor(feat_dim=world_model.feat_dim, action_dim=12).to(args.device)
    actor.load_state_dict(ckpt['actor'])
    world_model.eval()
    actor.eval()

    # 环境
    env = PseudoRealEnv(num_envs=1, device=args.device)
    print(f"[Collect] Created PseudoRealEnv (num_envs=1, obs_dim={env.obs_dim})")

    # Buffer
    buffer = ReplayBuffer(
        capacity=args.num_episodes * args.max_steps,
        obs_dim=env.obs_dim,
    )

    # 收集数据
    total_steps = 0
    for ep in range(args.num_episodes):
        obs = env.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(args.device)
        if obs.dim() > 1:
            obs = obs[0]

        ep_return = 0.0
        is_first = True  # 每个 episode 第一帧

        with torch.no_grad():
            state = world_model.rssm.initial_state(batch_size=1, device=args.device)
            is_first_env = torch.ones(1, dtype=torch.bool, device=args.device)
            last_actions = torch.zeros(1, env.action_dim, device=args.device)
            history_len = world_model._residual_config.get('history_len', 4)
            deter_history = torch.zeros(
                1, history_len, world_model._deter_dim, device=args.device
            )

            for t in range(args.max_steps):
                # 编码 + 动作
                embed = world_model.encode(obs.unsqueeze(0))
                state, _ = world_model.rssm.obs_step(
                    state, last_actions, embed, is_first_env, sample=True,
                )
                feat = world_model.rssm.get_feat(state)
                action = actor.sample(feat).squeeze(0)

                # Env step
                next_obs, reward, done, _ = env.step(action)

                # 存储
                buffer.add(
                    obs.cpu().numpy(),
                    action.cpu().numpy(),
                    float(reward) if torch.is_tensor(reward) else reward,
                    next_obs.cpu().numpy() if torch.is_tensor(next_obs) else next_obs,
                    bool(done) if torch.is_tensor(done) else done,
                    is_first=is_first,
                )
                is_first = False
                ep_return += float(reward) if torch.is_tensor(reward) else reward
                total_steps += 1

                # 更新 obs
                if torch.is_tensor(next_obs):
                    obs = next_obs[0] if next_obs.dim() > 1 else next_obs
                else:
                    obs = torch.from_numpy(next_obs).float().to(args.device)
                    if obs.dim() > 1:
                        obs = obs[0]

                # 更新 history 和 last_actions
                deter_history = torch.cat([
                    deter_history[:, 1:, :],
                    state['deter'].unsqueeze(1)
                ], dim=1)
                last_actions = action.detach()

                # done 时下一帧是 first
                is_first_env = torch.zeros(1, dtype=torch.bool, device=args.device)
                if done:
                    is_first_env = torch.ones(1, dtype=torch.bool, device=args.device)
                    is_first = True
                    break

        print(f"[Collect] Episode {ep + 1}/{args.num_episodes}: return={ep_return:.1f}, steps={t+1}")

    # 保存
    buffer.save(args.output)
    print(f"[Collect] Total steps: {total_steps}")
    print(f"[Collect] Saved to {args.output}")


if __name__ == '__main__':
    main()