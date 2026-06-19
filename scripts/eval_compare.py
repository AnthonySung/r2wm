"""
三策略对比评估脚本
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymapi  # noqa: F401
import torch  # noqa: E402

from envs import InaccurateSimEnv, PseudoRealEnv
from models.world_model import WorldModel
from models.actor import A1Actor
from evaluation.eval_protocol import three_policy_comparison, save_results


def main():
    parser = argparse.ArgumentParser(description='Three-policy comparison evaluation')
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/stage1_final.ckpt')
    parser.add_argument('--stage2_residual', type=str, default='checkpoints/stage2_final.ckpt')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # 加载 Stage 1 模型
    print(f"[Eval] Loading Stage 1 from {args.stage1_ckpt}")
    ckpt1 = torch.load(args.stage1_ckpt, map_location=args.device)
    world_model = WorldModel(obs_dim = 48, action_dim=12).to(args.device)
    world_model.load_state_dict(ckpt1['world_model'])
    actor = A1Actor(feat_dim=world_model.feat_dim, action_dim=12).to(args.device)
    actor.load_state_dict(ckpt1['actor'])

    # 加载 Stage 2 Residual
    print(f"[Eval] Loading Stage 2 Residual from {args.stage2_residual}")
    if os.path.exists(args.stage2_residual):
        ckpt2 = torch.load(args.stage2_residual, map_location=args.device)
        world_model.physical_residual.load_state_dict(ckpt2['physical_residual'])
        world_model.dynamics_residual.load_state_dict(ckpt2['dynamics_residual'])
        print(f"[Eval] Residual loaded")
    else:
        print(f"[Eval] Warning: Stage 2 residual not found, skip")

    # 环境
    pseudo_real_env = PseudoRealEnv(num_envs=args.num_envs, device=args.device)
    inaccurate_sim_env = InaccurateSimEnv(num_envs=args.num_envs, device=args.device)

    # 三策略对比
    summary = three_policy_comparison(
        actor, world_model,
        pseudo_real_env, inaccurate_sim_env,
        num_episodes=args.num_episodes,
        device=args.device,
    )

    # 保存
    save_results(summary)


if __name__ == '__main__':
    main()