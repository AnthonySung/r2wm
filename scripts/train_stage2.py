"""
Stage 2 训练脚本:在伪 real 数据上微调 Residual
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymapi  # noqa: F401
import torch  # noqa: E402

from envs import InaccurateSimEnv
from training.trainer import Trainer
from training.replay_buffer import ReplayBuffer


def load_config(path: str = 'configs/train.yaml') -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Stage 2 residual fine-tuning')
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/stage1_final.ckpt')
    parser.add_argument('--real_data', type=str, default='datasets/pseudo_real_data.npz')
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=4096)  # 仅用于创建 env 实例
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # 配置
    config = load_config(args.config)

    # 加载 Stage 1
    print(f"[Stage 2] Loading Stage 1 from {args.stage1_ckpt}")
    env = InaccurateSimEnv(num_envs=args.num_envs, device=args.device, headless=True)
    trainer = Trainer(env, config, device=args.device)
    trainer.load_checkpoint(args.stage1_ckpt)

    # 加载伪 real 数据
    print(f"[Stage 2] Loading real data from {args.real_data}")
    real_replay = ReplayBuffer.from_npz(
        args.real_data,
        obs_dim = 48,
        action_dim=12,
        device=args.device,
    )
    print(f"[Stage 2] Real data size: {len(real_replay)}")

    # 训练 Residual
    trainer.train_stage2_residual(
        real_replay=real_replay,
        total_steps=args.total_steps,
    )


if __name__ == '__main__':
    main()