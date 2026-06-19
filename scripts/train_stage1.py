"""
Stage 1 训练脚本:在 InaccurateSimEnv 上训练 WM + AC
"""

import os
import sys
import argparse
import yaml

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 关键:isaacgym 必须先 import(在 torch 之前)
from isaacgym import gymapi  # noqa: F401

import torch  # noqa: E402

from envs import InaccurateSimEnv  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def load_config(path: str = 'configs/train.yaml') -> dict:
    """加载训练配置"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Stage 1 training on InaccurateSim')
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--headless', action='store_true', default=True)
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 创建环境
    print(f"[Stage 1] Creating InaccurateSimEnv with {args.num_envs} envs")
    env = InaccurateSimEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
    )

    # 创建 Trainer
    trainer = Trainer(env, config, device=args.device)

    # 训练
    trainer.train_stage1(total_steps=args.total_steps)


if __name__ == '__main__':
    main()