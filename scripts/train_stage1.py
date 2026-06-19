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
    parser.add_argument('--amp_config', type=str, default='configs/amp.yaml',
                        help='AMP config (B1: AMPDiscriminator + AMPLoader + Normalizer)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP (use plain task reward)')
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--log_every', type=int, default=1000,
                        help='log every N steps (default 1000; use 100 for short tests)')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 加载 AMP 配置(B1)
    amp_config = None
    if not args.no_amp and args.amp_config is not None:
        import os as _os
        if _os.path.exists(args.amp_config):
            import yaml as _yaml
            with open(args.amp_config, 'r', encoding='utf-8') as f:
                _amp_full = _yaml.safe_load(f)
            amp_config = _amp_full.get('amp', None)
            print(f"[Stage 1] AMP config loaded from {args.amp_config}")
        else:
            print(f"[Stage 1] AMP config file not found: {args.amp_config}, AMP disabled")

    # 创建环境
    print(f"[Stage 1] Creating InaccurateSimEnv with {args.num_envs} envs")
    env = InaccurateSimEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        amp_config=amp_config,
    )

    # 创建 Trainer
    trainer = Trainer(env, config, device=args.device)

    # 训练
    trainer.train_stage1(total_steps=args.total_steps, log_every=args.log_every)


if __name__ == '__main__':
    main()