"""
评估 Stage 1 checkpoint 在 PseudoRealEnv 上的表现
不跑 Stage 2(还没训),只看 Zero-shot 性能
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# isaacgym 必须先 import
from isaacgym import gymapi  # noqa: F401
import torch  # noqa: E402

from envs import PseudoRealEnv
from models.world_model import WorldModel
from models.actor import A1Actor
from evaluation.eval_protocol import evaluate_policy


def infer_wm_params_from_state_dict(wm_sd):
    """
    从 ckpt['world_model'] state_dict 推断 WorldModel 参数。

    关键 keys:
      - rssm._gru.weight_ih: [3*hidden, hidden]  → hidden = shape[1], deter = shape[1]
      - rssm._imgs_stat_layer.weight: [stoch_dim*discrete, hidden] → stoch*discrete = shape[0]
      - encoder.net.0.weight: [hidden, obs_dim] → embed_dim = encoder.net.15.weight.shape[0] (最后一层)
    """
    # deter / hidden
    gru_shape = wm_sd['rssm._gru.weight_ih'].shape
    hidden = gru_shape[1]
    deter_dim = hidden
    # stoch_dim * discrete
    stoch_layer_shape = wm_sd['rssm._imgs_stat_layer.weight'].shape
    stoch_flat = stoch_layer_shape[0]
    # 默认 discrete=32,推 stoch_dim
    discrete = 32
    stoch_dim = stoch_flat // discrete
    # embed_dim: encoder 最后一层 Linear
    embed_keys = [k for k in wm_sd.keys() if k.startswith('encoder.net.') and k.endswith('.weight')]
    embed_dim = wm_sd[embed_keys[-1]].shape[0]
    return deter_dim, stoch_dim, discrete, hidden, embed_dim


def main():
    parser = argparse.ArgumentParser(description='Stage 1 evaluation')
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/stage1_final.ckpt')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='评估 episode 数 (默认 10, 服务器上 50 可能太慢)')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--use_residual', action='store_true',
                        help='是否启用 Residual (Stage 1 时 ≈0,通常 False)')
    parser.add_argument('--env_name', type=str, default='pseudo_real',
                        choices=['pseudo_real', 'inaccurate_sim'],
                        help='评估在哪个 env 上')
    args = parser.parse_args()

    # 加载 Stage 1 checkpoint
    print(f"[Eval] Loading Stage 1 from {args.stage1_ckpt}")
    ckpt = torch.load(args.stage1_ckpt, map_location=args.device)

    wm_sd = ckpt['world_model']
    deter_dim, stoch_dim, discrete, hidden, embed_dim = infer_wm_params_from_state_dict(wm_sd)
    print(f"[Eval] Inferred: deter={deter_dim}, stoch={stoch_dim}, discrete={discrete}, hidden={hidden}, embed={embed_dim}")

    # 创建 WorldModel
    world_model = WorldModel(
        obs_dim=48, action_dim=12,
        deter_dim=deter_dim, stoch_dim=stoch_dim, discrete=discrete,
        hidden=hidden, embed_dim=embed_dim, use_residual=True,
    ).to(args.device)

    missing, unexpected = world_model.load_state_dict(wm_sd, strict=False)
    if missing:
        print(f"[Eval] Missing keys (count={len(missing)}): first 5: {missing[:5]}")
    if unexpected:
        print(f"[Eval] Unexpected keys (count={len(unexpected)}): first 5: {unexpected[:5]}")
    print(f"[Eval] WorldModel loaded (strict=False)")

    # Actor
    actor_sd = ckpt['actor']
    feat_dim = deter_dim + stoch_dim * discrete
    actor = A1Actor(feat_dim=feat_dim, action_dim=12).to(args.device)
    actor.load_state_dict(actor_sd, strict=False)
    print(f"[Eval] Actor loaded (feat_dim={feat_dim})")

    # 环境
    if args.env_name == 'pseudo_real':
        from envs import PseudoRealEnv
        # 加载 amp config (跟 train 时一致)
        amp_config = None
        if os.path.exists('configs/amp.yaml'):
            import yaml as _yaml
            with open('configs/amp.yaml', 'r', encoding='utf-8') as f:
                amp_config = _yaml.safe_load(f).get('amp', None)
        env = PseudoRealEnv(num_envs=args.num_envs, device=args.device, amp_config=amp_config)
    else:
        from envs import InaccurateSimEnv
        amp_config = None
        if os.path.exists('configs/amp.yaml'):
            import yaml as _yaml
            with open('configs/amp.yaml', 'r', encoding='utf-8') as f:
                amp_config = _yaml.safe_load(f).get('amp', None)
        env = InaccurateSimEnv(num_envs=args.num_envs, device=args.device, amp_config=amp_config)
    print(f"[Eval] Environment: {args.env_name}")

    # 评估
    print(f"\n{'='*60}")
    print(f"评估 Stage 1 Zero-shot 在 {args.env_name} 上")
    print(f"{'='*60}")
    metrics = evaluate_policy(
        actor, world_model, env,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        use_residual=args.use_residual,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"📊 评估结果 ({args.env_name}, Zero-shot, use_residual={args.use_residual})")
    print(f"{'='*60}")
    print(f"  num_episodes:     {metrics['num_episodes']}")
    print(f"  mean_return:      {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  mean_ep_length:   {metrics['mean_episode_length']:.1f}")
    print(f"  success_rate:     {metrics['success_rate']:.2f} (return > 200)")
    print(f"  fall_rate:        {metrics['fall_rate']:.2f}")

    # 保存结果
    os.makedirs('results', exist_ok=True)
    path = f'results/eval_stage1_{args.env_name}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'env_name': args.env_name,
            'use_residual': args.use_residual,
            'stage1_ckpt': args.stage1_ckpt,
            'wm_params': {
                'deter_dim': deter_dim,
                'stoch_dim': stoch_dim,
                'discrete': discrete,
                'hidden': hidden,
                'embed_dim': embed_dim,
            },
            'metrics': metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Saved to {path}")

    env.close()


if __name__ == '__main__':
    main()
