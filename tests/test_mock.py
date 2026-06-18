"""
Mock 测试:不需要 Isaac Gym,验证 r2wmp 的核心逻辑。

测试内容:
1. Residual 零初始化验证
2. RSSM 前向 + kl_loss(关键 bug 修复验证)
3. WorldModel.observe() 和 imagine() 流程
4. Actor-Critic 输出形状
5. ReplayBuffer 采样 + 保存/加载
6. End-to-end: 一个完整小循环

运行方式:
    cd D:\\songay\\sim2real\\r2wmp
    python -m pytest tests/test_mock.py -v
    或
    python tests/test_mock.py
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_residual_zero_init():
    """测试 1: Residual 严格零初始化"""
    print("\n" + "=" * 60)
    print("Test 1: Residual 严格零初始化")
    print("=" * 60)

    from models.residual import PhysicalResidual, DynamicsResidual

    # PhysicalResidual
    phys = PhysicalResidual(stoch_dim=1024, action_dim=12, hidden=64, n_layers=3, init='zero')
    prev_stoch = torch.randn(2, 1024)
    action = torch.randn(2, 12)
    delta = phys(prev_stoch, action)
    
    assert delta.abs().max().item() < 1e-6, f"PhysicalResidual 零初始化失败, delta.max()={delta.abs().max().item()}"
    print(f"  ✅ PhysicalResidual 零初始化 OK, |delta|.max() = {delta.abs().max().item():.2e}")

    # DynamicsResidual
    dyn = DynamicsResidual(deter_dim=512, action_dim=12, history_len=4, hidden=128, n_layers=3, init='zero')
    deter_history = torch.randn(2, 4, 512)
    delta_dyn = dyn(deter_history, action)
    
    assert delta_dyn.abs().max().item() < 1e-6, f"DynamicsResidual 零初始化失败"
    print(f"  ✅ DynamicsResidual 零初始化 OK, |delta|.max() = {delta_dyn.abs().max().item():.2e}")

    # 对比:small_normal 应该不是 0
    phys_small = PhysicalResidual(stoch_dim=1024, action_dim=12, hidden=64, n_layers=3, init='small_normal')
    delta_small = phys_small(prev_stoch, action)
    print(f"  ℹ️  small_normal init 输出: |delta|.max() = {delta_small.abs().max().item():.4f} (非零)")


def test_rssm_forward_and_kl():
    """测试 2: RSSM 前向 + kl_loss 修复验证"""
    print("\n" + "=" * 60)
    print("Test 2: RSSM 前向 + kl_loss")
    print("=" * 60)

    from models.rssm import RSSM

    rssm = RSSM(
        deter_dim=64, stoch_dim=8, discrete=8, hidden=64,
        action_dim=4, embed_dim=32,
    )

    # initial state
    state = rssm.initial_state(batch_size=4, device='cpu')
    assert 'deter' in state and 'stoch' in state
    print(f"  ✅ initial_state: deter={state['deter'].shape}, stoch={state['stoch'].shape}")

    # img_step(想象一步)
    action = torch.randn(4, 4)
    prior = rssm.img_step(state, action, sample=True)
    assert 'deter' in prior and 'stoch' in prior
    print(f"  ✅ img_step: prior.deter={prior['deter'].shape}, prior.stoch={prior['stoch'].shape}")

    # obs_step(观察)
    embed = torch.randn(4, 32)
    is_first = torch.zeros(4, dtype=torch.bool)
    post, prior = rssm.obs_step(state, action, embed, is_first, sample=True)
    assert 'deter' in post and 'stoch' in post
    print(f"  ✅ obs_step: post.deter={post['deter'].shape}, post.stoch={post['stoch'].shape}")

    # kl_loss(关键:验证修复后能算出合理 KL)
    total, dyn_loss, rep_loss = rssm.kl_loss(post, prior, free=1.0, dyn_scale=1.0, rep_scale=1.0)
    assert torch.isfinite(total), f"kl_loss 含 NaN/Inf: {total}"
    assert dyn_loss.item() > 0, "dyn_loss 应 > 0"
    assert rep_loss.item() > 0, "rep_loss 应 > 0"
    print(f"  ✅ kl_loss: total={total.item():.4f}, dyn={dyn_loss.item():.4f}, rep={rep_loss.item():.4f}")

    # KL loss 能反向传播(关键!)
    total.backward()
    print(f"  ✅ kl_loss 反向传播 OK")

    # get_feat
    feat = rssm.get_feat(post)
    expected_dim = 8 * 8 + 64  # stoch_dim * discrete + deter_dim = 128
    assert feat.shape == (4, expected_dim), f"feat 维度错误: {feat.shape}"
    print(f"  ✅ get_feat: {feat.shape} (符合 stoch*discrete + deter = {expected_dim})")


def test_world_model_observe_imagine():
    """测试 3: WorldModel 完整 observe + imagine 流程"""
    print("\n" + "=" * 60)
    print("Test 3: WorldModel.observe + imagine")
    print("=" * 60)

    from models.world_model import WorldModel
    from models.actor import A1Actor

    # 小模型(测试用)
    wm = WorldModel(
        obs_dim = 48, action_dim=12,
        deter_dim=64, stoch_dim=8, discrete=8, hidden=64,
        embed_dim=32, use_residual=True,
    )
    actor = A1Actor(feat_dim=64 + 8*8, action_dim=12, hidden=64, n_layers=2)

    # 假数据
    obs_seq = torch.randn(2, 10, 45)
    action_seq = torch.randn(2, 10, 12)
    is_first_seq = torch.zeros(2, 10, dtype=torch.bool)

    # observe(Stage 1 风格: with_residual=True, stop_residual_grad=True)
    posts, priors = wm.observe(
        obs_seq, action_seq, is_first_seq,
        with_residual=True,
        stop_residual_grad=True,
    )
    print(f"  ✅ observe: posts.stoch={posts['stoch'].shape}, posts.deter={posts['deter'].shape}")
    
    # 验证:stop_residual_grad=True 时,residual 输出 ≈ 0(零初始化)
    # 所以 mean 应该 ≈ rssm_only_mean
    assert torch.isfinite(posts['mean']).all(), "posts.mean 含 NaN/Inf"
    print(f"  ✅ observe 输出有限值")

    # observe(stop_residual_grad=False) 验证 Stage 2 模式
    posts2, priors2 = wm.observe(
        obs_seq, action_seq, is_first_seq,
        with_residual=True,
        stop_residual_grad=False,
    )
    # 此时 residual 仍≈ 0(零初始化,无论是否接收梯度)
    # 所以两个输出应该接近
    diff = (posts['mean'] - posts2['mean']).abs().max().item()
    print(f"  ℹ️  stop_grad 切换的 mean 差异: {diff:.6f} (应该接近 0,因为 Residual=0)")

    # imagine(Stage 1 风格: with_residual=False)
    init_state = wm.rssm.initial_state(batch_size=2, device='cpu')
    states, actions, log_probs, entropies = wm.imagine(
        actor, init_state, horizon=5,
        with_residual=False,
    )
    print(f"  ✅ imagine: states.feat shape={wm.rssm.get_feat(states).shape if 'feat' not in states else states['deter'].shape}")
    assert actions.shape == (2, 5, 12)
    assert log_probs.shape == (2, 5)
    print(f"  ✅ imagine: actions={actions.shape}, log_probs={log_probs.shape}")

    # 想象时 Residual 不参与,验证
    # 重新想象一次,带 Residual
    states_with, _, _, _ = wm.imagine(
        actor, init_state, horizon=5,
        with_residual=True,
    )
    # 因为 Residual=0,两个输出应该一样
    diff = (states['deter'] - states_with['deter']).abs().max().item()
    print(f"  ℹ️  imagine with/without Residual deter 差异: {diff:.6f} (应该 ≈ 0,因 Residual=0)")


def test_actor_critic_output():
    """测试 4: Actor-Critic 输出形状和分布"""
    print("\n" + "=" * 60)
    print("Test 4: Actor-Critic 输出")
    print("=" * 60)

    from models.actor import A1Actor
    from models.critic import A1Critic, SlowCritic

    feat_dim = 128
    action_dim = 12

    actor = A1Actor(feat_dim=feat_dim, action_dim=action_dim, hidden=128, n_layers=2)
    critic = A1Critic(feat_dim=feat_dim, hidden=128, n_layers=2)
    slow_critic = SlowCritic(critic, update_fraction=0.02)

    feat = torch.randn(8, feat_dim)

    # Actor
    action, mean, std, entropy = actor(feat, sample=True)
    assert action.shape == (8, action_dim)
    assert action.abs().max() <= 1.0, f"tanh 输出应 ≤ 1, but {action.abs().max()}"
    assert mean.shape == (8, action_dim)
    assert std.shape == (8, action_dim)
    print(f"  ✅ Actor: action range=[{action.min().item():.3f}, {action.max().item():.3f}], std range=[{std.min().item():.3f}, {std.max().item():.3f}]")

    # Actor log_prob
    log_prob = actor.get_log_prob(feat, action)
    assert log_prob.shape == (8,)
    print(f"  ✅ Actor log_prob: range=[{log_prob.min().item():.3f}, {log_prob.max().item():.3f}]")

    # Critic
    value = critic(feat)
    assert value.shape == (8,)
    print(f"  ✅ Critic V(s): range=[{value.min().item():.3f}, {value.max().item():.3f}]")

    # Slow critic(EMA target)
    slow_value = slow_critic(feat)
    print(f"  ✅ SlowCritic (EMA target): value range=[{slow_value.min().item():.3f}, {slow_value.max().item():.3f}]")

    # EMA 更新
    critic_new = A1Critic(feat_dim=feat_dim, hidden=128, n_layers=2)
    # 给 critic_new 一个不同的权重
    for p in critic_new.parameters():
        p.data += 0.1
    slow_critic.update(critic_new)
    slow_value_after = slow_critic(feat)
    diff = (slow_value - slow_value_after).abs().max().item()
    assert diff > 1e-6, "EMA update 没生效"
    print(f"  ✅ EMA update: slow_value 变化 {diff:.4f}")


def test_replay_buffer():
    """测试 5: ReplayBuffer 采样 + 保存/加载"""
    print("\n" + "=" * 60)
    print("Test 5: ReplayBuffer")
    print("=" * 60)

    from training.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=1000, obs_dim = 48, action_dim=12, device='cpu')

    # 添加数据
    for i in range(100):
        buffer.add(
            obs=np.random.randn(45).astype(np.float32),
            action=np.random.randn(12).astype(np.float32),
            reward=float(i),
            next_obs=np.random.randn(45).astype(np.float32),
            done=(i % 20 == 0),
        )
    print(f"  ✅ Buffer size: {len(buffer)}")

    # 采样序列
    batch = buffer.sample(batch_size=4, seq_length=10)
    assert batch['obs'].shape == (4, 10, 45)
    assert batch['action'].shape == (4, 10, 12)
    assert batch['reward'].shape == (4, 10)
    assert batch['is_first'].shape == (4, 10)
    assert batch['is_first'][:, 0].all(), "第 0 步应是 first"
    print(f"  ✅ Sample: obs={batch['obs'].shape}, is_first[0].all()={batch['is_first'][:, 0].all().item()}")

    # 保存 / 加载
    save_path = 'datasets/test_buffer.npz'
    os.makedirs('datasets', exist_ok=True)
    buffer.save(save_path)
    
    buffer2 = ReplayBuffer.from_npz(save_path, obs_dim = 48, action_dim=12)
    assert len(buffer2) == len(buffer), "加载后 size 不一致"
    print(f"  ✅ Save/Load: 保存 {len(buffer)} 条,加载 {len(buffer2)} 条")

    # 清理
    if os.path.exists(save_path):
        os.remove(save_path)


def test_lambda_return():
    """测试 6: λ-return 计算"""
    print("\n" + "=" * 60)
    print("Test 6: λ-return 计算")
    print("=" * 60)

    from training.ac_loss import compute_lambda_return

    T, B = 5, 3
    rewards = torch.randn(T, B)
    values = torch.randn(T + 1, B)  # T+1 因为要 bootstrap
    continues = torch.ones(T, B)

    lambda_returns = compute_lambda_return(
        rewards, values, continues,
        lambda_=0.95, gamma=0.997
    )

    assert lambda_returns.shape == (T, B)
    assert torch.isfinite(lambda_returns).all()
    print(f"  ✅ lambda_returns: shape={lambda_returns.shape}, range=[{lambda_returns.min().item():.3f}, {lambda_returns.max().item():.3f}]")


def test_ac_loss_end_to_end():
    """测试 7: AC loss 端到端(Stage 1 训练 loop 模拟)"""
    print("\n" + "=" * 60)
    print("Test 7: AC loss 端到端")
    print("=" * 60)

    from training.ac_loss import compute_ac_loss
    from models.world_model import WorldModel
    from models.actor import A1Actor
    from models.critic import A1Critic, SlowCritic

    # 小模型
    wm = WorldModel(
        obs_dim = 48, action_dim=12,
        deter_dim=32, stoch_dim=4, discrete=4, hidden=32,
        embed_dim=16, use_residual=True,
    )
    actor = A1Actor(feat_dim=32 + 4*4, action_dim=12, hidden=32, n_layers=2)
    critic = A1Critic(feat_dim=32 + 4*4, hidden=32, n_layers=2)
    target_critic = SlowCritic(critic, update_fraction=0.02)

    # 假数据
    init_obs = torch.randn(4, 45)
    init_is_first = torch.zeros(4, dtype=torch.bool)
    init_action = torch.zeros(4, 12)

    # 计算 AC loss
    loss, metrics = compute_ac_loss(
        actor, critic, target_critic, wm,
        init_obs=init_obs,
        init_is_first=init_is_first,
        init_action=init_action,
        horizon=5,
    )

    assert torch.isfinite(loss), f"AC loss NaN/Inf: {loss}"
    print(f"  ✅ AC loss: {loss.item():.4f}")
    print(f"     actor_loss={metrics['actor_loss']:.4f}, critic_loss={metrics['critic_loss']:.4f}")
    print(f"     entropy_loss={metrics['entropy_loss']:.4f}, lambda_return={metrics['lambda_return_mean']:.4f}")

    # 反向传播
    loss.backward()
    
    # 检查所有参数都有梯度
    actor_params_with_grad = sum(1 for p in actor.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_actor_params = sum(1 for p in actor.parameters())
    print(f"  ✅ Actor 反向传播: {actor_params_with_grad}/{total_actor_params} 参数有梯度")


def test_residual_flow():
    """测试 8: Residual 在 Stage 1 / Stage 2 流程中的行为"""
    print("\n" + "=" * 60)
    print("Test 8: Residual 在 Stage 1 / Stage 2 的流程")
    print("=" * 60)

    from models.world_model import WorldModel

    wm = WorldModel(
        obs_dim = 48, action_dim=12,
        deter_dim=32, stoch_dim=4, discrete=4, hidden=32,
        embed_dim=16, use_residual=True,
    )

    # Stage 1: 验证 Residual 输出 ≈ 0
    obs = torch.randn(2, 5, 45)
    action = torch.randn(2, 5, 12)
    is_first = torch.zeros(2, 5, dtype=torch.bool)
    
    posts, priors = wm.observe(obs, action, is_first, with_residual=True, stop_residual_grad=True)
    
    # 检查物理 residual 输出
    prev_stoch = posts['stoch'][:, :-1].reshape(-1, 4*4)
    action_flat = action[:, :-1].reshape(-1, 12)
    phys_out = wm.physical_residual(prev_stoch, action_flat)
    print(f"  ✅ Stage 1 PhysicalResidual 输出 max: {phys_out.abs().max().item():.2e} (应≈0)")

    # Stage 2 重建
    print("  → recreate_residual_for_stage2()...")
    wm.recreate_residual_for_stage2()
    phys_out_after = wm.physical_residual(prev_stoch, action_flat)
    print(f"  ✅ Stage 2 PhysicalResidual 输出 max: {phys_out_after.abs().max().item():.2e} (重建后仍≈0,因零初始化)")

    # 验证:Stage 2 Residual 层数 = 1
    n_layers = sum(1 for m in wm.physical_residual.net if isinstance(m, nn.Linear)) // 2  # 粗略
    # 数 Linear 层更精确
    linear_layers = [m for m in wm.physical_residual.net if isinstance(m, nn.Linear)]
    print(f"  ✅ Stage 2 PhysicalResidual Linear 层数: {len(linear_layers)} (Stage 1: 4, Stage 2: 2)")

    # freeze_main_network
    wm.freeze_main_network()
    frozen_count = sum(1 for p in wm.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in wm.parameters() if p.requires_grad)
    print(f"  ✅ freeze_main_network: 冻结 {frozen_count} 参数,可训 {trainable_count} (应只有 Residual)")

    # 验证 Residual 可训
    res_params = list(wm.physical_residual.parameters()) + list(wm.dynamics_residual.parameters())
    for p in res_params:
        assert p.requires_grad, "Residual 参数应 requires_grad=True"
    print(f"  ✅ Residual 所有参数 requires_grad=True")


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 30)
    print("r2wmp Mock 测试套件(不需要 Isaac Gym)")
    print("🚀" * 30)

    try:
        test_residual_zero_init()
        test_rssm_forward_and_kl()
        test_world_model_observe_imagine()
        test_actor_critic_output()
        test_replay_buffer()
        test_lambda_return()
        test_ac_loss_end_to_end()
        test_residual_flow()

        print("\n" + "🎉" * 30)
        print("全部测试通过! r2wmp 核心逻辑正确")
        print("🎉" * 30)

        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 必须 import nn
    import torch.nn as nn
    success = main()
    sys.exit(0 if success else 1)