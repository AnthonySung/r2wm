#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/r2wm

python3 -c "
import torch
import sys
sys.path.insert(0, 'tests')
from models.world_model import WorldModel
from models.actor import A1Actor
from models.critic import A1Critic, SlowCritic
from training.ac_loss import compute_ac_loss

wm = WorldModel(obs_dim=48, action_dim=12, deter_dim=32, stoch_dim=4, discrete=4, hidden=32, embed_dim=16, use_residual=True)
actor = A1Actor(feat_dim=32+4*4, action_dim=12, hidden=32, n_layers=2)
critic = A1Critic(feat_dim=32+4*4, hidden=32, n_layers=2)
target_critic = SlowCritic(critic, update_fraction=0.02)

init_obs = torch.randn(4, 48)
init_is_first = torch.zeros(4, dtype=torch.bool)
init_action = torch.zeros(4, 12)

# 手动模拟 imagine 来 debug
state = wm.rssm.initial_state(batch_size=4, device='cpu')
embed = wm.encode(init_obs)
state, _ = wm.rssm.obs_step(state, init_action, embed, init_is_first, sample=True)
states, actions, log_probs, entropies = wm.imagine(actor, state, horizon=5, with_residual=False)

print('states.deter shape:', states['deter'].shape)
print('actions shape:', actions.shape)

feat_seq = wm.rssm.get_feat(states)
print('feat_seq shape:', feat_seq.shape)

# 这就是问题所在!
print('feat_seq.reshape(-1, feat_dim) shape:', feat_seq.reshape(-1, feat_seq.shape[-1]).shape)
values_seq = critic(feat_seq.reshape(-1, feat_seq.shape[-1])).reshape(5, -1)
print('values_seq shape:', values_seq.shape, '(应该是 [5, 4]=[horizon, B])')
print('B*horizon =', 4*5)
"