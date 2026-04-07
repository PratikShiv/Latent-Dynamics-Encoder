"""
PPO Update and utility functions

ppo_udate() supports 3 configuration
- Symmetric : No extra obs for policy or critic
- Asymmetic-priv (teacher) : obs_critic fed to critic only
- Asymmetric-z (student)   : policy_extra_bs (z) fed to both policy.evaluate and critic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto

class PPOMode(Enum):
    SYMMETRIC   = auto()    # Vanilla PPO
    TEACHER     = auto()    # Asymmetric: Privileged obs fed to critic only
    STUDENT     = auto()    # Asymmetric-z: Latent z fed to both actor and critic


class RunningMeanStd:
    """Welford's online algorithm for tracking mean and variance"""

    def __init__(self, shape=(), clip=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1 and self.mean.ndim == 1:
            batch = batch[np.newaxis, :]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2/total
        self.count = total

    def normalize(self, x):
        return np.clip(
            (x - self. mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8),
            -self.clip,
            self.clip,
        )

    def state_dict(self):
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count
        }
    
    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


# --------------------------------------------------------------------------------
"""
    Generalized Advantage Estimation

    Â+t = ∑_{l=0}^{∞} (γλ)^l · δ_{t+l}
    where δ_t = t_t + γ V(s_{t+1} - V(s_t))
"""

def comput_gae(rewards, values, dones, last_values, gamma, lam):
    # GAE for batched rollouts. All arrats have shape [T, N]

    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)

    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns

def ppo_update(policy, value_fn, batch, *,
               mode: PPOMode = PPOMode.SYMMETRIC,
               clip_ratio=0.2, entrpoy_coeff=0.01,
               pi_optimizer=None, vf_optimizer=None,
               update_epochs=10, minibatch_size=4096,
               max_grad_norm=0.5, device="cpu",
               pi_params=None, vf_params=None):
    # Run multiple epochs of PPO update on a collected batch
    
    obs_actor = torch.as_tensor(
        batch.get("obs_actor", batch.get("obs")), dtype=torch.float32, device=device)
    
    if mode == PPOMode.TEACHER:
        obs_priv = torch.as_tensor(batch["obs_critic_priv"], dtype=torch.float32, device=device)
        policy_extra = None
    elif mode == PPOMode.STUDENT:
        z = torch.as_tensor(batch["policy_extra_obs"], dtype=torch.float32, device=device)
        obs_priv = z
        policy_extra = z
    else:
        # Symmetric, vanilla PPO
        obs_priv = None
        policy_extra = None

    # Convert all to tensors
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    old_lp = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)

    # Normalize advantages with unit variance
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Parameters for both networks
    pi_clip_params = list(pi_params) if pi_params is not None else list(policy.parameters())
    vf_clip_params = list(vf_params) if vf_params is not None else list(value_fn.parameters())

    # Initial values
    N = obs_actor.shape[0]
    total_pi_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    total_clipfrac = 0.0
    n_updates = 0.0

    for _epochs in update_epochs:
        indices = torch.randperm(N, device=device)

        for start in range(0, N, minibatch_size):
            idx = indices[start:start + minibatch_size]
            mb_obs = obs_actor[idx]
            mb_act = actions[idx]
            mb_old_lp = old_lp[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]
            mb_priv = obs_priv[idx] if mode != PPOMode.SYMMETRIC else None
            mb_policy_extra = policy_extra[idx] if mode == PPOMode.STUDENT else None

            # Policy Loss
            if mode == PPOMode.STUDENT:
                new_lp, entropy = policy.evaluate(mb_obs, mb_policy_extra, mb_act)
            else:
                new_lp, entropy = policy.evalute(mb_obs, mb_act)

            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()
            pi_loss -= entrpoy_coeff * entropy.mean()

            pi_optimizer.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(pi_clip_params, max_grad_norm)
            pi_optimizer.step()

            # Value Loss
            if mode != PPOMode.SYMMETRIC:
                v_pred = value_fn(mb_obs, mb_priv)
            else:
                v_pred = value_fn(mb_obs)
            vf_loss = F.mse_loss(v_pred, mb_ret)

            vf_optimizer.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(vf_clip_params, max_grad_norm)
            vf_optimizer.step()

            # Total Losses
            total_pi_loss += pi_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.mean().item()

            with torch.no_grad():
                total_clipfrac += (torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()
            n_updates += 1

    return {
        "pi_loss": total_pi_loss / n_updates,
        "vf_loss": total_vf_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "clipfrac": total_clipfrac / n_updates,
    }