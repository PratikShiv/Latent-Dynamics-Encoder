"""
Train the student polict with adaptation encoder

Two stages:
    Stage 1 - Distillation (supervised):
        Freeze the teacher. Roll out the teacher in randomized envs.
        Train encoder + student jointly so the student's mean action matches the teacher's mean action
        loss = MSE(student_action, teacher_action) * lambda * ||z||^2

    Stage 2:
        Run PPO with PPOMode.STUDENT. The encoder produces z from history, the student acts on
        (obs, z), a fresh critic evaluates (obs,z). Closes the approximation gap

Usage:
    python -m training.train_student                                # both stages
    python -m training.train_student --stage distill                # Stage 1 only
    python -m training.train_student --stage finetune               # Stage 2 only        
"""

import argparse
import sys
import time
from collections import deque
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from envs.ant_environment import VelocityAntEnv
from envs.dynamics_config import DynamicsConfig
from models.encoder import AdaptationEncoder
from models.student import StudentPolicy
from models.teacher_model import TeacherPolicy, TeacherValueFn
from training.ppo import RunningMeanStd, comput_gae, ppo_update, PPOMode

# -------------------------------------------------------------------------------
# WandB
# -------------------------------------------------------------------------------

wandb = None
def _init_wandb(run_name="student-training"):
    global wandb
    import wandb as _wandb
    wandb = _wandb
    wandb.init(project="Latent-Dynamics-Encoder", name=run_name)
    wandb.define_metric("iteration")
    wandb.define_metric("*", step_metric="iteration")
    return wandb


# -------------------------------------------------------------------------------
# Config Loading
# -------------------------------------------------------------------------------

def load_config(path="training/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
    

# -------------------------------------------------------------------------------
# Privileged regression head + target normalization

# Without a direct supervisory signal, the encoder has no reason to encode
# dynamics: the teacher's action depends only on 'obs', so the student can match
# it by ignoring z. We add a small head that decodes z back to the privileged
# dynamics vector and train the encoder to *also* satisfy that regression.
# This gives z a concrete, dynamics-specific learning target
# -------------------------------------------------------------------------------

class PrivilegedHead(nn.Module):
    # Linear decoder from latent z -> privileged dynamics vector

    def __init__(self, latent_dim, privileged_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, privileged_dim)

    def forward(self, z):
        return self.linear(z)
    
def make_priv_scale(dyn_cfg: DynamicsConfig) -> np.ndarray:
    """
    Per-dimension scale for the 7-dim privileged_obs vector. Dividing the target
    by these brings each component into roughly unit range, so the MSE does not
    get dominated by whichever component happens to have the largest natural magnitude.

    Order matches VelocityAntEnv._get_privileged_obs():
        [friction_scale, mass_scale, action_delay_norm, obs_delay_norm,
        force_x, force_y, force_z]
    """

    fric_max = max(abs(dyn_cfg.friction_range[0]),
                   abs(dyn_cfg.friction_range[1]), 1e-3)
    mass_max = max(abs(dyn_cfg.mass_scale_range[0]),
                   abs(dyn_cfg.mass_scale_range[1]), 1e-3)
    force_max = max(dyn_cfg.external_force_range[1], 1e-3)

    return np.array([
        fric_max,
        mass_max,
        1.0,        # Action delay norm is already in [0,1]
        1.0,        # obs_delay_norm is already in [0,1]
        force_max,
        force_max,
        max(force_max * 0.5, 1e-3),
    ], dtype=np.float32)


# -------------------------------------------------------------------------------
# Env Factory (Same as train_teacher.py)
# -------------------------------------------------------------------------------

def make_env(dynamics_config, cmd_vx_range, cmd_vy_range, cmd_yaw_rate_range,
             max_episode_steps, randomization_seed, render_mode=None):
    print(f"cmd_vx_range: {cmd_vx_range}\n")
    print(f"cmd_vy_range: {cmd_vy_range}\n")
    print("--------------------------------------------------------------------------------")
    return VelocityAntEnv(
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        cmd_vx_range=cmd_vx_range,
        cmd_vy_range=cmd_vy_range,
        cmd_yaw_rate_range=cmd_yaw_rate_range,
        dynamics_config=dynamics_config,
        randomization_seed=randomization_seed
    )

def build_vec_env(e_cfg, dyn_config, seed, num_envs, render_mode=None):
    # Create AsyncVectorEnv with the given condif
    env_fns = [
        partial(
            make_env,
            dynamics_config=dyn_config,
            cmd_vx_range=tuple(e_cfg["cmd_vx_range"]),
            cmd_vy_range=tuple(e_cfg["cmd_vy_range"]),
            cmd_yaw_rate_range=tuple(e_cfg["cmd_yaw_rate_range"]),
            max_episode_steps=e_cfg["max_episode_steps"],
            randomization_seed = seed + i,
            render_mode=render_mode,
        )
        for i in range(num_envs)
    ]

    return gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=False,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


# -------------------------------------------------------------------------------
# Teacher checkpoint loading
# -------------------------------------------------------------------------------

def load_teacher(checkpoint_path, device="cpu"):
    """
    Load the frozen teacher policy and it's observation normalizer from Phase 1 checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]

    # Reconstruct the teacher and load weights
    cfg = ckpt.get("config", {})
    hidden = tuple(cfg.get("teacher", {}).get("hidden_size", [256, 256]))
    teacher = TeacherPolicy(obs_dim, act_dim, hidden_sizes=hidden).to(device)
    teacher.load_state_dict(ckpt["policy_state"])

    # Freeze - Not gradients will ever touch the teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Load the observation normalizer and freeze it
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.load_state_dict(ckpt["obs_rms"])

    return teacher, obs_rms, obs_dim, act_dim


# -------------------------------------------------------------------------------
# History Buffer
# -------------------------------------------------------------------------------

class HistoryBuffer:
    """
    Per environment ring buffer that stores the last H pairs.
    When an episode resets, the buffer clears the stale history from the prev. env.
    """

    def __init__(self, num_envs, history_length, obs_dim, act_dim):
        self.H = history_length
        self.obs_dim = obs_dim
        self.actdim = act_dim
        self.pair_dim = obs_dim + act_dim
        self.num_envs = num_envs

        # One deque per env., each holding up to H pairs
        self.buffers = [deque(maxlen=history_length) for _ in range(num_envs)]

    def append(self, obs, actions):
        for i in range(self.num_envs):
            pair = np.concatenate([obs[i], actions[i]]).astype(np.float32)
            self.buffers[i].append(pair)

    def reset_env(self, env_idx):
        self.buffers[env_idx].clear()

    def is_ready(self, env_idx):
        # Checks if this env has collected H timesteps
        return len(self.buffers[env_idx]) >= self.H
    
    def get_flat(self, env_idx):
        """
        Return the flattened history for one env: (H * pair_dim,)

        The oldest timestep comes first. If the bugger has exactly H entries:
        [path_{t-H+1}, path_{t-H+2}, ..., path_{t}]
        """
        buf = self.buffers[env_idx]
        stacked = np.array(list(buf), dtype=np.float32) # (H, pair_dim)
        return stacked.reshape(-1)                      # (H * pair_dim,)
    
    def get_flat_batch(self):
        # Return flattened histories for all envs
        out = np.zeros((self.num_envs, self.H * self.pair_dim), dtype=np.float32)
        for i in range(self.num_envs):
            if self.is_ready(i):
                out[i] = self.get_flat(i)
        return out
    
    
# -------------------------------------------------------------------------------
# STAGE 1: DISTILLATION
# -------------------------------------------------------------------------------

def distill(cfg, teacher, obs_rms, encoder, student, env, dyn_config, device="cpu"):
    """
    Supervised imitation + Privileged Regression

    At each timestep:
     1. Teacher sees normalized obs -> mean action (frozen target).
     2. Encoder sees history of last H (obs, action) pairs -> z.
     3. Student sees (obs, z) -> mean action.
     4. PrivilegedHead sees z -> predicted privileged dynamics.
     5. Loss = imit_MSE(student_act, teacher_act)
                + priv_coeff * priv_MSE(priv_pred, priv_target)
                + z_reg * ||z||^2
    
    The privileged regression term is what forces z to encode dynamics.
    """

    d_cfg = cfg["distillation"]
    enc_cfg = cfg["encoder"]

    num_envs = env.num_envs
    obs_dim = encoder.obs_dim
    act_dim = encoder.act_dim
    H = enc_cfg["history_length"]
    z_reg = d_cfg.get("z_reg_coeff", 1e-4)
    priv_coeff = d_cfg.get("priv_loss_coeff", 1.0)
    iterations = d_cfg["iterations"]
    batch_size = d_cfg["batch_size"]
    inner_epochs = d_cfg["inner_epochs"]

    privileged_dim = dyn_config.privileged_dim
    priv_head = PrivilegedHead(encoder.latent_dim, privileged_dim).to(device)
    priv_scale_np = make_priv_scale(dyn_config)
    priv_scale_t = torch.as_tensor(priv_scale_np, dtype=torch.float32, device=device)

    # Single Optimizer for both encoder and student parameters
    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(student.parameters())
        + list(priv_head.parameters()),
        lr=d_cfg["lr"],
    )

    # History buffer: one right buffer per enc
    hist_buf = HistoryBuffer(num_envs, H, obs_dim, act_dim)

    print(f"\n{'='*60}")
    print(f"  DISTILLATION - {iterations} iterations")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    print(f"  Student: {sum(p.numel() for p in student.parameters())} params")
    print(f"  PrivHead: {sum(p.numel() for p in priv_head.parameters())} params "
          f"(latent_dim={encoder.latent_dim} -> priv_dim={privileged_dim})")
    print(f"  History Length: {H}, Latent dim: {encoder.latent_dim}")
    print(f"  priv_coeff: {priv_coeff}  z_reg_coeff: {z_reg}")
    print(f"  priv_scale: {priv_scale_np.tolist()}")
    print(f"{'='*60}\n")

    # Rollout + Train Loop
    obs_raw, infos = env.reset()

    # Per-env privileged dynamics, refreshed on every reset. We track this
    # ourselves because the vectorized step() info on done steps may report the
    # NEW episode's dynamics, which would mislabel the LAST step of the old.
    priv_obs_per_env = np.array(
        infos["privileged_obs"], dtype=np.float32
    ).reshape(num_envs, privileged_dim).copy()
    
    total_steps = 0

    for it in range(1, iterations + 1):
        t0 = time.time()

        obs_buf = []
        hist_flat_buf = []
        teacher_act_buf = []
        priv_obs_buf = []

        steps_this_iter = 0

        while steps_this_iter < batch_size:
            # Normalize observations using teacher's frozen normalizer
            obs_norm = obs_rms.normalize(obs_raw).astype(np.float32)
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)

            # Teacher produces the target action (Deterministic mean, no gradient)
            with torch.no_grad():
                teacher_action = teacher.get_mean(obs_t)
            teacher_action_np = teacher_action.cpu().numpy()

            # Clip to action space bounds before stepping
            action_clipped = np.clip(
                teacher_action_np,
                env.single_action_space.low,
                env.single_action_space.high,
            )

            # Step the environment with the teacher's actions
            next_obs_raw, _reward, terminated, truncated, infos = env.step(action_clipped)
            done = np.logical_or(terminated, truncated)

            # Store (obs, action) into per-env history buffers
            hist_buf.append(obs_norm, teacher_action_np)

            # Collect training data from encs that have a full history window
            for i in range(num_envs):
                if hist_buf.is_ready(i):
                    obs_buf.append(obs_norm[i])
                    hist_flat_buf.append(hist_buf.get_flat(i))
                    teacher_act_buf.append(teacher_action_np[i])
                    priv_obs_buf.append(priv_obs_per_env[i].copy())


            # Episode Boundaries:
            done_idx = np.nonzero(done)[0]
            if done_idx.size > 0:
                next_priv = np.asarray(
                    infos["privileged_obs"], dtype=np.float32
                ).reshape(num_envs, privileged_dim)

                for idx in done_idx:
                    hist_buf.reset_env(idx)
                    priv_obs_per_env[idx] = next_priv[idx]

            obs_raw = next_obs_raw
            steps_this_iter += num_envs

        total_steps += steps_this_iter

        # Gradient Step
        # Stack collected samples into tensor
        obs_batch = torch.as_tensor(
            np.array(obs_buf), dtype=torch.float32, device=device)
        hist_batch = torch.as_tensor(
            np.array(hist_flat_buf), dtype=torch.float32, device=device)
        teacher_act_batch = torch.as_tensor(
            np.array(teacher_act_buf), dtype=torch.float32, device=device)
        
        # Forward: Encoder produces z from hsitory
        priv_obs_batch = torch.as_tensor(
            np.array(priv_obs_buf), dtype=torch.float32, device=device)
        
        for _ in range(inner_epochs):
            z = encoder(hist_batch)                 # (B, latent_dim)

            # Forward: Student produces its action prediction from (obs, z)
            student_action = student.get_mean(obs_batch, z)
            priv_pred = priv_head(z)
            priv_target_norm = priv_obs_batch / priv_scale_t
            priv_pred_norm = priv_pred / priv_scale_t

            # Loss: How far the student's actions are from teh teachers'
            # + L2 penalty on z to keep the latent bounded
            imitation_loss = F.mse_loss(student_action, teacher_act_batch)
            priv_loss = F.mse_loss(priv_pred_norm, priv_target_norm)
            z_reg_loss = z_reg * (z ** 2).mean()
            loss = imitation_loss + priv_coeff * priv_loss + z_reg_loss

            # Backprop through encoder + student jointly (teacher is frozen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dt = time.time() - t0
        
        
        # Logging:
        if it % 10 == 0 or it == 1:
            z_norm = z.detach().norm(dim=-1).mean().item()
            
            # Per dimension R^2 on this batch.
            with torch.no_grad():
                tgt = priv_obs_batch
                pred = priv_pred
                ss_res = ((tgt - pred) ** 2).mean(dim=0)
                ss_tot = ((tgt - tgt.mean(dim=0, keepdim=True)) ** 2).mean(dim=0) + 1e-8
                r2 = (1.0 - ss_res / ss_tot).cpu().numpy()

            print(
                f"[Distill {it:>5d}/{iterations}]  "
                f"imit_loss={imitation_loss.item():.5f}  "
                f"z_reg={z_reg_loss.item():.5f}  "
                f"z_norm={z_norm:.3f}  "
                f"R2[fric={r2[0]:+.2f} mass={r2[1]:+.2f} aDly={r2[2]:+.2f} oDly={r2[3]:+.2f}] "
                f"samples={len(obs_buf)}  "
                f"time={dt:.1f}s"
            )

        if wandb is not None:
            wandb.log({
                "iteration": it,
                "distill/imitation_loss": imitation_loss.item(),
                "distill/z_reg_loss": z_reg_loss.item(),
                "distill/priv_loss": priv_loss.item(),
                "distill/z_norm": z.detach().norm(dim=-1).mean().item(),
                "distill/total_loss": loss.item(),
                "distill/num_samples": len(obs_buf),
                "distill/total_env_steps": total_steps,
                "distill/r2_fric": r2[0],
                "distill/r2_mass": r2[1],
                "distill/r2_aDly": r2[2],
                "distill/r2_aDlx": r2[3],
            })

    return encoder, student, priv_head, priv_scale_np


    
# -------------------------------------------------------------------------------
# STAGE 2: RL FINE TUNING
# -------------------------------------------------------------------------------

def collect_student_rollouts(env, student, encoder, value_fn, batch_size,
                             gamma, lam, device, obs_rms, history_length):
    """
    Roll out the student policy in randomized envs, collecting
    (obs, z, action, reward, done) tuples for PPO.

    At each step:
        1. Encoder reads the history buffer -> z
        2. Student acts on (obs, z)
        3. Critic evaluates (obs, z)
        4. History buffer is updated and cleared on episode boundaries
    """

    num_envs = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    # Storage buffer for PPO batch
    obs_buf, z_buf, action_buf = [], [], []
    rewards_buf, dones_buf, log_probs_buf, values_buf = [], [], [], []

    # Episode statistics
    ep_returns, ep_lengths, ep_survivals = [], [], []
    ep_velocities_x, ep_energies = [], []

    # Per-env history ring buggers
    hist_buf = HistoryBuffer(num_envs, history_length, obs_dim, act_dim)

    obs_raw, _infos = env.reset()
    ep_ret = np.zeros(num_envs, dtype=np.float64)
    ep_len = np.zeros(num_envs, dtype=np.int32)
    ep_vx = [[] for _ in range(num_envs)]
    ep_eng = [[] for _ in range(num_envs)]

    total_steps = 0

    while total_steps < batch_size:
        obs_norm = obs_rms.normalize(obs_raw).astype(np.float32)
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)

        # Encoder: history -> z  (No gradient during rollout collection)
        hist_flat = hist_buf.get_flat_batch()
        hist_t = torch.as_tensor(hist_flat, dtype=torch.float32, device=device)

        with torch.no_grad():
            z_t = encoder(hist_t)                   # (num_envs, latent_dim)
            action, lp = student.act(obs_t, z_t)    # Sample Action
            val = value_fn(obs_t, z_t)              # Critic Value

        action_np = action.cpu().numpy()
        z_np = z_t.cpu().numpy()
        action_clipped = np.clip(action_np,
                                 env.single_action_space.low,
                                 env.single_action_space.high)
        
        # Step Environment
        next_obs_raw, reward, terminated, truncated, infos = env.step(action_clipped)
        done = np.logical_or(terminated, truncated)

        # Store (obs, action) into history buffer after stepping
        hist_buf.append(obs_norm, action_np)

        # Store rollout data
        obs_buf.append(obs_norm.copy())
        z_buf.append(z_np.copy())
        action_buf.append(action_np.copy())
        rewards_buf.append(reward.astype(np.float32))
        dones_buf.append(done.astype(np.float32))
        log_probs_buf.append(lp.cpu().numpy().astype(np.float32))
        values_buf.append(val.cpu().numpy().astype(np.float32))

        ep_ret += reward
        ep_len += 1
        total_steps += num_envs

        # Per-env episode tracking
        for i in range(num_envs):
            if "body_vx" in infos:
                ep_vx[i].append(float(np.asarray(infos["body_vx"]).ravel()[i]))
            if "energy" in infos:
                ep_eng[i].append(float(np.asarray(infos["energy"]).ravel()[i]))

        # Handle episode boundaries
        for idx in np.nonzero(done)[0]:
            ep_returns.append(float(ep_ret[idx]))
            ep_lengths.append(float(ep_len[idx]))
            ep_velocities_x.append(float(np.mean(ep_vx[idx])) if ep_vx[idx] else 0.0)
            ep_energies.append(float(np.mean(ep_eng[idx])) if ep_eng[idx] else 0.0)
            survived = not terminated[idx] if hasattr(terminated, '__getitem__') else not terminated
            ep_survivals.append(float(survived))

            # Reset per-env accumulators
            ep_ret[idx] = 0.0
            ep_len[idx] = 0
            ep_vx[idx] = []
            ep_eng[idx] = []

            # Clean history - New episode has different dynamics
            hist_buf.reset_env(idx)

        obs_raw = next_obs_raw

    # Bootstrap the last value
    last_obs_norm = obs_rms.normalize(obs_raw).astype(np.float32)
    last_hist = hist_buf.get_flat_batch()
    with torch.no_grad():
        last_z = encoder(torch.as_tensor(last_hist, dtype=torch.float32, device=device))
        last_values = value_fn(
            torch.as_tensor(last_obs_norm, dtype=torch.float32, device=device),
            last_z,
        ).cpu().numpy().astype(np.float32)

    # GAE Computation
    rewards_arr = np.asarray(rewards_buf, dtype=np.float32)
    values_arr = np.asarray(values_buf, dtype=np.float32)
    dones_arr = np.asarray(dones_buf, dtype=np.float32)

    advantages, returns = comput_gae(
        rewards_arr, values_arr, dones_arr, last_values,
        gamma=gamma, lam=lam,
    )

    # Flatten [T, N, ...] -> [T*N, ...]
    def flatten(arr, cols):
        a = np.asarray(arr, dtype=np.float32)
        return a.reshape(-1, cols) if cols > 0 else a.reshape(-1)
    
    latent_dim = encoder.latent_dim
    batch = {
        "obs_actor": flatten(obs_buf,obs_dim),
        "policy_extra_obs": flatten(z_buf, latent_dim),
        "actions": flatten(action_buf, act_dim),
        "log_probs": flatten(log_probs_buf, 0),
        "advantages": flatten(advantages, 0),
        "returns": flatten(returns, 0),
    }

    stats = {
        "mean_return": np.mean(ep_returns) if ep_returns else 0.0,
        "std_return": np.std(ep_returns) if ep_returns else 0.0,
        "mean_velocity_x": np.mean(ep_velocities_x) if ep_velocities_x else 0.0,
        "mean_energy": np.mean(ep_energies) if ep_energies else 0.0,
        "survival_rate": np.mean(ep_survivals) if ep_survivals else 0.0,
        "mean_ep_len": np.mean(ep_lengths) if ep_lengths else 0.0,
        "num_episodes": len(ep_returns),
    }

    return batch, stats, total_steps
    

def finetune(cfg, encoder, student, env, obs_rms, device="cpu"):
    """
    RL fine-tubing with PPO using PPOMode.Student

    The encoder is frozen - its weights do not change during fine tuning.
    Only the student policy nd a fresh critic are trained.

    This closes the fap betwen the distilled student and the actual task reward,
    compensating for the encoder;s lossy compression and the distribution shift
    from supervised -> on-policy
    """

    t_cfg = cfg["teacher"]
    f_cfg = cfg["finetune"]
    enc_cfg = cfg["encoder"]

    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    latent_dim = enc_cfg["latent_dim"]
    H = enc_cfg["history_length"]
    hidden = tuple(cfg["student"]["hidden_sizes"])

    # Freeze teh encoder so teh RL doesn't corrupt the learned dynamic encoding
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Fresh critic: Takes (obs, z) -> scalar value
    # Re-use TeacherValueFn architecture - It just concatenates 2 inputs
    # Input dim: obs_dim + latent_dim instead of obs_dim + privileged_dim
    value_fn = TeacherValueFn(obs_dim, latent_dim, hidden_sizes=hidden).to(device)

    # Optimizer for student policy and frresh critic only
    pi_optimizer = torch.optim.Adam(student.parameters(), lr=f_cfg["lr_actor"])
    vf_optimizer = torch.optim.Adam(value_fn.parameters(), lr=f_cfg["lr_critic"])

    total_steps_target = f_cfg["total_steps"]
    batch_size = t_cfg["batch_size"]
    gamma = t_cfg["gamma"]
    lam = t_cfg["lam"]

    # Debugging print statements
    print(f"\n{'='*60}")
    print(f"  RL FINE TUNING - {total_steps_target} env steps")
    print(f"  Encoder: FROZEN ({encoder.latent_dim} - dim latent)")
    print(f"  Student: {sum(p.numel() for p in student.parameters())} trainable params")
    print(f"  Critic: {sum(p.numel() for p in value_fn.parameters())} trainable params")
    print(f"{'='*60}\n")

    total_steps = 0
    best_reward = -np.inf
    it = 0
    save_dir = Path(t_cfg["save_dir"])

    while total_steps < total_steps_target:
        it += 1
        t0 = time.time()

        # Collect rollouts using the student policy
        batch, stats, steps = collect_student_rollouts(
            env, student, encoder, value_fn, batch_size=batch_size,
            gamma=gamma, lam=lam, device=device, obs_rms=obs_rms, history_length=H)
        total_steps += steps

        # PPO update using the STUDENT Mode Path
        info = ppo_update(
            student, value_fn, batch,
            mode=PPOMode.STUDENT,
            clip_ratio=t_cfg["clip_ratio"],
            entropy_coeff=t_cfg["entropy_coeff"],
            pi_optimizer=pi_optimizer,
            vf_optimizer=vf_optimizer,
            update_epochs=t_cfg["update_epochs"],
            minibatch_size=t_cfg["minibatch_size"],
            max_grad_norm=t_cfg["max_grad_norm"],
            device=device,
        )

        dt = time.time() - t0
        mr = stats["mean_return"]

        # Logging
        print(
            f"[Finetune {it:>4d}]  "
            f"steps={total_steps:>8d}/{total_steps_target}  "
            f"return={mr:>8.1f}  "
            f"survival={stats['survival_rate']:.2f}  "
            f"vx={stats['mean_velocity_x']:.3f}  "
            f"pi_loss={info['pi_loss']:.4f}  "
            f"time={dt:.1f}s"
        )

        if wandb is not None:
            wandb.log({
                "iteration": it,
                "finetune/total_env_steps": total_steps,
                "finetune/mean_return": mr,
                "finetune/survival_rate": stats["survival_rate"],
                "finetune/mean_velocity_x": stats["mean_velocity_x"],
                "finetune/mean_energy": stats["mean_energy"],
                "finetune/mean_ep_len": stats["mean_ep_len"],
                "finetune/pi_loss": info["pi_loss"],
                "finetune/vf_loss": info["vf_loss"],
                "finetune/entropy": info["entropy"],
                "finetune/clipfrac": info["clipfrac"],
            })

        # Checkpoint the best return
        if mr > best_reward:
            best_reward = mr
            save_student_checkpoint(
                save_dir / "student_finetuned.pt",
                encoder, student, value_fn, obs_rms, cfg,
                pi_optimizer, vf_optimizer, it, total_steps, best_reward,
            )

    return encoder, student, value_fn


# -------------------------------------------------------------------------------
# CHECKPOINT SAVE / LOAD
# -------------------------------------------------------------------------------

def save_student_checkpoint(path, encoder, student, value_fn, obs_rms, cfg,
                            pi_optimizer=None, vf_optimizer=None,
                            iteration=0, total_steps=0, best_reward=0.0,
                            priv_head = None, priv_scale = None):
    """
    Save encoder + student + critic + obs_rms to a single file.
    """
    print(f" ... Saving Student Checkpoint to {path} ...")

    data = {
        "encoder_state": encoder.state_dict(),
        "student_state": student.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "config": cfg,
        "iteration": iteration,
        "total_steps": total_steps,
        "best_reward": best_reward,
    }

    if value_fn is not None:
        data["value_fn_state"] = value_fn.state_dict()
    if pi_optimizer is not None:
        data["pi_optimizer_state"] = pi_optimizer.state_dict()
    if vf_optimizer is not None:
        data["vf_optimizer"] = vf_optimizer.state_dict()
    if priv_head is not None:
        data["priv_head_state"] = priv_head.state_dict()
    if priv_scale is not None:
        data["priv_obs_scale"] = np.asarray(priv_scale, dtype=np.float32)

    torch.save(data, path)

def load_student_checkpoint(path, obs_dim, act_dim, enc_cfg, stu_cfg, device="cpu"):
    """
    Load encoder + student from a checkpoint
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    encoder = AdaptationEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        history_length=enc_cfg["history_length"],
        latent_dim=enc_cfg["latent_dim"],
        hidden_size=tuple(enc_cfg["hidden_sizes"]),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])

    student = StudentPolicy(
        obs_dim=obs_dim,
        latent_dim=enc_cfg["latent_dim"],
        act_dim=act_dim,
        hidden_size=tuple(stu_cfg["hidden_sizes"]),
    ).to(device)
    student.load_state_dict(ckpt["student_state"])

    return encoder, student, ckpt


# -------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Student Training")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--teacher-ckpt", default="results/teacher_policy.pt")
    parser.add_argument("--stage", choices=["distill", "finetune", "both"], default="both")
    parser.add_argument("--render-mode", choices=["human"], default=None,
                        help="Render the environment during training. Not recommended for vectorized envs.")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e_cfg = cfg["env"]
    d_cfg = e_cfg["dynamics"]
    enc_cfg = cfg["encoder"]
    stu_cfg = cfg["student"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seed = cfg["teacher"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not args.no_wandb:
        _init_wandb(run_name=f"student-{args.stage}")

    # Load Teacher Mode (Frozen)
    print(f"Loading Teacher from {args.teacher_ckpt}")
    teacher, obs_rms, obs_dim, act_dim = load_teacher(args.teacher_ckpt, device)
    print(f"   obs_dim={obs_dim}  act_dim={act_dim}")

    # Build dynamic configs
    dyn_config = DynamicsConfig(
        friction_range = tuple(d_cfg["friction_range"]),
        mass_scale_range = tuple(d_cfg["mass_scale_range"]),
        action_delay_range = tuple(d_cfg["action_delay_range"]),
        obs_delay_range = tuple(d_cfg["obs_delay_range"]),
        external_force_range = tuple(d_cfg["external_force_range"])
    )

    # Build vectorized environment for parallel envs
    num_envs = e_cfg["num_envs"]
    env = build_vec_env(e_cfg, dyn_config, seed, num_envs, render_mode=args.render_mode)
    if args.render_mode == "human" and num_envs > 1:
        print("WARNING: Rendering multiple vectorized envs can be slow or unreliable. "
              "Use a single env for visualization if you want accurate behavior visualization.")
    print(f"Parallel Envs {env.num_envs}  render_mode={args.render_mode}")

    # Build encoder + Student
    latent_dim = enc_cfg["latent_dim"]
    H = enc_cfg["history_length"]

    encoder = AdaptationEncoder(
        obs_dim = obs_dim,
        act_dim = act_dim,
        history_length = H,
        latent_dim = latent_dim,
        hidden_size = tuple(enc_cfg["hidden_sizes"]),
    ).to(device)

    student = StudentPolicy(
        obs_dim = obs_dim,
        latent_dim = latent_dim,
        act_dim = act_dim,
        hidden_size = tuple(stu_cfg["hidden_sizes"]),
    ).to(device)

    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    print(f"Student: {sum(p.numel() for p in student.parameters())} params")

    save_dir = Path(cfg["teacher"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Distillation
    priv_head = None
    priv_scale = None

    if args.stage in ("distill", "both"):
        encoder, student, priv_head, priv_scale = distill(
            cfg, teacher, obs_rms, encoder, student, env, dyn_config, device)
        
        # Save distilled checkpoint (Before fine-tuning)
        save_student_checkpoint(
            save_dir / "student_distilled.pt",
            encoder, student, value_fn=None, obs_rms=obs_rms, cfg=cfg,
            priv_head=priv_head, priv_scale=priv_scale,
        )
        print(f"Distilled model saved to {save_dir / 'student_distilled.pt'}")

    # Load distilled checkpoint if only fine tuning
    if args.stage == "finetune":
        distilled_path = save_dir / "student_distilled.pt"
        print(f"Loading distilled model from {distilled_path}")
        encoder, student, distilled_ckpt = load_student_checkpoint(
            distilled_path, obs_dim, act_dim, enc_cfg, stu_cfg, device)
        
        # Carry forward the privileged head / scale if the distilled checkpoint
        # has them, so they end up in student_finetuned.pt as well
        if "priv_head_state" in distilled_ckpt:
            priv_head = PrivilegedHead(
                encoder.latent_dim, dyn_config.privileged_dim
            ).to(device)
            priv_head.load_state_dict(distilled_ckpt["priv_head_state"])
        if "priv_obs_scale" in distilled_ckpt:
            priv_scale = np.asarray(distilled_ckpt["priv_obs_scale"], dtype=np.float32)

    # Sage 2: RL finetuning
    if args.stage in ("finetune", "both"):
        encoder, student, value_fn = finetune(
            cfg, encoder, student, env, obs_rms, device)
        
        save_student_checkpoint(
            save_dir / "student_finetuned.pt",
            encoder, student, value_fn, obs_rms, cfg,
            priv_head=priv_head, priv_scale=priv_scale,
        )
        print(f"Fine-tuned model saved to {save_dir / 'student_finetuned.pt'}")

    env.close()
    if wandb is not None:
        wandb.finish()

    print("\n Phase 2 complete")

if __name__ == "__main__":
    main()