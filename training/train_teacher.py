"""
Phase 1: Train teh teacher policy with asymmetric actor-critic

The actor sees only proprioceptive obs
The critic sees obs + privileged dynamics info
Dynamics are randomized every episode reset
"""

import sys
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from envs.ant_environment import VelocityAntEnv
from envs.dynamics_config import DynamicsConfig
from models.teacher_model import TeacherPolicy, TeacherValueFn
from training.ppo import RunningMeanStd, comput_gae, ppo_update, PPOMode

# wandb is optional
wandb = None

def _init_wandb():
    """ Initialise Weights and Biases"""
    global wandb
    import wandb as _wandb
    wandb = _wandb

    wandb.init(
        project="Latent-Dynamics-Encoder",
    )
    wandb.define_metric("iteration")
    wandb.define_metric("*", step_metric="iteration")
    return wandb

# ---------------------------------------------------------------------------------------
# Load Configs
def load_config(path="training/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------------------
# Make environments

def make_env(dynamics_config, cmd_vx_range, cmd_vy_range, cmd_yaw_rate_range,
             max_episode_steps, randomization_seed, render_mode=None):
    return VelocityAntEnv(
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        cmd_vx_range=cmd_vx_range,
        cmd_vy_range=cmd_vy_range,
        cmd_yaw_rate_range=cmd_yaw_rate_range,
        dynamics_config=dynamics_config,
        randomization_seed=randomization_seed
    )

# -----------------------------------------------------------------------------
# Trajectory Collection.
def collect_rollouts(env, policy, value_fn, batch_size, gamma, lam, device,
                    obs_rms=None):
    # Roll out the policy, returning a batch dict and episode statistics
    # Return a single merged batch suitable for TRPO update

    num_envs = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    obs_actor_buf, obs_priv_buf, action_buf = [], [], []
    rewards_buf, dones_buf, log_probs_buf, values_buf = [], [], [], []
    raw_obs_buffer = []

    REWARD_TERM_KEYS = [
        "r_forward", "r_lateral", "r_vz", "r_yaw", "r_height", "r_orient", "r_energy",
        "r_smooth", "r_symmetry", "r_alive", "r_stand", "total_reward"]

    ep_returns, ep_lengths = [], []
    ep_velocities_x, ep_velocities_y, ep_energies, ep_survivals = [], [], [], []
    ep_reward_terms = {k: [] for k in REWARD_TERM_KEYS}
    
    obs_raw, infos = env.reset()
    priv_raw = np.stack(infos["privileged_obs"])

    ep_ret = np.zeros(num_envs, dtype=np.float64)
    ep_len = np.zeros(num_envs, dtype=np.int32)
    ep_velx = [[] for _ in range(num_envs)]
    ep_vely = [[] for _ in range(num_envs)]
    ep_eng = [[] for _ in range(num_envs)]

    total_steps = 0

    while total_steps < batch_size:
        raw_obs_buffer.append(obs_raw.copy())

        obs_norm = obs_rms.normalize(obs_raw).astype(np.float32) if obs_rms else obs_raw.astype(np.float32)
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
        priv_t = torch.as_tensor(priv_raw.astype(np.float32), dtype=torch.float32, device=device)

        with torch.no_grad():
            action, lp = policy.act(obs_t)
            val = value_fn(obs_t, priv_t)

        action_np = action.cpu().numpy()
        action_clipped = np.clip(action_np, env.single_action_space.low, env.single_action_space.high)

        next_obs_raw, reward, terminated, truncated, infos = env.step(action_clipped)
        done = np.logical_or(terminated,truncated)

        obs_actor_buf.append(obs_norm.copy())
        obs_priv_buf.append(priv_raw.astype(np.float32).copy()) 
        action_buf.append(action_np.copy()) # Store unclipped so log probs stay consistent
        rewards_buf.append(reward.astype(np.float32))
        dones_buf.append(done.astype(np.float32))
        log_probs_buf.append(lp.cpu().numpy().astype(np.float32))
        values_buf.append(val.cpu().numpy().astype(np.float32))
        
        ep_ret += reward
        ep_len += 1
        total_steps += num_envs
        
        for i in range(num_envs):
            if "body_vx" in infos:
                bvx = np.asarray(infos["body_vx"]).ravel()
                ep_velx[i].append(float(bvx[i]))
            if "body_vy" in infos:
                bvy = np.asarray(infos["body_vy"]).ravel()
                ep_vely[i].append(float(bvy[i]))
            if "energy" in infos:
                eng = np.asarray(infos["energy"]).ravel()
                ep_eng[i].append(float(eng[i]))
                
        for idx in np.nonzero(done)[0]:
            ep_returns.append(float(ep_ret[idx]))
            ep_lengths.append(int(ep_len[idx]))
            mean_velx= float(np.mean(ep_velx[idx])) if ep_velx[idx] else 0.0
            ep_velocities_x.append(mean_velx)
            mean_vely= float(np.mean(ep_vely[idx])) if ep_vely[idx] else 0.0
            ep_velocities_y.append(mean_vely)
            mean_eng = float(np.mean(ep_eng[idx])) if ep_eng[idx] else 0.0
            ep_energies.append(mean_eng)
            survived = not terminated[idx] if hasattr(terminated, '__getitem__') else not terminated
            ep_survivals.append(float(survived))

            # Append Reward terms
            if "episode_reward_terms" in infos:
                ert = infos["episode_reward_terms"]
                for rk in REWARD_TERM_KEYS:
                    if isinstance(ert, dict):
                        ep_reward_terms[rk].append(float(np.asarray(ert.get(rk, 0.0)).item()))
                    else:
                        ep_reward_terms[rk].append(float(ert[idx].get(rk, 0.0)))

            ep_ret[idx] = 0.0
            ep_len[idx] = 0.0
            ep_velx[idx] = []
            ep_vely[idx] = []
            ep_eng[idx] = []
        
        obs_raw = next_obs_raw
        priv_raw = np.stack(infos["privileged_obs"])

    # Update obs normalizer with all raaw observations seen this rollout
    if obs_rms is not None:
        obs_rms.update(np.asarray(raw_obs_buffer, dtype=np.float64).reshape(-1, obs_dim))

    # Bootstrap value for the last observation
    last_obs_norm = obs_rms.normalize(obs_raw).astype(np.float32) if obs_rms else obs_raw.astype(np.float32)
    last_priv = priv_raw.astype(np.float32)

    with torch.no_grad():
        last_values = value_fn(
            torch.as_tensor(last_obs_norm, dtype=torch.float32, device=device),
            torch.as_tensor(last_priv, dtype=torch.float32, device=device)
        ).cpu().numpy().astype(np.float32)

    rewards_arr = np.asarray(rewards_buf, dtype=np.float32)    # [T, N]
    values_arr = np.asarray(values_buf, dtype=np.float32)
    dones_arr = np.asarray(dones_buf, dtype=np.float32)

    advantages, returns = comput_gae(
        rewards_arr,
        values_arr,
        dones_arr,
        last_values,
        gamma=gamma,
        lam=lam,
    )

    def flatten(arr, cols):
        # Flatten [T, N, ...] -> [T*N, ....]
        a = np.asarray(arr, dtype=np.float32)
        return a.reshape(-1, cols) if cols > 0 else a.reshape(-1)

    batch = {
        "obs_actor": flatten(obs_actor_buf, obs_dim),
        "obs_critic_priv": flatten(obs_priv_buf, 7),
        "actions": flatten(action_buf, act_dim),
        "log_probs": flatten(log_probs_buf, 0),
        "advantages": flatten(advantages, 0),
        "returns": flatten(returns, 0)        
    }

    stats = {
        "mean_return": np.mean(ep_returns) if ep_returns else 0.0,
        "std_return": np.std(ep_returns) if ep_returns else 0.0,
        "mean_velocity_x": np.mean(ep_velocities_x) if ep_velocities_x else 0.0,
        "mean_velocity_y": np.mean(ep_velocities_y) if ep_velocities_y else 0.0,
        "mean_energy": np.mean(ep_energies) if ep_energies else 0.0,
        "survival_rate": np.mean(ep_survivals) if ep_survivals else 0.0,
        "mean_op_len": np.mean(ep_lengths) if ep_lengths else 0.0,
        "num_episodes": len(ep_returns),
    }

    for rk in REWARD_TERM_KEYS:
        vals = ep_reward_terms[rk]
        stats[rk] = float(np.mean(vals)) if vals else 0.0

    return batch, stats, total_steps



# ---------------------------------------------------------------------------------------
# Save / Load Checkpoints
def save_checkpoint(path, policy, value_fn, pi_optimizer,
                    vf_opt, iteration, total_steps, best_reward, obs_rms, cfg):
    print("... saving model ...")
    torch.save({
        "policy_state": policy.state_dict(),
        "value_fn_state": value_fn.state_dict(),
        "pi_optimizer_state": pi_optimizer.state_dict(),
        "vf_optimizer_state": vf_opt.state_dict(),
        "iteration": iteration,
        "total_steps": total_steps,
        "best_reward": best_reward,
        "obs_rms": obs_rms.state_dict(),
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
        "config": cfg,
    }, path)
    

# ------------------------------------------------------------------------------------
# Training Loop
def train(config_path="training/config.yaml"):
    cfg = load_config(config_path)
    t_cfg = cfg["teacher"]
    e_cfg = cfg["env"]
    d_cfg = e_cfg["dynamics"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device= "cpu"
    print(f"Device: {device}")

    seed = t_cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    _init_wandb()

    dyn_config = DynamicsConfig(
        friction_range = tuple(d_cfg["friction_range"]),
        mass_scale_range = tuple(d_cfg["mass_scale_range"]),
        action_delay_range = tuple(d_cfg["action_delay_range"]),
        obs_delay_range = tuple(d_cfg["obs_delay_range"]),
        external_force_range = tuple(d_cfg["external_force_range"])
    )

    # Vectorized Enviornment
    num_envs = e_cfg["num_envs"]
    env_fns = []
    for i in range(num_envs):
        env_fns.append(partial(
            make_env,
            dynamics_config = dyn_config,
            cmd_vx_range = tuple(e_cfg["cmd_vx_range"]),
            cmd_vy_range = tuple(e_cfg["cmd_vy_range"]),
            cmd_yaw_rate_range = tuple(e_cfg["cmd_yaw_rate_range"]),
            max_episode_steps = e_cfg["max_episode_steps"],
            randomization_seed = seed + i,
        ))

    # Use AsyncVectorEnv for parallel environment execution with pipes (shared_memory=False)
    env = gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=False,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    print(f"Parallel Envs: {env.num_envs}")

    obs_dim = env.single_observation_space.shape[0] # 30 base
    act_dim = env.single_action_space.shape[0]      # 8
    priv_dim = dyn_config.privileged_dim            # 7
    print(f"Obs dim: {obs_dim}  |  Act Dim: {act_dim}  |  Privileged dim: {priv_dim}")

    # Networks
    hidden = tuple(t_cfg["hidden_sizes"])
    policy = TeacherPolicy(obs_dim, act_dim, hidden_sizes=hidden).to(device)
    value_fn = TeacherValueFn(obs_dim, priv_dim, hidden_sizes=hidden).to(device)
    # Optimizers
    pi_optimizer = torch.optim.Adam(policy.parameters(), lr=t_cfg["lr_actor"])
    vf_optimizer = torch.optim.Adam(value_fn.parameters(), lr=t_cfg["lr_critic"])
    
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    save_dir = Path(t_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    best_reward = -np.inf
    
    # Main Loop
    for it in range(1, t_cfg["iterations"] + 1):
        t0 = time.time()

        batch, stats, steps_collected = collect_rollouts(
            env, policy, value_fn,
            batch_size=t_cfg["batch_size"],
            gamma=t_cfg["gamma"],
            lam=t_cfg["lam"],
            device=device,
            obs_rms=obs_rms
        )
        total_steps += steps_collected

        info = ppo_update(
            policy, value_fn, batch,
            mode=PPOMode.TEACHER,
            clip_ratio=t_cfg["clip_ratio"],
            entrpoy_coeff=t_cfg["entropy_coeff"],
            pi_optimizer=pi_optimizer,
            vf_optimizer=vf_optimizer,
            update_epochs=t_cfg["update_epochs"],
            minibatch_size=t_cfg["minibatch_size"],
            max_grad_norm=t_cfg["max_grad_norm"],
            device=device
        )
        
        dt = time.time() - t0
        mr = stats["mean_return"]

        # Logging
        wandb.log(
            {
                "iteration": it,
                "total_env_steps": total_steps,
                "mean_return": mr,

                "std_return": stats["std_return"],
                "mean_velocity_x": stats["mean_velocity_x"],
                "mean_velocity_y": stats["mean_velocity_y"],
                "mean_energy": stats["mean_energy"],
                "survival_rate": stats["survival_rate"],
                "mean_op_len": stats["mean_op_len"],
                "num_episodes": stats["num_episodes"],

                "pi_loss": info["pi_loss"],
                "vf_loss": info["vf_loss"],
                "entropy": info["entropy"],
                "clipfrac": info["clipfrac"],

                "reward/r_forward": stats["r_forward"],
                "reward/r_lateral": stats["r_lateral"],
                "reward/r_vz": stats["r_vz"],
                "reward/r_yaw": stats["r_yaw"],
                "reward/r_height": stats["r_height"],
                "reward/r_orient": stats["r_orient"],
                "reward/r_energy": stats["r_energy"],
                "reward/r_smooth": stats["r_smooth"],
                "reward/r_symmetry": stats["r_symmetry"],
                "reward/r_alive": stats["r_alive"],
                "reward/r_stand": stats["r_stand"],
                "reward/total_reward": stats["total_reward"],
                
                "timing/iter_seconds": dt,
            },
        )

        # Checkpoint
        if mr > best_reward:
            best_reward = mr
            save_checkpoint(
                save_dir / "teacher_policy.pt",
                policy, value_fn, pi_optimizer, vf_optimizer,
                it, total_steps, best_reward, obs_rms, cfg,
            )

    env.close()

    wandb.finish()

    print(f"\nDone. Best mean return {best_reward:.2f}")
    print("Models saved in {save_dir.resolve()}")

    return policy, value_fn


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "training/config.yaml"
    train(config_path)
