#!/usr/bin/env python3
"""
GUI evaluation of the best teacher policy.

Usage:
    python eval_teacher_gui.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

from envs.ant_environment import VelocityAntEnv
from envs.dynamics_config import DynamicsConfig
from models.teacher_model import TeacherPolicy
from training.ppo import RunningMeanStd


RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"

CHECKPOINT      = Path("results/teacher_policy.pt")
EPISODES        = 10
MAX_STEPS       = 1000
CMD_VX          = 1.0
CMD_VY          = 0.0
CMD_YAW_RATE    = 0.0


def cprint(color, text):
    print(color + text + RESET)


def load_checkpoint(path: Path, device: str):
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    cfg    = ckpt.get("config", {})
    hidden = tuple(cfg.get("teacher", {}).get("hidden_sizes", [256, 256]))

    policy = TeacherPolicy(ckpt["obs_dim"], ckpt["act_dim"], hidden_sizes=hidden).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    obs_rms = RunningMeanStd(shape=(ckpt["obs_dim"],))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    return policy, obs_rms, {
        "iteration":   ckpt.get("iteration", "?"),
        "total_steps": ckpt.get("total_steps", "?"),
        "best_reward": ckpt.get("best_reward", float("nan")),
        "obs_dim":     ckpt["obs_dim"],
        "act_dim":     ckpt["act_dim"],
    }


def run_episode(env, policy, obs_rms, device):
    obs, _ = env.reset()
    ep_return, ep_steps = 0.0, 0
    ep_vels, ep_energies = [], []

    while True:
        obs_norm = obs_rms.normalize(obs).astype(np.float32)
        obs_t    = torch.as_tensor(obs_norm, device=device).unsqueeze(0)

        with torch.no_grad():
            action = policy.get_mean(obs_t).squeeze(0).cpu().numpy()

        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_return += reward
        ep_steps  += 1
        if "body_vx" in info:
            ep_vels.append(float(info["body_vx"]))
        if "energy" in info:
            ep_energies.append(float(info["energy"]))

        if terminated or truncated:
            break

    return {
        "return":           ep_return,
        "steps":            ep_steps,
        "survived":         not terminated,
        "mean_vel":         float(np.mean(ep_vels))     if ep_vels     else 0.0,
        "mean_energy":      float(np.mean(ep_energies)) if ep_energies else 0.0,
        "friction":         env._friction_scale,
        "mass_scale":       env._mass_scale,
        "action_delay":     env._action_delay,
        "obs_delay":        env._obs_delay,
        "external Force":   env._ext_force
    }


HEADER = (
    f"{'Ep':>4} | {'Return':>9} | {'Steps':>5} | {'Vel':>6} | "
    f"{'Energy':>7} | {'Surv':>4} | {'Friction':>8} | {'Mass':>6} | {'Delay':>5}"
)
SEP = "-" * len(HEADER)


def print_row(ep_idx, stats):
    surv = (GREEN + "yes" + RESET) if stats["survived"] else (RED + "no " + RESET)
    print(
        f"{ep_idx:4d} | {stats['return']:9.2f} | {stats['steps']:5d} | "
        f"{stats['mean_vel']:6.3f} | {stats['mean_energy']:7.4f} | {surv}  | "
        f"{stats['friction']:8.3f} | {stats['mass_scale']:6.3f} | {stats['action_delay']:5d}"
    )


def print_summary(all_stats):
    returns   = [s["return"]   for s in all_stats]
    vels      = [s["mean_vel"] for s in all_stats]
    surv_rate = np.mean([s["survived"] for s in all_stats]) * 100
    print()
    cprint(BOLD, "── Summary " + "─" * (len(HEADER) - 11))
    print(
        f"  Episodes : {len(all_stats)}\n"
        f"  Return   : {np.mean(returns):8.2f} ± {np.std(returns):.2f}  "
        f"(min {np.min(returns):.1f}, max {np.max(returns):.1f})\n"
        f"  Mean vel : {np.mean(vels):8.3f}\n"
        f"  Survival : {surv_rate:5.1f}%"
    )
    print()


def main():
    if not CHECKPOINT.exists():
        cprint(RED, f"[error] Checkpoint not found: {CHECKPOINT}")
        cprint(YELLOW, "  Train first with: python -m training.train_teacher")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cprint(BOLD, f"\nLoading: {CHECKPOINT}")
    policy, obs_rms, meta = load_checkpoint(CHECKPOINT, device)
    print(
        f"  Iteration {meta['iteration']}  |  "
        f"{meta['total_steps']:,} env steps  |  "
        f"Best reward {meta['best_reward']:.2f}  |  "
        f"device={device}"
    )

    env = VelocityAntEnv(
        render_mode="human",
        max_episode_steps=MAX_STEPS,
        dynamics_config=DynamicsConfig(),
        fixed_command=(CMD_VX, CMD_VY, CMD_YAW_RATE),
        randomization_seed=0,
    )

    cprint(CYAN, f"\nRunning {EPISODES} episodes  [deterministic]  "
           f"[cmd vx={CMD_VX} vy={CMD_VY} yaw={CMD_YAW_RATE}]")
    print("  Close the MuJoCo window to stop early.\n")
    cprint(BOLD + CYAN, HEADER)
    print(SEP)

    all_stats = []
    try:
        for ep in range(1, EPISODES + 1):
            stats = run_episode(env, policy, obs_rms, device)
            all_stats.append(stats)
            print_row(ep, stats)
    except KeyboardInterrupt:
        cprint(YELLOW, "\n\nInterrupted.")
    finally:
        env.close()

    if all_stats:
        print_summary(all_stats)


if __name__ == "__main__":
    main()