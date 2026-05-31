"""
GUI Interface to test the trained student policy with the adaptation encoder.

This mirrors deployement on a real robot:
    1. A per-episode ring buffer collects the last H (obs_rms, action) pairs.
    2. Each step, the encoder maps the flattened history -> z
    3. The student acts on [obs_rms, z]. No privileged dynamic access.

Defaults to the fine-tuned checkpoint, falls back to the distilled one.

Usage:
    python -m evaluation.eval_student_policy
    python -m evaluation.eval_student_policy --checkpoint results/student_distilled.pt
    python -m evaluation.eval_student_policy --episodes 5 --cmd-vx 1.2 --stochastic
    python -m evaluation.eval_student_policy --no-render --no-randomize
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from envs.ant_environment import VelocityAntEnv
from envs.dynamics_config import DynamicsConfig
from models.encoder import AdaptationEncoder
from models.student import StudentPolicy
from training.ppo import RunningMeanStd

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"

DEFAULT_FINETUNED = Path("results/student_finetuned.pt")
DEFAULT_DISTILLED = Path("results/student_distilled.pt")

def cprint(color, text):
    print(color + text + RESET)

# ------------------------------------------------------------------
# Checkpoint Loading
# ------------------------------------------------------------------

def _infer_dims(ckpt, env):
    """Pull obs_dim / act_dim from the checkpoint if present, otherwise the env"""
    obs_dim = ckpt.get("obs_dim")
    act_dim = ckpt.get("act_dim")
    if obs_dim is None:
        obs_dim = env.observation_space.shape[0]
    if act_dim is None:
        act_dim = env.action_space.shape[0]
    return int(obs_dim), int(act_dim)

def load_student_checkpoint(path: Path, env, device: str):
    """
    Reconstruct the encoder + student + obs_rms from a checkpoint produced by
    training.train_student.save_student_checkpoint
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    enc_cfg = cfg.get("encoder", {})
    stu_cfg = cfg.get("student", {})

    H = int(enc_cfg.get("history_length", 50))
    latent_dim = int(enc_cfg.get("latent_dim", 8))
    enc_hidden = tuple(enc_cfg.get("hidden_size", [256, 256]))
    stu_hidden = tuple(stu_cfg.get("hidden_size", [256, 256]))

    obs_dim, act_dim = _infer_dims(ckpt, env)

    encoder = AdaptationEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        history_length=H,
        latent_dim=latent_dim,
        hidden_size=enc_hidden,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    student = StudentPolicy(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        act_dim=act_dim,
        hidden_size=stu_hidden,
    ).to(device)
    student.load_state_dict(ckpt["student_state"])
    student.eval()

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    meta = {
        "iteration":    ckpt.get("iteration", "?"),
        "total_steps":  ckpt.get("total_steps", "?"),
        "best_reward":  ckpt.get("best_reward", "?"),
        "history_length": H,
        "latent_dim": latent_dim,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "has_value_fn": "value_fn_state" in ckpt,
    }

    return encoder, student, obs_rms, meta


def resolve_checkpoint(arg_path: str | None) -> Path:
    if arg_path is not None:
        return Path(arg_path)
    if DEFAULT_FINETUNED.exists():
        return DEFAULT_FINETUNED
    if DEFAULT_DISTILLED.exists():
        cprint(YELLOW,
               f"[warn] {DEFAULT_FINETUNED} not found, falling back to {DEFAULT_DISTILLED}")
        return DEFAULT_DISTILLED
    return DEFAULT_FINETUNED


# ------------------------------------------------------------------
# Rollout
# ------------------------------------------------------------------

def run_episode(env, encoder, student, obs_rms, device, *, history_length, deterministic):
    """
    Single env rollout. Maintins a local history ring buffer that is cleared
    at every reset, just like the per-env buffer used during training.
    """
    obs, _info = env.reset()
    pair_dim = encoder.obs_dim + encoder.act_dim
    history = deque(maxlen=history_length)
    hist_flat_zeros = np.zeros(history_length * pair_dim, dtype=np.float32)

    ep_return = 0.0
    ep_steps = 0
    ep_vels = []
    ep_energies = []
    ep_z_norms = []
    terminated = False

    while True:
        obs_norm = obs_rms.normalize(obs).astype(np.float32)

        # Flatten history (oldest first). Zero padded during warmup
        if len(history) > 0:
            stacked = np.array(history, dtype=np.float32).reshape(-1)
            # Pad on the left, so the most recent pair stays last.
            if stacked.size < hist_flat_zeros.size:
                pad = np.zeros(hist_flat_zeros.size - stacked.size, dtype=np.float32)
                hist_flat = np.concatenate([pad, stacked])
            else:
                hist_flat = stacked
        else:
            hist_flat = hist_flat_zeros

        obs_t = torch.as_tensor(obs_norm, device=device).unsqueeze(0)
        hist_t = torch.as_tensor(hist_flat, device=device).unsqueeze(0)

        with torch.no_grad():
            z = encoder(hist_t)
            if deterministic:
                action_t = student.get_mean(obs_t, z)
            else:
                action_t, _ = student.act(obs_t, z)

        action = action_t.squeeze(0).cpu().numpy()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)

        # Append AFTER the step, matching training.
        history.append(np.concatenate([obs_norm, action.astype(np.float32)]))

        ep_return += reward
        ep_steps += 1
        ep_z_norms.append(float(np.linalg.norm(z.squeeze(0).cpu().numpy())))

        if "body_vx" in info:
            ep_vels.append(float(info["body_vx"]))
        if "energy" in info:
            ep_energies.append(float(info["energy"]))

        if terminated or truncated:
            break

    return {
        "return":   ep_return,
        "steps":    ep_steps,
        "survived": not terminated,
        "mean_vel": float(np.mean(ep_vels)) if ep_vels else 0.0,
        "mean_energy": float(np.mean(ep_energies)) if ep_energies else 0.0,
        "mean_z_norm": float(np.mean(ep_z_norms)) if ep_z_norms else 0.0,
        "friction": float(env._friction_scale),
        "mass_scale": float(env._mass_scale),
        "action_delay": int(env._action_delay),
        "obs_delay": int(env._obs_delay),
        "ext_force_mag": float(np.linalg.norm(env._ext_force)),
    }


# ------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained student policy in sim")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint (default: results/student_finetuned.pt, "
                              "falls back to results/student_distilled.pt).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cmd-vx", type=float, default=1.0)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-randomize", action="store_true")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    if not ckpt_path.exists():
        cprint(RED, f"[error] Checkpoint not found: {ckpt_path}")
        cprint(YELLOW, "   Train first with: python -m training.train_student")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optionally pin dynamics to nominal so we can isolate policy behavior
    if args.no_randomize:
        dyn_config = DynamicsConfig(
            friction_range=(1.0, 1.0),
            mass_scale_range=(1.0, 1.0),
            action_delay_range=(0, 0),
            obs_delay_range=(0, 0),
            external_force_range=(0.0, 0.0),
        )
    else:
        dyn_config = DynamicsConfig()

    env = VelocityAntEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_steps,
        dynamics_config=dyn_config,
        fixed_command=(args.cmd_vx, args.cmd_vy, args.cmd_yaw_rate),
        randomization_seed=args.seed,
    )

    cprint(BOLD, f"\n Loading: {ckpt_path}")
    encoder, student, obs_rms, meta = load_student_checkpoint(ckpt_path, env, device)

    best_reward = meta["best_reward"]
    try:
        best_reward_str = f"{float(best_reward):.2f}"
    except:
        best_reward_str = str(best_reward)

    print(
        f"   Iteration      : {meta['iteration']}\n",
        f"   Env Steps      : {meta['total_steps']}\n",
        f"   Best Reward    : {best_reward_str}\n",
        f"   History H      : {meta['history_length']}\n",
        f"   Latent d_z     : {meta['latent_dim']}\n",
        f"   obs_dim        : {meta['obs_dim']}   act_dim: {meta['act_dim']}\n",
        f"   has critic     : {meta['has_value_fn']}   device: {device}"
    )

    mode = "stochastic" if args.stochastic else "deterministic (mean)"
    dyn_mode = "nominal (no DR)" if args.no_randomize else "randomed"
    cprint(CYAN,
           f"\nRunning {args.episodes} episodes [{mode}]  [dynamics: {dyn_mode}]   "
           f"[cmd vx={args.cmd_vx} vy={args.cmd_vy} yaw={args.cmd_yaw_rate}]")
    
    if not args.no_render:
        print("  Close te MuJuCo window to stop early.\n")

    all_stats = []
    try:
        for ep in range(1, args.episodes + 1):
            stats = run_episode(
                env, encoder, student, obs_rms, device,
                history_length=meta["history_length"],
                deterministic=not args.stochastic,
            )
            all_stats.append(stats)
    except KeyboardInterrupt:
        cprint(YELLOW, "\n\nInterrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()