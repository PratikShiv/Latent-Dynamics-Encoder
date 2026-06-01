"""
Laten space analysis of the adaptation encoder.

Sweeps a grid of (friction_scale, mass_scale) values, for N episodes per cell with
the trained student policy, collects the encoder's latent z, and produces 3 outputs:

    1. UMAP of z, colored by friction_scale
    2. UMAP of z, colored by mass_scale
    3. Linear probing: Fut a linear regression from:
        a. z -> friction_scale
        b. z -> mass_scale

If the encoder has actually learned the dynamics, the UMAP should show a smooth
gradient and the linear probes should clear R^2 >= 0.85
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import umap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor

from envs.ant_environment import VelocityAntEnv
from envs.dynamics_config import DynamicsConfig
from evaluation.eval_student_policy import load_student_checkpoint, resolve_checkpoint

# -------------------------------------------------------
# Config
# -------------------------------------------------------

CHECKPOINT = None       # None -> finetuned, fallback to distilled
OUTPUT_DIR = Path("results/latent_analysis")
CACHE_PATH = OUTPUT_DIR / "cache.npz"
USE_CACHE = False   # Set True to skip rollouts and reuse cache

FRICTION_RANGE = (0.3, 1.5)
MASS_RANGE = (0.7, 1.4)
FRICTION_BINS = 6
MASS_BINS = 6
EPISODES_PER_CELL = 5

MAX_STEPS = 500
STRIDE = 20
WARMUP_STEPS = None
DETERMINISTIC = True

FIXED_CMD = (1.0, 0.0, 0.0)
SEED = 0


# -------------------------------------------------------
# Data Collection
# -------------------------------------------------------

def make_fixed_dyn_config(friction, mass_scale):
    return DynamicsConfig(
        friction_range=(friction, friction),
        mass_scale_range=(mass_scale, mass_scale),
        action_delay_range=(0, 0),
        obs_delay_range=(0, 0),
        external_force_range=(0.0, 0.0),
    )

def collect_episode_latent(env, encoder, student, obs_rms, device,
                           history_length, warmup_steps, stride, deterministic):
    obs, _info = env.reset()
    pair_dim = encoder.obs_dim + encoder.act_dim
    history = deque(maxlen=history_length)
    hist_flat_zeros = np.zeros(history_length * pair_dim, dtype=np.float32)

    zs = []
    step = 0

    while True:
        obs_norm = obs_rms.normalize(obs).astype(np.float32)
        
        if len(history) > 0:
            stacked = np.array(history, dtype=np.float32).reshape(-1)
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

        obs, _reward, terminated, truncated, info = env.step(action)
        history.append(np.concatenate([obs_norm, action.astype(np.float32)]))

        if step >= warmup_steps and (step - warmup_steps) % stride == 0:
            zs.append(z.squeeze(0).cpu().numpy())

        step += 1
        if terminated or truncated:
            break


    if not zs:
        return np.empty((0, encoder.latent_dim), dtype=np.float32)
    return np.stack(zs, axis=0).astype(np.float32)
    

def collect_dataset(env, encoder, student, obs_rms, device, friction_values,
                    mass_values, episodes_per_cell, history_length, warmup_steps,
                    stride, deterministic):
    Z_chunks, fr_chunks, ms_chunks, ep_chunks = [], [], [], []
    ep_idx = 0
    total_cells = len(friction_values) * len(mass_values)
    cell_i = 0

    for f in friction_values:
        for m in mass_values:
            cell_i += 1
            print(f"[cell {cell_i}/{total_cells}] friction={f:.3f} mass={m:.3f}")
            env.dyn_cfg = make_fixed_dyn_config(f, m)

            for _ in range(episodes_per_cell):
                z = collect_episode_latent(
                    env, encoder, student, obs_rms, device,
                    history_length=history_length,
                    warmup_steps=warmup_steps,
                    stride=stride,
                    deterministic=deterministic
                )
                if z.shape[0] == 0:
                    ep_idx += 1
                    continue
                
                Z_chunks.append(z)
                fr_chunks.append(np.full(z.shape[0], f, dtype=np.float32))
                ms_chunks.append(np.full(z.shape[0], m, dtype=np.float32))
                ep_chunks.append(np.full(z.shape[0], ep_idx, dtype=np.int32))
                ep_idx += 1

    if not Z_chunks:
        return (np.empty((0, encoder.laten_dim), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    
    return (np.concatenate(Z_chunks, axis=0),
            np.concatenate(fr_chunks, axis=0),
            np.concatenate(ms_chunks, axis=0),
            np.concatenate(ep_chunks, axis=0))


# -------------------------------------------------------
# Analysis
# -------------------------------------------------------

def run_umap(Z, seed=0):
    n_neighbors = min(15, max(2, Z.shape[0] // 5))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=seed,
    )
    return reducer.fit_transform(Z)

def plot_umap(emb, color, title, cbar_label, out_path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap=cmap, s=8, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def linear_probe(Z, target, groups, label, seed=0):
    # Split the data in 80/20
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(Z, target, groups))

    model = LinearRegression()
    model.fit(Z[train_idx], target[train_idx])
    r2_tr = r2_score(target[train_idx], model.predict(Z[train_idx]))
    r2_te = r2_score(target[test_idx], model.predict(Z[test_idx]))

    print(f"  {label:>16s}: R^2_train={r2_tr:6.3f}   R^2_test={r2_te:6.3f}  "
          f"(n_train={len(train_idx)}, n_test={len(test_idx)})")
    
    return r2_tr, r2_te
    
def nonlinear_probe(Z, target, groups, label, seed=0):
    # Split the data in 80/20
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(Z, target, groups))

    model = MLPRegressor(
        hidden_layer_sizes=(64, 64), max_iter=2000, random_state=seed)
    
    model.fit(Z[train_idx], target[train_idx])
    r2_tr = r2_score(target[train_idx], model.predict(Z[train_idx]))
    r2_te = r2_score(target[test_idx], model.predict(Z[test_idx]))

    print(f" [MLP] {label:>16s}: R^2_train={r2_tr:6.3f}   R^2_test={r2_te:6.3f}  "
          f"(n_train={len(train_idx)}, n_test={len(test_idx)})")
    
    return r2_tr, r2_te

# -------------------------------------------------------
# Main Loop
# -------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = VelocityAntEnv(
        render_mode=None,
        max_episode_steps=MAX_STEPS,
        dynamics_config=make_fixed_dyn_config(1.0, 1.0),
        fixed_command=FIXED_CMD,
        randomization_seed=SEED,
    )

    ckpt_path = resolve_checkpoint(CHECKPOINT)
    print(f" Loading Checkpoint: {ckpt_path}")
    encoder, student, obs_rms, meta = load_student_checkpoint(ckpt_path, env, device)
    warmup = WARMUP_STEPS if WARMUP_STEPS is not None else meta["history_length"]
    print(f" history={meta['history_length']} latent_dim={meta['latent_dim']} device={device}")
    print(f" warmup_steps={warmup} stride={STRIDE}")

    friction_values = np.linspace(*FRICTION_RANGE, FRICTION_BINS).round(4)
    mass_values = np.linspace(*MASS_RANGE, MASS_BINS).round(4)
    print(f" friction sweep: {friction_values.tolist()}")
    print(f" mass sweep: {mass_values.tolist()}")
    print(f" episodes/cell: {EPISODES_PER_CELL} "
          f"(total = {FRICTION_BINS * MASS_BINS * EPISODES_PER_CELL})")
    
    if USE_CACHE and CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        Z, friction, mass, groups = data["Z"], data["friction"], data["mass"], data["groups"]
    else:
        print("Collecting rollouts ...")
        Z, friction, mass, groups = collect_dataset(
            env, encoder, student, obs_rms, device,
            friction_values=friction_values,
            mass_values=mass_values,
            episodes_per_cell=EPISODES_PER_CELL,
            history_length=meta["history_length"],
            warmup_steps=warmup,
            stride=STRIDE,
            deterministic=DETERMINISTIC,
        )
        np.savez(CACHE_PATH, Z=Z, friction=friction, mass=mass, groups=groups)
        print(f"   cached -> {CACHE_PATH}")

    env.close()

    if Z.shape[0] == 0:
        print("ERROR: no latent samples collected.")
        return
    
    n_eps = int(groups.max()) + 1
    print(f"Dataset: Z {Z.shape} episodes={n_eps} samples/ep~{Z.shape[0]/n_eps:.1f}")

    # Plot 1+2: UMAP
    print("Fitting UMAP ...")
    emb = run_umap(Z, seed=SEED)

    umap_friction_path = OUTPUT_DIR / "umap_friction.png"
    umap_mass_path = OUTPUT_DIR / "umap_mass.png"
    plot_umap(emb, friction,
              "Latent z - UMAP, colored by friction scale",
              "friction scale", umap_friction_path, cmap="viridis")
    plot_umap(emb, mass,
              "Latent z - UMAP, colored by mass scale",
              "mass scale", umap_mass_path, cmap="plasma")
    print(f"  Saved -> {umap_friction_path}")
    print(f"  Saved -> {umap_mass_path}")

    # 3. Linear Probing
    print("Linear Probing (z -> dynamics):")
    fr_tr, fr_te = linear_probe(Z, friction, groups, "friction_scale", seed=SEED)
    ms_tr, ms_te = linear_probe(Z, mass, groups, "mass_scale", seed=SEED)

    # 4. Non-Linear Probing
    print("Linear Probing (z -> dynamics):")
    fr_tr_nl, fr_te_nl = nonlinear_probe(Z, friction, groups, "friction_scale", seed=SEED)
    ms_tr_nl, ms_te_nl = nonlinear_probe(Z, mass, groups, "mass_scale", seed=SEED)

    summary_path = OUTPUT_DIR / "linear_probe_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Linear Probe (z -> dynamics), group-aware 80/20 split.\n\n")
        f.write(f"friction_scale  R^2_train={fr_tr:.4f}  R^2_test={fr_te:.4f}\n")
        f.write(f"mass_scale  R^2_train={ms_tr:.4f}  R^2_test={ms_te:.4f}\n\n\n")

        f.write("Non-Linear Probe (z -> dynamics), group-aware 80/20 split.\n\n")
        f.write(f"friction_scale  R^2_train={fr_tr_nl:.4f}  R^2_test={fr_te_nl:.4f}\n")
        f.write(f"mass_scale  R^2_train={ms_tr_nl:.4f}  R^2_test={ms_te_nl:.4f}\n")
    print(f"  summary -> {summary_path}")

if __name__ == "__main__":
    main()