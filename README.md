# Latent-Dynamics-Encoder

A compact adaptation encoder for robot dynamics, trained to infer changing dynamics from history alone.
This repository reproduces and extends the RMA-style adaptation pipeline using a convolutional encoder over proprioceptive history, explicit latent analysis, and a two-stage student-teacher training framework.

## What this project is

This project learns a latent dynamics embedding `z` from only past observation-action history, without privileged dynamics access at inference time.
The learned latent is then used by a student policy to adapt control behavior to new dynamics in a randomized MuJoCo Ant environment.

Key features:
- 1D convolutional encoder over a history window of `H=50` past observation-action pairs
- Teacher-student distillation to transfer a privileged teacher policy into an adaptive student policy
- Privileged regression on latent `z` to force dynamics encoding
- Latent space analysis with UMAP and linear/non-linear probes

## Environment and randomization

The experiments use a velocity-controlled Ant environment with the robot constrained to walk in the +X direction.
Randomized dynamics during training include:
- friction scale in `[0.3, 1.5]`
- mass scale in `[0.7, 1.4]`
- action delay in `[0, 4]` simulation steps
- observation delay in `[0, 3]` simulation steps
- external force up to `3.0` N

The evaluation is performed on held-out dynamics configurations, with the student policy conditioned on the learned latent `z`.

## Approach

### Encoder architecture

The adaptation encoder compresses a history of the last `H=50` timesteps into a latent vector of size `d_z=16`.
It processes the flattened `(obs, action)` history with a 3-layer 1D convolutional backbone, then maps the convolutional output to the latent `z` vector.

This design is intentionally compact and temporal: it must discover dynamics information from how the robot state evolved, rather than from privileged state variables.

### Training pipeline

The training pipeline is two-stage:
1. Distillation stage
   - A frozen teacher policy is rolled out in randomized environments.
   - The student policy learns to match the teacher's actions while the encoder produces `z` from history.
   - A privileged dynamics regression loss forces the encoder to encode dynamics into `z`.
2. Fine-tuning stage
   - The student policy is further trained with PPO using `(obs, z)` as input.
   - The encoder remains frozen or is fine-tuned depending on the stage.

### Why privileged regression

Without a dedicated dynamics target, the student could ignore `z` and still mimic the teacher through observation alone.
To avoid this collapse, the model learns an auxiliary linear decoder from `z` to privileged dynamics variables such as friction and mass.
This gives `z` a concrete, interpretable structure.

## Results


The most informative project visuals are the latent-space UMAP plots below. They show how the learned encoder organizes dynamics variations in `z` space.

![UMAP colored by friction](results/latent_analysis/umap_friction.png)

*UMAP embedding of latent `z`, colored by friction scale.*

![UMAP colored by mass](results/latent_analysis/umap_mass.png)

*UMAP embedding of latent `z`, colored by mass scale.*

### Rollout

Rollout visualizations are a useful complement to the latent-space analysis. If you capture rendered episodes or GIFs, save them under `results/` and embed them here with names such as:

- `results/teacher_eval.gif`
- `results/student_eval.gif`


These visualizations help show how the student policy adapts online to new dynamics and how the encoder's latent state affects control behavior.

### Probing dynamics in latent space

Quantitative probes confirm that the encoder has learned a meaningful dynamics representation:
- `friction_scale` is encoded linearly with held-out test performance `R^2 ≈ 0.72`
- Linear and non-linear probes agree to within `0.01`, confirming that the latent representation is effectively linear for friction
- `mass_scale` is not recovered (`R^2 ≈ 0`) under both linear and non-linear probes

> Our conv-encoded latent representation linearly encodes friction with `R^2=0.72` on held out episodes (linear probe matches non-linear probe to within `0.01`, confirming linearity). Mass is not recovered (`R^2 ~ 0` across linear and non-linear probes), which we attribute to the limited number of body bounce cycles observable within the `H=50` history window.

### Published evaluation artifacts

Saved model checkpoints and analysis outputs are available in `results/`:
- `results/teacher_policy.pt`
- `results/student_distilled.pt`
- `results/student_finetuned.pt`
- `results/latent_analysis/umap_friction.png`
- `results/latent_analysis/umap_mass.png`
- `results/latent_analysis/linear_probe_summary.txt`

## How to run

Train the teacher policy:
```bash
python -m training.train_teacher
```

Train the student adaptation model:
```bash
python -m training.train_student
```

Evaluate latent structure and produce plots:
```bash
python -m evaluation.analyze_latent_space
```

Evaluate policies:
```bash
python -m evaluation.eval_teacher_policy --episode 10 --no-randomize --no-render
python -m evaluation.eval_student_policy --episode 10 --no-randomize --no-render
```

## Limitations and future work

- The current setup is restricted to forward walking in the +X direction.
- The history window uses only `H=50` timesteps, which may limit recoverability of dynamics like mass that require longer observation of bounce and contact patterns.
- Future work can extend to full 3D velocity targets `(Vx, Vy, yaw_rate)`, larger history windows, and transfer to more diverse tasks.

## Repository structure

- `envs/` — environment wrappers and randomized dynamics configuration
- `models/` — encoder, student policy, and teacher policy definitions
- `training/` — teacher training, student distillation, and PPO utilities
- `evaluation/` — latent analysis and policy evaluation scripts
- `results/` — saved checkpoints and latent analysis plots
