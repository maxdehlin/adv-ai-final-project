# ANTIDOTE

**Adversarial Noisy Trajectory Inference via Detection Of Trajectories for Trust Estimation**

Modifies Maximum Entropy IRL to be robust against poisoned demonstration datasets by assigning per-trajectory trust weights. Tested on `CarRacing-v3` with discrete actions.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install stable-baselines3 "gymnasium[box2d]>=1.0"
pip install scikit-learn numpy
```

---

## Full Pipeline

### 1. Train the expert policy

Train a PPO agent on the ground-truth CarRacing reward. This is the source of expert demonstrations.

```bash
python -m data_collection.train_expert
```

Default: 2M steps, saves to `models/expert_ppo.zip`. Checkpoints saved every 50k steps.

```bash
# Custom timesteps or output path
python -m data_collection.train_expert --timesteps 3000000 --out models/expert_ppo.zip

# Evaluate an existing model without retraining
python -m data_collection.train_expert --eval-only
python -m data_collection.train_expert --eval-only --render   # opens a window
```

Training logs: `logs/training/stdout.log` and `logs/training/train_log.csv`.

---

### 2. Collect expert demonstrations

Roll out the trained expert policy and save trajectories to disk.

```bash
python -m data_collection.collect_demos --policy expert --n 1000
# saves to data/raw/expert/
```

```bash
# Only keep episodes above a score threshold (recommended for clean expert data)
python -m data_collection.collect_demos --policy expert --n 1000 --min-score 850

# Append to existing data (avoids overwriting)
python -m data_collection.collect_demos --policy expert --n 200 --start-idx 1000
```

---

### 3. Collect poison demonstrations

**Option A — Automatic (random policy):**

```bash
python -m data_collection.collect_demos --policy random --n 300
# saves to data/raw/poison_random/
```

**Option B — Manual (drive badly yourself):**

```bash
python -m data_collection.teleop --n 20
# saves to data/raw/poison_human/
```

Controls: `←` steer left, `→` steer right, `↑` gas, `↓` brake, `R` discard & restart, `Q` quit and save.

Human-provided poison is prioritised over random poison when building datasets.

---

### 4. Build D0–D5 datasets

Mix expert and poison trajectories into six datasets at 0%–50% poison concentration.

```bash
# Preview counts without writing anything
python -m data_collection.build_datasets --preview

# Write manifests to data/datasets/
python -m data_collection.build_datasets
```

Produces `data/datasets/D0.json` through `D5.json`. Each manifest lists trajectory file paths and labels (expert/poison). Default total is 200 trajectories per dataset.

```bash
# Custom total or directories
python -m data_collection.build_datasets --total 200 --expert-dir data/raw/expert --random-dir data/raw/poison_random
```

---

### 5. Visualise policies and trajectories

```bash
# Watch the trained expert drive (3 episodes, live window)
python visualize.py

# Watch the random poison policy
python visualize.py --policy random

# Replay a saved trajectory with action labels overlaid
python visualize.py --trajectory data/raw/expert/traj_0000.npz

# More episodes or a different model
python visualize.py --model models/expert_ppo.zip --episodes 5
```

---

### 6. Run experiments

**No-poison behavior cloning sanity baseline:**

Before interpreting poisoned IRL results, check that clean expert demonstrations
can train a policy at all.

```bash
python -m experiments.no_poison_bc
```

**Preliminary (baseline MaxEnt IRL, no ANTIDOTE):**

Runs two conditions — 0% poison and 25% poison — and compares final RL agent scores.

```bash
python -m experiments.preliminary
```

```bash
# Tune iterations and RL training steps
python -m experiments.preliminary --irl-iters 1000 --rl-steps 500000 --n-eval 5

# Use the richer local-geometry/action-context feature set
python -m experiments.preliminary --features v2
```

Results saved to `results/preliminary/` as JSON files (one per condition).

---

## Project Structure

```
maxent_irl/
├── trajectory.py        # Trajectory dataclass
├── features.py          # CarRacingFeatures: 96×96×3 image → R^8
└── maxent_irl.py        # MaxEntIRL with per-trajectory trust weight hook

data_collection/
├── train_expert.py      # Train PPO expert policy
├── collect_demos.py     # Roll out any policy → .npz trajectory files
├── teleop.py            # Human keyboard play for manual poison demos
└── build_datasets.py    # Construct D0–D5 manifests from raw data

experiments/
└── preliminary.py       # Baseline MaxEnt IRL on 0% and 25% poison

visualize.py             # Watch policies and replay saved trajectories
```

**Data layout (gitignored):**
```
data/raw/expert/         # .npz files from expert rollout
data/raw/poison_random/  # .npz files from random policy
data/raw/poison_human/   # .npz files from human teleop
data/datasets/           # D0.json – D5.json manifests
models/                  # Saved PPO checkpoints
logs/                    # Training logs and CSVs
results/                 # Experiment output JSON files
```

---

## Trust Estimation Methods (coming soon)

| Method | Description |
|--------|-------------|
| β_OD | Outlier Detection — Isolation Forest on trajectory features |
| β_PC | Poison Classifier — binary classifier with clean anchor set |
| β_RC | Reward Consistency — EM-style iterative reweighting |
