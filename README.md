# ANTIDOTE

**Adversarial Noisy Trajectory Inference via Detection Of Trajectories for Trust Estimation**

Modifies Maximum Entropy IRL to be robust against poisoned demonstration datasets by assigning per-trajectory trust weights.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision
pip install stable-baselines3[extra] gymnasium[box2d]
pip install scikit-learn numpy
```

---

## Data Collection

### 1. Train the expert policy

```bash
python -m data_collection.train_expert
```

Trains PPO on `CarRacing-v3` (discrete actions) for 2M steps. Saves model to `models/expert_ppo.zip`. To customise:

```bash
python -m data_collection.train_expert --timesteps 3000000 --out models/expert_ppo.zip
```

Evaluate an existing model without retraining:

```bash
python -m data_collection.train_expert --eval-only --render
```

### 2. Collect expert demonstrations

```bash
python -m data_collection.collect_demos --policy expert --n 200
# saves to data/raw/expert/
```

### 3. Generate poison demonstrations

**Automatic (random policy):**

```bash
python -m data_collection.collect_demos --policy random --n 100
# saves to data/raw/poison_random/
```

**Manual (drive it yourself):**

```bash
python -m data_collection.teleop --n 20
# saves to data/raw/poison_human/
```

Controls: `← →` steer, `↑` gas, `↓` brake, `R` restart episode, `Q` quit.

Human-provided poison is prioritised over random poison when building datasets.

### 4. Build D0–D5 datasets

```bash
python -m data_collection.build_datasets --preview   # check counts without writing
python -m data_collection.build_datasets             # write manifests to data/datasets/
```

Produces six JSON manifests `D0.json`–`D5.json` with 0%–50% poison in 10% increments (200 trajectories each).

---

## Project Structure

```
maxent_irl/
├── trajectory.py       # Trajectory dataclass
├── features.py         # CarRacingFeatures: 96×96×3 image → R^8
└── maxent_irl.py       # MaxEntIRL with per-trajectory trust weight hook

data_collection/
├── train_expert.py     # Train PPO expert
├── collect_demos.py    # Roll out any policy → .npz trajectories
├── teleop.py           # Human keyboard play
└── build_datasets.py   # Construct D0–D5 manifests
```

---

## Trust Estimation Methods (coming soon)

| Method | Description |
|--------|-------------|
| β_OD | Outlier Detection — Isolation Forest on trajectory features |
| β_PC | Poison Classifier — binary classifier with clean anchor set |
| β_RC | Reward Consistency — EM-style iterative reweighting |
