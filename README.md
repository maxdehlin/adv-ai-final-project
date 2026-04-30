# ANTIDOTE

MaxEnt IRL with per-trajectory trust weights, robust to poisoned demonstrations. Tested on `CarRacing-v3` (discrete actions).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision stable-baselines3 "gymnasium[box2d]>=1.0" scikit-learn numpy
```

## Usage

```bash
# Train expert policy
python -m data_collection.train_expert

# Collect demos
python -m data_collection.collect_demos --policy expert --n 1000 --min-score 850
python -m data_collection.collect_demos --policy random --n 300
python -m data_collection.teleop --n 20   # manual poison (arrow keys)

# Replay a trajectory
python visualize.py --trajectory data/raw/expert/traj_0000.npz

# Run experiment
python -m experiments.antidote --dataset D3 --poison-name poison_stop
```

Results saved to `results/antidote/{poison_name}/{dataset}/`.

## Methods

| | Description |
|--|--|
| baseline | Unweighted MaxEnt IRL |
| β_OD | KNN outlier detection on action summaries |
| β_AE | Autoencoder reconstruction error |
| β_RC | EM reward-consistency reweighting |
