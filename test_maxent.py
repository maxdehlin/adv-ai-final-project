"""
Quick smoke test — no gym required.
Generates synthetic trajectories and verifies the IRL training loop runs.
"""
import numpy as np
from maxent_irl import Trajectory, MaxEntIRL

np.random.seed(0)

F = 8   # feature dim
N = 20  # trajectories
T = 50  # steps per trajectory

# Synthetic trajectories with random features
trajectories = []
for _ in range(N):
    feat = np.random.rand(T, F)
    traj = Trajectory(
        states=np.zeros((T, 96, 96, 3)),   # placeholder
        actions=np.zeros((T, 3)),
        features=feat,
    )
    traj.compute_feature_sum()
    trajectories.append(traj)

# --- Baseline: uniform weights ---
model = MaxEntIRL(feature_dim=F, lr=0.01)
history = model.train(trajectories, n_iter=200, verbose=True)
print(f"\nBaseline final θ: {model.theta}")

# --- ANTIDOTE hook: custom trust weights ---
model.reset()
weights = np.random.rand(N)          # replace with β_OD / β_PC / β_RC output
weights /= weights.sum()
history_w = model.train(trajectories, weights=weights, n_iter=200, verbose=True)
print(f"\nWeighted final θ: {model.theta}")
