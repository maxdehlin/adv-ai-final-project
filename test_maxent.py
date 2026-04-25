"""
Quick smoke test — no gym required.
Generates synthetic expert + background trajectories and verifies the IRL loop.
"""
import numpy as np
from maxent_irl import Trajectory, MaxEntIRL, CarRacingFeaturesV2

np.random.seed(0)

F    = 8   # feature dim
N    = 20  # demo trajectories
M    = 20  # background trajectories (random policy)
T    = 50  # steps per trajectory


def smoke_test_v2_features():
    extractor = CarRacingFeaturesV2()
    img = np.full((96, 96, 3), [100, 200, 100], dtype=np.uint8)  # grass
    img[:84, 38:58] = [100, 100, 100]  # centered road strip

    features = extractor(img, 3)  # gas
    assert features.shape == (extractor.feature_dim,)
    assert np.isfinite(features).all()

    names = extractor.feature_names
    assert features[names.index("action_gas")] == 1.0
    assert features[names.index("gas_on_road")] > 0.0


smoke_test_v2_features()

def make_trajs(n, feature_mean, feature_std):
    trajs = []
    for _ in range(n):
        feat = np.random.randn(T, F) * feature_std + feature_mean
        traj = Trajectory(
            states=np.zeros((T, 96, 96, 3)),
            actions=np.random.randint(0, 5, size=(T,)),
            features=feat,
        )
        traj.compute_feature_sum()
        trajs.append(traj)
    return trajs

# Expert demos: higher road_coverage (feature 0), lower grass (feature 1)
expert_mean = np.array([0.7, 0.1, 0.0, 0.05, 0.05, 0.1, 0.7, 0.0])
demo_trajs  = make_trajs(N, expert_mean, 0.05)

# Background: random policy — lower road coverage, uniform actions
bg_mean    = np.array([0.3, 0.4, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0])
bg_trajs   = make_trajs(M, bg_mean, 0.1)

# --- Baseline: uniform weights ---
model = MaxEntIRL(feature_dim=F, lr=0.05)
history = model.train(demo_trajs, bg_trajs, n_iter=200, verbose=False)
print(f"\nBaseline final theta: {model.theta}")
print("  (positive values = features more common in expert than background)")

# --- ANTIDOTE hook: custom trust weights ---
model.reset()
weights = np.random.rand(N)
weights /= weights.sum()
history_w = model.train(demo_trajs, bg_trajs, weights=weights, n_iter=200, verbose=False)
print(f"\nWeighted final theta: {model.theta}")
