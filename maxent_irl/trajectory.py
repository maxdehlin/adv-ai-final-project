from dataclasses import dataclass, field
import numpy as np


@dataclass
class Trajectory:
    """
    A single demonstration trajectory.

    states:       (T, *state_shape)  - raw observations at each timestep
    actions:      (T, *action_shape) - actions taken at each timestep
    features:     (T, F)             - feature vectors φ(s_t, a_t), filled by FeatureExtractor
    feature_sum:  (F,)               - Σ_t φ(s_t, a_t), the trajectory feature count vector
    """
    states: np.ndarray
    actions: np.ndarray
    features: np.ndarray = field(default=None)
    feature_sum: np.ndarray = field(default=None)

    def __len__(self):
        return len(self.states)

    def compute_feature_sum(self):
        """Cache the trajectory-level feature count vector Σ_t φ(s_t, a_t)."""
        assert self.features is not None, "Features must be set before calling compute_feature_sum."
        self.feature_sum = self.features.sum(axis=0)
        return self.feature_sum
