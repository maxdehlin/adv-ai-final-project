"""
Feature extractors: φ(s, a) → R^F

Swap in a different FeatureExtractor subclass to change the reward parameterization
without touching the IRL algorithm at all.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List
from .trajectory import Trajectory


class FeatureExtractor(ABC):
    """Abstract base class. All extractors must implement __call__ and feature_dim."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality F of the feature vector."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Names for each feature dimension."""
        return [f"feature_{i}" for i in range(self.feature_dim)]

    @abstractmethod
    def __call__(self, state: np.ndarray, action) -> np.ndarray:
        """
        Compute φ(s, a) for a single (state, action) pair.
        action may be an int (discrete) or np.ndarray (continuous).
        Returns a 1-D array of shape (F,).
        """
        ...

    def extract_trajectory(self, traj: Trajectory) -> Trajectory:
        """
        Fill traj.features and traj.feature_sum in-place.
        Returns the same trajectory for chaining.
        """
        traj.features = np.array([
            self(s, a) for s, a in zip(traj.states, traj.actions)
        ])
        traj.compute_feature_sum()
        return traj

    def extract_all(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Apply extract_trajectory to a list of trajectories."""
        return [self.extract_trajectory(t) for t in trajectories]


# ---------------------------------------------------------------------------
# CarRacing-v3 feature extractor  (continuous=False — 5 discrete actions)
#
# Actions:  0=do nothing  1=steer left  2=steer right  3=gas  4=brake
#
# CarRacing-v3 observations are 96×96×3 RGB images.
# These hand-crafted features are designed to distinguish expert from
# poisoned behavior without relying on ground-truth reward.
#
# Features (F = 8):
#   0  road_coverage     fraction of image pixels that are road (grey)
#   1  grass_coverage    fraction of image pixels that are grass (green)
#   2  center_deviation  how far road centre is from image centre [-1, 1]
#   3  action_nothing    1 if action == 0, else 0
#   4  action_steer_left  1 if action == 1, else 0
#   5  action_steer_right 1 if action == 2, else 0
#   6  action_gas         1 if action == 3, else 0
#   7  action_brake       1 if action == 4, else 0
# ---------------------------------------------------------------------------

# Pixel colour thresholds (empirical for CarRacing-v3)
_ROAD_GREY_LOW  = np.array([80,  80,  80])
_ROAD_GREY_HIGH = np.array([130, 130, 130])
_GRASS_GREEN_LOW  = np.array([90, 170, 90])
_GRASS_GREEN_HIGH = np.array([130, 230, 130])

V1_FEATURE_NAMES = [
    "road_coverage",
    "grass_coverage",
    "center_deviation",
    "action_nothing",
    "action_steer_left",
    "action_steer_right",
    "action_gas",
    "action_brake",
]

V2_EXTRA_FEATURE_NAMES = [
    "abs_center_deviation",
    "visual_motion",
    "road_coverage_delta",
    "grass_coverage_delta",
    "center_deviation_delta",
    "same_action_as_previous",
]


def _road_mask(img: np.ndarray) -> np.ndarray:
    return np.all((img >= _ROAD_GREY_LOW) & (img <= _ROAD_GREY_HIGH), axis=-1)


def _grass_mask(img: np.ndarray) -> np.ndarray:
    return np.all((img >= _GRASS_GREEN_LOW) & (img <= _GRASS_GREEN_HIGH), axis=-1)


def _center_deviation(img: np.ndarray) -> float:
    """Horizontal deviation of road centre from image centre (normalised to [-1, 1])."""
    road = _road_mask(img)
    cols_with_road = np.where(road.any(axis=0))[0]
    if len(cols_with_road) == 0:
        return 1.0          # completely off-road → max deviation
    road_centre = cols_with_road.mean()
    img_centre = img.shape[1] / 2.0
    return float((road_centre - img_centre) / img_centre)


def _visual_motion(state: np.ndarray, prev_state: np.ndarray | None) -> float:
    """Mean absolute frame difference, normalized to [0, 1]."""
    if prev_state is None:
        return 0.0
    current = np.asarray(state, dtype=np.float32)
    previous = np.asarray(prev_state, dtype=np.float32)
    return float(np.mean(np.abs(current - previous)) / 255.0)


N_ACTIONS = 5  # do nothing, steer left, steer right, gas, brake


class CarRacingFeatures(FeatureExtractor):
    """
    Feature extractor for CarRacing-v3 with continuous=False (5 discrete actions).

    Usage:
        extractor = CarRacingFeatures()
        traj = extractor.extract_trajectory(traj)
    """

    @property
    def feature_dim(self) -> int:
        return 8  # 3 visual features + 5 action one-hot

    @property
    def feature_names(self) -> list[str]:
        return V1_FEATURE_NAMES.copy()

    def __call__(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        state:  (96, 96, 3) uint8 image
        action: int in {0, 1, 2, 3, 4}
        """
        n_pixels = state.shape[0] * state.shape[1]

        road_coverage  = _road_mask(state).sum() / n_pixels
        grass_coverage = _grass_mask(state).sum() / n_pixels
        center_dev     = _center_deviation(state)

        one_hot = np.zeros(N_ACTIONS, dtype=np.float64)
        one_hot[int(action)] = 1.0

        return np.concatenate([
            [road_coverage, grass_coverage, center_dev],
            one_hot,
        ])


class CarRacingFeaturesV2(CarRacingFeatures):
    """
    Backward-compatible extension of CarRacingFeatures.

    The first 8 dimensions are exactly the V1 features. Extra dimensions add
    temporal and derived signals that help identify low-motion/stuck behavior,
    especially stop-poison trajectories.
    """

    @property
    def feature_dim(self) -> int:
        return len(V1_FEATURE_NAMES) + len(V2_EXTRA_FEATURE_NAMES)

    @property
    def feature_names(self) -> list[str]:
        return V1_FEATURE_NAMES.copy() + V2_EXTRA_FEATURE_NAMES.copy()

    def __call__(
        self,
        state: np.ndarray,
        action: int,
        prev_state: np.ndarray | None = None,
        prev_action: int | None = None,
    ) -> np.ndarray:
        v1 = super().__call__(state, action)

        road_coverage = float(v1[0])
        grass_coverage = float(v1[1])
        center_dev = float(v1[2])

        if prev_state is None:
            prev_road = road_coverage
            prev_grass = grass_coverage
            prev_center = center_dev
        else:
            n_pixels = prev_state.shape[0] * prev_state.shape[1]
            prev_road = float(_road_mask(prev_state).sum() / n_pixels)
            prev_grass = float(_grass_mask(prev_state).sum() / n_pixels)
            prev_center = _center_deviation(prev_state)

        same_action = 0.0 if prev_action is None else float(int(action) == int(prev_action))
        extras = np.array(
            [
                abs(center_dev),
                _visual_motion(state, prev_state),
                road_coverage - prev_road,
                grass_coverage - prev_grass,
                center_dev - prev_center,
                same_action,
            ],
            dtype=np.float64,
        )
        return np.concatenate([v1, extras])

    def extract_trajectory(self, traj: Trajectory) -> Trajectory:
        features = []
        prev_state = None
        prev_action = None
        for state, action in zip(traj.states, traj.actions):
            features.append(self(state, int(action), prev_state=prev_state, prev_action=prev_action))
            prev_state = state
            prev_action = int(action)

        traj.features = np.array(features)
        traj.compute_feature_sum()
        return traj
