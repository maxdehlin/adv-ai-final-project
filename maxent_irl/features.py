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

    @abstractmethod
    def __call__(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute φ(s, a) for a single (state, action) pair.
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
# CarRacing-v2 feature extractor
#
# CarRacing-v2 observations are 96×96×3 RGB images.
# These hand-crafted features are designed to distinguish expert from
# poisoned behavior without relying on ground-truth reward.
#
# Features (F = 8):
#   0  road_coverage     fraction of image pixels that are road (grey)
#   1  speed_x           horizontal velocity proxy (mean horizontal pixel flow placeholder)
#   2  speed_y           vertical velocity proxy
#   3  steering_magnitude |action[0]|  – how aggressively the agent steers
#   4  gas               action[1]
#   5  brake             action[2]
#   6  grass_coverage    fraction of image pixels that are grass (green)
#   7  center_deviation  how far the car is from the horizontal center of the image
#
# NOTE: speed_x / speed_y require two consecutive frames.  When only one
#       frame is available (first step), they default to 0.
# ---------------------------------------------------------------------------

# Pixel colour thresholds (empirical for CarRacing-v2)
_ROAD_GREY_LOW  = np.array([80,  80,  80])
_ROAD_GREY_HIGH = np.array([130, 130, 130])
_GRASS_GREEN_LOW  = np.array([90, 170, 90])
_GRASS_GREEN_HIGH = np.array([130, 230, 130])


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


class CarRacingFeatures(FeatureExtractor):
    """
    Feature extractor for CarRacing-v2 (96×96×3 observations).

    Usage:
        extractor = CarRacingFeatures()
        traj = extractor.extract_trajectory(traj)
    """

    def __init__(self):
        self._prev_state: np.ndarray = None

    @property
    def feature_dim(self) -> int:
        return 8

    def reset(self):
        """Call at the start of each trajectory to clear the previous-frame cache."""
        self._prev_state = None

    def __call__(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        state:  (96, 96, 3) uint8 image
        action: (3,)  [steering, gas, brake]
        """
        img = state.astype(np.float32)
        n_pixels = img.shape[0] * img.shape[1]

        road_coverage  = _road_mask(state).sum() / n_pixels
        grass_coverage = _grass_mask(state).sum() / n_pixels
        center_dev     = _center_deviation(state)

        # Optical-flow proxy: mean absolute pixel difference (very rough speed estimate)
        if self._prev_state is not None:
            diff = np.abs(img - self._prev_state.astype(np.float32))
            speed_x = diff[:, :, 0].mean() / 255.0
            speed_y = diff[:, :, 1].mean() / 255.0
        else:
            speed_x = 0.0
            speed_y = 0.0

        self._prev_state = state

        steering_mag = float(abs(action[0]))
        gas          = float(np.clip(action[1], 0, 1))
        brake        = float(np.clip(action[2], 0, 1))

        return np.array([
            road_coverage,
            speed_x,
            speed_y,
            steering_mag,
            gas,
            brake,
            grass_coverage,
            center_dev,
        ], dtype=np.float64)

    def extract_trajectory(self, traj: Trajectory) -> Trajectory:
        """Override to reset the frame cache at the start of each trajectory."""
        self.reset()
        return super().extract_trajectory(traj)
