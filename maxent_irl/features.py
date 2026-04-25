"""
Feature extractors: φ(s, a) → R^F

Swap in a different FeatureExtractor subclass to change the reward parameterization
without touching the IRL algorithm at all.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple
from .trajectory import Trajectory


class FeatureExtractor(ABC):
    """Abstract base class. All extractors must implement __call__ and feature_dim."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality F of the feature vector."""
        ...

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


# ---------------------------------------------------------------------------
# CarRacing-v3 feature extractor V2
#
# The original extractor is intentionally small, but most of its signal comes
# from global action frequencies. V2 adds local track geometry and action-state
# interaction features so a linear reward can represent useful driving rules:
# accelerate when pointed at/near the road, avoid accelerating off road, and
# steer toward the visible track.
# ---------------------------------------------------------------------------

_DRIVING_VIEW_BOTTOM_FRAC = 0.875  # crop out the lower dashboard area


def _driving_view(img: np.ndarray) -> np.ndarray:
    bottom = max(1, int(img.shape[0] * _DRIVING_VIEW_BOTTOM_FRAC))
    return img[:bottom]


def _roi_fraction(mask: np.ndarray, row0: float, row1: float, col0: float, col1: float) -> float:
    """Mean of a boolean mask inside a fractional image region."""
    h, w = mask.shape
    r0 = int(np.clip(round(row0 * h), 0, h))
    r1 = int(np.clip(round(row1 * h), 0, h))
    c0 = int(np.clip(round(col0 * w), 0, w))
    c1 = int(np.clip(round(col1 * w), 0, w))
    if r1 <= r0 or c1 <= c0:
        return 0.0
    return float(mask[r0:r1, c0:c1].mean())


def _band_center(road: np.ndarray, row0: float, row1: float) -> Tuple[Optional[float], float]:
    """Return weighted road center x and mean road coverage for a vertical band."""
    h, w = road.shape
    r0 = int(np.clip(round(row0 * h), 0, h))
    r1 = int(np.clip(round(row1 * h), 0, h))

    band = road[r0:r1]
    if band.size == 0:
        return None, 0.0

    counts = band.sum(axis=1).astype(np.float64)
    mean_coverage = float((counts / w).mean())
    valid = counts >= 3
    if not np.any(valid):
        return None, mean_coverage

    x = np.arange(w, dtype=np.float64)
    centers = (band[valid].astype(np.float64) @ x) / counts[valid]
    weights = counts[valid]
    return float(np.average(centers, weights=weights)), mean_coverage


def _track_geometry_from_road(road: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate signed center deviation, absolute deviation, and road heading.

    center_deviation > 0 means the visible road center is to the right.
    heading > 0 means the road center moves right as we look farther ahead.
    """
    _, w = road.shape

    near_center, near_coverage = _band_center(road, 0.55, 0.98)
    far_center, far_coverage = _band_center(road, 0.18, 0.55)

    if near_center is None:
        signed_dev = 0.0
        abs_dev = 1.0
    else:
        signed_dev = float(np.clip((near_center - (w / 2.0)) / (w / 2.0), -1.0, 1.0))
        abs_dev = abs(signed_dev)

    if near_center is None or far_center is None or near_coverage < 0.02 or far_coverage < 0.02:
        heading = 0.0
    else:
        heading = float(np.clip((far_center - near_center) / (w / 2.0), -1.0, 1.0))

    return signed_dev, abs_dev, heading


def _track_geometry(img: np.ndarray) -> Tuple[float, float, float]:
    return _track_geometry_from_road(_road_mask(_driving_view(img)))


class CarRacingFeaturesV2(FeatureExtractor):
    """
    Richer CarRacing-v3 features for learning a reward that tracks completion.

    Features are still computed from a single image/action pair, so this class
    can be used both for offline IRL and as the online reward wrapper feature
    extractor during PPO training.

    Feature groups:
      - local road/grass visibility around the car and lookahead region
      - signed/absolute road-center deviation and road heading
      - action priors
      - action-state interactions for gas, brake, and steering alignment
    """

    feature_names = (
        "road_near",
        "grass_near",
        "road_ahead",
        "grass_ahead",
        "center_deviation",
        "abs_center_deviation",
        "road_heading",
        "abs_road_heading",
        "action_nothing",
        "action_steer_left",
        "action_steer_right",
        "action_gas",
        "action_brake",
        "gas_on_road",
        "gas_off_road",
        "brake_on_road",
        "brake_off_road",
        "steer_toward_center",
        "steer_away_center",
        "steer_with_heading",
        "steer_against_heading",
    )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def __call__(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        state:  (96, 96, 3) uint8 image
        action: int in {0, 1, 2, 3, 4}
        """
        view = _driving_view(state)
        road = _road_mask(view)
        grass = _grass_mask(view)

        # Local patches are chosen around the car/front-of-car and the visible
        # lookahead region. They are more useful than whole-frame color counts.
        road_near = _roi_fraction(road, 0.60, 0.88, 0.32, 0.68)
        grass_near = _roi_fraction(grass, 0.60, 0.88, 0.32, 0.68)
        road_ahead = _roi_fraction(road, 0.25, 0.62, 0.20, 0.80)
        grass_ahead = _roi_fraction(grass, 0.25, 0.62, 0.20, 0.80)

        center_dev, abs_center_dev, heading = _track_geometry_from_road(road)
        abs_heading = abs(heading)

        one_hot = np.zeros(N_ACTIONS, dtype=np.float64)
        a = int(action)
        if a < 0 or a >= N_ACTIONS:
            raise ValueError(f"CarRacingFeaturesV2 expected action in [0, {N_ACTIONS - 1}], got {action}")
        one_hot[a] = 1.0

        action_nothing, action_left, action_right, action_gas, action_brake = one_hot

        off_road_near = 1.0 - road_near
        gas_on_road = action_gas * road_near
        gas_off_road = action_gas * off_road_near
        brake_on_road = action_brake * road_near
        brake_off_road = action_brake * off_road_near

        steer_toward_center = (
            action_left * max(-center_dev, 0.0)
            + action_right * max(center_dev, 0.0)
        )
        steer_away_center = (
            action_left * max(center_dev, 0.0)
            + action_right * max(-center_dev, 0.0)
        )
        steer_with_heading = (
            action_left * max(-heading, 0.0)
            + action_right * max(heading, 0.0)
        )
        steer_against_heading = (
            action_left * max(heading, 0.0)
            + action_right * max(-heading, 0.0)
        )

        return np.array([
            road_near,
            grass_near,
            road_ahead,
            grass_ahead,
            center_dev,
            abs_center_dev,
            heading,
            abs_heading,
            action_nothing,
            action_left,
            action_right,
            action_gas,
            action_brake,
            gas_on_road,
            gas_off_road,
            brake_on_road,
            brake_off_road,
            steer_toward_center,
            steer_away_center,
            steer_with_heading,
            steer_against_heading,
        ], dtype=np.float64)
