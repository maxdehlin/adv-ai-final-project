"""
Maximum Entropy Inverse Reinforcement Learning
Ziebart et al. (2008) — contrastive trajectory-level approximation.

Reward model:  r_θ(s, a) = θ · φ(s, a)          (linear, F parameters)

Trajectory return:  R(τ) = Σ_t r_θ(s_t, a_t) = θ · f(τ)

The partition function Z(θ) is approximated over a fixed set of BACKGROUND
trajectories (random policy rollouts), NOT the demonstration set itself.
This is required for a non-degenerate gradient signal.

Why: if Z is computed over the same demos being optimised, the gradient is
exactly zero at θ=0 (uniform softmax already matches empirical feature mean).
Using a separate background set breaks this symmetry and gives a meaningful
reward that distinguishes expert behavior from random behavior.

Objective (demo set D, background set B):
    L(θ) = Σ_{τ∈D} log P(τ | θ)
    P(τ | θ) = exp(R(τ)) / Z(θ),   Z(θ) = Σ_{τ'∈B} exp(R(τ'))

Gradient:
    ∇_θ L = Σ_{τ∈D} f(τ) - |D| · E_B[f]
    where E_B[f] = Σ_{τ'∈B} P(τ'|θ) f(τ')

Weighted variant (ANTIDOTE hook — pass weights to train / step):
    ∇_θ L_w = Σ_i w_i f(τ_i) - W · E_B[f]
    where W = Σ_i w_i
"""

import numpy as np
from typing import List, Optional, Callable
from .trajectory import Trajectory


class MaxEntIRL:
    """
    Linear MaxEnt IRL with background-trajectory partition function.

    Parameters
    ----------
    feature_dim : int
        Dimensionality F of the feature vector φ(s, a).
    lr : float
        Gradient ascent learning rate.
    l2 : float
        L2 regularisation on θ (keeps weights bounded; set 0 to disable).
    """

    def __init__(self, feature_dim: int, lr: float = 0.01, l2: float = 1e-4):
        self.feature_dim = feature_dim
        self.lr = lr
        self.l2 = l2
        self.theta = np.zeros(feature_dim)

    # ------------------------------------------------------------------
    # Core reward
    # ------------------------------------------------------------------

    def reward(self, features: np.ndarray) -> np.ndarray:
        """r_θ(s, a) = θ · φ(s, a).  features: (..., F) → scalar(s)."""
        return features @ self.theta

    def trajectory_return(self, traj: Trajectory) -> float:
        """R(τ) = θ · f(τ)."""
        return float(self.theta @ traj.feature_sum)

    # ------------------------------------------------------------------
    # Distribution over background trajectories  P(τ | θ)
    # ------------------------------------------------------------------

    def _bg_log_probs(self, background: List[Trajectory]) -> np.ndarray:
        """log P(τ_j | θ) for each background trajectory (logsumexp-stable)."""
        returns = np.array([self.trajectory_return(t) for t in background])
        log_Z = _logsumexp(returns)
        return returns - log_Z

    def _bg_probs(self, background: List[Trajectory]) -> np.ndarray:
        """P(τ_j | θ) over background set — (N_bg,) sums to 1."""
        return np.exp(self._bg_log_probs(background))

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def gradient(
        self,
        demo_trajs: List[Trajectory],
        background_trajs: List[Trajectory],
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute ∇_θ L (or ∇_θ L_w with trust weights).

        demo_trajs       : demonstrations (expert + possible poison)
        background_trajs : random-policy rollouts used for partition function Z
        weights          : (N_demo,) trust weights in [0,1], or None for uniform
                           ← THIS is the hook for ANTIDOTE trust estimation.

        Returns: (F,) gradient vector (ascent direction).
        """
        demo_features = np.array([t.feature_sum for t in demo_trajs])   # (N, F)
        bg_features   = np.array([t.feature_sum for t in background_trajs])  # (M, F)

        # --- empirical (weighted) feature expectations from demos ---
        if weights is None:
            f_empirical  = demo_features.mean(axis=0)
            total_weight = float(len(demo_trajs))
        else:
            w = np.asarray(weights, dtype=np.float64)
            W = w.sum()
            if W < 1e-12:
                raise ValueError("Sum of weights is effectively zero.")
            f_empirical  = (w[:, None] * demo_features).sum(axis=0) / W
            total_weight = W

        # --- expected feature counts under P(τ | θ) over background set ---
        bg_probs = self._bg_probs(background_trajs)               # (M,)
        f_expected = (bg_probs[:, None] * bg_features).sum(axis=0)  # (F,)

        grad = total_weight * (f_empirical - f_expected)

        if self.l2 > 0:
            grad -= self.l2 * self.theta

        return grad

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(
        self,
        demo_trajs: List[Trajectory],
        background_trajs: List[Trajectory],
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """One gradient-ascent step. Returns log-likelihood diagnostic."""
        grad = self.gradient(demo_trajs, background_trajs, weights)
        self.theta += self.lr * grad

        # Diagnostic: mean log P(τ_demo | θ) approximated via background Z
        log_Z   = _logsumexp(np.array([self.trajectory_return(t) for t in background_trajs]))
        returns = np.array([self.trajectory_return(t) for t in demo_trajs])
        log_probs = returns - log_Z
        if weights is None:
            return float(log_probs.mean())
        else:
            w = np.asarray(weights)
            return float((w * log_probs).sum() / w.sum())

    def train(
        self,
        demo_trajs: List[Trajectory],
        background_trajs: List[Trajectory],
        weights: Optional[np.ndarray] = None,
        n_iter: int = 1000,
        tol: float = 1e-6,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        verbose: bool = False,
    ) -> List[float]:
        """
        Run gradient ascent for up to n_iter steps.

        Parameters
        ----------
        demo_trajs       : demonstrations (may contain poison)
        background_trajs : random-policy rollouts for partition function
        weights          : (N_demo,) trust weights, or None for baseline
        n_iter           : max gradient steps
        tol              : stop early if ||Δθ|| < tol
        verbose          : print progress every 100 steps
        """
        _assert_features_extracted(demo_trajs)
        _assert_features_extracted(background_trajs)

        history = []
        for i in range(n_iter):
            theta_prev = self.theta.copy()
            ll = self.step(demo_trajs, background_trajs, weights)
            history.append(ll)

            if callback is not None:
                callback(i, ll, self.theta)

            if verbose and (i % 100 == 0 or i == n_iter - 1):
                print(f"  iter {i:4d}  ll={ll:.4f}  |θ|={np.linalg.norm(self.theta):.4f}")

            if np.linalg.norm(self.theta - theta_prev) < tol:
                if verbose:
                    print(f"  Converged at iter {i}.")
                break

        return history

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def score_trajectory(self, traj: Trajectory) -> float:
        """R(τ) = θ · f(τ)."""
        return self.trajectory_return(traj)

    def reset(self):
        """Re-initialise θ to zero."""
        self.theta = np.zeros(self.feature_dim)

    def __repr__(self):
        return (
            f"MaxEntIRL(F={self.feature_dim}, lr={self.lr}, l2={self.l2})\n"
            f"  θ = {np.array2string(self.theta, precision=4)}"
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _logsumexp(x: np.ndarray) -> float:
    c = x.max()
    return float(c + np.log(np.exp(x - c).sum()))


def _assert_features_extracted(trajectories: List[Trajectory]):
    for i, t in enumerate(trajectories):
        if t.feature_sum is None:
            raise ValueError(
                f"Trajectory {i} has no feature_sum. "
                "Run FeatureExtractor.extract_all(trajectories) first."
            )
