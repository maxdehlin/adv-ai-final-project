"""
Maximum Entropy Inverse Reinforcement Learning
Ziebart et al. (2008) — trajectory-level softmax approximation for continuous spaces.

Reward model:  r_θ(s, a) = θ · φ(s, a)          (linear, F parameters)

Trajectory return:  R(τ) = Σ_t r_θ(s_t, a_t) = θ · f(τ)
                    where f(τ) = Σ_t φ(s_t, a_t)

Trajectory distribution (softmax over finite demonstration set):
    P(τ_j | θ) = exp(R(τ_j)) / Z(θ),   Z(θ) = Σ_k exp(R(τ_k))

Log-likelihood:
    L(θ) = Σ_i log P(τ_i | θ)
          = Σ_i [R(τ_i)] - N · log Z(θ)

Gradient:
    ∇_θ L = Σ_i f(τ_i) - N · E_P[f(τ)]
           = N · (f_emp - f_expected)

Weighted variant (ANTIDOTE hook — pass weights to train / step):
    L_w(θ) = Σ_i w_i log P(τ_i | θ)
    ∇_θ L_w = Σ_i w_i f(τ_i) - (Σ_i w_i) · E_P[f(τ)]
             = W · (f_emp_w - f_expected)
    where W = Σ_i w_i and f_emp_w = Σ_i w_i f(τ_i) / W
"""

import numpy as np
from typing import List, Optional, Callable
from .trajectory import Trajectory


class MaxEntIRL:
    """
    Linear MaxEnt IRL with support for per-trajectory trust weights.

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

        # Reward parameters θ ∈ R^F  (initialise near zero with small noise)
        self.theta = np.zeros(feature_dim)

    # ------------------------------------------------------------------
    # Core reward
    # ------------------------------------------------------------------

    def reward(self, features: np.ndarray) -> np.ndarray:
        """
        r_θ(s, a) = θ · φ(s, a).

        features : (..., F) — broadcast-friendly.
        Returns  : (...)    scalar reward per feature vector.
        """
        return features @ self.theta

    def trajectory_return(self, traj: Trajectory) -> float:
        """R(τ) = θ · f(τ) = Σ_t r_θ(s_t, a_t)."""
        return float(self.theta @ traj.feature_sum)

    # ------------------------------------------------------------------
    # Distribution over trajectories  P(τ | θ)
    # ------------------------------------------------------------------

    def trajectory_log_probs(self, trajectories: List[Trajectory]) -> np.ndarray:
        """
        Compute log P(τ_j | θ) for each trajectory.

        Uses numerically stable log-sum-exp:
            log P(τ_j) = R(τ_j) - log Σ_k exp(R(τ_k))

        Returns: (N,) array of log-probabilities.
        """
        returns = np.array([self.trajectory_return(t) for t in trajectories])
        log_Z = _logsumexp(returns)
        return returns - log_Z

    def trajectory_probs(self, trajectories: List[Trajectory]) -> np.ndarray:
        """P(τ_j | θ)  — (N,) array, sums to 1."""
        return np.exp(self.trajectory_log_probs(trajectories))

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def gradient(
        self,
        trajectories: List[Trajectory],
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute ∇_θ L (or ∇_θ L_w with trust weights).

        weights : (N,) array in [0, 1], or None for uniform weights.
                  ← THIS is the hook for ANTIDOTE trust estimation.

        Returns: (F,) gradient vector (points in ascent direction).
        """
        N = len(trajectories)
        feature_sums = np.array([t.feature_sum for t in trajectories])  # (N, F)

        # --- empirical (weighted) feature expectations ---
        if weights is None:
            f_empirical = feature_sums.mean(axis=0)          # (F,)
            total_weight = float(N)
        else:
            w = np.asarray(weights, dtype=np.float64)
            W = w.sum()
            if W < 1e-12:
                raise ValueError("Sum of weights is effectively zero.")
            f_empirical = (w[:, None] * feature_sums).sum(axis=0) / W  # (F,)
            total_weight = W

        # --- expected feature counts under P(τ | θ) ---
        probs = self.trajectory_probs(trajectories)          # (N,)
        f_expected = (probs[:, None] * feature_sums).sum(axis=0)  # (F,)

        # ∇ = W · (f_emp - f_exp)  — scale by total weight to keep lr invariant
        grad = total_weight * (f_empirical - f_expected)

        # L2 regularisation (gradient of -½ λ ||θ||²)
        if self.l2 > 0:
            grad -= self.l2 * self.theta

        return grad

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(
        self,
        trajectories: List[Trajectory],
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        One gradient-ascent update step.

        Returns the current log-likelihood (or weighted log-likelihood) as a
        scalar diagnostic — useful for convergence monitoring.
        """
        grad = self.gradient(trajectories, weights)
        self.theta += self.lr * grad

        # Diagnostic: (weighted) log-likelihood
        log_probs = self.trajectory_log_probs(trajectories)
        if weights is None:
            return float(log_probs.sum())
        else:
            return float((np.asarray(weights) * log_probs).sum())

    def train(
        self,
        trajectories: List[Trajectory],
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
        trajectories : list of Trajectory (features must already be extracted)
        weights      : (N,) trust weights, or None for baseline MaxEnt IRL
        n_iter       : maximum number of gradient steps
        tol          : stop early if ||Δθ|| < tol
        callback     : optional fn(iter, log_likelihood, theta) called each step
        verbose      : print progress every 100 steps

        Returns
        -------
        history : list of log-likelihood values (one per step)
        """
        _assert_features_extracted(trajectories)

        history = []
        for i in range(n_iter):
            theta_prev = self.theta.copy()
            ll = self.step(trajectories, weights)
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
        """Unnormalised trajectory return R(τ) = θ · f(τ)."""
        return self.trajectory_return(traj)

    def reset(self):
        """Re-initialise θ to zero (for retraining from scratch)."""
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
    """Numerically stable log Σ exp(x_i)."""
    c = x.max()
    return float(c + np.log(np.exp(x - c).sum()))


def _assert_features_extracted(trajectories: List[Trajectory]):
    for i, t in enumerate(trajectories):
        if t.feature_sum is None:
            raise ValueError(
                f"Trajectory {i} has no feature_sum. "
                "Run FeatureExtractor.extract_all(trajectories) first."
            )
