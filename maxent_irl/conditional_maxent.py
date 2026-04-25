"""
State-conditional MaxEnt action model.

This is a diagnostic/training variant for discrete-action environments:

    P(a | s, theta) = exp(theta . phi(s, a)) / sum_a' exp(theta . phi(s, a'))

It is still a maximum-entropy model, but the partition function is over actions
available in the same observed state rather than over whole background
trajectories. For CarRacing-v3 with five discrete actions, this lets us ask the
right local question: does the reward prefer the expert action over the four
counterfactual actions in the same state?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConditionalMaxEntMetrics:
    n: int
    nll: float
    top1_accuracy: float
    expert_margin: float
    gas_argmax_rate: float
    predicted_action_freq: list[float]
    expert_action_freq: list[float]


class ConditionalMaxEntIRL:
    def __init__(self, feature_dim: int, lr: float = 0.5, l2: float = 1e-3):
        self.feature_dim = feature_dim
        self.lr = lr
        self.l2 = l2
        self.theta = np.zeros(feature_dim, dtype=np.float64)

    def logits(self, feature_tensor: np.ndarray) -> np.ndarray:
        return feature_tensor @ self.theta

    def probs(self, feature_tensor: np.ndarray) -> np.ndarray:
        logits = self.logits(feature_tensor)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def nll(
        self,
        feature_tensor: np.ndarray,
        expert_actions: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        probs = self.probs(feature_tensor)
        p_expert = probs[np.arange(len(expert_actions)), expert_actions]
        losses = -np.log(np.maximum(p_expert, 1e-12))
        if sample_weights is None:
            return float(losses.mean())
        weights = np.asarray(sample_weights, dtype=np.float64)
        return float((weights * losses).sum() / weights.sum())

    def gradient(
        self,
        feature_tensor: np.ndarray,
        expert_actions: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        probs = self.probs(feature_tensor)
        expert_features = feature_tensor[np.arange(len(expert_actions)), expert_actions]
        expected_features = (probs[:, :, None] * feature_tensor).sum(axis=1)
        per_sample_grad = expert_features - expected_features
        if sample_weights is None:
            grad = per_sample_grad.mean(axis=0)
        else:
            weights = np.asarray(sample_weights, dtype=np.float64)
            grad = (weights[:, None] * per_sample_grad).sum(axis=0) / weights.sum()
        if self.l2 > 0:
            grad -= self.l2 * self.theta
        return grad

    def step(
        self,
        feature_tensor: np.ndarray,
        expert_actions: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        grad = self.gradient(feature_tensor, expert_actions, sample_weights)
        self.theta += self.lr * grad
        return self.nll(feature_tensor, expert_actions, sample_weights)

    def train(
        self,
        feature_tensor: np.ndarray,
        expert_actions: np.ndarray,
        sample_weights: np.ndarray | None = None,
        n_iter: int = 1000,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> list[float]:
        history = []
        for i in range(n_iter):
            theta_prev = self.theta.copy()
            loss = self.step(feature_tensor, expert_actions, sample_weights)
            history.append(loss)

            if verbose and (i % 100 == 0 or i == n_iter - 1):
                print(f"  iter {i:4d}  nll={loss:.4f}  |theta|={np.linalg.norm(self.theta):.4f}")

            if np.linalg.norm(self.theta - theta_prev) < tol:
                if verbose:
                    print(f"  converged at iter {i}")
                break
        return history

    def evaluate(self, feature_tensor: np.ndarray, expert_actions: np.ndarray) -> ConditionalMaxEntMetrics:
        logits = self.logits(feature_tensor)
        pred = logits.argmax(axis=1)
        n_actions = feature_tensor.shape[1]
        expert_logits = logits[np.arange(len(expert_actions)), expert_actions]
        masked = logits.copy()
        masked[np.arange(len(expert_actions)), expert_actions] = -np.inf
        margins = expert_logits - masked.max(axis=1)

        pred_counts = np.bincount(pred, minlength=n_actions).astype(np.float64)
        expert_counts = np.bincount(expert_actions, minlength=n_actions).astype(np.float64)

        return ConditionalMaxEntMetrics(
            n=int(len(expert_actions)),
            nll=self.nll(feature_tensor, expert_actions),
            top1_accuracy=float(np.mean(pred == expert_actions)),
            expert_margin=float(margins.mean()),
            gas_argmax_rate=float(np.mean(pred == 3)) if n_actions > 3 else float("nan"),
            predicted_action_freq=(pred_counts / pred_counts.sum()).tolist(),
            expert_action_freq=(expert_counts / expert_counts.sum()).tolist(),
        )
