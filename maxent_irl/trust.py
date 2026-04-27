"""
Trajectory trust utilities for ANTIDOTE.

This module keeps the lightweight, non-neural trust estimators and shared
trajectory-summary helpers. The autoencoder method lives in
maxent_irl.autoencoder_trust because it depends on torch.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

N_ACTIONS = 5
_EPS = 1e-8
_LOGISTIC_CLIP = 60.0

SummaryMode = Literal["action", "v2"]


# ---------------------------------------------------------------------------
# Trajectory summaries
# ---------------------------------------------------------------------------


def action_summary(actions: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Summarize a trajectory using action statistics.

    These features catch scripted poison such as always-brake, random actions,
    and expert-then-bad trajectories whose second half changes behavior.
    """
    actions = _as_actions(actions)

    split = max(1, len(actions) // 2)
    first = actions[:split]
    second = actions[split:] if split < len(actions) else first

    overall_freq = _action_freq(actions)
    first_freq = _action_freq(first)
    second_freq = _action_freq(second)
    freq_delta = second_freq - first_freq

    switch_rate = float(np.mean(actions[1:] != actions[:-1])) if len(actions) > 1 else 0.0
    longest_run = _longest_run_fraction(actions)

    values = np.concatenate(
        [
            [len(actions) / 1000.0],
            overall_freq,
            first_freq,
            second_freq,
            freq_delta,
            [
                _entropy(overall_freq),
                _entropy(first_freq),
                _entropy(second_freq),
                switch_rate,
                longest_run,
            ],
        ]
    ).astype(np.float64)

    names = (
        ["length_frac"]
        + [f"action_{a}_freq" for a in range(N_ACTIONS)]
        + [f"first_action_{a}_freq" for a in range(N_ACTIONS)]
        + [f"second_action_{a}_freq" for a in range(N_ACTIONS)]
        + [f"delta_action_{a}_freq" for a in range(N_ACTIONS)]
        + [
            "action_entropy",
            "first_action_entropy",
            "second_action_entropy",
            "action_switch_rate",
            "longest_action_run_frac",
        ]
    )
    return values, names


def trajectory_summary(
    states: np.ndarray,
    actions: np.ndarray,
    mode: SummaryMode = "action",
    frame_stride: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a trajectory into a fixed-length trust-estimation vector.

    mode="action" is fast and uses action statistics only.
    mode="v2" appends sampled CarRacingFeatures mean/std summaries.
    """
    summary, names = action_summary(actions)
    if mode == "action":
        return summary, names
    if mode != "v2":
        raise ValueError(f"Unknown summary mode: {mode}")

    from .features import CarRacingFeatures

    actions = _as_actions(actions)
    extractor = CarRacingFeatures()
    stride = max(1, int(frame_stride))
    indices = np.arange(0, len(actions), stride)

    features = np.array([extractor(states[i], int(actions[i])) for i in indices])
    feature_summary = np.concatenate([features.mean(axis=0), features.std(axis=0)])
    feature_names = (
        [f"v2_mean_{name}" for name in extractor.feature_names]
        + [f"v2_std_{name}" for name in extractor.feature_names]
    )
    return np.concatenate([summary, feature_summary]), names + feature_names


# ---------------------------------------------------------------------------
# Outlier scores and diagnostics
# ---------------------------------------------------------------------------


def knn_outlier_scores(summaries: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Score each row by mean distance to its k nearest neighbors.

    Larger score means more outlier-like. Features are robust-standardized before
    distance computation.
    """
    x = robust_standardize(_as_2d_float(summaries, name="summaries"))
    n = len(x)
    if n < 2:
        raise ValueError("KNN outlier scoring needs at least two trajectories.")

    k_eff = min(max(1, int(k)), n - 1)
    dists = _pairwise_distances(x, x)
    np.fill_diagonal(dists, np.inf)

    nearest = np.partition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    return nearest.mean(axis=1)


def knn_reference_outlier_scores(
    summaries: np.ndarray,
    reference_summaries: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Score each row by distance to its k nearest reference trajectories.

    This is useful when a trusted reference set exists. The current main
    experiment does not use a clean reference set, but the helper remains useful
    for diagnostics.
    """
    samples = _as_2d_float(summaries, name="summaries")
    reference = _as_2d_float(reference_summaries, name="reference_summaries")
    if len(reference) < 1:
        raise ValueError("Reference KNN needs at least one reference trajectory.")
    if reference.shape[1] != samples.shape[1]:
        raise ValueError("Reference and sample summaries must have the same feature dimension.")

    ref_scaled, samples_scaled = robust_standardize_against_reference(reference, samples)
    k_eff = min(max(1, int(k)), len(reference))
    dists = _pairwise_distances(samples_scaled, ref_scaled)

    nearest = np.partition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    return nearest.mean(axis=1)


def classify_outliers(
    scores: np.ndarray,
    contamination: float | None = None,
    n_outliers: int | None = None,
) -> np.ndarray:
    """Return boolean predictions where True means predicted poison/outlier."""
    scores = np.asarray(scores, dtype=np.float64)
    n_outliers = _resolve_outlier_count(len(scores), contamination, n_outliers)

    pred = np.zeros(len(scores), dtype=bool)
    if n_outliers > 0:
        pred[np.argsort(scores)[-n_outliers:]] = True
    return pred


def binary_metrics(scores: np.ndarray, labels: list[str] | np.ndarray, pred_outlier: np.ndarray) -> dict:
    """Compute simple diagnostics using known labels. Poison is the positive class."""
    y_true = np.array([str(label) == "poison" for label in labels], dtype=bool)
    pred = np.asarray(pred_outlier, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)

    tp = int(np.sum(pred & y_true))
    fp = int(np.sum(pred & ~y_true))
    tn = int(np.sum(~pred & ~y_true))
    fn = int(np.sum(~pred & y_true))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "n": int(len(y_true)),
        "n_poison": int(np.sum(y_true)),
        "n_expert": int(np.sum(~y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auc": float(_auc_pairwise(scores, y_true)),
        "mean_expert_score": _mean_or_nan(scores[~y_true]),
        "mean_poison_score": _mean_or_nan(scores[y_true]),
    }


# ---------------------------------------------------------------------------
# Score-to-weight mappings
# ---------------------------------------------------------------------------


def scores_to_trust_weights(
    scores: np.ndarray,
    threshold: float | None = None,
    scale: float | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Convert outlier scores to trust weights in [0, 1].

    Higher outlier score means lower trust.
    """
    return _sigmoid_trust_weights(
        scores,
        threshold=threshold,
        scale=scale,
        temperature=temperature,
        higher_score_is_trusted=False,
    )


def reward_scores_to_trust_weights(
    scores: np.ndarray,
    threshold: float | None = None,
    scale: float | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Convert reward/log-likelihood scores to trust weights in [0, 1].

    Higher reward-consistency score means higher trust.
    """
    return _sigmoid_trust_weights(
        scores,
        threshold=threshold,
        scale=scale,
        temperature=temperature,
        higher_score_is_trusted=True,
    )


# ---------------------------------------------------------------------------
# Robust scaling
# ---------------------------------------------------------------------------


def robust_standardize(x: np.ndarray) -> np.ndarray:
    """Robust-standardize columns using median and IQR, with std fallback."""
    x = _as_2d_float(x, name="x")
    center, scale = _robust_center_scale(x)
    return (x - center) / scale


def robust_standardize_against_reference(
    reference: np.ndarray,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale reference and samples using robust statistics from reference only."""
    reference = _as_2d_float(reference, name="reference")
    samples = _as_2d_float(samples, name="samples")
    if reference.shape[1] != samples.shape[1]:
        raise ValueError("Reference and sample arrays must have the same feature dimension.")

    center, scale = _robust_center_scale(reference)
    return (reference - center) / scale, (samples - center) / scale


# ---------------------------------------------------------------------------
# High-level beta functions
# ---------------------------------------------------------------------------


def beta_OD(
    demo_trajs,
    contamination: float = 0.1,
    k: int = 5,
    summary_mode: SummaryMode = "action",
    frame_stride: int = 10,
) -> np.ndarray:
    """
    β_OD: unsupervised KNN outlier-detection trust weights.

    Returns one weight per demo trajectory. Higher weight means more trusted.
    """
    summaries = _trajectory_summaries(demo_trajs, mode=summary_mode, frame_stride=frame_stride)
    scores = knn_outlier_scores(summaries, k=k)
    threshold = _contamination_threshold(scores, contamination)
    return scores_to_trust_weights(scores, threshold=threshold)


def beta_RC(
    irl,
    demo_trajs,
    bg_trajs,
    K: int = 3,
    lam: float = 1.0,
    n_iter_per_step: int = 300,
    verbose: bool = False,
) -> np.ndarray:
    """
    β_RC: Reward Consistency trust weights.

    The loop starts with uniform weights, trains MaxEnt IRL, scores each
    trajectory by log P(tau_i | theta), updates beta from those scores, and
    repeats. The final irl.theta is trained with the returned beta weights.
    """
    if len(demo_trajs) == 0:
        raise ValueError("Reward Consistency needs at least one demo trajectory.")
    if len(bg_trajs) == 0:
        raise ValueError("Reward Consistency needs at least one background trajectory.")

    weights = np.ones(len(demo_trajs), dtype=np.float64)
    n_outer = max(1, int(K))

    for step in range(n_outer):
        _train_from_reset(irl, demo_trajs, bg_trajs, weights, n_iter_per_step, verbose)

        scores = reward_log_likelihood_scores(irl, demo_trajs, bg_trajs)
        weights = reward_scores_to_trust_weights(scores, temperature=lam)

        if verbose:
            print(
                f"  RC iter {step + 1}/{n_outer}: "
                f"weights min={weights.min():.3f} "
                f"max={weights.max():.3f} "
                f"mean={weights.mean():.3f}"
            )

    _train_from_reset(irl, demo_trajs, bg_trajs, weights, n_iter_per_step, verbose)
    return weights


def reward_log_likelihood_scores(irl, demo_trajs, bg_trajs) -> np.ndarray:
    """Score demos by log P(tau | theta) under the background partition set."""
    log_z = _logsumexp(np.array([irl.score_trajectory(t) for t in bg_trajs]))
    return np.array([irl.score_trajectory(t) - log_z for t in demo_trajs])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trajectory_summaries(
    trajectories,
    mode: SummaryMode = "action",
    frame_stride: int = 10,
) -> np.ndarray:
    return np.array(
        [
            trajectory_summary(t.states, t.actions, mode=mode, frame_stride=frame_stride)[0]
            for t in trajectories
        ],
        dtype=np.float64,
    )


def _sigmoid_trust_weights(
    scores: np.ndarray,
    *,
    threshold: float | None,
    scale: float | None,
    temperature: float,
    higher_score_is_trusted: bool,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return np.array([], dtype=np.float64)

    center = float(np.median(scores)) if threshold is None else float(threshold)
    denom = _score_scale(scores) if scale is None else max(float(scale), _EPS)
    z = np.clip((scores - center) / denom, -_LOGISTIC_CLIP, _LOGISTIC_CLIP)

    sign = -1.0 if higher_score_is_trusted else 1.0
    return 1.0 / (1.0 + np.exp(sign * float(temperature) * z))


def _score_scale(scores: np.ndarray) -> float:
    q25 = float(np.percentile(scores, 25))
    q75 = float(np.percentile(scores, 75))
    scale = q75 - q25
    if scale < _EPS:
        scale = float(np.std(scores))
    return scale if scale >= _EPS else 1.0


def _contamination_threshold(scores: np.ndarray, contamination: float | None) -> float | None:
    if contamination is None:
        return None
    contamination = float(np.clip(contamination, 0.0, 1.0))
    return float(np.percentile(scores, 100.0 * (1.0 - contamination)))


def _resolve_outlier_count(
    n_samples: int,
    contamination: float | None,
    n_outliers: int | None,
) -> int:
    if n_outliers is None:
        contamination = 0.1 if contamination is None else float(contamination)
        n_outliers = int(round(n_samples * contamination))
    return int(np.clip(n_outliers, 0, n_samples))


def _robust_center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = q75 - q25
    scale = np.where(scale < _EPS, np.std(x, axis=0), scale)
    scale = np.where(scale < _EPS, 1.0, scale)
    return center, scale


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norms = np.sum(a * a, axis=1, keepdims=True)
    b_norms = np.sum(b * b, axis=1, keepdims=True).T
    d2 = a_norms + b_norms - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _train_from_reset(
    irl,
    demo_trajs,
    bg_trajs,
    weights: np.ndarray,
    n_iter: int,
    verbose: bool,
) -> None:
    irl.reset()
    irl.train(demo_trajs, bg_trajs, weights=weights, n_iter=n_iter, verbose=verbose)


def _as_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if len(actions) == 0:
        raise ValueError("Cannot summarize an empty action sequence.")
    return actions


def _as_2d_float(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    return values


def _action_freq(actions: np.ndarray) -> np.ndarray:
    counts = np.bincount(actions, minlength=N_ACTIONS).astype(np.float64)[:N_ACTIONS]
    total = counts.sum()
    if total <= 0:
        return np.zeros(N_ACTIONS, dtype=np.float64)
    return counts / total


def _entropy(freq: np.ndarray) -> float:
    p = np.asarray(freq, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(N_ACTIONS))


def _longest_run_fraction(actions: np.ndarray) -> float:
    longest = 1
    current = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return float(max(longest, current) / len(actions))


def _auc_pairwise(scores: np.ndarray, y_true: np.ndarray) -> float:
    pos = scores[y_true]
    neg = scores[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    comparisons = pos[:, None] - neg[None, :]
    wins = np.sum(comparisons > 0)
    ties = np.sum(comparisons == 0)
    return float((wins + 0.5 * ties) / comparisons.size)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    center = float(values.max())
    return float(center + np.log(np.exp(values - center).sum()))
