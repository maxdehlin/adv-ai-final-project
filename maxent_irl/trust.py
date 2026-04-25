"""
Trajectory trust / outlier utilities.

This module is intentionally lightweight: the KNN detector uses only NumPy so
it can run anywhere the saved trajectory files can be loaded.
"""

from __future__ import annotations

import numpy as np

from .features import CarRacingFeaturesV2


N_ACTIONS = 5


def action_summary(actions: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Summarize a trajectory using action statistics.

    These features are deliberately good at catching targeted scripted poison:
    always-brake/gas/left policies, random policies, and expert-then-* policies
    whose second half has a very different action distribution.
    """
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if len(actions) == 0:
        raise ValueError("Cannot summarize an empty action sequence.")

    first = actions[: max(1, len(actions) // 2)]
    second = actions[len(actions) // 2 :]
    if len(second) == 0:
        second = first

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
    mode: str = "action",
    frame_stride: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a trajectory into a fixed-length vector for KNN outlier detection.

    mode="action" is fast and works well for scripted poison.
    mode="v2" appends sampled CarRacingFeaturesV2 mean/std summaries.
    """
    summary, names = action_summary(actions)
    if mode == "action":
        return summary, names
    if mode != "v2":
        raise ValueError(f"Unknown summary mode: {mode}")

    extractor = CarRacingFeaturesV2()
    stride = max(1, int(frame_stride))
    idx = np.arange(0, len(actions), stride)
    if len(idx) == 0:
        idx = np.array([0])

    features = np.array([extractor(states[i], int(actions[i])) for i in idx])
    v2_summary = np.concatenate([features.mean(axis=0), features.std(axis=0)])
    v2_names = (
        [f"v2_mean_{name}" for name in extractor.feature_names]
        + [f"v2_std_{name}" for name in extractor.feature_names]
    )
    return np.concatenate([summary, v2_summary]), names + v2_names


def knn_outlier_scores(summaries: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Score each row by mean distance to its k nearest neighbors.

    Larger score means more outlier-like. Features are robust-standardized
    before distance computation.
    """
    x = robust_standardize(np.asarray(summaries, dtype=np.float64))
    n = x.shape[0]
    if n < 2:
        raise ValueError("KNN outlier scoring needs at least two trajectories.")

    k_eff = min(max(1, int(k)), n - 1)
    sq_norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq_norms + sq_norms.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)

    nearest = np.partition(np.sqrt(d2), kth=k_eff - 1, axis=1)[:, :k_eff]
    return nearest.mean(axis=1)


def knn_reference_outlier_scores(
    summaries: np.ndarray,
    reference_summaries: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Score each row by distance to its k nearest clean-reference trajectories.

    This is usually better than mixed-set local-density KNN for clustered
    poison: if many poison trajectories are nearly identical, mixed KNN can
    incorrectly mark them as dense/inlier. A clean expert reference avoids that.
    """
    reference = np.asarray(reference_summaries, dtype=np.float64)
    samples = np.asarray(summaries, dtype=np.float64)
    if len(reference) < 1:
        raise ValueError("Reference KNN needs at least one reference trajectory.")

    ref_scaled, samples_scaled = robust_standardize_against_reference(reference, samples)
    k_eff = min(max(1, int(k)), len(reference))

    sample_norms = np.sum(samples_scaled * samples_scaled, axis=1, keepdims=True)
    ref_norms = np.sum(ref_scaled * ref_scaled, axis=1, keepdims=True).T
    d2 = sample_norms + ref_norms - 2.0 * (samples_scaled @ ref_scaled.T)
    np.maximum(d2, 0.0, out=d2)

    nearest = np.partition(np.sqrt(d2), kth=k_eff - 1, axis=1)[:, :k_eff]
    return nearest.mean(axis=1)


def classify_outliers(scores: np.ndarray, contamination: float | None = None, n_outliers: int | None = None) -> np.ndarray:
    """
    Return boolean predictions where True means predicted poison/outlier.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if n_outliers is None:
        if contamination is None:
            contamination = 0.1
        n_outliers = int(round(len(scores) * float(contamination)))
    n_outliers = int(np.clip(n_outliers, 0, len(scores)))

    pred = np.zeros(len(scores), dtype=bool)
    if n_outliers == 0:
        return pred

    outlier_idx = np.argsort(scores)[-n_outliers:]
    pred[outlier_idx] = True
    return pred


def scores_to_trust_weights(
    scores: np.ndarray,
    threshold: float | None = None,
    scale: float | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Convert outlier scores to soft trust weights in [0, 1].

    Higher outlier score -> lower trust. If threshold is provided, it should be
    the score boundary between trusted and outlier trajectories.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if threshold is None:
        threshold = float(np.median(scores))
    if scale is None:
        q25 = float(np.percentile(scores, 25))
        q75 = float(np.percentile(scores, 75))
        scale = q75 - q25
        if scale < 1e-8:
            scale = float(np.std(scores))
        if scale < 1e-8:
            scale = 1.0

    z = (scores - float(threshold)) / float(scale)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(float(temperature) * z))


def binary_metrics(scores: np.ndarray, labels: list[str] | np.ndarray, pred_outlier: np.ndarray) -> dict:
    """
    Compute simple diagnostics using known labels. Poison is the positive class.
    """
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
        "mean_expert_score": float(scores[~y_true].mean()) if np.any(~y_true) else float("nan"),
        "mean_poison_score": float(scores[y_true].mean()) if np.any(y_true) else float("nan"),
    }


def robust_standardize(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = q75 - q25
    scale = np.where(scale < 1e-8, np.std(x, axis=0), scale)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - median) / scale


def robust_standardize_against_reference(reference: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(reference, axis=0)
    q25 = np.percentile(reference, 25, axis=0)
    q75 = np.percentile(reference, 75, axis=0)
    scale = q75 - q25
    scale = np.where(scale < 1e-8, np.std(reference, axis=0), scale)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (reference - median) / scale, (samples - median) / scale


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
    if len(actions) == 0:
        return 0.0

    longest = 1
    current = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    longest = max(longest, current)
    return float(longest / len(actions))


def _auc_pairwise(scores: np.ndarray, y_true: np.ndarray) -> float:
    pos = scores[y_true]
    neg = scores[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    comparisons = pos[:, None] - neg[None, :]
    wins = np.sum(comparisons > 0)
    ties = np.sum(comparisons == 0)
    return float((wins + 0.5 * ties) / comparisons.size)
