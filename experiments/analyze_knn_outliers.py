"""
Diagnose KNN outlier detection on expert vs poison trajectories.

This does not train IRL or PPO. It answers one focused question:
do the candidate poison trajectories receive higher KNN outlier scores than
clean expert trajectories?

Usage:
    python -m experiments.analyze_knn_outliers --poison-dir data/raw/poison_random
    python -m experiments.analyze_knn_outliers --poison-dir data/raw/poison_expert_then_stop
    python -m experiments.analyze_knn_outliers --summary-mode v2 --frame-stride 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from maxent_irl.trust import (  # noqa: E402
    binary_metrics,
    classify_outliers,
    knn_reference_outlier_scores,
    knn_outlier_scores,
    scores_to_trust_weights,
    trajectory_summary,
)


def scan_paths(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "traj_*.npz")))


def sample_paths(paths: list[str], n: int, rng: np.random.Generator) -> list[str]:
    if n < 0 or n >= len(paths):
        return list(paths)
    idx = rng.choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


def load_manifest(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        manifest = json.load(f)
    paths = [entry["path"] for entry in manifest["trajectories"]]
    labels = [entry["label"] for entry in manifest["trajectories"]]
    return paths, labels


def load_dir_dataset(args) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(args.seed)
    all_expert_paths = scan_paths(args.expert_dir)
    if len(all_expert_paths) == 0:
        raise ValueError(f"No expert trajectories found in {args.expert_dir}")

    shuffled_expert = list(all_expert_paths)
    rng.shuffle(shuffled_expert)
    if args.n_expert < 0:
        expert_paths = shuffled_expert
        reference_pool = shuffled_expert
    else:
        expert_paths = shuffled_expert[: args.n_expert]
        reference_pool = shuffled_expert[args.n_expert :]

    if len(reference_pool) == 0:
        reference_pool = expert_paths
    reference_paths = sample_paths(reference_pool, args.n_reference, rng)

    poison_paths = sample_paths(scan_paths(args.poison_dir), args.n_poison, rng)

    if len(poison_paths) == 0:
        raise ValueError(f"No poison trajectories found in {args.poison_dir}")

    paths = expert_paths + poison_paths
    labels = ["expert"] * len(expert_paths) + ["poison"] * len(poison_paths)
    order = rng.permutation(len(paths))
    return [paths[i] for i in order], [labels[i] for i in order], reference_paths


def summarize_paths(paths: list[str], mode: str, frame_stride: int) -> tuple[np.ndarray, list[str]]:
    rows = []
    names = None

    for i, path in enumerate(paths, start=1):
        data = np.load(path)
        actions = data["actions"]
        states = data["states"] if mode == "v2" else np.empty((len(actions), 0))
        row, row_names = trajectory_summary(
            states=states,
            actions=actions,
            mode=mode,
            frame_stride=frame_stride,
        )
        rows.append(row)
        names = row_names

        if i == 1 or i % 25 == 0 or i == len(paths):
            print(f"  summarized {i:4d}/{len(paths)} trajectories")

    return np.vstack(rows), names or []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None, help="Optional D*.json manifest with expert/poison labels")
    parser.add_argument("--expert-dir", default="data/raw/expert")
    parser.add_argument("--poison-dir", default="data/raw/poison_random")
    parser.add_argument("--n-expert", type=int, default=180)
    parser.add_argument("--n-poison", type=int, default=20)
    parser.add_argument("--summary-mode", choices=["action", "v2"], default="action")
    parser.add_argument("--score-mode", choices=["expert-reference", "mixed"], default="expert-reference",
                        help="expert-reference scores distance to clean expert reference; mixed uses local density over all candidates")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-reference", type=int, default=100,
                        help="Number of clean expert reference trajectories for expert-reference KNN")
    parser.add_argument("--contamination", type=float, default=None,
                        help="Predicted outlier fraction. Default uses known poison count.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.manifest:
        paths, labels = load_manifest(args.manifest)
        reference_paths = [path for path, label in zip(paths, labels) if label == "expert"]
    else:
        paths, labels, reference_paths = load_dir_dataset(args)

    print(
        f"KNN outlier diagnostic: n={len(paths)}  "
        f"poison={labels.count('poison')}  summary={args.summary_mode}  "
        f"score={args.score_mode}  k={args.k}"
    )
    summaries, feature_names = summarize_paths(paths, args.summary_mode, args.frame_stride)
    if args.score_mode == "mixed":
        scores = knn_outlier_scores(summaries, k=args.k)
    else:
        print(f"Summarizing {len(reference_paths)} clean expert reference trajectories...")
        reference_summaries, _ = summarize_paths(reference_paths, args.summary_mode, args.frame_stride)
        scores = knn_reference_outlier_scores(summaries, reference_summaries, k=args.k)

    n_poison = labels.count("poison")
    pred = classify_outliers(
        scores,
        contamination=args.contamination,
        n_outliers=None if args.contamination is not None else n_poison,
    )
    if np.any(pred) and np.any(~pred):
        threshold = float((scores[pred].min() + scores[~pred].max()) / 2.0)
    else:
        threshold = None
    weights = scores_to_trust_weights(scores, threshold=threshold)
    metrics = binary_metrics(scores, labels, pred)

    expert_mask = np.array(labels) == "expert"
    poison_mask = ~expert_mask

    print("\nMetrics")
    print(f"  precision:          {metrics['precision']:.3f}")
    print(f"  recall:             {metrics['recall']:.3f}")
    print(f"  accuracy:           {metrics['accuracy']:.3f}")
    print(f"  auc:                {metrics['auc']:.3f}")
    print(f"  mean expert score:  {metrics['mean_expert_score']:.3f}")
    print(f"  mean poison score:  {metrics['mean_poison_score']:.3f}")
    print(f"  mean expert weight: {weights[expert_mask].mean():.3f}")
    print(f"  mean poison weight: {weights[poison_mask].mean():.3f}")
    print(f"  confusion:          tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']}")

    print("\nTop outliers")
    order = np.argsort(scores)[::-1][: args.top]
    for rank, idx in enumerate(order, start=1):
        flag = "PRED" if pred[idx] else "    "
        print(f"  {rank:2d}. {flag}  {labels[idx]:6s}  score={scores[idx]:7.3f}  weight={weights[idx]:.3f}  {paths[idx]}")

    result = {
        "paths": paths,
        "labels": labels,
        "scores": scores.tolist(),
        "weights": weights.tolist(),
        "predicted_outlier": pred.astype(bool).tolist(),
        "feature_names": feature_names,
        "metrics": metrics,
        "args": vars(args),
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved diagnostic to {args.out}")


if __name__ == "__main__":
    main()
