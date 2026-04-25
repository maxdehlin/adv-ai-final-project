"""
Diagnose autoencoder trust scoring on expert vs poison trajectories.

The autoencoder is trained only on clean expert reference trajectories. It then
scores candidate trajectories by reconstruction error:

    low reconstruction error  -> expert-like -> high trust weight
    high reconstruction error -> unusual     -> low trust weight

Usage:
    python -m experiments.analyze_autoencoder_trust --poison-dir data/raw/poison_random
    python -m experiments.analyze_autoencoder_trust --poison-dir data/raw/poison_expert_then_stop
    python -m experiments.analyze_autoencoder_trust --summary-mode v2 --frame-stride 25
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from maxent_irl.autoencoder_trust import fit_autoencoder_scores  # noqa: E402
from maxent_irl.trust import binary_metrics, classify_outliers, trajectory_summary  # noqa: E402


def scan_paths(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "traj_*.npz")))


def sample_paths(paths: list[str], n: int, rng: np.random.Generator) -> list[str]:
    if n < 0 or n >= len(paths):
        return list(paths)
    idx = rng.choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


def load_dir_dataset(args) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(args.seed)
    all_expert_paths = scan_paths(args.expert_dir)
    if len(all_expert_paths) == 0:
        raise ValueError(f"No expert trajectories found in {args.expert_dir}")

    shuffled_expert = list(all_expert_paths)
    rng.shuffle(shuffled_expert)
    candidate_expert = shuffled_expert[: args.n_expert] if args.n_expert >= 0 else shuffled_expert
    reference_pool = shuffled_expert[len(candidate_expert) :]
    if len(reference_pool) == 0:
        reference_pool = candidate_expert

    reference_paths = sample_paths(reference_pool, args.n_reference, rng)
    poison_paths = sample_paths(scan_paths(args.poison_dir), args.n_poison, rng)
    if len(poison_paths) == 0:
        raise ValueError(f"No poison trajectories found in {args.poison_dir}")

    paths = candidate_expert + poison_paths
    labels = ["expert"] * len(candidate_expert) + ["poison"] * len(poison_paths)
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
    parser.add_argument("--expert-dir", default="data/raw/expert")
    parser.add_argument("--poison-dir", default="data/raw/poison_random")
    parser.add_argument("--n-expert", type=int, default=180)
    parser.add_argument("--n-poison", type=int, default=20)
    parser.add_argument("--n-reference", type=int, default=200)
    parser.add_argument("--summary-mode", choices=["action", "v2"], default="action")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    paths, labels, reference_paths = load_dir_dataset(args)
    print(
        f"Autoencoder trust diagnostic: n={len(paths)}  poison={labels.count('poison')}  "
        f"reference={len(reference_paths)}  summary={args.summary_mode}"
    )

    print("Summarizing candidate trajectories...")
    summaries, feature_names = summarize_paths(paths, args.summary_mode, args.frame_stride)
    print("Summarizing clean expert reference trajectories...")
    reference_summaries, _ = summarize_paths(reference_paths, args.summary_mode, args.frame_stride)

    result = fit_autoencoder_scores(
        reference_summaries,
        summaries,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        val_frac=args.val_frac,
        seed=args.seed,
        device_arg=args.device,
        patience=args.patience,
    )

    n_poison = labels.count("poison")
    pred = classify_outliers(result.scores, n_outliers=n_poison)
    metrics = binary_metrics(result.scores, labels, pred)

    labels_arr = np.array(labels)
    expert_mask = labels_arr == "expert"
    poison_mask = labels_arr == "poison"

    print("\nTraining")
    print(f"  epochs run:         {len(result.train_losses)}")
    print(f"  final train loss:   {result.train_losses[-1]:.6f}")
    print(f"  final val loss:     {result.val_losses[-1]:.6f}")
    print(f"  trust threshold:    {result.threshold:.6f}")

    print("\nMetrics")
    print(f"  precision:          {metrics['precision']:.3f}")
    print(f"  recall:             {metrics['recall']:.3f}")
    print(f"  accuracy:           {metrics['accuracy']:.3f}")
    print(f"  auc:                {metrics['auc']:.3f}")
    print(f"  mean expert error:  {metrics['mean_expert_score']:.6f}")
    print(f"  mean poison error:  {metrics['mean_poison_score']:.6f}")
    print(f"  mean expert weight: {result.weights[expert_mask].mean():.3f}")
    print(f"  mean poison weight: {result.weights[poison_mask].mean():.3f}")
    print(f"  confusion:          tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']}")

    print("\nTop reconstruction errors")
    order = np.argsort(result.scores)[::-1][: args.top]
    for rank, idx in enumerate(order, start=1):
        flag = "PRED" if pred[idx] else "    "
        print(
            f"  {rank:2d}. {flag}  {labels[idx]:6s}  "
            f"error={result.scores[idx]:.6f}  weight={result.weights[idx]:.3f}  {paths[idx]}"
        )

    output = {
        "paths": paths,
        "labels": labels,
        "scores": result.scores.tolist(),
        "weights": result.weights.tolist(),
        "predicted_outlier": pred.astype(bool).tolist(),
        "feature_names": feature_names,
        "metrics": metrics,
        "train_losses": result.train_losses,
        "val_losses": result.val_losses,
        "threshold": result.threshold,
        "scale": result.scale,
        "args": vars(args),
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved diagnostic to {args.out}")


if __name__ == "__main__":
    main()
