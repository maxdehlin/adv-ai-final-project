"""
Train and diagnose state-conditional MaxEnt on expert action choices.

This is a fast pre-PPO sanity check for reward features. It trains:

    P(a | s) = softmax_a(theta . phi(s, a))

on expert states/actions, then reports whether the learned reward prefers the
expert action over counterfactual actions in the same state.

Usage:
    python -m experiments.diagnose_counterfactual_maxent --features v3
    python -m experiments.diagnose_counterfactual_maxent --features v2
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from maxent_irl import CarRacingFeaturesV2, CarRacingFeaturesV3  # noqa: E402
from maxent_irl.conditional_maxent import ConditionalMaxEntIRL  # noqa: E402


N_ACTIONS = 5


def make_extractor(name: str):
    if name == "v2":
        return CarRacingFeaturesV2()
    if name == "v3":
        return CarRacingFeaturesV3()
    raise ValueError(f"Unknown feature extractor: {name}")


def sample_state_actions(
    expert_dir: str,
    n_traj: int,
    frame_stride: int,
    max_states: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(expert_dir, "traj_*.npz")))
    if not paths:
        raise ValueError(f"No trajectories found in {expert_dir}")
    if n_traj > 0 and n_traj < len(paths):
        paths = [paths[i] for i in sorted(rng.choice(len(paths), size=n_traj, replace=False))]

    states = []
    actions = []
    stride = max(1, int(frame_stride))

    for path in paths:
        data = np.load(path)
        traj_states = data["states"]
        traj_actions = data["actions"].astype(np.int64)
        idx = np.arange(0, len(traj_actions), stride)
        for i in idx:
            states.append(traj_states[i])
            actions.append(int(traj_actions[i]))

    actions = np.array(actions, dtype=np.int64)
    if max_states > 0 and max_states < len(actions):
        keep = rng.choice(len(actions), size=max_states, replace=False)
        states = [states[i] for i in keep]
        actions = actions[keep]

    order = rng.permutation(len(actions))
    return [states[i] for i in order], actions[order]


def build_feature_tensor(states: list[np.ndarray], extractor) -> np.ndarray:
    rows = []
    for i, state in enumerate(states, start=1):
        rows.append(np.array([extractor(state, action) for action in range(N_ACTIONS)]))
        if i == 1 or i % 500 == 0 or i == len(states):
            print(f"  built counterfactual features {i:5d}/{len(states)}")
    return np.stack(rows, axis=0)


def split_train_val(features: np.ndarray, actions: np.ndarray, train_frac: float):
    n_train = int(round(len(actions) * train_frac))
    n_train = min(max(1, n_train), len(actions) - 1)
    return (
        features[:n_train],
        actions[:n_train],
        features[n_train:],
        actions[n_train:],
    )


def balanced_weights(actions: np.ndarray, n_actions: int = N_ACTIONS) -> np.ndarray:
    counts = np.bincount(actions, minlength=n_actions).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_actions * counts)
    sample_weights = weights[actions]
    return sample_weights / sample_weights.mean()


def print_metrics(label: str, metrics):
    print(f"\n{label}")
    print(f"  n:                    {metrics.n}")
    print(f"  nll:                  {metrics.nll:.4f}")
    print(f"  top1 expert accuracy: {metrics.top1_accuracy:.3f}")
    print(f"  expert margin:        {metrics.expert_margin:.3f}")
    print(f"  gas argmax rate:      {metrics.gas_argmax_rate:.3f}")
    print(f"  expert action freq:   {_fmt_freq(metrics.expert_action_freq)}")
    print(f"  pred action freq:     {_fmt_freq(metrics.predicted_action_freq)}")


def _fmt_freq(freqs):
    return "[" + ", ".join(f"{x:.2f}" for x in freqs) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-dir", default="data/raw/expert")
    parser.add_argument("--features", choices=["v2", "v3"], default="v3")
    parser.add_argument("--n-traj", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-states", type=int, default=6000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--class-balanced", action="store_true",
                        help="Reweight expert states so rare actions matter as much as gas.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    extractor = make_extractor(args.features)
    print(
        f"Counterfactual MaxEnt diagnostic: features={args.features}  "
        f"feature_dim={extractor.feature_dim}"
    )
    states, actions = sample_state_actions(
        expert_dir=args.expert_dir,
        n_traj=args.n_traj,
        frame_stride=args.frame_stride,
        max_states=args.max_states,
        seed=args.seed,
    )
    print(f"Loaded {len(actions)} sampled expert states.")
    features = build_feature_tensor(states, extractor)
    train_x, train_a, val_x, val_a = split_train_val(features, actions, args.train_frac)

    model = ConditionalMaxEntIRL(feature_dim=extractor.feature_dim, lr=args.lr, l2=args.l2)
    sample_weights = balanced_weights(train_a) if args.class_balanced else None
    print(f"\nTraining conditional MaxEnt for {args.iters} iterations...")
    model.train(train_x, train_a, sample_weights=sample_weights, n_iter=args.iters, verbose=args.verbose)

    print_metrics("Train metrics", model.evaluate(train_x, train_a))
    print_metrics("Held-out metrics", model.evaluate(val_x, val_a))

    print("\nLearned theta")
    for name, value in zip(extractor.feature_names, model.theta):
        print(f"  {name:24s} {value:9.3f}")


if __name__ == "__main__":
    main()
