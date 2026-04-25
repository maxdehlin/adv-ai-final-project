"""
No-poison behavior cloning baseline for CarRacing-v3.

This is a clean-data sanity baseline: train a policy directly from expert
trajectories, then evaluate it with the ground-truth CarRacing reward.

Why this baseline exists:
    If D0 cannot produce a competent policy, the poisoning experiment is not
    yet testing poison robustness. It is testing whether the downstream policy
    training setup can learn at all.

Usage:
    python -m experiments.no_poison_bc
    python -m experiments.no_poison_bc --max-frames 80000 --epochs 8
    python -m experiments.no_poison_bc --eval-only --model results/no_poison_bc/bc_policy.pt
"""

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


N_ACTIONS = 5


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """
    Convert CarRacing's 96x96x3 uint8 observation to a compact CHW float tensor.

    A 2x spatial downsample keeps the baseline fast and memory-light while
    preserving the track geometry needed for imitation.
    """
    obs = obs[::2, ::2]  # 48x48x3
    obs = obs.astype(np.float32) / 255.0
    return np.transpose(obs, (2, 0, 1))


def _scan_trajectories(expert_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(expert_dir, "traj_*.npz")))
    if not paths:
        raise ValueError(f"No trajectories found in {expert_dir}")
    return paths


def load_expert_frames(
    expert_dir: str,
    max_frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load up to max_frames state/action pairs from clean expert trajectories.

    Frames are sampled across shuffled trajectories and then shuffled again at
    the frame level. This avoids depending on one long contiguous rollout.
    """
    rng = np.random.default_rng(seed)
    paths = _scan_trajectories(expert_dir)
    rng.shuffle(paths)

    states = []
    actions = []
    remaining = max_frames

    for path in paths:
        if remaining <= 0:
            break

        data = np.load(path)
        traj_states = data["states"]
        traj_actions = data["actions"].astype(np.int64)

        if len(traj_actions) <= remaining:
            idx = np.arange(len(traj_actions))
        else:
            idx = rng.choice(len(traj_actions), size=remaining, replace=False)
            idx.sort()

        states.append(np.stack([_preprocess_obs(traj_states[i]) for i in idx]))
        actions.append(traj_actions[idx])
        remaining -= len(idx)

    x = np.concatenate(states, axis=0)
    y = np.concatenate(actions, axis=0)
    order = rng.permutation(len(y))
    return x[order], y[order]


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


def train_val_split(x: np.ndarray, y: np.ndarray, val_frac: float) -> Split:
    n_val = max(1, int(round(len(y) * val_frac)))
    return Split(
        x_train=x[n_val:],
        y_train=y[n_val:],
        x_val=x[:n_val],
        y_val=y[:n_val],
    )


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "This baseline requires torch. Install the project dependencies from "
            "README.md, then rerun this command."
        ) from exc
    return torch, nn, F, DataLoader, TensorDataset


def import_gym():
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Evaluation requires gymnasium[box2d]. Install the project dependencies "
            "from README.md, then rerun this command."
        ) from exc
    return gym


def build_policy(nn):
    class BCCnnPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 6 * 6, 256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, N_ACTIONS),
            )

        def forward(self, x):
            return self.head(self.conv(x))

    return BCCnnPolicy()


def class_weights(y: np.ndarray, torch, device):
    counts = np.bincount(y, minlength=N_ACTIONS).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.sqrt(counts.sum() / (N_ACTIONS * counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_bc(args):
    torch, nn, F, DataLoader, TensorDataset = import_torch()

    _seed_everything(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading up to {args.max_frames:,} clean expert frames from {args.expert_dir}...")
    x, y = load_expert_frames(args.expert_dir, args.max_frames, args.seed)
    split = train_val_split(x, y, args.val_frac)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_policy(nn).to(device)

    train_ds = TensorDataset(
        torch.tensor(split.x_train, dtype=torch.float32),
        torch.tensor(split.y_train, dtype=torch.long),
    )
    val_x = torch.tensor(split.x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(split.y_val, dtype=torch.long, device=device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(split.y_train, torch, device) if args.balanced_loss else None

    print(f"Training BC policy on {len(split.y_train):,} frames; validating on {len(split.y_val):,}.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        correct = 0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += int(yb.numel())

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = F.cross_entropy(val_logits, val_y, weight=weights)
            val_acc = float((val_logits.argmax(dim=1) == val_y).float().mean().item())

        print(
            f"  epoch {epoch:02d}  "
            f"train_loss={np.mean(losses):.4f}  train_acc={correct / seen:.3f}  "
            f"val_loss={float(val_loss.cpu()):.4f}  val_acc={val_acc:.3f}"
        )

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_shape": [3, 48, 48],
            "n_actions": N_ACTIONS,
            "max_frames": args.max_frames,
            "balanced_loss": args.balanced_loss,
            "seed": args.seed,
        },
        args.model,
    )
    print(f"Saved BC policy to {args.model}")
    return model, device


def load_bc_model(model_path: str, device_arg: Optional[str]):
    torch, nn, _F, _DataLoader, _TensorDataset = import_torch()
    device = torch.device(device_arg if device_arg else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(model_path, map_location=device)
    model = build_policy(nn).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, torch, device


def evaluate(model, torch, device, n_eval: int, seed: int, render: bool = False) -> list[float]:
    gym = import_gym()
    render_mode = "human" if render else None
    scores = []

    for ep in range(n_eval):
        env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0

        while not done:
            x = torch.tensor(_preprocess_obs(obs)[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                action = int(model(x).argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated

        env.close()
        scores.append(total)
        print(f"  eval ep {ep + 1:02d}: {total:.1f}")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-dir", default="data/raw/expert")
    parser.add_argument("--results-dir", default="results/no_poison_bc")
    parser.add_argument("--model", default="results/no_poison_bc/bc_policy.pt")
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--balanced-loss", action="store_true")
    parser.add_argument("--unbalanced-loss", action="store_false", dest="balanced_loss")
    parser.set_defaults(balanced_loss=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        model, torch, device = load_bc_model(args.model, args.device)
    else:
        model, device = train_bc(args)
        torch, *_ = import_torch()

    scores = evaluate(model, torch, device, args.n_eval, args.seed, args.render)
    result = {
        "baseline": "no_poison_behavior_cloning",
        "expert_dir": args.expert_dir,
        "model": args.model,
        "max_frames": args.max_frames,
        "epochs": args.epochs,
        "balanced_loss": args.balanced_loss,
        "eval_scores": scores,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "seed": args.seed,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "bc_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Mean score: {result['mean_score']:.1f} +/- {result['std_score']:.1f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
