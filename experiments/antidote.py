"""
ANTIDOTE experiment runner — all four methods on a single dataset.

Runs baseline, β_OD, β_AE, β_RC on one of D0–D5, then evaluates the learned
reward function DIRECTLY by scoring held-out expert and poison trajectories.

No RL training is performed. The reward quality is measured by:
  - mean r_θ on held-out expert trajectories
  - mean r_θ on held-out poison trajectories
  - separation = mean_expert_score − mean_poison_score  (higher is better)
  - AUC = P(r_θ(expert) > r_θ(poison))  (higher is better, 0.5 = chance)

Results are saved under:  results/antidote/{poison_name}/{dataset}/

Usage:
    python -m experiments.antidote --dataset D1 --poison-name poison_stop
    python -m experiments.antidote --dataset D3 --poison-name poison_random --poison-dir data/raw/poison_random
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import json
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from maxent_irl import Trajectory, CarRacingFeatures, CarRacingFeaturesV2, MaxEntIRL
from maxent_irl.trust import beta_OD, beta_RC
from maxent_irl.autoencoder_trust import beta_AE
from data_collection.collect_demos import load_trajectory

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASETS = {
    "D0": {"n_expert": 200, "n_poison": 0},
    "D1": {"n_expert": 180, "n_poison": 20},
    "D2": {"n_expert": 160, "n_poison": 40},
    "D3": {"n_expert": 140, "n_poison": 60},
    "D4": {"n_expert": 120, "n_poison": 80},
    "D5": {"n_expert": 100, "n_poison": 100},
}

ALL_METHODS = ["baseline", "OD", "AE", "RC"]

# Held-out eval set: fixed indices never used in demos.
# 1000 expert files available; demos sample outside 950–999.
EVAL_EXPERT_INDICES = list(range(950, 1000))  # 50 held-out expert trajectories

# When the eval poison dir is the SAME as the training poison dir, we randomly
# reserve N_EVAL_POISON_SAME files as held-out and exclude them from the demo
# pool, guaranteeing no train/eval poison overlap.
# When the dirs differ we do a random sample of N_EVAL_POISON_CROSS files.
N_EVAL_POISON_SAME  = 10   # same-dir held-out (e.g. poison_stop → poison_stop)
N_EVAL_POISON_CROSS = 50   # cross-dir held-out (e.g. poison_stop → poison_random)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_path(path, extractor):
    states, actions = load_trajectory(path)
    traj = Trajectory(states=states, actions=actions)
    extractor.extract_trajectory(traj)
    return traj


def _relpath(path: str) -> str:
    return os.path.relpath(path, start=os.getcwd())


def load_dir(directory, n, extractor, label="", return_paths=False):
    paths = sorted(glob.glob(os.path.join(directory, "traj_*.npz")))
    if len(paths) < n:
        raise ValueError(f"Need {n} trajectories in {directory}, found {len(paths)}")
    paths = random.sample(paths, n)
    trajs = [_load_path(p, extractor) for p in paths]
    print(f"  Loaded {len(trajs)} {label} trajectories from {directory}")
    if return_paths:
        return trajs, paths
    return trajs


def load_fixed(directory, indices, extractor, label=""):
    trajs = []
    for i in indices:
        p = os.path.join(directory, f"traj_{i:04d}.npz")
        trajs.append(_load_path(p, extractor))
    print(f"  Loaded {len(trajs)} {label} trajectories (fixed indices)")
    return trajs


def make_extractor(feature_version: str):
    if feature_version == "v1":
        return CarRacingFeatures()
    if feature_version == "v2":
        return CarRacingFeaturesV2()
    raise ValueError(f"Unknown feature version: {feature_version}")


# ---------------------------------------------------------------------------
# Direct reward evaluation (no RL)
# ---------------------------------------------------------------------------


def eval_reward_direct(irl_model, held_out_expert, held_out_poison):
    """
    Score held-out trajectories with the learned reward r_θ(τ) = θ · f(τ).

    Returns a dict with separation and AUC metrics.
    """
    expert_scores = np.array([irl_model.score_trajectory(t) for t in held_out_expert])
    poison_scores = np.array([irl_model.score_trajectory(t) for t in held_out_poison])

    mean_expert = float(expert_scores.mean())
    mean_poison = float(poison_scores.mean()) if len(poison_scores) > 0 else float("nan")
    separation  = mean_expert - mean_poison if len(poison_scores) > 0 else float("nan")

    if len(poison_scores) > 0:
        comparisons = expert_scores[:, None] - poison_scores[None, :]
        wins = int(np.sum(comparisons > 0))
        ties = int(np.sum(comparisons == 0))
        auc  = float((wins + 0.5 * ties) / comparisons.size)
    else:
        auc = float("nan")

    print(f"  Expert scores: mean={mean_expert:.4f}  (n={len(expert_scores)})")
    if len(poison_scores) > 0:
        print(f"  Poison scores: mean={mean_poison:.4f}  (n={len(poison_scores)})")
        print(f"  Separation:    {separation:.4f}")
        print(f"  AUC:           {auc:.4f}")

    return {
        "mean_expert_score": mean_expert,
        "mean_poison_score": mean_poison,
        "separation":        separation,
        "auc":               auc,
        "n_held_out_expert": len(expert_scores),
        "n_held_out_poison": len(poison_scores),
        "expert_scores":     expert_scores.tolist(),
        "poison_scores":     poison_scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        required=True, choices=list(DATASETS))
    parser.add_argument("--expert-dir",     default="data/raw/expert")
    parser.add_argument("--poison-dir",     default="data/raw/poison_stop")
    parser.add_argument("--eval-poison-dir", default=None,
                        help="Held-out poison dir. Defaults to --poison-dir (same-dir split).")
    parser.add_argument("--poison-name",    default=None,
                        help="Label used in the results path. Defaults to basename of --poison-dir.")
    parser.add_argument("--background-dir", default="data/raw/background")
    parser.add_argument("--n-background",   type=int, default=50)
    parser.add_argument("--methods",        default="baseline,OD,AE,RC")
    parser.add_argument("--irl-iters",      type=int, default=1000)
    parser.add_argument("--feature-version", default="v1", choices=["v1", "v2"],
                        help="Reward feature extractor. v1 is the original 8-D feature set.")
    parser.add_argument("--ae-epochs",      type=int, default=200)
    parser.add_argument("--ae-summary-mode", default="action", choices=["action", "v2"])
    parser.add_argument("--ae-frame-stride", type=int, default=10)
    parser.add_argument("--rc-k",           type=int, default=3,
                        help="Number of Reward Consistency outer reweighting iterations.")
    parser.add_argument("--results-dir",    default="results/antidote")
    parser.add_argument("--seed",           type=int, default=None,
                        help="Random seed for reproducible sampling/training. Default: random each run.")
    args = parser.parse_args()

    # Resolve derived args
    if args.eval_poison_dir is None:
        args.eval_poison_dir = args.poison_dir
    if args.poison_name is None:
        args.poison_name = os.path.basename(args.poison_dir.rstrip("/"))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown_methods = [m for m in methods if m not in ALL_METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods {unknown_methods}. Choose from {ALL_METHODS}.")
    cfg = DATASETS[args.dataset]
    n_expert, n_poison = cfg["n_expert"], cfg["n_poison"]
    poison_pct = n_poison / 200

    same_dir = os.path.realpath(args.poison_dir) == os.path.realpath(args.eval_poison_dir)

    results_dir = os.path.join(args.results_dir, args.poison_name, args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset:     {args.dataset}  "
          f"({n_expert} expert + {n_poison} poison = {poison_pct*100:.0f}%)")
    print(f"  Poison:      {args.poison_name}  ({'same-dir split' if same_dir else 'cross-dir eval'})")
    print(f"  Train poison dir: {args.poison_dir}")
    print(f"  Eval poison dir:  {args.eval_poison_dir}")
    if not same_dir:
        print("  WARNING: held-out poison differs from training poison; metrics are cross-poison eval.")
    print(f"  Seed:        {args.seed if args.seed is not None else 'random'}")
    print(f"  Methods:     {methods}")
    print(f"  Results →    {results_dir}")
    print(f"{'='*60}")

    extractor = make_extractor(args.feature_version)

    # --- Load data ---
    print("\n[DATA] Loading trajectories...")

    # Expert demo pool: exclude held-out eval indices only. Trust estimation
    # methods receive no clean side data.
    excluded = set(EVAL_EXPERT_INDICES)
    all_expert_paths = sorted(glob.glob(os.path.join(args.expert_dir, "traj_*.npz")))
    demo_pool_paths = [
        p for p in all_expert_paths
        if int(os.path.basename(p).split("_")[1].split(".")[0]) not in excluded
    ]
    if len(demo_pool_paths) < n_expert:
        raise ValueError(f"Need {n_expert} demo expert trajectories, "
                         f"found {len(demo_pool_paths)} after exclusions.")
    sampled_expert_paths = random.sample(demo_pool_paths, n_expert)
    expert_trajs = [_load_path(p, extractor) for p in sampled_expert_paths]
    print(f"  Loaded {len(expert_trajs)} expert trajectories")

    # Poison demo pool. If eval uses the same dir, reserve a random held-out
    # subset first so train and eval poison never overlap.
    all_poison_paths = sorted(glob.glob(os.path.join(args.poison_dir, "traj_*.npz")))
    if same_dir:
        n_eval_poison = min(N_EVAL_POISON_SAME, len(all_poison_paths))
        held_out_poison_paths = random.sample(all_poison_paths, n_eval_poison)
        held_out_poison_set = set(held_out_poison_paths)
        demo_poison_paths = [p for p in all_poison_paths if p not in held_out_poison_set]
    else:
        demo_poison_paths = all_poison_paths
        held_out_poison_paths = None  # will random-sample from eval_poison_dir

    poison_trajs = []
    sampled_poison_paths = []
    if n_poison > 0:
        avail = len(demo_poison_paths)
        actual_n_poison = min(n_poison, avail)
        if actual_n_poison < n_poison:
            print(f"  WARNING: requested {n_poison} poison demos but only "
                  f"{avail} available after held-out split; using {actual_n_poison}.")
        sampled_poison_paths = random.sample(demo_poison_paths, actual_n_poison)
        poison_trajs = [_load_path(p, extractor) for p in sampled_poison_paths]
        print(f"  Loaded {len(poison_trajs)} poison trajectories")

    demo_trajs = expert_trajs + poison_trajs
    random.shuffle(demo_trajs)

    bg_trajs = load_dir(args.background_dir, args.n_background, extractor, "background")

    # Held-out eval set
    held_out_expert = load_fixed(args.expert_dir, EVAL_EXPERT_INDICES,
                                 extractor, "held-out expert")
    if same_dir:
        held_out_poison = [_load_path(p, extractor) for p in held_out_poison_paths]
        print(f"  Loaded {len(held_out_poison)} held-out poison trajectories (same-dir random split)")
    else:
        held_out_poison, held_out_poison_paths = load_dir(
            args.eval_poison_dir,
            N_EVAL_POISON_CROSS,
            extractor,
            "held-out poison",
            return_paths=True,
        )

    overlap = set(sampled_poison_paths) & set(held_out_poison_paths)
    if overlap:
        raise RuntimeError(f"Train/eval poison overlap detected: {sorted(overlap)}")

    print(f"  Demos: {len(demo_trajs)}  Background: {len(bg_trajs)}  "
          f"feature_dim={extractor.feature_dim}  feature_version={args.feature_version}")
    print(f"  Held-out eval: {len(held_out_expert)} expert + "
          f"{len(held_out_poison)} poison")

    # --- Run each method ---
    all_results = []

    for method in methods:
        print(f"\n{'─'*60}")
        print(f"  Method: {method}")
        print(f"{'─'*60}")

        irl = MaxEntIRL(feature_dim=extractor.feature_dim, lr=0.05, l2=1e-4)

        if method == "baseline":
            print("  [IRL] Training MaxEnt IRL (uniform weights)...")
            irl.train(demo_trajs, bg_trajs, weights=None,
                      n_iter=args.irl_iters, verbose=True)

        elif method == "OD":
            print("  [β_OD] Computing KNN outlier weights...")
            weights = beta_OD(demo_trajs, contamination=max(0.05, poison_pct))
            print(f"  Weights: min={weights.min():.3f}  max={weights.max():.3f}  "
                  f"mean={weights.mean():.3f}")
            print("  [IRL] Training MaxEnt IRL with β_OD weights...")
            irl.train(demo_trajs, bg_trajs, weights=weights,
                      n_iter=args.irl_iters, verbose=True)

        elif method == "AE":
            print("  [β_AE] Training autoencoder on mixed demo summaries...")
            weights = beta_AE(
                demo_trajs,
                summary_mode=args.ae_summary_mode,
                frame_stride=args.ae_frame_stride,
                epochs=args.ae_epochs,
                seed=args.seed,
            )
            print(f"  Weights: min={weights.min():.3f}  max={weights.max():.3f}  "
                  f"mean={weights.mean():.3f}")
            print("  [IRL] Training MaxEnt IRL with β_AE weights...")
            irl.train(demo_trajs, bg_trajs, weights=weights,
                      n_iter=args.irl_iters, verbose=True)

        elif method == "RC":
            print("  [β_RC] Running EM reward-consistency loop...")
            weights = beta_RC(irl, demo_trajs, bg_trajs,
                              K=args.rc_k, lam=1.0, n_iter_per_step=300, verbose=True)
            print(f"  Final RC weights: min={weights.min():.3f}  "
                  f"max={weights.max():.3f}  mean={weights.mean():.3f}")

        print(f"  Final θ: {np.array2string(irl.theta, precision=3)}")

        # --- Direct reward evaluation ---
        print("  [EVAL] Scoring held-out trajectories...")
        eval_metrics = eval_reward_direct(irl, held_out_expert, held_out_poison)

        result = {
            "dataset":      args.dataset,
            "method":       method,
            "poison_name":  args.poison_name,
            "poison_dir":   args.poison_dir,
            "train_poison_dir": args.poison_dir,
            "eval_poison_dir": args.eval_poison_dir,
            "eval_poison_same_dir": same_dir,
            "eval_poison_split": "same_dir_random" if same_dir else "cross_dir_random",
            "n_expert":     n_expert,
            "n_poison":     len(poison_trajs),
            "n_requested_poison": n_poison,
            "n_available_train_poison": len(demo_poison_paths),
            "n_requested_held_out_poison": N_EVAL_POISON_SAME if same_dir else N_EVAL_POISON_CROSS,
            "poison_pct":   len(poison_trajs) / (n_expert + len(poison_trajs)) if poison_trajs else 0.0,
            "irl_iters":    args.irl_iters,
            "feature_version": args.feature_version,
            "feature_names": extractor.feature_names,
            "rc_k":        args.rc_k if method == "RC" else None,
            "seed":         args.seed,
            "theta":        irl.theta.tolist(),
            "train_poison_paths": [_relpath(p) for p in sampled_poison_paths],
            "held_out_poison_paths": [_relpath(p) for p in held_out_poison_paths],
            **eval_metrics,
        }

        out_path = os.path.join(results_dir, f"{method}_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")
        all_results.append(result)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {args.dataset} ({poison_pct*100:.0f}% poison)")
    print(f"{'='*60}")
    print(f"  {'Method':<12} {'Separation':>12}  {'AUC':>8}")
    print(f"  {'-'*35}")
    for r in all_results:
        sep = r["separation"]
        auc = r["auc"]
        sep_str = f"{sep:.4f}" if sep == sep else "   n/a"
        auc_str = f"{auc:.4f}" if auc == auc else "   n/a"
        print(f"  {r['method']:<12} {sep_str:>12}  {auc_str:>8}")


if __name__ == "__main__":
    main()
