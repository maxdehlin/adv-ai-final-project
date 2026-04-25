"""
Preliminary MaxEnt IRL experiment — baseline only (no ANTIDOTE weighting).

Runs two conditions:
  - D0:   0% poison
  - D_10: 10% poison  (sampled from data/raw/poison_random/)

For each condition:
  1. Load trajectories and extract features
  2. Train MaxEnt IRL (uniform weights)
  3. Train a PPO agent using the learned reward
  4. Evaluate the PPO agent using the ground-truth CarRacing reward
  5. Report scores

Usage:
    python -m experiments.preliminary
    python -m experiments.preliminary --irl-iters 500 --rl-steps 300000
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import os
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from maxent_irl import (
    Trajectory,
    FeatureExtractor,
    CarRacingFeatures,
    CarRacingFeaturesV2,
    CarRacingFeaturesV3,
    MaxEntIRL,
    ConditionalMaxEntIRL,
)
from data_collection.collect_demos import load_trajectory


N_ACTIONS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_extractor(feature_set: str) -> FeatureExtractor:
    if feature_set == "v1":
        return CarRacingFeatures()
    if feature_set == "v2":
        return CarRacingFeaturesV2()
    if feature_set == "v3":
        return CarRacingFeaturesV3()
    raise ValueError(f"Unknown feature set: {feature_set}")


def load_trajectories_from_dir(
    directory: str,
    n: int,
    extractor: FeatureExtractor,
    label: str = "",
    extract_features: bool = True,
) -> list[Trajectory]:
    import glob

    paths = sorted(glob.glob(os.path.join(directory, "traj_*.npz")))
    if len(paths) < n:
        raise ValueError(f"Need {n} trajectories in {directory}, found {len(paths)}")
    paths = random.sample(paths, n)
    trajs = []
    for p in paths:
        states, actions = load_trajectory(p)
        traj = Trajectory(states=states, actions=actions)
        if extract_features:
            extractor.extract_trajectory(traj)
        trajs.append(traj)
    print(f"  Loaded {len(trajs)} {label} trajectories from {directory}")
    return trajs


def build_dataset(
    n_expert: int,
    n_poison: int,
    expert_dir: str,
    poison_dir: str,
    extractor: FeatureExtractor,
    extract_features: bool = True,
) -> list[Trajectory]:
    trajs = []
    trajs += load_trajectories_from_dir(
        expert_dir, n_expert, extractor, "expert", extract_features=extract_features
    )
    if n_poison > 0:
        trajs += load_trajectories_from_dir(
            poison_dir, n_poison, extractor, "poison", extract_features=extract_features
        )
    random.shuffle(trajs)
    return trajs


def build_counterfactual_dataset(
    trajectories: list[Trajectory],
    extractor: FeatureExtractor,
    frame_stride: int,
    max_states: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = []
    expert_actions = []
    stride = max(1, int(frame_stride))

    for traj in trajectories:
        idx = np.arange(0, len(traj.actions), stride)
        for i in idx:
            action = int(traj.actions[i])
            if 0 <= action < N_ACTIONS:
                states.append(traj.states[i])
                expert_actions.append(action)

    expert_actions = np.asarray(expert_actions, dtype=np.int64)
    if len(expert_actions) == 0:
        raise ValueError("No valid sampled state/action pairs for conditional MaxEnt.")

    if max_states > 0 and max_states < len(expert_actions):
        keep = rng.choice(len(expert_actions), size=max_states, replace=False)
        states = [states[i] for i in keep]
        expert_actions = expert_actions[keep]

    order = rng.permutation(len(expert_actions))
    states = [states[i] for i in order]
    expert_actions = expert_actions[order]

    rows = []
    for i, state in enumerate(states, start=1):
        rows.append(np.array([extractor(state, action) for action in range(N_ACTIONS)]))
        if i == 1 or i % 1000 == 0 or i == len(states):
            print(f"  built counterfactual features {i:5d}/{len(states)}")

    return np.stack(rows, axis=0), expert_actions


def class_balanced_weights(
    actions: np.ndarray,
    exponent: float,
    n_actions: int = N_ACTIONS,
) -> np.ndarray:
    counts = np.bincount(actions, minlength=n_actions).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_actions * counts)
    weights = weights ** float(exponent)
    sample_weights = weights[actions]
    return sample_weights / sample_weights.mean()


def metrics_to_dict(metrics) -> dict:
    return {
        "n": metrics.n,
        "nll": metrics.nll,
        "top1_accuracy": metrics.top1_accuracy,
        "expert_margin": metrics.expert_margin,
        "gas_argmax_rate": metrics.gas_argmax_rate,
        "predicted_action_freq": metrics.predicted_action_freq,
        "expert_action_freq": metrics.expert_action_freq,
    }


def print_conditional_metrics(metrics):
    print(f"  conditional n={metrics.n}")
    print(f"  nll={metrics.nll:.4f}")
    print(f"  top1 expert accuracy={metrics.top1_accuracy:.3f}")
    print(f"  expert margin={metrics.expert_margin:.3f}")
    print(f"  gas argmax rate={metrics.gas_argmax_rate:.3f}")
    print(f"  expert action freq={_fmt_freq(metrics.expert_action_freq)}")
    print(f"  pred action freq={_fmt_freq(metrics.predicted_action_freq)}")


def _fmt_freq(freqs):
    return "[" + ", ".join(f"{x:.2f}" for x in freqs) + "]"


# ---------------------------------------------------------------------------
# Reward wrapper: replaces env reward with learned IRL reward
# ---------------------------------------------------------------------------


class LearnedRewardEnv(gym.Wrapper):
    """Swap out the ground-truth reward for r_θ(s, a) = θ · φ(s, a)."""

    def __init__(self, env, irl_model: MaxEntIRL, extractor: FeatureExtractor):
        super().__init__(env)
        self.irl = irl_model
        self.extractor = extractor

    def step(self, action):
        obs, _gt_reward, terminated, truncated, info = self.env.step(action)
        features = self.extractor(obs, int(action))
        learned_reward = float(self.irl.theta @ features)
        return obs, learned_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_condition(
    name: str,
    n_expert: int,
    n_poison: int,
    expert_dir: str,
    poison_dir: str,
    background_dir: str,
    n_background: int,
    irl_iters: int,
    rl_steps: int,
    n_eval: int,
    results_dir: str,
    seed: int,
    feature_set: str,
    objective: str,
    conditional_frame_stride: int,
    conditional_max_states: int,
    class_balanced: bool,
    balance_exponent: float,
    conditional_lr: float,
    conditional_l2: float,
):
    random.seed(seed)
    np.random.seed(seed)
    print(f"\n{'='*60}")
    print(f"  Condition: {name}  ({n_expert} expert + {n_poison} poison)")
    print(f"  Objective: {objective}")
    print(f"{'='*60}")

    extractor = make_extractor(feature_set)

    # --- 1. Load dataset ---
    print("\n[1/4] Loading trajectories...")
    extract_features = objective == "trajectory"
    demo_trajs = build_dataset(
        n_expert,
        n_poison,
        expert_dir,
        poison_dir,
        extractor,
        extract_features=extract_features,
    )
    bg_trajs = []
    if objective == "trajectory":
        bg_trajs = load_trajectories_from_dir(
            background_dir, n_background, extractor, "background"
        )
    print(
        f"  Demos: {len(demo_trajs)}  Background: {len(bg_trajs)}  "
        f"features={feature_set}  feature_dim={extractor.feature_dim}"
    )

    # --- 2. Train MaxEnt IRL ---
    conditional_metrics = None
    if objective == "trajectory":
        print(f"\n[2/4] Training trajectory MaxEnt IRL ({irl_iters} iterations)...")
        irl = MaxEntIRL(feature_dim=extractor.feature_dim, lr=0.05, l2=1e-4)
        history = irl.train(demo_trajs, bg_trajs, n_iter=irl_iters, verbose=True)
    else:
        print(f"\n[2/4] Training conditional MaxEnt IRL ({irl_iters} iterations)...")
        feature_tensor, expert_actions = build_counterfactual_dataset(
            demo_trajs,
            extractor,
            frame_stride=conditional_frame_stride,
            max_states=conditional_max_states,
            seed=seed,
        )
        sample_weights = None
        if class_balanced:
            sample_weights = class_balanced_weights(expert_actions, exponent=balance_exponent)
            print(f"  Using class-balanced action weights (exponent={balance_exponent:.2f}).")

        irl = ConditionalMaxEntIRL(
            feature_dim=extractor.feature_dim,
            lr=conditional_lr,
            l2=conditional_l2,
        )
        history = irl.train(
            feature_tensor,
            expert_actions,
            sample_weights=sample_weights,
            n_iter=irl_iters,
            verbose=True,
        )
        conditional_metrics = irl.evaluate(feature_tensor, expert_actions)
        print_conditional_metrics(conditional_metrics)
    print(f"  Final theta: {np.array2string(irl.theta, precision=3)}")

    # --- 3. Train RL with learned reward ---
    print(f"\n[3/4] Training PPO with learned reward ({rl_steps:,} steps)...")

    def make_learned_env():
        base = gym.make("CarRacing-v3", continuous=False)
        return LearnedRewardEnv(base, irl, make_extractor(feature_set))

    train_env = VecTransposeImage(make_vec_env(make_learned_env, n_envs=4))
    rl_model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        verbose=0,
    )
    rl_model.learn(total_timesteps=rl_steps)
    train_env.close()

    # Save RL model
    os.makedirs(results_dir, exist_ok=True)
    rl_path = os.path.join(results_dir, f"{name}_rl_model.zip")
    rl_model.save(rl_path)

    # --- 4. Evaluate with ground-truth reward ---
    print(f"\n[4/4] Evaluating with ground-truth reward ({n_eval} episodes)...")
    eval_env = gym.make("CarRacing-v3", continuous=False)
    scores = []
    for ep in range(n_eval):
        obs, _ = eval_env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
        print(f"  ep {ep+1:2d}: {total:.1f}")
    eval_env.close()

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    print(f"\n  --> Mean: {mean_score:.1f} ± {std_score:.1f}")

    result = {
        "condition": name,
        "n_expert": n_expert,
        "n_poison": n_poison,
        "n_background": n_background,
        "feature_set": feature_set,
        "feature_dim": extractor.feature_dim,
        "objective": objective,
        "class_balanced": class_balanced,
        "balance_exponent": balance_exponent,
        "conditional_frame_stride": conditional_frame_stride,
        "conditional_max_states": conditional_max_states,
        "poison_pct": (
            n_poison / (n_expert + n_poison) if (n_expert + n_poison) > 0 else 0
        ),
        "irl_iters": irl_iters,
        "rl_steps": rl_steps,
        "theta": irl.theta.tolist(),
        "irl_final_ll": float(history[-1]) if history else None,
        "irl_metric_name": "nll" if objective == "conditional" else "ll",
        "conditional_metrics": (
            metrics_to_dict(conditional_metrics) if conditional_metrics is not None else None
        ),
        "eval_scores": scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "seed": seed,
    }

    out_path = os.path.join(results_dir, f"{name}_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-dir",     default="data/raw/expert")
    parser.add_argument("--poison-dir",     default="data/raw/poison_human_stop")
    parser.add_argument("--background-dir", default="data/raw/background",
                        help="Random-policy trajectories used for partition function Z(θ).")
    parser.add_argument("--n-background",   type=int, default=100)
    parser.add_argument("--results-dir",    default="results/preliminary")
    parser.add_argument("--n-expert",       type=int, default=200)
    parser.add_argument("--irl-iters",      type=int, default=1000)
    parser.add_argument("--rl-steps",       type=int, default=500_000)
    parser.add_argument("--n-eval",         type=int, default=5)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--features",       choices=["v1", "v2", "v3"], default="v2",
                        help="Feature extractor to use for IRL reward learning.")
    parser.add_argument("--objective",      choices=["trajectory", "conditional"], default="trajectory",
                        help="MaxEnt objective: old trajectory/background objective or action-counterfactual conditional objective.")
    parser.add_argument("--conditional-frame-stride", type=int, default=10,
                        help="Use every Nth sampled trajectory frame for conditional MaxEnt.")
    parser.add_argument("--conditional-max-states", type=int, default=6000,
                        help="Maximum state/action pairs used by conditional MaxEnt; <=0 uses all sampled pairs.")
    parser.add_argument("--class-balanced", action="store_true",
                        help="For conditional MaxEnt, reweight states so rare expert actions matter more.")
    parser.add_argument("--balance-exponent", type=float, default=1.0,
                        help="Class-balancing strength. 1.0 is full inverse-frequency; 0.5 is milder.")
    parser.add_argument("--conditional-lr", type=float, default=0.5)
    parser.add_argument("--conditional-l2", type=float, default=1e-3)
    args = parser.parse_args()

    n_total     = args.n_expert   # 200
    n_poison_10 = 20              # all 20 stop demos
    n_expert_10 = n_total - n_poison_10  # 180

    results = []

    # D0: 0% poison
    results.append(
        run_condition(
            name="D0_0pct",
            n_expert=n_total,
            n_poison=0,
            expert_dir=args.expert_dir,
            poison_dir=args.poison_dir,
            background_dir=args.background_dir,
            n_background=args.n_background,
            irl_iters=args.irl_iters,
            rl_steps=args.rl_steps,
            n_eval=args.n_eval,
            results_dir=args.results_dir,
            seed=args.seed,
            feature_set=args.features,
            objective=args.objective,
            conditional_frame_stride=args.conditional_frame_stride,
            conditional_max_states=args.conditional_max_states,
            class_balanced=args.class_balanced,
            balance_exponent=args.balance_exponent,
            conditional_lr=args.conditional_lr,
            conditional_l2=args.conditional_l2,
        )
    )

    # D10: 10% poison
    results.append(
        run_condition(
            name="D10_10pct",
            n_expert=n_expert_10,
            n_poison=n_poison_10,
            expert_dir=args.expert_dir,
            poison_dir=args.poison_dir,
            background_dir=args.background_dir,
            n_background=args.n_background,
            irl_iters=args.irl_iters,
            rl_steps=args.rl_steps,
            n_eval=args.n_eval,
            results_dir=args.results_dir,
            seed=args.seed,
            feature_set=args.features,
            objective=args.objective,
            conditional_frame_stride=args.conditional_frame_stride,
            conditional_max_states=args.conditional_max_states,
            class_balanced=args.class_balanced,
            balance_exponent=args.balance_exponent,
            conditional_lr=args.conditional_lr,
            conditional_l2=args.conditional_l2,
        )
    )

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':<15} {'Poison%':>8}  {'Mean Score':>12}  {'Std':>8}")
    print(f"  {'-'*45}")
    for r in results:
        print(
            f"  {r['condition']:<15} {r['poison_pct']*100:>7.0f}%  "
            f"{r['mean_score']:>12.1f}  {r['std_score']:>8.1f}"
        )


if __name__ == "__main__":
    main()
