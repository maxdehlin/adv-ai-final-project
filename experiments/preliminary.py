"""
Preliminary MaxEnt IRL experiment — baseline only (no ANTIDOTE weighting).

Runs two conditions:
  - D0:   0% poison
  - D_25: 25% poison  (sampled from data/raw/poison_random/)

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

from maxent_irl import Trajectory, CarRacingFeatures, MaxEntIRL
from data_collection.collect_demos import load_trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trajectories_from_dir(directory: str, n: int, extractor: CarRacingFeatures,
                                label: str = "") -> list[Trajectory]:
    import glob
    paths = sorted(glob.glob(os.path.join(directory, "traj_*.npz")))
    if len(paths) < n:
        raise ValueError(f"Need {n} trajectories in {directory}, found {len(paths)}")
    paths = random.sample(paths, n)
    trajs = []
    for p in paths:
        states, actions = load_trajectory(p)
        traj = Trajectory(states=states, actions=actions)
        extractor.extract_trajectory(traj)
        trajs.append(traj)
    print(f"  Loaded {len(trajs)} {label} trajectories from {directory}")
    return trajs


def build_dataset(n_expert: int, n_poison: int,
                  expert_dir: str, poison_dir: str,
                  extractor: CarRacingFeatures) -> list[Trajectory]:
    trajs = []
    trajs += load_trajectories_from_dir(expert_dir, n_expert, extractor, "expert")
    if n_poison > 0:
        trajs += load_trajectories_from_dir(poison_dir, n_poison, extractor, "poison")
    random.shuffle(trajs)
    return trajs


# ---------------------------------------------------------------------------
# Reward wrapper: replaces env reward with learned IRL reward
# ---------------------------------------------------------------------------

class LearnedRewardEnv(gym.Wrapper):
    """Swap out the ground-truth reward for r_θ(s, a) = θ · φ(s, a)."""

    def __init__(self, env, irl_model: MaxEntIRL, extractor: CarRacingFeatures):
        super().__init__(env)
        self.irl   = irl_model
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
    irl_iters: int,
    rl_steps: int,
    n_eval: int,
    results_dir: str,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    print(f"\n{'='*60}")
    print(f"  Condition: {name}  ({n_expert} expert + {n_poison} poison)")
    print(f"{'='*60}")

    extractor = CarRacingFeatures()

    # --- 1. Load dataset ---
    print("\n[1/4] Loading trajectories...")
    trajs = build_dataset(n_expert, n_poison, expert_dir, poison_dir, extractor)
    print(f"  Total: {len(trajs)} trajectories, feature_dim={extractor.feature_dim}")

    # --- 2. Train MaxEnt IRL ---
    print(f"\n[2/4] Training MaxEnt IRL ({irl_iters} iterations)...")
    irl = MaxEntIRL(feature_dim=extractor.feature_dim, lr=0.05, l2=1e-4)
    history = irl.train(trajs, n_iter=irl_iters, verbose=True)
    print(f"  Final θ: {np.array2string(irl.theta, precision=3)}")

    # --- 3. Train RL with learned reward ---
    print(f"\n[3/4] Training PPO with learned reward ({rl_steps:,} steps)...")

    def make_learned_env():
        base = gym.make("CarRacing-v3", continuous=False)
        return LearnedRewardEnv(base, irl, CarRacingFeatures())

    train_env = VecTransposeImage(make_vec_env(make_learned_env, n_envs=4))
    rl_model = PPO("CnnPolicy", train_env, n_steps=512, batch_size=128,
                   n_epochs=10, learning_rate=3e-4, verbose=0)
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
    std_score  = float(np.std(scores))
    print(f"\n  --> Mean: {mean_score:.1f} ± {std_score:.1f}")

    result = {
        "condition": name,
        "n_expert": n_expert,
        "n_poison": n_poison,
        "poison_pct": n_poison / (n_expert + n_poison) if (n_expert + n_poison) > 0 else 0,
        "irl_iters": irl_iters,
        "rl_steps": rl_steps,
        "theta": irl.theta.tolist(),
        "irl_final_ll": float(history[-1]) if history else None,
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
    parser.add_argument("--expert-dir",  default="data/raw/expert")
    parser.add_argument("--poison-dir",  default="data/raw/poison_random")
    parser.add_argument("--results-dir", default="results/preliminary")
    parser.add_argument("--n-expert",    type=int,   default=200)
    parser.add_argument("--irl-iters",   type=int,   default=1000)
    parser.add_argument("--rl-steps",    type=int,   default=500_000)
    parser.add_argument("--n-eval",      type=int,   default=5)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    n_total  = args.n_expert
    n_poison_25 = int(round(n_total * 0.25))
    n_expert_25 = n_total - n_poison_25

    results = []

    # D0: 0% poison
    results.append(run_condition(
        name="D0_0pct",
        n_expert=n_total,
        n_poison=0,
        expert_dir=args.expert_dir,
        poison_dir=args.poison_dir,
        irl_iters=args.irl_iters,
        rl_steps=args.rl_steps,
        n_eval=args.n_eval,
        results_dir=args.results_dir,
        seed=args.seed,
    ))

    # D_25: 25% poison
    results.append(run_condition(
        name="D25_25pct",
        n_expert=n_expert_25,
        n_poison=n_poison_25,
        expert_dir=args.expert_dir,
        poison_dir=args.poison_dir,
        irl_iters=args.irl_iters,
        rl_steps=args.rl_steps,
        n_eval=args.n_eval,
        results_dir=args.results_dir,
        seed=args.seed,
    ))

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':<15} {'Poison%':>8}  {'Mean Score':>12}  {'Std':>8}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['condition']:<15} {r['poison_pct']*100:>7.0f}%  "
              f"{r['mean_score']:>12.1f}  {r['std_score']:>8.1f}")


if __name__ == "__main__":
    main()
