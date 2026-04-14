"""
Roll out a policy and save trajectories to disk.

Each trajectory is saved as a compressed .npz file:
    states:  (T, 96, 96, 3) uint8
    actions: (T,)           int

Usage:
    # Collect 200 expert demos
    python -m data_collection.collect_demos --policy expert --n 200 --out data/raw/expert

    # Collect 100 random-policy poison demos
    python -m data_collection.collect_demos --policy random --n 100 --out data/raw/poison_random

    # Load a specific model path
    python -m data_collection.collect_demos --policy expert --model models/expert_ppo.zip --n 50
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


def _make_env(render: bool = False):
    render_mode = "human" if render else None
    return gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)


def _random_policy(_obs):
    return np.random.randint(0, 5)


def _load_expert(model_path: str):
    model = PPO.load(model_path)
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


def collect(
    policy_fn,
    n_episodes: int,
    out_dir: str,
    max_steps: int = 1000,
    render: bool = False,
    start_idx: int = 0,
    min_score: float = None,
    max_attempts: int = None,
) -> list[str]:
    """
    Roll out policy_fn and save trajectories to out_dir.

    If min_score is set, episodes below that threshold are discarded and
    re-rolled until n_episodes qualifying trajectories are collected.
    max_attempts caps total rollouts to prevent infinite loops (default 5×n).

    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    env = _make_env(render=render)
    saved_paths = []

    if max_attempts is None:
        max_attempts = n_episodes * 5 if min_score is not None else n_episodes

    attempts = 0
    saved = 0

    while saved < n_episodes and attempts < max_attempts:
        obs, _ = env.reset()
        states, actions, rewards = [], [], []
        done = False
        step = 0

        while not done and step < max_steps:
            action = policy_fn(obs)
            states.append(obs.copy())
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
            step += 1

        attempts += 1
        episode_score = float(sum(rewards))

        if min_score is not None and episode_score < min_score:
            print(f"  [attempt {attempts:3d}] score={episode_score:7.1f}  DISCARDED (< {min_score})")
            continue

        states_arr  = np.array(states,  dtype=np.uint8)
        actions_arr = np.array(actions, dtype=np.int8)

        idx  = start_idx + saved
        path = os.path.join(out_dir, f"traj_{idx:04d}.npz")
        np.savez_compressed(path, states=states_arr, actions=actions_arr)
        saved_paths.append(path)
        saved += 1

        print(f"  [{saved:3d}/{n_episodes}] score={episode_score:7.1f}  saved {path}  (T={step})")

    env.close()

    if saved < n_episodes:
        print(f"\nWARNING: only collected {saved}/{n_episodes} episodes "
              f"after {attempts} attempts (min_score={min_score})")
    else:
        print(f"\nDone. {saved} trajectories saved to {out_dir}  "
              f"({attempts} total attempts)")

    return saved_paths


def load_trajectory(path: str):
    """Load a saved trajectory. Returns (states, actions)."""
    data = np.load(path)
    return data["states"], data["actions"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", choices=["expert", "random"], required=True,
        help="'expert' loads a trained PPO model; 'random' uses a uniform random policy",
    )
    parser.add_argument("--model", type=str, default="models/expert_ppo.zip",
                        help="Path to PPO model (only used when --policy expert)")
    parser.add_argument("--n", type=int, default=200, help="Number of episodes to collect")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: data/raw/expert or data/raw/poison_random)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting index for filenames (useful for appending to existing data)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Discard episodes below this score (e.g. 850 for quality filtering)")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Max rollout attempts when using --min-score (default: 5x n)")
    args = parser.parse_args()

    if args.policy == "expert":
        print(f"Loading expert model from {args.model}...")
        policy_fn = _load_expert(args.model)
        out_dir = args.out or "data/raw/expert"
    else:
        policy_fn = _random_policy
        out_dir = args.out or "data/raw/poison_random"

    print(f"Collecting {args.n} episodes → {out_dir}"
          + (f"  (min_score={args.min_score})" if args.min_score else ""))
    paths = collect(
        policy_fn,
        n_episodes=args.n,
        out_dir=out_dir,
        max_steps=args.max_steps,
        render=args.render,
        start_idx=args.start_idx,
        min_score=args.min_score,
        max_attempts=args.max_attempts,
    )
    print(f"\nDone. {len(paths)} trajectories saved to {out_dir}")
