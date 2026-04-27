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

    # Collect scripted targeted poison demos
    python -m data_collection.collect_demos --policy gas --n 100 --out data/raw/poison_gas
    python -m data_collection.collect_demos --policy expert-then-stop --switch-step 250 --n 50

    # Load a specific model path
    python -m data_collection.collect_demos --policy expert --model models/expert_ppo.zip --n 50
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


ACTION_NOTHING = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2
ACTION_GAS = 3
ACTION_BRAKE = 4


def _make_env(render: bool = False):
    render_mode = "human" if render else None
    return gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)


def _random_policy(_obs):
    return np.random.randint(0, 5)


def _stop_policy(_obs):
    """Always brake (action 4). Car slows and stops — clearly suboptimal."""
    return ACTION_BRAKE


def _nothing_policy(_obs):
    """Never act. Useful as a low-effort poison baseline."""
    return ACTION_NOTHING


def _gas_policy(_obs):
    """Always accelerate. Biases action statistics toward gas but fails turns."""
    return ACTION_GAS


def _left_policy(_obs):
    """Always steer left. Creates targeted steering-bias poison."""
    return ACTION_LEFT


def _right_policy(_obs):
    """Always steer right. Creates targeted steering-bias poison."""
    return ACTION_RIGHT


class ZigZagPolicy:
    """Accelerates while alternating left/right steering on a fixed period."""

    def __init__(self, period: int = 12):
        self.period = max(1, int(period))
        self.t = 0

    def reset(self):
        self.t = 0

    def __call__(self, _obs):
        phase = (self.t // self.period) % 2
        self.t += 1
        return ACTION_GAS if self.t % 2 == 0 else (ACTION_LEFT if phase == 0 else ACTION_RIGHT)


class ExpertThenPolicy:
    """
    A stealthier targeted poison: follow the expert at first, then sabotage.

    The prefix makes trajectories look less trivially random, while the suffix
    injects a targeted bad behavior such as stopping, turning, or accelerating
    through turns.
    """

    def __init__(self, model_path: str, switch_step: int, poison_action: int):
        self.model = PPO.load(model_path)
        self.switch_step = max(0, int(switch_step))
        self.poison_action = int(poison_action)
        self.t = 0

    def reset(self):
        self.t = 0

    def __call__(self, obs):
        if self.t < self.switch_step:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = self.poison_action
        self.t += 1
        return action


SCRIPTED_POLICIES = {
    "stop": _stop_policy,
    "nothing": _nothing_policy,
    "gas": _gas_policy,
    "left": _left_policy,
    "right": _right_policy,
}

EXPERT_THEN_ACTIONS = {
    "expert-then-stop": ACTION_BRAKE,
    "expert-then-gas": ACTION_GAS,
    "expert-then-left": ACTION_LEFT,
    "expert-then-right": ACTION_RIGHT,
}


def _load_expert(model_path: str):
    model = PPO.load(model_path)
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


def _default_out_dir(policy_name: str) -> str:
    if policy_name == "expert":
        return "data/raw/expert"
    if policy_name == "random":
        return "data/raw/poison_random"
    if policy_name == "stop":
        return "data/raw/poison_human_stop"
    return f"data/raw/poison_{policy_name.replace('-', '_')}"


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
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
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
    policy_choices = (
        ["expert", "random", "zigzag"]
        + sorted(SCRIPTED_POLICIES.keys())
        + sorted(EXPERT_THEN_ACTIONS.keys())
    )
    parser.add_argument(
        "--policy", choices=policy_choices, required=True,
        help="Policy to roll out: expert, random, simple scripted poison, or expert-then-* targeted poison.",
    )
    parser.add_argument("--model", type=str, default="models/expert_ppo.zip",
                        help="Path to PPO model (used by expert and expert-then-* policies)")
    parser.add_argument("--switch-step", type=int, default=250,
                        help="Step where expert-then-* policies switch from expert to poison action")
    parser.add_argument("--zigzag-period", type=int, default=12,
                        help="Number of steps before zigzag switches steering direction")
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
        out_dir = args.out or _default_out_dir(args.policy)
    elif args.policy == "random":
        policy_fn = _random_policy
        out_dir = args.out or _default_out_dir(args.policy)
    elif args.policy == "zigzag":
        policy_fn = ZigZagPolicy(period=args.zigzag_period)
        out_dir = args.out or _default_out_dir(args.policy)
    elif args.policy in EXPERT_THEN_ACTIONS:
        print(f"Loading expert model from {args.model}...")
        policy_fn = ExpertThenPolicy(
            model_path=args.model,
            switch_step=args.switch_step,
            poison_action=EXPERT_THEN_ACTIONS[args.policy],
        )
        out_dir = args.out or _default_out_dir(args.policy)
    else:
        policy_fn = SCRIPTED_POLICIES[args.policy]
        out_dir = args.out or _default_out_dir(args.policy)

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
