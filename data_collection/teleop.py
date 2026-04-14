"""
Human-controlled demonstration collection for CarRacing-v3 (discrete).

Drive the car yourself using arrow keys. Each completed episode is saved
as a trajectory file that can be used as poison or expert data.

Controls:
    ←  Steer left   (action 1)
    →  Steer right  (action 2)
    ↑  Gas           (action 3)
    ↓  Brake         (action 4)
    (no key)         Do nothing (action 0)

    R   Discard current episode and restart
    Q   Quit and save all completed episodes

Usage:
    python -m data_collection.teleop
    python -m data_collection.teleop --out data/raw/poison_human --n 20
"""

import argparse
import os
import numpy as np
import gymnasium as gym


def _get_action_from_keys(keys) -> int:
    import pygame
    if keys[pygame.K_LEFT]:
        return 2
    if keys[pygame.K_RIGHT]:
        return 1
    if keys[pygame.K_UP]:
        return 3
    if keys[pygame.K_DOWN]:
        return 4
    return 0


def run_teleop(out_dir: str, n_episodes: int, max_steps: int = 1000, start_idx: int = 0):
    import pygame

    os.makedirs(out_dir, exist_ok=True)
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")

    saved_paths = []
    ep = 0

    print("=" * 50)
    print("  CarRacing Teleop")
    print("  Arrow keys to drive.")
    print("  R = restart episode,  Q = quit")
    print("=" * 50)

    while ep < n_episodes:
        obs, _ = env.reset()
        states, actions = [], []
        done = False
        step = 0
        discarded = False

        print(f"\nEpisode {ep + 1}/{n_episodes} — drive!")

        while not done and step < max_steps:
            env.render()
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_q]:
                print("Quit.")
                env.close()
                return saved_paths

            if keys[pygame.K_r]:
                print("  Episode discarded, restarting.")
                discarded = True
                break

            action = _get_action_from_keys(keys)
            states.append(obs.copy())
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

        if discarded or len(states) == 0:
            continue

        states_arr  = np.array(states,  dtype=np.uint8)
        actions_arr = np.array(actions, dtype=np.int8)

        idx  = start_idx + ep
        path = os.path.join(out_dir, f"traj_{idx:04d}.npz")
        np.savez_compressed(path, states=states_arr, actions=actions_arr)
        saved_paths.append(path)
        print(f"  Saved {path}  (T={step})")
        ep += 1

    env.close()
    print(f"\nDone. {len(saved_paths)} trajectories saved to {out_dir}")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/raw/poison_human")
    parser.add_argument("--n", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()

    run_teleop(
        out_dir=args.out,
        n_episodes=args.n,
        max_steps=args.max_steps,
        start_idx=args.start_idx,
    )
