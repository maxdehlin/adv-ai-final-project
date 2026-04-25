"""
Watch a policy drive in CarRacing-v3 (discrete).

Usage:
    # Watch the trained expert
    python visualize.py

    # Watch the random (poison) policy
    python visualize.py --policy random

    # Watch a specific saved model
    python visualize.py --model models/expert_ppo.zip --episodes 3

    # Replay a saved trajectory file
    python visualize.py --trajectory data/raw/expert/traj_0000.npz
"""

import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


ACTION_NAMES = {0: "NOTHING", 1: "RIGHT", 2: "LEFT", 3: "GAS", 4: "BRAKE"}


def _make_env():
    return gym.make("CarRacing-v3", continuous=False, render_mode="human")


def watch_policy(model_path: str, n_episodes: int, delay: float = 0.0):
    env = _make_env()
    model = PPO.load(model_path)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\nEpisode {ep + 1}/{n_episodes}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            if delay:
                time.sleep(delay)

        print(f"  Score: {total_reward:.1f}  |  Steps: {step}")

    env.close()


def watch_random(n_episodes: int, delay: float = 0.03):
    env = _make_env()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\nEpisode {ep + 1}/{n_episodes}  [random policy]")
        while not done:
            action = np.random.randint(0, 5)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            if delay:
                time.sleep(delay)

        print(f"  Score: {total_reward:.1f}  |  Steps: {step}")

    env.close()


def replay_trajectory(path: str, delay: float = 0.03):
    """Replay a saved .npz trajectory frame by frame."""
    data = np.load(path)
    states  = data["states"]   # (T, 96, 96, 3)
    actions = data["actions"]  # (T,)
    T = len(states)

    # Use rgb_array mode and render manually via pygame
    import pygame
    pygame.init()
    scale = 6
    screen = pygame.display.set_mode((96 * scale, 96 * scale))
    pygame.display.set_caption(f"Trajectory replay: {path}")
    clock = pygame.time.Clock()

    print(f"\nReplaying {path}  ({T} steps)")
    total_reward = 0.0

    env = gym.make("CarRacing-v3", continuous=False)
    obs, _ = env.reset()

    for t in range(T):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                return

        action = int(actions[t])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Render the stored frame (not env frame) so it's faithful to the recording
        frame = states[t]
        surf = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))   # pygame is (W, H, C)
        )
        surf = pygame.transform.scale(surf, (96 * scale, 96 * scale))
        screen.blit(surf, (0, 0))

        # Overlay action and step info
        font = pygame.font.SysFont("monospace", 14)
        screen.blit(font.render(f"step {t+1}/{T}  action: {ACTION_NAMES[action]}", True, (255, 255, 0)), (8, 8))
        screen.blit(font.render(f"reward: {total_reward:.1f}", True, (255, 255, 0)), (8, 26))
        pygame.display.flip()

        if delay:
            clock.tick(1.0 / delay)

    print(f"  Total reward: {total_reward:.1f}")
    time.sleep(1.5)
    pygame.quit()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",     choices=["expert", "random"], default="expert")
    parser.add_argument("--model",      type=str,   default="models/expert_ppo.zip")
    parser.add_argument("--episodes",   type=int,   default=3)
    parser.add_argument("--delay",      type=float, default=0.0,
                        help="Seconds between frames (0 = as fast as possible)")
    parser.add_argument("--trajectory", type=str,   default=None,
                        help="Path to a .npz trajectory file to replay")
    args = parser.parse_args()

    if args.trajectory:
        replay_trajectory(args.trajectory, delay=args.delay or 0.03)
    elif args.policy == "random":
        watch_random(n_episodes=args.episodes, delay=args.delay or 0.03)
    else:
        watch_policy(args.model, n_episodes=args.episodes, delay=args.delay)
