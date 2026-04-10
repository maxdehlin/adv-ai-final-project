"""
Train a PPO expert on CarRacing-v2 (discrete) and save the model.

Usage:
    python -m data_collection.train_expert
    python -m data_collection.train_expert --timesteps 3000000 --out models/expert_ppo.zip
    python -m data_collection.train_expert --eval-only --model models/expert_ppo.zip
"""

import argparse
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def make_env():
    return gym.make("CarRacing-v2", continuous=False)


def train(timesteps: int, out_path: str, eval_freq: int = 50_000):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Vectorised training env (4 parallel)
    train_env = VecTransposeImage(make_vec_env(make_env, n_envs=4))

    # Single eval env
    eval_env = VecTransposeImage(make_vec_env(make_env, n_envs=1))

    callbacks = [
        EvalCallback(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=5,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=eval_freq,
            save_path=os.path.dirname(out_path),
            name_prefix="expert_ppo_ckpt",
        ),
    ]

    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    print(f"Training for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(out_path)
    print(f"Model saved to {out_path}")

    train_env.close()
    eval_env.close()
    return model


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v2", continuous=False, render_mode=render_mode)
    model = PPO.load(model_path)

    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        scores.append(total_reward)
        print(f"  Episode {ep+1:2d}: score = {total_reward:.1f}")

    env.close()
    print(f"\nMean score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--out", type=str, default="models/expert_ppo.zip")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", type=str, default="models/expert_ppo.zip")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model, render=args.render)
    else:
        train(args.timesteps, args.out)
        print("\nEvaluating trained model...")
        evaluate(args.out)
