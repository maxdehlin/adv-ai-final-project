"""
Train a PPO expert on CarRacing-v3 (discrete) and save the model.

Usage:
    python -m data_collection.train_expert
    python -m data_collection.train_expert --timesteps 3000000 --out models/expert_ppo.zip
    python -m data_collection.train_expert --eval-only --model models/expert_ppo.zip
"""

import argparse
import csv
import datetime
import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage


LOG_DIR  = "logs/training"
LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")


class TrainingLogger(BaseCallback):
    """
    Logs per-update metrics to stdout and a CSV file:
        timestep, elapsed_s, mean_ep_reward, mean_ep_len,
        policy_loss, value_loss, entropy_loss, approx_kl
    """

    def __init__(self, log_path: str = LOG_FILE, print_freq: int = 10):
        super().__init__(verbose=0)
        self.log_path  = log_path
        self.print_freq = print_freq   # print every N updates
        self._start_time = None
        self._update = 0
        self._csv_file = None
        self._writer   = None

    def _on_training_start(self):
        self._start_time = time.time()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._csv_file = open(self.log_path, "w", newline="")
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow([
            "timestep", "elapsed_s", "updates",
            "mean_ep_reward", "mean_ep_len",
            "policy_loss", "value_loss", "entropy_loss", "approx_kl",
        ])
        print(f"\n{'='*60}")
        print(f"  Training started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  Log: {self.log_path}")
        print(f"{'='*60}")
        print(f"{'Step':>10}  {'Elapsed':>8}  {'MeanRew':>9}  {'EpLen':>7}  "
              f"{'PolicyLoss':>11}  {'ValueLoss':>10}  {'Entropy':>9}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        self._update += 1
        elapsed = time.time() - self._start_time
        ts      = self.num_timesteps

        # --- fetch logger values (SB3 stores them in self.logger) ---
        def _get(key, default=float("nan")):
            try:
                return self.logger.name_to_value[key]
            except (KeyError, AttributeError):
                return default

        mean_rew    = _get("rollout/ep_rew_mean")
        mean_len    = _get("rollout/ep_len_mean")
        policy_loss = _get("train/policy_gradient_loss")
        value_loss  = _get("train/value_loss")
        entropy     = _get("train/entropy_loss")
        approx_kl   = _get("train/approx_kl")

        self._writer.writerow([
            ts, f"{elapsed:.1f}", self._update,
            f"{mean_rew:.2f}", f"{mean_len:.1f}",
            f"{policy_loss:.5f}", f"{value_loss:.5f}",
            f"{entropy:.5f}", f"{approx_kl:.5f}",
        ])
        self._csv_file.flush()

        if self._update % self.print_freq == 0:
            print(f"{ts:>10,}  {elapsed:>7.0f}s  {mean_rew:>9.2f}  {mean_len:>7.1f}  "
                  f"{policy_loss:>11.5f}  {value_loss:>10.5f}  {entropy:>9.5f}")

    def _on_training_end(self):
        elapsed = time.time() - self._start_time
        print(f"\nTraining finished in {elapsed:.0f}s ({elapsed/3600:.2f}h)")
        if self._csv_file:
            self._csv_file.close()


def make_env():
    return gym.make("CarRacing-v3", continuous=False)


def train(timesteps: int, out_path: str, eval_freq: int = 50_000):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_env = VecTransposeImage(make_vec_env(make_env, n_envs=4))
    eval_env  = VecTransposeImage(make_vec_env(make_env, n_envs=1))

    callbacks = [
        TrainingLogger(log_path=LOG_FILE),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(out_path),
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
        verbose=0,                        # silenced — TrainingLogger handles output
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
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
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
