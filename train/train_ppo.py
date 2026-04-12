"""
PPO Training Script for 5G Network Slicing Dynamic Pricing
============================================================
Trains a Proximal Policy Optimization agent on the NetworkSlicingEnv.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from env.network_slicing_env import NetworkSlicingEnv
from train.config import ENV_CONFIG, PPO_CONFIG, EVAL_CONFIG
from train.train_sac import evaluate_policy, EpisodeLogCallback


def train_ppo(seed=42, total_timesteps=None):
    """Train PPO and return model + training log."""
    print(f"\n{'='*60}")
    print(f"PPO Training (seed={seed})")
    print(f"{'='*60}")

    env = Monitor(NetworkSlicingEnv(config=ENV_CONFIG))
    ts = total_timesteps or PPO_CONFIG["total_timesteps"]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=PPO_CONFIG["learning_rate"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        clip_range=PPO_CONFIG["clip_range"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        gamma=PPO_CONFIG["gamma"],
        policy_kwargs=PPO_CONFIG["policy_kwargs"],
        seed=seed,
        verbose=0,
    )

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=ts, callback=callback, progress_bar=True)

    return model, callback


def main():
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)

    seed = EVAL_CONFIG["seeds"][0]
    model, cb = train_ppo(seed=seed)
    train_rewards = cb.episode_rewards

    model_path = f"experiments/models/ppo_seed{seed}"
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    n_eval = EVAL_CONFIG['n_eval_episodes']
    print(f"\nEvaluating ({n_eval} episodes)...")
    results = evaluate_policy(
        model, ENV_CONFIG, n_eval, seed)

    rewards = [r["total_reward"] for r in results]
    revenues = [r["total_revenue"] for r in results]
    penalties = [r["total_penalty"] for r in results]

    print(f"\n{'='*60}")
    print(f"PPO Evaluation Results (seed={seed})")
    print(f"{'='*60}")
    print(f"  Reward:  {np.mean(rewards):>14,.0f}"
          f" ± {np.std(rewards):>10,.0f}")
    print(f"  Revenue: {np.mean(revenues):>14,.0f}"
          f" ± {np.std(revenues):>10,.0f}")
    print(f"  Penalty: {np.mean(penalties):>14,.0f}"
          f" ± {np.std(penalties):>10,.0f}")
    nu = np.mean([r['final_N_U'] for r in results])
    ne = np.mean([r['final_N_E'] for r in results])
    print(f"  Final N_U: {nu:>8,.0f}")
    print(f"  Final N_E: {ne:>8,.0f}")

    save_data = {
        "train_rewards": train_rewards,
        "train_final_N_U": cb.episode_final_N_U,
        "train_final_N_E": cb.episode_final_N_E,
        "eval_results": [
            {k: v for k, v in r.items() if k != "trajectory"}
            for r in results],
    }
    result_path = f"experiments/results/ppo_seed{seed}.json"
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()
