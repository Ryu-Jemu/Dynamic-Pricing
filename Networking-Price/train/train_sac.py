"""
SAC Training Script for 5G Network Slicing Dynamic Pricing
============================================================
Trains a Soft Actor-Critic agent on the NetworkSlicingEnv.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from env.network_slicing_env import NetworkSlicingEnv
from train.config import ENV_CONFIG, SAC_CONFIG, EVAL_CONFIG


class EpisodeLogCallback(BaseCallback):
    """Logs episode reward at each episode end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                ep_num = len(self.episode_rewards)
                if ep_num % 50 == 0:
                    recent = self.episode_rewards[-50:]
                    print(f"  Episode {ep_num}: "
                          f"mean_reward={np.mean(recent):,.0f} "
                          f"± {np.std(recent):,.0f}")
        return True


def evaluate_policy(model, env_config, n_episodes=20, seed=42):
    """Evaluate a trained policy over n_episodes."""
    env = NetworkSlicingEnv(config=env_config)
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        total_revenue = 0.0
        total_penalty = 0.0
        trajectory = []

        for t in range(env.T):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_revenue += info["revenue"]
            total_penalty += info["penalty"]
            if t % 100 == 0 or t == env.T - 1:
                trajectory.append({
                    "t": t,
                    "F_U": info["F_U"], "p_U": info["p_U"],
                    "F_E": info["F_E"], "p_E": info["p_E"],
                    "N_U": info["N_U"], "N_E": info["N_E"],
                    "eta_U": info["eta_U"], "eta_E": info["eta_E"],
                    "revenue": info["revenue"], "penalty": info["penalty"],
                })

        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "total_revenue": total_revenue,
            "total_penalty": total_penalty,
            "final_N_U": info["N_U"],
            "final_N_E": info["N_E"],
            "trajectory": trajectory,
        })

    return results


def train_sac(seed=42, total_timesteps=None):
    """Train SAC and return model + training log."""
    print(f"\n{'='*60}")
    print(f"SAC Training (seed={seed})")
    print(f"{'='*60}")

    env = Monitor(NetworkSlicingEnv(config=ENV_CONFIG))
    ts = total_timesteps or SAC_CONFIG["total_timesteps"]

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=SAC_CONFIG["learning_rate"],
        batch_size=SAC_CONFIG["batch_size"],
        buffer_size=SAC_CONFIG["buffer_size"],
        tau=SAC_CONFIG["tau"],
        gamma=SAC_CONFIG["gamma"],
        ent_coef=SAC_CONFIG["ent_coef"],
        policy_kwargs=SAC_CONFIG["policy_kwargs"],
        seed=seed,
        verbose=0,
    )

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=ts, callback=callback, progress_bar=True)

    return model, callback.episode_rewards


def main():
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)

    seed = EVAL_CONFIG["seeds"][0]
    model, train_rewards = train_sac(seed=seed)

    # Save model
    model_path = f"experiments/models/sac_seed{seed}"
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # Evaluate
    print(f"\nEvaluating ({EVAL_CONFIG['n_eval_episodes']} episodes)...")
    results = evaluate_policy(model, ENV_CONFIG, EVAL_CONFIG["n_eval_episodes"], seed)

    rewards = [r["total_reward"] for r in results]
    revenues = [r["total_revenue"] for r in results]
    penalties = [r["total_penalty"] for r in results]

    print(f"\n{'='*60}")
    print(f"SAC Evaluation Results (seed={seed})")
    print(f"{'='*60}")
    print(f"  Reward:  {np.mean(rewards):>14,.0f} ± {np.std(rewards):>10,.0f}")
    print(f"  Revenue: {np.mean(revenues):>14,.0f} ± {np.std(revenues):>10,.0f}")
    print(f"  Penalty: {np.mean(penalties):>14,.0f} ± {np.std(penalties):>10,.0f}")
    print(f"  Final N_U: {np.mean([r['final_N_U'] for r in results]):>8,.0f}")
    print(f"  Final N_E: {np.mean([r['final_N_E'] for r in results]):>8,.0f}")

    # Save results
    save_data = {
        "train_rewards": train_rewards,
        "eval_results": [{k: v for k, v in r.items() if k != "trajectory"}
                         for r in results],
        "config": {"env": str(ENV_CONFIG), "sac": str(SAC_CONFIG)},
    }
    result_path = f"experiments/results/sac_seed{seed}.json"
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()
