"""
Baseline Evaluation Script
===========================
Evaluates 4 non-RL baselines on the NetworkSlicingEnv.
(RL baselines — 2PT-SAC, Flat-SAC, Myopic-SAC — are trained separately)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env.network_slicing_env import NetworkSlicingEnv
from train.config import ENV_CONFIG, EVAL_CONFIG, REFERENCE_ACTION


def run_baseline(env_config, policy_fn, name, n_episodes=20, seed=42):
    """Run a fixed-policy baseline for n_episodes and collect metrics."""
    env = NetworkSlicingEnv(config=env_config)
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        total_revenue = 0.0
        total_penalty = 0.0

        for t in range(env.T):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_revenue += info["revenue"]
            total_penalty += info["penalty"]

        results.append({
            "total_reward": total_reward,
            "total_revenue": total_revenue,
            "total_penalty": total_penalty,
            "final_N_U": info["N_U"],
            "final_N_E": info["N_E"],
        })

    rewards = [r["total_reward"] for r in results]
    revenues = [r["total_revenue"] for r in results]
    penalties = [r["total_penalty"] for r in results]

    print(f"\n  {name}:")
    print(f"    Reward:  {np.mean(rewards):>14,.0f} ± {np.std(rewards):>10,.0f}")
    print(f"    Revenue: {np.mean(revenues):>14,.0f} ± {np.std(revenues):>10,.0f}")
    print(f"    Penalty: {np.mean(penalties):>14,.0f} ± {np.std(penalties):>10,.0f}")
    print(f"    N_U={np.mean([r['final_N_U'] for r in results]):.0f}, "
          f"N_E={np.mean([r['final_N_E'] for r in results]):.0f}")

    return {
        "name": name,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_revenue": float(np.mean(revenues)),
        "mean_penalty": float(np.mean(penalties)),
        "mean_final_N_U": float(np.mean([r["final_N_U"] for r in results])),
        "mean_final_N_E": float(np.mean([r["final_N_E"] for r in results])),
        "raw_results": results,
    }


def main():
    os.makedirs("experiments/results", exist_ok=True)
    ref = np.array(REFERENCE_ACTION, dtype=np.float32)
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("Baseline Evaluation")
    print("=" * 60)

    all_results = {}

    # 1. Static-Heuristic: fixed reference prices
    all_results["static_heuristic"] = run_baseline(
        ENV_CONFIG,
        policy_fn=lambda obs: ref,
        name="Static-Heuristic (reference prices)",
        n_episodes=EVAL_CONFIG["n_eval_episodes"],
    )

    # 2. Random: uniform random actions
    all_results["random"] = run_baseline(
        ENV_CONFIG,
        policy_fn=lambda obs: rng.uniform(0, 1, size=4).astype(np.float32),
        name="Random",
        n_episodes=EVAL_CONFIG["n_eval_episodes"],
    )

    # 3. Zero-price: F=0, p=0
    all_results["zero_price"] = run_baseline(
        ENV_CONFIG,
        policy_fn=lambda obs: np.zeros(4, dtype=np.float32),
        name="Zero-Price (F=0, p=0)",
        n_episodes=EVAL_CONFIG["n_eval_episodes"],
    )

    # 4. Max-price: F=max, p=max
    all_results["max_price"] = run_baseline(
        ENV_CONFIG,
        policy_fn=lambda obs: np.ones(4, dtype=np.float32),
        name="Max-Price (F=100, p=20)",
        n_episodes=EVAL_CONFIG["n_eval_episodes"],
    )

    # Save
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "raw_results"}
                 for k, v in all_results.items()}
    with open("experiments/results/baselines.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: experiments/results/baselines.json")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Baseline':<35} {'Reward':>14} {'Revenue':>14} {'Penalty':>14}")
    print(f"{'-'*77}")
    for name, r in all_results.items():
        print(f"{r['name']:<35} {r['mean_reward']:>14,.0f} "
              f"{r['mean_revenue']:>14,.0f} {r['mean_penalty']:>14,.0f}")


if __name__ == "__main__":
    main()
