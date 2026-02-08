"""
SAC Training Script: reward_type comparison (log vs tanh vs linear).

Usage:
  python scripts/train_sac.py                        # default: log, 1 seed
  python scripts/train_sac.py --reward_type tanh     # specific type
  python scripts/train_sac.py --all --seeds 3        # full comparison: 3 types × 3 seeds

Requirements:
  pip install stable-baselines3 shimmy gymnasium

Step 3 experiment design:
  - 3 reward_types × 3 seeds × 100k timesteps
  - SAC with ent_coef="auto" (α auto-tuning compensates reward scale)
  - Metrics: episode_reward_mean, profit, N_active, V_rate
  - Output: artifacts/{run_id}/results.json + tensorboard logs

Academic basis:
  Haarnoja et al. (2018): SAC with ent_coef="auto" automatically adjusts
  entropy temperature α to compensate reward scale changes. This means
  switching from tanh to log should not require hyperparameter tuning —
  α adapts to maintain the entropy-reward balance.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.utils import load_config, save_config, ensure_artifacts_dir
from src.envs.oran_slicing_env import OranSlicingEnv
from src.models.calibrate import calibrate_reward_scale, run_random_rollouts


# ──────────────────────────────────────────────────────────
# Calibration per reward_type
# ──────────────────────────────────────────────────────────

CALIBRATED_SCALES = {
    # Pre-computed from 20 random rollout episodes (seed=42)
    # See calibrate.py for methodology
    "log":    724_177.0,    # median(|profit|)
    "tanh":   1_364_050.0,  # p95(|profit|)
    "linear": 852_531.0,    # p95 / (clip × 0.8)
}


def make_env(cfg: Dict[str, Any], seed: int):
    """Create OranSlicingEnv with given config."""
    def _init():
        env = OranSlicingEnv(cfg, seed=seed)
        return env
    return _init


def run_experiment(
    reward_type: str,
    seed: int,
    total_timesteps: int,
    cfg_path: str,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Run single SAC training experiment.

    Returns dict with training metrics.
    """
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("  pip install stable-baselines3 shimmy")
        sys.exit(1)

    # Load and modify config
    cfg = load_config(cfg_path)
    cfg["economics"]["reward_type"] = reward_type
    cfg["economics"]["profit_scale"] = CALIBRATED_SCALES[reward_type]

    run_name = f"{reward_type}_seed{seed}"
    run_dir = artifacts_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    save_config(cfg, run_dir / "config.yaml")

    # Create environments
    train_env = Monitor(OranSlicingEnv(cfg, seed=seed))
    eval_env = Monitor(OranSlicingEnv(cfg, seed=seed + 1000))

    # SAC hyperparameters
    # Key: ent_coef="auto" — α auto-tuning compensates reward scale
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",          # Critical: auto-adjusts for reward scale
        target_entropy="auto",
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
        seed=seed,
        verbose=0,
        tensorboard_log=str(run_dir / "tb"),
    )

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=2_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    print(f"  Training {run_name}: {total_timesteps} timesteps...")
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name=run_name,
    )
    train_time = time.time() - t0

    # Evaluate final model
    eval_rewards = []
    eval_profits = []
    eval_n_active = []

    for ep in range(10):
        obs, info = eval_env.reset(seed=seed + 2000 + ep)
        total_reward = 0
        ep_profits = []
        ep_n = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            ep_profits.append(info.get("profit", 0))
            ep_n.append(info.get("N_active_eMBB", 0) + info.get("N_active_URLLC", 0))
            done = terminated or truncated
        eval_rewards.append(total_reward)
        eval_profits.append(np.mean(ep_profits))
        eval_n_active.append(np.mean(ep_n))

    # Results
    results = {
        "reward_type": reward_type,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "profit_scale": CALIBRATED_SCALES[reward_type],
        "train_time_sec": train_time,
        "eval_reward_mean": float(np.mean(eval_rewards)),
        "eval_reward_std": float(np.std(eval_rewards)),
        "eval_profit_mean": float(np.mean(eval_profits)),
        "eval_profit_std": float(np.std(eval_profits)),
        "eval_n_active_mean": float(np.mean(eval_n_active)),
        "final_ent_coef": float(model.ent_coef_tensor.item())
            if hasattr(model, "ent_coef_tensor") else None,
    }

    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    model.save(str(run_dir / "final_model"))
    train_env.close()
    eval_env.close()

    return results


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAC reward_type comparison")
    parser.add_argument("--reward_type", type=str, default="log",
                        choices=["tanh", "linear", "log"])
    parser.add_argument("--all", action="store_true",
                        help="Run all 3 reward_types")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per type")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps per run")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--artifacts", type=str, default=None,
                        help="Artifacts directory (default: auto)")
    args = parser.parse_args()

    # Artifacts directory
    if args.artifacts:
        artifacts_dir = Path(args.artifacts)
    else:
        artifacts_dir = ensure_artifacts_dir(
            load_config(args.config),
            run_id=f"step3_comparison_{int(time.time())}"
        )

    print(f"Artifacts: {artifacts_dir}")

    # Determine experiment matrix
    reward_types = ["tanh", "linear", "log"] if args.all else [args.reward_type]
    seeds = list(range(args.seeds))

    all_results = []

    for rt in reward_types:
        for s in seeds:
            seed = 42 + s * 100
            result = run_experiment(
                reward_type=rt,
                seed=seed,
                total_timesteps=args.timesteps,
                cfg_path=args.config,
                artifacts_dir=artifacts_dir,
            )
            all_results.append(result)
            print(f"  → {rt}/seed{seed}: "
                  f"eval_reward={result['eval_reward_mean']:.2f}±{result['eval_reward_std']:.2f}, "
                  f"profit={result['eval_profit_mean']:,.0f}, "
                  f"N_active={result['eval_n_active_mean']:.1f}")

    # Save combined results
    with open(artifacts_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Type':<8} {'Reward Mean':>12} {'Reward Std':>11} {'Profit Mean':>13} {'N_active':>10} {'ent_coef':>10}")
    print("-" * 80)
    for rt in reward_types:
        rt_results = [r for r in all_results if r["reward_type"] == rt]
        rm = np.mean([r["eval_reward_mean"] for r in rt_results])
        rs = np.mean([r["eval_reward_std"] for r in rt_results])
        pm = np.mean([r["eval_profit_mean"] for r in rt_results])
        nm = np.mean([r["eval_n_active_mean"] for r in rt_results])
        ec = np.mean([r["final_ent_coef"] for r in rt_results if r["final_ent_coef"]])
        print(f"{rt:<8} {rm:>12.2f} {rs:>11.2f} {pm:>13,.0f} {nm:>10.1f} {ec:>10.4f}")
    print("=" * 80)

    print(f"\nAll results saved to {artifacts_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
