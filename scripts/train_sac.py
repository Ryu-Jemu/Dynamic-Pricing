"""
Comparison training script: random vs trained SAC.

Usage:
  python scripts/train_sac.py --config config/calibrated.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("oran.scripts.train")


def run_random_baseline(cfg: Dict[str, Any], n_episodes: int = 5,
                        seed: int = 42) -> Dict[str, Any]:
    """Run random policy episodes for baseline comparison."""
    from src.envs.oran_slicing_env import OranSlicingEnv

    rng = np.random.default_rng(seed)
    all_rewards: List[float] = []
    all_profits: List[float] = []
    all_final_users: List[int] = []

    for ep in tqdm(range(n_episodes), desc="Random baseline"):
        ep_seed = int(rng.integers(0, 2**31))
        env = OranSlicingEnv(cfg, seed=ep_seed)
        obs, info = env.reset(seed=ep_seed)

        ep_rewards = []
        ep_profits = []
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rewards.append(reward)
            ep_profits.append(info.get("profit", 0))
            done = terminated or truncated

        all_rewards.extend(ep_rewards)
        all_profits.extend(ep_profits)
        final_n = (env.pool.active_count("eMBB")
                    + env.pool.active_count("URLLC"))
        all_final_users.append(final_n)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_profit": float(np.mean(all_profits)),
        "mean_final_users": float(np.mean(all_final_users)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparison: random vs SAC training"
    )
    parser.add_argument("--config", type=str, default="config/calibrated.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )

    from src.models.utils import load_config

    cfg_path = args.config
    if not Path(cfg_path).exists():
        cfg_path = "config/default.yaml"
    cfg = load_config(cfg_path)

    logger.info("Running random baseline (%d episodes)...", args.episodes)
    results = run_random_baseline(cfg, n_episodes=args.episodes)

    logger.info("Random baseline results:")
    for k, v in results.items():
        logger.info("  %s = %.4f", k, v)


if __name__ == "__main__":
    main()
