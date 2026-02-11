"""
SB3 SAC training for 5G O-RAN 3-Part Tariff environment (§12).

Features:
  - Automatic device selection: MPS (Apple) → CUDA → CPU  [MPS_PYTORCH]
  - ent_coef="auto"  [SB3 SAC documentation]
  - tqdm progress bar (Requirement 16)
  - Per-step CSV logging for later analysis

Usage:
  python -m oran3pt.train --config config/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .utils import load_config, select_device
from .env import OranSlicingPricingEnv

logger = logging.getLogger("oran3pt.train")


class CSVLogger:
    """Append step-level info dicts to a CSV file."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None
        self._writer = None
        self._fields: Optional[List[str]] = None

    def log(self, record: Dict[str, Any]) -> None:
        if self._writer is None:
            self._fields = list(record.keys())
            self._f = open(self._path, "w", newline="")
            self._writer = csv.DictWriter(self._f, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow(record)
        self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None
            self._writer = None


def _run_random_baseline(cfg: Dict[str, Any], total_steps: int,
                         csv_path: str, users_csv: Optional[str] = None,
                         seed: int = 42) -> Dict[str, float]:
    """Fallback training loop with random actions (no SB3 needed)."""
    env = OranSlicingPricingEnv(cfg, users_csv=users_csv, seed=seed)
    obs, _ = env.reset(seed=seed)
    csvlog = CSVLogger(csv_path)
    rewards: List[float] = []

    for step in tqdm(range(total_steps), desc="Random baseline"):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        csvlog.log(info)
        rewards.append(reward)
        if terminated or truncated:
            obs, _ = env.reset()

    csvlog.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def train(cfg: Dict[str, Any],
          users_csv: Optional[str] = None,
          output_dir: str = "outputs") -> Path:
    """Train SAC agent; return path to saved model."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = select_device()
    logger.info("Training device: %s", device)

    tc = cfg.get("training", {})
    total_timesteps = tc.get("total_timesteps", 200_000)

    csv_path = str(out / "train_log.csv")

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback

        env = OranSlicingPricingEnv(cfg, users_csv=users_csv)
        csvlog = CSVLogger(csv_path)

        class _LogCallback(BaseCallback):
            def __init__(self, csvl: CSVLogger):
                super().__init__()
                self.csvl = csvl

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                if infos:
                    self.csvl.log(infos[0])
                return True

        model = SAC(
            "MlpPolicy", env,
            learning_rate=tc.get("learning_rate", 3e-4),
            batch_size=tc.get("batch_size", 256),
            buffer_size=tc.get("buffer_size", 100_000),
            gamma=tc.get("gamma", 0.99),
            tau=tc.get("tau", 0.005),
            ent_coef=tc.get("ent_coef", "auto"),
            train_freq=tc.get("train_freq", 1),
            gradient_steps=tc.get("gradient_steps", 1),
            device=device,
            verbose=0,
        )

        cb = _LogCallback(csvlog)
        model.learn(total_timesteps=total_timesteps,
                    callback=cb, progress_bar=True)

        model_path = out / "best_model"
        model.save(str(model_path))
        csvlog.close()
        logger.info("Model saved → %s", model_path)
        return model_path

    except ImportError:
        logger.warning("SB3 not installed — running random baseline.")
        stats = _run_random_baseline(cfg, total_timesteps, csv_path,
                                     users_csv=users_csv)
        logger.info("Random baseline: %s", stats)
        return out / "train_log.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC (3-part tariff)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--users", default="data/users_init.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(name)s] %(message)s")
    cfg = load_config(args.config)
    users_csv = args.users if Path(args.users).exists() else None
    train(cfg, users_csv=users_csv, output_dir=args.output)


if __name__ == "__main__":
    main()
