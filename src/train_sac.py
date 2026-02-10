"""
Training script for O-RAN Slicing + Pricing with SB3 SAC.

Usage:
  python -m src.train_sac --config config/calibrated.yaml

References:
  [SAC][SB3_SAC][SB3_TIPS][MPS_PYTORCH][MPS_APPLE][TQDM]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("oran.train")


class CSVLogCallback:
    """SB3 callback that logs per-step info dict to CSV."""

    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._fieldnames: Optional[List[str]] = None

    def _on_step(self, info: Dict[str, Any], step: int) -> None:
        record = {"step": step}
        for key in ["month", "rho_URLLC", "profit", "revenue",
                     "cost_total", "reward"]:
            if key in info:
                record[key] = info[key]
        for sname in ["eMBB", "URLLC"]:
            for prefix in ["fee", "N_active", "N_post_churn", "joins",
                           "churns", "V_rate", "avg_T", "rho_util", "topups"]:
                flat_key = f"{prefix}_{sname}"
                if flat_key in info:
                    record[flat_key] = info[flat_key]
        if self._writer is None:
            self._fieldnames = list(record.keys())
            self._file = open(self.log_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(record)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class SB3CSVCallback:
    """Adapts CSVLogCallback to SB3's BaseCallback protocol."""

    def __init__(self, csv_logger: CSVLogCallback) -> None:
        self.csv_logger = csv_logger
        self.n_calls = 0

    def __call__(self, locals_dict: Dict[str, Any],
                 globals_dict: Dict[str, Any]) -> bool:
        self.n_calls += 1
        infos = locals_dict.get("infos", [])
        if infos:
            self.csv_logger._on_step(infos[0], self.n_calls)
        return True


def train_sac(cfg: Dict[str, Any], run_dir: Path) -> Path:
    """Train SAC agent and return path to saved model."""
    from src.envs.oran_slicing_env import OranSlicingEnv
    from src.models.utils import select_device

    device = select_device()
    logger.info("Training device: %s", device)

    train_cfg = cfg.get("training", {})
    total_timesteps = train_cfg.get("total_timesteps", 50000)
    lr = train_cfg.get("learning_rate", 0.0003)
    batch_size = train_cfg.get("batch_size", 256)
    buffer_size = train_cfg.get("buffer_size", 50000)
    gamma = train_cfg.get("gamma", 0.99)
    tau = train_cfg.get("tau", 0.005)
    ent_coef = train_cfg.get("ent_coef", "auto")
    train_freq = train_cfg.get("train_freq", 1)
    gradient_steps = train_cfg.get("gradient_steps", 1)
    n_repeats = train_cfg.get("n_repeats", 3)

    best_model_path = run_dir / "best_model"
    all_rewards: List[List[float]] = []

    for rep in range(n_repeats):
        logger.info("=== Training repeat %d/%d ===", rep + 1, n_repeats)
        env = OranSlicingEnv(cfg)

        csv_logger = CSVLogCallback(str(run_dir / f"train_log_rep{rep}.csv"))

        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import BaseCallback

            class _CB(BaseCallback):
                def __init__(self, csv_log: CSVLogCallback):
                    super().__init__()
                    self.csv_log = csv_log

                def _on_step(self) -> bool:
                    infos = self.locals.get("infos", [])
                    if infos:
                        self.csv_log._on_step(infos[0], self.num_timesteps)
                    return True

            model = SAC(
                "MlpPolicy", env,
                learning_rate=lr, batch_size=batch_size,
                buffer_size=buffer_size, gamma=gamma, tau=tau,
                ent_coef=ent_coef, train_freq=train_freq,
                gradient_steps=gradient_steps,
                device=device, verbose=0,
            )

            callback = _CB(csv_logger)
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True,
            )

            model.save(str(best_model_path))
            logger.info("Model saved: %s", best_model_path)

        except ImportError:
            logger.warning(
                "stable-baselines3 not installed. "
                "Running random policy training loop instead."
            )
            obs, info = env.reset()
            ep_rewards = []
            for step in tqdm(range(total_timesteps), desc=f"Rep {rep}"):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                csv_logger._on_step(info, step)
                ep_rewards.append(reward)
                if terminated or truncated:
                    obs, info = env.reset()
            all_rewards.append(ep_rewards)

        finally:
            csv_logger.close()

    if all_rewards:
        means = [np.mean(r) for r in all_rewards]
        logger.info(
            "Training complete. Mean reward across repeats: %.4f ± %.4f",
            np.mean(means), np.std(means),
        )

    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC for O-RAN slicing")
    parser.add_argument("--config", type=str, default="config/calibrated.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )

    from src.models.utils import load_config, ensure_artifacts_dir

    cfg_path = args.config
    if not Path(cfg_path).exists():
        cfg_path = "config/default.yaml"
        logger.warning("Calibrated config not found, using %s", cfg_path)

    cfg = load_config(cfg_path)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = ensure_artifacts_dir(cfg)

    logger.info("Artifacts directory: %s", run_dir)
    train_sac(cfg, run_dir)


if __name__ == "__main__":
    main()
