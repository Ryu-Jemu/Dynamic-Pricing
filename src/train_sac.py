"""
SAC training script (Section 18).

Uses SB3 SAC with MlpPolicy, ent_coef="auto".
Device: prefer mps if available; fallback to CPU.  [MPS_PYTORCH][MPS_APPLE]
Progress: tqdm for training.  [TQDM]
Evaluation: multiple repeats, report mean±std.  [SB3_TIPS]
Outputs: model checkpoint, CSV training log, config snapshot.

Usage:
  python -m src.train_sac --config config/default.yaml
  python -m src.train_sac --config config/calibrated.yaml

References:
  [SAC]       https://arxiv.org/abs/1801.01290
  [SB3_SAC]   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
  [SB3_TIPS]  https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
  [MPS_PYTORCH] https://pytorch.org/docs/stable/notes/mps.html
  [MPS_APPLE]   https://developer.apple.com/metal/pytorch/
  [TQDM]      https://tqdm.github.io/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from tqdm import tqdm

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.envs.oran_slicing_env import ORANSlicingEnv
from src.models.utils import (
    load_config,
    save_config,
    select_device,
    setup_logger,
)

logger = logging.getLogger("oran.train")


# =====================================================================
# Custom callbacks
# =====================================================================

class TQDMProgressCallback(BaseCallback):
    """tqdm progress bar for SB3 training.  [TQDM]"""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.pbar: Optional[tqdm] = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()


class CSVLogCallback(BaseCallback):
    """Log episode metrics to CSV for later plotting."""

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        self._file = None
        self._writer = None
        self._header_written = False

    def _on_training_start(self):
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)

    def _on_step(self) -> bool:
        # Log when episode ends (info has episode stats from Monitor)
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                row_dict = {
                    "timestep": self.num_timesteps,
                    "ep_reward": ep.get("r", 0),
                    "ep_length": ep.get("l", 0),
                }
                # Add env info if available
                for key in [
                    "profit", "revenue", "cost_total",
                    "mean_rho_util", "penalty",
                ]:
                    if key in info:
                        row_dict[key] = info[key]

                # Add slice-specific info
                for key in ["fees", "N_active", "V_rates", "n_joins", "n_churns"]:
                    if key in info and isinstance(info[key], dict):
                        for s, v in info[key].items():
                            row_dict[f"{key}_{s}"] = v

                if not self._header_written:
                    self._writer.writerow(row_dict.keys())
                    self._header_written = True
                self._writer.writerow(row_dict.values())
                self._file.flush()
        return True

    def _on_training_end(self):
        if self._file is not None:
            self._file.close()


# =====================================================================
# Training function
# =====================================================================

def train(config_path: str) -> str:
    """Train SAC agent and return the run directory path.

    Returns:
        str: path to artifacts/<run_id>/
    """
    cfg = load_config(config_path)
    tc = cfg.get("training", {})

    # ---- Run ID and artifact directory ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = cfg.get("artifacts", {}).get("base_dir", "artifacts")
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logger ----
    setup_logger("oran", logging.INFO)
    setup_logger("oran.train", logging.INFO)

    # ---- Device [MPS_PYTORCH][MPS_APPLE] ----
    device = select_device()
    logger.info("Device: %s", device)
    logger.info("Run dir: %s", run_dir)

    # ---- Save config snapshot ----
    save_config(cfg, run_dir / "config.yaml")
    shutil.copy2(config_path, run_dir / "config_source.yaml")

    # ---- Environments ----
    train_env = Monitor(ORANSlicingEnv(cfg))
    eval_env = Monitor(ORANSlicingEnv(cfg))

    # ---- SAC model [SAC][SB3_SAC] ----
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=tc.get("learning_rate", 3e-4),
        batch_size=tc.get("batch_size", 256),
        buffer_size=tc.get("buffer_size", 50000),
        gamma=tc.get("gamma", 0.99),
        tau=tc.get("tau", 0.005),
        ent_coef=tc.get("ent_coef", "auto"),
        train_freq=tc.get("train_freq", 1),
        gradient_steps=tc.get("gradient_steps", 1),
        verbose=0,
        device=device,
    )

    total_timesteps = tc.get("total_timesteps", 50000)
    logger.info("SAC model created: %d timesteps", total_timesteps)

    # ---- Callbacks ----
    csv_log_path = str(run_dir / "train_log.csv")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=tc.get("eval_freq_steps", 5000),
        n_eval_episodes=tc.get("eval_episodes", 5),
        deterministic=True,
        verbose=0,
    )

    progress_callback = TQDMProgressCallback(total_timesteps)
    csv_callback = CSVLogCallback(csv_log_path)

    # ---- Train ----
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, eval_callback, csv_callback],
        log_interval=tc.get("log_interval", 10),
    )
    train_time = time.time() - t0

    # ---- Save final model ----
    model.save(str(run_dir / "final_model"))
    logger.info("Training complete in %.1f seconds", train_time)

    # ---- Save run metadata ----
    meta = {
        "run_id": run_id,
        "config_path": config_path,
        "device": device,
        "total_timesteps": total_timesteps,
        "train_time_sec": round(train_time, 1),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Artifacts saved to: %s", run_dir)
    return str(run_dir)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC (Section 18)")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config (use calibrated.yaml for calibrated run)",
    )
    args = parser.parse_args()

    run_dir = train(args.config)
    print(f"\nRun directory: {run_dir}")
