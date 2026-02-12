"""
SB3 SAC training for 5G O-RAN 3-Part Tariff environment (§12).

REVISION 4 — Fixes:
  [F1] Explicit SB3 import diagnostic (not silent fallback)
  [F2] Structured logging of training metrics
  [F5] CSVLogger crash on SB3 episode-boundary keys (terminal_observation, episode)

Root cause of [F5]:
  SB3 injects 'terminal_observation' (numpy array) and 'episode' (dict with
  keys r, l, t) into the info dict when an episode terminates.  The CSVLogger
  initialised its DictWriter fieldnames from the *first* step's info dict,
  which did not contain these keys.  On step ~346 (first episode boundary),
  DictWriter raised:
    ValueError: dict contains fields not in fieldnames: 'terminal_observation', 'episode'
  This was caught by the broad `except Exception` and silently fell back to
  random baseline — meaning NO learning occurred.

Fix:
  1. CSVLogger uses `extrasaction='ignore'` so unknown columns are silently
     skipped instead of raising.
  2. _LogCallback filters out non-scalar SB3 keys before logging, since
     'terminal_observation' is a numpy array and not CSV-serialisable anyway.

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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .utils import load_config, select_device
from .env import OranSlicingPricingEnv

logger = logging.getLogger("oran3pt.train")


# [F1] Check SB3 availability at module level with diagnostic info
_SB3_AVAILABLE = False
_SB3_IMPORT_ERROR = None
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError as e:
    _SB3_IMPORT_ERROR = str(e)

# ── Keys injected by SB3 at episode boundaries ──────────────────────
# These are NOT part of the env's info dict and must be excluded from
# CSV logging.  'terminal_observation' is a numpy array (not scalar),
# 'episode' is a dict {'r': float, 'l': int, 't': float}.
# See: stable_baselines3/common/vec_env/base_vec_env.py
_SB3_INJECTED_KEYS = frozenset({
    "terminal_observation",
    "episode",
    "TimeLimit.truncated",
    "_final_observation",
    "_final_info",
})


class CSVLogger:
    """Append step-level info dicts to a CSV file.

    [F5] Uses extrasaction='ignore' so that any unexpected keys
    (e.g. SB3-injected episode metadata) are silently skipped
    rather than raising ValueError.
    """

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
            # [F5] extrasaction='ignore' — skip keys not in original fieldnames
            self._writer = csv.DictWriter(
                self._f, fieldnames=self._fields, extrasaction="ignore"
            )
            self._writer.writeheader()
        self._writer.writerow(record)
        self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None
            self._writer = None


def _filter_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Remove SB3-injected non-scalar keys from the info dict.

    [F5] 'terminal_observation' is a numpy array and 'episode' is a
    nested dict — neither is CSV-serialisable.  We strip them here
    so the CSVLogger only receives flat scalar data.
    """
    return {k: v for k, v in info.items() if k not in _SB3_INJECTED_KEYS}


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

    # [F1] FIX: Explicit check with clear error message
    if not _SB3_AVAILABLE:
        logger.error(
            "═══════════════════════════════════════════════════════════\n"
            "  stable-baselines3 is NOT installed.\n"
            "  Import error: %s\n"
            "  \n"
            "  To fix, run:\n"
            "    pip install 'stable-baselines3[extra]>=2.3.0'\n"
            "  \n"
            "  Falling back to random baseline (NO LEARNING).\n"
            "═══════════════════════════════════════════════════════════",
            _SB3_IMPORT_ERROR
        )
        stats = _run_random_baseline(cfg, total_timesteps, csv_path,
                                     users_csv=users_csv)
        logger.info("Random baseline stats: %s", stats)
        return out / "train_log.csv"

    # SB3 is available — proceed with SAC training
    logger.info("SB3 available — training SAC for %d timesteps", total_timesteps)

    env = OranSlicingPricingEnv(cfg, users_csv=users_csv)
    csvlog = CSVLogger(csv_path)

    class _LogCallback(BaseCallback):
        """Log per-step info to CSV, filtering out SB3-injected keys."""

        def __init__(self, csvl: CSVLogger):
            super().__init__()
            self.csvl = csvl

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if infos:
                # [F5] Filter out non-scalar SB3 keys before CSV logging
                self.csvl.log(_filter_info(infos[0]))
            return True

    try:
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
        logger.info("Model saved → %s.zip", model_path)
        return model_path

    except Exception as e:
        csvlog.close()
        logger.error("SAC training failed: %s", e, exc_info=True)
        logger.info("Falling back to random baseline.")
        stats = _run_random_baseline(cfg, total_timesteps, csv_path,
                                     users_csv=users_csv)
        logger.info("Random baseline stats: %s", stats)
        return out / "train_log.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC (3-part tariff)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--users", default="data/users_init.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(name)s] %(message)s")

    # [F1] Print SB3 availability at startup
    if _SB3_AVAILABLE:
        import stable_baselines3
        logger.info("SB3 version: %s", stable_baselines3.__version__)
    else:
        logger.warning("SB3 NOT available: %s", _SB3_IMPORT_ERROR)

    cfg = load_config(args.config)
    users_csv = args.users if Path(args.users).exists() else None
    train(cfg, users_csv=users_csv, output_dir=args.output)


if __name__ == "__main__":
    main()