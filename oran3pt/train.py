"""
SB3 SAC training for 5G O-RAN 3-Part Tariff environment (§12).

REVISION 5 — Enhancements:
  [E7] Linear learning rate schedule  [Loshchilov & Hutter, ICLR 2019]
  [E9] EvalCallback with best-model checkpointing [Henderson 2018]
  [E9] Multi-seed training loop
  Prior revisions (v1–v4):
  [F1] Explicit SB3 import diagnostic (not silent fallback)
  [F5] CSVLogger crash on SB3 episode-boundary keys

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


# [F1] Check SB3 availability at module level
_SB3_AVAILABLE = False
_SB3_IMPORT_ERROR = None
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    _SB3_AVAILABLE = True
except ImportError as e:
    _SB3_IMPORT_ERROR = str(e)

# ── Keys injected by SB3 at episode boundaries ──────────────────────
_SB3_INJECTED_KEYS = frozenset({
    "terminal_observation",
    "episode",
    "TimeLimit.truncated",
    "_final_observation",
    "_final_info",
})


class CSVLogger:
    """Append step-level info dicts to a CSV file.

    [F5] Uses extrasaction='ignore' so SB3-injected keys are skipped.
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
    """Remove SB3-injected non-scalar keys from the info dict."""
    return {k: v for k, v in info.items() if k not in _SB3_INJECTED_KEYS}


def _make_lr_schedule(lr_start: float, lr_end: float):
    """[E7] Linear LR decay  [Loshchilov & Hutter, ICLR 2019].

    Returns a callable that SB3 accepts for learning_rate parameter.
    progress_remaining goes from 1.0 (start) → 0.0 (end of training).
    """
    def _schedule(progress_remaining: float) -> float:
        return lr_end + (lr_start - lr_end) * progress_remaining
    return _schedule


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


def train_single_seed(cfg: Dict[str, Any],
                      users_csv: Optional[str] = None,
                      output_dir: str = "outputs",
                      seed: int = 0) -> Path:
    """Train a single SAC seed; return path to best model."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = select_device()
    logger.info("Training seed=%d  device=%s", seed, device)

    tc = cfg.get("training", {})
    total_timesteps = tc.get("total_timesteps", 1_000_000)
    lr_start = tc.get("learning_rate", 3e-4)
    lr_end = tc.get("learning_rate_end", 1e-5)
    lr_schedule_type = tc.get("lr_schedule", "linear")

    csv_path = str(out / f"train_log_seed{seed}.csv")

    # [F1] Explicit check
    if not _SB3_AVAILABLE:
        logger.error(
            "═══════════════════════════════════════════════════════\n"
            "  stable-baselines3 is NOT installed.\n"
            "  Import error: %s\n"
            "  To fix:  pip install 'stable-baselines3[extra]>=2.3.0'\n"
            "  Falling back to random baseline (NO LEARNING).\n"
            "═══════════════════════════════════════════════════════",
            _SB3_IMPORT_ERROR
        )
        stats = _run_random_baseline(cfg, total_timesteps, csv_path,
                                     users_csv=users_csv, seed=seed)
        logger.info("Random baseline stats: %s", stats)
        return out / f"train_log_seed{seed}.csv"

    logger.info("SB3 available — training SAC for %d timesteps (seed %d)",
                total_timesteps, seed)

    env = OranSlicingPricingEnv(cfg, users_csv=users_csv, seed=seed)
    csvlog = CSVLogger(csv_path)

    class _LogCallback(BaseCallback):
        def __init__(self, csvl: CSVLogger):
            super().__init__()
            self.csvl = csvl

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if infos:
                self.csvl.log(_filter_info(infos[0]))
            return True

    # [E7] Learning rate schedule
    if lr_schedule_type == "linear":
        learning_rate = _make_lr_schedule(lr_start, lr_end)
        logger.info("  LR schedule: linear %g -> %g", lr_start, lr_end)
    else:
        learning_rate = lr_start
        logger.info("  LR: constant %g", lr_start)

    try:
        model = SAC(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            batch_size=tc.get("batch_size", 256),
            buffer_size=tc.get("buffer_size", 200_000),
            gamma=tc.get("gamma", 0.995),
            tau=tc.get("tau", 0.005),
            ent_coef=tc.get("ent_coef", "auto"),
            train_freq=tc.get("train_freq", 1),
            gradient_steps=tc.get("gradient_steps", 1),
            device=device,
            seed=seed,
            verbose=0,
        )

        # [E9] EvalCallback — checkpoint best model during training
        eval_env = OranSlicingPricingEnv(cfg, users_csv=users_csv,
                                         seed=seed + 10000)
        eval_freq = tc.get("eval_freq", 10000)
        n_eval_eps = tc.get("n_eval_episodes", 5)
        best_model_name = f"best_model_seed{seed}"

        # best_model_save_name added in SB3 2.4.0; fall back if absent
        eval_cb_kwargs = dict(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_eps,
            deterministic=True,
            verbose=0,
        )
        import inspect
        if "best_model_save_name" in inspect.signature(EvalCallback.__init__).parameters:
            eval_cb_kwargs["best_model_save_path"] = str(out)
            eval_cb_kwargs["best_model_save_name"] = best_model_name
        else:
            # Fallback: save into seed-specific subdirectory to avoid overwriting
            seed_dir = out / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            eval_cb_kwargs["best_model_save_path"] = str(seed_dir)
            best_model_name = "best_model"  # default name within subdirectory
            logger.info("SB3 EvalCallback lacks best_model_save_name; "
                        "saving to %s/best_model.zip", seed_dir)
        eval_cb = EvalCallback(**eval_cb_kwargs)

        log_cb = _LogCallback(csvlog)
        model.learn(total_timesteps=total_timesteps,
                    callback=[log_cb, eval_cb],
                    progress_bar=True)

        # Also save final model
        final_path = out / f"final_model_seed{seed}"
        model.save(str(final_path))
        csvlog.close()
        logger.info("Seed %d done.  Best -> %s.zip  Final -> %s.zip",
                     seed, best_model_name, final_path)

        best_zip = out / f"{best_model_name}.zip"
        # Fallback: check seed-specific subdirectory if not found in out/
        if not best_zip.exists():
            best_zip = out / f"seed{seed}" / "best_model.zip"
        if best_zip.exists():
            return best_zip
        return Path(str(final_path) + ".zip")

    except Exception as e:
        csvlog.close()
        logger.error("SAC training failed (seed %d): %s", seed, e,
                     exc_info=True)
        logger.info("Falling back to random baseline.")
        stats = _run_random_baseline(cfg, total_timesteps, csv_path,
                                     users_csv=users_csv, seed=seed)
        logger.info("Random baseline stats: %s", stats)
        return out / f"train_log_seed{seed}.csv"


def train(cfg: Dict[str, Any],
          users_csv: Optional[str] = None,
          output_dir: str = "outputs") -> Path:
    """[E9] Multi-seed training loop.  Returns path to overall best model."""
    tc = cfg.get("training", {})
    n_seeds = tc.get("n_seeds", 1)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_paths: List[Path] = []
    for seed_idx in range(n_seeds):
        logger.info("======== Training seed %d / %d ========",
                     seed_idx + 1, n_seeds)
        p = train_single_seed(cfg, users_csv=users_csv,
                              output_dir=output_dir, seed=seed_idx)
        best_paths.append(p)

    # Copy seed-0 best model as the canonical "best_model.zip" for eval
    canonical = out / "best_model.zip"
    if best_paths and best_paths[0].exists():
        import shutil
        shutil.copy2(best_paths[0], canonical)
        logger.info("Canonical best model -> %s", canonical)

    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC (3-part tariff)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--users", default="data/users_init.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override n_seeds from config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(name)s] %(message)s")

    if _SB3_AVAILABLE:
        import stable_baselines3
        logger.info("SB3 version: %s", stable_baselines3.__version__)
    else:
        logger.warning("SB3 NOT available: %s", _SB3_IMPORT_ERROR)

    cfg = load_config(args.config)
    if args.seeds is not None:
        cfg.setdefault("training", {})["n_seeds"] = args.seeds
    users_csv = args.users if Path(args.users).exists() else None
    train(cfg, users_csv=users_csv, output_dir=args.output)


if __name__ == "__main__":
    main()
