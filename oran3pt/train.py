"""
SB3 SAC training for 5G O-RAN 3-Part Tariff environment (§12).

REVISION 10 — Changes from v9:
  [EP1] 1-Cycle continuous episode mode support
        SB3 auto-handles truncated=True via ReplayBuffer
        n_eval_episodes increased (5→20) for shorter episodes
        [Pardo ICML 2018; Wan arXiv 2025]
  Prior revisions (v1–v9):
  [M2] Curriculum: fraction-based phase boundary  [Narvekar 2020; Bengio 2009]
  [M6] Lagrangian QoS callback for pviol_E        [Tessler 2019; Stooke 2020]
  [M7] train_freq/gradient_steps 1→4              [Fedus 2020]
  [R3] Curriculum learning (Phase 1: no churn/join; Phase 2: full dynamics)
       [Narvekar et al., JMLR 2020; Bengio et al., ICML 2009]
  [R8] Higher initial entropy coefficient for broader exploration
       [Zhou et al., ICLR 2022]
  [E7] Linear learning rate schedule  [Loshchilov & Hutter, ICLR 2019]
  [E9] EvalCallback with best-model checkpointing [Henderson 2018]
  [E9] Multi-seed training loop
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
    """[E7] Linear LR decay  [Loshchilov & Hutter, ICLR 2019]."""
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


if _SB3_AVAILABLE:
    class _CurriculumCallback(BaseCallback):
        """[R3][D5] Multi-phase curriculum learning callback.

        Supports 2-phase (legacy) and 3-phase (D5) configurations.
        3-phase:
          Phase 1: no churn/join, no Lagrangian — pricing/capacity isolation
          Phase 2: full dynamics, boosted Lagrangian — QoS-focused
          Phase 3: full dynamics, normal Lagrangian — full optimization

        References:
          [Narvekar et al., JMLR 2020]  — Task-specific curriculum
          [Bengio et al., ICML 2009]    — Difficulty ordering
          [Achiam et al., ICML 2017]    — CPO feasible region
        """

        def __init__(self, phases: List[Dict], total_timesteps: int,
                     verbose: int = 0):
            super().__init__(verbose)
            self._phases = phases
            # Compute cumulative step boundaries
            self._boundaries: List[int] = [0]
            cumulative = 0
            for p in phases:
                cumulative += int(p["fraction"] * total_timesteps)
                self._boundaries.append(cumulative)
            self._current_phase = 0

        def _on_step(self) -> bool:
            while (self._current_phase < len(self._phases) - 1
                   and self.num_timesteps
                   >= self._boundaries[self._current_phase + 1]):
                self._current_phase += 1
                phase_cfg = self._phases[self._current_phase]
                # [F6] Use .unwrapped to bypass SB3 Monitor wrapper
                raw_env = self.training_env.envs[0]
                env = getattr(raw_env, 'unwrapped', raw_env)
                if hasattr(env, 'set_curriculum_phase'):
                    phase_val = 0 if phase_cfg.get("churn_join", True) else 1
                    env.set_curriculum_phase(phase_val)
                if hasattr(env, 'set_lagrangian_boost'):
                    env.set_lagrangian_boost(
                        phase_cfg.get("lagrangian_boost", 1.0))
                logger.info(
                    "[D5] Curriculum: → Phase %d at step %d "
                    "(churn_join=%s, lagrangian_boost=%.1f)",
                    self._current_phase + 1, self.num_timesteps,
                    phase_cfg.get("churn_join", True),
                    phase_cfg.get("lagrangian_boost", 1.0))
            return True

    class _LagrangianPIDCallback(BaseCallback):
        """[M6][D2] PID-Lagrangian for pviol_E constraint.

        Replaces simple dual ascent with PID control for faster,
        more stable convergence to the feasible region.

        References:
          [Stooke et al., ICLR 2020]   — PID Lagrangian for responsive safety
          [Tessler et al., ICML 2019]  — RCPO baseline
          [Boyd & Vandenberghe, 2004]  — Dual convergence theory
          [Paternain et al., CDC 2019] — Safe RL via primal-dual
        """

        def __init__(self, threshold: float = 0.15,
                     Kp: float = 0.05, Ki: float = 0.005,
                     Kd: float = 0.01,
                     update_freq: int = 200,
                     lambda_max: float = 10.0,
                     verbose: int = 0) -> None:
            super().__init__(verbose)
            self._threshold = threshold
            self._Kp = Kp
            self._Ki = Ki
            self._Kd = Kd
            self._update_freq = update_freq
            self._lambda_max = lambda_max
            self.lambda_val: float = 0.0
            self._error_integral: float = 0.0
            self._prev_error: float = 0.0
            self._pviol_buffer: List[float] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if infos and "pviol_E" in infos[0]:
                self._pviol_buffer.append(float(infos[0]["pviol_E"]))

            if len(self._pviol_buffer) >= self._update_freq:
                mean_pviol = float(np.mean(self._pviol_buffer))
                error = mean_pviol - self._threshold

                # PID update
                self._error_integral += error
                error_derivative = error - self._prev_error
                delta = (self._Kp * error
                         + self._Ki * self._error_integral
                         + self._Kd * error_derivative)
                self.lambda_val = max(0.0, min(
                    self._lambda_max, self.lambda_val + delta))
                self._prev_error = error
                self._pviol_buffer.clear()

                # [F6] Use .unwrapped to bypass SB3 Monitor wrapper
                raw_env = self.training_env.envs[0]
                env = getattr(raw_env, 'unwrapped', raw_env)
                if hasattr(env, 'set_lagrangian_lambda'):
                    env.set_lagrangian_lambda(self.lambda_val)
                    if self.verbose > 0:
                        logger.info(
                            "[D2] PID Lagrangian: λ=%.4f  "
                            "mean_pviol_E=%.4f  error=%.4f",
                            self.lambda_val, mean_pviol, error)
            return True


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

    curriculum_cfg = tc.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    # [D5] 3-phase curriculum
    curriculum_phases = curriculum_cfg.get("phases", [
        {"fraction": 0.15, "churn_join": False, "lagrangian_boost": 0.0},
        {"fraction": 0.25, "churn_join": True, "lagrangian_boost": 2.0},
        {"fraction": 0.60, "churn_join": True, "lagrangian_boost": 1.0},
    ])

    ent_coef_init = tc.get("ent_coef_init", None)
    ent_coef = tc.get("ent_coef", "auto")

    csv_path = str(out / f"train_log_seed{seed}.csv")

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

    initial_phase = 1 if curriculum_enabled else 0
    env = OranSlicingPricingEnv(
        cfg, users_csv=users_csv, seed=seed,
        curriculum_phase=initial_phase)
    if curriculum_enabled:
        phase_desc = ", ".join(
            f"P{i+1}({p['fraction']:.0%})" for i, p in enumerate(curriculum_phases))
        logger.info("[D5] Curriculum enabled: %d phases [%s]",
                    len(curriculum_phases), phase_desc)

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

    if lr_schedule_type == "linear":
        learning_rate = _make_lr_schedule(lr_start, lr_end)
        logger.info("  LR schedule: linear %g -> %g", lr_start, lr_end)
    else:
        learning_rate = lr_start
        logger.info("  LR: constant %g", lr_start)

    if ent_coef_init is not None:
        effective_ent_coef = ent_coef_init
        logger.info("  [R8] ent_coef_init: %g (will transition to '%s')",
                    ent_coef_init, ent_coef)
    else:
        effective_ent_coef = ent_coef

    try:
        model = SAC(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            batch_size=tc.get("batch_size", 256),
            buffer_size=tc.get("buffer_size", 200_000),
            gamma=tc.get("gamma", 0.995),
            tau=tc.get("tau", 0.005),
            ent_coef=effective_ent_coef,
            train_freq=tc.get("train_freq", 1),
            gradient_steps=tc.get("gradient_steps", 1),
            device=device,
            seed=seed,
            verbose=0,
        )

        eval_env = OranSlicingPricingEnv(cfg, users_csv=users_csv,
                                         seed=seed + 10000)
        eval_freq = tc.get("eval_freq", 10000)
        n_eval_eps = tc.get("n_eval_episodes", 5)
        best_model_name = f"best_model_seed{seed}"

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
            seed_dir = out / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            eval_cb_kwargs["best_model_save_path"] = str(seed_dir)
            best_model_name = "best_model"
            logger.info("SB3 EvalCallback lacks best_model_save_name; "
                        "saving to %s/best_model.zip", seed_dir)
        eval_cb = EvalCallback(**eval_cb_kwargs)

        callbacks = [_LogCallback(csvlog), eval_cb]

        if curriculum_enabled:
            callbacks.append(_CurriculumCallback(
                curriculum_phases, total_timesteps))

        # [M6][D2] PID Lagrangian QoS constraint callback
        lag_cfg = cfg.get("lagrangian_qos", {})
        if lag_cfg.get("enabled", False):
            lag_cb = _LagrangianPIDCallback(
                threshold=lag_cfg.get("pviol_E_threshold", 0.15),
                Kp=lag_cfg.get("Kp", lag_cfg.get("lr_lambda", 0.05)),
                Ki=lag_cfg.get("Ki", 0.005),
                Kd=lag_cfg.get("Kd", 0.01),
                update_freq=lag_cfg.get("update_freq", 200),
                lambda_max=lag_cfg.get("lambda_max", 10.0),
            )
            callbacks.append(lag_cb)
            logger.info("[D2] PID Lagrangian enabled: "
                        "Kp=%.4f  Ki=%.4f  Kd=%.4f  λ_max=%.1f  freq=%d",
                        lag_cb._Kp, lag_cb._Ki, lag_cb._Kd,
                        lag_cb._lambda_max, lag_cb._update_freq)

        model.learn(total_timesteps=total_timesteps,
                    callback=callbacks,
                    progress_bar=True)

        final_path = out / f"final_model_seed{seed}"
        model.save(str(final_path))
        csvlog.close()
        logger.info("Seed %d done.  Best -> %s.zip  Final -> %s.zip",
                     seed, best_model_name, final_path)

        best_zip = out / f"{best_model_name}.zip"
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

    canonical = out / "best_model.zip"
    if best_paths and best_paths[0].exists():
        if best_paths[0].resolve() != canonical.resolve():
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
