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
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    [HI-4] Supports context manager protocol (__enter__/__exit__)
    to ensure file handle is closed even on exceptions.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None
        self._writer = None
        self._fields: Optional[List[str]] = None

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

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
            # [FIX-S3] Boost decay state for gradual phase transitions
            self._prev_boost: float = phases[0].get("lagrangian_boost", 0.0)
            self._decay_start: int = 0
            self._decay_steps: int = 0
            self._decay_target: float = 0.0

        def _on_step(self) -> bool:
            while (self._current_phase < len(self._phases) - 1
                   and self.num_timesteps
                   >= self._boundaries[self._current_phase + 1]):
                prev_boost = self._phases[self._current_phase].get(
                    "lagrangian_boost", 1.0)
                self._current_phase += 1
                phase_cfg = self._phases[self._current_phase]
                # [F6] Use .unwrapped to bypass SB3 Monitor wrapper
                raw_env = self.training_env.envs[0]
                env = getattr(raw_env, 'unwrapped', raw_env)
                if hasattr(env, 'set_curriculum_phase'):
                    phase_val = 0 if phase_cfg.get("churn_join", True) else 1
                    env.set_curriculum_phase(phase_val)
                # [FIX-S3] Check for boost_decay_steps
                target_boost = phase_cfg.get("lagrangian_boost", 1.0)
                decay_steps = phase_cfg.get("boost_decay_steps", 0)
                if decay_steps > 0 and prev_boost != target_boost:
                    self._prev_boost = prev_boost
                    self._decay_start = self.num_timesteps
                    self._decay_steps = decay_steps
                    self._decay_target = target_boost
                    # Start at prev_boost; linear interpolation below
                    if hasattr(env, 'set_lagrangian_boost'):
                        env.set_lagrangian_boost(prev_boost)
                else:
                    self._decay_steps = 0   # no decay for this phase
                    if hasattr(env, 'set_lagrangian_boost'):
                        env.set_lagrangian_boost(target_boost)
                logger.info(
                    "[D5] Curriculum: → Phase %d at step %d "
                    "(churn_join=%s, lagrangian_boost=%.1f%s)",
                    self._current_phase + 1, self.num_timesteps,
                    phase_cfg.get("churn_join", True),
                    target_boost,
                    f", decay from {prev_boost:.1f} over {decay_steps} steps"
                    if decay_steps > 0 else "")

            # [FIX-S3] Gradual boost decay interpolation
            if self._decay_steps > 0:
                elapsed = self.num_timesteps - self._decay_start
                if elapsed < self._decay_steps:
                    alpha = elapsed / self._decay_steps
                    current_boost = (self._prev_boost
                                     + alpha * (self._decay_target
                                                - self._prev_boost))
                    raw_env = self.training_env.envs[0]
                    env = getattr(raw_env, 'unwrapped', raw_env)
                    if hasattr(env, 'set_lagrangian_boost'):
                        env.set_lagrangian_boost(current_boost)
                else:
                    # Decay complete
                    raw_env = self.training_env.envs[0]
                    env = getattr(raw_env, 'unwrapped', raw_env)
                    if hasattr(env, 'set_lagrangian_boost'):
                        env.set_lagrangian_boost(self._decay_target)
                    self._decay_steps = 0

            return True

    class _LagrangianPIDCallback(BaseCallback):
        """[M6][D2] PID-Lagrangian for pviol_E constraint.

        Replaces simple dual ascent with PID control for faster,
        more stable convergence to the feasible region.

        [CR-2] Anti-windup: integral term clamped asymmetrically.
        Positive bound = lambda_max / Ki (prevents overshoot).
        [FIX-C2] Negative bound = -lambda_max * 0.05 (prevents
        integral from sinking deep into negative territory in the
        feasible region, which causes sluggish lambda recovery).

        [FIX-C1] eval_env parameter propagates lambda to EvalCallback's
        evaluation environment for constraint-aware model selection.

        [I-1b] lambda_min > 0 ensures policy always has a minimum
        QoS incentive, preventing pviol_E drift in eval.

        References:
          [Stooke et al., ICLR 2020 §3.2] — PID Lagrangian + integral windup
          [Tessler et al., ICML 2019]  — RCPO baseline
          [Boyd & Vandenberghe, 2004]  — Dual convergence theory
          [Mao et al., arXiv 2025]     — PID not plug-and-play; windup key issue
          [Paternain et al., CDC 2019; TAC 2022] — Positive dual lower bound
        """

        def __init__(self, threshold: float = 0.15,
                     Kp: float = 0.05, Ki: float = 0.005,
                     Kd: float = 0.01,
                     update_freq: int = 200,
                     lambda_max: float = 10.0,
                     lambda_min: float = 0.0,
                     eval_env=None,
                     verbose: int = 0) -> None:
            super().__init__(verbose)
            self._threshold = threshold
            self._Kp = Kp
            self._Ki = Ki
            self._Kd = Kd
            self._update_freq = update_freq
            self._lambda_max = lambda_max
            self._lambda_min = lambda_min  # [I-1b] [Paternain CDC 2019]
            self.lambda_val: float = lambda_min
            self._error_integral: float = 0.0
            self._prev_error: float = 0.0
            self._pviol_buffer: List[float] = []
            self._eval_env = eval_env  # [FIX-C1] propagate λ to eval_env
            # [CR-2] Anti-windup: positive integral bound = lambda_max / Ki
            # [FIX-C2] Asymmetric: negative bound = -lambda_max * 0.05
            # Recovery from integral_min at typical error +0.1 takes ~5
            # PID updates (was ~20 with * 0.2; ~thousands with symmetric)
            # [Stooke ICLR 2020 §3.2; Mao arXiv 2025]
            self._integral_max: float = lambda_max / max(Ki, 1e-8)
            self._integral_min: float = -lambda_max * 0.05

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if infos and "pviol_E" in infos[0]:
                self._pviol_buffer.append(float(infos[0]["pviol_E"]))

            if len(self._pviol_buffer) >= self._update_freq:
                mean_pviol = float(np.mean(self._pviol_buffer))
                error = mean_pviol - self._threshold

                # [CR-2][I-1a] PID update with asymmetric integral clamping
                # Positive windup: ±integral_max (standard)
                # Negative windup: integral_min = -10% of integral_max
                # [Stooke ICLR 2020 §3.2; Mao arXiv 2025]
                self._error_integral = max(
                    self._integral_min,
                    min(self._integral_max,
                        self._error_integral + error))
                error_derivative = error - self._prev_error
                delta = (self._Kp * error
                         + self._Ki * self._error_integral
                         + self._Kd * error_derivative)
                # [I-1b] lambda_min ensures minimum QoS incentive
                # [Paternain CDC 2019; TAC 2022]
                self.lambda_val = max(self._lambda_min, min(
                    self._lambda_max, self.lambda_val + delta))
                self._prev_error = error
                self._pviol_buffer.clear()

                # [F6] Use .unwrapped to bypass SB3 Monitor wrapper
                raw_env = self.training_env.envs[0]
                env = getattr(raw_env, 'unwrapped', raw_env)
                if hasattr(env, 'set_lagrangian_lambda'):
                    env.set_lagrangian_lambda(self.lambda_val)
                    # [FIX-C1] Propagate λ to eval_env for constraint-aware
                    # model selection [Tessler ICML 2019 §3; Stooke 2020 §3.2]
                    if self._eval_env is not None:
                        eval_base = getattr(
                            self._eval_env, 'unwrapped', self._eval_env)
                        if hasattr(eval_base, 'set_lagrangian_lambda'):
                            eval_base.set_lagrangian_lambda(self.lambda_val)
                    if self.verbose > 0:
                        logger.info(
                            "[D2] PID Lagrangian: λ=%.4f  "
                            "mean_pviol_E=%.4f  error=%.4f",
                            self.lambda_val, mean_pviol, error)
            return True

    class _EarlyStoppingCallback(BaseCallback):
        """[OPT-C] Eval reward plateau detection → early stopping.

        Monitors EvalCallback.best_mean_reward. If no improvement exceeds
        min_improvement for patience consecutive evaluations after
        min_timesteps, stops training.

        References:
          [Prechelt 2002] Early stopping methodology
          [Henderson AAAI 2018] Training budget allocation
        """

        def __init__(self, eval_callback: EvalCallback,
                     patience: int = 10,
                     min_timesteps: int = 500000,
                     min_improvement: float = 0.01,
                     verbose: int = 0):
            super().__init__(verbose)
            self._eval_cb = eval_callback
            self._patience = patience
            self._min_ts = min_timesteps
            self._min_imp = min_improvement
            self._best_reward = -np.inf
            self._stale_count = 0
            self._last_eval_calls = 0

        def _on_step(self) -> bool:
            if self.num_timesteps < self._min_ts:
                return True
            # Only check when a new evaluation has occurred
            if self._eval_cb.n_calls == self._last_eval_calls:
                return True
            self._last_eval_calls = self._eval_cb.n_calls

            current = self._eval_cb.best_mean_reward
            if self._best_reward < 0:
                threshold = self._best_reward * (1.0 - self._min_imp)
            else:
                threshold = self._best_reward * (1.0 + self._min_imp)

            if current > threshold:
                self._best_reward = current
                self._stale_count = 0
            else:
                self._stale_count += 1

            if self._stale_count >= self._patience:
                logger.info("[OPT-C] Early stopping at step %d "
                           "(no improvement for %d evals, best=%.4f)",
                           self.num_timesteps, self._patience,
                           self._best_reward)
                return False  # stops model.learn()
            return True


def train_single_seed(cfg: Dict[str, Any],
                      users_csv: Optional[str] = None,
                      output_dir: str = "outputs",
                      seed: int = 0) -> Tuple[Path, float]:
    """[OPT-H] Train a single SAC seed; return (path, best_eval_reward)."""
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
        return out / f"train_log_seed{seed}.csv", stats["mean_reward"]

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

    # [OPT-B] SB3 "auto_X" format: auto-tuning enabled + init value X
    # float 전달 시 auto-tuning이 영구 비활성화됨 (SB3 sac.py:180-195)
    # [Haarnoja ICML 2018 §5.1; Wang & Ni, ICML AutoML 2020]
    if ent_coef_init is not None and ent_coef == "auto":
        effective_ent_coef = f"auto_{ent_coef_init}"
        logger.info("  [OPT-B] ent_coef: '%s' (auto-tune, init=%.2f, "
                    "target_entropy=-5)", effective_ent_coef, ent_coef_init)
    elif ent_coef_init is not None:
        effective_ent_coef = ent_coef_init
        logger.info("  ent_coef: %g (fixed, no auto-tune)", ent_coef_init)
    else:
        effective_ent_coef = ent_coef

    try:
        # [I-6a] target_entropy controls final exploration level
        # Default -dim(A)=-5 is conservative; -3.0 for smoother convergence
        # [Haarnoja ICML 2018 §5; Zhou ICLR 2022]
        sac_kwargs: Dict[str, Any] = dict(
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
        target_entropy = tc.get("target_entropy", None)
        if target_entropy is not None:
            sac_kwargs["target_entropy"] = target_entropy
            logger.info("  [I-6a] target_entropy: %.1f", target_entropy)
        model = SAC("MlpPolicy", env, **sac_kwargs)

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

        # [OPT-C] Early stopping callback
        es_cfg = tc.get("early_stopping", {})
        if es_cfg.get("enabled", False):
            es_cb = _EarlyStoppingCallback(
                eval_callback=eval_cb,
                patience=es_cfg.get("patience", 10),
                min_timesteps=es_cfg.get("min_timesteps", 500000),
                min_improvement=es_cfg.get("min_improvement", 0.01),
            )
            callbacks.append(es_cb)
            logger.info("[OPT-C] Early stopping: patience=%d, "
                       "min_timesteps=%d, min_improvement=%.3f",
                       es_cb._patience, es_cb._min_ts, es_cb._min_imp)

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
                lambda_min=lag_cfg.get("lambda_min", 0.0),
                eval_env=eval_env,  # [FIX-C1] constraint-aware model selection
            )
            callbacks.append(lag_cb)
            logger.info("[D2] PID Lagrangian enabled: "
                        "Kp=%.4f  Ki=%.4f  Kd=%.4f  λ_max=%.1f  freq=%d",
                        lag_cb._Kp, lag_cb._Ki, lag_cb._Kd,
                        lag_cb._lambda_max, lag_cb._update_freq)

        model.learn(total_timesteps=total_timesteps,
                    callback=callbacks,
                    progress_bar=True)

        # [OPT-H] Retrieve best eval reward for cross-seed comparison
        best_reward = float(eval_cb.best_mean_reward)

        final_path = out / f"final_model_seed{seed}"
        model.save(str(final_path))

        # [V11-4] Save Lagrangian PID state for eval restoration
        # [Stooke ICLR 2020 §3.2; Tessler ICML 2019]
        lagrangian_state_path = out / f"lagrangian_state_seed{seed}.json"
        lag_state = {"lambda": 0.0, "integral": 0.0, "prev_error": 0.0,
                     "threshold": 0.15}
        for cb in callbacks:
            if isinstance(cb, _LagrangianPIDCallback):
                lag_state = {
                    "lambda": cb.lambda_val,
                    "integral": cb._error_integral,
                    "prev_error": cb._prev_error,
                    "threshold": cb._threshold,
                }
                break
        import json
        with open(lagrangian_state_path, "w") as f:
            json.dump(lag_state, f, indent=2)
        logger.info("[V11-4] Lagrangian state saved: λ=%.4f → %s",
                    lag_state["lambda"], lagrangian_state_path)

        csvlog.close()
        logger.info("Seed %d done.  Best -> %s.zip  Final -> %s.zip  "
                     "best_reward=%.4f", seed, best_model_name, final_path,
                     best_reward)

        best_zip = out / f"{best_model_name}.zip"
        if not best_zip.exists():
            best_zip = out / f"seed{seed}" / "best_model.zip"
        if best_zip.exists():
            return best_zip, best_reward
        return Path(str(final_path) + ".zip"), best_reward

    except Exception as e:
        csvlog.close()
        logger.error("SAC training failed (seed %d): %s", seed, e,
                     exc_info=True)
        logger.info("Falling back to random baseline.")
        stats = _run_random_baseline(cfg, total_timesteps, csv_path,
                                     users_csv=users_csv, seed=seed)
        logger.info("Random baseline stats: %s", stats)
        return out / f"train_log_seed{seed}.csv", stats["mean_reward"]


def _seed_worker(seed: int, cfg: Dict[str, Any],
                 users_csv: Optional[str],
                 output_dir: str,
                 result_queue: mp.Queue) -> None:
    """[OPT-A] Parallel seed training worker.

    Each seed is fully independent: separate env, model, buffer, CSV.
    macOS spawn mode ensures MPS safety.
    [Henderson AAAI 2018] multi-seed reproducibility.
    """
    try:
        path, reward = train_single_seed(cfg, users_csv, output_dir, seed)
        result_queue.put((seed, str(path), reward))
    except Exception as e:
        logger.error("Seed %d failed: %s", seed, e)
        result_queue.put((seed, None, float('-inf')))


def _train_sequential(cfg: Dict[str, Any],
                      users_csv: Optional[str],
                      output_dir: str,
                      n_seeds: int) -> Tuple[Path, List[Tuple[int, str, float]]]:
    """Sequential multi-seed training (fallback or n_seeds=1)."""
    results: List[Tuple[int, str, float]] = []
    for seed_idx in range(n_seeds):
        logger.info("======== Training seed %d / %d ========",
                     seed_idx + 1, n_seeds)
        p, reward = train_single_seed(cfg, users_csv=users_csv,
                                      output_dir=output_dir, seed=seed_idx)
        results.append((seed_idx, str(p), reward))
    return Path(output_dir), results


def _train_parallel(cfg: Dict[str, Any],
                    users_csv: Optional[str],
                    output_dir: str,
                    n_seeds: int) -> Tuple[Path, List[Tuple[int, str, float]]]:
    """[OPT-A] Parallel multi-seed training via multiprocessing.

    Each seed runs in a separate process. macOS spawn mode ensures
    MPS safety. Memory: ~150MB/process × n_seeds.
    [Henderson AAAI 2018] multi-seed reproducibility.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_parallel = cfg.get("training", {}).get("max_parallel", n_seeds)
    result_queue = mp.Queue()

    processes = []
    for seed_idx in range(min(n_seeds, max_parallel)):
        p = mp.Process(
            target=_seed_worker,
            args=(seed_idx, cfg, users_csv, output_dir, result_queue))
        p.start()
        processes.append(p)

    results: List[Tuple[int, str, float]] = []
    for _ in range(n_seeds):
        results.append(result_queue.get())
    for p in processes:
        p.join()

    return out, results


def train(cfg: Dict[str, Any],
          users_csv: Optional[str] = None,
          output_dir: str = "outputs") -> Path:
    """[E9][OPT-A][OPT-H] Multi-seed training with parallel support.

    Returns path to canonical best_model.zip (selected by highest eval reward).
    """
    tc = cfg.get("training", {})
    n_seeds = tc.get("n_seeds", 1)
    parallel = tc.get("parallel_seeds", False)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if n_seeds > 1 and parallel and _SB3_AVAILABLE:
        _, results = _train_parallel(cfg, users_csv, output_dir, n_seeds)
    else:
        _, results = _train_sequential(cfg, users_csv, output_dir, n_seeds)

    # [OPT-H] Select best model across all seeds by eval reward
    valid_results = [(s, p, r) for s, p, r in results if p is not None]
    if not valid_results:
        logger.warning("No valid seed results. No best_model.zip created.")
        return out / "best_model.zip"

    best_seed, best_path_str, best_reward = max(
        valid_results, key=lambda x: x[2])
    canonical = out / "best_model.zip"
    best_path = Path(best_path_str)

    if best_path.exists() and best_path.resolve() != canonical.resolve():
        shutil.copy2(best_path, canonical)

    # [V11-4] Also copy Lagrangian state for best seed
    best_lag = out / f"lagrangian_state_seed{best_seed}.json"
    canonical_lag = out / "lagrangian_state.json"
    if best_lag.exists() and best_lag.resolve() != canonical_lag.resolve():
        shutil.copy2(best_lag, canonical_lag)
        logger.info("[V11-4] Lagrangian state → %s", canonical_lag)

    logger.info("[OPT-H] Best model: seed %d (reward=%.4f) -> %s",
               best_seed, best_reward, canonical)
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC (3-part tariff)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--override", default=None,
                        help="[CR-3] Override config (e.g. config/production.yaml)")
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

    cfg = load_config(args.config, override_path=args.override)
    if args.seeds is not None:
        cfg.setdefault("training", {})["n_seeds"] = args.seeds
    users_csv = args.users if Path(args.users).exists() else None
    train(cfg, users_csv=users_csv, output_dir=args.output)


if __name__ == "__main__":
    main()
