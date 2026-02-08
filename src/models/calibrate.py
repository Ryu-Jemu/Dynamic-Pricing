"""
Calibration module (Section 19).

Avoids arbitrary hardcoded numbers by fitting parameters to scenario
targets via quantile matching and optimization.

Three calibration routines:
  1) calibrate_demand()        — fit lognormal mu, sigma  [LOGNORMAL_TNET]
  2) calibrate_market()        — fit churn offset U_outside [CHURN_SLR][DISCONF_PDF]
  3) calibrate_reward_scale()  — random rollouts → profit_scale [SB3_TIPS]

Usage:
  python -m src.models.calibrate --config config/default.yaml

References:
  [LOGNORMAL_TNET] https://dl.acm.org/doi/10.1109/TNET.2021.3059542
  [CHURN_SLR]      https://link.springer.com/article/10.1007/s11301-023-00335-7
  [DISCONF_PDF]    https://accesson.kr/ijcon/assets/pdf/55438/journal-21-1-11.pdf
  [SB3_TIPS]       https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
"""

"""
Calibration module for reward scale and demand parameters.

Section 19 — Calibration:
  19.1  Demand calibration: fit lognormal (mu, sigma) to target quantiles
  19.2  Market calibration: adjust logistic coefficients to target churn
  19.3  Reward scale calibration: set profit_scale from random rollouts

Step 2 integration — reward scale calibration:
  The calibration of profit_scale must account for the selected reward_type.

  ┌─────────┬──────────────────────────────────────────────────────────┐
  │ Type    │ Calibration strategy                                     │
  ├─────────┼──────────────────────────────────────────────────────────┤
  │ tanh    │ scale = p95(|profit|) from random rollouts  [ORIGINAL]  │
  │         │ Ensures tanh(P/S) ≈ 0.85 at p95 → room for growth      │
  │         │ Problem: saturates above 3×scale → poor discrimination  │
  ├─────────┼──────────────────────────────────────────────────────────┤
  │ linear  │ scale = p95(|profit|) / target_max_reward               │
  │         │ Ensures clip boundary covers the operating range         │
  │         │ target_max_reward = reward_clip * 0.8 (80% headroom)    │
  ├─────────┼──────────────────────────────────────────────────────────┤
  │ log     │ scale = median(|profit|)  [PERCENTILE-BASED]            │
  │         │ log(1 + |P|/S) with S = median gives r ≈ 0.69 at       │
  │         │ median profit, ~2.4 at p95 — well within clip range.    │
  │         │ Key: log never saturates, so scale choice affects        │
  │         │ compression rate, not saturation point.                  │
  └─────────┴──────────────────────────────────────────────────────────┘

  Rationale for percentile-based calibration for log:
    Unlike tanh where scale determines the saturation boundary,
    for log reward the scale only controls the compression rate.
    Using median (p50) as scale ensures:
      1. Typical profits map to r ≈ 0.69 (moderate reward)
      2. High profits (p95) map to r ≈ 2.0-3.0 (strong signal)
      3. No saturation at any profit level (log property)
    This is more robust than p95-based scaling because log's
    unbounded nature makes it insensitive to outliers.

  Academic basis:
    - Schaul et al. (DeepMind, 2021): return-based scaling uses
      running statistics (mean/std) of returns.
    - PARS (ICLR 2025): reward scaling + layer norm for stability.
    - AN-SAC (Gao et al., JNFA 2025): adaptive normalization.

References:
  [SB3_TIPS]  https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("oran.calibrate")


def calibrate_reward_scale(
    profit_samples: np.ndarray,
    reward_type: str = "log",
    reward_clip: float = 2.0,
    min_scale: float = 1.0,
) -> Dict[str, Any]:
    """Calibrate profit_scale from random rollout profit samples.

    Parameters
    ----------
    profit_samples : ndarray
        Array of monthly profit values from random rollouts.
        Should be collected over multiple episodes.
    reward_type : str
        "tanh", "linear", or "log"
    reward_clip : float
        Clipping bound (used by linear mode to compute scale)
    min_scale : float
        Floor value for profit_scale to prevent division by zero

    Returns
    -------
    dict with keys:
        profit_scale : float    Calibrated scale value
        method : str            Description of calibration method
        stats : dict            Diagnostic statistics
    """
    if len(profit_samples) == 0:
        logger.warning("Empty profit_samples; returning min_scale=%.1f", min_scale)
        return {
            "profit_scale": min_scale,
            "method": "empty_fallback",
            "stats": {},
        }

    abs_profits = np.abs(profit_samples)
    abs_profits = abs_profits[np.isfinite(abs_profits)]

    if len(abs_profits) == 0:
        logger.warning("All profit samples non-finite; returning min_scale")
        return {
            "profit_scale": min_scale,
            "method": "nonfinite_fallback",
            "stats": {},
        }

    # Compute statistics
    p50 = float(np.percentile(abs_profits, 50))
    p75 = float(np.percentile(abs_profits, 75))
    p95 = float(np.percentile(abs_profits, 95))
    p99 = float(np.percentile(abs_profits, 99))
    mean_abs = float(np.mean(abs_profits))
    std_abs = float(np.std(abs_profits))

    stats = {
        "n_samples": len(profit_samples),
        "n_valid": len(abs_profits),
        "mean_abs_profit": mean_abs,
        "std_abs_profit": std_abs,
        "p50_abs_profit": p50,
        "p75_abs_profit": p75,
        "p95_abs_profit": p95,
        "p99_abs_profit": p99,
    }

    if reward_type == "tanh":
        # ── Original: p95-based ──────────────────────────────────
        # scale = p95(|P|) so that tanh(P/S) ≈ 0.85 at p95.
        # This was the original calibration strategy.
        scale = max(p95, min_scale)
        method = "tanh: scale = p95(|profit|)"

        # Diagnostic: check saturation
        r_at_p95 = float(np.tanh(p95 / scale))
        r_at_p99 = float(np.tanh(p99 / scale))
        stats["r_at_p95"] = r_at_p95
        stats["r_at_p99"] = r_at_p99
        if r_at_p99 > 0.999:
            logger.warning(
                "tanh calibration: r(p99)=%.4f → saturation likely. "
                "Consider switching to reward_type='log'.",
                r_at_p99,
            )

    elif reward_type == "linear":
        # ── Linear: ensure clip covers operating range ───────────
        # scale = p95(|P|) / (reward_clip × 0.8)
        # The 0.8 factor provides 20% headroom before clipping.
        target_r = reward_clip * 0.8
        scale = max(p95 / max(target_r, 0.1), min_scale)
        method = f"linear: scale = p95 / (clip×0.8) = {p95:.0f} / {target_r:.1f}"

        # Diagnostic: check clip boundary
        clip_profit = scale * reward_clip
        frac_clipped = float(np.mean(abs_profits > clip_profit))
        stats["clip_profit_boundary"] = clip_profit
        stats["frac_clipped"] = frac_clipped
        if frac_clipped > 0.10:
            logger.warning(
                "linear calibration: %.1f%% of profits exceed clip boundary "
                "(%.0f KRW). Consider increasing reward_clip.",
                frac_clipped * 100, clip_profit,
            )

    elif reward_type == "log":
        # ── Step 2: Percentile-based for log ─────────────────────
        # scale = median(|P|)
        # log(1 + |P|/S) with S = median gives:
        #   - At median profit: r = log(2) ≈ 0.693
        #   - At p95 profit: r = log(1 + p95/p50), typically ~2-3
        # Key advantage: log never saturates, so scale choice only
        # affects compression rate, not saturation boundary.
        scale = max(p50, min_scale)
        method = "log: scale = median(|profit|)"

        # Diagnostic: expected reward range
        r_at_p50 = float(np.log1p(p50 / scale))
        r_at_p95 = float(np.log1p(p95 / scale))
        r_at_p99 = float(np.log1p(p99 / scale))
        stats["r_at_p50"] = r_at_p50
        stats["r_at_p95"] = r_at_p95
        stats["r_at_p99"] = r_at_p99

        # Check if reward exceeds clip
        if r_at_p99 > reward_clip:
            logger.info(
                "log calibration: r(p99)=%.2f exceeds reward_clip=%.1f. "
                "%.1f%% of rewards will be clipped (acceptable for log).",
                r_at_p99, reward_clip,
                float(np.mean(np.log1p(abs_profits / scale) > reward_clip)) * 100,
            )

    else:
        raise ValueError(f"Unknown reward_type: '{reward_type}'")

    profit_scale = float(scale)

    logger.info(
        "Calibrated profit_scale=%.0f (reward_type=%s, method=%s)",
        profit_scale, reward_type, method,
    )

    return {
        "profit_scale": profit_scale,
        "method": method,
        "stats": stats,
    }


def run_random_rollouts(
    env_cls: type,
    cfg: Dict[str, Any],
    n_episodes: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Collect profit samples from random policy rollouts.

    Parameters
    ----------
    env_cls : type
        Environment class (OranSlicingEnv)
    cfg : dict
        Full configuration dictionary
    n_episodes : int
        Number of episodes to run
    seed : int
        Random seed for reproducibility

    Returns
    -------
    ndarray : all monthly profit values across episodes
    """
    all_profits = []
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        env = env_cls(cfg, seed=ep_seed)
        obs, info = env.reset(seed=ep_seed)

        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if "profit" in info:
                all_profits.append(info["profit"])
            done = terminated or truncated

    return np.array(all_profits, dtype=np.float64)


def calibrate_from_config(
    env_cls: type,
    cfg: Dict[str, Any],
    n_episodes: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full calibration pipeline: rollouts → profit_scale.

    Parameters
    ----------
    env_cls : type
        Environment class
    cfg : dict
        Configuration (economics.reward_type read from here)
    n_episodes : int or None
        Override calibration.reward_scale_episodes from config
    seed : int

    Returns
    -------
    dict with calibrated profit_scale, method, stats
    """
    cal_cfg = cfg.get("calibration", {})
    if n_episodes is None:
        n_episodes = cal_cfg.get("reward_scale_episodes", 20)

    econ_cfg = cfg.get("economics", {})
    reward_type = econ_cfg.get("reward_type", "log")
    reward_clip = econ_cfg.get("reward_clip", 2.0)

    logger.info(
        "Running %d random rollouts for reward_type=%s calibration...",
        n_episodes, reward_type,
    )

    profit_samples = run_random_rollouts(env_cls, cfg, n_episodes, seed)

    result = calibrate_reward_scale(
        profit_samples,
        reward_type=reward_type,
        reward_clip=reward_clip,
    )

    return result
