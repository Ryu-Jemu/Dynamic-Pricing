"""
Calibration module (Section 19).

Avoids arbitrary hardcoded numbers by fitting parameters to scenario
targets via quantile matching and optimization.

Three calibration routines:
  1) calibrate_demand()        — fit lognormal mu, sigma  [LOGNORMAL_TNET]
  2) calibrate_market()        — fit churn offset U_outside [CHURN_SLR][DISCONF_PDF]
  3) calibrate_reward_scale()  — random rollouts → profit_scale [SB3_TIPS]

Usage (CLI):
  python -m src.models.calibrate --config config/default.yaml

Output:
  config/calibrated.yaml  (merged default + calibrated overrides)

References:
  [LOGNORMAL_TNET] IEEE/ACM Trans. Networking 2021
  [CHURN_SLR]      Springer SLR 2023
  [DISCONF_PDF]    AccessON 2021
  [SB3_TIPS]       SB3 RL Tips and Tricks
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import optimize

from .utils import load_config, save_config, merge_configs, sigmoid, compute_price_bounds
from .demand import DemandModel

logger = logging.getLogger("oran.calibrate")


# =====================================================================
# 19.1  Demand calibration  [LOGNORMAL_TNET]
# =====================================================================

def calibrate_demand(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fit lognormal (mu, sigma) per slice to target p50/p90 quantiles."""
    demand_cfg = cfg.get("demand", {})
    updates: Dict[str, Any] = {"demand": {}}

    for sname in cfg.get("slices", {}).get("names", ["eMBB", "URLLC"]):
        sc = demand_cfg.get(sname, {})
        target_p50 = sc.get("target_p50_gb", 10.0)
        target_p90 = sc.get("target_p90_gb", 35.0)

        mu, sigma = DemandModel.fit_lognormal_quantiles(target_p50, target_p90)

        updates["demand"][sname] = {"mu": mu, "sigma": sigma}

        logger.info(
            "Demand calibration [%s]: mu=%.4f, sigma=%.4f "
            "(target p50=%.1f, p90=%.1f)",
            sname, mu, sigma, target_p50, target_p90,
        )

    return updates


# =====================================================================
# 19.2  Market calibration  [CHURN_SLR][DISCONF_PDF]
# =====================================================================

def _baseline_churn_rate(
    beta_price: float,
    beta_qos: float,
    beta_sw: float,
    U_outside: float,
    F_baseline: float,
    T_act_baseline: float,
    seg_cfg: Dict[str, Any],
    seg_names: List[str],
    seg_probs: List[float],
    price_norm: float = 70000.0,
) -> float:
    """Compute expected baseline monthly churn rate (analytical)."""
    sensitivity = seg_cfg.get("sensitivity", {})
    weighted_churn = 0.0

    for seg_name, seg_prob in zip(seg_names, seg_probs):
        s = sensitivity.get(seg_name, {})
        w_price = s.get("w_price", 1.0)
        w_qos = s.get("w_qos", 1.0)
        sw_cost = s.get("sw_cost", 0.5)
        b_u = s.get("b_u", 0.0)

        logit = (
            b_u
            - beta_price * w_price * (F_baseline / price_norm)
            + beta_qos * w_qos * np.log1p(max(T_act_baseline, 0.0))
            - beta_sw * sw_cost
            - U_outside
        )
        p_stay = float(sigmoid(logit))
        p_churn = 1.0 - p_stay
        weighted_churn += seg_prob * p_churn

    return weighted_churn


def calibrate_market(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Calibrate U_outside per slice to match target churn rates."""
    mc = cfg.get("market", {})
    seg_cfg = cfg.get("segments", {})
    seg_names = seg_cfg.get("names", ["light", "mid", "heavy", "qos_sensitive"])
    seg_probs = seg_cfg.get("proportions", [0.25, 0.40, 0.25, 0.10])
    price_bounds = compute_price_bounds(cfg)

    beta_price = mc.get("beta_price", 0.5)
    beta_qos = mc.get("beta_qos", 0.3)
    beta_sw = mc.get("beta_sw", 0.2)
    price_norm = mc.get("price_norm", 70000.0)

    U_outside_per_slice: Dict[str, float] = {}

    for sname in cfg.get("slices", {}).get("names", ["eMBB", "URLLC"]):
        target_churn = mc.get(f"target_churn_rate_{sname}", 0.03)

        pb = price_bounds[sname]
        F_baseline = (pb["F_min"] + pb["F_max"]) / 2.0

        slo_key = f"SLO_T_user_{sname}_mbps"
        T_baseline = cfg.get("sla", {}).get(slo_key, 10.0) * 2.0

        def residual(U_out: float) -> float:
            rate = _baseline_churn_rate(
                beta_price=beta_price,
                beta_qos=beta_qos,
                beta_sw=beta_sw,
                U_outside=U_out,
                F_baseline=F_baseline,
                T_act_baseline=T_baseline,
                seg_cfg=seg_cfg,
                seg_names=seg_names,
                seg_probs=seg_probs,
                price_norm=price_norm,
            )
            return rate - target_churn

        try:
            U_opt = float(optimize.brentq(residual, -20.0, 20.0, xtol=1e-6))
        except ValueError:
            logger.warning(
                "brentq failed for %s; using U_outside=0.0 as fallback.",
                sname,
            )
            U_opt = 0.0

        U_outside_per_slice[sname] = U_opt

        actual_rate = _baseline_churn_rate(
            beta_price=beta_price, beta_qos=beta_qos, beta_sw=beta_sw,
            U_outside=U_opt, F_baseline=F_baseline,
            T_act_baseline=T_baseline, seg_cfg=seg_cfg,
            seg_names=seg_names, seg_probs=seg_probs,
            price_norm=price_norm,
        )
        logger.info(
            "Market calibration [%s]: U_outside=%.4f → churn=%.4f "
            "(target=%.4f)",
            sname, U_opt, actual_rate, target_churn,
        )

    U_global = float(np.mean(list(U_outside_per_slice.values())))

    return {
        "market": {
            "beta_price": beta_price,
            "beta_qos": beta_qos,
            "beta_sw": beta_sw,
            "U_outside": U_global,
            "U_outside_per_slice": U_outside_per_slice,
        }
    }


# =====================================================================
# 19.3  Reward scale calibration  [SB3_TIPS]
# =====================================================================

def calibrate_reward_scale_from_samples(
    profit_samples: np.ndarray,
    reward_type: str = "log",
    reward_clip: float = 2.0,
    min_scale: float = 1.0,
) -> Dict[str, Any]:
    """Calibrate profit_scale from pre-collected profit samples."""
    if len(profit_samples) == 0:
        logger.warning("Empty profit_samples; returning min_scale=%.1f", min_scale)
        return {"profit_scale": min_scale, "method": "empty_fallback", "stats": {}}

    abs_profits = np.abs(profit_samples)
    abs_profits = abs_profits[np.isfinite(abs_profits)]

    if len(abs_profits) == 0:
        logger.warning("All profit samples non-finite; returning min_scale")
        return {"profit_scale": min_scale, "method": "nonfinite_fallback", "stats": {}}

    p50 = float(np.percentile(abs_profits, 50))
    p75 = float(np.percentile(abs_profits, 75))
    p95 = float(np.percentile(abs_profits, 95))
    p99 = float(np.percentile(abs_profits, 99))
    mean_abs = float(np.mean(abs_profits))

    stats = {
        "n_samples": len(profit_samples), "n_valid": len(abs_profits),
        "mean_abs_profit": mean_abs, "p50": p50, "p75": p75,
        "p95": p95, "p99": p99,
    }

    if reward_type == "tanh":
        scale = max(p95, min_scale)
        method = "tanh: scale = p95(|profit|)"
    elif reward_type == "linear":
        target_r = reward_clip * 0.8
        scale = max(p95 / max(target_r, 0.1), min_scale)
        method = f"linear: scale = p95 / (clip×0.8)"
    elif reward_type == "log":
        scale = max(p50, min_scale)
        method = "log: scale = median(|profit|)"
    else:
        raise ValueError(f"Unknown reward_type: '{reward_type}'")

    logger.info("Calibrated profit_scale=%.0f (%s)", scale, method)
    return {"profit_scale": float(scale), "method": method, "stats": stats}


def run_random_rollouts(
    env_cls: type,
    cfg: Dict[str, Any],
    n_episodes: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Collect profit samples from random policy rollouts."""
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


def calibrate_reward_scale(cfg: Dict[str, Any],
                           n_episodes: Optional[int] = None,
                           seed: int = 42) -> Dict[str, Any]:
    """Full reward-scale calibration pipeline: rollouts → profit_scale."""
    from ..envs.oran_slicing_env import OranSlicingEnv

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

    profit_samples = run_random_rollouts(OranSlicingEnv, cfg, n_episodes, seed)

    result = calibrate_reward_scale_from_samples(
        profit_samples, reward_type=reward_type, reward_clip=reward_clip,
    )

    return {
        "economics": {
            "profit_scale": result["profit_scale"],
        }
    }


# Backward-compatible alias
calibrate_from_config = calibrate_reward_scale


# =====================================================================
# CLI entry point
# =====================================================================

def main() -> None:
    """Run full calibration pipeline and write calibrated.yaml."""
    parser = argparse.ArgumentParser(
        description="Calibrate demand, market, and reward_scale (Section 19)"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--output", type=str, default="config/calibrated.yaml",
        help="Path for calibrated config output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)

    logger.info("=== Step 1/3: Demand calibration ===")
    demand_updates = calibrate_demand(cfg)
    cfg = merge_configs(cfg, demand_updates)

    logger.info("=== Step 2/3: Market calibration ===")
    market_updates = calibrate_market(cfg)
    cfg = merge_configs(cfg, market_updates)

    logger.info("=== Step 3/3: Reward scale calibration ===")
    reward_updates = calibrate_reward_scale(cfg)
    cfg = merge_configs(cfg, reward_updates)

    save_config(cfg, args.output)
    logger.info("Calibrated config saved: %s", args.output)


if __name__ == "__main__":
    main()
