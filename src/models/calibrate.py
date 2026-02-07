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

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import optimize

from .demand import DemandConfig, DemandModel
from .market import MarketModel
from .pools import UserPoolManager
from .radio import RadioConfig, RadioModel
from .topup import TopUpModel
from .economics import EconomicsModel
from .utils import (
    load_config,
    merge_configs,
    save_config,
    setup_logger,
    compute_price_bounds,
    sigmoid,
)

logger = logging.getLogger("oran.calibrate")


# =====================================================================
# 1)  calibrate_demand   [LOGNORMAL_TNET]
# =====================================================================

def calibrate_demand(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fit lognormal (mu, sigma) per slice via quantile matching.

    Uses p50 and p90 targets from config.  The analytical relationship:
      p50 = exp(mu)
      p90 = exp(mu + sigma * z_0.90)
    gives a closed-form solution.

    Returns updated config dict fragment with calibrated mu, sigma.
    """
    updates: Dict[str, Any] = {"demand": {}}
    demand_cfg = cfg.get("demand", {})
    tol = cfg.get("calibration", {}).get("demand_tolerance", 0.10)

    for sname in cfg.get("slices", {}).get("names", ["eMBB", "URLLC"]):
        sc = demand_cfg.get(sname, {})
        p50 = sc.get("target_p50_gb", 10.0)
        p90 = sc.get("target_p90_gb", 35.0)
        target_mean = sc.get("target_mean_gb", 15.0)

        mu, sigma = DemandModel.fit_lognormal_quantiles(p50, p90)

        fitted_p50 = DemandModel.lognormal_median(mu, sigma)
        fitted_p90 = DemandModel.lognormal_quantile(mu, sigma, 0.90)
        fitted_mean = DemandModel.lognormal_mean(mu, sigma)

        p50_err = abs(fitted_p50 - p50) / max(p50, 1e-9)
        p90_err = abs(fitted_p90 - p90) / max(p90, 1e-9)

        logger.info(
            "calibrate_demand [%s]: mu=%.4f sigma=%.4f | "
            "p50=%.2f(target=%.2f,err=%.4f) p90=%.2f(target=%.2f,err=%.4f) "
            "mean=%.2f(target=%.2f)",
            sname, mu, sigma,
            fitted_p50, p50, p50_err, fitted_p90, p90, p90_err,
            fitted_mean, target_mean,
        )

        if p50_err > tol or p90_err > tol:
            logger.warning(
                "calibrate_demand [%s]: quantile error exceeds tolerance",
                sname,
            )

        updates["demand"][sname] = {"mu": mu, "sigma": sigma}

    return updates


# =====================================================================
# 2)  calibrate_market   [CHURN_SLR][DISCONF_PDF]
# =====================================================================

def _baseline_churn_rate(
    beta_price: float,
    beta_qos: float,
    beta_sw: float,
    U_outside: float,
    F_baseline: float,
    T_act_baseline: float,
    seg_cfg: Dict[str, Any],
    seg_names: list,
    seg_probs: list,
    price_norm: float = 10000.0,
) -> float:
    """Compute expected baseline churn rate (no disconfirmation).

    Population-weighted average over segments.
    """
    sens_cfg = seg_cfg.get("sensitivity", {})

    total_churn = 0.0
    for seg_name, seg_prob in zip(seg_names, seg_probs):
        s = sens_cfg.get(seg_name, {})
        w_price = s.get("w_price", 1.0)
        w_qos = s.get("w_qos", 1.0)
        sw_cost = s.get("sw_cost", 0.5)
        b_u = s.get("b_u", 0.0)

        logit_stay = (
            b_u
            - beta_price * w_price * (F_baseline / price_norm)
            + beta_qos * w_qos * np.log1p(max(T_act_baseline, 0.0))
            - beta_sw * sw_cost
            - U_outside
        )
        p_stay = float(sigmoid(logit_stay))
        total_churn += seg_prob * (1.0 - p_stay)

    return total_churn


def calibrate_market(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fit churn model by solving for U_outside offset.

    Strategy:
      - Keep beta_price, beta_qos, beta_sw, beta_disc at initial values
        (these control sensitivity/elasticity around the operating point).
      - Solve for U_outside such that the population-averaged baseline
        churn rate matches the target for each slice.
      - Final U_outside is averaged across slices.

    This is equivalent to finding the logit intercept that sets the
    correct operating point.  [CHURN_SLR][DISCONF_PDF]

    Returns updated config dict fragment.
    """
    market_cfg = cfg.get("market", {})
    seg_cfg = cfg.get("segments", {})
    seg_names = seg_cfg.get("names", ["light", "mid", "heavy", "qos_sensitive"])
    seg_probs = seg_cfg.get("proportions", [0.25, 0.40, 0.25, 0.10])
    tol = cfg.get("calibration", {}).get("market_tolerance", 0.15)
    price_norm = market_cfg.get("price_norm", 10000.0)

    beta_price = market_cfg.get("beta_price", 0.5)
    beta_qos = market_cfg.get("beta_qos", 0.3)
    beta_sw = market_cfg.get("beta_sw", 0.2)

    price_bounds = compute_price_bounds(cfg)
    calibrated = {}

    for sname in cfg.get("slices", {}).get("names", ["eMBB", "URLLC"]):
        target_churn = market_cfg.get(f"target_churn_rate_{sname}", 0.03)

        pb = price_bounds[sname]
        F_baseline = (pb["F_min"] + pb["F_max"]) / 2.0

        slo_key = f"SLO_T_user_{sname}_mbps"
        T_baseline = cfg.get("sla", {}).get(slo_key, 10.0) * 2.0

        # Solve: find U_outside such that churn_rate = target
        def objective(U_out: float) -> float:
            rate = _baseline_churn_rate(
                beta_price, beta_qos, beta_sw, U_out,
                F_baseline, T_baseline,
                seg_cfg, seg_names, seg_probs, price_norm,
            )
            return rate - target_churn

        # U_outside is subtracted in logit_stay.
        # Large negative U_outside → higher logit_stay → lower churn.
        # Bracket: U_outside in [-50, 50]
        try:
            U_opt = optimize.brentq(objective, -50.0, 50.0, maxiter=500)
        except ValueError:
            logger.warning(
                "calibrate_market [%s]: bracket [-50,50] failed; "
                "trying [-200,200]", sname,
            )
            try:
                U_opt = optimize.brentq(objective, -200.0, 200.0, maxiter=500)
            except ValueError:
                logger.error(
                    "calibrate_market [%s]: cannot find U_outside; "
                    "using 0.0", sname,
                )
                U_opt = 0.0

        actual_rate = _baseline_churn_rate(
            beta_price, beta_qos, beta_sw, U_opt,
            F_baseline, T_baseline,
            seg_cfg, seg_names, seg_probs, price_norm,
        )
        err = abs(actual_rate - target_churn) / max(target_churn, 1e-9)

        logger.info(
            "calibrate_market [%s]: U_outside=%.4f | "
            "churn_rate=%.4f (target=%.4f, rel_err=%.4f) | "
            "F_base=%.0f T_base=%.1f",
            sname, U_opt, actual_rate, target_churn, err,
            F_baseline, T_baseline,
        )

        if err > tol:
            logger.warning(
                "calibrate_market [%s]: calibration error %.4f exceeds "
                "tolerance %.4f", sname, err, tol,
            )

        calibrated[sname] = {
            "U_outside": U_opt,
            "actual_churn_rate": actual_rate,
        }

    # Store per-slice U_outside for accurate per-slice churn
    U_per_slice = {
        sname: v["U_outside"] for sname, v in calibrated.items()
    }
    avg_U = float(np.mean(list(U_per_slice.values())))

    updates = {
        "market": {
            "beta_price": beta_price,
            "beta_qos": beta_qos,
            "beta_sw": beta_sw,
            "beta_disc": market_cfg.get("beta_disc", 0.4),
            "U_outside": avg_U,
            "U_outside_per_slice": U_per_slice,
            "_calibration_info": {
                "per_slice": calibrated,
                "avg_U_outside": avg_U,
            },
        }
    }
    return updates


# =====================================================================
# 3)  calibrate_reward_scale   [SB3_TIPS]
# =====================================================================

def _run_random_episode(cfg: Dict[str, Any], rng: np.random.Generator) -> float:
    """Run one episode with random actions; return final cumulative profit."""
    radio = RadioModel(RadioConfig.from_config(cfg))
    demand_model = DemandModel(DemandConfig.from_config(cfg))
    pool_mgr = UserPoolManager.from_config(cfg, rng=rng)
    market = MarketModel(cfg)
    econ = EconomicsModel(cfg)
    topup_model = TopUpModel(cfg)
    price_bounds = compute_price_bounds(cfg)

    K = cfg["time"]["inner_loop_K"]
    episode_len = cfg["time"]["episode_len_months"]
    topup_price = cfg.get("topup", {}).get("price_krw", 11000)

    Q_mid, v_cap_mid = {}, {}
    for sname in ["eMBB", "URLLC"]:
        plans = cfg["plans"][sname]
        mid = plans[len(plans) // 2]
        Q_mid[sname] = mid["Q_gb_month"]
        v_cap_mid[sname] = mid["v_cap_mbps"]

    monthly_profits = []

    for month in range(episode_len):
        rho_URLLC = float(rng.uniform(0.05, 0.95))
        fees = {}
        for sname in ["eMBB", "URLLC"]:
            pb = price_bounds[sname]
            fees[sname] = float(rng.uniform(pb["F_min"], pb["F_max"]))

        # Join
        pool_mgr.reset_monthly_fields()
        for sname in ["eMBB", "URLLC"]:
            n_avail = pool_mgr.inactive_count(sname)
            n_join = market.sample_joins(sname, n_avail, rng=rng)
            candidates = [
                u.user_id for u in pool_mgr.inactive_pool.values()
                if u.slice == sname
            ][:n_join]
            pool_mgr.join(candidates)

        N_active = {s: pool_mgr.active_count(s) for s in ["eMBB", "URLLC"]}

        # Demand
        for sname in ["eMBB", "URLLC"]:
            users = pool_mgr.get_active_users(sname)
            if not users:
                continue
            segs = np.array([u.segment for u in users])
            D = demand_model.sample_demand(sname, len(users), segs, rng=rng)
            for i, u in enumerate(users):
                u.D_u = D[i]
                u.T_exp = topup_model.apply_throttle(
                    u.D_u, Q_mid[sname], 100.0, v_cap_mid[sname]
                )

        # Inner loop
        avg_T_steps = {"eMBB": [], "URLLC": []}
        rho_utils = []
        for k in range(K):
            users_e = pool_mgr.get_active_users("eMBB")
            users_u = pool_mgr.get_active_users("URLLC")
            T_exp_e = np.array([u.T_exp for u in users_e]) if users_e else np.array([])
            T_exp_u = np.array([u.T_exp for u in users_u]) if users_u else np.array([])

            result = radio.inner_step(
                N_active_eMBB=N_active["eMBB"],
                N_active_URLLC=N_active["URLLC"],
                rho_URLLC=rho_URLLC,
                T_exp_users_eMBB=T_exp_e,
                T_exp_users_URLLC=T_exp_u,
            )
            avg_T_steps["eMBB"].append(result["avg_T_act_eMBB"])
            avg_T_steps["URLLC"].append(result["avg_T_act_URLLC"])
            rho_utils.append(
                (result["rho_util_eMBB"] + result["rho_util_URLLC"]) / 2
            )

        mean_rho = float(np.mean(rho_utils)) if rho_utils else 0.0
        V_rates = {}
        for sname in ["eMBB", "URLLC"]:
            arr = np.array(avg_T_steps[sname])
            V_rates[sname] = econ.sla.compute_violation_rate(arr, sname)

        # Update user fields + churn
        for sname in ["eMBB", "URLLC"]:
            users = pool_mgr.get_active_users(sname)
            avg_t = float(np.mean(avg_T_steps[sname])) if avg_T_steps[sname] else 0.0
            for u in users:
                u.T_act_avg = avg_t
            market.update_disconfirmation(users)
            churned = market.sample_churns(users, fees[sname], rng=rng)
            pool_mgr.churn(churned)

        # Profit
        profit_result = econ.compute_profit(
            fees=fees, N_active=N_active,
            n_topups={"eMBB": 0, "URLLC": 0},
            topup_price=topup_price,
            V_rates=V_rates,
            mean_rho_util=mean_rho,
            avg_load=mean_rho,
        )
        monthly_profits.append(profit_result["profit"])

    return float(np.mean(monthly_profits))


def calibrate_reward_scale(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Random policy rollouts → profit_scale = p95(|Pi|).  [SB3_TIPS]

    Runs N episodes with random actions, collects per-episode mean
    monthly profit, and sets profit_scale to p95 of |mean_profit|.
    """
    n_episodes = cfg.get("calibration", {}).get("reward_scale_episodes", 20)

    logger.info("calibrate_reward_scale: running %d random episodes...",
                n_episodes)

    profits = []
    for ep in range(n_episodes):
        rng = np.random.default_rng()
        try:
            mean_profit = _run_random_episode(cfg, rng)
            profits.append(abs(mean_profit))
            if (ep + 1) % 5 == 0 or ep == 0:
                logger.info("  episode %d/%d: |mean_profit|=%.0f",
                            ep + 1, n_episodes, abs(mean_profit))
        except Exception as e:
            logger.warning("  episode %d failed: %s", ep + 1, e)

    if not profits:
        logger.error("calibrate_reward_scale: all episodes failed")
        return {"economics": {"profit_scale": 1e6}}

    profits_arr = np.array(profits)
    p95 = float(np.percentile(profits_arr, 95))
    profit_scale = max(p95, 1.0)

    logger.info(
        "calibrate_reward_scale: p95(|profit|)=%.0f → profit_scale=%.0f "
        "(mean=%.0f, std=%.0f)",
        p95, profit_scale, profits_arr.mean(), profits_arr.std(),
    )

    return {
        "economics": {
            "profit_scale": profit_scale,
            "_calibration_info": {
                "n_episodes": len(profits),
                "p95_abs_profit": p95,
                "mean_abs_profit": float(profits_arr.mean()),
            },
        }
    }


# =====================================================================
# Run all calibrations
# =====================================================================

def run_all_calibrations(config_path: str) -> Dict[str, Any]:
    """Run all calibration routines and save calibrated.yaml."""
    cfg = load_config(config_path)
    config_dir = Path(config_path).parent

    logger.info("=" * 60)
    logger.info("Starting calibration pipeline")
    logger.info("=" * 60)

    # 1) Demand
    logger.info("--- calibrate_demand ---")
    demand_updates = calibrate_demand(cfg)
    cfg = merge_configs(cfg, demand_updates)

    # 2) Market
    logger.info("--- calibrate_market ---")
    market_updates = calibrate_market(cfg)
    cfg = merge_configs(cfg, market_updates)

    # 3) Reward scale
    logger.info("--- calibrate_reward_scale ---")
    reward_updates = calibrate_reward_scale(cfg)
    cfg = merge_configs(cfg, reward_updates)

    # Save
    out_path = config_dir / "calibrated.yaml"
    save_config(cfg, out_path)
    logger.info("Calibrated config saved to: %s", out_path)
    logger.info("=" * 60)

    return cfg


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration (Section 19)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    setup_logger("oran", logging.INFO)
    setup_logger("oran.calibrate", logging.INFO)

    run_all_calibrations(args.config)
