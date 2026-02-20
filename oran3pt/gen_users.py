"""
Synthetic user population generator (Requirement 6 / §13).

Produces ``data/users_init.csv`` with per-user lognormal traffic parameters,
price/QoS sensitivities, switching cost, and CLV discount rate.

Parameter calibration policy (§13.3):
  - Lognormal (mu, sigma) fitted to target p50/p90 daily usage per slice.
  - Segment-level sensitivities drawn from literature-grounded ranges.

References:
  [Grubb AER 2009]       — 3-part tariff user heterogeneity
  [Nevo et al. 2015]     — broadband usage heterogeneity
  [Gupta JSR 2006]       — CLV discount rate
  [CHURN logit]          — switching-cost distributions
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .utils import load_config, fit_lognormal_quantiles

logger = logging.getLogger("oran3pt.gen_users")


def generate_users(cfg: Dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """[HI-2] Vectorized user generation — numpy batch operations replace
    Python for-loop. Output schema and statistical properties are identical.
    [Grubb AER 2009; Nevo 2016 — user heterogeneity]
    """
    rng = np.random.default_rng(seed)
    pop = cfg["population"]
    N = pop["N_total"]
    N_act = pop["N_active_init"]
    frac_u = pop["frac_urllc"]

    seg_names = pop["segments"]["names"]
    seg_probs = pop["segments"]["proportions"]
    psens_map = pop["segments"]["price_sensitivity"]
    qsens_map = pop["segments"]["qos_sensitivity"]
    swcost_map = pop["segments"]["switching_cost"]

    # Calibrate lognormal params from config targets
    traf = cfg["traffic"]
    mu_u, sig_u = fit_lognormal_quantiles(
        traf["URLLC"]["target_p50_gb_day"], traf["URLLC"]["target_p90_gb_day"])
    mu_e, sig_e = fit_lognormal_quantiles(
        traf["eMBB"]["target_p50_gb_day"], traf["eMBB"]["target_p90_gb_day"])

    # Vectorized slice assignment
    slices = np.where(rng.random(N) < frac_u, "URLLC", "eMBB")

    # Vectorized segment assignment
    segments = rng.choice(seg_names, size=N, p=seg_probs)

    # Vectorized per-user noise on lognormal params (±10% std)
    u_mu_u = mu_u + rng.normal(0, abs(mu_u) * 0.10, N)
    u_sig_u = np.maximum(0.05, sig_u + rng.normal(0, sig_u * 0.10, N))
    u_mu_e = mu_e + rng.normal(0, abs(mu_e) * 0.10, N)
    u_sig_e = np.maximum(0.05, sig_e + rng.normal(0, sig_e * 0.10, N))

    # Vectorized sensitivities: look up segment values, add noise
    ps_base = np.array([psens_map[s] for s in segments])
    qs_base = np.array([qsens_map[s] for s in segments])
    sc_base = np.array([swcost_map[s] for s in segments])

    ps = ps_base + rng.normal(0, 0.05, N)
    qs = qs_base + rng.normal(0, 0.05, N)
    sc = np.maximum(0.0, sc_base + rng.normal(0, 0.05, N))
    dr = 0.01 + rng.uniform(-0.002, 0.002, N)

    is_active = np.where(np.arange(N) < N_act, 1, 0)

    # ── [WTP-CHURN] WTP 분포 샘플링 ────────────────────────────────────
    # [Nevo et al. Econometrica 2016; Train & Weeks JChoice 2005]
    wtp_cfg = cfg.get("wtp_model", {})
    if wtp_cfg.get("enabled", False):
        wtp_base = np.zeros(N)
        wtp_total = np.zeros(N)
        wtp_decay = np.zeros(N)
        income = np.zeros(N)
        outside = np.zeros(N)
        loyalty = np.zeros(N)

        # [FIX-S3] Vectorized WTP sampling per slice×segment batch
        # [Nevo et al. 2016; Bolton 2003; Deaton 1980; Train 2009; Koszegi 2006]
        for sl_key, sl_label in [("urllc", "URLLC"), ("embb", "eMBB")]:
            for seg_key in seg_names:
                mask = (slices == sl_label) & (segments == seg_key)
                n = int(mask.sum())
                if n == 0:
                    continue
                dp = wtp_cfg[sl_key][seg_key]
                wtp_base[mask] = rng.lognormal(
                    dp["mu_base"], dp["sigma_base"], n)
                wtp_total[mask] = rng.lognormal(
                    dp["mu_total"], dp["sigma_total"], n)

                sp = wtp_cfg["segments"][seg_key]
                wtp_decay[mask] = rng.uniform(*sp["wtp_decay_rate"], n)
                income[mask] = rng.uniform(*sp["income_proxy"], n)
                outside[mask] = rng.uniform(*sp["outside_option"], n)
                loyalty[mask] = rng.uniform(*sp["loyalty_inertia"], n)

        # Consistency constraint: wtp_total >= wtp_base * 1.05
        wtp_total = np.maximum(wtp_total, wtp_base * 1.05)
    else:
        # WTP model disabled → NaN columns (backward compat)
        wtp_base = np.full(N, np.nan)
        wtp_total = np.full(N, np.nan)
        wtp_decay = np.full(N, np.nan)
        income = np.full(N, np.nan)
        outside = np.full(N, np.nan)
        loyalty = np.full(N, np.nan)

    return pd.DataFrame({
        "user_id": np.arange(N),
        "slice": slices,
        "segment": segments,
        "mu_urllc": np.round(u_mu_u, 5),
        "sigma_urllc": np.round(u_sig_u, 5),
        "mu_embb": np.round(u_mu_e, 5),
        "sigma_embb": np.round(u_sig_e, 5),
        "price_sensitivity": np.round(ps, 4),
        "qos_sensitivity": np.round(qs, 4),
        "switching_cost": np.round(sc, 4),
        "clv_discount_rate": np.round(dr, 5),
        "is_active_init": is_active,
        "wtp_base_fee": np.round(wtp_base, 0),
        "wtp_total_bill": np.round(wtp_total, 0),
        "wtp_decay_rate": np.round(wtp_decay, 4),
        "income_proxy": np.round(income, 4),
        "outside_option": np.round(outside, 4),
        "loyalty_inertia": np.round(loyalty, 4),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic user CSV")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="data/users_init.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(name)s] %(message)s")

    cfg = load_config(args.config)
    df = generate_users(cfg, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %d users → %s", len(df), out)
    logger.info("  Active: %d  Inactive: %d",
                int(df["is_active_init"].sum()),
                int((1 - df["is_active_init"]).sum()))
    logger.info("  URLLC: %d  eMBB: %d",
                int((df["slice"] == "URLLC").sum()),
                int((df["slice"] == "eMBB").sum()))


if __name__ == "__main__":
    main()
