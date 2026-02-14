"""
Synthetic user population generator (Requirement 6 / §13).

Produces ``data/users_init.csv`` with per-user lognormal traffic parameters,
price/QoS sensitivities, switching cost, CLV discount rate, and spatial
coordinates within the cell coverage area.

Parameter calibration policy (§13.3):
  - Lognormal (mu, sigma) fitted to target p50/p90 daily usage per slice.
  - Segment-level sensitivities drawn from literature-grounded ranges.

[M13a] Spatial coordinate generation:
  - Each user assigned persistent (x, y) within circular cell.
  - Segment-aware radial distribution: QoS-sensitive users cluster near tower,
    price-sensitive users toward cell edge.
  - sqrt-uniform radial sampling for uniform area density.

References:
  [Grubb AER 2009]       — 3-part tariff user heterogeneity
  [Nevo et al. 2015]     — broadband usage heterogeneity
  [Gupta JSR 2006]       — CLV discount rate
  [CHURN logit]          — switching-cost distributions
  [3GPP TR 38.913 §6.1]  — Urban Macro user spatial distribution
  [Devroye 1986]         — sqrt-uniform radial sampling for disk
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
    rng = np.random.default_rng(seed)
    pop = cfg["population"]
    N = pop["N_total"]
    N_act = pop["N_active_init"]
    frac_u = pop["frac_urllc"]

    seg_names = pop["segments"]["names"]
    seg_probs = pop["segments"]["proportions"]
    psens = pop["segments"]["price_sensitivity"]
    qsens = pop["segments"]["qos_sensitivity"]
    swcost = pop["segments"]["switching_cost"]

    # [M13a] Spatial config — segment-aware radial distribution
    spatial = pop.get("spatial", {})
    cell_radius = spatial.get("cell_radius", 20.0)
    default_radii = {
        "qos_sensitive": [2.0, 10.0],
        "balanced": [3.0, 18.0],
        "price_sensitive": [8.0, 20.0],
    }
    seg_radii = spatial.get("segment_radii", default_radii)

    # Calibrate lognormal params from config targets
    traf = cfg["traffic"]
    mu_u, sig_u = fit_lognormal_quantiles(
        traf["URLLC"]["target_p50_gb_day"], traf["URLLC"]["target_p90_gb_day"])
    mu_e, sig_e = fit_lognormal_quantiles(
        traf["eMBB"]["target_p50_gb_day"], traf["eMBB"]["target_p90_gb_day"])

    rows = []
    for uid in range(N):
        # Assign slice
        sl = "URLLC" if rng.random() < frac_u else "eMBB"
        # Assign segment
        seg = rng.choice(seg_names, p=seg_probs)

        # Per-user noise on lognormal params (±10 % std)
        u_mu_u = mu_u + rng.normal(0, abs(mu_u) * 0.10)
        u_sig_u = max(0.05, sig_u + rng.normal(0, sig_u * 0.10))
        u_mu_e = mu_e + rng.normal(0, abs(mu_e) * 0.10)
        u_sig_e = max(0.05, sig_e + rng.normal(0, sig_e * 0.10))

        # Sensitivities with small noise
        ps = psens[seg] + rng.normal(0, 0.05)
        qs = qsens[seg] + rng.normal(0, 0.05)
        sc = max(0.0, swcost[seg] + rng.normal(0, 0.05))
        dr = 0.01 + rng.uniform(-0.002, 0.002)   # CLV discount rate ~ 1 %

        is_active = 1 if uid < N_act else 0

        # [M13a] Spatial coordinates within cell  [3GPP TR 38.913; Devroye 1986]
        r_range = seg_radii.get(seg, [3.0, 18.0])
        r_min, r_max = float(r_range[0]), min(float(r_range[1]), cell_radius)
        r = float(np.sqrt(rng.uniform(r_min**2, r_max**2)))
        theta = float(rng.uniform(0, 2 * np.pi))
        x = round(r * np.cos(theta), 3)
        y = round(r * np.sin(theta), 3)

        rows.append({
            "user_id": uid,
            "slice": sl,
            "segment": seg,
            "mu_urllc": round(u_mu_u, 5),
            "sigma_urllc": round(u_sig_u, 5),
            "mu_embb": round(u_mu_e, 5),
            "sigma_embb": round(u_sig_e, 5),
            "price_sensitivity": round(ps, 4),
            "qos_sensitivity": round(qs, 4),
            "switching_cost": round(sc, 4),
            "clv_discount_rate": round(dr, 5),
            "is_active_init": is_active,
            "x": x,
            "y": y,
        })

    return pd.DataFrame(rows)


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
