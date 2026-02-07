"""
Data cap, throttle, and top-up model (Section 11).

Carrier plans commonly throttle speed after exceeding monthly quota.
  [TWORLD_18][TWORLD_127]

Top-up: max 1 per month, logit purchase probability (calibrated).

Will be fully implemented in Step 2.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .utils import sigmoid


class TopUpModel:
    """Top-up and throttle logic (Section 11).

    11.1  If D_u > Q_s → T_exp_u = min(T_base, v_cap_s)
    11.2  P(topup) = σ(κ_top * (ΔU_top - w_price * Price_top))
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        tu = cfg.get("topup", {})
        self.enabled = tu.get("enabled", True)
        self.once_per_month = tu.get("topup_once_per_month", True)
        self.price_krw = tu.get("price_krw", 11000)
        self.data_gb = tu.get("data_gb", 5.0)
        self.kappa_top = tu.get("kappa_top", 1.0)
        self.w_price_topup = tu.get("w_price_topup", 0.5)

    def apply_throttle(
        self,
        D_u: float,
        Q_s: float,
        T_base: float,
        v_cap_mbps: float,
    ) -> float:
        """Return user's expected throughput after data-cap check.

        If D_u > Q_s → min(T_base, v_cap_s)  [TWORLD_18][TWORLD_127]
        """
        if D_u > Q_s:
            return min(T_base, v_cap_mbps)
        return T_base

    def decide_topup(
        self,
        delta_utility: float,
        w_price_user: float,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        """Sample top-up purchase decision (max 1/month).

        P(topup) = σ(κ_top * (ΔU_top - w_price * Price_top))
        """
        if not self.enabled:
            return False
        if rng is None:
            rng = np.random.default_rng()

        logit = self.kappa_top * (delta_utility - self.w_price_topup * self.price_krw / 10000.0)
        prob = float(sigmoid(logit))
        return bool(rng.random() < prob)
