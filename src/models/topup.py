"""
Data cap, throttle, and top-up model (Section 11).

References:
  [TWORLD_18][TWORLD_127]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .utils import sigmoid


class TopUpModel:
    """Top-up and throttle logic (Section 11)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        tu = cfg.get("topup", {})
        self.enabled = tu.get("enabled", True)
        self.once_per_month = tu.get("topup_once_per_month", True)
        self.price_krw = tu.get("price_krw", 11000)
        self.data_gb = tu.get("data_gb", 5.0)
        self.kappa_top = tu.get("kappa_top", 1.0)
        self.w_price_topup = tu.get("w_price_topup", 0.5)

    def apply_throttle(self, D_u: float, Q_s: float,
                       T_base: float, v_cap_mbps: float) -> float:
        if D_u > Q_s:
            return min(T_base, v_cap_mbps)
        return T_base

    def decide_topup(self, delta_utility: float, w_price_user: float,
                     rng: Optional[np.random.Generator] = None) -> bool:
        if not self.enabled:
            return False
        if rng is None:
            rng = np.random.default_rng()
        logit = self.kappa_top * (
            delta_utility - self.w_price_topup * self.price_krw / 10000.0
        )
        prob = float(sigmoid(logit))
        return bool(rng.random() < prob)
