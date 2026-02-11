"""
Data cap, throttle, and top-up model (Section 11).

FIX F1: Throttle now transitions throughput ceiling from v_max → v_cap.
  Before: apply_throttle simply clamped T_base to v_cap.
  After:  apply_throttle returns the new ceiling AND sets user.throttled.

  The throughput ceiling in the radio inner loop changes:
    Before cap exceeded:  T_ceil = v_max_mbps  (e.g., 300 Mbps)
    After cap exceeded:   T_ceil = v_cap_mbps  (e.g., 3 Mbps)

  This separation is consistent with how Korean mobile carriers enforce
  data caps: after Q_gb is consumed, the OCS/PCRF system enforces a
  speed restriction (속도제한) to v_cap_mbps.  [TWORLD_18][TWORLD_127]

References:
  [TWORLD_18][TWORLD_127]
  [TS23503] 3GPP TS 23.503 — Policy and Charging Control framework
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
                       v_max_mbps: float, v_cap_mbps: float) -> float:
        """Determine throughput ceiling based on data cap status.

        FIX F1: Returns the appropriate throughput ceiling:
          - v_max_mbps if user has NOT exceeded data cap
          - v_cap_mbps if user HAS exceeded data cap

        Args:
            D_u: User's monthly data consumption (GB).
            Q_s: Plan data cap (GB).
            v_max_mbps: Pre-cap peak throughput.
            v_cap_mbps: Post-cap throttle speed.

        Returns:
            Throughput ceiling (Mbps) for subsequent radio computation.
        """
        if D_u > Q_s:
            return v_cap_mbps
        return v_max_mbps

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
