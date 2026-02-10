"""
Market model: join/churn with disconfirmation + churn determinants (Section 13).

References:
  [CHURN_SLR]   https://link.springer.com/article/10.1007/s11301-023-00335-7
  [DISCONF_PDF]  https://accesson.kr/ijcon/assets/pdf/55438/journal-21-1-11.pdf
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import sigmoid
from .pools import User

logger = logging.getLogger("oran.market")


class MarketModel:
    """Join and churn dynamics (Section 13)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        mc = cfg.get("market", {})

        self.beta_price: float = mc.get("beta_price", 0.5)
        self.beta_qos: float = mc.get("beta_qos", 0.3)
        self.beta_disc: float = mc.get("beta_disc", 0.4)
        self.beta_sw: float = mc.get("beta_sw", 0.2)
        self.U_outside: float = mc.get("U_outside", 0.0)
        self.U_outside_per_slice: Dict[str, float] = mc.get(
            "U_outside_per_slice", {}
        )
        self.delta_max: float = mc.get("delta_max", 50.0)

        self.lambda_join: Dict[str, float] = {
            "eMBB": mc.get("lambda_join_eMBB", 8.0),
            "URLLC": mc.get("lambda_join_URLLC", 3.0),
        }
        self.join_cap: Dict[str, int] = {
            "eMBB": mc.get("join_cap_eMBB", 25),
            "URLLC": mc.get("join_cap_URLLC", 10),
        }
        # D7: price_norm should match actual fee scale so price term doesn't dominate
        # Default uses midpoint of typical fee range (~70k KRW) instead of 10k
        self._price_norm: float = mc.get("price_norm", 70000.0)

    def compute_disconfirmation(self, T_exp: float, T_act_avg: float) -> float:
        delta = max(0.0, T_exp - T_act_avg)
        return min(delta, self.delta_max)

    def compute_stay_logit(self, user: User, F_s: float,
                           T_act_avg: float, delta_disc: float) -> float:
        delta_clamped = min(delta_disc, self.delta_max)
        U_out = self.U_outside_per_slice.get(user.slice, self.U_outside)

        logit = (
            user.b_u
            - self.beta_price * user.w_price * (F_s / self._price_norm)
            + self.beta_qos * user.w_qos * np.log1p(max(T_act_avg, 0.0))
            - self.beta_disc * delta_clamped
            - self.beta_sw * user.sw_cost
            - U_out
        )
        return float(logit)

    def compute_churn_prob(self, user: User, F_s: float,
                           T_act_avg: float, delta_disc: float) -> float:
        logit = self.compute_stay_logit(user, F_s, T_act_avg, delta_disc)
        p_stay = float(sigmoid(logit))
        return 1.0 - p_stay

    def sample_joins(self, slice_name: str, n_available: int,
                     rng: Optional[np.random.Generator] = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        lam = self.lambda_join.get(slice_name, 5.0)
        cap = self.join_cap.get(slice_name, 15)
        n_join = int(rng.poisson(lam))
        n_join = min(n_join, cap, n_available)
        return max(n_join, 0)

    def sample_churns(self, active_users: List[User], F_s: float,
                      rng: Optional[np.random.Generator] = None) -> List[int]:
        if rng is None:
            rng = np.random.default_rng()
        churned_ids: List[int] = []
        for user in active_users:
            p_churn = self.compute_churn_prob(
                user=user, F_s=F_s,
                T_act_avg=user.T_act_avg,
                delta_disc=user.delta_disc,
            )
            user.stay_prob = 1.0 - p_churn
            if rng.random() < p_churn:
                churned_ids.append(user.user_id)
        return churned_ids

    def update_disconfirmation(self, users: List[User]) -> None:
        for user in users:
            user.delta_disc = self.compute_disconfirmation(
                T_exp=user.T_exp, T_act_avg=user.T_act_avg,
            )
