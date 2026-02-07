"""
Market model: join/churn with disconfirmation + churn determinants (Section 13).

Uses logistic churn/join with drivers supported by:
  - disconfirmation/confirmation affecting continuance [DISCONF_PDF]
  - churn determinants: service quality, satisfaction, switching costs [CHURN_SLR]

13.1  Disconfirmation:
  Δ_disc_u = max(0, T_exp_u - T_act_u_month_avg)
  Soft-saturate: min(Δ_disc_u, Δ_max)

13.2  Churn probability (logit; calibrated):
  P(stay_u) = σ( b_u
                  - β_price * w_price * F_s_norm
                  + β_qos   * w_qos   * log(1 + T_act_avg)
                  - β_disc  * min(Δ_disc, Δ_max)
                  - β_sw    * sw_cost
                  - U_outside )
  P(churn_u) = 1 - P(stay_u)

  All betas are calibrated; do NOT hardcode as "truth".  [CHURN_SLR][DISCONF_PDF]

13.3  Join model (Poisson):
  J_s ~ Poisson(lambda_join_s) with join_cap_s
  Calibrate lambda_join_s to match scenario target growth/steady-state.

References:
  [CHURN_SLR]   https://link.springer.com/article/10.1007/s11301-023-00335-7
  [DISCONF_PDF]  https://accesson.kr/ijcon/assets/pdf/55438/journal-21-1-11.pdf
  [SB3_TIPS]    https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import sigmoid
from .pools import User

logger = logging.getLogger("oran.market")


class MarketModel:
    """Join and churn dynamics (Section 13).

    All coefficients (beta_*) are calibrated in calibrate.py.
    Initial values in config are starting guesses only.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        mc = cfg.get("market", {})

        # ---- Churn logistic coefficients (calibrated) [CHURN_SLR][DISCONF_PDF] ----
        self.beta_price: float = mc.get("beta_price", 0.5)
        self.beta_qos: float = mc.get("beta_qos", 0.3)
        self.beta_disc: float = mc.get("beta_disc", 0.4)
        self.beta_sw: float = mc.get("beta_sw", 0.2)
        self.U_outside: float = mc.get("U_outside", 0.0)
        # Per-slice U_outside override (from calibration)
        self.U_outside_per_slice: Dict[str, float] = mc.get(
            "U_outside_per_slice", {}
        )
        self.delta_max: float = mc.get("delta_max", 50.0)

        # ---- Join model (Poisson) [CHURN_SLR] ----
        self.lambda_join: Dict[str, float] = {
            "eMBB": mc.get("lambda_join_eMBB", 8.0),
            "URLLC": mc.get("lambda_join_URLLC", 3.0),
        }
        self.join_cap: Dict[str, int] = {
            "eMBB": mc.get("join_cap_eMBB", 25),
            "URLLC": mc.get("join_cap_URLLC", 10),
        }

        # ---- Price normalization (for logit stability) ----
        # Normalize fee by 10,000 KRW unit so logit inputs stay O(1)
        self._price_norm: float = mc.get("price_norm", 10000.0)

    # =================================================================
    # 13.1  Disconfirmation  [DISCONF_PDF]
    # =================================================================

    def compute_disconfirmation(
        self,
        T_exp: float,
        T_act_avg: float,
    ) -> float:
        """Monthly disconfirmation, soft-saturated.

        Δ_disc = min( max(0, T_exp - T_act_avg), Δ_max )
        """
        delta = max(0.0, T_exp - T_act_avg)
        return min(delta, self.delta_max)

    # =================================================================
    # 13.2  Churn probability  [CHURN_SLR][DISCONF_PDF]
    # =================================================================

    def compute_stay_logit(
        self,
        user: User,
        F_s: float,
        T_act_avg: float,
        delta_disc: float,
    ) -> float:
        """Compute the logit (log-odds) of staying.

        logit(P_stay) = b_u
                        - β_price * w_price * (F_s / price_norm)
                        + β_qos   * w_qos   * log(1 + T_act_avg)
                        - β_disc  * min(Δ_disc, Δ_max)
                        - β_sw    * sw_cost
                        - U_outside
        """
        delta_clamped = min(delta_disc, self.delta_max)

        # Use per-slice U_outside if available, else global
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

    def compute_churn_prob(
        self,
        user: User,
        F_s: float,
        T_act_avg: float,
        delta_disc: float,
    ) -> float:
        """Compute P(churn) for a single user.  [CHURN_SLR][DISCONF_PDF]

        P(stay) = σ(logit)
        P(churn) = 1 - P(stay)
        """
        logit = self.compute_stay_logit(user, F_s, T_act_avg, delta_disc)
        p_stay = float(sigmoid(logit))
        return 1.0 - p_stay

    # =================================================================
    # 13.3  Join model (Poisson)  [CHURN_SLR]
    # =================================================================

    def sample_joins(
        self,
        slice_name: str,
        n_available: int,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Sample number of joins for a slice.

        J_s ~ Poisson(lambda_join_s), capped at join_cap_s
        and at the number of available inactive users.
        """
        if rng is None:
            rng = np.random.default_rng()

        lam = self.lambda_join.get(slice_name, 5.0)
        cap = self.join_cap.get(slice_name, 15)

        n_join = int(rng.poisson(lam))
        n_join = min(n_join, cap, n_available)
        return max(n_join, 0)

    # =================================================================
    # Batch churn sampling
    # =================================================================

    def sample_churns(
        self,
        active_users: List[User],
        F_s: float,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """Return user_ids of users who churn this month.

        For each active user, compute P(churn) using the user's
        current monthly fields (T_act_avg, delta_disc), then sample.
        """
        if rng is None:
            rng = np.random.default_rng()

        churned_ids: List[int] = []
        for user in active_users:
            p_churn = self.compute_churn_prob(
                user=user,
                F_s=F_s,
                T_act_avg=user.T_act_avg,
                delta_disc=user.delta_disc,
            )
            # Store stay_prob on user for logging
            user.stay_prob = 1.0 - p_churn

            if rng.random() < p_churn:
                churned_ids.append(user.user_id)

        return churned_ids

    # =================================================================
    # Batch disconfirmation update
    # =================================================================

    def update_disconfirmation(self, users: List[User]) -> None:
        """Update Δ_disc for a list of users using their current fields.

        Reads user.T_exp and user.T_act_avg, writes user.delta_disc.
        """
        for user in users:
            user.delta_disc = self.compute_disconfirmation(
                T_exp=user.T_exp,
                T_act_avg=user.T_act_avg,
            )
