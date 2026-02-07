"""
O-RAN 1-Cell Slicing + Pricing Gymnasium Environment.

Sections 16–17 of HybridPrompt.md.

Action space (Section 6.1):
  Box([-1,-1,-1], [1,1,1], float32)  →  (F_eMBB, F_URLLC, rho_URLLC)

Observation space (Section 16):
  Fixed-order float32 vector, finite bounds, no NaN/Inf.

Step order (Section 17 — fixed monthly):
  1) Map action → F_eMBB, F_URLLC, rho_URLLC
  2) Join: sample joins, inactive→active
  3) Generate monthly demand D_u
  4) Inner loop k=1..K
  5) Aggregate monthly KPIs
  6) Top-up + disconfirmation
  7) Churn: active→churned
  8) Profit/reward
  9) Log

References:
  [SB3_SAC]   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
  [SB3_TIPS]  https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.models.radio import RadioConfig, RadioModel
from src.models.demand import DemandConfig, DemandModel
from src.models.pools import UserPoolManager
from src.models.market import MarketModel
from src.models.economics import EconomicsModel
from src.models.topup import TopUpModel
from src.models.safety import sanitize_obs, validate_action, validate_state
from src.models.utils import compute_price_bounds

logger = logging.getLogger("oran.env")

# ---- Observation vector layout (Section 16) ----
# Index : Field                : Normalization
#  0    : month / episode_len  : [0, 1]
#  1    : N_active_eMBB / N_max: [0, 1]
#  2    : N_active_URLLC / N_max: [0, 1]
#  3    : joins_eMBB / join_cap: [0, 1]
#  4    : joins_URLLC / join_cap: [0, 1]
#  5    : churns_eMBB / N_max  : [0, 1]
#  6    : churns_URLLC / N_max : [0, 1]
#  7    : rho_URLLC            : [0.05, 0.95]
#  8    : F_eMBB / F_max_eMBB  : [0, 2]
#  9    : F_URLLC / F_max_URLLC: [0, 2]
# 10    : mean_rho_util        : [0, 1]
# 11    : avg_T_eMBB / T_cap   : [0, 1]
# 12    : avg_T_URLLC / T_cap  : [0, 1]
# 13    : V_rate_eMBB          : [0, 1]
# 14    : V_rate_URLLC         : [0, 1]
# 15    : profit_norm (tanh)   : [-1, 1]
OBS_DIM = 16
OBS_LOW = np.zeros(OBS_DIM, dtype=np.float32)
OBS_HIGH = np.ones(OBS_DIM, dtype=np.float32)
# Adjust known ranges
OBS_LOW[7] = 0.05   # rho_URLLC min
OBS_HIGH[7] = 0.95  # rho_URLLC max
OBS_HIGH[8] = 2.0   # normalized fee can exceed 1
OBS_HIGH[9] = 2.0
OBS_LOW[15] = -1.0  # tanh profit


class ORANSlicingEnv(gym.Env):
    """O-RAN 1-Cell Slicing + Pricing Environment.

    Monthly steps, 50-month episodes.
    Action: Box([-1,-1,-1],[1,1,1]) → (F_eMBB, F_URLLC, rho_URLLC)

    Compatible with SB3 SAC.  [SB3_SAC][SB3_TIPS]
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Spaces (Section 6.1) [SB3_SAC] ----
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, shape=(OBS_DIM,), dtype=np.float32,
        )

        # ---- Price bounds (Section 6.2) ----
        self.price_bounds = compute_price_bounds(cfg)

        # ---- Time (Section 5) ----
        self.episode_len: int = cfg["time"]["episode_len_months"]
        self.K: int = cfg["time"]["inner_loop_K"]

        # ---- Models ----
        self.radio = RadioModel(RadioConfig.from_config(cfg))
        self.demand_model = DemandModel(DemandConfig.from_config(cfg))
        self.market = MarketModel(cfg)
        self.econ = EconomicsModel(cfg)
        self.topup_model = TopUpModel(cfg)

        # ---- Plan catalog: use middle plan for Q, v_cap ----
        self.Q_mid: Dict[str, float] = {}
        self.v_cap_mid: Dict[str, float] = {}
        for sname in ["eMBB", "URLLC"]:
            plans = cfg["plans"][sname]
            mid = plans[len(plans) // 2]
            self.Q_mid[sname] = mid["Q_gb_month"]
            self.v_cap_mid[sname] = mid["v_cap_mbps"]

        self.topup_price: float = cfg.get("topup", {}).get("price_krw", 11000)

        # ---- Normalization constants ----
        self.N_max: float = float(
            cfg.get("population", {}).get("inactive_pool_size", 2000)
        )
        self.T_cap: float = float(
            cfg.get("radio", {}).get("T_cell_cap_mbps", 1000)
        )
        self.F_max_norm: Dict[str, float] = {
            s: self.price_bounds[s]["F_max"] for s in ["eMBB", "URLLC"]
        }

        # ---- Episode state ----
        self.rng: Optional[np.random.Generator] = None
        self.pool_mgr: Optional[UserPoolManager] = None
        self.month: int = 0
        self._last_info: Dict[str, Any] = {}

    # =================================================================
    # reset
    # =================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to month 0 with fresh user pools.

        No global seed fixing for training/eval (Section 5).
        """
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.pool_mgr = UserPoolManager.from_config(self.cfg, rng=self.rng)
        self.month = 0

        obs = self._build_obs_initial()
        self._last_info = {"month": 0}
        return obs, self._last_info

    # =================================================================
    # step  (Section 17 — fixed order)
    # =================================================================

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one monthly step.

        Returns: (obs, reward, terminated, truncated, info)
        """
        assert self.pool_mgr is not None, "Call reset() first"
        assert self.rng is not None

        # ---- Safety: validate action [SB3_TIPS] ----
        action = np.asarray(action, dtype=np.float32)
        action_ok, action_violations = validate_action(action)
        penalty = 0.0
        if not action_ok:
            action = np.clip(np.nan_to_num(action, 0.0), -1.0, 1.0)
            penalty += 1.0

        # ============================================================
        # 1) Map action → F_eMBB, F_URLLC, rho_URLLC  (Section 6.1)
        # ============================================================
        fees: Dict[str, float] = {}
        for idx, sname in enumerate(["eMBB", "URLLC"]):
            pb = self.price_bounds[sname]
            mid = (pb["F_min"] + pb["F_max"]) / 2.0
            half = (pb["F_max"] - pb["F_min"]) / 2.0
            fees[sname] = float(np.clip(
                mid + half * action[idx], pb["F_min"], pb["F_max"],
            ))

        rho_URLLC = float(np.clip(0.5 + 0.45 * action[2], 0.05, 0.95))

        # ============================================================
        # 2) Join  (Section 17.2)
        # ============================================================
        self.pool_mgr.reset_monthly_fields()
        n_joins: Dict[str, int] = {}
        for sname in ["eMBB", "URLLC"]:
            n_avail = self.pool_mgr.inactive_count(sname)
            n_join = self.market.sample_joins(sname, n_avail, rng=self.rng)
            candidates = [
                u.user_id for u in self.pool_mgr.inactive_pool.values()
                if u.slice == sname
            ][:n_join]
            self.pool_mgr.join(candidates)
            n_joins[sname] = len(candidates)

        N_active = {
            s: self.pool_mgr.active_count(s) for s in ["eMBB", "URLLC"]
        }

        # ============================================================
        # 3) Generate monthly demand  (Section 17.3)
        # ============================================================
        for sname in ["eMBB", "URLLC"]:
            users = self.pool_mgr.get_active_users(sname)
            if not users:
                continue
            segs = np.array([u.segment for u in users])
            D = self.demand_model.sample_demand(
                sname, len(users), segs, rng=self.rng,
            )
            T_base = 100.0  # base expected throughput before throttle
            for i, u in enumerate(users):
                u.D_u = D[i]
                u.T_exp = self.topup_model.apply_throttle(
                    u.D_u, self.Q_mid[sname], T_base, self.v_cap_mid[sname],
                )

        # ============================================================
        # 4) Inner loop k=1..K  (Section 17.4)
        # ============================================================
        avg_T_steps: Dict[str, list] = {"eMBB": [], "URLLC": []}
        rho_utils: list = []

        for k in range(self.K):
            users_e = self.pool_mgr.get_active_users("eMBB")
            users_u = self.pool_mgr.get_active_users("URLLC")
            T_exp_e = (
                np.array([u.T_exp for u in users_e])
                if users_e else np.array([])
            )
            T_exp_u = (
                np.array([u.T_exp for u in users_u])
                if users_u else np.array([])
            )

            result = self.radio.inner_step(
                N_active_eMBB=N_active["eMBB"],
                N_active_URLLC=N_active["URLLC"],
                rho_URLLC=rho_URLLC,
                T_exp_users_eMBB=T_exp_e,
                T_exp_users_URLLC=T_exp_u,
            )
            avg_T_steps["eMBB"].append(result["avg_T_act_eMBB"])
            avg_T_steps["URLLC"].append(result["avg_T_act_URLLC"])
            rho_utils.append(
                (result["rho_util_eMBB"] + result["rho_util_URLLC"]) / 2.0
            )

        # ============================================================
        # 5) Aggregate monthly KPIs  (Section 17.5)
        # ============================================================
        mean_rho = float(np.mean(rho_utils)) if rho_utils else 0.0
        avg_T: Dict[str, float] = {}
        V_rates: Dict[str, float] = {}
        for sname in ["eMBB", "URLLC"]:
            arr = np.array(avg_T_steps[sname])
            avg_T[sname] = float(arr.mean()) if len(arr) > 0 else 0.0
            V_rates[sname] = self.econ.sla.compute_violation_rate(arr, sname)

        # ============================================================
        # 6) Top-up + disconfirmation  (Section 17.6)
        # ============================================================
        n_topups: Dict[str, int] = {"eMBB": 0, "URLLC": 0}
        for sname in ["eMBB", "URLLC"]:
            users = self.pool_mgr.get_active_users(sname)
            for u in users:
                u.T_act_avg = avg_T[sname]
            # Top-up for exceeders
            for u in users:
                if u.D_u > self.Q_mid[sname] and not u.topup_flag:
                    delta_utility = max(0.0, u.T_exp - u.T_act_avg)
                    if self.topup_model.decide_topup(
                        delta_utility, u.w_price, rng=self.rng,
                    ):
                        u.topup_flag = True
                        n_topups[sname] += 1
            # Disconfirmation update
            self.market.update_disconfirmation(users)

        # ============================================================
        # 7) Churn  (Section 17.7)
        # ============================================================
        n_churns: Dict[str, int] = {}
        for sname in ["eMBB", "URLLC"]:
            users = self.pool_mgr.get_active_users(sname)
            churned_ids = self.market.sample_churns(
                users, fees[sname], rng=self.rng,
            )
            self.pool_mgr.churn(churned_ids)
            n_churns[sname] = len(churned_ids)

        # ============================================================
        # 8) Profit & reward  (Section 17.8)
        # ============================================================
        # State validation [SB3_TIPS]
        state_penalty, state_violations = validate_state(
            fees=fees,
            N_active=N_active,
            rho_urllc=rho_URLLC,
            rho_util={"eMBB": mean_rho, "URLLC": mean_rho},
        )
        penalty += state_penalty

        profit_result = self.econ.compute_profit(
            fees=fees,
            N_active=N_active,
            n_topups=n_topups,
            topup_price=self.topup_price,
            V_rates=V_rates,
            mean_rho_util=mean_rho,
            avg_load=mean_rho,
        )
        reward = self.econ.compute_reward(
            profit_result["profit"], penalty=penalty,
        )

        # ============================================================
        # 9) Build observation & info  (Section 17.9)
        # ============================================================
        self.month += 1
        terminated = self.month >= self.episode_len
        truncated = False

        # Post-churn active counts for obs
        N_active_post = {
            s: self.pool_mgr.active_count(s) for s in ["eMBB", "URLLC"]
        }

        obs = self._build_obs(
            month=self.month,
            N_active=N_active_post,
            n_joins=n_joins,
            n_churns=n_churns,
            rho_URLLC=rho_URLLC,
            fees=fees,
            mean_rho=mean_rho,
            avg_T=avg_T,
            V_rates=V_rates,
            profit=profit_result["profit"],
        )

        info = {
            "month": self.month,
            "fees": fees,
            "rho_URLLC": rho_URLLC,
            "N_active": N_active,
            "N_active_post_churn": N_active_post,
            "n_joins": n_joins,
            "n_churns": n_churns,
            "n_topups": n_topups,
            "avg_T": avg_T,
            "V_rates": V_rates,
            "mean_rho_util": mean_rho,
            "penalty": penalty,
            **profit_result,
        }
        self._last_info = info

        return obs, float(reward), terminated, truncated, info

    # =================================================================
    # Observation builders
    # =================================================================

    def _build_obs_initial(self) -> np.ndarray:
        """Build initial observation at month 0."""
        N_e = self.pool_mgr.active_count("eMBB") if self.pool_mgr else 0
        N_u = self.pool_mgr.active_count("URLLC") if self.pool_mgr else 0
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = 0.0  # month=0
        obs[1] = N_e / max(self.N_max, 1.0)
        obs[2] = N_u / max(self.N_max, 1.0)
        obs[7] = 0.5   # default rho_URLLC
        obs[8] = 0.5   # mid-range fee
        obs[9] = 0.5
        return sanitize_obs(obs)

    def _build_obs(
        self,
        month: int,
        N_active: Dict[str, int],
        n_joins: Dict[str, int],
        n_churns: Dict[str, int],
        rho_URLLC: float,
        fees: Dict[str, float],
        mean_rho: float,
        avg_T: Dict[str, float],
        V_rates: Dict[str, float],
        profit: float,
    ) -> np.ndarray:
        """Build observation vector (Section 16)."""
        eps_len = max(self.episode_len, 1)
        obs = np.array([
            month / eps_len,                                    #  0
            N_active.get("eMBB", 0) / max(self.N_max, 1.0),   #  1
            N_active.get("URLLC", 0) / max(self.N_max, 1.0),  #  2
            n_joins.get("eMBB", 0) / max(self.market.join_cap.get("eMBB", 25), 1),  #  3
            n_joins.get("URLLC", 0) / max(self.market.join_cap.get("URLLC", 10), 1), # 4
            n_churns.get("eMBB", 0) / max(self.N_max, 1.0),   #  5
            n_churns.get("URLLC", 0) / max(self.N_max, 1.0),  #  6
            rho_URLLC,                                          #  7
            fees.get("eMBB", 0) / max(self.F_max_norm["eMBB"], 1.0),   #  8
            fees.get("URLLC", 0) / max(self.F_max_norm["URLLC"], 1.0), #  9
            mean_rho,                                           # 10
            avg_T.get("eMBB", 0) / max(self.T_cap, 1.0),      # 11
            avg_T.get("URLLC", 0) / max(self.T_cap, 1.0),     # 12
            V_rates.get("eMBB", 0),                             # 13
            V_rates.get("URLLC", 0),                            # 14
            float(np.tanh(profit / max(abs(self.econ.profit_scale), 1.0))),  # 15
        ], dtype=np.float32)

        return sanitize_obs(obs)
