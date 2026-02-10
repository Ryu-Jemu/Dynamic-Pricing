"""
O-RAN 1-Cell Slicing + Pricing Environment (SB3 SAC, Monthly).

Gymnasium-compatible environment implementing Sections 4–17 of HybridPrompt.md.

Action space (continuous, 3D — §6.1):
  a[0] → F_eMBB    ∈ [F_min_eMBB, F_max_eMBB]       [FIX M5]
  a[1] → F_URLLC   ∈ [F_min_URLLC, F_max_URLLC]     [FIX M5]
  a[2] → rho_URLLC ∈ [rho_min, rho_max]              [FIX M5]

Observation space (float32, shape=(16,) — §16):
  See _build_obs() for full vector layout.

Step order (§17 — FIX M1):
  1) Map action → F_eMBB, F_URLLC, rho_URLLC
  2) Join: sample joins, move inactive→active                    [FIX M1]
  3) Generate monthly demand D_u for actives
  4) Inner loop k=1..K:
     - compute rho_util_k per slice                              [FIX M4]
     - compute T_eff_k, T_act_u,k
     - track per-step avg throughput for SLA                     [FIX M3]
  5) Aggregate monthly KPIs
  6) Top-up: throttle + top-up decision + disconfirmation        [FIX M2]
  7) Churn: sample churn, move active→churned                   [FIX M1]
  8) Compute revenue/cost/profit/reward

T_exp set from plan-based v_cap_mbps, not hardcoded 100.        [FIX M7]

References:
  [SAC][SB3_SAC][SB3_TIPS][TS38104][TS38214][TSDI_KEYNOTE]
  [LOGNORMAL_TNET][CONG_5G_PMC][LTE_LOAD_TPUT]
  [CHURN_SLR][DISCONF_PDF][VERIZON_SLA][BS_POWER]
  [TWORLD_18][TWORLD_127]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..models.radio import RadioConfig, RadioModel
from ..models.demand import DemandConfig, DemandModel
from ..models.market import MarketModel
from ..models.pools import UserPoolManager
from ..models.economics import EconomicsModel
from ..models.topup import TopUpModel
from ..models.safety import validate_state, validate_action, sanitize_obs
from ..models.utils import compute_price_bounds, safe_clip

logger = logging.getLogger("oran.env")


class OranSlicingEnv(gym.Env):
    """O-RAN 1-Cell Slicing + Pricing environment.

    See module docstring for full specification references.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict[str, Any], seed: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

        # --- Time ---
        self.episode_len = cfg.get("time", {}).get("episode_len_months", 50)
        self.K = cfg.get("time", {}).get("inner_loop_K", 30)

        # --- Models ---
        self.radio_cfg = RadioConfig.from_config(cfg)
        self.radio = RadioModel(self.radio_cfg)
        self.demand_cfg = DemandConfig.from_config(cfg)
        self.demand = DemandModel(self.demand_cfg)
        self.market = MarketModel(cfg)
        self.economics = EconomicsModel(cfg)
        self.topup = TopUpModel(cfg)                              # [FIX M2]

        # --- Price bounds (§6.2) ---
        self.price_bounds = compute_price_bounds(cfg)

        # --- Action space (continuous [-1, 1]^3 → mapped) ---
        ac = cfg.get("action", {})
        self.rho_min = ac.get("rho_urllc_min", 0.05)
        self.rho_max = ac.get("rho_urllc_max", 0.95)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

        # --- Observation space (§16) ---
        obs_cfg = cfg.get("observation", {})
        self._obs_clip_min = obs_cfg.get("clip_min", -10.0)
        self._obs_clip_max = obs_cfg.get("clip_max", 10.0)
        self.observation_space = spaces.Box(
            low=self._obs_clip_min, high=self._obs_clip_max,
            shape=(16,), dtype=np.float32,
        )

        # --- State ---
        self.pool: Optional[UserPoolManager] = None
        self.month: int = 0
        self._last_info: Dict[str, Any] = {}

        # --- Previous step values for observation ---
        self._prev_fee_eMBB: float = 0.0
        self._prev_fee_URLLC: float = 0.0
        self._prev_profit: float = 0.0
        self._prev_revenue: float = 0.0
        self._prev_cost: float = 0.0

        # --- Phase 3: Warm-up (M8) ---
        warmup = cfg.get("warmup", {})
        self._warmup_months: int = warmup.get("months", 10)
        self._warmup_price_limit: float = warmup.get("price_limit_frac", 0.10)
        self._warmup_early_limit: float = warmup.get("early_limit_frac", 0.05)
        self._warmup_early_months: int = warmup.get("early_months", 3)

        # --- Phase 3: Churned recycling (D9) ---
        mc = cfg.get("market", {})
        self._churn_recycle_cooldown: int = mc.get("churn_recycle_cooldown", 3)

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.pool = UserPoolManager.from_config(self.cfg, rng=self._rng)
        self.month = 0

        # Set initial fees to midpoint of bounds
        self._prev_fee_eMBB = (
            self.price_bounds["eMBB"]["F_min"]
            + self.price_bounds["eMBB"]["F_max"]
        ) / 2.0
        self._prev_fee_URLLC = (
            self.price_bounds["URLLC"]["F_min"]
            + self.price_bounds["URLLC"]["F_max"]
        ) / 2.0
        self._prev_profit = 0.0
        self._prev_revenue = 0.0
        self._prev_cost = 0.0

        self._last_info = {}
        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.month += 1
        info = self._run_month(action)
        self._last_info = info

        obs = self._build_obs()
        reward = info.get("reward", 0.0)
        terminated = (self.month >= self.episode_len)
        truncated = False

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # FIX M5: Action mapping per §6.1
    # ------------------------------------------------------------------
    def _map_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        """Map [-1,1]^3 to environment parameters per §6.1.

        a[0] → F_eMBB    ∈ [F_min_eMBB, F_max_eMBB]
        a[1] → F_URLLC   ∈ [F_min_URLLC, F_max_URLLC]
        a[2] → rho_URLLC ∈ [rho_min, rho_max]

        Returns (fee_eMBB, fee_URLLC, rho_URLLC).
        """
        a = np.clip(action, -1.0, 1.0)

        pb_e = self.price_bounds["eMBB"]
        fee_eMBB = pb_e["F_min"] + (a[0] + 1.0) / 2.0 * (pb_e["F_max"] - pb_e["F_min"])

        pb_u = self.price_bounds["URLLC"]
        fee_URLLC = pb_u["F_min"] + (a[1] + 1.0) / 2.0 * (pb_u["F_max"] - pb_u["F_min"])

        rho_URLLC = self.rho_min + (a[2] + 1.0) / 2.0 * (self.rho_max - self.rho_min)

        return float(fee_eMBB), float(fee_URLLC), float(rho_URLLC)

    def _clamp_fee(self, fee: float, prev_fee: float, limit_frac: float) -> float:
        """M8: Clamp fee change to ±limit_frac of previous fee during warm-up."""
        if prev_fee <= 0:
            return fee
        max_delta = prev_fee * limit_frac
        return float(np.clip(fee, prev_fee - max_delta, prev_fee + max_delta))

    # ------------------------------------------------------------------
    # Main monthly step — §17 compliant
    # ------------------------------------------------------------------
    def _run_month(self, action: np.ndarray) -> Dict[str, Any]:
        """Execute one monthly step per §17 step order.

        Step order:
          1) Map action
          2) Join (inactive → active)        [FIX M1]
          3) Generate demand
          4) Inner loop k=1..K               [FIX M3: per-step SLA tracking]
                                             [FIX M4: per-slice rho_util]
          5) Aggregate monthly KPIs
          6) Top-up + disconfirmation        [FIX M2]
          7) Churn (active → churned)        [FIX M1]
          8) Profit/reward
        """

        # ══════════════════════════════════════════════════════════════
        # 1) Map action  [FIX M5: §6.1 order]
        # ══════════════════════════════════════════════════════════════
        fee_eMBB, fee_URLLC, rho_URLLC = self._map_action(action)

        # M8: Warm-up price clamping — limit fee changes in early months
        if self.month <= self._warmup_months:
            limit = self._warmup_early_limit if self.month <= self._warmup_early_months \
                else self._warmup_price_limit
            fee_eMBB = self._clamp_fee(fee_eMBB, self._prev_fee_eMBB, limit)
            fee_URLLC = self._clamp_fee(fee_URLLC, self._prev_fee_URLLC, limit)

        # D10: Validate action
        action_valid, action_violations = validate_action(action)

        # Reset monthly fields for all active users
        self.pool.reset_monthly_fields()

        # ══════════════════════════════════════════════════════════════
        # 2) Join: sample joins, move inactive → active  [FIX M1]
        #    §17 step 2: Join BEFORE demand generation
        # ══════════════════════════════════════════════════════════════
        n_avail_eMBB = self.pool.inactive_count("eMBB")
        n_avail_URLLC = self.pool.inactive_count("URLLC")
        n_join_eMBB = self.market.sample_joins("eMBB", n_avail_eMBB, rng=self._rng)
        n_join_URLLC = self.market.sample_joins("URLLC", n_avail_URLLC, rng=self._rng)

        inactive_eMBB = [
            u for u in self.pool.inactive_pool.values() if u.slice == "eMBB"
        ]
        inactive_URLLC = [
            u for u in self.pool.inactive_pool.values() if u.slice == "URLLC"
        ]
        join_ids_eMBB = [u.user_id for u in inactive_eMBB[:n_join_eMBB]]
        join_ids_URLLC = [u.user_id for u in inactive_URLLC[:n_join_URLLC]]
        self.pool.join(join_ids_eMBB)
        self.pool.join(join_ids_URLLC)

        # ══════════════════════════════════════════════════════════════
        # 3) Generate monthly demand for all active users (including new joins)
        # ══════════════════════════════════════════════════════════════
        users_eMBB = self.pool.get_active_users("eMBB")
        users_URLLC = self.pool.get_active_users("URLLC")
        n_eMBB = len(users_eMBB)
        n_URLLC = len(users_URLLC)

        if n_eMBB > 0:
            seg_eMBB = np.array([u.segment for u in users_eMBB])
            D_eMBB = self.demand.sample_demand(
                "eMBB", n_eMBB, segments=seg_eMBB, rng=self._rng,
            )
            for i, u in enumerate(users_eMBB):
                u.D_u = D_eMBB[i]

        if n_URLLC > 0:
            seg_URLLC = np.array([u.segment for u in users_URLLC])
            D_URLLC = self.demand.sample_demand(
                "URLLC", n_URLLC, segments=seg_URLLC, rng=self._rng,
            )
            for i, u in enumerate(users_URLLC):
                u.D_u = D_URLLC[i]

        # ══════════════════════════════════════════════════════════════
        # 4) Inner loop k=1..K
        #    [FIX M7]: T_exp from plan v_cap (set by reset_monthly)
        #    [FIX M4]: Track rho_util per slice separately
        #    [FIX M3]: Track per-step avg throughput for SLA violation
        # ══════════════════════════════════════════════════════════════
        T_exp_eMBB = np.array(
            [u.T_exp for u in users_eMBB], dtype=np.float64,
        ) if n_eMBB > 0 else np.array([], dtype=np.float64)

        T_exp_URLLC = np.array(
            [u.T_exp for u in users_URLLC], dtype=np.float64,
        ) if n_URLLC > 0 else np.array([], dtype=np.float64)

        T_act_sum_eMBB = np.zeros(n_eMBB)
        T_act_sum_URLLC = np.zeros(n_URLLC)

        # [FIX M4] Per-slice accumulators
        rho_util_accum_eMBB = 0.0
        rho_util_accum_URLLC = 0.0

        # [FIX M3] Per-step slice-average throughput arrays for SLA
        step_avg_T_eMBB = np.zeros(self.K)
        step_avg_T_URLLC = np.zeros(self.K)

        for k in range(self.K):
            result = self.radio.inner_step(
                N_active_eMBB=n_eMBB,
                N_active_URLLC=n_URLLC,
                rho_URLLC=rho_URLLC,
                T_exp_users_eMBB=T_exp_eMBB,
                T_exp_users_URLLC=T_exp_URLLC,
            )

            if n_eMBB > 0:
                T_act_sum_eMBB += result["T_act_eMBB"]
                step_avg_T_eMBB[k] = result["avg_T_act_eMBB"]    # [FIX M3]
            if n_URLLC > 0:
                T_act_sum_URLLC += result["T_act_URLLC"]
                step_avg_T_URLLC[k] = result["avg_T_act_URLLC"]  # [FIX M3]

            # [FIX M4] Per-slice accumulation
            rho_util_accum_eMBB += result["rho_util_eMBB"]
            rho_util_accum_URLLC += result["rho_util_URLLC"]

        # ══════════════════════════════════════════════════════════════
        # 5) Aggregate monthly KPIs
        # ══════════════════════════════════════════════════════════════
        K = max(self.K, 1)

        # [FIX M4] Separate per-slice mean utilization
        mean_rho_util_eMBB = rho_util_accum_eMBB / K
        mean_rho_util_URLLC = rho_util_accum_URLLC / K

        if n_eMBB > 0:
            T_act_avg_eMBB = T_act_sum_eMBB / K
            for i, u in enumerate(users_eMBB):
                u.T_act_avg = T_act_avg_eMBB[i]
            avg_T_eMBB = float(T_act_avg_eMBB.mean())
        else:
            avg_T_eMBB = 0.0

        if n_URLLC > 0:
            T_act_avg_URLLC = T_act_sum_URLLC / K
            for i, u in enumerate(users_URLLC):
                u.T_act_avg = T_act_avg_URLLC[i]
            avg_T_URLLC = float(T_act_avg_URLLC.mean())
        else:
            avg_T_URLLC = 0.0

        # [FIX M3] SLA violation = fraction of inner steps below SLO
        # §12.1: V_s = (#violating inner steps) / K
        V_rate_eMBB = self.economics.sla.compute_violation_rate(
            step_avg_T_eMBB, "eMBB",
        )
        V_rate_URLLC = self.economics.sla.compute_violation_rate(
            step_avg_T_URLLC, "URLLC",
        )

        # ══════════════════════════════════════════════════════════════
        # 6) Top-up + throttle + disconfirmation  [FIX M2]
        #    §17 step 6: "Optional: top-up (max 1) and update T_exp/Δ_disc"
        # ══════════════════════════════════════════════════════════════
        n_topups_eMBB = 0
        n_topups_URLLC = 0

        if self.topup.enabled:
            for u in users_eMBB:
                # Check if user exceeded data cap
                if u.D_u > u.Q_gb:
                    # Apply throttle: reduce T_exp to v_cap
                    u.T_exp = self.topup.apply_throttle(
                        u.D_u, u.Q_gb, u.T_exp, u.v_cap_mbps,
                    )
                    # Top-up decision
                    delta_util = u.T_act_avg - u.v_cap_mbps  # utility loss
                    if not u.topup_flag and self.topup.decide_topup(
                        delta_util, u.w_price, rng=self._rng,
                    ):
                        u.topup_flag = True
                        u.T_exp = u.v_cap_mbps  # restore after top-up
                        u.Q_gb += self.topup.data_gb  # extend cap
                        n_topups_eMBB += 1

            for u in users_URLLC:
                if u.D_u > u.Q_gb:
                    u.T_exp = self.topup.apply_throttle(
                        u.D_u, u.Q_gb, u.T_exp, u.v_cap_mbps,
                    )
                    delta_util = u.T_act_avg - u.v_cap_mbps
                    if not u.topup_flag and self.topup.decide_topup(
                        delta_util, u.w_price, rng=self._rng,
                    ):
                        u.topup_flag = True
                        u.T_exp = u.v_cap_mbps
                        u.Q_gb += self.topup.data_gb
                        n_topups_URLLC += 1

        # 6b. Disconfirmation update (after top-up adjusts T_exp)
        self.market.update_disconfirmation(users_eMBB)
        self.market.update_disconfirmation(users_URLLC)

        # ══════════════════════════════════════════════════════════════
        # 7) Churn: sample churn, move active → churned  [FIX M1]
        #    §17 step 7: Churn AFTER top-up/disconfirmation
        # ══════════════════════════════════════════════════════════════
        churn_ids_eMBB = self.market.sample_churns(
            users_eMBB, fee_eMBB, rng=self._rng,
        )
        churn_ids_URLLC = self.market.sample_churns(
            users_URLLC, fee_URLLC, rng=self._rng,
        )
        n_churn_eMBB = len(churn_ids_eMBB)
        n_churn_URLLC = len(churn_ids_URLLC)
        self.pool.churn(churn_ids_eMBB)
        self.pool.churn(churn_ids_URLLC)

        # Post-churn active counts
        n_post_eMBB = self.pool.active_count("eMBB")
        n_post_URLLC = self.pool.active_count("URLLC")

        # D9: Recycle churned users back to inactive pool after cooldown
        n_recycled = self.pool.recycle_churned(
            cooldown_months=self._churn_recycle_cooldown,
        )

        # ══════════════════════════════════════════════════════════════
        # 8) Profit & reward
        #    [FIX M4]: Use total cell utilization for energy cost
        #    [D10]: Safety validation
        #    [M9]: Shaped reward with retention, efficiency, churn, volatility
        # ══════════════════════════════════════════════════════════════
        # For resource cost: use weighted-average utilization
        # For energy: use total PRB utilization (sum of slice shares)
        rho_eMBB_share = 1.0 - rho_URLLC
        total_cell_util = (
            mean_rho_util_eMBB * rho_eMBB_share
            + mean_rho_util_URLLC * rho_URLLC
        )
        mean_rho_util_combined = (mean_rho_util_eMBB + mean_rho_util_URLLC) / 2.0

        profit_result = self.economics.compute_profit(
            fees={"eMBB": fee_eMBB, "URLLC": fee_URLLC},
            N_active={"eMBB": n_eMBB, "URLLC": n_URLLC},
            n_topups={"eMBB": n_topups_eMBB, "URLLC": n_topups_URLLC},
            topup_price=self.topup.price_krw,
            V_rates={"eMBB": V_rate_eMBB, "URLLC": V_rate_URLLC},
            mean_rho_util=mean_rho_util_combined,
            avg_load=total_cell_util,
        )

        profit = profit_result["profit"]

        # D10: Safety validation and penalty
        safety_penalty, safety_violations = validate_state(
            N_active={"eMBB": n_eMBB, "URLLC": n_URLLC},
            rho_util={"eMBB": mean_rho_util_eMBB, "URLLC": mean_rho_util_URLLC},
            fees={"eMBB": fee_eMBB, "URLLC": fee_URLLC},
            rho_urllc=rho_URLLC,
        )

        # M9: Compute reward shaping terms
        retention_bonus = self.economics.compute_retention_bonus(
            N_active={"eMBB": n_post_eMBB, "URLLC": n_post_URLLC},
        )
        efficiency_bonus = self.economics.compute_efficiency_bonus(
            rho_util={"eMBB": mean_rho_util_eMBB, "URLLC": mean_rho_util_URLLC},
        )
        churn_penalty = self.economics.compute_churn_penalty(
            n_churns={"eMBB": n_churn_eMBB, "URLLC": n_churn_URLLC},
            N_active={"eMBB": n_eMBB, "URLLC": n_URLLC},
        )
        volatility_penalty = self.economics.compute_volatility_penalty(
            fee_delta={
                "eMBB": fee_eMBB - self._prev_fee_eMBB,
                "URLLC": fee_URLLC - self._prev_fee_URLLC,
            },
        )

        reward = self.economics.compute_reward(
            profit, penalty=safety_penalty,
            shaping={
                "retention_bonus": retention_bonus,
                "efficiency_bonus": efficiency_bonus,
                "churn_penalty": churn_penalty,
                "volatility_penalty": volatility_penalty,
            },
        )

        # Update state for next obs
        self._prev_fee_eMBB = fee_eMBB
        self._prev_fee_URLLC = fee_URLLC
        self._prev_profit = profit
        self._prev_revenue = profit_result["revenue"]
        self._prev_cost = profit_result["cost_total"]

        # ══════════════════════════════════════════════════════════════
        # 9) Info dict (flat keys)
        # ══════════════════════════════════════════════════════════════
        info = {
            "month": self.month,
            "fee_eMBB": fee_eMBB,
            "fee_URLLC": fee_URLLC,
            "rho_URLLC": rho_URLLC,
            "N_active_eMBB": n_eMBB,
            "N_active_URLLC": n_URLLC,
            "N_post_churn_eMBB": n_post_eMBB,
            "N_post_churn_URLLC": n_post_URLLC,
            "joins_eMBB": len(join_ids_eMBB),
            "joins_URLLC": len(join_ids_URLLC),
            "churns_eMBB": n_churn_eMBB,
            "churns_URLLC": n_churn_URLLC,
            "rho_util_eMBB": mean_rho_util_eMBB,        # [FIX M4]
            "rho_util_URLLC": mean_rho_util_URLLC,      # [FIX M4]
            "avg_T_eMBB": avg_T_eMBB,
            "avg_T_URLLC": avg_T_URLLC,
            "V_rate_eMBB": V_rate_eMBB,                  # [FIX M3]
            "V_rate_URLLC": V_rate_URLLC,                # [FIX M3]
            "topups_eMBB": n_topups_eMBB,                # [FIX M2]
            "topups_URLLC": n_topups_URLLC,              # [FIX M2]
            "recycled": n_recycled,                      # [D9]
            "retention_bonus": retention_bonus,           # [M9]
            "efficiency_bonus": efficiency_bonus,         # [M9]
            "churn_penalty_val": churn_penalty,           # [M9]
            "volatility_penalty": volatility_penalty,     # [M9]
            "safety_penalty": safety_penalty,             # [D10]
            "revenue": profit_result["revenue"],
            "cost_total": profit_result["cost_total"],
            "profit": profit,
            "reward": reward,
        }

        self.pool.assert_invariants()
        return info

    # ------------------------------------------------------------------
    # Observation builder (§16)
    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        """Build observation vector (§16). Shape = (16,)."""
        profit_scale = max(abs(self.economics.profit_scale), 1.0)

        if self.pool is None:
            return np.zeros(16, dtype=np.float32)

        n_eMBB = self.pool.active_count("eMBB")
        n_URLLC = self.pool.active_count("URLLC")

        obs = np.array([
            self.month / max(self.episode_len, 1),           # 0: normalized month
            n_eMBB / 200.0,                                  # 1: N_active_eMBB norm
            n_URLLC / 100.0,                                 # 2: N_active_URLLC norm
            self._last_info.get("joins_eMBB", 0) / 25.0,    # 3: joins eMBB norm
            self._last_info.get("joins_URLLC", 0) / 10.0,   # 4: joins URLLC norm
            self._last_info.get("churns_eMBB", 0) / 25.0,   # 5: churns eMBB norm
            self._last_info.get("churns_URLLC", 0) / 10.0,  # 6: churns URLLC norm
            self._last_info.get("rho_URLLC", 0.5),           # 7: rho_URLLC
            self._prev_fee_eMBB / 100000.0,                  # 8: fee_eMBB norm
            self._prev_fee_URLLC / 100000.0,                 # 9: fee_URLLC norm
            self._last_info.get("rho_util_eMBB", 0.0),       # 10: rho_util eMBB [M4]
            self._last_info.get("avg_T_eMBB", 0.0) / 100.0,  # 11: avg T eMBB
            self._last_info.get("avg_T_URLLC", 0.0) / 100.0, # 12: avg T URLLC
            self._last_info.get("V_rate_eMBB", 0.0),         # 13: V_rate eMBB
            self._last_info.get("V_rate_URLLC", 0.0),        # 14: V_rate URLLC
            self._prev_profit / profit_scale,                 # 15: norm profit
        ], dtype=np.float32)

        # D10: Use safety module's sanitize_obs for NaN/Inf protection
        obs = sanitize_obs(obs, self._obs_clip_min, self._obs_clip_max)
        return obs
