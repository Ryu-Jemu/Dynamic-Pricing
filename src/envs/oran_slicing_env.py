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
"""
O-RAN 1-Cell Slicing + Pricing Gymnasium Environment.

Action space (Box, dim=3):
  a[0] → rho_URLLC  ∈ [rho_min, rho_max]
  a[1] → fee_eMBB   ∈ [F_min, F_max]
  a[2] → fee_URLLC  ∈ [F_min, F_max]

Observation space (Box, dim=16):
  [N_eMBB_norm, N_URLLC_norm, rho_URLLC, fee_eMBB_norm, fee_URLLC_norm,
   revenue_norm, cost_norm, V_eMBB, V_URLLC, T_avg_eMBB_norm,
   T_avg_URLLC_norm, rho_util_eMBB, rho_util_URLLC, month_sin, month_cos,
   profit_norm]

Step 2 changes:
  - obs[15]: Was hardcoded as `profit / profit_scale` (effectively tanh-
    scale normalization). Now uses EconomicsModel._compute_reward_raw()
    so the observation normalizer is consistent with the reward function.
  - Reward: Delegates to EconomicsModel.compute_reward() which dispatches
    based on config-driven reward_type ∈ {tanh, linear, log}.
  - EconomicsModel instantiated from config, replacing inline economics.

  Rationale (obs[15] fix):
    The original obs[15] = profit / profit_scale was designed assuming
    tanh reward where |r| ≤ 1, so dividing by scale kept obs in a
    reasonable range. With log reward, the raw reward can exceed 1.0,
    so using the same _compute_reward_raw ensures the observation
    accurately reflects what the agent receives as reward signal.
    This prevents information mismatch between obs and reward.

References:
  [SB3_TIPS]  https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
  [SAC_ORIG]  Haarnoja et al., ICML 2018
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..models.utils import (
    compute_price_bounds, safe_clip, safe_divide, load_config
)
from ..models.radio import RadioConfig, RadioModel
from ..models.demand import DemandConfig, DemandModel
from ..models.pools import UserPoolManager
from ..models.market import MarketModel
from ..models.economics import EconomicsModel
from ..models.market import MarketModel

logger = logging.getLogger("oran.env")


class OranSlicingEnv(gym.Env):
    """O-RAN 1-Cell Network Slicing + Pricing environment.

    Step 2 integration:
      - EconomicsModel handles reward computation with configurable
        reward_type (tanh / linear / log).
      - obs[15] uses EconomicsModel._compute_reward_raw for consistency.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Dict[str, Any],
                 seed: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

        # Sub-models
        self.radio_cfg = RadioConfig.from_config(cfg)
        self.radio = RadioModel(self.radio_cfg)
        self.demand_cfg = DemandConfig.from_config(cfg)
        self.demand = DemandModel(self.demand_cfg)

        # ── Step 2: EconomicsModel (replaces inline economics) ────
        self.economics = EconomicsModel(cfg)

        # ── Step 3: MarketModel (churn/join) ──────────────────────
        self.market = MarketModel(cfg)

        # Price bounds
        self.price_bounds = compute_price_bounds(cfg)

        # Action space: [rho_URLLC, fee_eMBB, fee_URLLC] mapped from [-1,1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation space: 16-dim
        obs_clip = cfg.get("observation", {}).get("clip_max", 10.0)
        self.observation_space = spaces.Box(
            low=-obs_clip, high=obs_clip, shape=(16,), dtype=np.float32
        )

        # Convenience aliases from economics
        self.profit_scale = self.economics.profit_scale
        self.lambda_penalty = self.economics.lambda_penalty
        self.reward_clip = self.economics.reward_clip
        self.reward_type = self.economics.reward_type

        # Energy (delegated to economics.energy)
        # Radio
        self.unit_cost_prb = self.economics.unit_cost_prb

        # SLA (delegated to economics.sla)

        # Topup
        self.topup_cfg = cfg.get("topup", {})

        # Episode config
        time_cfg = cfg.get("time", {})
        self.episode_len = time_cfg.get("episode_len_months", 50)
        self.inner_K = time_cfg.get("inner_loop_K", 30)

        # Normalization constants
        pop = cfg.get("population", {})
        self.N_max_eMBB = pop.get("N0_eMBB", 120) * 3.0
        self.N_max_URLLC = pop.get("N0_URLLC", 30) * 3.0
        self.T_norm = self.radio_cfg.T_cell_cap_mbps

        # State
        self.month = 0
        self.pool: Optional[UserPoolManager] = None

        # Monthly tracking
        self._monthly_info: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────
    # Action mapping
    # ──────────────────────────────────────────────────────────

    def _map_action(self, action: np.ndarray) -> Dict[str, float]:
        """Map [-1, 1]^3 → physical action values."""
        a = np.clip(action, -1.0, 1.0).astype(np.float64)

        rho_min = self.cfg["action"]["rho_urllc_min"]
        rho_max = self.cfg["action"]["rho_urllc_max"]
        rho_urllc = rho_min + (a[0] + 1.0) / 2.0 * (rho_max - rho_min)

        b_embb = self.price_bounds["eMBB"]
        fee_eMBB = (b_embb["F_min"] + b_embb["F_max"]) / 2.0 \
            + a[1] * (b_embb["F_max"] - b_embb["F_min"]) / 2.0

        b_urllc = self.price_bounds["URLLC"]
        fee_URLLC = (b_urllc["F_min"] + b_urllc["F_max"]) / 2.0 \
            + a[2] * (b_urllc["F_max"] - b_urllc["F_min"]) / 2.0

        return {
            "rho_URLLC": float(np.clip(rho_urllc, rho_min, rho_max)),
            "fee_eMBB": float(np.clip(fee_eMBB, b_embb["F_min"], b_embb["F_max"])),
            "fee_URLLC": float(np.clip(fee_URLLC, b_urllc["F_min"], b_urllc["F_max"])),
        }

    # ──────────────────────────────────────────────────────────
    # Observation  — STEP 2 FIX: obs[15]
    # ──────────────────────────────────────────────────────────

    def _get_obs(self, action_dict: Optional[Dict[str, float]] = None,
                 info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Build 16-dim observation vector.

        Step 2 change — obs[15]:
          BEFORE: info.get("profit", 0) / max(self.profit_scale, 1)
            This was implicitly tied to tanh normalization and did not
            reflect the actual reward signal for log/linear modes.

          AFTER: EconomicsModel._compute_reward_raw(profit, reward_type, ...)
            Uses the same transformation as the reward function, ensuring
            the agent's observation of "normalized profit" matches the
            reward it will actually receive (before penalty subtraction).
        """
        if action_dict is None:
            action_dict = {"rho_URLLC": 0.2, "fee_eMBB": 69000, "fee_URLLC": 55000}
        if info is None:
            info = {}

        n_embb = self.pool.active_count("eMBB") if self.pool else 0
        n_urllc = self.pool.active_count("URLLC") if self.pool else 0

        b_embb = self.price_bounds["eMBB"]
        b_urllc = self.price_bounds["URLLC"]

        # ── Step 2 FIX: obs[15] uses _compute_reward_raw ──────────
        # Before: profit / profit_scale  (hardcoded, tanh-scale)
        # After:  _compute_reward_raw(profit, reward_type, scale, clip)
        # This ensures obs[15] is consistent with the actual reward
        # signal for any reward_type.
        profit = info.get("profit", 0.0)
        profit_norm = EconomicsModel._compute_reward_raw(
            profit,
            self.reward_type,
            self.profit_scale,
            self.reward_clip,
        )

        obs = np.array([
            n_embb / max(self.N_max_eMBB, 1),                          # 0
            n_urllc / max(self.N_max_URLLC, 1),                        # 1
            action_dict["rho_URLLC"],                                   # 2
            (action_dict["fee_eMBB"] - b_embb["F_min"])
            / max(b_embb["F_max"] - b_embb["F_min"], 1),              # 3
            (action_dict["fee_URLLC"] - b_urllc["F_min"])
            / max(b_urllc["F_max"] - b_urllc["F_min"], 1),            # 4
            info.get("revenue", 0) / max(self.profit_scale, 1),        # 5
            info.get("cost_total", 0) / max(self.profit_scale, 1),     # 6
            info.get("V_rate_eMBB", 0),                                # 7
            info.get("V_rate_URLLC", 0),                               # 8
            info.get("avg_T_eMBB", 0) / max(self.T_norm, 1),          # 9
            info.get("avg_T_URLLC", 0) / max(self.T_norm, 1),         # 10
            info.get("rho_util_eMBB", 0),                              # 11
            info.get("rho_util_URLLC", 0),                             # 12
            np.sin(2 * np.pi * self.month / 12),                       # 13
            np.cos(2 * np.pi * self.month / 12),                       # 14
            profit_norm,                                                # 15  ← STEP 2 FIX
        ], dtype=np.float64)

        clip_v = self.cfg.get("observation", {}).get("clip_max", 10.0)
        obs = safe_clip(obs, -clip_v, clip_v)
        return obs.astype(np.float32)

    # ──────────────────────────────────────────────────────────
    # Monthly step
    # ──────────────────────────────────────────────────────────

    def _run_month(self, action_dict: Dict[str, float]) -> Dict[str, Any]:
        """Execute one monthly step."""
        rho_URLLC = action_dict["rho_URLLC"]
        fee_eMBB = action_dict["fee_eMBB"]
        fee_URLLC = action_dict["fee_URLLC"]

        # 1. Get active users
        users_eMBB = self.pool.get_active_users("eMBB")
        users_URLLC = self.pool.get_active_users("URLLC")
        n_eMBB = len(users_eMBB)
        n_URLLC = len(users_URLLC)

        # 2. Sample demand
        segs_eMBB = np.array([u.segment for u in users_eMBB]) if n_eMBB > 0 else np.array([])
        segs_URLLC = np.array([u.segment for u in users_URLLC]) if n_URLLC > 0 else np.array([])

        demand_eMBB = self.demand.sample_demand("eMBB", n_eMBB, segs_eMBB, self._rng)
        demand_URLLC = self.demand.sample_demand("URLLC", n_URLLC, segs_URLLC, self._rng)

        for i, u in enumerate(users_eMBB):
            u.D_u = demand_eMBB[i]
        for i, u in enumerate(users_URLLC):
            u.D_u = demand_URLLC[i]

        # 3. Inner loop — radio model
        T_exp_eMBB = np.array([100.0] * n_eMBB) if n_eMBB > 0 else np.array([])
        T_exp_URLLC = np.array([100.0] * n_URLLC) if n_URLLC > 0 else np.array([])

        T_accum_eMBB = np.zeros(n_eMBB)
        T_accum_URLLC = np.zeros(n_URLLC)

        rho_util_accum = 0.0

        for k in range(self.inner_K):
            result = self.radio.inner_step(
                n_eMBB, n_URLLC, rho_URLLC, T_exp_eMBB, T_exp_URLLC
            )
            T_accum_eMBB += result["T_act_eMBB"] if n_eMBB > 0 else 0
            T_accum_URLLC += result["T_act_URLLC"] if n_URLLC > 0 else 0
            rho_util_accum += (result["rho_util_eMBB"] + result["rho_util_URLLC"]) / 2.0

        mean_rho_util = rho_util_accum / max(self.inner_K, 1)

        # Per-user average throughput over inner loop
        if n_eMBB > 0:
            avg_T_per_user_eMBB = T_accum_eMBB / self.inner_K
            for i, u in enumerate(users_eMBB):
                u.T_act_avg = avg_T_per_user_eMBB[i]
        if n_URLLC > 0:
            avg_T_per_user_URLLC = T_accum_URLLC / self.inner_K
            for i, u in enumerate(users_URLLC):
                u.T_act_avg = avg_T_per_user_URLLC[i]

        avg_T_eMBB = float(np.mean(avg_T_per_user_eMBB)) if n_eMBB > 0 else 0.0
        avg_T_URLLC = float(np.mean(avg_T_per_user_URLLC)) if n_URLLC > 0 else 0.0

        # 4. SLA violations
        V_count_eMBB = sum(1 for u in users_eMBB if u.T_act_avg < self.economics.sla.SLO_T["eMBB"])
        V_count_URLLC = sum(1 for u in users_URLLC if u.T_act_avg < self.economics.sla.SLO_T["URLLC"])
        V_rate_eMBB = V_count_eMBB / max(n_eMBB, 1)
        V_rate_URLLC = V_count_URLLC / max(n_URLLC, 1)

        # 5. Revenue & cost via EconomicsModel
        topup_price = self.topup_cfg.get("price_krw", 11000)
        econ_result = self.economics.compute_profit(
            fees={"eMBB": fee_eMBB, "URLLC": fee_URLLC},
            N_active={"eMBB": n_eMBB, "URLLC": n_URLLC},
            n_topups={"eMBB": 0, "URLLC": 0},
            topup_price=topup_price,
            V_rates={"eMBB": V_rate_eMBB, "URLLC": V_rate_URLLC},
            mean_rho_util=mean_rho_util,
            avg_load=mean_rho_util,
        )

        profit = econ_result["profit"]
        penalty = V_rate_eMBB + V_rate_URLLC

        # 6. Churn/Join via MarketModel (Step 3 integration)
        # 6a. Update disconfirmation for all active users
        self.market.update_disconfirmation(users_eMBB)
        self.market.update_disconfirmation(users_URLLC)

        # 6b. Sample churns
        churn_ids_eMBB = self.market.sample_churns(
            users_eMBB, fee_eMBB, self._rng
        )
        churn_ids_URLLC = self.market.sample_churns(
            users_URLLC, fee_URLLC, self._rng
        )
        n_churn_eMBB = len(churn_ids_eMBB)
        n_churn_URLLC = len(churn_ids_URLLC)
        self.pool.churn(churn_ids_eMBB)
        self.pool.churn(churn_ids_URLLC)

        # 6c. Sample joins from inactive pool
        n_avail_eMBB = self.pool.inactive_count("eMBB")
        n_avail_URLLC = self.pool.inactive_count("URLLC")
        n_join_eMBB = self.market.sample_joins("eMBB", n_avail_eMBB, self._rng)
        n_join_URLLC = self.market.sample_joins("URLLC", n_avail_URLLC, self._rng)

        inactive_eMBB = [uid for uid, u in self.pool.inactive_pool.items()
                         if u.slice == "eMBB"]
        inactive_URLLC = [uid for uid, u in self.pool.inactive_pool.items()
                          if u.slice == "URLLC"]

        join_ids_eMBB = list(self._rng.choice(
            inactive_eMBB,
            size=min(n_join_eMBB, len(inactive_eMBB)),
            replace=False
        )) if inactive_eMBB and n_join_eMBB > 0 else []

        join_ids_URLLC = list(self._rng.choice(
            inactive_URLLC,
            size=min(n_join_URLLC, len(inactive_URLLC)),
            replace=False
        )) if inactive_URLLC and n_join_URLLC > 0 else []

        self.pool.join(join_ids_eMBB)
        self.pool.join(join_ids_URLLC)

        n_post_eMBB = self.pool.active_count("eMBB")
        n_post_URLLC = self.pool.active_count("URLLC")

        # Reset monthly user fields
        self.pool.reset_monthly_fields()

        info = {
            "month": self.month,
            "profit": profit,
            "revenue": econ_result["revenue"],
            "cost_total": econ_result["cost_total"],
            "cost_energy": econ_result["cost_energy"],
            "cost_sla_total": econ_result["cost_sla_total"],
            "cost_resource": econ_result["cost_resource"],
            "mean_rho_util": mean_rho_util,
            "rho_URLLC": rho_URLLC,
            "rho_util_eMBB": mean_rho_util,   # simplified
            "rho_util_URLLC": mean_rho_util,
            "penalty": penalty,
            "fee_eMBB": fee_eMBB,
            "fee_URLLC": fee_URLLC,
            "N_active_eMBB": n_eMBB,
            "N_active_URLLC": n_URLLC,
            "N_post_churn_eMBB": n_post_eMBB,
            "N_post_churn_URLLC": n_post_URLLC,
            "joins_eMBB": len(join_ids_eMBB),
            "joins_URLLC": len(join_ids_URLLC),
            "churns_eMBB": n_churn_eMBB,
            "churns_URLLC": n_churn_URLLC,
            "topups_eMBB": 0,
            "topups_URLLC": 0,
            "V_rate_eMBB": V_rate_eMBB,
            "V_rate_URLLC": V_rate_URLLC,
            "avg_T_eMBB": avg_T_eMBB,
            "avg_T_URLLC": avg_T_URLLC,
        }
        return info

    # ──────────────────────────────────────────────────────────
    # Gym interface
    # ──────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.month = 0
        self.pool = UserPoolManager.from_config(self.cfg, rng=self._rng)

        obs = self._get_obs()
        return obs, {"month": 0}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.month += 1

        action_dict = self._map_action(action)
        info = self._run_month(action_dict)

        # ── Step 2: Delegates to EconomicsModel.compute_reward ──
        reward = self.economics.compute_reward(
            info["profit"], info["penalty"]
        )
        info["reward"] = reward

        obs = self._get_obs(action_dict, info)

        terminated = (self.month >= self.episode_len)
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if self._monthly_info:
            logger.info("Month %d: profit=%.0f, reward=%.4f",
                        self.month,
                        self._monthly_info.get("profit", 0),
                        self._monthly_info.get("reward", 0))
