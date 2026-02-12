"""
O-RAN 1-Cell · 3-Part Tariff · 2 Slices · Online MDP  (§§3–11)

REVISION 7 — Improvements from training evaluation:
  [R4] Per-dimension action smoothing    [Dalal et al., NeurIPS 2018]
  [R5] Observation dim 20 → 22          [Dulac-Arnold et al., JMLR 2021]
  [R6] Population-aware reward term      [Mguni 2019; Zheng 2022]
  Prior revisions (v1–v6):
  [E4] Observation dim 16 → 20 (load factors, allowance utilisation)
  [E6] CLV-aware reward shaping (retention penalty)
  [E8] Stronger action smoothing (weight from config)
  [C3] Removed price_norm re-multiplication in churn/join logits
  [C4] Demand-price elasticity  [Nevo et al., Econometrica 2016]
  [C5] Action smoothing penalty  [Dulac-Arnold et al., JMLR 2021]

Gymnasium-compatible environment for SB3 SAC.

Action (5-D continuous, §4):
  a = [F_U, p_U^over, F_E, p_E^over, ρ_U]

Observation (22-D, §3.2) — see ``_build_obs`` for layout.
  [R5] Two new features (in addition to E4's four):
    20: over_rev_E / (p_over_E × N_E + ε)  — eMBB overage revenue rate
    21: (T − cycle_step) / T                — days remaining in cycle

Revenue per step (§5.2  — online accrual):
  BaseRev_t  = (F_U·N_U + F_E·N_E) / T
  OverRev_t  = Σ_s p_s^over · ΔOver_s(t)
  Revenue_t  = BaseRev_t + OverRev_t

Cost per step (§9):
  Cost_t = c_opex·N_active + c_energy·(L_U+L_E) + c_cac·N_join + SLA_penalty

Market (§10):
  Logit-based churn/join every step, expectation-only or stochastic mode.
  [R3] Curriculum mode: Phase 1 disables churn/join for stable learning.

Reward (§15 + §11b + §15c):
  [R6] reward = log_profit − smooth_penalty − retention_penalty + pop_bonus
  pop_bonus = β_pop × (N_active / N_total − target_ratio)
  [Mguni et al., AAMAS 2019; Zheng et al., Science Advances 2022]

References:
  [Haarnoja 2018]    SAC
  [Grubb AER 2009]   3-part tariff
  [Nevo 2016]        Broadband tariffs / usage coupling  (Econometrica)
  [TS 23.503]        Usage monitoring thresholds
  [Huang IoT-J 2020] URLLC priority / coexistence
  [Gupta JSR 2006]   CLV
  [ITU Teletraffic]  Poisson arrivals
  [Dulac-Arnold 2021] Challenges of Real-World RL
  [Ahn 2006]         Telecom churn determinants
  [Ng 1999]          Policy invariance under reward transformations
  [Dalal 2018]       Safe exploration — per-dimension constraints
  [Mguni 2019]       Population stability in economic simulations
  [Zheng 2022]       The AI Economist — population welfare terms
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .utils import load_config, sigmoid, safe_clip

logger = logging.getLogger("oran3pt.env")


class OranSlicingPricingEnv(gym.Env):
    """5G O-RAN single-cell pricing environment with 3-part tariff."""

    metadata = {"render_modes": []}

    # ── constructor ───────────────────────────────────────────────────
    def __init__(self, cfg: Dict[str, Any],
                 users_csv: Optional[str] = None,
                 seed: Optional[int] = None,
                 curriculum_phase: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._curriculum_phase = curriculum_phase

        # Time (§3.1)
        tc = cfg["time"]
        self.T: int = tc["steps_per_cycle"]
        self.n_cycles: int = tc["episode_cycles"]
        self.episode_len: int = self.T * self.n_cycles

        # Action bounds (§4)
        ac = cfg["action"]
        self._a_lo = np.array([
            ac["F_U_min"], ac["p_over_U_min"],
            ac["F_E_min"], ac["p_over_E_min"],
            ac["rho_U_min"],
        ], dtype=np.float64)
        self._a_hi = np.array([
            ac["F_U_max"], ac["p_over_U_max"],
            ac["F_E_max"], ac["p_over_E_max"],
            ac["rho_U_max"],
        ], dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

        # Observation (§3.2)  [R5] 22-D
        obs_cfg = cfg.get("observation", {})
        self._obs_dim: int = obs_cfg.get("dim", 22)
        self._obs_lo = obs_cfg.get("clip_min", -10.0)
        self._obs_hi = obs_cfg.get("clip_max", 10.0)
        self.observation_space = spaces.Box(
            self._obs_lo, self._obs_hi, shape=(self._obs_dim,), dtype=np.float32)

        # Tariff allowances (§5.1)
        tar = cfg["tariff"]
        self.Q_U: float = tar["Q_U_gb"]
        self.Q_E: float = tar["Q_E_gb"]

        # Radio (§7)
        rad = cfg["radio"]
        self.C_total: float = rad["C_total_gb_per_step"]
        self.kappa_U: float = rad.get("kappa_U", 1.2)

        # QoS (§8)
        qos = cfg["qos"]
        self.alpha_cong: float = qos["alpha_congestion"]
        self.lambda_U: float = qos["lambda_U"]
        self.lambda_E: float = qos["lambda_E"]

        # Cost (§9)
        cc = cfg["cost"]
        self.c_opex: float = cc["c_opex_per_user"]
        self.c_energy: float = cc["c_energy_per_gb"]
        self.c_cac: float = cc["c_cac_per_join"]

        # Market (§10)
        mc = cfg["market"]
        self.b0_churn: float = mc["beta0_churn"]
        self.bp_churn: float = mc["beta_p_churn"]
        self.bq_churn: float = mc["beta_q_churn"]
        self.bsw_churn: float = mc["beta_sw_churn"]
        self.b0_join: float = mc["beta0_join"]
        self.bp_join: float = mc["beta_p_join"]
        self.bq_join: float = mc["beta_q_join"]
        self.market_mode: str = mc.get("mode", "stochastic")
        self._price_norm: float = mc.get("price_norm", 120000.0)

        # Demand-price elasticity (§6b)  [C4]
        de = cfg.get("demand_elasticity", {})
        self._demand_elast_enabled: bool = de.get("enabled", False)
        self._eps_U: float = de.get("epsilon_U", 0.15)
        self._eps_E: float = de.get("epsilon_E", 0.30)
        self._pref_U: float = de.get("p_ref_U", 2500.0)
        self._pref_E: float = de.get("p_ref_E", 1500.0)
        self._demand_floor: float = de.get("floor", 0.5)

        # Action smoothing (§15b)  [C5][E8][R4]
        sm = cfg.get("action_smoothing", {})
        self._smooth_enabled: bool = sm.get("enabled", False)
        self._smooth_weight: float = sm.get("weight", 0.05)
        smooth_weights = sm.get("weights", None)
        if smooth_weights is not None and len(smooth_weights) == 5:
            self._smooth_weights = np.array(smooth_weights, dtype=np.float64)
        else:
            self._smooth_weights = np.full(5, self._smooth_weight, dtype=np.float64)

        # CLV reward shaping (§11b)  [E6][R2]
        clv_rs = cfg.get("clv_reward_shaping", {})
        self._clv_rs_enabled: bool = clv_rs.get("enabled", False)
        self._clv_rs_alpha: float = clv_rs.get("alpha_retention", 2.0)
        self._clv_rs_warmup: int = clv_rs.get("warmup_steps", 100)

        # [R6] Population-aware reward
        pop_rw = cfg.get("population_reward", {})
        self._pop_reward_enabled: bool = pop_rw.get("enabled", False)
        self._pop_beta: float = pop_rw.get("beta_pop", 0.1)
        self._pop_target_ratio: float = pop_rw.get("target_ratio", 0.4)

        # CLV (§11)
        clv_cfg = cfg.get("clv", {})
        self.clv_enabled: bool = clv_cfg.get("enabled", True)
        self.clv_horizon: int = clv_cfg.get("horizon_months", 24)
        self.clv_d: float = clv_cfg.get("discount_rate_monthly", 0.01)

        # Load users
        self._users_csv_path = users_csv
        self._load_users(users_csv)

        # Reward normalisation constants
        self._reward_scale = max(
            self._a_hi[0] * self.N_total / self.T, 1.0)

        # Per-episode state (initialised in reset)
        self.t: int = 0
        self._active_mask: np.ndarray = np.array([])
        self._cycle_usage_U: float = 0.0
        self._cycle_usage_E: float = 0.0
        self._prev_over_U: float = 0.0
        self._prev_over_E: float = 0.0
        self._prev_action: np.ndarray = np.zeros(5, dtype=np.float64)
        self._prev_revenue: float = 0.0
        self._prev_cost: float = 0.0
        self._prev_profit: float = 0.0
        self._prev_n_join: int = 0
        self._prev_n_churn: int = 0
        self._prev_pviol_U: float = 0.0
        self._prev_pviol_E: float = 0.0
        self._prev_L_U: float = 0.0
        self._prev_L_E: float = 0.0
        self._prev_C_U: float = 1.0
        self._prev_C_E: float = 1.0
        self._prev_over_rev_E: float = 0.0

    def _load_users(self, csv_path: Optional[str]) -> None:
        if csv_path is not None and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        else:
            from .gen_users import generate_users
            df = generate_users(self.cfg, seed=int(self._rng.integers(0, 2**31)))

        self._users = df
        self.N_total: int = len(df)
        self._slice_is_U = (df["slice"].values == "URLLC").astype(np.float64)
        self._slice_is_E = 1.0 - self._slice_is_U
        self._mu_u = df["mu_urllc"].values.astype(np.float64)
        self._sig_u = df["sigma_urllc"].values.astype(np.float64)
        self._mu_e = df["mu_embb"].values.astype(np.float64)
        self._sig_e = df["sigma_embb"].values.astype(np.float64)
        self._psens = df["price_sensitivity"].values.astype(np.float64)
        self._qsens = df["qos_sensitivity"].values.astype(np.float64)
        self._swcost = df["switching_cost"].values.astype(np.float64)
        self._clv_dr = df["clv_discount_rate"].values.astype(np.float64)
        self._init_active = df["is_active_init"].values.astype(bool)

    def set_curriculum_phase(self, phase: int) -> None:
        self._curriculum_phase = phase

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self._active_mask = self._init_active.copy()
        self._cycle_usage_U = 0.0
        self._cycle_usage_E = 0.0
        self._prev_over_U = 0.0
        self._prev_over_E = 0.0
        mid = (self._a_lo + self._a_hi) / 2.0
        self._prev_action = mid.copy()
        self._prev_revenue = 0.0
        self._prev_cost = 0.0
        self._prev_profit = 0.0
        self._prev_n_join = 0
        self._prev_n_churn = 0
        self._prev_pviol_U = 0.0
        self._prev_pviol_E = 0.0
        self._prev_L_U = 0.0
        self._prev_L_E = 0.0
        self._prev_C_U = self.C_total * 0.5 * self.kappa_U
        self._prev_C_E = self.C_total * 0.5
        self._prev_over_rev_E = 0.0
        return self._build_obs(), {}

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.t += 1
        info = self._run_step(action)
        obs = self._build_obs()
        reward = info["reward"]
        terminated = self.t >= self.episode_len
        return obs, reward, terminated, False, info

    def _map_action(self, raw: np.ndarray) -> np.ndarray:
        a = np.clip(raw, -1.0, 1.0).astype(np.float64)
        return self._a_lo + (a + 1.0) / 2.0 * (self._a_hi - self._a_lo)

    @property
    def _cycle_step(self) -> int:
        return (self.t - 1) % self.T

    @property
    def _is_cycle_start(self) -> bool:
        return self._cycle_step == 0

    def _run_step(self, raw_action: np.ndarray) -> Dict[str, Any]:
        a = self._map_action(raw_action)
        F_U, p_over_U, F_E, p_over_E, rho_U = a

        if self._is_cycle_start:
            self._cycle_usage_U = 0.0
            self._cycle_usage_E = 0.0
            self._prev_over_U = 0.0
            self._prev_over_E = 0.0

        if self._curriculum_phase == 1:
            n_join, n_churn = 0, 0
        else:
            n_join, n_churn = self._market_step(F_U, p_over_U, F_E, p_over_E)

        N_act = int(self._active_mask.sum())
        N_U = int((self._active_mask * self._slice_is_U).sum())
        N_E = N_act - N_U

        L_U, L_E = self._generate_traffic(p_over_U, p_over_E)

        self._cycle_usage_U += L_U
        self._cycle_usage_E += L_E

        C_U = rho_U * self.C_total * self.kappa_U
        C_E = (1.0 - rho_U) * self.C_total
        C_U = max(C_U, 1e-6)
        C_E = max(C_E, 1e-6)

        pviol_U = float(sigmoid(self.alpha_cong * (L_U / C_U - 1.0)))
        pviol_E = float(sigmoid(self.alpha_cong * (L_E / C_E - 1.0)))

        base_rev = (F_U * N_U + F_E * N_E) / self.T

        cur_over_U = max(self._cycle_usage_U - self.Q_U * N_U, 0.0)
        cur_over_E = max(self._cycle_usage_E - self.Q_E * N_E, 0.0)
        delta_over_U = max(cur_over_U - self._prev_over_U, 0.0)
        delta_over_E = max(cur_over_E - self._prev_over_E, 0.0)
        self._prev_over_U = cur_over_U
        self._prev_over_E = cur_over_E

        over_rev_U = p_over_U * delta_over_U
        over_rev_E = p_over_E * delta_over_E
        over_rev = over_rev_U + over_rev_E
        revenue = base_rev + over_rev

        cost_opex = self.c_opex * N_act
        cost_energy = self.c_energy * (L_U + L_E)
        cost_cac = self.c_cac * n_join
        sla_penalty = self.lambda_U * pviol_U + self.lambda_E * pviol_E
        cost_total = cost_opex + cost_energy + cost_cac + sla_penalty

        profit = revenue - cost_total

        smooth_penalty = 0.0
        if self._smooth_enabled and self.t > 1:
            a_norm = (a - self._a_lo) / np.maximum(self._a_hi - self._a_lo, 1e-8)
            prev_norm = (self._prev_action - self._a_lo) / np.maximum(
                self._a_hi - self._a_lo, 1e-8)
            smooth_penalty = float(
                np.sum(self._smooth_weights * (a_norm - prev_norm) ** 2))

        retention_penalty = 0.0
        if (self._clv_rs_enabled and self.t > self._clv_rs_warmup
                and N_act > 0):
            retention_penalty = self._clv_rs_alpha * (n_churn / max(N_act, 1))

        pop_bonus = 0.0
        if self._pop_reward_enabled:
            pop_bonus = self._pop_beta * (
                N_act / max(self.N_total, 1) - self._pop_target_ratio)

        reward = self._compute_reward(
            profit, smooth_penalty, retention_penalty, pop_bonus)

        self._prev_action = a.copy()
        self._prev_revenue = revenue
        self._prev_cost = cost_total
        self._prev_profit = profit
        self._prev_n_join = n_join
        self._prev_n_churn = n_churn
        self._prev_pviol_U = pviol_U
        self._prev_pviol_E = pviol_E
        self._prev_L_U = L_U
        self._prev_L_E = L_E
        self._prev_C_U = C_U
        self._prev_C_E = C_E
        self._prev_over_rev_E = over_rev_E

        return {
            "step": self.t,
            "cycle": (self.t - 1) // self.T + 1,
            "cycle_step": self._cycle_step,
            "F_U": F_U, "p_over_U": p_over_U,
            "F_E": F_E, "p_over_E": p_over_E,
            "rho_U": rho_U,
            "N_active": N_act, "N_U": N_U, "N_E": N_E,
            "N_inactive": self.N_total - N_act,
            "n_join": n_join, "n_churn": n_churn,
            "L_U": L_U, "L_E": L_E,
            "C_U": C_U, "C_E": C_E,
            "pviol_U": pviol_U, "pviol_E": pviol_E,
            "base_rev": base_rev, "over_rev": over_rev,
            "over_rev_E": over_rev_E,
            "revenue": revenue,
            "cost_opex": cost_opex, "cost_energy": cost_energy,
            "cost_cac": cost_cac, "sla_penalty": sla_penalty,
            "cost_total": cost_total,
            "profit": profit, "reward": reward,
            "smooth_penalty": smooth_penalty,
            "retention_penalty": retention_penalty,
            "pop_bonus": pop_bonus,
            "cycle_usage_U": self._cycle_usage_U,
            "cycle_usage_E": self._cycle_usage_E,
        }

    def _generate_traffic(self, p_over_U: float = 0.0,
                          p_over_E: float = 0.0) -> Tuple[float, float]:
        act = self._active_mask
        n = int(act.sum())
        if n == 0:
            return 0.0, 0.0

        traf_cfg_u = self.cfg["traffic"]["URLLC"]
        traf_cfg_e = self.cfg["traffic"]["eMBB"]

        raw_u = self._rng.lognormal(self._mu_u, self._sig_u)
        raw_u = np.clip(raw_u, traf_cfg_u["D_min_gb"], traf_cfg_u["D_max_gb"])

        raw_e = self._rng.lognormal(self._mu_e, self._sig_e)
        raw_e = np.clip(raw_e, traf_cfg_e["D_min_gb"], traf_cfg_e["D_max_gb"])

        if self._demand_elast_enabled:
            mult_u = max(self._demand_floor,
                         1.0 - self._eps_U * (p_over_U / self._pref_U - 1.0))
            mult_e = max(self._demand_floor,
                         1.0 - self._eps_E * (p_over_E / self._pref_E - 1.0))
            raw_u *= mult_u
            raw_e *= mult_e

        raw_u *= act * self._slice_is_U
        raw_e *= act * self._slice_is_E

        return float(raw_u.sum()), float(raw_e.sum())

    def _market_step(self, F_U: float, p_over_U: float,
                     F_E: float, p_over_E: float
                     ) -> Tuple[int, int]:
        N_act = int(self._active_mask.sum())
        N_inact = self.N_total - N_act

        P_sig = (F_U + F_E) / (2.0 * self._price_norm)
        Q_sig = 1.0 - (self._prev_pviol_U + self._prev_pviol_E) / 2.0

        churn_logits = (
            self.b0_churn
            + self.bp_churn * self._psens * P_sig
            - self.bq_churn * self._qsens * Q_sig
            - self.bsw_churn * self._swcost
        )
        p_churn_all = sigmoid(churn_logits)
        p_churn_active = p_churn_all * self._active_mask

        E_churn = float(p_churn_active.sum())

        join_logits = (
            self.b0_join
            - self.bp_join * self._psens * P_sig
            + self.bq_join * self._qsens * Q_sig
        )
        p_join_all = sigmoid(join_logits)
        p_join_inactive = p_join_all * (~self._active_mask).astype(np.float64)

        E_join = float(p_join_inactive.sum())

        if self.market_mode == "expectation":
            n_churn = min(int(round(E_churn)), N_act)
            n_join = min(int(round(E_join)), N_inact)
            if n_churn > 0:
                churn_scores = p_churn_active.copy()
                churn_scores[~self._active_mask] = -1.0
                churn_idx = np.argsort(churn_scores)[-n_churn:]
                self._active_mask[churn_idx] = False
            if n_join > 0:
                join_scores = p_join_inactive.copy()
                join_scores[self._active_mask] = -1.0
                join_idx = np.argsort(join_scores)[-n_join:]
                self._active_mask[join_idx] = True
        else:
            n_churn_raw = int(self._rng.poisson(max(E_churn, 0.0)))
            n_churn = min(n_churn_raw, N_act)
            n_join_raw = int(self._rng.poisson(max(E_join, 0.0)))
            n_join = min(n_join_raw, N_inact)

            if n_churn > 0:
                active_idx = np.where(self._active_mask)[0]
                probs = p_churn_all[active_idx]
                probs = probs / max(probs.sum(), 1e-12)
                chosen = self._rng.choice(
                    active_idx, size=min(n_churn, len(active_idx)),
                    replace=False, p=probs)
                self._active_mask[chosen] = False

            if n_join > 0:
                inact_idx = np.where(~self._active_mask)[0]
                probs = p_join_all[inact_idx]
                probs = probs / max(probs.sum(), 1e-12)
                chosen = self._rng.choice(
                    inact_idx, size=min(n_join, len(inact_idx)),
                    replace=False, p=probs)
                self._active_mask[chosen] = True

        return n_join, n_churn

    def _compute_reward(self, profit: float,
                        smooth_penalty: float = 0.0,
                        retention_penalty: float = 0.0,
                        pop_bonus: float = 0.0) -> float:
        if not np.isfinite(profit):
            profit = 0.0
        r = float(np.sign(profit) * np.log1p(abs(profit) / self._reward_scale))
        r -= smooth_penalty
        r -= retention_penalty
        r += pop_bonus
        return float(np.clip(r, -2.0, 2.0))

    def _build_obs(self) -> np.ndarray:
        N_act = int(self._active_mask.sum())
        N_U = int((self._active_mask * self._slice_is_U).sum())
        N_E = N_act - N_U

        allow_cap_U = max(self.Q_U * N_U, 1e-6)
        allow_cap_E = max(self.Q_E * N_E, 1e-6)
        allow_util_U = min(self._cycle_usage_U / allow_cap_U, 5.0)
        allow_util_E = min(self._cycle_usage_E / allow_cap_E, 5.0)

        load_factor_U = min(self._prev_L_U / max(self._prev_C_U, 1e-6), 5.0)
        load_factor_E = min(self._prev_L_E / max(self._prev_C_E, 1e-6), 5.0)

        p_over_E = self._prev_action[3] if self.t > 0 else 1.0
        over_rev_rate_E = self._prev_over_rev_E / max(p_over_E * N_E + 1e-6, 1e-6)
        over_rev_rate_E = min(over_rev_rate_E, 5.0)

        days_remaining = (self.T - self._cycle_step) / max(self.T, 1)

        obs = np.array([
            N_act / max(self.N_total, 1),
            (self.N_total - N_act) / max(self.N_total, 1),
            self._prev_n_join / max(self.N_total * 0.05, 1.0),
            self._prev_n_churn / max(self.N_total * 0.05, 1.0),
            self._prev_pviol_U,
            self._prev_pviol_E,
            self._prev_revenue / max(self._reward_scale, 1.0),
            self._prev_cost / max(self._reward_scale, 1.0),
            self._prev_profit / max(self._reward_scale, 1.0),
            self._prev_action[0] / self._a_hi[0],
            self._prev_action[1] / self._a_hi[1],
            self._prev_action[2] / self._a_hi[2],
            self._prev_action[3] / self._a_hi[3],
            self._prev_action[4],
            (self.t % self.T) / max(self.T, 1),
            self.t / max(self.episode_len, 1),
            allow_util_U,
            allow_util_E,
            load_factor_U,
            load_factor_E,
            over_rev_rate_E,
            days_remaining,
        ], dtype=np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=self._obs_hi, neginf=self._obs_lo)
        obs = np.clip(obs, self._obs_lo, self._obs_hi)
        return obs

    def compute_clv(self, monthly_cashflow: float,
                    retention_prob: float) -> float:
        clv = 0.0
        for k in range(self.clv_horizon):
            clv += monthly_cashflow * (retention_prob ** k) / ((1.0 + self.clv_d) ** k)
        return clv
