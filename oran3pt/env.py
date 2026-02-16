"""
O-RAN 1-Cell · 3-Part Tariff · 2 Slices · Constrained MDP  (§§3–11)

REVISION 10 — Changes from v9:
  [EP1] 1-Cycle Continuous Episode Design      [Pardo ICML 2018; Wan arXiv 2025]
        - episode_cycles 24→1, episode_mode="continuous"
        - truncated=True (not terminated) at cycle boundary
        - Population persists across reset; only billing accumulators reset
        - obs[15] redefined: episode_progress → churn_rate_ema
        - _total_steps global counter for warmup
        - Eliminates value function bias from terminated=True at step 720
  Prior revisions (v1–v9):
  [D1] NSACF-style admission control          [3GPP TS 23.501 §5.2.3; Caballero JSAC 2019]
  [D2] Hierarchical action timing              [Vezhnevets ICML 2017; Bacon AAAI 2017]
  [D4] Slice-specific QoS signal per user     [Kim & Yoon 2004; Ahn 2006]
  [D5] Observation dim 22 → 24                [Dulac-Arnold JMLR 2021]
  [D6] pviol_E EMA in obs[22]                 [Dulac-Arnold JMLR 2021]
  [M3] Convex SLA penalty for eMBB            [Tessler 2019; Paternain 2019]
  [M4] rho_U_max 0.60 → 0.35                 [Huang IoT-J 2020]
  [M5] beta_pop 0.1 → 0.3                    [Mguni 2019; Zheng 2022]
  [M6] Lagrangian safety layer                [Tessler 2019; Stooke 2020]
  [R4] Per-dimension action smoothing         [Dalal NeurIPS 2018]
  [R5] Observation dim 20 → 22               [Dulac-Arnold JMLR 2021]
  [R6] Population-aware reward term           [Mguni 2019; Zheng 2022]
  [C4] Demand-price elasticity                [Nevo et al. Econometrica 2016]
  [C5] Action smoothing penalty               [Dulac-Arnold JMLR 2021]

Gymnasium-compatible environment for SB3 SAC.

Formal Problem Structure — Constrained MDP (CMDP) [Altman 1999]:
  State:   s_t ∈ ℝ²⁴  (normalised observation vector)
  Action:  a_t ∈ ℝ⁵   [F_U, p_U^over, F_E, p_E^over, ρ_U]
  Reward:  r(s,a) = sign(π)·log1p(|π|/s) − penalties + bonuses
  Constraint:  E[pviol_E] ≤ ε_QoS  (Lagrangian dual ascent [M6])
  Transition:  P(s'|s,a) from market logit + lognormal traffic

The agent solves: max_π E[Σ γ^t r_t]  s.t.  E[Σ γ^t g_t] ≤ 0
where g_t = pviol_E − ε_QoS.
  [Tessler ICML 2019; Stooke ICLR 2020; Boyd & Vandenberghe 2004]

Action (5-D continuous, §4):
  a = [F_U, p_U^over, F_E, p_E^over, ρ_U]
  [D2] Pricing dims (a[0:4]) update only at cycle boundaries when
       hierarchical_actions.enabled=true. ρ_U (a[4]) updates every step.

Observation (24-D, §3.2):
  Indices 0–21: see ``_build_obs`` for layout
  [D6] 22: pviol_E_ema (EMA α=0.3 trend signal)
  [D5] 23: load_headroom_E

Revenue per step (§5.2 — online accrual):
  BaseRev_t = (F_U·N_U + F_E·N_E) / T
  OverRev_t = Σ_s p_s^over · ΔOver_s(t)

Cost per step (§9):
  Cost_t = c_opex·N_active + c_energy·(L_U+L_E) + c_cac·N_join + SLA
  [M3] SLA = λ_U·pviol_U + λ_E·pviol_E^γ  (convex)

Reward (§15 + §11b + §15c + §15d):
  reward = log_profit − smooth_penalty − retention_penalty
           + pop_bonus − lagrangian_penalty

Market (§10):
  Logit-based churn/join. [D1] Joins gated by load-aware admission control.

References:
  [Haarnoja 2018]    SAC
  [Grubb AER 2009]   3-part tariff
  [Nevo 2016]        Broadband tariffs / usage coupling  (Econometrica)
  [Dulac-Arnold 2021] Challenges of Real-World RL
  [Tessler 2019]     Reward Constrained Policy Optimization (RCPO)
  [Stooke 2020]      Responsive Safety in RL
  [Altman 1999]      Constrained Markov Decision Processes
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .utils import sigmoid

logger = logging.getLogger("oran3pt.env")


class OranSlicingPricingEnv(gym.Env):
    """O-RAN single-cell CMDP with 3-part tariff and admission control."""

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

        # [EP1] Episode mode: "continuous" or "episodic" (legacy)
        # [Pardo ICML 2018; Wan arXiv 2025]
        self._episode_mode: str = tc.get("episode_mode", "episodic")
        self._total_steps: int = 0       # global step counter (persists across resets)
        self._first_reset_done: bool = False  # track if first reset has occurred

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

        # Observation (§3.2)  [D5] 24-D
        obs_cfg = cfg.get("observation", {})
        self._obs_dim: int = obs_cfg.get("dim", 24)
        self._obs_lo = obs_cfg.get("clip_min", -10.0)
        self._obs_hi = obs_cfg.get("clip_max", 10.0)
        self.observation_space = spaces.Box(
            self._obs_lo, self._obs_hi, shape=(self._obs_dim,),
            dtype=np.float32)

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
        self._gamma_sla_E: float = qos.get("gamma_sla_E", 1.0)

        # [M6] Lagrangian safety layer
        lag_cfg = cfg.get("lagrangian_qos", {})
        self._lagrangian_enabled: bool = lag_cfg.get("enabled", False)
        self._lagrangian_lambda: float = 0.0
        self._pviol_E_threshold: float = lag_cfg.get(
            "pviol_E_threshold", 0.15)

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

        # Demand-price elasticity (§6b)
        de = cfg.get("demand_elasticity", {})
        self._demand_elast_enabled: bool = de.get("enabled", False)
        self._eps_U: float = de.get("epsilon_U", 0.15)
        self._eps_E: float = de.get("epsilon_E", 0.30)
        self._pref_U: float = de.get("p_ref_U", 2500.0)
        self._pref_E: float = de.get("p_ref_E", 1500.0)
        self._demand_floor: float = de.get("floor", 0.5)

        # Action smoothing (§15b)
        sm = cfg.get("action_smoothing", {})
        self._smooth_enabled: bool = sm.get("enabled", False)
        self._smooth_weight: float = sm.get("weight", 0.05)
        smooth_weights = sm.get("weights", None)
        if smooth_weights is not None and len(smooth_weights) == 5:
            self._smooth_weights = np.array(
                smooth_weights, dtype=np.float64)
        else:
            self._smooth_weights = np.full(
                5, self._smooth_weight, dtype=np.float64)

        # CLV reward shaping (§11b)
        clv_rs = cfg.get("clv_reward_shaping", {})
        self._clv_rs_enabled: bool = clv_rs.get("enabled", False)
        self._clv_rs_alpha: float = clv_rs.get("alpha_retention", 2.0)
        self._clv_rs_warmup: int = clv_rs.get("warmup_steps", 100)

        # Population-aware reward (§15c)
        pop_rw = cfg.get("population_reward", {})
        self._pop_reward_enabled: bool = pop_rw.get("enabled", False)
        self._pop_beta: float = pop_rw.get("beta_pop", 0.1)
        self._pop_target_ratio: float = pop_rw.get("target_ratio", 0.4)

        # ── [D1] Admission control (NSACF) ───────────────────────────
        # 3GPP TS 23.501 §5.2.3; Caballero et al. IEEE JSAC 2019
        ac_cfg = cfg.get("admission_control", {})
        self._ac_enabled: bool = ac_cfg.get("enabled", False)
        self._ac_load_threshold: float = ac_cfg.get(
            "load_threshold", 0.85)
        self._ac_pviol_ceiling: float = ac_cfg.get(
            "pviol_ceiling", 0.30)
        self._ac_per_user_E: float = ac_cfg.get(
            "per_user_load_estimate_E_gb", 1.5)
        self._ac_per_user_U: float = ac_cfg.get(
            "per_user_load_estimate_U_gb", 0.15)

        # ── [D2] Hierarchical action timing ──────────────────────────
        # Vezhnevets et al. ICML 2017; Bacon et al. AAAI 2017
        ha_cfg = cfg.get("hierarchical_actions", {})
        self._hier_enabled: bool = ha_cfg.get("enabled", False)

        # Load users
        self._users_csv_path = users_csv
        self._load_users(users_csv)

        # Reward normalisation
        self._reward_scale = max(
            self._a_hi[0] * self.N_total / self.T, 1.0)

        # Per-episode state
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
        self._prev_n_rejected: int = 0       # [D1]
        self._cycle_pricing = np.zeros(4, dtype=np.float64)  # [D2]
        self._last_churned_ids: list = []   # [M13c]
        self._last_joined_ids: list = []    # [M13c]
        self._pviol_E_ema: float = 0.0     # [D6] EMA for obs trend
        self._lagrangian_boost: float = 1.0  # [D5] curriculum phase boost
        self._churn_rate_ema: float = 0.0  # [EP1] churn rate EMA for obs[15]

    def _load_users(self, csv_path: Optional[str]) -> None:
        if csv_path is not None and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        else:
            from .gen_users import generate_users
            df = generate_users(
                self.cfg, seed=int(self._rng.integers(0, 2**31)))
        self._users = df
        self.N_total: int = len(df)
        self._slice_is_U = (
            df["slice"].values == "URLLC").astype(np.float64)
        self._slice_is_E = 1.0 - self._slice_is_U
        self._mu_u = df["mu_urllc"].values.astype(np.float64)
        self._sig_u = df["sigma_urllc"].values.astype(np.float64)
        self._mu_e = df["mu_embb"].values.astype(np.float64)
        self._sig_e = df["sigma_embb"].values.astype(np.float64)
        self._psens = df["price_sensitivity"].values.astype(np.float64)
        self._qsens = df["qos_sensitivity"].values.astype(np.float64)
        self._swcost = df["switching_cost"].values.astype(np.float64)
        self._init_active = df["is_active_init"].values.astype(bool)

    # ── Public setters ────────────────────────────────────────────────

    def set_curriculum_phase(self, phase: int) -> None:
        self._curriculum_phase = phase

    def set_lagrangian_lambda(self, lambda_val: float) -> None:
        """[M6] Update Lagrangian multiplier from dual ascent."""
        self._lagrangian_lambda = lambda_val

    def set_lagrangian_boost(self, boost: float) -> None:
        """[D5] Set Lagrangian penalty boost factor for curriculum phases."""
        self._lagrangian_boost = boost

    # ── Gymnasium interface ───────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # [EP1] Continuous mode: preserve population and financial state
        # across resets (billing-cycle boundaries). Full init only on first
        # reset or in legacy episodic mode.
        # [Pardo ICML 2018] — truncation boundary; [Wan arXiv 2025] — continuing
        is_continuous = (self._episode_mode == "continuous"
                         and self._first_reset_done)

        self.t = 0

        # Always reset billing accumulators (cycle boundary)
        self._cycle_usage_U = 0.0
        self._cycle_usage_E = 0.0
        self._prev_over_U = 0.0
        self._prev_over_E = 0.0
        self._last_churned_ids = []   # [M13c]
        self._last_joined_ids = []    # [M13c]

        if not is_continuous:
            # Full reset — first call or episodic mode
            self._active_mask = self._init_active.copy()
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
            self._prev_n_rejected = 0
            self._cycle_pricing = mid[:4].copy()
            self._pviol_E_ema = 0.0       # [D6]
            self._churn_rate_ema = 0.0     # [EP1]
        # else: continuous mode — population, prev_action, pviol_E_ema,
        #       financial state, churn_rate_ema all persist

        self._first_reset_done = True
        return self._build_obs(), {}

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.t += 1
        self._total_steps += 1    # [EP1] global counter persists across resets
        info = self._run_step(action)
        obs = self._build_obs()
        reward = info["reward"]

        # [EP1] Continuous mode: truncated=True (not terminated) at cycle end
        # SB3 ReplayBuffer bootstraps V(s') for truncated transitions
        # [Pardo ICML 2018; Gymnasium API]
        done = self.t >= self.episode_len
        if self._episode_mode == "continuous":
            terminated = False
            truncated = done
        else:
            terminated = done
            truncated = False

        return obs, reward, terminated, truncated, info

    # ── Action mapping ────────────────────────────────────────────────

    def _map_action(self, raw: np.ndarray) -> np.ndarray:
        a = np.clip(raw, -1.0, 1.0).astype(np.float64)
        return self._a_lo + (a + 1.0) / 2.0 * (self._a_hi - self._a_lo)

    @property
    def _cycle_step(self) -> int:
        return (self.t - 1) % self.T

    @property
    def _is_cycle_start(self) -> bool:
        return self._cycle_step == 0

    # ── [D1] Admission control ────────────────────────────────────────

    def _admission_gate(self, n_candidates: int,
                        candidate_idx: np.ndarray,
                        C_E: float, L_E: float,
                        C_U: float, L_U: float
                        ) -> Tuple[int, int]:
        """NSACF-style load-aware admission gating.

        For each join candidate, predict marginal load impact.
        Reject if admitting pushes projected load ratio above threshold
        or if current pviol_E already exceeds the pviol ceiling.

        Returns (n_admitted, n_rejected).

        References:
          [3GPP TS 23.501 §5.2.3] — Network Slice Admission Control Function
          [Caballero et al., IEEE JSAC 2019] — Admission control for slicing
          [Samdanis et al., IEEE CommMag 2016] — Slice brokering
        """
        if not self._ac_enabled or n_candidates == 0:
            return n_candidates, 0

        # Block all joins when pviol already exceeds ceiling
        if self._prev_pviol_E > self._ac_pviol_ceiling:
            return 0, n_candidates

        n_admitted = 0
        proj_L_E = L_E
        proj_L_U = L_U

        for i in range(n_candidates):
            idx = candidate_idx[i]
            is_urllc = self._slice_is_U[idx] > 0.5
            marginal = (self._ac_per_user_U if is_urllc
                        else self._ac_per_user_E)

            if is_urllc:
                new_ratio_U = (proj_L_U + marginal) / max(C_U, 1e-6)
                new_ratio_E = proj_L_E / max(C_E, 1e-6)
            else:
                new_ratio_E = (proj_L_E + marginal) / max(C_E, 1e-6)
                new_ratio_U = proj_L_U / max(C_U, 1e-6)

            if (new_ratio_E > self._ac_load_threshold
                    or new_ratio_U > self._ac_load_threshold):
                break  # remaining candidates also rejected

            n_admitted += 1
            if is_urllc:
                proj_L_U += marginal
            else:
                proj_L_E += marginal

        return n_admitted, n_candidates - n_admitted

    # ── Core step ─────────────────────────────────────────────────────

    def _run_step(self, raw_action: np.ndarray) -> Dict[str, Any]:
        a = self._map_action(raw_action)
        F_U, p_over_U, F_E, p_over_E, rho_U = a

        # [D2] Hierarchical: lock pricing at cycle start
        if self._hier_enabled:
            if self._is_cycle_start:
                self._cycle_pricing = np.array(
                    [F_U, p_over_U, F_E, p_over_E], dtype=np.float64)
            else:
                F_U, p_over_U, F_E, p_over_E = self._cycle_pricing

        # Reset cycle accumulators
        if self._is_cycle_start:
            self._cycle_usage_U = 0.0
            self._cycle_usage_E = 0.0
            self._prev_over_U = 0.0
            self._prev_over_E = 0.0

        # [M13c] Reset per-step user event lists
        self._last_churned_ids = []
        self._last_joined_ids = []

        # Market dynamics
        if self._curriculum_phase == 1:
            n_join, n_churn, n_rejected = 0, 0, 0
        else:
            n_join, n_churn, n_rejected = self._market_step(
                F_U, p_over_U, F_E, p_over_E, rho_U)

        N_act = int(self._active_mask.sum())
        N_U = int((self._active_mask * self._slice_is_U).sum())
        N_E = N_act - N_U

        # Traffic generation
        L_U, L_E = self._generate_traffic(p_over_U, p_over_E)
        self._cycle_usage_U += L_U
        self._cycle_usage_E += L_E

        # Capacity allocation
        C_U = rho_U * self.C_total * self.kappa_U
        C_E = (1.0 - rho_U) * self.C_total
        C_U = max(C_U, 1e-6)
        C_E = max(C_E, 1e-6)

        # QoS violation
        pviol_U = float(sigmoid(
            self.alpha_cong * (L_U / C_U - 1.0)))
        pviol_E = float(sigmoid(
            self.alpha_cong * (L_E / C_E - 1.0)))

        # [D6] Update pviol_E exponential moving average (α=0.3)
        self._pviol_E_ema = 0.3 * pviol_E + 0.7 * self._pviol_E_ema

        # [EP1] Update churn rate EMA for obs[15] (α=0.3)
        churn_rate = n_churn / max(N_act, 1)
        self._churn_rate_ema = 0.3 * churn_rate + 0.7 * self._churn_rate_ema

        # Revenue
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

        # Cost
        cost_opex = self.c_opex * N_act
        cost_energy = self.c_energy * (L_U + L_E)
        cost_cac = self.c_cac * n_join
        sla_penalty = (self.lambda_U * pviol_U
                       + self.lambda_E * (pviol_E ** self._gamma_sla_E))
        cost_total = cost_opex + cost_energy + cost_cac + sla_penalty
        profit = revenue - cost_total

        # ── Reward shaping ────────────────────────────────────────────
        smooth_penalty = 0.0
        if self._smooth_enabled and self.t > 1:
            a_full = np.array([F_U, p_over_U, F_E, p_over_E, rho_U])
            a_range = np.maximum(self._a_hi - self._a_lo, 1e-8)
            a_norm = (a_full - self._a_lo) / a_range
            prev_norm = (self._prev_action - self._a_lo) / a_range
            smooth_penalty = float(np.sum(
                self._smooth_weights * (a_norm - prev_norm) ** 2))

        retention_penalty = 0.0
        # [EP1] Use _total_steps for warmup — self.t resets every episode
        if (self._clv_rs_enabled
                and self._total_steps > self._clv_rs_warmup
                and N_act > 0):
            retention_penalty = self._clv_rs_alpha * (
                n_churn / max(N_act, 1))

        pop_bonus = 0.0
        if self._pop_reward_enabled:
            pop_bonus = self._pop_beta * (
                N_act / max(self.N_total, 1)
                - self._pop_target_ratio)

        lagrangian_penalty = 0.0
        if self._lagrangian_enabled and self._lagrangian_lambda > 0.0:
            lagrangian_penalty = self._lagrangian_lambda * max(
                0.0, pviol_E - self._pviol_E_threshold)
            lagrangian_penalty *= self._lagrangian_boost  # [D5] curriculum boost

        reward = self._compute_reward(
            profit, smooth_penalty, retention_penalty,
            pop_bonus, lagrangian_penalty)

        # Store state for next step
        self._prev_action = np.array(
            [F_U, p_over_U, F_E, p_over_E, rho_U], dtype=np.float64)
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
        self._prev_n_rejected = n_rejected

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
            "n_rejected": n_rejected,
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
            "lagrangian_penalty": lagrangian_penalty,
            "cycle_usage_U": self._cycle_usage_U,
            "cycle_usage_E": self._cycle_usage_E,
            "churned_user_ids": self._last_churned_ids,   # [M13c]
            "joined_user_ids": self._last_joined_ids,     # [M13c]
        }

    # ── Traffic ───────────────────────────────────────────────────────

    def _generate_traffic(self, p_over_U: float = 0.0,
                          p_over_E: float = 0.0
                          ) -> Tuple[float, float]:
        act = self._active_mask
        if int(act.sum()) == 0:
            return 0.0, 0.0

        traf_u = self.cfg["traffic"]["URLLC"]
        traf_e = self.cfg["traffic"]["eMBB"]

        raw_u = self._rng.lognormal(self._mu_u, self._sig_u)
        raw_u = np.clip(raw_u, traf_u["D_min_gb"], traf_u["D_max_gb"])
        raw_e = self._rng.lognormal(self._mu_e, self._sig_e)
        raw_e = np.clip(raw_e, traf_e["D_min_gb"], traf_e["D_max_gb"])

        if self._demand_elast_enabled:
            mult_u = max(self._demand_floor,
                         1.0 - self._eps_U * (
                             p_over_U / self._pref_U - 1.0))
            mult_e = max(self._demand_floor,
                         1.0 - self._eps_E * (
                             p_over_E / self._pref_E - 1.0))
            raw_u *= mult_u
            raw_e *= mult_e

        raw_u *= act * self._slice_is_U
        raw_e *= act * self._slice_is_E
        return float(raw_u.sum()), float(raw_e.sum())

    # ── Market dynamics ───────────────────────────────────────────────

    def _market_step(self, F_U: float, p_over_U: float,
                     F_E: float, p_over_E: float,
                     rho_U: float = 0.2
                     ) -> Tuple[int, int, int]:
        """Market step with [D1] admission control. Returns (join, churn, rejected)."""
        N_act = int(self._active_mask.sum())
        N_inact = self.N_total - N_act

        P_sig = (F_U + F_E) / (2.0 * self._price_norm)
        # [D4] Slice-specific QoS signal — each user sees own slice's QoS
        # [Kim & Yoon 2004; Ahn 2006]
        Q_sig_users = np.where(
            self._slice_is_U,
            1.0 - self._prev_pviol_U,    # URLLC users: own slice QoS
            1.0 - self._prev_pviol_E     # eMBB users: own slice QoS
        )

        # Churn logit
        churn_logits = (
            self.b0_churn
            + self.bp_churn * self._psens * P_sig
            - self.bq_churn * self._qsens * Q_sig_users  # [D4] per-user
            - self.bsw_churn * self._swcost)
        p_churn_all = sigmoid(churn_logits)
        p_churn_active = p_churn_all * self._active_mask
        E_churn = float(p_churn_active.sum())

        # Join logit
        join_logits = (
            self.b0_join
            - self.bp_join * self._psens * P_sig
            + self.bq_join * self._qsens * Q_sig_users)   # [D4] per-user
        p_join_all = sigmoid(join_logits)
        p_join_inactive = p_join_all * (
            ~self._active_mask).astype(np.float64)
        E_join = float(p_join_inactive.sum())

        n_rejected = 0
        C_U = rho_U * self.C_total * self.kappa_U
        C_E = (1.0 - rho_U) * self.C_total

        if self.market_mode == "expectation":
            n_churn = min(int(round(E_churn)), N_act)
            n_join_cand = min(int(round(E_join)), N_inact)

            # Churn first (frees capacity)
            if n_churn > 0:
                scores = p_churn_active.copy()
                scores[~self._active_mask] = -1.0
                idx = np.argsort(scores)[-n_churn:]
                self._active_mask[idx] = False
                self._last_churned_ids = idx.tolist()  # [M13c]

            # [D1] Admission gate
            if n_join_cand > 0:
                scores = p_join_inactive.copy()
                scores[self._active_mask] = -1.0
                cand_idx = np.argsort(scores)[-n_join_cand:]
                n_join, n_rejected = self._admission_gate(
                    n_join_cand, cand_idx,
                    C_E, self._prev_L_E, C_U, self._prev_L_U)
                if n_join > 0:
                    self._active_mask[cand_idx[:n_join]] = True
                    self._last_joined_ids = cand_idx[:n_join].tolist()  # [M13c]
            else:
                n_join = 0
        else:
            # Stochastic mode
            n_churn_raw = int(self._rng.poisson(max(E_churn, 0.0)))
            n_churn = min(n_churn_raw, N_act)
            n_join_raw = int(self._rng.poisson(max(E_join, 0.0)))
            n_join_cand = min(n_join_raw, N_inact)

            if n_churn > 0:
                active_idx = np.where(self._active_mask)[0]
                probs = p_churn_all[active_idx]
                probs = probs / max(probs.sum(), 1e-12)
                chosen = self._rng.choice(
                    active_idx,
                    size=min(n_churn, len(active_idx)),
                    replace=False, p=probs)
                self._active_mask[chosen] = False
                self._last_churned_ids = chosen.tolist()  # [M13c]

            if n_join_cand > 0:
                inact_idx = np.where(~self._active_mask)[0]
                probs = p_join_all[inact_idx]
                probs = probs / max(probs.sum(), 1e-12)
                cand_idx = self._rng.choice(
                    inact_idx,
                    size=min(n_join_cand, len(inact_idx)),
                    replace=False, p=probs)
                n_join, n_rejected = self._admission_gate(
                    len(cand_idx), cand_idx,
                    C_E, self._prev_L_E, C_U, self._prev_L_U)
                if n_join > 0:
                    self._active_mask[cand_idx[:n_join]] = True
                    self._last_joined_ids = cand_idx[:n_join].tolist()  # [M13c]
            else:
                n_join = 0

        return n_join, n_churn, n_rejected

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self, profit: float,
                        smooth_penalty: float = 0.0,
                        retention_penalty: float = 0.0,
                        pop_bonus: float = 0.0,
                        lagrangian_penalty: float = 0.0) -> float:
        if not np.isfinite(profit):
            profit = 0.0
        r = float(np.sign(profit) * np.log1p(
            abs(profit) / self._reward_scale))
        r -= smooth_penalty
        r -= retention_penalty
        r += pop_bonus
        # [D2] Lagrangian penalty applied OUTSIDE base reward clip
        # to preserve constraint gradient  [Paternain CDC 2019]
        r_base = float(np.clip(r, -2.0, 2.0))
        r_final = r_base - lagrangian_penalty
        return float(np.clip(r_final, -4.0, 4.0))

    # ── Observation ───────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        N_act = int(self._active_mask.sum())
        N_inact = self.N_total - N_act
        N_U = int((self._active_mask * self._slice_is_U).sum())
        N_E = N_act - N_U

        obs = np.zeros(self._obs_dim, dtype=np.float32)

        obs[0] = N_act / max(self.N_total, 1)
        obs[1] = N_inact / max(self.N_total, 1)
        obs[2] = self._prev_n_join / max(self.N_total * 0.05, 1)
        obs[3] = self._prev_n_churn / max(self.N_total * 0.05, 1)
        obs[4] = self._prev_pviol_U
        obs[5] = self._prev_pviol_E
        obs[6] = self._prev_revenue / max(self._reward_scale, 1)
        obs[7] = self._prev_cost / max(self._reward_scale, 1)
        obs[8] = self._prev_profit / max(self._reward_scale, 1)

        a_range = np.maximum(self._a_hi - self._a_lo, 1e-8)
        obs[9:14] = ((self._prev_action - self._a_lo) / a_range
                      ).astype(np.float32)

        obs[14] = ((self.t - 1) % self.T) / max(self.T, 1)
        # [EP1] obs[15]: churn_rate_ema (continuous) or episode_progress (episodic)
        # In continuous mode, episode_progress is spurious (no seasonality).
        # churn_rate_ema provides actionable market signal.
        # [Pardo ICML 2018; Dulac-Arnold JMLR 2021]
        if self._episode_mode == "continuous":
            obs[15] = self._churn_rate_ema
        else:
            obs[15] = self.t / max(self.episode_len, 1)
        obs[16] = self._cycle_usage_U / max(
            self.Q_U * max(N_U, 1), 1e-6)
        obs[17] = self._cycle_usage_E / max(
            self.Q_E * max(N_E, 1), 1e-6)
        obs[18] = self._prev_L_U / max(self._prev_C_U, 1e-6)
        obs[19] = self._prev_L_E / max(self._prev_C_E, 1e-6)
        obs[20] = self._prev_over_rev_E / max(
            self._prev_action[3] * max(N_E, 1), 1e-6)
        obs[21] = (self.T - self._cycle_step) / max(self.T, 1)

        # [D5][D6] v9 extended features
        if self._obs_dim >= 24:
            obs[22] = self._pviol_E_ema          # [D6] pviol_E trend (EMA)
            obs[23] = max(0.0, 1.0 - self._prev_L_E / max(
                self._prev_C_E, 1e-6))            # load headroom

        return np.clip(obs, self._obs_lo, self._obs_hi)