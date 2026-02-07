"""
Economics module: SLA credits, energy cost, profit, and reward.

Section 12 — SLA/SLO violation & credits:
  12.1  V_s = (#violating inner steps) / K
  12.2  C_sla_s = credit_frac(V_s) * F_s * N_active_s
        Credit tiers are scenario assumptions informed by MRC-based
        credit structures.  [VERIZON_SLA][SOLARWINDS_SLA_SLO]

Section 14 — Energy model:
  P_kW(load) = P0 + (P1 - P0) * load   [BS_POWER][BS_POWER_MEAS]
  C_energy = P_avg_kW * hours_month * elec_price_KRW_per_kWh
  Fixed electricity unit price (no TOU variation).

Section 15 — Profit and reward:
  15.1  Rev = Σ_active F_s + Σ_topup Price_top
        Cost = C_energy + Σ_s C_sla_s + C_resource
        Pi = Rev - Cost       (CAC is removed)
  15.2  C_resource = unit_cost_prb * PRB_total * mean(rho_util_k)
  15.3  r = tanh(Pi / profit_scale) - λ_penalty * penalty  [SB3_TIPS]

References:
  [VERIZON_SLA]       https://www.verizon.com/business/service_guide/reg/cp_mgn_plus_sla_2020AUG17_mk.pdf
  [SOLARWINDS_SLA_SLO] https://www.solarwinds.com/sre-best-practices/sla-vs-slo
  [BS_POWER]          https://arxiv.org/abs/1411.1571
  [BS_POWER_MEAS]     https://www.mdpi.com/1424-8220/12/4/4281
  [SB3_TIPS]          https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("oran.economics")


# =====================================================================
# 12) SLA / SLO model  [VERIZON_SLA][SOLARWINDS_SLA_SLO]
# =====================================================================

class SLAModel:
    """SLA/SLO violation measurement and credit computation (Section 12).

    Uses internal SLO targets to detect violations; credit tiers
    are scenario assumptions informed by MRC-based credit structures.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        sla = cfg.get("sla", {})
        self.SLO_T: Dict[str, float] = {
            "eMBB": sla.get("SLO_T_user_eMBB_mbps", 10.0),
            "URLLC": sla.get("SLO_T_user_URLLC_mbps", 5.0),
        }
        # Credit tier table (scenario; configurable) [VERIZON_SLA]
        # Each entry: {"threshold": V_max, "fraction": credit_frac}
        # Table is assumed sorted ascending by threshold.
        self.credit_tiers: List[Dict[str, float]] = sla.get("credit_tiers", [
            {"threshold": 0.05, "fraction": 0.00},
            {"threshold": 0.15, "fraction": 0.05},
            {"threshold": 0.30, "fraction": 0.10},
            {"threshold": 1.00, "fraction": 0.20},
        ])
        self.credit_cap: float = sla.get("credit_cap_fraction", 0.30)

    # -----------------------------------------------------------------
    # 12.1  Violation rate (monthly)
    # -----------------------------------------------------------------

    def compute_violation_rate(
        self,
        avg_throughputs_per_step: np.ndarray,
        slice_name: str,
    ) -> float:
        """Fraction of inner steps where avg throughput < SLO.

        Parameters
        ----------
        avg_throughputs_per_step : ndarray shape (K,)
            Per-step slice-level average throughput (Mbps).
        slice_name : str

        Returns
        -------
        float : V_s = (#violating steps) / K, in [0, 1].
        """
        if len(avg_throughputs_per_step) == 0:
            return 0.0

        slo = self.SLO_T.get(slice_name, 0.0)
        violations = np.sum(avg_throughputs_per_step < slo)
        K = len(avg_throughputs_per_step)
        return float(violations / K)

    # -----------------------------------------------------------------
    # 12.2  Credit computation (MRC-based)  [VERIZON_SLA]
    # -----------------------------------------------------------------

    def _credit_fraction(self, V_s: float) -> float:
        """Look up credit fraction from tier table.

        Tier table (scenario defaults):
          V_s ≤ 0.05 → 0%
          0.05 < V_s ≤ 0.15 → 5%
          0.15 < V_s ≤ 0.30 → 10%
          V_s > 0.30 → 20%
        """
        frac = 0.0
        for tier in self.credit_tiers:
            if V_s <= tier["threshold"]:
                frac = tier["fraction"]
                break
            frac = tier["fraction"]  # last tier catches all above
        return frac

    def compute_credit(
        self,
        V_s: float,
        F_s: float,
        N_active_s: int,
    ) -> float:
        """SLA credit (KRW) for one slice.

        C_sla_s = credit_frac(V_s) * F_s * N_active_s
        Cap: total credit ≤ credit_cap * F_s * N_active_s
        """
        frac = self._credit_fraction(V_s)
        mrc = F_s * max(N_active_s, 0)
        credit = frac * mrc
        cap = self.credit_cap * mrc
        return min(credit, cap)


# =====================================================================
# 14) Energy model  [BS_POWER][BS_POWER_MEAS]
# =====================================================================

class EnergyModel:
    """Base station energy cost (Section 14).

    P_kW(load) = P0 + (P1 - P0) * load
    C_energy = P_avg_kW * hours_month * elec_price_KRW_per_kWh

    P0, P1 are scenario params with literature-reasonable ranges.
    Fixed electricity unit price (no TOU variation).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        e = cfg.get("energy", {})
        self.P0_kw: float = e.get("P0_kw", 0.8)
        self.P1_kw: float = e.get("P1_kw", 1.4)
        self.hours_per_month: float = e.get("hours_per_month", 730.0)
        self.elec_price: float = e.get("elec_price_krw_per_kwh", 120.0)

    def power_kw(self, load: float) -> float:
        """Instantaneous power (kW) at given load ∈ [0, 1].

        P(load) = P0 + (P1 - P0) * load  [BS_POWER]
        """
        load = float(np.clip(load, 0.0, 1.0))
        return self.P0_kw + (self.P1_kw - self.P0_kw) * load

    def compute_cost(self, avg_load: float) -> float:
        """Monthly energy cost (KRW).

        C_energy = P_avg_kW * hours_month * elec_price
        """
        p_avg = self.power_kw(avg_load)
        return p_avg * self.hours_per_month * self.elec_price


# =====================================================================
# 15) Economics — Revenue, Cost, Profit, Reward  [SB3_TIPS]
# =====================================================================

class EconomicsModel:
    """Full economics: revenue, cost, profit, reward (Section 15).

    15.1  Rev  = Σ_active F_s + Σ_topup Price_top
          Cost = C_energy + Σ_s C_sla_s + C_resource
          Pi   = Rev - Cost         (CAC removed)

    15.2  C_resource = unit_cost_prb * PRB_total * mean(rho_util_k)

    15.3  r = tanh(Pi / profit_scale) - λ_penalty * penalty
          profit_scale calibrated via random rollouts.  [SB3_TIPS]
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        ec = cfg.get("economics", {})
        self.unit_cost_prb: float = ec.get("unit_cost_prb_krw", 50.0)
        self.profit_scale: float = ec.get("profit_scale", 1.0)
        self.lambda_penalty: float = ec.get("lambda_penalty", 10.0)
        self.reward_clip: float = ec.get("reward_clip", 2.0)
        self.prb_total: int = cfg.get("radio", {}).get("prb_total", 273)

        self.sla = SLAModel(cfg)
        self.energy = EnergyModel(cfg)

    # -----------------------------------------------------------------
    # 15.2  Resource cost
    # -----------------------------------------------------------------

    def compute_resource_cost(self, mean_rho_util: float) -> float:
        """C_resource = unit_cost_prb * PRB_total * mean(rho_util_k)"""
        return self.unit_cost_prb * self.prb_total * max(mean_rho_util, 0.0)

    # -----------------------------------------------------------------
    # 15.1  Profit
    # -----------------------------------------------------------------

    def compute_profit(
        self,
        fees: Dict[str, float],
        N_active: Dict[str, int],
        n_topups: Dict[str, int],
        topup_price: float,
        V_rates: Dict[str, float],
        mean_rho_util: float,
        avg_load: float,
    ) -> Dict[str, float]:
        """Compute monthly profit breakdown (Section 15.1).

        Parameters
        ----------
        fees : dict   {"eMBB": F_eMBB, "URLLC": F_URLLC}
        N_active : dict  active users per slice
        n_topups : dict  number of top-ups per slice
        topup_price : float  KRW per top-up pack
        V_rates : dict  violation rates per slice
        mean_rho_util : float  average utilization across inner steps
        avg_load : float  overall cell load for energy model

        Returns
        -------
        dict with keys:
          revenue, revenue_sub, revenue_topup,
          cost_energy, cost_sla_eMBB, cost_sla_URLLC, cost_sla_total,
          cost_resource, cost_total,
          profit
        """
        # ---- Revenue ----
        rev_sub = sum(
            fees.get(s, 0.0) * max(N_active.get(s, 0), 0)
            for s in ["eMBB", "URLLC"]
        )
        rev_topup = sum(
            n_topups.get(s, 0) * topup_price
            for s in ["eMBB", "URLLC"]
        )
        revenue = rev_sub + rev_topup

        # ---- Costs ----
        # Energy [BS_POWER][BS_POWER_MEAS]
        cost_energy = self.energy.compute_cost(avg_load)

        # SLA credits [VERIZON_SLA]
        cost_sla: Dict[str, float] = {}
        for s in ["eMBB", "URLLC"]:
            cost_sla[s] = self.sla.compute_credit(
                V_s=V_rates.get(s, 0.0),
                F_s=fees.get(s, 0.0),
                N_active_s=N_active.get(s, 0),
            )
        cost_sla_total = sum(cost_sla.values())

        # Resource cost (Section 15.2)
        cost_resource = self.compute_resource_cost(mean_rho_util)

        cost_total = cost_energy + cost_sla_total + cost_resource

        # ---- Profit ----
        profit = revenue - cost_total

        return {
            "revenue": revenue,
            "revenue_sub": rev_sub,
            "revenue_topup": rev_topup,
            "cost_energy": cost_energy,
            "cost_sla_eMBB": cost_sla["eMBB"],
            "cost_sla_URLLC": cost_sla["URLLC"],
            "cost_sla_total": cost_sla_total,
            "cost_resource": cost_resource,
            "cost_total": cost_total,
            "profit": profit,
        }

    # -----------------------------------------------------------------
    # 15.3  Reward  [SB3_TIPS]
    # -----------------------------------------------------------------

    def compute_reward(
        self,
        profit: float,
        penalty: float = 0.0,
    ) -> float:
        """Reward: tanh(Pi / profit_scale) - λ_penalty * penalty.

        profit_scale is calibrated from random rollouts to avoid
        arbitrary scaling.  [SB3_TIPS]

        Final reward is clipped to [-reward_clip, reward_clip].
        """
        if not np.isfinite(profit):
            logger.warning("Non-finite profit: %s → using 0", profit)
            profit = 0.0

        # Avoid division by zero in profit_scale
        scale = max(abs(self.profit_scale), 1.0)
        raw = float(np.tanh(profit / scale))

        r = raw - self.lambda_penalty * penalty
        r = float(np.clip(r, -self.reward_clip, self.reward_clip))
        return r
