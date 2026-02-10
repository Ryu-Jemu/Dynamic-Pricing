"""
Economics module: SLA credits, energy cost, profit, and reward.

Sections 12, 14, 15 of HybridPrompt.md.

FIX F3: Added alpha_vrate — explicit SLA violation penalty in reward shaping.
  Previously, V_rate was only indirectly penalized via SLA credits (which
  were constant at V=1.0 due to the v_cap < SLO bug).  Now, the reward
  includes a direct V_rate penalty term:

    r += −α_vrate × mean(V_rate_eMBB, V_rate_URLLC)

  This provides continuous gradient signal for QoS improvement, which is
  essential after fixing the V_rate=1.0 bug so the agent can learn to
  balance profit against QoS.

  Academic basis:
    [NG_1999]        Ng, Harada, Russell, "Policy Invariance Under Reward
                     Transformations: Theory and Application to Reward
                     Shaping," ICML 1999 — potential-based shaping preserves
                     optimal policy; our additive term is a reasonable
                     auxiliary signal given the sparse SLA credit structure.
    [HENDERSON_2018] Henderson et al., "Deep RL that Matters," AAAI 2018 —
                     reward scale and shaping strongly affect learning.

FIX F4: Increased alpha_churn from 0.20 to 0.40.
  Previous analysis showed churn at 2.6× target (eMBB) despite low
  alpha_churn.  A stronger penalty makes the revenue-vs-retention
  tradeoff more balanced, preventing the agent from always choosing
  maximum fees.

  Academic basis:
    [CHURN_SLR]   Ahmad et al., "Customer Churn Prediction in Telecom:
                  A Systematic Literature Review," Management Review
                  Quarterly, 2023 — customer retention value typically
                  5–25× acquisition cost; justifies stronger churn penalty.

References:
  [VERIZON_SLA] [BS_POWER] [SB3_TIPS] [NG_1999] [HENDERSON_2018] [CHURN_SLR]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("oran.economics")

_VALID_REWARD_TYPES = ("tanh", "linear", "log")


class SLAModel:
    """SLA/SLO violation measurement and credit computation (Section 12)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        sla = cfg.get("sla", {})
        self.SLO_T: Dict[str, float] = {
            "eMBB": sla.get("SLO_T_user_eMBB_mbps", 3.0),
            "URLLC": sla.get("SLO_T_user_URLLC_mbps", 2.0),
        }
        self.credit_tiers: List[Dict[str, float]] = sla.get("credit_tiers", [
            {"threshold": 0.05, "fraction": 0.00},
            {"threshold": 0.15, "fraction": 0.05},
            {"threshold": 0.30, "fraction": 0.10},
            {"threshold": 1.00, "fraction": 0.20},
        ])
        self.credit_cap: float = sla.get("credit_cap_fraction", 0.30)

    def compute_violation_rate(self, avg_throughputs_per_step: np.ndarray,
                               slice_name: str) -> float:
        if len(avg_throughputs_per_step) == 0:
            return 0.0
        slo = self.SLO_T.get(slice_name, 0.0)
        violations = np.sum(avg_throughputs_per_step < slo)
        K = len(avg_throughputs_per_step)
        return float(violations / K)

    def _credit_fraction(self, V_s: float) -> float:
        frac = 0.0
        for tier in self.credit_tiers:
            if V_s <= tier["threshold"]:
                frac = tier["fraction"]
                break
            frac = tier["fraction"]
        return frac

    def compute_credit(self, V_s: float, F_s: float, N_active_s: int) -> float:
        frac = self._credit_fraction(V_s)
        mrc = F_s * max(N_active_s, 0)
        credit = frac * mrc
        cap = self.credit_cap * mrc
        return min(credit, cap)


class EnergyModel:
    """Base station energy cost (Section 14)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        e = cfg.get("energy", {})
        self.P0_kw: float = e.get("P0_kw", 0.8)
        self.P1_kw: float = e.get("P1_kw", 1.4)
        self.hours_per_month: float = e.get("hours_per_month", 730.0)
        self.elec_price: float = e.get("elec_price_krw_per_kwh", 120.0)

    def power_kw(self, load: float) -> float:
        load = float(np.clip(load, 0.0, 1.0))
        return self.P0_kw + (self.P1_kw - self.P0_kw) * load

    def compute_cost(self, avg_load: float) -> float:
        p_avg = self.power_kw(avg_load)
        return p_avg * self.hours_per_month * self.elec_price


class EconomicsModel:
    """Full economics: revenue, cost, profit, reward (Section 15).

    Phase 3 (M9) + FIX F3/F4: Enhanced reward with auxiliary shaping terms:
      r = f(profit/scale)
          + α_ret   × retention_bonus
          + α_eff   × efficiency_bonus
          − α_churn × churn_penalty
          − α_vol   × price_volatility
          − α_vrate × vrate_penalty        [FIX F3: NEW]
          − λ       × safety_penalty
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        ec = cfg.get("economics", {})
        self.unit_cost_prb: float = ec.get("unit_cost_prb_krw", 50.0)
        self.profit_scale: float = ec.get("profit_scale", 5000000.0)
        self.lambda_penalty: float = ec.get("lambda_penalty", 10.0)
        self.reward_clip: float = ec.get("reward_clip", 2.0)
        self.prb_total: int = cfg.get("radio", {}).get("prb_total", 273)

        self.reward_type: str = ec.get("reward_type", "log")
        if self.reward_type not in _VALID_REWARD_TYPES:
            raise ValueError(
                f"Unknown reward_type: '{self.reward_type}'. "
                f"Must be one of {_VALID_REWARD_TYPES}"
            )

        # M9 + FIX F3/F4: Reward shaping weights
        shaping = ec.get("reward_shaping", {})
        self.alpha_retention: float = shaping.get("alpha_retention", 0.15)
        self.alpha_efficiency: float = shaping.get("alpha_efficiency", 0.10)
        # FIX F4: alpha_churn increased 0.20 → 0.40 [CHURN_SLR]
        self.alpha_churn: float = shaping.get("alpha_churn", 0.40)
        self.alpha_volatility: float = shaping.get("alpha_volatility", 0.10)
        # FIX F3: NEW — explicit V_rate penalty [NG_1999]
        self.alpha_vrate: float = shaping.get("alpha_vrate", 0.30)

        # Reference user counts for retention normalization
        pop = cfg.get("population", {})
        self._N_ref_eMBB: int = pop.get("N0_eMBB", 120)
        self._N_ref_URLLC: int = pop.get("N0_URLLC", 30)

        self.sla = SLAModel(cfg)
        self.energy = EnergyModel(cfg)

    def compute_resource_cost(self, mean_rho_util: float) -> float:
        return self.unit_cost_prb * self.prb_total * max(mean_rho_util, 0.0)

    def compute_profit(self, fees: Dict[str, float], N_active: Dict[str, int],
                       n_topups: Dict[str, int], topup_price: float,
                       V_rates: Dict[str, float], mean_rho_util: float,
                       avg_load: float) -> Dict[str, float]:
        rev_sub = sum(
            fees.get(s, 0.0) * max(N_active.get(s, 0), 0)
            for s in ["eMBB", "URLLC"]
        )
        rev_topup = sum(
            n_topups.get(s, 0) * topup_price
            for s in ["eMBB", "URLLC"]
        )
        revenue = rev_sub + rev_topup

        cost_energy = self.energy.compute_cost(avg_load)
        cost_sla: Dict[str, float] = {}
        for s in ["eMBB", "URLLC"]:
            cost_sla[s] = self.sla.compute_credit(
                V_s=V_rates.get(s, 0.0),
                F_s=fees.get(s, 0.0),
                N_active_s=N_active.get(s, 0),
            )
        cost_sla_total = sum(cost_sla.values())
        cost_resource = self.compute_resource_cost(mean_rho_util)
        cost_total = cost_energy + cost_sla_total + cost_resource
        profit = revenue - cost_total

        return {
            "revenue": revenue, "revenue_sub": rev_sub,
            "revenue_topup": rev_topup, "cost_energy": cost_energy,
            "cost_sla_eMBB": cost_sla["eMBB"],
            "cost_sla_URLLC": cost_sla["URLLC"],
            "cost_sla_total": cost_sla_total,
            "cost_resource": cost_resource,
            "cost_total": cost_total, "profit": profit,
        }

    @staticmethod
    def _compute_reward_raw(profit: float, reward_type: str,
                            profit_scale: float, reward_clip: float) -> float:
        scale = max(abs(profit_scale), 1.0)
        if reward_type == "tanh":
            return float(np.tanh(profit / scale))
        elif reward_type == "linear":
            return float(np.clip(profit / scale, -reward_clip, reward_clip))
        elif reward_type == "log":
            return float(np.sign(profit) * np.log1p(abs(profit) / scale))
        else:
            raise ValueError(f"Unknown reward_type: '{reward_type}'.")

    def compute_retention_bonus(self, N_active: Dict[str, int]) -> float:
        """M9: Reward for maintaining user base relative to initial count."""
        ratio_e = min(N_active.get("eMBB", 0) / max(self._N_ref_eMBB, 1), 1.5)
        ratio_u = min(N_active.get("URLLC", 0) / max(self._N_ref_URLLC, 1), 1.5)
        return 0.7 * ratio_e + 0.3 * ratio_u

    def compute_efficiency_bonus(self, rho_util: Dict[str, float]) -> float:
        """M9: Reward for good resource utilization (0.3-0.8 is ideal)."""
        bonus = 0.0
        for s in ["eMBB", "URLLC"]:
            rho = rho_util.get(s, 0.0)
            bonus += np.exp(-((rho - 0.55) ** 2) / (2 * 0.15 ** 2))
        return bonus / 2.0

    def compute_churn_penalty(self, n_churns: Dict[str, int],
                              N_active: Dict[str, int]) -> float:
        """M9 + FIX F4: Penalty for churn rate exceeding target.

        FIX F4: Threshold lowered from 5% to 3.5% (midpoint of eMBB/URLLC
        targets) so the penalty activates earlier, discouraging the agent
        from accepting high churn for maximum fees.
        """
        total_churn = sum(n_churns.get(s, 0) for s in ["eMBB", "URLLC"])
        total_active = sum(max(N_active.get(s, 0), 1) for s in ["eMBB", "URLLC"])
        churn_rate = total_churn / total_active
        # FIX F4: threshold 0.05 → 0.035 (closer to calibration targets)
        return max(0.0, churn_rate - 0.035)

    def compute_volatility_penalty(self, fee_delta: Dict[str, float]) -> float:
        """M9: Penalty for large price changes between months."""
        total = 0.0
        for s in ["eMBB", "URLLC"]:
            norm_delta = abs(fee_delta.get(s, 0.0)) / 70000.0
            total += norm_delta
        return total / 2.0

    def compute_vrate_penalty(self, V_rates: Dict[str, float]) -> float:
        """FIX F3: Explicit SLA violation rate penalty.  [NG_1999]

        Returns the weighted average V_rate across slices, giving the
        agent a continuous gradient signal to minimize violations.

        URLLC violations are weighted more heavily (0.6 vs 0.4 for eMBB)
        to reflect the higher reliability requirement of URLLC slices.
        [TS23501] §5.7 specifies that URLLC has stricter QoS requirements.
        """
        v_embb = V_rates.get("eMBB", 0.0)
        v_urllc = V_rates.get("URLLC", 0.0)
        return 0.4 * v_embb + 0.6 * v_urllc

    def compute_reward(self, profit: float, penalty: float = 0.0,
                       shaping: Optional[Dict[str, float]] = None) -> float:
        """Compute shaped reward (M9 enhanced + FIX F3/F4).

        Args:
            profit: Raw monthly profit.
            penalty: Safety penalty from validate_state.
            shaping: Optional dict with keys:
                retention_bonus, efficiency_bonus, churn_penalty,
                volatility_penalty, vrate_penalty  [FIX F3: NEW]
        """
        if not np.isfinite(profit):
            logger.warning("Non-finite profit: %s → using 0", profit)
            profit = 0.0
        raw = self._compute_reward_raw(
            profit, self.reward_type, self.profit_scale, self.reward_clip,
        )

        # M9 + FIX F3: Add shaping terms
        if shaping is not None:
            raw += self.alpha_retention * shaping.get("retention_bonus", 0.0)
            raw += self.alpha_efficiency * shaping.get("efficiency_bonus", 0.0)
            raw -= self.alpha_churn * shaping.get("churn_penalty", 0.0)
            raw -= self.alpha_volatility * shaping.get("volatility_penalty", 0.0)
            # FIX F3: NEW — V_rate penalty
            raw -= self.alpha_vrate * shaping.get("vrate_penalty", 0.0)

        r = raw - self.lambda_penalty * penalty
        r = float(np.clip(r, -self.reward_clip, self.reward_clip))
        return r
