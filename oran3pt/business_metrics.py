"""
Business KPI computation utilities for the O-RAN 3PT dashboard.

Transforms step-level rollout data (rollout_log.csv) and CLV analysis
(clv_report.csv) into business-oriented metrics suitable for executive
reporting and non-technical stakeholders.

All monetary values are in KRW. Time conversion: 1 step = 1 day,
T = steps_per_cycle = 30 steps = 1 month (billing cycle).

References:
  [Grubb AER 2009]      3-part tariff
  [Gupta JSR 2006]      CLV computation
  [Nevo 2016]           Demand elasticity / overage pricing
  [Kim & Yoon 2004]     Churn determinants
  [Henderson AAAI 2018] Baseline comparison methodology
  [Dulac-Arnold 2021]   Operational dashboard design
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ── Section A: Executive KPI Strip ─────────────────────────────────

def compute_executive_kpis(df: pd.DataFrame,
                           clv_df: Optional[pd.DataFrame] = None,
                           T: int = 30) -> Dict[str, float]:
    """Compute 6 executive KPI values from rollout data.

    Args:
        df: rollout_log.csv DataFrame.
        clv_df: clv_report.csv DataFrame (1 row).
        T: steps per billing cycle (default 30 = 1 month).

    Returns:
        Dict with keys: monthly_profit_M_KRW, profit_margin_pct,
        monthly_retention_pct, arpu_K_KRW, sla_compliance_pct, clv_K_KRW.
    """
    mean_profit = df["profit"].mean()
    mean_revenue = df["revenue"].mean()
    mean_N = df["N_active"].mean()
    mean_churn_rate = (df["n_churn"] / df["N_active"].clip(lower=1)).mean()
    mean_pviol_E = df["pviol_E"].mean()

    clv_val = 0.0
    if clv_df is not None and len(clv_df) > 0 and "CLV_per_user" in clv_df.columns:
        clv_val = float(clv_df["CLV_per_user"].iloc[0])

    return {
        "monthly_profit_M_KRW": mean_profit * T / 1e6,
        "profit_margin_pct": float(df["profit_margin"].mean() * 100)
            if "profit_margin" in df.columns
            else (mean_profit / max(mean_revenue, 1e-6)) * 100,
        "monthly_retention_pct": (1.0 - mean_churn_rate) ** T * 100,
        "arpu_K_KRW": (mean_revenue * T) / max(mean_N, 1) / 1e3,
        "sla_compliance_pct": (1.0 - mean_pviol_E) * 100,
        "clv_K_KRW": clv_val / 1e3,
    }


# ── Section B: Profitability Analysis ──────────────────────────────

def compute_pl_waterfall(df: pd.DataFrame,
                         T: int = 30) -> Dict[str, float]:
    """Compute monthly P&L waterfall components.

    Derives per-slice revenue breakdown from existing columns:
      base_rev_U = F_U * N_U / T
      base_rev_E = F_E * N_E / T
      over_rev_U = over_rev - over_rev_E

    Returns monthly values (step mean × T).
    """
    # Derive per-slice revenue
    base_rev_U = df["F_U"] * df["N_U"] / T
    base_rev_E = df["F_E"] * df["N_E"] / T
    over_rev_U = df["over_rev"] - df.get("over_rev_E", 0.0)

    over_rev_E = df["over_rev_E"] if "over_rev_E" in df.columns else (
        df["over_rev"] - over_rev_U)

    return {
        # Revenue components (monthly)
        "base_rev_U": float(base_rev_U.mean() * T),
        "base_rev_E": float(base_rev_E.mean() * T),
        "over_rev_U": float(over_rev_U.mean() * T),
        "over_rev_E": float(over_rev_E.mean() * T),
        "total_revenue": float(df["revenue"].mean() * T),
        # Cost components (monthly)
        "cost_opex": float(df["cost_opex"].mean() * T),
        "cost_energy": float(df["cost_energy"].mean() * T),
        "cost_cac": float(df["cost_cac"].mean() * T),
        "sla_penalty": float(df["sla_penalty"].mean() * T),
        "total_cost": float(df["cost_total"].mean() * T),
        # Bottom line
        "net_profit": float(df["profit"].mean() * T),
    }


def compute_revenue_breakdown(df: pd.DataFrame,
                               T: int = 30) -> Dict[str, float]:
    """Compute revenue source ratios (4-way: URLLC/eMBB × base/overage).

    Returns percentage shares summing to 100%.
    """
    base_rev_U = (df["F_U"] * df["N_U"] / T).mean()
    base_rev_E = (df["F_E"] * df["N_E"] / T).mean()
    over_rev_E = df["over_rev_E"].mean() if "over_rev_E" in df.columns else 0.0
    total_over = df["over_rev"].mean()
    over_rev_U = total_over - over_rev_E

    total = base_rev_U + base_rev_E + over_rev_U + over_rev_E
    total = max(total, 1e-6)

    return {
        "base_U_pct": base_rev_U / total * 100,
        "base_E_pct": base_rev_E / total * 100,
        "over_U_pct": over_rev_U / total * 100,
        "over_E_pct": over_rev_E / total * 100,
        "base_total_pct": (base_rev_U + base_rev_E) / total * 100,
        "over_total_pct": (over_rev_U + over_rev_E) / total * 100,
    }


def compute_slice_economics(df: pd.DataFrame,
                             cfg: Optional[Dict[str, Any]] = None,
                             T: int = 30) -> Dict[str, Any]:
    """Compute per-slice unit economics table.

    Returns dict with 'urllc' and 'embb' sub-dicts containing
    subscriber count, fees, ARPU, and revenue contribution.
    """
    mean_N_U = df["N_U"].mean()
    mean_N_E = df["N_E"].mean()
    mean_F_U = df["F_U"].mean()
    mean_F_E = df["F_E"].mean()
    mean_p_over_U = df["p_over_U"].mean()
    mean_p_over_E = df["p_over_E"].mean()

    # Per-slice revenue
    rev_U = (df["F_U"] * df["N_U"] / T).mean()
    over_rev_E = df["over_rev_E"].mean() if "over_rev_E" in df.columns else 0.0
    rev_E = (df["F_E"] * df["N_E"] / T).mean() + over_rev_E
    rev_U += (df["over_rev"].mean() - over_rev_E)  # add URLLC overage

    total_rev = df["revenue"].mean()
    total_rev = max(total_rev, 1e-6)

    # Data allowances from config
    Q_U = 5.0
    Q_E = 50.0
    if cfg:
        Q_U = cfg.get("tariff", {}).get("Q_U_gb", 5.0)
        Q_E = cfg.get("tariff", {}).get("Q_E_gb", 50.0)

    return {
        "urllc": {
            "subscribers": round(mean_N_U, 1),
            "base_fee_KRW": round(mean_F_U),
            "allowance_gb": Q_U,
            "overage_price_KRW_GB": round(mean_p_over_U),
            "arpu_K_KRW": round(rev_U * T / max(mean_N_U, 1) / 1e3, 1),
            "revenue_share_pct": round(rev_U / total_rev * 100, 1),
        },
        "embb": {
            "subscribers": round(mean_N_E, 1),
            "base_fee_KRW": round(mean_F_E),
            "allowance_gb": Q_E,
            "overage_price_KRW_GB": round(mean_p_over_E),
            "arpu_K_KRW": round(rev_E * T / max(mean_N_E, 1) / 1e3, 1),
            "revenue_share_pct": round(rev_E / total_rev * 100, 1),
        },
    }


# ── Section C: Customer Market Analysis ────────────────────────────

def compute_monthly_subscribers(df: pd.DataFrame,
                                T: int = 30) -> pd.DataFrame:
    """Aggregate step-level data to monthly subscriber metrics.

    Returns DataFrame with columns: month, avg_active, total_joins,
    total_churns, net_change, avg_N_U, avg_N_E.
    """
    df_c = df.copy()
    df_c["month"] = (df_c["step"] - 1) // T + 1

    monthly = df_c.groupby("month").agg(
        avg_active=("N_active", "mean"),
        total_joins=("n_join", "sum"),
        total_churns=("n_churn", "sum"),
        avg_N_U=("N_U", "mean"),
        avg_N_E=("N_E", "mean"),
    ).reset_index()
    monthly["net_change"] = monthly["total_joins"] - monthly["total_churns"]
    return monthly


def compute_price_churn_correlation(df: pd.DataFrame) -> Dict[str, float]:
    """Compute correlation between pricing actions and churn rate.

    Returns Pearson correlation coefficients.
    """
    churn_rate = df["n_churn"] / df["N_active"].clip(lower=1)
    return {
        "corr_FE_churn": float(df["F_E"].corr(churn_rate)),
        "corr_FU_churn": float(df["F_U"].corr(churn_rate)),
        "corr_poverE_churn": float(df["p_over_E"].corr(churn_rate)),
        "corr_poverU_churn": float(df["p_over_U"].corr(churn_rate)),
    }


# ── Section D: Service Quality ─────────────────────────────────────

def compute_sla_compliance(df: pd.DataFrame) -> Dict[str, float]:
    """Compute SLA compliance metrics per slice.

    Returns compliance percentages and violation statistics.
    """
    return {
        "urllc_compliance_pct": (1.0 - df["pviol_U"].mean()) * 100,
        "embb_compliance_pct": (1.0 - df["pviol_E"].mean()) * 100,
        "urllc_max_pviol": float(df["pviol_U"].max()),
        "embb_max_pviol": float(df["pviol_E"].max()),
        "embb_gt15_pct": float((df["pviol_E"] > 0.15).mean() * 100),
        "embb_gt50_pct": float((df["pviol_E"] > 0.50).mean() * 100),
    }


def compute_capacity_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute capacity utilization and headroom per slice.

    Returns utilization percentages and remaining capacity.
    """
    urllc_util = (df["L_U"] / df["C_U"].clip(lower=1e-6)).mean()
    embb_util = (df["L_E"] / df["C_E"].clip(lower=1e-6)).mean()

    return {
        "urllc_util_pct": urllc_util * 100,
        "embb_util_pct": embb_util * 100,
        "urllc_headroom_pct": (1.0 - urllc_util) * 100,
        "embb_headroom_pct": (1.0 - embb_util) * 100,
        "mean_L_U": float(df["L_U"].mean()),
        "mean_C_U": float(df["C_U"].mean()),
        "mean_L_E": float(df["L_E"].mean()),
        "mean_C_E": float(df["C_E"].mean()),
    }


def estimate_additional_capacity(df: pd.DataFrame,
                                  target_util: float = 0.70
                                  ) -> Dict[str, Any]:
    """Estimate additional subscribers at target utilization.

    Uses average per-user traffic to project remaining capacity.
    """
    mean_L_E = df["L_E"].mean()
    mean_C_E = df["C_E"].mean()
    mean_N_E = df["N_E"].mean()

    per_user_traffic = mean_L_E / max(mean_N_E, 1)
    remaining = max(0, mean_C_E * target_util - mean_L_E)
    additional = remaining / max(per_user_traffic, 1e-6)

    mean_L_U = df["L_U"].mean()
    mean_C_U = df["C_U"].mean()
    mean_N_U = df["N_U"].mean()
    per_user_U = mean_L_U / max(mean_N_U, 1)
    remaining_U = max(0, mean_C_U * target_util - mean_L_U)
    additional_U = remaining_U / max(per_user_U, 1e-6)

    return {
        "additional_embb_users": int(additional),
        "additional_urllc_users": int(additional_U),
        "target_utilization_pct": target_util * 100,
        "per_user_traffic_E_gb": round(per_user_traffic, 2),
        "per_user_traffic_U_gb": round(per_user_U, 3),
    }


# ── Section E: AI Performance ─────────────────────────────────────

def compute_annual_projection(df: pd.DataFrame,
                               T: int = 30,
                               n_cells: int = 100
                               ) -> Dict[str, float]:
    """Project annualized financials from evaluation data.

    Args:
        df: rollout_log.csv DataFrame.
        T: steps per cycle (30).
        n_cells: number of cells for scaling projection.

    Returns:
        Dict with annual_profit, scaled profit, etc.
    """
    monthly_profit = df["profit"].mean() * T
    annual_profit = monthly_profit * 12
    monthly_revenue = df["revenue"].mean() * T
    annual_revenue = monthly_revenue * 12

    return {
        "monthly_profit": monthly_profit,
        "annual_profit_single_cell": annual_profit,
        "annual_profit_scaled": annual_profit * n_cells,
        "n_cells": n_cells,
        "monthly_revenue": monthly_revenue,
        "annual_revenue_single_cell": annual_revenue,
    }


# ── Section F: Strategy Summary ────────────────────────────────────

def compute_pricing_strategy_summary(df: pd.DataFrame,
                                      T: int = 30
                                      ) -> Dict[str, Any]:
    """Compute summary statistics for the pricing strategy text block.

    Returns mean/std of all pricing actions and key performance metrics.
    """
    sla_E = (1.0 - df["pviol_E"].mean()) * 100
    sla_U = (1.0 - df["pviol_U"].mean()) * 100
    mean_rho = df["rho_U"].mean()

    return {
        "F_U_mean": float(df["F_U"].mean()),
        "F_U_std": float(df["F_U"].std()),
        "F_E_mean": float(df["F_E"].mean()),
        "F_E_std": float(df["F_E"].std()),
        "p_over_U_mean": float(df["p_over_U"].mean()),
        "p_over_U_std": float(df["p_over_U"].std()),
        "p_over_E_mean": float(df["p_over_E"].mean()),
        "p_over_E_std": float(df["p_over_E"].std()),
        "rho_U_mean": mean_rho,
        "rho_U_std": float(df["rho_U"].std()),
        "sla_E_pct": sla_E,
        "sla_U_pct": sla_U,
        "mean_N_active": float(df["N_active"].mean()),
        "mean_N_U": float(df["N_U"].mean()),
        "mean_N_E": float(df["N_E"].mean()),
    }


# ── Aggregate all metrics ──────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame,
                        clv_df: Optional[pd.DataFrame] = None,
                        cfg: Optional[Dict[str, Any]] = None,
                        T: int = 30) -> Dict[str, Any]:
    """Compute all business metrics from rollout data.

    Convenience function that calls all compute_* functions and
    returns a single nested dict suitable for JSON injection.
    """
    return {
        "kpis": compute_executive_kpis(df, clv_df, T),
        "pl_waterfall": compute_pl_waterfall(df, T),
        "revenue_breakdown": compute_revenue_breakdown(df, T),
        "slice_economics": compute_slice_economics(df, cfg, T),
        "monthly_subscribers": compute_monthly_subscribers(df, T).to_dict(
            orient="records"),
        "price_churn_corr": compute_price_churn_correlation(df),
        "sla_compliance": compute_sla_compliance(df),
        "capacity": compute_capacity_analysis(df),
        "additional_capacity": estimate_additional_capacity(df),
        "annual_projection": compute_annual_projection(df, T),
        "strategy": compute_pricing_strategy_summary(df, T),
    }
