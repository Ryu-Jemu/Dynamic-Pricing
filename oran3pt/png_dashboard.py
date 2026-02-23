"""
Comprehensive PNG Dashboard Generator (§16c).

Generates 7 themed PNG sheets (28 panels) covering ALL metrics recorded
during SAC training and evaluation.  Designed for beginner readability
with annotated thresholds, config-driven bounds, and colorblind-safe
palette.

REVISION 12.5 — [M11] + [DASH-2] individual panel output.

Sheets:
  1. Financial Overview (revenue, cost, profit, margins)
  2. Reward Decomposition (reward signal + shaping components)
  3. Market Dynamics (users, churn, join, per-slice)
  4. Pricing Actions (F_U, F_E, p_over_U, p_over_E, rho_U)
  5. Network & QoS (load vs capacity, violation probabilities)
  6. Billing & Usage (cycle usage vs allowance, overage, elasticity)
  7. Multi-Seed Convergence (cross-seed envelope, training only)

Usage:
  python -m oran3pt.png_dashboard --output outputs --config config/default.yaml
  python -m oran3pt.png_dashboard --output outputs --mode eval --dpi 200
  python -m oran3pt.png_dashboard --output outputs --layout individual
  python -m oran3pt.png_dashboard --output outputs --layout both

References:
  [Wong Nat.Methods 2011]   Color-blind safe palette
  [Henderson AAAI 2018]     Multi-seed convergence analysis
  [Dulac-Arnold JMLR 2021]  Observability for real-world RL
  [Tufte 2001]              Data-ink ratio, small multiples
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("oran3pt.png_dashboard")

# ── Color-blind safe palette [Wong Nat.Methods 2011] ──────────────────
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
CYAN = "#56B4E9"
YELLOW = "#F0E442"
GRAY = "#999999"
DARK_BLUE = "#003f5c"
DARK_RED = "#a4262c"
LIGHT_RED = "#D55E00"
LIGHT_GREEN = "#009E73"

# Semantic color map (consistent across all sheets)
_COLORS = {
    "revenue": BLUE,
    "cost": RED,
    "profit": GREEN,
    "reward": ORANGE,
    "base_rev": BLUE,
    "over_rev": DARK_BLUE,
    "cost_opex": RED,
    "cost_energy": ORANGE,
    "cost_cac": PURPLE,
    "sla_penalty": YELLOW,
    "smooth_penalty": ORANGE,
    "retention_penalty": RED,
    "pop_bonus": GREEN,
    "lagrangian_penalty": PURPLE,
    "pviol_U": RED,
    "pviol_E": BLUE,
    "N_active": BLUE,
    "N_U": CYAN,
    "N_E": PURPLE,
    "n_join": GREEN,
    "n_churn": RED,
    "F_U": BLUE,
    "F_E": PURPLE,
    "p_over_U": CYAN,
    "p_over_E": ORANGE,
    "rho_U": CYAN,
    "L_U": CYAN,
    "L_E": BLUE,
    "C_U": GREEN,
    "C_E": RED,
}

# Reference strings per sheet group
_REF_01 = ("Refs: [Grubb AER 2009] 3-Part Tariff | "
           "[Nevo Econometrica 2016] Demand Elasticity | "
           "[Tessler ICML 2019] CMDP")
_REF_02 = ("Refs: [Dalal NeurIPS 2018] Action Smoothing | "
           "[Wiewiora ICML 2003] Reward Shaping | "
           "[Mguni AAMAS 2019] Population Reward")
_REF_03 = ("Refs: [Kim & Yoon 2004] Churn Determinants | "
           "[Ahn 2006] Partial Defection | "
           "[Gupta JSR 2006] CLV")
_REF_04 = ("Refs: [Grubb AER 2009] Base Fee | "
           "[Nevo 2016] Overage Price | "
           "[Huang IoT-J 2020] PRB Allocation")
_REF_05 = ("Refs: [3GPP TS 38.104] QoS | "
           "[Tessler ICML 2019] CMDP | "
           "[Huang IoT-J 2020] URLLC-eMBB Coexistence")
_REF_06 = ("Refs: [Nevo Econometrica 2016] Demand Elasticity | "
           "[3GPP TS 23.503] Usage Monitoring | "
           "[Grubb AER 2009] 3-Part Tariff")
_REF_07 = ("Refs: [Henderson AAAI 2018] Multi-Seed Evaluation | "
           "[Narvekar JMLR 2020] Curriculum Learning | "
           "[Haarnoja ICML 2018] SAC")


def _setup_style() -> None:
    """Configure matplotlib rcParams for clean, readable charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": True,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.framealpha": 0.8,
        "lines.linewidth": 1.5,
        "figure.dpi": 100,
    })


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def _load_training_data(output_dir: str) -> Dict[int, pd.DataFrame]:
    """Load all train_log_seed{i}.csv files.

    Returns:
        Dict mapping seed_id -> DataFrame.
    """
    out = Path(output_dir)
    seed_files = sorted(out.glob("train_log_seed*.csv"))
    result = {}
    for f in seed_files:
        stem = f.stem  # e.g. "train_log_seed0"
        try:
            seed_id = int(stem.replace("train_log_seed", ""))
        except ValueError:
            continue
        df = pd.read_csv(f)
        logger.info("  Loaded %s: %d rows, %d cols", f.name, len(df), len(df.columns))
        result[seed_id] = df
    return result


def _load_eval_data(output_dir: str) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]
]:
    """Load rollout_log.csv, eval_summary.csv, clv_report.csv.

    Returns:
        (rollout_df, summary_df, clv_df) — any may be None if missing.
    """
    out = Path(output_dir)
    rollout = summary = clv = None

    rp = out / "rollout_log.csv"
    if rp.exists():
        rollout = pd.read_csv(rp)
        logger.info("  Loaded rollout_log.csv: %d rows", len(rollout))

    sp = out / "eval_summary.csv"
    if sp.exists():
        summary = pd.read_csv(sp)

    cp = out / "clv_report.csv"
    if cp.exists():
        clv = pd.read_csv(cp)

    return rollout, summary, clv


def _detect_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign episode IDs by detecting step counter resets.

    Reuses logic from html_dashboard.py:49-58.
    """
    df = df.copy()
    if "step" not in df.columns:
        raise ValueError("CSV must contain 'step' column")
    resets = (df["step"] <= df["step"].shift(1).fillna(0)).astype(int)
    df["_episode"] = resets.cumsum()
    return df


def _episode_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate step-level data to episode-level means/sums.

    Returns DataFrame indexed by episode with aggregated columns.
    """
    df = _detect_episodes(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    skip = {"_episode", "repeat"}
    cols = [c for c in numeric_cols if c not in skip]

    agg = df.groupby("_episode")[cols].agg(["mean", "std", "sum", "min", "max"])
    # Flatten multi-level columns
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg.index.name = "episode"
    return agg


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived diagnostic columns if not present."""
    df = df.copy()
    if "L_U" in df.columns and "C_U" in df.columns:
        df["urllc_util"] = df["L_U"] / df["C_U"].clip(lower=1e-6)
    if "L_E" in df.columns and "C_E" in df.columns:
        df["embb_util"] = df["L_E"] / df["C_E"].clip(lower=1e-6)
    if "sla_penalty" in df.columns and "revenue" in df.columns:
        df["sla_revenue_ratio"] = df["sla_penalty"] / df["revenue"].clip(lower=1e-6)
    if "profit" in df.columns and "revenue" in df.columns:
        df["profit_margin"] = df["profit"] / df["revenue"].clip(lower=1e-6)
    if "n_join" in df.columns and "n_churn" in df.columns:
        df["population_delta"] = df["n_join"] - df["n_churn"]
    return df


def _rolling_smooth(series: pd.Series, window: int = 100) -> pd.Series:
    """Compute rolling mean, filling initial NaN values with expanding mean."""
    rolled = series.rolling(window, min_periods=1).mean()
    return rolled


def _col(name: str) -> str:
    """Get color for a metric name."""
    return _COLORS.get(name, GRAY)


def _safe_get(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Safely get a column, returning a constant series if missing."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _annotate_threshold(ax, y_val: float, label: str,
                        color: str = GRAY, linestyle: str = "--") -> None:
    """Add a horizontal threshold line with text annotation."""
    ax.axhline(y_val, color=color, linestyle=linestyle, alpha=0.6, linewidth=1.0)
    ax.text(0.02, y_val, f"  {label}",
            transform=ax.get_yaxis_transform(),
            fontsize=7, color=color, va="bottom")


def _callout(ax, text: str) -> None:
    """Add a summary callout in the top-right corner."""
    ax.text(0.97, 0.95, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRAY, alpha=0.85))


def _unavailable(ax, label: str = "Not available") -> None:
    """Show 'not available' message for missing data."""
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color=GRAY)
    ax.set_xticks([])
    ax.set_yticks([])


# ══════════════════════════════════════════════════════════════════════
# PANEL FUNCTIONS — atomic rendering units [DASH-2]
# Each function draws on a single matplotlib Axes object.
# ══════════════════════════════════════════════════════════════════════

# ── Sheet 1: Financial Overview ──────────────────────────────────────

def _panel_01_1_revenue_cost_profit(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 1.1: Revenue, Cost, and Profit trajectory."""
    x = g.index
    ax.plot(x, g["revenue"] / 1e6, color=_col("revenue"), label="Revenue", lw=2)
    ax.plot(x, g["cost_total"] / 1e6, color=_col("cost"), label="Cost", lw=2)
    ax.plot(x, g["profit"] / 1e6, color=_col("profit"), label="Profit", lw=2)
    ax.fill_between(x, 0, g["profit"] / 1e6,
                    where=g["profit"] > 0, alpha=0.08, color=GREEN)
    ax.fill_between(x, 0, g["profit"] / 1e6,
                    where=g["profit"] < 0, alpha=0.08, color=RED)
    ax.set_title("[1.1] Revenue, Cost, and Profit")
    xlabel = "Episode" if mode == "training" else "Step"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("M KRW")
    ax.legend(loc="best")
    mean_profit = g["profit"].mean()
    _callout(ax, f"Mean Profit: {mean_profit/1e6:.2f}M KRW")


def _panel_01_2_revenue_decomposition(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 1.2: Revenue Decomposition (base + overage)."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    base = _safe_get(g, "base_rev")
    over = _safe_get(g, "over_rev")
    if base.sum() > 0 or over.sum() > 0:
        ax.fill_between(x, 0, base / 1e6,
                        color=_col("base_rev"), alpha=0.6, label="Base Revenue (Fees)")
        ax.fill_between(x, base / 1e6, (base + over) / 1e6,
                        color=_col("over_rev"), alpha=0.6, label="Overage Revenue")
        ax.plot(x, g["revenue"] / 1e6, color="black", lw=1, ls="--",
                alpha=0.5, label="Total Revenue")
        ax.set_title("[1.2] Revenue Decomposition")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("M KRW")
        ax.legend(loc="best")
        base_pct = base.sum() / max(g["revenue"].sum(), 1e-6) * 100
        _callout(ax, f"Base: {base_pct:.0f}% | Overage: {100-base_pct:.0f}%")
    else:
        _unavailable(ax, "Revenue breakdown not available (pre-v7 data)")
        ax.set_title("[1.2] Revenue Decomposition")


def _panel_01_3_cost_breakdown(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 1.3: Cost Breakdown (OPEX, Energy, CAC, SLA)."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    opex = _safe_get(g, "cost_opex")
    energy = _safe_get(g, "cost_energy")
    cac = _safe_get(g, "cost_cac")
    sla = _safe_get(g, "sla_penalty")
    if opex.sum() > 0 or energy.sum() > 0:
        bottom = np.zeros(len(x))
        for col_name, series, color in [
            ("OPEX", opex, _col("cost_opex")),
            ("Energy", energy, _col("cost_energy")),
            ("CAC", cac, _col("cost_cac")),
            ("SLA Penalty", sla, _col("sla_penalty")),
        ]:
            vals = series.values / 1e6
            ax.fill_between(x, bottom, bottom + vals,
                            color=color, alpha=0.6, label=col_name)
            bottom = bottom + vals
        ax.plot(x, g["cost_total"] / 1e6, color="black", lw=1, ls="--",
                alpha=0.5, label="Total Cost")
        ax.set_title("[2.1] Cost Breakdown")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("M KRW")
        ax.legend(loc="best")
    else:
        _unavailable(ax, "Cost breakdown not available")
        ax.set_title("[2.1] Cost Breakdown")


def _panel_01_4_profitability_ratios(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 1.4: Profit Margin & SLA/Revenue Ratio (dual axis)."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    margin = g["profit"] / g["revenue"].clip(lower=1e-6) * 100
    ax.plot(x, margin, color=_col("profit"), label="Profit Margin (%)", lw=2)
    ax.set_ylabel("Profit Margin (%)", color=_col("profit"))
    _annotate_threshold(ax, 0, "Break-even", color=GRAY)

    if "sla_penalty" in g.columns:
        ax2 = ax.twinx()
        sla_ratio = g["sla_penalty"] / g["revenue"].clip(lower=1e-6) * 100
        ax2.plot(x, sla_ratio, color=_col("sla_penalty"),
                 ls="--", alpha=0.8, label="SLA/Revenue (%)")
        ax2.set_ylabel("SLA/Revenue (%)", color=_col("sla_penalty"))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend(loc="best")

    ax.set_title("[2.2] Profitability Ratios")
    ax.set_xlabel(xlabel)
    _callout(ax, f"Mean Margin: {margin.mean():.1f}%")


# ── Sheet 2: Reward Decomposition ────────────────────────────────────

def _panel_02_1_reward_over_time(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 2.1: Reward over time with rolling mean."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    ax.plot(x, g["reward"], color=_col("reward"), alpha=0.4, lw=0.8,
            label="Reward (raw)")
    smoothed = _rolling_smooth(g["reward"],
                               window=max(len(g) // 20, 5))
    ax.plot(x, smoothed, color=_col("reward"), lw=2.5,
            label="Reward (smoothed)")
    ax.set_title("[1.1] Total Reward Over Time")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reward")
    _annotate_threshold(ax, 0, "Zero reward", color=GRAY)
    ax.legend(loc="best")
    _callout(ax, f"Mean: {g['reward'].mean():.4f}")


def _panel_02_2_reward_shaping_components(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 2.2: Reward shaping components stacked area."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    components = []
    for col_name, label in [
        ("smooth_penalty", "Smooth Penalty"),
        ("retention_penalty", "Retention Penalty"),
        ("lagrangian_penalty", "Lagrangian Penalty"),
        ("pop_bonus", "Population Bonus"),
    ]:
        if col_name in g.columns and g[col_name].abs().sum() > 1e-10:
            components.append((col_name, label))

    if components:
        for col_name, label in components:
            vals = g[col_name]
            if "penalty" in col_name:
                ax.fill_between(x, 0, -vals, alpha=0.5,
                                color=_col(col_name), label=f"-{label}")
            else:
                ax.fill_between(x, 0, vals, alpha=0.5,
                                color=_col(col_name), label=label)
        ax.set_title("[1.2] Reward Shaping Components")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Magnitude")
        ax.legend(loc="best")
        _annotate_threshold(ax, 0, "Zero", color=GRAY)
    else:
        _unavailable(ax, "Reward components not available")
        ax.set_title("[1.2] Reward Shaping Components")


def _panel_02_3_reward_distribution(
    ax, raw_df: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 2.3: Reward distribution histogram."""
    rewards = raw_df["reward"].dropna()
    ax.hist(rewards, bins=60, color=_col("reward"), alpha=0.7,
            edgecolor="white", density=True)
    mean_r = rewards.mean()
    p10 = np.percentile(rewards, 10)
    p90 = np.percentile(rewards, 90)
    ax.axvline(mean_r, color=RED, linestyle="--", lw=2,
               label=f"Mean = {mean_r:.4f}")
    ax.axvline(p10, color=GRAY, linestyle=":", lw=1,
               label=f"P10 = {p10:.4f}")
    ax.axvline(p90, color=GRAY, linestyle=":", lw=1,
               label=f"P90 = {p90:.4f}")
    ax.set_title("[2.1] Reward Distribution")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.legend(loc="best")


def _panel_02_4_penalty_bonus_trajectories(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 2.4: Individual penalty/bonus time series."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    has_any = False
    for col_name, label, ls in [
        ("smooth_penalty", "Smooth Penalty", "-"),
        ("retention_penalty", "Retention Penalty", "--"),
        ("lagrangian_penalty", "Lagrangian Penalty", "-."),
        ("pop_bonus", "Pop Bonus", "-"),
    ]:
        if col_name in g.columns and g[col_name].abs().sum() > 1e-10:
            ax.plot(x, g[col_name], color=_col(col_name),
                    ls=ls, label=label, lw=1.5)
            has_any = True
    if has_any:
        ax.set_title("[2.2] Penalty & Bonus Trajectories")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        _annotate_threshold(ax, 0, "Zero", color=GRAY)
    else:
        _unavailable(ax, "Penalty/bonus data not available")
        ax.set_title("[2.2] Penalty & Bonus Trajectories")


# ── Sheet 3: Market Dynamics ─────────────────────────────────────────

def _panel_03_1_active_users(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 3.1: Active Users vs Target."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    N_total = cfg.get("population", {}).get("N_total", 500)
    target_ratio = cfg.get("population_reward", {}).get("target_ratio", 0.4)
    ax.plot(x, g["N_active"], color=_col("N_active"), lw=2, label="N_active")
    _annotate_threshold(ax, N_total, f"N_total = {N_total}", color=GRAY)
    _annotate_threshold(ax, target_ratio * N_total,
                        f"Target = {target_ratio*N_total:.0f} "
                        f"({target_ratio*100:.0f}%)", color=GREEN)
    ax.set_title("[1.1] Active User Population")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")
    ax.legend(loc="best")
    _callout(ax, f"Mean: {g['N_active'].mean():.0f} users")


def _panel_03_2_per_slice_subscribers(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 3.2: Per-Slice Subscribers (URLLC/eMBB)."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    if "N_U" in g.columns and "N_E" in g.columns:
        ax.plot(x, g["N_U"], color=_col("N_U"), lw=2, label="N_U (URLLC)")
        ax.plot(x, g["N_E"], color=_col("N_E"), lw=2, label="N_E (eMBB)")
        ax.set_title("[1.2] Per-Slice Subscribers")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Users")
        ax.legend(loc="best")
        _callout(ax, f"URLLC: {g['N_U'].mean():.0f} | eMBB: {g['N_E'].mean():.0f}")
    else:
        _unavailable(ax, "Per-slice user counts not available")
        ax.set_title("[1.2] Per-Slice Subscribers")


def _panel_03_3_market_flows(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 3.3: Market Flows — Joins & Churns."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    width = max(1, (x.max() - x.min()) / len(x) * 0.35)
    ax.bar(x, g["n_join"], width=width, color=_col("n_join"),
           alpha=0.7, label="Joins")
    ax.bar(x, -g["n_churn"], width=width, color=_col("n_churn"),
           alpha=0.7, label="Churns (inverted)")
    net_flow = g["n_join"] - g["n_churn"]
    ax.plot(x, net_flow, color=BLUE, lw=1.5, label="Net Flow")
    _annotate_threshold(ax, 0, "Zero net flow", color=GRAY)
    ax.set_title("[2.1] Market Flows: Joins & Churns")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users per step")
    ax.legend(loc="best")


def _panel_03_4_churn_rate(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 3.4: Churn Rate Over Time."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    churn_rate = g["n_churn"] / g["N_active"].clip(lower=1)
    ax.plot(x, churn_rate * 100, color=_col("n_churn"), lw=1.5,
            label="Churn Rate (%)")
    churn_target = cfg.get("calibration", {}).get("churn_target_monthly", 0.03)
    T = cfg.get("time", {}).get("steps_per_cycle", 30)
    n_cycles = cfg.get("time", {}).get("episode_cycles", 24)
    daily_churn_target = 1 - (1 - churn_target) ** (1 / (T * n_cycles))
    _annotate_threshold(ax, daily_churn_target * 100,
                        f"Target: {daily_churn_target*100:.3f}%/step",
                        color=GREEN)
    ax.set_title("[2.2] Churn Rate Over Time")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Churn Rate (%)")
    ax.legend(loc="best")
    _callout(ax, f"Mean: {churn_rate.mean()*100:.3f}%/step")


# ── Sheet 4: Pricing Actions ─────────────────────────────────────────

def _panel_04_1_urllc_pricing(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 4.1: URLLC Pricing (F_U + p_over_U dual axis)."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    acfg = cfg.get("action", {})
    ax.plot(x, g["F_U"] / 1e3, color=_col("F_U"), lw=2, label="F_U (K KRW)")
    ax.fill_between(x, acfg.get("F_U_min", 30000) / 1e3,
                    acfg.get("F_U_max", 140000) / 1e3,
                    color=_col("F_U"), alpha=0.06, label="F_U bounds")
    ax.set_ylabel("F_U (K KRW)", color=_col("F_U"))
    ax2 = ax.twinx()
    ax2.plot(x, g["p_over_U"], color=_col("p_over_U"), lw=1.5,
             ls="--", label="p_over_U (KRW/GB)")
    ax2.set_ylabel("p_over_U (KRW/GB)", color=_col("p_over_U"))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.set_title("[1.1] URLLC Pricing")
    ax.set_xlabel(xlabel)


def _panel_04_2_embb_pricing(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 4.2: eMBB Pricing (F_E + p_over_E dual axis)."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    acfg = cfg.get("action", {})
    ax.plot(x, g["F_E"] / 1e3, color=_col("F_E"), lw=2, label="F_E (K KRW)")
    ax.fill_between(x, acfg.get("F_E_min", 35000) / 1e3,
                    acfg.get("F_E_max", 100000) / 1e3,
                    color=_col("F_E"), alpha=0.06, label="F_E bounds")
    ax.set_ylabel("F_E (K KRW)", color=_col("F_E"))
    ax2 = ax.twinx()
    ax2.plot(x, g["p_over_E"], color=_col("p_over_E"), lw=1.5,
             ls="--", label="p_over_E (KRW/GB)")
    ax2.set_ylabel("p_over_E (KRW/GB)", color=_col("p_over_E"))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.set_title("[1.2] eMBB Pricing")
    ax.set_xlabel(xlabel)


def _panel_04_3_rho_u_allocation(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 4.3: rho_U Resource Allocation."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    acfg = cfg.get("action", {})
    rho_min = acfg.get("rho_U_min", 0.03)
    rho_max = acfg.get("rho_U_max", 0.10)
    ax.plot(x, g["rho_U"], color=_col("rho_U"), lw=2, label="rho_U")
    ax.fill_between(x, rho_min, rho_max,
                    color=_col("rho_U"), alpha=0.08,
                    label=f"Bounds [{rho_min:.2f}, {rho_max:.2f}]")
    ax.set_title("[2.1] URLLC PRB Allocation (rho_U)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rho_U")
    ax.legend(loc="best")
    _callout(ax, f"Mean: {g['rho_U'].mean():.3f}")


def _panel_04_4_action_distributions(
    ax, raw_df: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 4.4: Action Distributions (normalized box plots)."""
    acfg = cfg.get("action", {})
    action_data = []
    action_labels = []
    action_colors = []
    for col, label, color in [
        ("F_U", "F_U\n(K KRW)", _col("F_U")),
        ("p_over_U", "p_over_U\n(KRW/GB)", _col("p_over_U")),
        ("F_E", "F_E\n(K KRW)", _col("F_E")),
        ("p_over_E", "p_over_E\n(KRW/GB)", _col("p_over_E")),
        ("rho_U", "rho_U", _col("rho_U")),
    ]:
        if col in raw_df.columns:
            vals = raw_df[col].dropna()
            lo = acfg.get(f"{col}_min", vals.min()) if col != "rho_U" \
                else acfg.get("rho_U_min", 0.03)
            hi = acfg.get(f"{col}_max", vals.max()) if col != "rho_U" \
                else acfg.get("rho_U_max", 0.10)
            if col.startswith("F_"):
                lo = acfg.get(f"{col}_min", vals.min())
                hi = acfg.get(f"{col}_max", vals.max())
            elif col.startswith("p_over"):
                lo = acfg.get(f"{col}_min", vals.min())
                hi = acfg.get(f"{col}_max", vals.max())
            norm_vals = (vals - lo) / max(hi - lo, 1e-6)
            action_data.append(norm_vals.values)
            action_labels.append(label)
            action_colors.append(color)

    if action_data:
        bp = ax.boxplot(action_data, tick_labels=action_labels, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, color in zip(bp["boxes"], action_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title("[2.2] Action Distributions (Normalized to [0,1])")
        ax.set_ylabel("Normalized Action Value")
        _annotate_threshold(ax, 0.5, "Midpoint", color=GRAY)


# ── Sheet 5: Network & QoS ──────────────────────────────────────────

def _panel_05_1_urllc_load_capacity(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 5.1: URLLC Load vs Capacity."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    if "L_U" in g.columns and "C_U" in g.columns:
        ax.plot(x, g["L_U"], color=_col("L_U"), lw=1.5, label="L_U (load)")
        ax.plot(x, g["C_U"], color=_col("C_U"), lw=1.5, label="C_U (capacity)")
        ax.fill_between(x, g["L_U"], g["C_U"],
                        where=g["L_U"] > g["C_U"],
                        color=RED, alpha=0.2, label="Overload zone")
        ax.fill_between(x, g["L_U"], g["C_U"],
                        where=g["L_U"] <= g["C_U"],
                        color=GREEN, alpha=0.05)
        ax.set_title("[1.1] URLLC Load vs Capacity")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GB/step")
        ax.legend(loc="best")
        headroom = ((g["C_U"] - g["L_U"]) / g["C_U"].clip(lower=1e-6)).mean() * 100
        _callout(ax, f"Mean headroom: {headroom:.0f}%")
    else:
        _unavailable(ax, "URLLC load/capacity not available")
        ax.set_title("[1.1] URLLC Load vs Capacity")


def _panel_05_2_embb_load_capacity(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 5.2: eMBB Load vs Capacity."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    if "L_E" in g.columns and "C_E" in g.columns:
        ax.plot(x, g["L_E"], color=_col("L_E"), lw=1.5, label="L_E (load)")
        ax.plot(x, g["C_E"], color=_col("C_E"), lw=1.5, label="C_E (capacity)")
        ax.fill_between(x, g["L_E"], g["C_E"],
                        where=g["L_E"] > g["C_E"],
                        color=RED, alpha=0.2, label="Overload zone")
        ax.fill_between(x, g["L_E"], g["C_E"],
                        where=g["L_E"] <= g["C_E"],
                        color=GREEN, alpha=0.05)
        ax.set_title("[1.2] eMBB Load vs Capacity")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GB/step")
        ax.legend(loc="best")
        overload_pct = (g["L_E"] > g["C_E"]).mean() * 100
        _callout(ax, f"Overload: {overload_pct:.1f}% of steps")
    else:
        _unavailable(ax, "eMBB load/capacity not available")
        ax.set_title("[1.2] eMBB Load vs Capacity")


def _panel_05_3_qos_violation_prob(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 5.3: QoS Violation Probability."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    pviol_threshold = cfg.get("lagrangian_qos", {}).get("pviol_E_threshold", 0.08)
    ax.plot(x, g["pviol_U"], color=_col("pviol_U"), lw=2,
            label="pviol_U (URLLC)")
    ax.plot(x, g["pviol_E"], color=_col("pviol_E"), lw=2,
            label="pviol_E (eMBB)")
    _annotate_threshold(ax, pviol_threshold,
                        f"Lagrangian threshold = {pviol_threshold}",
                        color=DARK_RED)
    ax.fill_between(x, pviol_threshold, g["pviol_E"],
                    where=g["pviol_E"] > pviol_threshold,
                    color=RED, alpha=0.15, label="Constraint violation")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("[2.1] QoS Violation Probability")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("p_viol")
    ax.legend(loc="best")
    _callout(ax, f"Mean pviol_E: {g['pviol_E'].mean():.3f}")


def _panel_05_4_slice_utilization(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 5.4: Slice Utilization Ratios."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    if "L_U" in g.columns and "C_U" in g.columns:
        util_U = g["L_U"] / g["C_U"].clip(lower=1e-6)
        ax.plot(x, util_U, color=_col("L_U"), lw=1.5,
                label="URLLC utilization (L_U/C_U)")
    if "L_E" in g.columns and "C_E" in g.columns:
        util_E = g["L_E"] / g["C_E"].clip(lower=1e-6)
        ax.plot(x, util_E, color=_col("L_E"), lw=1.5,
                label="eMBB utilization (L_E/C_E)")
    _annotate_threshold(ax, 1.0, "Congestion boundary (1.0)", color=DARK_RED)
    ax.set_title("[2.2] Slice Utilization Ratios")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Load / Capacity")
    ax.legend(loc="best")


# ── Sheet 6: Billing & Usage ─────────────────────────────────────────

def _panel_06_1_urllc_cycle_usage(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 6.1: URLLC Cycle Usage vs Allowance."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    Q_U = cfg.get("tariff", {}).get("Q_U_gb", 5.0)
    if "cycle_usage_U" in g.columns and "N_U" in g.columns:
        ax.plot(x, g["cycle_usage_U"], color=_col("L_U"), lw=1.5,
                label="Cycle Usage (GB)")
        allowance_U = Q_U * g["N_U"]
        ax.plot(x, allowance_U, color=GREEN, ls="--", lw=1,
                label=f"Allowance (Q_U={Q_U}GB x N_U)")
        ax.fill_between(x, g["cycle_usage_U"], allowance_U,
                        where=g["cycle_usage_U"] > allowance_U,
                        color=RED, alpha=0.15, label="Overage zone")
        ax.set_title("[1.1] URLLC: Cycle Usage vs Allowance")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GB")
        ax.legend(loc="best")
    elif "cycle_usage_U" in g.columns:
        ax.plot(x, g["cycle_usage_U"], color=_col("L_U"), lw=1.5,
                label="Cycle Usage (GB)")
        ax.set_title("[1.1] URLLC: Cycle Usage")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GB")
        ax.legend(loc="best")
    else:
        _unavailable(ax, "Cycle usage data not available")
        ax.set_title("[1.1] URLLC: Cycle Usage vs Allowance")


def _panel_06_2_embb_cycle_usage(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 6.2: eMBB Cycle Usage vs Allowance."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    Q_E = cfg.get("tariff", {}).get("Q_E_gb", 50.0)
    if "cycle_usage_E" in g.columns and "N_E" in g.columns:
        ax.plot(x, g["cycle_usage_E"], color=_col("L_E"), lw=1.5,
                label="Cycle Usage (GB)")
        allowance_E = Q_E * g["N_E"]
        ax.plot(x, allowance_E, color=GREEN, ls="--", lw=1,
                label=f"Allowance (Q_E={Q_E}GB x N_E)")
        ax.fill_between(x, g["cycle_usage_E"], allowance_E,
                        where=g["cycle_usage_E"] > allowance_E,
                        color=RED, alpha=0.15, label="Overage zone")
        ax.set_title("[1.2] eMBB: Cycle Usage vs Allowance")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GB")
        ax.legend(loc="best")
    elif "cycle_usage_E" in g.columns:
        ax.plot(x, g["cycle_usage_E"], color=_col("L_E"), lw=1.5,
                label="Cycle Usage (GB)")
        ax.set_title("[1.2] eMBB: Cycle Usage")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GB")
        ax.legend(loc="best")
    else:
        _unavailable(ax, "Cycle usage data not available")
        ax.set_title("[1.2] eMBB: Cycle Usage vs Allowance")


def _panel_06_3_overage_revenue(
    ax, g: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 6.3: Overage Revenue by Slice."""
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    if "over_rev" in g.columns and "over_rev_E" in g.columns:
        over_rev_U = g["over_rev"] - g["over_rev_E"]
        ax.fill_between(x, 0, over_rev_U / 1e3,
                        color=_col("L_U"), alpha=0.6, label="URLLC Overage")
        ax.fill_between(x, over_rev_U / 1e3,
                        (over_rev_U + g["over_rev_E"]) / 1e3,
                        color=_col("L_E"), alpha=0.6, label="eMBB Overage")
        ax.set_title("[2.1] Overage Revenue by Slice")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("K KRW")
        ax.legend(loc="best")
        total_over = g["over_rev"].sum()
        total_rev = g.get("revenue", g["over_rev"]).sum()
        over_pct = total_over / max(total_rev, 1e-6) * 100
        _callout(ax, f"Overage share: {over_pct:.1f}% of revenue")
    elif "over_rev" in g.columns:
        ax.plot(x, g["over_rev"] / 1e3, color=_col("over_rev"),
                lw=1.5, label="Total Overage Revenue")
        ax.set_title("[2.1] Overage Revenue")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("K KRW")
        ax.legend(loc="best")
    else:
        _unavailable(ax, "Overage revenue data not available")
        ax.set_title("[2.1] Overage Revenue by Slice")


def _panel_06_4_demand_elasticity(
    ax, raw_df: pd.DataFrame, cfg: Dict[str, Any], mode: str,
) -> None:
    """Panel 6.4: Demand Elasticity scatter (p_over_E vs L_E)."""
    if "p_over_E" in raw_df.columns and "L_E" in raw_df.columns:
        sample = raw_df[["p_over_E", "L_E"]].dropna()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        ax.scatter(sample["p_over_E"], sample["L_E"],
                   c=sample.index, cmap="viridis",
                   alpha=0.3, s=8, edgecolors="none")
        ax.set_title("[2.2] Demand Elasticity: Overage Price vs eMBB Load")
        ax.set_xlabel("p_over_E (KRW/GB)")
        ax.set_ylabel("L_E (GB/step)")
        epsilon_E = cfg.get("demand_elasticity", {}).get("epsilon_E", 0.30)
        ax.text(0.97, 0.05,
                f"epsilon_E = {epsilon_E} [Nevo 2016]",
                transform=ax.transAxes, ha="right", fontsize=8,
                color=GRAY)
    else:
        _unavailable(ax, "Price/load data not available for scatter")
        ax.set_title("[2.2] Demand Elasticity Scatter")


# ── Sheet 7: Multi-Seed Convergence ──────────────────────────────────

def _prepare_convergence_data(
    seed_dfs: Dict[int, pd.DataFrame],
) -> Tuple[Dict[int, pd.DataFrame], int, int]:
    """Pre-compute per-seed episode aggregations for convergence panels.

    Returns:
        (seed_episodes, n_seeds, max_episodes)
    """
    seed_episodes: Dict[int, pd.DataFrame] = {}
    for sid, df in seed_dfs.items():
        df_ep = _detect_episodes(df)
        ep_agg = df_ep.groupby("_episode").agg(
            reward_mean=("reward", "mean"),
            profit_mean=("profit", "mean"),
            pviol_E_mean=("pviol_E", "mean"),
            rho_U_mean=("rho_U", "mean"),
        )
        seed_episodes[sid] = ep_agg

    n_seeds = len(seed_episodes)
    max_episodes = max(len(ep) for ep in seed_episodes.values()) if seed_episodes else 0
    return seed_episodes, n_seeds, max_episodes


def _convergence_envelope(
    ax, seed_episodes: Dict[int, pd.DataFrame],
    metric: str, n_seeds: int, max_episodes: int,
    cfg: Dict[str, Any], title: str, ylabel: str, color: str,
    show_legend: bool = False,
) -> None:
    """Shared logic for convergence envelope plots (Sheet 7 panels)."""
    all_series = []
    for sid in sorted(seed_episodes.keys()):
        ep = seed_episodes[sid]
        vals = ep[metric].values
        if n_seeds > 1:
            ax.plot(range(len(vals)), vals, color=color, alpha=0.25, lw=0.7,
                    label="Individual seeds" if sid == min(seed_episodes.keys()) else "")
        padded = np.full(max_episodes, np.nan)
        padded[:len(vals)] = vals
        all_series.append(padded)

    arr = np.array(all_series)
    mean_vals = np.nanmean(arr, axis=0)
    episodes = np.arange(max_episodes)

    if n_seeds == 1:
        ax.plot(episodes, mean_vals, color=color, lw=2.5,
                label=f"Seed {min(seed_episodes.keys())}")
    else:
        p25 = np.nanpercentile(arr, 25, axis=0)
        p75 = np.nanpercentile(arr, 75, axis=0)
        ax.plot(episodes, mean_vals, color=color, lw=2.5,
                label="Mean (all seeds)")
        ax.fill_between(episodes, p25, p75, color=color, alpha=0.15,
                        label="P25\u2013P75 envelope")

    # Curriculum phase boundaries
    cur_cfg = cfg.get("training", {}).get("curriculum", {})
    phases = cur_cfg.get("phases", [])
    cumul = 0.0
    for pi, phase in enumerate(phases[:-1]):
        cumul += phase.get("fraction", 0)
        boundary_ep = int(max_episodes * cumul)
        if 0 < boundary_ep < max_episodes:
            ax.axvline(boundary_ep, color=GRAY, ls=":", alpha=0.6)
            ax.text(boundary_ep + 1, ax.get_ylim()[1] * 0.95,
                    f"Phase {pi+1}\u2192{pi+2}", fontsize=7, color=GRAY)

    # pviol_E threshold
    pviol_threshold = cfg.get("lagrangian_qos", {}).get("pviol_E_threshold", 0.08)
    if metric == "pviol_E_mean":
        _annotate_threshold(ax, pviol_threshold,
                            f"Threshold = {pviol_threshold}",
                            color=DARK_RED)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(loc="best")


def _panel_07_1_reward_convergence(
    ax, seed_episodes: Dict[int, pd.DataFrame],
    cfg: Dict[str, Any], n_seeds: int, max_episodes: int,
) -> None:
    """Panel 7.1: Reward Convergence."""
    _convergence_envelope(ax, seed_episodes, "reward_mean",
                          n_seeds, max_episodes, cfg,
                          "[1.1] Reward Convergence", "Reward",
                          _col("reward"), show_legend=True)


def _panel_07_2_profit_convergence(
    ax, seed_episodes: Dict[int, pd.DataFrame],
    cfg: Dict[str, Any], n_seeds: int, max_episodes: int,
) -> None:
    """Panel 7.2: Profit Convergence."""
    _convergence_envelope(ax, seed_episodes, "profit_mean",
                          n_seeds, max_episodes, cfg,
                          "[1.2] Profit Convergence", "Profit (KRW)",
                          _col("profit"))


def _panel_07_3_pviol_convergence(
    ax, seed_episodes: Dict[int, pd.DataFrame],
    cfg: Dict[str, Any], n_seeds: int, max_episodes: int,
) -> None:
    """Panel 7.3: eMBB QoS Violation Convergence."""
    _convergence_envelope(ax, seed_episodes, "pviol_E_mean",
                          n_seeds, max_episodes, cfg,
                          "[2.1] eMBB QoS Violation Convergence", "pviol_E",
                          _col("pviol_E"))


def _panel_07_4_rho_convergence(
    ax, seed_episodes: Dict[int, pd.DataFrame],
    cfg: Dict[str, Any], n_seeds: int, max_episodes: int,
) -> None:
    """Panel 7.4: Resource Allocation Convergence."""
    _convergence_envelope(ax, seed_episodes, "rho_U_mean",
                          n_seeds, max_episodes, cfg,
                          "[2.2] Resource Allocation Convergence", "rho_U",
                          _col("rho_U"))


# ══════════════════════════════════════════════════════════════════════
# INDIVIDUAL PANEL SAVE HELPER [DASH-2]
# ══════════════════════════════════════════════════════════════════════

def _save_individual_panel(
    panel_fn: Callable,
    panel_args: tuple,
    out_path: Path,
    dpi: int,
    figsize: Tuple[float, float] = (10, 7),
    ref_text: Optional[str] = None,
) -> Path:
    """Create a single-axis figure, call the panel function, and save.

    [DASH-2] Each panel is rendered as a standalone PNG file with
    optional reference citation at the bottom.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    panel_fn(ax, *panel_args)
    if ref_text:
        fig.text(0.5, 0.01, ref_text, ha="center", fontsize=7, color="gray")
        plt.tight_layout(rect=[0, 0.03, 1, 1.0])
    else:
        plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════
# SHEET RENDERERS (thin wrappers calling panel functions)
# ══════════════════════════════════════════════════════════════════════

def _render_sheet_01_financial(
    g: pd.DataFrame, cfg: Dict[str, Any],
    out_dir: Path, dpi: int, mode: str,
) -> Path:
    """Sheet 1: Financial Overview — Revenue, Cost, Profit, Margins."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sheet 1: Financial Performance Overview",
                 fontsize=14, fontweight="bold", y=0.98)

    _panel_01_1_revenue_cost_profit(axes[0, 0], g, cfg, mode)
    _panel_01_2_revenue_decomposition(axes[0, 1], g, cfg, mode)
    _panel_01_3_cost_breakdown(axes[1, 0], g, cfg, mode)
    _panel_01_4_profitability_ratios(axes[1, 1], g, cfg, mode)

    fig.text(0.5, 0.01, _REF_01, ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "01_financial_overview.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_sheet_02_reward(
    g: pd.DataFrame, raw_df: pd.DataFrame, cfg: Dict[str, Any],
    out_dir: Path, dpi: int, mode: str,
) -> Path:
    """Sheet 2: Reward Decomposition — signal analysis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sheet 2: Reward Signal Decomposition",
                 fontsize=14, fontweight="bold", y=0.98)

    _panel_02_1_reward_over_time(axes[0, 0], g, cfg, mode)
    _panel_02_2_reward_shaping_components(axes[0, 1], g, cfg, mode)
    _panel_02_3_reward_distribution(axes[1, 0], raw_df, cfg, mode)
    _panel_02_4_penalty_bonus_trajectories(axes[1, 1], g, cfg, mode)

    fig.text(0.5, 0.01, _REF_02, ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "02_reward_decomposition.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_sheet_03_market(
    g: pd.DataFrame, cfg: Dict[str, Any],
    out_dir: Path, dpi: int, mode: str,
) -> Path:
    """Sheet 3: Market Dynamics — population and churn."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sheet 3: Market Dynamics & User Population",
                 fontsize=14, fontweight="bold", y=0.98)

    _panel_03_1_active_users(axes[0, 0], g, cfg, mode)
    _panel_03_2_per_slice_subscribers(axes[0, 1], g, cfg, mode)
    _panel_03_3_market_flows(axes[1, 0], g, cfg, mode)
    _panel_03_4_churn_rate(axes[1, 1], g, cfg, mode)

    fig.text(0.5, 0.01, _REF_03, ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "03_market_dynamics.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_sheet_04_actions(
    g: pd.DataFrame, raw_df: pd.DataFrame, cfg: Dict[str, Any],
    out_dir: Path, dpi: int, mode: str,
) -> Path:
    """Sheet 4: Pricing Actions — agent strategy."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sheet 4: Agent Pricing Strategy",
                 fontsize=14, fontweight="bold", y=0.98)

    _panel_04_1_urllc_pricing(axes[0, 0], g, cfg, mode)
    _panel_04_2_embb_pricing(axes[0, 1], g, cfg, mode)
    _panel_04_3_rho_u_allocation(axes[1, 0], g, cfg, mode)
    _panel_04_4_action_distributions(axes[1, 1], raw_df, cfg, mode)

    fig.text(0.5, 0.01, _REF_04, ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "04_pricing_actions.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_sheet_05_network(
    g: pd.DataFrame, cfg: Dict[str, Any],
    out_dir: Path, dpi: int, mode: str,
) -> Path:
    """Sheet 5: Network & QoS — load, capacity, violations."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sheet 5: Network Quality & Load Management",
                 fontsize=14, fontweight="bold", y=0.98)

    _panel_05_1_urllc_load_capacity(axes[0, 0], g, cfg, mode)
    _panel_05_2_embb_load_capacity(axes[0, 1], g, cfg, mode)
    _panel_05_3_qos_violation_prob(axes[1, 0], g, cfg, mode)
    _panel_05_4_slice_utilization(axes[1, 1], g, cfg, mode)

    fig.text(0.5, 0.01, _REF_05, ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "05_network_qos.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_sheet_06_billing(
    g: pd.DataFrame, raw_df: pd.DataFrame, cfg: Dict[str, Any],
    out_dir: Path, dpi: int, mode: str,
) -> Path:
    """Sheet 6: Billing & Usage — cycle usage, overage, elasticity."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sheet 6: Billing Cycle & Traffic Analysis",
                 fontsize=14, fontweight="bold", y=0.98)

    _panel_06_1_urllc_cycle_usage(axes[0, 0], g, cfg, mode)
    _panel_06_2_embb_cycle_usage(axes[0, 1], g, cfg, mode)
    _panel_06_3_overage_revenue(axes[1, 0], g, cfg, mode)
    _panel_06_4_demand_elasticity(axes[1, 1], raw_df, cfg, mode)

    fig.text(0.5, 0.01, _REF_06, ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "06_billing_usage.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_sheet_07_convergence(
    seed_dfs: Dict[int, pd.DataFrame], cfg: Dict[str, Any],
    out_dir: Path, dpi: int,
) -> Optional[Path]:
    """Sheet 7: Multi-Seed Convergence (training mode only)."""
    import matplotlib.pyplot as plt

    if len(seed_dfs) < 1:
        logger.info("  No training data — skipping Sheet 7")
        return None

    seed_episodes, n_seeds, max_episodes = _prepare_convergence_data(seed_dfs)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    seed_label = "1 Seed" if n_seeds == 1 else f"{n_seeds} Seeds"
    fig.suptitle(f"Sheet 7: Training Convergence ({seed_label})",
                 fontsize=14, fontweight="bold", y=0.98)

    _panel_07_1_reward_convergence(axes[0, 0], seed_episodes, cfg, n_seeds, max_episodes)
    _panel_07_2_profit_convergence(axes[0, 1], seed_episodes, cfg, n_seeds, max_episodes)
    _panel_07_3_pviol_convergence(axes[1, 0], seed_episodes, cfg, n_seeds, max_episodes)
    _panel_07_4_rho_convergence(axes[1, 1], seed_episodes, cfg, n_seeds, max_episodes)

    fig.text(0.5, 0.01, _REF_07, ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "07_multi_seed_convergence.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# INDIVIDUAL PANEL GENERATION [DASH-2]
# ══════════════════════════════════════════════════════════════════════

# Panel registry: (function, filename, ref_text, uses_raw_df)
_PANELS_SHEET_1_6 = [
    # Sheet 1
    (_panel_01_1_revenue_cost_profit, "01_1_revenue_cost_profit.png", _REF_01, False),
    (_panel_01_2_revenue_decomposition, "01_2_revenue_decomposition.png", _REF_01, False),
    (_panel_01_3_cost_breakdown, "01_3_cost_breakdown.png", _REF_01, False),
    (_panel_01_4_profitability_ratios, "01_4_profitability_ratios.png", _REF_01, False),
    # Sheet 2
    (_panel_02_1_reward_over_time, "02_1_reward_over_time.png", _REF_02, False),
    (_panel_02_2_reward_shaping_components, "02_2_reward_shaping_components.png", _REF_02, False),
    (_panel_02_3_reward_distribution, "02_3_reward_distribution.png", _REF_02, True),
    (_panel_02_4_penalty_bonus_trajectories, "02_4_penalty_bonus_trajectories.png", _REF_02, False),
    # Sheet 3
    (_panel_03_1_active_users, "03_1_active_users.png", _REF_03, False),
    (_panel_03_2_per_slice_subscribers, "03_2_per_slice_subscribers.png", _REF_03, False),
    (_panel_03_3_market_flows, "03_3_market_flows.png", _REF_03, False),
    (_panel_03_4_churn_rate, "03_4_churn_rate.png", _REF_03, False),
    # Sheet 4
    (_panel_04_1_urllc_pricing, "04_1_urllc_pricing.png", _REF_04, False),
    (_panel_04_2_embb_pricing, "04_2_embb_pricing.png", _REF_04, False),
    (_panel_04_3_rho_u_allocation, "04_3_rho_u_allocation.png", _REF_04, False),
    (_panel_04_4_action_distributions, "04_4_action_distributions.png", _REF_04, True),
    # Sheet 5
    (_panel_05_1_urllc_load_capacity, "05_1_urllc_load_capacity.png", _REF_05, False),
    (_panel_05_2_embb_load_capacity, "05_2_embb_load_capacity.png", _REF_05, False),
    (_panel_05_3_qos_violation_prob, "05_3_qos_violation_prob.png", _REF_05, False),
    (_panel_05_4_slice_utilization, "05_4_slice_utilization.png", _REF_05, False),
    # Sheet 6
    (_panel_06_1_urllc_cycle_usage, "06_1_urllc_cycle_usage.png", _REF_06, False),
    (_panel_06_2_embb_cycle_usage, "06_2_embb_cycle_usage.png", _REF_06, False),
    (_panel_06_3_overage_revenue, "06_3_overage_revenue.png", _REF_06, False),
    (_panel_06_4_demand_elasticity, "06_4_demand_elasticity.png", _REF_06, True),
]

_PANELS_SHEET_7 = [
    (_panel_07_1_reward_convergence, "07_1_reward_convergence.png", _REF_07),
    (_panel_07_2_profit_convergence, "07_2_profit_convergence.png", _REF_07),
    (_panel_07_3_pviol_convergence, "07_3_pviol_convergence.png", _REF_07),
    (_panel_07_4_rho_convergence, "07_4_rho_convergence.png", _REF_07),
]


def _generate_individual_panels(
    g: pd.DataFrame, raw_df: pd.DataFrame,
    cfg: Dict[str, Any], out: Path, dpi: int, mode: str,
    seed_dfs: Optional[Dict[int, pd.DataFrame]] = None,
) -> List[Path]:
    """Generate all 28 individual panel PNGs. [DASH-2]"""
    generated: List[Path] = []

    for panel_fn, fname, ref, uses_raw in _PANELS_SHEET_1_6:
        data = raw_df if uses_raw else g
        p = _save_individual_panel(
            panel_fn, (data, cfg, mode),
            out / fname, dpi, ref_text=ref)
        generated.append(p)

    if seed_dfs and len(seed_dfs) > 0:
        seed_episodes, n_seeds, max_episodes = _prepare_convergence_data(seed_dfs)
        for panel_fn, fname, ref in _PANELS_SHEET_7:
            p = _save_individual_panel(
                panel_fn, (seed_episodes, cfg, n_seeds, max_episodes),
                out / fname, dpi, ref_text=ref)
            generated.append(p)

    return generated


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

def generate_all_pngs(
    output_dir: str = "outputs",
    config_path: str = "config/default.yaml",
    mode: str = "auto",
    dpi: int = 180,
    layout: str = "composite",
) -> List[Path]:
    """Generate all themed PNG dashboard sheets.

    Args:
        output_dir: Directory containing CSV logs and where PNGs are saved.
        config_path: Path to YAML config (for action bounds, thresholds).
        mode: ``"training"`` reads train_log CSVs,
              ``"eval"`` reads rollout_log.csv,
              ``"auto"`` detects available files.
        dpi: Resolution for output PNGs.
        layout: ``"composite"`` (7 sheets, default),
                ``"individual"`` (28 individual panels),
                ``"both"`` (7 + 28 = 35 files).

    Returns:
        List of Paths to generated PNG files.

    References:
        [Wong Nat.Methods 2011]   Color-blind safe palette
        [Henderson AAAI 2018]     Multi-seed analysis
        [Dulac-Arnold JMLR 2021]  RL observability
        [Tufte 2001]              Visual display principles
    """
    from oran3pt.utils import load_config

    _setup_style()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load config for thresholds and bounds
    cfg = load_config(config_path)
    logger.info("Config loaded from %s", config_path)

    # Determine mode
    has_training = len(list(out.glob("train_log_seed*.csv"))) > 0
    has_eval = (out / "rollout_log.csv").exists()

    if mode == "auto":
        # Prefer eval data if available, also generate training convergence
        if has_eval:
            mode = "eval"
        elif has_training:
            mode = "training"
        else:
            logger.error("No CSV data found in %s", output_dir)
            return []

    generated: List[Path] = []

    if mode == "eval" and has_eval:
        logger.info("=== Generating Evaluation Dashboard ===")
        rollout, _, _ = _load_eval_data(output_dir)
        if rollout is None:
            logger.error("rollout_log.csv detected but could not be loaded from %s",
                         output_dir)
            return []

        rollout = _add_derived_columns(rollout)

        # For eval: aggregate by step across repeats
        g = rollout.groupby("step").mean(numeric_only=True)

        seed_dfs = None
        if has_training:
            seed_dfs = _load_training_data(output_dir)

        if layout in ("composite", "both"):
            generated.append(
                _render_sheet_01_financial(g, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_02_reward(g, rollout, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_03_market(g, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_04_actions(g, rollout, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_05_network(g, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_06_billing(g, rollout, cfg, out, dpi, mode))
            if seed_dfs:
                p = _render_sheet_07_convergence(seed_dfs, cfg, out, dpi)
                if p:
                    generated.append(p)

        if layout in ("individual", "both"):
            generated.extend(
                _generate_individual_panels(g, rollout, cfg, out, dpi, mode,
                                            seed_dfs=seed_dfs))

    elif mode == "training" and has_training:
        logger.info("=== Generating Training Dashboard ===")
        seed_dfs = _load_training_data(output_dir)

        # [DASH-1] Use best seed from training_metadata.json for sheets 1-6
        primary_seed = min(seed_dfs.keys())  # default fallback
        meta_path = out / "training_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as _f:
                    _meta = json.load(_f)
                candidate = _meta.get("best_seed", primary_seed)
                if candidate in seed_dfs:
                    primary_seed = candidate
            except Exception:
                pass  # graceful fallback
        logger.info("  Primary seed for Sheets 1-6: seed %d", primary_seed)
        primary_df = seed_dfs[primary_seed]
        primary_df = _add_derived_columns(primary_df)

        # Episode-level aggregation
        primary_df = _detect_episodes(primary_df)
        g = primary_df.groupby("_episode").mean(numeric_only=True)

        if layout in ("composite", "both"):
            generated.append(
                _render_sheet_01_financial(g, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_02_reward(g, primary_df, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_03_market(g, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_04_actions(g, primary_df, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_05_network(g, cfg, out, dpi, mode))
            generated.append(
                _render_sheet_06_billing(g, primary_df, cfg, out, dpi, mode))
            p = _render_sheet_07_convergence(seed_dfs, cfg, out, dpi)
            if p:
                generated.append(p)

        if layout in ("individual", "both"):
            generated.extend(
                _generate_individual_panels(g, primary_df, cfg, out, dpi, mode,
                                            seed_dfs=seed_dfs))

    logger.info("=== Generated %d PNG files ===", len(generated))
    for p in generated:
        logger.info("  %s", p)

    return generated


def main() -> None:
    """CLI entry point for PNG dashboard generation."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive PNG dashboard sheets [M11]")
    parser.add_argument(
        "--output", default="outputs",
        help="Directory containing CSV logs (default: outputs)")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config (default: config/default.yaml)")
    parser.add_argument(
        "--mode", default="auto", choices=["training", "eval", "auto"],
        help="Data mode: training, eval, or auto-detect (default: auto)")
    parser.add_argument(
        "--layout", default="composite",
        choices=["composite", "individual", "both"],
        help="Output layout: composite (7 sheets), individual (28 panels), "
             "or both (default: composite)")
    parser.add_argument(
        "--dpi", type=int, default=180,
        help="Output resolution in DPI (default: 180)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s] %(message)s",
    )

    paths = generate_all_pngs(
        output_dir=args.output,
        config_path=args.config,
        mode=args.mode,
        dpi=args.dpi,
        layout=args.layout,
    )
    print(f"\nGenerated {len(paths)} PNG files:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
