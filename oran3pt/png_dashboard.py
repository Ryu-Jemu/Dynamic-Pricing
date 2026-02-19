"""
Comprehensive PNG Dashboard Generator (§16c).

Generates 7 themed PNG sheets (28 panels) covering ALL metrics recorded
during SAC training and evaluation.  Designed for beginner readability
with annotated thresholds, config-driven bounds, and colorblind-safe
palette.

REVISION 9 — New module  [M11].

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

References:
  [Wong Nat.Methods 2011]   Color-blind safe palette
  [Henderson AAAI 2018]     Multi-seed convergence analysis
  [Dulac-Arnold JMLR 2021]  Observability for real-world RL
  [Tufte 2001]              Data-ink ratio, small multiples
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# SHEET RENDERERS
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
    x = g.index

    # (1,1) Revenue / Cost / Profit
    ax = axes[0, 0]
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

    # (1,2) Revenue Decomposition
    ax = axes[0, 1]
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

    # (2,1) Cost Breakdown
    ax = axes[1, 0]
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

    # (2,2) Profit Margin & SLA/Revenue Ratio
    ax = axes[1, 1]
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

    fig.text(0.5, 0.01,
             "Refs: [Grubb AER 2009] 3-Part Tariff | "
             "[Nevo Econometrica 2016] Demand Elasticity | "
             "[Tessler ICML 2019] CMDP",
             ha="center", fontsize=7, color="gray")

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
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"

    # (1,1) Reward over time with rolling mean
    ax = axes[0, 0]
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

    # (1,2) Reward components stacked area
    ax = axes[0, 1]
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
            # Penalties are subtracted (negative effect), bonus is positive
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

    # (2,1) Reward distribution histogram
    ax = axes[1, 0]
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

    # (2,2) Individual penalty/bonus time series
    ax = axes[1, 1]
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

    fig.text(0.5, 0.01,
             "Refs: [Dalal NeurIPS 2018] Action Smoothing | "
             "[Wiewiora ICML 2003] Reward Shaping | "
             "[Mguni AAMAS 2019] Population Reward",
             ha="center", fontsize=7, color="gray")

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
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    N_total = cfg.get("population", {}).get("N_total", 500)
    target_ratio = cfg.get("population_reward", {}).get("target_ratio", 0.4)

    # (1,1) Active Users vs Target
    ax = axes[0, 0]
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

    # (1,2) Per-Slice Users
    ax = axes[0, 1]
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

    # (2,1) Join / Churn / Net Flow
    ax = axes[1, 0]
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

    # (2,2) Churn Rate & Population Delta
    ax = axes[1, 1]
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

    fig.text(0.5, 0.01,
             "Refs: [Kim & Yoon 2004] Churn Determinants | "
             "[Ahn 2006] Partial Defection | "
             "[Gupta JSR 2006] CLV",
             ha="center", fontsize=7, color="gray")

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
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    acfg = cfg.get("action", {})

    # (1,1) URLLC Pricing
    ax = axes[0, 0]
    ax.plot(x, g["F_U"] / 1e3, color=_col("F_U"), lw=2, label="F_U (K KRW)")
    ax.fill_between(x, acfg.get("F_U_min", 30000) / 1e3,
                    acfg.get("F_U_max", 90000) / 1e3,
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

    # (1,2) eMBB Pricing
    ax = axes[0, 1]
    ax.plot(x, g["F_E"] / 1e3, color=_col("F_E"), lw=2, label="F_E (K KRW)")
    ax.fill_between(x, acfg.get("F_E_min", 40000) / 1e3,
                    acfg.get("F_E_max", 150000) / 1e3,
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

    # (2,1) rho_U Resource Allocation
    ax = axes[1, 0]
    rho_min = acfg.get("rho_U_min", 0.05)
    rho_max = acfg.get("rho_U_max", 0.35)
    ax.plot(x, g["rho_U"], color=_col("rho_U"), lw=2, label="rho_U")
    ax.fill_between(x, rho_min, rho_max,
                    color=_col("rho_U"), alpha=0.08,
                    label=f"Bounds [{rho_min:.2f}, {rho_max:.2f}]")
    ax.set_title("[2.1] URLLC PRB Allocation (rho_U)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rho_U")
    ax.legend(loc="best")
    _callout(ax, f"Mean: {g['rho_U'].mean():.3f}")

    # (2,2) Action Distributions (box plots)
    ax = axes[1, 1]
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
            # Normalize to [0,1] for comparable box plots
            vals = raw_df[col].dropna()
            lo = acfg.get(f"{col}_min", vals.min()) if col != "rho_U" \
                else acfg.get("rho_U_min", 0.05)
            hi = acfg.get(f"{col}_max", vals.max()) if col != "rho_U" \
                else acfg.get("rho_U_max", 0.35)
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

    fig.text(0.5, 0.01,
             "Refs: [Grubb AER 2009] Base Fee | "
             "[Nevo 2016] Overage Price | "
             "[Huang IoT-J 2020] PRB Allocation",
             ha="center", fontsize=7, color="gray")

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
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    pviol_threshold = cfg.get("lagrangian_qos", {}).get("pviol_E_threshold", 0.15)

    # (1,1) URLLC Load vs Capacity
    ax = axes[0, 0]
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

    # (1,2) eMBB Load vs Capacity
    ax = axes[0, 1]
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

    # (2,1) QoS Violation Probabilities
    ax = axes[1, 0]
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

    # (2,2) Utilization Ratios
    ax = axes[1, 1]
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

    fig.text(0.5, 0.01,
             "Refs: [3GPP TS 38.104] QoS | "
             "[Tessler ICML 2019] CMDP | "
             "[Huang IoT-J 2020] URLLC-eMBB Coexistence",
             ha="center", fontsize=7, color="gray")

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
    x = g.index
    xlabel = "Episode" if mode == "training" else "Step"
    Q_U = cfg.get("tariff", {}).get("Q_U_gb", 5.0)
    Q_E = cfg.get("tariff", {}).get("Q_E_gb", 50.0)

    # (1,1) URLLC Cycle Usage vs Allowance
    ax = axes[0, 0]
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

    # (1,2) eMBB Cycle Usage vs Allowance
    ax = axes[0, 1]
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

    # (2,1) Overage Revenue by Slice
    ax = axes[1, 0]
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

    # (2,2) Demand Elasticity: p_over_E vs L_E scatter
    ax = axes[1, 1]
    if "p_over_E" in raw_df.columns and "L_E" in raw_df.columns:
        sample = raw_df[["p_over_E", "L_E"]].dropna()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        sc = ax.scatter(sample["p_over_E"], sample["L_E"],
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

    fig.text(0.5, 0.01,
             "Refs: [Nevo Econometrica 2016] Demand Elasticity | "
             "[3GPP TS 23.503] Usage Monitoring | "
             "[Grubb AER 2009] 3-Part Tariff",
             ha="center", fontsize=7, color="gray")

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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    n_seeds = len(seed_dfs)
    fig.suptitle(f"Sheet 7: Training Convergence ({n_seeds} Seeds)",
                 fontsize=14, fontweight="bold", y=0.98)

    # Aggregate each seed to episode level
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

    # Build aligned arrays
    max_episodes = max(len(ep) for ep in seed_episodes.values())

    metrics_config = [
        ("reward_mean", "[1.1] Reward Convergence", "Reward", _col("reward")),
        ("profit_mean", "[1.2] Profit Convergence", "Profit (KRW)", _col("profit")),
        ("pviol_E_mean", "[2.1] eMBB QoS Violation Convergence", "pviol_E",
         _col("pviol_E")),
        ("rho_U_mean", "[2.2] Resource Allocation Convergence", "rho_U",
         _col("rho_U")),
    ]

    for idx, (metric, title, ylabel, color) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]

        # Collect per-seed series
        all_series = []
        for sid in sorted(seed_episodes.keys()):
            ep = seed_episodes[sid]
            vals = ep[metric].values
            ax.plot(range(len(vals)), vals, color=color, alpha=0.25, lw=0.7,
                    label=f"Seed {sid}" if sid == min(seed_episodes.keys()) else "")
            # Pad to max_episodes for envelope computation
            padded = np.full(max_episodes, np.nan)
            padded[:len(vals)] = vals
            all_series.append(padded)

        arr = np.array(all_series)
        mean_vals = np.nanmean(arr, axis=0)
        p25 = np.nanpercentile(arr, 25, axis=0)
        p75 = np.nanpercentile(arr, 75, axis=0)

        episodes = np.arange(max_episodes)
        ax.plot(episodes, mean_vals, color=color, lw=2.5,
                label="Mean (all seeds)")
        ax.fill_between(episodes, p25, p75, color=color, alpha=0.15,
                        label="P25–P75 envelope")

        # [ME-4] Curriculum phase boundaries — use phases list (v9+)
        cur_cfg = cfg.get("training", {}).get("curriculum", {})
        phases = cur_cfg.get("phases", [])
        if phases:
            cumul = 0.0
            for pi, phase in enumerate(phases[:-1]):
                cumul += phase.get("fraction", 0)
                boundary_ep = int(max_episodes * cumul)
                if 0 < boundary_ep < max_episodes:
                    ax.axvline(boundary_ep, color=GRAY, ls=":", alpha=0.6)
                    ax.text(boundary_ep + 1, ax.get_ylim()[1] * 0.95,
                            f"Phase {pi+1}→{pi+2}", fontsize=7, color=GRAY)
        else:
            # Legacy fallback: single phase1_fraction boundary
            curriculum_frac = cur_cfg.get("phase1_fraction", 0.20)
            phase1_ep = int(max_episodes * curriculum_frac)
            if 0 < phase1_ep < max_episodes:
                ax.axvline(phase1_ep, color=GRAY, ls=":", alpha=0.6)
                ax.text(phase1_ep + 1, ax.get_ylim()[1] * 0.95,
                        "Phase 1→2", fontsize=7, color=GRAY)

        # Specific thresholds
        pviol_threshold = cfg.get("lagrangian_qos", {}).get(
            "pviol_E_threshold", 0.15)
        if metric == "pviol_E_mean":
            _annotate_threshold(ax, pviol_threshold,
                                f"Threshold = {pviol_threshold}",
                                color=DARK_RED)

        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        # Only show legend for first subplot to avoid clutter
        if idx == 0:
            ax.legend(loc="best")

    fig.text(0.5, 0.01,
             "Refs: [Henderson AAAI 2018] Multi-Seed Evaluation | "
             "[Narvekar JMLR 2020] Curriculum Learning | "
             "[Haarnoja ICML 2018] SAC",
             ha="center", fontsize=7, color="gray")

    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = out_dir / "07_multi_seed_convergence.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

def generate_all_pngs(
    output_dir: str = "outputs",
    config_path: str = "config/default.yaml",
    mode: str = "auto",
    dpi: int = 180,
) -> List[Path]:
    """Generate all themed PNG dashboard sheets.

    Args:
        output_dir: Directory containing CSV logs and where PNGs are saved.
        config_path: Path to YAML config (for action bounds, thresholds).
        mode: ``"training"`` reads train_log CSVs,
              ``"eval"`` reads rollout_log.csv,
              ``"auto"`` detects available files.
        dpi: Resolution for output PNGs.

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
        logger.info("=== Generating Evaluation Dashboard (7 sheets) ===")
        rollout, _, _ = _load_eval_data(output_dir)
        assert rollout is not None

        rollout = _add_derived_columns(rollout)

        # For eval: aggregate by step across repeats
        g = rollout.groupby("step").mean(numeric_only=True)

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

        # Also generate convergence if training data exists
        if has_training:
            seed_dfs = _load_training_data(output_dir)
            p = _render_sheet_07_convergence(seed_dfs, cfg, out, dpi)
            if p:
                generated.append(p)

    elif mode == "training" and has_training:
        logger.info("=== Generating Training Dashboard (7 sheets) ===")
        seed_dfs = _load_training_data(output_dir)

        # Use seed 0 as primary for sheets 1-6, aggregated to episode level
        primary_seed = min(seed_dfs.keys())
        primary_df = seed_dfs[primary_seed]
        primary_df = _add_derived_columns(primary_df)

        # Episode-level aggregation
        primary_df = _detect_episodes(primary_df)
        g = primary_df.groupby("_episode").mean(numeric_only=True)

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

    logger.info("=== Generated %d PNG sheets ===", len(generated))
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
    )
    print(f"\nGenerated {len(paths)} PNG dashboard sheets:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
