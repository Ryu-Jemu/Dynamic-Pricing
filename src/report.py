"""
Report generation from evaluation CSV — Enhanced Dashboard (v2).

Generates a professional multi-panel dashboard with:
  Panel 1: Convergence tracking (reward + profit dual-axis, rolling mean)
  Panel 2: Churn analysis (per-slice rates vs calibration targets)
  Panel 3: Cost breakdown (stacked area: OPEX, CAC, energy, SLA, resource)
  Panel 4: PRB efficiency (utilization + rho_URLLC allocation trajectory)
  Panel 5: Economic metrics (revenue vs cost, profit margin %)
  Panel 6: SLA compliance (V_rate time series, throughput vs SLO)

Design principles:
  - Color-blind safe palette (Wong 2011, Nature Methods)
  - Minimal chartjunk (Tufte, "Visual Display of Quantitative Information")
  - Grid lines and reference markers for interpretability
  - Dual-axis plots where appropriate for multi-scale comparison

Usage:
  python -m src.report --run_dir artifacts/<run_id>

References:
  [WONG_2011]  Wong, "Points of view: Color blindness," Nat. Methods 2011
  [TUFTE_2001] Tufte, "The Visual Display of Quantitative Information," 2001
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("oran.report")

# ---------------------------------------------------------------------------
# Color-blind safe palette [WONG_2011]
# ---------------------------------------------------------------------------
COLORS = {
    "blue":     "#0072B2",
    "orange":   "#E69F00",
    "green":    "#009E73",
    "red":      "#D55E00",
    "purple":   "#CC79A7",
    "cyan":     "#56B4E9",
    "yellow":   "#F0E442",
    "black":    "#000000",
    "gray":     "#999999",
    "lightgray": "#DDDDDD",
}


def _setup_axes_style(ax, title: str, xlabel: str, ylabel: str,
                      grid: bool = True) -> None:
    """Apply consistent styling to axis. [TUFTE_2001]"""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def generate_report(run_dir: str) -> None:
    """Generate summary report and enhanced dashboard from eval CSV."""
    run_path = Path(run_dir)
    csv_files = list(run_path.glob("eval*.csv"))

    if not csv_files:
        logger.warning("No eval CSV found in %s", run_dir)
        return

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required for report generation")
        return

    csv_path = csv_files[0]
    logger.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # Text report (report.md)
    # ------------------------------------------------------------------
    monthly_mean = df.groupby("month").mean(numeric_only=True)

    report_path = run_path / "report.md"
    with open(report_path, "w") as f:
        f.write("# O-RAN Slicing Evaluation Report\n\n")
        f.write(f"**Source**: {csv_path.name}\n")
        f.write(f"**Records**: {len(df)}\n")
        f.write(f"**Repeats**: {df['repeat'].nunique()}\n\n")

        f.write("## Key Metrics (Episode End)\n\n")
        last_month = df["month"].max()
        final = df[df["month"] == last_month]

        f.write(f"- Mean profit (final month): {final['profit'].mean():,.0f} KRW\n")
        f.write(f"- Mean reward (final month): {final['reward'].mean():.4f}\n")
        f.write(f"- Mean N_active_eMBB: {final['N_active_eMBB'].mean():.1f}\n")
        f.write(f"- Mean N_active_URLLC: {final['N_active_URLLC'].mean():.1f}\n")
        f.write(f"- Mean fee_eMBB: {final['fee_eMBB'].mean():,.0f} KRW\n")
        f.write(f"- Mean fee_URLLC: {final['fee_URLLC'].mean():,.0f} KRW\n")
        f.write(f"- Mean V_rate_eMBB: {final['V_rate_eMBB'].mean():.4f}\n")
        f.write(f"- Mean V_rate_URLLC: {final['V_rate_URLLC'].mean():.4f}\n\n")

        # Cost breakdown summary
        if "cost_opex" in df.columns:
            f.write("## Cost Structure (FIX F6/F7)\n\n")
            f.write(f"- Mean OPEX (final month): {final.get('cost_opex', pd.Series([0])).mean():,.0f} KRW\n")
            f.write(f"- Mean CAC (final month): {final.get('cost_cac', pd.Series([0])).mean():,.0f} KRW\n")
            f.write(f"- Mean cost_total: {final['cost_total'].mean():,.0f} KRW\n")
            total_rev = final["revenue"].mean()
            total_cost = final["cost_total"].mean()
            if total_rev > 0:
                margin = (total_rev - total_cost) / total_rev * 100
                f.write(f"- Profit margin: {margin:.1f}%\n\n")

        # Churn analysis
        f.write("## Churn Analysis (FIX F5)\n\n")
        for s in ["eMBB", "URLLC"]:
            churns = final[f"churns_{s}"].mean()
            n_active = final[f"N_active_{s}"].mean()
            churn_rate = churns / max(n_active, 1) * 100
            f.write(f"- {s} churn rate (final): {churn_rate:.1f}%\n")
        f.write("\n")

        f.write("## Monthly Summary\n\n")
        summary_cols = ["profit", "reward", "fee_eMBB", "fee_URLLC",
                        "N_active_eMBB", "N_active_URLLC",
                        "V_rate_eMBB", "V_rate_URLLC"]
        available_cols = [c for c in summary_cols if c in monthly_mean.columns]
        f.write(monthly_mean[available_cols].to_markdown())
        f.write("\n")

    logger.info("Report written: %s", report_path)

    # ------------------------------------------------------------------
    # Enhanced multi-panel dashboard
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(4, 3, figsize=(22, 20))
    fig.suptitle("O-RAN Slicing + Pricing — Dense Evaluation Dashboard",
                 fontsize=17, fontweight="bold", y=0.995)
    months = monthly_mean.index
    window = max(3, len(monthly_mean) // 12)

    def _has(*cols: str) -> bool:
        return all(c in monthly_mean.columns for c in cols)

    def _mark_unavailable(ax, title: str) -> None:
        _setup_axes_style(ax, title, "Month", "Value")
        ax.text(
            0.5, 0.5, "No compatible columns in eval.csv",
            ha="center", va="center", fontsize=10,
            transform=ax.transAxes, color=COLORS["gray"],
        )

    # P1 KPI snapshot (text card)
    ax = axes[0, 0]
    ax.set_axis_off()
    last_month = int(df["month"].max())
    final_month_df = df[df["month"] == last_month]
    rev_final = final_month_df["revenue"].mean() if "revenue" in final_month_df else np.nan
    profit_final = final_month_df["profit"].mean() if "profit" in final_month_df else np.nan
    cost_final = final_month_df["cost_total"].mean() if "cost_total" in final_month_df else np.nan
    margin_final = (
        (profit_final / max(rev_final, 1.0)) * 100.0
        if np.isfinite(rev_final) and np.isfinite(profit_final) else np.nan
    )
    kpi_lines = [
        "P1 KPI Snapshot",
        f"Records          : {len(df):,}",
        f"Repeats          : {df['repeat'].nunique()}",
        f"Months           : {int(df['month'].min())}..{last_month}",
        "",
        f"Final Revenue    : {rev_final/1e6:,.2f} M KRW" if np.isfinite(rev_final) else "Final Revenue    : N/A",
        f"Final Cost       : {cost_final/1e6:,.2f} M KRW" if np.isfinite(cost_final) else "Final Cost       : N/A",
        f"Final Profit     : {profit_final/1e6:,.2f} M KRW" if np.isfinite(profit_final) else "Final Profit     : N/A",
        f"Final Margin     : {margin_final:,.2f} %" if np.isfinite(margin_final) else "Final Margin     : N/A",
        "",
        f"Reward Mean±Std  : {df['reward'].mean():.4f} ± {df['reward'].std():.4f}" if "reward" in df else "Reward Mean±Std  : N/A",
        f"Profit Mean±Std  : {df['profit'].mean()/1e6:,.2f} ± {df['profit'].std()/1e6:,.2f} M KRW" if "profit" in df else "Profit Mean±Std  : N/A",
    ]
    ax.text(
        0.02, 0.98, "\n".join(kpi_lines), ha="left", va="top",
        fontsize=10.5, family="monospace",
        bbox={"facecolor": "#F7F7F7", "edgecolor": COLORS["lightgray"], "pad": 10},
        transform=ax.transAxes,
    )

    # P2 Reward + Profit convergence
    ax = axes[0, 1]
    if _has("reward", "profit"):
        _setup_axes_style(ax, "P2 Convergence: Reward + Profit", "Month", "Reward")
        for rep in df["repeat"].unique():
            rd = df[df["repeat"] == rep]
            ax.plot(rd["month"], rd["reward"], alpha=0.15, color=COLORS["blue"], linewidth=0.8)
        reward_roll = monthly_mean["reward"].rolling(window, min_periods=1).mean()
        ax.plot(months, reward_roll, color=COLORS["blue"], linewidth=2.4, label="Reward rolling")
        ax2 = ax.twinx()
        profit_roll = monthly_mean["profit"].rolling(window, min_periods=1).mean()
        ax2.plot(months, profit_roll / 1e6, color=COLORS["orange"], linewidth=2.0, linestyle="--", label="Profit rolling")
        ax2.set_ylabel("Profit (M KRW)", color=COLORS["orange"], fontsize=10)
        ax2.tick_params(axis="y", labelcolor=COLORS["orange"], labelsize=9)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="lower right", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P2 Convergence: Reward + Profit")

    # P3 Economic flow: revenue/cost/profit
    ax = axes[0, 2]
    if _has("revenue", "cost_total", "profit"):
        _setup_axes_style(ax, "P3 Revenue vs Cost vs Profit", "Month", "M KRW")
        rev = monthly_mean["revenue"] / 1e6
        cost = monthly_mean["cost_total"] / 1e6
        profit = monthly_mean["profit"] / 1e6
        ax.plot(months, rev, color=COLORS["blue"], linewidth=2.0, label="Revenue")
        ax.plot(months, cost, color=COLORS["red"], linewidth=2.0, label="Cost")
        ax.plot(months, profit, color=COLORS["green"], linewidth=1.8, linestyle="-.", label="Profit")
        ax.fill_between(months, 0, profit, where=(profit >= 0), alpha=0.12, color=COLORS["green"])
        ax.fill_between(months, 0, profit, where=(profit < 0), alpha=0.12, color=COLORS["red"])
        ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8, linestyle=":")
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P3 Revenue vs Cost vs Profit")

    # P4 Fee policy + spread
    ax = axes[1, 0]
    if _has("fee_eMBB", "fee_URLLC"):
        _setup_axes_style(ax, "P4 Subscription Fees + Spread", "Month", "Fee (KRW)")
        fee_e = monthly_mean["fee_eMBB"]
        fee_u = monthly_mean["fee_URLLC"]
        ax.plot(months, fee_e, color=COLORS["blue"], linewidth=2.0, label="fee eMBB")
        ax.plot(months, fee_u, color=COLORS["red"], linewidth=2.0, label="fee URLLC")
        spread = fee_u - fee_e
        ax2 = ax.twinx()
        ax2.plot(months, spread, color=COLORS["purple"], linewidth=1.6, linestyle="--", label="URLLC - eMBB")
        ax2.set_ylabel("Spread (KRW)", fontsize=10, color=COLORS["purple"])
        ax2.tick_params(axis="y", labelcolor=COLORS["purple"], labelsize=9)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P4 Subscription Fees + Spread")

    # P5 Active users + post-churn
    ax = axes[1, 1]
    if _has("N_active_eMBB", "N_active_URLLC"):
        _setup_axes_style(ax, "P5 Active Users & Post-churn Population", "Month", "Users")
        ne = monthly_mean["N_active_eMBB"]
        nu = monthly_mean["N_active_URLLC"]
        ax.plot(months, ne, color=COLORS["blue"], linewidth=1.8, label="N_active eMBB")
        ax.plot(months, nu, color=COLORS["red"], linewidth=1.8, label="N_active URLLC")
        ax.plot(months, ne + nu, color=COLORS["black"], linewidth=2.0, linestyle="--", label="N_active total")
        if _has("N_post_churn_eMBB", "N_post_churn_URLLC"):
            npc = monthly_mean["N_post_churn_eMBB"] + monthly_mean["N_post_churn_URLLC"]
            ax.plot(months, npc, color=COLORS["gray"], linewidth=1.6, linestyle=":", label="N_post_churn total")
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P5 Active Users & Post-churn Population")

    # P6 Flow dynamics: joins vs churns
    ax = axes[1, 2]
    if _has("joins_eMBB", "joins_URLLC", "churns_eMBB", "churns_URLLC"):
        _setup_axes_style(ax, "P6 User Flow: Joins vs Churns", "Month", "Users")
        joins_tot = monthly_mean["joins_eMBB"] + monthly_mean["joins_URLLC"]
        churn_tot = monthly_mean["churns_eMBB"] + monthly_mean["churns_URLLC"]
        net = joins_tot - churn_tot
        ax.plot(months, joins_tot, color=COLORS["green"], linewidth=2.0, label="Joins total")
        ax.plot(months, churn_tot, color=COLORS["red"], linewidth=2.0, label="Churns total")
        ax.bar(months, net, color=np.where(net >= 0, COLORS["cyan"], COLORS["orange"]), alpha=0.3, width=0.8, label="Net flow")
        ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8, linestyle=":")
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P6 User Flow: Joins vs Churns")

    # P7 Churn rate vs targets
    ax = axes[2, 0]
    if _has("churns_eMBB", "churns_URLLC", "N_active_eMBB", "N_active_URLLC"):
        _setup_axes_style(ax, "P7 Churn Rate vs Calibration Targets", "Month", "Churn rate")
        churn_e = monthly_mean["churns_eMBB"] / monthly_mean["N_active_eMBB"].clip(lower=1)
        churn_u = monthly_mean["churns_URLLC"] / monthly_mean["N_active_URLLC"].clip(lower=1)
        ax.plot(months, churn_e, color=COLORS["blue"], linewidth=1.8, label="eMBB actual")
        ax.plot(months, churn_u, color=COLORS["red"], linewidth=1.8, label="URLLC actual")
        ax.axhline(0.03, color=COLORS["blue"], linewidth=1.0, linestyle=":", alpha=0.8, label="eMBB target 3%")
        ax.axhline(0.04, color=COLORS["red"], linewidth=1.0, linestyle=":", alpha=0.8, label="URLLC target 4%")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P7 Churn Rate vs Calibration Targets")

    # P8 SLA + throughput
    ax = axes[2, 1]
    if _has("V_rate_eMBB", "V_rate_URLLC"):
        _setup_axes_style(ax, "P8 SLA Violation + Throughput", "Month", "V_rate")
        ax.plot(months, monthly_mean["V_rate_eMBB"], color=COLORS["blue"], linewidth=1.8, label="V_rate eMBB")
        ax.plot(months, monthly_mean["V_rate_URLLC"], color=COLORS["red"], linewidth=1.8, label="V_rate URLLC")
        ax.axhline(0.05, color=COLORS["gray"], linewidth=1.0, linestyle=":", alpha=0.7, label="Credit tier 1")
        ax.axhline(0.15, color=COLORS["gray"], linewidth=1.0, linestyle="-.", alpha=0.6, label="Credit tier 2")
        ax.set_ylim(0, 1.0)
        if _has("avg_T_eMBB", "avg_T_URLLC"):
            ax2 = ax.twinx()
            ax2.plot(months, monthly_mean["avg_T_eMBB"], color=COLORS["cyan"], linewidth=1.6, linestyle="--", label="T_avg eMBB")
            ax2.plot(months, monthly_mean["avg_T_URLLC"], color=COLORS["purple"], linewidth=1.6, linestyle="--", label="T_avg URLLC")
            ax2.set_ylabel("Throughput (Mbps)", fontsize=10)
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="best", framealpha=0.85)
        else:
            ax.legend(fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P8 SLA Violation + Throughput")

    # P9 PRB utilization + allocation
    ax = axes[2, 2]
    if _has("rho_util_eMBB", "rho_util_URLLC", "rho_URLLC"):
        _setup_axes_style(ax, "P9 PRB Utilization & Allocation", "Month", "Utilization / Share")
        ax.plot(months, monthly_mean["rho_util_eMBB"], color=COLORS["blue"], linewidth=1.8, label="rho_util eMBB")
        ax.plot(months, monthly_mean["rho_util_URLLC"], color=COLORS["red"], linewidth=1.8, label="rho_util URLLC")
        ax.plot(months, monthly_mean["rho_URLLC"], color=COLORS["green"], linewidth=2.0, linestyle="--", label="rho_URLLC alloc")
        total_util = (monthly_mean["rho_util_eMBB"] + monthly_mean["rho_util_URLLC"]).clip(lower=0, upper=1.5)
        ax.plot(months, total_util, color=COLORS["black"], linewidth=1.4, linestyle=":", label="util total")
        ax.axhspan(0.3, 0.8, alpha=0.06, color=COLORS["green"])
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P9 PRB Utilization & Allocation")

    # P10 Top-up behavior
    ax = axes[3, 0]
    if _has("topups_eMBB", "topups_URLLC", "N_active_eMBB", "N_active_URLLC"):
        _setup_axes_style(ax, "P10 Top-up Counts + Top-up Ratio", "Month", "Count")
        te = monthly_mean["topups_eMBB"]
        tu = monthly_mean["topups_URLLC"]
        tt = te + tu
        na = monthly_mean["N_active_eMBB"] + monthly_mean["N_active_URLLC"]
        topup_ratio = tt / na.clip(lower=1)
        ax.plot(months, te, color=COLORS["blue"], linewidth=1.6, label="topups eMBB")
        ax.plot(months, tu, color=COLORS["red"], linewidth=1.6, label="topups URLLC")
        ax.plot(months, tt, color=COLORS["black"], linewidth=1.8, linestyle="--", label="topups total")
        ax2 = ax.twinx()
        ax2.plot(months, topup_ratio, color=COLORS["orange"], linewidth=1.6, linestyle=":", label="topup ratio")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.set_ylabel("Ratio", fontsize=10, color=COLORS["orange"])
        ax2.tick_params(axis="y", labelcolor=COLORS["orange"], labelsize=9)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P10 Top-up Counts + Top-up Ratio")

    # P11 Cross-repeat stability
    ax = axes[3, 1]
    if _has("reward", "profit"):
        _setup_axes_style(ax, "P11 Cross-repeat Stability (std)", "Month", "Reward std")
        by_month = df.groupby("month")
        reward_std = by_month["reward"].std().fillna(0.0)
        profit_std = (by_month["profit"].std() / 1e6).fillna(0.0)
        ax.plot(reward_std.index, reward_std.values, color=COLORS["blue"], linewidth=2.0, label="Reward std")
        ax.fill_between(reward_std.index, 0, reward_std.values, color=COLORS["blue"], alpha=0.12)
        ax2 = ax.twinx()
        ax2.plot(profit_std.index, profit_std.values, color=COLORS["red"], linewidth=2.0, linestyle="--", label="Profit std (M KRW)")
        ax2.set_ylabel("Profit std (M KRW)", fontsize=10, color=COLORS["red"])
        ax2.tick_params(axis="y", labelcolor=COLORS["red"], labelsize=9)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="best", framealpha=0.85)
    else:
        _mark_unavailable(ax, "P11 Cross-repeat Stability (std)")

    # P12 Correlation matrix
    ax = axes[3, 2]
    corr_cols = [
        "reward", "profit", "revenue", "cost_total",
        "fee_eMBB", "fee_URLLC",
        "N_active_eMBB", "N_active_URLLC",
        "V_rate_eMBB", "V_rate_URLLC",
        "rho_URLLC", "rho_util_eMBB", "rho_util_URLLC",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    if len(corr_cols) >= 4:
        cmat = df[corr_cols].corr(numeric_only=True)
        im = ax.imshow(cmat.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_title("P12 Metric Correlation Heatmap", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(len(corr_cols)))
        ax.set_yticks(np.arange(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=75, ha="right", fontsize=7)
        ax.set_yticklabels(corr_cols, fontsize=7)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                v = cmat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    else:
        _mark_unavailable(ax, "P12 Metric Correlation Heatmap")

    # Final layout
    plt.tight_layout(rect=[0, 0, 1, 0.982])
    plt.subplots_adjust(hspace=0.38, wspace=0.28)

    plot_path = run_path / "eval_dashboard.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    logger.info("Dashboard saved: %s", plot_path)

    # Also save individual panels for paper/presentation use
    _save_individual_panels(df, monthly_mean, run_path)


def _save_individual_panels(df, monthly_mean, run_path: Path) -> None:
    """Save high-res individual panels for paper figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- Individual: Active Users ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _setup_axes_style(ax, "Active Users Over Time", "Month", "User Count")
    months = monthly_mean.index

    if "N_active_eMBB" in monthly_mean.columns:
        ax.plot(months, monthly_mean["N_active_eMBB"],
                color=COLORS["blue"], linewidth=2, label="eMBB")
    if "N_active_URLLC" in monthly_mean.columns:
        ax.plot(months, monthly_mean["N_active_URLLC"],
                color=COLORS["red"], linewidth=2, label="URLLC")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "active_users.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close()

    # --- Individual: Subscription Fees ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _setup_axes_style(ax, "Monthly Subscription Fees", "Month", "Fee (KRW)")
    if "fee_eMBB" in monthly_mean.columns:
        ax.plot(months, monthly_mean["fee_eMBB"],
                color=COLORS["blue"], linewidth=2, label="eMBB")
    if "fee_URLLC" in monthly_mean.columns:
        ax.plot(months, monthly_mean["fee_URLLC"],
                color=COLORS["red"], linewidth=2, label="URLLC")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "fees.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close()

    # --- Individual: Joins vs Churns ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _setup_axes_style(ax, "Joins vs Churns (eMBB)", "Month", "Count")
    if "joins_eMBB" in monthly_mean.columns:
        ax.plot(months, monthly_mean["joins_eMBB"],
                color=COLORS["green"], linewidth=2, label="Joins eMBB")
    if "churns_eMBB" in monthly_mean.columns:
        ax.plot(months, monthly_mean["churns_eMBB"],
                color=COLORS["red"], linewidth=2, label="Churns eMBB")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(plots_dir / "joins_churns.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close()

    logger.info("Individual plots saved to: %s", plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )
    generate_report(args.run_dir)


if __name__ == "__main__":
    main()
