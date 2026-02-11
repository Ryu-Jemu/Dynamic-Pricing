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

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle("O-RAN Slicing + Pricing — Evaluation Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)

    # ══════════════════════════════════════════════════════════════════
    # Panel 1: Convergence Tracking (Reward + Profit dual-axis)
    # ══════════════════════════════════════════════════════════════════
    ax1 = axes[0, 0]
    _setup_axes_style(ax1, "① Convergence: Reward & Profit",
                      "Month", "Reward")

    # Per-repeat reward traces (faded)
    for rep in df["repeat"].unique():
        rd = df[df["repeat"] == rep]
        ax1.plot(rd["month"], rd["reward"], alpha=0.2,
                 color=COLORS["blue"], linewidth=0.8)

    # Rolling mean reward
    window = max(3, len(monthly_mean) // 10)
    reward_roll = monthly_mean["reward"].rolling(window, min_periods=1).mean()
    ax1.plot(monthly_mean.index, reward_roll,
             color=COLORS["blue"], linewidth=2.5, label="Reward (rolling)")

    # Dual axis: profit
    ax1b = ax1.twinx()
    profit_roll = monthly_mean["profit"].rolling(window, min_periods=1).mean()
    ax1b.plot(monthly_mean.index, profit_roll / 1e6,
              color=COLORS["orange"], linewidth=2.0, linestyle="--",
              label="Profit (M₩, rolling)")
    ax1b.set_ylabel("Profit (M KRW)", fontsize=10, color=COLORS["orange"])
    ax1b.tick_params(axis="y", labelcolor=COLORS["orange"], labelsize=9)
    ax1b.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower right", fontsize=8, framealpha=0.8)

    # ══════════════════════════════════════════════════════════════════
    # Panel 2: Churn Analysis (per-slice rate vs targets)
    # ══════════════════════════════════════════════════════════════════
    ax2 = axes[0, 1]
    _setup_axes_style(ax2, "② Churn Rate vs Calibration Targets",
                      "Month", "Churn Rate")

    # Compute per-month churn rate
    for s, color, target in [("eMBB", COLORS["blue"], 0.03),
                              ("URLLC", COLORS["red"], 0.04)]:
        churn_col = f"churns_{s}"
        n_col = f"N_active_{s}"
        if churn_col in monthly_mean.columns and n_col in monthly_mean.columns:
            churn_rate = monthly_mean[churn_col] / monthly_mean[n_col].clip(lower=1)
            ax2.plot(monthly_mean.index, churn_rate,
                     color=color, linewidth=1.8, label=f"{s} actual")
            ax2.axhline(y=target, color=color, linewidth=1.2,
                        linestyle=":", alpha=0.7, label=f"{s} target ({target:.0%})")

    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.8)

    # Shade violation zone
    ax2.axhspan(0.04, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0.04 else 0.10,
                alpha=0.08, color=COLORS["red"], label="_nolegend_")

    # ══════════════════════════════════════════════════════════════════
    # Panel 3: Cost Breakdown (stacked area)
    # ══════════════════════════════════════════════════════════════════
    ax3 = axes[1, 0]
    _setup_axes_style(ax3, "③ Monthly Cost Breakdown",
                      "Month", "Cost (M KRW)")

    months = monthly_mean.index

    # Build cost components (use available columns)
    cost_components = {}
    cost_colors = []
    cost_labels = []

    if "cost_opex" in monthly_mean.columns:
        cost_components["OPEX"] = monthly_mean["cost_opex"].values / 1e6
        cost_colors.append(COLORS["blue"])
        cost_labels.append("OPEX [F6]")

    if "cost_cac" in monthly_mean.columns:
        cost_components["CAC"] = monthly_mean["cost_cac"].values / 1e6
        cost_colors.append(COLORS["orange"])
        cost_labels.append("CAC [F7]")

    # Derive energy and SLA from total if individual columns missing
    # Use revenue - profit = cost_total for reference
    if "cost_total" in monthly_mean.columns:
        total_cost = monthly_mean["cost_total"].values / 1e6
        accounted = sum(cost_components.values()) if cost_components else np.zeros(len(months))
        remaining = total_cost - accounted
        remaining = np.maximum(remaining, 0)

        cost_components["Energy+SLA+Resource"] = remaining
        cost_colors.append(COLORS["green"])
        cost_labels.append("Energy+SLA+Resource")

    if cost_components:
        stacked = np.row_stack(list(cost_components.values()))
        ax3.stackplot(months, stacked, labels=cost_labels,
                      colors=cost_colors, alpha=0.7)
        ax3.legend(fontsize=8, loc="upper left", framealpha=0.8)
    else:
        # Fallback: single cost line
        if "cost_total" in monthly_mean.columns:
            ax3.plot(months, monthly_mean["cost_total"] / 1e6,
                     color=COLORS["red"], linewidth=2, label="Total Cost")
            ax3.legend(fontsize=8)

    ax3.set_ylim(bottom=0)

    # ══════════════════════════════════════════════════════════════════
    # Panel 4: PRB Efficiency (utilization + allocation)
    # ══════════════════════════════════════════════════════════════════
    ax4 = axes[1, 1]
    _setup_axes_style(ax4, "④ PRB Utilization & URLLC Allocation",
                      "Month", "Utilization / Share")

    if "rho_util_eMBB" in monthly_mean.columns:
        ax4.plot(months, monthly_mean["rho_util_eMBB"],
                 color=COLORS["blue"], linewidth=1.8, label="ρ_util eMBB")
    if "rho_util_URLLC" in monthly_mean.columns:
        ax4.plot(months, monthly_mean["rho_util_URLLC"],
                 color=COLORS["red"], linewidth=1.8, label="ρ_util URLLC")
    if "rho_URLLC" in monthly_mean.columns:
        ax4.plot(months, monthly_mean["rho_URLLC"],
                 color=COLORS["green"], linewidth=2.0, linestyle="--",
                 label="ρ_URLLC (allocation)")

    # Ideal utilization band (0.3–0.8)
    ax4.axhspan(0.3, 0.8, alpha=0.06, color=COLORS["green"],
                label="Efficient zone (0.3–0.8)")
    ax4.set_ylim(0, 1.0)
    ax4.legend(fontsize=8, loc="upper right", framealpha=0.8)

    # ══════════════════════════════════════════════════════════════════
    # Panel 5: Economic Metrics (Revenue vs Cost, Margin %)
    # ══════════════════════════════════════════════════════════════════
    ax5 = axes[2, 0]
    _setup_axes_style(ax5, "⑤ Revenue, Cost & Profit Margin",
                      "Month", "Amount (M KRW)")

    if "revenue" in monthly_mean.columns:
        ax5.plot(months, monthly_mean["revenue"] / 1e6,
                 color=COLORS["blue"], linewidth=2.0, label="Revenue")
    if "cost_total" in monthly_mean.columns:
        ax5.plot(months, monthly_mean["cost_total"] / 1e6,
                 color=COLORS["red"], linewidth=2.0, label="Cost")

    # Fill profit area
    if "revenue" in monthly_mean.columns and "cost_total" in monthly_mean.columns:
        rev = monthly_mean["revenue"].values / 1e6
        cost = monthly_mean["cost_total"].values / 1e6
        ax5.fill_between(months, cost, rev,
                         where=(rev >= cost), alpha=0.15,
                         color=COLORS["green"], label="Profit (+)")
        ax5.fill_between(months, cost, rev,
                         where=(rev < cost), alpha=0.15,
                         color=COLORS["red"], label="Loss (−)")

    ax5.set_ylim(bottom=0)
    ax5.legend(fontsize=8, loc="upper left", framealpha=0.8)

    # Dual axis: margin %
    ax5b = ax5.twinx()
    if "revenue" in monthly_mean.columns and "profit" in monthly_mean.columns:
        margin = (monthly_mean["profit"] / monthly_mean["revenue"].clip(lower=1)) * 100
        ax5b.plot(months, margin,
                  color=COLORS["purple"], linewidth=1.5, linestyle=":",
                  label="Margin %")
        ax5b.set_ylabel("Profit Margin (%)", fontsize=10,
                        color=COLORS["purple"])
        ax5b.tick_params(axis="y", labelcolor=COLORS["purple"], labelsize=9)
        ax5b.spines["top"].set_visible(False)
        ax5b.legend(fontsize=8, loc="upper right", framealpha=0.8)

    # ══════════════════════════════════════════════════════════════════
    # Panel 6: SLA Compliance (V_rate + Throughput vs SLO)
    # ══════════════════════════════════════════════════════════════════
    ax6 = axes[2, 1]
    _setup_axes_style(ax6, "⑥ SLA Violation Rates",
                      "Month", "V_rate")

    if "V_rate_eMBB" in monthly_mean.columns:
        ax6.plot(months, monthly_mean["V_rate_eMBB"],
                 color=COLORS["blue"], linewidth=1.8, label="V_rate eMBB")

        # Shade per-repeat spread
        for rep in df["repeat"].unique():
            rd = df[df["repeat"] == rep]
            ax6.plot(rd["month"], rd["V_rate_eMBB"],
                     alpha=0.12, color=COLORS["blue"], linewidth=0.5)

    if "V_rate_URLLC" in monthly_mean.columns:
        ax6.plot(months, monthly_mean["V_rate_URLLC"],
                 color=COLORS["red"], linewidth=1.8, label="V_rate URLLC")

        for rep in df["repeat"].unique():
            rd = df[df["repeat"] == rep]
            ax6.plot(rd["month"], rd["V_rate_URLLC"],
                     alpha=0.12, color=COLORS["red"], linewidth=0.5)

    # SLA credit activation thresholds
    ax6.axhline(y=0.05, color=COLORS["gray"], linewidth=1.0,
                linestyle=":", alpha=0.6, label="Credit tier 1 (5%)")
    ax6.axhline(y=0.15, color=COLORS["gray"], linewidth=1.0,
                linestyle="-.", alpha=0.5, label="Credit tier 2 (15%)")

    ax6.set_ylim(0, 1.0)
    ax6.legend(fontsize=8, loc="upper right", framealpha=0.8)

    # ══════════════════════════════════════════════════════════════════
    # Final layout
    # ══════════════════════════════════════════════════════════════════
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

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