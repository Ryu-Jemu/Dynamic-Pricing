"""
Report generation script (Section 18).

Reads eval_monthly.csv and eval_summary.json from a run directory,
produces:
  - 6 PNG plots (profit, fees, rho, utilization, violation rates, churn/join)
  - report.md summary

Usage:
  python -m src.report --run_dir artifacts/<run_id>

References:
  [SB3_TIPS] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (headless safe)
import matplotlib.pyplot as plt

from src.models.utils import setup_logger

logger = logging.getLogger("oran.report")

# ---- Plot style ----
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "lines.linewidth": 1.5,
})

COLORS = {
    "eMBB": "#2196F3",
    "URLLC": "#FF5722",
    "profit": "#4CAF50",
    "revenue": "#2196F3",
    "cost": "#F44336",
    "rho": "#9C27B0",
    "util": "#FF9800",
}


def _load_data(run_dir: str):
    """Load eval CSV and summary JSON."""
    run_path = Path(run_dir)

    csv_path = run_path / "eval_monthly.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No eval_monthly.csv in {run_dir}")
    df = pd.read_csv(csv_path)

    summary_path = run_path / "eval_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    meta_path = run_path / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return df, summary, meta


def _repeats_stats(df: pd.DataFrame, col: str):
    """Compute per-month mean and std across repeats."""
    grouped = df.groupby("month")[col]
    return grouped.mean(), grouped.std().fillna(0)


def _plot_with_band(ax, months, mean, std, color, label, alpha=0.15):
    """Plot mean line with ±1σ shaded band."""
    ax.plot(months, mean, color=color, label=label)
    ax.fill_between(months, mean - std, mean + std, color=color, alpha=alpha)


# =====================================================================
# Individual plot functions
# =====================================================================

def plot_profit(df: pd.DataFrame, out_dir: Path):
    """Plot 1: Monthly profit, revenue, cost."""
    fig, ax = plt.subplots()
    for col, label, color in [
        ("profit", "Profit", COLORS["profit"]),
        ("revenue", "Revenue", COLORS["revenue"]),
        ("cost_total", "Total Cost", COLORS["cost"]),
    ]:
        if col in df.columns:
            m, s = _repeats_stats(df, col)
            _plot_with_band(ax, m.index, m.values, s.values, color, label)

    ax.set_xlabel("Month")
    ax.set_ylabel("KRW")
    ax.set_title("Monthly Profit / Revenue / Cost")
    ax.legend()
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(out_dir / "plot_profit.png", dpi=150)
    plt.close(fig)
    logger.info("Saved plot_profit.png")


def plot_fees(df: pd.DataFrame, out_dir: Path):
    """Plot 2: Monthly fees per slice."""
    fig, ax = plt.subplots()
    for sname, color in [("eMBB", COLORS["eMBB"]), ("URLLC", COLORS["URLLC"])]:
        col = f"fee_{sname}"
        if col in df.columns:
            m, s = _repeats_stats(df, col)
            _plot_with_band(ax, m.index, m.values, s.values, color, f"F_{sname}")

    ax.set_xlabel("Month")
    ax.set_ylabel("Fee (KRW/month)")
    ax.set_title("Monthly Subscription Fees by Slice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_fees.png", dpi=150)
    plt.close(fig)
    logger.info("Saved plot_fees.png")


def plot_rho(df: pd.DataFrame, out_dir: Path):
    """Plot 3: PRB share (rho_URLLC) and mean utilization."""
    fig, ax1 = plt.subplots()

    if "rho_URLLC" in df.columns:
        m, s = _repeats_stats(df, "rho_URLLC")
        _plot_with_band(ax1, m.index, m.values, s.values, COLORS["rho"], "ρ_URLLC")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("ρ_URLLC (PRB share)")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    if "mean_rho_util" in df.columns:
        m, s = _repeats_stats(df, "mean_rho_util")
        _plot_with_band(ax2, m.index, m.values, s.values, COLORS["util"], "Mean ρ_util")
    ax2.set_ylabel("Mean Utilization (ρ_util)")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("PRB Share & Utilization")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_rho.png", dpi=150)
    plt.close(fig)
    logger.info("Saved plot_rho.png")


def plot_utilization(df: pd.DataFrame, out_dir: Path):
    """Plot 4: Average per-user throughput per slice."""
    fig, ax = plt.subplots()
    for sname, color in [("eMBB", COLORS["eMBB"]), ("URLLC", COLORS["URLLC"])]:
        col = f"avg_T_{sname}"
        if col in df.columns:
            m, s = _repeats_stats(df, col)
            _plot_with_band(ax, m.index, m.values, s.values, color, f"T_avg {sname}")

    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Throughput (Mbps)")
    ax.set_title("Average Per-User Throughput by Slice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_throughput.png", dpi=150)
    plt.close(fig)
    logger.info("Saved plot_throughput.png")


def plot_violations(df: pd.DataFrame, out_dir: Path):
    """Plot 5: SLA violation rates per slice."""
    fig, ax = plt.subplots()
    for sname, color in [("eMBB", COLORS["eMBB"]), ("URLLC", COLORS["URLLC"])]:
        col = f"V_rate_{sname}"
        if col in df.columns:
            m, s = _repeats_stats(df, col)
            _plot_with_band(ax, m.index, m.values, s.values, color, f"V_{sname}")

    ax.set_xlabel("Month")
    ax.set_ylabel("Violation Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SLA Violation Rates by Slice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_violations.png", dpi=150)
    plt.close(fig)
    logger.info("Saved plot_violations.png")


def plot_churn_join(df: pd.DataFrame, out_dir: Path):
    """Plot 6: Churn and join counts per slice."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, sname in enumerate(["eMBB", "URLLC"]):
        ax = axes[idx]
        for col_suffix, label, ls in [
            (f"joins_{sname}", "Joins", "-"),
            (f"churns_{sname}", "Churns", "--"),
        ]:
            if col_suffix in df.columns:
                m, s = _repeats_stats(df, col_suffix)
                color = COLORS[sname]
                ax.plot(m.index, m.values, color=color, linestyle=ls, label=label)
                ax.fill_between(m.index, (m - s).values, (m + s).values,
                                color=color, alpha=0.1)

        # Also plot active users on twin axis
        n_col = f"N_active_{sname}"
        if n_col in df.columns:
            ax2 = ax.twinx()
            m2, s2 = _repeats_stats(df, n_col)
            ax2.plot(m2.index, m2.values, color="gray", linestyle=":", label="N_active")
            ax2.set_ylabel("N_active", color="gray")

        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.set_title(f"{sname}: Joins / Churns / Active")
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "plot_churn_join.png", dpi=150)
    plt.close(fig)
    logger.info("Saved plot_churn_join.png")


# =====================================================================
# Report generation
# =====================================================================

def generate_report(run_dir: str) -> str:
    """Generate all plots and report.md.

    Returns path to report.md.
    """
    run_path = Path(run_dir)
    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    df, summary, meta = _load_data(run_dir)

    # ---- Generate plots ----
    plot_profit(df, plots_dir)
    plot_fees(df, plots_dir)
    plot_rho(df, plots_dir)
    plot_utilization(df, plots_dir)
    plot_violations(df, plots_dir)
    plot_churn_join(df, plots_dir)

    # ---- Generate report.md ----
    report_lines = [
        "# O-RAN 1-Cell Slicing + Pricing — Evaluation Report",
        "",
        "## Run Information",
        "",
        f"- **Run ID**: {meta.get('run_id', 'N/A')}",
        f"- **Device**: {meta.get('device', summary.get('device', 'N/A'))}",
        f"- **Total timesteps**: {meta.get('total_timesteps', 'N/A')}",
        f"- **Training time**: {meta.get('train_time_sec', 'N/A')} sec",
        f"- **Model source**: {summary.get('model_source', 'N/A')}",
        f"- **Eval repeats**: {summary.get('n_repeats', 'N/A')}",
        "",
        "## Evaluation Summary (mean ± std across repeats)",
        "",
    ]

    # Summary table
    stat_keys = [
        ("mean_reward", "Mean Monthly Reward", "{:.4f}"),
        ("total_reward", "Total Episode Reward", "{:.2f}"),
        ("mean_profit", "Mean Monthly Profit (KRW)", "{:,.0f}"),
        ("total_profit", "Total Episode Profit (KRW)", "{:,.0f}"),
        ("final_N_eMBB", "Final Active eMBB", "{:.0f}"),
        ("final_N_URLLC", "Final Active URLLC", "{:.0f}"),
        ("mean_V_eMBB", "Mean Violation Rate eMBB", "{:.4f}"),
        ("mean_V_URLLC", "Mean Violation Rate URLLC", "{:.4f}"),
    ]

    report_lines.append("| Metric | Mean | Std | Min | Max |")
    report_lines.append("|--------|------|-----|-----|-----|")
    for key, label, fmt in stat_keys:
        if key in summary:
            s = summary[key]
            row = (
                f"| {label} "
                f"| {fmt.format(s['mean'])} "
                f"| {fmt.format(s['std'])} "
                f"| {fmt.format(s['min'])} "
                f"| {fmt.format(s['max'])} |"
            )
            report_lines.append(row)

    report_lines.extend([
        "",
        "## Plots",
        "",
        "### 1. Profit / Revenue / Cost",
        "![Profit](plots/plot_profit.png)",
        "",
        "### 2. Subscription Fees by Slice",
        "![Fees](plots/plot_fees.png)",
        "",
        "### 3. PRB Share & Utilization",
        "![Rho](plots/plot_rho.png)",
        "",
        "### 4. Average Per-User Throughput",
        "![Throughput](plots/plot_throughput.png)",
        "",
        "### 5. SLA Violation Rates",
        "![Violations](plots/plot_violations.png)",
        "",
        "### 6. Churn / Join / Active Users",
        "![ChurnJoin](plots/plot_churn_join.png)",
        "",
        "---",
        "",
        "## Configuration",
        "",
        "See `config.yaml` in the run directory for the full "
        "configuration used for this run.",
        "",
        "## Methodology",
        "",
        "- **Algorithm**: SAC (Soft Actor-Critic) via Stable-Baselines3",
        "- **Policy**: MlpPolicy with automatic entropy coefficient",
        "- **Evaluation**: Deterministic policy, N repeats, "
        "mean±std reported (no global seeds)",
        "- **Calibration**: demand (lognormal quantile fit), "
        "market (logistic churn offset), reward scale (random rollout p95)",
        "",
        "## References",
        "",
        "- [SAC] Haarnoja et al., Soft Actor-Critic, 2018",
        "- [SB3] Stable-Baselines3 documentation",
        "- [TS 38.104] 3GPP NR radio parameters",
        "",
        "*Report generated automatically by `src/report.py`*",
    ])

    report_path = run_path / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info("Report saved: %s", report_path)
    print(f"\nReport: {report_path}")
    print(f"Plots:  {plots_dir}/")

    return str(report_path)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots and report (Section 18)"
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to run directory (e.g. artifacts/20250207_143000)",
    )
    args = parser.parse_args()

    setup_logger("oran", logging.INFO)
    setup_logger("oran.report", logging.INFO)

    generate_report(args.run_dir)
