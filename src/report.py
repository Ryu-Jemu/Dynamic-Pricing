"""
Report generation from evaluation CSV.

Reads eval.csv, generates summary statistics and plots.

Usage:
  python -m src.report --run_dir artifacts/<run_id>

References:
  [SB3_TIPS]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("oran.report")


def generate_report(run_dir: str) -> None:
    """Generate summary report from eval CSV."""
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

    # Summary stats per month
    monthly = df.groupby("month").agg({
        "profit": ["mean", "std"],
        "reward": ["mean", "std"],
        "fee_eMBB": "mean",
        "fee_URLLC": "mean",
        "N_active_eMBB": "mean",
        "N_active_URLLC": "mean",
        "joins_eMBB": "mean",
        "churns_eMBB": "mean",
        "V_rate_eMBB": "mean",
        "V_rate_URLLC": "mean",
    })

    report_path = run_path / "report.md"
    with open(report_path, "w") as f:
        f.write("# O-RAN Slicing Evaluation Report\n\n")
        f.write(f"**Source**: {csv_path.name}\n")
        f.write(f"**Records**: {len(df)}\n")
        f.write(f"**Repeats**: {df['repeat'].nunique()}\n\n")

        f.write("## Key Metrics (Episode End)\n\n")
        last_month = df["month"].max()
        final = df[df["month"] == last_month]
        f.write(f"- Mean profit (final month): {final['profit'].mean():.0f} KRW\n")
        f.write(f"- Mean reward (final month): {final['reward'].mean():.4f}\n")
        f.write(f"- Mean N_active_eMBB: {final['N_active_eMBB'].mean():.1f}\n")
        f.write(f"- Mean N_active_URLLC: {final['N_active_URLLC'].mean():.1f}\n")
        f.write(f"- Mean fee_eMBB: {final['fee_eMBB'].mean():.0f} KRW\n")
        f.write(f"- Mean fee_URLLC: {final['fee_URLLC'].mean():.0f} KRW\n")
        f.write(f"- Mean V_rate_eMBB: {final['V_rate_eMBB'].mean():.4f}\n")
        f.write(f"- Mean V_rate_URLLC: {final['V_rate_URLLC'].mean():.4f}\n\n")

        f.write("## Monthly Summary\n\n")
        f.write(monthly.to_markdown())
        f.write("\n")

    logger.info("Report written: %s", report_path)

    # Generate plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # Plot 1: Profit over time
        ax = axes[0, 0]
        for rep in df["repeat"].unique():
            rep_data = df[df["repeat"] == rep]
            ax.plot(rep_data["month"], rep_data["profit"], alpha=0.5)
        ax.set_title("Monthly Profit")
        ax.set_xlabel("Month")
        ax.set_ylabel("Profit (KRW)")

        # Plot 2: Fees over time
        ax = axes[0, 1]
        monthly_mean = df.groupby("month").mean(numeric_only=True)
        ax.plot(monthly_mean.index, monthly_mean["fee_eMBB"], label="eMBB")
        ax.plot(monthly_mean.index, monthly_mean["fee_URLLC"], label="URLLC")
        ax.set_title("Mean Fees")
        ax.set_xlabel("Month")
        ax.set_ylabel("Fee (KRW)")
        ax.legend()

        # Plot 3: Active users
        ax = axes[1, 0]
        ax.plot(monthly_mean.index, monthly_mean["N_active_eMBB"], label="eMBB")
        ax.plot(monthly_mean.index, monthly_mean["N_active_URLLC"], label="URLLC")
        ax.set_title("Active Users")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.legend()

        # Plot 4: Joins/Churns
        ax = axes[1, 1]
        ax.plot(monthly_mean.index, monthly_mean["joins_eMBB"], label="Joins eMBB")
        ax.plot(monthly_mean.index, monthly_mean["churns_eMBB"], label="Churns eMBB")
        ax.set_title("Joins vs Churns (eMBB)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.legend()

        # Plot 5: Violation rates
        ax = axes[2, 0]
        ax.plot(monthly_mean.index, monthly_mean["V_rate_eMBB"], label="eMBB")
        ax.plot(monthly_mean.index, monthly_mean["V_rate_URLLC"], label="URLLC")
        ax.set_title("SLA Violation Rates")
        ax.set_xlabel("Month")
        ax.set_ylabel("V_rate")
        ax.legend()

        # Plot 6: Reward
        ax = axes[2, 1]
        for rep in df["repeat"].unique():
            rep_data = df[df["repeat"] == rep]
            ax.plot(rep_data["month"], rep_data["reward"], alpha=0.5)
        ax.set_title("Reward")
        ax.set_xlabel("Month")
        ax.set_ylabel("Reward")

        plt.tight_layout()
        plot_path = run_path / "eval_plots.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("Plots saved: %s", plot_path)

    except ImportError:
        logger.warning("matplotlib not available; skipping plots")


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
