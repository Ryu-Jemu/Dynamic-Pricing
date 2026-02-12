"""
Streamlit evaluation dashboard (Requirement 14).

REVISION 7 — Enhancements:
  [R5] Added overage rate tracking
  [R6] Added population bonus tracking
  Prior revisions:
  [E4] Added load factor visualisations (P7b)
  [E6] Added retention penalty tracking
  [F4] width="stretch" (Streamlit deprecation fix)

Panels:
  1. Profit / Revenue / Cost time series
  2. Active users / Join / Churn
  3. QoS violation probabilities
  4. Action trajectories
  5. Distributions (profit, rho_U)
  6. CLV summary table
  7. Load factors
  8. Reward over time

Design:
  - Color-blind safe palette  [Wong, Nat. Methods 2011]

Usage:
  streamlit run oran3pt/dashboard_app.py -- --data outputs/rollout_log.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Color-blind safe palette [WONG_2011] ──
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
CYAN   = "#56B4E9"


def _make_static_dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    """Matplotlib fallback when Streamlit is unavailable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    g = df.groupby("step").mean(numeric_only=True)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("O-RAN 3-Part Tariff — Evaluation Dashboard (v7)",
                 fontsize=14, fontweight="bold")

    # P1: Profit / Revenue / Cost
    ax = axes[0, 0]
    ax.plot(g.index, g["revenue"] / 1e6, color=BLUE, label="Revenue")
    ax.plot(g.index, g["cost_total"] / 1e6, color=RED, label="Cost")
    ax.plot(g.index, g["profit"] / 1e6, color=GREEN, label="Profit")
    ax.set_title("P1  Profit / Revenue / Cost")
    ax.set_xlabel("Step"); ax.set_ylabel("M KRW")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P2: Active users / Join / Churn
    ax = axes[0, 1]
    ax.plot(g.index, g["N_active"], color=BLUE, label="N_active")
    ax2 = ax.twinx()
    ax2.plot(g.index, g["n_join"], color=GREEN, alpha=0.7, label="Joins")
    ax2.plot(g.index, g["n_churn"], color=RED, alpha=0.7, label="Churns")
    ax.set_title("P2  Users / Joins / Churns")
    ax.set_xlabel("Step"); ax.set_ylabel("N_active")
    ax2.set_ylabel("Joins / Churns")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # P3: QoS violation
    ax = axes[1, 0]
    ax.plot(g.index, g["pviol_U"], color=RED, label="p_viol URLLC")
    ax.plot(g.index, g["pviol_E"], color=BLUE, label="p_viol eMBB")
    ax.set_title("P3  QoS Violation Probability")
    ax.set_xlabel("Step"); ax.set_ylabel("p_viol")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P4: Actions
    ax = axes[1, 1]
    ax.plot(g.index, g["F_U"] / 1e3, color=BLUE, label="F_U (K)")
    ax.plot(g.index, g["F_E"] / 1e3, color=RED, label="F_E (K)")
    ax2 = ax.twinx()
    ax2.plot(g.index, g["rho_U"], color=GREEN, linestyle="--", label="rho_U")
    ax.set_title("P4  Action Trajectories")
    ax.set_xlabel("Step"); ax.set_ylabel("Fee (K KRW)")
    ax2.set_ylabel("rho_U")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(alpha=0.3)

    # P5: Profit distribution
    ax = axes[2, 0]
    ax.hist(df["profit"] / 1e6, bins=50, color=GREEN, alpha=0.7, edgecolor="white")
    ax.axvline(df["profit"].mean() / 1e6, color=RED, linestyle="--",
               label=f"mean={df['profit'].mean()/1e6:.2f}M")
    ax.set_title("P5  Profit Distribution")
    ax.set_xlabel("Profit (M KRW)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P6: rho_U distribution
    ax = axes[2, 1]
    ax.hist(df["rho_U"], bins=50, color=CYAN, alpha=0.7, edgecolor="white")
    ax.set_title("P6  rho_U Distribution")
    ax.set_xlabel("rho_U"); ax.set_ylabel("Count"); ax.grid(alpha=0.3)

    # P7: Load factors  [E4]
    ax = axes[3, 0]
    if "L_E" in g.columns and "C_E" in g.columns:
        ax.plot(g.index, g["L_E"], color=BLUE, alpha=0.7, lw=0.8, label="L_E (load)")
        ax.plot(g.index, g["C_E"], color=RED, lw=1.0, label="C_E (capacity)")
        ax.fill_between(g.index, g["L_E"], g["C_E"],
                        where=g["L_E"] > g["C_E"], color=RED, alpha=0.15)
        ax.set_title("P7  eMBB Load vs Capacity")
        ax.set_xlabel("Step"); ax.set_ylabel("GB/day")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Load data not available", transform=ax.transAxes,
                ha='center')

    # P8: Reward over time
    ax = axes[3, 1]
    ax.plot(g.index, g["reward"], color=ORANGE, alpha=0.7, lw=0.8)
    ax.set_title("P8  Reward Over Time")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = out_dir / "eval_dashboard.png"
    plt.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Dashboard saved -> {fig_path}")


def _run_streamlit(csv_path: str) -> None:
    """Streamlit interactive dashboard."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.set_page_config(page_title="O-RAN 3PT Dashboard v7", layout="wide")
    st.title("O-RAN 3-Part Tariff — Evaluation Dashboard (v7)")

    df = pd.read_csv(csv_path)
    g = df.groupby("step").mean(numeric_only=True)

    # ── KPI cards ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Profit", f"{df['profit'].mean()/1e6:.2f} M KRW")
    col2.metric("Mean N_active", f"{df['N_active'].mean():.0f}")
    col3.metric("Mean p_viol eMBB", f"{df['pviol_E'].mean():.3f}")
    col4.metric("Mean Reward", f"{df['reward'].mean():.4f}")

    # P1
    st.subheader("P1  Profit / Revenue / Cost")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g["revenue"]/1e6,
                             name="Revenue", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=g.index, y=g["cost_total"]/1e6,
                             name="Cost", line=dict(color=RED)))
    fig.add_trace(go.Scatter(x=g.index, y=g["profit"]/1e6,
                             name="Profit", line=dict(color=GREEN)))
    fig.update_layout(xaxis_title="Step", yaxis_title="M KRW", height=350)
    st.plotly_chart(fig, width="stretch")

    # P2 / P3
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("P2  Active Users & Market Flow")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=g.index, y=g["N_active"],
                                  name="N_active", line=dict(color=BLUE)),
                       secondary_y=False)
        fig2.add_trace(go.Scatter(x=g.index, y=g["n_join"],
                                  name="Joins", line=dict(color=GREEN, dash="dot")),
                       secondary_y=True)
        fig2.add_trace(go.Scatter(x=g.index, y=g["n_churn"],
                                  name="Churns", line=dict(color=RED, dash="dot")),
                       secondary_y=True)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, width="stretch")

    with c2:
        st.subheader("P3  QoS Violation Probability")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=g.index, y=g["pviol_U"],
                                  name="p_viol URLLC", line=dict(color=RED)))
        fig3.add_trace(go.Scatter(x=g.index, y=g["pviol_E"],
                                  name="p_viol eMBB", line=dict(color=BLUE)))
        fig3.update_layout(yaxis_range=[0, 1], height=300)
        st.plotly_chart(fig3, width="stretch")

    # P4: Action trajectories
    st.subheader("P4  Action Trajectories")
    act_cols = [("F_U", BLUE), ("F_E", RED), ("p_over_U", ORANGE),
                ("p_over_E", PURPLE), ("rho_U", GREEN)]
    tabs = st.tabs([c[0] for c in act_cols])
    for tab, (col, color) in zip(tabs, act_cols):
        with tab:
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=g.index, y=g[col],
                                       line=dict(color=color)))
            fig_a.update_layout(xaxis_title="Step", yaxis_title=col, height=250)
            st.plotly_chart(fig_a, width="stretch")

    # P5 / P6
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("P5  Profit Distribution")
        fig5 = go.Figure(data=[go.Histogram(
            x=df["profit"]/1e6, nbinsx=50,
            marker_color=GREEN, opacity=0.75)])
        fig5.update_layout(xaxis_title="Profit (M KRW)", height=300)
        st.plotly_chart(fig5, width="stretch")
    with c2:
        st.subheader("P6  rho_U Distribution")
        fig6 = go.Figure(data=[go.Histogram(
            x=df["rho_U"], nbinsx=50,
            marker_color=CYAN, opacity=0.75)])
        fig6.update_layout(xaxis_title="rho_U", height=300)
        st.plotly_chart(fig6, width="stretch")

    # P7: Load factors  [E4]
    if "L_E" in g.columns and "C_E" in g.columns:
        st.subheader("P7  eMBB Load vs Capacity  [E4]")
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=g.index, y=g["L_E"],
                                  name="L_E (load)", line=dict(color=BLUE)))
        fig7.add_trace(go.Scatter(x=g.index, y=g["C_E"],
                                  name="C_E (capacity)", line=dict(color=RED)))
        fig7.update_layout(xaxis_title="Step", yaxis_title="GB/day", height=300)
        st.plotly_chart(fig7, width="stretch")

    # P8: CLV table
    clv_path = Path(csv_path).parent / "clv_report.csv"
    if clv_path.exists():
        st.subheader("P8  CLV Summary  [Gupta JSR 2006]")
        st.dataframe(pd.read_csv(clv_path))

    st.caption(
        "References: [Haarnoja 2018] SAC | [Grubb 2009] 3-part tariff | "
        "[Gupta 2006] CLV | [Dulac-Arnold 2021] Real-world RL | "
        "[Ng 1999] Reward shaping | [Henderson 2018] Multi-seed eval | "
        "[Dalal 2018] Per-dim constraints | [Mguni 2019] Population reward"
    )


def main() -> None:
    csv_path = "outputs/rollout_log.csv"

    for i, arg in enumerate(sys.argv):
        if arg == "--data" and i + 1 < len(sys.argv):
            csv_path = sys.argv[i + 1]

    csv_p = Path(csv_path)
    if not csv_p.exists():
        print(f"Data file not found: {csv_path}")
        print("Run evaluation first: python -m oran3pt.eval")
        return

    try:
        import streamlit
        _run_streamlit(csv_path)
    except ImportError:
        print("Streamlit not available — generating static PNG dashboard.")
        df = pd.read_csv(csv_path)
        _make_static_dashboard(df, csv_p.parent)


if __name__ == "__main__":
    main()
