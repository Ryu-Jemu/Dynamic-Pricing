"""
HTML Convergence Dashboard Generator (§16b).

Reads step-level training logs and produces a self-contained HTML dashboard
with Chart.js visualisations of SAC training convergence.

REVISION 7 — New module.

Panels (10 charts):
  1. Financial Growth (revenue, profit, N_active)
  2. Network Efficiency (traffic volume, load factor)
  3. Reward Maximisation (mean reward P10–P90, p_viol)
  4. URLLC Pricing (F_U ± σ, p_over_U)
  5. eMBB Pricing (F_E ± σ, p_over_E)
  6. ρ_U Convergence (± σ band)
  7. User Retention (joins, churns, net Δ)
  8. Daily Load Risk (avg / peak by cycle day)
  9. Traffic Efficiency Scatter (L_E vs load factor)
  10. Episode Load Profile (step-level)

Usage:
  python -m oran3pt.html_dashboard --csv outputs/train_log_seed0.csv
  python -m oran3pt.html_dashboard --csv outputs/rollout_log.csv --output outputs/dashboard.html

References:
  [Wong Nat.Methods 2011]  Color-blind safe palette
  [Haarnoja 2018]          SAC
  [Henderson 2018]         Multi-seed evaluation
  [Dulac-Arnold 2021]      Real-world RL
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("oran3pt.html_dashboard")

# Template directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_FILE = _TEMPLATE_DIR / "convergence_dashboard.html"


def _detect_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign episode IDs by detecting step counter resets."""
    df = df.copy()
    if "step" not in df.columns:
        raise ValueError("CSV must contain 'step' column")

    # Episode boundary: step goes back to 1 (or decreases)
    resets = (df["step"] <= df["step"].shift(1).fillna(0)).astype(int)
    df["_episode"] = resets.cumsum()
    return df


def _compute_episode_aggregates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate step-level data into episode-level metrics."""
    episodes = []
    steps_per_episode = 0

    for ep_id, g in df.groupby("_episode"):
        n = len(g)
        steps_per_episode = n

        rec: Dict[str, Any] = {
            "episode": int(ep_id),
            "global_step": int((ep_id + 1) * n),
        }

        # Reward statistics
        rec["mean_reward"] = float(g["reward"].mean())
        rec["total_reward"] = float(g["reward"].sum())
        rec["reward_std"] = float(g["reward"].std())
        rec["reward_p10"] = float(np.percentile(g["reward"], 10))
        rec["reward_p90"] = float(np.percentile(g["reward"], 90))

        # Financial
        rec["total_profit"] = float(g["profit"].sum())
        rec["mean_profit"] = float(g["profit"].mean())
        rec["mean_revenue"] = float(g["revenue"].mean())
        rec["mean_cost"] = float(g["cost_total"].mean())
        rec["mean_base_rev"] = float(g["base_rev"].mean()) if "base_rev" in g else 0.0
        rec["mean_over_rev"] = float(g["over_rev"].mean()) if "over_rev" in g else 0.0
        mean_rev = rec["mean_revenue"]
        rec["profit_margin"] = float(rec["mean_profit"] / mean_rev) if mean_rev > 0 else 0.0

        # Actions — mean and std
        for col, key in [("F_U", "F_U"), ("F_E", "F_E"),
                         ("p_over_U", "p_over_U"), ("p_over_E", "p_over_E"),
                         ("rho_U", "rho_U")]:
            if col in g.columns:
                rec[f"mean_{key}"] = float(g[col].mean())
                rec[f"std_{key}"] = float(g[col].std())
            else:
                rec[f"mean_{key}"] = 0.0
                rec[f"std_{key}"] = 0.0

        # Users
        rec["mean_N_active"] = float(g["N_active"].mean())
        rec["init_N_active"] = int(g["N_active"].iloc[0])
        rec["final_N_active"] = int(g["N_active"].iloc[-1])
        rec["total_churn"] = int(g["n_churn"].sum())
        rec["total_join"] = int(g["n_join"].sum())
        rec["net_user_change"] = rec["total_join"] - rec["total_churn"]

        # QoS
        rec["mean_pviol_U"] = float(g["pviol_U"].mean())
        rec["mean_pviol_E"] = float(g["pviol_E"].mean())
        rec["pviol_E_gt50"] = float((g["pviol_E"] > 0.5).mean())

        # Load
        if "L_E" in g.columns and "C_E" in g.columns:
            rec["mean_L_E"] = float(g["L_E"].mean())
            rec["mean_C_E"] = float(g["C_E"].mean())
            load_factors = g["L_E"] / g["C_E"].clip(lower=1e-6)
            rec["mean_load_factor_E"] = float(load_factors.mean())
            rec["peak_load_factor_E"] = float(load_factors.max())
        else:
            rec["mean_L_E"] = 0.0
            rec["mean_C_E"] = 1.0
            rec["mean_load_factor_E"] = 0.0
            rec["peak_load_factor_E"] = 0.0

        if "L_U" in g.columns and "C_U" in g.columns:
            rec["mean_L_U"] = float(g["L_U"].mean())
            rec["mean_C_U"] = float(g["C_U"].mean())
        else:
            rec["mean_L_U"] = 0.0
            rec["mean_C_U"] = 1.0

        # Cost breakdown
        rec["mean_opex"] = float(g["cost_opex"].mean()) if "cost_opex" in g else 0.0
        rec["mean_energy"] = float(g["cost_energy"].mean()) if "cost_energy" in g else 0.0
        rec["mean_cac"] = float(g["cost_cac"].mean()) if "cost_cac" in g else 0.0
        rec["mean_sla"] = float(g["sla_penalty"].mean()) if "sla_penalty" in g else 0.0

        # Reward components
        rec["mean_smooth_penalty"] = float(
            g["smooth_penalty"].mean()) if "smooth_penalty" in g else 0.0
        rec["mean_retention_penalty"] = float(
            g["retention_penalty"].mean()) if "retention_penalty" in g else 0.0

        episodes.append(rec)

    return episodes


def _extract_last_episode(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract step-level data from the last episode for micro-analysis."""
    last_ep_id = df["_episode"].max()
    g = df[df["_episode"] == last_ep_id].copy()

    # Compute load_factor_E per step
    if "L_E" in g.columns and "C_E" in g.columns:
        g["load_factor_E"] = g["L_E"] / g["C_E"].clip(lower=1e-6)
    else:
        g["load_factor_E"] = 0.0

    fields = [
        "step", "cycle", "cycle_step",
        "F_U", "p_over_U", "F_E", "p_over_E", "rho_U",
        "N_active", "n_join", "n_churn",
        "L_U", "L_E", "C_U", "C_E",
        "pviol_U", "pviol_E",
        "load_factor_E",
        "revenue", "profit", "reward",
    ]
    available = [f for f in fields if f in g.columns]
    records = g[available].to_dict(orient="records")

    # Round floats for JSON size
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float):
                rec[k] = round(v, 10)

    return records


def _build_json(data: Any) -> str:
    """Compact JSON serialisation (no unnecessary whitespace)."""
    return json.dumps(data, separators=(",", ":"))


def generate_html_dashboard(
    csv_path: str,
    output_path: str = "outputs/training_convergence_dashboard.html",
    seed: int = 0,
    revision: str = "7",
    template_path: Optional[str] = None,
) -> Path:
    """Generate HTML convergence dashboard from a training/eval CSV.

    Args:
        csv_path: Path to step-level CSV (train_log_seed*.csv or rollout_log.csv).
        output_path: Where to write the HTML file.
        seed: Seed number for display in header/footer.
        revision: Revision label for footer.
        template_path: Override template file path.

    Returns:
        Path to generated HTML file.
    """
    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read template
    tmpl_p = Path(template_path) if template_path else _TEMPLATE_FILE
    if not tmpl_p.exists():
        raise FileNotFoundError(
            f"Dashboard template not found: {tmpl_p}\n"
            f"Expected at: {_TEMPLATE_FILE}")
    template = tmpl_p.read_text(encoding="utf-8")

    # Read and process CSV
    logger.info("Reading training log: %s", csv_path)
    df = pd.read_csv(csv_p)
    logger.info("  %d rows, %d columns", len(df), len(df.columns))

    # Detect episodes
    df = _detect_episodes(df)
    n_episodes = int(df["_episode"].nunique())
    n_steps = len(df)
    logger.info("  Detected %d episodes (%d total steps)", n_episodes, n_steps)

    # Compute aggregates
    ep_data = _compute_episode_aggregates(df)
    last_ep = _extract_last_episode(df)
    logger.info("  EP_DATA: %d episodes, LAST_EP: %d steps", len(ep_data), len(last_ep))

    # Build subtitle
    subtitle = (
        f"O-RAN 5G 3-Part Tariff · Seed {seed} · "
        f"{n_steps:,} Steps / {n_episodes} Episodes"
    )
    footer_left = f"O-RAN 3-Part Tariff · SAC Seed {seed} · Revision {revision}"

    # Inject data into template
    html = template
    html = html.replace("__EP_DATA__", _build_json(ep_data))
    html = html.replace("__LAST_EP__", _build_json(last_ep))
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__FOOTER_LEFT__", footer_left)

    # Write output
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html, encoding="utf-8")
    logger.info("Dashboard written → %s (%d KB)", out_p, len(html) // 1024)

    return out_p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML convergence dashboard from training CSV")
    parser.add_argument(
        "--csv", default="outputs/train_log_seed0.csv",
        help="Path to step-level training or eval CSV")
    parser.add_argument(
        "--output", default="outputs/training_convergence_dashboard.html",
        help="Output HTML file path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default="7")
    parser.add_argument(
        "--template", default=None,
        help="Override dashboard template path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s] %(message)s")

    generate_html_dashboard(
        csv_path=args.csv,
        output_path=args.output,
        seed=args.seed,
        revision=args.revision,
        template_path=args.template,
    )


if __name__ == "__main__":
    main()
