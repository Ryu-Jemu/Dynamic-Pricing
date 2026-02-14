"""
3D Simulation Environment Dashboard Generator (§16d).

Produces a self-contained HTML file with an interactive Three.js 3D
visualisation of the Best Model's evaluation behavior.  Follows the
same architecture as html_dashboard.py: Python reads CSV, computes
step-level data, injects JSON into an HTML template.

REVISION 9 — [M13e] Per-user spatial visualisation.
  - Reads users_init.csv for persistent (x, y) coordinates and segment data
  - Reads user_events_log.csv for step-by-step join/churn events
  - Injects USER_DATA, USER_EVENTS, USER_META into template
  - Fallback: works without user data (aggregate-only mode)

Prior:
  [M12] Original 3D dashboard module.

Scene elements:
  - Central base station tower
  - Coverage dome
  - URLLC / eMBB slice sectors (arc ~ rho_U)
  - Individual user meshes with persistent positions [M13]
  - Join/churn animations [M13]
  - Traffic flow particles [M13]
  - Load bars (green → red gradient)
  - QoS violation overlays
  - HUD panels for pricing, financial, and reward data
  - Timeline animation with play/pause/speed controls
  - Welcome overlay for first-time viewers [M13]
  - Mini-map with 2D user distribution [M13]

Usage:
  python -m oran3pt.sim3d_dashboard --csv outputs/rollout_log.csv
  python -m oran3pt.sim3d_dashboard --csv outputs/rollout_log.csv \\
      --users data/users_init.csv --events outputs/user_events_log.csv

References:
  [Wong Nat.Methods 2011]   Color-blind safe palette
  [Dulac-Arnold JMLR 2021]  Observability for real-world RL
  [3GPP TR 38.913 §6.1]     User spatial distribution
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("oran3pt.sim3d_dashboard")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_FILE = _TEMPLATE_DIR / "sim3d_dashboard.html"


def _select_best_repeat(df: pd.DataFrame) -> pd.DataFrame:
    """Select the repeat with highest cumulative profit."""
    if "repeat" not in df.columns:
        return df

    profits = df.groupby("repeat")["profit"].sum()
    best = profits.idxmax()
    logger.info("  Selected repeat %d (profit=%.0f)", best, profits[best])
    return df[df["repeat"] == best].copy().reset_index(drop=True)


def _extract_step_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract per-step data for 3D visualisation."""
    fields = [
        "step", "cycle", "cycle_step",
        "F_U", "p_over_U", "F_E", "p_over_E", "rho_U",
        "N_active", "N_U", "N_E", "N_inactive",
        "n_join", "n_churn",
        "L_U", "L_E", "C_U", "C_E",
        "pviol_U", "pviol_E",
        "revenue", "cost_total", "profit", "reward",
        "base_rev", "over_rev",
    ]
    available = [f for f in fields if f in df.columns]
    records = df[available].to_dict(orient="records")

    # Round floats for compactness
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float):
                rec[k] = round(v, 6)
            elif isinstance(v, (np.floating, np.integer)):
                rec[k] = round(float(v), 6)

    return records


def _compute_normalization(df: pd.DataFrame) -> Dict[str, float]:
    """Compute min/max ranges for normalizing 3D scene properties."""
    norms: Dict[str, float] = {}
    for col in ["revenue", "profit", "N_active", "L_U", "L_E", "C_U", "C_E"]:
        if col in df.columns:
            norms[f"{col}_min"] = float(df[col].min())
            norms[f"{col}_max"] = float(df[col].max())
    norms["N_total"] = float(df["N_active"].max() + df.get(
        "N_inactive", pd.Series(0)).max()) if "N_inactive" in df.columns else 500.0
    return norms


def _load_user_data(csv_path: Optional[str]) -> List[Dict[str, Any]]:
    """[M13e] Load per-user static data (coordinates, slice, segment).

    Returns empty list if file not found or missing required columns.
    """
    if csv_path is None:
        return []
    p = Path(csv_path)
    if not p.exists():
        logger.warning("Users CSV not found: %s — falling back to aggregate mode", p)
        return []

    df = pd.read_csv(p)
    required = {"user_id", "x", "y", "slice", "segment"}
    if not required.issubset(set(df.columns)):
        logger.warning("Users CSV missing spatial columns (x, y) — aggregate mode")
        return []

    users = []
    for _, row in df.iterrows():
        users.append({
            "id": int(row["user_id"]),
            "x": round(float(row["x"]), 3),
            "y": round(float(row["y"]), 3),
            "slice": str(row["slice"]),
            "segment": str(row["segment"]),
        })
    logger.info("  Loaded %d users with spatial coordinates", len(users))
    return users


def _load_user_events(csv_path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """[M13e] Load per-step user events (join/churn).

    Returns dict keyed by step number (as string for JSON).
    Empty dict if file not found.
    """
    if csv_path is None:
        return {}
    p = Path(csv_path)
    if not p.exists():
        logger.warning("Events CSV not found: %s — no per-user events", p)
        return {}

    df = pd.read_csv(p)
    required = {"step", "event_type", "user_id"}
    if not required.issubset(set(df.columns)):
        logger.warning("Events CSV missing required columns — skipping")
        return {}

    events: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        step_key = str(int(row["step"]))
        if step_key not in events:
            events[step_key] = []
        events[step_key].append({
            "type": str(row["event_type"]),
            "uid": int(row["user_id"]),
        })
    logger.info("  Loaded user events across %d steps", len(events))
    return events


def generate_3d_dashboard(
    csv_path: str,
    output_path: str = "outputs/sim3d_dashboard.html",
    config_path: Optional[str] = None,
    template_path: Optional[str] = None,
    users_csv: Optional[str] = None,
    events_csv: Optional[str] = None,
) -> Path:
    """Generate 3D simulation dashboard from evaluation CSV.

    Args:
        csv_path: Path to rollout_log.csv or similar step-level CSV.
        output_path: Where to write the HTML file.
        config_path: Optional YAML config for metadata.
        template_path: Override template file path.
        users_csv: [M13e] Path to users_init.csv with spatial coordinates.
        events_csv: [M13e] Path to user_events_log.csv.

    Returns:
        Path to generated HTML file.
    """
    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read template
    tmpl_p = Path(template_path) if template_path else _TEMPLATE_FILE
    if not tmpl_p.exists():
        raise FileNotFoundError(f"3D template not found: {tmpl_p}")
    template = tmpl_p.read_text(encoding="utf-8")

    # Read CSV
    logger.info("Reading evaluation log: %s", csv_path)
    df = pd.read_csv(csv_p)
    logger.info("  %d rows, %d columns", len(df), len(df.columns))

    # Select best repeat
    df = _select_best_repeat(df)
    n_steps = len(df)
    logger.info("  Using %d steps for 3D visualisation", n_steps)

    # Extract data
    step_data = _extract_step_data(df)
    norms = _compute_normalization(df)

    # [M13e] Load per-user data
    user_data = _load_user_data(users_csv)
    user_events = _load_user_events(events_csv)

    # Build metadata
    meta = {
        "n_steps": n_steps,
        "n_cycles": int(df["cycle"].max()) if "cycle" in df.columns else 0,
        "steps_per_cycle": int(
            df.groupby("cycle").size().median()) if "cycle" in df.columns else 30,
    }

    # [M13e] User metadata
    user_meta = {
        "N_total": len(user_data) if user_data else int(norms.get("N_total", 500)),
        "N_active_init": len([u for u in user_data if True]),  # overridden from events
        "cell_radius": 20.0,
        "has_spatial": len(user_data) > 0,
    }
    # Count initial active from events
    init_events = user_events.get("0", [])
    user_meta["N_active_init"] = len(
        [e for e in init_events if e["type"] == "initial_active"]
    ) if init_events else int(norms.get("N_active_min", 200))

    # Inject into template
    html = template
    html = html.replace("__STEP_DATA__",
                         json.dumps(step_data, separators=(",", ":")))
    html = html.replace("__NORMS__",
                         json.dumps(norms, separators=(",", ":")))
    html = html.replace("__META__",
                         json.dumps(meta, separators=(",", ":")))
    html = html.replace("__USER_DATA__",
                         json.dumps(user_data, separators=(",", ":")))
    html = html.replace("__USER_EVENTS__",
                         json.dumps(user_events, separators=(",", ":")))
    html = html.replace("__USER_META__",
                         json.dumps(user_meta, separators=(",", ":")))

    # Write output
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html, encoding="utf-8")
    logger.info("3D Dashboard written → %s (%d KB)", out_p, len(html) // 1024)

    return out_p


def main() -> None:
    """CLI entry point for 3D dashboard generation."""
    parser = argparse.ArgumentParser(
        description="Generate 3D simulation dashboard [M12][M13]")
    parser.add_argument(
        "--csv", default="outputs/rollout_log.csv",
        help="Path to step-level evaluation CSV")
    parser.add_argument(
        "--output", default="outputs/sim3d_dashboard.html",
        help="Output HTML file path")
    parser.add_argument(
        "--config", default=None,
        help="Optional YAML config for metadata")
    parser.add_argument(
        "--template", default=None,
        help="Override template path")
    parser.add_argument(
        "--users", default=None,
        help="[M13] Path to users_init.csv with spatial coordinates")
    parser.add_argument(
        "--events", default=None,
        help="[M13] Path to user_events_log.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s] %(message)s",
    )

    generate_3d_dashboard(
        csv_path=args.csv,
        output_path=args.output,
        config_path=args.config,
        template_path=args.template,
        users_csv=args.users,
        events_csv=args.events,
    )


if __name__ == "__main__":
    main()
