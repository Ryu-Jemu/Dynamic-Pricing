"""
Business Dashboard Generator — executive-level KPI report.

Reads evaluation outputs (rollout_log.csv, clv_report.csv) and generates
a self-contained HTML dashboard designed for non-technical stakeholders
(C-level, finance, marketing, network operations).

Follows the same template-injection pattern as html_dashboard.py:
load CSV → compute metrics → inject JSON into HTML template →
write single-file dashboard.

Usage:
  python -m oran3pt.business_dashboard \\
      --csv outputs/rollout_log.csv \\
      --clv outputs/clv_report.csv \\
      --config config/default.yaml \\
      --output outputs/business_dashboard.html

References:
  [Grubb AER 2009]      3-part tariff economic model
  [Gupta JSR 2006]      CLV methodology
  [Wong Nat.Methods 2011] Color-blind safe palette
  [Henderson AAAI 2018]  Baseline comparison
  [Dulac-Arnold 2021]    Operational dashboard design
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .business_metrics import compute_all_metrics, compute_monthly_subscribers
from .utils import load_config

logger = logging.getLogger("oran3pt.business_dashboard")

# Template directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_FILE = _TEMPLATE_DIR / "business_dashboard.html"


def _build_json(data: Any) -> str:
    """Compact JSON serialisation for template injection."""

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(data, separators=(",", ":"), default=_default)


def _round_nested(obj: Any, decimals: int = 4) -> Any:
    """Recursively round floats in nested dict/list structures."""
    if isinstance(obj, dict):
        return {k: _round_nested(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_nested(v, decimals) for v in obj]
    if isinstance(obj, float):
        return round(obj, decimals)
    return obj


def generate_business_dashboard(
    csv_path: str,
    output_path: str = "outputs/business_dashboard.html",
    clv_path: Optional[str] = None,
    config_path: Optional[str] = None,
    template_path: Optional[str] = None,
) -> Path:
    """Generate HTML business dashboard from evaluation CSV data.

    Args:
        csv_path: Path to rollout_log.csv.
        output_path: Where to write the HTML file.
        clv_path: Path to clv_report.csv (optional).
        config_path: Path to YAML config (optional, for tariff info).
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
            f"Business dashboard template not found: {tmpl_p}\n"
            f"Expected at: {_TEMPLATE_FILE}")
    template = tmpl_p.read_text(encoding="utf-8")

    # Read rollout data
    logger.info("Reading rollout log: %s", csv_path)
    df = pd.read_csv(csv_p)
    logger.info("  %d rows, %d columns", len(df), len(df.columns))

    # Read CLV report
    clv_df = None
    if clv_path and Path(clv_path).exists():
        clv_df = pd.read_csv(clv_path)
        logger.info("  CLV report: %s", clv_path)

    # Read config
    cfg = None
    T = 30  # default steps per cycle
    if config_path and Path(config_path).exists():
        cfg = load_config(config_path)
        T = cfg.get("time", {}).get("steps_per_cycle", 30)
        logger.info("  Config: %s (T=%d)", config_path, T)

    # Compute all business metrics
    metrics = compute_all_metrics(df, clv_df, cfg, T)
    metrics = _round_nested(metrics)

    # Monthly subscriber data (separate for chart rendering)
    monthly = compute_monthly_subscribers(df, T)
    monthly_data = _round_nested(monthly.to_dict(orient="records"))

    # Build metadata strings
    n_steps = len(df)
    n_repeats = df["repeat"].nunique() if "repeat" in df.columns else 1
    mean_N = df["N_active"].mean()

    subtitle = (
        f"5G Single Cell · {n_steps:,} Steps · "
        f"{n_repeats} Repeats · "
        f"평균 {mean_N:.0f} 가입자"
    )
    footer_left = "O-RAN 3-Part Tariff · SAC Agent · Business Dashboard"
    import datetime
    footer_right = f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}"

    # Inject data into template
    html = template
    html = html.replace("__METRICS__", _build_json(metrics))
    html = html.replace("__MONTHLY__", _build_json(monthly_data))
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__FOOTER_LEFT__", footer_left)
    html = html.replace("__FOOTER_RIGHT__", footer_right)

    # Write output
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html, encoding="utf-8")
    logger.info("Business dashboard written -> %s (%d KB)",
                out_p, len(html) // 1024)

    return out_p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate business KPI dashboard from evaluation data")
    parser.add_argument(
        "--csv", default="outputs/rollout_log.csv",
        help="Path to rollout_log.csv")
    parser.add_argument(
        "--clv", default=None,
        help="Path to clv_report.csv")
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config (for tariff/time parameters)")
    parser.add_argument(
        "--output", default="outputs/business_dashboard.html",
        help="Output HTML file path")
    parser.add_argument(
        "--template", default=None,
        help="Override dashboard template path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s] %(message)s")

    # Auto-detect CLV and config paths if not specified
    csv_dir = Path(args.csv).parent
    clv_path = args.clv
    if clv_path is None:
        candidate = csv_dir / "clv_report.csv"
        if candidate.exists():
            clv_path = str(candidate)

    config_path = args.config
    if config_path is None:
        candidate = Path("config/default.yaml")
        if candidate.exists():
            config_path = str(candidate)

    generate_business_dashboard(
        csv_path=args.csv,
        output_path=args.output,
        clv_path=clv_path,
        config_path=config_path,
        template_path=args.template,
    )


if __name__ == "__main__":
    main()
