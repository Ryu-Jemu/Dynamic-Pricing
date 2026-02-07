"""
Evaluation script (Section 18).

Runs multiple evaluation repeats (N=3–5) with the trained SAC model.
Reports mean±std over repeats (no global seeds).  [SB3_TIPS]
Outputs CSV logs with per-month KPIs for each repeat.

Usage:
  python -m src.eval --config config/default.yaml --run_dir artifacts/<run_id>

If --run_dir is not specified, uses the most recent run in artifacts/.

References:
  [SB3_TIPS] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
  [TQDM]     https://tqdm.github.io/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from stable_baselines3 import SAC

from src.envs.oran_slicing_env import ORANSlicingEnv
from src.models.utils import load_config, select_device, setup_logger

logger = logging.getLogger("oran.eval")


def find_latest_run(base_dir: str = "artifacts") -> str:
    """Find the most recent run directory."""
    p = Path(base_dir)
    if not p.exists():
        raise FileNotFoundError(f"No artifacts directory: {base_dir}")
    runs = sorted([d for d in p.iterdir() if d.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {base_dir}")
    return str(runs[-1])


def evaluate_episode(
    model: SAC,
    env: ORANSlicingEnv,
    deterministic: bool = True,
) -> List[Dict[str, Any]]:
    """Run one evaluation episode, return per-month records."""
    obs, _ = env.reset()
    records: List[Dict[str, Any]] = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        record = {
            "month": info.get("month", 0),
            "reward": reward,
            "profit": info.get("profit", 0),
            "revenue": info.get("revenue", 0),
            "cost_total": info.get("cost_total", 0),
            "cost_energy": info.get("cost_energy", 0),
            "cost_sla_total": info.get("cost_sla_total", 0),
            "cost_resource": info.get("cost_resource", 0),
            "mean_rho_util": info.get("mean_rho_util", 0),
            "rho_URLLC": info.get("rho_URLLC", 0),
            "penalty": info.get("penalty", 0),
        }

        # Slice-level metrics
        for sname in ["eMBB", "URLLC"]:
            record[f"fee_{sname}"] = info.get("fees", {}).get(sname, 0)
            record[f"N_active_{sname}"] = info.get("N_active", {}).get(sname, 0)
            record[f"N_post_churn_{sname}"] = info.get("N_active_post_churn", {}).get(sname, 0)
            record[f"joins_{sname}"] = info.get("n_joins", {}).get(sname, 0)
            record[f"churns_{sname}"] = info.get("n_churns", {}).get(sname, 0)
            record[f"topups_{sname}"] = info.get("n_topups", {}).get(sname, 0)
            record[f"V_rate_{sname}"] = info.get("V_rates", {}).get(sname, 0)
            record[f"avg_T_{sname}"] = info.get("avg_T", {}).get(sname, 0)

        records.append(record)

    return records


def evaluate(
    config_path: str,
    run_dir: str,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    """Run multi-repeat evaluation.

    Returns:
        dict with summary statistics and path to CSV.
    """
    cfg = load_config(config_path)
    run_path = Path(run_dir)

    # ---- Load model ----
    device = select_device()
    # Try best model first, then final
    best_path = run_path / "best_model" / "best_model.zip"
    final_path = run_path / "final_model.zip"

    if best_path.exists():
        model_path = str(best_path)
        model_source = "best"
    elif final_path.exists():
        model_path = str(final_path)
        model_source = "final"
    else:
        raise FileNotFoundError(
            f"No model found in {run_dir}. "
            f"Looked for {best_path} and {final_path}"
        )

    logger.info("Loading model from: %s (%s)", model_path, model_source)
    model = SAC.load(model_path, device=device)

    # ---- Evaluate ----
    env = ORANSlicingEnv(cfg)
    all_records: List[List[Dict]] = []
    repeat_summaries: List[Dict[str, float]] = []

    for rep in tqdm(range(n_repeats), desc="Eval repeats"):
        records = evaluate_episode(model, env, deterministic=True)
        all_records.append(records)

        # Per-episode summary
        rewards = [r["reward"] for r in records]
        profits = [r["profit"] for r in records]
        repeat_summaries.append({
            "repeat": rep,
            "total_reward": sum(rewards),
            "mean_reward": float(np.mean(rewards)),
            "total_profit": sum(profits),
            "mean_profit": float(np.mean(profits)),
            "final_N_eMBB": records[-1].get("N_post_churn_eMBB", 0),
            "final_N_URLLC": records[-1].get("N_post_churn_URLLC", 0),
            "mean_V_eMBB": float(np.mean([r["V_rate_eMBB"] for r in records])),
            "mean_V_URLLC": float(np.mean([r["V_rate_URLLC"] for r in records])),
        })

    # ---- Write per-month CSV ----
    csv_path = run_path / "eval_monthly.csv"
    if all_records:
        fieldnames = ["repeat"] + list(all_records[0][0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rep_idx, records in enumerate(all_records):
                for rec in records:
                    row = {"repeat": rep_idx, **rec}
                    writer.writerow(row)
    logger.info("Eval CSV saved: %s", csv_path)

    # ---- Aggregate mean±std across repeats [SB3_TIPS] ----
    summary = {}
    for key in ["total_reward", "mean_reward", "total_profit", "mean_profit",
                "final_N_eMBB", "final_N_URLLC", "mean_V_eMBB", "mean_V_URLLC"]:
        vals = [s[key] for s in repeat_summaries]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    summary["n_repeats"] = n_repeats
    summary["model_source"] = model_source
    summary["device"] = device

    # Save summary JSON
    summary_path = run_path / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Eval summary saved: %s", summary_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model: {model_source} | Device: {device} | Repeats: {n_repeats}")
    print(f"  Mean reward:  {summary['mean_reward']['mean']:.4f} "
          f"± {summary['mean_reward']['std']:.4f}")
    print(f"  Mean profit:  {summary['mean_profit']['mean']:,.0f} "
          f"± {summary['mean_profit']['std']:,.0f} KRW")
    print(f"  Total profit: {summary['total_profit']['mean']:,.0f} "
          f"± {summary['total_profit']['std']:,.0f} KRW")
    print(f"  Final N_eMBB: {summary['final_N_eMBB']['mean']:.0f} "
          f"± {summary['final_N_eMBB']['std']:.0f}")
    print(f"  Final N_URLLC:{summary['final_N_URLLC']['mean']:.0f} "
          f"± {summary['final_N_URLLC']['std']:.0f}")
    print(f"  Mean V_eMBB:  {summary['mean_V_eMBB']['mean']:.4f}")
    print(f"  Mean V_URLLC: {summary['mean_V_URLLC']['mean']:.4f}")
    print("=" * 60)

    return summary


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained SAC (Section 18)")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Run directory (default: most recent in artifacts/)",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=None,
        help="Number of eval repeats (default: from config)",
    )
    args = parser.parse_args()

    setup_logger("oran", logging.INFO)
    setup_logger("oran.eval", logging.INFO)

    cfg = load_config(args.config)
    run_dir = args.run_dir or find_latest_run(
        cfg.get("artifacts", {}).get("base_dir", "artifacts")
    )
    n_repeats = args.n_repeats or cfg.get("training", {}).get("n_repeats", 3)

    evaluate(args.config, run_dir, n_repeats)
