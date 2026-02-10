"""
Evaluation script for O-RAN Slicing + Pricing agent.

Runs trained SAC policy for N repeats, collects per-month metrics,
and writes CSV for report generation.

Usage:
  python -m src.eval --config config/calibrated.yaml \\
                     --model artifacts/<run_id>/best_model.zip \\
                     --output artifacts/<run_id>/eval.csv \\
                     --repeats 3

References:
  [SB3_SAC] [SB3_TIPS]
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("oran.eval")


def evaluate_episode(env, model, repeat_id: int) -> List[Dict[str, Any]]:
    """Run one full episode and collect per-month records.

    Parameters
    ----------
    env : OranSlicingEnv
    model : SB3 SAC model (or None for random policy)
    repeat_id : int

    Returns
    -------
    List of dicts, one per month.
    """
    obs, info = env.reset()
    records: List[Dict[str, Any]] = []
    done = False

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── FIX C4: Use flat keys matching env's info dict ──
        # The env returns flat keys like info["fee_eMBB"], NOT nested
        # dicts like info["fees"]["eMBB"].
        record: Dict[str, Any] = {
            "repeat": repeat_id,
            "month": info.get("month", 0),
            "reward": reward,
            "profit": info.get("profit", 0),
            "revenue": info.get("revenue", 0),
            "cost_total": info.get("cost_total", 0),
            "rho_URLLC": info.get("rho_URLLC", 0),
        }

        for sname in ["eMBB", "URLLC"]:
            # Flat key access — matches env._run_month() output
            record[f"fee_{sname}"] = info.get(f"fee_{sname}", 0)
            record[f"N_active_{sname}"] = info.get(f"N_active_{sname}", 0)
            record[f"N_post_churn_{sname}"] = info.get(f"N_post_churn_{sname}", 0)
            record[f"joins_{sname}"] = info.get(f"joins_{sname}", 0)
            record[f"churns_{sname}"] = info.get(f"churns_{sname}", 0)
            record[f"V_rate_{sname}"] = info.get(f"V_rate_{sname}", 0)
            record[f"avg_T_{sname}"] = info.get(f"avg_T_{sname}", 0)
            record[f"rho_util_{sname}"] = info.get(f"rho_util_{sname}", 0)
            record[f"topups_{sname}"] = info.get(f"topups_{sname}", 0)

        records.append(record)

    return records


def run_evaluation(cfg: Dict[str, Any], model_path: Optional[str] = None,
                   n_repeats: int = 3, output_path: str = "eval.csv") -> Path:
    """Run full evaluation and write CSV."""
    # ── FIX C1: Correct class name ──
    from src.envs.oran_slicing_env import OranSlicingEnv

    env = OranSlicingEnv(cfg)

    # Load model if provided
    model = None
    if model_path is not None:
        try:
            from stable_baselines3 import SAC
            model = SAC.load(model_path, env=env)
            logger.info("Loaded model from %s", model_path)
        except Exception as e:
            logger.warning("Could not load model (%s); using random policy.", e)

    all_records: List[Dict[str, Any]] = []
    for rep in tqdm(range(n_repeats), desc="Eval repeats"):
        records = evaluate_episode(env, model, repeat_id=rep)
        all_records.extend(records)

    # Write CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    logger.info("Evaluation complete: %d records → %s", len(all_records), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SAC agent")
    parser.add_argument("--config", type=str, default="config/calibrated.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .zip")
    parser.add_argument("--output", type=str, default="artifacts/eval.csv")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )

    from src.models.utils import load_config
    cfg_path = args.config
    cfg = load_config(cfg_path)

    run_evaluation(cfg, model_path=args.model,
                   n_repeats=args.repeats, output_path=args.output)


if __name__ == "__main__":
    main()
