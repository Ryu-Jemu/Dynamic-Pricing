"""
Evaluation script — export rollout log + summary (§14 data source).

REVISION 7 — Changes:
  Updated for new info keys (pop_bonus, over_rev_E)
  Prior revisions:
  [E9] Multi-seed model selection (evaluates best model across seeds)
  [F3] Fixed pandas FutureWarning in CLV groupby.apply

Outputs:
  outputs/rollout_log.csv   — per-step metrics across repeats
  outputs/eval_summary.csv  — aggregate statistics
  outputs/clv_report.csv    — CLV analysis

Usage:
  python -m oran3pt.eval --config config/default.yaml --model outputs/best_model.zip
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import load_config
from .env import OranSlicingPricingEnv

logger = logging.getLogger("oran3pt.eval")


def evaluate_episode(env: OranSlicingPricingEnv,
                     model, repeat_id: int) -> List[Dict[str, Any]]:
    obs, _ = env.reset()
    records: List[Dict[str, Any]] = []
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        info["repeat"] = repeat_id
        records.append(info)
        done = terminated or truncated
    return records


def run_evaluation(cfg: Dict[str, Any],
                   model_path: Optional[str] = None,
                   users_csv: Optional[str] = None,
                   n_repeats: int = 5,
                   output_dir: str = "outputs") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = OranSlicingPricingEnv(cfg, users_csv=users_csv)

    model = None
    if model_path and Path(model_path).exists():
        try:
            from stable_baselines3 import SAC
            mp = model_path if model_path.endswith(".zip") else model_path + ".zip"
            if Path(mp).exists():
                model = SAC.load(mp, env=env)
                logger.info("Loaded model from %s", mp)
            else:
                logger.warning("Model file not found: %s — using random.", mp)
        except Exception as e:
            logger.warning("Could not load model (%s) — using random.", e)

    all_records: List[Dict[str, Any]] = []
    for rep in tqdm(range(n_repeats), desc="Eval repeats"):
        records = evaluate_episode(env, model, repeat_id=rep)
        all_records.extend(records)

    rollout_path = out / "rollout_log.csv"
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(rollout_path, index=False)
        logger.info("Rollout log: %d rows -> %s", len(df), rollout_path)

        summary_path = out / "eval_summary.csv"
        num_cols = df.select_dtypes(include="number").columns
        mean_vals = df[num_cols].mean()
        std_vals = df[num_cols].std()
        summary = pd.DataFrame({"mean": mean_vals, "std": std_vals})
        summary.to_csv(summary_path)
        logger.info("Summary -> %s", summary_path)

        if cfg.get("clv", {}).get("enabled", True):
            _compute_clv_report(cfg, df, out)

    return rollout_path


def _compute_clv_report(cfg: Dict[str, Any], df: pd.DataFrame,
                        out: Path) -> None:
    clv_cfg = cfg.get("clv", {})
    H = clv_cfg.get("horizon_months", 24)
    d = clv_cfg.get("discount_rate_monthly", 0.01)

    T = cfg["time"]["steps_per_cycle"]

    monthly_profit = df.groupby("repeat").apply(
        lambda g: g.groupby(g["step"].apply(lambda s: (s - 1) // T))["profit"].sum().mean(),
        include_groups=False,
    )
    mean_monthly_profit = float(monthly_profit.mean())

    mean_N = df["N_active"].mean()
    cf_per_user = mean_monthly_profit / max(mean_N, 1)

    churn_per_step = df["n_churn"].mean() / max(df["N_active"].mean(), 1)
    monthly_churn = 1.0 - (1.0 - churn_per_step) ** T
    retention = max(1.0 - monthly_churn, 0.01)

    clv = 0.0
    for k in range(H):
        clv += cf_per_user * (retention ** k) / ((1.0 + d) ** k)

    clv_path = out / "clv_report.csv"
    pd.DataFrame([{
        "mean_monthly_profit": mean_monthly_profit,
        "mean_N_active": mean_N,
        "cf_per_user_month": cf_per_user,
        "monthly_churn": monthly_churn,
        "monthly_retention": retention,
        "CLV_per_user": clv,
        "horizon_months": H,
        "discount_rate": d,
    }]).to_csv(clv_path, index=False)
    logger.info("CLV report -> %s  (CLV=%.0f KRW)", clv_path, clv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="outputs/best_model")
    parser.add_argument("--users", default="data/users_init.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(name)s] %(message)s")
    cfg = load_config(args.config)
    users_csv = args.users if Path(args.users).exists() else None
    run_evaluation(cfg, model_path=args.model, users_csv=users_csv,
                   n_repeats=args.repeats, output_dir=args.output)


if __name__ == "__main__":
    main()
