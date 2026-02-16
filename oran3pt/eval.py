"""
Evaluation script — export rollout log + summary (§14 data source).

REVISION 10 — Changes from v9:
  [EP1] Chained episode evaluation for continuous mode
        In continuous mode (episode_cycles=1), chains 24 episodes into
        one repeat to produce 720-row rollout logs identical to v9 format.
        Population state persists across chained episodes.
        [Pardo ICML 2018; Gupta JSR 2006 — 24-month CLV horizon]
  Prior revisions:
  [M13d] Per-user event logging (user_events_log.csv) for 3D dashboard
         [Dulac-Arnold 2021 — per-user observability]
  [M8] Derived diagnostic columns (utilisation, margins, pop delta)
       [Dulac-Arnold 2021 — observability for real-world RL]
  [E9] Multi-seed model selection (evaluates best model across seeds)
  [F3] Fixed pandas FutureWarning in CLV groupby.apply

Outputs:
  outputs/rollout_log.csv       — per-step metrics across repeats
  outputs/eval_summary.csv      — aggregate statistics
  outputs/clv_report.csv        — CLV analysis
  outputs/user_events_log.csv   — [M13d] per-user join/churn events

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
                     model, repeat_id: int,
                     n_chains: int = 1,
                     ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run evaluation episodes, returning (records, user_events).

    [EP1] In continuous mode, chains `n_chains` consecutive episodes
    into one repeat. Population persists across episodes via env's
    continuous reset. Step numbers are renumbered globally (1..n_chains*T)
    so rollout_log.csv has the same 720-row format as v9.

    [M13d] user_events contains per-user join/churn events for the
    3D dashboard visualization.
    """
    obs, _ = env.reset()
    records: List[Dict[str, Any]] = []
    user_events: List[Dict[str, Any]] = []
    global_step = 0

    # [M13d] Record initial active users
    for uid in range(env.N_total):
        if env._active_mask[uid]:
            user_events.append({
                "step": 0, "event_type": "initial_active", "user_id": int(uid),
            })

    for chain_idx in range(n_chains):
        if chain_idx > 0:
            # In continuous mode, reset preserves population state
            obs, _ = env.reset()

        done = False
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # [M13d] Extract user events before DataFrame conversion
            churned = info.pop("churned_user_ids", [])
            joined = info.pop("joined_user_ids", [])

            # [EP1] Renumber step globally across chained episodes
            global_step += 1
            info["step"] = global_step
            info["cycle"] = (global_step - 1) // env.T + 1

            for uid in churned:
                user_events.append({
                    "step": global_step, "event_type": "churn",
                    "user_id": int(uid),
                })
            for uid in joined:
                user_events.append({
                    "step": global_step, "event_type": "join",
                    "user_id": int(uid),
                })

            info["repeat"] = repeat_id
            records.append(info)
            done = terminated or truncated

    return records, user_events


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

    # [EP1] In continuous mode (episode_cycles=1), chain 24 episodes
    # per repeat to produce 720 rows matching v9 rollout format.
    # CLV computation uses (step-1)//T grouping which works correctly.
    # [Gupta JSR 2006] — 24-month CLV horizon
    episode_mode = cfg.get("time", {}).get("episode_mode", "episodic")
    if episode_mode == "continuous":
        clv_horizon = cfg.get("clv", {}).get("horizon_months", 24)
        n_chains = clv_horizon  # 24 episodes chained = 24 billing cycles
    else:
        n_chains = 1

    all_records: List[Dict[str, Any]] = []
    all_user_events: dict[int, List[Dict[str, Any]]] = {}  # [M13d] per-repeat
    for rep in tqdm(range(n_repeats), desc="Eval repeats"):
        records, user_events = evaluate_episode(
            env, model, repeat_id=rep, n_chains=n_chains)
        all_records.extend(records)
        all_user_events[rep] = user_events

    rollout_path = out / "rollout_log.csv"
    if all_records:
        df = pd.DataFrame(all_records)

        # [M8] Derived diagnostic columns
        df["urllc_util"] = df["L_U"] / df["C_U"].clip(lower=1e-6)
        df["embb_util"] = df["L_E"] / df["C_E"].clip(lower=1e-6)
        df["sla_revenue_ratio"] = df["sla_penalty"] / df["revenue"].clip(lower=1e-6)
        df["profit_margin"] = df["profit"] / df["revenue"].clip(lower=1e-6)
        df["population_delta"] = df["n_join"] - df["n_churn"]

        df.to_csv(rollout_path, index=False)
        logger.info("Rollout log: %d rows -> %s", len(df), rollout_path)

        # [M13d] Write user events for best repeat (highest cumulative profit)
        profits_per_rep = df.groupby("repeat")["profit"].sum()
        best_rep = int(profits_per_rep.idxmax())
        best_events = all_user_events.get(best_rep, [])
        if best_events:
            events_path = out / "user_events_log.csv"
            pd.DataFrame(best_events).to_csv(events_path, index=False)
            logger.info("User events log: %d events (repeat %d) -> %s",
                        len(best_events), best_rep, events_path)

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
