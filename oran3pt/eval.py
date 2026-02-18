"""
Evaluation script — export rollout log + summary (§14 data source).

REVISION 10 — Changes from v9:
  [EP1] Chained episode evaluation for continuous mode
        In continuous mode (episode_cycles=1), chains 24 episodes into
        one repeat to produce 720-row rollout logs identical to v9 format.
        Population state persists across chained episodes.
        [Pardo ICML 2018; Gupta JSR 2006 — 24-month CLV horizon]
  Prior revisions:
  [M8] Derived diagnostic columns (utilisation, margins, pop delta)
       [Dulac-Arnold 2021 — observability for real-world RL]
  [E9] Multi-seed model selection (evaluates best model across seeds)
  [F3] Fixed pandas FutureWarning in CLV groupby.apply

  [M15] Automatic dashboard generation after evaluation
        Calls html_dashboard, png_dashboard, business_dashboard
        with graceful fallback on import/runtime errors.
        --no-dashboard flag to suppress.
        [Dulac-Arnold JMLR 2021 — post-training observability]

Outputs:
  outputs/rollout_log.csv       — per-step metrics across repeats
  outputs/eval_summary.csv      — aggregate statistics
  outputs/clv_report.csv        — CLV analysis
  outputs/training_convergence_dashboard.html  — [M15] HTML convergence
  outputs/01..07_*.png          — [M15] PNG dashboard sheets (7)
  outputs/business_dashboard.html — [M15] business KPI dashboard

Usage:
  python -m oran3pt.eval --config config/default.yaml --model outputs/best_model.zip
  python -m oran3pt.eval --no-dashboard   # skip dashboard generation
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
                     ) -> List[Dict[str, Any]]:
    """Run evaluation episodes, returning records.

    [EP1] In continuous mode, chains `n_chains` consecutive episodes
    into one repeat. Population persists across episodes via env's
    continuous reset. Step numbers are renumbered globally (1..n_chains*T)
    so rollout_log.csv has the same 720-row format as v9.
    """
    obs, _ = env.reset()
    records: List[Dict[str, Any]] = []
    global_step = 0

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

            # [EP1] Renumber step globally across chained episodes
            global_step += 1
            info["step"] = global_step
            info["cycle"] = (global_step - 1) // env.T + 1

            info["repeat"] = repeat_id
            records.append(info)
            done = terminated or truncated

    return records


def run_evaluation(cfg: Dict[str, Any],
                   model_path: Optional[str] = None,
                   users_csv: Optional[str] = None,
                   n_repeats: int = 5,
                   output_dir: str = "outputs",
                   config_path: Optional[str] = None,
                   generate_dashboard: bool = True) -> Path:
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

    # [ME-3] Per-repeat seed ensures statistically independent evaluations
    # Each repeat uses seed = base_seed + repeat_id for reproducibility
    # [Henderson AAAI 2018] — independent evaluation runs
    base_seed = cfg.get("training", {}).get("eval_base_seed", 10000)
    all_records: List[Dict[str, Any]] = []
    for rep in tqdm(range(n_repeats), desc="Eval repeats"):
        env.reset(seed=base_seed + rep)
        records = evaluate_episode(
            env, model, repeat_id=rep, n_chains=n_chains)
        all_records.extend(records)

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

        summary_path = out / "eval_summary.csv"
        num_cols = df.select_dtypes(include="number").columns
        mean_vals = df[num_cols].mean()
        std_vals = df[num_cols].std()
        summary = pd.DataFrame({"mean": mean_vals, "std": std_vals})
        summary.to_csv(summary_path)
        logger.info("Summary -> %s", summary_path)

        if cfg.get("clv", {}).get("enabled", True):
            _compute_clv_report(cfg, df, out)

    # [M15] Generate dashboards after all CSV outputs are written
    if generate_dashboard:
        generate_dashboards(cfg, output_dir=output_dir,
                            config_path=config_path)

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


def generate_dashboards(cfg: Dict[str, Any],
                        output_dir: str = "outputs",
                        config_path: Optional[str] = None,
                        ) -> List[Path]:
    """[M15] Generate all post-evaluation dashboards.

    Produces (when input CSVs exist):
      1. HTML convergence dashboard  (from train_log_seed0.csv)
      2. PNG dashboard sheets × 7    (from train/eval CSVs)  [M11]
      3. Business KPI dashboard      (from rollout_log.csv)  [M14]

    Each generator is wrapped in try/except so one failure does not
    block the others.  Imports are deferred to avoid hard dependencies.

    [Dulac-Arnold JMLR 2021] — post-training observability
    [Henderson AAAI 2018]    — multi-seed convergence analysis
    """
    out = Path(output_dir)
    generated: List[Path] = []

    # ── 1. HTML Convergence Dashboard ──
    train_csv = out / "train_log_seed0.csv"
    fallback_csv = out / "rollout_log.csv"
    src_csv = str(train_csv) if train_csv.exists() else (
        str(fallback_csv) if fallback_csv.exists() else None)
    if src_csv is not None:
        try:
            from .html_dashboard import generate_html_dashboard
            p = generate_html_dashboard(
                csv_path=src_csv,
                output_path=str(out / "training_convergence_dashboard.html"),
                seed=0, revision="10")
            generated.append(Path(p))
            logger.info("[M15] HTML convergence dashboard -> %s", p)
        except Exception as e:
            logger.warning("[M15] HTML dashboard failed: %s", e)

    # ── 2. PNG Dashboard Sheets [M11] ──
    try:
        from .png_dashboard import generate_all_pngs
        cp = config_path or "config/default.yaml"
        pngs = generate_all_pngs(
            output_dir=output_dir, config_path=cp,
            mode="auto", dpi=180)
        generated.extend(pngs)
        logger.info("[M15] PNG sheets: %d files", len(pngs))
    except Exception as e:
        logger.warning("[M15] PNG dashboard failed: %s", e)

    # ── 3. Business KPI Dashboard [M14] ──
    rollout_csv = out / "rollout_log.csv"
    if rollout_csv.exists():
        try:
            from .business_dashboard import generate_business_dashboard
            clv_csv = out / "clv_report.csv"
            p = generate_business_dashboard(
                csv_path=str(rollout_csv),
                output_path=str(out / "business_dashboard.html"),
                clv_path=str(clv_csv) if clv_csv.exists() else None,
                config_path=config_path)
            generated.append(Path(p))
            logger.info("[M15] Business dashboard -> %s", p)
        except Exception as e:
            logger.warning("[M15] Business dashboard failed: %s", e)

    logger.info("[M15] Dashboard generation complete: %d files", len(generated))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="outputs/best_model")
    parser.add_argument("--users", default="data/users_init.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-dashboard", action="store_true", default=False,
                        help="Skip dashboard generation after evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(name)s] %(message)s")
    cfg = load_config(args.config)
    users_csv = args.users if Path(args.users).exists() else None
    run_evaluation(cfg, model_path=args.model, users_csv=users_csv,
                   n_repeats=args.repeats, output_dir=args.output,
                   config_path=args.config,
                   generate_dashboard=not args.no_dashboard)


if __name__ == "__main__":
    main()
