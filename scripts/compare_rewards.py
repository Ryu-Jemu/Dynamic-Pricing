#!/usr/bin/env python3
"""Compare reward mappings for calibrated reward profiles.

This script keeps the same calibrated profile values that were previously
stored as YAML-like text in this file, and prints reward values for a set
of reference profits.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List

# Random rollout profit statistics (reference)
ROLLOUT_STATS = {
    "n_episodes": 20,
    "n_samples": 1000,
    "seed": 42,
    "abs_profit_mean": 929_984,
    "abs_profit_p50": 706_866,
    "abs_profit_p75": 937_849,
    "abs_profit_p95": 1_392_116,
    "abs_profit_p99": 10_021_107,
}

# Calibrated reward profiles
PROFILES: Dict[str, Dict[str, float | str]] = {
    "log": {
        "reward_type": "log",
        "profit_scale": 724_177.0,
        "reward_clip": 2.0,
    },
    "tanh": {
        "reward_type": "tanh",
        "profit_scale": 1_364_050.0,
        "reward_clip": 2.0,
    },
    "linear": {
        "reward_type": "linear",
        "profit_scale": 852_531.0,
        "reward_clip": 2.0,
    },
}

DEFAULT_PROFITS = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]


def compute_reward(profit: float, reward_type: str, scale: float, clip: float) -> float:
    """Compute reward with the same formula family used in configuration."""
    scale = max(abs(scale), 1.0)
    x = profit / scale

    if reward_type == "tanh":
        raw = math.tanh(x)
    elif reward_type == "linear":
        raw = x
    elif reward_type == "log":
        raw = math.copysign(math.log1p(abs(x)), x)
    else:
        raise ValueError(f"Unsupported reward_type: {reward_type}")

    return max(-clip, min(clip, raw))


def parse_profits(value: str) -> List[float]:
    """Parse comma-separated profit values."""
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("No profit values provided")
    return [float(v) for v in items]


def fmt_num(n: float) -> str:
    return f"{n:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare calibrated reward profiles")
    parser.add_argument(
        "--profits",
        type=str,
        default=",".join(str(p) for p in DEFAULT_PROFITS),
        help="Comma-separated profit values in KRW",
    )
    args = parser.parse_args()

    profits = parse_profits(args.profits)

    print("Rollout Stats")
    print("-" * 80)
    for key, value in ROLLOUT_STATS.items():
        print(f"{key:>16}: {value:,}" if isinstance(value, int) else f"{key:>16}: {value}")

    print("\nReward Comparison")
    print("-" * 80)
    header = ["profile", "type", "scale", "clip"] + [f"P={fmt_num(p)}" for p in profits]
    print(" | ".join(header))
    print("-" * 80)

    for profile_name, spec in PROFILES.items():
        reward_type = str(spec["reward_type"])
        scale = float(spec["profit_scale"])
        clip = float(spec["reward_clip"])

        row = [
            profile_name,
            reward_type,
            f"{scale:,.1f}",
            f"{clip:.1f}",
        ]

        for p in profits:
            r = compute_reward(p, reward_type=reward_type, scale=scale, clip=clip)
            row.append(f"{r:.4f}")

        print(" | ".join(row))


if __name__ == "__main__":
    main()
