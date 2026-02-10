"""
Numerical safety and validation module.

Ensures SB3 compatibility by catching NaN/Inf and invalid states.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("oran.safety")


def assert_finite(arr: np.ndarray, name: str = "array") -> None:
    if not np.all(np.isfinite(arr)):
        bad_mask = ~np.isfinite(arr)
        bad_count = int(np.sum(bad_mask))
        raise ValueError(
            f"Non-finite values in '{name}': {bad_count} element(s) "
            f"(sample: {arr[bad_mask][:5]})"
        )


def sanitize_obs(obs: np.ndarray, clip_min: float = -10.0,
                 clip_max: float = 10.0) -> np.ndarray:
    obs = np.nan_to_num(obs, nan=0.0, posinf=clip_max, neginf=clip_min)
    obs = np.clip(obs, clip_min, clip_max)
    return obs.astype(np.float32)


def validate_action(action: np.ndarray) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    if not np.all(np.isfinite(action)):
        violations.append("action_nan_inf")
    if action.shape != (3,):
        violations.append(f"action_shape_{action.shape}")
    return (len(violations) == 0), violations


def validate_state(N_active: Dict[str, int], rho_util: Dict[str, float],
                   fees: Dict[str, float], rho_urllc: float) -> Tuple[float, List[str]]:
    penalty = 0.0
    violations: List[str] = []

    if rho_urllc < 0.01 or rho_urllc > 0.99:
        violations.append("rho_urllc_near_boundary")
        penalty += 1.0

    for s in ["eMBB", "URLLC"]:
        if fees.get(s, 0) < 0:
            violations.append(f"negative_fee_{s}")
            penalty += 1.0
        if N_active.get(s, 0) < 0:
            violations.append(f"negative_N_active_{s}")
            penalty += 1.0
        rho = rho_util.get(s, 0.0)
        if not np.isfinite(rho):
            violations.append(f"rho_util_nan_inf_{s}")
            penalty += 1.0

    if violations:
        logger.warning("State violations: %s (penalty=%.2f)", violations, penalty)
    return penalty, violations


def safe_reward(raw_reward: float, penalty: float = 0.0,
                lambda_penalty: float = 10.0, clip_abs: float = 2.0) -> float:
    if not np.isfinite(raw_reward):
        logger.warning("Non-finite raw reward: %s -> replaced with -clip", raw_reward)
        raw_reward = -clip_abs
    r = raw_reward - lambda_penalty * penalty
    r = float(np.clip(r, -clip_abs, clip_abs))
    return r
