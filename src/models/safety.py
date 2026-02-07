"""
Numerical safety and validation module.

Ensures SB3 compatibility by catching NaN/Inf and invalid states.
All hard penalties are logged for debugging.

References:
  [SB3_TIPS] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("oran.safety")


# ---------------------------------------------------------------------------
# Finite-check helpers
# ---------------------------------------------------------------------------

def assert_finite(arr: np.ndarray, name: str = "array") -> None:
    """Raise ValueError if any element is NaN or Inf. [SB3_TIPS]"""
    if not np.all(np.isfinite(arr)):
        bad_mask = ~np.isfinite(arr)
        bad_count = int(np.sum(bad_mask))
        raise ValueError(
            f"Non-finite values in '{name}': {bad_count} element(s) "
            f"(sample: {arr[bad_mask][:5]})"
        )


def sanitize_obs(obs: np.ndarray,
                 clip_min: float = -10.0,
                 clip_max: float = 10.0) -> np.ndarray:
    """Replace NaN/Inf and clip observation to finite bounds. [SB3_TIPS]"""
    obs = np.nan_to_num(obs, nan=0.0, posinf=clip_max, neginf=clip_min)
    obs = np.clip(obs, clip_min, clip_max)
    return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

def validate_action(action: np.ndarray) -> Tuple[bool, List[str]]:
    """Validate raw SB3 action array.

    Returns (is_valid, list_of_violation_strings).
    """
    violations: List[str] = []
    if not np.all(np.isfinite(action)):
        violations.append("action_nan_inf")
    if action.shape != (3,):
        violations.append(f"action_shape_{action.shape}")
    return (len(violations) == 0), violations


# ---------------------------------------------------------------------------
# State validation (per-step)
# ---------------------------------------------------------------------------

def validate_state(
    N_active: Dict[str, int],
    rho_util: Dict[str, float],
    fees: Dict[str, float],
    rho_urllc: float,
) -> Tuple[float, List[str]]:
    """Check state variables and return (penalty, violation_list).

    Penalty > 0 indicates a hard violation that should be subtracted
    from reward.  [SB3_TIPS]
    """
    penalty = 0.0
    violations: List[str] = []

    # 0-PRB check
    if rho_urllc < 0.01 or rho_urllc > 0.99:
        violations.append("rho_urllc_near_boundary")
        penalty += 1.0

    for s in ["eMBB", "URLLC"]:
        # Negative fees
        if fees.get(s, 0) < 0:
            violations.append(f"negative_fee_{s}")
            penalty += 1.0

        # Negative active users
        if N_active.get(s, 0) < 0:
            violations.append(f"negative_N_active_{s}")
            penalty += 1.0

        # Utilization sanity
        rho = rho_util.get(s, 0.0)
        if not np.isfinite(rho):
            violations.append(f"rho_util_nan_inf_{s}")
            penalty += 1.0

    if violations:
        logger.warning("State violations: %s (penalty=%.2f)", violations, penalty)

    return penalty, violations


# ---------------------------------------------------------------------------
# Reward safety wrapper
# ---------------------------------------------------------------------------

def safe_reward(raw_reward: float,
                penalty: float = 0.0,
                lambda_penalty: float = 10.0,
                clip_abs: float = 2.0) -> float:
    """Apply penalty and clip reward to finite range. [SB3_TIPS]

    r_final = clip( raw_reward - lambda_penalty * penalty, -clip_abs, clip_abs )
    """
    if not np.isfinite(raw_reward):
        logger.warning("Non-finite raw reward: %s → replaced with -clip", raw_reward)
        raw_reward = -clip_abs

    r = raw_reward - lambda_penalty * penalty
    r = float(np.clip(r, -clip_abs, clip_abs))
    return r
