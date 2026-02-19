"""
Shared utilities — config I/O, math, device selection, calibration.

References:
  [SB3_TIPS]     SB3 RL Tips and Tricks — reward normalisation
  [MPS_PYTORCH]  https://docs.pytorch.org/docs/stable/notes/mps.html
  [LOGNORMAL]    scipy.stats.lognorm parameterisation
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from scipy import stats

logger = logging.getLogger("oran3pt.utils")

# ── Config I/O ────────────────────────────────────────────────────────

def load_config(path: str | Path,
                override_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config, optionally deep-merging an override file on top.

    [CR-3] Override support enables production.yaml to selectively
    override default.yaml values without duplicating the full config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if override_path is not None:
        override_path = Path(override_path)
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict.

    [CR-3] Leaf values in override replace base values.
    Nested dicts are merged recursively.
    """
    result = base.copy()
    for k, v in override.items():
        if (k in result and isinstance(result[k], dict)
                and isinstance(v, dict)):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

# ── Math ──────────────────────────────────────────────────────────────

def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def safe_clip(x, lo=-1e8, hi=1e8):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)
    return np.clip(x, lo, hi)

# ── Lognormal calibration  [scipy.stats.lognorm] ─────────────────────

def fit_lognormal_quantiles(p50: float, p90: float) -> tuple[float, float]:
    """Return (mu, sigma) of the underlying Normal so that
    exp(Normal(mu,sigma)) matches the given median and 90-th percentile."""
    if p50 <= 0 or p90 <= p50:
        raise ValueError(f"Need 0 < p50 < p90; got p50={p50}, p90={p90}")
    z90 = float(stats.norm.ppf(0.90))
    mu = np.log(p50)
    sigma = (np.log(p90) - mu) / z90
    if sigma <= 0:
        raise ValueError(f"Fitted sigma={sigma:.4f} <= 0")
    return float(mu), float(sigma)

# ── Device selection  [MPS_PYTORCH] ───────────────────────────────────

def select_device() -> str:
    """Best available PyTorch device: mps → cuda → cpu."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
