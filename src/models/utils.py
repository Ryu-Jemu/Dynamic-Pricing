"""
Shared utility functions for O-RAN slicing simulation.

Provides:
- YAML config loading/merging
- Logging helpers
- Common math (sigmoid, safe_clip, etc.)
- Device selection (MPS/CPU) [MPS_PYTORCH][MPS_APPLE]

References:
  [SB3_TIPS] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
  [MPS_PYTORCH] https://docs.pytorch.org/docs/stable/notes/mps.html
  [MPS_APPLE] https://developer.apple.com/metal/pytorch/
"""

from __future__ import annotations

import logging
import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Save dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into *base* (override wins on conflicts)."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


# ---------------------------------------------------------------------------
# Price bounds derived from plan catalog  (Section 6.2)
# ---------------------------------------------------------------------------

def compute_price_bounds(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Return per-slice agent-controlled fee bounds.

    F_s in [0.8 * F_min_catalog, 1.2 * F_max_catalog]
    """
    explore = cfg["action"]["price_explore_factor"]  # 0.2
    bounds: Dict[str, Dict[str, float]] = {}
    for sname in cfg["slices"]["names"]:
        plans = cfg["plans"][sname]
        fees = [p["F_krw_month"] for p in plans]
        f_min_cat = min(fees)
        f_max_cat = max(fees)
        bounds[sname] = {
            "F_min": (1.0 - explore) * f_min_cat,
            "F_max": (1.0 + explore) * f_max_cat,
        }
    return bounds


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def safe_clip(x: np.ndarray | float,
              lo: float = -1e8,
              hi: float = 1e8) -> np.ndarray:
    """Clip and replace NaN/Inf with 0.0."""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)
    return np.clip(x, lo, hi)


def safe_divide(numerator: np.ndarray | float,
                denominator: np.ndarray | float,
                default: float = 0.0) -> np.ndarray:
    """Element-wise division; returns *default* where denominator is 0."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), default)
    return out


# ---------------------------------------------------------------------------
# Device selection  [MPS_PYTORCH][MPS_APPLE]
# ---------------------------------------------------------------------------

def select_device() -> str:
    """Return best available PyTorch device string."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str = "oran",
                 level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Artifacts directory helpers
# ---------------------------------------------------------------------------

def ensure_artifacts_dir(cfg: Dict[str, Any],
                         run_id: Optional[str] = None) -> Path:
    """Create and return artifacts/<run_id>/ directory."""
    import datetime
    base = Path(cfg.get("artifacts", {}).get("base_dir", "artifacts"))
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
