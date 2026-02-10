"""
Demand model: per-user monthly data usage (GB).

Traffic volumes are modeled with log-normal distributions.  [LOGNORMAL_TNET]

Per-user monthly demand:
  D_user_s ~ LogNormal(mu_s, sigma_s)  truncated to [D_min_s, D_max_s]

Parameters (mu, sigma) are NOT hardcoded:
  - Fitted by calibrate.py via quantile matching.  [LOGNORMAL_TNET]

Segments (max 4): light, mid, heavy, qos_sensitive
Truncation ranges (scenario defaults, configurable):
  eMBB:  D_min=0.2GB,  D_max=300GB
  URLLC: D_min=0.05GB, D_max=50GB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats


@dataclass
class SliceDemandConfig:
    """Demand parameters for a single slice."""
    mu: float
    sigma: float
    D_min_gb: float
    D_max_gb: float
    target_mean_gb: float = 0.0
    target_p50_gb: float = 0.0
    target_p90_gb: float = 0.0


@dataclass
class DemandConfig:
    """Full demand configuration for all slices."""
    slices: Dict[str, SliceDemandConfig] = field(default_factory=dict)
    segment_scales: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DemandConfig":
        demand_cfg = cfg.get("demand", {})
        seg_cfg = cfg.get("segments", {})

        slices = {}
        for sname in cfg.get("slices", {}).get("names", ["eMBB", "URLLC"]):
            sc = demand_cfg.get(sname, {})
            slices[sname] = SliceDemandConfig(
                mu=sc.get("mu", 2.0),
                sigma=sc.get("sigma", 0.8),
                D_min_gb=sc.get("D_min_gb", 0.1),
                D_max_gb=sc.get("D_max_gb", 300.0),
                target_mean_gb=sc.get("target_mean_gb", 10.0),
                target_p50_gb=sc.get("target_p50_gb", 7.0),
                target_p90_gb=sc.get("target_p90_gb", 25.0),
            )

        segment_scales = seg_cfg.get("demand_scale", {
            "light": 0.5, "mid": 1.0, "heavy": 2.0, "qos_sensitive": 1.0,
        })
        return cls(slices=slices, segment_scales=segment_scales)


class DemandModel:
    """Log-normal demand generator with truncation and segment scaling."""

    def __init__(self, config: DemandConfig) -> None:
        self.cfg = config

    def sample_demand(self, slice_name: str, n_users: int,
                      segments: Optional[np.ndarray] = None,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if n_users <= 0:
            return np.array([], dtype=np.float64)
        if rng is None:
            rng = np.random.default_rng()

        sc = self.cfg.slices[slice_name]
        raw = rng.lognormal(mean=sc.mu, sigma=sc.sigma, size=n_users)

        if segments is not None:
            scales = np.array([
                self.cfg.segment_scales.get(str(seg), 1.0)
                for seg in segments
            ], dtype=np.float64)
            raw = raw * scales

        demand = np.clip(raw, sc.D_min_gb, sc.D_max_gb)
        return demand

    @staticmethod
    def lognormal_mean(mu: float, sigma: float) -> float:
        return np.exp(mu + sigma ** 2 / 2.0)

    @staticmethod
    def lognormal_median(mu: float, sigma: float) -> float:
        return np.exp(mu)

    @staticmethod
    def lognormal_quantile(mu: float, sigma: float, q: float) -> float:
        return float(stats.lognorm.ppf(q, s=sigma, scale=np.exp(mu)))

    @staticmethod
    def fit_lognormal_quantiles(target_p50: float, target_p90: float) -> tuple[float, float]:
        """Fit mu, sigma from p50 and p90 targets via quantile matching."""
        if target_p50 <= 0 or target_p90 <= 0:
            raise ValueError("Quantile targets must be positive.")
        if target_p90 <= target_p50:
            raise ValueError("p90 must be greater than p50.")

        z90 = stats.norm.ppf(0.90)
        mu = np.log(target_p50)
        sigma = (np.log(target_p90) - mu) / z90

        if sigma <= 0:
            raise ValueError(
                f"Fitted sigma={sigma:.4f} <= 0; check targets "
                f"(p50={target_p50}, p90={target_p90})."
            )
        return float(mu), float(sigma)

    def validate_params(self, slice_name: str, tolerance: float = 0.10) -> Dict[str, Any]:
        sc = self.cfg.slices[slice_name]
        analytical_mean = self.lognormal_mean(sc.mu, sc.sigma)
        analytical_p50 = self.lognormal_median(sc.mu, sc.sigma)
        analytical_p90 = self.lognormal_quantile(sc.mu, sc.sigma, 0.90)

        def rel_err(actual: float, target: float) -> float:
            if abs(target) < 1e-12:
                return 0.0
            return abs(actual - target) / abs(target)

        results = {
            "mean": {"analytical": analytical_mean, "target": sc.target_mean_gb,
                     "rel_error": rel_err(analytical_mean, sc.target_mean_gb)},
            "p50": {"analytical": analytical_p50, "target": sc.target_p50_gb,
                    "rel_error": rel_err(analytical_p50, sc.target_p50_gb)},
            "p90": {"analytical": analytical_p90, "target": sc.target_p90_gb,
                    "rel_error": rel_err(analytical_p90, sc.target_p90_gb)},
        }

        passed = (
            results["p50"]["rel_error"] <= tolerance
            and results["p90"]["rel_error"] <= tolerance
        )
        return {"passed": passed, "details": results}
