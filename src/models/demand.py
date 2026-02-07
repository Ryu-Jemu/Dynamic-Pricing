"""
Demand model: per-user monthly data usage (GB).

Traffic volumes are modeled with log-normal distributions.
Claim: internet traffic volumes are often well modeled by
log-normal / heavy-tailed distributions.  [LOGNORMAL_TNET]

Per-user monthly demand:
  D_user_s ~ LogNormal(mu_s, sigma_s)  truncated to [D_min_s, D_max_s]

Parameters (mu, sigma) are NOT hardcoded:
  - They MUST be fitted by calibrate.py via quantile matching or MLE
    to scenario targets (mean / p50 / p90).  [LOGNORMAL_TNET]
  - Initial guesses live in config/default.yaml; calibrated values
    are written to config/calibrated.yaml.

Segments (max 4):
  light, mid, heavy, qos_sensitive
  Each segment applies a demand_scale multiplier.  [LOGNORMAL_TNET]

Truncation ranges (scenario defaults, configurable):
  eMBB:  D_min=0.2GB,  D_max=300GB
  URLLC: D_min=0.05GB, D_max=50GB
Rationale: plan-bucket realism.  [TWORLD_18][TWORLD_127]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats


@dataclass
class SliceDemandConfig:
    """Demand parameters for a single slice."""

    mu: float               # log-normal mu (calibrated)
    sigma: float            # log-normal sigma (calibrated)
    D_min_gb: float         # truncation lower bound
    D_max_gb: float         # truncation upper bound
    target_mean_gb: float = 0.0   # calibration target
    target_p50_gb: float = 0.0
    target_p90_gb: float = 0.0


@dataclass
class DemandConfig:
    """Full demand configuration for all slices."""

    slices: Dict[str, SliceDemandConfig] = field(default_factory=dict)
    segment_scales: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DemandConfig":
        """Build DemandConfig from full YAML config dict."""
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
            "light": 0.5,
            "mid": 1.0,
            "heavy": 2.0,
            "qos_sensitive": 1.0,
        })

        return cls(slices=slices, segment_scales=segment_scales)


class DemandModel:
    """Log-normal demand generator with truncation and segment scaling.

    References:
      [LOGNORMAL_TNET] IEEE/ACM Trans. Networking 2021
      [TWORLD_18][TWORLD_127] Public plan pages (truncation realism)
    """

    def __init__(self, config: DemandConfig) -> None:
        self.cfg = config

    # -----------------------------------------------------------------
    # Core sampling
    # -----------------------------------------------------------------

    def sample_demand(
        self,
        slice_name: str,
        n_users: int,
        segments: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample monthly demand (GB) for n_users in a slice.

        Parameters
        ----------
        slice_name : str
            "eMBB" or "URLLC"
        n_users : int
            Number of active users to sample for.
        segments : ndarray of str, shape (n_users,), optional
            Segment labels per user (for demand_scale).
            If None, all users use scale=1.0.
        rng : numpy Generator, optional
            If None, uses numpy.random.default_rng() (no global seed).

        Returns
        -------
        ndarray of float64, shape (n_users,) — monthly demand in GB.
        """
        if n_users <= 0:
            return np.array([], dtype=np.float64)

        if rng is None:
            rng = np.random.default_rng()

        sc = self.cfg.slices[slice_name]

        # Sample from underlying log-normal
        raw = rng.lognormal(mean=sc.mu, sigma=sc.sigma, size=n_users)

        # Apply segment scaling
        if segments is not None:
            scales = np.array([
                self.cfg.segment_scales.get(str(seg), 1.0)
                for seg in segments
            ], dtype=np.float64)
            raw = raw * scales

        # Truncate to [D_min, D_max]  [TWORLD_18][TWORLD_127]
        demand = np.clip(raw, sc.D_min_gb, sc.D_max_gb)

        return demand

    # -----------------------------------------------------------------
    # Analytical helpers (for calibration)
    # -----------------------------------------------------------------

    @staticmethod
    def lognormal_mean(mu: float, sigma: float) -> float:
        """Analytical mean of LogNormal(mu, sigma).

        E[X] = exp(mu + sigma^2 / 2)
        """
        return np.exp(mu + sigma ** 2 / 2.0)

    @staticmethod
    def lognormal_median(mu: float, sigma: float) -> float:
        """Analytical median (p50) of LogNormal(mu, sigma).

        Median = exp(mu)
        """
        return np.exp(mu)

    @staticmethod
    def lognormal_quantile(mu: float, sigma: float, q: float) -> float:
        """Quantile of LogNormal(mu, sigma).

        Q(q) = exp(mu + sigma * Phi^{-1}(q))
        """
        return float(stats.lognorm.ppf(q, s=sigma, scale=np.exp(mu)))

    @staticmethod
    def fit_lognormal_quantiles(
        target_p50: float,
        target_p90: float,
    ) -> tuple[float, float]:
        """Fit mu, sigma from p50 and p90 targets via quantile matching.

        p50 = exp(mu)         →  mu = ln(p50)
        p90 = exp(mu + sigma * z_0.90)
            →  sigma = (ln(p90) - mu) / z_0.90

        where z_0.90 = Phi^{-1}(0.90) ≈ 1.2816.

        Returns (mu, sigma).
        """
        if target_p50 <= 0 or target_p90 <= 0:
            raise ValueError("Quantile targets must be positive.")
        if target_p90 <= target_p50:
            raise ValueError("p90 must be greater than p50.")

        z90 = stats.norm.ppf(0.90)  # ≈ 1.2816
        mu = np.log(target_p50)
        sigma = (np.log(target_p90) - mu) / z90

        if sigma <= 0:
            raise ValueError(
                f"Fitted sigma={sigma:.4f} <= 0; check targets "
                f"(p50={target_p50}, p90={target_p90})."
            )

        return float(mu), float(sigma)

    # -----------------------------------------------------------------
    # Validation (used by calibration tests)
    # -----------------------------------------------------------------

    def validate_params(self, slice_name: str, tolerance: float = 0.10) -> Dict[str, Any]:
        """Check that analytical stats match calibration targets.

        Returns dict with 'passed' flag and detailed comparison.
        """
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

        # p50 and p90 should match closely (quantile-matched);
        # mean may deviate more due to truncation effects
        passed = (
            results["p50"]["rel_error"] <= tolerance
            and results["p90"]["rel_error"] <= tolerance
        )

        return {"passed": passed, "details": results}
