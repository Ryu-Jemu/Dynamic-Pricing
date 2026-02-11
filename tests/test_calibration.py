"""
Tests for calibration module (§20 — test_calibration.py).

Tests demand, market, and reward_scale calibration routines
against scenario targets within tolerance.

References:
  [LOGNORMAL_TNET][CHURN_SLR][SB3_TIPS]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.utils import load_config
from src.models.calibrate import (
    calibrate_demand,
    calibrate_market,
    _baseline_churn_rate,
    calibrate_reward_scale,
    calibrate_reward_scale_from_samples,
)
from src.models.demand import DemandModel, DemandConfig

logger = logging.getLogger("oran.test.calibration")


@pytest.fixture
def cfg():
    """Load default config for testing."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    return load_config(str(config_path))


# =====================================================================
# Test Demand Calibration  [LOGNORMAL_TNET]
# =====================================================================

class TestCalibrateDemand:
    """Test demand parameter fitting via quantile matching."""

    def test_demand_calibration_returns_updates(self, cfg):
        updates = calibrate_demand(cfg)

        assert "demand" in updates
        for sname in ["eMBB", "URLLC"]:
            assert sname in updates["demand"]
            assert "mu" in updates["demand"][sname]
            assert "sigma" in updates["demand"][sname]
            assert updates["demand"][sname]["sigma"] > 0

    def test_demand_params_match_targets(self, cfg):
        updates = calibrate_demand(cfg)
        tolerance = cfg.get("calibration", {}).get("demand_tolerance", 0.10)

        for sname in ["eMBB", "URLLC"]:
            mu = updates["demand"][sname]["mu"]
            sigma = updates["demand"][sname]["sigma"]
            target_p50 = cfg["demand"][sname]["target_p50_gb"]
            target_p90 = cfg["demand"][sname]["target_p90_gb"]

            actual_p50 = DemandModel.lognormal_median(mu, sigma)
            actual_p90 = DemandModel.lognormal_quantile(mu, sigma, 0.90)

            assert abs(actual_p50 - target_p50) / target_p50 <= tolerance, \
                f"{sname} p50: {actual_p50:.2f} vs target {target_p50:.2f}"
            assert abs(actual_p90 - target_p90) / target_p90 <= tolerance, \
                f"{sname} p90: {actual_p90:.2f} vs target {target_p90:.2f}"

    def test_demand_calibration_finite(self, cfg):
        updates = calibrate_demand(cfg)
        for sname in ["eMBB", "URLLC"]:
            mu = updates["demand"][sname]["mu"]
            sigma = updates["demand"][sname]["sigma"]
            assert np.isfinite(mu), f"{sname} mu is not finite"
            assert np.isfinite(sigma), f"{sname} sigma is not finite"


# =====================================================================
# Test Market Calibration  [CHURN_SLR][DISCONF_PDF]
# =====================================================================

class TestCalibrateMarket:
    """Test market parameter calibration."""

    def test_baseline_churn_rate_computes(self, cfg):
        seg_cfg = cfg.get("segments", {})
        seg_names = seg_cfg.get("names", ["light", "mid", "heavy", "qos_sensitive"])
        seg_probs = seg_cfg.get("proportions", [0.25, 0.40, 0.25, 0.10])

        rate = _baseline_churn_rate(
            beta_price=0.5, beta_qos=0.3, beta_sw=0.2,
            U_outside=0.0,
            F_baseline=70000.0,
            T_act_baseline=20.0,
            seg_cfg=seg_cfg,
            seg_names=seg_names,
            seg_probs=seg_probs,
        )

        assert 0.0 <= rate <= 1.0, f"Churn rate {rate} out of range"
        assert np.isfinite(rate), "Churn rate not finite"

    def test_market_calibration_returns_updates(self, cfg):
        updates = calibrate_market(cfg)

        assert "market" in updates
        assert "U_outside" in updates["market"]
        assert "U_outside_per_slice" in updates["market"]
        assert isinstance(updates["market"]["U_outside_per_slice"], dict)

    def test_market_calibration_matches_targets(self, cfg):
        updates = calibrate_market(cfg)
        tolerance = cfg.get("calibration", {}).get("market_tolerance", 0.15)
        seg_cfg = cfg.get("segments", {})
        seg_names = seg_cfg.get("names", ["light", "mid", "heavy", "qos_sensitive"])
        seg_probs = seg_cfg.get("proportions", [0.25, 0.40, 0.25, 0.10])

        from src.models.utils import compute_price_bounds
        price_bounds = compute_price_bounds(cfg)

        for sname in ["eMBB", "URLLC"]:
            target = cfg["market"].get(f"target_churn_rate_{sname}", 0.03)
            U_opt = updates["market"]["U_outside_per_slice"][sname]
            pb = price_bounds[sname]
            F_baseline = (pb["F_min"] + pb["F_max"]) / 2.0
            slo_key = f"SLO_T_user_{sname}_mbps"
            T_baseline = cfg.get("sla", {}).get(slo_key, 10.0) * 2.0

            actual = _baseline_churn_rate(
                beta_price=updates["market"]["beta_price"],
                beta_qos=updates["market"]["beta_qos"],
                beta_sw=updates["market"]["beta_sw"],
                U_outside=U_opt,
                F_baseline=F_baseline,
                T_act_baseline=T_baseline,
                seg_cfg=seg_cfg,
                seg_names=seg_names,
                seg_probs=seg_probs,
                price_norm=cfg.get("market", {}).get("price_norm", 70000.0),
            )

            assert abs(actual - target) / max(target, 1e-6) <= tolerance, \
                f"{sname}: actual churn {actual:.4f} vs target {target:.4f}"


# =====================================================================
# Test Reward Scale Calibration  [SB3_TIPS]
# =====================================================================

class TestCalibrateRewardScale:
    """Test reward_scale calibration."""

    def test_reward_scale_from_samples(self):
        rng = np.random.default_rng(42)
        profits = rng.normal(500000, 200000, size=500)

        result = calibrate_reward_scale_from_samples(
            profits, reward_type="log", reward_clip=2.0,
        )

        assert "profit_scale" in result
        assert result["profit_scale"] > 0
        assert np.isfinite(result["profit_scale"])

    def test_reward_scale_from_samples_empty(self):
        result = calibrate_reward_scale_from_samples(
            np.array([]), reward_type="log",
        )
        assert result["profit_scale"] >= 1.0

    def test_reward_scale_full_pipeline(self, cfg):
        result = calibrate_reward_scale(cfg, n_episodes=2, seed=42)

        assert "economics" in result
        assert "profit_scale" in result["economics"]
        assert result["economics"]["profit_scale"] > 0
        assert np.isfinite(result["economics"]["profit_scale"])


# =====================================================================
# Integration: Full calibration pipeline
# =====================================================================

class TestFullCalibration:
    """Test that all three calibrations compose correctly."""

    def test_full_pipeline_produces_valid_config(self, cfg):
        from src.models.utils import merge_configs

        demand_updates = calibrate_demand(cfg)
        cfg_1 = merge_configs(cfg, demand_updates)

        market_updates = calibrate_market(cfg_1)
        cfg_2 = merge_configs(cfg_1, market_updates)

        assert cfg_2["demand"]["eMBB"]["mu"] == demand_updates["demand"]["eMBB"]["mu"]
        assert cfg_2["market"]["U_outside"] == market_updates["market"]["U_outside"]

        reward_updates = calibrate_reward_scale(cfg_2, n_episodes=1, seed=42)
        cfg_3 = merge_configs(cfg_2, reward_updates)

        assert cfg_3["economics"]["profit_scale"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
