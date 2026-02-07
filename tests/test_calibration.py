"""
test_calibration.py — Demand + market + reward_scale meet targets
within tolerance.

Tests are deterministic by construction (local fixed RNG or analytical).
  [SB3_TIPS]

References:
  [LOGNORMAL_TNET] https://dl.acm.org/doi/10.1109/TNET.2021.3059542
  [CHURN_SLR]      https://link.springer.com/article/10.1007/s11301-023-00335-7
  [SB3_TIPS]       RL Tips and Tricks
"""

import unittest
import numpy as np

from src.models.utils import load_config, compute_price_bounds, merge_configs
from src.models.demand import DemandConfig, DemandModel
from src.models.market import MarketModel
from src.models.pools import User
from src.models.calibrate import (
    calibrate_demand,
    calibrate_market,
    _baseline_churn_rate,
)


class TestCalibrateDemand(unittest.TestCase):
    """Test demand calibration (Section 19.1).  [LOGNORMAL_TNET]"""

    def setUp(self):
        self.cfg = load_config("config/default.yaml")
        self.tol = self.cfg.get("calibration", {}).get(
            "demand_tolerance", 0.10
        )

    def test_calibrate_demand_quantile_match(self):
        """Calibrated mu, sigma must yield p50/p90 within tolerance."""
        updates = calibrate_demand(self.cfg)
        cfg_cal = merge_configs(self.cfg, updates)

        for sname in ["eMBB", "URLLC"]:
            sc = cfg_cal["demand"][sname]
            mu = sc["mu"]
            sigma = sc["sigma"]
            target_p50 = self.cfg["demand"][sname]["target_p50_gb"]
            target_p90 = self.cfg["demand"][sname]["target_p90_gb"]

            fitted_p50 = DemandModel.lognormal_median(mu, sigma)
            fitted_p90 = DemandModel.lognormal_quantile(mu, sigma, 0.90)

            err_p50 = abs(fitted_p50 - target_p50) / target_p50
            err_p90 = abs(fitted_p90 - target_p90) / target_p90

            self.assertLessEqual(err_p50, self.tol,
                                 f"{sname} p50 error {err_p50:.4f}")
            self.assertLessEqual(err_p90, self.tol,
                                 f"{sname} p90 error {err_p90:.4f}")

    def test_calibrated_sigma_positive(self):
        """Calibrated sigma must be positive."""
        updates = calibrate_demand(self.cfg)
        for sname in ["eMBB", "URLLC"]:
            sigma = updates["demand"][sname]["sigma"]
            self.assertGreater(sigma, 0.0)

    def test_sample_from_calibrated_params(self):
        """Empirical statistics from calibrated params should be close
        to targets (large sample).  [LOGNORMAL_TNET]"""
        updates = calibrate_demand(self.cfg)
        cfg_cal = merge_configs(self.cfg, updates)
        rng = np.random.default_rng(42)

        dc = DemandConfig.from_config(cfg_cal)
        dm = DemandModel(dc)

        for sname in ["eMBB", "URLLC"]:
            D = dm.sample_demand(sname, 50000, rng=rng)
            target_p50 = self.cfg["demand"][sname]["target_p50_gb"]
            target_p90 = self.cfg["demand"][sname]["target_p90_gb"]

            emp_p50 = float(np.median(D))
            emp_p90 = float(np.percentile(D, 90))

            # 15% tolerance for empirical (truncation + sampling noise)
            err_p50 = abs(emp_p50 - target_p50) / target_p50
            err_p90 = abs(emp_p90 - target_p90) / target_p90
            self.assertLessEqual(err_p50, 0.15,
                                 f"{sname} emp p50={emp_p50:.2f}")
            self.assertLessEqual(err_p90, 0.15,
                                 f"{sname} emp p90={emp_p90:.2f}")


class TestCalibrateMarket(unittest.TestCase):
    """Test market calibration (Section 19.2).  [CHURN_SLR][DISCONF_PDF]"""

    def setUp(self):
        self.cfg = load_config("config/default.yaml")
        self.tol = self.cfg.get("calibration", {}).get(
            "market_tolerance", 0.15
        )

    def test_calibrate_market_churn_rate(self):
        """Calibrated U_outside must yield baseline churn within tolerance."""
        updates = calibrate_market(self.cfg)
        cfg_cal = merge_configs(self.cfg, updates)

        seg_cfg = self.cfg.get("segments", {})
        seg_names = seg_cfg.get("names",
                                ["light", "mid", "heavy", "qos_sensitive"])
        seg_probs = seg_cfg.get("proportions", [0.25, 0.40, 0.25, 0.10])
        price_bounds = compute_price_bounds(self.cfg)

        cal_market = cfg_cal["market"]
        U_per_slice = cal_market.get("U_outside_per_slice", {})

        for sname in ["eMBB", "URLLC"]:
            target_churn = self.cfg["market"].get(
                f"target_churn_rate_{sname}", 0.03
            )
            pb = price_bounds[sname]
            F_baseline = (pb["F_min"] + pb["F_max"]) / 2.0
            slo_key = f"SLO_T_user_{sname}_mbps"
            T_baseline = self.cfg["sla"].get(slo_key, 10.0) * 2.0

            # Use per-slice U_outside for accurate verification
            U_out = U_per_slice.get(sname, cal_market["U_outside"])

            actual = _baseline_churn_rate(
                beta_price=cal_market["beta_price"],
                beta_qos=cal_market["beta_qos"],
                beta_sw=cal_market["beta_sw"],
                U_outside=U_out,
                F_baseline=F_baseline,
                T_act_baseline=T_baseline,
                seg_cfg=seg_cfg,
                seg_names=seg_names,
                seg_probs=seg_probs,
            )
            err = abs(actual - target_churn) / max(target_churn, 1e-9)

            self.assertLessEqual(
                err, self.tol,
                f"{sname}: actual={actual:.4f}, target={target_churn}, "
                f"err={err:.4f}"
            )

    def test_calibrated_betas_positive(self):
        """Beta_price and beta_qos must remain positive."""
        updates = calibrate_market(self.cfg)
        m = updates["market"]
        self.assertGreater(m["beta_price"], 0.0)
        self.assertGreater(m["beta_qos"], 0.0)

    def test_churn_direction_after_calibration(self):
        """After calibration, churn direction must be correct:
        higher price → more churn, higher QoS → less churn."""
        updates = calibrate_market(self.cfg)
        cfg_cal = merge_configs(self.cfg, updates)
        market = MarketModel(cfg_cal)

        user = User(
            user_id=0, slice="eMBB", segment="mid",
            w_price=1.0, w_qos=1.0, sw_cost=0.5, b_u=0.0,
        )

        p_base = market.compute_churn_prob(user, 55000, 20.0, 0.0)
        p_high_price = market.compute_churn_prob(user, 100000, 20.0, 0.0)
        p_high_qos = market.compute_churn_prob(user, 55000, 80.0, 0.0)
        p_high_disc = market.compute_churn_prob(user, 55000, 20.0, 30.0)

        self.assertGreater(p_high_price, p_base,
                           "Higher price should increase churn")
        self.assertLess(p_high_qos, p_base,
                        "Higher QoS should decrease churn")
        self.assertGreater(p_high_disc, p_base,
                           "Higher disconfirmation should increase churn")

    def test_calibrated_churn_is_reasonable(self):
        """Calibrated churn at baseline should be in [0.001, 0.20]."""
        updates = calibrate_market(self.cfg)
        cfg_cal = merge_configs(self.cfg, updates)
        market = MarketModel(cfg_cal)

        user = User(
            user_id=0, slice="eMBB", segment="mid",
            w_price=1.0, w_qos=1.0, sw_cost=0.5, b_u=0.0,
        )
        p_churn = market.compute_churn_prob(user, 60000, 20.0, 0.0)
        self.assertGreater(p_churn, 0.001,
                           f"Churn too low: {p_churn}")
        self.assertLess(p_churn, 0.20,
                        f"Churn too high: {p_churn}")


class TestCalibrateRewardScale(unittest.TestCase):
    """Test reward scale calibration (Section 19.3).  [SB3_TIPS]"""

    def test_profit_scale_positive(self):
        """profit_scale from calibration must be positive and finite."""
        cfg = load_config("config/default.yaml")
        demand_updates = calibrate_demand(cfg)
        cfg = merge_configs(cfg, demand_updates)
        market_updates = calibrate_market(cfg)
        cfg = merge_configs(cfg, market_updates)

        cfg["calibration"]["reward_scale_episodes"] = 3
        from src.models.calibrate import calibrate_reward_scale
        reward_updates = calibrate_reward_scale(cfg)

        ps = reward_updates["economics"]["profit_scale"]
        self.assertGreater(ps, 0.0)
        self.assertTrue(np.isfinite(ps))

    def test_scaled_reward_in_range(self):
        """tanh(Pi/scale) should be in [-1, 1] for typical profits."""
        profit_scale = 5e6
        for profit in [-1e7, -1e6, 0, 1e6, 5e6, 1e7]:
            r = float(np.tanh(profit / profit_scale))
            self.assertGreaterEqual(r, -1.0)
            self.assertLessEqual(r, 1.0)
            self.assertTrue(np.isfinite(r))


class TestFitLognormalQuantiles(unittest.TestCase):
    """Test the quantile fitting function directly."""

    def test_exact_recovery(self):
        """fit_lognormal_quantiles should recover p50 and p90 exactly."""
        for p50, p90 in [(10.0, 35.0), (2.0, 7.0), (50.0, 150.0), (0.5, 2.0)]:
            mu, sigma = DemandModel.fit_lognormal_quantiles(p50, p90)
            recovered_p50 = DemandModel.lognormal_median(mu, sigma)
            recovered_p90 = DemandModel.lognormal_quantile(mu, sigma, 0.90)
            self.assertAlmostEqual(recovered_p50, p50, places=4)
            self.assertAlmostEqual(recovered_p90, p90, places=4)

    def test_invalid_inputs(self):
        """Should raise on invalid quantile targets."""
        with self.assertRaises(ValueError):
            DemandModel.fit_lognormal_quantiles(-1.0, 5.0)
        with self.assertRaises(ValueError):
            DemandModel.fit_lognormal_quantiles(10.0, 5.0)


if __name__ == "__main__":
    unittest.main()
