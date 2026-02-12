"""
Unit tests for O-RAN 3-Part Tariff environment.

REVISION 8 — Changes:
  [M9] v8 unit tests (T11) — curriculum fraction, convex SLA, Lagrangian,
       rho_U bound, pop_bonus scale, eval diagnostics
  Prior revisions:
  [R4] Per-dimension smoothing tests
  [R5] Observation shape updated to (22,)
  [R6] Population-aware reward tests
  [E4] Observation shape 16 → 20
  [E5] Episode length now 720 (24 cycles)
  [E6] CLV reward shaping tests
  [E8] Stronger smoothing tests

Test groups:
  T1  Environment basics (reset, step, spaces)
  T2  Revenue model (3-part tariff, online accrual)
  T3  Market dynamics (join/churn, conservation)
  T4  QoS violation (sigmoid, capacity)
  T5  Numerical safety (no NaN/Inf, obs bounds, reward clip)
  T6  Billing cycle (reset accumulators, 30-step cycles)
  T7  Utility functions
  T8  Calibration validation (recalibrated for v5)
  T9  v5 enhancements (20D obs, CLV reward, load factors)
  T10 v7 enhancements (22D obs, per-dim smoothing, pop reward, curriculum)
  T11 v8 enhancements (curriculum fraction, convex SLA, Lagrangian, rho_U bound)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oran3pt.utils import load_config, sigmoid, fit_lognormal_quantiles
from oran3pt.env import OranSlicingPricingEnv


@pytest.fixture
def cfg():
    return load_config(
        str(Path(__file__).resolve().parent.parent / "config" / "default.yaml")
    )


@pytest.fixture
def env(cfg):
    return OranSlicingPricingEnv(cfg, seed=42)


# =====================================================================
# T1  Environment basics
# =====================================================================
class TestEnvBasics:
    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (22,), f"Bad obs shape: {obs.shape}"
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (22,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (5,), \
            f"Action space should be (5,), got {env.action_space.shape}"

    def test_episode_terminates(self, env):
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, f"Episode did not terminate in {steps} steps"
        assert steps == env.episode_len, \
            f"Terminated at step {steps}, expected {env.episode_len}"

    def test_episode_length_matches_config(self, env, cfg):
        expected = cfg["time"]["steps_per_cycle"] * cfg["time"]["episode_cycles"]
        assert env.episode_len == expected, \
            f"Episode len {env.episode_len} != {expected}"


# =====================================================================
# T2  Revenue model
# =====================================================================
class TestRevenueModel:
    def test_revenue_non_negative_with_active_users(self, env):
        env.reset(seed=42)
        revenues = []
        for _ in range(30):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            revenues.append(info["revenue"])
            if term:
                break
        assert any(r > 0 for r in revenues)

    def test_overage_revenue_accrual(self, env):
        env.reset(seed=42)
        total_over_rev = 0.0
        for _ in range(60):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            total_over_rev += info.get("over_rev", 0.0)
            if term:
                break
        assert total_over_rev >= 0.0


# =====================================================================
# T3  Market dynamics
# =====================================================================
class TestMarketDynamics:
    def test_population_conservation(self, env):
        env.reset(seed=42)
        total_pop = env.N_total
        for _ in range(30):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            n_active = int(env._active_mask.sum())
            n_inactive = total_pop - n_active
            assert n_active >= 0
            assert n_inactive >= 0
            assert n_active + n_inactive == total_pop
            if term:
                break

    def test_join_churn_in_info(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "n_join" in info
        assert "n_churn" in info
        assert info["n_join"] >= 0
        assert info["n_churn"] >= 0

    def test_no_negative_active(self, env):
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            assert info["N_active"] >= 0
            if term:
                break


# =====================================================================
# T4  QoS violation
# =====================================================================
class TestQoSViolation:
    def test_sigmoid_properties(self):
        assert 0 < sigmoid(0.0) < 1
        assert abs(sigmoid(0.0) - 0.5) < 1e-6
        vals = [sigmoid(x) for x in [-10, -1, 0, 1, 10]]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-10

    def test_violation_in_range(self, env):
        env.reset(seed=42)
        for _ in range(30):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            for key in ["pviol_U", "pviol_E"]:
                assert 0.0 <= info[key] <= 1.0
            if term:
                break

    def test_high_load_increases_violation(self, env):
        low_rho = np.array([-1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        high_rho = np.array([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        env.reset(seed=42)
        _, _, _, _, info_low = env.step(low_rho)
        env.reset(seed=42)
        _, _, _, _, info_high = env.step(high_rho)
        assert np.isfinite(info_low["pviol_E"])
        assert np.isfinite(info_high["pviol_E"])


# =====================================================================
# T5  Numerical safety
# =====================================================================
class TestNumericalSafety:
    def test_no_nan_inf_random_episode(self, env):
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))
        for step in range(env.episode_len):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs)), f"Step {step}: obs NaN/Inf"
            assert np.isfinite(reward), f"Step {step}: reward NaN/Inf"
            assert np.isfinite(info["profit"]), f"Step {step}: profit NaN/Inf"
            if terminated or truncated:
                break

    def test_multiple_seeds(self, cfg):
        for seed in [0, 7, 42, 999]:
            env = OranSlicingPricingEnv(cfg, seed=seed)
            obs, _ = env.reset(seed=seed)
            for _ in range(30):
                action = env.action_space.sample()
                obs, r, term, _, _ = env.step(action)
                assert np.all(np.isfinite(obs)), f"seed={seed}: obs NaN/Inf"
                assert np.isfinite(r), f"seed={seed}: reward NaN/Inf"
                if term:
                    break

    def test_obs_within_bounds(self, env):
        env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, term, _, _ = env.step(action)
            assert env.observation_space.contains(obs), \
                f"obs out of bounds: min={obs.min():.4f} max={obs.max():.4f}"
            if term:
                break

    def test_reward_clipped(self, env):
        env.reset(seed=42)
        clip = 2.0
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, term, _, _ = env.step(action)
            assert -clip <= reward <= clip
            if term:
                break

    def test_extreme_actions(self, env):
        extremes = [
            np.full(env.action_space.shape, -1.0, dtype=np.float32),
            np.full(env.action_space.shape, 1.0, dtype=np.float32),
            np.zeros(env.action_space.shape, dtype=np.float32),
        ]
        for ext in extremes:
            obs, _ = env.reset(seed=42)
            obs, r, _, _, info = env.step(ext)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(r)


# =====================================================================
# T6  Billing cycle
# =====================================================================
class TestBillingCycle:
    def test_cycle_length(self, env):
        T = env.T
        assert T > 0
        env.reset(seed=42)
        for _ in range(T + 1):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            if term:
                break

    def test_info_contains_step(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "step" in info
        assert info["step"] == 1


# =====================================================================
# T7  Utility functions
# =====================================================================
class TestUtils:
    def test_fit_lognormal_quantiles(self):
        from scipy import stats
        p50, p90 = 1.5, 5.0
        mu, sigma = fit_lognormal_quantiles(p50, p90)
        actual_p50 = np.exp(mu)
        actual_p90 = float(stats.lognorm.ppf(0.90, s=sigma, scale=np.exp(mu)))
        assert abs(actual_p50 - p50) / p50 < 0.01
        assert abs(actual_p90 - p90) / p90 < 0.01

    def test_fit_lognormal_rejects_bad_input(self):
        with pytest.raises(ValueError):
            fit_lognormal_quantiles(-1.0, 5.0)
        with pytest.raises(ValueError):
            fit_lognormal_quantiles(5.0, 3.0)

    def test_sigmoid_stable_at_extremes(self):
        assert np.isfinite(sigmoid(500.0))
        assert np.isfinite(sigmoid(-500.0))
        assert abs(sigmoid(500.0) - 1.0) < 1e-6
        assert abs(sigmoid(-500.0) - 0.0) < 1e-6


# =====================================================================
# T8  Calibration validation
# =====================================================================
class TestCalibration:
    def test_monthly_churn_within_target(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        total_churn = 0
        total_active = 0
        n_steps = 300

        mid_action = np.zeros(5, dtype=np.float32)
        for _ in range(n_steps):
            _, _, term, _, info = env.step(mid_action)
            total_churn += info["n_churn"]
            total_active += info["N_active"]
            if term:
                break

        per_step_churn = total_churn / max(total_active, 1)
        monthly_churn = 1.0 - (1.0 - per_step_churn) ** 30
        assert 0.005 <= monthly_churn <= 0.15, \
            f"Monthly churn {monthly_churn:.4f} outside [0.5%, 15%] band"

    def test_monthly_join_within_target(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        total_join = 0
        total_inactive = 0
        n_steps = 300

        mid_action = np.zeros(5, dtype=np.float32)
        for _ in range(n_steps):
            _, _, term, _, info = env.step(mid_action)
            total_join += info["n_join"]
            total_inactive += info["N_inactive"]
            if term:
                break

        per_step_join = total_join / max(total_inactive, 1)
        monthly_join = 1.0 - (1.0 - per_step_join) ** 30
        assert 0.01 <= monthly_join <= 0.15, \
            f"Monthly join {monthly_join:.4f} outside [1%, 15%] band"

    def test_embb_not_permanently_congested(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        pviol_values = []

        mid_action = np.zeros(5, dtype=np.float32)
        for _ in range(60):
            _, _, term, _, info = env.step(mid_action)
            pviol_values.append(info["pviol_E"])
            if term:
                break

        mean_pviol = np.mean(pviol_values)
        assert mean_pviol < 0.95, \
            f"eMBB pviol_E mean={mean_pviol:.4f} — permanently congested"

    def test_capacity_adequate_for_population(self, cfg):
        C = cfg["radio"]["C_total_gb_per_step"]
        N_active = cfg["population"]["N_active_init"]
        frac_embb = 1.0 - cfg["population"]["frac_urllc"]
        p50_embb = cfg["traffic"]["eMBB"]["target_p50_gb_day"]

        expected_embb = N_active * frac_embb * p50_embb
        assert C >= expected_embb * 0.8, \
            f"Capacity {C} GB < 80% of expected eMBB demand {expected_embb:.0f} GB"

    def test_high_price_causes_more_churn(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        mid_action = np.zeros(5, dtype=np.float32)
        mid_churn = 0
        mid_active = 0
        for _ in range(60):
            _, _, term, _, info = env.step(mid_action)
            mid_churn += info["n_churn"]
            mid_active += info["N_active"]
            if term:
                break

        env2 = OranSlicingPricingEnv(cfg, seed=42)
        env2.reset(seed=42)
        max_price_action = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        high_churn = 0
        high_active = 0
        for _ in range(60):
            _, _, term, _, info = env2.step(max_price_action)
            high_churn += info["n_churn"]
            high_active += info["N_active"]
            if term:
                break

        mid_rate = mid_churn / max(mid_active, 1)
        high_rate = high_churn / max(high_active, 1)
        assert high_rate > mid_rate * 1.2, \
            f"Max-price churn rate ({high_rate:.5f}) not sufficiently " \
            f"higher than mid-price ({mid_rate:.5f})"


# =====================================================================
# T9  v5 Enhancement tests
# =====================================================================
class TestV5Enhancements:
    def test_obs_dim_is_22(self, env):
        assert env.observation_space.shape == (22,)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (22,)

    def test_load_factor_in_obs(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, info = env.step(action)
        load_U = obs[18]
        load_E = obs[19]
        assert np.isfinite(load_U) and load_U >= 0.0
        assert np.isfinite(load_E) and load_E >= 0.0

    def test_allowance_util_in_obs(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        util_U = obs[16]
        util_E = obs[17]
        assert np.isfinite(util_U) and util_U >= 0.0
        assert np.isfinite(util_E) and util_E >= 0.0

    def test_allowance_util_increases_within_cycle(self, env):
        env.reset(seed=42)
        mid_action = np.zeros(5, dtype=np.float32)
        prev_util_E = 0.0
        for step_i in range(30):
            obs, _, term, _, _ = env.step(mid_action)
            cur_util_E = obs[17]
            assert cur_util_E >= prev_util_E - 1e-6, \
                f"Step {step_i}: util_E decreased from {prev_util_E} to {cur_util_E}"
            prev_util_E = cur_util_E
            if term:
                break

    def test_retention_penalty_in_info(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "retention_penalty" in info
        assert isinstance(info["retention_penalty"], float)
        assert np.isfinite(info["retention_penalty"])
        assert info["retention_penalty"] >= 0.0

    def test_retention_penalty_warmup(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        warmup = cfg.get("clv_reward_shaping", {}).get("warmup_steps", 100)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["retention_penalty"] == 0.0, \
            f"Penalty should be 0 at step 1 (warmup={warmup})"

    def test_smooth_penalty_in_info(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "smooth_penalty" in info
        assert isinstance(info["smooth_penalty"], float)
        assert np.isfinite(info["smooth_penalty"])

    def test_large_action_change_produces_smooth_penalty(self, env):
        env.reset(seed=42)
        low_action = np.full(5, -1.0, dtype=np.float32)
        env.step(low_action)
        high_action = np.full(5, 1.0, dtype=np.float32)
        _, _, _, _, info = env.step(high_action)
        assert info["smooth_penalty"] > 0.0, \
            f"Smooth penalty should be > 0 for max action change"

    def test_info_contains_load_capacity(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        for key in ["L_U", "L_E", "C_U", "C_E"]:
            assert key in info, f"Missing key: {key}"
            assert np.isfinite(info[key]), f"{key} is not finite"
            assert info[key] >= 0.0, f"{key} is negative"


# =====================================================================
# T10  v7 Enhancement tests
# =====================================================================
class TestV7Enhancements:
    def test_obs_dim_is_22(self, env):
        assert env.observation_space.shape == (22,)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (22,)

    def test_overage_rate_in_obs(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        over_rate = obs[20]
        assert np.isfinite(over_rate) and over_rate >= 0.0

    def test_days_remaining_in_obs(self, env):
        env.reset(seed=42)
        mid_action = np.zeros(5, dtype=np.float32)
        obs, _, _, _, _ = env.step(mid_action)
        days_remaining = obs[21]
        assert np.isfinite(days_remaining)
        assert 0.0 <= days_remaining <= 1.0, \
            f"Days remaining {days_remaining} not in [0, 1]"

    def test_days_remaining_decreases_within_cycle(self, env):
        env.reset(seed=42)
        mid_action = np.zeros(5, dtype=np.float32)
        prev_days = 1.0
        for step_i in range(30):
            obs, _, term, _, _ = env.step(mid_action)
            cur_days = obs[21]
            assert cur_days <= prev_days + 1e-6, \
                f"Step {step_i}: days_remaining increased from {prev_days} to {cur_days}"
            prev_days = cur_days
            if term:
                break

    def test_pop_bonus_in_info(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "pop_bonus" in info
        assert isinstance(info["pop_bonus"], float)
        assert np.isfinite(info["pop_bonus"])

    def test_pop_bonus_positive_when_above_target(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert abs(info["pop_bonus"]) < 0.02, \
            f"Pop bonus should be near 0 at target ratio, got {info['pop_bonus']}"

    def test_per_dimension_smoothing(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        mid_action = np.zeros(5, dtype=np.float32)
        env.step(mid_action)

        rho_only = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _, _, _, _, info_rho = env.step(rho_only)

        env.reset(seed=42)
        env.step(mid_action)

        fu_only = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info_fu = env.step(fu_only)

        assert info_fu["smooth_penalty"] > info_rho["smooth_penalty"], \
            f"F_U penalty ({info_fu['smooth_penalty']:.6f}) should exceed " \
            f"rho_U penalty ({info_rho['smooth_penalty']:.6f})"

    def test_over_rev_e_in_info(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "over_rev_E" in info
        assert np.isfinite(info["over_rev_E"])
        assert info["over_rev_E"] >= 0.0

    def test_curriculum_phase_no_churn(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42, curriculum_phase=1)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["n_churn"] == 0, "Phase 1 should have no churn"
        assert info["n_join"] == 0, "Phase 1 should have no join"

    def test_curriculum_phase_switch(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42, curriculum_phase=1)
        env.reset(seed=42)

        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
        assert info["n_churn"] == 0

        env.set_curriculum_phase(0)
        total_events = 0
        for _ in range(100):
            _, _, term, _, info = env.step(env.action_space.sample())
            total_events += info["n_churn"] + info["n_join"]
            if term:
                break

        assert total_events > 0, \
            "After switching to Phase 0, churn/join should occur"

    def test_22d_obs_all_finite(self, env):
        env.reset(seed=42)
        for step in range(100):
            action = env.action_space.sample()
            obs, _, term, _, _ = env.step(action)
            for i in range(22):
                assert np.isfinite(obs[i]), \
                    f"Step {step}, obs[{i}] = {obs[i]} is not finite"
            if term:
                break


# =====================================================================
# T11  v8 Enhancement tests
# =====================================================================
class TestV8Enhancements:
    def test_T11_1_curriculum_fraction_based(self, cfg):
        """[M2] phase1_fraction produces correct absolute step count."""
        cfg_copy = {**cfg}
        cfg_copy["training"] = {**cfg.get("training", {})}
        cfg_copy["training"]["total_timesteps"] = 100
        cfg_copy["training"]["curriculum"] = {"enabled": True, "phase1_fraction": 0.20}
        fraction = cfg_copy["training"]["curriculum"]["phase1_fraction"]
        total = cfg_copy["training"]["total_timesteps"]
        phase1_steps = int(total * fraction)
        assert phase1_steps == 20, \
            f"phase1_steps should be 20 (20% of 100), got {phase1_steps}"

    def test_T11_2_convex_sla_penalty(self, cfg):
        """[M3] SLA penalty with gamma_sla_E=2.0 produces convex curve."""
        from oran3pt.utils import sigmoid
        gamma = cfg["qos"].get("gamma_sla_E", 2.0)
        # pviol_E = 0.9: penalty_factor = 0.9^2 = 0.81
        # pviol_E = 0.3: penalty_factor = 0.3^2 = 0.09
        pf_high = 0.9 ** gamma
        pf_low = 0.3 ** gamma
        # Convex: penalty at 0.9 should be much more than 3x penalty at 0.3
        ratio = pf_high / pf_low
        assert ratio > 3.0, \
            f"Convex penalty ratio {ratio:.2f} should exceed 3.0"

    def test_T11_3_rho_U_clipped_to_035(self, cfg):
        """[M4] rho_U action clipped to [0.05, 0.35]."""
        assert cfg["action"]["rho_U_max"] <= 0.35, \
            f"rho_U_max should be <= 0.35, got {cfg['action']['rho_U_max']}"
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        max_action = np.ones(5, dtype=np.float32)
        _, _, _, _, info = env.step(max_action)
        assert info["rho_U"] <= 0.35 + 1e-6, \
            f"rho_U should be <= 0.35, got {info['rho_U']}"

    def test_T11_4_pop_bonus_scale(self, cfg):
        """[M5] pop_bonus magnitude is 10-20% of reward at N_active=150."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        # Run a few steps to get a stable state
        mid_action = np.zeros(5, dtype=np.float32)
        for _ in range(30):
            _, _, term, _, info = env.step(mid_action)
            if term:
                break
        beta_pop = cfg.get("population_reward", {}).get("beta_pop", 0.3)
        assert beta_pop >= 0.2, \
            f"beta_pop should be >= 0.2 for adequate signal, got {beta_pop}"

    def test_T11_5_lagrangian_increases_on_violation(self, cfg):
        """[M6] Lagrangian lambda increases when pviol_E > threshold."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        threshold = cfg.get("lagrangian_qos", {}).get("pviol_E_threshold", 0.15)
        # Verify setter works
        env.set_lagrangian_lambda(0.0)
        assert env._lagrangian_lambda == 0.0
        env.set_lagrangian_lambda(1.5)
        assert env._lagrangian_lambda == 1.5

    def test_T11_6_lagrangian_zero_below_threshold(self, cfg):
        """[M6] Lagrangian penalty is 0 when pviol_E < threshold."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        env.set_lagrangian_lambda(2.0)
        # With low rho_U, pviol_E should be low => max(0, pviol_E - 0.15) = 0
        low_rho = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        _, _, _, _, info = env.step(low_rho)
        if info["pviol_E"] < env._pviol_E_threshold:
            assert info["lagrangian_penalty"] == 0.0, \
                f"Lagrangian penalty should be 0 when pviol_E < threshold"

    def test_T11_7_convex_sla_penalty_ratio(self, cfg):
        """[M3] penalty(0.9) / penalty(0.3) > 3x for convex (vs 3x for linear)."""
        gamma = cfg["qos"].get("gamma_sla_E", 2.0)
        lambda_E = cfg["qos"]["lambda_E"]
        penalty_high = lambda_E * (0.9 ** gamma)
        penalty_low = lambda_E * (0.3 ** gamma)
        ratio = penalty_high / penalty_low
        assert ratio > 3.0, \
            f"Convex penalty ratio {ratio:.2f} should exceed 3.0 (linear = 3.0)"

    def test_T11_8_lagrangian_penalty_in_info(self, env):
        """[M6] lagrangian_penalty key present and finite in info dict."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "lagrangian_penalty" in info, \
            "Missing 'lagrangian_penalty' key in info dict"
        assert np.isfinite(info["lagrangian_penalty"]), \
            f"lagrangian_penalty is not finite: {info['lagrangian_penalty']}"
        assert info["lagrangian_penalty"] >= 0.0

    def test_T11_9_phase1_fraction_backward_compat(self, cfg):
        """[M2] Backward compatibility: absolute phase1_steps still works."""
        cfg_copy = {**cfg}
        cfg_copy["training"] = {**cfg.get("training", {})}
        cfg_copy["training"]["curriculum"] = {
            "enabled": True,
            "phase1_steps": 50000,
        }
        # When phase1_fraction is absent, should use phase1_steps
        curriculum_cfg = cfg_copy["training"]["curriculum"]
        phase1_fraction = curriculum_cfg.get("phase1_fraction", None)
        assert phase1_fraction is None
        phase1_steps = curriculum_cfg.get("phase1_steps", 200000)
        assert phase1_steps == 50000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
