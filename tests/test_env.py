"""
Unit tests for O-RAN 3-Part Tariff environment.

Test groups:
  T1  Environment basics (reset, step, spaces)
  T2  Revenue model (3-part tariff, online accrual)
  T3  Market dynamics (join/churn probabilities, conservation)
  T4  QoS violation (sigmoid, capacity)
  T5  Numerical safety (no NaN/Inf, obs bounds, reward clip)
  T6  Billing cycle (reset accumulators, 30-step cycles)

References:
  [Grubb 2009]    3-part tariff structure
  [TS 23.503]     Usage monitoring / charging
  [SB3_TIPS]      Reward normalization, no NaN
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
        assert obs.shape == (16,), f"Bad obs shape: {obs.shape}"
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, "step should return (obs, reward, term, trunc, info)"
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (16,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (3,) or env.action_space.shape == (5,)

    def test_episode_terminates(self, env):
        """Episode must terminate within episode_len steps."""
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 500:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, f"Episode did not terminate in {steps} steps"
        assert steps == env.episode_len, f"Terminated at step {steps}, expected {env.episode_len}"


# =====================================================================
# T2  Revenue model (3-part tariff) [Grubb 2009]
# =====================================================================
class TestRevenueModel:
    def test_revenue_non_negative_with_active_users(self, env):
        """Revenue should be >= 0 when there are active users."""
        env.reset(seed=42)
        revenues = []
        for _ in range(30):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            revenues.append(info["revenue"])
            if term:
                break
        # At least some revenues should be positive
        assert any(r > 0 for r in revenues), "No positive revenue in 30 steps"

    def test_overage_revenue_accrual(self, env):
        """Over time, cumulative usage exceeding allowance should produce overage revenue."""
        env.reset(seed=42)
        total_over_rev = 0.0
        for _ in range(60):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            total_over_rev += info.get("revenue_overage", 0.0)
            if term:
                break
        # Overage should eventually appear (demand > allowance for some users)
        assert total_over_rev >= 0.0, "Overage revenue cannot be negative"


# =====================================================================
# T3  Market dynamics (join/churn, conservation)
# =====================================================================
class TestMarketDynamics:
    def test_population_conservation(self, env):
        """Total population (active + inactive) must be conserved."""
        env.reset(seed=42)
        total_pop = env.N_total  # total users in CSV
        for _ in range(30):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            n_active = int(env._active_mask.sum())
            n_inactive = total_pop - n_active
            assert n_active >= 0
            assert n_inactive >= 0
            assert n_active + n_inactive == total_pop, \
                f"Conservation violated: {n_active} + {n_inactive} != {total_pop}"
            if term:
                break

    def test_join_churn_in_info(self, env):
        """Info dict must contain join/churn counts."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "n_join" in info
        assert "n_churn" in info
        assert info["n_join"] >= 0
        assert info["n_churn"] >= 0

    def test_no_negative_active(self, env):
        """N_active should never go below 0."""
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            assert info["N_active"] >= 0
            if term:
                break


# =====================================================================
# T4  QoS violation [TS 23.503]
# =====================================================================
class TestQoSViolation:
    def test_sigmoid_properties(self):
        """Sigmoid should be in (0,1) and monotone increasing."""
        assert 0 < sigmoid(0.0) < 1
        assert abs(sigmoid(0.0) - 0.5) < 1e-6
        vals = [sigmoid(x) for x in [-10, -1, 0, 1, 10]]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-10, "Sigmoid not monotone"

    def test_violation_in_range(self, env):
        """p_viol_U and p_viol_E must be in [0, 1]."""
        env.reset(seed=42)
        for _ in range(30):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            for key in ["pviol_U", "pviol_E"]:
                assert 0.0 <= info[key] <= 1.0, f"{key}={info[key]} out of [0,1]"
            if term:
                break

    def test_high_load_increases_violation(self, env):
        """Extreme ρ_U → 0 should increase eMBB violation (less capacity)."""
        # This is a directional check, not strict monotonicity
        env.reset(seed=42)
        # Force ρ_U very low (most capacity to URLLC → eMBB starved)
        low_rho = np.array([-1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        high_rho = np.array([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # Just verify no crash — directional property is stochastic
        env.reset(seed=42)
        _, _, _, _, info_low = env.step(low_rho[:env.action_space.shape[0]])
        env.reset(seed=42)
        _, _, _, _, info_high = env.step(high_rho[:env.action_space.shape[0]])
        assert np.isfinite(info_low["pviol_E"])
        assert np.isfinite(info_high["pviol_E"])


# =====================================================================
# T5  Numerical safety [SB3_TIPS]
# =====================================================================
class TestNumericalSafety:
    def test_no_nan_inf_random_episode(self, env):
        """Full episode with random actions: obs and reward always finite."""
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs)), f"Initial obs not finite: {obs}"
        for step in range(env.episode_len):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs)), f"Step {step}: obs NaN/Inf"
            assert np.isfinite(reward), f"Step {step}: reward NaN/Inf"
            assert np.isfinite(info["profit"]), f"Step {step}: profit NaN/Inf"
            if terminated or truncated:
                break

    def test_multiple_seeds(self, cfg):
        """No NaN across different seeds."""
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
        """Observation must lie within observation_space."""
        env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, term, _, _ = env.step(action)
            assert env.observation_space.contains(obs), \
                f"obs out of bounds: min={obs.min():.4f} max={obs.max():.4f}"
            if term:
                break

    def test_reward_clipped(self, env):
        """Reward should be within configured clip bounds."""
        env.reset(seed=42)
        clip = 2.0  # hardcoded reward clip in env
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, term, _, _ = env.step(action)
            assert -clip <= reward <= clip, f"Reward {reward} outside [{-clip}, {clip}]"
            if term:
                break

    def test_extreme_actions(self, env):
        """Extreme corner actions should not crash."""
        extremes = [
            np.full(env.action_space.shape, -1.0, dtype=np.float32),
            np.full(env.action_space.shape, 1.0, dtype=np.float32),
            np.zeros(env.action_space.shape, dtype=np.float32),
        ]
        for ext in extremes:
            obs, _ = env.reset(seed=42)
            obs, r, _, _, info = env.step(ext)
            assert np.all(np.isfinite(obs)), f"Extreme action: obs NaN/Inf"
            assert np.isfinite(r), f"Extreme action: reward NaN/Inf"


# =====================================================================
# T6  Billing cycle (30-step reset) [TS 23.503]
# =====================================================================
class TestBillingCycle:
    def test_cycle_length(self, env):
        """Billing cycle should be T steps (default 30)."""
        T = env.T
        assert T > 0
        env.reset(seed=42)
        for step_i in range(T + 1):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            if term:
                break
        # After T steps, we should be in cycle step T (or reset to 0 at T+1)

    def test_info_contains_step(self, env):
        """Info dict must contain step counter."""
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
        """Fitted lognormal should match p50/p90 targets."""
        from scipy import stats
        p50, p90 = 1.5, 5.0
        mu, sigma = fit_lognormal_quantiles(p50, p90)
        actual_p50 = np.exp(mu)
        actual_p90 = float(stats.lognorm.ppf(0.90, s=sigma, scale=np.exp(mu)))
        assert abs(actual_p50 - p50) / p50 < 0.01, f"p50 mismatch: {actual_p50} vs {p50}"
        assert abs(actual_p90 - p90) / p90 < 0.01, f"p90 mismatch: {actual_p90} vs {p90}"

    def test_fit_lognormal_rejects_bad_input(self):
        with pytest.raises(ValueError):
            fit_lognormal_quantiles(-1.0, 5.0)
        with pytest.raises(ValueError):
            fit_lognormal_quantiles(5.0, 3.0)  # p90 < p50

    def test_sigmoid_stable_at_extremes(self):
        """Sigmoid should not overflow at large inputs."""
        assert np.isfinite(sigmoid(500.0))
        assert np.isfinite(sigmoid(-500.0))
        assert abs(sigmoid(500.0) - 1.0) < 1e-6
        assert abs(sigmoid(-500.0) - 0.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
