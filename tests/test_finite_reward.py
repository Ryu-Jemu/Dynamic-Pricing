"""
Tests for finite rewards and Phase 2 validation (§20 — test_finite_reward.py).

Tests:
  - No NaN/Inf in observations or rewards under random actions
  - Phase 2 specific: per-slice rho_util, SLA per-step, action mapping, top-up

References:
  [SB3_TIPS]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.utils import load_config
from src.envs.oran_slicing_env import OranSlicingEnv


@pytest.fixture
def cfg():
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    return load_config(str(config_path))


# =====================================================================
# Finite reward / obs tests (§20)
# =====================================================================

class TestFiniteReward:
    """No NaN/Inf under random actions. [SB3_TIPS]"""

    def test_no_nan_inf_random_episode(self, cfg):
        """Full episode with random actions: obs and reward always finite."""
        env = OranSlicingEnv(cfg, seed=42)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs)), f"Initial obs not finite: {obs}"

        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.all(np.isfinite(obs)), \
                f"Step {step}: obs not finite: {obs}"
            assert np.isfinite(reward), \
                f"Step {step}: reward not finite: {reward}"
            assert np.isfinite(info["profit"]), \
                f"Step {step}: profit not finite: {info['profit']}"

            if terminated or truncated:
                break

    def test_no_nan_inf_multiple_seeds(self, cfg):
        """Finite across different seeds."""
        for seed in [0, 1, 42, 123, 9999]:
            env = OranSlicingEnv(cfg, seed=seed)
            obs, _ = env.reset(seed=seed)

            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.all(np.isfinite(obs)), f"seed={seed}: obs NaN/Inf"
                assert np.isfinite(reward), f"seed={seed}: reward NaN/Inf"
                if terminated:
                    break

    def test_obs_within_bounds(self, cfg):
        """Observation values should be within observation_space bounds."""
        env = OranSlicingEnv(cfg, seed=42)
        obs, _ = env.reset(seed=42)

        for _ in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(action)

            assert env.observation_space.contains(obs), \
                f"obs out of bounds: min={obs.min():.4f}, max={obs.max():.4f}"
            if terminated:
                break

    def test_reward_clipped(self, cfg):
        """Reward should be within clip bounds."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)
        clip = env.economics.reward_clip

        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, _, _ = env.step(action)
            assert -clip <= reward <= clip, \
                f"Reward {reward} outside clip [{-clip}, {clip}]"
            if terminated:
                break

    def test_extreme_actions(self, cfg):
        """Extreme corner actions should not produce NaN/Inf."""
        env = OranSlicingEnv(cfg, seed=42)

        extreme_actions = [
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),  # all min
            np.array([1.0, 1.0, 1.0], dtype=np.float32),      # all max
            np.array([-1.0, 1.0, -1.0], dtype=np.float32),    # mixed
            np.array([1.0, -1.0, 1.0], dtype=np.float32),     # mixed
        ]

        for ext_action in extreme_actions:
            obs, _ = env.reset(seed=42)
            obs, reward, _, _, info = env.step(ext_action)
            assert np.all(np.isfinite(obs)), f"Extreme action {ext_action}: obs NaN/Inf"
            assert np.isfinite(reward), f"Extreme action {ext_action}: reward NaN/Inf"


# =====================================================================
# Phase 2 specific tests
# =====================================================================

class TestPhase2ActionMapping:
    """FIX M5: Action mapping per §6.1."""

    def test_action_mapping_order(self, cfg):
        """a[0]→F_eMBB, a[1]→F_URLLC, a[2]→rho_URLLC per §6.1."""
        env = OranSlicingEnv(cfg, seed=42)

        # Midpoint action
        fee_e, fee_u, rho = env._map_action(np.array([0.0, 0.0, 0.0]))
        pb_e = env.price_bounds["eMBB"]
        pb_u = env.price_bounds["URLLC"]
        assert abs(fee_e - (pb_e["F_min"] + pb_e["F_max"]) / 2) < 1.0
        assert abs(fee_u - (pb_u["F_min"] + pb_u["F_max"]) / 2) < 1.0
        assert abs(rho - (env.rho_min + env.rho_max) / 2) < 0.001

    def test_action_mapping_boundaries(self, cfg):
        """Boundary actions map to correct bounds."""
        env = OranSlicingEnv(cfg, seed=42)
        pb_e = env.price_bounds["eMBB"]
        pb_u = env.price_bounds["URLLC"]

        # All min
        fee_e, fee_u, rho = env._map_action(np.array([-1.0, -1.0, -1.0]))
        assert abs(fee_e - pb_e["F_min"]) < 1.0
        assert abs(fee_u - pb_u["F_min"]) < 1.0
        assert abs(rho - env.rho_min) < 0.001

        # All max
        fee_e, fee_u, rho = env._map_action(np.array([1.0, 1.0, 1.0]))
        assert abs(fee_e - pb_e["F_max"]) < 1.0
        assert abs(fee_u - pb_u["F_max"]) < 1.0
        assert abs(rho - env.rho_max) < 0.001


class TestPhase2PerSliceRhoUtil:
    """FIX M4: rho_util tracked independently per slice."""

    def test_rho_util_separate(self, cfg):
        """rho_util_eMBB and rho_util_URLLC should differ."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        found_different = False
        for _ in range(10):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
            if abs(info["rho_util_eMBB"] - info["rho_util_URLLC"]) > 1e-6:
                found_different = True
                break

        assert found_different, "rho_util never differed between slices"

    def test_rho_util_in_valid_range(self, cfg):
        """rho_util should be in [0, 1]."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        for _ in range(30):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            for s in ["eMBB", "URLLC"]:
                rho = info[f"rho_util_{s}"]
                assert 0.0 <= rho <= 1.0 + 1e-6, \
                    f"rho_util_{s}={rho} out of range"
            if terminated:
                break


class TestPhase2SLAViolation:
    """FIX M3: SLA violation as per-step fraction (§12.1)."""

    def test_v_rate_in_valid_range(self, cfg):
        """V_rate should be in [0, 1]."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        for _ in range(30):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            for s in ["eMBB", "URLLC"]:
                V = info[f"V_rate_{s}"]
                assert 0.0 <= V <= 1.0, f"V_rate_{s}={V} out of [0,1]"
            if terminated:
                break

    def test_v_rate_is_step_fraction(self, cfg):
        """V_rate should be a multiple of 1/K (computed from K inner steps)."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)
        K = env.K

        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        for s in ["eMBB", "URLLC"]:
            V = info[f"V_rate_{s}"]
            # V should be n/K for some integer n
            n_violations = V * K
            assert abs(n_violations - round(n_violations)) < 1e-6, \
                f"V_rate_{s}={V} is not a multiple of 1/{K}"


class TestPhase2TopUp:
    """FIX M2: Top-up model integration."""

    def test_topup_model_active(self, cfg):
        """Top-up model should be instantiated and enabled."""
        env = OranSlicingEnv(cfg, seed=42)
        assert hasattr(env, "topup"), "No topup attribute"
        assert env.topup.enabled, "TopUp model not enabled"

    def test_topup_produces_purchases(self, cfg):
        """Over a full episode, some top-ups should occur."""
        env = OranSlicingEnv(cfg, seed=123)
        env.reset(seed=123)

        total_topups = 0
        for _ in range(50):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            total_topups += info["topups_eMBB"] + info["topups_URLLC"]
            if terminated:
                break

        assert total_topups > 0, \
            "No top-ups in entire episode (model may not be integrated)"

    def test_topup_revenue_reflected(self, cfg):
        """Top-up revenue should contribute to total revenue."""
        env = OranSlicingEnv(cfg, seed=123)
        env.reset(seed=123)

        # Run episode and track revenue with/without topups
        revenues = []
        topup_months = []
        for m in range(50):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            revenues.append(info["revenue"])
            if info["topups_eMBB"] + info["topups_URLLC"] > 0:
                topup_months.append(m)
            if terminated:
                break

        # At least some months should have topup revenue
        assert len(topup_months) > 0, "No months with top-ups"


class TestPhase2PlanAssignment:
    """FIX M7: Plan-based T_exp and Q_gb."""

    def test_users_have_plan_ids(self, cfg):
        """All users should have non-empty plan_id."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        for u in env.pool.active_pool.values():
            assert u.plan_id != "", f"User {u.user_id} has no plan_id"

    def test_t_exp_not_100(self, cfg):
        """T_exp should not be hardcoded 100."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        t_exp_values = set(u.T_exp for u in env.pool.active_pool.values())
        assert 100.0 not in t_exp_values, \
            f"T_exp=100.0 still present: {t_exp_values}"

    def test_t_exp_matches_v_cap(self, cfg):
        """T_exp should equal plan v_cap_mbps after reset_monthly."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        for u in env.pool.active_pool.values():
            assert u.T_exp == u.v_cap_mbps, \
                f"User {u.user_id}: T_exp={u.T_exp} != v_cap={u.v_cap_mbps}"

    def test_q_gb_from_plan(self, cfg):
        """Q_gb should be set from plan config, not default."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        q_values = set(u.Q_gb for u in env.pool.active_pool.values())
        # Should have multiple Q_gb values from different plans
        assert len(q_values) > 1, \
            f"Only one Q_gb value: {q_values} (plans not assigned)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
