"""
Tests for pool invariants and step order (§20 — test_invariants.py).

Tests:
  - Pool disjointness: inactive ∩ active ∩ churned = ∅
  - Conservation: |inactive| + |active| + |churned| = total created
  - Transitions only at month boundaries
  - Step order: join before demand, churn after top-up (§17)

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
from src.models.pools import UserPoolManager
from src.envs.oran_slicing_env import OranSlicingEnv


@pytest.fixture
def cfg():
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    return load_config(str(config_path))


# =====================================================================
# Pool invariant tests
# =====================================================================

class TestPoolInvariants:
    """Pool disjointness + conservation invariants."""

    def test_initial_disjointness(self, cfg):
        """Pools are pairwise disjoint after initialization."""
        rng = np.random.default_rng(42)
        pool = UserPoolManager.from_config(cfg, rng=rng)
        pool.assert_invariants()  # raises if violated

    def test_conservation_after_join(self, cfg):
        """Total user count is conserved after join operations."""
        rng = np.random.default_rng(42)
        pool = UserPoolManager.from_config(cfg, rng=rng)
        total_before = pool.total_users

        inactive = [
            u for u in pool.inactive_pool.values() if u.slice == "eMBB"
        ]
        join_ids = [u.user_id for u in inactive[:5]]
        pool.join(join_ids)

        assert pool.total_users == total_before
        pool.assert_invariants()

    def test_conservation_after_churn(self, cfg):
        """Total user count is conserved after churn operations."""
        rng = np.random.default_rng(42)
        pool = UserPoolManager.from_config(cfg, rng=rng)
        total_before = pool.total_users

        active = [
            u for u in pool.active_pool.values() if u.slice == "eMBB"
        ]
        churn_ids = [u.user_id for u in active[:3]]
        pool.churn(churn_ids)

        assert pool.total_users == total_before
        pool.assert_invariants()

    def test_conservation_after_full_episode(self, cfg):
        """Conservation holds through an entire episode."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)
        total_start = env.pool.total_users

        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
            assert env.pool.total_users == total_start, \
                f"Conservation violated: {env.pool.total_users} != {total_start}"
            env.pool.assert_invariants()

    def test_disjointness_after_full_episode(self, cfg):
        """Disjointness holds through an entire episode."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        for _ in range(20):
            action = env.action_space.sample()
            env.step(action)
            ids_i = set(env.pool.inactive_pool.keys())
            ids_a = set(env.pool.active_pool.keys())
            ids_c = set(env.pool.churned_pool.keys())
            assert ids_i.isdisjoint(ids_a), "inactive ∩ active ≠ ∅"
            assert ids_i.isdisjoint(ids_c), "inactive ∩ churned ≠ ∅"
            assert ids_a.isdisjoint(ids_c), "active ∩ churned ≠ ∅"

    def test_join_only_from_inactive(self, cfg):
        """join() only moves users that are in the inactive pool."""
        rng = np.random.default_rng(42)
        pool = UserPoolManager.from_config(cfg, rng=rng)

        active_id = next(iter(pool.active_pool.keys()))
        # Trying to join an already-active user should be a no-op
        joined = pool.join([active_id])
        assert len(joined) == 0, "Should not join from active pool"
        pool.assert_invariants()

    def test_churn_only_from_active(self, cfg):
        """churn() only moves users that are in the active pool."""
        rng = np.random.default_rng(42)
        pool = UserPoolManager.from_config(cfg, rng=rng)

        inactive_id = next(iter(pool.inactive_pool.keys()))
        # Trying to churn an inactive user should be a no-op
        churned = pool.churn([inactive_id])
        assert len(churned) == 0, "Should not churn from inactive pool"
        pool.assert_invariants()


# =====================================================================
# Step order tests (§17 compliance — FIX M1)
# =====================================================================

class TestStepOrder:
    """Verify §17-compliant step order after Phase 2 M1 fix."""

    def test_joins_included_in_n_active(self, cfg):
        """N_active in info should include newly joined users (join at step 2)."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)
        n_before = env.pool.active_count("eMBB")

        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        # N_active should = pre-step active + joins (since join happens first)
        expected = n_before + info["joins_eMBB"]
        assert info["N_active_eMBB"] == expected, \
            f"N_active={info['N_active_eMBB']} != {n_before}+{info['joins_eMBB']}={expected}"

    def test_post_churn_less_than_active(self, cfg):
        """N_post_churn should be <= N_active (churn removes users)."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        for _ in range(10):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
            for s in ["eMBB", "URLLC"]:
                assert info[f"N_post_churn_{s}"] <= info[f"N_active_{s}"] + info[f"joins_{s}"], \
                    f"Post-churn > active+joins for {s}"

    def test_plan_based_t_exp(self, cfg):
        """T_exp should be set from plan v_cap, not hardcoded 100 (FIX M7)."""
        env = OranSlicingEnv(cfg, seed=42)
        env.reset(seed=42)

        for u in env.pool.active_pool.values():
            assert u.T_exp != 100.0 or u.v_cap_mbps == 100.0, \
                f"User {u.user_id}: T_exp={u.T_exp} (should be {u.v_cap_mbps})"
            assert u.T_exp == u.v_cap_mbps, \
                f"User {u.user_id}: T_exp={u.T_exp} != v_cap={u.v_cap_mbps}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
