"""
test_invariants.py — Pool disjointness + conservation; transitions only
at month boundary.

Tests are deterministic by construction (fixed inputs or local fixed RNG).
  [SB3_TIPS]

Section 20:
  - pool disjointness: pools are pairwise disjoint
  - conservation: total_users = |inactive| + |active| + |churned|
  - transitions only at month boundaries via join/churn events
"""

import unittest
import numpy as np

from src.models.utils import load_config
from src.models.pools import UserPoolManager, User


class TestPoolInvariants(unittest.TestCase):
    """Test pool disjointness, conservation, and transition rules."""

    def setUp(self):
        """Initialize pools with a local fixed RNG (test only)."""
        self.cfg = load_config("config/default.yaml")
        self.rng = np.random.default_rng(12345)  # local test RNG
        self.pm = UserPoolManager.from_config(self.cfg, rng=self.rng)

    # ---------------------------------------------------------------
    # Disjointness
    # ---------------------------------------------------------------

    def test_initial_disjointness(self):
        """Pools must be pairwise disjoint after initialization."""
        ids_i = set(self.pm.inactive_pool.keys())
        ids_a = set(self.pm.active_pool.keys())
        ids_c = set(self.pm.churned_pool.keys())

        self.assertTrue(ids_i.isdisjoint(ids_a),
                        f"inactive ∩ active = {ids_i & ids_a}")
        self.assertTrue(ids_i.isdisjoint(ids_c),
                        f"inactive ∩ churned = {ids_i & ids_c}")
        self.assertTrue(ids_a.isdisjoint(ids_c),
                        f"active ∩ churned = {ids_a & ids_c}")

    def test_disjointness_after_join(self):
        """Disjointness must hold after join events."""
        # Pick 10 inactive users
        inactive_ids = list(self.pm.inactive_pool.keys())[:10]
        self.pm.join(inactive_ids)
        self.pm.assert_invariants()  # raises if violated

    def test_disjointness_after_churn(self):
        """Disjointness must hold after churn events."""
        active_ids = list(self.pm.active_pool.keys())[:5]
        self.pm.churn(active_ids)
        self.pm.assert_invariants()

    def test_disjointness_after_join_then_churn(self):
        """Disjointness must hold after join followed by churn."""
        # Join 10
        inactive_ids = list(self.pm.inactive_pool.keys())[:10]
        joined = self.pm.join(inactive_ids)
        self.assertEqual(len(joined), 10)

        # Churn 5 of them
        joined_ids = [u.user_id for u in joined[:5]]
        churned = self.pm.churn(joined_ids)
        self.assertEqual(len(churned), 5)

        self.pm.assert_invariants()

    # ---------------------------------------------------------------
    # Conservation
    # ---------------------------------------------------------------

    def test_initial_conservation(self):
        """total_users = |inactive| + |active| + |churned| at init."""
        total = (len(self.pm.inactive_pool)
                 + len(self.pm.active_pool)
                 + len(self.pm.churned_pool))
        self.assertEqual(total, self.pm.total_users)

    def test_conservation_after_transitions(self):
        """Conservation after multiple join/churn cycles."""
        total_before = self.pm.total_users

        # Join 15
        inactive_ids = list(self.pm.inactive_pool.keys())[:15]
        self.pm.join(inactive_ids)
        self.assertEqual(self.pm.total_users, total_before)

        # Churn 8
        active_ids = list(self.pm.active_pool.keys())[:8]
        self.pm.churn(active_ids)
        self.assertEqual(self.pm.total_users, total_before)

    def test_no_user_duplication(self):
        """No user ID should appear in more than one pool."""
        all_ids = (
            list(self.pm.inactive_pool.keys())
            + list(self.pm.active_pool.keys())
            + list(self.pm.churned_pool.keys())
        )
        self.assertEqual(len(all_ids), len(set(all_ids)),
                         "Duplicate user IDs found across pools")

    # ---------------------------------------------------------------
    # Transition correctness
    # ---------------------------------------------------------------

    def test_join_moves_inactive_to_active(self):
        """Join must move user from inactive to active (not copy)."""
        uid = list(self.pm.inactive_pool.keys())[0]
        self.assertIn(uid, self.pm.inactive_pool)
        self.assertNotIn(uid, self.pm.active_pool)

        self.pm.join([uid])

        self.assertNotIn(uid, self.pm.inactive_pool)
        self.assertIn(uid, self.pm.active_pool)

    def test_churn_moves_active_to_churned(self):
        """Churn must move user from active to churned (not copy)."""
        uid = list(self.pm.active_pool.keys())[0]
        self.assertIn(uid, self.pm.active_pool)
        self.assertNotIn(uid, self.pm.churned_pool)

        self.pm.churn([uid])

        self.assertNotIn(uid, self.pm.active_pool)
        self.assertIn(uid, self.pm.churned_pool)

    def test_join_nonexistent_is_safe(self):
        """Joining a user not in inactive pool should be silently skipped."""
        fake_id = 999999
        result = self.pm.join([fake_id])
        self.assertEqual(len(result), 0)
        self.pm.assert_invariants()

    def test_churn_nonexistent_is_safe(self):
        """Churning a user not in active pool should be silently skipped."""
        fake_id = 999999
        result = self.pm.churn([fake_id])
        self.assertEqual(len(result), 0)
        self.pm.assert_invariants()

    # ---------------------------------------------------------------
    # User field immutability
    # ---------------------------------------------------------------

    def test_user_immutable_fields_preserved(self):
        """user_id, slice, segment must not change across transitions."""
        uid = list(self.pm.inactive_pool.keys())[0]
        user_before = self.pm.inactive_pool[uid]
        slice_before = user_before.slice
        seg_before = user_before.segment

        self.pm.join([uid])
        user_after = self.pm.active_pool[uid]

        self.assertEqual(user_after.user_id, uid)
        self.assertEqual(user_after.slice, slice_before)
        self.assertEqual(user_after.segment, seg_before)

    # ---------------------------------------------------------------
    # Monthly reset
    # ---------------------------------------------------------------

    def test_monthly_reset(self):
        """Monthly reset should zero transient fields for active users."""
        users = list(self.pm.active_pool.values())[:3]
        for u in users:
            u.D_u = 50.0
            u.T_act_avg = 25.0
            u.delta_disc = 10.0

        self.pm.reset_monthly_fields()

        for u in users:
            self.assertEqual(u.D_u, 0.0)
            self.assertEqual(u.T_act_avg, 0.0)
            self.assertEqual(u.delta_disc, 0.0)


if __name__ == "__main__":
    unittest.main()
