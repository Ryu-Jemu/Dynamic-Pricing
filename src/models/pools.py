"""
User pool management with strict invariants.

Maintains three disjoint pools with unique user IDs:
  - inactive_pool:  potential subscribers (available for join)
  - active_pool:    currently subscribed users
  - churned_pool:   users who have left

Strict invariants (asserted and tested):
  1. Disjointness: pools are pairwise disjoint.
  2. Conservation: total_users = |inactive| + |active| + |churned|
  3. Transitions only at month boundaries via join/churn events.

Per-user fields:
  Immutable: user_id, slice, segment
  Sensitivities: w_price, w_qos, sw_cost, b_u
  Monthly (reset each step): D_u, topup_flag, T_act_avg, delta_disc, stay_prob

References:
  [SB3_TIPS] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
  [CHURN_SLR] Springer SLR (2023)
  [DISCONF_PDF] AccessON (2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("oran.pools")


@dataclass
class User:
    """Single user record.

    Immutable fields set at creation; monthly fields reset each step.
    """

    # Immutable
    user_id: int
    slice: str               # "eMBB" or "URLLC"
    segment: str             # "light", "mid", "heavy", "qos_sensitive"

    # Sensitivities [CHURN_SLR]
    w_price: float = 1.0
    w_qos: float = 1.0
    sw_cost: float = 0.5
    b_u: float = 0.0

    # Monthly fields (reset each step)
    D_u: float = 0.0          # monthly demand (GB)
    topup_flag: bool = False
    T_act_avg: float = 0.0    # monthly avg delivered throughput (Mbps)
    T_exp: float = 100.0      # expected throughput (Mbps); may be throttled
    delta_disc: float = 0.0   # disconfirmation [DISCONF_PDF]
    stay_prob: float = 1.0

    def reset_monthly(self) -> None:
        """Reset monthly transient fields at start of each step."""
        self.D_u = 0.0
        self.topup_flag = False
        self.T_act_avg = 0.0
        self.T_exp = 100.0
        self.delta_disc = 0.0
        self.stay_prob = 1.0


class UserPoolManager:
    """Manages the three disjoint user pools.

    All pool transitions go through explicit methods:
      - join():  inactive → active
      - churn(): active → churned
    """

    def __init__(self) -> None:
        self._inactive: Dict[int, User] = {}
        self._active: Dict[int, User] = {}
        self._churned: Dict[int, User] = {}
        self._next_id: int = 0
        self._total_created: int = 0

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def inactive_pool(self) -> Dict[int, User]:
        return self._inactive

    @property
    def active_pool(self) -> Dict[int, User]:
        return self._active

    @property
    def churned_pool(self) -> Dict[int, User]:
        return self._churned

    @property
    def total_users(self) -> int:
        return len(self._inactive) + len(self._active) + len(self._churned)

    def active_count(self, slice_name: Optional[str] = None) -> int:
        if slice_name is None:
            return len(self._active)
        return sum(1 for u in self._active.values() if u.slice == slice_name)

    def inactive_count(self, slice_name: Optional[str] = None) -> int:
        if slice_name is None:
            return len(self._inactive)
        return sum(1 for u in self._inactive.values() if u.slice == slice_name)

    def get_active_users(self, slice_name: str) -> List[User]:
        """Return list of active users in a given slice."""
        return [u for u in self._active.values() if u.slice == slice_name]

    # -----------------------------------------------------------------
    # User creation (initial population setup)
    # -----------------------------------------------------------------

    def create_user(
        self,
        slice_name: str,
        segment: str,
        pool: str = "inactive",
        sensitivities: Optional[Dict[str, float]] = None,
    ) -> User:
        """Create a new user and place into specified pool.

        Parameters
        ----------
        pool : str
            One of "inactive", "active".  (Cannot create directly into churned.)
        """
        uid = self._next_id
        self._next_id += 1
        self._total_created += 1

        sens = sensitivities or {}
        user = User(
            user_id=uid,
            slice=slice_name,
            segment=segment,
            w_price=sens.get("w_price", 1.0),
            w_qos=sens.get("w_qos", 1.0),
            sw_cost=sens.get("sw_cost", 0.5),
            b_u=sens.get("b_u", 0.0),
        )

        if pool == "inactive":
            self._inactive[uid] = user
        elif pool == "active":
            self._active[uid] = user
        else:
            raise ValueError(f"Invalid initial pool: {pool}")

        return user

    # -----------------------------------------------------------------
    # Pool transitions (month boundary only)
    # -----------------------------------------------------------------

    def join(self, user_ids: List[int]) -> List[User]:
        """Move users from inactive → active (join event).

        Returns list of joined users.
        Silently skips IDs not found in inactive pool.
        """
        joined = []
        for uid in user_ids:
            if uid in self._inactive:
                user = self._inactive.pop(uid)
                user.reset_monthly()
                self._active[uid] = user
                joined.append(user)
        return joined

    def churn(self, user_ids: List[int]) -> List[User]:
        """Move users from active → churned (churn event).

        Returns list of churned users.
        Silently skips IDs not found in active pool.
        """
        churned = []
        for uid in user_ids:
            if uid in self._active:
                user = self._active.pop(uid)
                self._churned[uid] = user
                churned.append(user)
        return churned

    # -----------------------------------------------------------------
    # Monthly reset
    # -----------------------------------------------------------------

    def reset_monthly_fields(self) -> None:
        """Reset transient monthly fields for all active users."""
        for user in self._active.values():
            user.reset_monthly()

    # -----------------------------------------------------------------
    # Invariant checks
    # -----------------------------------------------------------------

    def assert_invariants(self) -> None:
        """Assert pool disjointness and conservation.

        Raises AssertionError if invariants are violated.
        """
        ids_inactive = set(self._inactive.keys())
        ids_active = set(self._active.keys())
        ids_churned = set(self._churned.keys())

        # Disjointness
        assert ids_inactive.isdisjoint(ids_active), (
            f"inactive ∩ active = {ids_inactive & ids_active}"
        )
        assert ids_inactive.isdisjoint(ids_churned), (
            f"inactive ∩ churned = {ids_inactive & ids_churned}"
        )
        assert ids_active.isdisjoint(ids_churned), (
            f"active ∩ churned = {ids_active & ids_churned}"
        )

        # Conservation
        total = len(ids_inactive) + len(ids_active) + len(ids_churned)
        all_ids = ids_inactive | ids_active | ids_churned
        assert total == len(all_ids), (
            f"Conservation violation: sum of pools={total}, unique IDs={len(all_ids)}"
        )

    # -----------------------------------------------------------------
    # Population initialization (from config)
    # -----------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
    ) -> "UserPoolManager":
        """Initialize pools from config (Section 8.1).

        Creates:
          - N0_eMBB active eMBB users
          - N0_URLLC active URLLC users
          - inactive_pool_size inactive users (split proportionally)
        """
        if rng is None:
            rng = np.random.default_rng()

        manager = cls()
        pop_cfg = cfg.get("population", {})
        seg_cfg = cfg.get("segments", {})

        N0_eMBB = pop_cfg.get("N0_eMBB", 120)
        N0_URLLC = pop_cfg.get("N0_URLLC", 30)
        inactive_size = pop_cfg.get("inactive_pool_size", 2000)

        seg_names = seg_cfg.get("names", ["light", "mid", "heavy", "qos_sensitive"])
        seg_probs = seg_cfg.get("proportions", [0.25, 0.40, 0.25, 0.10])
        sens_cfg = seg_cfg.get("sensitivity", {})

        def _make_users(n: int, slice_name: str, pool: str) -> None:
            segs = rng.choice(seg_names, size=n, p=seg_probs)
            for seg in segs:
                s = sens_cfg.get(seg, {})
                # Add small noise to sensitivities
                sensitivities = {
                    "w_price": s.get("w_price", 1.0) + rng.normal(0, 0.05),
                    "w_qos": s.get("w_qos", 1.0) + rng.normal(0, 0.05),
                    "sw_cost": max(0.0, s.get("sw_cost", 0.5) + rng.normal(0, 0.05)),
                    "b_u": s.get("b_u", 0.0) + rng.normal(0, 0.1),
                }
                manager.create_user(slice_name, seg, pool=pool,
                                    sensitivities=sensitivities)

        # Active users (Section 8.1)
        _make_users(N0_eMBB, "eMBB", "active")
        _make_users(N0_URLLC, "URLLC", "active")

        # Inactive pool — split roughly 70% eMBB, 30% URLLC (scenario)
        n_inactive_embb = int(inactive_size * 0.7)
        n_inactive_urllc = inactive_size - n_inactive_embb
        _make_users(n_inactive_embb, "eMBB", "inactive")
        _make_users(n_inactive_urllc, "URLLC", "inactive")

        manager.assert_invariants()
        logger.info(
            "Pools initialized: active(eMBB=%d, URLLC=%d), inactive=%d",
            N0_eMBB, N0_URLLC, inactive_size,
        )
        return manager
