"""
User pool management with strict invariants (Section 8).

Maintains three disjoint pools with unique user IDs:
  - inactive_pool:  potential subscribers (available for join)
  - active_pool:    currently subscribed users
  - churned_pool:   users who have left

Strict invariants (asserted and tested):
  1. Disjointness: pools are pairwise disjoint.
  2. Conservation: total_users = |inactive| + |active| + |churned|
  3. Transitions only at month boundaries via join/churn events.

References:
  [SB3_TIPS] [CHURN_SLR] [DISCONF_PDF]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("oran.pools")


@dataclass
class User:
    """Single user record."""

    # Immutable
    user_id: int
    slice: str
    segment: str

    # Plan assignment (§7) — set during creation, immutable per user
    plan_id: str = ""
    Q_gb: float = 50.0          # monthly data cap from plan
    v_cap_mbps: float = 3.0     # post-cap throttle speed from plan

    # Sensitivities [CHURN_SLR]
    w_price: float = 1.0
    w_qos: float = 1.0
    sw_cost: float = 0.5
    b_u: float = 0.0

    # Monthly fields (reset each step)
    D_u: float = 0.0
    topup_flag: bool = False
    T_act_avg: float = 0.0
    T_exp: float = 3.0          # §10.3/§11: plan-based, NOT hardcoded 100
    delta_disc: float = 0.0
    stay_prob: float = 1.0

    def reset_monthly(self) -> None:
        """Reset monthly fields. T_exp resets to plan's v_cap (not 100)."""
        self.D_u = 0.0
        self.topup_flag = False
        self.T_act_avg = 0.0
        self.T_exp = self.v_cap_mbps   # §10.3: plan-based expected throughput
        self.delta_disc = 0.0
        self.stay_prob = 1.0


class UserPoolManager:
    """Manages the three disjoint user pools."""

    def __init__(self) -> None:
        self._inactive: Dict[int, User] = {}
        self._active: Dict[int, User] = {}
        self._churned: Dict[int, User] = {}
        self._next_id: int = 0
        self._total_created: int = 0

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
        return [u for u in self._active.values() if u.slice == slice_name]

    def create_user(self, slice_name: str, segment: str,
                    pool: str = "inactive",
                    sensitivities: Optional[Dict[str, float]] = None,
                    plan_id: str = "",
                    Q_gb: float = 50.0,
                    v_cap_mbps: float = 3.0) -> User:
        uid = self._next_id
        self._next_id += 1
        self._total_created += 1

        sens = sensitivities or {}
        user = User(
            user_id=uid, slice=slice_name, segment=segment,
            plan_id=plan_id, Q_gb=Q_gb, v_cap_mbps=v_cap_mbps,
            w_price=sens.get("w_price", 1.0),
            w_qos=sens.get("w_qos", 1.0),
            sw_cost=sens.get("sw_cost", 0.5),
            b_u=sens.get("b_u", 0.0),
        )
        # Initialize T_exp from plan
        user.T_exp = v_cap_mbps

        if pool == "inactive":
            self._inactive[uid] = user
        elif pool == "active":
            self._active[uid] = user
        else:
            raise ValueError(f"Invalid initial pool: {pool}")
        return user

    def join(self, user_ids: List[int]) -> List[User]:
        joined = []
        for uid in user_ids:
            if uid in self._inactive:
                user = self._inactive.pop(uid)
                user.reset_monthly()
                self._active[uid] = user
                joined.append(user)
        return joined

    def churn(self, user_ids: List[int]) -> List[User]:
        churned = []
        for uid in user_ids:
            if uid in self._active:
                user = self._active.pop(uid)
                user._churn_cooldown = 0  # track months since churn
                self._churned[uid] = user
                churned.append(user)
        return churned

    def recycle_churned(self, cooldown_months: int = 3) -> int:
        """Move churned users back to inactive pool after cooldown (D9).

        Prevents inactive pool depletion over long episodes.
        Users who churned >= cooldown_months ago are recycled.

        Args:
            cooldown_months: Minimum months before a churned user can rejoin.

        Returns:
            Number of users recycled.
        """
        to_recycle = []
        for uid, user in self._churned.items():
            cd = getattr(user, '_churn_cooldown', 0)
            if cd >= cooldown_months:
                to_recycle.append(uid)
            else:
                user._churn_cooldown = cd + 1

        for uid in to_recycle:
            user = self._churned.pop(uid)
            user.reset_monthly()
            # Reset churn-related state so user arrives fresh
            user.delta_disc = 0.0
            user.stay_prob = 1.0
            self._inactive[uid] = user

        return len(to_recycle)

    def reset_monthly_fields(self) -> None:
        for user in self._active.values():
            user.reset_monthly()

    def assert_invariants(self) -> None:
        ids_inactive = set(self._inactive.keys())
        ids_active = set(self._active.keys())
        ids_churned = set(self._churned.keys())

        assert ids_inactive.isdisjoint(ids_active), \
            f"inactive ∩ active = {ids_inactive & ids_active}"
        assert ids_inactive.isdisjoint(ids_churned), \
            f"inactive ∩ churned = {ids_inactive & ids_churned}"
        assert ids_active.isdisjoint(ids_churned), \
            f"active ∩ churned = {ids_active & ids_churned}"

        total = len(ids_inactive) + len(ids_active) + len(ids_churned)
        all_ids = ids_inactive | ids_active | ids_churned
        assert total == len(all_ids), \
            f"Conservation violation: sum={total}, unique={len(all_ids)}"

    @classmethod
    def from_config(cls, cfg: Dict[str, Any],
                    rng: Optional[np.random.Generator] = None) -> "UserPoolManager":
        if rng is None:
            rng = np.random.default_rng()

        manager = cls()
        pop_cfg = cfg.get("population", {})
        seg_cfg = cfg.get("segments", {})
        plans_cfg = cfg.get("plans", {})

        N0_eMBB = pop_cfg.get("N0_eMBB", 120)
        N0_URLLC = pop_cfg.get("N0_URLLC", 30)
        inactive_size = pop_cfg.get("inactive_pool_size", 2000)

        seg_names = seg_cfg.get("names", ["light", "mid", "heavy", "qos_sensitive"])
        seg_probs = seg_cfg.get("proportions", [0.25, 0.40, 0.25, 0.10])
        sens_cfg = seg_cfg.get("sensitivity", {})

        # §7: Segment → plan tier mapping
        # light → basic (index 0), mid → standard (1),
        # heavy/qos_sensitive → premium (2)
        _seg_to_plan_idx = {
            "light": 0, "mid": 1, "heavy": 2, "qos_sensitive": 2,
        }

        def _get_plan(slice_name: str, segment: str) -> Dict[str, Any]:
            """Look up plan for a given slice+segment."""
            slice_plans = plans_cfg.get(slice_name, [])
            if not slice_plans:
                return {"plan_id": "", "Q_gb_month": 50.0, "v_cap_mbps": 3.0}
            idx = _seg_to_plan_idx.get(segment, 1)
            idx = min(idx, len(slice_plans) - 1)
            return slice_plans[idx]

        def _make_users(n: int, slice_name: str, pool: str) -> None:
            segs = rng.choice(seg_names, size=n, p=seg_probs)
            for seg in segs:
                s = sens_cfg.get(seg, {})
                sensitivities = {
                    "w_price": s.get("w_price", 1.0) + rng.normal(0, 0.05),
                    "w_qos": s.get("w_qos", 1.0) + rng.normal(0, 0.05),
                    "sw_cost": max(0.0, s.get("sw_cost", 0.5) + rng.normal(0, 0.05)),
                    "b_u": s.get("b_u", 0.0) + rng.normal(0, 0.1),
                }
                plan = _get_plan(slice_name, seg)
                manager.create_user(
                    slice_name, seg, pool=pool,
                    sensitivities=sensitivities,
                    plan_id=plan.get("plan_id", ""),
                    Q_gb=plan.get("Q_gb_month", 50.0),
                    v_cap_mbps=plan.get("v_cap_mbps", 3.0),
                )

        _make_users(N0_eMBB, "eMBB", "active")
        _make_users(N0_URLLC, "URLLC", "active")

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
