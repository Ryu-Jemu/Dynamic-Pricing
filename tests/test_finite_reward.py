"""
test_finite_reward.py — No NaN/Inf under random actions.

Run multiple random episodes using all modules and verify that every
computed value (reward, profit, observation components) is finite.

Tests are deterministic by construction (local fixed RNG).
  [SB3_TIPS]
"""

import unittest
import numpy as np

from src.models.utils import load_config, compute_price_bounds
from src.models.radio import RadioConfig, RadioModel
from src.models.demand import DemandConfig, DemandModel
from src.models.pools import UserPoolManager
from src.models.market import MarketModel
from src.models.economics import EconomicsModel
from src.models.topup import TopUpModel
from src.models.safety import sanitize_obs, validate_action, safe_reward


class TestFiniteReward(unittest.TestCase):
    """Ensure no NaN/Inf under random actions over multi-step episodes."""

    def setUp(self):
        self.cfg = load_config("config/default.yaml")

    def _run_random_episode(self, seed: int, n_months: int = 10):
        """Run a short random episode and collect all values.

        Returns list of dicts, one per month, with all KPIs.
        """
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        radio = RadioModel(RadioConfig.from_config(cfg))
        demand_model = DemandModel(DemandConfig.from_config(cfg))
        pool_mgr = UserPoolManager.from_config(cfg, rng=rng)
        market = MarketModel(cfg)
        econ = EconomicsModel(cfg)
        topup_model = TopUpModel(cfg)
        price_bounds = compute_price_bounds(cfg)

        K = cfg["time"]["inner_loop_K"]
        topup_price = cfg.get("topup", {}).get("price_krw", 11000)

        # Representative plan params
        Q_mid, v_cap_mid = {}, {}
        for sname in ["eMBB", "URLLC"]:
            plans = cfg["plans"][sname]
            mid = plans[len(plans) // 2]
            Q_mid[sname] = mid["Q_gb_month"]
            v_cap_mid[sname] = mid["v_cap_mbps"]

        monthly_records = []

        for month in range(n_months):
            # Random action in [-1, 1]^3
            raw_action = rng.uniform(-1, 1, size=3).astype(np.float32)

            # Map action → fees, rho
            rho_URLLC = float(np.clip(
                0.5 + 0.45 * raw_action[2], 0.05, 0.95
            ))
            fees = {}
            for idx, sname in enumerate(["eMBB", "URLLC"]):
                pb = price_bounds[sname]
                mid_fee = (pb["F_min"] + pb["F_max"]) / 2
                half_range = (pb["F_max"] - pb["F_min"]) / 2
                fees[sname] = float(np.clip(
                    mid_fee + half_range * raw_action[idx],
                    pb["F_min"], pb["F_max"],
                ))

            # Join
            pool_mgr.reset_monthly_fields()
            for sname in ["eMBB", "URLLC"]:
                n_avail = pool_mgr.inactive_count(sname)
                n_join = market.sample_joins(sname, n_avail, rng=rng)
                candidates = [
                    u.user_id for u in pool_mgr.inactive_pool.values()
                    if u.slice == sname
                ][:n_join]
                pool_mgr.join(candidates)

            N_active = {s: pool_mgr.active_count(s) for s in ["eMBB", "URLLC"]}

            # Demand + throttle
            for sname in ["eMBB", "URLLC"]:
                users = pool_mgr.get_active_users(sname)
                if not users:
                    continue
                segs = np.array([u.segment for u in users])
                D = demand_model.sample_demand(sname, len(users), segs, rng=rng)
                for i, u in enumerate(users):
                    u.D_u = D[i]
                    u.T_exp = topup_model.apply_throttle(
                        u.D_u, Q_mid[sname], 100.0, v_cap_mid[sname]
                    )

            # Inner loop
            avg_T_steps = {"eMBB": [], "URLLC": []}
            rho_utils = []

            for k in range(K):
                users_e = pool_mgr.get_active_users("eMBB")
                users_u = pool_mgr.get_active_users("URLLC")
                T_exp_e = np.array([u.T_exp for u in users_e]) if users_e else np.array([])
                T_exp_u = np.array([u.T_exp for u in users_u]) if users_u else np.array([])

                result = radio.inner_step(
                    N_active_eMBB=N_active["eMBB"],
                    N_active_URLLC=N_active["URLLC"],
                    rho_URLLC=rho_URLLC,
                    T_exp_users_eMBB=T_exp_e,
                    T_exp_users_URLLC=T_exp_u,
                )
                avg_T_steps["eMBB"].append(result["avg_T_act_eMBB"])
                avg_T_steps["URLLC"].append(result["avg_T_act_URLLC"])
                rho_utils.append(
                    (result["rho_util_eMBB"] + result["rho_util_URLLC"]) / 2
                )

            mean_rho = float(np.mean(rho_utils)) if rho_utils else 0.0
            V_rates = {}
            for sname in ["eMBB", "URLLC"]:
                arr = np.array(avg_T_steps[sname])
                V_rates[sname] = econ.sla.compute_violation_rate(arr, sname)

            # Update user fields
            for sname in ["eMBB", "URLLC"]:
                users = pool_mgr.get_active_users(sname)
                avg_t = float(np.mean(avg_T_steps[sname])) if avg_T_steps[sname] else 0.0
                for u in users:
                    u.T_act_avg = avg_t
                market.update_disconfirmation(users)
                churned = market.sample_churns(users, fees[sname], rng=rng)
                pool_mgr.churn(churned)

            # Profit & reward
            profit_result = econ.compute_profit(
                fees=fees, N_active=N_active,
                n_topups={"eMBB": 0, "URLLC": 0},
                topup_price=topup_price,
                V_rates=V_rates,
                mean_rho_util=mean_rho,
                avg_load=mean_rho,
            )
            reward = econ.compute_reward(profit_result["profit"])

            monthly_records.append({
                "month": month,
                "reward": reward,
                "profit": profit_result["profit"],
                "revenue": profit_result["revenue"],
                "cost_total": profit_result["cost_total"],
                "V_eMBB": V_rates["eMBB"],
                "V_URLLC": V_rates["URLLC"],
                "mean_rho": mean_rho,
                "N_eMBB": N_active["eMBB"],
                "N_URLLC": N_active["URLLC"],
                "rho_URLLC": rho_URLLC,
                "F_eMBB": fees["eMBB"],
                "F_URLLC": fees["URLLC"],
            })

        return monthly_records

    # ---------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------

    def test_finite_reward_seed_0(self):
        """All rewards finite with seed 0."""
        records = self._run_random_episode(seed=0, n_months=15)
        for r in records:
            self.assertTrue(
                np.isfinite(r["reward"]),
                f"Non-finite reward at month {r['month']}: {r['reward']}"
            )

    def test_finite_reward_seed_42(self):
        """All rewards finite with seed 42."""
        records = self._run_random_episode(seed=42, n_months=15)
        for r in records:
            self.assertTrue(np.isfinite(r["reward"]))

    def test_finite_reward_seed_999(self):
        """All rewards finite with seed 999."""
        records = self._run_random_episode(seed=999, n_months=15)
        for r in records:
            self.assertTrue(np.isfinite(r["reward"]))

    def test_finite_profit_all_seeds(self):
        """Profit must be finite across multiple seeds."""
        for seed in [0, 42, 123, 456, 789]:
            records = self._run_random_episode(seed=seed, n_months=10)
            for r in records:
                self.assertTrue(
                    np.isfinite(r["profit"]),
                    f"Non-finite profit at seed={seed} month={r['month']}"
                )

    def test_all_kpis_finite(self):
        """Every KPI must be finite."""
        records = self._run_random_episode(seed=7, n_months=10)
        for r in records:
            for key, val in r.items():
                if isinstance(val, (float, int)):
                    self.assertTrue(
                        np.isfinite(val),
                        f"Non-finite {key}={val} at month {r['month']}"
                    )

    def test_reward_bounded(self):
        """Reward must be within [-clip, clip]."""
        clip = self.cfg.get("economics", {}).get("reward_clip", 2.0)
        records = self._run_random_episode(seed=11, n_months=20)
        for r in records:
            self.assertGreaterEqual(r["reward"], -clip)
            self.assertLessEqual(r["reward"], clip)

    def test_sanitize_obs_handles_extremes(self):
        """sanitize_obs must handle NaN, Inf, extreme values."""
        obs = np.array([np.nan, np.inf, -np.inf, 1e20, -1e20, 0.5],
                       dtype=np.float32)
        sanitized = sanitize_obs(obs)
        self.assertTrue(np.all(np.isfinite(sanitized)))
        self.assertEqual(sanitized.dtype, np.float32)

    def test_safe_reward_handles_nan(self):
        """safe_reward must return finite value for NaN input."""
        r = safe_reward(float("nan"), penalty=0.0)
        self.assertTrue(np.isfinite(r))

    def test_validate_action_nan(self):
        """validate_action must flag NaN."""
        ok, violations = validate_action(np.array([np.nan, 0.0, 0.0]))
        self.assertFalse(ok)
        self.assertIn("action_nan_inf", violations)


if __name__ == "__main__":
    unittest.main()
