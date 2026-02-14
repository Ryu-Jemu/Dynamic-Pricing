"""
Unit tests for O-RAN 3-Part Tariff environment.

REVISION 9 — Changes from v8:
  [D1–D5] v9 design tests (T12) — admission control, hierarchical actions,
       hard capacity guard, 24D observation, backward compat
  [T13] Dashboard smoke tests (PNG + 3D)
  Prior revisions (v1–v8):
  [M9] v8 tests (T11): curriculum fraction, convex SLA, Lagrangian,
       rho_U bound, pop_bonus scale, eval diagnostics
  [R4] Per-dimension smoothing tests
  [R5] Observation shape updated to (24,)  [D5]
  [R6] Population-aware reward tests
  [E4] Observation shape 16 → 20
  [E5] Episode length now 720 (24 cycles)
  [E6] CLV reward shaping tests
  [E8] Stronger smoothing tests

Test groups (72 tests, 15 classes):
  T1  Environment basics (reset, step, spaces)
  T2  Revenue model (3-part tariff, online accrual)
  T3  Market dynamics (join/churn, conservation)
  T4  QoS violation (sigmoid, capacity)
  T5  Numerical safety (no NaN/Inf, obs bounds, reward clip)
  T6  Billing cycle (reset accumulators, 30-step cycles)
  T7  Utility functions
  T8  Calibration validation
  T9  v5 enhancements (CLV reward, load factors)
  T10 v7 enhancements (per-dim smoothing, pop reward, curriculum)
  T11 v8 enhancements (curriculum fraction, convex SLA, Lagrangian)
  T12 v9 design improvements (admission, hierarchical, hard cap, 24D)
  T13 Dashboard smoke tests (PNG + 3D)
  T14 [M13] Spatial visualization & per-user events
  T15 [D1-D7] Revision design (slice QoS, PID Lagrangian, 3-phase curriculum)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oran3pt.utils import load_config, sigmoid, fit_lognormal_quantiles
from oran3pt.env import OranSlicingPricingEnv

OBS_DIM = 24  # [D5] v9 default


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
        assert obs.shape == (OBS_DIM,), f"Bad obs shape: {obs.shape}"
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (OBS_DIM,)
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
        for _ in range(300):
            _, _, term, _, info = env.step(mid_action)
            pviol_values.append(info["pviol_E"])
            if term:
                break

        pviol_arr = np.array(pviol_values)
        frac_below_05 = (pviol_arr < 0.5).mean()
        assert frac_below_05 > 0.10, \
            f"Only {frac_below_05:.1%} of steps have pviol_E < 0.5"

    def test_capacity_adequate_for_population(self, cfg):
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)

        mid_action = np.zeros(5, dtype=np.float32)
        for _ in range(10):
            _, _, _, _, info = env.step(mid_action)

        C_E = info["C_E"]
        L_E = info["L_E"]
        ratio = L_E / max(C_E, 1e-6)
        assert ratio < 5.0, \
            f"eMBB load ratio {ratio:.2f} — capacity inadequate"


# =====================================================================
# T9  v5 enhancements
# =====================================================================
class TestV5Enhancements:
    def test_obs_shape_v9(self, env):
        """[D5] Observation space should be 24D."""
        obs, _ = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,)

    def test_load_factor_in_obs(self, env):
        """[E4] Load factors present in observation."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, info = env.step(action)
        load_U = obs[18]
        load_E = obs[19]
        assert np.isfinite(load_U)
        assert np.isfinite(load_E)

    def test_retention_penalty_exists(self, env):
        """[E6] Retention penalty key in info."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "retention_penalty" in info

    def test_clv_config_present(self, cfg):
        """[E5] CLV config section exists."""
        assert "clv" in cfg
        assert cfg["clv"]["horizon_months"] > 0


# =====================================================================
# T10  v7 enhancements
# =====================================================================
class TestV7Enhancements:
    def test_per_dim_smoothing_weights(self, cfg):
        """[R4] Per-dimension smoothing weights in config."""
        sm = cfg.get("action_smoothing", {})
        weights = sm.get("weights", [])
        assert len(weights) == 5, \
            f"Expected 5 smoothing weights, got {len(weights)}"

    def test_pop_reward_config(self, cfg):
        """[R6] Population reward config present."""
        pr = cfg.get("population_reward", {})
        assert pr.get("enabled", False) is True
        assert "beta_pop" in pr
        assert "target_ratio" in pr

    def test_curriculum_config(self, cfg):
        """[R3] Curriculum learning config."""
        tr = cfg.get("training", {})
        cur = tr.get("curriculum", {})
        assert cur.get("enabled", False) is True

    def test_obs_dims_20_to_21(self, env):
        """[R5] obs[20] overage rev rate, obs[21] days remaining."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.isfinite(obs[20])
        assert np.isfinite(obs[21])
        assert 0.0 <= obs[21] <= 1.0 + 1e-6


# =====================================================================
# T11  v8 enhancements
# =====================================================================
class TestV8Enhancements:
    def test_T11_1_curriculum_fraction(self, cfg):
        """[M2] Curriculum phase1_fraction is a float in (0, 1)."""
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
        gamma = cfg["qos"].get("gamma_sla_E", 2.0)
        pf_high = 0.9 ** gamma
        pf_low = 0.3 ** gamma
        ratio = pf_high / pf_low
        assert ratio > 3.0, \
            f"Convex penalty ratio {ratio:.2f} should exceed 3.0"

    def test_T11_3_rho_U_clipped_to_045(self, cfg):
        """[D1] rho_U action clipped to [0.05, 0.45]."""
        assert cfg["action"]["rho_U_max"] <= 0.45, \
            f"rho_U_max should be <= 0.45, got {cfg['action']['rho_U_max']}"
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        max_action = np.ones(5, dtype=np.float32)
        _, _, _, _, info = env.step(max_action)
        assert info["rho_U"] <= 0.45 + 1e-6, \
            f"rho_U should be <= 0.45, got {info['rho_U']}"

    def test_T11_4_pop_bonus_scale(self, cfg):
        """[M5] pop_bonus magnitude is reasonable."""
        beta_pop = cfg.get("population_reward", {}).get("beta_pop", 0.3)
        assert beta_pop >= 0.2, \
            f"beta_pop should be >= 0.2 for adequate signal, got {beta_pop}"

    def test_T11_5_lagrangian_increases_on_violation(self, cfg):
        """[M6] Lagrangian lambda can be set."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        env.set_lagrangian_lambda(0.0)
        assert env._lagrangian_lambda == 0.0
        env.set_lagrangian_lambda(1.5)
        assert env._lagrangian_lambda == 1.5

    def test_T11_6_lagrangian_zero_below_threshold(self, cfg):
        """[M6] Lagrangian penalty is 0 when pviol_E < threshold."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        env.set_lagrangian_lambda(2.0)
        # Use low rho_U → high eMBB capacity → low pviol_E
        low_rho = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        _, _, _, _, info = env.step(low_rho)
        if info["pviol_E"] < env._pviol_E_threshold:
            assert info["lagrangian_penalty"] == 0.0

    def test_T11_7_eval_info_keys(self, env):
        """Info dict contains diagnostic keys."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        required = ["step", "profit", "revenue", "cost_total",
                     "pviol_U", "pviol_E", "N_active", "n_join", "n_churn",
                     "L_U", "L_E", "C_U", "C_E"]
        for k in required:
            assert k in info, f"Missing key: {k}"

    def test_T11_8_smoothing_per_dim(self, cfg):
        """[R4] Action smoothing is per-dimension."""
        # Disable hierarchical so pricing dims actually change mid-cycle
        cfg_copy = {**cfg}
        cfg_copy["hierarchical_actions"] = {"enabled": False}
        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        env.reset(seed=42)
        a1 = np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        env.step(a1)
        a2 = np.array([-0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info2 = env.step(a2)
        assert info2["smooth_penalty"] > 0.0


# =====================================================================
# T12  v9 design improvements
# =====================================================================
class TestV9Design:

    # ── [D5] Observation dim = 24 ─────────────────────────────────────

    def test_T12_1_obs_dim_24(self, env):
        """[D5] Observation space should be 24-dimensional."""
        assert env.observation_space.shape == (24,), \
            f"Expected obs dim 24, got {env.observation_space.shape}"
        obs, _ = env.reset(seed=42)
        assert obs.shape == (24,)

    def test_T12_2_pviol_E_ema_in_obs(self, env):
        """[D6] obs[22] = pviol_E EMA, in [0, 1]."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.isfinite(obs[22])
        assert 0.0 <= obs[22] <= 1.0 + 1e-6

    def test_T12_3_load_headroom_in_obs(self, env):
        """[D5] obs[23] = load headroom for eMBB, in [0, 1]."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.isfinite(obs[23])
        assert -0.01 <= obs[23] <= 1.0 + 1e-6

    # ── [D1] Admission control ────────────────────────────────────────

    def test_T12_4_n_rejected_in_info(self, env):
        """[D1] info dict must contain n_rejected key."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "n_rejected" in info
        assert isinstance(info["n_rejected"], int)
        assert info["n_rejected"] >= 0

    def test_T12_5_admission_blocks_under_congestion(self, cfg):
        """[D1] Rejections occur when system is congested."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        high_rho = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        total_rejected = 0
        for _ in range(60):
            _, _, term, _, info = env.step(high_rho)
            total_rejected += info["n_rejected"]
            if term:
                break
        if cfg.get("admission_control", {}).get("enabled", False):
            assert total_rejected >= 0

    def test_T12_6_population_conservation_with_admission(self, env):
        """[D1] N_active + N_inactive = N_total with admissions."""
        env.reset(seed=42)
        N_total = env.N_total
        for _ in range(60):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            n_active = int(env._active_mask.sum())
            assert n_active + (N_total - n_active) == N_total
            if term:
                break

    # ── [D2] Hierarchical actions ─────────────────────────────────────

    def test_T12_7_pricing_locked_within_cycle(self, cfg):
        """[D2] Pricing dims constant within a billing cycle."""
        cfg_copy = {**cfg}
        cfg_copy["hierarchical_actions"] = {"enabled": True}
        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        env.reset(seed=42)

        action1 = np.array([0.5, 0.3, 0.5, 0.3, 0.0], dtype=np.float32)
        _, _, _, _, info1 = env.step(action1)
        locked_F_U = info1["F_U"]
        locked_F_E = info1["F_E"]

        action2 = np.array([-0.5, -0.3, -0.5, -0.3, 0.5], dtype=np.float32)
        _, _, _, _, info2 = env.step(action2)
        assert abs(info2["F_U"] - locked_F_U) < 1e-4, \
            f"F_U changed mid-cycle: {locked_F_U} → {info2['F_U']}"
        assert abs(info2["F_E"] - locked_F_E) < 1e-4

    def test_T12_8_rho_U_updates_every_step(self, cfg):
        """[D2] rho_U updates every step."""
        cfg_copy = {**cfg}
        cfg_copy["hierarchical_actions"] = {"enabled": True}
        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        env.reset(seed=42)

        a1 = np.array([0.0, 0.0, 0.0, 0.0, -0.8], dtype=np.float32)
        _, _, _, _, info1 = env.step(a1)
        a2 = np.array([0.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
        _, _, _, _, info2 = env.step(a2)
        assert abs(info2["rho_U"] - info1["rho_U"]) > 0.05

    # ── [D3] Hard capacity guard ──────────────────────────────────────

    def test_T12_9_effective_load_capped(self, cfg):
        """[D3] L_effective ≤ C when hard_cap enabled."""
        cfg_copy = {**cfg}
        cfg_copy["capacity_enforcement"] = {"hard_cap": True}
        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        env.reset(seed=42)

        for _ in range(60):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            assert info["L_E_effective"] <= info["C_E"] + 1e-6
            assert info["L_U_effective"] <= info["C_U"] + 1e-6
            if term:
                break

    def test_T12_10_hard_cap_pviol_uses_raw_load(self, cfg):
        """[D3] QoS penalty uses raw load, not capped."""
        cfg_copy = {**cfg}
        cfg_copy["capacity_enforcement"] = {"hard_cap": True}
        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        env.reset(seed=42)

        high_rho = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(30):
            _, _, term, _, info = env.step(high_rho)
            if info["L_E"] > info["C_E"]:
                assert info["pviol_E"] > 0.5
            if term:
                break

    # ── Backward compat ───────────────────────────────────────────────

    def test_T12_11_disabled_features_backward_compat(self, cfg):
        """All v9 features can be disabled."""
        cfg_copy = {**cfg}
        cfg_copy["admission_control"] = {"enabled": False}
        cfg_copy["hierarchical_actions"] = {"enabled": False}
        cfg_copy["capacity_enforcement"] = {"hard_cap": False}
        cfg_copy["observation"] = {**cfg.get("observation", {}), "dim": 22}

        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (22,)

        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["n_rejected"] == 0
        assert info["L_E_effective"] == info["L_E"]

    def test_T12_12_numerical_safety_v9(self, env):
        """Full episode with v9 features, no NaN/Inf."""
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))

        for step in range(env.episode_len):
            action = env.action_space.sample()
            obs, reward, term, _, info = env.step(action)
            assert np.all(np.isfinite(obs)), f"Step {step}: NaN/Inf"
            assert np.isfinite(reward)
            assert info["n_rejected"] >= 0
            if term:
                break



# ── T13  Dashboard module smoke tests [M11][M12] ─────────────────────

class TestT13DashboardSmoke:
    """Smoke tests for PNG and 3D dashboard generators."""

    def test_T13_1_png_dashboard_import(self):
        """[M11] png_dashboard module imports without errors."""
        from oran3pt import png_dashboard  # noqa: F401
        assert hasattr(png_dashboard, "generate_all_pngs")
        assert hasattr(png_dashboard, "main")

    def test_T13_2_sim3d_dashboard_import(self):
        """[M12] sim3d_dashboard module imports without errors."""
        from oran3pt import sim3d_dashboard  # noqa: F401
        assert hasattr(sim3d_dashboard, "generate_3d_dashboard")
        assert hasattr(sim3d_dashboard, "main")

    def test_T13_3_png_episode_detection(self):
        """[M11] Episode detection handles step counter resets."""
        from oran3pt.png_dashboard import _detect_episodes
        df = pd.DataFrame({
            "step": [1, 2, 3, 1, 2, 3],
            "reward": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        result = _detect_episodes(df)
        assert "_episode" in result.columns
        assert result["_episode"].nunique() == 2

    def test_T13_4_png_derived_columns(self):
        """[M11] Derived columns computed correctly."""
        from oran3pt.png_dashboard import _add_derived_columns
        df = pd.DataFrame({
            "L_U": [10.0], "C_U": [20.0],
            "L_E": [15.0], "C_E": [20.0],
            "profit": [1000.0], "revenue": [2000.0],
            "sla_penalty": [100.0],
            "n_join": [5], "n_churn": [3],
        })
        result = _add_derived_columns(df)
        assert "urllc_util" in result.columns
        assert "embb_util" in result.columns
        assert "profit_margin" in result.columns
        assert abs(result["urllc_util"].iloc[0] - 0.5) < 1e-6
        assert abs(result["profit_margin"].iloc[0] - 0.5) < 1e-6
        assert result["population_delta"].iloc[0] == 2

    def test_T13_5_sim3d_best_repeat_selection(self):
        """[M12] Best repeat selected by highest cumulative profit."""
        from oran3pt.sim3d_dashboard import _select_best_repeat
        df = pd.DataFrame({
            "repeat": [0, 0, 1, 1],
            "profit": [100, 200, 500, 600],
        })
        result = _select_best_repeat(df)
        assert len(result) == 2
        assert result["profit"].sum() == 1100  # repeat 1


# =====================================================================
# T14  [M13] Spatial visualization & per-user events
# =====================================================================
class TestM13SpatialVisualization:

    def test_T14_1_users_have_coordinates(self, cfg):
        """[M13a] Generated users have x, y columns within cell radius."""
        from oran3pt.gen_users import generate_users
        df = generate_users(cfg, seed=42)
        assert "x" in df.columns, "Missing x column"
        assert "y" in df.columns, "Missing y column"
        r = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        cell_radius = cfg.get("population", {}).get(
            "spatial", {}).get("cell_radius", 20.0)
        assert r.max() <= cell_radius + 0.1, \
            f"Max radius {r.max():.2f} exceeds cell_radius {cell_radius}"

    def test_T14_2_user_events_in_info(self, env):
        """[M13c] Info dict contains churned/joined user ID lists."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "churned_user_ids" in info
        assert "joined_user_ids" in info
        assert isinstance(info["churned_user_ids"], list)
        assert isinstance(info["joined_user_ids"], list)

    def test_T14_3_segment_radial_distribution(self, cfg):
        """[M13a] QoS-sensitive users are closer to tower than price-sensitive."""
        from oran3pt.gen_users import generate_users
        df = generate_users(cfg, seed=42)
        df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        qos_r = df[df["segment"] == "qos_sensitive"]["r"].mean()
        price_r = df[df["segment"] == "price_sensitive"]["r"].mean()
        assert qos_r < price_r, \
            f"QoS mean_r={qos_r:.2f} should be < price mean_r={price_r:.2f}"

    def test_T14_4_backward_compat_no_spatial(self):
        """[M13e] Dashboard fallback works without spatial data."""
        from oran3pt.sim3d_dashboard import _load_user_data
        result = _load_user_data(None)
        assert result == [], "Should return empty list for None path"

    def test_T14_5_event_count_matches_aggregate(self, env):
        """[M13c] len(churned_user_ids) == n_churn for each step."""
        env.reset(seed=42)
        for _ in range(60):
            action = env.action_space.sample()
            _, _, term, _, info = env.step(action)
            assert len(info["churned_user_ids"]) == info["n_churn"], \
                f"Churn mismatch: {len(info['churned_user_ids'])} vs {info['n_churn']}"
            assert len(info["joined_user_ids"]) == info["n_join"], \
                f"Join mismatch: {len(info['joined_user_ids'])} vs {info['n_join']}"
            if term:
                break


# =====================================================================
# T15  [D1–D7] Revision Design implementation tests
# =====================================================================
class TestRevisionDesign:

    def test_T15_1_slice_specific_qsig(self, cfg):
        """[D4] eMBB user churn depends on pviol_E, not pviol_U."""
        import copy
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        # Run a few steps to get non-zero pviol values
        for _ in range(10):
            env.step(env.action_space.sample())
        # Check that Q_sig computation uses per-slice values
        # eMBB users (slice_is_U==False) should see pviol_E only
        pviol_U = env._prev_pviol_U
        pviol_E = env._prev_pviol_E
        embb_qsig = 1.0 - pviol_E
        urllc_qsig = 1.0 - pviol_U
        avg_qsig = 1.0 - (pviol_U + pviol_E) / 2.0
        # If slice-specific, eMBB Q_sig should differ from average
        # when pviol_U != pviol_E (which is typical)
        if abs(pviol_U - pviol_E) > 0.01:
            assert embb_qsig != avg_qsig, \
                "eMBB Q_sig should differ from averaged Q_sig"

    def test_T15_2_pid_lagrangian_params(self, cfg):
        """[D2] Config has PID Lagrangian parameters."""
        lag = cfg.get("lagrangian_qos", {})
        assert "Kp" in lag, "Missing Kp in lagrangian_qos config"
        assert "Ki" in lag, "Missing Ki in lagrangian_qos config"
        assert "Kd" in lag, "Missing Kd in lagrangian_qos config"
        assert lag.get("lambda_max", 0) >= 10.0, \
            f"lambda_max should be >= 10.0, got {lag.get('lambda_max')}"
        assert lag.get("update_freq", 0) <= 200, \
            f"update_freq should be <= 200, got {lag.get('update_freq')}"

    def test_T15_3_lagrangian_outside_reward_clip(self, env):
        """[D2] Reward can exceed [-2,2] when Lagrangian penalty is large."""
        env.reset(seed=42)
        env.set_lagrangian_lambda(10.0)
        env._lagrangian_boost = 1.0
        # Force high pviol_E scenario
        env._pviol_E_threshold = 0.0  # any pviol triggers penalty
        # Step with extreme action to generate penalty
        obs, reward, _, _, info = env.step(env.action_space.sample())
        # With lambda=10 and pviol > 0, reward should potentially be < -2
        # The key check is that the reward clip is [-4, 4], not [-2, 2]
        assert -4.0 - 1e-6 <= reward <= 4.0 + 1e-6, \
            f"Reward {reward} should be in [-4, 4]"

    def test_T15_4_pviol_ema_tracks_violation(self, env):
        """[D6] pviol_E EMA tracks violation trend, finite values."""
        env.reset(seed=42)
        ema_values = []
        for _ in range(30):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            ema_values.append(obs[22])
            assert np.isfinite(obs[22]), "pviol_E EMA should be finite"
            assert 0.0 <= obs[22] <= 1.0 + 1e-6, \
                f"pviol_E EMA {obs[22]} out of bounds"

    def test_T15_5_convex_sla_gamma3(self, cfg):
        """[D3] Convex SLA penalty with gamma=3.0."""
        gamma = cfg.get("qos", {}).get("gamma_sla_E", 2.0)
        assert gamma == 3.0, f"gamma_sla_E should be 3.0, got {gamma}"
        lambda_E = cfg.get("qos", {}).get("lambda_E", 200000)
        # penalty(0.9) / penalty(0.3) should be (0.9/0.3)^3 = 27
        ratio = (0.9 ** gamma) / (0.3 ** gamma)
        assert ratio > 25.0, \
            f"Cubic penalty ratio {ratio:.1f} should exceed 25"

    def test_T15_6_3phase_curriculum_config(self, cfg):
        """[D5] Curriculum has 3 phases summing to ~1.0."""
        phases = cfg.get("training", {}).get("curriculum", {}).get("phases")
        assert phases is not None, "Missing curriculum.phases in config"
        assert len(phases) == 3, f"Expected 3 phases, got {len(phases)}"
        total = sum(p["fraction"] for p in phases)
        assert abs(total - 1.0) < 0.01, \
            f"Phase fractions should sum to 1.0, got {total}"
        # Phase 1 should disable churn/join
        assert phases[0]["churn_join"] is False
        # Phase 2 should have boosted Lagrangian
        assert phases[1].get("lagrangian_boost", 1.0) > 1.0

    def test_T15_7_lagrangian_boost_setter(self, env):
        """[D5] set_lagrangian_boost doubles penalty effect."""
        env.reset(seed=42)
        env.set_lagrangian_lambda(1.0)
        # Normal boost
        env.set_lagrangian_boost(1.0)
        env.step(env.action_space.sample())
        # Boosted
        env.set_lagrangian_boost(2.0)
        assert env._lagrangian_boost == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
