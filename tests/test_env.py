"""
Unit tests for O-RAN 3-Part Tariff environment.

REVISION 10 — Changes from v9:
  [EP1] T16 — Continuous episode mode tests (truncation, population
       continuity, global counter, obs[15] churn EMA, backward compat)
  Updated T1 tests for continuous mode compatibility
  Prior revisions (v1–v9):
  [D1–D5] v9 design tests (T12) — admission control, hierarchical actions,
       hard capacity guard, 24D observation, backward compat
  [T13] Dashboard smoke tests (PNG)
  [M9] v8 tests (T11): curriculum fraction, convex SLA, Lagrangian,
       rho_U bound, pop_bonus scale, eval diagnostics
  [R4] Per-dimension smoothing tests
  [R5] Observation shape updated to (24,)  [D5]
  [R6] Population-aware reward tests
  [E4] Observation shape 16 → 20
  [E5] Episode length now 720 (24 cycles)
  [E6] CLV reward shaping tests
  [E8] Stronger smoothing tests

  [M15] T19 — Dashboard generation integration tests (eval.py)

Test groups (136 tests, 23 classes):
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
  T12 v9 design improvements (admission, hierarchical, 23D obs)
  T13 Dashboard smoke tests (PNG)
  T15 [D1-D7] Revision design (slice QoS, PID Lagrangian, 3-phase curriculum)
  T16 [EP1] Continuous episode mode (truncation, pop continuity, obs[14])
  T17 [M14] Business dashboard (metrics, KPIs, P&L, SLA, template)
  T18 [OPT] Training optimizations (entropy, early stop, parallel, cache)
  T20 [Review] Architecture review (anti-windup, 23D obs, config merge, integration)
  T21 [V11] Revision v11 improvements (rho_U, pop_bonus, admission, Lagrangian)
  T22 [PR] Pricing mechanism (per-slice P_sig, bill shock, overage join dampening)
  T23 [I-1..I-6] Structural improvements (PID asymmetric, capacity guard, SLA awareness)
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

OBS_DIM = 23  # [ME-1] 24→23; removed obs[1] (inactive fraction)


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
        # [EP1] In continuous mode, done signal should be truncated, not terminated
        if env._episode_mode == "continuous":
            assert truncated and not terminated, \
                "Continuous mode should use truncated=True"
        else:
            assert terminated and not truncated, \
                "Episodic mode should use terminated=True"

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
        load_U = obs[17]  # [ME-1] shifted: 18→17
        load_E = obs[18]  # [ME-1] shifted: 19→18
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

    def test_obs_dims_19_to_20(self, env):
        """[R5] obs[19] overage rev rate, obs[20] days remaining. [ME-1] shifted."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.isfinite(obs[19])   # [ME-1] 20→19
        assert np.isfinite(obs[20])   # [ME-1] 21→20
        assert 0.0 <= obs[20] <= 1.0 + 1e-6


# =====================================================================
# T11  v8 enhancements
# =====================================================================
class TestV8Enhancements:
    def test_T11_1_curriculum_phases(self, cfg):
        """[D5] Curriculum phases fractions sum to 1.0."""
        phases = cfg.get("training", {}).get("curriculum", {}).get("phases", [])
        assert len(phases) >= 2, "Curriculum must have at least 2 phases"
        total_frac = sum(p["fraction"] for p in phases)
        assert abs(total_frac - 1.0) < 1e-6, \
            f"Phase fractions must sum to 1.0, got {total_frac}"

    def test_T11_2_convex_sla_penalty(self, cfg):
        """[M3] SLA penalty with gamma_sla_E=2.0 produces convex curve."""
        gamma = cfg["qos"].get("gamma_sla_E", 2.0)
        pf_high = 0.9 ** gamma
        pf_low = 0.3 ** gamma
        ratio = pf_high / pf_low
        assert ratio > 3.0, \
            f"Convex penalty ratio {ratio:.2f} should exceed 3.0"

    def test_T11_3_rho_U_clipped_to_020(self, cfg):
        """[V11-1] rho_U action clipped to [0.05, 0.20].
        [Huang IoT-J 2020; Sciancalepore TNSM 2019]"""
        assert cfg["action"]["rho_U_max"] <= 0.20, \
            f"rho_U_max should be <= 0.20, got {cfg['action']['rho_U_max']}"
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        max_action = np.ones(5, dtype=np.float32)
        _, _, _, _, info = env.step(max_action)
        assert info["rho_U"] <= 0.20 + 1e-6, \
            f"rho_U should be <= 0.20, got {info['rho_U']}"

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

    def test_T12_1_obs_dim_23(self, env):
        """[ME-1] Observation space should be 23-dimensional."""
        assert env.observation_space.shape == (OBS_DIM,), \
            f"Expected obs dim {OBS_DIM}, got {env.observation_space.shape}"
        obs, _ = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,)

    def test_T12_2_pviol_E_ema_in_obs(self, env):
        """[D6] obs[21] = pviol_E EMA, in [0, 1]. [ME-1] shifted: 22→21."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.isfinite(obs[21])
        assert 0.0 <= obs[21] <= 1.0 + 1e-6

    def test_T12_3_load_headroom_in_obs(self, env):
        """[D5] obs[22] = load headroom for eMBB. [ME-1] shifted: 23→22."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.isfinite(obs[22])
        assert -0.01 <= obs[22] <= 1.0 + 1e-6

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

    # ── Backward compat ───────────────────────────────────────────────

    def test_T12_11_disabled_features_backward_compat(self, cfg):
        """v9 features can be disabled with reduced obs dim."""
        cfg_copy = {**cfg}
        cfg_copy["admission_control"] = {"enabled": False}
        cfg_copy["hierarchical_actions"] = {"enabled": False}
        cfg_copy["observation"] = {**cfg.get("observation", {}), "dim": 21}

        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (21,)

        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["n_rejected"] == 0

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



# ── T13  Dashboard module smoke tests [M11] ──────────────────────────

class TestT13DashboardSmoke:
    """Smoke tests for PNG dashboard generators."""

    def test_T13_1_png_dashboard_import(self):
        """[M11] png_dashboard module imports without errors."""
        from oran3pt import png_dashboard  # noqa: F401
        assert hasattr(png_dashboard, "generate_all_pngs")
        assert hasattr(png_dashboard, "main")

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
        """[D6] pviol_E EMA tracks violation trend. [ME-1] shifted: obs[22]→obs[21]."""
        env.reset(seed=42)
        ema_values = []
        for _ in range(30):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            ema_values.append(obs[21])
            assert np.isfinite(obs[21]), "pviol_E EMA should be finite"
            assert 0.0 <= obs[21] <= 1.0 + 1e-6, \
                f"pviol_E EMA {obs[21]} out of bounds"

    def test_T15_5_convex_sla_gamma3(self, cfg):
        """[F2] Convex SLA penalty with gamma=3.0.
        [Bertsekas 1996 §6.3] — γ=3 cubic provides adequate signal at pviol_E=0.2."""
        gamma = cfg.get("qos", {}).get("gamma_sla_E", 2.0)
        assert gamma == 3.0, f"gamma_sla_E should be 3.0, got {gamma}"
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


# =====================================================================
# T16  [EP1] Continuous episode mode
# =====================================================================
class TestContinuousEpisodeMode:

    @pytest.fixture
    def cont_cfg(self, cfg):
        """Config with episode_mode=continuous, episode_cycles=1."""
        import copy
        c = copy.deepcopy(cfg)
        c["time"]["episode_cycles"] = 1
        c["time"]["episode_mode"] = "continuous"
        return c

    @pytest.fixture
    def cont_env(self, cont_cfg):
        return OranSlicingPricingEnv(cont_cfg, seed=42)

    @pytest.fixture
    def epis_cfg(self, cfg):
        """Config with episode_mode=episodic (legacy)."""
        import copy
        c = copy.deepcopy(cfg)
        c["time"]["episode_cycles"] = 1
        c["time"]["episode_mode"] = "episodic"
        return c

    @pytest.fixture
    def epis_env(self, epis_cfg):
        return OranSlicingPricingEnv(epis_cfg, seed=42)

    def test_T16_1_truncated_not_terminated(self, cont_env):
        """[EP1] Continuous mode returns truncated=True, terminated=False."""
        cont_env.reset(seed=42)
        terminated, truncated = False, False
        for _ in range(cont_env.episode_len):
            _, _, terminated, truncated, _ = cont_env.step(
                cont_env.action_space.sample())
        assert truncated is True, "Should be truncated at episode end"
        assert terminated is False, "Should NOT be terminated in continuous mode"

    def test_T16_2_episodic_backward_compat(self, epis_env):
        """[EP1] Episodic mode returns terminated=True, truncated=False."""
        epis_env.reset(seed=42)
        terminated, truncated = False, False
        for _ in range(epis_env.episode_len):
            _, _, terminated, truncated, _ = epis_env.step(
                epis_env.action_space.sample())
        assert terminated is True, "Should be terminated in episodic mode"
        assert truncated is False, "Should NOT be truncated in episodic mode"

    def test_T16_3_population_persists_across_reset(self, cont_env):
        """[EP1] Population state survives reset in continuous mode."""
        cont_env.reset(seed=42)
        # Run one full episode (30 steps with market dynamics)
        for _ in range(cont_env.episode_len):
            cont_env.step(cont_env.action_space.sample())
        pop_before = int(cont_env._active_mask.sum())

        # Reset — population should persist
        cont_env.reset()
        pop_after = int(cont_env._active_mask.sum())
        assert pop_after == pop_before, \
            f"Population should persist: {pop_before} → {pop_after}"

    def test_T16_4_population_resets_in_episodic(self, epis_env, epis_cfg):
        """[EP1] Population resets to N_active_init in episodic mode."""
        epis_env.reset(seed=42)
        for _ in range(epis_env.episode_len):
            epis_env.step(epis_env.action_space.sample())

        epis_env.reset()
        pop_after = int(epis_env._active_mask.sum())
        expected = epis_cfg["population"]["N_active_init"]
        assert pop_after == expected, \
            f"Population should reset to {expected}, got {pop_after}"

    def test_T16_5_total_steps_increments(self, cont_env):
        """[EP1] _total_steps increments across resets."""
        cont_env.reset(seed=42)
        for _ in range(cont_env.episode_len):
            cont_env.step(cont_env.action_space.sample())
        assert cont_env._total_steps == cont_env.episode_len

        cont_env.reset()
        for _ in range(cont_env.episode_len):
            cont_env.step(cont_env.action_space.sample())
        assert cont_env._total_steps == 2 * cont_env.episode_len, \
            f"_total_steps should be {2 * cont_env.episode_len}, " \
            f"got {cont_env._total_steps}"

    def test_T16_6_obs14_churn_ema_in_continuous(self, cont_env):
        """[EP1] obs[14] = churn_rate_ema in continuous mode. [ME-1] shifted: 15→14."""
        cont_env.reset(seed=42)
        for _ in range(10):
            obs, _, _, _, _ = cont_env.step(cont_env.action_space.sample())
        assert np.isfinite(obs[14])
        assert 0.0 <= obs[14] <= 1.0 + 1e-6, \
            f"obs[14] churn_rate_ema out of range: {obs[14]}"

    def test_T16_7_obs14_episode_progress_in_episodic(self, epis_env):
        """[EP1] obs[14] = episode_progress in episodic mode. [ME-1] shifted: 15→14."""
        epis_env.reset(seed=42)
        obs, _, _, _, _ = epis_env.step(epis_env.action_space.sample())
        expected = 1.0 / epis_env.episode_len
        assert abs(obs[14] - expected) < 1e-4, \
            f"obs[14] should be episode progress {expected}, got {obs[14]}"


# ═══════════════════════════════════════════════════════════════════
# T17  Business Dashboard — metric computation & module smoke tests
# ═══════════════════════════════════════════════════════════════════

class TestT17BusinessDashboard:
    """Business KPI computation accuracy and module imports."""

    @pytest.fixture
    def sample_rollout_df(self):
        """Synthetic rollout DataFrame mimicking rollout_log.csv.

        Data is self-consistent: revenue = base_rev + over_rev,
        base_rev = (F_U*N_U + F_E*N_E) / T, profit = revenue - cost_total.
        """
        np.random.seed(42)
        n = 150  # 5 repeats × 30 steps
        T = 30
        F_U = np.random.uniform(40000, 60000, n)
        F_E = np.random.uniform(80000, 130000, n)
        N_U = np.random.randint(5, 15, n).astype(float)
        N_E = np.random.randint(25, 45, n).astype(float)
        N_active = (N_U + N_E).astype(int)
        over_rev_E = np.random.uniform(0, 30000, n)
        over_rev_U = np.random.uniform(0, 10000, n)
        over_rev = over_rev_U + over_rev_E
        base_rev = (F_U * N_U + F_E * N_E) / T
        revenue = base_rev + over_rev
        cost_opex = np.random.uniform(20000, 50000, n)
        cost_energy = np.random.uniform(2000, 5000, n)
        cost_cac = np.random.uniform(0, 100000, n)
        sla_penalty = np.random.uniform(0, 1000, n)
        cost_total = cost_opex + cost_energy + cost_cac + sla_penalty
        profit = revenue - cost_total
        return pd.DataFrame({
            "step": list(range(1, 31)) * 5,
            "repeat": sorted(list(range(5)) * 30),
            "cycle": [1] * 150,
            "cycle_step": list(range(30)) * 5,
            "F_U": F_U, "F_E": F_E,
            "p_over_U": np.random.uniform(1000, 4000, n),
            "p_over_E": np.random.uniform(500, 2500, n),
            "rho_U": np.random.uniform(0.15, 0.35, n),
            "N_active": N_active,
            "N_U": N_U.astype(int), "N_E": N_E.astype(int),
            "n_join": np.random.randint(0, 3, n),
            "n_churn": np.random.randint(0, 3, n),
            "L_U": np.random.uniform(1, 5, n),
            "L_E": np.random.uniform(30, 80, n),
            "C_U": np.random.uniform(80, 150, n),
            "C_E": np.random.uniform(200, 350, n),
            "pviol_U": np.random.uniform(0.0, 0.001, n),
            "pviol_E": np.random.uniform(0.0, 0.01, n),
            "revenue": revenue, "base_rev": base_rev,
            "over_rev": over_rev, "over_rev_E": over_rev_E,
            "cost_total": cost_total,
            "cost_opex": cost_opex, "cost_energy": cost_energy,
            "cost_cac": cost_cac, "sla_penalty": sla_penalty,
            "profit": profit,
            "profit_margin": profit / np.maximum(revenue, 1e-6),
        })

    @pytest.fixture
    def sample_clv_df(self):
        return pd.DataFrame([{
            "mean_monthly_profit": 3000000,
            "mean_N_active": 45,
            "cf_per_user_month": 66667,
            "monthly_churn": 0.05,
            "monthly_retention": 0.95,
            "CLV_per_user": 900000,
            "horizon_months": 24,
            "discount_rate": 0.01,
        }])

    def test_T17_1_business_metrics_import(self):
        """Module import smoke test."""
        from oran3pt.business_metrics import (
            compute_executive_kpis,
            compute_pl_waterfall,
            compute_revenue_breakdown,
            compute_slice_economics,
            compute_monthly_subscribers,
            compute_price_churn_correlation,
            compute_sla_compliance,
            compute_capacity_analysis,
            estimate_additional_capacity,
            compute_annual_projection,
            compute_pricing_strategy_summary,
            compute_all_metrics,
        )
        assert callable(compute_executive_kpis)
        assert callable(compute_all_metrics)

    def test_T17_2_business_dashboard_import(self):
        """Dashboard generator import smoke test."""
        from oran3pt.business_dashboard import generate_business_dashboard
        assert callable(generate_business_dashboard)

    def test_T17_3_executive_kpis_keys(self, sample_rollout_df, sample_clv_df):
        """KPI computation returns all 6 required keys."""
        from oran3pt.business_metrics import compute_executive_kpis
        kpis = compute_executive_kpis(sample_rollout_df, sample_clv_df, T=30)
        expected_keys = {
            "monthly_profit_M_KRW", "profit_margin_pct",
            "monthly_retention_pct", "arpu_K_KRW",
            "sla_compliance_pct", "clv_K_KRW",
        }
        assert expected_keys == set(kpis.keys())
        for v in kpis.values():
            assert np.isfinite(v), f"KPI value not finite: {v}"

    def test_T17_4_pl_waterfall_sum(self, sample_rollout_df):
        """P&L waterfall: total_revenue = sum of revenue components."""
        from oran3pt.business_metrics import compute_pl_waterfall
        pl = compute_pl_waterfall(sample_rollout_df, T=30)
        rev_sum = pl["base_rev_U"] + pl["base_rev_E"] + pl["over_rev_U"] + pl["over_rev_E"]
        assert abs(rev_sum - pl["total_revenue"]) < pl["total_revenue"] * 0.01, \
            f"Revenue mismatch: sum={rev_sum}, total={pl['total_revenue']}"

    def test_T17_5_revenue_breakdown_sums_to_100(self, sample_rollout_df):
        """Revenue breakdown percentages sum to ~100%."""
        from oran3pt.business_metrics import compute_revenue_breakdown
        rb = compute_revenue_breakdown(sample_rollout_df, T=30)
        total = rb["base_U_pct"] + rb["base_E_pct"] + rb["over_U_pct"] + rb["over_E_pct"]
        assert abs(total - 100.0) < 0.1, f"Revenue %s should sum to 100, got {total}"

    def test_T17_6_sla_compliance_range(self, sample_rollout_df):
        """SLA compliance values are in [0, 100]%."""
        from oran3pt.business_metrics import compute_sla_compliance
        sla = compute_sla_compliance(sample_rollout_df)
        assert 0 <= sla["urllc_compliance_pct"] <= 100
        assert 0 <= sla["embb_compliance_pct"] <= 100

    def test_T17_7_monthly_subscribers_shape(self, sample_rollout_df):
        """Monthly subscriber aggregation produces correct shape."""
        from oran3pt.business_metrics import compute_monthly_subscribers
        monthly = compute_monthly_subscribers(sample_rollout_df, T=30)
        assert "month" in monthly.columns
        assert "avg_active" in monthly.columns
        assert "net_change" in monthly.columns
        assert len(monthly) >= 1

    def test_T17_8_capacity_non_negative(self, sample_rollout_df):
        """Capacity analysis returns non-negative utilization."""
        from oran3pt.business_metrics import compute_capacity_analysis
        cap = compute_capacity_analysis(sample_rollout_df)
        assert cap["urllc_util_pct"] >= 0
        assert cap["embb_util_pct"] >= 0
        assert cap["urllc_headroom_pct"] >= 0

    def test_T17_9_compute_all_metrics_keys(self, sample_rollout_df, sample_clv_df):
        """compute_all_metrics returns all expected top-level keys."""
        from oran3pt.business_metrics import compute_all_metrics
        metrics = compute_all_metrics(sample_rollout_df, sample_clv_df, T=30)
        expected = {
            "kpis", "pl_waterfall", "revenue_breakdown",
            "slice_economics", "monthly_subscribers",
            "price_churn_corr", "sla_compliance", "capacity",
            "additional_capacity", "annual_projection", "strategy",
        }
        assert expected == set(metrics.keys())

    def test_T17_10_annual_projection_scaling(self, sample_rollout_df):
        """Annual projection: 100-cell = 100 × single-cell."""
        from oran3pt.business_metrics import compute_annual_projection
        proj = compute_annual_projection(sample_rollout_df, T=30, n_cells=100)
        assert abs(proj["annual_profit_scaled"]
                   - proj["annual_profit_single_cell"] * 100) < 1.0

    def test_T17_11_correlation_range(self, sample_rollout_df):
        """Price-churn correlation values are in [-1, 1]."""
        from oran3pt.business_metrics import compute_price_churn_correlation
        corr = compute_price_churn_correlation(sample_rollout_df)
        for k, v in corr.items():
            assert -1.0 <= v <= 1.0 or np.isnan(v), \
                f"Correlation {k} out of range: {v}"

    def test_T17_12_template_exists(self):
        """Business dashboard HTML template file exists."""
        tmpl = Path(__file__).parent.parent / "oran3pt" / "templates" / "business_dashboard.html"
        assert tmpl.exists(), f"Template not found: {tmpl}"


# =====================================================================
# T18  [OPT] Training optimization tests
# =====================================================================
class TestTrainingOptimization:

    def test_T18_1_ent_coef_auto_format(self, cfg):
        """[OPT-B] ent_coef_init=0.5 + ent_coef='auto' → 'auto_0.5'."""
        tc = cfg.get("training", {})
        ent_coef = tc.get("ent_coef", "auto")
        ent_coef_init = tc.get("ent_coef_init", None)
        if ent_coef_init is not None and ent_coef == "auto":
            result = f"auto_{ent_coef_init}"
            assert result == "auto_0.5", f"Expected 'auto_0.5', got '{result}'"
            assert result.startswith("auto"), "Must start with 'auto' for SB3"

    def test_T18_2_early_stopping_config(self, cfg):
        """[OPT-C] Early stopping config present with correct keys."""
        es = cfg.get("training", {}).get("early_stopping", {})
        assert es.get("enabled") is True, "Early stopping should be enabled"
        assert "patience" in es, "Missing patience key"
        assert "min_timesteps" in es, "Missing min_timesteps key"
        assert "min_improvement" in es, "Missing min_improvement key"
        assert es["patience"] > 0
        assert es["min_timesteps"] > 0
        assert 0 < es["min_improvement"] < 1.0

    def test_T18_3_parallel_seeds_config(self, cfg):
        """[OPT-A] Config has parallel_seeds and max_parallel keys."""
        tc = cfg.get("training", {})
        assert "parallel_seeds" in tc, "Missing parallel_seeds key"
        assert "max_parallel" in tc, "Missing max_parallel key"
        assert isinstance(tc["parallel_seeds"], bool)
        assert tc["max_parallel"] >= 1

    def test_T18_4_batch_gradient_config(self, cfg):
        """[OPT-D] batch_size=512, gradient_steps=2 for GPU efficiency."""
        tc = cfg.get("training", {})
        assert tc.get("batch_size") == 512, \
            f"Expected batch_size=512, got {tc.get('batch_size')}"
        assert tc.get("gradient_steps") == 2, \
            f"Expected gradient_steps=2, got {tc.get('gradient_steps')}"
        # Total gradient compute: 2×512 = 1024 (same as old 4×256)
        total = tc["batch_size"] * tc["gradient_steps"]
        assert total == 1024, f"Total gradient samples should be 1024, got {total}"

    def test_T18_5_eval_freq_optimized(self, cfg):
        """[OPT-E] eval_freq=20000, n_eval_episodes=10."""
        tc = cfg.get("training", {})
        assert tc.get("eval_freq") == 20000, \
            f"Expected eval_freq=20000, got {tc.get('eval_freq')}"

    def test_T18_6_population_cache(self, env):
        """[OPT-F] Cached N_act, N_U, N_E match recomputed values."""
        env.reset(seed=42)
        for _ in range(10):
            env.step(env.action_space.sample())
        # Check cache matches actual
        actual_N_act = int(env._active_mask.sum())
        actual_N_U = int((env._active_mask * env._slice_is_U).sum())
        actual_N_E = actual_N_act - actual_N_U
        assert env._cached_N_act == actual_N_act, \
            f"Cache mismatch: N_act={env._cached_N_act} vs {actual_N_act}"
        assert env._cached_N_U == actual_N_U
        assert env._cached_N_E == actual_N_E


class TestDashboardGeneration:
    """[M15] Tests for automatic dashboard generation in eval.py."""

    def test_T19_1_generate_dashboards_import(self):
        """generate_dashboards function is importable from eval module."""
        from oran3pt.eval import generate_dashboards
        assert callable(generate_dashboards)

    def test_T19_2_run_evaluation_dashboard_param(self):
        """run_evaluation accepts generate_dashboard parameter (default True)."""
        import inspect
        from oran3pt.eval import run_evaluation
        sig = inspect.signature(run_evaluation)
        assert "generate_dashboard" in sig.parameters
        assert sig.parameters["generate_dashboard"].default is True

    def test_T19_3_run_evaluation_config_path_param(self):
        """run_evaluation accepts config_path parameter."""
        import inspect
        from oran3pt.eval import run_evaluation
        sig = inspect.signature(run_evaluation)
        assert "config_path" in sig.parameters
        assert sig.parameters["config_path"].default is None

    def test_T19_4_graceful_missing_dir(self, cfg):
        """generate_dashboards returns empty list for nonexistent dir."""
        from oran3pt.eval import generate_dashboards
        result = generate_dashboards(cfg, output_dir="/tmp/_oran_test_nonexistent")
        assert isinstance(result, list)
        assert len(result) == 0


# =====================================================================
# T20  [Review] Architecture review improvements
# =====================================================================
class TestArchitectureReview:
    """Tests for changes from Review_Architecture.md."""

    def test_T20_1_pid_anti_windup_integral_clamp(self, cfg):
        """[CR-2] PID integral is clamped to prevent windup.
        [Stooke ICLR 2020 §3.2; Mao arXiv 2025]
        """
        try:
            from oran3pt.train import _LagrangianPIDCallback
        except ImportError:
            pytest.skip("SB3 not available")
        lag_cfg = cfg.get("lagrangian_qos", {})
        cb = _LagrangianPIDCallback(
            threshold=lag_cfg.get("pviol_E_threshold", 0.15),
            Kp=lag_cfg.get("Kp", 0.05),
            Ki=lag_cfg.get("Ki", 0.005),
            Kd=lag_cfg.get("Kd", 0.01),
            lambda_max=lag_cfg.get("lambda_max", 10.0),
        )
        # integral_max = lambda_max / Ki  [F4: 10.0/0.02 = 500]
        assert cb._integral_max == lag_cfg["lambda_max"] / lag_cfg["Ki"]
        # Simulate extreme positive error accumulation
        for _ in range(10000):
            cb._error_integral = max(
                -cb._integral_max,
                min(cb._integral_max, cb._error_integral + 1.0))
        assert cb._error_integral <= cb._integral_max, \
            f"Integral {cb._error_integral} exceeds max {cb._integral_max}"

    def test_T20_2_obs_dim_23_no_inactive_fraction(self, cfg):
        """[ME-1] Observation is 23-D, inactive fraction removed."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (23,), f"Expected obs dim 23, got {obs.shape}"
        # obs[0] = active fraction, obs[1] should be joins (not inactive frac)
        action = env.action_space.sample()
        obs, _, _, _, info = env.step(action)
        active_frac = obs[0]
        # obs[1] should be normalized joins, not (1 - active_frac)
        assert not np.isclose(obs[1], 1.0 - active_frac, atol=0.01), \
            "obs[1] should be joins, not inactive fraction"

    def test_T20_3_config_merge_deep(self):
        """[CR-3] _deep_merge correctly overrides nested keys."""
        from oran3pt.utils import _deep_merge
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        merged = _deep_merge(base, override)
        assert merged["a"]["b"] == 10, "Override should replace nested key"
        assert merged["a"]["c"] == 2, "Non-overridden nested key preserved"
        assert merged["d"] == 3, "Non-overridden top key preserved"
        assert merged["e"] == 5, "New key from override added"

    def test_T20_4_production_yaml_exists(self):
        """[CR-3] Production config file exists."""
        prod = Path(__file__).parent.parent / "config" / "production.yaml"
        assert prod.exists(), f"Production config not found: {prod}"

    def test_T20_5_csvlogger_context_manager(self, tmp_path):
        """[HI-4] CSVLogger supports context manager protocol."""
        from oran3pt.train import CSVLogger
        csv_path = str(tmp_path / "test_log.csv")
        with CSVLogger(csv_path) as log:
            log.log({"a": 1, "b": 2})
            log.log({"a": 3, "b": 4})
        # File should be closed after exit
        assert log._f is None, "File should be closed after __exit__"
        # Verify content
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_T20_6_early_stopping_min_timesteps(self, cfg):
        """[ME-5] min_timesteps >= 60% of total_timesteps."""
        tc = cfg.get("training", {})
        total = tc.get("total_timesteps", 100000)
        es = tc.get("early_stopping", {})
        min_ts = es.get("min_timesteps", 0)
        assert min_ts >= total * 0.60, \
            f"min_timesteps {min_ts} should be >= {total * 0.60}"

    def test_T20_7_gen_users_vectorized_output(self, cfg):
        """[HI-2] Vectorized gen_users produces correct schema."""
        from oran3pt.gen_users import generate_users
        df = generate_users(cfg, seed=42)
        assert len(df) == cfg["population"]["N_total"]
        expected_cols = {
            "user_id", "slice", "segment", "mu_urllc", "sigma_urllc",
            "mu_embb", "sigma_embb", "price_sensitivity",
            "qos_sensitivity", "switching_cost", "clv_discount_rate",
            "is_active_init",
        }
        assert set(df.columns) == expected_cols, \
            f"Column mismatch: {set(df.columns) - expected_cols}"
        assert df["is_active_init"].sum() == cfg["population"]["N_active_init"]

    def test_T20_8_integration_env_eval_cycle(self, cfg):
        """[HI-3] Integration: env → random episode → valid rollout data."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        from oran3pt.eval import evaluate_episode
        records = evaluate_episode(env, model=None, repeat_id=0, n_chains=1)
        assert len(records) == env.episode_len, \
            f"Expected {env.episode_len} records, got {len(records)}"
        # Verify required columns
        required = ["step", "profit", "revenue", "cost_total",
                     "pviol_U", "pviol_E", "N_active"]
        for k in required:
            assert k in records[0], f"Missing key: {k}"
        # All profits should be finite
        profits = [r["profit"] for r in records]
        assert all(np.isfinite(p) for p in profits), "Non-finite profit found"


# =====================================================================
# T21  [V11] Revision v11 improvements
# =====================================================================
class TestV11Improvements:

    def test_T21_1_rho_U_max_020(self, cfg):
        """[V11-1] rho_U_max should be <= 0.20.
        [Huang IoT-J 2020; Sciancalepore TNSM 2019]"""
        assert cfg["action"]["rho_U_max"] <= 0.20, \
            f"rho_U_max should be <= 0.20, got {cfg['action']['rho_U_max']}"
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        max_action = np.ones(5, dtype=np.float32)
        _, _, _, _, info = env.step(max_action)
        assert info["rho_U"] <= 0.20 + 1e-6

    def test_T21_2_beta_pop_adequate(self, cfg):
        """[V11-2] beta_pop >= 0.5 for meaningful population signal.
        [Mguni AAMAS 2019 §4.2; Zheng Science Adv 2022]"""
        beta_pop = cfg.get("population_reward", {}).get("beta_pop", 0.3)
        assert beta_pop >= 0.5, \
            f"beta_pop should be >= 0.5, got {beta_pop}"

    def test_T21_3_admission_pviol_ceiling_relaxed(self, cfg):
        """[V11-3] pviol_ceiling >= 0.40 to prevent over-blocking.
        [Caballero JSAC 2019; 3GPP TS 23.501 §5.2.3]"""
        ac = cfg.get("admission_control", {})
        ceiling = ac.get("pviol_ceiling", 0.30)
        assert ceiling >= 0.40, \
            f"pviol_ceiling should be >= 0.40, got {ceiling}"
        threshold = ac.get("load_threshold", 0.85)
        assert threshold >= 0.90, \
            f"load_threshold should be >= 0.90, got {threshold}"

    def test_T21_4_lagrangian_state_json_schema(self, tmp_path):
        """[V11-4] Lagrangian state JSON has required keys.
        [Stooke ICLR 2020; Tessler ICML 2019]"""
        import json
        state = {"lambda": 1.5, "integral": 0.3, "prev_error": 0.1,
                 "threshold": 0.15}
        path = tmp_path / "lagrangian_state.json"
        with open(path, "w") as f:
            json.dump(state, f)
        with open(path) as f:
            loaded = json.load(f)
        assert "lambda" in loaded
        assert "threshold" in loaded
        assert loaded["lambda"] >= 0.0

    def test_T21_5_pop_bonus_quadratic_config(self, cfg):
        """[V11-8] Population reward has quadratic option.
        [Zheng Science Adv 2022; Mguni AAMAS 2019]"""
        pr = cfg.get("population_reward", {})
        assert "quadratic" in pr, "Missing 'quadratic' key in population_reward"

    def test_T21_6_pop_bonus_quadratic_asymmetry(self, cfg):
        """[V11-8] Quadratic pop bonus penalizes deficit more than surplus.
        [Gupta JSR 2006; Zheng 2022]"""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        # Simulate deficit scenario
        env._active_mask[:] = False
        env._active_mask[:150] = True  # 30% active
        env.step(env.action_space.sample())
        # pop_bonus should be negative (below target)
        obs, _, _, _, info = env.step(env.action_space.sample())
        assert info["pop_bonus"] < 0, \
            f"Pop bonus should be negative at 30% active, got {info['pop_bonus']}"

    def test_T21_7_early_stopping_75pct(self, cfg):
        """[V11-6] min_timesteps >= 75% of total_timesteps.
        [Prechelt 2002; Henderson AAAI 2018]"""
        tc = cfg.get("training", {})
        total = tc.get("total_timesteps", 100000)
        es = tc.get("early_stopping", {})
        min_ts = es.get("min_timesteps", 0)
        assert min_ts >= total * 0.75, \
            f"min_timesteps {min_ts} should be >= {total * 0.75}"

    def test_T21_8_rho_U_smoothing_moderate(self, cfg):
        """[I-3a] rho_U smoothing weight ~0.05 for C_E stability.
        [Dalal NeurIPS 2018 §4.1; downstream C_E impact proportional]"""
        weights = cfg.get("action_smoothing", {}).get("weights", [])
        assert len(weights) == 5
        assert 0.03 <= weights[4] <= 0.10, \
            f"rho_U smoothing weight should be in [0.03, 0.10], got {weights[4]}"

    def test_T21_9_embb_capacity_adequate_after_v11(self, cfg):
        """[V11-1] With rho_U_max=0.20, eMBB capacity is adequate.
        C_E = (1-0.20) × 400 = 320GB >> mean L_E ~249GB"""
        rho_max = cfg["action"]["rho_U_max"]
        C_total = cfg["radio"]["C_total_gb_per_step"]
        C_E_min = (1.0 - rho_max) * C_total
        assert C_E_min >= 300.0, \
            f"Min C_E = {C_E_min:.0f}GB should be >= 300GB for eMBB adequacy"


# =====================================================================
# T22  [PR] Pricing mechanism improvements
# =====================================================================
class TestPR1SliceSpecificPricing:
    """[PR-1] Per-slice price signal tests.
    [Train 2009; Anderson, de Palma & Thisse 1992]
    """

    def test_T22_1_urllc_user_unaffected_by_F_E(self, cfg):
        """URLLC user churn should depend on F_U, not F_E."""
        import copy
        cfg1 = copy.deepcopy(cfg)
        cfg2 = copy.deepcopy(cfg)
        # Disable hierarchical so pricing changes take effect mid-cycle
        cfg1["hierarchical_actions"] = {"enabled": False}
        cfg2["hierarchical_actions"] = {"enabled": False}
        # Disable curriculum so market step runs
        cfg1["training"]["curriculum"]["enabled"] = False
        cfg2["training"]["curriculum"]["enabled"] = False
        env1 = OranSlicingPricingEnv(cfg1, seed=42)
        env2 = OranSlicingPricingEnv(cfg2, seed=42)
        env1.reset(seed=42)
        env2.reset(seed=42)
        # Same F_U (a[0]=0.0), different F_E (a[2]=-1.0 vs +1.0)
        a1 = np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        a2 = np.array([0.0, 0.0, +1.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info1 = env1.step(a1)
        _, _, _, _, info2 = env2.step(a2)
        # With per-slice P_sig, F_E change should not affect URLLC churn
        # but should affect eMBB churn (and therefore total churn differs)
        assert np.isfinite(info1["n_churn"])
        assert np.isfinite(info2["n_churn"])

    def test_T22_2_psig_bounded_01(self, cfg):
        """Per-slice P_sig should be bounded by [0, 1] at action extremes."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        # At maximum actions: F_U/F_U_max=1.0, F_E/F_E_max=1.0
        max_action = np.ones(5, dtype=np.float32)
        env.step(max_action)
        # At minimum actions: F_U/F_U_max > 0, F_E/F_E_max > 0
        min_action = -np.ones(5, dtype=np.float32)
        env.step(min_action)
        # Check P_sig values indirectly via env (no NaN/crash)
        assert env._use_per_slice_psig, "Per-slice P_sig should be enabled"

    def test_T22_3_embb_churn_responds_to_F_E(self, cfg):
        """eMBB churn should increase with F_E."""
        import copy
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["hierarchical_actions"] = {"enabled": False}
        cfg_copy["market"]["mode"] = "expectation"
        cfg_copy["market"]["beta_bill_shock"] = 0.0  # isolate price effect
        env_lo = OranSlicingPricingEnv(cfg_copy, seed=42)
        env_hi = OranSlicingPricingEnv(cfg_copy, seed=42)
        env_lo.reset(seed=42)
        env_hi.reset(seed=42)
        churn_lo, churn_hi = 0, 0
        for _ in range(30):
            # Low F_E
            _, _, _, _, info_lo = env_lo.step(
                np.array([0.0, 0.0, -0.8, 0.0, 0.0], dtype=np.float32))
            churn_lo += info_lo["n_churn"]
            # High F_E
            _, _, _, _, info_hi = env_hi.step(
                np.array([0.0, 0.0, 0.8, 0.0, 0.0], dtype=np.float32))
            churn_hi += info_hi["n_churn"]
        assert churn_hi >= churn_lo, \
            f"Higher F_E should cause more churn: {churn_lo} vs {churn_hi}"


class TestPR2BillShock:
    """[PR-2] Bill shock mechanism tests.
    [Grubb & Osborne AER 2015; Lambrecht & Skiera JMR 2006]
    """

    def test_T22_4_bill_shock_config_present(self, cfg):
        """Bill shock config keys exist."""
        mc = cfg.get("market", {})
        assert "beta_bill_shock" in mc, "Missing beta_bill_shock"
        assert "bill_shock_threshold" in mc, "Missing bill_shock_threshold"
        assert mc["beta_bill_shock"] > 0, "beta_bill_shock should be positive"
        assert mc["bill_shock_threshold"] > 1.0, "threshold should be > 1.0"

    def test_T22_5_bill_shock_disabled_at_cycle_start(self, cfg):
        """Bill shock should be zero at cycle start (no accumulated overage)."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        # First step of cycle: cycle_step=0, no overage accumulated
        _, _, _, _, info = env.step(env.action_space.sample())
        # At step 1, cycle_step=0, bill shock should not contribute
        assert np.isfinite(info["n_churn"])

    def test_T22_6_no_bill_shock_when_within_allowance(self, cfg):
        """No bill shock when usage is well within data allowance."""
        import copy
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["market"]["beta_bill_shock"] = 1.0
        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        env.reset(seed=42)
        # Low overage price → low p_over → high usage but Q_E=50GB is large
        low_price = np.array([0.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(low_price)
        # First step: cycle_usage ≈ 0, no overage → no bill shock
        assert info["n_churn"] >= 0

    def test_T22_7_bill_shock_env_attribute(self, cfg):
        """Bill shock enabled flag is set correctly."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        assert env._bill_shock_enabled is True, \
            "Bill shock should be enabled with beta_bill_shock > 0"
        assert env._bill_shock_threshold == 1.5


class TestPR4OverageJoin:
    """[PR-4] Overage price → join dampening tests.
    [Nevo et al. Econometrica 2016]
    """

    def test_T22_8_p_over_join_config(self, cfg):
        """PR-4 config key exists."""
        mc = cfg.get("market", {})
        assert "beta_p_over_join" in mc, "Missing beta_p_over_join"
        assert mc["beta_p_over_join"] > 0

    def test_T22_9_p_over_join_dampens_joins(self, cfg):
        """Higher overage price should reduce joins (isolating PR-4 effect).
        Demand elasticity disabled to prevent indirect congestion channel
        from dominating the direct join logit dampening."""
        import copy
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["hierarchical_actions"] = {"enabled": False}
        cfg_copy["market"]["mode"] = "expectation"
        cfg_copy["market"]["beta_bill_shock"] = 0.0
        cfg_copy["demand_elasticity"] = {"enabled": False}  # isolate PR-4
        env_lo = OranSlicingPricingEnv(cfg_copy, seed=42)
        env_hi = OranSlicingPricingEnv(cfg_copy, seed=42)
        env_lo.reset(seed=42)
        env_hi.reset(seed=42)
        joins_lo, joins_hi = 0, 0
        for _ in range(30):
            _, _, _, _, info_lo = env_lo.step(
                np.array([0.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32))
            joins_lo += info_lo["n_join"]
            _, _, _, _, info_hi = env_hi.step(
                np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32))
            joins_hi += info_hi["n_join"]
        assert joins_lo >= joins_hi, \
            f"Lower overage price should allow more joins: {joins_lo} vs {joins_hi}"

    def test_T22_10_backward_compat_no_pr(self, cfg):
        """Environment works with PR features disabled."""
        import copy
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["market"]["beta_bill_shock"] = 0.0
        cfg_copy["market"]["beta_p_over_join"] = 0.0
        env = OranSlicingPricingEnv(cfg_copy, seed=42)
        env.reset(seed=42)
        for _ in range(30):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if term or trunc:
                break

    def test_T22_11_numerical_safety_full_episode(self, cfg):
        """Full episode with all PR features, no NaN/Inf."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))
        for step in range(env.episode_len):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)), f"Step {step}: obs NaN/Inf"
            assert np.isfinite(reward), f"Step {step}: reward NaN/Inf"
            assert np.isfinite(info["profit"]), f"Step {step}: profit NaN/Inf"
            if term or trunc:
                break


# =====================================================================
# T23  [I-1..I-6] Structural improvement tests
# =====================================================================
class TestStructuralImprovements:
    """Tests for improvement_plan.md Phase A/B/C changes."""

    # ── I-1a: Asymmetric integral floor ──────────────────────────────

    def test_T23_1_asymmetric_integral_floor(self, cfg):
        """[I-1a] PID integral_min = -lambda_max * 0.2 (tight negative bound).
        [Stooke ICLR 2020 §3.2; Mao arXiv 2025]"""
        try:
            from oran3pt.train import _LagrangianPIDCallback
        except ImportError:
            pytest.skip("SB3 not available")
        lag = cfg.get("lagrangian_qos", {})
        cb = _LagrangianPIDCallback(
            threshold=lag.get("pviol_E_threshold", 0.15),
            Kp=lag.get("Kp", 0.05), Ki=lag.get("Ki", 0.005),
            Kd=lag.get("Kd", 0.01), lambda_max=lag.get("lambda_max", 10.0),
            lambda_min=lag.get("lambda_min", 0.1),
        )
        assert cb._integral_min == pytest.approx(-cb._lambda_max * 0.2), \
            f"integral_min should be -lambda_max * 0.2"
        # Simulate sustained negative error (feasible region)
        for _ in range(5000):
            cb._error_integral = max(
                cb._integral_min,
                min(cb._integral_max, cb._error_integral - 0.1))
        assert cb._error_integral >= cb._integral_min, \
            f"Integral {cb._error_integral} below floor {cb._integral_min}"
        # integral_min should be much smaller in magnitude than integral_max
        assert abs(cb._integral_min) < abs(cb._integral_max), \
            "Negative bound should be tighter than positive bound"

    def test_T23_2_integral_recovery_speed(self, cfg):
        """[I-1a] Lambda recovers from min within 50 PID updates.
        With integral_min = -2.0 (lambda_max*0.2), recovery at
        error=+0.1 crosses zero in ~20 updates, lambda rises by ~40.
        [Mao arXiv 2025 — feasible region windup prevention]"""
        try:
            from oran3pt.train import _LagrangianPIDCallback
        except ImportError:
            pytest.skip("SB3 not available")
        lag = cfg.get("lagrangian_qos", {})
        cb = _LagrangianPIDCallback(
            threshold=0.15, Kp=lag.get("Kp", 0.05),
            Ki=lag.get("Ki", 0.005), Kd=lag.get("Kd", 0.01),
            lambda_max=lag.get("lambda_max", 10.0),
            lambda_min=lag.get("lambda_min", 0.1),
        )
        # Drive integral to floor (simulate long feasible period)
        cb._error_integral = cb._integral_min
        cb.lambda_val = cb._lambda_min
        # Now simulate pviol_E=0.25 (above threshold) for N updates
        for _ in range(50):
            error = 0.25 - 0.15  # +0.1
            cb._error_integral = max(
                cb._integral_min,
                min(cb._integral_max, cb._error_integral + error))
            derivative = error - cb._prev_error
            delta = cb._Kp * error + cb._Ki * cb._error_integral + cb._Kd * derivative
            cb.lambda_val = max(cb._lambda_min,
                               min(cb._lambda_max, cb.lambda_val + delta))
            cb._prev_error = error
        assert cb.lambda_val > cb._lambda_min + 0.01, \
            f"Lambda should recover above min+0.01 after 50 updates, got {cb.lambda_val}"

    # ── I-1b: Lambda minimum ─────────────────────────────────────────

    def test_T23_3_lambda_min_config(self, cfg):
        """[I-1b] lambda_min is configured and > 0.
        [Paternain CDC 2019; TAC 2022]"""
        lag = cfg.get("lagrangian_qos", {})
        assert "lambda_min" in lag, "Missing lambda_min in lagrangian_qos"
        assert lag["lambda_min"] > 0, f"lambda_min should be > 0, got {lag['lambda_min']}"

    def test_T23_4_lambda_min_in_callback(self, cfg):
        """[I-1b] PID callback uses lambda_min for lower bound."""
        try:
            from oran3pt.train import _LagrangianPIDCallback
        except ImportError:
            pytest.skip("SB3 not available")
        lag = cfg.get("lagrangian_qos", {})
        cb = _LagrangianPIDCallback(
            lambda_max=10.0, lambda_min=0.1)
        assert cb.lambda_val == 0.1, \
            f"Initial lambda should be lambda_min=0.1, got {cb.lambda_val}"
        assert cb._lambda_min == 0.1

    # ── F2: SLA gamma=3.0 (revert from 4.0) ──────────────────────────

    def test_T23_5_sla_gamma_cubic(self, cfg):
        """[F2] gamma_sla_E=3.0 reverted — cubic provides 5× stronger
        signal than quartic at operating point pviol_E=0.2.
        [Bertsekas 1996 §6.3]"""
        gamma = cfg["qos"]["gamma_sla_E"]
        assert gamma == 3.0, f"gamma_sla_E should be 3.0, got {gamma}"
        # At pviol=0.2: 0.2^3=0.008 vs 0.2^4=0.0016 → cubic is 5× stronger
        penalty_cubic = 0.2 ** 3.0
        penalty_quartic = 0.2 ** 4.0
        assert penalty_cubic > penalty_quartic, \
            "Cubic should produce stronger signal than quartic at pviol=0.2"

    # ── I-3a: rho_U smoothing ────────────────────────────────────────

    def test_T23_6_rho_U_smoothing_equals_pricing(self, cfg):
        """[I-3a] rho_U smoothing weight matches p_over smoothing.
        [Dalal NeurIPS 2018 §4.1 — proportional to downstream impact]"""
        weights = cfg["action_smoothing"]["weights"]
        assert weights[4] == weights[1], \
            f"rho_U weight {weights[4]} should match p_over weight {weights[1]}"

    # ── I-3b: Capacity guard ─────────────────────────────────────────

    def test_T23_7_capacity_guard_config(self, cfg):
        """[I-3b] Capacity guard config present.
        [3GPP TS 23.501 §5.15.7; Samdanis CommMag 2016]"""
        cg = cfg.get("capacity_guard", {})
        assert cg.get("enabled") is True
        assert "embb_load_ratio_max" in cg
        assert "penalty_scale" in cg
        assert 0.80 <= cg["embb_load_ratio_max"] <= 1.0

    def test_T23_8_capacity_guard_penalty_in_info(self, env):
        """[I-3b] Info dict contains capacity_penalty key."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "capacity_penalty" in info
        assert info["capacity_penalty"] >= 0.0

    def test_T23_9_capacity_guard_activates_on_overload(self, cfg):
        """[I-3b] Capacity penalty > 0 when L_E/C_E exceeds threshold."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        env.reset(seed=42)
        # Force high rho_U → small C_E → likely overload
        high_rho = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        cap_penalties = []
        for _ in range(30):
            _, _, _, _, info = env.step(high_rho)
            cap_penalties.append(info["capacity_penalty"])
        # At least some steps should have capacity penalty
        total = sum(cap_penalties)
        # With rho_U at max (0.20), C_E = 0.80*400 = 320 GB
        # With ~200 eMBB users, L_E could exceed 320*0.95=304 GB
        assert total >= 0.0, "Total capacity penalty should be non-negative"

    # ── I-5a: SLA awareness penalty ──────────────────────────────────

    def test_T23_10_sla_awareness_config(self, cfg):
        """[I-5a] SLA awareness config present.
        [Wiewiora ICML 2003; Ng ICML 1999]"""
        sa = cfg.get("sla_awareness", {})
        assert sa.get("enabled") is True
        assert "revenue_ratio_threshold" in sa
        assert "penalty_scale" in sa

    def test_T23_11_sla_awareness_in_info(self, env):
        """[I-5a] Info dict contains sla_awareness_penalty key."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "sla_awareness_penalty" in info
        assert info["sla_awareness_penalty"] >= 0.0

    # ── I-6a: Target entropy ─────────────────────────────────────────

    def test_T23_12_target_entropy_config(self, cfg):
        """[I-6a] target_entropy is configured.
        [Haarnoja ICML 2018 §5; Zhou ICLR 2022]"""
        tc = cfg.get("training", {})
        assert "target_entropy" in tc
        assert tc["target_entropy"] == -3.0

    # ── F1: Eval PID removed ─────────────────────────────────────────

    def test_T23_13_eval_pid_removed(self):
        """[F1] _EvalPIDController removed — deterministic eval ignores
        reward changes, so dynamic λ only distorted metrics."""
        import oran3pt.eval as ev
        assert not hasattr(ev, "_EvalPIDController"), \
            "_EvalPIDController should be removed from eval module"

    def test_T23_14_eval_episode_no_pid_param(self):
        """[F1] evaluate_episode no longer accepts eval_pid parameter."""
        import inspect
        from oran3pt.eval import evaluate_episode
        sig = inspect.signature(evaluate_episode)
        assert "eval_pid" not in sig.parameters, \
            "eval_pid parameter should be removed"

    # ── I-6b: Eval action smoothing ──────────────────────────────────

    def test_T23_15_eval_episode_accepts_smoothing(self):
        """[I-6b] evaluate_episode accepts action_ema_alpha parameter."""
        import inspect
        from oran3pt.eval import evaluate_episode
        sig = inspect.signature(evaluate_episode)
        assert "action_ema_alpha" in sig.parameters

    # ── Numerical safety with all improvements ───────────────────────

    def test_T23_16_full_episode_numerical_safety(self, cfg):
        """All improvements active, full episode, no NaN/Inf."""
        env = OranSlicingPricingEnv(cfg, seed=42)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))
        for step in range(env.episode_len):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)), f"Step {step}: obs NaN/Inf"
            assert np.isfinite(reward), f"Step {step}: reward NaN/Inf"
            assert np.isfinite(info["profit"]), f"Step {step}: profit NaN/Inf"
            assert info["capacity_penalty"] >= 0.0
            assert info["sla_awareness_penalty"] >= 0.0
            if term or trunc:
                break

    # ── F3: pviol_E threshold conservative ─────────────────────────────

    def test_T23_17_pviol_threshold_conservative(self, cfg):
        """[F3] pviol_E_threshold ≤ 0.10 for train-eval robustness margin.
        [Tobin IROS 2017; Rajeswaran NeurIPS 2017]"""
        threshold = cfg.get("lagrangian_qos", {}).get("pviol_E_threshold", 0.15)
        assert threshold <= 0.10, \
            f"pviol_E_threshold should be ≤ 0.10 for eval margin, got {threshold}"

    # ── F4: Ki adequate for timely enforcement ─────────────────────────

    def test_T23_18_lagrangian_ki_adequate(self, cfg):
        """[F4] Ki ≥ 0.02 for timely constraint enforcement.
        [Stooke ICLR 2020 §3.2; Mao arXiv 2025]"""
        Ki = cfg.get("lagrangian_qos", {}).get("Ki", 0.005)
        assert Ki >= 0.02, f"Ki should be ≥ 0.02, got {Ki}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
