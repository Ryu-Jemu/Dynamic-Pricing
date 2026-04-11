"""
Environment Verification Tests (Level 1)
==========================================
Tests match the verified numerical values from 6-fold Audit:
  - E[q_U]=3.4903, E[q_E]=29.9641
  - P(q>Q_bar): URLLC 19.44%, eMBB 32.69%
  - P_dep = sigma(-10.77) = 2.1020e-5
  - Monthly churn = 1.502%
  - E[B_U]=$55.59, E[B_E]=$81.67
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats
from env.network_slicing_env import NetworkSlicingEnv


def test_lognormal_usage():
    """Verify LogNormal E[q] and Var[q] match analytical values."""
    rng = np.random.default_rng(42)
    n_samples = 500_000

    # URLLC: mu=1.0, sigma^2=0.5
    samples_u = rng.lognormal(1.0, np.sqrt(0.5), n_samples)
    expected_u = np.exp(1.0 + 0.5 / 2)  # 3.4903
    assert abs(np.mean(samples_u) - expected_u) / expected_u < 0.01, \
        f"URLLC E[q] mismatch: {np.mean(samples_u):.4f} vs {expected_u:.4f}"

    # eMBB: mu=3.0, sigma^2=0.8
    samples_e = rng.lognormal(3.0, np.sqrt(0.8), n_samples)
    expected_e = np.exp(3.0 + 0.8 / 2)  # 29.9641
    assert abs(np.mean(samples_e) - expected_e) / expected_e < 0.02, \
        f"eMBB E[q] mismatch: {np.mean(samples_e):.4f} vs {expected_e:.4f}"

    # P(q > Q_bar)
    p_exceed_u = np.mean(samples_u > 5.0)
    p_exceed_e = np.mean(samples_e > 30.0)
    assert abs(p_exceed_u - 0.1944) < 0.015, \
        f"URLLC P(q>5) mismatch: {p_exceed_u:.4f} vs 0.1944"
    assert abs(p_exceed_e - 0.3269) < 0.015, \
        f"eMBB P(q>30) mismatch: {p_exceed_e:.4f} vs 0.3269"

    print("  [PASS] LogNormal usage: E[q] and P(q>Q_bar) verified")


def test_departure_probability():
    """Verify sigma(-10.77) = 2.1020e-5 and monthly churn = 1.502%."""
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # At reference prices (F_tilde=1, p_tilde=1), E[eta_U]=0.95, E[eta_E]=0.90
    x_u = -9.72 + 1.0 * 1.0 + 0.8 * 1.0 - 3.0 * 0.95
    x_e = -12.12 + 1.0 * 1.0 + 0.8 * 1.0 - 0.5 * 0.90
    assert abs(x_u - (-10.77)) < 0.01, f"URLLC logit mismatch: {x_u}"
    assert abs(x_e - (-10.77)) < 0.01, f"eMBB logit mismatch: {x_e}"

    P_dep = sigmoid(-10.77)
    assert abs(P_dep - 2.1020e-5) / 2.1020e-5 < 0.01, \
        f"P_dep mismatch: {P_dep:.6e} vs 2.1020e-5"

    monthly_churn = 1.0 - (1.0 - P_dep) ** 720
    assert abs(monthly_churn - 0.01502) < 0.001, \
        f"Monthly churn mismatch: {monthly_churn:.5f} vs 0.01502"

    print("  [PASS] Departure: sigma(-10.77) and monthly churn verified")


def test_arrival_probability():
    """Verify arrival probabilities and expected counts."""
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # At reference prices
    P_arr_u = sigmoid(2.0 - 0.8 - 0.6)  # sigma(0.6)
    P_arr_e = sigmoid(2.5 - 0.8 - 0.6)  # sigma(1.1)

    assert abs(P_arr_u - 0.6457) < 0.001, f"P_arr_U mismatch: {P_arr_u:.4f}"
    assert abs(P_arr_e - 0.7503) < 0.001, f"P_arr_E mismatch: {P_arr_e:.4f}"

    E_new_u = 0.05 * P_arr_u
    E_new_e = 0.15 * P_arr_e
    assert abs(E_new_u - 0.0323) < 0.001, f"E[N_new_U] mismatch: {E_new_u:.4f}"
    assert abs(E_new_e - 0.1125) < 0.001, f"E[N_new_E] mismatch: {E_new_e:.4f}"

    print("  [PASS] Arrival: P_arr and E[N_new] verified")


def test_expected_bill():
    """Verify E[B] using lognormal call formula."""
    def call_formula(mu, sigma2, K):
        sigma = np.sqrt(sigma2)
        E_q = np.exp(mu + sigma2 / 2)
        d1 = (mu + sigma2 - np.log(K)) / sigma
        d2 = d1 - sigma
        return E_q * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)

    # URLLC: F=50, Q_bar=5, p=10
    overage_u = call_formula(1.0, 0.5, 5.0)
    E_bill_u = 50.0 + overage_u * 10.0
    assert abs(E_bill_u - 55.5863) < 0.01, \
        f"E[B_U] mismatch: {E_bill_u:.4f} vs 55.5863"

    # eMBB: F=30, Q_bar=30, p=5
    overage_e = call_formula(3.0, 0.8, 30.0)
    E_bill_e = 30.0 + overage_e * 5.0
    assert abs(E_bill_e - 81.6712) < 0.05, \
        f"E[B_E] mismatch: {E_bill_e:.4f} vs 81.6712"

    print("  [PASS] Expected bill: E[B_U]=$55.59, E[B_E]=$81.67 verified")


def test_env_basic_step():
    """Test that the environment runs and produces valid outputs."""
    env = NetworkSlicingEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (4,), f"Obs shape mismatch: {obs.shape}"
    # After normalization: N_U/N_init_U = 1000/1000 = 1.0
    assert abs(obs[0] - 1.0) < 0.01, f"N_U/N_init mismatch: {obs[0]}"
    assert abs(obs[1] - 1.0) < 0.01, f"N_E/N_init mismatch: {obs[1]}"

    # Reference price action: F_U=50/100=0.5, p_U=10/20=0.5, F_E=30/100=0.3, p_E=5/20=0.25
    action = np.array([0.5, 0.5, 0.3, 0.25], dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)

    assert obs2.shape == (4,), "Step obs shape mismatch"
    assert not terminated, "Should not terminate at t=1"
    assert reward > 0, f"Reward should be positive at reference prices: {reward}"
    assert info["revenue"] > 0, "Revenue should be positive"
    assert info["N_U"] > 0, "N_U should be positive"
    assert info["N_E"] > 0, "N_E should be positive"

    print("  [PASS] Environment basic step: reset, step, obs, reward verified")


def test_env_full_episode():
    """Run a full 720-step episode and check final state."""
    env = NetworkSlicingEnv()
    obs, _ = env.reset(seed=123)

    total_reward = 0.0
    action = np.array([0.5, 0.5, 0.3, 0.25], dtype=np.float32)

    for t in range(720):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    assert terminated, "Should terminate at t=720"
    assert total_reward > 0, f"Total reward should be positive: {total_reward}"
    # With reward_scale=1e-5, total should be ~3100 (not ~311M raw)
    assert 1000 < total_reward < 10000, \
        f"Scaled total_reward out of expected range: {total_reward:.0f}"

    # N should stay roughly stable (not explode or collapse)
    assert 800 < info["N_U"] < 1500, f"N_U out of range: {info['N_U']}"
    assert 4000 < info["N_E"] < 6000, f"N_E out of range: {info['N_E']}"

    print(f"  [PASS] Full episode: total_reward={total_reward:,.0f}, "
          f"N_U={info['N_U']:.0f}, N_E={info['N_E']:.0f}")


def test_edge_case_zero_price():
    """F=0, p=0 should give zero revenue."""
    env = NetworkSlicingEnv()
    env.reset(seed=42)
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["revenue"] == 0.0, f"Revenue should be 0 at zero prices: {info['revenue']}"
    print("  [PASS] Edge case: zero price → zero revenue")


if __name__ == "__main__":
    print("=" * 60)
    print("Environment Verification Tests (Level 1)")
    print("=" * 60)
    test_lognormal_usage()
    test_departure_probability()
    test_arrival_probability()
    test_expected_bill()
    test_env_basic_step()
    test_env_full_episode()
    test_edge_case_zero_price()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
