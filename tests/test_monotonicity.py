"""
Tests for monotonicity properties (§20 — test_monotonicity.py).

Tests:
  - Increasing N_active (thus rho_util) must not increase T_eff
  - Violations must not decrease when offered load increases

References:
  [CONG_5G_PMC][LTE_LOAD_TPUT]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.utils import load_config
from src.models.radio import RadioConfig, RadioModel


@pytest.fixture
def cfg():
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    return load_config(str(config_path))


@pytest.fixture
def radio(cfg):
    rc = RadioConfig.from_config(cfg)
    return RadioModel(rc)


class TestThroughputMonotonicity:
    """T_eff must decrease (or stay flat) as utilization increases."""

    def test_t_eff_decreases_with_n_active(self, radio):
        """More users sharing fixed PRBs → lower per-user throughput."""
        rho_URLLC = 0.3
        T_exp = np.array([5.0])  # single user, plan-based T_exp

        prev_avg = float("inf")
        for n_eMBB in [10, 30, 60, 100, 150, 200]:
            T_exp_arr = np.array([5.0] * n_eMBB)
            T_exp_urllc = np.array([3.0] * 10)
            result = radio.inner_step(
                N_active_eMBB=n_eMBB,
                N_active_URLLC=10,
                rho_URLLC=rho_URLLC,
                T_exp_users_eMBB=T_exp_arr,
                T_exp_users_URLLC=T_exp_urllc,
            )
            avg_T = result["avg_T_act_eMBB"]
            assert avg_T <= prev_avg + 1e-6, \
                f"T_eff increased from {prev_avg:.4f} to {avg_T:.4f} at N={n_eMBB}"
            prev_avg = avg_T

    def test_rho_util_increases_with_n_active(self, radio):
        """More users → higher utilization."""
        rho_URLLC = 0.3
        prev_rho = -1.0

        for n_eMBB in [10, 30, 60, 100, 150]:
            T_exp_arr = np.array([5.0] * n_eMBB)
            T_exp_urllc = np.array([3.0] * 10)
            result = radio.inner_step(
                N_active_eMBB=n_eMBB,
                N_active_URLLC=10,
                rho_URLLC=rho_URLLC,
                T_exp_users_eMBB=T_exp_arr,
                T_exp_users_URLLC=T_exp_urllc,
            )
            rho = result["rho_util_eMBB"]
            assert rho >= prev_rho - 1e-6, \
                f"rho_util decreased from {prev_rho:.4f} to {rho:.4f} at N={n_eMBB}"
            prev_rho = rho

    def test_urllc_slice_allocation_monotone(self, radio):
        """Increasing rho_URLLC → more PRBs for URLLC, fewer for eMBB."""
        n_eMBB = 80
        n_URLLC = 20
        T_exp_e = np.array([5.0] * n_eMBB)
        T_exp_u = np.array([3.0] * n_URLLC)

        prev_T_urllc = 0.0
        prev_T_embb = float("inf")

        for rho in [0.1, 0.2, 0.3, 0.5, 0.7]:
            result = radio.inner_step(
                N_active_eMBB=n_eMBB,
                N_active_URLLC=n_URLLC,
                rho_URLLC=rho,
                T_exp_users_eMBB=T_exp_e,
                T_exp_users_URLLC=T_exp_u,
            )
            T_urllc = result["avg_T_act_URLLC"]
            T_embb = result["avg_T_act_eMBB"]

            # URLLC throughput should increase with more PRBs
            assert T_urllc >= prev_T_urllc - 1e-6, \
                f"URLLC T decreased at rho={rho}"
            # eMBB throughput should decrease with fewer PRBs
            assert T_embb <= prev_T_embb + 1e-6, \
                f"eMBB T increased at rho={rho}"
            prev_T_urllc = T_urllc
            prev_T_embb = T_embb


class TestViolationMonotonicity:
    """SLA violations should not decrease when load increases."""

    def test_violation_increases_with_load(self, cfg, radio):
        """More users → higher violation rate (lower per-user throughput)."""
        from src.models.economics import SLAModel
        sla = SLAModel(cfg)
        rho_URLLC = 0.3
        K = 30

        prev_V = -1.0
        for n_eMBB in [10, 50, 100, 150, 200]:
            step_avgs = np.zeros(K)
            T_exp_e = np.array([5.0] * n_eMBB)
            T_exp_u = np.array([3.0] * 10)

            for k in range(K):
                result = radio.inner_step(
                    N_active_eMBB=n_eMBB,
                    N_active_URLLC=10,
                    rho_URLLC=rho_URLLC,
                    T_exp_users_eMBB=T_exp_e,
                    T_exp_users_URLLC=T_exp_u,
                )
                step_avgs[k] = result["avg_T_act_eMBB"]

            V = sla.compute_violation_rate(step_avgs, "eMBB")
            assert V >= prev_V - 1e-6, \
                f"V_rate decreased from {prev_V:.4f} to {V:.4f} at N={n_eMBB}"
            prev_V = V


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
