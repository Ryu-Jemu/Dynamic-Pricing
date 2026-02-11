"""
Tests for monotonicity properties (§20 — test_monotonicity.py).

Tests:
  - Increasing N_active (thus rho_util) must not increase T_eff
  - Violations must not decrease when offered load increases

FIX F1: inner_step parameter renamed from T_exp_users_* to T_ceil_users_*
to clarify that these are throughput ceilings (v_max_mbps).

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

        prev_avg = float("inf")
        for n_eMBB in [10, 30, 60, 100, 150, 200]:
            # FIX F1: T_ceil = v_max_mbps (pre-cap peak), not v_cap
            T_ceil_arr = np.array([150.0] * n_eMBB)  # eMBB basic v_max
            T_ceil_urllc = np.array([100.0] * 10)     # URLLC standard v_max
            result = radio.inner_step(
                N_active_eMBB=n_eMBB,
                N_active_URLLC=10,
                rho_URLLC=rho_URLLC,
                T_ceil_users_eMBB=T_ceil_arr,
                T_ceil_users_URLLC=T_ceil_urllc,
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
            T_ceil_arr = np.array([150.0] * n_eMBB)
            T_ceil_urllc = np.array([100.0] * 10)
            result = radio.inner_step(
                N_active_eMBB=n_eMBB,
                N_active_URLLC=10,
                rho_URLLC=rho_URLLC,
                T_ceil_users_eMBB=T_ceil_arr,
                T_ceil_users_URLLC=T_ceil_urllc,
            )
            rho = result["rho_util_eMBB"]
            assert rho >= prev_rho - 1e-6, \
                f"rho_util decreased from {prev_rho:.4f} to {rho:.4f} at N={n_eMBB}"
            prev_rho = rho

    def test_urllc_slice_allocation_monotone(self, radio):
        """Increasing rho_URLLC → more PRBs for URLLC, fewer for eMBB."""
        n_eMBB = 80
        n_URLLC = 20
        T_ceil_e = np.array([300.0] * n_eMBB)   # eMBB standard v_max
        T_ceil_u = np.array([100.0] * n_URLLC)   # URLLC standard v_max

        prev_T_urllc = 0.0
        prev_T_embb = float("inf")

        for rho in [0.1, 0.2, 0.3, 0.5, 0.7]:
            result = radio.inner_step(
                N_active_eMBB=n_eMBB,
                N_active_URLLC=n_URLLC,
                rho_URLLC=rho,
                T_ceil_users_eMBB=T_ceil_e,
                T_ceil_users_URLLC=T_ceil_u,
            )
            T_urllc = result["avg_T_act_URLLC"]
            T_embb = result["avg_T_act_eMBB"]

            assert T_urllc >= prev_T_urllc - 1e-6, \
                f"URLLC T decreased at rho={rho}"
            assert T_embb <= prev_T_embb + 1e-6, \
                f"eMBB T increased at rho={rho}"
            prev_T_urllc = T_urllc
            prev_T_embb = T_embb


class TestViolationMonotonicity:
    """SLA violations should not decrease when load increases."""

    def test_violation_increases_with_load(self, cfg, radio):
        """More users → higher violation rate."""
        from src.models.economics import SLAModel
        sla = SLAModel(cfg)
        rho_URLLC = 0.3
        K = 30

        prev_V = -1.0
        for n_eMBB in [10, 50, 100, 150, 200]:
            step_avgs = np.zeros(K)
            # FIX F1: Use v_max as ceiling (high enough to not constrain)
            T_ceil_e = np.array([150.0] * n_eMBB)
            T_ceil_u = np.array([100.0] * 10)

            for k in range(K):
                result = radio.inner_step(
                    N_active_eMBB=n_eMBB,
                    N_active_URLLC=10,
                    rho_URLLC=rho_URLLC,
                    T_ceil_users_eMBB=T_ceil_e,
                    T_ceil_users_URLLC=T_ceil_u,
                )
                step_avgs[k] = result["avg_T_act_eMBB"]

            V = sla.compute_violation_rate(step_avgs, "eMBB")
            assert V >= prev_V - 1e-6, \
                f"V_rate decreased from {prev_V:.4f} to {V:.4f} at N={n_eMBB}"
            prev_V = V


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
