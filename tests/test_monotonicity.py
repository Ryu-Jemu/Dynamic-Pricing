"""
test_monotonicity.py — Increasing N_active (thus rho_util) must NOT
increase T_eff; violations must not decrease when offered load increases.

Deterministic by construction (fixed inputs, no RNG needed).
  [SB3_TIPS]

References:
  [CONG_5G_PMC] https://pmc.ncbi.nlm.nih.gov/articles/PMC10346256/
  [LTE_LOAD_TPUT] https://aircconline.com/ijcnc/V9N6/9617cnc03.pdf
"""

import unittest
import numpy as np

from src.models.utils import load_config
from src.models.radio import RadioConfig, RadioModel
from src.models.economics import SLAModel


class TestRadioMonotonicity(unittest.TestCase):
    """Test monotone degradation in the radio model."""

    def setUp(self):
        self.cfg = load_config("config/default.yaml")
        self.rc = RadioConfig.from_config(self.cfg)
        self.radio = RadioModel(self.rc)

    # ---------------------------------------------------------------
    # 10.1  rho_util monotonicity  [CONG_5G_PMC]
    # ---------------------------------------------------------------

    def test_rho_util_monotone_embb(self):
        """rho_util must be non-decreasing in N_active for eMBB."""
        N_values = [0, 1, 10, 50, 100, 150, 200, 300, 500]
        rhos = [self.radio.compute_rho_util(n, "eMBB") for n in N_values]
        for i in range(len(rhos) - 1):
            self.assertLessEqual(
                rhos[i], rhos[i + 1],
                f"rho_util not monotone: rho({N_values[i]})={rhos[i]} "
                f"> rho({N_values[i+1]})={rhos[i+1]}"
            )

    def test_rho_util_monotone_urllc(self):
        """rho_util must be non-decreasing in N_active for URLLC."""
        N_values = [0, 1, 5, 20, 50, 80, 100, 200]
        rhos = [self.radio.compute_rho_util(n, "URLLC") for n in N_values]
        for i in range(len(rhos) - 1):
            self.assertLessEqual(rhos[i], rhos[i + 1])

    def test_rho_util_bounds(self):
        """rho_util must always be in [0.0, rho_util_max]."""
        for n in [0, 1, 100, 1000, 10000]:
            for sname in ["eMBB", "URLLC"]:
                rho = self.radio.compute_rho_util(n, sname)
                self.assertGreaterEqual(rho, 0.0)
                self.assertLessEqual(rho, self.rc.rho_util_max)

    # ---------------------------------------------------------------
    # 10.2  T_eff monotone degradation  [CONG_5G_PMC][LTE_LOAD_TPUT]
    # ---------------------------------------------------------------

    def test_T_eff_decreasing_in_rho(self):
        """T_eff must be strictly decreasing as rho_util increases."""
        rho_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.98]
        T_effs = [self.radio.compute_T_eff(rho) for rho in rho_values]
        for i in range(len(T_effs) - 1):
            self.assertGreater(
                T_effs[i], T_effs[i + 1],
                f"T_eff not decreasing: T_eff({rho_values[i]})={T_effs[i]} "
                f"<= T_eff({rho_values[i+1]})={T_effs[i+1]}"
            )

    def test_T_eff_nonnegative(self):
        """T_eff must be non-negative for all valid rho."""
        for rho in [0.0, 0.5, 0.98, 1.0]:
            T_eff = self.radio.compute_T_eff(rho)
            self.assertGreaterEqual(T_eff, 0.0)

    def test_T_eff_at_zero_load(self):
        """At zero load, T_eff = T_cell_cap_mbps."""
        T_eff = self.radio.compute_T_eff(0.0)
        self.assertAlmostEqual(T_eff, self.rc.T_cell_cap_mbps, places=4)

    # ---------------------------------------------------------------
    # Combined: more users → lower T_eff
    # ---------------------------------------------------------------

    def test_more_users_lower_T_eff(self):
        """Increasing N_active → higher rho → lower T_eff (end-to-end)."""
        N_low = 20
        N_high = 180

        rho_low = self.radio.compute_rho_util(N_low, "eMBB")
        rho_high = self.radio.compute_rho_util(N_high, "eMBB")

        T_eff_low = self.radio.compute_T_eff(rho_low)
        T_eff_high = self.radio.compute_T_eff(rho_high)

        self.assertGreater(T_eff_low, T_eff_high,
                           "More users should yield lower T_eff")

    # ---------------------------------------------------------------
    # Violation rate must not decrease with increasing load
    # ---------------------------------------------------------------

    def test_violations_nondecreasing_with_load(self):
        """Violations should not decrease when offered load increases.

        With more users (higher rho_util), per-user throughput drops,
        so SLO violations should not decrease.
        [CONG_5G_PMC][LTE_LOAD_TPUT]
        """
        sla = SLAModel(self.cfg)
        slo_embb = sla.SLO_T["eMBB"]  # 10 Mbps
        K = 30
        rho_URLLC = 0.3

        N_list = [20, 60, 120, 200, 300]
        violation_rates = []

        for N_embb in N_list:
            avg_T_per_step = []
            for k in range(K):
                # All users have high T_exp (not throttled)
                T_exp = np.full(N_embb, 100.0)
                result = self.radio.inner_step(
                    N_active_eMBB=N_embb,
                    N_active_URLLC=10,
                    rho_URLLC=rho_URLLC,
                    T_exp_users_eMBB=T_exp,
                    T_exp_users_URLLC=np.full(10, 50.0),
                )
                avg_T_per_step.append(result["avg_T_act_eMBB"])

            V = sla.compute_violation_rate(np.array(avg_T_per_step), "eMBB")
            violation_rates.append(V)

        # Violation rate should be non-decreasing as N increases
        for i in range(len(violation_rates) - 1):
            self.assertLessEqual(
                violation_rates[i], violation_rates[i + 1],
                f"Violations decreased when load increased: "
                f"V(N={N_list[i]})={violation_rates[i]:.3f} > "
                f"V(N={N_list[i+1]})={violation_rates[i+1]:.3f}"
            )

    # ---------------------------------------------------------------
    # Per-user throughput capped by T_exp
    # ---------------------------------------------------------------

    def test_T_act_capped_by_T_exp(self):
        """Per-user throughput must not exceed T_exp_user."""
        T_exp = 5.0  # low expected throughput
        T_eff = 1000.0  # high cell capacity
        slice_share = 1.0
        N_slice = 1

        T_act = self.radio.compute_T_act_user(T_eff, slice_share, N_slice, T_exp)
        self.assertLessEqual(T_act, T_exp)

    def test_T_act_no_div_by_zero(self):
        """N_slice=0 must not cause division by zero."""
        T_act = self.radio.compute_T_act_user(
            T_eff=500.0, slice_share=0.5, N_slice=0, T_exp_user=100.0
        )
        self.assertTrue(np.isfinite(T_act))
        self.assertGreaterEqual(T_act, 0.0)


if __name__ == "__main__":
    unittest.main()
