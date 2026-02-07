"""
Radio capacity proxy and congestion model.

Implements a macro-level capacity abstraction (not full PHY MCS/TBS chain).
Full PHY mapping requires TS 38.214 MCS/TBS chain; we use a monotone
degradation proxy instead, documented as a simplification.

Key equations (Section 10):
  10.1  rho_util_k = clip(c_load * (N_active_k / N_ref), 0.0, 0.98)
  10.2  T_eff_k    = T_cell_cap_mbps * (1 - rho_util_k)       [monotone]
  10.3  T_act_u,k  = min(T_eff_k * share_s / max(N_s, 1), T_exp_u,k)

Monotonicity: increasing utilization must NOT increase T_eff.
  [CONG_5G_PMC][LTE_LOAD_TPUT]

Standards context:
  [TS38104] bandwidth=100MHz, SCS=30kHz → PRB_total=273
  [TS38214][TSDI_KEYNOTE] PHY throughput depends on MCS/TBS chain (not implemented)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class RadioConfig:
    """Immutable radio configuration derived from YAML config.

    Standards-pinned:
      bandwidth_mhz = 100   [TS38104]
      scs_khz       = 30    [TS38104]
      prb_total     = 273   [TS38104][TSDI_KEYNOTE]

    Scenario parameters:
      T_cell_cap_mbps, c_load, N_ref_*, rho_util_max
    """

    # Standards-pinned [TS38104]
    bandwidth_mhz: int = 100
    scs_khz: int = 30
    prb_total: int = 273

    # Scenario / calibratable
    T_cell_cap_mbps: float = 1000.0
    c_load: float = 1.0
    N_ref_eMBB: float = 200.0
    N_ref_URLLC: float = 100.0
    rho_util_max: float = 0.98

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RadioConfig":
        """Build RadioConfig from full YAML config dict."""
        radio = cfg.get("radio", {})
        rm = cfg.get("radio_model", {})
        return cls(
            bandwidth_mhz=radio.get("bandwidth_mhz", 100),
            scs_khz=radio.get("scs_khz", 30),
            prb_total=radio.get("prb_total", 273),
            T_cell_cap_mbps=rm.get("T_cell_cap_mbps", 1000.0),
            c_load=rm.get("c_load", 1.0),
            N_ref_eMBB=rm.get("N_ref_eMBB", 200.0),
            N_ref_URLLC=rm.get("N_ref_URLLC", 100.0),
            rho_util_max=rm.get("rho_util_max", 0.98),
        )


class RadioModel:
    """Macro-level radio capacity model for a single NR cell.

    This is a *proxy* model, not a full PHY simulation.
    See module docstring for equations and references.
    """

    def __init__(self, config: RadioConfig) -> None:
        self.cfg = config

    # -----------------------------------------------------------------
    # 10.1  Offered load → utilization  [CONG_5G_PMC][LTE_LOAD_TPUT]
    # -----------------------------------------------------------------

    def compute_rho_util(
        self,
        N_active: int,
        slice_name: str,
    ) -> float:
        """Compute slice-level utilization from number of active users.

        rho_util = clip(c_load * N_active / N_ref, 0.0, rho_util_max)

        Monotone in N_active by construction.
        """
        if slice_name == "eMBB":
            n_ref = self.cfg.N_ref_eMBB
        elif slice_name == "URLLC":
            n_ref = self.cfg.N_ref_URLLC
        else:
            raise ValueError(f"Unknown slice: {slice_name}")

        # Guard against n_ref == 0
        if n_ref <= 0:
            n_ref = 1.0

        rho = self.cfg.c_load * (N_active / n_ref)
        rho = float(np.clip(rho, 0.0, self.cfg.rho_util_max))
        return rho

    # -----------------------------------------------------------------
    # 10.2  Effective cell capacity under congestion  [CONG_5G_PMC]
    # -----------------------------------------------------------------

    def compute_T_eff(self, rho_util: float) -> float:
        """Effective cell throughput at given utilization.

        T_eff = T_cell_cap_mbps * (1 - rho_util)

        MONOTONICITY GUARANTEE: ∂T_eff/∂rho_util < 0 always.
        This is the core congestion degradation model.
        """
        rho_util = float(np.clip(rho_util, 0.0, self.cfg.rho_util_max))
        t_eff = self.cfg.T_cell_cap_mbps * (1.0 - rho_util)
        return max(t_eff, 0.0)

    # -----------------------------------------------------------------
    # 10.3  Per-user delivered throughput proxy
    # -----------------------------------------------------------------

    def compute_T_act_user(
        self,
        T_eff: float,
        slice_share: float,
        N_slice: int,
        T_exp_user: float,
    ) -> float:
        """Per-user delivered throughput at one inner step.

        T_act_u,k = min(T_eff * share_s / max(N_s, 1), T_exp_u,k)

        Parameters
        ----------
        T_eff : float
            Effective cell capacity (Mbps) at this inner step.
        slice_share : float
            PRB share allocated to this slice (0..1).
        N_slice : int
            Number of active users in this slice.
        T_exp_user : float
            User's expected / maximum throughput (Mbps).
            (May be throttled by data cap, Section 11.)

        Returns
        -------
        float : Delivered throughput (Mbps), non-negative.
        """
        n = max(N_slice, 1)  # prevent div-by-zero [SB3_TIPS]
        fair_share = T_eff * slice_share / n
        t_act = min(fair_share, T_exp_user)
        return max(t_act, 0.0)

    # -----------------------------------------------------------------
    # Combined inner-step computation (convenience)
    # -----------------------------------------------------------------

    def inner_step(
        self,
        N_active_eMBB: int,
        N_active_URLLC: int,
        rho_URLLC: float,
        T_exp_users_eMBB: np.ndarray,
        T_exp_users_URLLC: np.ndarray,
    ) -> Dict[str, Any]:
        """Run one inner-step k of the radio model.

        Parameters
        ----------
        N_active_eMBB, N_active_URLLC : int
            Active user counts per slice.
        rho_URLLC : float
            URLLC PRB share (action-controlled).
        T_exp_users_eMBB, T_exp_users_URLLC : ndarray shape (N_s,)
            Per-user expected throughput arrays (Mbps).

        Returns
        -------
        dict with keys:
          rho_util_eMBB, rho_util_URLLC : float
          T_eff_eMBB, T_eff_URLLC : float  (Mbps)
          T_act_eMBB, T_act_URLLC : ndarray (Mbps)  per-user
          avg_T_act_eMBB, avg_T_act_URLLC : float
        """
        rho_eMBB_share = 1.0 - rho_URLLC

        # Utilization (Section 10.1)
        rho_util_eMBB = self.compute_rho_util(N_active_eMBB, "eMBB")
        rho_util_URLLC = self.compute_rho_util(N_active_URLLC, "URLLC")

        # Effective capacity (Section 10.2)
        T_eff_eMBB = self.compute_T_eff(rho_util_eMBB)
        T_eff_URLLC = self.compute_T_eff(rho_util_URLLC)

        # Per-user throughput (Section 10.3)
        T_act_eMBB = np.array([
            self.compute_T_act_user(T_eff_eMBB, rho_eMBB_share,
                                    N_active_eMBB, t_exp)
            for t_exp in T_exp_users_eMBB
        ], dtype=np.float64) if N_active_eMBB > 0 else np.array([], dtype=np.float64)

        T_act_URLLC = np.array([
            self.compute_T_act_user(T_eff_URLLC, rho_URLLC,
                                    N_active_URLLC, t_exp)
            for t_exp in T_exp_users_URLLC
        ], dtype=np.float64) if N_active_URLLC > 0 else np.array([], dtype=np.float64)

        avg_eMBB = float(T_act_eMBB.mean()) if len(T_act_eMBB) > 0 else 0.0
        avg_URLLC = float(T_act_URLLC.mean()) if len(T_act_URLLC) > 0 else 0.0

        return {
            "rho_util_eMBB": rho_util_eMBB,
            "rho_util_URLLC": rho_util_URLLC,
            "T_eff_eMBB": T_eff_eMBB,
            "T_eff_URLLC": T_eff_URLLC,
            "T_act_eMBB": T_act_eMBB,
            "T_act_URLLC": T_act_URLLC,
            "avg_T_act_eMBB": avg_eMBB,
            "avg_T_act_URLLC": avg_URLLC,
        }
