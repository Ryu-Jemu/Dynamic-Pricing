"""
eMBB QoS Model - Throughput-Based

Implements throughput calculation and QoS evaluation for eMBB services.

Reference:
- 3GPP TS 38.214: Physical layer procedures for data
- 3GPP TS 22.261: Service requirements (eMBB: 100 Mbps DL experienced)
- 3GPP TR 38.913: Scenarios and requirements (Table 7.1)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# 3GPP TS 38.214 Table 5.1.3.1-1: MCS Index Table 1 for PDSCH
# (SINR range, Modulation, Code Rate, Spectral Efficiency)
MCS_TABLE_64QAM = [
    # (sinr_min, sinr_max, modulation, code_rate, spectral_efficiency)
    (-10.0, -5.0, "QPSK", 0.08, 0.15),
    (-5.0, -2.0, "QPSK", 0.12, 0.23),
    (-2.0, 0.0, "QPSK", 0.19, 0.38),
    (0.0, 2.0, "QPSK", 0.30, 0.60),
    (2.0, 4.0, "QPSK", 0.44, 0.88),
    (4.0, 6.0, "QPSK", 0.59, 1.18),
    (6.0, 8.0, "16QAM", 0.37, 1.48),
    (8.0, 10.0, "16QAM", 0.48, 1.91),
    (10.0, 12.0, "16QAM", 0.60, 2.41),
    (12.0, 14.0, "64QAM", 0.46, 2.73),
    (14.0, 16.0, "64QAM", 0.50, 3.02),
    (16.0, 18.0, "64QAM", 0.55, 3.32),
    (18.0, 20.0, "64QAM", 0.60, 3.61),
    (20.0, 22.0, "64QAM", 0.65, 3.90),
    (22.0, 24.0, "64QAM", 0.75, 4.52),
    (24.0, 26.0, "64QAM", 0.85, 5.12),
    (26.0, 28.0, "64QAM", 0.93, 5.55),
]

# Extended MCS table with 256QAM (Table 5.1.3.1-2)
MCS_TABLE_256QAM = [
    # Higher SINR entries with 256QAM
    (24.0, 26.0, "256QAM", 0.57, 4.52),
    (26.0, 28.0, "256QAM", 0.63, 5.03),
    (28.0, 30.0, "256QAM", 0.70, 5.55),
    (30.0, 32.0, "256QAM", 0.78, 6.23),
    (32.0, 34.0, "256QAM", 0.85, 6.79),
    (34.0, 40.0, "256QAM", 0.93, 7.41),
]


class SpectralEfficiencyCalculator:
    """
    Calculate spectral efficiency based on SINR and MCS mapping.
    """
    
    def __init__(self, enable_256qam: bool = True):
        """
        Initialize SE calculator.
        
        Args:
            enable_256qam: Whether to use 256QAM for high SINR
        """
        self.enable_256qam = enable_256qam
        
        # Build combined MCS table
        self.mcs_table = MCS_TABLE_64QAM.copy()
        if enable_256qam:
            # Replace high SINR entries with 256QAM
            self.mcs_table = [
                entry for entry in self.mcs_table if entry[0] < 24.0
            ]
            self.mcs_table.extend(MCS_TABLE_256QAM)
    
    def get_spectral_efficiency(self, sinr_db: float) -> Tuple[float, str, float]:
        """
        Get spectral efficiency for given SINR.
        
        Args:
            sinr_db: SINR in dB
        
        Returns:
            Tuple of (spectral_efficiency, modulation, code_rate)
        """
        # Find matching MCS entry
        for sinr_min, sinr_max, mod, cr, se in self.mcs_table:
            if sinr_min <= sinr_db < sinr_max:
                return se, mod, cr
        
        # Handle edge cases
        if sinr_db < self.mcs_table[0][0]:
            entry = self.mcs_table[0]
            return entry[4], entry[2], entry[3]
        else:
            entry = self.mcs_table[-1]
            return entry[4], entry[2], entry[3]
    
    def get_practical_se(
        self, 
        sinr_db: float, 
        efficiency_factor: float = 0.75
    ) -> float:
        """
        Get practical spectral efficiency with overhead factor.
        
        Args:
            sinr_db: SINR in dB
            efficiency_factor: Factor to account for overhead (default 0.75)
        
        Returns:
            Practical spectral efficiency in bits/symbol
        """
        se, _, _ = self.get_spectral_efficiency(sinr_db)
        return se * efficiency_factor


class EMMBQoSModel:
    """
    eMBB Quality of Service model.
    
    Evaluates throughput-based QoS for eMBB services.
    
    Requirements (3GPP TR 38.913):
    - Experienced data rate: 100 Mbps DL (urban)
    - User plane latency: 4 ms
    - Reliability: 99.9% (for general service)
    """
    
    def __init__(
        self,
        throughput_requirement_mbps: float = 50.0,
        latency_tolerance_ms: float = 100.0,
        reliability_requirement: float = 0.999,
        scs_khz: int = 30,
        useful_re_per_prb: int = 148,
        enable_256qam: bool = True
    ):
        """
        Initialize eMBB QoS model.
        
        Args:
            throughput_requirement_mbps: Required DL throughput in Mbps
            latency_tolerance_ms: Maximum acceptable latency in ms
            reliability_requirement: Target reliability
            scs_khz: Subcarrier spacing in kHz
            useful_re_per_prb: Useful RE per PRB per slot
            enable_256qam: Enable 256QAM for high SINR
        """
        self.throughput_requirement_mbps = throughput_requirement_mbps
        self.latency_tolerance_ms = latency_tolerance_ms
        self.reliability_requirement = reliability_requirement
        self.scs_khz = scs_khz
        self.useful_re_per_prb = useful_re_per_prb
        
        # Slot duration
        mu = int(np.log2(scs_khz / 15))
        self.slot_duration_ms = 1.0 / (2 ** mu)
        self.slots_per_second = int(1000 / self.slot_duration_ms)
        
        # Spectral efficiency calculator
        self.se_calc = SpectralEfficiencyCalculator(enable_256qam)
        
        # PRB bandwidth
        self.prb_bandwidth_hz = 12 * scs_khz * 1000  # 12 subcarriers × SCS
    
    def calculate_throughput_mbps(
        self,
        sinr_db: float,
        allocated_prb: int,
        efficiency_factor: float = 0.75
    ) -> float:
        """
        Calculate achievable throughput in Mbps.
        
        Throughput = SE × PRB_BW × N_PRB × (1 - overhead)
        
        Args:
            sinr_db: SINR in dB
            allocated_prb: Number of PRBs allocated
            efficiency_factor: Overhead factor
        
        Returns:
            Throughput in Mbps
        """
        if allocated_prb <= 0:
            return 0.0
        
        # Get spectral efficiency
        se = self.se_calc.get_practical_se(sinr_db, efficiency_factor)
        
        # Calculate throughput
        # Throughput = SE (bits/symbol) × Symbols per second
        # Symbols per second = RE per PRB × Slots per second × N_PRB
        re_per_second = self.useful_re_per_prb * self.slots_per_second * allocated_prb
        throughput_bps = se * re_per_second
        throughput_mbps = throughput_bps / 1e6
        
        return throughput_mbps
    
    def evaluate_qos(
        self,
        sinr_db: float,
        allocated_prb: int
    ) -> Tuple[bool, float, Dict]:
        """
        Evaluate if eMBB QoS requirements are met.
        
        Args:
            sinr_db: User's SINR in dB
            allocated_prb: Number of PRBs allocated
        
        Returns:
            Tuple of (qos_satisfied, violation_metric, details)
        """
        # Calculate achievable throughput
        throughput_mbps = self.calculate_throughput_mbps(sinr_db, allocated_prb)
        
        # Check if requirement is met
        throughput_ok = throughput_mbps >= self.throughput_requirement_mbps
        
        # Violation metric: ratio of shortfall (0 if satisfied, positive otherwise)
        if throughput_ok:
            violation_metric = 0.0
        else:
            # Normalized shortfall
            violation_metric = (
                (self.throughput_requirement_mbps - throughput_mbps) /
                self.throughput_requirement_mbps
            )
            violation_metric = np.clip(violation_metric, 0.0, 1.0)
        
        # Get MCS info
        se, mod, cr = self.se_calc.get_spectral_efficiency(sinr_db)
        
        details = {
            "throughput_ok": throughput_ok,
            "achieved_throughput_mbps": throughput_mbps,
            "required_throughput_mbps": self.throughput_requirement_mbps,
            "throughput_ratio": throughput_mbps / max(self.throughput_requirement_mbps, 1e-6),
            "spectral_efficiency": se,
            "modulation": mod,
            "code_rate": cr,
            "sinr_db": sinr_db,
            "allocated_prb": allocated_prb
        }
        
        return throughput_ok, violation_metric, details
    
    def get_minimum_prb(
        self,
        sinr_db: float
    ) -> int:
        """
        Calculate minimum PRBs needed to satisfy throughput requirement.
        
        Args:
            sinr_db: User's SINR in dB
        
        Returns:
            Minimum number of PRBs required
        """
        # Binary search for minimum PRB
        for n_prb in range(1, 100):
            satisfied, _, _ = self.evaluate_qos(sinr_db, n_prb)
            if satisfied:
                return n_prb
        
        return 100  # Cap


class EMMBViolationCalculator:
    """
    Calculate eMBB QoS violations for resource allocation.
    """
    
    def __init__(self, qos_model: EMMBQoSModel):
        self.qos_model = qos_model
    
    def batch_evaluate(
        self,
        sinr_dbs: np.ndarray,
        allocated_prbs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate QoS for multiple users.
        
        Args:
            sinr_dbs: Array of SINR values
            allocated_prbs: Array of PRB allocations
        
        Returns:
            Tuple of (satisfied_array, violations_array)
        """
        n_users = len(sinr_dbs)
        satisfied = np.zeros(n_users, dtype=bool)
        violations = np.zeros(n_users)
        
        for i in range(n_users):
            sat, viol, _ = self.qos_model.evaluate_qos(
                sinr_dbs[i], int(allocated_prbs[i])
            )
            satisfied[i] = sat
            violations[i] = viol
        
        return satisfied, violations


class UserThroughputTracker:
    """
    Track throughput history for a single user for Proportional Fair scheduling.
    
    This is a per-user tracker; create one instance per user.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of samples for moving average
        """
        self.window_size = window_size
        self.history: List[float] = []
    
    def update(self, throughput: float) -> None:
        """Record throughput sample."""
        self.history.append(throughput)
        
        # Keep window size
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_average(self, default: float = 1.0) -> float:
        """Get average throughput."""
        if len(self.history) == 0:
            return default
        return np.mean(self.history)
    
    def reset(self) -> None:
        """Clear history."""
        self.history.clear()


if __name__ == "__main__":
    print("=" * 60)
    print("eMBB QoS Model - Throughput Test")
    print("=" * 60)
    
    # Create QoS model
    qos_model = EMMBQoSModel(
        throughput_requirement_mbps=50.0,
        scs_khz=30
    )
    
    print(f"\nConfiguration:")
    print(f"  Throughput requirement: {qos_model.throughput_requirement_mbps} Mbps")
    print(f"  SCS: {qos_model.scs_khz} kHz")
    print(f"  Slot duration: {qos_model.slot_duration_ms} ms")
    print(f"  Slots per second: {qos_model.slots_per_second}")
    
    print("\nSpectral Efficiency vs SINR:")
    se_calc = SpectralEfficiencyCalculator()
    for sinr in [-5, 0, 5, 10, 15, 20, 25, 30, 35]:
        se, mod, cr = se_calc.get_spectral_efficiency(sinr)
        print(f"  {sinr:3d} dB: SE={se:.2f} bits/sym ({mod}, R={cr:.2f})")
    
    print("\nThroughput vs SINR and PRB:")
    for sinr_db in [5, 10, 15, 20, 25]:
        print(f"\n  SINR = {sinr_db} dB:")
        for n_prb in [1, 2, 5, 10, 20]:
            satisfied, viol, details = qos_model.evaluate_qos(sinr_db, n_prb)
            status = "✓" if satisfied else "✗"
            tput = details["achieved_throughput_mbps"]
            print(f"    {n_prb:2d} PRB: {status} {tput:.1f} Mbps")
    
    print("\nMinimum PRB for 50 Mbps:")
    for sinr_db in [5, 10, 15, 20, 25, 30]:
        min_prb = qos_model.get_minimum_prb(sinr_db)
        print(f"  SINR={sinr_db:2d}dB: min_PRB={min_prb}")
