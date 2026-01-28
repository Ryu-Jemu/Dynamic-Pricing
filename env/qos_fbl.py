"""
URLLC QoS Model - Finite Block Length (FBL) Theory

Implements reliability and latency analysis for URLLC services
using Finite Block Length information theory.

Reference:
- Polyanskiy et al. (2010): "Channel Coding Rate in the Finite Blocklength Regime"
  IEEE Transactions on Information Theory, Vol. 56, No. 5
- 3GPP TR 38.913: 5G requirements (URLLC: 1ms latency, 99.999% reliability)
- 3GPP TS 22.261: Service requirements for 5G
"""

import numpy as np
from scipy.stats import norm
from scipy.special import erfinv
from typing import Tuple, Optional


class FBLCapacity:
    """
    Finite Block Length (FBL) capacity calculator.
    
    In URLLC scenarios with short packets, Shannon capacity is not achievable.
    FBL theory provides the achievable rate for finite blocklength transmissions.
    
    Key formula (Normal Approximation):
    R ≈ C - √(V/n) * Q^{-1}(ε)
    
    where:
    - R: Achievable rate (bits/channel use)
    - C: Shannon capacity
    - V: Channel dispersion
    - n: Blocklength (symbols)
    - ε: Target error probability
    - Q^{-1}: Inverse Q-function
    """
    
    @staticmethod
    def shannon_capacity(sinr_linear: float) -> float:
        """
        Calculate Shannon capacity.
        
        C = log2(1 + SINR)
        
        Args:
            sinr_linear: SINR in linear scale
        
        Returns:
            Capacity in bits per channel use
        """
        return np.log2(1 + sinr_linear)
    
    @staticmethod
    def channel_dispersion(sinr_linear: float) -> float:
        """
        Calculate channel dispersion for AWGN channel.
        
        V = (1 - 1/(1+SINR)^2) * (log2(e))^2
        
        Reference: Polyanskiy et al. (2010), Eq. (296)
        
        Args:
            sinr_linear: SINR in linear scale
        
        Returns:
            Channel dispersion
        """
        log2_e = np.log2(np.e)
        return (1 - 1 / (1 + sinr_linear)**2) * (log2_e ** 2)
    
    @staticmethod
    def q_function_inverse(epsilon: float) -> float:
        """
        Calculate inverse Q-function.
        
        Q^{-1}(ε) = √2 * erfinv(1 - 2ε)
        
        Args:
            epsilon: Error probability (0 < ε < 0.5)
        
        Returns:
            Q^{-1}(ε)
        """
        # Use scipy's ppf (percent point function) for inverse normal CDF
        # Q^{-1}(ε) = Φ^{-1}(1-ε) where Φ is standard normal CDF
        return norm.ppf(1 - epsilon)
    
    @staticmethod
    def mimo_diversity_gain(n_tx: int = 4, n_rx: int = 4) -> float:
        """
        Calculate effective MIMO diversity/multiplexing gain for FBL regime.
        
        In the finite blocklength regime, MIMO provides both diversity gain
        (improved reliability) and multiplexing gain (increased rate).
        
        Reference:
            Yang et al. (2014): "Quasi-Static Multiple-Antenna Fading Channels
            at Finite Blocklength", IEEE Trans. Information Theory
            
            - At high SNR: multiplexing gain ≈ min(n_tx, n_rx)
            - At low SNR: diversity gain ≈ n_tx × n_rx
            - Practical gain is between these extremes
        
        Args:
            n_tx: Number of transmit antennas
            n_rx: Number of receive antennas
            
        Returns:
            Effective SINR multiplier for FBL calculations
        """
        # Conservative estimate: sqrt of spatial streams
        # This accounts for practical implementation losses
        # Full multiplexing gain: min(n_tx, n_rx)
        # Practical gain: sqrt of this (accounting for CSI errors, etc.)
        spatial_streams = min(n_tx, n_rx)
        
        # Use sqrt for conservative estimate in FBL regime
        # This is supported by measurements showing ~40-60% of theoretical gain
        return np.sqrt(spatial_streams)
    
    @staticmethod
    def achievable_rate_mimo(
        sinr_linear: float,
        blocklength: int,
        target_error_prob: float,
        n_tx: int = 4,
        n_rx: int = 4
    ) -> float:
        """
        Calculate FBL achievable rate with MIMO diversity/multiplexing gain.
        
        Applies effective SINR scaling based on antenna configuration.
        
        Reference:
            Yang et al. (2014): MIMO FBL capacity analysis
            Collins et al. (2018): Practical MIMO FBL bounds
        
        Args:
            sinr_linear: SINR in linear scale (single antenna equivalent)
            blocklength: Number of channel uses
            target_error_prob: Target BLER
            n_tx: Number of transmit antennas
            n_rx: Number of receive antennas
            
        Returns:
            Achievable rate in bits per channel use
        """
        # Apply MIMO gain to effective SINR
        mimo_gain = FBLCapacity.mimo_diversity_gain(n_tx, n_rx)
        effective_sinr = sinr_linear * mimo_gain
        
        # Calculate FBL rate with effective SINR
        return FBLCapacity.achievable_rate(
            effective_sinr, blocklength, target_error_prob
        )
    
    @staticmethod
    def achievable_rate(
        sinr_linear: float,
        blocklength: int,
        target_error_prob: float
    ) -> float:
        """
        Calculate FBL achievable rate (Normal Approximation).
        
        R ≈ C - √(V/n) * Q^{-1}(ε)
        
        Args:
            sinr_linear: SINR in linear scale
            blocklength: Number of channel uses (symbols)
            target_error_prob: Target block error probability
        
        Returns:
            Achievable rate in bits per channel use (non-negative)
        """
        if blocklength <= 0:
            return 0.0
        
        if sinr_linear <= 0:
            return 0.0
        
        # Shannon capacity
        C = FBLCapacity.shannon_capacity(sinr_linear)
        
        # Channel dispersion
        V = FBLCapacity.channel_dispersion(sinr_linear)
        
        # Q-inverse
        Q_inv = FBLCapacity.q_function_inverse(target_error_prob)
        
        # FBL rate
        rate = C - np.sqrt(V / blocklength) * Q_inv
        
        return max(0.0, rate)
    
    @staticmethod
    def required_blocklength(
        sinr_linear: float,
        rate_required: float,
        target_error_prob: float
    ) -> int:
        """
        Calculate minimum blocklength required to achieve target rate.
        
        Derived from: R = C - √(V/n) * Q^{-1}(ε)
        Solving for n: n = V * (Q^{-1}(ε) / (C - R))^2
        
        Args:
            sinr_linear: SINR in linear scale
            rate_required: Required rate in bits per channel use
            target_error_prob: Target block error probability
        
        Returns:
            Minimum blocklength (channel uses)
        """
        C = FBLCapacity.shannon_capacity(sinr_linear)
        
        if rate_required >= C:
            return np.inf  # Cannot achieve rate higher than capacity
        
        V = FBLCapacity.channel_dispersion(sinr_linear)
        Q_inv = FBLCapacity.q_function_inverse(target_error_prob)
        
        n = V * (Q_inv / (C - rate_required)) ** 2
        
        return int(np.ceil(n))
    
    @staticmethod
    def error_probability(
        sinr_linear: float,
        blocklength: int,
        rate: float
    ) -> float:
        """
        Calculate achievable error probability for given rate.
        
        ε ≈ Q((C - R) * √(n/V))
        
        Args:
            sinr_linear: SINR in linear scale
            blocklength: Blocklength
            rate: Transmission rate in bits per channel use
        
        Returns:
            Block error probability
        """
        if blocklength <= 0:
            return 1.0
        
        C = FBLCapacity.shannon_capacity(sinr_linear)
        V = FBLCapacity.channel_dispersion(sinr_linear)
        
        if V <= 0:
            return 0.0 if rate <= C else 1.0
        
        # Q-function argument
        z = (C - rate) * np.sqrt(blocklength / V)
        
        # Q-function = 1 - Φ(z)
        error_prob = 1 - norm.cdf(z)
        
        return np.clip(error_prob, 0.0, 1.0)


class URLLCQoSModel:
    """
    URLLC Quality of Service model.
    
    Evaluates whether URLLC requirements can be met given
    PRB allocation and channel conditions.
    
    Requirements (3GPP TS 22.261):
    - Latency: 1 ms user plane
    - Reliability: 99.999% (BLER ≤ 10^-5)
    - Packet size: 32 bytes (typical)
    """
    
    def __init__(
        self,
        latency_requirement_ms: float = 1.0,
        reliability_requirement: float = 0.99999,
        packet_size_bytes: int = 32,
        scs_khz: int = 30,
        useful_re_per_prb: int = 148
    ):
        """
        Initialize URLLC QoS model.
        
        Args:
            latency_requirement_ms: Latency budget in ms
            reliability_requirement: Target reliability (e.g., 0.99999)
            packet_size_bytes: Packet size in bytes
            scs_khz: Subcarrier spacing in kHz
            useful_re_per_prb: Useful RE per PRB per slot
        """
        self.latency_requirement_ms = latency_requirement_ms
        self.reliability_requirement = reliability_requirement
        self.target_bler = 1 - reliability_requirement  # e.g., 1e-5
        self.packet_size_bytes = packet_size_bytes
        self.packet_size_bits = packet_size_bytes * 8
        self.scs_khz = scs_khz
        self.useful_re_per_prb = useful_re_per_prb
        
        # Slot duration (ms)
        mu = int(np.log2(scs_khz / 15))
        self.slot_duration_ms = 1.0 / (2 ** mu)
        
        # Maximum slots within latency budget
        self.max_slots = int(latency_requirement_ms / self.slot_duration_ms)
        
        # FBL calculator
        self.fbl = FBLCapacity()
    
    def calculate_blocklength(self, n_prb: int, n_slots: int = 1) -> int:
        """
        Calculate total blocklength (symbols) for given PRB allocation.
        
        Args:
            n_prb: Number of PRBs allocated
            n_slots: Number of slots used
        
        Returns:
            Total blocklength (channel uses)
        """
        return self.useful_re_per_prb * n_prb * n_slots
    
    def calculate_required_rate(self, blocklength: int) -> float:
        """
        Calculate required rate to transmit packet within blocklength.
        
        Args:
            blocklength: Available channel uses
        
        Returns:
            Required rate in bits per channel use
        """
        if blocklength <= 0:
            return np.inf
        return self.packet_size_bits / blocklength
    
    def evaluate_qos(
        self,
        sinr_db: float,
        allocated_prb: int,
        n_slots: int = 1
    ) -> Tuple[bool, float, dict]:
        """
        Evaluate if URLLC QoS requirements can be met.
        
        Args:
            sinr_db: User's SINR in dB
            allocated_prb: Number of PRBs allocated
            n_slots: Number of transmission slots
        
        Returns:
            Tuple of (qos_satisfied, violation_probability, details)
        """
        sinr_linear = 10 ** (sinr_db / 10)
        
        # Check latency constraint
        actual_latency_ms = n_slots * self.slot_duration_ms
        latency_ok = actual_latency_ms <= self.latency_requirement_ms
        
        # Calculate blocklength
        blocklength = self.calculate_blocklength(allocated_prb, n_slots)
        
        if blocklength <= 0:
            return False, 1.0, {
                "latency_ok": latency_ok,
                "reliability_ok": False,
                "achievable_bler": 1.0,
                "target_bler": self.target_bler,
                "blocklength": 0,
                "required_rate": np.inf,
                "achievable_rate": 0.0
            }
        
        # Required rate to transmit packet
        required_rate = self.calculate_required_rate(blocklength)
        
        # Achievable rate with FBL
        achievable_rate = self.fbl.achievable_rate(
            sinr_linear, blocklength, self.target_bler
        )
        
        # Calculate actual BLER at required rate
        if achievable_rate >= required_rate:
            # Can achieve target BLER
            achievable_bler = self.target_bler
        else:
            # Calculate BLER at required rate
            achievable_bler = self.fbl.error_probability(
                sinr_linear, blocklength, required_rate
            )
        
        reliability_ok = achievable_bler <= self.target_bler
        qos_satisfied = latency_ok and reliability_ok
        
        # Violation probability is the gap from target
        if qos_satisfied:
            violation_prob = achievable_bler
        else:
            # QoS violated - return actual BLER as violation metric
            violation_prob = achievable_bler
        
        details = {
            "latency_ok": latency_ok,
            "actual_latency_ms": actual_latency_ms,
            "reliability_ok": reliability_ok,
            "achievable_bler": achievable_bler,
            "target_bler": self.target_bler,
            "blocklength": blocklength,
            "required_rate": required_rate,
            "achievable_rate": achievable_rate,
            "sinr_db": sinr_db,
            "allocated_prb": allocated_prb
        }
        
        return qos_satisfied, violation_prob, details
    
    def get_minimum_prb(
        self,
        sinr_db: float,
        n_slots: int = 1
    ) -> int:
        """
        Calculate minimum PRBs needed to satisfy QoS.
        
        Args:
            sinr_db: User's SINR in dB
            n_slots: Number of slots available
        
        Returns:
            Minimum number of PRBs required
        """
        sinr_linear = 10 ** (sinr_db / 10)
        
        # Calculate required rate first
        # Then find blocklength that achieves this rate with target BLER
        required_blocklength = self.fbl.required_blocklength(
            sinr_linear, 
            self.packet_size_bits / 1000,  # Approximate rate
            self.target_bler
        )
        
        # Convert to PRBs
        re_per_slot = self.useful_re_per_prb * n_slots
        if re_per_slot <= 0:
            return np.inf
        
        min_prb = int(np.ceil(required_blocklength / re_per_slot))
        
        # Iteratively verify
        for n_prb in range(1, 100):
            satisfied, _, _ = self.evaluate_qos(sinr_db, n_prb, n_slots)
            if satisfied:
                return n_prb
        
        return 100  # Cap at reasonable maximum


class URLLCViolationCalculator:
    """
    Calculate URLLC QoS violation probabilities for resource allocation decisions.
    """
    
    def __init__(self, qos_model: URLLCQoSModel):
        self.qos_model = qos_model
    
    def batch_evaluate(
        self,
        sinr_dbs: np.ndarray,
        allocated_prbs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate QoS for multiple users.
        
        Args:
            sinr_dbs: Array of SINR values in dB
            allocated_prbs: Array of PRB allocations
        
        Returns:
            Tuple of (qos_satisfied_array, violation_probs_array)
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


if __name__ == "__main__":
    print("=" * 60)
    print("URLLC QoS Model - FBL Theory Test")
    print("=" * 60)
    
    # Create QoS model
    qos_model = URLLCQoSModel(
        latency_requirement_ms=1.0,
        reliability_requirement=0.99999,
        packet_size_bytes=32
    )
    
    print(f"\nConfiguration:")
    print(f"  Latency requirement: {qos_model.latency_requirement_ms} ms")
    print(f"  Reliability requirement: {qos_model.reliability_requirement}")
    print(f"  Target BLER: {qos_model.target_bler}")
    print(f"  Packet size: {qos_model.packet_size_bytes} bytes")
    print(f"  Slot duration: {qos_model.slot_duration_ms} ms")
    print(f"  Max slots within latency: {qos_model.max_slots}")
    
    print("\nQoS Evaluation vs SINR and PRB:")
    for sinr_db in [5, 10, 15, 20, 25]:
        print(f"\n  SINR = {sinr_db} dB:")
        for n_prb in [1, 2, 3, 4]:
            satisfied, viol_prob, details = qos_model.evaluate_qos(sinr_db, n_prb)
            status = "✓" if satisfied else "✗"
            print(f"    {n_prb} PRB: {status} BLER={details['achievable_bler']:.2e}")
    
    print("\nMinimum PRB Requirements:")
    for sinr_db in [5, 10, 15, 20, 25, 30]:
        min_prb = qos_model.get_minimum_prb(sinr_db)
        print(f"  SINR={sinr_db:2d}dB: min_PRB={min_prb}")
