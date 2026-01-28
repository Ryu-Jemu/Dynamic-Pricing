"""
3GPP TR 38.901 Channel Model - UMi-Street Canyon

Implements path loss, shadowing, and SINR calculations for 5G NR
based on 3GPP TR 38.901 V17.0.0 (2022-03).

Reference:
- 3GPP TR 38.901: "Study on channel model for frequencies from 0.5 to 100 GHz"
  Section 7.4.1: UMi-Street Canyon scenario
  Table 7.4.1-1: Path loss models
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm


class ChannelModel3GPP38901:
    """
    3GPP TR 38.901 UMi-Street Canyon channel model.
    
    Implements:
    - LOS probability (Table 7.4.2-1)
    - Path loss (Table 7.4.1-1)
    - Shadowing (Table 7.4.1-1)
    - O2I penetration loss (optional)
    """
    
    def __init__(
        self,
        frequency_ghz: float = 3.5,
        h_bs: float = 10.0,  # BS height in meters
        h_ut: float = 1.5,   # UE height in meters
        scenario: str = "UMi-Street Canyon"
    ):
        """
        Initialize channel model.
        
        Args:
            frequency_ghz: Carrier frequency in GHz (0.5-100 GHz supported)
            h_bs: Base station antenna height in meters
            h_ut: User terminal antenna height in meters
            scenario: Channel scenario (currently only UMi-Street Canyon)
        """
        self.frequency_ghz = frequency_ghz
        self.frequency_hz = frequency_ghz * 1e9
        self.h_bs = h_bs
        self.h_ut = h_ut
        self.scenario = scenario
        
        # Effective antenna heights for UMi (Table 7.4.1-1, Note 1)
        self.h_e = 1.0  # Effective environment height
        self.h_bs_eff = self.h_bs - self.h_e
        self.h_ut_eff = self.h_ut - self.h_e
        
        # Breakpoint distance (Table 7.4.1-1)
        self.d_bp = self._calculate_breakpoint_distance()
        
        # Shadowing standard deviations (Table 7.4.1-1)
        self.sigma_los = 4.0   # LOS shadowing std [dB]
        self.sigma_nlos = 7.82  # NLOS shadowing std [dB]
    
    def _calculate_breakpoint_distance(self) -> float:
        """
        Calculate breakpoint distance for UMi.
        
        d'_BP = 4 * h'_BS * h'_UT * fc / c
        
        Reference: 3GPP TR 38.901 Table 7.4.1-1, Note 1
        """
        c = 3e8  # Speed of light
        d_bp_prime = (
            4 * self.h_bs_eff * self.h_ut_eff * 
            self.frequency_hz / c
        )
        return d_bp_prime
    
    def calculate_los_probability(self, d_2d: float) -> float:
        """
        Calculate LOS probability for UMi-Street Canyon.
        
        Reference: 3GPP TR 38.901 Table 7.4.2-1
        
        P_LOS = 1                                    if d_2D <= 18m
              = (18/d_2D) + exp(-d_2D/36)*(1-18/d_2D) otherwise
        
        Args:
            d_2d: 2D distance in meters
        
        Returns:
            LOS probability [0, 1]
        """
        if d_2d <= 18.0:
            return 1.0
        else:
            return (18.0 / d_2d) + np.exp(-d_2d / 36.0) * (1.0 - 18.0 / d_2d)
    
    def calculate_path_loss_los(self, d_3d: float, d_2d: float) -> float:
        """
        Calculate LOS path loss for UMi-Street Canyon.
        
        Reference: 3GPP TR 38.901 Table 7.4.1-1
        
        PL_UMi-LOS = PL1 if 10m <= d_2D <= d'_BP
                   = PL2 if d'_BP < d_2D <= 5000m
        
        PL1 = 32.4 + 21*log10(d_3D) + 20*log10(fc)
        PL2 = 32.4 + 40*log10(d_3D) + 20*log10(fc) - 9.5*log10(d'^2_BP + (h_BS-h_UT)^2)
        
        Args:
            d_3d: 3D distance in meters
            d_2d: 2D distance in meters
        
        Returns:
            Path loss in dB
        """
        fc_ghz = self.frequency_ghz
        
        if d_2d <= self.d_bp:
            # PL1: Before breakpoint
            pl = 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz)
        else:
            # PL2: After breakpoint
            h_diff = self.h_bs - self.h_ut
            pl = (32.4 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz) -
                  9.5 * np.log10(self.d_bp**2 + h_diff**2))
        
        return pl
    
    def calculate_path_loss_nlos(self, d_3d: float, d_2d: float) -> float:
        """
        Calculate NLOS path loss for UMi-Street Canyon.
        
        Reference: 3GPP TR 38.901 Table 7.4.1-1
        
        PL_UMi-NLOS = max(PL_UMi-LOS, PL'_UMi-NLOS)
        
        PL'_UMi-NLOS = 35.3*log10(d_3D) + 22.4 + 21.3*log10(fc) - 0.3*(h_UT - 1.5)
        
        Args:
            d_3d: 3D distance in meters
            d_2d: 2D distance in meters
        
        Returns:
            Path loss in dB
        """
        fc_ghz = self.frequency_ghz
        
        # NLOS path loss formula
        pl_nlos_prime = (
            35.3 * np.log10(d_3d) + 22.4 + 
            21.3 * np.log10(fc_ghz) - 
            0.3 * (self.h_ut - 1.5)
        )
        
        # LOS path loss (for comparison)
        pl_los = self.calculate_path_loss_los(d_3d, d_2d)
        
        # NLOS = max(LOS, NLOS')
        return max(pl_los, pl_nlos_prime)
    
    def calculate_path_loss(
        self, 
        d_2d: float, 
        los_state: Optional[bool] = None
    ) -> Tuple[float, bool, float]:
        """
        Calculate path loss with probabilistic LOS/NLOS.
        
        Args:
            d_2d: 2D distance in meters (horizontal)
            los_state: Override LOS state (None = probabilistic)
        
        Returns:
            Tuple of (path_loss_db, is_los, shadowing_db)
        """
        # Minimum distance
        d_2d = max(d_2d, 10.0)
        
        # 3D distance
        h_diff = self.h_bs - self.h_ut
        d_3d = np.sqrt(d_2d**2 + h_diff**2)
        
        # Determine LOS state
        if los_state is None:
            p_los = self.calculate_los_probability(d_2d)
            is_los = np.random.random() < p_los
        else:
            is_los = los_state
        
        # Calculate path loss
        if is_los:
            path_loss = self.calculate_path_loss_los(d_3d, d_2d)
            sigma = self.sigma_los
        else:
            path_loss = self.calculate_path_loss_nlos(d_3d, d_2d)
            sigma = self.sigma_nlos
        
        # Add shadowing
        shadowing = np.random.normal(0, sigma)
        
        return path_loss, is_los, shadowing
    
    def calculate_sinr(
        self,
        d_2d: float,
        tx_power_dbm: float = 43.0,
        bandwidth_hz: float = 20e6,
        noise_figure_db: float = 7.0,
        interference_dbm: float = -np.inf,
        los_state: Optional[bool] = None,
        include_shadowing: bool = True
    ) -> Tuple[float, bool]:
        """
        Calculate SINR for a user.
        
        SINR = P_rx / (N + I)
        
        Args:
            d_2d: 2D distance in meters
            tx_power_dbm: Transmit power in dBm
            bandwidth_hz: System bandwidth in Hz
            noise_figure_db: Receiver noise figure in dB
            interference_dbm: Total interference in dBm (optional)
            los_state: Override LOS state
            include_shadowing: If False, exclude shadow fading (deterministic mode)
        
        Returns:
            Tuple of (sinr_db, is_los)
        """
        # Get path loss and shadowing
        path_loss_db, is_los, shadowing_db = self.calculate_path_loss(d_2d, los_state)
        
        # Apply shadowing only if enabled
        if not include_shadowing:
            shadowing_db = 0.0
        
        # Received power
        rx_power_dbm = tx_power_dbm - path_loss_db - shadowing_db
        
        # Thermal noise power
        thermal_noise_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz)
        
        # Total noise power (noise + interference)
        noise_power_dbm = thermal_noise_dbm + noise_figure_db
        
        if interference_dbm > -np.inf:
            # Add interference
            noise_linear = 10 ** (noise_power_dbm / 10)
            interference_linear = 10 ** (interference_dbm / 10)
            total_noise_dbm = 10 * np.log10(noise_linear + interference_linear)
        else:
            total_noise_dbm = noise_power_dbm
        
        # SINR
        sinr_db = rx_power_dbm - total_noise_dbm
        
        # Clip to realistic range
        sinr_db = np.clip(sinr_db, -10.0, 40.0)
        
        return sinr_db, is_los


class UserChannel:
    """
    Encapsulates channel state for a single user.
    """
    
    def __init__(
        self,
        user_id: int,
        distance_m: float,
        channel_model: ChannelModel3GPP38901,
        tx_power_dbm: float = 43.0,
        bandwidth_hz: float = 20e6,
        noise_figure_db: float = 7.0
    ):
        self.user_id = user_id
        self.distance_m = distance_m
        self.channel_model = channel_model
        self.tx_power_dbm = tx_power_dbm
        self.bandwidth_hz = bandwidth_hz
        self.noise_figure_db = noise_figure_db
        
        # Current channel state
        self.sinr_db: float = 0.0
        self.is_los: bool = True
        self.path_loss_db: float = 0.0
        self.shadowing_db: float = 0.0
        
        # Update channel
        self.update_channel()
    
    def update_channel(self) -> None:
        """Update channel state (call each time slot or as needed)."""
        self.sinr_db, self.is_los = self.channel_model.calculate_sinr(
            d_2d=self.distance_m,
            tx_power_dbm=self.tx_power_dbm,
            bandwidth_hz=self.bandwidth_hz,
            noise_figure_db=self.noise_figure_db
        )
    
    def get_sinr_linear(self) -> float:
        """Get SINR in linear scale."""
        return 10 ** (self.sinr_db / 10)


def generate_user_position(
    cell_radius_m: float = 200.0,
    min_distance_m: float = 10.0
) -> Tuple[float, float]:
    """
    Generate random user position with uniform distribution in cell.
    
    Uses area-weighted sampling for uniform spatial distribution.
    
    Args:
        cell_radius_m: Cell radius in meters
        min_distance_m: Minimum distance from BS
    
    Returns:
        Tuple of (distance_m, angle_rad)
    """
    # Uniform distribution in disk requires r^2 sampling
    r_min_sq = (min_distance_m / cell_radius_m) ** 2
    r_normalized = np.sqrt(np.random.uniform(r_min_sq, 1.0))
    distance_m = r_normalized * cell_radius_m
    
    angle_rad = np.random.uniform(0, 2 * np.pi)
    
    return distance_m, angle_rad


if __name__ == "__main__":
    # Test channel model
    print("=" * 60)
    print("3GPP TR 38.901 UMi-Street Canyon Channel Model Test")
    print("=" * 60)
    
    channel = ChannelModel3GPP38901(frequency_ghz=3.5)
    
    print(f"\nConfiguration:")
    print(f"  Frequency: {channel.frequency_ghz} GHz")
    print(f"  BS height: {channel.h_bs} m")
    print(f"  UE height: {channel.h_ut} m")
    print(f"  Breakpoint distance: {channel.d_bp:.1f} m")
    
    print("\nLOS Probability vs Distance:")
    for d in [10, 20, 50, 100, 150, 200]:
        p_los = channel.calculate_los_probability(d)
        print(f"  {d:3d}m: P_LOS = {p_los:.3f}")
    
    print("\nSINR Distribution (1000 samples at each distance):")
    for d in [20, 50, 100, 150, 200]:
        sinrs = []
        for _ in range(1000):
            sinr, _ = channel.calculate_sinr(d_2d=d)
            sinrs.append(sinr)
        print(f"  {d:3d}m: SINR = {np.mean(sinrs):.1f} ± {np.std(sinrs):.1f} dB")
