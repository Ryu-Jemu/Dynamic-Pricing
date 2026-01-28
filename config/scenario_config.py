"""
5G O-RAN Network Slicing Simulation - Scenario Configuration

All economic and network parameters are defined as SCENARIO INPUTS with documentation.
No arbitrary constants - all values are either:
1. Derived from 3GPP standards
2. Documented from industry reports/literature
3. Explicitly marked as configurable scenario inputs

References:
- 3GPP TS 38.101-1: NR User Equipment radio transmission and reception
- 3GPP TS 38.214: Physical layer procedures for data
- 3GPP TR 38.901: Channel model for frequencies from 0.5 to 100 GHz
- McKinsey 5G Economics Report (2021)
- GSMA Network Economics (2023)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class SystemConfig:
    """
    System-level configuration derived from 3GPP standards.
    
    Reference: 3GPP TS 38.101-1, TS 38.104
    
    Supports deterministic (fixed) channel scenarios for focused 
    pricing optimization studies.
    
    Reference:
    - IEEE TMC 2023 (Cai et al.): Single gNB DRL resource allocation
    - Wiley 2024 (Khani et al.): SAC adapts to dynamic conditions within single BS
    """
    # Frequency band (n78 band for 5G NR)
    frequency_ghz: float = 3.5
    
    # System bandwidth - scenario input
    # Options: 5, 10, 15, 20, 25, 50, 100 MHz
    bandwidth_mhz: int = 20
    
    # Subcarrier spacing - scenario input
    # Options: 15, 30, 60 kHz for FR1
    scs_khz: int = 30
    
    # NOTE: N_RB is NOT hardcoded here - it will be derived from 
    # 3GPP TS 38.101-1 Table 5.3.2-1 in nr_prb_table.py
    
    # gNB parameters (3GPP TS 38.104)
    gnb_tx_power_dbm: float = 43.0  # 20W typical macro
    gnb_height_m: float = 10.0
    ue_height_m: float = 1.5
    
    # Cell configuration
    cell_radius_m: float = 200.0  # UMi scenario
    min_distance_m: float = 10.0
    
    # Antenna configuration
    antenna_config: str = "4T4R"  # MIMO configuration
    
    # Noise parameters
    noise_figure_db: float = 7.0
    thermal_noise_dbm_hz: float = -174.0  # dBm/Hz at 290K
    
    # === ENVIRONMENT FIXATION SETTINGS ===
    # Reference: IEEE TMC 2023 - Single BS pricing optimization
    # When True, external factors (channel, user positions) are fixed
    # to focus RL exploration on pricing strategy optimization
    enable_deterministic_channel: bool = False
    
    # Fixed user distance distribution (meters from BS)
    # Used when enable_deterministic_channel=True
    # Format: (distance_m, count) for each user type
    fixed_urllc_distances: Optional[Tuple[float, ...]] = None
    fixed_embb_distances: Optional[Tuple[float, ...]] = None
    
    # Fixed SINR values (dB) - overrides channel model
    # If None, SINR is calculated from distance
    fixed_urllc_sinr_db: Optional[float] = None  # e.g., 20.0 dB
    fixed_embb_sinr_db: Optional[float] = None   # e.g., 15.0 dB
    
    # Shadow fading mode
    # "stochastic": Normal shadow fading (default)
    # "fixed": No shadow fading (deterministic path loss only)
    # "scenario": Pre-defined shadow fading values
    shadow_fading_mode: str = "stochastic"
    
    # Interference model
    # "none": Single-cell, no inter-cell interference
    # "fixed": Fixed interference margin (dB)
    # "stochastic": Random interference from neighboring cells
    interference_mode: str = "none"
    fixed_interference_margin_db: float = 3.0  # Used when mode="fixed"


@dataclass
class URLLCConfig:
    """
    URLLC (Ultra-Reliable Low-Latency Communication) slice configuration.
    
    Reference: 3GPP TS 22.261, TR 38.913
    """
    # User population bounds - scenario input
    initial_users: int = 10
    min_users: int = 1
    max_users: int = 25
    
    # QoS requirements (3GPP TS 22.261 Table 7.1-1)
    latency_requirement_ms: float = 1.0  # User plane latency
    reliability_requirement: float = 0.99999  # 99.999% = 1-1e-5
    target_bler: float = 1e-5  # Block Error Rate target
    
    # Packet characteristics (typical URLLC applications)
    packet_size_bytes: int = 32  # Small packets for control/IoT
    packet_arrival_rate_per_slot: float = 0.1  # Poisson rate
    
    # PRB requirements (derived from FBL analysis)
    min_prb_per_user: int = 2  # Minimum guarantee
    
    # Service characteristics
    service_type: str = "B2B"
    applications: tuple = (
        "Industrial Automation",
        "Remote Surgery", 
        "Autonomous Vehicles",
        "Smart Grid Control"
    )


@dataclass
class EMMBConfig:
    """
    eMBB (enhanced Mobile Broadband) slice configuration.
    
    Reference: 3GPP TS 22.261, TR 38.913
    
    Includes Multi-User Diversity scheduling parameters.
    
    Reference:
    - IEEE/ACM Trans. Netw. 2020 (Anand et al.): Joint URLLC/eMBB scheduling
    - IEEE Xplore 2022 (D-PF): Demand-based Proportional Fairness
    - arXiv 2020 (Yin et al.): Dynamic proportional fairness for multi-user diversity
    """
    # User population bounds - scenario input
    initial_users: int = 50
    min_users: int = 5
    max_users: int = 100
    
    # QoS requirements
    throughput_requirement_mbps: float = 50.0  # DL experienced data rate
    latency_tolerance_ms: float = 100.0  # More relaxed than URLLC
    reliability_requirement: float = 0.999  # 99.9%
    
    # Traffic characteristics
    avg_data_rate_mbps: float = 20.0  # Average per user
    peak_to_avg_ratio: float = 3.0
    
    # PRB requirements
    min_prb_per_user: int = 1  # Minimum guarantee
    
    # === MULTI-USER DIVERSITY SCHEDULING PARAMETERS ===
    # Reference: IEEE/ACM Trans. Netw. 2020 - Proportional Fair with diversity
    
    # Enable multi-user diversity gain in spectral efficiency calculation
    enable_multiuser_diversity: bool = True
    
    # Proportional Fair scheduling window (slots for averaging)
    pf_averaging_window: int = 100
    
    # Fairness-throughput tradeoff parameter (alpha in PF metric)
    # alpha=0: Max throughput (no fairness)
    # alpha=1: Standard PF
    # alpha=2: Max-min fairness
    pf_fairness_alpha: float = 1.0
    
    # Multi-user diversity gain model
    # "opportunistic": Select best channel user (max diversity gain)
    # "proportional_fair": Balance fairness and diversity
    # "round_robin": No diversity exploitation
    diversity_scheduling_mode: str = "proportional_fair"
    
    # Minimum scheduling interval to ensure fairness (ms)
    max_starvation_time_ms: float = 100.0
    
    # Service characteristics
    service_type: str = "B2C"
    applications: tuple = (
        "4K/8K Video Streaming",
        "Cloud Gaming",
        "AR/VR Applications",
        "Video Conferencing"
    )


@dataclass
class ThreePartTariffConfig:
    """
    Three-Part Tariff pricing configuration.
    
    Structure: Revenue = F + max(0, U - D) × p
    - F: Access fee (fixed monthly/weekly fee)
    - D: Allowance (included data)
    - p: Overage price per MB
    
    References:
    - Fibich et al. (2017) Operations Research - Optimal Three-Part Tariff
    - AT&T Mobile Share Value Plans
    - Verizon 5G Network Slice pricing (2024)
    """
    
    # URLLC Pricing (B2B Premium) - scenario inputs with documentation
    urllc_base_fee_hourly: float = 50.0  # $/hour - premium B2B service
    urllc_allowance_mb: float = 10.0  # MB per billing cycle
    urllc_overage_price_per_mb: float = 0.50  # $/MB for overage
    
    # eMBB Pricing (B2C Standard) - scenario inputs
    embb_base_fee_hourly: float = 5.0  # $/hour
    embb_allowance_mb: float = 300.0  # MB per billing cycle
    embb_overage_price_per_mb: float = 0.02  # $/MB for overage
    
    # Price adjustment bounds for RL agent
    # Reference: ±20% dynamic adjustment range (McKinsey 2021)
    price_factor_min: float = 0.8
    price_factor_max: float = 1.2
    
    # Billing cycle configuration
    billing_cycle_hours: int = 168  # 1 week = 168 hours
    
    # Price bounds for action space mapping
    urllc_fee_bounds: Tuple[float, float] = (40.0, 60.0)  # F bounds
    urllc_overage_bounds: Tuple[float, float] = (0.40, 0.60)  # p bounds
    embb_fee_bounds: Tuple[float, float] = (4.0, 6.0)
    embb_overage_bounds: Tuple[float, float] = (0.016, 0.024)


@dataclass
class DemandConfig:
    """
    User demand model configuration (arrivals and churn).
    
    Arrivals: Non-Homogeneous Poisson Process (NHPP)
    Churn: Hazard/Logistic model with overage burden
    
    References:
    - ACM IMC 2023: Mobile network traffic analysis
    - Scientific Reports 2024: Telecom churn prediction
    - MDPI Algorithms 2024: Churn modeling
    """
    
    # === ARRIVAL PARAMETERS ===
    # Base arrival rates (users per hour) - calibrated from literature
    urllc_base_arrival_rate: float = 0.2  # Lower for B2B
    embb_base_arrival_rate: float = 1.0   # Higher for B2C
    
    # Time profile parameters (periodic patterns)
    # URLLC: Business hours peak (09:00-18:00)
    urllc_peak_hours: Tuple[int, int] = (9, 18)
    urllc_peak_multiplier: float = 1.2
    
    # eMBB: Evening peak (18:00-23:00)
    embb_peak_hours: Tuple[int, int] = (18, 23)
    embb_peak_multiplier: float = 2.0
    
    # Price elasticity of demand - scenario inputs
    # Reference: ScienceDirect 2024, FERDI 2021
    urllc_price_elasticity_fee: float = 0.3  # Inelastic (mission-critical)
    urllc_price_elasticity_overage: float = 0.2
    embb_price_elasticity_fee: float = 1.2  # Elastic (consumer)
    embb_price_elasticity_overage: float = 1.5
    
    # QoS effect on arrivals
    urllc_qos_sensitivity: float = 0.5
    embb_qos_sensitivity: float = 0.3
    
    # === CHURN PARAMETERS ===
    # Logistic/Hazard model coefficients
    # P(churn) = sigmoid(b0 + bV*Viol + bC*ConsecViol + bO*Overage + bF*Fee)
    
    # URLLC churn (B2B - lower base, higher QoS sensitivity)
    urllc_churn_b0: float = -6.0  # Base log-odds (low base churn)
    urllc_churn_bV: float = 2.0   # QoS violation sensitivity
    urllc_churn_bC: float = 1.5   # Consecutive violation penalty
    urllc_churn_bO: float = 0.5   # Overage payment sensitivity (lower for B2B)
    urllc_churn_bF: float = 0.01  # Fee sensitivity (very low for B2B)
    
    # eMBB churn (B2C - higher base, price sensitive)
    embb_churn_b0: float = -5.0   # Base log-odds
    embb_churn_bV: float = 1.0    # QoS violation sensitivity
    embb_churn_bC: float = 1.2    # Consecutive violation penalty
    embb_churn_bO: float = 2.0    # Overage payment sensitivity (high for B2C)
    embb_churn_bF: float = 0.1    # Fee sensitivity
    
    # Churn check interval
    urllc_churn_check_hours: int = 6  # B2B SLA review cycle
    embb_churn_check_hours: int = 3   # More frequent for B2C


@dataclass 
class CostConfig:
    """
    Operational cost model configuration.
    
    C_total = C_energy + C_spectrum + C_backhaul + C_fixed + C_acquisition
    
    References:
    - MDPI ECO6G 2022: 5G energy consumption
    - PMC 2024: OPEX breakdown
    - GSMA 2023: Network economics
    """
    
    # === ENERGY COST MODEL ===
    # P(t) = P_idle + (P_max - P_idle) * load^k
    # Reference: PMC 2021 - 5G BS power consumption
    
    power_idle_kw: float = 1.0    # Idle power consumption
    power_max_kw: float = 4.0     # Max power at full load
    power_load_exponent: float = 1.5  # Non-linear load factor
    
    # Electricity price - scenario input (varies by region)
    # Reference: Korea average industrial rate
    electricity_price_per_kwh: float = 0.10  # $/kWh
    
    # === SPECTRUM/INFRASTRUCTURE AMORTIZATION ===
    # Convert annual costs to hourly
    # Reference: GSMA 2023 TCO analysis
    
    annual_spectrum_cost: float = 50000.0  # $/year for spectrum license
    annual_infrastructure_opex: float = 30000.0  # $/year maintenance
    
    # === BACKHAUL COST ===
    # Reference: GSMA 2023 - data transport costs
    backhaul_cost_per_gb: float = 0.01  # $/GB
    
    # === FIXED OPERATIONAL COST ===
    # Personnel, site rental, etc.
    fixed_hourly_cost: float = 10.0  # $/hour
    
    # === CUSTOMER ACQUISITION COST (optional) ===
    # Reference: Industry benchmark
    customer_acquisition_cost: float = 50.0  # $/new user
    enable_acquisition_cost: bool = True


@dataclass
class CMDPConfig:
    """
    Constrained MDP configuration for Primal-Dual SAC.
    
    Objective: max E[sum gamma^t * r_t]
    Subject to: E[Viol_u] <= delta_u, E[Viol_e] <= delta_e
    
    Supports per-state (time-varying) constraints for peak/off-peak hours.
    
    Reference:
    - UC Davis CLARA (2024): Constrained RL for network slicing
    - Computer Communications 2024: Two-timescale resource allocation
    - IEEE Trans. Control 2022: Learning in CMDPs with per-state constraints
    """
    
    # Discount factor
    gamma: float = 0.99
    
    # === BASE CONSTRAINT THRESHOLDS ===
    urllc_violation_threshold: float = 0.001  # 0.1% max violation rate
    embb_violation_threshold: float = 0.01    # 1% max violation rate
    churn_threshold: float = 0.02             # 2% max churn rate (optional)
    
    # === PER-STATE (TIME-VARYING) CONSTRAINTS ===
    # Reference: Computer Communications 2024 - two-timescale approach
    enable_time_varying_constraints: bool = True
    
    # Peak hours definition (business hours for URLLC, evening for eMBB)
    urllc_peak_hours: Tuple[int, int] = (9, 18)   # 09:00-18:00
    embb_peak_hours: Tuple[int, int] = (18, 23)   # 18:00-23:00
    
    # Tighter constraints during peak hours (stricter QoS)
    urllc_peak_violation_threshold: float = 0.0005  # 0.05% during peak
    embb_peak_violation_threshold: float = 0.005    # 0.5% during peak
    
    # Relaxed constraints during off-peak (allow more flexibility)
    urllc_offpeak_violation_threshold: float = 0.002  # 0.2% during off-peak
    embb_offpeak_violation_threshold: float = 0.02    # 2% during off-peak
    
    # === ENABLE/DISABLE CONSTRAINTS ===
    enable_urllc_constraint: bool = True
    enable_embb_constraint: bool = True
    enable_churn_constraint: bool = False
    
    # === LAGRANGE MULTIPLIER SETTINGS ===
    lambda_init: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 100.0
    lambda_lr: float = 0.001  # Learning rate for dual variables
    
    def get_constraint_threshold(
        self, 
        slice_type: str, 
        hour_of_day: int
    ) -> float:
        """
        Get time-varying constraint threshold based on current hour.
        
        Args:
            slice_type: "URLLC" or "eMBB"
            hour_of_day: Current hour (0-23)
            
        Returns:
            Applicable violation threshold
        """
        if not self.enable_time_varying_constraints:
            if slice_type == "URLLC":
                return self.urllc_violation_threshold
            else:
                return self.embb_violation_threshold
        
        if slice_type == "URLLC":
            start, end = self.urllc_peak_hours
            if start <= hour_of_day < end:
                return self.urllc_peak_violation_threshold
            else:
                return self.urllc_offpeak_violation_threshold
        else:  # eMBB
            start, end = self.embb_peak_hours
            if start <= hour_of_day < end:
                return self.embb_peak_violation_threshold
            else:
                return self.embb_offpeak_violation_threshold


@dataclass
class SACConfig:
    """
    SAC (Soft Actor-Critic) hyperparameters.
    
    Reference: Haarnoja et al. ICML 2018
    """
    
    learning_rate: float = 3e-4
    buffer_size: int = 200000
    batch_size: int = 512
    gamma: float = 0.99
    tau: float = 0.005
    
    # Entropy coefficient
    ent_coef: str = "auto"  # Automatic temperature adjustment
    target_entropy: str = "auto"  # -dim(A)
    
    # Network architecture
    net_arch: tuple = (256, 256, 128)
    
    # Training settings
    learning_starts: int = 2000
    train_freq: int = 1
    gradient_steps: int = 1
    
    # Total training
    total_timesteps: int = 750000  # Increased for 6D action space
    
    # Evaluation
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    
    # Random seed
    seed: int = 42


@dataclass
class SimulationConfig:
    """
    Overall simulation configuration.
    """
    
    # Time scales
    step_duration_hours: int = 1  # RL step = 1 hour
    episode_length_hours: int = 168  # 1 week
    
    # Billing cycle
    billing_cycle_hours: int = 168  # Weekly billing
    
    # Random seed
    seed: int = 42
    
    # Logging
    log_frequency: int = 100  # Log every N steps
    save_frequency: int = 10000  # Save model every N steps
    
    # Paths
    log_dir: str = "./logs"
    model_dir: str = "./models"
    result_dir: str = "./results"


def get_default_config() -> Dict:
    """
    Get default configuration dictionary with all components.
    """
    return {
        "system": SystemConfig(),
        "urllc": URLLCConfig(),
        "embb": EMMBConfig(),
        "pricing": ThreePartTariffConfig(),
        "demand": DemandConfig(),
        "cost": CostConfig(),
        "cmdp": CMDPConfig(),
        "sac": SACConfig(),
        "simulation": SimulationConfig(),
    }


@dataclass
class ScenarioConfig:
    """
    Complete scenario configuration that bundles all sub-configs.
    
    This class provides a convenient way to pass all configuration
    to the environment and training scripts.
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    urllc: URLLCConfig = field(default_factory=URLLCConfig)
    embb: EMMBConfig = field(default_factory=EMMBConfig)
    tariff: ThreePartTariffConfig = field(default_factory=ThreePartTariffConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    cmdp: CMDPConfig = field(default_factory=CMDPConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ScenarioConfig':
        """Create ScenarioConfig from dictionary."""
        return cls(
            system=config_dict.get("system", SystemConfig()),
            urllc=config_dict.get("urllc", URLLCConfig()),
            embb=config_dict.get("embb", EMMBConfig()),
            tariff=config_dict.get("pricing", ThreePartTariffConfig()),
            demand=config_dict.get("demand", DemandConfig()),
            cost=config_dict.get("cost", CostConfig()),
            cmdp=config_dict.get("cmdp", CMDPConfig()),
            sac=config_dict.get("sac", SACConfig()),
            simulation=config_dict.get("simulation", SimulationConfig()),
        )


if __name__ == "__main__":
    # Print default configuration
    config = get_default_config()
    for name, cfg in config.items():
        print(f"\n=== {name.upper()} ===")
        for key, value in cfg.__dict__.items():
            print(f"  {key}: {value}")
