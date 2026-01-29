"""
5G O-RAN Network Slicing CMDP Environment

CRITICAL FIX (2026-01-28):
- Reward function redesigned for positive profit optimization
- CLV adjustment weight significantly reduced
- QoS violation calculation corrected
- User dynamics balanced with capacity

Reference:
- 3GPP standards for 5G NR
- SafeSlice (2024): CMDP for network slicing
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import warnings

# Import project modules
from config.scenario_config import (
    SystemConfig, URLLCConfig, EMMBConfig,
    ThreePartTariffConfig, DemandConfig, CostConfig, 
    CMDPConfig, SACConfig, SimulationConfig, ScenarioConfig, get_default_config
)
from env.scheduler import (
    TwoStageScheduler, UserAllocationState, SchedulingResult,
    NRResourceGrid
)
from env.qos_embb import EMMBQoSModel


@dataclass
class User:
    """Complete user state for simulation."""
    user_id: int
    slice_type: str
    distance_m: float
    angle_rad: float
    sinr_db: float = 15.0
    is_los: bool = True
    qos_violations_total: int = 0
    qos_violations_24h: int = 0
    consecutive_violations: int = 0
    satisfaction: float = 1.0
    hourly_usage_mb: float = 0.0
    cumulative_usage_mb: float = 0.0
    overage_payment_total: float = 0.0
    allocated_prb: int = 0
    subscription_hour: int = 0
    hours_active: int = 0


@dataclass
class StepInfo:
    """Detailed information from environment step."""
    revenue_total: float = 0.0
    revenue_urllc: float = 0.0
    revenue_embb: float = 0.0
    revenue_access_fee: float = 0.0
    revenue_overage: float = 0.0
    cost_total: float = 0.0
    cost_energy: float = 0.0
    cost_spectrum: float = 0.0
    cost_backhaul: float = 0.0
    cost_fixed: float = 0.0
    cost_acquisition: float = 0.0
    profit: float = 0.0
    urllc_violation_rate: float = 0.0
    embb_violation_rate: float = 0.0
    urllc_violations: int = 0
    embb_violations: int = 0
    urllc_arrivals: int = 0
    embb_arrivals: int = 0
    urllc_churns: int = 0
    embb_churns: int = 0
    n_urllc_users: int = 0
    n_embb_users: int = 0
    prb_utilization: float = 0.0
    urllc_prb_used: int = 0
    embb_prb_used: int = 0
    urllc_fee_factor: float = 1.0
    urllc_overage_factor: float = 1.0
    embb_fee_factor: float = 1.0
    embb_overage_factor: float = 1.0
    constraint_urllc: float = 0.0
    constraint_embb: float = 0.0


def action_to_price_factors(
    action: np.ndarray,
    fee_range: Tuple[float, float] = (0.8, 1.2),
    overage_range: Tuple[float, float] = (0.8, 1.2)
) -> Dict[str, float]:
    """Convert action [-1, 1] to price factors."""
    def scale(value, low, high):
        return low + (value + 1) * (high - low) / 2
    
    return {
        'urllc_fee': scale(action[0], fee_range[0], fee_range[1]),
        'urllc_overage': scale(action[1], overage_range[0], overage_range[1]),
        'embb_fee': scale(action[2], fee_range[0], fee_range[1]),
        'embb_overage': scale(action[3], overage_range[0], overage_range[1])
    }


class SimpleChannelModel:
    """Simplified 3GPP TR 38.901 UMi channel model."""
    
    def __init__(self, frequency_ghz: float = 3.5, h_bs: float = 10.0, h_ut: float = 1.5):
        self.fc = frequency_ghz
        self.h_bs = h_bs
        self.h_ut = h_ut
    
    def calculate_sinr(self, distance_m: float) -> Tuple[float, bool]:
        """Calculate SINR based on distance."""
        # Simplified path loss model
        d3d = np.sqrt(distance_m**2 + (self.h_bs - self.h_ut)**2)
        d3d = max(d3d, 10.0)
        
        # LOS probability
        p_los = min(18/distance_m, 1) * (1 - np.exp(-distance_m/36)) + np.exp(-distance_m/36)
        is_los = np.random.random() < p_los
        
        if is_los:
            pl = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(self.fc)
            shadowing = np.random.normal(0, 4.0)
        else:
            pl = 35.3 + 22.4 * np.log10(d3d) + 21.3 * np.log10(self.fc)
            shadowing = np.random.normal(0, 7.82)
        
        # SINR calculation (simplified)
        tx_power = 43.0  # dBm
        noise = -174 + 10 * np.log10(20e6) + 7  # Noise + NF
        sinr_db = tx_power - pl - shadowing - noise
        
        # Clamp to realistic range
        sinr_db = np.clip(sinr_db, -5, 35)
        
        return sinr_db, is_los


class SimpleTariffManager:
    """Simplified tariff and billing management."""
    
    def __init__(
        self,
        urllc_base_fee: float = 50.0,
        urllc_allowance_mb: float = 10.0,
        urllc_base_overage: float = 0.50,
        embb_base_fee: float = 5.0,
        embb_allowance_mb: float = 300.0,
        embb_base_overage: float = 0.02,
        billing_cycle_hours: int = 168
    ):
        self.urllc_base_fee = urllc_base_fee
        self.urllc_allowance_mb = urllc_allowance_mb
        self.urllc_base_overage = urllc_base_overage
        self.embb_base_fee = embb_base_fee
        self.embb_allowance_mb = embb_allowance_mb
        self.embb_base_overage = embb_base_overage
        self.billing_cycle_hours = billing_cycle_hours
        
        # Current price factors
        self.urllc_fee_factor = 1.0
        self.urllc_overage_factor = 1.0
        self.embb_fee_factor = 1.0
        self.embb_overage_factor = 1.0
        
        # User allowance tracking
        self.user_allowances: Dict[int, Dict] = {}
    
    def add_user(self, user_id: int, slice_type: str, hour: int):
        """Register new user."""
        if slice_type == "URLLC":
            allowance = self.urllc_allowance_mb
        else:
            allowance = self.embb_allowance_mb
        
        self.user_allowances[user_id] = {
            "slice_type": slice_type,
            "remaining_allowance": allowance,
            "cycle_start": hour
        }
    
    def remove_user(self, user_id: int):
        """Remove user."""
        if user_id in self.user_allowances:
            del self.user_allowances[user_id]
    
    def update_prices(
        self,
        urllc_fee_factor: float,
        urllc_overage_factor: float,
        embb_fee_factor: float,
        embb_overage_factor: float
    ):
        """Update price factors."""
        self.urllc_fee_factor = urllc_fee_factor
        self.urllc_overage_factor = urllc_overage_factor
        self.embb_fee_factor = embb_fee_factor
        self.embb_overage_factor = embb_overage_factor
    
    def bill_user(self, user_id: int, usage_mb: float, hour: int) -> Tuple[float, float, float]:
        """Calculate billing for user. Returns (total_revenue, overage_charge, remaining_allowance)."""
        if user_id not in self.user_allowances:
            return 0.0, 0.0, 0.0
        
        user_data = self.user_allowances[user_id]
        slice_type = user_data["slice_type"]
        
        # Check billing cycle reset
        cycle_elapsed = (hour - user_data["cycle_start"]) % self.billing_cycle_hours
        if cycle_elapsed == 0 and hour > user_data["cycle_start"]:
            # Reset allowance
            if slice_type == "URLLC":
                user_data["remaining_allowance"] = self.urllc_allowance_mb
            else:
                user_data["remaining_allowance"] = self.embb_allowance_mb
        
        # Calculate fees
        if slice_type == "URLLC":
            hourly_fee = (self.urllc_base_fee * self.urllc_fee_factor) / self.billing_cycle_hours
            overage_rate = self.urllc_base_overage * self.urllc_overage_factor
        else:
            hourly_fee = (self.embb_base_fee * self.embb_fee_factor) / self.billing_cycle_hours
            overage_rate = self.embb_base_overage * self.embb_overage_factor
        
        # Calculate overage
        remaining = user_data["remaining_allowance"]
        if usage_mb <= remaining:
            overage_charge = 0.0
            user_data["remaining_allowance"] -= usage_mb
        else:
            overage_mb = usage_mb - remaining
            overage_charge = overage_mb * overage_rate
            user_data["remaining_allowance"] = 0.0
        
        total_revenue = hourly_fee + overage_charge
        return total_revenue, overage_charge, user_data["remaining_allowance"]
    
    def get_allowance_stats(self, slice_type: str) -> Dict:
        """Get allowance statistics for a slice."""
        users = [u for u in self.user_allowances.values() if u["slice_type"] == slice_type]
        if not users:
            return {"mean_overage_ratio": 0.0, "near_limit_fraction": 0.0}
        
        if slice_type == "URLLC":
            total_allowance = self.urllc_allowance_mb
        else:
            total_allowance = self.embb_allowance_mb
        
        used_ratios = [1 - u["remaining_allowance"]/total_allowance for u in users]
        near_limit = sum(1 for r in used_ratios if r > 0.8) / len(users)
        
        return {
            "mean_overage_ratio": np.mean(used_ratios),
            "near_limit_fraction": near_limit
        }


class SimpleArrivalManager:
    """Simplified NHPP arrival model."""
    
    def __init__(
        self,
        urllc_base_rate: float = 0.2,
        embb_base_rate: float = 1.0
    ):
        self.urllc_base_rate = urllc_base_rate
        self.embb_base_rate = embb_base_rate
    
    def generate_arrivals(
        self,
        hour: int,
        urllc_fee_factor: float = 1.0,
        urllc_overage_factor: float = 1.0,
        urllc_violation_rate: float = 0.0,
        embb_fee_factor: float = 1.0,
        embb_overage_factor: float = 1.0,
        embb_violation_rate: float = 0.0
    ) -> Tuple[int, int]:
        """Generate arrivals with price and QoS sensitivity."""
        hour_of_day = hour % 24
        
        # URLLC: Business hours peak
        if 9 <= hour_of_day <= 17:
            urllc_mult = 1.5
        else:
            urllc_mult = 0.5
        
        # eMBB: Evening peak
        if 18 <= hour_of_day <= 23:
            embb_mult = 1.5
        elif 9 <= hour_of_day <= 18:
            embb_mult = 1.0
        else:
            embb_mult = 0.3
        
        # Price elasticity
        urllc_price_effect = max(0.5, 1.5 - 0.3 * (urllc_fee_factor - 1.0))
        embb_price_effect = max(0.3, 1.5 - 0.8 * (embb_fee_factor - 1.0))
        
        # QoS effect
        urllc_qos_effect = max(0.3, 1.0 - urllc_violation_rate * 5)
        embb_qos_effect = max(0.5, 1.0 - embb_violation_rate * 3)
        
        # Calculate rates
        urllc_rate = self.urllc_base_rate * urllc_mult * urllc_price_effect * urllc_qos_effect
        embb_rate = self.embb_base_rate * embb_mult * embb_price_effect * embb_qos_effect
        
        # Poisson arrivals
        urllc_arrivals = np.random.poisson(urllc_rate)
        embb_arrivals = np.random.poisson(embb_rate)
        
        return urllc_arrivals, embb_arrivals


class SimpleChurnManager:
    """Simplified hazard-based churn model."""
    
    def __init__(self):
        # Base hourly churn probability (very low)
        self.urllc_base_churn = 0.0001  # ~0.7% weekly
        self.embb_base_churn = 0.0005   # ~3.5% weekly
        
        self.user_states: Dict[int, Dict] = {}
    
    def register_user(self, user_id: int, slice_type: str):
        """Register user for churn tracking."""
        self.user_states[user_id] = {
            "slice_type": slice_type,
            "violations": 0,
            "consecutive_violations": 0,
            "hours_active": 0
        }
    
    def remove_user(self, user_id: int):
        """Remove user."""
        if user_id in self.user_states:
            del self.user_states[user_id]
    
    def update_user(self, user_id: int, violated: bool):
        """Update user state."""
        if user_id not in self.user_states:
            return
        
        state = self.user_states[user_id]
        state["hours_active"] += 1
        
        if violated:
            state["violations"] += 1
            state["consecutive_violations"] += 1
        else:
            state["consecutive_violations"] = 0
    
    def evaluate_all_users(
        self,
        urllc_fee_factor: float = 1.0,
        embb_fee_factor: float = 1.0,
        hour: int = 0
    ) -> Tuple[List[int], List[int]]:
        """Evaluate churn for all users. Only check churn every 24 hours."""
        urllc_churned = []
        embb_churned = []
        
        # Only evaluate churn every 24 hours
        if hour % 24 != 0:
            return urllc_churned, embb_churned
        
        for user_id, state in list(self.user_states.items()):
            slice_type = state["slice_type"]
            
            # Base daily churn probability
            if slice_type == "URLLC":
                base_prob = self.urllc_base_churn * 24
                fee_factor = urllc_fee_factor
            else:
                base_prob = self.embb_base_churn * 24
                fee_factor = embb_fee_factor
            
            # Modify based on factors
            # Higher fees increase churn
            fee_effect = 1.0 + 0.5 * max(0, fee_factor - 1.0)
            
            # Consecutive violations increase churn
            violation_effect = 1.0 + 0.3 * min(state["consecutive_violations"], 5)
            
            # Tenure reduces churn (loyalty)
            tenure_effect = max(0.5, 1.0 - 0.001 * state["hours_active"])
            
            # Final probability
            churn_prob = base_prob * fee_effect * violation_effect * tenure_effect
            churn_prob = min(0.1, churn_prob)  # Cap at 10% daily
            
            if np.random.random() < churn_prob:
                if slice_type == "URLLC":
                    urllc_churned.append(user_id)
                else:
                    embb_churned.append(user_id)
        
        return urllc_churned, embb_churned


class SimpleCostManager:
    """Simplified cost model."""
    
    def __init__(
        self,
        energy_cost_per_kwh: float = 0.12,
        base_power_kw: float = 2.0,
        power_per_prb_w: float = 10.0,
        spectrum_cost_per_mhz_hour: float = 0.1,
        backhaul_cost_per_gb: float = 0.005,
        fixed_cost_per_hour: float = 5.0,
        customer_acquisition_cost: float = 20.0
    ):
        self.energy_cost_per_kwh = energy_cost_per_kwh
        self.base_power_kw = base_power_kw
        self.power_per_prb_w = power_per_prb_w
        self.spectrum_cost_per_mhz_hour = spectrum_cost_per_mhz_hour
        self.backhaul_cost_per_gb = backhaul_cost_per_gb
        self.fixed_cost_per_hour = fixed_cost_per_hour
        self.customer_acquisition_cost = customer_acquisition_cost
    
    def calculate_hourly_cost(
        self,
        prb_used: int,
        bandwidth_mhz: float = 20.0,
        total_data_mb: float = 0.0,
        new_customers: int = 0
    ) -> Dict[str, float]:
        """Calculate hourly operational cost."""
        # Energy cost
        power_kw = self.base_power_kw + (prb_used * self.power_per_prb_w) / 1000
        energy_cost = power_kw * self.energy_cost_per_kwh
        
        # Spectrum cost (amortized)
        spectrum_cost = bandwidth_mhz * self.spectrum_cost_per_mhz_hour
        
        # Backhaul cost
        backhaul_cost = (total_data_mb / 1000) * self.backhaul_cost_per_gb
        
        # Fixed cost
        fixed_cost = self.fixed_cost_per_hour
        
        # Acquisition cost
        acquisition_cost = new_customers * self.customer_acquisition_cost
        
        total_cost = energy_cost + spectrum_cost + backhaul_cost + fixed_cost + acquisition_cost
        
        return {
            "energy": energy_cost,
            "spectrum": spectrum_cost,
            "backhaul": backhaul_cost,
            "fixed": fixed_cost,
            "acquisition": acquisition_cost,
            "total": total_cost
        }


class NetworkSlicingCMDPEnv(gym.Env):
    """
    5G O-RAN Network Slicing CMDP Environment.
    
    CRITICAL FIX:
    - Reward function properly scaled for positive profits
    - CLV adjustment weight reduced from 0.1 to 0.01
    - QoS evaluation uses soft thresholds
    - User dynamics balanced with capacity
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Load configuration
        if config is None:
            config_dict = get_default_config()
            config = ScenarioConfig.from_dict(config_dict)
        
        self.config = config
        self.sys_cfg = config.system
        self.urllc_cfg = config.urllc
        self.embb_cfg = config.embb
        self.tariff_cfg = config.tariff
        self.demand_cfg = config.demand
        self.cost_cfg = config.cost
        self.cmdp_cfg = config.cmdp
        self.sim_cfg = config.simulation
        
        self.render_mode = render_mode
        
        # Initialize components
        self._setup_components()
        self._define_spaces()
        
        # Episode state
        self.current_hour = 0
        self.episode_length = self.sim_cfg.episode_length_hours
        
        # User tracking
        self.urllc_users: Dict[int, User] = {}
        self.embb_users: Dict[int, User] = {}
        self.next_user_id = 0
        
        # Price factors
        self.urllc_fee_factor = 1.0
        self.urllc_overage_factor = 1.0
        self.embb_fee_factor = 1.0
        self.embb_overage_factor = 1.0
        
        # Cumulative metrics
        self.cumulative_revenue = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_churn = 0
        self.cumulative_arrivals = 0
        self.cumulative_violations_urllc = 0
        self.cumulative_violations_embb = 0
        
        # History for observation
        self.profit_history = deque(maxlen=24)
        self.revenue_history = deque(maxlen=24)
        self.cost_history = deque(maxlen=24)
        self.violation_history = deque(maxlen=24)
        
        # Step info
        self._step_info = StepInfo()
        self.last_info = None
    
    def _setup_components(self):
        """Initialize all simulation components."""
        # Resource grid
        self.resource_grid = NRResourceGrid(
            bandwidth_mhz=self.sys_cfg.bandwidth_mhz,
            scs_khz=self.sys_cfg.scs_khz
        )
        self.n_rb = self.resource_grid.n_rb
        
        # Channel model
        self.channel_model = SimpleChannelModel(
            frequency_ghz=self.sys_cfg.frequency_ghz,
            h_bs=self.sys_cfg.gnb_height_m,
            h_ut=self.sys_cfg.ue_height_m
        )
        
        # QoS models
        self.embb_qos = EMMBQoSModel(
            throughput_requirement_mbps=self.embb_cfg.throughput_requirement_mbps,
            latency_tolerance_ms=self.embb_cfg.latency_tolerance_ms,
            reliability_requirement=self.embb_cfg.reliability_requirement,
            scs_khz=self.sys_cfg.scs_khz,
            soft_qos_threshold=0.70  # FIXED: Reduced from 0.80 for realistic operation
        )
        
        # Scheduler
        self.scheduler = TwoStageScheduler(
            resource_grid=self.resource_grid,
            urllc_config=self.urllc_cfg,
            embb_config=self.embb_cfg
        )
        
        # Tariff manager
        self.tariff_manager = SimpleTariffManager(
            urllc_base_fee=self.tariff_cfg.urllc_base_fee_hourly,
            urllc_allowance_mb=self.tariff_cfg.urllc_allowance_mb,
            urllc_base_overage=self.tariff_cfg.urllc_overage_price_per_mb,
            embb_base_fee=self.tariff_cfg.embb_base_fee_hourly,
            embb_allowance_mb=self.tariff_cfg.embb_allowance_mb,
            embb_base_overage=self.tariff_cfg.embb_overage_price_per_mb,
            billing_cycle_hours=self.tariff_cfg.billing_cycle_hours
        )
        
        # Arrival manager
        self.arrival_manager = SimpleArrivalManager(
            urllc_base_rate=self.demand_cfg.urllc_base_arrival_rate,
            embb_base_rate=self.demand_cfg.embb_base_arrival_rate
        )
        
        # Churn manager
        self.churn_manager = SimpleChurnManager()
        
        # Cost manager
        self.cost_manager = SimpleCostManager(
            energy_cost_per_kwh=self.cost_cfg.energy_cost_per_kwh,
            base_power_kw=self.cost_cfg.base_power_kw,
            power_per_prb_w=self.cost_cfg.power_per_prb_w,
            spectrum_cost_per_mhz_hour=self.cost_cfg.spectrum_cost_per_mhz_hour,
            backhaul_cost_per_gb=self.cost_cfg.backhaul_cost_per_gb,
            fixed_cost_per_hour=self.cost_cfg.fixed_cost_per_hour,
            customer_acquisition_cost=self.cost_cfg.customer_acquisition_cost
        )
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation: 32 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float32
        )
        
        # Action: 4 dimensions (price factors)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_hour = 0
        self.urllc_users.clear()
        self.embb_users.clear()
        self.next_user_id = 0
        
        # Reset price factors
        self.urllc_fee_factor = 1.0
        self.urllc_overage_factor = 1.0
        self.embb_fee_factor = 1.0
        self.embb_overage_factor = 1.0
        
        # Reset cumulative metrics
        self.cumulative_revenue = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_churn = 0
        self.cumulative_arrivals = 0
        self.cumulative_violations_urllc = 0
        self.cumulative_violations_embb = 0
        
        # Reset history
        self.profit_history.clear()
        self.revenue_history.clear()
        self.cost_history.clear()
        self.violation_history.clear()
        
        # Reset managers
        self.tariff_manager = SimpleTariffManager(
            urllc_base_fee=self.tariff_cfg.urllc_base_fee_hourly,
            urllc_allowance_mb=self.tariff_cfg.urllc_allowance_mb,
            urllc_base_overage=self.tariff_cfg.urllc_overage_price_per_mb,
            embb_base_fee=self.tariff_cfg.embb_base_fee_hourly,
            embb_allowance_mb=self.tariff_cfg.embb_allowance_mb,
            embb_base_overage=self.tariff_cfg.embb_overage_price_per_mb,
            billing_cycle_hours=self.tariff_cfg.billing_cycle_hours
        )
        self.churn_manager = SimpleChurnManager()
        
        # Initialize users
        self._initialize_users()
        
        # Get initial observation
        obs = self._get_observation()
        info = {"initial": True}
        
        return obs, info
    
    def _initialize_users(self):
        """Initialize starting users."""
        for _ in range(self.urllc_cfg.initial_users):
            self._add_user("URLLC")
        
        for _ in range(self.embb_cfg.initial_users):
            self._add_user("eMBB")
    
    def _add_user(self, slice_type: str) -> User:
        """Add a new user to the network."""
        user_id = self.next_user_id
        self.next_user_id += 1
        
        # Generate random location
        distance, angle = self._generate_user_location()
        
        # Calculate SINR
        sinr_db, is_los = self.channel_model.calculate_sinr(distance)
        
        user = User(
            user_id=user_id,
            slice_type=slice_type,
            distance_m=distance,
            angle_rad=angle,
            sinr_db=sinr_db,
            is_los=is_los,
            subscription_hour=self.current_hour
        )
        
        if slice_type == "URLLC":
            self.urllc_users[user_id] = user
        else:
            self.embb_users[user_id] = user
        
        self.tariff_manager.add_user(user_id, slice_type, self.current_hour)
        self.churn_manager.register_user(user_id, slice_type)
        
        return user
    
    def _remove_user(self, user_id: int, slice_type: str):
        """Remove a user from the network."""
        if slice_type == "URLLC":
            if user_id in self.urllc_users:
                del self.urllc_users[user_id]
        else:
            if user_id in self.embb_users:
                del self.embb_users[user_id]
        
        self.tariff_manager.remove_user(user_id)
        self.churn_manager.remove_user(user_id)
        self.scheduler.remove_user(user_id)
    
    def _generate_user_location(self) -> Tuple[float, float]:
        """Generate random user location within cell."""
        cell_radius = self.sys_cfg.cell_radius_m
        min_distance = 10.0
        
        r = cell_radius * np.sqrt(
            self.np_random.uniform((min_distance/cell_radius)**2, 1.0)
        )
        theta = self.np_random.uniform(0, 2 * np.pi)
        
        return r, theta
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # 1. Decode action
        price_factors = action_to_price_factors(action)
        self.urllc_fee_factor = price_factors['urllc_fee']
        self.urllc_overage_factor = price_factors['urllc_overage']
        self.embb_fee_factor = price_factors['embb_fee']
        self.embb_overage_factor = price_factors['embb_overage']
        
        self.tariff_manager.update_prices(
            self.urllc_fee_factor, self.urllc_overage_factor,
            self.embb_fee_factor, self.embb_overage_factor
        )
        
        # 2. Update channels
        self._update_channels()
        
        # 3. Generate traffic
        self._generate_traffic()
        
        # 4. Schedule resources
        scheduling_result = self._allocate_resources()
        
        # 5. Evaluate QoS
        urllc_violations, embb_violations = self._evaluate_qos(scheduling_result)
        
        # 6. Bill users
        revenue_info = self._bill_users()
        
        # 7. Calculate costs
        cost_info = self._calculate_costs(scheduling_result)
        
        # 8. Process churn
        churn_info = self._process_churn()
        
        # 9. Process arrivals
        arrival_info = self._process_arrivals()
        
        # 10. Calculate reward
        profit = revenue_info['total'] - cost_info['total']
        reward = self._calculate_reward(profit, arrival_info, churn_info)
        
        # 11. Calculate constraints
        n_urllc = len(self.urllc_users)
        n_embb = len(self.embb_users)
        urllc_viol_rate = urllc_violations / max(1, n_urllc)
        embb_viol_rate = embb_violations / max(1, n_embb)
        
        constraint_urllc = urllc_viol_rate - self.cmdp_cfg.urllc_violation_threshold
        constraint_embb = embb_viol_rate - self.cmdp_cfg.embb_violation_threshold
        
        # 12. Update history
        self._update_history(profit, revenue_info['total'], cost_info['total'],
                           urllc_viol_rate, embb_viol_rate)
        
        # 13. Advance time
        self.current_hour += 1
        
        # 14. Build info
        info = self._build_info(
            revenue_info, cost_info, profit,
            urllc_violations, embb_violations,
            urllc_viol_rate, embb_viol_rate,
            scheduling_result, churn_info, arrival_info,
            constraint_urllc, constraint_embb
        )
        self.last_info = info
        
        # 15. Get observation
        obs = self._get_observation()
        
        # 16. Check termination
        terminated = False
        truncated = self.current_hour >= self.episode_length
        
        return obs, reward, terminated, truncated, info
    
    def _update_channels(self):
        """Update channel conditions for all users.
        
        FIX: Uses mean-reversion model instead of random walk.
        SINR fluctuates around base value determined by distance,
        preventing cumulative drift that caused QoS violations.
        """
        for user in list(self.urllc_users.values()) + list(self.embb_users.values()):
            # Calculate base SINR from distance (stable reference)
            base_sinr, _ = self.channel_model.calculate_sinr(user.distance_m)
            
            # Apply bounded shadow fading around base SINR
            # Standard deviation of ~2 dB is typical for slow fading
            shadow_fading = self.np_random.normal(0, 2.0)
            user.sinr_db = np.clip(base_sinr + shadow_fading, -5, 35)
    
    def _generate_traffic(self):
        """Generate traffic usage for all users."""
        for user in self.urllc_users.values():
            base_usage = self.tariff_cfg.urllc_allowance_mb / 168
            variation = self.np_random.exponential(0.2) * base_usage
            user.hourly_usage_mb = base_usage + variation
            user.cumulative_usage_mb += user.hourly_usage_mb
        
        for user in self.embb_users.values():
            hour_of_day = self.current_hour % 24
            if 18 <= hour_of_day <= 23:
                multiplier = 1.5
            elif 9 <= hour_of_day <= 18:
                multiplier = 1.0
            else:
                multiplier = 0.5
            
            base_usage = (self.tariff_cfg.embb_allowance_mb / 168) * multiplier
            variation = self.np_random.exponential(0.5) * base_usage
            user.hourly_usage_mb = base_usage + variation
            user.cumulative_usage_mb += user.hourly_usage_mb
    
    def _allocate_resources(self) -> SchedulingResult:
        """Allocate PRBs using two-stage scheduler."""
        urllc_states = [
            UserAllocationState(
                user_id=u.user_id,
                slice_type="URLLC",
                sinr_db=u.sinr_db,
                distance_m=u.distance_m,
                consecutive_violations=u.consecutive_violations,
                qos_violations_24h=u.qos_violations_24h
            )
            for u in self.urllc_users.values()
        ]
        
        embb_states = [
            UserAllocationState(
                user_id=u.user_id,
                slice_type="eMBB",
                sinr_db=u.sinr_db,
                distance_m=u.distance_m,
                avg_throughput_mbps=5.0
            )
            for u in self.embb_users.values()
        ]
        
        recent_urllc_viol = self._get_recent_violation_rate("URLLC")
        recent_embb_viol = self._get_recent_violation_rate("eMBB")
        
        result = self.scheduler.schedule(
            urllc_users=urllc_states,
            embb_users=embb_states,
            urllc_violation_rate=recent_urllc_viol,
            embb_violation_rate=recent_embb_viol
        )
        
        for user in self.urllc_users.values():
            user.allocated_prb = result.urllc_allocation.allocations.get(user.user_id, 0)
        
        for user in self.embb_users.values():
            user.allocated_prb = result.embb_allocation.allocations.get(user.user_id, 0)
        
        return result
    
    def _evaluate_qos(self, scheduling_result: SchedulingResult) -> Tuple[int, int]:
        """Evaluate QoS for all users."""
        urllc_violations = 0
        embb_violations = 0
        
        # URLLC: Simplified reliability-based evaluation
        for user in self.urllc_users.values():
            # Higher SINR and more PRBs = lower violation probability
            sinr_factor = max(0, (user.sinr_db - 5) / 30)  # 0 at 5dB, 1 at 35dB
            prb_factor = min(1, user.allocated_prb / 3)  # 0 at 0 PRB, 1 at 3+ PRB
            
            # Base violation probability
            base_viol_prob = 0.001  # Target: 0.1%
            
            # Adjusted probability
            viol_prob = base_viol_prob / max(0.1, sinr_factor * prb_factor)
            viol_prob = min(0.1, viol_prob)  # Cap at 10%
            
            if self.np_random.random() < viol_prob:
                urllc_violations += 1
                user.qos_violations_total += 1
                user.qos_violations_24h += 1
                user.consecutive_violations += 1
                self.churn_manager.update_user(user.user_id, True)
            else:
                user.consecutive_violations = 0
                self.churn_manager.update_user(user.user_id, False)
        
        # eMBB: Throughput-based evaluation with SOFT threshold
        for user in self.embb_users.values():
            satisfied, _, details = self.embb_qos.evaluate_qos(
                sinr_db=user.sinr_db,
                allocated_prb=user.allocated_prb
            )
            
            if not satisfied:
                embb_violations += 1
                user.qos_violations_total += 1
                user.qos_violations_24h += 1
                user.consecutive_violations += 1
                self.churn_manager.update_user(user.user_id, True)
            else:
                user.consecutive_violations = 0
                self.churn_manager.update_user(user.user_id, False)
            
            if 'achieved_throughput_mbps' in details:
                self.scheduler.update_embb_throughput(
                    user.user_id, details['achieved_throughput_mbps']
                )
        
        self.cumulative_violations_urllc += urllc_violations
        self.cumulative_violations_embb += embb_violations
        
        return urllc_violations, embb_violations
    
    def _bill_users(self) -> Dict[str, float]:
        """Bill all users."""
        total_revenue = 0.0
        urllc_revenue = 0.0
        embb_revenue = 0.0
        access_fee_total = 0.0
        overage_total = 0.0
        
        for user in self.urllc_users.values():
            revenue, overage, _ = self.tariff_manager.bill_user(
                user.user_id, user.hourly_usage_mb, self.current_hour
            )
            total_revenue += revenue
            urllc_revenue += revenue
            access_fee = revenue - overage
            access_fee_total += access_fee
            overage_total += overage
        
        for user in self.embb_users.values():
            revenue, overage, _ = self.tariff_manager.bill_user(
                user.user_id, user.hourly_usage_mb, self.current_hour
            )
            total_revenue += revenue
            embb_revenue += revenue
            access_fee = revenue - overage
            access_fee_total += access_fee
            overage_total += overage
        
        self.cumulative_revenue += total_revenue
        
        return {
            'total': total_revenue,
            'urllc': urllc_revenue,
            'embb': embb_revenue,
            'access_fee': access_fee_total,
            'overage': overage_total
        }
    
    def _calculate_costs(self, scheduling_result: SchedulingResult) -> Dict[str, float]:
        """Calculate operational costs."""
        total_data_mb = sum(u.hourly_usage_mb for u in 
                          list(self.urllc_users.values()) + list(self.embb_users.values()))
        
        new_customers = getattr(self, '_recent_arrivals', 0)
        
        cost_info = self.cost_manager.calculate_hourly_cost(
            prb_used=scheduling_result.total_prb_used,
            bandwidth_mhz=self.sys_cfg.bandwidth_mhz,
            total_data_mb=total_data_mb,
            new_customers=new_customers
        )
        
        self.cumulative_cost += cost_info['total']
        
        return cost_info
    
    def _process_churn(self) -> Dict[str, int]:
        """Process user churn."""
        urllc_churned, embb_churned = self.churn_manager.evaluate_all_users(
            urllc_fee_factor=self.urllc_fee_factor,
            embb_fee_factor=self.embb_fee_factor,
            hour=self.current_hour
        )
        
        for user_id in urllc_churned:
            self._remove_user(user_id, "URLLC")
        
        for user_id in embb_churned:
            self._remove_user(user_id, "eMBB")
        
        total_churned = len(urllc_churned) + len(embb_churned)
        self.cumulative_churn += total_churned
        
        return {
            'urllc': len(urllc_churned),
            'embb': len(embb_churned),
            'total': total_churned
        }
    
    def _process_arrivals(self) -> Dict[str, int]:
        """Process new user arrivals."""
        urllc_viol_rate = self._get_recent_violation_rate("URLLC")
        embb_viol_rate = self._get_recent_violation_rate("eMBB")
        
        urllc_arrivals, embb_arrivals = self.arrival_manager.generate_arrivals(
            hour=self.current_hour,
            urllc_fee_factor=self.urllc_fee_factor,
            urllc_overage_factor=self.urllc_overage_factor,
            urllc_violation_rate=urllc_viol_rate,
            embb_fee_factor=self.embb_fee_factor,
            embb_overage_factor=self.embb_overage_factor,
            embb_violation_rate=embb_viol_rate
        )
        
        max_urllc = self.urllc_cfg.max_users
        max_embb = self.embb_cfg.max_users
        
        urllc_arrivals = min(urllc_arrivals, max_urllc - len(self.urllc_users))
        embb_arrivals = min(embb_arrivals, max_embb - len(self.embb_users))
        
        for _ in range(urllc_arrivals):
            self._add_user("URLLC")
        
        for _ in range(embb_arrivals):
            self._add_user("eMBB")
        
        total_arrivals = urllc_arrivals + embb_arrivals
        self.cumulative_arrivals += total_arrivals
        self._recent_arrivals = total_arrivals
        
        return {
            'urllc': urllc_arrivals,
            'embb': embb_arrivals,
            'total': total_arrivals
        }
    
    def _calculate_reward(
        self,
        profit: float,
        arrival_info: Dict,
        churn_info: Dict
    ) -> float:
        """
        Calculate reward with FIXED scaling.
        
        CRITICAL FIX:
        - Reward primarily based on profit
        - CLV adjustment weight reduced to 0.01 (was 0.1)
        - Better normalization for stable learning
        """
        # Expected max revenue per hour
        max_revenue = (self.urllc_cfg.max_users * self.tariff_cfg.urllc_base_fee_hourly / 168 +
                      self.embb_cfg.max_users * self.tariff_cfg.embb_base_fee_hourly / 168)
        
        # Normalize profit to roughly [-1, 1] range
        # Expected profit is typically 30-50% of revenue
        expected_profit = max_revenue * 0.4
        normalized_profit = profit / max(1.0, expected_profit)
        
        # CLV adjustment (SIGNIFICANTLY REDUCED)
        clv_weight = 0.01  # CRITICAL: Reduced from 0.1
        
        net_customers = arrival_info['total'] - churn_info['total']
        clv_adjustment = clv_weight * net_customers * 0.1  # Small bonus for net growth
        
        # Final reward
        reward = normalized_profit + clv_adjustment
        
        return float(reward)
    
    def _get_recent_violation_rate(self, slice_type: str) -> float:
        """Get recent violation rate for a slice."""
        if slice_type == "URLLC":
            users = self.urllc_users
        else:
            users = self.embb_users
        
        if not users:
            return 0.0
        
        total_violations = sum(u.qos_violations_24h for u in users.values())
        rate = total_violations / (len(users) * 24)
        return min(1.0, rate)
    
    def _update_history(self, profit: float, revenue: float, cost: float,
                       urllc_viol: float, embb_viol: float):
        """Update rolling history."""
        self.profit_history.append(profit)
        self.revenue_history.append(revenue)
        self.cost_history.append(cost)
        self.violation_history.append((urllc_viol, embb_viol))
        
        # Decay 24h violation counters
        if self.current_hour % 24 == 0:
            for user in list(self.urllc_users.values()) + list(self.embb_users.values()):
                user.qos_violations_24h = max(0, user.qos_violations_24h - 1)
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = np.zeros(32, dtype=np.float32)
        
        n_urllc = len(self.urllc_users)
        n_embb = len(self.embb_users)
        
        # [0-5] URLLC state
        obs[0] = n_urllc / max(1, self.urllc_cfg.max_users)
        obs[1] = self._get_recent_violation_rate("URLLC")
        obs[2] = np.mean([u.sinr_db for u in self.urllc_users.values()]) / 40 if n_urllc > 0 else 0.5
        obs[3] = sum(u.allocated_prb for u in self.urllc_users.values()) / max(1, self.n_rb * 0.3)
        obs[4] = self.urllc_fee_factor
        obs[5] = self.urllc_overage_factor
        
        # [6-11] eMBB state
        obs[6] = n_embb / max(1, self.embb_cfg.max_users)
        obs[7] = self._get_recent_violation_rate("eMBB")
        obs[8] = np.mean([u.sinr_db for u in self.embb_users.values()]) / 40 if n_embb > 0 else 0.5
        obs[9] = sum(u.allocated_prb for u in self.embb_users.values()) / max(1, self.n_rb * 0.7)
        obs[10] = self.embb_fee_factor
        obs[11] = self.embb_overage_factor
        
        # [12-17] System state
        total_prb_used = (sum(u.allocated_prb for u in self.urllc_users.values()) +
                        sum(u.allocated_prb for u in self.embb_users.values()))
        obs[12] = total_prb_used / self.n_rb
        
        recent_revenue = np.mean(list(self.revenue_history)) if self.revenue_history else 0
        recent_cost = np.mean(list(self.cost_history)) if self.cost_history else 0
        recent_profit = np.mean(list(self.profit_history)) if self.profit_history else 0
        
        obs[13] = recent_revenue / 100
        obs[14] = recent_profit / 50
        obs[15] = self.cumulative_churn / max(1, self.current_hour + 1)
        obs[16] = self.cumulative_arrivals / max(1, self.current_hour + 1)
        obs[17] = recent_cost / 50
        
        # [18-21] Allowance state
        urllc_stats = self.tariff_manager.get_allowance_stats("URLLC")
        embb_stats = self.tariff_manager.get_allowance_stats("eMBB")
        
        obs[18] = urllc_stats.get('mean_overage_ratio', 0)
        obs[19] = embb_stats.get('mean_overage_ratio', 0)
        obs[20] = urllc_stats.get('near_limit_fraction', 0)
        obs[21] = embb_stats.get('near_limit_fraction', 0)
        
        # [22-25] Time features
        hour_of_day = self.current_hour % 24
        day_of_week = (self.current_hour // 24) % 7
        
        obs[22] = np.sin(2 * np.pi * hour_of_day / 24)
        obs[23] = np.cos(2 * np.pi * hour_of_day / 24)
        obs[24] = np.sin(2 * np.pi * day_of_week / 7)
        obs[25] = np.cos(2 * np.pi * day_of_week / 7)
        
        # [26-31] History features
        obs[26] = np.sum(list(self.revenue_history)) / 1000 if self.revenue_history else 0
        
        if self.violation_history:
            avg_viol = np.mean([v[0] + v[1] for v in self.violation_history])
            obs[27] = avg_viol
        
        obs[28] = self.cumulative_churn / 100
        obs[29] = self.cumulative_arrivals / 100
        obs[30] = np.sum(list(self.profit_history)) / 500 if self.profit_history else 0
        obs[31] = np.sum(list(self.cost_history)) / 100 if self.cost_history else 0
        
        return obs
    
    def _build_info(
        self,
        revenue_info: Dict,
        cost_info: Dict,
        profit: float,
        urllc_violations: int,
        embb_violations: int,
        urllc_viol_rate: float,
        embb_viol_rate: float,
        scheduling_result: SchedulingResult,
        churn_info: Dict,
        arrival_info: Dict,
        constraint_urllc: float,
        constraint_embb: float
    ) -> Dict[str, Any]:
        """Build step info dictionary."""
        return {
            'revenue_total': revenue_info['total'],
            'revenue_urllc': revenue_info['urllc'],
            'revenue_embb': revenue_info['embb'],
            'revenue_access_fee': revenue_info['access_fee'],
            'revenue_overage': revenue_info['overage'],
            'cost_total': cost_info['total'],
            'cost_energy': cost_info['energy'],
            'cost_spectrum': cost_info['spectrum'],
            'cost_backhaul': cost_info['backhaul'],
            'cost_fixed': cost_info['fixed'],
            'cost_acquisition': cost_info['acquisition'],
            'profit': profit,
            'urllc_violation_rate': urllc_viol_rate,
            'embb_violation_rate': embb_viol_rate,
            'urllc_violations': urllc_violations,
            'embb_violations': embb_violations,
            'urllc_arrivals': arrival_info['urllc'],
            'embb_arrivals': arrival_info['embb'],
            'urllc_churns': churn_info['urllc'],
            'embb_churns': churn_info['embb'],
            'n_urllc_users': len(self.urllc_users),
            'n_embb_users': len(self.embb_users),
            'prb_utilization': scheduling_result.total_prb_utilization,
            'urllc_prb_used': scheduling_result.urllc_allocation.total_allocated,
            'embb_prb_used': scheduling_result.embb_allocation.total_allocated,
            'urllc_fee_factor': self.urllc_fee_factor,
            'urllc_overage_factor': self.urllc_overage_factor,
            'embb_fee_factor': self.embb_fee_factor,
            'embb_overage_factor': self.embb_overage_factor,
            'constraint_urllc': constraint_urllc,
            'constraint_embb': constraint_embb
        }
    
    def get_constraint_values(self) -> np.ndarray:
        """Get current constraint values for CMDP."""
        if self.last_info is None:
            return np.zeros(2)
        
        return np.array([
            self.last_info['constraint_urllc'],
            self.last_info['constraint_embb']
        ])
    
    def render(self):
        """Render environment state."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            print(f"\n{'='*60}")
            print(f"Hour {self.current_hour} / {self.episode_length}")
            print(f"{'='*60}")
            
            if self.last_info:
                info = self.last_info
                print(f"\nUsers: URLLC={info['n_urllc_users']}, eMBB={info['n_embb_users']}")
                print(f"Revenue: ${info['revenue_total']:.2f}")
                print(f"Cost: ${info['cost_total']:.2f}")
                print(f"Profit: ${info['profit']:.2f}")
                print(f"\nQoS Violations:")
                print(f"  URLLC: {info['urllc_violation_rate']:.4f} (target: {self.cmdp_cfg.urllc_violation_threshold})")
                print(f"  eMBB: {info['embb_violation_rate']:.4f} (target: {self.cmdp_cfg.embb_violation_threshold})")
                print(f"\nPRB Utilization: {info['prb_utilization']:.2%}")


if __name__ == "__main__":
    print("=" * 70)
    print("Network Slicing CMDP Environment Test (FIXED)")
    print("=" * 70)
    
    env = NetworkSlicingCMDPEnv(render_mode="human")
    
    print(f"\nConfiguration:")
    print(f"  Total PRBs: {env.n_rb}")
    print(f"  URLLC users: {env.urllc_cfg.initial_users} initial, {env.urllc_cfg.max_users} max")
    print(f"  eMBB users: {env.embb_cfg.initial_users} initial, {env.embb_cfg.max_users} max")
    print(f"  eMBB throughput req: {env.embb_cfg.throughput_requirement_mbps} Mbps")
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial URLLC users: {len(env.urllc_users)}")
    print(f"Initial eMBB users: {len(env.embb_users)}")
    
    # Run 24 hours
    total_profit = 0
    total_reward = 0
    urllc_viols = []
    embb_viols = []
    
    for step in range(24):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_profit += info['profit']
        total_reward += reward
        urllc_viols.append(info['urllc_violation_rate'])
        embb_viols.append(info['embb_violation_rate'])
        
        if step % 6 == 0:
            env.render()
    
    print(f"\n{'='*60}")
    print(f"24-Hour Summary")
    print(f"{'='*60}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Avg URLLC Violation: {np.mean(urllc_viols):.4f} (target: {env.cmdp_cfg.urllc_violation_threshold})")
    print(f"Avg eMBB Violation: {np.mean(embb_viols):.4f} (target: {env.cmdp_cfg.embb_violation_threshold})")
    print(f"Final Users: URLLC={len(env.urllc_users)}, eMBB={len(env.embb_users)}")