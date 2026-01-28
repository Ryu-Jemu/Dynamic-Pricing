"""
5G O-RAN Network Slicing CMDP Environment

A Gymnasium environment for Constrained Markov Decision Process (CMDP)
based network slicing with three-part tariff pricing optimization.

Key Features:
- Three-part tariff pricing (Access fee F, Allowance D, Overage price p)
- CMDP formulation with QoS constraints
- Online user arrivals (NHPP) and churn (hazard model)
- 3GPP-compliant channel and QoS models
- Decomposed cost model (energy, spectrum, backhaul)

References:
- 3GPP TR 38.901: Channel models
- 3GPP TS 38.101: NR resource grid
- Fibich et al. (2017): Three-part tariff theory
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
from env.nr_prb_table import NRResourceGrid, get_n_rb
from env.channel_38901 import ChannelModel3GPP38901, UserChannel
from env.qos_fbl import URLLCQoSModel, FBLCapacity
from env.qos_embb import EMMBQoSModel, SpectralEfficiencyCalculator
from env.scheduler import (
    TwoStageScheduler, UserAllocationState, SchedulingResult
)
from models.tariff_three_part import (
    DynamicTariffManager, TariffPlan, action_to_price_factors
)
from models.arrivals_nhpp import SliceArrivalManager
from models.churn_hazard import ChurnManager
from models.costs import CostManager


@dataclass
class User:
    """Complete user state for simulation."""
    user_id: int
    slice_type: str  # "URLLC" or "eMBB"
    
    # Location and channel
    distance_m: float
    angle_rad: float
    sinr_db: float = 15.0
    is_los: bool = True
    
    # QoS tracking
    qos_violations_total: int = 0
    qos_violations_24h: int = 0
    consecutive_violations: int = 0
    satisfaction: float = 1.0
    
    # Usage and billing
    hourly_usage_mb: float = 0.0
    cumulative_usage_mb: float = 0.0
    overage_payment_total: float = 0.0
    
    # Scheduling
    allocated_prb: int = 0
    
    # Timing
    subscription_hour: int = 0
    hours_active: int = 0


@dataclass
class StepInfo:
    """Detailed information from environment step."""
    # Revenue breakdown
    revenue_total: float = 0.0
    revenue_urllc: float = 0.0
    revenue_embb: float = 0.0
    revenue_access_fee: float = 0.0
    revenue_overage: float = 0.0
    
    # Cost breakdown
    cost_total: float = 0.0
    cost_energy: float = 0.0
    cost_spectrum: float = 0.0
    cost_backhaul: float = 0.0
    cost_fixed: float = 0.0
    cost_acquisition: float = 0.0
    
    # Profit
    profit: float = 0.0
    
    # QoS metrics
    urllc_violation_rate: float = 0.0
    embb_violation_rate: float = 0.0
    urllc_violations: int = 0
    embb_violations: int = 0
    
    # User dynamics
    urllc_arrivals: int = 0
    embb_arrivals: int = 0
    urllc_churns: int = 0
    embb_churns: int = 0
    n_urllc_users: int = 0
    n_embb_users: int = 0
    
    # Resource utilization
    prb_utilization: float = 0.0
    urllc_prb_used: int = 0
    embb_prb_used: int = 0
    
    # Pricing
    urllc_fee_factor: float = 1.0
    urllc_overage_factor: float = 1.0
    embb_fee_factor: float = 1.0
    embb_overage_factor: float = 1.0
    
    # Constraint values (for CMDP)
    constraint_urllc: float = 0.0  # Violation rate - threshold
    constraint_embb: float = 0.0


class NetworkSlicingCMDPEnv(gym.Env):
    """
    5G O-RAN Network Slicing CMDP Environment.
    
    Observation Space (32 dimensions):
        [0-5]   URLLC state: users, violation_rate, avg_sinr, prb_util, fee_factor, overage_factor
        [6-11]  eMBB state: users, violation_rate, avg_sinr, prb_util, fee_factor, overage_factor
        [12-17] System: total_prb_util, revenue_norm, profit_norm, churn_rate, arrival_rate, cost_norm
        [18-21] Allowance: urllc_overage_ratio, embb_overage_ratio, urllc_near_limit, embb_near_limit
        [22-25] Time: hour_sin, hour_cos, day_sin, day_cos
        [26-31] History: 24h_revenue, 24h_violations, cum_churn, cum_arrival, 24h_profit, 24h_cost
    
    Action Space (4 dimensions, continuous [-1, 1]):
        [0] URLLC access fee factor adjustment
        [1] URLLC overage price factor adjustment
        [2] eMBB access fee factor adjustment
        [3] eMBB overage price factor adjustment
        
        Mapped to [0.8, 1.2] price factors
    
    Reward: Profit = Revenue - Cost (normalized)
    
    Constraints (CMDP):
        - URLLC violation rate <= threshold (default 0.001 = 0.1%)
        - eMBB violation rate <= threshold (default 0.01 = 1%)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the CMDP environment.
        
        Args:
            config: Scenario configuration (uses defaults if None)
            render_mode: Rendering mode
        """
        super().__init__()
        
        # Load configuration
        self.config = config or ScenarioConfig()
        self.render_mode = render_mode
        
        # Extract sub-configs
        self.sys_cfg = self.config.system
        self.urllc_cfg = self.config.urllc
        self.embb_cfg = self.config.embb
        self.tariff_cfg = self.config.tariff
        self.demand_cfg = self.config.demand
        self.cost_cfg = self.config.cost
        self.cmdp_cfg = self.config.cmdp
        
        # Initialize resource grid
        self.n_rb = get_n_rb(
            bandwidth_mhz=self.sys_cfg.bandwidth_mhz,
            scs_khz=self.sys_cfg.scs_khz
        )
        self.resource_grid = NRResourceGrid(
            bandwidth_mhz=self.sys_cfg.bandwidth_mhz,
            scs_khz=self.sys_cfg.scs_khz
        )
        
        # Initialize models
        self._init_models()
        
        # Define spaces
        self._define_spaces()
        
        # Episode parameters
        self.episode_length = 168  # 1 week in hours
        self.current_hour = 0
        
        # User tracking
        self.urllc_users: Dict[int, User] = {}
        self.embb_users: Dict[int, User] = {}
        self.next_user_id = 1
        
        # History tracking
        self.revenue_history = deque(maxlen=24)
        self.violation_history = deque(maxlen=24)
        self.profit_history = deque(maxlen=24)
        self.cost_history = deque(maxlen=24)
        
        # Cumulative metrics
        self.cumulative_churn = 0
        self.cumulative_arrivals = 0
        self.cumulative_revenue = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_violations_urllc = 0
        self.cumulative_violations_embb = 0
        
        # Current price factors
        self.urllc_fee_factor = 1.0
        self.urllc_overage_factor = 1.0
        self.embb_fee_factor = 1.0
        self.embb_overage_factor = 1.0
        
        # Last step info
        self.last_info: Optional[StepInfo] = None
    
    def _init_models(self):
        """Initialize all sub-models."""
        # Channel model (3GPP TR 38.901 UMi-Street Canyon)
        self.channel_model = ChannelModel3GPP38901(
            frequency_ghz=self.sys_cfg.frequency_ghz,
            h_bs=self.sys_cfg.gnb_height_m,
            h_ut=self.sys_cfg.ue_height_m
        )
        # Store tx_power and cell_radius for SINR calculations
        self.tx_power_dbm = self.sys_cfg.gnb_tx_power_dbm
        self.cell_radius_m = self.sys_cfg.cell_radius_m
        
        # QoS models - use individual parameters from config
        self.urllc_qos = URLLCQoSModel(
            latency_requirement_ms=self.urllc_cfg.latency_requirement_ms,
            reliability_requirement=self.urllc_cfg.reliability_requirement,
            packet_size_bytes=self.urllc_cfg.packet_size_bytes,
            scs_khz=self.sys_cfg.scs_khz
        )
        self.embb_qos = EMMBQoSModel(
            throughput_requirement_mbps=self.embb_cfg.throughput_requirement_mbps,
            latency_tolerance_ms=self.embb_cfg.latency_tolerance_ms,
            reliability_requirement=self.embb_cfg.reliability_requirement,
            scs_khz=self.sys_cfg.scs_khz
        )
        
        # Scheduler - uses config objects
        self.scheduler = TwoStageScheduler(
            resource_grid=self.resource_grid,
            urllc_config=self.urllc_cfg,
            embb_config=self.embb_cfg
        )
        
        # Tariff manager - use base values from tariff config
        self.tariff_manager = DynamicTariffManager(
            urllc_base_fee=self.tariff_cfg.urllc_base_fee_hourly,
            urllc_allowance_mb=self.tariff_cfg.urllc_allowance_mb,
            urllc_base_overage=self.tariff_cfg.urllc_overage_price_per_mb,
            embb_base_fee=self.tariff_cfg.embb_base_fee_hourly,
            embb_allowance_mb=self.tariff_cfg.embb_allowance_mb,
            embb_base_overage=self.tariff_cfg.embb_overage_price_per_mb,
            billing_cycle_hours=self.tariff_cfg.billing_cycle_hours
        )
        
        # Arrival model - use defaults (configured for URLLC B2B, eMBB B2C patterns)
        self.arrival_manager = SliceArrivalManager()
        
        # Churn model - use defaults
        self.churn_manager = ChurnManager()
        
        # Cost model - use defaults
        self.cost_manager = CostManager()
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation space: 32 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float32
        )
        
        # Action space: 4 continuous dimensions [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Constraint space for CMDP (2 constraints)
        self.constraint_dim = 2
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Reset time
        self.current_hour = 0
        
        # Clear users
        self.urllc_users.clear()
        self.embb_users.clear()
        self.next_user_id = 1
        
        # Reset tariff manager
        self.tariff_manager.reset()
        
        # Reset history
        self.revenue_history.clear()
        self.violation_history.clear()
        self.profit_history.clear()
        self.cost_history.clear()
        
        # Reset cumulative metrics
        self.cumulative_churn = 0
        self.cumulative_arrivals = 0
        self.cumulative_revenue = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_violations_urllc = 0
        self.cumulative_violations_embb = 0
        
        # Reset price factors
        self.urllc_fee_factor = 1.0
        self.urllc_overage_factor = 1.0
        self.embb_fee_factor = 1.0
        self.embb_overage_factor = 1.0
        
        # Initialize users
        self._spawn_initial_users()
        
        # Get initial observation
        obs = self._get_observation()
        info = {"initial": True}
        
        return obs, info
    
    def _spawn_initial_users(self):
        """Spawn initial users at episode start."""
        # URLLC users
        n_urllc = self.urllc_cfg.initial_users
        for _ in range(n_urllc):
            self._add_user("URLLC")
        
        # eMBB users
        n_embb = self.embb_cfg.initial_users
        for _ in range(n_embb):
            self._add_user("eMBB")
    
    def _add_user(self, slice_type: str) -> User:
        """Add a new user to the network."""
        user_id = self.next_user_id
        self.next_user_id += 1
        
        # Generate random location
        distance, angle = self._generate_user_location()
        
        # Calculate initial SINR
        sinr_db, is_los = self.channel_model.calculate_sinr(distance)
        
        # Create user
        user = User(
            user_id=user_id,
            slice_type=slice_type,
            distance_m=distance,
            angle_rad=angle,
            sinr_db=sinr_db,
            is_los=is_los,
            subscription_hour=self.current_hour
        )
        
        # Add to appropriate dict
        if slice_type == "URLLC":
            self.urllc_users[user_id] = user
        else:
            self.embb_users[user_id] = user
        
        # Register with tariff manager
        self.tariff_manager.add_user(user_id, slice_type, self.current_hour)
        
        # Register with churn manager
        self.churn_manager.register_user(user_id, slice_type)
        
        return user
    
    def _remove_user(self, user_id: int, slice_type: str):
        """Remove a user from the network (churn)."""
        if slice_type == "URLLC":
            if user_id in self.urllc_users:
                del self.urllc_users[user_id]
        else:
            if user_id in self.embb_users:
                del self.embb_users[user_id]
        
        # Remove from tariff manager (only needs user_id)
        self.tariff_manager.remove_user(user_id)
        
        # Remove from churn manager
        self.churn_manager.remove_user(user_id)
        
        # Remove from scheduler tracking
        self.scheduler.remove_user(user_id)
    
    def _generate_user_location(self) -> Tuple[float, float]:
        """Generate random user location within cell."""
        cell_radius = self.sys_cfg.cell_radius_m
        min_distance = 10.0  # Minimum distance from gNB
        
        # Uniform distribution over cell area
        r = cell_radius * np.sqrt(
            self.np_random.uniform((min_distance/cell_radius)**2, 1.0)
        )
        theta = self.np_random.uniform(0, 2 * np.pi)
        
        return r, theta
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step (1 hour).
        
        Args:
            action: [urllc_fee_adj, urllc_overage_adj, embb_fee_adj, embb_overage_adj]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Decode action to price factors
        price_factors = action_to_price_factors(action)
        self.urllc_fee_factor = price_factors['urllc_fee']
        self.urllc_overage_factor = price_factors['urllc_overage']
        self.embb_fee_factor = price_factors['embb_fee']
        self.embb_overage_factor = price_factors['embb_overage']
        
        # Update tariff manager
        self.tariff_manager.update_prices(
            urllc_fee_factor=self.urllc_fee_factor,
            urllc_overage_factor=self.urllc_overage_factor,
            embb_fee_factor=self.embb_fee_factor,
            embb_overage_factor=self.embb_overage_factor
        )
        
        # 2. Update channels for all users
        self._update_channels()
        
        # 3. Generate traffic usage
        self._generate_traffic()
        
        # 4. PRB allocation and scheduling
        scheduling_result = self._allocate_resources()
        
        # 5. Evaluate QoS
        urllc_violations, embb_violations = self._evaluate_qos(scheduling_result)
        
        # 6. Bill users and calculate revenue
        revenue_info = self._bill_users()
        
        # 7. Calculate costs
        cost_info = self._calculate_costs(scheduling_result)
        
        # 8. Process churn
        churn_info = self._process_churn()
        
        # 9. Process arrivals
        arrival_info = self._process_arrivals()
        
        # 10. Calculate reward and constraints
        profit = revenue_info['total'] - cost_info['total']
        reward = self._calculate_reward(profit)
        
        # Constraint values with per-state (time-varying) thresholds
        # Reference: UC Davis CLARA 2024, Computer Communications 2024
        n_urllc = len(self.urllc_users)
        n_embb = len(self.embb_users)
        urllc_viol_rate = urllc_violations / max(1, n_urllc)
        embb_viol_rate = embb_violations / max(1, n_embb)
        
        # Get time-varying constraint thresholds based on current hour
        hour_of_day = self.current_hour % 24
        urllc_threshold = self.cmdp_cfg.get_constraint_threshold("URLLC", hour_of_day)
        embb_threshold = self.cmdp_cfg.get_constraint_threshold("eMBB", hour_of_day)
        
        constraint_urllc = urllc_viol_rate - urllc_threshold
        constraint_embb = embb_viol_rate - embb_threshold
        
        # 11. Update history
        self._update_history(profit, revenue_info['total'], cost_info['total'],
                           urllc_viol_rate, embb_viol_rate)
        
        # 12. Advance time
        self.current_hour += 1
        
        # 13. Check termination
        terminated = False
        truncated = self.current_hour >= self.episode_length
        
        # 14. Build info dict
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
        
        return obs, reward, terminated, truncated, info
    
    def _update_channels(self):
        """
        Update channel state for all users.
        
        Supports deterministic channel mode for focused pricing optimization.
        Reference: IEEE TMC 2023 - Single gNB pricing optimization
        """
        # Check if deterministic channel mode is enabled
        deterministic_mode = getattr(self.sys_cfg, 'enable_deterministic_channel', False)
        shadow_mode = getattr(self.sys_cfg, 'shadow_fading_mode', 'stochastic')
        
        for user in list(self.urllc_users.values()) + list(self.embb_users.values()):
            if deterministic_mode:
                # Fixed channel mode - no mobility, no fading
                # Use fixed SINR if specified, otherwise maintain current
                slice_type = user.slice_type
                if slice_type == "URLLC" and self.sys_cfg.fixed_urllc_sinr_db is not None:
                    user.sinr_db = self.sys_cfg.fixed_urllc_sinr_db
                elif slice_type == "eMBB" and self.sys_cfg.fixed_embb_sinr_db is not None:
                    user.sinr_db = self.sys_cfg.fixed_embb_sinr_db
                # No distance update in deterministic mode
            else:
                # Stochastic mode - normal mobility and fading
                # Slight distance variation (mobility)
                user.distance_m += self.np_random.normal(0, 2.0)
                user.distance_m = np.clip(user.distance_m, 10, self.sys_cfg.cell_radius_m)
                
                # Recalculate SINR based on shadow fading mode
                if shadow_mode == "fixed":
                    # No shadow fading - deterministic path loss only
                    user.sinr_db, user.is_los = self.channel_model.calculate_sinr(
                        user.distance_m, 
                        include_shadowing=False
                    )
                else:
                    # Normal stochastic channel
                    user.sinr_db, user.is_los = self.channel_model.calculate_sinr(user.distance_m)
    
    def _generate_traffic(self):
        """Generate hourly traffic usage for each user."""
        for user in self.urllc_users.values():
            # URLLC: Small packets, relatively constant usage
            base_usage = self.tariff_cfg.urllc_allowance_mb / 168  # Spread over week
            variation = self.np_random.exponential(0.2) * base_usage
            user.hourly_usage_mb = base_usage + variation
            user.cumulative_usage_mb += user.hourly_usage_mb
        
        for user in self.embb_users.values():
            # eMBB: Larger variation, time-of-day effects
            hour_of_day = self.current_hour % 24
            
            # Peak usage evening hours
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
        # Build user allocation states
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
                avg_throughput_mbps=50.0  # Will be updated by scheduler
            )
            for u in self.embb_users.values()
        ]
        
        # Get recent violation rates
        recent_urllc_viol = self._get_recent_violation_rate("URLLC")
        recent_embb_viol = self._get_recent_violation_rate("eMBB")
        
        # Execute scheduling
        result = self.scheduler.schedule(
            urllc_users=urllc_states,
            embb_users=embb_states,
            urllc_violation_rate=recent_urllc_viol,
            embb_violation_rate=recent_embb_viol
        )
        
        # Update user PRB allocations
        for user in self.urllc_users.values():
            user.allocated_prb = result.urllc_allocation.allocations.get(user.user_id, 0)
        
        for user in self.embb_users.values():
            user.allocated_prb = result.embb_allocation.allocations.get(user.user_id, 0)
        
        return result
    
    def _evaluate_qos(self, scheduling_result: SchedulingResult) -> Tuple[int, int]:
        """Evaluate QoS for all users and count violations."""
        urllc_violations = 0
        embb_violations = 0
        
        # URLLC QoS evaluation
        for user in self.urllc_users.values():
            satisfied, viol_prob, _ = self.urllc_qos.evaluate_qos(
                sinr_db=user.sinr_db,
                allocated_prb=user.allocated_prb
            )
            
            # Probabilistic violation
            if self.np_random.random() < viol_prob:
                urllc_violations += 1
                user.qos_violations_total += 1
                user.qos_violations_24h += 1
                user.consecutive_violations += 1
            else:
                user.consecutive_violations = 0
        
        # eMBB QoS evaluation
        for user in self.embb_users.values():
            satisfied, violation_metric, details = self.embb_qos.evaluate_qos(
                sinr_db=user.sinr_db,
                allocated_prb=user.allocated_prb
            )
            
            if not satisfied:
                embb_violations += 1
                user.qos_violations_total += 1
                user.qos_violations_24h += 1
                user.consecutive_violations += 1
            else:
                user.consecutive_violations = 0
            
            # Update throughput tracker
            if 'achieved_throughput_mbps' in details:
                self.scheduler.update_embb_throughput(
                    user.user_id, 
                    details['achieved_throughput_mbps']
                )
        
        # Update cumulative
        self.cumulative_violations_urllc += urllc_violations
        self.cumulative_violations_embb += embb_violations
        
        return urllc_violations, embb_violations
    
    def _bill_users(self) -> Dict[str, float]:
        """Bill all users and calculate revenue."""
        total_revenue = 0.0
        urllc_revenue = 0.0
        embb_revenue = 0.0
        access_fee_revenue = 0.0
        overage_revenue = 0.0
        
        # Bill URLLC users
        for user in self.urllc_users.values():
            revenue, overage, _ = self.tariff_manager.bill_user(
                user_id=user.user_id,
                usage_mb=user.hourly_usage_mb,
                current_hour=self.current_hour
            )
            total_revenue += revenue
            urllc_revenue += revenue
            user.overage_payment_total += overage
            
            # Estimate access fee vs overage split
            effective_fee = self.tariff_cfg.urllc_base_fee_hourly * self.urllc_fee_factor
            access_fee_revenue += effective_fee
            overage_revenue += overage
        
        # Bill eMBB users
        for user in self.embb_users.values():
            revenue, overage, _ = self.tariff_manager.bill_user(
                user_id=user.user_id,
                usage_mb=user.hourly_usage_mb,
                current_hour=self.current_hour
            )
            total_revenue += revenue
            embb_revenue += revenue
            user.overage_payment_total += overage
            
            effective_fee = self.tariff_cfg.embb_base_fee_hourly * self.embb_fee_factor
            access_fee_revenue += effective_fee
            overage_revenue += overage
        
        self.cumulative_revenue += total_revenue
        
        return {
            'total': total_revenue,
            'urllc': urllc_revenue,
            'embb': embb_revenue,
            'access_fee': access_fee_revenue,
            'overage': overage_revenue
        }
    
    def _calculate_costs(self, scheduling_result: SchedulingResult) -> Dict[str, float]:
        """Calculate operational costs."""
        # PRB utilization
        prb_util = scheduling_result.total_prb_utilization
        
        # Total data (approximate)
        total_data_mb = sum(u.hourly_usage_mb for u in self.urllc_users.values())
        total_data_mb += sum(u.hourly_usage_mb for u in self.embb_users.values())
        
        # New users this hour (from recent arrivals)
        new_users = getattr(self, '_recent_arrivals', 0)
        
        # Calculate costs
        total_cost, breakdown = self.cost_manager.calculate_hourly_cost(
            prb_utilization=prb_util,
            total_data_mb=total_data_mb,
            new_users=new_users
        )
        
        self.cumulative_cost += total_cost
        
        return {
            'total': total_cost,
            'energy': breakdown['energy'],
            'spectrum': breakdown['spectrum'],
            'backhaul': breakdown['backhaul'],
            'fixed': breakdown['fixed'],
            'acquisition': breakdown.get('acquisition', 0.0)
        }
    
    def _process_churn(self) -> Dict[str, int]:
        """Process user churn."""
        urllc_churned, embb_churned = self.churn_manager.evaluate_all_users(
            urllc_fee_factor=self.urllc_fee_factor,
            embb_fee_factor=self.embb_fee_factor,
            hour=self.current_hour
        )
        
        # Remove churned users
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
        # Get recent violation rates for QoS effect
        urllc_viol_rate = self._get_recent_violation_rate("URLLC")
        embb_viol_rate = self._get_recent_violation_rate("eMBB")
        
        # Generate arrivals using unified API
        urllc_arrivals, embb_arrivals = self.arrival_manager.generate_arrivals(
            hour=self.current_hour,
            urllc_fee_factor=self.urllc_fee_factor,
            urllc_overage_factor=self.urllc_overage_factor,
            urllc_violation_rate=urllc_viol_rate,
            embb_fee_factor=self.embb_fee_factor,
            embb_overage_factor=self.embb_overage_factor,
            embb_violation_rate=embb_viol_rate
        )
        
        # Apply user limits
        max_urllc = self.urllc_cfg.max_users
        max_embb = self.embb_cfg.max_users
        
        urllc_arrivals = min(urllc_arrivals, max_urllc - len(self.urllc_users))
        embb_arrivals = min(embb_arrivals, max_embb - len(self.embb_users))
        
        # Add new users
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
    
    def _calculate_reward(self, profit: float) -> float:
        """
        Calculate reward with Customer Lifetime Value (CLV) consideration.
        
        Enhanced reward function that accounts for:
        1. Immediate profit (revenue - cost)
        2. CLV loss from churned customers
        3. CLV gain from new customers
        
        Reference:
            The CLV approach penalizes short-term profit maximization that
            leads to customer loss, encouraging sustainable pricing strategies.
            
            CLV = Σ_{t=0}^∞ (margin × retention^t) / (1+r)^t
            Simplified: CLV ≈ margin × (retention / (1 - retention + r))
        """
        # Expected max profit (rough estimate)
        max_users = self.urllc_cfg.max_users + self.embb_cfg.max_users
        max_revenue = (self.urllc_cfg.max_users * self.tariff_cfg.urllc_base_fee_hourly +
                      self.embb_cfg.max_users * self.tariff_cfg.embb_base_fee_hourly)
        
        # Base reward: normalized profit
        base_reward = profit / max(100.0, max_revenue * 0.5)
        
        # CLV-based adjustment
        clv_adjustment = self._calculate_clv_adjustment()
        
        # Combine: base profit - CLV loss penalty
        reward = base_reward - clv_adjustment
        
        return float(reward)
    
    def _calculate_clv_adjustment(self) -> float:
        """
        Calculate CLV-based reward adjustment for churn/arrivals.
        
        Estimates the long-term value impact of customer dynamics.
        
        Reference:
            CLV literature suggests using 6-12 month horizon for telecom.
            We use expected monthly revenue × average lifetime months.
        """
        # Get recent churn and arrival counts
        recent_urllc_churns = self._step_info.urllc_churns if hasattr(self, '_step_info') else 0
        recent_embb_churns = self._step_info.embb_churns if hasattr(self, '_step_info') else 0
        recent_urllc_arrivals = self._step_info.urllc_arrivals if hasattr(self, '_step_info') else 0
        recent_embb_arrivals = self._step_info.embb_arrivals if hasattr(self, '_step_info') else 0
        
        # Skip if no dynamics
        if recent_urllc_churns + recent_embb_churns + recent_urllc_arrivals + recent_embb_arrivals == 0:
            return 0.0
        
        # CLV parameters (could be moved to config)
        # Average customer lifetime in months (literature: 12-24 months for telecom)
        urllc_lifetime_months = 18.0  # B2B longer retention
        embb_lifetime_months = 12.0   # B2C shorter retention
        
        # Profit margin (typically 30-50% for telecom)
        profit_margin = 0.35
        
        # Discount rate (monthly, ~10% annual → ~0.8% monthly)
        monthly_discount_rate = 0.008
        
        # Calculate monthly revenue per user type
        urllc_monthly_revenue = self.tariff_cfg.urllc_base_fee_hourly * 24 * 30  # hourly to monthly
        embb_monthly_revenue = self.tariff_cfg.embb_base_fee_hourly * 24 * 30
        
        # Simplified CLV: margin × monthly_revenue × lifetime_months
        # (ignoring discounting for simplicity in RL context)
        urllc_clv = profit_margin * urllc_monthly_revenue * urllc_lifetime_months / 12  # Annualized
        embb_clv = profit_margin * embb_monthly_revenue * embb_lifetime_months / 12
        
        # CLV loss from churn
        clv_loss = (recent_urllc_churns * urllc_clv + 
                   recent_embb_churns * embb_clv)
        
        # CLV gain from arrivals (new customers have full CLV potential)
        # Apply 50% discount for uncertainty about new customer retention
        clv_gain = 0.5 * (recent_urllc_arrivals * urllc_clv + 
                         recent_embb_arrivals * embb_clv)
        
        # Net CLV impact (loss - gain)
        # Normalize by max_revenue to keep scale consistent
        max_revenue = (self.urllc_cfg.max_users * self.tariff_cfg.urllc_base_fee_hourly +
                      self.embb_cfg.max_users * self.tariff_cfg.embb_base_fee_hourly)
        
        # CLV weight factor (controls how much CLV affects reward)
        # 0.1 means CLV impact is ~10% of immediate profit impact
        clv_weight = 0.1
        
        clv_adjustment = clv_weight * (clv_loss - clv_gain) / max(100.0, max_revenue * 0.5)
        
        return clv_adjustment
    
    def _get_recent_violation_rate(self, slice_type: str) -> float:
        """Get recent violation rate for a slice."""
        if slice_type == "URLLC":
            users = self.urllc_users
        else:
            users = self.embb_users
        
        if not users:
            return 0.0
        
        total_violations = sum(u.qos_violations_24h for u in users.values())
        # Violations per user per hour, normalized
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
        for user in list(self.urllc_users.values()) + list(self.embb_users.values()):
            if self.current_hour % 24 == 0:
                user.qos_violations_24h = max(0, user.qos_violations_24h - 1)
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = np.zeros(32, dtype=np.float32)
        
        n_urllc = len(self.urllc_users)
        n_embb = len(self.embb_users)
        
        # [0-5] URLLC state
        obs[0] = n_urllc / max(1, self.urllc_cfg.max_users)
        obs[1] = self._get_recent_violation_rate("URLLC")
        obs[2] = np.mean([u.sinr_db for u in self.urllc_users.values()]) / 40 if n_urllc > 0 else 0
        obs[3] = sum(u.allocated_prb for u in self.urllc_users.values()) / max(1, self.n_rb * 0.5)
        obs[4] = self.urllc_fee_factor
        obs[5] = self.urllc_overage_factor
        
        # [6-11] eMBB state
        obs[6] = n_embb / max(1, self.embb_cfg.max_users)
        obs[7] = self._get_recent_violation_rate("eMBB")
        obs[8] = np.mean([u.sinr_db for u in self.embb_users.values()]) / 40 if n_embb > 0 else 0
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
        
        obs[13] = recent_revenue / 1000  # Normalize
        obs[14] = recent_profit / 500
        obs[15] = self.cumulative_churn / max(1, self.current_hour + 1)
        obs[16] = self.cumulative_arrivals / max(1, self.current_hour + 1)
        obs[17] = recent_cost / 100
        
        # [18-21] Allowance state
        urllc_overage_stats = self.tariff_manager.get_allowance_stats("URLLC")
        embb_overage_stats = self.tariff_manager.get_allowance_stats("eMBB")
        
        obs[18] = urllc_overage_stats.get('mean_overage_ratio', 0)
        obs[19] = embb_overage_stats.get('mean_overage_ratio', 0)
        obs[20] = urllc_overage_stats.get('near_limit_fraction', 0)
        obs[21] = embb_overage_stats.get('near_limit_fraction', 0)
        
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
        else:
            obs[27] = 0
        
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
        info = StepInfo(
            revenue_total=revenue_info['total'],
            revenue_urllc=revenue_info['urllc'],
            revenue_embb=revenue_info['embb'],
            revenue_access_fee=revenue_info['access_fee'],
            revenue_overage=revenue_info['overage'],
            
            cost_total=cost_info['total'],
            cost_energy=cost_info['energy'],
            cost_spectrum=cost_info['spectrum'],
            cost_backhaul=cost_info['backhaul'],
            cost_fixed=cost_info['fixed'],
            cost_acquisition=cost_info.get('acquisition', 0),
            
            profit=profit,
            
            urllc_violation_rate=urllc_viol_rate,
            embb_violation_rate=embb_viol_rate,
            urllc_violations=urllc_violations,
            embb_violations=embb_violations,
            
            urllc_arrivals=arrival_info['urllc'],
            embb_arrivals=arrival_info['embb'],
            urllc_churns=churn_info['urllc'],
            embb_churns=churn_info['embb'],
            n_urllc_users=len(self.urllc_users),
            n_embb_users=len(self.embb_users),
            
            prb_utilization=scheduling_result.total_prb_utilization,
            urllc_prb_used=scheduling_result.urllc_allocation.total_allocated,
            embb_prb_used=scheduling_result.embb_allocation.total_allocated,
            
            urllc_fee_factor=self.urllc_fee_factor,
            urllc_overage_factor=self.urllc_overage_factor,
            embb_fee_factor=self.embb_fee_factor,
            embb_overage_factor=self.embb_overage_factor,
            
            constraint_urllc=constraint_urllc,
            constraint_embb=constraint_embb
        )
        
        return info.__dict__
    
    def get_constraint_values(self) -> np.ndarray:
        """
        Get current constraint values for CMDP.
        
        Returns:
            Array of constraint values [urllc_constraint, embb_constraint]
            Negative = satisfied, Positive = violated
        """
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
                print(f"Arrivals: URLLC={info['urllc_arrivals']}, eMBB={info['embb_arrivals']}")
                print(f"Churns: URLLC={info['urllc_churns']}, eMBB={info['embb_churns']}")
                
                print(f"\nRevenue: ${info['revenue_total']:.2f}")
                print(f"  Access fee: ${info['revenue_access_fee']:.2f}")
                print(f"  Overage: ${info['revenue_overage']:.2f}")
                
                print(f"\nCost: ${info['cost_total']:.2f}")
                print(f"Profit: ${info['profit']:.2f}")
                
                print(f"\nQoS Violations:")
                print(f"  URLLC: {info['urllc_violation_rate']:.4f} (threshold: {self.cmdp_cfg.urllc_violation_threshold})")
                print(f"  eMBB: {info['embb_violation_rate']:.4f} (threshold: {self.cmdp_cfg.embb_violation_threshold})")
                
                print(f"\nPRB Utilization: {info['prb_utilization']:.2%}")
                
                print(f"\nPrice Factors:")
                print(f"  URLLC: fee={info['urllc_fee_factor']:.2f}, overage={info['urllc_overage_factor']:.2f}")
                print(f"  eMBB: fee={info['embb_fee_factor']:.2f}, overage={info['embb_overage_factor']:.2f}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Network Slicing CMDP Environment Test")
    print("=" * 70)
    
    # Create environment
    env = NetworkSlicingCMDPEnv(render_mode="human")
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Total PRBs: {env.n_rb}")
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial URLLC users: {len(env.urllc_users)}")
    print(f"Initial eMBB users: {len(env.embb_users)}")
    
    # Run a few steps
    total_profit = 0
    for step in range(24):  # 24 hours
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_profit += info['profit']
        
        if step % 6 == 0:
            env.render()
    
    print(f"\n{'='*60}")
    print(f"24-Hour Summary")
    print(f"{'='*60}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Cumulative Revenue: ${env.cumulative_revenue:.2f}")
    print(f"Cumulative Cost: ${env.cumulative_cost:.2f}")
    print(f"Total Arrivals: {env.cumulative_arrivals}")
    print(f"Total Churns: {env.cumulative_churn}")
    print(f"Final Users: URLLC={len(env.urllc_users)}, eMBB={len(env.embb_users)}")
    
    # Get constraints
    constraints = env.get_constraint_values()
    print(f"\nConstraint Values: URLLC={constraints[0]:.4f}, eMBB={constraints[1]:.4f}")
    print("(Negative = satisfied, Positive = violated)")
    
    print("\n" + "=" * 70)
    print("Environment test completed!")
