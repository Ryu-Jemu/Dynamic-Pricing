"""
Two-Stage PRB Scheduler for 5G O-RAN Network Slicing

Stage 1: Slice-level allocation
    - Guarantee minimum PRB to URLLC for reliability/latency
    - Distribute remaining PRBs to eMBB based on demand
    
Stage 2: In-slice scheduling
    - URLLC: Priority-based with consecutive-violation protection
    - eMBB: Proportional Fair (PF) scheduling

References:
    - 3GPP TS 38.214: Physical layer procedures for data
    - Wang et al. (2024): DRL-based URLLC/eMBB scheduling, Wiley
    - Springer LNCS 2021: Preemptive priority queuing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from config.scenario_config import URLLCConfig, EMMBConfig
from env.qos_fbl import URLLCQoSModel
from env.qos_embb import EMMBQoSModel, UserThroughputTracker
from env.nr_prb_table import NRResourceGrid


@dataclass
class UserAllocationState:
    """Per-user allocation state for scheduling decisions."""
    user_id: int
    slice_type: str  # "URLLC" or "eMBB"
    sinr_db: float
    distance_m: float
    consecutive_violations: int = 0
    qos_violations_24h: int = 0
    avg_throughput_mbps: float = 1.0  # For PF scheduling (eMBB)
    priority_score: float = 0.0
    allocated_prb: int = 0


@dataclass
class SliceAllocationResult:
    """Result of slice-level PRB allocation."""
    urllc_prb: int
    embb_prb: int
    reserved_prb: int
    urllc_utilization: float
    embb_utilization: float


@dataclass 
class UserAllocationResult:
    """Result of per-user PRB allocation."""
    allocations: Dict[int, int]  # user_id -> allocated_prb
    total_allocated: int
    utilization: float
    unserved_users: List[int]


class SliceLevelAllocator:
    """
    Stage 1: Slice-level PRB allocation.
    
    Guarantees minimum PRB for URLLC based on user count and QoS requirements,
    then distributes remaining PRBs to eMBB based on demand.
    """
    
    def __init__(
        self,
        total_prb: int,
        urllc_min_prb_per_user: int = 2,
        embb_min_prb_per_user: int = 1,
        reserved_prb_fraction: float = 0.05,
        urllc_priority_weight: float = 1.5
    ):
        """
        Args:
            total_prb: Total PRBs available (from 3GPP table)
            urllc_min_prb_per_user: Minimum PRB guarantee per URLLC user
            embb_min_prb_per_user: Minimum PRB guarantee per eMBB user
            reserved_prb_fraction: Fraction of PRBs to reserve for burst handling
            urllc_priority_weight: Priority weight for URLLC slice
        """
        self.total_prb = total_prb
        self.urllc_min_per_user = urllc_min_prb_per_user
        self.embb_min_per_user = embb_min_prb_per_user
        self.reserved_fraction = reserved_prb_fraction
        self.urllc_priority = urllc_priority_weight
        
        # Minimum reserved PRBs (at least 3 for flexibility)
        self.min_reserved = max(3, int(total_prb * reserved_prb_fraction))
    
    def allocate(
        self,
        n_urllc_users: int,
        n_embb_users: int,
        urllc_violation_rate: float = 0.0,
        embb_violation_rate: float = 0.0
    ) -> SliceAllocationResult:
        """
        Allocate PRBs to slices based on user counts and QoS status.
        
        Algorithm:
        1. Compute URLLC minimum requirement
        2. Adjust based on violation rate (give more if violations high)
        3. Allocate remaining to eMBB
        4. Ensure minimum reserved PRBs
        
        Args:
            n_urllc_users: Number of active URLLC users
            n_embb_users: Number of active eMBB users
            urllc_violation_rate: Recent URLLC QoS violation rate [0,1]
            embb_violation_rate: Recent eMBB QoS violation rate [0,1]
            
        Returns:
            SliceAllocationResult with PRB allocations
        """
        available_prb = self.total_prb - self.min_reserved
        
        # Step 1: URLLC base allocation
        urllc_base_need = n_urllc_users * self.urllc_min_per_user
        
        # Step 2: URLLC boost based on violation rate
        # If violations are high, allocate more PRBs (up to 50% extra)
        urllc_boost_factor = 1.0 + 0.5 * min(urllc_violation_rate * 10, 1.0)
        urllc_need = int(urllc_base_need * urllc_boost_factor)
        
        # Step 3: eMBB base allocation
        embb_base_need = n_embb_users * self.embb_min_per_user
        embb_boost_factor = 1.0 + 0.3 * min(embb_violation_rate * 5, 1.0)
        embb_need = int(embb_base_need * embb_boost_factor)
        
        # Step 4: Resolve contention
        total_need = urllc_need + embb_need
        
        if total_need <= available_prb:
            # Sufficient PRBs: allocate as needed, distribute excess
            urllc_prb = urllc_need
            embb_prb = embb_need
            excess = available_prb - total_need
            
            # Distribute excess proportionally with URLLC priority
            if n_urllc_users > 0 and n_embb_users > 0:
                urllc_share = self.urllc_priority / (self.urllc_priority + 1.0)
                urllc_extra = int(excess * urllc_share * 0.3)  # URLLC gets less extra
                embb_extra = excess - urllc_extra
                urllc_prb += urllc_extra
                embb_prb += embb_extra
            elif n_urllc_users > 0:
                urllc_prb += excess
            else:
                embb_prb += excess
        else:
            # Insufficient PRBs: URLLC has strict priority
            urllc_prb = min(urllc_need, available_prb)
            embb_prb = max(0, available_prb - urllc_prb)
        
        # Ensure non-negative
        urllc_prb = max(0, urllc_prb)
        embb_prb = max(0, embb_prb)
        reserved_prb = self.total_prb - urllc_prb - embb_prb
        
        # Compute utilization
        urllc_util = urllc_prb / max(1, urllc_need) if urllc_need > 0 else 1.0
        embb_util = embb_prb / max(1, embb_need) if embb_need > 0 else 1.0
        
        return SliceAllocationResult(
            urllc_prb=urllc_prb,
            embb_prb=embb_prb,
            reserved_prb=reserved_prb,
            urllc_utilization=min(1.0, urllc_util),
            embb_utilization=min(1.0, embb_util)
        )


class URLLCInSliceScheduler:
    """
    Stage 2a: URLLC in-slice scheduling.
    
    Priority-based allocation with:
    - Consecutive violation protection (boost priority for users with recent violations)
    - SINR-aware allocation (low SINR users may need more PRBs)
    - Minimum guarantee enforcement
    """
    
    def __init__(
        self,
        min_prb_per_user: int = 2,
        violation_priority_boost: float = 2.0,
        sinr_threshold_db: float = 10.0
    ):
        """
        Args:
            min_prb_per_user: Minimum PRBs to guarantee per user
            violation_priority_boost: Priority multiplier for users with violations
            sinr_threshold_db: SINR below which users get priority
        """
        self.min_prb = min_prb_per_user
        self.violation_boost = violation_priority_boost
        self.sinr_threshold = sinr_threshold_db
    
    def compute_priority(self, user: UserAllocationState) -> float:
        """
        Compute scheduling priority for a URLLC user.
        
        Priority factors:
        1. Consecutive violations (highest priority)
        2. Low SINR (needs more resources)
        3. Recent violation history
        
        Returns:
            Priority score (higher = more urgent)
        """
        priority = 1.0
        
        # Factor 1: Consecutive violations (exponential boost)
        if user.consecutive_violations > 0:
            priority *= self.violation_boost ** min(user.consecutive_violations, 3)
        
        # Factor 2: Low SINR penalty
        if user.sinr_db < self.sinr_threshold:
            sinr_factor = 1.0 + (self.sinr_threshold - user.sinr_db) / 10.0
            priority *= sinr_factor
        
        # Factor 3: 24h violation history
        priority *= 1.0 + 0.1 * min(user.qos_violations_24h, 10)
        
        return priority
    
    def allocate(
        self,
        users: List[UserAllocationState],
        available_prb: int
    ) -> UserAllocationResult:
        """
        Allocate PRBs to URLLC users with priority scheduling.
        
        Algorithm:
        1. Compute priority for each user
        2. Sort by priority (descending)
        3. Allocate minimum PRBs to all users if possible
        4. Distribute remaining PRBs by priority
        
        Args:
            users: List of URLLC user states
            available_prb: Total PRBs available for URLLC slice
            
        Returns:
            UserAllocationResult with per-user allocations
        """
        if not users:
            return UserAllocationResult(
                allocations={},
                total_allocated=0,
                utilization=0.0,
                unserved_users=[]
            )
        
        allocations = {u.user_id: 0 for u in users}
        remaining_prb = available_prb
        
        # Compute priorities
        for user in users:
            user.priority_score = self.compute_priority(user)
        
        # Sort by priority (descending)
        sorted_users = sorted(users, key=lambda u: u.priority_score, reverse=True)
        
        # Phase 1: Minimum guarantee (by priority order)
        for user in sorted_users:
            if remaining_prb >= self.min_prb:
                allocations[user.user_id] = self.min_prb
                remaining_prb -= self.min_prb
            elif remaining_prb > 0:
                allocations[user.user_id] = remaining_prb
                remaining_prb = 0
            else:
                break
        
        # Phase 2: Extra allocation for high-priority users
        if remaining_prb > 0:
            # Give extra PRBs to users with consecutive violations
            for user in sorted_users:
                if remaining_prb <= 0:
                    break
                if user.consecutive_violations > 0:
                    extra = min(2, remaining_prb)  # Max 2 extra PRBs
                    allocations[user.user_id] += extra
                    remaining_prb -= extra
        
        # Phase 3: Distribute remaining evenly
        if remaining_prb > 0 and len(users) > 0:
            extra_per_user = remaining_prb // len(users)
            if extra_per_user > 0:
                for user in users:
                    allocations[user.user_id] += extra_per_user
                remaining_prb -= extra_per_user * len(users)
        
        # Identify unserved users
        unserved = [u.user_id for u in users if allocations[u.user_id] == 0]
        total_allocated = sum(allocations.values())
        utilization = total_allocated / available_prb if available_prb > 0 else 0.0
        
        return UserAllocationResult(
            allocations=allocations,
            total_allocated=total_allocated,
            utilization=utilization,
            unserved_users=unserved
        )


class EMMBInSliceScheduler:
    """
    Stage 2b: eMBB in-slice scheduling with Multi-User Diversity.
    
    Proportional Fair (PF) scheduling with multi-user diversity gain:
    - PF metric = instantaneous_rate / average_rate
    - Multi-user diversity: exploits channel variations across users
    - Balances throughput and fairness
    - Minimum PRB guarantee per user
    
    Reference:
    - IEEE/ACM Trans. Netw. 2020 (Anand et al.): Joint URLLC/eMBB scheduling
    - IEEE Xplore 2022 (D-PF): Demand-based Proportional Fairness
    - arXiv 2025: Multi-user content diversity for significant UX gains
    """
    
    def __init__(
        self,
        min_prb_per_user: int = 1,
        pf_alpha: float = 1.0,
        pf_beta: float = 1.0,
        enable_diversity: bool = True,
        diversity_mode: str = "proportional_fair"
    ):
        """
        Args:
            min_prb_per_user: Minimum PRBs to guarantee per user
            pf_alpha: Exponent for instantaneous rate
            pf_beta: Exponent for average rate (fairness factor)
            enable_diversity: Enable multi-user diversity gain
            diversity_mode: "opportunistic", "proportional_fair", or "round_robin"
        """
        self.min_prb = min_prb_per_user
        self.alpha = pf_alpha
        self.beta = pf_beta
        self.enable_diversity = enable_diversity
        self.diversity_mode = diversity_mode
        
        # Track scheduling history for fairness
        self.last_scheduled: Dict[int, int] = {}  # user_id -> last_slot
        self.current_slot = 0
    
    def compute_multiuser_diversity_gain(
        self, 
        users: List[UserAllocationState],
        target_user_id: int
    ) -> float:
        """
        Compute multi-user diversity gain for opportunistic scheduling.
        
        When multiple users have varying channel conditions, scheduling
        users with temporarily good channels improves spectral efficiency.
        
        Diversity gain ≈ log(K) for K users with i.i.d. Rayleigh fading
        
        Reference: 
        - IEEE Trans. Comm. 2005: Multi-user diversity in fading channels
        - arXiv 2025: Significant gains from exploiting multi-user diversity
        
        Args:
            users: List of all eMBB users
            target_user_id: User to compute gain for
            
        Returns:
            Diversity gain factor (≥1.0)
        """
        if not self.enable_diversity or len(users) <= 1:
            return 1.0
        
        # Find target user's SINR
        target_sinr = None
        sinrs = []
        for u in users:
            sinrs.append(u.sinr_db)
            if u.user_id == target_user_id:
                target_sinr = u.sinr_db
        
        if target_sinr is None:
            return 1.0
        
        # Compute diversity gain based on SINR percentile
        # Users with above-average channel get diversity bonus
        sinrs_array = np.array(sinrs)
        mean_sinr = np.mean(sinrs_array)
        std_sinr = np.std(sinrs_array) + 1e-6  # Avoid division by zero
        
        # Z-score of target user's SINR
        z_score = (target_sinr - mean_sinr) / std_sinr
        
        # Diversity gain: higher for users with better-than-average channels
        # Bounded between 0.5 and 2.0 to prevent extreme allocations
        K = len(users)
        base_diversity_gain = 1.0 + 0.1 * np.log(K)  # log(K) scaling
        
        if self.diversity_mode == "opportunistic":
            # Full diversity exploitation
            diversity_gain = base_diversity_gain * (1.0 + 0.3 * z_score)
        elif self.diversity_mode == "proportional_fair":
            # Balanced diversity (default)
            diversity_gain = base_diversity_gain * (1.0 + 0.15 * z_score)
        else:  # round_robin
            diversity_gain = 1.0
        
        return np.clip(diversity_gain, 0.5, 2.0)
    
    def compute_pf_metric(
        self, 
        user: UserAllocationState,
        users: Optional[List[UserAllocationState]] = None
    ) -> float:
        """
        Compute Proportional Fair scheduling metric with multi-user diversity.
        
        PF metric = R_inst^alpha × DiversityGain / R_avg^beta
        
        where R_inst is estimated from SINR (Shannon capacity proxy)
        
        Args:
            user: Target user state
            users: All users (for diversity calculation)
        """
        # Instantaneous rate estimate (Shannon capacity)
        sinr_linear = 10 ** (user.sinr_db / 10)
        r_inst = np.log2(1 + sinr_linear)
        
        # Average throughput (from tracking)
        r_avg = max(0.1, user.avg_throughput_mbps)
        
        # Multi-user diversity gain
        if users is not None and self.enable_diversity:
            diversity_gain = self.compute_multiuser_diversity_gain(users, user.user_id)
        else:
            diversity_gain = 1.0
        
        # PF metric with diversity
        metric = (r_inst ** self.alpha) * diversity_gain / (r_avg ** self.beta)
        
        return metric
    
    def allocate(
        self,
        users: List[UserAllocationState],
        available_prb: int
    ) -> UserAllocationResult:
        """
        Allocate PRBs to eMBB users with Proportional Fair scheduling
        and multi-user diversity gain.
        
        Algorithm:
        1. Compute PF metric with diversity gain for each user
        2. Allocate minimum PRBs first
        3. Distribute remaining PRBs proportionally to PF metric
        
        Args:
            users: List of eMBB user states
            available_prb: Total PRBs available for eMBB slice
            
        Returns:
            UserAllocationResult with per-user allocations
        """
        if not users:
            return UserAllocationResult(
                allocations={},
                total_allocated=0,
                utilization=0.0,
                unserved_users=[]
            )
        
        allocations = {u.user_id: 0 for u in users}
        remaining_prb = available_prb
        
        # Compute PF metrics with diversity gain
        pf_metrics = {}
        for user in users:
            pf_metrics[user.user_id] = self.compute_pf_metric(user, users)
        
        # Phase 1: Minimum guarantee
        for user in users:
            if remaining_prb >= self.min_prb:
                allocations[user.user_id] = self.min_prb
                remaining_prb -= self.min_prb
            elif remaining_prb > 0:
                allocations[user.user_id] = remaining_prb
                remaining_prb = 0
            else:
                break
        
        # Phase 2: Proportional fair distribution of remaining PRBs
        if remaining_prb > 0:
            total_metric = sum(pf_metrics.values())
            if total_metric > 0:
                for user in users:
                    share = pf_metrics[user.user_id] / total_metric
                    extra = int(remaining_prb * share)
                    allocations[user.user_id] += extra
                
                # Handle rounding remainder
                allocated_extra = sum(allocations.values()) - len(users) * self.min_prb
                if allocated_extra < remaining_prb:
                    # Give remainder to highest PF metric user
                    best_user = max(users, key=lambda u: pf_metrics[u.user_id])
                    allocations[best_user.user_id] += remaining_prb - allocated_extra
        
        # Identify unserved users
        unserved = [u.user_id for u in users if allocations[u.user_id] == 0]
        total_allocated = sum(allocations.values())
        utilization = total_allocated / available_prb if available_prb > 0 else 0.0
        
        return UserAllocationResult(
            allocations=allocations,
            total_allocated=total_allocated,
            utilization=min(1.0, utilization),
            unserved_users=unserved
        )


@dataclass
class SchedulingResult:
    """Complete scheduling result for one time step."""
    # Slice-level
    slice_allocation: SliceAllocationResult
    
    # Per-user
    urllc_allocation: UserAllocationResult
    embb_allocation: UserAllocationResult
    
    # Aggregate metrics
    total_prb_used: int
    total_prb_utilization: float
    urllc_users_served: int
    embb_users_served: int


class TwoStageScheduler:
    """
    Complete two-stage PRB scheduler combining slice-level and in-slice allocation.
    """
    
    def __init__(
        self,
        resource_grid: NRResourceGrid,
        urllc_config: Optional[URLLCConfig] = None,
        embb_config: Optional[EMMBConfig] = None
    ):
        """
        Args:
            resource_grid: NR resource grid with PRB count
            urllc_config: URLLC QoS configuration
            embb_config: eMBB QoS configuration
        """
        self.resource_grid = resource_grid
        self.total_prb = resource_grid.n_rb
        
        # Default configs if not provided
        self.urllc_config = urllc_config or URLLCConfig()
        self.embb_config = embb_config or EMMBConfig()
        
        # Initialize allocators
        self.slice_allocator = SliceLevelAllocator(
            total_prb=self.total_prb,
            urllc_min_prb_per_user=2,
            embb_min_prb_per_user=1
        )
        
        self.urllc_scheduler = URLLCInSliceScheduler(
            min_prb_per_user=2,
            violation_priority_boost=2.0
        )
        
        # Initialize eMBB scheduler with multi-user diversity settings
        enable_diversity = getattr(self.embb_config, 'enable_multiuser_diversity', True)
        diversity_mode = getattr(self.embb_config, 'diversity_scheduling_mode', 'proportional_fair')
        pf_alpha = getattr(self.embb_config, 'pf_fairness_alpha', 1.0)
        
        self.embb_scheduler = EMMBInSliceScheduler(
            min_prb_per_user=1,
            pf_alpha=pf_alpha,
            pf_beta=1.0,
            enable_diversity=enable_diversity,
            diversity_mode=diversity_mode
        )
        
        # Throughput trackers for PF scheduling
        self.embb_throughput_trackers: Dict[int, UserThroughputTracker] = {}
    
    def update_embb_throughput(self, user_id: int, throughput_mbps: float):
        """Update average throughput for PF scheduling."""
        if user_id not in self.embb_throughput_trackers:
            self.embb_throughput_trackers[user_id] = UserThroughputTracker(
                window_size=24
            )
        self.embb_throughput_trackers[user_id].update(throughput_mbps)
    
    def get_embb_avg_throughput(self, user_id: int) -> float:
        """Get average throughput for a user."""
        if user_id in self.embb_throughput_trackers:
            return self.embb_throughput_trackers[user_id].get_average()
        return 1.0  # Default
    
    def remove_user(self, user_id: int):
        """Remove user from tracking (on churn)."""
        if user_id in self.embb_throughput_trackers:
            del self.embb_throughput_trackers[user_id]
    
    def schedule(
        self,
        urllc_users: List[UserAllocationState],
        embb_users: List[UserAllocationState],
        urllc_violation_rate: float = 0.0,
        embb_violation_rate: float = 0.0
    ) -> SchedulingResult:
        """
        Execute complete two-stage scheduling.
        
        Args:
            urllc_users: URLLC user states
            embb_users: eMBB user states
            urllc_violation_rate: Recent URLLC violation rate
            embb_violation_rate: Recent eMBB violation rate
            
        Returns:
            Complete scheduling result
        """
        # Update eMBB user average throughputs for PF
        for user in embb_users:
            user.avg_throughput_mbps = self.get_embb_avg_throughput(user.user_id)
        
        # Stage 1: Slice-level allocation
        slice_result = self.slice_allocator.allocate(
            n_urllc_users=len(urllc_users),
            n_embb_users=len(embb_users),
            urllc_violation_rate=urllc_violation_rate,
            embb_violation_rate=embb_violation_rate
        )
        
        # Stage 2a: URLLC in-slice scheduling
        urllc_result = self.urllc_scheduler.allocate(
            users=urllc_users,
            available_prb=slice_result.urllc_prb
        )
        
        # Stage 2b: eMBB in-slice scheduling
        embb_result = self.embb_scheduler.allocate(
            users=embb_users,
            available_prb=slice_result.embb_prb
        )
        
        # Compute aggregate metrics
        total_used = urllc_result.total_allocated + embb_result.total_allocated
        total_util = total_used / self.total_prb if self.total_prb > 0 else 0.0
        
        return SchedulingResult(
            slice_allocation=slice_result,
            urllc_allocation=urllc_result,
            embb_allocation=embb_result,
            total_prb_used=total_used,
            total_prb_utilization=total_util,
            urllc_users_served=len(urllc_users) - len(urllc_result.unserved_users),
            embb_users_served=len(embb_users) - len(embb_result.unserved_users)
        )


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Two-Stage PRB Scheduler Test")
    print("=" * 70)
    
    # Create resource grid (20 MHz @ 30 kHz SCS -> 51 PRB)
    from env.nr_prb_table import get_n_rb
    n_rb = get_n_rb(bandwidth_mhz=20, scs_khz=30)
    print(f"\nTotal PRBs: {n_rb} (20 MHz @ 30 kHz SCS)")
    
    resource_grid = NRResourceGrid(bandwidth_mhz=20, scs_khz=30)
    scheduler = TwoStageScheduler(resource_grid)
    
    # Create test users
    np.random.seed(42)
    
    # URLLC users (10 users)
    urllc_users = []
    for i in range(10):
        user = UserAllocationState(
            user_id=1000 + i,
            slice_type="URLLC",
            sinr_db=np.random.uniform(5, 25),
            distance_m=np.random.uniform(20, 150),
            consecutive_violations=np.random.choice([0, 0, 0, 1, 2]),
            qos_violations_24h=np.random.randint(0, 5)
        )
        urllc_users.append(user)
    
    # eMBB users (50 users)
    embb_users = []
    for i in range(50):
        user = UserAllocationState(
            user_id=2000 + i,
            slice_type="eMBB",
            sinr_db=np.random.uniform(5, 30),
            distance_m=np.random.uniform(20, 200),
            avg_throughput_mbps=np.random.uniform(10, 100)
        )
        embb_users.append(user)
    
    # Run scheduling
    print("\n--- Scheduling with 10 URLLC + 50 eMBB users ---")
    result = scheduler.schedule(
        urllc_users=urllc_users,
        embb_users=embb_users,
        urllc_violation_rate=0.05,
        embb_violation_rate=0.02
    )
    
    print(f"\nSlice-Level Allocation:")
    print(f"  URLLC PRBs: {result.slice_allocation.urllc_prb}")
    print(f"  eMBB PRBs: {result.slice_allocation.embb_prb}")
    print(f"  Reserved PRBs: {result.slice_allocation.reserved_prb}")
    
    print(f"\nURLLC Allocation:")
    print(f"  Users served: {result.urllc_users_served}/{len(urllc_users)}")
    print(f"  Total PRBs: {result.urllc_allocation.total_allocated}")
    print(f"  Utilization: {result.urllc_allocation.utilization:.2%}")
    
    print(f"\neMBB Allocation:")
    print(f"  Users served: {result.embb_users_served}/{len(embb_users)}")
    print(f"  Total PRBs: {result.embb_allocation.total_allocated}")
    print(f"  Utilization: {result.embb_allocation.utilization:.2%}")
    
    print(f"\nAggregate:")
    print(f"  Total PRBs used: {result.total_prb_used}/{n_rb}")
    print(f"  Overall utilization: {result.total_prb_utilization:.2%}")
    
    # Test high-load scenario
    print("\n--- High Load Scenario (25 URLLC + 100 eMBB) ---")
    
    urllc_high = [UserAllocationState(
        user_id=3000 + i, slice_type="URLLC",
        sinr_db=np.random.uniform(5, 25),
        distance_m=np.random.uniform(20, 150),
        consecutive_violations=np.random.choice([0, 1, 2, 3])
    ) for i in range(25)]
    
    embb_high = [UserAllocationState(
        user_id=4000 + i, slice_type="eMBB",
        sinr_db=np.random.uniform(5, 30),
        distance_m=np.random.uniform(20, 200),
        avg_throughput_mbps=np.random.uniform(10, 100)
    ) for i in range(100)]
    
    result_high = scheduler.schedule(
        urllc_users=urllc_high,
        embb_users=embb_high,
        urllc_violation_rate=0.15,  # Higher violation rate
        embb_violation_rate=0.10
    )
    
    print(f"\nSlice-Level Allocation:")
    print(f"  URLLC PRBs: {result_high.slice_allocation.urllc_prb}")
    print(f"  eMBB PRBs: {result_high.slice_allocation.embb_prb}")
    print(f"  Reserved PRBs: {result_high.slice_allocation.reserved_prb}")
    
    print(f"\nURLLC: {result_high.urllc_users_served}/{len(urllc_high)} served")
    print(f"eMBB: {result_high.embb_users_served}/{len(embb_high)} served")
    print(f"Overall utilization: {result_high.total_prb_utilization:.2%}")
    
    # URLLC allocation detail
    print("\n--- URLLC Priority Analysis ---")
    for user in sorted(urllc_high, key=lambda u: u.priority_score, reverse=True)[:5]:
        alloc = result_high.urllc_allocation.allocations.get(user.user_id, 0)
        print(f"  User {user.user_id}: priority={user.priority_score:.2f}, "
              f"violations={user.consecutive_violations}, "
              f"SINR={user.sinr_db:.1f} dB, PRBs={alloc}")
    
    print("\n" + "=" * 70)
    print("Scheduler test completed successfully!")
