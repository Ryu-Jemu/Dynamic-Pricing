"""
Two-Stage PRB Scheduler for 5G O-RAN Network Slicing

CRITICAL FIX (2026-01-28):
- Stage 1: Improved slice-level allocation with realistic capacity
- Stage 2: Enhanced per-user PRB distribution for QoS achievement
- Added dynamic PRB adjustment based on SINR

References:
- 3GPP TS 38.214: Physical layer procedures for data
- Wang et al. (2024): DRL-based URLLC/eMBB scheduling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

# Forward declarations for type hints
try:
    from config.scenario_config import URLLCConfig, EMMBConfig
    from env.qos_embb import UserThroughputTracker
except ImportError:
    URLLCConfig = None
    EMMBConfig = None
    UserThroughputTracker = None


@dataclass
class UserAllocationState:
    """Per-user allocation state for scheduling decisions."""
    user_id: int
    slice_type: str  # "URLLC" or "eMBB"
    sinr_db: float
    distance_m: float
    consecutive_violations: int = 0
    qos_violations_24h: int = 0
    avg_throughput_mbps: float = 1.0
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
    
    FIXED: Better balance between URLLC guarantee and eMBB fairness.
    """
    
    def __init__(
        self,
        total_prb: int,
        urllc_min_prb_per_user: int = 3,  # INCREASED from 2
        embb_min_prb_per_user: int = 2,    # INCREASED from 1
        reserved_prb_fraction: float = 0.02,  # REDUCED from 0.05
        urllc_priority_weight: float = 2.0  # INCREASED from 1.5
    ):
        """
        Args:
            total_prb: Total PRBs available
            urllc_min_prb_per_user: Minimum PRB guarantee per URLLC user
            embb_min_prb_per_user: Minimum PRB guarantee per eMBB user
            reserved_prb_fraction: Fraction of PRBs to reserve
            urllc_priority_weight: Priority weight for URLLC slice
        """
        self.total_prb = total_prb
        self.urllc_min_per_user = urllc_min_prb_per_user
        self.embb_min_per_user = embb_min_prb_per_user
        self.reserved_fraction = reserved_prb_fraction
        self.urllc_priority = urllc_priority_weight
        
        # Minimum reserved PRBs (reduced)
        self.min_reserved = max(1, int(total_prb * reserved_prb_fraction))
    
    def allocate(
        self,
        n_urllc_users: int,
        n_embb_users: int,
        urllc_violation_rate: float = 0.0,
        embb_violation_rate: float = 0.0
    ) -> SliceAllocationResult:
        """
        Allocate PRBs to slices based on user counts and QoS status.
        
        FIXED: More balanced allocation ensuring both slices can achieve QoS.
        """
        available_prb = self.total_prb - self.min_reserved
        
        # Step 1: URLLC base allocation (strict guarantee)
        urllc_base_need = n_urllc_users * self.urllc_min_per_user
        
        # Step 2: URLLC boost based on violation rate (aggressive)
        # If violations are high, significantly increase allocation
        urllc_boost_factor = 1.0 + min(urllc_violation_rate * 20, 1.0)  # Up to 2x
        urllc_need = int(urllc_base_need * urllc_boost_factor)
        
        # Step 3: eMBB base allocation
        embb_base_need = n_embb_users * self.embb_min_per_user
        embb_boost_factor = 1.0 + min(embb_violation_rate * 5, 0.5)  # Up to 1.5x
        embb_need = int(embb_base_need * embb_boost_factor)
        
        # Step 4: Allocation with URLLC priority
        total_need = urllc_need + embb_need
        
        if total_need <= available_prb:
            # Sufficient PRBs: allocate as needed
            urllc_prb = urllc_need
            embb_prb = embb_need
            excess = available_prb - total_need
            
            # Distribute excess primarily to eMBB (needs more for throughput)
            if n_urllc_users > 0 and n_embb_users > 0:
                # eMBB gets 70% of excess, URLLC gets 30%
                embb_extra = int(excess * 0.7)
                urllc_extra = excess - embb_extra
                urllc_prb += urllc_extra
                embb_prb += embb_extra
            elif n_embb_users > 0:
                embb_prb += excess
            else:
                urllc_prb += excess
        else:
            # Insufficient PRBs: URLLC has strict priority
            # But still ensure eMBB gets minimum viable allocation
            urllc_prb = min(urllc_need, int(available_prb * 0.5))  # Cap at 50%
            embb_prb = available_prb - urllc_prb
            
            # If URLLC absolutely needs more, take from eMBB
            if urllc_violation_rate > 0.01:  # High URLLC violations
                urllc_prb = min(urllc_need, int(available_prb * 0.7))
                embb_prb = available_prb - urllc_prb
        
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
    
    Priority-based with consecutive-violation protection.
    """
    
    def __init__(
        self,
        min_prb_per_user: int = 3,  # INCREASED from 2
        violation_priority_boost: float = 3.0  # INCREASED from 2.0
    ):
        self.min_prb = min_prb_per_user
        self.violation_boost = violation_priority_boost
    
    def compute_priority(self, user: UserAllocationState) -> float:
        """
        Compute scheduling priority for URLLC user.
        
        Higher priority for:
        - Users with consecutive violations (urgent)
        - Users with worse channel (need more resources)
        """
        # Base priority from SINR (inverted - worse channel = higher priority)
        sinr_priority = max(0, (30 - user.sinr_db) / 30)
        
        # Violation boost (exponential for consecutive violations)
        violation_priority = self.violation_boost ** min(user.consecutive_violations, 3)
        
        # 24h violation history boost
        history_boost = 1.0 + 0.2 * min(user.qos_violations_24h, 5)
        
        return sinr_priority * violation_priority * history_boost
    
    def allocate(
        self,
        users: List[UserAllocationState],
        available_prb: int
    ) -> UserAllocationResult:
        """
        Allocate PRBs to URLLC users with priority-based scheduling.
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
        
        # Phase 2: Extra allocation for users with consecutive violations
        if remaining_prb > 0:
            for user in sorted_users:
                if remaining_prb <= 0:
                    break
                if user.consecutive_violations > 0:
                    # Give 1 extra PRB per consecutive violation (up to 2)
                    extra = min(user.consecutive_violations, 2, remaining_prb)
                    allocations[user.user_id] += extra
                    remaining_prb -= extra
        
        # Phase 3: Distribute remaining evenly
        if remaining_prb > 0 and len(users) > 0:
            extra_per_user = remaining_prb // len(users)
            if extra_per_user > 0:
                for user in users:
                    allocations[user.user_id] += extra_per_user
                remaining_prb -= extra_per_user * len(users)
            
            # Give remainder to highest priority user
            if remaining_prb > 0:
                allocations[sorted_users[0].user_id] += remaining_prb
        
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
    
    FIXED: Better PRB distribution considering SINR-dependent throughput.
    """
    
    def __init__(
        self,
        min_prb_per_user: int = 2,  # INCREASED from 1
        pf_alpha: float = 1.0,
        pf_beta: float = 1.0,
        enable_diversity: bool = True,
        diversity_mode: str = "proportional_fair"
    ):
        self.min_prb = min_prb_per_user
        self.alpha = pf_alpha
        self.beta = pf_beta
        self.enable_diversity = enable_diversity
        self.diversity_mode = diversity_mode
    
    def compute_sinr_weight(self, sinr_db: float) -> float:
        """
        Compute SINR-based weight for PRB allocation.
        
        Users with lower SINR need more PRBs to achieve same throughput.
        """
        # Reference SINR (15 dB is moderate)
        ref_sinr = 15.0
        
        # Weight inversely proportional to SINR (normalized)
        # Low SINR users need more resources
        weight = max(0.5, ref_sinr / max(sinr_db, 1.0))
        return min(2.0, weight)  # Cap at 2x
    
    def compute_pf_metric(
        self,
        user: UserAllocationState,
        users: Optional[List[UserAllocationState]] = None
    ) -> float:
        """
        Compute Proportional Fair metric with SINR consideration.
        
        FIXED: Better balance between fairness and efficiency.
        """
        # Instantaneous rate estimate (Shannon capacity)
        sinr_linear = 10 ** (user.sinr_db / 10)
        r_inst = np.log2(1 + sinr_linear)
        
        # Average throughput (from tracking)
        r_avg = max(0.1, user.avg_throughput_mbps)
        
        # Multi-user diversity gain
        diversity_gain = 1.0
        if users is not None and self.enable_diversity and len(users) > 1:
            # Simple diversity gain based on channel variation
            sinrs = [u.sinr_db for u in users]
            diversity_gain = 1.0 + 0.1 * np.std(sinrs) / max(1, np.mean(sinrs))
        
        # PF metric with diversity
        metric = (r_inst ** self.alpha) * diversity_gain / (r_avg ** self.beta)
        
        return metric
    
    def allocate(
        self,
        users: List[UserAllocationState],
        available_prb: int
    ) -> UserAllocationResult:
        """
        Allocate PRBs to eMBB users with Proportional Fair scheduling.
        
        CRITICAL FIX: Ensure ALL users get minimum 1 PRB before any extras.
        This prevents the 0 PRB starvation issue that caused ~50% violations.
        """
        if not users:
            return UserAllocationResult(
                allocations={},
                total_allocated=0,
                utilization=0.0,
                unserved_users=[]
            )
        
        n_users = len(users)
        allocations = {u.user_id: 0 for u in users}
        remaining_prb = available_prb
        
        # Phase 0: GUARANTEE minimum 1 PRB to ALL users first
        # This is the critical fix to prevent 0 PRB starvation
        min_per_user = max(1, min(remaining_prb // n_users, self.min_prb))
        for user in users:
            allocations[user.user_id] = min_per_user
        remaining_prb -= min_per_user * n_users
        
        # Early exit if no remaining PRBs
        if remaining_prb <= 0:
            unserved = [u.user_id for u in users if allocations[u.user_id] == 0]
            total_allocated = sum(allocations.values())
            return UserAllocationResult(
                allocations=allocations,
                total_allocated=total_allocated,
                utilization=total_allocated / available_prb if available_prb > 0 else 0.0,
                unserved_users=unserved
            )
        
        # Compute PF metrics and SINR weights
        pf_metrics = {}
        sinr_weights = {}
        for user in users:
            pf_metrics[user.user_id] = self.compute_pf_metric(user, users)
            sinr_weights[user.user_id] = self.compute_sinr_weight(user.sinr_db)
        
        # Phase 1: Additional PRBs for low-SINR users (need more for same throughput)
        # Give extra to users with SINR below reference
        ref_sinr = 15.0
        low_sinr_users = [u for u in users if u.sinr_db < ref_sinr]
        if low_sinr_users and remaining_prb > 0:
            extra_for_low_sinr = min(remaining_prb, len(low_sinr_users))
            for user in sorted(low_sinr_users, key=lambda u: u.sinr_db):
                if remaining_prb <= 0:
                    break
                allocations[user.user_id] += 1
                remaining_prb -= 1
        
        # Phase 2: Proportional fair distribution of remaining PRBs
        if remaining_prb > 0:
            total_metric = sum(pf_metrics.values())
            if total_metric > 0:
                # Calculate shares first without modifying
                shares = {}
                for user in users:
                    shares[user.user_id] = int(remaining_prb * pf_metrics[user.user_id] / total_metric)
                
                # Apply shares
                for user in users:
                    allocations[user.user_id] += shares[user.user_id]
                
                # Handle rounding remainder
                allocated_so_far = sum(allocations.values())
                remaining_after = available_prb - allocated_so_far
                if remaining_after > 0:
                    # Distribute to users with highest PF metric
                    sorted_by_metric = sorted(users, key=lambda u: pf_metrics[u.user_id], reverse=True)
                    for i, user in enumerate(sorted_by_metric):
                        if i >= remaining_after:
                            break
                        allocations[user.user_id] += 1
        
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


class NRResourceGrid:
    """Simple NR resource grid for PRB calculations."""
    
    def __init__(self, bandwidth_mhz: int = 20, scs_khz: int = 30):
        self.bandwidth_mhz = bandwidth_mhz
        self.scs_khz = scs_khz
        self.n_rb = self._calculate_n_rb()
    
    def _calculate_n_rb(self) -> int:
        """Calculate number of RBs from 3GPP tables."""
        # Simplified: 20 MHz @ 30 kHz = 51 RBs
        prb_table = {
            (20, 30): 51,
            (50, 30): 133,
            (100, 30): 273,
        }
        return prb_table.get((self.bandwidth_mhz, self.scs_khz), 51)


class TwoStageScheduler:
    """
    Complete two-stage PRB scheduler combining slice-level and in-slice allocation.
    
    FIXED: Better coordination between stages for QoS achievement.
    """
    
    def __init__(
        self,
        resource_grid: NRResourceGrid,
        urllc_config: Optional[object] = None,
        embb_config: Optional[object] = None
    ):
        self.resource_grid = resource_grid
        self.total_prb = resource_grid.n_rb
        
        # Initialize allocators with FIXED parameters
        self.slice_allocator = SliceLevelAllocator(
            total_prb=self.total_prb,
            urllc_min_prb_per_user=3,  # INCREASED
            embb_min_prb_per_user=2    # INCREASED
        )
        
        self.urllc_scheduler = URLLCInSliceScheduler(
            min_prb_per_user=3,  # INCREASED
            violation_priority_boost=3.0  # INCREASED
        )
        
        self.embb_scheduler = EMMBInSliceScheduler(
            min_prb_per_user=2,  # INCREASED
            pf_alpha=1.0,
            pf_beta=1.0,
            enable_diversity=True,
            diversity_mode="proportional_fair"
        )
        
        # Throughput trackers for PF scheduling
        self.embb_throughput_trackers: Dict[int, object] = {}
    
    def update_embb_throughput(self, user_id: int, throughput_mbps: float):
        """Update average throughput for PF scheduling."""
        if user_id not in self.embb_throughput_trackers:
            # Create simple tracker
            self.embb_throughput_trackers[user_id] = {"history": [], "window": 24}
        
        tracker = self.embb_throughput_trackers[user_id]
        tracker["history"].append(throughput_mbps)
        if len(tracker["history"]) > tracker["window"]:
            tracker["history"].pop(0)
    
    def get_embb_avg_throughput(self, user_id: int) -> float:
        """Get average throughput for a user."""
        if user_id in self.embb_throughput_trackers:
            history = self.embb_throughput_trackers[user_id]["history"]
            if history:
                return np.mean(history)
        return 1.0
    
    def remove_user(self, user_id: int):
        """Remove user from tracking."""
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
        """
        n_urllc = len(urllc_users)
        n_embb = len(embb_users)
        
        # Update eMBB users with tracked throughput
        for user in embb_users:
            user.avg_throughput_mbps = self.get_embb_avg_throughput(user.user_id)
        
        # Stage 1: Slice-level allocation
        slice_result = self.slice_allocator.allocate(
            n_urllc_users=n_urllc,
            n_embb_users=n_embb,
            urllc_violation_rate=urllc_violation_rate,
            embb_violation_rate=embb_violation_rate
        )
        
        # Stage 2a: URLLC in-slice
        urllc_result = self.urllc_scheduler.allocate(
            users=urllc_users,
            available_prb=slice_result.urllc_prb
        )
        
        # Stage 2b: eMBB in-slice
        embb_result = self.embb_scheduler.allocate(
            users=embb_users,
            available_prb=slice_result.embb_prb
        )
        
        # Update user allocation states
        for user in urllc_users:
            user.allocated_prb = urllc_result.allocations.get(user.user_id, 0)
        
        for user in embb_users:
            user.allocated_prb = embb_result.allocations.get(user.user_id, 0)
        
        # Compute aggregate metrics
        total_used = urllc_result.total_allocated + embb_result.total_allocated
        total_util = total_used / self.total_prb if self.total_prb > 0 else 0.0
        
        return SchedulingResult(
            slice_allocation=slice_result,
            urllc_allocation=urllc_result,
            embb_allocation=embb_result,
            total_prb_used=total_used,
            total_prb_utilization=total_util,
            urllc_users_served=n_urllc - len(urllc_result.unserved_users),
            embb_users_served=n_embb - len(embb_result.unserved_users)
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Two-Stage PRB Scheduler Test (FIXED)")
    print("=" * 70)
    
    # Create resource grid
    resource_grid = NRResourceGrid(bandwidth_mhz=20, scs_khz=30)
    print(f"\nTotal PRBs: {resource_grid.n_rb}")
    
    scheduler = TwoStageScheduler(resource_grid)
    
    # Test with FIXED user counts
    np.random.seed(42)
    
    # URLLC users (5 users - REDUCED)
    urllc_users = [
        UserAllocationState(
            user_id=1000 + i,
            slice_type="URLLC",
            sinr_db=np.random.uniform(10, 25),
            distance_m=np.random.uniform(20, 150),
            consecutive_violations=np.random.choice([0, 0, 0, 1]),
        )
        for i in range(5)
    ]
    
    # eMBB users (20 users - REDUCED)
    embb_users = [
        UserAllocationState(
            user_id=2000 + i,
            slice_type="eMBB",
            sinr_db=np.random.uniform(8, 25),
            distance_m=np.random.uniform(20, 200),
            avg_throughput_mbps=np.random.uniform(2, 8)
        )
        for i in range(20)
    ]
    
    # Run scheduling
    print(f"\n--- Scheduling with {len(urllc_users)} URLLC + {len(embb_users)} eMBB users ---")
    result = scheduler.schedule(
        urllc_users=urllc_users,
        embb_users=embb_users,
        urllc_violation_rate=0.01,
        embb_violation_rate=0.05
    )
    
    print(f"\nSlice-Level Allocation:")
    print(f"  URLLC PRBs: {result.slice_allocation.urllc_prb}")
    print(f"  eMBB PRBs: {result.slice_allocation.embb_prb}")
    print(f"  Reserved PRBs: {result.slice_allocation.reserved_prb}")
    
    print(f"\nURLLC Allocation:")
    print(f"  Users served: {result.urllc_users_served}/{len(urllc_users)}")
    print(f"  PRBs per user: {result.slice_allocation.urllc_prb / max(1, len(urllc_users)):.1f}")
    
    print(f"\neMBB Allocation:")
    print(f"  Users served: {result.embb_users_served}/{len(embb_users)}")
    print(f"  PRBs per user: {result.slice_allocation.embb_prb / max(1, len(embb_users)):.1f}")
    
    print(f"\nOverall utilization: {result.total_prb_utilization:.2%}")
    
    # Verify eMBB can achieve QoS
    print("\n--- eMBB QoS Verification ---")
    from env.qos_embb import EMMBQoSModel
    qos_model = EMMBQoSModel(throughput_requirement_mbps=5.0)
    
    satisfied_count = 0
    for user in embb_users:
        alloc = result.embb_allocation.allocations.get(user.user_id, 0)
        sat, _, details = qos_model.evaluate_qos(user.sinr_db, alloc)
        if sat:
            satisfied_count += 1
    
    print(f"  eMBB users satisfied: {satisfied_count}/{len(embb_users)}")
    print(f"  Expected violation rate: {100*(1-satisfied_count/len(embb_users)):.1f}%")