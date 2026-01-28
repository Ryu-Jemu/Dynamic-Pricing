"""
Environment package for 5G O-RAN Network Slicing RL Simulation.

Contains:
- 3GPP-compliant channel models (TR 38.901)
- NR resource grid utilities (TS 38.101)
- QoS models for URLLC (FBL) and eMBB
- Two-stage PRB scheduler
- CMDP Gymnasium environment
"""

from .nr_prb_table import (
    NRResourceGrid,
    get_n_rb,
    get_prb_bandwidth_hz,
    get_slot_duration_ms,
    get_slots_per_second,
)

from .channel_38901 import (
    ChannelModel3GPP38901,
    UserChannel,
)

from .qos_fbl import (
    URLLCQoSModel,
    FBLCapacity,
    URLLCViolationCalculator,
)

from .qos_embb import (
    EMMBQoSModel,
    SpectralEfficiencyCalculator,
    EMMBViolationCalculator,
    UserThroughputTracker,
)

from .scheduler import (
    TwoStageScheduler,
    UserAllocationState,
    SchedulingResult,
    SliceAllocationResult,
)

from .network_slicing_cmdp_env import (
    NetworkSlicingCMDPEnv,
    User,
    StepInfo,
)

__all__ = [
    # NR PRB Table
    "NRResourceGrid",
    "get_n_rb",
    "get_prb_bandwidth_hz",
    "get_slot_duration_ms",
    "get_slots_per_second",
    # Channel Model
    "ChannelModel3GPP38901",
    "UserChannel",
    # URLLC QoS
    "URLLCQoSModel",
    "FBLCapacity",
    "URLLCViolationCalculator",
    # eMBB QoS
    "EMMBQoSModel",
    "SpectralEfficiencyCalculator",
    "EMMBViolationCalculator",
    "UserThroughputTracker",
    # Scheduler
    "TwoStageScheduler",
    "UserAllocationState",
    "SchedulingResult",
    "SliceAllocationResult",
    # Environment
    "NetworkSlicingCMDPEnv",
    "User",
    "StepInfo",
]
