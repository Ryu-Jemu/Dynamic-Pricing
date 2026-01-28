"""
Training package for 5G O-RAN Network Slicing RL Simulation.

Contains:
- Primal-dual Lagrangian SAC trainer for CMDP
- Constraint handling with learned multipliers
- Baseline implementations (fixed pricing, penalty SAC)
"""

from .train_cmdp_sac import (
    CMDPSACTrainer,
    PrimalDualConfig,
    TrainingResult,
    train_cmdp_sac,
    train_penalty_sac,
    train_fixed_pricing,
)

__all__ = [
    "CMDPSACTrainer",
    "PrimalDualConfig",
    "TrainingResult",
    "train_cmdp_sac",
    "train_penalty_sac",
    "train_fixed_pricing",
]
