"""
Economic models package for 5G O-RAN Network Slicing RL Simulation.

Contains:
- Three-part tariff pricing model
- NHPP arrival model with price/QoS effects
- Hawkes self-exciting arrival model (optional)
- Hazard-based churn model with overage burden
- Decomposed cost model (energy, spectrum, backhaul)
"""

from .tariff_three_part import (
    SliceType,
    TariffPlan,
    UserBillingState,
    DynamicTariffManager,
)

from .arrivals_nhpp import (
    TimeProfile,
    PriceEffect,
    QoSEffect,
    ArrivalModelConfig,
    NHPPArrivalModel,
    SliceArrivalManager,
)

from .arrivals_hawkes import (
    HawkesParameters,
    HawkesArrivalModel,
    HawkesCalibrator,
)

from .churn_hazard import (
    ChurnCoefficients,
    UserChurnState,
    ChurnModel,
    SliceChurnConfig,
    ChurnManager,
)

from .costs import (
    EnergyModelConfig,
    SpectrumCostConfig,
    BackhaulCostConfig,
    FixedCostConfig,
    AcquisitionCostConfig,
    EnergyModel,
    BackhaulModel,
    CostManager,
)

__all__ = [
    # Tariff
    "SliceType",
    "TariffPlan",
    "UserBillingState",
    "DynamicTariffManager",
    # Arrivals (NHPP)
    "TimeProfile",
    "PriceEffect",
    "QoSEffect",
    "ArrivalModelConfig",
    "NHPPArrivalModel",
    "SliceArrivalManager",
    # Arrivals (Hawkes)
    "HawkesParameters",
    "HawkesArrivalModel",
    "HawkesCalibrator",
    # Churn
    "ChurnCoefficients",
    "UserChurnState",
    "ChurnModel",
    "SliceChurnConfig",
    "ChurnManager",
    # Costs
    "EnergyModelConfig",
    "SpectrumCostConfig",
    "BackhaulCostConfig",
    "FixedCostConfig",
    "AcquisitionCostConfig",
    "EnergyModel",
    "BackhaulModel",
    "CostManager",
]
