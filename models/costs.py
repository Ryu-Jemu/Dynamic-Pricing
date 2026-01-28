"""
Operational Cost Model

Implements cost calculation for 5G network operation.

C_total = C_energy + C_spectrum + C_backhaul + C_fixed + C_acquisition

Components are decomposed into measurable factors with documented sources.

References:
- MDPI ECO6G 2022: 5G energy consumption modeling
- PMC 2024: OPEX breakdown (25% of total, 90% is energy)
- PMC 2021: 5G BS power consumption (4kW+ for macro)
- GSMA 2023: Network economics and TCO analysis
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnergyModelConfig:
    """
    Energy consumption model configuration.
    
    Uses load-dependent power model:
    P(t) = P_idle + (P_max - P_idle) × load^k
    
    Reference: PMC 2021 - 5G BS power measurements
    """
    # Power parameters (kW)
    power_idle_kw: float = 1.0      # Idle power (BS on, no traffic)
    power_max_kw: float = 4.0       # Maximum power at full load
    power_exponent: float = 1.5     # Non-linear load scaling
    
    # Electricity pricing ($/kWh) - scenario input
    electricity_price: float = 0.10


@dataclass
class SpectrumCostConfig:
    """
    Spectrum license and infrastructure cost configuration.
    
    Converts annual/capex costs to hourly opex.
    
    Reference: GSMA 2023 - 5G TCO analysis
    """
    # Annual costs ($)
    annual_spectrum_license: float = 50000.0   # Spectrum license fee
    annual_infrastructure: float = 30000.0     # Maintenance, upgrades
    
    # Derived hourly costs
    @property
    def hourly_spectrum_cost(self) -> float:
        return self.annual_spectrum_license / (365 * 24)
    
    @property
    def hourly_infrastructure_cost(self) -> float:
        return self.annual_infrastructure / (365 * 24)


@dataclass
class BackhaulCostConfig:
    """
    Backhaul (data transport) cost configuration.
    
    Reference: GSMA 2023 - Backhaul cost benchmarks
    """
    # Cost per data unit ($/GB)
    cost_per_gb: float = 0.01  # Scenario input


@dataclass
class FixedCostConfig:
    """
    Fixed operational costs (non-traffic dependent).
    
    Includes: personnel, site rental, administration
    """
    hourly_fixed_cost: float = 10.0  # $/hour


@dataclass
class AcquisitionCostConfig:
    """
    Customer acquisition cost configuration.
    
    Reference: Industry benchmarks for CAC
    """
    cost_per_customer: float = 50.0  # $/new user
    enabled: bool = True


class EnergyModel:
    """
    Calculate energy costs based on network load.
    
    Power model: P(t) = P_idle + (P_max - P_idle) × load^k
    """
    
    def __init__(self, config: EnergyModelConfig):
        self.config = config
    
    def calculate_power_kw(self, load_fraction: float) -> float:
        """
        Calculate instantaneous power consumption.
        
        Args:
            load_fraction: Network load as fraction [0, 1]
        
        Returns:
            Power consumption in kW
        """
        load = np.clip(load_fraction, 0.0, 1.0)
        
        power = (
            self.config.power_idle_kw + 
            (self.config.power_max_kw - self.config.power_idle_kw) * 
            (load ** self.config.power_exponent)
        )
        
        return power
    
    def calculate_hourly_cost(self, load_fraction: float) -> float:
        """
        Calculate hourly energy cost.
        
        Args:
            load_fraction: Average load over the hour
        
        Returns:
            Energy cost in $
        """
        power_kw = self.calculate_power_kw(load_fraction)
        energy_kwh = power_kw * 1  # 1 hour
        cost = energy_kwh * self.config.electricity_price
        
        return cost
    
    def get_breakdown(self, load_fraction: float) -> Dict[str, float]:
        """Get detailed energy cost breakdown."""
        power = self.calculate_power_kw(load_fraction)
        return {
            "load_fraction": load_fraction,
            "power_kw": power,
            "energy_kwh": power,
            "cost_per_kwh": self.config.electricity_price,
            "hourly_cost": power * self.config.electricity_price
        }


class BackhaulModel:
    """
    Calculate backhaul (data transport) costs.
    """
    
    def __init__(self, config: BackhaulCostConfig):
        self.config = config
    
    def calculate_hourly_cost(self, data_gb: float) -> float:
        """
        Calculate backhaul cost for data volume.
        
        Args:
            data_gb: Data volume in GB
        
        Returns:
            Backhaul cost in $
        """
        return data_gb * self.config.cost_per_gb
    
    def convert_mb_to_gb(self, data_mb: float) -> float:
        """Convert MB to GB."""
        return data_mb / 1024


class CostManager:
    """
    Manages all cost calculations for network operation.
    """
    
    def __init__(
        self,
        energy_config: Optional[EnergyModelConfig] = None,
        spectrum_config: Optional[SpectrumCostConfig] = None,
        backhaul_config: Optional[BackhaulCostConfig] = None,
        fixed_config: Optional[FixedCostConfig] = None,
        acquisition_config: Optional[AcquisitionCostConfig] = None
    ):
        """
        Initialize cost manager with all components.
        """
        self.energy_config = energy_config or EnergyModelConfig()
        self.spectrum_config = spectrum_config or SpectrumCostConfig()
        self.backhaul_config = backhaul_config or BackhaulCostConfig()
        self.fixed_config = fixed_config or FixedCostConfig()
        self.acquisition_config = acquisition_config or AcquisitionCostConfig()
        
        # Component models
        self.energy_model = EnergyModel(self.energy_config)
        self.backhaul_model = BackhaulModel(self.backhaul_config)
        
        # Cost history for analysis
        self.cost_history = []
    
    def calculate_hourly_cost(
        self,
        prb_utilization: float,
        total_data_mb: float,
        new_users: int = 0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total hourly operational cost.
        
        Args:
            prb_utilization: PRB utilization ratio [0, 1]
            total_data_mb: Total data transmitted in MB
            new_users: Number of newly acquired users
        
        Returns:
            Tuple of (total_cost, breakdown_dict)
        """
        # Energy cost (load-dependent)
        energy_cost = self.energy_model.calculate_hourly_cost(prb_utilization)
        
        # Spectrum/Infrastructure cost (fixed hourly)
        spectrum_cost = self.spectrum_config.hourly_spectrum_cost
        infrastructure_cost = self.spectrum_config.hourly_infrastructure_cost
        
        # Backhaul cost (data-dependent)
        data_gb = self.backhaul_model.convert_mb_to_gb(total_data_mb)
        backhaul_cost = self.backhaul_model.calculate_hourly_cost(data_gb)
        
        # Fixed operational cost
        fixed_cost = self.fixed_config.hourly_fixed_cost
        
        # Customer acquisition cost (optional)
        if self.acquisition_config.enabled:
            acquisition_cost = new_users * self.acquisition_config.cost_per_customer
        else:
            acquisition_cost = 0.0
        
        # Total
        total_cost = (
            energy_cost + 
            spectrum_cost + 
            infrastructure_cost + 
            backhaul_cost + 
            fixed_cost + 
            acquisition_cost
        )
        
        breakdown = {
            "energy": energy_cost,
            "spectrum": spectrum_cost,
            "infrastructure": infrastructure_cost,
            "backhaul": backhaul_cost,
            "fixed": fixed_cost,
            "acquisition": acquisition_cost,
            "total": total_cost
        }
        
        # Record history
        self.cost_history.append(breakdown)
        
        return total_cost, breakdown
    
    def calculate_marginal_costs(self) -> Dict[str, float]:
        """
        Calculate marginal costs for decision making.
        
        Marginal cost per unit helps inform pricing decisions.
        """
        return {
            "marginal_cost_per_prb_percent": (
                (self.energy_config.power_max_kw - self.energy_config.power_idle_kw) *
                self.energy_config.electricity_price / 100
            ),
            "marginal_cost_per_gb": self.backhaul_config.cost_per_gb,
            "marginal_cost_per_user": (
                self.acquisition_config.cost_per_customer 
                if self.acquisition_config.enabled else 0.0
            )
        }
    
    def get_cost_statistics(self) -> Dict[str, float]:
        """Get cost statistics from history."""
        if not self.cost_history:
            return {}
        
        totals = [h["total"] for h in self.cost_history]
        
        return {
            "total_accumulated_cost": sum(totals),
            "mean_hourly_cost": np.mean(totals),
            "std_hourly_cost": np.std(totals),
            "min_hourly_cost": min(totals),
            "max_hourly_cost": max(totals),
            "mean_energy_fraction": np.mean([
                h["energy"] / max(h["total"], 1e-6) 
                for h in self.cost_history
            ])
        }
    
    def get_cost_breakdown_summary(self) -> Dict[str, float]:
        """Get average cost breakdown by component."""
        if not self.cost_history:
            return {}
        
        components = ["energy", "spectrum", "infrastructure", 
                     "backhaul", "fixed", "acquisition"]
        
        return {
            component: np.mean([h[component] for h in self.cost_history])
            for component in components
        }


def estimate_expected_hourly_cost(
    expected_prb_util: float = 0.5,
    expected_data_mb: float = 500.0,
    expected_new_users: float = 1.0,
    cost_manager: Optional[CostManager] = None
) -> Dict[str, float]:
    """
    Estimate expected hourly costs for planning.
    
    Args:
        expected_prb_util: Expected average PRB utilization
        expected_data_mb: Expected hourly data volume (MB)
        expected_new_users: Expected new users per hour
        cost_manager: Cost manager (uses default if None)
    
    Returns:
        Expected cost breakdown
    """
    if cost_manager is None:
        cost_manager = CostManager()
    
    _, breakdown = cost_manager.calculate_hourly_cost(
        expected_prb_util, expected_data_mb, int(expected_new_users)
    )
    
    return breakdown


if __name__ == "__main__":
    print("=" * 60)
    print("Cost Model Test")
    print("=" * 60)
    
    # Create cost manager with default settings
    manager = CostManager()
    
    print("\nCost Configuration:")
    print(f"  Energy - Idle: {manager.energy_config.power_idle_kw}kW, "
          f"Max: {manager.energy_config.power_max_kw}kW")
    print(f"  Electricity: ${manager.energy_config.electricity_price}/kWh")
    print(f"  Spectrum: ${manager.spectrum_config.hourly_spectrum_cost:.4f}/hr")
    print(f"  Infrastructure: ${manager.spectrum_config.hourly_infrastructure_cost:.4f}/hr")
    print(f"  Backhaul: ${manager.backhaul_config.cost_per_gb}/GB")
    print(f"  Fixed: ${manager.fixed_config.hourly_fixed_cost}/hr")
    print(f"  CAC: ${manager.acquisition_config.cost_per_customer}/user")
    
    # Test cost calculation at different loads
    print("\nCost vs PRB Utilization (500 MB, 1 new user):")
    for load in [0.0, 0.25, 0.5, 0.75, 1.0]:
        total, breakdown = manager.calculate_hourly_cost(
            prb_utilization=load,
            total_data_mb=500,
            new_users=1
        )
        print(f"  {load*100:3.0f}% load: Total=${total:.2f} "
              f"(Energy=${breakdown['energy']:.3f})")
    
    # Cost vs data volume
    print("\nCost vs Data Volume (50% load, 1 new user):")
    for data_mb in [100, 500, 1000, 5000, 10000]:
        total, breakdown = manager.calculate_hourly_cost(
            prb_utilization=0.5,
            total_data_mb=data_mb,
            new_users=1
        )
        print(f"  {data_mb:5.0f} MB: Total=${total:.2f} "
              f"(Backhaul=${breakdown['backhaul']:.3f})")
    
    # Marginal costs
    print("\nMarginal Costs:")
    marginal = manager.calculate_marginal_costs()
    for key, value in marginal.items():
        print(f"  {key}: ${value:.4f}")
    
    # Expected hourly cost
    print("\nExpected Hourly Cost (baseline scenario):")
    expected = estimate_expected_hourly_cost(
        expected_prb_util=0.5,
        expected_data_mb=500,
        expected_new_users=1
    )
    for component, value in expected.items():
        print(f"  {component}: ${value:.4f}")
    
    # Simulate a week
    print("\n" + "=" * 60)
    print("Week Simulation")
    print("=" * 60)
    
    manager = CostManager()  # Fresh manager
    np.random.seed(42)
    
    for hour in range(168):
        # Varying load pattern
        load = 0.3 + 0.4 * np.sin(2 * np.pi * hour / 24) + np.random.uniform(0, 0.2)
        load = np.clip(load, 0.1, 0.9)
        
        # Data volume varies with load
        data_mb = 200 + 600 * load + np.random.normal(0, 50)
        data_mb = max(50, data_mb)
        
        # New users (Poisson)
        new_users = np.random.poisson(1)
        
        manager.calculate_hourly_cost(load, data_mb, new_users)
    
    print("\nWeekly Statistics:")
    stats = manager.get_cost_statistics()
    for key, value in stats.items():
        if "fraction" in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: ${value:.2f}")
    
    print("\nAverage Breakdown:")
    breakdown = manager.get_cost_breakdown_summary()
    for component, value in breakdown.items():
        print(f"  {component}: ${value:.4f}/hr")
