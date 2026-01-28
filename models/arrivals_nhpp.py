"""
User Arrival Model - Non-Homogeneous Poisson Process (NHPP)

Implements time-varying arrival rates with price and QoS effects.

λ(t) = λ₀ × T(t) × P(F, p) × Q(V)

where:
- λ₀: Base arrival rate
- T(t): Time profile (periodic patterns)
- P(F, p): Price effect
- Q(V): QoS effect

References:
- ACM IMC 2023: Mobile network traffic analysis, Poisson validation
- Nature Comm. Eng. 2023: 5G network traffic prediction
- ScienceDirect 2024: Price elasticity in telecommunications
- FERDI 2021: Mobile data demand elasticity
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class TimeProfile:
    """
    Time-varying intensity profile for arrivals.
    
    Implements periodic patterns based on hour-of-day and day-of-week.
    """
    
    def __init__(
        self,
        peak_hours: Tuple[int, int] = (9, 18),
        peak_multiplier: float = 1.5,
        weekend_factor: float = 0.7,
        use_sinusoidal: bool = False
    ):
        """
        Initialize time profile.
        
        Args:
            peak_hours: (start, end) of peak hours (24h format)
            peak_multiplier: Multiplier during peak hours
            weekend_factor: Factor for weekend (day 5, 6)
            use_sinusoidal: Use smooth sinusoidal profile instead of step
        """
        self.peak_start, self.peak_end = peak_hours
        self.peak_multiplier = peak_multiplier
        self.weekend_factor = weekend_factor
        self.use_sinusoidal = use_sinusoidal
    
    def get_hourly_factor(self, hour: int) -> float:
        """
        Get time profile factor for given hour.
        
        Args:
            hour: Simulation hour (0-indexed)
        
        Returns:
            Time profile multiplier
        """
        hour_of_day = hour % 24
        day_of_week = (hour // 24) % 7  # 0-6, assume Monday start
        
        # Day factor
        if day_of_week >= 5:  # Weekend
            day_factor = self.weekend_factor
        else:
            day_factor = 1.0
        
        # Hour factor
        if self.use_sinusoidal:
            # Smooth sinusoidal profile
            # Peak at mid_peak_hour, trough at night
            mid_peak = (self.peak_start + self.peak_end) / 2
            hour_factor = 1.0 + (self.peak_multiplier - 1.0) * \
                          (1 + np.cos(2 * np.pi * (hour_of_day - mid_peak) / 24)) / 2
        else:
            # Step function
            if self.peak_start <= hour_of_day < self.peak_end:
                hour_factor = self.peak_multiplier
            else:
                hour_factor = 1.0
        
        return day_factor * hour_factor


class PriceEffect:
    """
    Price elasticity effect on arrival rate.
    
    Uses constant elasticity demand model:
    P(F, p) = (F_ref/F)^ε_F × (p_ref/p)^ε_p
    
    Lower prices → higher arrivals (price elasticity is positive)
    """
    
    def __init__(
        self,
        fee_elasticity: float = 0.3,
        overage_elasticity: float = 0.2,
        reference_fee_factor: float = 1.0,
        reference_overage_factor: float = 1.0
    ):
        """
        Initialize price effect model.
        
        Args:
            fee_elasticity: Elasticity of demand w.r.t. access fee
            overage_elasticity: Elasticity w.r.t. overage price
            reference_fee_factor: Reference fee factor (baseline)
            reference_overage_factor: Reference overage factor
        """
        self.fee_elasticity = fee_elasticity
        self.overage_elasticity = overage_elasticity
        self.ref_fee = reference_fee_factor
        self.ref_overage = reference_overage_factor
    
    def get_price_factor(
        self,
        fee_factor: float,
        overage_factor: float,
        expected_overage_prob: float = 0.0
    ) -> float:
        """
        Calculate price effect on arrivals.
        
        Args:
            fee_factor: Current fee factor (1.0 = base price)
            overage_factor: Current overage factor
            expected_overage_prob: Expected probability user will pay overage
                                   (weights the overage elasticity)
        
        Returns:
            Price effect multiplier
        """
        # Fee effect - always applies
        fee_effect = (self.ref_fee / max(fee_factor, 0.1)) ** self.fee_elasticity
        
        # Overage effect - weighted by expected overage probability
        # New users care about overage less if allowance is generous
        overage_effect = (self.ref_overage / max(overage_factor, 0.1)) ** \
                        (self.overage_elasticity * expected_overage_prob)
        
        return fee_effect * overage_effect


class QoSEffect:
    """
    QoS quality effect on arrival rate.
    
    Poor QoS reduces new arrivals (reputation/word-of-mouth effect).
    Q(V) = exp(-α × V)
    """
    
    def __init__(self, sensitivity: float = 0.5):
        """
        Args:
            sensitivity: Sensitivity to QoS violations (α)
        """
        self.sensitivity = sensitivity
    
    def get_qos_factor(self, violation_rate: float) -> float:
        """
        Calculate QoS effect on arrivals.
        
        Args:
            violation_rate: Current QoS violation rate [0, 1]
        
        Returns:
            QoS effect multiplier
        """
        return np.exp(-self.sensitivity * violation_rate)


@dataclass
class ArrivalModelConfig:
    """Configuration for arrival model."""
    base_rate: float = 1.0
    peak_hours: Tuple[int, int] = (9, 18)
    peak_multiplier: float = 1.5
    weekend_factor: float = 0.7
    fee_elasticity: float = 0.3
    overage_elasticity: float = 0.2
    qos_sensitivity: float = 0.5


class NHPPArrivalModel:
    """
    Non-Homogeneous Poisson Process arrival model.
    
    Generates arrivals with time-varying, price-dependent, QoS-affected intensity.
    """
    
    def __init__(
        self,
        base_rate: float,
        time_profile: TimeProfile,
        price_effect: PriceEffect,
        qos_effect: QoSEffect,
        max_arrivals_per_hour: int = 100
    ):
        """
        Initialize NHPP arrival model.
        
        Args:
            base_rate: Base arrival rate (λ₀) per hour
            time_profile: Time profile component
            price_effect: Price effect component
            qos_effect: QoS effect component
            max_arrivals_per_hour: Cap on arrivals per hour
        """
        self.base_rate = base_rate
        self.time_profile = time_profile
        self.price_effect = price_effect
        self.qos_effect = qos_effect
        self.max_arrivals = max_arrivals_per_hour
        
        # Track history for analysis
        self.arrival_history: List[Dict] = []
    
    @classmethod
    def from_config(cls, config: ArrivalModelConfig) -> 'NHPPArrivalModel':
        """Create model from configuration."""
        return cls(
            base_rate=config.base_rate,
            time_profile=TimeProfile(
                config.peak_hours,
                config.peak_multiplier,
                config.weekend_factor
            ),
            price_effect=PriceEffect(
                config.fee_elasticity,
                config.overage_elasticity
            ),
            qos_effect=QoSEffect(config.qos_sensitivity)
        )
    
    def get_intensity(
        self,
        hour: int,
        fee_factor: float,
        overage_factor: float,
        violation_rate: float,
        expected_overage_prob: float = 0.1
    ) -> float:
        """
        Calculate arrival intensity for given conditions.
        
        Args:
            hour: Current hour
            fee_factor: Current fee factor
            overage_factor: Current overage factor
            violation_rate: Current QoS violation rate
            expected_overage_prob: Expected overage probability
        
        Returns:
            Arrival intensity (λ)
        """
        # Component factors
        time_factor = self.time_profile.get_hourly_factor(hour)
        price_factor = self.price_effect.get_price_factor(
            fee_factor, overage_factor, expected_overage_prob
        )
        qos_factor = self.qos_effect.get_qos_factor(violation_rate)
        
        # Combined intensity
        intensity = self.base_rate * time_factor * price_factor * qos_factor
        
        return intensity
    
    def generate_arrivals(
        self,
        hour: int,
        fee_factor: float,
        overage_factor: float,
        violation_rate: float,
        expected_overage_prob: float = 0.1
    ) -> int:
        """
        Generate number of arrivals for current hour.
        
        Args:
            hour: Current hour
            fee_factor: Current fee factor
            overage_factor: Current overage factor
            violation_rate: Current QoS violation rate
            expected_overage_prob: Expected overage probability
        
        Returns:
            Number of new arrivals (Poisson distributed)
        """
        intensity = self.get_intensity(
            hour, fee_factor, overage_factor, 
            violation_rate, expected_overage_prob
        )
        
        # Poisson sample
        n_arrivals = np.random.poisson(intensity)
        n_arrivals = min(n_arrivals, self.max_arrivals)
        
        # Record for analysis
        self.arrival_history.append({
            "hour": hour,
            "intensity": intensity,
            "arrivals": n_arrivals,
            "fee_factor": fee_factor,
            "overage_factor": overage_factor,
            "violation_rate": violation_rate
        })
        
        return n_arrivals
    
    def get_statistics(self) -> Dict[str, float]:
        """Get arrival statistics from history."""
        if not self.arrival_history:
            return {}
        
        arrivals = [h["arrivals"] for h in self.arrival_history]
        intensities = [h["intensity"] for h in self.arrival_history]
        
        return {
            "total_arrivals": sum(arrivals),
            "mean_arrivals_per_hour": np.mean(arrivals),
            "std_arrivals": np.std(arrivals),
            "mean_intensity": np.mean(intensities),
            "max_arrivals": max(arrivals),
        }


class SliceArrivalManager:
    """
    Manages arrivals for both URLLC and eMBB slices.
    """
    
    def __init__(
        self,
        urllc_config: Optional[ArrivalModelConfig] = None,
        embb_config: Optional[ArrivalModelConfig] = None
    ):
        """
        Initialize arrival manager for both slices.
        
        Args:
            urllc_config: URLLC arrival configuration
            embb_config: eMBB arrival configuration
        """
        # Default URLLC config (B2B: business hours, price-inelastic)
        if urllc_config is None:
            urllc_config = ArrivalModelConfig(
                base_rate=0.2,
                peak_hours=(9, 18),
                peak_multiplier=1.2,
                weekend_factor=0.5,
                fee_elasticity=0.3,
                overage_elasticity=0.2,
                qos_sensitivity=0.5
            )
        
        # Default eMBB config (B2C: evening peak, price-elastic)
        if embb_config is None:
            embb_config = ArrivalModelConfig(
                base_rate=1.0,
                peak_hours=(18, 23),
                peak_multiplier=2.0,
                weekend_factor=1.2,
                fee_elasticity=1.2,
                overage_elasticity=1.5,
                qos_sensitivity=0.3
            )
        
        self.urllc_model = NHPPArrivalModel.from_config(urllc_config)
        self.embb_model = NHPPArrivalModel.from_config(embb_config)
    
    def generate_arrivals(
        self,
        hour: int,
        urllc_fee_factor: float,
        urllc_overage_factor: float,
        urllc_violation_rate: float,
        embb_fee_factor: float,
        embb_overage_factor: float,
        embb_violation_rate: float,
        urllc_overage_prob: float = 0.1,
        embb_overage_prob: float = 0.3
    ) -> Tuple[int, int]:
        """
        Generate arrivals for both slices.
        
        Returns:
            Tuple of (urllc_arrivals, embb_arrivals)
        """
        urllc_arrivals = self.urllc_model.generate_arrivals(
            hour, urllc_fee_factor, urllc_overage_factor,
            urllc_violation_rate, urllc_overage_prob
        )
        
        embb_arrivals = self.embb_model.generate_arrivals(
            hour, embb_fee_factor, embb_overage_factor,
            embb_violation_rate, embb_overage_prob
        )
        
        return urllc_arrivals, embb_arrivals


if __name__ == "__main__":
    print("=" * 60)
    print("NHPP Arrival Model Test")
    print("=" * 60)
    
    # Create arrival manager
    manager = SliceArrivalManager()
    
    # Simulate one week
    print("\nSimulating one week of arrivals...")
    
    urllc_total = 0
    embb_total = 0
    
    for hour in range(168):
        urllc, embb = manager.generate_arrivals(
            hour=hour,
            urllc_fee_factor=1.0,
            urllc_overage_factor=1.0,
            urllc_violation_rate=0.01,
            embb_fee_factor=1.0,
            embb_overage_factor=1.0,
            embb_violation_rate=0.05
        )
        urllc_total += urllc
        embb_total += embb
        
        if hour % 24 == 0:
            day = hour // 24
            print(f"  Day {day}: URLLC={urllc_total}, eMBB={embb_total}")
    
    print(f"\nWeek totals: URLLC={urllc_total}, eMBB={embb_total}")
    
    # Statistics
    print("\nURLLC Statistics:")
    stats = manager.urllc_model.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\neMBB Statistics:")
    stats = manager.embb_model.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Test price effect
    print("\n" + "=" * 60)
    print("Price Effect Test")
    print("=" * 60)
    
    print("\neMBB arrivals vs price factor (hour 20, evening peak):")
    for fee_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        model = NHPPArrivalModel.from_config(ArrivalModelConfig(
            base_rate=1.0,
            peak_hours=(18, 23),
            peak_multiplier=2.0,
            fee_elasticity=1.2
        ))
        
        arrivals = []
        for _ in range(1000):
            n = model.generate_arrivals(20, fee_factor, 1.0, 0.0)
            arrivals.append(n)
        
        print(f"  Fee factor {fee_factor:.1f}: mean={np.mean(arrivals):.2f} arrivals/hr")
