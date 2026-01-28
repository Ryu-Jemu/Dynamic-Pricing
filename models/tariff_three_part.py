"""
Three-Part Tariff Pricing Model

Implements the three-part tariff pricing structure for 5G network slices.

Revenue = F + max(0, U - D) × p

where:
- F: Access fee (fixed monthly/weekly fee)
- D: Allowance (included data per billing cycle)
- U: Actual usage
- p: Overage price per unit

References:
- Fibich et al. (2017): "Optimal Three-Part Tariff Plans", Operations Research
- Bhargava & Gangwar (2018): "On the Optimality of Three-Part Tariffs"
- AT&T Mobile Share Value Plans, Verizon 5G Network Slice pricing (2024)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from enum import Enum


class SliceType(Enum):
    """Network slice types."""
    URLLC = "URLLC"
    EMBB = "eMBB"


@dataclass
class TariffPlan:
    """
    Three-Part Tariff plan definition.
    
    Attributes:
        name: Plan name
        slice_type: URLLC or eMBB
        access_fee: Fixed fee per billing period ($)
        allowance: Included usage per billing period (MB)
        overage_price: Price per MB over allowance ($)
        billing_cycle_hours: Length of billing cycle in hours
    """
    name: str
    slice_type: SliceType
    access_fee: float  # $ per billing cycle
    allowance: float   # MB per billing cycle
    overage_price: float  # $ per MB
    billing_cycle_hours: int = 168  # 1 week default
    
    def __post_init__(self):
        """Validate tariff parameters."""
        if self.access_fee < 0:
            raise ValueError("Access fee cannot be negative")
        if self.allowance < 0:
            raise ValueError("Allowance cannot be negative")
        if self.overage_price < 0:
            raise ValueError("Overage price cannot be negative")
    
    def calculate_bill(self, usage_mb: float, remaining_allowance: float) -> Tuple[float, float]:
        """
        Calculate bill for given usage.
        
        Args:
            usage_mb: Data usage in MB for this period
            remaining_allowance: Remaining allowance from billing cycle
        
        Returns:
            Tuple of (total_bill, overage_charge)
        """
        overage_mb = max(0, usage_mb - remaining_allowance)
        overage_charge = overage_mb * self.overage_price
        
        # Note: Access fee is typically charged once per billing cycle
        # For hourly billing, we prorate it
        hourly_access_fee = self.access_fee / self.billing_cycle_hours
        
        total_bill = hourly_access_fee + overage_charge
        return total_bill, overage_charge
    
    def get_hourly_access_fee(self) -> float:
        """Get prorated hourly access fee."""
        return self.access_fee / self.billing_cycle_hours


@dataclass
class UserBillingState:
    """
    Track user's billing state within a billing cycle.
    
    Attributes:
        user_id: Unique user identifier
        slice_type: User's slice type
        cumulative_usage_mb: Total usage in current billing cycle
        remaining_allowance_mb: Remaining free usage
        total_bill: Total bill accumulated this cycle
        overage_payments: Total overage charges this cycle
        billing_cycle_start: Hour when current cycle started
    """
    user_id: int
    slice_type: SliceType
    cumulative_usage_mb: float = 0.0
    remaining_allowance_mb: float = 0.0
    total_bill: float = 0.0
    overage_payments: float = 0.0
    billing_cycle_start: int = 0
    
    def reset_for_new_cycle(self, allowance: float, current_hour: int) -> None:
        """Reset billing state for new cycle."""
        self.cumulative_usage_mb = 0.0
        self.remaining_allowance_mb = allowance
        self.total_bill = 0.0
        self.overage_payments = 0.0
        self.billing_cycle_start = current_hour
    
    def record_usage(
        self, 
        usage_mb: float, 
        tariff: TariffPlan
    ) -> Tuple[float, float]:
        """
        Record usage and calculate charges.
        
        Args:
            usage_mb: Usage for this hour
            tariff: Applicable tariff plan
        
        Returns:
            Tuple of (hourly_bill, overage_charge)
        """
        # Calculate charges
        hourly_bill, overage_charge = tariff.calculate_bill(
            usage_mb, self.remaining_allowance_mb
        )
        
        # Update state
        self.cumulative_usage_mb += usage_mb
        self.remaining_allowance_mb = max(0, self.remaining_allowance_mb - usage_mb)
        self.total_bill += hourly_bill
        self.overage_payments += overage_charge
        
        return hourly_bill, overage_charge
    
    def get_overage_ratio(self) -> float:
        """
        Get ratio of overage payments to total bill.
        
        Useful for churn modeling - high overage ratio indicates price dissatisfaction.
        """
        if self.total_bill <= 0:
            return 0.0
        return self.overage_payments / self.total_bill


class DynamicTariffManager:
    """
    Manages dynamic three-part tariff pricing controlled by RL agent.
    
    The RL agent adjusts:
    - F: Access fee (within bounds)
    - p: Overage price (within bounds)
    
    D (allowance) is typically fixed per slice type.
    """
    
    def __init__(
        self,
        urllc_base_fee: float = 50.0,
        urllc_allowance_mb: float = 10.0,
        urllc_base_overage: float = 0.50,
        embb_base_fee: float = 5.0,
        embb_allowance_mb: float = 300.0,
        embb_base_overage: float = 0.02,
        fee_adjustment_range: Tuple[float, float] = (0.8, 1.2),
        overage_adjustment_range: Tuple[float, float] = (0.8, 1.2),
        billing_cycle_hours: int = 168
    ):
        """
        Initialize tariff manager.
        
        Args:
            urllc_base_fee: Base access fee for URLLC ($ per billing cycle)
            urllc_allowance_mb: Allowance for URLLC (MB)
            urllc_base_overage: Base overage price for URLLC ($/MB)
            embb_base_fee: Base access fee for eMBB
            embb_allowance_mb: Allowance for eMBB
            embb_base_overage: Base overage price for eMBB
            fee_adjustment_range: (min, max) multiplier for fees
            overage_adjustment_range: (min, max) multiplier for overage
            billing_cycle_hours: Length of billing cycle
        """
        # Base parameters (scenario inputs)
        self.urllc_base_fee = urllc_base_fee
        self.urllc_allowance_mb = urllc_allowance_mb
        self.urllc_base_overage = urllc_base_overage
        
        self.embb_base_fee = embb_base_fee
        self.embb_allowance_mb = embb_allowance_mb
        self.embb_base_overage = embb_base_overage
        
        # Adjustment ranges
        self.fee_min, self.fee_max = fee_adjustment_range
        self.overage_min, self.overage_max = overage_adjustment_range
        
        self.billing_cycle_hours = billing_cycle_hours
        
        # Current price factors (RL agent controls these)
        self.urllc_fee_factor = 1.0
        self.urllc_overage_factor = 1.0
        self.embb_fee_factor = 1.0
        self.embb_overage_factor = 1.0
        
        # User billing states
        self.user_states: Dict[int, UserBillingState] = {}
    
    def reset(self) -> None:
        """Reset tariff manager to initial state."""
        # Reset price factors
        self.urllc_fee_factor = 1.0
        self.urllc_overage_factor = 1.0
        self.embb_fee_factor = 1.0
        self.embb_overage_factor = 1.0
        
        # Clear user billing states
        self.user_states.clear()
    
    def update_prices(
        self,
        urllc_fee_factor: float,
        urllc_overage_factor: float,
        embb_fee_factor: float,
        embb_overage_factor: float
    ) -> None:
        """
        Update price factors from RL agent action.
        
        Args:
            urllc_fee_factor: Multiplier for URLLC access fee
            urllc_overage_factor: Multiplier for URLLC overage price
            embb_fee_factor: Multiplier for eMBB access fee
            embb_overage_factor: Multiplier for eMBB overage price
        """
        # Clip to valid range
        self.urllc_fee_factor = np.clip(urllc_fee_factor, self.fee_min, self.fee_max)
        self.urllc_overage_factor = np.clip(urllc_overage_factor, self.overage_min, self.overage_max)
        self.embb_fee_factor = np.clip(embb_fee_factor, self.fee_min, self.fee_max)
        self.embb_overage_factor = np.clip(embb_overage_factor, self.overage_min, self.overage_max)
    
    def get_current_tariff(self, slice_type: SliceType) -> TariffPlan:
        """
        Get current tariff plan with dynamic pricing applied.
        
        Args:
            slice_type: URLLC or eMBB
        
        Returns:
            Current TariffPlan with adjusted prices
        """
        if slice_type == SliceType.URLLC:
            return TariffPlan(
                name="URLLC Dynamic",
                slice_type=slice_type,
                access_fee=self.urllc_base_fee * self.urllc_fee_factor,
                allowance=self.urllc_allowance_mb,
                overage_price=self.urllc_base_overage * self.urllc_overage_factor,
                billing_cycle_hours=self.billing_cycle_hours
            )
        else:
            return TariffPlan(
                name="eMBB Dynamic",
                slice_type=slice_type,
                access_fee=self.embb_base_fee * self.embb_fee_factor,
                allowance=self.embb_allowance_mb,
                overage_price=self.embb_base_overage * self.embb_overage_factor,
                billing_cycle_hours=self.billing_cycle_hours
            )
    
    def register_user(
        self, 
        user_id: int, 
        slice_type: SliceType, 
        current_hour: int
    ) -> None:
        """
        Register new user with initial billing state.
        
        Args:
            user_id: User identifier
            slice_type: User's slice type
            current_hour: Current simulation hour
        """
        allowance = (
            self.urllc_allowance_mb if slice_type == SliceType.URLLC 
            else self.embb_allowance_mb
        )
        
        self.user_states[user_id] = UserBillingState(
            user_id=user_id,
            slice_type=slice_type,
            remaining_allowance_mb=allowance,
            billing_cycle_start=current_hour
        )
    
    def add_user(
        self, 
        user_id: int, 
        slice_type_str: str, 
        current_hour: int
    ) -> None:
        """
        Alias for register_user that accepts string slice type.
        
        Args:
            user_id: User identifier
            slice_type_str: User's slice type as string ("URLLC" or "eMBB")
            current_hour: Current simulation hour
        """
        slice_type = SliceType.URLLC if slice_type_str == "URLLC" else SliceType.EMBB
        self.register_user(user_id, slice_type, current_hour)
    
    def remove_user(self, user_id: int) -> None:
        """Remove user from billing system."""
        if user_id in self.user_states:
            del self.user_states[user_id]
    
    def bill_user(
        self, 
        user_id: int, 
        usage_mb: float,
        current_hour: int
    ) -> Tuple[float, float, float]:
        """
        Bill user for hourly usage.
        
        Args:
            user_id: User identifier
            usage_mb: Usage for this hour in MB
            current_hour: Current simulation hour
        
        Returns:
            Tuple of (hourly_revenue, overage_charge, remaining_allowance)
        """
        if user_id not in self.user_states:
            return 0.0, 0.0, 0.0
        
        state = self.user_states[user_id]
        tariff = self.get_current_tariff(state.slice_type)
        
        # Check for new billing cycle
        hours_since_start = current_hour - state.billing_cycle_start
        if hours_since_start >= self.billing_cycle_hours:
            state.reset_for_new_cycle(tariff.allowance, current_hour)
        
        # Calculate bill
        hourly_revenue, overage_charge = state.record_usage(usage_mb, tariff)
        
        return hourly_revenue, overage_charge, state.remaining_allowance_mb
    
    def get_user_overage_ratio(self, user_id: int) -> float:
        """Get user's overage payment ratio (for churn modeling)."""
        if user_id not in self.user_states:
            return 0.0
        return self.user_states[user_id].get_overage_ratio()
    
    def get_slice_revenue_breakdown(
        self,
        user_ids: List[int],
        usages_mb: List[float],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Calculate revenue breakdown for a set of users.
        
        Returns:
            Dictionary with 'total_revenue', 'access_fees', 'overage_charges'
        """
        total_revenue = 0.0
        access_fees = 0.0
        overage_charges = 0.0
        
        for user_id, usage_mb in zip(user_ids, usages_mb):
            revenue, overage, _ = self.bill_user(user_id, usage_mb, current_hour)
            total_revenue += revenue
            overage_charges += overage
            
            # Access fee component
            if user_id in self.user_states:
                tariff = self.get_current_tariff(self.user_states[user_id].slice_type)
                access_fees += tariff.get_hourly_access_fee()
        
        return {
            "total_revenue": total_revenue,
            "access_fees": access_fees,
            "overage_charges": overage_charges
        }
    
    def get_price_state(self) -> Dict[str, float]:
        """
        Get current pricing state for observation.
        
        Returns:
            Dictionary with current price factors and effective prices
        """
        return {
            "urllc_fee_factor": self.urllc_fee_factor,
            "urllc_overage_factor": self.urllc_overage_factor,
            "embb_fee_factor": self.embb_fee_factor,
            "embb_overage_factor": self.embb_overage_factor,
            "urllc_effective_fee": self.urllc_base_fee * self.urllc_fee_factor,
            "urllc_effective_overage": self.urllc_base_overage * self.urllc_overage_factor,
            "embb_effective_fee": self.embb_base_fee * self.embb_fee_factor,
            "embb_effective_overage": self.embb_base_overage * self.embb_overage_factor,
        }
    
    def get_allowance_stats(self, slice_type: SliceType) -> Dict[str, float]:
        """
        Get statistics on remaining allowances for users of a slice.
        
        Useful for state observation - shows distribution of users near overage.
        """
        remaining = []
        for state in self.user_states.values():
            if state.slice_type == slice_type:
                remaining.append(state.remaining_allowance_mb)
        
        if not remaining:
            return {
                "mean_remaining": 0.0,
                "pct_in_overage": 0.0,
                "pct_near_overage": 0.0  # Less than 20% remaining
            }
        
        remaining = np.array(remaining)
        allowance = (
            self.urllc_allowance_mb if slice_type == SliceType.URLLC 
            else self.embb_allowance_mb
        )
        
        return {
            "mean_remaining": np.mean(remaining),
            "pct_in_overage": np.mean(remaining <= 0),
            "pct_near_overage": np.mean(remaining < 0.2 * allowance)
        }


def action_to_price_factors(
    action: np.ndarray,
    fee_range: Tuple[float, float] = (0.8, 1.2),
    overage_range: Tuple[float, float] = (0.8, 1.2)
) -> Dict[str, float]:
    """
    Convert RL action [-1, 1] to price factors.
    
    Args:
        action: Array of 4 values in [-1, 1]
            [urllc_fee, urllc_overage, embb_fee, embb_overage]
        fee_range: (min, max) for fee factors
        overage_range: (min, max) for overage factors
    
    Returns:
        Dict with keys: urllc_fee, urllc_overage, embb_fee, embb_overage
    """
    def scale(value, low, high):
        """Scale [-1, 1] to [low, high]."""
        return low + (value + 1) * (high - low) / 2
    
    return {
        'urllc_fee': scale(action[0], fee_range[0], fee_range[1]),
        'urllc_overage': scale(action[1], overage_range[0], overage_range[1]),
        'embb_fee': scale(action[2], fee_range[0], fee_range[1]),
        'embb_overage': scale(action[3], overage_range[0], overage_range[1])
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Three-Part Tariff Model Test")
    print("=" * 60)
    
    # Create tariff manager
    manager = DynamicTariffManager()
    
    print("\nBase Tariffs:")
    for slice_type in [SliceType.URLLC, SliceType.EMBB]:
        tariff = manager.get_current_tariff(slice_type)
        print(f"\n{slice_type.value}:")
        print(f"  Access Fee: ${tariff.access_fee:.2f}/week (${tariff.get_hourly_access_fee():.4f}/hr)")
        print(f"  Allowance: {tariff.allowance:.0f} MB")
        print(f"  Overage Price: ${tariff.overage_price:.4f}/MB")
    
    # Simulate user billing
    print("\n" + "=" * 60)
    print("User Billing Simulation")
    print("=" * 60)
    
    # Register users
    manager.register_user(1, SliceType.URLLC, 0)
    manager.register_user(2, SliceType.EMBB, 0)
    
    # URLLC user with low usage (within allowance)
    print("\nURLLC User (low usage - 0.5 MB/hr):")
    for hour in range(24):
        revenue, overage, remaining = manager.bill_user(1, 0.5, hour)
        if hour % 6 == 0:
            print(f"  Hour {hour:2d}: Revenue=${revenue:.4f}, Overage=${overage:.4f}, Remaining={remaining:.1f}MB")
    
    # eMBB user with high usage (will hit overage)
    print("\neMBB User (high usage - 20 MB/hr):")
    manager.register_user(3, SliceType.EMBB, 0)
    for hour in range(24):
        revenue, overage, remaining = manager.bill_user(3, 20.0, hour)
        if hour % 6 == 0:
            print(f"  Hour {hour:2d}: Revenue=${revenue:.4f}, Overage=${overage:.4f}, Remaining={remaining:.1f}MB")
    
    # Test dynamic pricing
    print("\n" + "=" * 60)
    print("Dynamic Pricing Test")
    print("=" * 60)
    
    # Simulate RL action
    action = np.array([0.5, -0.3, 0.2, 0.8])  # Various adjustments
    factors = action_to_price_factors(action)
    print(f"\nAction: {action}")
    print(f"Price factors: {factors}")
    
    manager.update_prices(*factors)
    print(f"\nUpdated prices:")
    state = manager.get_price_state()
    for key, value in state.items():
        print(f"  {key}: {value:.4f}")
