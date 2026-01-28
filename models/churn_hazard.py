"""
User Churn Model - Hazard/Logistic Model

Implements churn probability calculation with QoS and price sensitivity.

P(churn) = σ(b₀ + b_V×V + b_C×C + b_O×O + b_F×F)

where:
- σ: Sigmoid function
- b₀: Base log-odds (intercept)
- V: QoS violation rate
- C: Consecutive violations count
- O: Overage payment burden
- F: Fee level

References:
- Scientific Reports 2024: Telecom churn prediction (91.66% accuracy)
- MDPI Algorithms 2024: ML-based churn modeling in telecom
- ScienceDirect 2024 (RICO): QoS-satisfaction based churn
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)


@dataclass
class ChurnCoefficients:
    """
    Logistic regression coefficients for churn model.
    
    All coefficients should be estimated/calibrated from data or
    based on documented literature values.
    
    Reference:
        Scientific Reports 2024: Key churn predictors include tenure,
        QoS violations, and overage burden.
    """
    b0: float = -6.0       # Base log-odds (intercept)
    b_violation: float = 2.0  # QoS violation sensitivity
    b_consecutive: float = 1.5  # Consecutive violation penalty
    b_overage: float = 1.0    # Overage payment sensitivity
    b_fee: float = 0.05       # Fee sensitivity
    b_tenure: float = -0.01   # Tenure effect (negative = longer tenure reduces churn)
    
    def validate(self) -> bool:
        """Validate coefficient signs are economically sensible."""
        # All coefficients except b0 and b_tenure should be positive 
        # (violations, overage, fees increase churn)
        # b_tenure should be negative (longer tenure reduces churn)
        return (
            self.b_violation >= 0 and
            self.b_consecutive >= 0 and
            self.b_overage >= 0 and
            self.b_fee >= 0 and
            self.b_tenure <= 0  # Tenure reduces churn
        )


@dataclass
class UserChurnState:
    """
    Track user state relevant to churn prediction.
    """
    user_id: int
    slice_type: str
    
    # QoS history
    total_violations: int = 0
    consecutive_violations: int = 0
    recent_violation_rate: float = 0.0
    
    # Billing history
    total_overage_payments: float = 0.0
    recent_overage_payment: float = 0.0
    overage_to_base_ratio: float = 0.0  # Overage burden
    
    # Subscription history
    hours_subscribed: int = 0
    satisfaction_score: float = 1.0
    
    # Track for violation rate calculation
    violation_history: List[bool] = field(default_factory=list)
    history_window: int = 24  # Hours for rolling window
    
    def update_qos(self, violated: bool) -> None:
        """Update QoS violation tracking."""
        if violated:
            self.total_violations += 1
            self.consecutive_violations += 1
        else:
            self.consecutive_violations = 0
        
        # Update rolling history
        self.violation_history.append(violated)
        if len(self.violation_history) > self.history_window:
            self.violation_history.pop(0)
        
        # Calculate recent violation rate
        if self.violation_history:
            self.recent_violation_rate = sum(self.violation_history) / len(self.violation_history)
    
    def update_billing(
        self, 
        overage_payment: float, 
        base_fee: float
    ) -> None:
        """Update billing state."""
        self.recent_overage_payment = overage_payment
        self.total_overage_payments += overage_payment
        
        # Overage burden relative to base fee
        if base_fee > 0:
            self.overage_to_base_ratio = overage_payment / base_fee
        else:
            self.overage_to_base_ratio = 0.0
    
    def update_subscription(self) -> None:
        """Increment subscription time."""
        self.hours_subscribed += 1
    
    def update_satisfaction(
        self,
        violation_rate: float,
        overage_burden: float,
        price_deviation: float = 0.0
    ) -> None:
        """
        Update satisfaction score.
        
        Satisfaction decreases with violations and overage burden.
        """
        # Weight factors (could be calibrated)
        w_viol = 0.4
        w_overage = 0.3
        w_price = 0.3
        
        # Components
        viol_component = 1 - violation_rate
        overage_component = 1 / (1 + overage_burden)  # Diminishes with burden
        price_component = 1 - abs(price_deviation)
        
        self.satisfaction_score = (
            w_viol * viol_component +
            w_overage * overage_component +
            w_price * price_component
        )
        self.satisfaction_score = np.clip(self.satisfaction_score, 0, 1)


class ChurnModel:
    """
    Logistic/Hazard model for user churn prediction.
    
    Calculates probability of churn based on user state and pricing.
    Includes tenure effect based on academic literature.
    
    Reference:
        Scientific Reports 2024: Tenure is a key predictor of churn,
        with early-stage customers showing 2-3x higher churn risk.
    """
    
    def __init__(
        self,
        coefficients: ChurnCoefficients,
        check_interval_hours: int = 6,
        max_churn_prob: float = 0.1
    ):
        """
        Initialize churn model.
        
        Args:
            coefficients: Model coefficients
            check_interval_hours: How often to evaluate churn
            max_churn_prob: Maximum allowed churn probability per check
        """
        self.coefficients = coefficients
        self.check_interval = check_interval_hours
        self.max_prob = max_churn_prob
        
        if not coefficients.validate():
            raise ValueError("Invalid coefficient signs")
    
    @staticmethod
    def tenure_factor(hours_subscribed: int) -> float:
        """
        Calculate tenure-based risk factor for churn.
        
        Longer tenure reduces churn risk due to:
        - Switching costs (learned usage patterns, bundled services)
        - Loyalty/habit formation
        - Selection effect (dissatisfied customers already left)
        
        Reference:
            Scientific Reports 2024: Early-stage customers (< 90 days)
            show 2-3x higher churn rates than established customers.
        
        Args:
            hours_subscribed: Total hours since subscription start
            
        Returns:
            Tenure factor (>1 for new customers, <1 for loyal customers)
        """
        days = hours_subscribed / 24
        
        # Piecewise linear decay based on literature patterns
        if days < 30:
            # First month: highest churn risk (acquisition phase)
            # Factor: 2.0 → 1.5 over 30 days
            return 2.0 - 0.5 * (days / 30)
        elif days < 90:
            # 1-3 months: elevated risk (evaluation phase)
            # Factor: 1.5 → 1.0 over 60 days
            return 1.5 - 0.5 * ((days - 30) / 60)
        elif days < 180:
            # 3-6 months: moderate risk (commitment phase)
            # Factor: 1.0 → 0.7 over 90 days
            return 1.0 - 0.3 * ((days - 90) / 90)
        elif days < 365:
            # 6-12 months: low risk (loyalty phase)
            # Factor: 0.7 → 0.5 over 185 days
            return 0.7 - 0.2 * ((days - 180) / 185)
        else:
            # 1+ year: loyal customer (retention phase)
            # Factor: 0.5 (minimum, very low churn risk)
            return 0.5
    
    def calculate_churn_probability(
        self,
        user_state: UserChurnState,
        fee_factor: float = 1.0
    ) -> float:
        """
        Calculate churn probability for a user.
        
        Incorporates tenure effect as multiplicative factor on base probability.
        
        Args:
            user_state: User's current state
            fee_factor: Current fee factor (1.0 = base price)
        
        Returns:
            Churn probability [0, max_prob]
        """
        c = self.coefficients
        
        # Linear combination of features
        log_odds = (
            c.b0 +
            c.b_violation * user_state.recent_violation_rate +
            c.b_consecutive * user_state.consecutive_violations +
            c.b_overage * user_state.overage_to_base_ratio +
            c.b_fee * (fee_factor - 1.0) +  # Deviation from base
            c.b_tenure * (user_state.hours_subscribed / 24)  # Days subscribed
        )
        
        # Sigmoid to probability
        base_prob = sigmoid(log_odds)
        
        # Apply tenure factor (multiplicative adjustment)
        tenure_adj = self.tenure_factor(user_state.hours_subscribed)
        prob = base_prob * tenure_adj
        
        # Cap at maximum
        return min(prob, self.max_prob)
    
    def should_check_churn(self, hour: int) -> bool:
        """Check if churn should be evaluated at this hour."""
        return hour % self.check_interval == 0
    
    def evaluate_churn(
        self,
        user_state: UserChurnState,
        fee_factor: float,
        hour: int
    ) -> Tuple[bool, float]:
        """
        Evaluate if user churns.
        
        Args:
            user_state: User state
            fee_factor: Current fee factor
            hour: Current hour
        
        Returns:
            Tuple of (churned, probability)
        """
        if not self.should_check_churn(hour):
            return False, 0.0
        
        prob = self.calculate_churn_probability(user_state, fee_factor)
        churned = np.random.random() < prob
        
        return churned, prob


@dataclass
class SliceChurnConfig:
    """Configuration for slice-specific churn model."""
    coefficients: ChurnCoefficients
    check_interval_hours: int = 6
    max_churn_prob: float = 0.1


class ChurnManager:
    """
    Manages churn evaluation for both URLLC and eMBB slices.
    """
    
    def __init__(
        self,
        urllc_config: Optional[SliceChurnConfig] = None,
        embb_config: Optional[SliceChurnConfig] = None
    ):
        """
        Initialize churn manager.
        
        Args:
            urllc_config: URLLC churn configuration
            embb_config: eMBB churn configuration
        """
        # Default URLLC config (B2B: low base churn, high QoS sensitivity)
        if urllc_config is None:
            urllc_config = SliceChurnConfig(
                coefficients=ChurnCoefficients(
                    b0=-6.0,           # Very low base churn
                    b_violation=2.0,    # High QoS sensitivity
                    b_consecutive=1.5,  # Strong consecutive penalty
                    b_overage=0.5,      # Lower overage sensitivity (B2B)
                    b_fee=0.01          # Low fee sensitivity (committed contracts)
                ),
                check_interval_hours=6,  # SLA review cycle
                max_churn_prob=0.02      # 2% max per check
            )
        
        # Default eMBB config (B2C: higher base, price sensitive)
        if embb_config is None:
            embb_config = SliceChurnConfig(
                coefficients=ChurnCoefficients(
                    b0=-5.0,           # Higher base churn
                    b_violation=1.0,    # Lower QoS sensitivity
                    b_consecutive=1.2,  # Moderate consecutive penalty
                    b_overage=2.0,      # High overage sensitivity (B2C)
                    b_fee=0.1           # Price sensitive
                ),
                check_interval_hours=3,  # More frequent checks
                max_churn_prob=0.03      # 3% max per check
            )
        
        self.urllc_model = ChurnModel(
            urllc_config.coefficients,
            urllc_config.check_interval_hours,
            urllc_config.max_churn_prob
        )
        
        self.embb_model = ChurnModel(
            embb_config.coefficients,
            embb_config.check_interval_hours,
            embb_config.max_churn_prob
        )
        
        # Track user states
        self.user_states: Dict[int, UserChurnState] = {}
        
        # Churn history for analysis
        self.churn_history: List[Dict] = []
    
    def register_user(self, user_id: int, slice_type: str) -> None:
        """Register new user."""
        self.user_states[user_id] = UserChurnState(
            user_id=user_id,
            slice_type=slice_type
        )
    
    def remove_user(self, user_id: int) -> None:
        """Remove churned user."""
        if user_id in self.user_states:
            del self.user_states[user_id]
    
    def update_user_qos(self, user_id: int, violated: bool) -> None:
        """Update user QoS state."""
        if user_id in self.user_states:
            self.user_states[user_id].update_qos(violated)
    
    def update_user_billing(
        self, 
        user_id: int, 
        overage_payment: float,
        base_fee: float
    ) -> None:
        """Update user billing state."""
        if user_id in self.user_states:
            self.user_states[user_id].update_billing(overage_payment, base_fee)
    
    def get_model_for_slice(self, slice_type: str) -> ChurnModel:
        """Get appropriate churn model for slice."""
        if slice_type.upper() == "URLLC":
            return self.urllc_model
        else:
            return self.embb_model
    
    def evaluate_all_users(
        self,
        urllc_fee_factor: float,
        embb_fee_factor: float,
        hour: int
    ) -> Tuple[List[int], List[int]]:
        """
        Evaluate churn for all users.
        
        Args:
            urllc_fee_factor: Current URLLC fee factor
            embb_fee_factor: Current eMBB fee factor
            hour: Current simulation hour
        
        Returns:
            Tuple of (urllc_churned_ids, embb_churned_ids)
        """
        urllc_churned = []
        embb_churned = []
        
        for user_id, state in list(self.user_states.items()):
            # Update subscription time
            state.update_subscription()
            
            # Get appropriate model and fee factor
            model = self.get_model_for_slice(state.slice_type)
            fee_factor = (
                urllc_fee_factor if state.slice_type.upper() == "URLLC"
                else embb_fee_factor
            )
            
            # Evaluate churn
            churned, prob = model.evaluate_churn(state, fee_factor, hour)
            
            if churned:
                # Record churn event
                self.churn_history.append({
                    "hour": hour,
                    "user_id": user_id,
                    "slice_type": state.slice_type,
                    "probability": prob,
                    "violation_rate": state.recent_violation_rate,
                    "consecutive_violations": state.consecutive_violations,
                    "overage_ratio": state.overage_to_base_ratio,
                    "hours_subscribed": state.hours_subscribed
                })
                
                if state.slice_type.upper() == "URLLC":
                    urllc_churned.append(user_id)
                else:
                    embb_churned.append(user_id)
                
                # Remove from tracking
                self.remove_user(user_id)
        
        return urllc_churned, embb_churned
    
    def get_user_churn_probability(self, user_id: int, fee_factor: float) -> float:
        """Get current churn probability for a user."""
        if user_id not in self.user_states:
            return 0.0
        
        state = self.user_states[user_id]
        model = self.get_model_for_slice(state.slice_type)
        return model.calculate_churn_probability(state, fee_factor)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get churn statistics."""
        if not self.churn_history:
            return {
                "total_churned": 0,
                "urllc_churned": 0,
                "embb_churned": 0,
            }
        
        urllc_churned = sum(
            1 for h in self.churn_history if h["slice_type"].upper() == "URLLC"
        )
        embb_churned = len(self.churn_history) - urllc_churned
        
        avg_hours = np.mean([h["hours_subscribed"] for h in self.churn_history])
        avg_viol = np.mean([h["violation_rate"] for h in self.churn_history])
        
        return {
            "total_churned": len(self.churn_history),
            "urllc_churned": urllc_churned,
            "embb_churned": embb_churned,
            "avg_hours_before_churn": avg_hours,
            "avg_violation_rate_at_churn": avg_viol
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Churn Model Test")
    print("=" * 60)
    
    # Create churn manager
    manager = ChurnManager()
    
    # Register test users
    for i in range(5):
        manager.register_user(i, "URLLC")
    for i in range(5, 25):
        manager.register_user(i, "eMBB")
    
    print(f"\nRegistered users: {len(manager.user_states)}")
    
    # Simulate different QoS scenarios
    print("\nSimulating 168 hours with varying QoS...")
    
    np.random.seed(42)
    
    for hour in range(168):
        # Update QoS randomly
        for user_id, state in manager.user_states.items():
            # URLLC has higher QoS, eMBB more violations
            if state.slice_type == "URLLC":
                violated = np.random.random() < 0.02  # 2% violation rate
            else:
                violated = np.random.random() < 0.10  # 10% violation rate
            
            manager.update_user_qos(user_id, violated)
            
            # Simulate billing (some overage)
            if state.slice_type == "eMBB":
                overage = np.random.exponential(0.5)  # Random overage
                manager.update_user_billing(user_id, overage, 5.0)
        
        # Evaluate churn
        urllc_churned, embb_churned = manager.evaluate_all_users(
            urllc_fee_factor=1.0,
            embb_fee_factor=1.0,
            hour=hour
        )
        
        if urllc_churned or embb_churned:
            print(f"  Hour {hour}: URLLC churned={urllc_churned}, eMBB churned={embb_churned}")
    
    print(f"\nRemaining users: {len(manager.user_states)}")
    
    # Statistics
    print("\nChurn Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test churn probability calculation
    print("\n" + "=" * 60)
    print("Churn Probability Analysis")
    print("=" * 60)
    
    # Create a fresh user for testing
    test_state = UserChurnState(user_id=999, slice_type="eMBB")
    model = manager.embb_model
    
    print("\neMBB churn probability vs violation rate:")
    for viol_rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
        test_state.recent_violation_rate = viol_rate
        test_state.consecutive_violations = 0
        test_state.overage_to_base_ratio = 0
        prob = model.calculate_churn_probability(test_state, 1.0)
        print(f"  Violation rate {viol_rate:.1f}: P(churn) = {prob:.4f}")
    
    print("\neMBB churn probability vs overage burden:")
    for overage in [0.0, 0.5, 1.0, 2.0, 5.0]:
        test_state.recent_violation_rate = 0.1
        test_state.consecutive_violations = 0
        test_state.overage_to_base_ratio = overage
        prob = model.calculate_churn_probability(test_state, 1.0)
        print(f"  Overage ratio {overage:.1f}: P(churn) = {prob:.4f}")
