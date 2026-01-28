"""
Hawkes Process Arrival Model for 5G O-RAN Network Slicing

Implements self-exciting point process for user arrivals, capturing
burstiness and clustering effects that NHPP cannot model.

Model:
    λ(t) = μ(t) + Σ_{t_i < t} α * exp(-β * (t - t_i))
    
    where:
    - μ(t): baseline intensity (from NHPP time profile)
    - α: excitation strength (jump size per event)
    - β: decay rate (controls memory)
    - t_i: past event times

References:
    - Hawkes, A.G. (1971). Spectra of some self-exciting and mutually
      exciting point processes. Biometrika, 58(1), 83-90.
    - Laub, P.J., et al. (2015). Hawkes processes. arXiv:1507.02822.
    - Bacry, E., et al. (2015). Hawkes processes in finance. Market
      Microstructure and Liquidity, 1(1).

Author: Research Implementation
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


class ServiceType(Enum):
    """Service type enum for network slicing."""
    URLLC = "URLLC"
    eMBB = "eMBB"


@dataclass
class HawkesParameters:
    """Parameters for Hawkes process model."""
    
    # Baseline intensity (events per hour)
    mu_0: float = 1.0
    
    # Excitation strength (intensity jump per event)
    alpha: float = 0.3
    
    # Decay rate (1/hour) - controls memory duration
    beta: float = 2.0
    
    # Time profile parameters (24-hour periodic)
    time_profile: np.ndarray = field(
        default_factory=lambda: np.ones(24)
    )
    
    # Price elasticity for baseline
    price_elasticity_fee: float = 0.3
    price_elasticity_overage: float = 0.2
    
    # QoS effect on baseline
    qos_sensitivity: float = 0.1
    
    # Reference price factors
    ref_fee_factor: float = 1.0
    ref_overage_factor: float = 1.0
    
    @property
    def branching_ratio(self) -> float:
        """
        Branching ratio n = α/β.
        For stationarity, require n < 1.
        """
        return self.alpha / self.beta if self.beta > 0 else float('inf')
    
    @property
    def is_stationary(self) -> bool:
        """Check if process is stationary (n < 1)."""
        return self.branching_ratio < 1.0
    
    def validate(self) -> bool:
        """Validate parameter constraints."""
        if self.mu_0 < 0:
            raise ValueError("Baseline intensity must be non-negative")
        if self.alpha < 0:
            raise ValueError("Excitation strength must be non-negative")
        if self.beta <= 0:
            raise ValueError("Decay rate must be positive")
        if not self.is_stationary:
            raise ValueError(
                f"Non-stationary: branching ratio {self.branching_ratio:.3f} >= 1"
            )
        return True


class HawkesArrivalModel:
    """
    Hawkes self-exciting point process for user arrivals.
    
    Captures burstiness and clustering in arrivals that NHPP cannot model,
    particularly useful during peak hours or promotional events.
    """
    
    def __init__(
        self,
        service_type: ServiceType,
        params: Optional[HawkesParameters] = None
    ):
        """
        Initialize Hawkes arrival model.
        
        Args:
            service_type: URLLC or eMBB
            params: Hawkes parameters (uses defaults if None)
        """
        self.service_type = service_type
        self.params = params or self._default_params()
        self.params.validate()
        
        # Event history within current billing cycle
        self._event_times: List[float] = []
        self._current_time: float = 0.0
        
        # Statistics tracking
        self._total_arrivals: int = 0
        self._intensity_history: List[float] = []
    
    def _default_params(self) -> HawkesParameters:
        """Get default parameters based on service type."""
        if self.service_type == ServiceType.URLLC:
            return HawkesParameters(
                mu_0=0.15,           # Lower baseline for B2B
                alpha=0.2,           # Moderate clustering
                beta=3.0,            # Fast decay (20 min half-life)
                price_elasticity_fee=0.3,
                price_elasticity_overage=0.15,
                qos_sensitivity=0.15,
                time_profile=self._business_hours_profile()
            )
        else:  # eMBB
            return HawkesParameters(
                mu_0=0.8,            # Higher baseline for B2C
                alpha=0.4,           # Stronger clustering (social effects)
                beta=2.0,            # Slower decay (30 min half-life)
                price_elasticity_fee=1.0,
                price_elasticity_overage=0.8,
                qos_sensitivity=0.08,
                time_profile=self._consumer_hours_profile()
            )
    
    def _business_hours_profile(self) -> np.ndarray:
        """Generate business hours time profile (9-18)."""
        profile = np.zeros(24)
        for h in range(24):
            if 9 <= h < 18:
                profile[h] = 1.2
            elif 8 <= h < 9 or 18 <= h < 19:
                profile[h] = 0.8
            else:
                profile[h] = 0.3
        return profile / profile.mean()  # Normalize
    
    def _consumer_hours_profile(self) -> np.ndarray:
        """Generate consumer hours time profile (evening peak)."""
        profile = np.zeros(24)
        for h in range(24):
            if 19 <= h < 23:
                profile[h] = 1.8
            elif 12 <= h < 14:
                profile[h] = 1.3
            elif 7 <= h < 12 or 14 <= h < 19:
                profile[h] = 1.0
            elif 23 <= h or h < 2:
                profile[h] = 0.7
            else:
                profile[h] = 0.3
        return profile / profile.mean()  # Normalize
    
    def _baseline_intensity(
        self,
        hour_of_day: int,
        fee_factor: float = 1.0,
        overage_factor: float = 1.0,
        violation_rate: float = 0.0
    ) -> float:
        """
        Compute baseline intensity μ(t) including price and QoS effects.
        
        μ(t) = μ₀ × T(h) × P_fee × P_overage × Q
        """
        p = self.params
        
        # Time profile
        T_h = p.time_profile[hour_of_day % 24]
        
        # Price effects (demand decreases with higher prices)
        P_fee = (p.ref_fee_factor / max(fee_factor, 0.1)) ** p.price_elasticity_fee
        P_overage = (p.ref_overage_factor / max(overage_factor, 0.1)) ** p.price_elasticity_overage
        
        # QoS effect (demand decreases with violations)
        Q = np.exp(-p.qos_sensitivity * violation_rate)
        
        return p.mu_0 * T_h * P_fee * P_overage * Q
    
    def _excitation_kernel(self, dt: float) -> float:
        """
        Compute excitation kernel value: α * exp(-β * dt).
        
        Args:
            dt: Time since event (hours)
        
        Returns:
            Kernel value
        """
        if dt < 0:
            return 0.0
        return self.params.alpha * np.exp(-self.params.beta * dt)
    
    def _compute_intensity(
        self,
        t: float,
        hour_of_day: int,
        fee_factor: float = 1.0,
        overage_factor: float = 1.0,
        violation_rate: float = 0.0
    ) -> float:
        """
        Compute intensity at time t.
        
        λ(t) = μ(t) + Σ_{t_i < t} α * exp(-β * (t - t_i))
        """
        # Baseline intensity
        mu_t = self._baseline_intensity(
            hour_of_day, fee_factor, overage_factor, violation_rate
        )
        
        # Excitation from past events
        excitation = sum(
            self._excitation_kernel(t - t_i)
            for t_i in self._event_times
            if t_i < t
        )
        
        return mu_t + excitation
    
    def generate_arrivals(
        self,
        current_hour: int,
        fee_factor: float = 1.0,
        overage_factor: float = 1.0,
        violation_rate: float = 0.0,
        dt: float = 1.0
    ) -> int:
        """
        Generate arrivals for the next time step using thinning algorithm.
        
        Ogata's modified thinning algorithm for Hawkes process.
        
        Args:
            current_hour: Current simulation hour
            fee_factor: Price factor for access fee [0.8, 1.2]
            overage_factor: Price factor for overage rate [0.8, 1.2]
            violation_rate: Recent QoS violation rate [0, 1]
            dt: Time step duration in hours
        
        Returns:
            Number of new arrivals
        """
        hour_of_day = current_hour % 24
        arrivals = []
        
        t = 0.0  # Relative time within step
        
        # Upper bound on intensity (conservative)
        lambda_star = self._compute_intensity(
            self._current_time + t, hour_of_day,
            fee_factor, overage_factor, violation_rate
        ) * 2.0 + 1.0
        
        while t < dt:
            # Generate next candidate time
            u = np.random.random()
            w = -np.log(u) / lambda_star
            t_candidate = t + w
            
            if t_candidate >= dt:
                break
            
            # Compute actual intensity at candidate time
            lambda_t = self._compute_intensity(
                self._current_time + t_candidate, hour_of_day,
                fee_factor, overage_factor, violation_rate
            )
            
            # Accept/reject
            if np.random.random() < lambda_t / lambda_star:
                # Accept arrival
                event_time = self._current_time + t_candidate
                arrivals.append(event_time)
                self._event_times.append(event_time)
                
                # Update upper bound (intensity jumps)
                lambda_star = lambda_t + self.params.alpha
            
            t = t_candidate
        
        # Update time
        self._current_time += dt
        
        # Prune old events (beyond memory horizon)
        memory_horizon = 5.0 / self.params.beta  # ~5 time constants
        self._event_times = [
            t_i for t_i in self._event_times
            if self._current_time - t_i < memory_horizon
        ]
        
        # Track statistics
        self._total_arrivals += len(arrivals)
        current_intensity = self._compute_intensity(
            self._current_time, hour_of_day,
            fee_factor, overage_factor, violation_rate
        )
        self._intensity_history.append(current_intensity)
        
        return len(arrivals)
    
    def reset(self):
        """Reset model state for new episode."""
        self._event_times = []
        self._current_time = 0.0
        self._total_arrivals = 0
        self._intensity_history = []
    
    def get_current_intensity(
        self,
        hour_of_day: int,
        fee_factor: float = 1.0,
        overage_factor: float = 1.0,
        violation_rate: float = 0.0
    ) -> float:
        """Get current intensity for observation state."""
        return self._compute_intensity(
            self._current_time, hour_of_day,
            fee_factor, overage_factor, violation_rate
        )
    
    def get_statistics(self) -> Dict:
        """Get model statistics."""
        return {
            'total_arrivals': self._total_arrivals,
            'current_time': self._current_time,
            'active_events': len(self._event_times),
            'avg_intensity': np.mean(self._intensity_history) if self._intensity_history else 0,
            'branching_ratio': self.params.branching_ratio,
        }


class HawkesCalibrator:
    """
    Calibrate Hawkes process parameters from arrival data.
    
    Uses maximum likelihood estimation (MLE) with the log-likelihood:
    
    L(θ) = Σ log(λ(t_i)) - ∫_0^T λ(t) dt
    """
    
    def __init__(self, service_type: ServiceType):
        """Initialize calibrator for service type."""
        self.service_type = service_type
        self._fitted_params: Optional[HawkesParameters] = None
    
    def fit(
        self,
        event_times: np.ndarray,
        T_max: float,
        hour_sequence: Optional[np.ndarray] = None,
        initial_params: Optional[HawkesParameters] = None
    ) -> HawkesParameters:
        """
        Fit Hawkes parameters using MLE.
        
        Args:
            event_times: Array of arrival times
            T_max: Observation period length
            hour_sequence: Hour of day for each event (optional)
            initial_params: Starting point for optimization
        
        Returns:
            Fitted HawkesParameters
        """
        if len(event_times) < 10:
            print("Warning: Few events, using default parameters")
            return HawkesParameters()
        
        # Sort events
        events = np.sort(event_times)
        
        # Initial values
        if initial_params:
            x0 = [initial_params.mu_0, initial_params.alpha, initial_params.beta]
        else:
            # Heuristic initialization
            avg_rate = len(events) / T_max
            x0 = [avg_rate * 0.7, 0.3, 2.0]
        
        def neg_log_likelihood(params):
            mu, alpha, beta = params
            
            if mu <= 0 or alpha < 0 or beta <= 0:
                return 1e10
            if alpha / beta >= 0.99:  # Stationarity
                return 1e10
            
            # Intensity at each event
            log_lambdas = []
            for i, t_i in enumerate(events):
                # Baseline
                lambda_i = mu
                
                # Excitation from past events
                for j in range(i):
                    lambda_i += alpha * np.exp(-beta * (t_i - events[j]))
                
                if lambda_i > 0:
                    log_lambdas.append(np.log(lambda_i))
                else:
                    return 1e10
            
            # Integral term: ∫ λ(t) dt = μT + (α/β) Σ (1 - exp(-β(T-t_i)))
            integral = mu * T_max
            for t_i in events:
                integral += (alpha / beta) * (1 - np.exp(-beta * (T_max - t_i)))
            
            return integral - sum(log_lambdas)
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[(1e-6, None), (0, None), (1e-6, None)]
        )
        
        if result.success:
            mu_hat, alpha_hat, beta_hat = result.x
            
            self._fitted_params = HawkesParameters(
                mu_0=mu_hat,
                alpha=alpha_hat,
                beta=beta_hat
            )
            
            print(f"Fitted Hawkes parameters:")
            print(f"  μ₀ = {mu_hat:.4f}")
            print(f"  α = {alpha_hat:.4f}")
            print(f"  β = {beta_hat:.4f}")
            print(f"  Branching ratio = {alpha_hat/beta_hat:.4f}")
            
            return self._fitted_params
        else:
            print(f"Optimization failed: {result.message}")
            return HawkesParameters()
    
    def goodness_of_fit(
        self,
        event_times: np.ndarray,
        T_max: float
    ) -> Dict:
        """
        Assess goodness of fit using residual analysis.
        
        Transforms event times using compensator (time-rescaling theorem).
        If model is correct, transformed times should be Poisson(1).
        """
        if self._fitted_params is None:
            raise ValueError("Must fit model first")
        
        events = np.sort(event_times)
        p = self._fitted_params
        
        # Compute compensator (integrated intensity)
        compensators = []
        for i, t_i in enumerate(events):
            # Λ(t_i) = ∫_0^{t_i} λ(s) ds
            Lambda = p.mu_0 * t_i
            for j in range(i):
                Lambda += (p.alpha / p.beta) * (
                    1 - np.exp(-p.beta * (t_i - events[j]))
                )
            compensators.append(Lambda)
        
        # Inter-arrival times in transformed time
        compensators = np.array(compensators)
        inter_arrivals = np.diff(compensators)
        
        # Should be Exp(1) if model is correct
        from scipy.stats import kstest, expon
        ks_stat, ks_pvalue = kstest(inter_arrivals, 'expon', args=(0, 1))
        
        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mean_inter_arrival': inter_arrivals.mean(),
            'std_inter_arrival': inter_arrivals.std(),
            'model_fit': 'good' if ks_pvalue > 0.05 else 'poor'
        }


def create_arrival_model(
    service_type: ServiceType,
    use_hawkes: bool = False,
    params: Optional[HawkesParameters] = None
):
    """
    Factory function to create arrival model.
    
    Args:
        service_type: URLLC or eMBB
        use_hawkes: If True, use Hawkes; else use NHPP
        params: Optional parameters
    
    Returns:
        Arrival model (HawkesArrivalModel or NHPPArrivalModel)
    """
    if use_hawkes:
        return HawkesArrivalModel(service_type, params)
    else:
        # Import NHPP model
        from models.arrivals_nhpp import NHPPArrivalModel
        return NHPPArrivalModel(service_type)


if __name__ == "__main__":
    # Demonstration
    print("=" * 60)
    print("Hawkes Process Arrival Model Demonstration")
    print("=" * 60)
    
    # Create models for both service types
    urllc_model = HawkesArrivalModel(ServiceType.URLLC)
    embb_model = HawkesArrivalModel(ServiceType.eMBB)
    
    print(f"\nURLLC Hawkes Parameters:")
    print(f"  μ₀ = {urllc_model.params.mu_0}")
    print(f"  α = {urllc_model.params.alpha}")
    print(f"  β = {urllc_model.params.beta}")
    print(f"  Branching ratio = {urllc_model.params.branching_ratio:.3f}")
    print(f"  Stationary: {urllc_model.params.is_stationary}")
    
    print(f"\neMBB Hawkes Parameters:")
    print(f"  μ₀ = {embb_model.params.mu_0}")
    print(f"  α = {embb_model.params.alpha}")
    print(f"  β = {embb_model.params.beta}")
    print(f"  Branching ratio = {embb_model.params.branching_ratio:.3f}")
    print(f"  Stationary: {embb_model.params.is_stationary}")
    
    # Simulate one week
    print("\n" + "-" * 40)
    print("Simulating 168 hours (1 week)...")
    print("-" * 40)
    
    urllc_arrivals = []
    embb_arrivals = []
    
    for hour in range(168):
        # Simulate with varying price factors
        fee_factor = 1.0 + 0.1 * np.sin(2 * np.pi * hour / 24)
        
        u_arr = urllc_model.generate_arrivals(
            hour, fee_factor=fee_factor, violation_rate=0.02
        )
        e_arr = embb_model.generate_arrivals(
            hour, fee_factor=fee_factor, violation_rate=0.05
        )
        
        urllc_arrivals.append(u_arr)
        embb_arrivals.append(e_arr)
    
    print(f"\nURLLC Statistics:")
    stats = urllc_model.get_statistics()
    print(f"  Total arrivals: {stats['total_arrivals']}")
    print(f"  Average intensity: {stats['avg_intensity']:.3f}")
    print(f"  Active events in memory: {stats['active_events']}")
    
    print(f"\neMBB Statistics:")
    stats = embb_model.get_statistics()
    print(f"  Total arrivals: {stats['total_arrivals']}")
    print(f"  Average intensity: {stats['avg_intensity']:.3f}")
    print(f"  Active events in memory: {stats['active_events']}")
    
    # Demonstrate calibration
    print("\n" + "-" * 40)
    print("Calibration Demonstration")
    print("-" * 40)
    
    # Generate synthetic data
    np.random.seed(42)
    true_params = HawkesParameters(mu_0=0.5, alpha=0.3, beta=2.0)
    
    test_model = HawkesArrivalModel(ServiceType.eMBB, true_params)
    
    all_events = []
    t = 0.0
    for hour in range(168):
        n = test_model.generate_arrivals(hour)
        for _ in range(n):
            all_events.append(t + np.random.random())
        t += 1.0
    
    print(f"\nGenerated {len(all_events)} events")
    print(f"True parameters: μ₀={true_params.mu_0}, α={true_params.alpha}, β={true_params.beta}")
    
    # Fit model
    calibrator = HawkesCalibrator(ServiceType.eMBB)
    fitted = calibrator.fit(np.array(all_events), T_max=168.0)
    
    # Goodness of fit
    gof = calibrator.goodness_of_fit(np.array(all_events), T_max=168.0)
    print(f"\nGoodness of fit:")
    print(f"  KS statistic: {gof['ks_statistic']:.4f}")
    print(f"  KS p-value: {gof['ks_pvalue']:.4f}")
    print(f"  Model fit: {gof['model_fit']}")
    
    print("\n" + "=" * 60)
    print("Hawkes Process Model Ready")
    print("=" * 60)
