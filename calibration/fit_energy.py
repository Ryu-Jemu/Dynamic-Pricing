"""
Energy Model Calibration for 5G O-RAN Network Slicing Simulation.

Estimates parameters for gNB power consumption model:
P(t) = P_idle + (P_max - P_idle) × load^k

Methods:
- Least squares fitting from measurement data
- Literature-based parameter estimation
- Model validation and diagnostics

References:
- PMC 2021: 5G BS power consumption measurements
- MDPI ECO6G 2022: Energy efficiency in 5G
- Ericsson 2023: 5G network energy performance
"""

import numpy as np
from scipy import optimize
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings


@dataclass
class PowerMeasurement:
    """Single power consumption measurement."""
    load_fraction: float  # PRB utilization [0, 1]
    power_kw: float  # Measured power consumption (kW)
    temperature_c: float = 25.0  # Ambient temperature (optional)
    timestamp: float = 0.0  # Time of measurement


@dataclass
class EnergyParameters:
    """Estimated energy model parameters."""
    # Power model: P = P_idle + (P_max - P_idle) × load^k
    p_idle: float  # Idle power consumption (kW)
    p_max: float  # Maximum power at full load (kW)
    k: float  # Load exponent (non-linearity factor)
    
    # Confidence intervals
    p_idle_ci: Tuple[float, float]
    p_max_ci: Tuple[float, float]
    k_ci: Tuple[float, float]
    
    # Temperature coefficient (optional)
    temp_coef: float  # Power increase per °C above 25°C
    
    # Model diagnostics
    r_squared: float  # Coefficient of determination
    rmse: float  # Root mean squared error
    mae: float  # Mean absolute error
    n_measurements: int


class EnergyCalibrator:
    """
    Calibrates gNB power consumption model parameters.
    
    The power model is:
    P(load, T) = [P_idle + (P_max - P_idle) × load^k] × (1 + α × (T - 25))
    
    Where:
    - P_idle: Idle power (no traffic)
    - P_max: Maximum power at full load
    - k: Non-linearity exponent (k > 1 means super-linear growth)
    - α: Temperature coefficient
    """
    
    def __init__(self):
        """Initialize calibrator with literature-based priors."""
        self.measurements: List[PowerMeasurement] = []
        self.parameters: Optional[EnergyParameters] = None
        
        # Literature-based priors for 5G NR macro gNB
        # Reference: PMC 2021, MDPI ECO6G 2022
        self.prior_p_idle = (0.5, 1.5)  # kW
        self.prior_p_max = (3.0, 5.0)   # kW
        self.prior_k = (1.0, 2.0)        # Usually 1.3-1.7
        self.prior_temp_coef = (0.01, 0.03)  # 1-3% per °C
    
    def add_measurement(self, measurement: PowerMeasurement):
        """Add a single measurement."""
        if not 0 <= measurement.load_fraction <= 1:
            warnings.warn(f"Load fraction {measurement.load_fraction} outside [0,1]")
        self.measurements.append(measurement)
    
    def add_measurements(self, measurements: List[PowerMeasurement]):
        """Add multiple measurements."""
        for m in measurements:
            self.add_measurement(m)
    
    def generate_synthetic_data(
        self,
        n_measurements: int = 200,
        true_p_idle: float = 1.0,
        true_p_max: float = 4.0,
        true_k: float = 1.5,
        noise_std: float = 0.05,
        seed: int = 42
    ) -> List[PowerMeasurement]:
        """
        Generate synthetic power measurements for calibration validation.
        
        Args:
            n_measurements: Number of measurements to generate
            true_p_idle: True idle power (kW)
            true_p_max: True max power (kW)
            true_k: True load exponent
            noise_std: Measurement noise standard deviation
            seed: Random seed
        """
        np.random.seed(seed)
        
        measurements = []
        
        for i in range(n_measurements):
            # Generate load fraction (more samples at low and high loads)
            if np.random.random() < 0.3:
                load = np.random.uniform(0, 0.2)  # Low load
            elif np.random.random() < 0.5:
                load = np.random.uniform(0.8, 1.0)  # High load
            else:
                load = np.random.uniform(0, 1.0)  # Full range
            
            # Temperature (mostly around 25°C)
            temp = np.random.normal(25, 5)
            temp = np.clip(temp, 10, 40)
            
            # True power (no temperature effect for simplicity)
            true_power = true_p_idle + (true_p_max - true_p_idle) * (load ** true_k)
            
            # Add measurement noise
            measured_power = true_power + np.random.normal(0, noise_std)
            measured_power = max(measured_power, 0.1)  # Physical constraint
            
            measurements.append(PowerMeasurement(
                load_fraction=load,
                power_kw=measured_power,
                temperature_c=temp,
                timestamp=float(i)
            ))
        
        return measurements
    
    def fit_least_squares(self) -> EnergyParameters:
        """
        Fit power model using nonlinear least squares.
        
        Minimizes: Σ (P_measured - P_model)²
        """
        if len(self.measurements) < 10:
            warnings.warn("Limited data. Consider adding more measurements.")
        
        # Extract data
        loads = np.array([m.load_fraction for m in self.measurements])
        powers = np.array([m.power_kw for m in self.measurements])
        temps = np.array([m.temperature_c for m in self.measurements])
        
        def power_model(params, load, temp=25.0):
            """Power consumption model."""
            p_idle, p_max, k = params[:3]
            temp_coef = params[3] if len(params) > 3 else 0.0
            
            base_power = p_idle + (p_max - p_idle) * (load ** k)
            temp_factor = 1 + temp_coef * (temp - 25.0)
            
            return base_power * temp_factor
        
        def residuals(params):
            """Residual function for optimization."""
            predicted = power_model(params, loads, temps)
            return powers - predicted
        
        # Initial guess from priors
        x0 = np.array([
            np.mean(self.prior_p_idle),
            np.mean(self.prior_p_max),
            np.mean(self.prior_k),
            0.02  # Initial temp coef
        ])
        
        # Bounds
        bounds = (
            [0.1, 1.0, 0.5, 0.0],    # Lower bounds
            [3.0, 10.0, 3.0, 0.1]    # Upper bounds
        )
        
        # Fit using least squares
        result = optimize.least_squares(
            residuals,
            x0,
            bounds=bounds,
            method='trf'  # Trust Region Reflective
        )
        
        params = result.x
        p_idle, p_max, k, temp_coef = params
        
        # Compute predictions and residuals
        predictions = power_model(params, loads, temps)
        residuals_final = powers - predictions
        
        # Model diagnostics
        ss_res = np.sum(residuals_final ** 2)
        ss_tot = np.sum((powers - np.mean(powers)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        rmse = np.sqrt(np.mean(residuals_final ** 2))
        mae = np.mean(np.abs(residuals_final))
        
        # Estimate confidence intervals using Jacobian
        try:
            # Approximate covariance from Jacobian
            J = result.jac
            sigma_sq = ss_res / (len(loads) - 4)  # Residual variance
            cov = sigma_sq * np.linalg.inv(J.T @ J)
            std_errors = np.sqrt(np.diag(cov))
            
            z = 1.96
            p_idle_ci = (p_idle - z*std_errors[0], p_idle + z*std_errors[0])
            p_max_ci = (p_max - z*std_errors[1], p_max + z*std_errors[1])
            k_ci = (k - z*std_errors[2], k + z*std_errors[2])
        except:
            # Fallback to prior-based CIs
            p_idle_ci = self.prior_p_idle
            p_max_ci = self.prior_p_max
            k_ci = self.prior_k
        
        self.parameters = EnergyParameters(
            p_idle=p_idle,
            p_max=p_max,
            k=k,
            p_idle_ci=p_idle_ci,
            p_max_ci=p_max_ci,
            k_ci=k_ci,
            temp_coef=temp_coef,
            r_squared=r_squared,
            rmse=rmse,
            mae=mae,
            n_measurements=len(self.measurements)
        )
        
        return self.parameters
    
    def fit_from_literature(
        self,
        bs_type: str = "macro_5g",
        mimo_config: str = "4T4R"
    ) -> EnergyParameters:
        """
        Set parameters based on published literature values.
        
        This is useful when actual measurements are unavailable.
        
        References:
        - PMC 2021: "Energy Consumption of 5G BS"
        - MDPI ECO6G 2022: "5G Energy Efficiency"
        - Ericsson 2023: "5G Network Energy"
        
        Args:
            bs_type: Type of base station ("macro_5g", "micro_5g", "small_cell")
            mimo_config: MIMO configuration ("4T4R", "8T8R", "32T32R", "64T64R")
        """
        # Literature-based parameters
        params_table = {
            "macro_5g": {
                "4T4R": (1.0, 4.0, 1.5),    # (P_idle, P_max, k)
                "8T8R": (1.2, 5.0, 1.5),
                "32T32R": (1.5, 7.0, 1.4),
                "64T64R": (2.0, 10.0, 1.3),
            },
            "micro_5g": {
                "4T4R": (0.3, 1.5, 1.6),
                "8T8R": (0.4, 2.0, 1.5),
            },
            "small_cell": {
                "4T4R": (0.1, 0.5, 1.7),
            }
        }
        
        if bs_type not in params_table:
            warnings.warn(f"Unknown BS type {bs_type}, using macro_5g")
            bs_type = "macro_5g"
        
        if mimo_config not in params_table[bs_type]:
            warnings.warn(f"Unknown MIMO config {mimo_config}, using 4T4R")
            mimo_config = "4T4R"
        
        p_idle, p_max, k = params_table[bs_type][mimo_config]
        
        # Typical uncertainty: ±20%
        margin = 0.2
        
        self.parameters = EnergyParameters(
            p_idle=p_idle,
            p_max=p_max,
            k=k,
            p_idle_ci=(p_idle*(1-margin), p_idle*(1+margin)),
            p_max_ci=(p_max*(1-margin), p_max*(1+margin)),
            k_ci=(k*(1-margin/2), k*(1+margin/2)),
            temp_coef=0.02,  # 2% per °C typical
            r_squared=np.nan,
            rmse=np.nan,
            mae=np.nan,
            n_measurements=0
        )
        
        return self.parameters
    
    def compute_power_curve(
        self,
        load_range: Tuple[float, float] = (0, 1),
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute power consumption curve across load range.
        
        Returns:
            Dictionary with 'load', 'power', 'power_lower', 'power_upper'
        """
        if self.parameters is None:
            raise ValueError("Must fit parameters first")
        
        p = self.parameters
        loads = np.linspace(load_range[0], load_range[1], n_points)
        
        # Central estimate
        power = p.p_idle + (p.p_max - p.p_idle) * (loads ** p.k)
        
        # Lower bound (using lower CI values)
        power_lower = p.p_idle_ci[0] + (p.p_max_ci[0] - p.p_idle_ci[0]) * (loads ** p.k_ci[1])
        
        # Upper bound
        power_upper = p.p_idle_ci[1] + (p.p_max_ci[1] - p.p_idle_ci[1]) * (loads ** p.k_ci[0])
        
        return {
            'load': loads,
            'power': power,
            'power_lower': power_lower,
            'power_upper': power_upper
        }
    
    def compute_efficiency_curve(self) -> Dict[str, np.ndarray]:
        """
        Compute energy efficiency (bits per Joule) vs load.
        
        Assumes throughput scales linearly with load.
        """
        if self.parameters is None:
            raise ValueError("Must fit parameters first")
        
        # Throughput scaling
        # At full load: assume 1 Gbps for 5G macro
        max_throughput_gbps = 1.0
        
        curve = self.compute_power_curve()
        loads = curve['load']
        power = curve['power']
        
        # Throughput (Gbps)
        throughput = max_throughput_gbps * loads
        
        # Energy efficiency (Gbps/kW = bits per Joule × 10^6)
        efficiency = np.zeros_like(loads)
        nonzero = power > 0
        efficiency[nonzero] = throughput[nonzero] / power[nonzero]
        
        # Marginal efficiency (derivative)
        delta_throughput = np.diff(throughput)
        delta_power = np.diff(power)
        marginal = np.zeros_like(loads)
        marginal[1:] = delta_throughput / np.maximum(delta_power, 1e-6)
        
        return {
            'load': loads,
            'efficiency': efficiency,
            'marginal_efficiency': marginal
        }
    
    def report(self) -> str:
        """Generate calibration report."""
        if self.parameters is None:
            return "No parameters fitted yet."
        
        p = self.parameters
        
        # Compute key metrics
        power_at_50pct = p.p_idle + (p.p_max - p.p_idle) * (0.5 ** p.k)
        power_range = p.p_max - p.p_idle
        
        report = f"""
========================================
ENERGY MODEL CALIBRATION REPORT
========================================

Model: P(load) = P_idle + (P_max - P_idle) × load^k

ESTIMATED PARAMETERS:
---------------------
Idle power (P_idle): {p.p_idle:.3f} kW
  95% CI: [{p.p_idle_ci[0]:.3f}, {p.p_idle_ci[1]:.3f}]

Max power (P_max): {p.p_max:.3f} kW
  95% CI: [{p.p_max_ci[0]:.3f}, {p.p_max_ci[1]:.3f}]

Load exponent (k): {p.k:.3f}
  95% CI: [{p.k_ci[0]:.3f}, {p.k_ci[1]:.3f}]

Temperature coefficient: {p.temp_coef:.4f} /°C
  (Power increases {p.temp_coef*100:.1f}% per °C above 25°C)

MODEL FIT:
----------
R² (coefficient of determination): {p.r_squared:.4f}
RMSE: {p.rmse:.4f} kW
MAE: {p.mae:.4f} kW
N measurements: {p.n_measurements}

POWER CONSUMPTION PROFILE:
--------------------------
"""
        # Power at different load levels
        for load in [0, 0.25, 0.5, 0.75, 1.0]:
            power = p.p_idle + (p.p_max - p.p_idle) * (load ** p.k)
            bar_len = int(power / p.p_max * 30)
            bar = '█' * bar_len
            report += f"  Load {load*100:3.0f}%: {power:.2f} kW  {bar}\n"
        
        report += f"""
KEY METRICS:
------------
- Dynamic range: {power_range:.2f} kW ({power_range/p.p_idle*100:.0f}% of idle)
- Power at 50% load: {power_at_50pct:.2f} kW
- Non-linearity: {'Super-linear (k>1)' if p.k > 1 else 'Sub-linear (k<1)' if p.k < 1 else 'Linear'}

COST IMPLICATIONS (at $0.10/kWh):
---------------------------------
"""
        # Hourly costs at different loads
        for load in [0.25, 0.5, 0.75, 1.0]:
            power = p.p_idle + (p.p_max - p.p_idle) * (load ** p.k)
            cost_hourly = power * 0.10
            cost_daily = cost_hourly * 24
            report += f"  Load {load*100:.0f}%: ${cost_hourly:.3f}/hour, ${cost_daily:.2f}/day\n"
        
        return report


if __name__ == "__main__":
    # Demo: Synthetic data calibration and literature-based
    print("=" * 60)
    print("ENERGY MODEL CALIBRATION DEMONSTRATION")
    print("=" * 60)
    
    # Method 1: Fit from synthetic measurements
    print("\n--- Method 1: Synthetic Measurement Data ---")
    calibrator = EnergyCalibrator()
    
    synthetic_data = calibrator.generate_synthetic_data(
        n_measurements=200,
        true_p_idle=1.0,
        true_p_max=4.0,
        true_k=1.5
    )
    calibrator.add_measurements(synthetic_data)
    
    params = calibrator.fit_least_squares()
    print(calibrator.report())
    
    # Method 2: Literature-based
    print("\n--- Method 2: Literature-Based Parameters ---")
    calibrator2 = EnergyCalibrator()
    
    for bs_type, mimo in [("macro_5g", "4T4R"), ("macro_5g", "64T64R"), ("micro_5g", "4T4R")]:
        print(f"\n{bs_type} with {mimo}:")
        params2 = calibrator2.fit_from_literature(bs_type=bs_type, mimo_config=mimo)
        print(f"  P_idle = {params2.p_idle:.2f} kW")
        print(f"  P_max  = {params2.p_max:.2f} kW")
        print(f"  k      = {params2.k:.2f}")
    
    # Power curve visualization data
    print("\n--- Power Curve Data ---")
    curve = calibrator.compute_power_curve()
    print("Load (%) | Power (kW)")
    print("-" * 25)
    for i in range(0, 101, 20):
        idx = i
        if idx < len(curve['load']):
            print(f"  {curve['load'][idx]*100:5.0f}   |   {curve['power'][idx]:.3f}")
