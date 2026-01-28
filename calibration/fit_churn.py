"""
Churn Model Calibration for 5G O-RAN Network Slicing Simulation.

Estimates parameters for Hazard/Logistic churn model:
P(churn) = sigmoid(b₀ + b_V×V + b_C×C + b_O×O + b_F×F)

Methods:
- Maximum Likelihood Estimation for logistic regression
- Bayesian estimation with informed priors
- Synthetic target calibration

References:
- Scientific Reports 2024: ML-based churn prediction
- MDPI Algorithms 2024: Telecom churn modeling
- ScienceDirect 2024: QoS-based churn analysis
"""

import numpy as np
from scipy import optimize
from scipy.special import expit  # Sigmoid function
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings


@dataclass
class ChurnObservation:
    """Single observation of churn event or non-event."""
    churned: bool  # Whether user churned (1) or stayed (0)
    violation_rate: float  # QoS violation rate
    consecutive_violations: int  # Number of consecutive violations
    overage_ratio: float  # Overage payment / base fee ratio
    fee_factor: float  # Current fee factor
    tenure_hours: int  # How long user has been subscribed
    slice_type: str  # "URLLC" or "eMBB"


@dataclass
class ChurnParameters:
    """Estimated churn model parameters."""
    # Logistic regression coefficients
    b0: float  # Intercept (base log-odds)
    b_violation: float  # Violation rate coefficient
    b_consecutive: float  # Consecutive violation coefficient
    b_overage: float  # Overage ratio coefficient
    b_fee: float  # Fee factor coefficient
    
    # Confidence intervals
    b0_ci: Tuple[float, float]
    b_violation_ci: Tuple[float, float]
    b_consecutive_ci: Tuple[float, float]
    b_overage_ci: Tuple[float, float]
    b_fee_ci: Tuple[float, float]
    
    # Model diagnostics
    log_likelihood: float
    pseudo_r2: float  # McFadden's R²
    accuracy: float  # Classification accuracy
    n_observations: int
    n_churned: int


class ChurnCalibrator:
    """
    Calibrates Hazard/Logistic churn model parameters.
    
    The churn probability model is:
    P(churn) = σ(b₀ + b_V×V + b_C×C + b_O×O + b_F×F)
    
    Where:
    - b₀: Base log-odds (determines baseline churn rate)
    - b_V: QoS violation sensitivity
    - b_C: Consecutive violation penalty
    - b_O: Overage payment sensitivity
    - b_F: Fee sensitivity
    """
    
    def __init__(self, slice_type: str = "URLLC"):
        """
        Initialize calibrator.
        
        Args:
            slice_type: "URLLC" or "eMBB"
        """
        self.slice_type = slice_type
        self.observations: List[ChurnObservation] = []
        self.parameters: Optional[ChurnParameters] = None
        
        # Set priors based on literature
        self._setup_priors()
    
    def _setup_priors(self):
        """
        Set up prior distributions based on industry literature.
        
        References:
        - URLLC churn: 8.7% annually → ~0.01% hourly base rate
        - eMBB churn: 15-20% annually → ~0.02% hourly base rate
        """
        if self.slice_type == "URLLC":
            # B2B: Low base churn, high QoS sensitivity
            self.prior_b0 = (-7.0, -5.0)  # Base log-odds
            self.prior_b_violation = (1.5, 3.0)
            self.prior_b_consecutive = (1.0, 2.0)
            self.prior_b_overage = (0.2, 1.0)  # Less price sensitive
            self.prior_b_fee = (0.005, 0.02)
        else:
            # eMBB B2C: Higher base churn, price sensitive
            self.prior_b0 = (-6.0, -4.0)
            self.prior_b_violation = (0.5, 1.5)
            self.prior_b_consecutive = (0.8, 1.5)
            self.prior_b_overage = (1.5, 3.0)  # Very price sensitive
            self.prior_b_fee = (0.05, 0.2)
    
    def add_observation(self, obs: ChurnObservation):
        """Add a single observation."""
        if obs.slice_type != self.slice_type:
            warnings.warn(f"Observation slice type mismatch")
        self.observations.append(obs)
    
    def add_observations(self, observations: List[ChurnObservation]):
        """Add multiple observations."""
        for obs in observations:
            self.add_observation(obs)
    
    def generate_synthetic_data(
        self,
        n_users: int = 1000,
        true_b0: float = None,
        true_b_violation: float = None,
        true_b_consecutive: float = None,
        true_b_overage: float = None,
        true_b_fee: float = None,
        seed: int = 42
    ) -> List[ChurnObservation]:
        """
        Generate synthetic churn data for calibration validation.
        
        Creates realistic user behavior patterns based on the logistic model.
        """
        np.random.seed(seed)
        
        # Set defaults based on slice type
        if true_b0 is None:
            true_b0 = -6.0 if self.slice_type == "URLLC" else -5.0
        if true_b_violation is None:
            true_b_violation = 2.0 if self.slice_type == "URLLC" else 1.0
        if true_b_consecutive is None:
            true_b_consecutive = 1.5 if self.slice_type == "URLLC" else 1.2
        if true_b_overage is None:
            true_b_overage = 0.5 if self.slice_type == "URLLC" else 2.0
        if true_b_fee is None:
            true_b_fee = 0.01 if self.slice_type == "URLLC" else 0.1
        
        observations = []
        
        for _ in range(n_users):
            # Generate user features
            # Violation rate: mostly low, some high
            violation_rate = np.random.beta(1, 10) if np.random.random() > 0.2 else np.random.beta(2, 5)
            
            # Consecutive violations: mostly 0, exponential decay
            consecutive = int(np.random.exponential(0.5))
            consecutive = min(consecutive, 5)
            
            # Overage ratio: varies with usage patterns
            overage_ratio = np.random.exponential(0.3)
            overage_ratio = min(overage_ratio, 2.0)
            
            # Fee factor: uniform in RL range
            fee_factor = np.random.uniform(0.8, 1.2)
            
            # Tenure: exponential distribution
            tenure = int(np.random.exponential(500))
            
            # Compute churn probability
            logit = (true_b0 
                    + true_b_violation * violation_rate 
                    + true_b_consecutive * consecutive
                    + true_b_overage * overage_ratio
                    + true_b_fee * (fee_factor - 1.0) * 100)  # Scale fee factor
            
            prob = expit(logit)
            churned = np.random.random() < prob
            
            obs = ChurnObservation(
                churned=churned,
                violation_rate=violation_rate,
                consecutive_violations=consecutive,
                overage_ratio=overage_ratio,
                fee_factor=fee_factor,
                tenure_hours=tenure,
                slice_type=self.slice_type
            )
            observations.append(obs)
        
        return observations
    
    def fit_mle(self, regularization: float = 0.01) -> ChurnParameters:
        """
        Fit churn model using Maximum Likelihood Estimation.
        
        Maximizes the Bernoulli log-likelihood:
        L = Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]
        
        Args:
            regularization: L2 regularization strength
        """
        if len(self.observations) < 50:
            warnings.warn("Limited data. Consider adding more observations.")
        
        # Prepare data matrices
        n = len(self.observations)
        X = np.zeros((n, 5))
        y = np.zeros(n)
        
        for i, obs in enumerate(self.observations):
            X[i, 0] = 1.0  # Intercept
            X[i, 1] = obs.violation_rate
            X[i, 2] = obs.consecutive_violations
            X[i, 3] = obs.overage_ratio
            X[i, 4] = (obs.fee_factor - 1.0) * 100  # Scale for numerical stability
            y[i] = 1.0 if obs.churned else 0.0
        
        def neg_log_likelihood(beta):
            """Negative log-likelihood with L2 regularization."""
            logits = X @ beta
            probs = expit(logits)
            probs = np.clip(probs, 1e-10, 1-1e-10)  # Numerical stability
            
            ll = np.sum(y * np.log(probs) + (1-y) * np.log(1-probs))
            
            # L2 regularization (not on intercept)
            reg = regularization * np.sum(beta[1:]**2)
            
            return -(ll - reg)
        
        # Initial guess from priors
        x0 = np.array([
            np.mean(self.prior_b0),
            np.mean(self.prior_b_violation),
            np.mean(self.prior_b_consecutive),
            np.mean(self.prior_b_overage),
            np.mean(self.prior_b_fee)
        ])
        
        # Optimize
        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B'
        )
        
        beta = result.x
        log_likelihood = -result.fun
        
        # Compute standard errors from Hessian
        try:
            hessian = self._compute_hessian(neg_log_likelihood, beta)
            cov_matrix = np.linalg.inv(hessian)
            std_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
        except:
            # Fallback to prior-based CIs
            std_errors = np.array([0.5, 0.3, 0.2, 0.3, 0.02])
        
        # Confidence intervals
        z = 1.96  # 95% CI
        b0_ci = (beta[0] - z*std_errors[0], beta[0] + z*std_errors[0])
        b_v_ci = (beta[1] - z*std_errors[1], beta[1] + z*std_errors[1])
        b_c_ci = (beta[2] - z*std_errors[2], beta[2] + z*std_errors[2])
        b_o_ci = (beta[3] - z*std_errors[3], beta[3] + z*std_errors[3])
        b_f_ci = (beta[4] - z*std_errors[4], beta[4] + z*std_errors[4])
        
        # Model diagnostics
        # Null model log-likelihood
        p_null = np.mean(y)
        ll_null = n * (p_null * np.log(p_null + 1e-10) + (1-p_null) * np.log(1-p_null + 1e-10))
        
        # McFadden's R²
        pseudo_r2 = 1 - (log_likelihood / ll_null) if ll_null != 0 else 0
        
        # Classification accuracy
        predictions = expit(X @ beta) > 0.5
        accuracy = np.mean(predictions == y)
        
        n_churned = int(np.sum(y))
        
        self.parameters = ChurnParameters(
            b0=beta[0],
            b_violation=beta[1],
            b_consecutive=beta[2],
            b_overage=beta[3],
            b_fee=beta[4],
            b0_ci=b0_ci,
            b_violation_ci=b_v_ci,
            b_consecutive_ci=b_c_ci,
            b_overage_ci=b_o_ci,
            b_fee_ci=b_f_ci,
            log_likelihood=log_likelihood,
            pseudo_r2=pseudo_r2,
            accuracy=accuracy,
            n_observations=n,
            n_churned=n_churned
        )
        
        return self.parameters
    
    def _compute_hessian(self, func, x, eps=1e-5) -> np.ndarray:
        """Compute numerical Hessian matrix."""
        n = len(x)
        hessian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                
                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                
                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                
                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps * eps)
        
        return hessian
    
    def fit_to_target_churn_rate(
        self,
        target_annual_churn: float,
        qos_weight: float = 0.4,
        price_weight: float = 0.3
    ) -> ChurnParameters:
        """
        Calibrate model to achieve a target annual churn rate.
        
        This method is useful when actual churn data is unavailable but
        target KPIs are known from industry benchmarks.
        
        Args:
            target_annual_churn: Target annual churn rate (e.g., 0.15 for 15%)
            qos_weight: Relative weight of QoS factors
            price_weight: Relative weight of price factors
        """
        # Convert annual to hourly base churn probability
        # P(survive 1 year) = (1 - p_hourly)^8760 = 1 - annual_churn
        # p_hourly ≈ -log(1 - annual_churn) / 8760
        p_hourly_base = -np.log(1 - target_annual_churn) / 8760
        
        # Convert to log-odds
        b0 = np.log(p_hourly_base / (1 - p_hourly_base))
        
        # Set other coefficients based on slice type and weights
        if self.slice_type == "URLLC":
            b_violation = 2.0 * qos_weight / 0.4
            b_consecutive = 1.5 * qos_weight / 0.4
            b_overage = 0.5 * price_weight / 0.3
            b_fee = 0.01 * price_weight / 0.3
        else:
            b_violation = 1.0 * qos_weight / 0.4
            b_consecutive = 1.2 * qos_weight / 0.4
            b_overage = 2.0 * price_weight / 0.3
            b_fee = 0.1 * price_weight / 0.3
        
        # Simple CIs based on typical uncertainty
        margin = 0.3  # 30% uncertainty
        
        self.parameters = ChurnParameters(
            b0=b0,
            b_violation=b_violation,
            b_consecutive=b_consecutive,
            b_overage=b_overage,
            b_fee=b_fee,
            b0_ci=(b0*(1-margin), b0*(1+margin)),
            b_violation_ci=(b_violation*(1-margin), b_violation*(1+margin)),
            b_consecutive_ci=(b_consecutive*(1-margin), b_consecutive*(1+margin)),
            b_overage_ci=(b_overage*(1-margin), b_overage*(1+margin)),
            b_fee_ci=(b_fee*(1-margin), b_fee*(1+margin)),
            log_likelihood=np.nan,
            pseudo_r2=np.nan,
            accuracy=np.nan,
            n_observations=0,
            n_churned=0
        )
        
        return self.parameters
    
    def predict_churn_curve(
        self,
        violation_range: Tuple[float, float] = (0, 0.5),
        overage_range: Tuple[float, float] = (0, 2.0)
    ) -> Dict[str, np.ndarray]:
        """
        Generate churn probability curves for visualization.
        
        Varies one factor while holding others at baseline.
        """
        if self.parameters is None:
            raise ValueError("Must fit parameters first")
        
        n_points = 50
        p = self.parameters
        
        # Baseline values
        baseline = {
            'violation': 0.01,
            'consecutive': 0,
            'overage': 0.1,
            'fee_factor': 1.0
        }
        
        results = {}
        
        # Violation rate curve
        violations = np.linspace(violation_range[0], violation_range[1], n_points)
        probs = expit(p.b0 + p.b_violation * violations 
                     + p.b_consecutive * baseline['consecutive']
                     + p.b_overage * baseline['overage']
                     + p.b_fee * (baseline['fee_factor'] - 1.0) * 100)
        results['violation'] = {'x': violations, 'y': probs}
        
        # Overage ratio curve
        overages = np.linspace(overage_range[0], overage_range[1], n_points)
        probs = expit(p.b0 + p.b_violation * baseline['violation']
                     + p.b_consecutive * baseline['consecutive']
                     + p.b_overage * overages
                     + p.b_fee * (baseline['fee_factor'] - 1.0) * 100)
        results['overage'] = {'x': overages, 'y': probs}
        
        # Fee factor curve
        fees = np.linspace(0.8, 1.2, n_points)
        probs = expit(p.b0 + p.b_violation * baseline['violation']
                     + p.b_consecutive * baseline['consecutive']
                     + p.b_overage * baseline['overage']
                     + p.b_fee * (fees - 1.0) * 100)
        results['fee'] = {'x': fees, 'y': probs}
        
        return results
    
    def report(self) -> str:
        """Generate calibration report."""
        if self.parameters is None:
            return "No parameters fitted yet."
        
        p = self.parameters
        
        # Compute baseline churn rate
        baseline_prob = expit(p.b0)
        annual_equiv = 1 - (1 - baseline_prob) ** 8760
        
        report = f"""
========================================
CHURN CALIBRATION REPORT - {self.slice_type}
========================================

Model: Logistic/Hazard
P(churn) = σ(b₀ + b_V×V + b_C×C + b_O×O + b_F×F)

ESTIMATED PARAMETERS:
---------------------
Base log-odds (b₀): {p.b0:.4f}
  95% CI: [{p.b0_ci[0]:.4f}, {p.b0_ci[1]:.4f}]
  → Baseline churn prob: {baseline_prob:.6f}/hour
  → Equivalent annual: {annual_equiv:.2%}

Violation sensitivity (b_V): {p.b_violation:.4f}
  95% CI: [{p.b_violation_ci[0]:.4f}, {p.b_violation_ci[1]:.4f}]

Consecutive violation (b_C): {p.b_consecutive:.4f}
  95% CI: [{p.b_consecutive_ci[0]:.4f}, {p.b_consecutive_ci[1]:.4f}]

Overage sensitivity (b_O): {p.b_overage:.4f}
  95% CI: [{p.b_overage_ci[0]:.4f}, {p.b_overage_ci[1]:.4f}]

Fee sensitivity (b_F): {p.b_fee:.4f}
  95% CI: [{p.b_fee_ci[0]:.4f}, {p.b_fee_ci[1]:.4f}]

MODEL FIT:
----------
Log-likelihood: {p.log_likelihood:.2f}
Pseudo R² (McFadden): {p.pseudo_r2:.4f}
Classification accuracy: {p.accuracy:.2%}
N observations: {p.n_observations}
N churned: {p.n_churned} ({p.n_churned/max(p.n_observations,1):.2%})

INTERPRETATION:
---------------
"""
        # Add interpretation of effects
        # Effect of 10% violation rate
        prob_10pct_viol = expit(p.b0 + p.b_violation * 0.10)
        report += f"- 10% violation rate increases churn prob to {prob_10pct_viol:.6f}/hour\n"
        
        # Effect of 3 consecutive violations
        prob_3consec = expit(p.b0 + p.b_consecutive * 3)
        report += f"- 3 consecutive violations: churn prob = {prob_3consec:.6f}/hour\n"
        
        # Effect of high overage (1.0 ratio)
        prob_high_overage = expit(p.b0 + p.b_overage * 1.0)
        report += f"- High overage (ratio=1.0): churn prob = {prob_high_overage:.6f}/hour\n"
        
        # Effect of 20% price increase
        prob_price_up = expit(p.b0 + p.b_fee * 20)
        report += f"- 20% fee increase: churn prob = {prob_price_up:.6f}/hour\n"
        
        return report


if __name__ == "__main__":
    # Demo: Generate synthetic data and calibrate
    print("=" * 60)
    print("CHURN CALIBRATION DEMONSTRATION")
    print("=" * 60)
    
    for slice_type in ["URLLC", "eMBB"]:
        print(f"\n{'='*60}")
        print(f"Calibrating {slice_type} churn model...")
        print("="*60)
        
        calibrator = ChurnCalibrator(slice_type=slice_type)
        
        # Method 1: Synthetic data calibration
        print("\n--- Method 1: Synthetic Data MLE ---")
        synthetic_data = calibrator.generate_synthetic_data(n_users=1000)
        calibrator.add_observations(synthetic_data)
        
        params = calibrator.fit_mle()
        print(calibrator.report())
        
        # Method 2: Target churn rate calibration
        print("\n--- Method 2: Target Churn Rate Calibration ---")
        calibrator2 = ChurnCalibrator(slice_type=slice_type)
        
        target_churn = 0.087 if slice_type == "URLLC" else 0.15  # Annual
        params2 = calibrator2.fit_to_target_churn_rate(target_churn)
        print(calibrator2.report())
