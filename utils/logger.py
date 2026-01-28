"""
Logging and Metrics Tracking Utilities.

Provides:
- SimulationLogger: Structured logging for training
- MetricsTracker: Real-time metrics collection and aggregation
- CSV logging for episode metrics
"""

import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from collections import deque
import numpy as np


@dataclass
class StepMetrics:
    """Metrics collected at each simulation step."""
    step: int
    hour: int
    
    # Financial
    revenue: float
    cost: float
    profit: float
    
    # Users
    urllc_users: int
    embb_users: int
    arrivals: int
    churns: int
    
    # QoS
    urllc_violations: int
    embb_violations: int
    urllc_violation_rate: float
    embb_violation_rate: float
    
    # Pricing
    urllc_fee_factor: float
    urllc_overage_factor: float
    embb_fee_factor: float
    embb_overage_factor: float
    
    # PRB utilization
    prb_utilization: float


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for a complete episode."""
    episode: int
    timesteps: int
    
    # Financial totals
    total_profit: float
    total_revenue: float
    total_cost: float
    avg_hourly_profit: float
    
    # QoS aggregates
    urllc_violation_rate: float
    embb_violation_rate: float
    urllc_violations_total: int
    embb_violations_total: int
    
    # User aggregates
    avg_urllc_users: float
    avg_embb_users: float
    total_arrivals: int
    total_churns: int
    churn_rate: float
    
    # Pricing averages
    avg_urllc_fee_factor: float
    avg_urllc_overage_factor: float
    avg_embb_fee_factor: float
    avg_embb_overage_factor: float
    
    # Constraint satisfaction
    urllc_constraint_satisfied: bool
    embb_constraint_satisfied: bool
    
    # Cost breakdown
    energy_cost: float = 0.0
    spectrum_cost: float = 0.0
    backhaul_cost: float = 0.0
    fixed_cost: float = 0.0
    
    # Lagrange multipliers (for CMDP)
    lambda_urllc: float = 0.0
    lambda_embb: float = 0.0


class MetricsTracker:
    """
    Tracks and aggregates simulation metrics.
    
    Provides real-time metrics collection, rolling statistics,
    and episode-level aggregation.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        urllc_threshold: float = 0.001,
        embb_threshold: float = 0.01
    ):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for statistics
            urllc_threshold: URLLC violation constraint threshold
            embb_threshold: eMBB violation constraint threshold
        """
        self.window_size = window_size
        self.urllc_threshold = urllc_threshold
        self.embb_threshold = embb_threshold
        
        # Step-level storage (current episode)
        self.current_episode_steps: List[StepMetrics] = []
        
        # Episode-level storage
        self.episodes: List[EpisodeMetrics] = []
        
        # Rolling statistics
        self.profit_history = deque(maxlen=window_size)
        self.urllc_violation_history = deque(maxlen=window_size)
        self.embb_violation_history = deque(maxlen=window_size)
        
        # Current episode counters
        self.episode_count = 0
        self.step_count = 0
        
        # Constraint tracking
        self.constraint_violations = {
            'urllc': 0,
            'embb': 0
        }
    
    def log_step(
        self,
        step: int,
        hour: int,
        revenue: float,
        cost: float,
        urllc_users: int,
        embb_users: int,
        arrivals: int,
        churns: int,
        urllc_violations: int,
        embb_violations: int,
        urllc_fee_factor: float,
        urllc_overage_factor: float,
        embb_fee_factor: float,
        embb_overage_factor: float,
        prb_utilization: float
    ) -> StepMetrics:
        """Log metrics for a single step."""
        profit = revenue - cost
        
        # Compute violation rates
        urllc_rate = urllc_violations / max(urllc_users, 1)
        embb_rate = embb_violations / max(embb_users, 1)
        
        metrics = StepMetrics(
            step=step,
            hour=hour,
            revenue=revenue,
            cost=cost,
            profit=profit,
            urllc_users=urllc_users,
            embb_users=embb_users,
            arrivals=arrivals,
            churns=churns,
            urllc_violations=urllc_violations,
            embb_violations=embb_violations,
            urllc_violation_rate=urllc_rate,
            embb_violation_rate=embb_rate,
            urllc_fee_factor=urllc_fee_factor,
            urllc_overage_factor=urllc_overage_factor,
            embb_fee_factor=embb_fee_factor,
            embb_overage_factor=embb_overage_factor,
            prb_utilization=prb_utilization
        )
        
        self.current_episode_steps.append(metrics)
        self.step_count += 1
        
        # Update rolling histories
        self.profit_history.append(profit)
        self.urllc_violation_history.append(urllc_rate)
        self.embb_violation_history.append(embb_rate)
        
        return metrics
    
    def end_episode(
        self,
        lambda_urllc: float = 0.0,
        lambda_embb: float = 0.0,
        cost_breakdown: Optional[Dict[str, float]] = None
    ) -> EpisodeMetrics:
        """
        Finalize episode and compute aggregated metrics.
        
        Args:
            lambda_urllc: Current URLLC Lagrange multiplier
            lambda_embb: Current eMBB Lagrange multiplier
            cost_breakdown: Optional cost component breakdown
        """
        steps = self.current_episode_steps
        
        if not steps:
            raise ValueError("No steps logged for this episode")
        
        n_steps = len(steps)
        
        # Financial aggregates
        total_revenue = sum(s.revenue for s in steps)
        total_cost = sum(s.cost for s in steps)
        total_profit = total_revenue - total_cost
        avg_hourly_profit = total_profit / n_steps
        
        # QoS aggregates
        total_urllc_violations = sum(s.urllc_violations for s in steps)
        total_embb_violations = sum(s.embb_violations for s in steps)
        total_urllc_users = sum(s.urllc_users for s in steps)
        total_embb_users = sum(s.embb_users for s in steps)
        
        urllc_violation_rate = total_urllc_violations / max(total_urllc_users, 1)
        embb_violation_rate = total_embb_violations / max(total_embb_users, 1)
        
        # User aggregates
        avg_urllc_users = np.mean([s.urllc_users for s in steps])
        avg_embb_users = np.mean([s.embb_users for s in steps])
        total_arrivals = sum(s.arrivals for s in steps)
        total_churns = sum(s.churns for s in steps)
        
        total_user_hours = total_urllc_users + total_embb_users
        churn_rate = total_churns / max(total_user_hours, 1)
        
        # Pricing averages
        avg_urllc_fee = np.mean([s.urllc_fee_factor for s in steps])
        avg_urllc_overage = np.mean([s.urllc_overage_factor for s in steps])
        avg_embb_fee = np.mean([s.embb_fee_factor for s in steps])
        avg_embb_overage = np.mean([s.embb_overage_factor for s in steps])
        
        # Constraint satisfaction
        urllc_satisfied = urllc_violation_rate <= self.urllc_threshold
        embb_satisfied = embb_violation_rate <= self.embb_threshold
        
        # Track constraint violations
        if not urllc_satisfied:
            self.constraint_violations['urllc'] += 1
        if not embb_satisfied:
            self.constraint_violations['embb'] += 1
        
        # Cost breakdown
        if cost_breakdown:
            energy_cost = cost_breakdown.get('energy', 0)
            spectrum_cost = cost_breakdown.get('spectrum', 0)
            backhaul_cost = cost_breakdown.get('backhaul', 0)
            fixed_cost = cost_breakdown.get('fixed', 0)
        else:
            energy_cost = spectrum_cost = backhaul_cost = fixed_cost = 0
        
        episode_metrics = EpisodeMetrics(
            episode=self.episode_count,
            timesteps=n_steps,
            total_profit=total_profit,
            total_revenue=total_revenue,
            total_cost=total_cost,
            avg_hourly_profit=avg_hourly_profit,
            urllc_violation_rate=urllc_violation_rate,
            embb_violation_rate=embb_violation_rate,
            urllc_violations_total=total_urllc_violations,
            embb_violations_total=total_embb_violations,
            avg_urllc_users=avg_urllc_users,
            avg_embb_users=avg_embb_users,
            total_arrivals=total_arrivals,
            total_churns=total_churns,
            churn_rate=churn_rate,
            avg_urllc_fee_factor=avg_urllc_fee,
            avg_urllc_overage_factor=avg_urllc_overage,
            avg_embb_fee_factor=avg_embb_fee,
            avg_embb_overage_factor=avg_embb_overage,
            urllc_constraint_satisfied=urllc_satisfied,
            embb_constraint_satisfied=embb_satisfied,
            energy_cost=energy_cost,
            spectrum_cost=spectrum_cost,
            backhaul_cost=backhaul_cost,
            fixed_cost=fixed_cost,
            lambda_urllc=lambda_urllc,
            lambda_embb=lambda_embb
        )
        
        self.episodes.append(episode_metrics)
        self.episode_count += 1
        
        # Reset step storage
        self.current_episode_steps = []
        
        return episode_metrics
    
    def get_rolling_stats(self) -> Dict[str, float]:
        """Get rolling statistics over recent history."""
        return {
            'rolling_profit_mean': np.mean(self.profit_history) if self.profit_history else 0,
            'rolling_profit_std': np.std(self.profit_history) if self.profit_history else 0,
            'rolling_urllc_violation': np.mean(self.urllc_violation_history) if self.urllc_violation_history else 0,
            'rolling_embb_violation': np.mean(self.embb_violation_history) if self.embb_violation_history else 0,
        }
    
    def get_constraint_satisfaction_rate(self) -> Dict[str, float]:
        """Get constraint satisfaction rates over all episodes."""
        if not self.episodes:
            return {'urllc': 1.0, 'embb': 1.0}
        
        n_episodes = len(self.episodes)
        return {
            'urllc': 1 - self.constraint_violations['urllc'] / n_episodes,
            'embb': 1 - self.constraint_violations['embb'] / n_episodes
        }


class SimulationLogger:
    """
    Structured logging for simulation training.
    
    Provides:
    - Console logging with configurable verbosity
    - CSV logging for metrics
    - JSON logging for configuration
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        console_level: int = logging.INFO,
        enable_csv: bool = True
    ):
        """
        Initialize simulation logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name for this experiment
            console_level: Logging level for console output
            enable_csv: Whether to enable CSV logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Create experiment subdirectory
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger(f"simulation.{experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.experiment_dir / "training.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # CSV logging
        self.enable_csv = enable_csv
        self.csv_file = None
        self.csv_writer = None
        if enable_csv:
            self._init_csv()
        
        self.logger.info(f"Simulation logger initialized: {experiment_name}")
    
    def _init_csv(self):
        """Initialize CSV file for metrics logging."""
        csv_path = self.experiment_dir / "episode_metrics.csv"
        self.csv_file = open(csv_path, 'w', newline='')
        
        # Define CSV columns
        columns = [
            'episode', 'timesteps', 'total_profit', 'total_revenue', 'total_cost',
            'avg_hourly_profit', 'urllc_violation_rate', 'embb_violation_rate',
            'urllc_violations_total', 'embb_violations_total',
            'avg_urllc_users', 'avg_embb_users', 'total_arrivals', 'total_churns',
            'churn_rate', 'avg_urllc_fee_factor', 'avg_urllc_overage_factor',
            'avg_embb_fee_factor', 'avg_embb_overage_factor',
            'urllc_constraint_satisfied', 'embb_constraint_satisfied',
            'energy_cost', 'spectrum_cost', 'backhaul_cost', 'fixed_cost',
            'lambda_urllc', 'lambda_embb'
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=columns)
        self.csv_writer.writeheader()
    
    def log_episode(self, metrics: EpisodeMetrics):
        """Log episode metrics to CSV and console."""
        # Console summary
        self.logger.info(
            f"Episode {metrics.episode}: "
            f"Profit=${metrics.total_profit:.0f}, "
            f"URLLC viol={metrics.urllc_violation_rate*100:.3f}%, "
            f"eMBB viol={metrics.embb_violation_rate*100:.2f}%, "
            f"Users={metrics.avg_urllc_users:.1f}/{metrics.avg_embb_users:.1f}"
        )
        
        # CSV logging
        if self.enable_csv and self.csv_writer:
            self.csv_writer.writerow(asdict(metrics))
            self.csv_file.flush()
    
    def log_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file."""
        config_path = self.experiment_dir / "config.json"
        
        # Convert dataclasses to dicts
        serializable_config = {}
        for key, value in config.items():
            if hasattr(value, '__dict__'):
                serializable_config[key] = value.__dict__
            else:
                serializable_config[key] = value
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def close(self):
        """Close logger and files."""
        if self.csv_file:
            self.csv_file.close()
        
        # Remove handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


if __name__ == "__main__":
    # Demo: Metrics tracking and logging
    print("=" * 60)
    print("LOGGER AND METRICS TRACKER DEMONSTRATION")
    print("=" * 60)
    
    # Initialize logger
    logger = SimulationLogger(
        log_dir="/home/claude/5G_ORAN_RL_Simulation/logs",
        experiment_name="demo_run",
        console_level=logging.INFO
    )
    
    # Initialize metrics tracker
    tracker = MetricsTracker(
        window_size=50,
        urllc_threshold=0.001,
        embb_threshold=0.01
    )
    
    # Simulate a few episodes
    np.random.seed(42)
    
    for episode in range(5):
        logger.info(f"Starting episode {episode}")
        
        # Simulate steps
        for step in range(168):  # 1 week
            hour = step % 24
            
            # Generate synthetic metrics
            tracker.log_step(
                step=step,
                hour=hour,
                revenue=500 + np.random.normal(0, 50),
                cost=200 + np.random.normal(0, 20),
                urllc_users=10 + np.random.randint(-2, 3),
                embb_users=50 + np.random.randint(-5, 6),
                arrivals=np.random.poisson(2),
                churns=np.random.poisson(1),
                urllc_violations=np.random.poisson(0.1),
                embb_violations=np.random.poisson(0.5),
                urllc_fee_factor=1.0 + np.random.normal(0, 0.05),
                urllc_overage_factor=1.0 + np.random.normal(0, 0.05),
                embb_fee_factor=1.0 + np.random.normal(0, 0.05),
                embb_overage_factor=1.0 + np.random.normal(0, 0.05),
                prb_utilization=0.7 + np.random.normal(0, 0.1)
            )
        
        # End episode
        episode_metrics = tracker.end_episode()
        logger.log_episode(episode_metrics)
        
        # Print rolling stats
        rolling = tracker.get_rolling_stats()
        print(f"  Rolling profit: ${rolling['rolling_profit_mean']:.0f} ± ${rolling['rolling_profit_std']:.0f}")
    
    # Print summary
    print("\nConstraint satisfaction rates:")
    rates = tracker.get_constraint_satisfaction_rate()
    print(f"  URLLC: {rates['urllc']:.1%}")
    print(f"  eMBB: {rates['embb']:.1%}")
    
    # Close logger
    logger.close()
    
    print(f"\nLogs saved to {logger.experiment_dir}")
