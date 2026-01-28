"""
Plotting and Analysis Module for 5G O-RAN Network Slicing Simulation.

Provides comprehensive visualization and analysis of:
- Training metrics (profit, revenue, cost, violations)
- Constraint satisfaction (URLLC/eMBB SLA compliance)
- Pricing dynamics (fee factors, overage rates)
- User dynamics (arrivals, churn, population)
- Cost breakdown analysis
- Baseline comparisons

Dependencies:
- matplotlib
- seaborn (optional, for enhanced visuals)
- pandas
- numpy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import json
import csv
from datetime import datetime

# Conditional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import PercentFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class EpisodeMetrics:
    """Metrics collected during a single episode."""
    episode: int
    timesteps: int
    
    # Financial metrics
    total_profit: float
    total_revenue: float
    total_cost: float
    avg_hourly_profit: float
    
    # QoS metrics
    urllc_violation_rate: float
    embb_violation_rate: float
    urllc_violations_total: int
    embb_violations_total: int
    
    # User metrics
    avg_urllc_users: float
    avg_embb_users: float
    total_arrivals: int
    total_churns: int
    churn_rate: float
    
    # Pricing metrics
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


@dataclass
class TrainingLog:
    """Collection of training metrics over time."""
    episodes: List[EpisodeMetrics] = field(default_factory=list)
    
    # Aggregated metrics
    timestep_profits: List[float] = field(default_factory=list)
    timestep_violations: List[Tuple[float, float]] = field(default_factory=list)
    
    # Lagrange multipliers (for CMDP)
    lambda_urllc_history: List[float] = field(default_factory=list)
    lambda_embb_history: List[float] = field(default_factory=list)


class ResultsAnalyzer:
    """
    Analyzes and visualizes simulation results.
    
    Supports loading from CSV logs and generating comprehensive reports.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize analyzer.
        
        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = Path(log_dir)
        self.training_log = TrainingLog()
        self.baseline_results: Dict[str, TrainingLog] = {}
        
        # Style settings
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
        if HAS_SEABORN:
            sns.set_palette("husl")
    
    def load_episode_log(self, filepath: str) -> None:
        """Load episode metrics from CSV file."""
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            metrics = EpisodeMetrics(
                episode=int(row.get('episode', 0)),
                timesteps=int(row.get('timesteps', 168)),
                total_profit=float(row.get('total_profit', 0)),
                total_revenue=float(row.get('total_revenue', 0)),
                total_cost=float(row.get('total_cost', 0)),
                avg_hourly_profit=float(row.get('avg_hourly_profit', 0)),
                urllc_violation_rate=float(row.get('urllc_violation_rate', 0)),
                embb_violation_rate=float(row.get('embb_violation_rate', 0)),
                urllc_violations_total=int(row.get('urllc_violations_total', 0)),
                embb_violations_total=int(row.get('embb_violations_total', 0)),
                avg_urllc_users=float(row.get('avg_urllc_users', 0)),
                avg_embb_users=float(row.get('avg_embb_users', 0)),
                total_arrivals=int(row.get('total_arrivals', 0)),
                total_churns=int(row.get('total_churns', 0)),
                churn_rate=float(row.get('churn_rate', 0)),
                avg_urllc_fee_factor=float(row.get('avg_urllc_fee_factor', 1.0)),
                avg_urllc_overage_factor=float(row.get('avg_urllc_overage_factor', 1.0)),
                avg_embb_fee_factor=float(row.get('avg_embb_fee_factor', 1.0)),
                avg_embb_overage_factor=float(row.get('avg_embb_overage_factor', 1.0)),
                urllc_constraint_satisfied=bool(row.get('urllc_constraint_satisfied', True)),
                embb_constraint_satisfied=bool(row.get('embb_constraint_satisfied', True)),
                energy_cost=float(row.get('energy_cost', 0)),
                spectrum_cost=float(row.get('spectrum_cost', 0)),
                backhaul_cost=float(row.get('backhaul_cost', 0)),
                fixed_cost=float(row.get('fixed_cost', 0))
            )
            self.training_log.episodes.append(metrics)
    
    def load_baseline(self, name: str, filepath: str) -> None:
        """Load baseline results for comparison."""
        baseline_log = TrainingLog()
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            metrics = EpisodeMetrics(
                episode=int(row.get('episode', 0)),
                timesteps=int(row.get('timesteps', 168)),
                total_profit=float(row.get('total_profit', 0)),
                total_revenue=float(row.get('total_revenue', 0)),
                total_cost=float(row.get('total_cost', 0)),
                avg_hourly_profit=float(row.get('avg_hourly_profit', 0)),
                urllc_violation_rate=float(row.get('urllc_violation_rate', 0)),
                embb_violation_rate=float(row.get('embb_violation_rate', 0)),
                urllc_violations_total=int(row.get('urllc_violations_total', 0)),
                embb_violations_total=int(row.get('embb_violations_total', 0)),
                avg_urllc_users=float(row.get('avg_urllc_users', 0)),
                avg_embb_users=float(row.get('avg_embb_users', 0)),
                total_arrivals=int(row.get('total_arrivals', 0)),
                total_churns=int(row.get('total_churns', 0)),
                churn_rate=float(row.get('churn_rate', 0)),
                avg_urllc_fee_factor=float(row.get('avg_urllc_fee_factor', 1.0)),
                avg_urllc_overage_factor=float(row.get('avg_urllc_overage_factor', 1.0)),
                avg_embb_fee_factor=float(row.get('avg_embb_fee_factor', 1.0)),
                avg_embb_overage_factor=float(row.get('avg_embb_overage_factor', 1.0)),
                urllc_constraint_satisfied=bool(row.get('urllc_constraint_satisfied', True)),
                embb_constraint_satisfied=bool(row.get('embb_constraint_satisfied', True))
            )
            baseline_log.episodes.append(metrics)
        
        self.baseline_results[name] = baseline_log
    
    def get_summary_statistics(self) -> Dict:
        """Compute summary statistics from training log."""
        if not self.training_log.episodes:
            return {}
        
        episodes = self.training_log.episodes
        
        # Use last 20% of episodes for final performance
        n_final = max(1, len(episodes) // 5)
        final_episodes = episodes[-n_final:]
        
        stats = {
            'total_episodes': len(episodes),
            'final_avg_profit': np.mean([e.total_profit for e in final_episodes]),
            'final_std_profit': np.std([e.total_profit for e in final_episodes]),
            'final_avg_revenue': np.mean([e.total_revenue for e in final_episodes]),
            'final_avg_cost': np.mean([e.total_cost for e in final_episodes]),
            'final_urllc_violation': np.mean([e.urllc_violation_rate for e in final_episodes]),
            'final_embb_violation': np.mean([e.embb_violation_rate for e in final_episodes]),
            'final_urllc_constraint_rate': np.mean([e.urllc_constraint_satisfied for e in final_episodes]),
            'final_embb_constraint_rate': np.mean([e.embb_constraint_satisfied for e in final_episodes]),
            'final_churn_rate': np.mean([e.churn_rate for e in final_episodes]),
            'final_avg_urllc_users': np.mean([e.avg_urllc_users for e in final_episodes]),
            'final_avg_embb_users': np.mean([e.avg_embb_users for e in final_episodes]),
        }
        
        return stats
    
    def export_summary_csv(self, filepath: str) -> None:
        """Export summary statistics to CSV."""
        stats = self.get_summary_statistics()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in stats.items():
                writer.writerow([key, value])


def plot_training_curves(
    analyzer: ResultsAnalyzer,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot training curves showing profit, revenue, cost over episodes.
    
    Args:
        analyzer: ResultsAnalyzer with loaded data
        save_path: Optional path to save figure
        show: Whether to display figure
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None
    
    episodes = analyzer.training_log.episodes
    if not episodes:
        print("No episode data to plot")
        return None
    
    # Extract data
    episode_nums = [e.episode for e in episodes]
    profits = [e.total_profit for e in episodes]
    revenues = [e.total_revenue for e in episodes]
    costs = [e.total_cost for e in episodes]
    
    # Compute rolling averages
    window = min(20, len(episodes) // 10 + 1)
    profit_smooth = pd.Series(profits).rolling(window, min_periods=1).mean()
    revenue_smooth = pd.Series(revenues).rolling(window, min_periods=1).mean()
    cost_smooth = pd.Series(costs).rolling(window, min_periods=1).mean()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Profit curve
    ax1 = axes[0, 0]
    ax1.plot(episode_nums, profits, alpha=0.3, color='green', label='Episode')
    ax1.plot(episode_nums, profit_smooth, color='green', linewidth=2, label=f'Rolling Mean (w={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Profit ($)')
    ax1.set_title('Episode Profit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Revenue and Cost
    ax2 = axes[0, 1]
    ax2.plot(episode_nums, revenue_smooth, color='blue', linewidth=2, label='Revenue')
    ax2.plot(episode_nums, cost_smooth, color='red', linewidth=2, label='Cost')
    ax2.fill_between(episode_nums, cost_smooth, revenue_smooth, alpha=0.3, color='green', label='Profit')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Amount ($)')
    ax2.set_title('Revenue vs Cost')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Hourly profit
    ax3 = axes[1, 0]
    hourly_profits = [e.avg_hourly_profit for e in episodes]
    hourly_smooth = pd.Series(hourly_profits).rolling(window, min_periods=1).mean()
    ax3.plot(episode_nums, hourly_profits, alpha=0.3, color='purple')
    ax3.plot(episode_nums, hourly_smooth, color='purple', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Hourly Profit ($)')
    ax3.set_title('Hourly Profit')
    ax3.grid(True, alpha=0.3)
    
    # Distribution of final profits
    ax4 = axes[1, 1]
    n_final = max(1, len(episodes) // 5)
    final_profits = [e.total_profit for e in episodes[-n_final:]]
    ax4.hist(final_profits, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax4.axvline(np.mean(final_profits), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(final_profits):.0f}')
    ax4.set_xlabel('Total Profit ($)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Final {n_final} Episodes Profit Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_constraint_satisfaction(
    analyzer: ResultsAnalyzer,
    urllc_threshold: float = 0.001,
    embb_threshold: float = 0.01,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot constraint satisfaction analysis.
    
    Args:
        analyzer: ResultsAnalyzer with loaded data
        urllc_threshold: URLLC violation threshold (e.g., 0.001 = 0.1%)
        embb_threshold: eMBB violation threshold
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    episodes = analyzer.training_log.episodes
    if not episodes:
        return None
    
    episode_nums = [e.episode for e in episodes]
    urllc_violations = [e.urllc_violation_rate for e in episodes]
    embb_violations = [e.embb_violation_rate for e in episodes]
    
    window = min(20, len(episodes) // 10 + 1)
    urllc_smooth = pd.Series(urllc_violations).rolling(window, min_periods=1).mean()
    embb_smooth = pd.Series(embb_violations).rolling(window, min_periods=1).mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # URLLC violations
    ax1 = axes[0, 0]
    ax1.plot(episode_nums, urllc_violations, alpha=0.3, color='red')
    ax1.plot(episode_nums, urllc_smooth, color='red', linewidth=2, label='Rolling Mean')
    ax1.axhline(y=urllc_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({urllc_threshold*100:.2f}%)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Violation Rate')
    ax1.set_title('URLLC QoS Violation Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # eMBB violations
    ax2 = axes[0, 1]
    ax2.plot(episode_nums, embb_violations, alpha=0.3, color='orange')
    ax2.plot(episode_nums, embb_smooth, color='orange', linewidth=2, label='Rolling Mean')
    ax2.axhline(y=embb_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({embb_threshold*100:.1f}%)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Violation Rate')
    ax2.set_title('eMBB QoS Violation Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Constraint satisfaction rate
    ax3 = axes[1, 0]
    n_final = max(1, len(episodes) // 5)
    urllc_sat_rate = np.mean([e.urllc_constraint_satisfied for e in episodes[-n_final:]])
    embb_sat_rate = np.mean([e.embb_constraint_satisfied for e in episodes[-n_final:]])
    both_sat_rate = np.mean([e.urllc_constraint_satisfied and e.embb_constraint_satisfied for e in episodes[-n_final:]])
    
    bars = ax3.bar(['URLLC', 'eMBB', 'Both'], [urllc_sat_rate, embb_sat_rate, both_sat_rate], 
                   color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.95, color='gray', linestyle='--', label='95% Target')
    ax3.set_ylabel('Satisfaction Rate')
    ax3.set_title(f'Constraint Satisfaction (Last {n_final} Episodes)')
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    
    # Add value labels
    for bar, val in zip(bars, [urllc_sat_rate, embb_sat_rate, both_sat_rate]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Violation distribution
    ax4 = axes[1, 1]
    final_urllc = [e.urllc_violation_rate for e in episodes[-n_final:]]
    final_embb = [e.embb_violation_rate for e in episodes[-n_final:]]
    
    positions = [1, 2]
    bp = ax4.boxplot([final_urllc, final_embb], positions=positions, widths=0.6)
    ax4.axhline(y=urllc_threshold, color='red', linestyle='--', alpha=0.7, label=f'URLLC Threshold')
    ax4.axhline(y=embb_threshold, color='orange', linestyle='--', alpha=0.7, label=f'eMBB Threshold')
    ax4.set_xticks(positions)
    ax4.set_xticklabels(['URLLC', 'eMBB'])
    ax4.set_ylabel('Violation Rate')
    ax4.set_title(f'Violation Rate Distribution (Last {n_final} Episodes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_pricing_analysis(
    analyzer: ResultsAnalyzer,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot pricing factor analysis.
    
    Shows how the RL agent adjusts fee and overage factors over time.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    episodes = analyzer.training_log.episodes
    if not episodes:
        return None
    
    episode_nums = [e.episode for e in episodes]
    
    # Extract pricing data
    urllc_fee = [e.avg_urllc_fee_factor for e in episodes]
    urllc_overage = [e.avg_urllc_overage_factor for e in episodes]
    embb_fee = [e.avg_embb_fee_factor for e in episodes]
    embb_overage = [e.avg_embb_overage_factor for e in episodes]
    
    window = min(20, len(episodes) // 10 + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # URLLC pricing
    ax1 = axes[0, 0]
    ax1.plot(episode_nums, pd.Series(urllc_fee).rolling(window, min_periods=1).mean(), 
             color='blue', linewidth=2, label='Fee Factor')
    ax1.plot(episode_nums, pd.Series(urllc_overage).rolling(window, min_periods=1).mean(),
             color='red', linewidth=2, label='Overage Factor')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(episode_nums, 0.8, 1.2, alpha=0.1, color='gray')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Price Factor')
    ax1.set_title('URLLC Pricing Factors')
    ax1.legend()
    ax1.set_ylim(0.7, 1.3)
    ax1.grid(True, alpha=0.3)
    
    # eMBB pricing
    ax2 = axes[0, 1]
    ax2.plot(episode_nums, pd.Series(embb_fee).rolling(window, min_periods=1).mean(),
             color='blue', linewidth=2, label='Fee Factor')
    ax2.plot(episode_nums, pd.Series(embb_overage).rolling(window, min_periods=1).mean(),
             color='red', linewidth=2, label='Overage Factor')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(episode_nums, 0.8, 1.2, alpha=0.1, color='gray')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Price Factor')
    ax2.set_title('eMBB Pricing Factors')
    ax2.legend()
    ax2.set_ylim(0.7, 1.3)
    ax2.grid(True, alpha=0.3)
    
    # Final pricing distribution
    n_final = max(1, len(episodes) // 5)
    
    ax3 = axes[1, 0]
    final_urllc_fee = [e.avg_urllc_fee_factor for e in episodes[-n_final:]]
    final_urllc_overage = [e.avg_urllc_overage_factor for e in episodes[-n_final:]]
    ax3.hist(final_urllc_fee, bins=15, alpha=0.7, label='Fee Factor', color='blue')
    ax3.hist(final_urllc_overage, bins=15, alpha=0.7, label='Overage Factor', color='red')
    ax3.axvline(x=1.0, color='gray', linestyle='--')
    ax3.set_xlabel('Price Factor')
    ax3.set_ylabel('Count')
    ax3.set_title(f'URLLC Final Pricing Distribution (n={n_final})')
    ax3.legend()
    
    ax4 = axes[1, 1]
    final_embb_fee = [e.avg_embb_fee_factor for e in episodes[-n_final:]]
    final_embb_overage = [e.avg_embb_overage_factor for e in episodes[-n_final:]]
    ax4.hist(final_embb_fee, bins=15, alpha=0.7, label='Fee Factor', color='blue')
    ax4.hist(final_embb_overage, bins=15, alpha=0.7, label='Overage Factor', color='red')
    ax4.axvline(x=1.0, color='gray', linestyle='--')
    ax4.set_xlabel('Price Factor')
    ax4.set_ylabel('Count')
    ax4.set_title(f'eMBB Final Pricing Distribution (n={n_final})')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_user_dynamics(
    analyzer: ResultsAnalyzer,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot user dynamics: arrivals, churn, population stability.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    episodes = analyzer.training_log.episodes
    if not episodes:
        return None
    
    episode_nums = [e.episode for e in episodes]
    window = min(20, len(episodes) // 10 + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # User population
    ax1 = axes[0, 0]
    urllc_users = pd.Series([e.avg_urllc_users for e in episodes]).rolling(window, min_periods=1).mean()
    embb_users = pd.Series([e.avg_embb_users for e in episodes]).rolling(window, min_periods=1).mean()
    ax1.plot(episode_nums, urllc_users, color='red', linewidth=2, label='URLLC')
    ax1.plot(episode_nums, embb_users, color='blue', linewidth=2, label='eMBB')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Users')
    ax1.set_title('User Population Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Arrivals and Churns
    ax2 = axes[0, 1]
    arrivals = pd.Series([e.total_arrivals for e in episodes]).rolling(window, min_periods=1).mean()
    churns = pd.Series([e.total_churns for e in episodes]).rolling(window, min_periods=1).mean()
    ax2.plot(episode_nums, arrivals, color='green', linewidth=2, label='Arrivals')
    ax2.plot(episode_nums, churns, color='red', linewidth=2, label='Churns')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Count per Episode')
    ax2.set_title('Arrivals vs Churns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Churn rate
    ax3 = axes[1, 0]
    churn_rates = [e.churn_rate for e in episodes]
    churn_smooth = pd.Series(churn_rates).rolling(window, min_periods=1).mean()
    ax3.plot(episode_nums, churn_rates, alpha=0.3, color='red')
    ax3.plot(episode_nums, churn_smooth, color='red', linewidth=2)
    ax3.axhline(y=0.02, color='gray', linestyle='--', label='2% Target')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Churn Rate')
    ax3.set_title('Churn Rate Over Training')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Net user growth
    ax4 = axes[1, 1]
    net_growth = [e.total_arrivals - e.total_churns for e in episodes]
    net_smooth = pd.Series(net_growth).rolling(window, min_periods=1).mean()
    ax4.plot(episode_nums, net_growth, alpha=0.3, color='purple')
    ax4.plot(episode_nums, net_smooth, color='purple', linewidth=2)
    ax4.axhline(y=0, color='gray', linestyle='--')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Net Growth (Arrivals - Churns)')
    ax4.set_title('Net User Growth')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_cost_breakdown(
    analyzer: ResultsAnalyzer,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot cost breakdown analysis.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    episodes = analyzer.training_log.episodes
    if not episodes:
        return None
    
    # Aggregate cost components from last episodes
    n_final = max(1, len(episodes) // 5)
    final_episodes = episodes[-n_final:]
    
    # Check if cost breakdown is available
    if final_episodes[0].energy_cost == 0 and final_episodes[0].total_cost > 0:
        # No breakdown available, show total only
        fig, ax = plt.subplots(figsize=(10, 6))
        costs = [e.total_cost for e in final_episodes]
        ax.hist(costs, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Total Cost ($)')
        ax.set_ylabel('Count')
        ax.set_title(f'Cost Distribution (Last {n_final} Episodes)')
        ax.axvline(np.mean(costs), color='blue', linestyle='--', label=f'Mean: ${np.mean(costs):.0f}')
        ax.legend()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cost breakdown pie chart
        ax1 = axes[0]
        avg_energy = np.mean([e.energy_cost for e in final_episodes])
        avg_spectrum = np.mean([e.spectrum_cost for e in final_episodes])
        avg_backhaul = np.mean([e.backhaul_cost for e in final_episodes])
        avg_fixed = np.mean([e.fixed_cost for e in final_episodes])
        
        labels = ['Energy', 'Spectrum', 'Backhaul', 'Fixed']
        sizes = [avg_energy, avg_spectrum, avg_backhaul, avg_fixed]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Filter out zero values
        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if non_zero:
            labels, sizes, colors = zip(*non_zero)
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Average Cost Breakdown (Last {n_final} Episodes)')
        
        # Cost trend over time
        ax2 = axes[1]
        episode_nums = [e.episode for e in episodes]
        total_costs = [e.total_cost for e in episodes]
        window = min(20, len(episodes) // 10 + 1)
        cost_smooth = pd.Series(total_costs).rolling(window, min_periods=1).mean()
        
        ax2.plot(episode_nums, total_costs, alpha=0.3, color='red')
        ax2.plot(episode_nums, cost_smooth, color='red', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Cost ($)')
        ax2.set_title('Total Cost Over Training')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    analyzer: ResultsAnalyzer,
    metric: str = 'profit',
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot comparison between main results and baselines.
    
    Args:
        analyzer: ResultsAnalyzer with loaded data and baselines
        metric: Metric to compare ('profit', 'violation', 'churn')
        save_path: Optional path to save figure
        show: Whether to display
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if not analyzer.training_log.episodes:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get metric data
    def get_metric_data(log: TrainingLog) -> List[float]:
        if metric == 'profit':
            return [e.total_profit for e in log.episodes]
        elif metric == 'violation':
            return [e.urllc_violation_rate + e.embb_violation_rate for e in log.episodes]
        elif metric == 'churn':
            return [e.churn_rate for e in log.episodes]
        else:
            return [e.total_profit for e in log.episodes]
    
    # Training curves comparison
    ax1 = axes[0]
    main_data = get_metric_data(analyzer.training_log)
    window = min(20, len(main_data) // 10 + 1)
    episodes = list(range(len(main_data)))
    
    ax1.plot(episodes, pd.Series(main_data).rolling(window, min_periods=1).mean(),
             linewidth=2, label='CMDP-SAC')
    
    for name, baseline_log in analyzer.baseline_results.items():
        baseline_data = get_metric_data(baseline_log)
        baseline_smooth = pd.Series(baseline_data).rolling(window, min_periods=1).mean()
        ax1.plot(range(len(baseline_data)), baseline_smooth, linewidth=2, label=name)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(metric.capitalize())
    ax1.set_title(f'{metric.capitalize()} Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final performance comparison
    ax2 = axes[1]
    n_final = max(1, len(main_data) // 5)
    
    names = ['CMDP-SAC']
    means = [np.mean(main_data[-n_final:])]
    stds = [np.std(main_data[-n_final:])]
    
    for name, baseline_log in analyzer.baseline_results.items():
        baseline_data = get_metric_data(baseline_log)
        n_baseline = max(1, len(baseline_data) // 5)
        names.append(name)
        means.append(np.mean(baseline_data[-n_baseline:]))
        stds.append(np.std(baseline_data[-n_baseline:]))
    
    x = np.arange(len(names))
    bars = ax2.bar(x, means, yerr=stds, capsize=5, color=['green'] + ['gray']*(len(names)-1),
                   alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel(metric.capitalize())
    ax2.set_title(f'Final {metric.capitalize()} Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def generate_full_report(
    analyzer: ResultsAnalyzer,
    output_dir: str = "./results",
    show: bool = False
) -> None:
    """
    Generate complete analysis report with all plots.
    
    Args:
        analyzer: ResultsAnalyzer with loaded data
        output_dir: Directory to save plots and report
        show: Whether to display plots interactively
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating analysis report...")
    
    # Generate all plots
    print("  - Training curves...")
    plot_training_curves(analyzer, save_path=output_path / "training_curves.png", show=show)
    
    print("  - Constraint satisfaction...")
    plot_constraint_satisfaction(analyzer, save_path=output_path / "constraints.png", show=show)
    
    print("  - Pricing analysis...")
    plot_pricing_analysis(analyzer, save_path=output_path / "pricing.png", show=show)
    
    print("  - User dynamics...")
    plot_user_dynamics(analyzer, save_path=output_path / "user_dynamics.png", show=show)
    
    print("  - Cost breakdown...")
    plot_cost_breakdown(analyzer, save_path=output_path / "costs.png", show=show)
    
    if analyzer.baseline_results:
        print("  - Baseline comparison...")
        plot_comparison(analyzer, metric='profit', save_path=output_path / "comparison_profit.png", show=show)
        plot_comparison(analyzer, metric='violation', save_path=output_path / "comparison_violation.png", show=show)
    
    # Export summary statistics
    print("  - Summary statistics...")
    analyzer.export_summary_csv(output_path / "summary_stats.csv")
    
    # Generate text report
    stats = analyzer.get_summary_statistics()
    
    report = f"""
================================================================================
5G O-RAN NETWORK SLICING RL SIMULATION - ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

TRAINING SUMMARY
----------------
Total Episodes: {stats.get('total_episodes', 'N/A')}

FINAL PERFORMANCE (Last 20% of episodes)
----------------------------------------
Average Profit: ${stats.get('final_avg_profit', 0):.2f} (±${stats.get('final_std_profit', 0):.2f})
Average Revenue: ${stats.get('final_avg_revenue', 0):.2f}
Average Cost: ${stats.get('final_avg_cost', 0):.2f}

CONSTRAINT SATISFACTION
-----------------------
URLLC Violation Rate: {stats.get('final_urllc_violation', 0)*100:.4f}%
eMBB Violation Rate: {stats.get('final_embb_violation', 0)*100:.4f}%
URLLC Constraint Satisfaction: {stats.get('final_urllc_constraint_rate', 0)*100:.1f}%
eMBB Constraint Satisfaction: {stats.get('final_embb_constraint_rate', 0)*100:.1f}%

USER METRICS
------------
Average URLLC Users: {stats.get('final_avg_urllc_users', 0):.1f}
Average eMBB Users: {stats.get('final_avg_embb_users', 0):.1f}
Churn Rate: {stats.get('final_churn_rate', 0)*100:.2f}%

GENERATED FILES
---------------
- training_curves.png
- constraints.png
- pricing.png
- user_dynamics.png
- costs.png
- summary_stats.csv
{'- comparison_profit.png' if analyzer.baseline_results else ''}
{'- comparison_violation.png' if analyzer.baseline_results else ''}

================================================================================
"""
    
    with open(output_path / "report.txt", 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    # Demo: Create sample data and generate plots
    print("=" * 60)
    print("EVALUATION MODULE DEMONSTRATION")
    print("=" * 60)
    
    # Create analyzer
    analyzer = ResultsAnalyzer()
    
    # Generate synthetic episode data for demonstration
    print("\nGenerating synthetic training data...")
    np.random.seed(42)
    
    for ep in range(100):
        # Simulate improving performance over training
        progress = ep / 100
        
        metrics = EpisodeMetrics(
            episode=ep,
            timesteps=168,
            total_profit=50000 + 20000 * progress + np.random.normal(0, 5000),
            total_revenue=80000 + 10000 * progress + np.random.normal(0, 3000),
            total_cost=30000 - 5000 * progress + np.random.normal(0, 2000),
            avg_hourly_profit=300 + 150 * progress + np.random.normal(0, 30),
            urllc_violation_rate=max(0, 0.01 - 0.008 * progress + np.random.normal(0, 0.002)),
            embb_violation_rate=max(0, 0.05 - 0.04 * progress + np.random.normal(0, 0.01)),
            urllc_violations_total=int(max(0, 5 - 4 * progress + np.random.normal(0, 1))),
            embb_violations_total=int(max(0, 20 - 15 * progress + np.random.normal(0, 3))),
            avg_urllc_users=10 + 5 * progress + np.random.normal(0, 1),
            avg_embb_users=50 + 20 * progress + np.random.normal(0, 5),
            total_arrivals=int(100 + 50 * progress + np.random.normal(0, 10)),
            total_churns=int(max(0, 30 - 20 * progress + np.random.normal(0, 5))),
            churn_rate=max(0, 0.03 - 0.02 * progress + np.random.normal(0, 0.005)),
            avg_urllc_fee_factor=1.0 + 0.1 * np.sin(ep / 10) + np.random.normal(0, 0.02),
            avg_urllc_overage_factor=1.0 + 0.05 * np.cos(ep / 10) + np.random.normal(0, 0.02),
            avg_embb_fee_factor=1.0 - 0.05 * progress + np.random.normal(0, 0.02),
            avg_embb_overage_factor=0.95 + 0.1 * progress + np.random.normal(0, 0.02),
            urllc_constraint_satisfied=np.random.random() < (0.7 + 0.25 * progress),
            embb_constraint_satisfied=np.random.random() < (0.6 + 0.35 * progress),
            energy_cost=5000 + np.random.normal(0, 500),
            spectrum_cost=6000,
            backhaul_cost=2000 + np.random.normal(0, 200),
            fixed_cost=17000
        )
        analyzer.training_log.episodes.append(metrics)
    
    print(f"Generated {len(analyzer.training_log.episodes)} synthetic episodes")
    
    # Print summary
    stats = analyzer.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Generate plots (without showing in headless environment)
    print("\nGenerating plots...")
    output_dir = Path("/home/claude/5G_ORAN_RL_Simulation/results/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if HAS_MATPLOTLIB:
        plot_training_curves(analyzer, save_path=str(output_dir / "training_curves.png"), show=False)
        plot_constraint_satisfaction(analyzer, save_path=str(output_dir / "constraints.png"), show=False)
        plot_pricing_analysis(analyzer, save_path=str(output_dir / "pricing.png"), show=False)
        plot_user_dynamics(analyzer, save_path=str(output_dir / "user_dynamics.png"), show=False)
        plot_cost_breakdown(analyzer, save_path=str(output_dir / "costs.png"), show=False)
        print(f"\nPlots saved to {output_dir}")
    else:
        print("matplotlib not available - skipping plot generation")
    
    # Export summary
    analyzer.export_summary_csv(str(output_dir / "summary_stats.csv"))
    print(f"Summary statistics exported to {output_dir / 'summary_stats.csv'}")
