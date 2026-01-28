"""
Evaluation module for 5G O-RAN Network Slicing Simulation.

Provides:
- Training metrics visualization
- Performance analysis plots
- Constraint satisfaction analysis
- Comparison with baselines
"""

from .plots import (
    plot_training_curves,
    plot_constraint_satisfaction,
    plot_pricing_analysis,
    plot_user_dynamics,
    plot_cost_breakdown,
    plot_comparison,
    ResultsAnalyzer
)

__all__ = [
    'plot_training_curves',
    'plot_constraint_satisfaction',
    'plot_pricing_analysis',
    'plot_user_dynamics',
    'plot_cost_breakdown',
    'plot_comparison',
    'ResultsAnalyzer'
]
