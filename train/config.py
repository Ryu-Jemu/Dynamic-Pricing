"""
Centralized configuration for all training and evaluation.
All parameter values are verified through 9-fold Audit.
"""

# ── Environment Parameters (Audit-verified) ──────────────────────
ENV_CONFIG = {
    # Tariff reference
    "F_ref": [50.0, 30.0],
    "p_ref": [10.0, 5.0],
    "Q_bar": [5.0, 30.0],
    # Action bounds
    "F_max": [100.0, 100.0],
    "p_max": [20.0, 20.0],
    # Traffic (LogNormal)
    "mu": [1.0, 3.0],
    "sigma2": [0.5, 0.8],
    # Departure
    "gamma0": [-9.72, -12.12],
    "gamma_F": 1.0,
    "gamma_p": 0.8,
    "gamma_eta": [3.0, 0.5],
    # Arrival
    "beta0": [2.0, 2.5],
    "beta_F": 0.8,
    "beta_p": 0.6,
    "lambda_max": [0.05, 0.15],
    # QoS
    "eta_low": [0.90, 0.80],
    "eta_high": [1.0, 1.0],
    "eta_tgt": [0.99999, 0.90],
    # Penalty
    "w": [500.0, 50.0],
    # Reward scaling (Audit recommendation: stabilize critic learning)
    "reward_scale": 1e-5,
    # MDP
    "T": 720,
    "gamma": 0.99,
    # Initial state
    "N_init": [1000.0, 5000.0],
    "eta_init": [0.95, 0.90],
}

# ── SAC Hyperparameters ──────────────────────────────────────────
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "policy_kwargs": dict(net_arch=[256, 256, 256]),
    "total_timesteps": 720 * 500,   # 500 episodes
    "seed": 42,
}

# ── PPO Hyperparameters ──────────────────────────────────────────
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_epochs": 10,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "policy_kwargs": dict(net_arch=[256, 256, 256]),
    "total_timesteps": 720 * 500,   # 500 episodes
    "seed": 42,
}

# ── Evaluation ───────────────────────────────────────────────────
EVAL_CONFIG = {
    "n_eval_episodes": 20,
    "seeds": [42, 123, 456, 789, 1024],
}

# ── Reference action for Static-Heuristic baseline ───────────────
# F_U=50/100=0.5, p_U=10/20=0.5, F_E=30/100=0.3, p_E=5/20=0.25
REFERENCE_ACTION = [0.5, 0.5, 0.3, 0.25]
