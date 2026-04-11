"""
5G Network Slicing Dynamic Pricing Environment
================================================
Gymnasium environment implementing the MDP from Model Spec (F1-F11).

Slices: URLLC (s=0) and eMBB (s=1)
State:  (N_U, N_E, eta_U_prev, eta_E_prev) -- 4-dim
Action: (F_U, p_U, F_E, p_E)               -- 4-dim continuous [0,1] scaled
Reward: r_t = Revenue - QoS Penalty
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NetworkSlicingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}

        # --- Slices: index 0 = URLLC, index 1 = eMBB ---
        self.n_slices = 2

        # Tariff reference (normalization)
        self.F_ref = np.array(cfg.get("F_ref", [50.0, 30.0]))
        self.p_ref = np.array(cfg.get("p_ref", [10.0, 5.0]))
        self.Q_bar = np.array(cfg.get("Q_bar", [5.0, 30.0]))

        # Action bounds
        self.F_max = np.array(cfg.get("F_max", [100.0, 100.0]))
        self.p_max = np.array(cfg.get("p_max", [20.0, 20.0]))

        # Traffic (LogNormal parameters)
        self.mu = np.array(cfg.get("mu", [1.0, 3.0]))
        self.sigma2 = np.array(cfg.get("sigma2", [0.5, 0.8]))
        self.sigma = np.sqrt(self.sigma2)

        # Departure coefficients (F3)
        self.gamma0 = np.array(cfg.get("gamma0", [-9.72, -12.12]))
        self.gamma_F = cfg.get("gamma_F", 1.0)
        self.gamma_p = cfg.get("gamma_p", 0.8)
        self.gamma_eta = np.array(cfg.get("gamma_eta", [3.0, 0.5]))

        # Arrival coefficients (F4, F5)
        self.beta0 = np.array(cfg.get("beta0", [2.0, 2.5]))
        self.beta_F = cfg.get("beta_F", 0.8)
        self.beta_p = cfg.get("beta_p", 0.6)
        self.lambda_max = np.array(cfg.get("lambda_max", [0.05, 0.15]))

        # QoS parameters (F7)
        self.eta_low = np.array(cfg.get("eta_low", [0.90, 0.80]))
        self.eta_high = np.array(cfg.get("eta_high", [1.0, 1.0]))
        self.eta_tgt = np.array(cfg.get("eta_tgt", [0.99999, 0.90]))

        # Penalty weights (F9)
        self.w = np.array(cfg.get("w", [500.0, 50.0]))

        # MDP parameters
        self.T = cfg.get("T", 720)
        self.discount = cfg.get("gamma", 0.99)

        # Initial state
        self.N_init = np.array(cfg.get("N_init", [1000.0, 5000.0]))
        self.eta_init = np.array(cfg.get("eta_init", [0.95, 0.90]))

        # Normalization constants (Audit recommendation: eliminate scale imbalance)
        self.reward_scale = cfg.get("reward_scale", 1e-5)

        # --- Spaces ---
        # Action: 4-dim continuous [0, 1], scaled to actual values in step()
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # State: (N_U/N_init_U, N_E/N_init_E, eta_U_prev, eta_E_prev)
        # All dimensions normalized to ~[0, 2] range
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _scale_action(self, action):
        """Scale [0,1] action to actual F and p values."""
        F = np.array([action[0] * self.F_max[0], action[2] * self.F_max[1]])
        p = np.array([action[1] * self.p_max[0], action[3] * self.p_max[1]])
        return F, p

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_init.copy().astype(np.float64)
        self.eta_prev = self.eta_init.copy().astype(np.float64)
        self.t = 0
        self.info_log = {}
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize N by initial values to eliminate scale imbalance
        return np.array(
            [self.N[0] / self.N_init[0],
             self.N[1] / self.N_init[1],
             self.eta_prev[0],
             self.eta_prev[1]],
            dtype=np.float32,
        )

    def step(self, action):
        self.t += 1
        F, p = self._scale_action(action)

        # Normalized prices (for departure/arrival formulas)
        F_tilde = F / self.F_ref
        p_tilde = p / self.p_ref

        # ============ Phase B: Subscriber Dynamics ============

        # Step 3: Departure (F3)
        dep_logit = (
            self.gamma0
            + self.gamma_F * F_tilde
            + self.gamma_p * p_tilde
            - self.gamma_eta * self.eta_prev
        )
        P_dep = self._sigmoid(dep_logit)  # shape (2,)

        N_leave = np.zeros(2)
        for s in range(2):
            if self.N[s] > 0:
                N_leave[s] = self.np_random.binomial(
                    int(self.N[s]), float(P_dep[s])
                )
        N_surv = self.N - N_leave

        # Step 4: Arrival (F4, F5)
        arr_logit = self.beta0 - self.beta_F * F_tilde - self.beta_p * p_tilde
        P_arr = self._sigmoid(arr_logit)
        lam = self.lambda_max * P_arr
        N_new = np.array(
            [self.np_random.poisson(float(lam[s])) for s in range(2)],
            dtype=np.float64,
        )

        # Active users (F6)
        N_active = N_surv + N_new

        # ============ Phase C: Usage and QoS ============

        # Step 5: Usage generation and billing (F1, F2)
        total_revenue = 0.0
        for s in range(2):
            n = int(N_active[s])
            if n > 0:
                # Generate usage for all active users
                usage = self.np_random.lognormal(
                    mean=float(self.mu[s]),
                    sigma=float(self.sigma[s]),
                    size=n,
                )
                # Compute bills (F1)
                overage = np.maximum(0.0, usage - self.Q_bar[s])
                bills = F[s] + overage * p[s]
                total_revenue += np.sum(bills)

        # Step 6: QoS realization (F7)
        eta = np.array([
            self.np_random.uniform(float(self.eta_low[s]), float(self.eta_high[s]))
            for s in range(2)
        ])

        # ============ Phase D: Learning Signal ============

        # Step 7: Reward (F8-F10)
        penalty = 0.0
        for s in range(2):
            shortfall = max(0.0, self.eta_tgt[s] - eta[s])
            penalty += self.w[s] * N_active[s] * shortfall

        reward = (total_revenue - penalty) * self.reward_scale

        # Step 8: State transition (F11)
        self.N = N_active.copy()
        self.eta_prev = eta.copy()

        terminated = self.t >= self.T
        truncated = False

        info = {
            "revenue": total_revenue,
            "penalty": penalty,
            "N_U": N_active[0],
            "N_E": N_active[1],
            "eta_U": eta[0],
            "eta_E": eta[1],
            "F_U": F[0],
            "p_U": p[0],
            "F_E": F[1],
            "p_E": p[1],
            "P_dep_U": P_dep[0],
            "P_dep_E": P_dep[1],
            "N_leave_U": N_leave[0],
            "N_leave_E": N_leave[1],
            "N_new_U": N_new[0],
            "N_new_E": N_new[1],
        }

        return self._get_obs(), float(reward), terminated, truncated, info
