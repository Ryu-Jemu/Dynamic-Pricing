"""
CMDP SAC Training Script - Primal-Dual Lagrangian Method

Implements Soft Actor-Critic with learned Lagrange multipliers for
constrained optimization in 5G network slicing.

Objective: max E[Σ γ^t r_t] subject to E[C_i(s,a)] ≤ d_i for all constraints

Uses primal-dual gradient descent:
- Primal (policy): Minimize -J(π) + λ^T C(π)
- Dual (multipliers): λ ← max(0, λ + α_λ (C(π) - d))

References:
- Haarnoja et al. (2018): Soft Actor-Critic
- Tessler et al. (2019): Reward Constrained Policy Optimization
- SafeSlice (2024): CMDP for network slicing
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise

from env.network_slicing_cmdp_env import NetworkSlicingCMDPEnv
from config.scenario_config import get_default_config, SACConfig, CMDPConfig


@dataclass
class LagrangeMultiplierConfig:
    """Configuration for Lagrange multiplier updates."""
    initial_urllc_lambda: float = 1.0
    initial_embb_lambda: float = 1.0
    lambda_lr: float = 0.01  # Learning rate for multipliers
    lambda_max: float = 100.0  # Maximum multiplier value
    lambda_min: float = 0.0  # Minimum multiplier value
    constraint_threshold_urllc: float = 0.001  # 0.1% violation allowed
    constraint_threshold_embb: float = 0.01  # 1% violation allowed


class LagrangeMultiplierManager:
    """
    Manages Lagrange multipliers for CMDP constraints.
    
    Uses dual gradient ascent to update multipliers based on constraint violations.
    """
    
    def __init__(self, config: LagrangeMultiplierConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Initialize multipliers as learnable parameters
        self.lambda_urllc = torch.tensor(
            [config.initial_urllc_lambda], 
            requires_grad=True, 
            dtype=torch.float32,
            device=device
        )
        self.lambda_embb = torch.tensor(
            [config.initial_embb_lambda],
            requires_grad=True,
            dtype=torch.float32,
            device=device
        )
        
        # Optimizer for multipliers
        self.optimizer = Adam(
            [self.lambda_urllc, self.lambda_embb],
            lr=config.lambda_lr
        )
        
        # History for logging
        self.history = {
            "lambda_urllc": [],
            "lambda_embb": [],
            "urllc_violation": [],
            "embb_violation": [],
            "constraint_satisfied": []
        }
    
    def update(
        self,
        urllc_violation_rate: float,
        embb_violation_rate: float
    ) -> Tuple[float, float]:
        """
        Update Lagrange multipliers using dual gradient ascent.
        
        λ ← max(0, λ + α (C - d))
        
        Args:
            urllc_violation_rate: Current URLLC violation rate
            embb_violation_rate: Current eMBB violation rate
            
        Returns:
            Updated (lambda_urllc, lambda_embb)
        """
        # Constraint violations (C - d)
        urllc_slack = urllc_violation_rate - self.config.constraint_threshold_urllc
        embb_slack = embb_violation_rate - self.config.constraint_threshold_embb
        
        # Gradient ascent on multipliers
        self.optimizer.zero_grad()
        
        # Loss for dual problem: -λ * (C - d) (we want to maximize, so negate)
        dual_loss = -(
            self.lambda_urllc * urllc_slack + 
            self.lambda_embb * embb_slack
        )
        dual_loss.backward()
        self.optimizer.step()
        
        # Project to valid range
        with torch.no_grad():
            self.lambda_urllc.clamp_(self.config.lambda_min, self.config.lambda_max)
            self.lambda_embb.clamp_(self.config.lambda_min, self.config.lambda_max)
        
        # Log
        self.history["lambda_urllc"].append(self.lambda_urllc.item())
        self.history["lambda_embb"].append(self.lambda_embb.item())
        self.history["urllc_violation"].append(urllc_violation_rate)
        self.history["embb_violation"].append(embb_violation_rate)
        self.history["constraint_satisfied"].append(
            urllc_slack <= 0 and embb_slack <= 0
        )
        
        return self.lambda_urllc.item(), self.lambda_embb.item()
    
    def get_penalty(
        self,
        urllc_violation: float,
        embb_violation: float
    ) -> float:
        """
        Calculate Lagrangian penalty term.
        
        L_penalty = λ_u * V_u + λ_e * V_e
        """
        penalty = (
            self.lambda_urllc.item() * urllc_violation +
            self.lambda_embb.item() * embb_violation
        )
        return penalty
    
    def get_state(self) -> Dict:
        """Get current multiplier state."""
        return {
            "lambda_urllc": self.lambda_urllc.item(),
            "lambda_embb": self.lambda_embb.item()
        }
    
    def save(self, path: str):
        """Save multiplier state."""
        state = {
            "lambda_urllc": self.lambda_urllc.detach().cpu().numpy().tolist(),
            "lambda_embb": self.lambda_embb.detach().cpu().numpy().tolist(),
            "config": asdict(self.config),
            "history": self.history
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: str):
        """Load multiplier state."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.lambda_urllc = torch.tensor(
            state["lambda_urllc"],
            requires_grad=True,
            dtype=torch.float32,
            device=self.device
        )
        self.lambda_embb = torch.tensor(
            state["lambda_embb"],
            requires_grad=True,
            dtype=torch.float32,
            device=self.device
        )
        self.history = state.get("history", self.history)


class CMDPRewardWrapper:
    """
    Wraps environment reward with Lagrangian penalty.
    
    Modified reward: r' = r - λ_u * V_u - λ_e * V_e
    """
    
    def __init__(
        self,
        lagrange_manager: LagrangeMultiplierManager,
        penalty_scale: float = 1.0
    ):
        self.lagrange_manager = lagrange_manager
        self.penalty_scale = penalty_scale
    
    def modify_reward(
        self,
        original_reward: float,
        info: Dict
    ) -> float:
        """
        Modify reward with Lagrangian penalty.
        
        Args:
            original_reward: Original environment reward (profit)
            info: Step info containing violation rates
            
        Returns:
            Modified reward with constraint penalty
        """
        urllc_viol = info.get("urllc_violation_rate", 0.0)
        embb_viol = info.get("embb_violation_rate", 0.0)
        
        penalty = self.lagrange_manager.get_penalty(urllc_viol, embb_viol)
        
        modified_reward = original_reward - self.penalty_scale * penalty
        
        return modified_reward


class CMDPSACCallback(BaseCallback):
    """
    Custom callback for CMDP-SAC training.
    
    - Updates Lagrange multipliers periodically
    - Logs constraint satisfaction
    - Saves checkpoints
    """
    
    def __init__(
        self,
        lagrange_manager: LagrangeMultiplierManager,
        update_freq: int = 1000,  # Update multipliers every N steps
        log_dir: str = "./logs",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.lagrange_manager = lagrange_manager
        self.update_freq = update_freq
        self.log_dir = log_dir
        
        # Rolling statistics
        self.urllc_violations = deque(maxlen=update_freq)
        self.embb_violations = deque(maxlen=update_freq)
        self.rewards = deque(maxlen=update_freq)
        self.profits = deque(maxlen=update_freq)
        
        # Episode tracking
        self.episode_count = 0
        self.step_in_episode = 0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Get info from environment
        info = self.locals.get("infos", [{}])[0]
        
        # Collect statistics
        self.urllc_violations.append(info.get("urllc_violation_rate", 0.0))
        self.embb_violations.append(info.get("embb_violation_rate", 0.0))
        self.rewards.append(self.locals.get("rewards", [0.0])[0])
        self.profits.append(info.get("profit", 0.0))
        
        self.step_in_episode += 1
        
        # Check for episode end
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            self.step_in_episode = 0
        
        # Update multipliers periodically
        if self.n_calls % self.update_freq == 0 and len(self.urllc_violations) > 0:
            avg_urllc = np.mean(self.urllc_violations)
            avg_embb = np.mean(self.embb_violations)
            
            lambda_u, lambda_e = self.lagrange_manager.update(avg_urllc, avg_embb)
            
            if self.verbose >= 1:
                avg_reward = np.mean(self.rewards)
                avg_profit = np.mean(self.profits)
                print(f"\n[Step {self.n_calls}] Multiplier Update:")
                print(f"  λ_URLLC: {lambda_u:.4f}, λ_eMBB: {lambda_e:.4f}")
                print(f"  Avg URLLC viol: {avg_urllc:.4f}, Avg eMBB viol: {avg_embb:.4f}")
                print(f"  Avg reward: {avg_reward:.2f}, Avg profit: ${avg_profit:.2f}")
        
        return True
    
    def _on_training_end(self):
        """Called at end of training."""
        # Save final multiplier state
        save_path = os.path.join(self.log_dir, "lagrange_multipliers.json")
        self.lagrange_manager.save(save_path)
        
        if self.verbose >= 1:
            print(f"\nTraining complete. Lagrange multipliers saved to {save_path}")


class CMDPSACTrainer:
    """
    Main trainer class for CMDP-SAC.
    
    Orchestrates:
    - Environment creation
    - SAC agent training
    - Lagrange multiplier updates
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        env_config: Optional[Dict] = None,
        sac_config: Optional[SACConfig] = None,
        lagrange_config: Optional[LagrangeMultiplierConfig] = None,
        log_dir: str = "./logs",
        device: str = "auto",
        seed: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            env_config: Environment configuration
            sac_config: SAC hyperparameters
            lagrange_config: Lagrange multiplier configuration
            log_dir: Directory for logs and checkpoints
            device: Training device ("auto", "cpu", "cuda", "mps")
            seed: Random seed
        """
        self.log_dir = log_dir
        self.seed = seed
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "tensorboard"), exist_ok=True)
        
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Configurations
        self.sac_config = sac_config or SACConfig()
        self.lagrange_config = lagrange_config or LagrangeMultiplierConfig()
        
        # Create environment
        self.env_config = env_config or {}
        self.env = self._create_env()
        self.eval_env = self._create_env()
        
        # Create Lagrange multiplier manager
        self.lagrange_manager = LagrangeMultiplierManager(
            self.lagrange_config,
            device=self.device if self.device != "mps" else "cpu"
        )
        
        # Create reward wrapper
        self.reward_wrapper = CMDPRewardWrapper(self.lagrange_manager)
        
        # Create SAC agent
        self.model = self._create_model()
        
        # Training state
        self.training_history = {
            "episode_rewards": [],
            "episode_profits": [],
            "episode_urllc_violations": [],
            "episode_embb_violations": [],
            "constraint_satisfaction_rate": []
        }
    
    def _create_env(self) -> NetworkSlicingCMDPEnv:
        """Create and configure environment."""
        config = get_default_config()
        
        # Apply any custom config
        for key, value in self.env_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        env = NetworkSlicingCMDPEnv(config)
        env.reset(seed=self.seed)
        
        return env
    
    def _create_model(self) -> SAC:
        """Create SAC model with configuration."""
        policy_kwargs = {
            "net_arch": list(self.sac_config.net_arch)
        }
        
        model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.sac_config.learning_rate,
            buffer_size=self.sac_config.buffer_size,
            batch_size=self.sac_config.batch_size,
            gamma=self.sac_config.gamma,
            tau=self.sac_config.tau,
            ent_coef=self.sac_config.ent_coef,
            learning_starts=self.sac_config.learning_starts,
            train_freq=self.sac_config.train_freq,
            gradient_steps=self.sac_config.gradient_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(self.log_dir, "tensorboard"),
            verbose=1,
            seed=self.seed,
            device=self.device if self.device != "mps" else "cpu"
        )
        
        return model
    
    def train(
        self,
        total_timesteps: int = 500000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = 50000,
        multiplier_update_freq: int = 1000
    ):
        """
        Train the CMDP-SAC agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: Frequency of evaluation
            n_eval_episodes: Number of episodes per evaluation
            save_freq: Frequency of checkpoint saving
            multiplier_update_freq: Frequency of Lagrange multiplier updates
        """
        print("=" * 60)
        print("CMDP-SAC Training")
        print("=" * 60)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Device: {self.device}")
        print(f"Log directory: {self.log_dir}")
        print("=" * 60)
        
        # Create callbacks
        cmdp_callback = CMDPSACCallback(
            self.lagrange_manager,
            update_freq=multiplier_update_freq,
            log_dir=self.log_dir,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.log_dir, "best_model"),
            log_path=os.path.join(self.log_dir, "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1
        )
        
        # Start training
        start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[cmdp_callback, eval_callback],
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/3600:.2f} hours")
        
        # Save final model
        self.save(os.path.join(self.log_dir, "final_model"))
        
        return self.model
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate trained agent.
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating over {n_episodes} episodes...")
        
        metrics = {
            "episode_rewards": [],
            "episode_profits": [],
            "episode_lengths": [],
            "urllc_violation_rates": [],
            "embb_violation_rates": [],
            "constraint_satisfied": []
        }
        
        for ep in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_profit = 0
            episode_urllc_viols = []
            episode_embb_viols = []
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_profit += info.get("profit", 0)
                episode_urllc_viols.append(info.get("urllc_violation_rate", 0))
                episode_embb_viols.append(info.get("embb_violation_rate", 0))
                steps += 1
            
            avg_urllc = np.mean(episode_urllc_viols)
            avg_embb = np.mean(episode_embb_viols)
            
            metrics["episode_rewards"].append(episode_reward)
            metrics["episode_profits"].append(episode_profit)
            metrics["episode_lengths"].append(steps)
            metrics["urllc_violation_rates"].append(avg_urllc)
            metrics["embb_violation_rates"].append(avg_embb)
            metrics["constraint_satisfied"].append(
                avg_urllc <= self.lagrange_config.constraint_threshold_urllc and
                avg_embb <= self.lagrange_config.constraint_threshold_embb
            )
        
        # Summary statistics
        summary = {
            "mean_reward": np.mean(metrics["episode_rewards"]),
            "std_reward": np.std(metrics["episode_rewards"]),
            "mean_profit": np.mean(metrics["episode_profits"]),
            "std_profit": np.std(metrics["episode_profits"]),
            "mean_urllc_violation": np.mean(metrics["urllc_violation_rates"]),
            "mean_embb_violation": np.mean(metrics["embb_violation_rates"]),
            "constraint_satisfaction_rate": np.mean(metrics["constraint_satisfied"])
        }
        
        print("\nEvaluation Results:")
        print(f"  Mean reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"  Mean profit: ${summary['mean_profit']:.2f} ± {summary['std_profit']:.2f}")
        print(f"  URLLC violation rate: {summary['mean_urllc_violation']:.4f}")
        print(f"  eMBB violation rate: {summary['mean_embb_violation']:.4f}")
        print(f"  Constraint satisfaction: {summary['constraint_satisfaction_rate']:.1%}")
        
        return {"metrics": metrics, "summary": summary}
    
    def save(self, path: str):
        """Save model and training state."""
        self.model.save(path)
        self.lagrange_manager.save(f"{path}_lagrange.json")
        
        # Save training history
        with open(f"{path}_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and training state."""
        self.model = SAC.load(path, env=self.env, device=self.device)
        
        lagrange_path = f"{path}_lagrange.json"
        if os.path.exists(lagrange_path):
            self.lagrange_manager.load(lagrange_path)
        
        history_path = f"{path}_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        print(f"Model loaded from {path}")


def train_cmdp_sac(
    total_timesteps: int = 500000,
    log_dir: str = "./logs/cmdp_sac",
    seed: int = 42,
    device: str = "auto"
):
    """
    Convenience function to train CMDP-SAC agent.
    
    Args:
        total_timesteps: Total training steps
        log_dir: Directory for logs
        seed: Random seed
        device: Training device
        
    Returns:
        Trained model and evaluation results
    """
    # Custom configurations
    sac_config = SACConfig(
        learning_rate=3e-4,
        buffer_size=200000,
        batch_size=512,
        gamma=0.99,
        tau=0.005,
        net_arch=(256, 256, 128),
        learning_starts=2000,
        train_freq=1,
        gradient_steps=1
    )
    
    lagrange_config = LagrangeMultiplierConfig(
        initial_urllc_lambda=1.0,
        initial_embb_lambda=1.0,
        lambda_lr=0.01,
        constraint_threshold_urllc=0.001,
        constraint_threshold_embb=0.01
    )
    
    # Create trainer
    trainer = CMDPSACTrainer(
        sac_config=sac_config,
        lagrange_config=lagrange_config,
        log_dir=log_dir,
        device=device,
        seed=seed
    )
    
    # Train
    model = trainer.train(
        total_timesteps=total_timesteps,
        eval_freq=10000,
        n_eval_episodes=5,
        save_freq=50000,
        multiplier_update_freq=1000
    )
    
    # Evaluate
    results = trainer.evaluate(n_episodes=10)
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CMDP-SAC for Network Slicing")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--log-dir", type=str, default="./logs/cmdp_sac", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    model, results = train_cmdp_sac(
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        seed=args.seed,
        device=args.device
    )
