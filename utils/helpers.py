"""
Common Helper Functions for 5G O-RAN Network Slicing Simulation.

Provides:
- Random seed management
- Device detection (CUDA/MPS/CPU)
- Learning rate schedules
- Configuration I/O
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import asdict, is_dataclass
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - Python random
    - NumPy
    - PyTorch (if available)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set PyTorch seeds
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    """
    Detect and return the best available device.
    
    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA: {device_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = "cpu"
            print("Using CPU")
        
        return device
    
    except ImportError:
        print("PyTorch not available. Using CPU.")
        return "cpu"


def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Starting value
        final_value: Ending value (default 0)
    
    Returns:
        Schedule function: progress -> learning_rate
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (start) to 0 (end).
        """
        return final_value + (initial_value - final_value) * progress_remaining
    
    return schedule


def exponential_schedule(
    initial_value: float,
    decay_rate: float = 0.99
) -> Callable[[float], float]:
    """
    Exponential learning rate schedule.
    
    Args:
        initial_value: Starting value
        decay_rate: Decay rate per step
    
    Returns:
        Schedule function: progress -> learning_rate
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (start) to 0 (end).
        """
        # Convert progress to steps (assuming 1M total steps)
        steps = (1 - progress_remaining) * 1_000_000
        return initial_value * (decay_rate ** (steps / 10000))
    
    return schedule


def cosine_schedule(
    initial_value: float,
    final_value: float = 0.0
) -> Callable[[float], float]:
    """
    Cosine annealing learning rate schedule.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
    
    Returns:
        Schedule function: progress -> learning_rate
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (start) to 0 (end).
        """
        return final_value + 0.5 * (initial_value - final_value) * (
            1 + np.cos(np.pi * (1 - progress_remaining))
        )
    
    return schedule


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.
    
    Handles dataclasses and numpy arrays.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def serialize(obj):
        """Custom serializer for non-JSON types."""
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    # Convert config
    serializable_config = {}
    for key, value in config.items():
        try:
            serializable_config[key] = serialize(value)
        except Exception:
            serializable_config[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2, default=str)


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to JSON config file
    
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    return config


def format_number(value: float, precision: int = 2) -> str:
    """
    Format number with appropriate suffix (K, M, B).
    
    Args:
        value: Number to format
        precision: Decimal precision
    
    Returns:
        Formatted string
    """
    if abs(value) >= 1e9:
        return f"{value/1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_moving_average(values: list, window: int = 10) -> list:
    """
    Compute moving average of a list.
    
    Args:
        values: List of values
        window: Window size
    
    Returns:
        List of moving averages
    """
    if len(values) < window:
        return values
    
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i+1]))
    
    return result


def compute_exponential_moving_average(
    values: list,
    alpha: float = 0.1
) -> list:
    """
    Compute exponential moving average.
    
    Args:
        values: List of values
        alpha: Smoothing factor (0 < alpha <= 1)
    
    Returns:
        List of EMA values
    """
    if not values:
        return []
    
    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    
    return ema


def clip_gradients(model, max_norm: float = 1.0) -> float:
    """
    Clip gradients of a PyTorch model.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    
    Returns:
        Total gradient norm before clipping
    """
    try:
        import torch
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0
        
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return total_norm.item()
    except ImportError:
        return 0.0


def print_model_summary(model) -> None:
    """
    Print summary of a PyTorch model.
    
    Args:
        model: PyTorch model
    """
    try:
        import torch
        
        print("\nModel Summary:")
        print("=" * 60)
        
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            print(f"{name}: {param.shape} ({num_params:,} params)")
        
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    except ImportError:
        print("PyTorch not available")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training if no improvement
    for a specified number of episodes.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of episodes to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = None
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.best_value = None
        self.counter = 0
        self.should_stop = False


if __name__ == "__main__":
    # Demo: Helper functions
    print("=" * 60)
    print("HELPER FUNCTIONS DEMONSTRATION")
    print("=" * 60)
    
    # Set seed
    print("\n1. Setting random seed...")
    set_seed(42)
    print(f"  Random int: {random.randint(0, 100)}")
    print(f"  NumPy random: {np.random.rand():.4f}")
    
    # Device detection
    print("\n2. Device detection...")
    device = get_device()
    
    # Learning rate schedules
    print("\n3. Learning rate schedules...")
    linear = linear_schedule(1e-3, 1e-5)
    exponential = exponential_schedule(1e-3, 0.99)
    cosine = cosine_schedule(1e-3, 1e-5)
    
    print("  Progress | Linear   | Exponential | Cosine")
    print("  " + "-" * 50)
    for progress in [1.0, 0.75, 0.5, 0.25, 0.0]:
        print(f"  {progress:.2f}     | {linear(progress):.2e} | {exponential(progress):.2e} | {cosine(progress):.2e}")
    
    # Number formatting
    print("\n4. Number formatting...")
    for val in [123, 1234, 12345, 123456, 1234567, 12345678, 123456789]:
        print(f"  {val:>12} -> {format_number(val)}")
    
    # Time formatting
    print("\n5. Time formatting...")
    for secs in [45, 120, 3723, 7265]:
        print(f"  {secs:>5} seconds -> {format_time(secs)}")
    
    # Config I/O
    print("\n6. Config save/load...")
    test_config = {
        'learning_rate': 3e-4,
        'batch_size': 256,
        'network': [256, 256, 128],
        'seed': 42
    }
    
    config_path = "/home/claude/5G_ORAN_RL_Simulation/logs/test_config.json"
    save_config(test_config, config_path)
    loaded_config = load_config(config_path)
    print(f"  Saved and loaded config: {loaded_config}")
    
    # Early stopping
    print("\n7. Early stopping...")
    early_stop = EarlyStopping(patience=3, mode='max')
    values = [100, 105, 103, 104, 102, 101, 100]
    
    for i, val in enumerate(values):
        should_stop = early_stop(val)
        status = "STOP" if should_stop else "continue"
        print(f"  Episode {i+1}: value={val}, best={early_stop.best_value}, counter={early_stop.counter}, {status}")
        if should_stop:
            break
    
    # Moving averages
    print("\n8. Moving averages...")
    data = [10, 12, 15, 14, 18, 20, 19, 22, 25, 24]
    ma = compute_moving_average(data, window=3)
    ema = compute_exponential_moving_average(data, alpha=0.3)
    
    print("  Original | MA(3)  | EMA(0.3)")
    print("  " + "-" * 30)
    for i in range(len(data)):
        print(f"  {data[i]:8.2f} | {ma[i]:6.2f} | {ema[i]:6.2f}")
    
    print("\nAll helper functions working correctly!")
