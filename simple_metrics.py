"""
Simple, lightweight metrics for chess model training.

This module provides extremely basic metrics that can be displayed during training
with minimal computational overhead.
"""

import time
from collections import deque

class SimpleMetrics:
    """
    Extremely lightweight metrics for real-time monitoring during training.
    
    These metrics:
    1. Add virtually no computational overhead
    2. Use only data already computed during training
    3. Provide basic indicators of model improvement
    """
    
    def __init__(self, window_size=100):
        """
        Initialize the simple metrics tracker.
        
        Args:
            window_size (int): Number of data points to keep in the sliding window
        """
        self.window_size = window_size
        
        # Use deques for efficient sliding window
        self.loss_values = deque(maxlen=window_size)
        self.reward_values = deque(maxlen=window_size)
        self.q_values = deque(maxlen=window_size)
        self.exploration_rates = deque(maxlen=window_size)
        
        # Simple counters
        self.positions_seen = 0
        self.optimization_steps = 0
        self.start_time = time.time()
    
    def update_loss(self, loss_value):
        """Update loss metric."""
        if loss_value is not None:
            self.loss_values.append(loss_value)
            self.optimization_steps += 1
    
    def update_reward(self, reward):
        """Update reward metric."""
        self.reward_values.append(reward)
    
    def update_q_value(self, q_value):
        """Update Q-value metric."""
        self.q_values.append(q_value)
    
    def update_exploration(self, epsilon):
        """Update exploration rate metric."""
        self.exploration_rates.append(epsilon)
    
    def increment_positions(self, count=1):
        """Increment the positions counter."""
        self.positions_seen += count
    
    def get_metrics_string(self):
        """
        Get a simple string representation of current metrics.
        
        Returns:
            str: Formatted metrics string
        """
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        # Calculate positions per second
        pos_per_sec = self.positions_seen / elapsed if elapsed > 0 else 0
        
        # Calculate average metrics
        avg_loss = sum(self.loss_values) / max(len(self.loss_values), 1) if self.loss_values else 0
        avg_reward = sum(self.reward_values) / max(len(self.reward_values), 1) if self.reward_values else 0
        avg_q = sum(self.q_values) / max(len(self.q_values), 1) if self.q_values else 0
        
        # Get current values (most recent)
        current_loss = self.loss_values[-1] if self.loss_values else 0
        current_reward = self.reward_values[-1] if self.reward_values else 0
        current_q = self.q_values[-1] if self.q_values else 0
        current_epsilon = self.exploration_rates[-1] if self.exploration_rates else 0
        
        # Format the metrics string
        metrics_str = (
            f"\n--- TRAINING METRICS ---\n"
            f"Time: {hours:02d}:{minutes:02d}:{seconds:02d} | "
            f"Positions: {self.positions_seen} ({pos_per_sec:.1f}/s)\n"
            f"Loss: {current_loss:.4f} (avg: {avg_loss:.4f}) | "
            f"Reward: {current_reward:.2f} (avg: {avg_reward:.2f})\n"
            f"Q-value: {current_q:.3f} (avg: {avg_q:.3f}) | "
            f"Exploration: {current_epsilon:.3f}\n"
            f"------------------------"
        )
        
        return metrics_str
    
    def print_metrics(self):
        """Print current metrics to the console."""
        print(self.get_metrics_string())
