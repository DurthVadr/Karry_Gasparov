"""
Visualization module for chess model training metrics.

This module provides functions for visualizing the metrics collected
during chess model training, helping to understand model improvement
over time.
"""

import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_training_metrics(metrics, output_path=None):
    """
    Plot the training metrics over time.

    Args:
        metrics (SimpleMetrics): SimpleMetrics instance with training data
        output_path (str, optional): Path to save the plot
    """
    try:
        # Get metrics data
        loss_values = list(metrics.loss_values)
        reward_values = list(metrics.reward_values)
        q_values = list(metrics.q_values)
        exploration_rates = list(metrics.exploration_rates)

        # Check if we have data to plot
        if not any([loss_values, reward_values, q_values, exploration_rates]):
            print("No metrics data available to plot")
            return

        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot loss
        if loss_values:
            ax = axs[0, 0]
            ax.plot(loss_values, marker='o', linestyle='-', color='red', label='Loss')
            ax.set_title("Loss Over Time")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Loss Value")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Plot reward
        if reward_values:
            ax = axs[0, 1]
            ax.plot(reward_values, marker='o', linestyle='-', color='green', label='Reward')
            ax.set_title("Reward Over Time")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward Value")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Plot Q-values
        if q_values:
            ax = axs[1, 0]
            ax.plot(q_values, marker='o', linestyle='-', color='blue', label='Q-Value')
            ax.set_title("Q-Value Over Time")
            ax.set_xlabel("Step")
            ax.set_ylabel("Q-Value")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Plot exploration rate
        if exploration_rates:
            ax = axs[1, 1]
            ax.plot(exploration_rates, marker='o', linestyle='-', color='purple', label='Exploration Rate')
            ax.set_title("Exploration Rate Over Time")
            ax.set_xlabel("Step")
            ax.set_ylabel("Epsilon")
            ax.set_ylim([0, 1])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        plt.tight_layout()

        # Save the plot if output_path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Training metrics plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Error plotting training metrics: {e}")
        import traceback
        traceback.print_exc()

def plot_performance_summary(metrics, output_path=None):
    """
    Create a summary plot of the model's performance metrics.

    Args:
        metrics (SimpleMetrics): SimpleMetrics instance with training data
        output_path (str, optional): Path to save the plot
    """
    try:
        # Calculate summary statistics
        avg_loss = sum(metrics.loss_values) / max(len(metrics.loss_values), 1) if metrics.loss_values else 0
        avg_reward = sum(metrics.reward_values) / max(len(metrics.reward_values), 1) if metrics.reward_values else 0
        avg_q = sum(metrics.q_values) / max(len(metrics.q_values), 1) if metrics.q_values else 0

        # Calculate positions per second
        elapsed = time.time() - metrics.start_time
        pos_per_sec = metrics.positions_seen / elapsed if elapsed > 0 else 0

        # Create a bar chart for the summary metrics
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define metrics to show
        metric_names = ['Loss', 'Reward', 'Q-Value', 'Positions/sec']
        metric_values = [avg_loss, avg_reward, avg_q, pos_per_sec]

        # Create bar chart
        bars = ax.bar(metric_names, metric_values, color=['red', 'green', 'blue', 'purple'])

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Add title and labels
        ax.set_title('Model Performance Summary')
        ax.set_ylabel('Value')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save the plot if output_path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Performance summary plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Error creating performance summary: {e}")
        import traceback
        traceback.print_exc()

def plot_training_progress(metrics, output_path=None):
    """
    Plot overall training progress including loss, reward, and speed.

    Args:
        metrics (SimpleMetrics): SimpleMetrics instance with training data
        output_path (str, optional): Path to save the plot
    """
    try:
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Plot loss trend
        if metrics.loss_values:
            ax = axs[0]
            # Use moving average to smooth the curve
            window_size = min(10, len(metrics.loss_values))
            if window_size > 1:
                loss_smooth = np.convolve(list(metrics.loss_values),
                                         np.ones(window_size)/window_size,
                                         mode='valid')
                ax.plot(loss_smooth, color='red', linewidth=2, label='Loss (Smoothed)')
            ax.plot(list(metrics.loss_values), color='red', alpha=0.3, label='Loss (Raw)')
            ax.set_title("Loss Trend")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Loss Value")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Plot reward trend
        if metrics.reward_values:
            ax = axs[1]
            # Use moving average to smooth the curve
            window_size = min(5, len(metrics.reward_values))
            if window_size > 1:
                reward_smooth = np.convolve(list(metrics.reward_values),
                                           np.ones(window_size)/window_size,
                                           mode='valid')
                ax.plot(reward_smooth, color='green', linewidth=2, label='Reward (Smoothed)')
            ax.plot(list(metrics.reward_values), color='green', alpha=0.3, label='Reward (Raw)')
            ax.set_title("Reward Trend")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward Value")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Plot exploration rate
        if metrics.exploration_rates:
            ax = axs[2]
            ax.plot(list(metrics.exploration_rates), color='purple', label='Exploration Rate')
            ax.set_title("Exploration Rate Decay")
            ax.set_xlabel("Step")
            ax.set_ylabel("Epsilon")
            ax.set_ylim([0, 1])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        plt.tight_layout()

        # Save the plot if output_path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Training progress plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Error plotting training progress: {e}")
        import traceback
        traceback.print_exc()
