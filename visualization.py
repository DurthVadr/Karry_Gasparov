"""
Visualization module for chess reinforcement learning.

This module provides functions for visualizing training progress and model performance.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for plotting
import matplotlib.pyplot as plt

def plot_training_progress(training_stats, output_path=None):
    """
    Plot training progress metrics with enhanced visualizations.
    
    Args:
        training_stats (dict): Dictionary containing training statistics
        output_path (str, optional): Path to save the plot
    """
    try:
        # Determine number of subplots needed
        num_plots = 4
        if 'win_rates' in training_stats and training_stats['win_rates']:
            num_plots = 5
        
        # Create figure with subplots
        _, axs = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))
        
        # Plot episode rewards
        axs[0].plot(training_stats['episode_rewards'], color='blue')
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        
        # Add moving average for rewards
        if len(training_stats['episode_rewards']) > 10:
            window_size = min(50, len(training_stats['episode_rewards']) // 5)
            rewards_avg = [sum(training_stats['episode_rewards'][max(0, i-window_size):i+1]) /
                          min(window_size, i+1) for i in range(len(training_stats['episode_rewards']))]
            axs[0].plot(rewards_avg, color='red', linestyle='--', label=f'{window_size}-episode moving avg')
            axs[0].legend()
        
        # Plot episode lengths
        axs[1].plot(training_stats['episode_lengths'], color='green')
        axs[1].set_title('Episode Lengths')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Length')
        
        # Plot losses
        if training_stats['losses']:
            # Use log scale for losses to better visualize changes
            axs[2].plot(training_stats['losses'], color='orange', alpha=0.5)
            axs[2].set_title('Training Loss')
            axs[2].set_xlabel('Optimization Step')
            axs[2].set_ylabel('Loss')
            
            # Add smoothed loss curve
            if len(training_stats['losses']) > 100:
                window_size = min(500, len(training_stats['losses']) // 10)
                losses_avg = []
                for i in range(len(training_stats['losses'])):
                    start_idx = max(0, i - window_size)
                    losses_avg.append(sum(training_stats['losses'][start_idx:i+1]) / (i - start_idx + 1))
                axs[2].plot(losses_avg, color='red', linewidth=2, label=f'{window_size}-step moving avg')
                axs[2].legend()
        
        # Plot average Q-values
        if 'avg_q_values' in training_stats and training_stats['avg_q_values']:
            axs[3].plot(training_stats['avg_q_values'], color='purple')
            axs[3].set_title('Average Q-Values')
            axs[3].set_xlabel('Step')
            axs[3].set_ylabel('Q-Value')
        
        # Plot win rates if available
        if 'win_rates' in training_stats and training_stats['win_rates']:
            axs[4].plot(training_stats['win_rates'], color='brown', marker='o', linestyle='-')
            axs[4].set_title('Win Rate Against Stockfish')
            axs[4].set_xlabel('Evaluation')
            axs[4].set_ylabel('Win Rate')
            axs[4].set_ylim([0, 1])
            axs[4].axhline(y=0.6, color='green', linestyle='--', label='Level-up Threshold')
            axs[4].legend()
        
        plt.tight_layout()
        
        # Save the plot if output_path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Enhanced training progress plot saved to {output_path}")
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        print(f"Error plotting training progress: {e}")
        import traceback
        traceback.print_exc()
        # Continue without plotting if matplotlib is not available

def plot_evaluation_results(results, output_path=None):
    """
    Plot model evaluation results against different Stockfish levels.
    
    Args:
        results (dict): Dictionary containing evaluation results
        output_path (str, optional): Path to save the plot
    """
    try:
        # Extract data from results
        levels = sorted(results.keys())
        scores = [results[level]['score'] for level in levels]
        win_rates = [results[level]['win_rate'] for level in levels]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot scores
        ax1.plot(levels, scores, marker='o', color='blue', linestyle='-')
        ax1.set_title('Model Performance Against Stockfish')
        ax1.set_xlabel('Stockfish Level')
        ax1.set_ylabel('Score (wins + 0.5*draws)')
        ax1.set_ylim([0, 1])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add threshold line for competitive play
        ax1.axhline(y=0.45, color='green', linestyle='--', label='Competitive Threshold (45%)')
        ax1.legend()
        
        # Plot win rates
        ax2.plot(levels, win_rates, marker='s', color='red', linestyle='-')
        ax2.set_title('Win Rate Against Stockfish')
        ax2.set_xlabel('Stockfish Level')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim([0, 1])
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the plot if output_path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Evaluation results plot saved to {output_path}")
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        print(f"Error plotting evaluation results: {e}")
        import traceback
        traceback.print_exc()
