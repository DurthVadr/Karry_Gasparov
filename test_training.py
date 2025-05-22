"""
Test script for training with random move opponents.

This script tests the training process with random move opponents
and position diversity improvements.
"""

import os
import torch
import chess
import random
import time
import argparse
from drl_agent import DQN
from position_diversity import PositionDiversity
from position_clustering import PositionClusterCache
from reward import RewardCalculator
from training import SelfPlayTrainer
from hyperparameters import get_optimized_hyperparameters
from main import ChessTrainer

def test_training(num_episodes=50, stockfish_path=None):
    """
    Test training with random move opponents.
    
    Args:
        num_episodes (int): Number of episodes to train
        stockfish_path (str): Path to Stockfish executable
    """
    print("\n=== Testing Training with Random Move Opponents ===")
    
    # Create trainer
    trainer = ChessTrainer(stockfish_path=stockfish_path)
    
    # Override hyperparameters for testing
    trainer.hyperparams['curriculum']['random_move_pct'] = 0.25
    trainer.hyperparams['curriculum']['random_win_rate_threshold'] = 0.70
    trainer.hyperparams['self_play']['use_opening_book'] = True
    trainer.hyperparams['self_play']['use_tablebase'] = True
    trainer.hyperparams['async_eval']['use_position_clustering'] = True
    
    # Initialize reward calculator with position clustering
    trainer.reward_calculator = RewardCalculator(
        stockfish_path=stockfish_path,
        use_async=True,
        num_workers=4,
        stockfish_eval_frequency=0.1,  # Lower frequency for faster testing
        use_position_clustering=True,
        cache_size=1000,
        similarity_threshold=0.85
    )
    
    # Run training
    print(f"\nRunning training for {num_episodes} episodes...")
    trainer.train_self_play(
        num_episodes=num_episodes,
        stockfish_levels=[1],  # Only test against Stockfish level 1
        batch_size=32,
        save_interval=25,
        eval_interval=50,
        target_level=1
    )
    
    print("\nTraining test completed successfully!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test training with random move opponents")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to train")
    parser.add_argument("--stockfish", type=str, default=None, help="Path to Stockfish executable")
    args = parser.parse_args()
    
    test_training(num_episodes=args.episodes, stockfish_path=args.stockfish)

if __name__ == "__main__":
    main()
