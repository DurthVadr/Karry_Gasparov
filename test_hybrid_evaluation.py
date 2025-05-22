"""
Test script to measure self-play speed with hybrid Stockfish evaluation.

This script runs a benchmark to measure the positions per second during self-play
with the hybrid evaluation approach (1% normal, 10% quality episodes).
"""

import os
import time
import torch
import argparse
import numpy as np
from main import ChessTrainer

def run_hybrid_test(stockfish_path, batch_size=512, num_episodes=100):
    """
    Run a test of the hybrid evaluation approach.
    
    Args:
        stockfish_path (str): Path to Stockfish executable
        batch_size (int): Batch size to use for training
        num_episodes (int): Number of episodes to run
        
    Returns:
        dict: Dictionary with test results
    """
    print(f"\n=== Testing Hybrid Evaluation with batch_size={batch_size} ===")
    
    # Create a trainer with the specified parameters
    trainer = ChessTrainer(stockfish_path=stockfish_path, gpu_type="rtx_4070")
    
    # Ensure hybrid evaluation is enabled
    trainer.reward_calculator.hybrid_eval_enabled = True
    trainer.reward_calculator.low_freq_rate = 0.01  # 1% for normal episodes
    trainer.reward_calculator.high_freq_rate = 0.1  # 10% for quality episodes
    trainer.reward_calculator.high_freq_interval = 10  # Every 10th episode is quality
    
    # Run self-play training
    start_time = time.time()
    
    # Run self-play training
    trainer.train_self_play(
        num_episodes=num_episodes,
        stockfish_levels=[1],  # Use only level 1 for speed testing
        batch_size=batch_size,
        save_interval=num_episodes + 1,  # Don't save during test
        eval_interval=num_episodes + 1,  # Don't evaluate during test
        target_level=1
    )
    
    # Calculate elapsed time and positions per second
    elapsed_time = time.time() - start_time
    total_positions = sum(trainer.training_stats['episode_lengths'])
    positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0
    
    # Calculate average positions per second for normal and quality episodes
    normal_episodes = [i for i in range(num_episodes) if i % 10 != 0]
    quality_episodes = [i for i in range(num_episodes) if i % 10 == 0]
    
    normal_positions = sum([trainer.training_stats['episode_lengths'][i] for i in normal_episodes])
    quality_positions = sum([trainer.training_stats['episode_lengths'][i] for i in quality_episodes])
    
    normal_time = elapsed_time * (len(normal_episodes) / num_episodes)
    quality_time = elapsed_time * (len(quality_episodes) / num_episodes)
    
    normal_speed = normal_positions / normal_time if normal_time > 0 else 0
    quality_speed = quality_positions / quality_time if quality_time > 0 else 0
    
    print(f"Completed {num_episodes} episodes with {total_positions} positions in {elapsed_time:.2f} seconds")
    print(f"Overall speed: {positions_per_second:.2f} positions/second")
    print(f"Normal episodes (1%): {normal_speed:.2f} positions/second")
    print(f"Quality episodes (10%): {quality_speed:.2f} positions/second")
    
    # Clean up
    trainer.close()
    
    return {
        "overall_speed": positions_per_second,
        "normal_speed": normal_speed,
        "quality_speed": quality_speed,
        "total_positions": total_positions,
        "elapsed_time": elapsed_time
    }

def main():
    """Main function to run the hybrid evaluation test."""
    parser = argparse.ArgumentParser(description="Test hybrid Stockfish evaluation approach")
    parser.add_argument("--stockfish_path", default="/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish",
                        help="Path to Stockfish executable")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to run for the test")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size to use for training")
    
    args = parser.parse_args()
    
    # Print system information
    print("\n=== System Information ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run the hybrid evaluation test
    results = run_hybrid_test(
        args.stockfish_path,
        batch_size=args.batch_size,
        num_episodes=args.num_episodes
    )
    
    # Print summary
    print("\n=== Hybrid Evaluation Test Results ===")
    print(f"Overall Speed: {results['overall_speed']:.2f} positions/second")
    print(f"Normal Episodes (1%): {results['normal_speed']:.2f} positions/second")
    print(f"Quality Episodes (10%): {results['quality_speed']:.2f} positions/second")
    print(f"Speedup Ratio: {results['normal_speed'] / results['quality_speed']:.2f}x")

if __name__ == "__main__":
    main()
