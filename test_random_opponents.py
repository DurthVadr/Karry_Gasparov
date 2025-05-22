"""
Test script for random move opponents and position diversity.

This script tests the random move opponents and position diversity improvements.
"""

import os
import torch
import chess
import random
import time
import argparse
from drl_agent import DQN, board_to_tensor, create_move_mask
from position_diversity import PositionDiversity
from position_clustering import PositionClusterCache
from reward import RewardCalculator
from training import SelfPlayTrainer
from hyperparameters import get_optimized_hyperparameters

class TestTrainer:
    """Simple trainer class for testing."""
    
    def __init__(self, model_dir="models", stockfish_path=None):
        """Initialize the test trainer."""
        self.model_dir = model_dir
        self.stockfish_path = stockfish_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Get optimized hyperparameters
        self.hyperparams = get_optimized_hyperparameters()
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            stockfish_path=stockfish_path,
            use_async=True,
            num_workers=4,
            use_position_clustering=True,
            cache_size=1000,
            similarity_threshold=0.85
        )
        
        # Training stats
        self.training_stats = {}
        
        # Other parameters
        self.batch_size = 64
        self.max_moves = 100
        self.early_stopping_no_progress = 30

def test_random_opponents():
    """Test random move opponents."""
    print("\n=== Testing Random Move Opponents ===")
    
    # Create test trainer
    trainer = TestTrainer()
    
    # Create self-play trainer
    self_play_trainer = SelfPlayTrainer(trainer)
    
    # Print initial random move percentage
    print(f"Initial random move percentage: {self_play_trainer.current_random_move_pct:.2f}")
    
    # Test random move selection
    board = chess.Board()
    state = board_to_tensor(board, trainer.device)
    mask = create_move_mask(board, trainer.device)
    
    # Test with different random percentages
    for pct in [1.0, 0.5, 0.0]:
        self_play_trainer.current_random_move_pct = pct
        print(f"\nTesting with random move percentage: {pct:.2f}")
        
        # Make 10 moves
        random_count = 0
        model_count = 0
        
        for _ in range(100):
            # Get legal moves
            legal_moves = list(board.legal_moves)
            
            # Decide whether to make a random move
            if random.random() < self_play_trainer.current_random_move_pct:
                # Random move
                random_count += 1
            else:
                # Model move
                model_count += 1
        
        print(f"Random moves: {random_count}, Model moves: {model_count}")
        print(f"Actual random percentage: {random_count / (random_count + model_count):.2f}")
    
    print("\nRandom move opponents test completed successfully!")

def test_position_diversity():
    """Test position diversity."""
    print("\n=== Testing Position Diversity ===")
    
    # Create position diversity module
    position_diversity = PositionDiversity()
    
    # Test opening book
    print("\nTesting opening book positions:")
    for _ in range(3):
        board = position_diversity.get_random_opening_position()
        print(f"Opening position: {board.fen()}")
        print(f"Move number: {board.fullmove_number}")
        print(f"To move: {'White' if board.turn == chess.WHITE else 'Black'}")
        print(f"Piece count: {len(board.piece_map())}")
        print()
    
    # Test endgame tablebase
    print("\nTesting endgame tablebase positions:")
    for _ in range(3):
        board = position_diversity.get_random_endgame_position()
        print(f"Endgame position: {board.fen()}")
        print(f"Move number: {board.fullmove_number}")
        print(f"To move: {'White' if board.turn == chess.WHITE else 'Black'}")
        print(f"Piece count: {len(board.piece_map())}")
        print()
    
    print("Position diversity test completed successfully!")

def test_position_clustering():
    """Test position clustering for cache."""
    print("\n=== Testing Position Clustering Cache ===")
    
    # Create position cluster cache
    cache = PositionClusterCache(cache_size=1000, similarity_threshold=0.85)
    
    # Create some test positions
    positions = []
    
    # Starting position
    positions.append(chess.Board())
    
    # Similar positions with small differences
    board = chess.Board()
    board.push_san("e4")
    positions.append(board.copy())
    
    board.push_san("e5")
    positions.append(board.copy())
    
    board.push_san("Nf3")
    positions.append(board.copy())
    
    # Add positions to cache
    for i, board in enumerate(positions):
        cache.put(board, i * 100)  # Use index * 100 as dummy evaluation
    
    # Print cache stats
    print("\nCache stats after adding positions:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test cache hits
    print("\nTesting cache hits:")
    
    # Exact match
    board = chess.Board()
    value, hit_type = cache.get(board)
    print(f"Exact match - Value: {value}, Hit type: {hit_type}")
    
    # Similar position (not exact match)
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")  # This move makes it different from cached positions
    value, hit_type = cache.get(board)
    print(f"Similar position - Value: {value}, Hit type: {hit_type}")
    
    # Very different position
    board = chess.Board()
    board.push_san("d4")  # Different opening
    value, hit_type = cache.get(board)
    print(f"Different position - Value: {value}, Hit type: {hit_type}")
    
    print("\nPosition clustering cache test completed successfully!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test random move opponents and position diversity")
    parser.add_argument("--test", choices=["random", "diversity", "clustering", "all"], 
                        default="all", help="Which test to run")
    args = parser.parse_args()
    
    if args.test == "random" or args.test == "all":
        test_random_opponents()
    
    if args.test == "diversity" or args.test == "all":
        test_position_diversity()
    
    if args.test == "clustering" or args.test == "all":
        test_position_clustering()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
