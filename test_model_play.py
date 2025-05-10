"""
Test script to play a few moves with the model
"""

import os
import torch
import sys
from drl_agent import DQN, ChessAgent
import chess

def test_model_play(model_path, num_moves=5):
    """Test if a model can play a few moves"""
    print(f"Testing model play from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} does not exist!")
        return False
    
    # Create a board and agent
    board = chess.Board()
    agent = ChessAgent(model_path=model_path)
    print("Agent created with model")
    
    # Play a few moves
    for i in range(num_moves):
        print(f"\nMove {i+1}:")
        print(board.unicode())
        
        # Get a move from the agent
        move = agent.select_move(board)
        print(f"Selected move: {move.uci()}")
        
        # Make the move
        board.push(move)
    
    # Show final board state
    print("\nFinal board state:")
    print(board.unicode())
    
    return True

def test_model_vs_random(model_path, num_moves=10):
    """Test model playing against random moves"""
    print(f"Testing model vs random from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} does not exist!")
        return False
    
    # Create a board and agent
    board = chess.Board()
    agent = ChessAgent(model_path=model_path)
    print("Agent created with model")
    
    # Play alternating moves
    for i in range(num_moves):
        print(f"\nMove {i+1}:")
        print(board.unicode())
        
        if board.is_game_over():
            print("Game over!")
            break
        
        # Model plays as black (odd moves)
        if i % 2 == 1:
            move = agent.select_move(board)
            print(f"Model move: {move.uci()}")
        # Random plays as white (even moves)
        else:
            import random
            move = random.choice(list(board.legal_moves))
            print(f"Random move: {move.uci()}")
        
        # Make the move
        board.push(move)
    
    # Show final board state
    print("\nFinal board state:")
    print(board.unicode())
    
    return True

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default path
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/model_pgn_checkpoint.pt"
    
    # Test model playing a few moves
    print("=== Testing model playing solo ===")
    test_model_play(model_path)
    
    print("\n\n=== Testing model vs random ===")
    test_model_vs_random(model_path)
