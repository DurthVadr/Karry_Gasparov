import chess
import random
import os
import sys
import time

# Import the model integration
from model_integration import ModelIntegration

def test_model_without_gui():
    """Test the chess model without GUI"""
    print("Testing chess model without GUI...")
    
    # Create a new board
    board = chess.Board()
    
    # Create model integration
    model_integration = ModelIntegration()
    
    # Try to load a model (if it exists)
    model_path = "/models/model_final.pt"
    if os.path.exists(model_path):
        model = model_integration.load_model(model_path)
    else:
        print(f"Model file {model_path} not found. Using random model.")
        model = model_integration.current_model
    
    print(f"Using model: {model.name}")
    print("\nInitial board:")
    print(board.unicode())
    
    # Play a few moves
    for i in range(10):
        if board.is_game_over():
            result = board.result()
            print(f"\nGame over! Result: {result}")
            break
        
        print(f"\nMove {i+1} - {'White' if board.turn == chess.WHITE else 'Black'} to play")
        
        # Get a move from the model
        start_time = time.time()
        move = model_integration.get_move(board)
        end_time = time.time()
        
        # Print the move and thinking time
        print(f"Move: {move.uci()} (thinking time: {end_time - start_time:.2f}s)")
        
        # Make the move
        board.push(move)
        print(board.unicode())
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_model_without_gui()
