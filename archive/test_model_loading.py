"""
Test script to verify model loading
"""

import os
import torch
import sys
from drl_agent import DQN, ChessAgent
import chess

def test_model_loading(model_path):
    """Test if a model can be loaded correctly"""
    print(f"Testing model loading from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} does not exist!")
        return False
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    print(f"Model file size: {file_size:.2f} MB")
    
    # Try to load the model
    try:
        # Create a DQN instance
        model = DQN()
        print("Created DQN instance")
        
        # Load the state dict
        print(f"Loading state dict from {model_path}...")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"State dict loaded, keys: {list(state_dict.keys())}")
        
        # Check if the state dict matches the model architecture
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
        
        # Test the model with a simple board
        board = chess.Board()
        agent = ChessAgent(model_path=model_path)
        print("Agent created")
        
        # Get a move
        move = agent.select_move(board)
        print(f"Selected move: {move.uci()}")
        
        return True
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default path
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/model_pgn_checkpoint.pt"
    success = test_model_loading(model_path)
    print(f"Model loading {'successful' if success else 'failed'}")
