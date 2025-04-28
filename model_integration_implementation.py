import chess
import random
import os
import sys
import torch
import numpy as np

# Import the DQN and helper functions from drl_agent.py
from drl_agent import DQN, board_to_tensor, create_move_mask, ChessAgent

class RandomModel:
    """A simple model that makes random moves"""
    def __init__(self):
        self.name = "Random Model"
    
    def get_move(self, board):
        """Return a random legal move"""
        return random.choice(list(board.legal_moves))

class DRLChessModel:
    """Wrapper for the deep reinforcement learning chess model"""
    def __init__(self, model_path=None):
        self.name = "Deep Reinforcement Learning Chess Bot"
        self.agent = ChessAgent(model_path=model_path)
        
        # If model_path is provided but doesn't exist, print a warning
        if model_path and not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    def get_move(self, board):
        """Get a move from the DRL agent"""
        return self.agent.select_move(board)

class ModelIntegration:
    """Class to handle integration with different chess models"""
    def __init__(self):
        self.current_model = RandomModel()
    
    def load_model(self, model_path):
        """Load a model from a file"""
        # Check file extension to determine model type
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            # PyTorch model (our DRL model)
            try:
                self.current_model = DRLChessModel(model_path=model_path)
                print(f"Loaded DRL model from {model_path}")
                return self.current_model
            except Exception as e:
                print(f"Error loading DRL model: {str(e)}")
                # Fall back to random model
                self.current_model = RandomModel()
                return self.current_model
        elif model_path.endswith('.h5'):
            # TensorFlow/Keras model (not implemented in this version)
            print("TensorFlow/Keras models are not supported in this version.")
            # Fall back to random model
            self.current_model = RandomModel()
            return self.current_model
        else:
            # Unknown model type
            print(f"Unknown model type for file {model_path}")
            # Fall back to random model
            self.current_model = RandomModel()
            return self.current_model
    
    def get_move(self, board):
        """Get a move from the current model"""
        return self.current_model.get_move(board)

# Example usage
if __name__ == "__main__":
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
    
    # Get a move from the model
    move = model_integration.get_move(board)
    
    # Print the move
    print(f"Model: {model.name}")
    print(f"Move: {move.uci()}")
