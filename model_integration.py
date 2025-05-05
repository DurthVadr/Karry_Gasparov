"""
Model Integration Module

This module provides an interface between the chess GUI and various chess models.
It handles loading different types of models and provides a consistent interface
for getting moves from them.

The main components are:
1. RandomModel: A fallback model that makes random legal moves
2. DRLChessModel: A wrapper for the deep reinforcement learning chess model
3. ModelIntegration: The main class that handles loading and using models

Usage:
    model_integration = ModelIntegration()
    model = model_integration.load_model("path/to/model.pt")
    move = model.get_move(board)
"""

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
    """
    Handles integration between the chess GUI and various chess models.

    This class provides a unified interface for loading and using different types
    of chess models. It supports loading PyTorch models (.pt, .pth) and provides
    a fallback to a random move generator if loading fails.

    Attributes:
        current_model: The currently loaded chess model
    """
    def __init__(self):
        """Initialize with a random model as default"""
        self.current_model = RandomModel()

    def load_model(self, model_path):
        """
        Load a chess model from a file.

        Args:
            model_path (str): Path to the model file

        Returns:
            The loaded model object

        The method determines the model type based on the file extension:
        - .pt/.pth: PyTorch models (DRL chess model)
        - .h5: TensorFlow/Keras models (not currently supported)

        If loading fails, it falls back to a random model.
        """
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
        """
        Get a move from the current model for the given board position.

        Args:
            board (chess.Board): The current chess position

        Returns:
            chess.Move: A legal chess move
        """
        return self.current_model.get_move(board)
