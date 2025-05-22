"""
Simple position diversity module for chess reinforcement learning.

This module provides basic functions for generating diverse chess positions.
"""

import os
import random
import chess
import json

class PositionDiversity:
    """
    Provides basic chess positions for training.
    """

    def __init__(self):
        """Initialize the position diversity module."""
        # Initialize opening book
        self.opening_positions = self._create_opening_book()

        # Initialize endgame positions
        self.endgame_positions = self._create_endgame_positions()

    def _create_opening_book(self):
        """
        Create a collection of common opening positions from JSON file.

        Returns:
            list: List of chess.Board objects with opening positions
        """
        openings = []

        # Path to the JSON file with chess positions
        json_file_path = "chess_positions.json"

        # Check if the JSON file exists
        if not os.path.exists(json_file_path):
            print(f"Warning: Chess positions file {json_file_path} not found.")
            return openings

        try:
            # Load positions from JSON file
            with open(json_file_path, 'r') as f:
                positions_data = json.load(f)

            # Process opening positions
            if 'openings' in positions_data:
                for opening in positions_data['openings']:
                    try:
                        # Create board from FEN
                        fen = opening.get('fen')
                        if not fen:
                            continue

                        board = chess.Board(fen)

                        # Validate the position
                        if board.is_valid() and self._is_position_legal(board):
                            openings.append(board)
                        else:
                            print(f"Warning: Skipping invalid opening position: {opening.get('name', 'unnamed')}")
                    except Exception as e:
                        print(f"Error processing opening position {opening.get('name', 'unnamed')}: {e}")

            # Process middlegame positions if available
            if 'middlegames' in positions_data:
                for middlegame in positions_data['middlegames']:
                    try:
                        # Create board from FEN
                        fen = middlegame.get('fen')
                        if not fen:
                            continue

                        board = chess.Board(fen)

                        # Validate the position
                        if board.is_valid() and self._is_position_legal(board):
                            openings.append(board)
                        else:
                            print(f"Warning: Skipping invalid middlegame position: {middlegame.get('name', 'unnamed')}")
                    except Exception as e:
                        print(f"Error processing middlegame position {middlegame.get('name', 'unnamed')}: {e}")

        except Exception as e:
            print(f"Error loading chess positions from {json_file_path}: {e}")

        return openings

    def _create_endgame_positions(self):
        """
        Create a collection of common endgame positions from JSON file.

        Returns:
            list: List of chess.Board objects with endgame positions
        """
        endgames = []

        # Path to the JSON file with chess positions
        json_file_path = "chess_positions.json"

        # Check if the JSON file exists
        if not os.path.exists(json_file_path):
            print(f"Warning: Chess positions file {json_file_path} not found.")
            return endgames

        try:
            # Load positions from JSON file
            with open(json_file_path, 'r') as f:
                positions_data = json.load(f)

            # Process endgame positions
            if 'endgames' in positions_data:
                for endgame in positions_data['endgames']:
                    try:
                        # Create board from FEN
                        fen = endgame.get('fen')
                        if not fen:
                            continue

                        board = chess.Board(fen)

                        # Validate the position
                        if board.is_valid() and self._is_position_legal(board):
                            endgames.append(board)
                        else:
                            print(f"Warning: Skipping invalid endgame position: {endgame.get('name', 'unnamed')}")
                    except Exception as e:
                        print(f"Error processing endgame position {endgame.get('name', 'unnamed')}: {e}")

        except Exception as e:
            print(f"Error loading chess positions from {json_file_path}: {e}")

        return endgames

    def get_random_opening_position(self):
        """
        Get a random position from the opening book.

        Returns:
            chess.Board: A random opening position
        """
        if not self.opening_positions:
            # Return a new board if no openings are available
            return chess.Board()

        # Select a random opening position
        position = random.choice(self.opening_positions)
        return position.copy()

    def get_random_endgame_position(self):
        """
        Get a random position from the endgame tablebase.

        Returns:
            chess.Board: A random endgame position
        """
        if not self.endgame_positions:
            # Return a new board if no endgames are available
            return chess.Board()

        # Select a random endgame position
        position = random.choice(self.endgame_positions)
        return position.copy()

    # Position clustering methods removed to simplify the codebase

    def _is_position_legal(self, board):
        """
        Simple validation for chess positions.

        Args:
            board (chess.Board): The board position to validate

        Returns:
            bool: True if the position is legal, False otherwise
        """
        try:
            # Check for kings
            if not board.pieces(chess.KING, chess.WHITE) or not board.pieces(chess.KING, chess.BLACK):
                return False  # Both sides must have a king

            # Check if kings are adjacent (illegal position)
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)

            if white_king_square is not None and black_king_square is not None:
                king_distance = chess.square_distance(white_king_square, black_king_square)
                if king_distance < 2:
                    return False  # Kings cannot be adjacent

            return True

        except Exception as e:
            print(f"Error validating position: {e}")
            return False
