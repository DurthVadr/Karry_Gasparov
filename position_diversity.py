"""
Position diversity module for chess reinforcement learning.

This module provides functions for improving position diversity in training:
1. Opening book positions
2. Endgame tablebase positions
3. Position clustering for improved cache hit rates
"""

import os
import random
import chess
from collections import defaultdict
import json

class PositionDiversity:
    """
    Provides diverse chess positions for training.

    This class implements:
    1. Opening book positions from standard openings
    2. Endgame tablebase positions for common endgames
    3. Position clustering for improved cache hit rates
    """

    def __init__(self):
        """Initialize the position diversity module."""
        # Initialize opening book
        self.opening_positions = self._create_opening_book()

        # Initialize endgame positions
        self.endgame_positions = self._create_endgame_positions()

        # Initialize position clusters
        self.position_clusters = defaultdict(list)
        self.cluster_count = 0

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

    def add_to_cluster(self, board, data):
        """
        Add a position to the appropriate cluster.

        Args:
            board (chess.Board): The board position
            data: Data to associate with this position

        Returns:
            int: Cluster ID
        """
        # Note: We're not using the position hash directly, but using board similarity instead

        # Find the most similar existing cluster
        best_cluster = None
        best_similarity = 0

        for cluster_id, positions in self.position_clusters.items():
            if not positions:
                continue

            # Check similarity with the first position in the cluster
            similarity = self._calculate_similarity(board, positions[0][0])

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id

        # If we found a similar enough cluster, add to it
        if best_cluster is not None and best_similarity >= 0.85:
            self.position_clusters[best_cluster].append((board.copy(), data))
            return best_cluster

        # Otherwise, create a new cluster
        new_cluster_id = self.cluster_count
        self.cluster_count += 1
        self.position_clusters[new_cluster_id].append((board.copy(), data))
        return new_cluster_id

    def get_cluster_data(self, cluster_id):
        """
        Get all data from a specific cluster.

        Args:
            cluster_id (int): The cluster ID

        Returns:
            list: List of data items in the cluster
        """
        if cluster_id not in self.position_clusters:
            return []

        return [item[1] for item in self.position_clusters[cluster_id]]

    def _get_position_hash(self, board):
        """
        Create a simple hash of the board position.

        Args:
            board (chess.Board): The board position

        Returns:
            str: A hash string representing the position
        """
        # Use the FEN string without move counters as a hash
        fen_parts = board.fen().split(' ')
        return ' '.join(fen_parts[:4])

    def _is_position_legal(self, board):
        """
        Perform additional validation checks on a chess position.

        Args:
            board (chess.Board): The board position to validate

        Returns:
            bool: True if the position is legal, False otherwise
        """
        try:
            # Check for kings
            if not board.pieces(chess.KING, chess.WHITE) or not board.pieces(chess.KING, chess.BLACK):
                return False  # Both sides must have a king

            # Check for too many pieces of each type
            piece_counts = {
                chess.PAWN: 0,
                chess.KNIGHT: 0,
                chess.BISHOP: 0,
                chess.ROOK: 0,
                chess.QUEEN: 0,
                chess.KING: 0
            }

            for piece_type in piece_counts:
                white_count = len(board.pieces(piece_type, chess.WHITE))
                black_count = len(board.pieces(piece_type, chess.BLACK))
                piece_counts[piece_type] = white_count + black_count

            # Validate piece counts
            if piece_counts[chess.PAWN] > 16:
                return False  # Too many pawns
            if piece_counts[chess.KNIGHT] > 4:
                return False  # Too many knights
            if piece_counts[chess.BISHOP] > 4:
                return False  # Too many bishops
            if piece_counts[chess.ROOK] > 4:
                return False  # Too many rooks
            if piece_counts[chess.QUEEN] > 2:
                return False  # Too many queens
            if piece_counts[chess.KING] != 2:
                return False  # Must have exactly 2 kings

            # Check if kings are adjacent (illegal position)
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)

            if white_king_square is not None and black_king_square is not None:
                king_distance = chess.square_distance(white_king_square, black_king_square)
                if king_distance < 2:
                    return False  # Kings cannot be adjacent

            # Check if the side not to move is in check (illegal position)
            if board.turn == chess.WHITE and board.is_check():
                return False  # Black just moved but left white king in check
            elif board.turn == chess.BLACK and board.is_check():
                return False  # White just moved but left black king in check

            return True

        except Exception as e:
            print(f"Error validating position: {e}")
            return False

    def _calculate_similarity(self, board1, board2):
        """
        Calculate similarity between two board positions.

        Args:
            board1 (chess.Board): First board position
            board2 (chess.Board): Second board position

        Returns:
            float: Similarity score (0-1)
        """
        # Convert boards to piece maps
        pieces1 = board1.piece_map()
        pieces2 = board2.piece_map()

        # Count matching pieces
        matches = 0
        total = max(len(pieces1), len(pieces2))

        if total == 0:
            return 1.0  # Empty boards are identical

        for square, piece1 in pieces1.items():
            if square in pieces2 and pieces2[square] == piece1:
                matches += 1

        return matches / total
