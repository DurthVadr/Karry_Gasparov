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
import chess.pgn
import numpy as np
from collections import defaultdict
import io

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
        Create a collection of common opening positions.
        
        Returns:
            list: List of chess.Board objects with opening positions
        """
        openings = []
        
        # Common openings in PGN format
        opening_pgns = [
            # Open games
            "1. e4 e5",  # King's Pawn Opening
            "1. e4 e5 2. Nf3 Nc6 3. Bb5",  # Ruy Lopez
            "1. e4 e5 2. Nf3 Nc6 3. Bc4",  # Italian Game
            "1. e4 e5 2. Nf3 Nc6 3. d4",  # Scotch Game
            "1. e4 e5 2. f4",  # King's Gambit
            
            # Semi-open games
            "1. e4 c5",  # Sicilian Defense
            "1. e4 c6",  # Caro-Kann Defense
            "1. e4 e6",  # French Defense
            "1. e4 d6",  # Pirc Defense
            "1. e4 d5",  # Scandinavian Defense
            
            # Closed games
            "1. d4 d5",  # Queen's Pawn Game
            "1. d4 d5 2. c4",  # Queen's Gambit
            "1. d4 Nf6",  # Indian Defense
            "1. d4 Nf6 2. c4 e6",  # Queen's Indian
            "1. d4 Nf6 2. c4 g6",  # King's Indian
            
            # Flank openings
            "1. c4",  # English Opening
            "1. Nf3",  # Réti Opening
            "1. f4",  # Bird's Opening
            "1. b3",  # Larsen's Opening
        ]
        
        # Convert PGN strings to board positions
        for pgn_str in opening_pgns:
            pgn = io.StringIO(pgn_str)
            game = chess.pgn.read_game(pgn)
            board = game.board()
            
            # Apply all moves to get the position
            for move in game.mainline_moves():
                board.push(move)
            
            # Add the position to our collection
            openings.append(board.copy())
        
        return openings
    
    def _create_endgame_positions(self):
        """
        Create a collection of common endgame positions.
        
        Returns:
            list: List of chess.Board objects with endgame positions
        """
        endgames = []
        
        # Common endgame positions in FEN format
        endgame_fens = [
            # King and pawn vs king
            "4k3/4P3/4K3/8/8/8/8/8 w - - 0 1",  # White king and pawn vs black king
            "8/8/8/8/8/4k3/4p3/4K3 b - - 0 1",  # Black king and pawn vs white king
            
            # Rook endgames
            "8/8/8/8/8/2k5/2p5/2K1R3 w - - 0 1",  # Lucena position
            "8/8/8/8/8/2k5/8/2K1R3 w - - 0 1",  # Philidor position
            
            # Queen vs pawn
            "8/8/8/8/8/2k5/2p5/2K1Q3 w - - 0 1",  # Queen vs pawn
            
            # Bishop and knight checkmate
            "8/8/8/8/8/2k5/8/2K1BN2 w - - 0 1",  # Bishop and knight vs king
            
            # Two bishops checkmate
            "8/8/8/8/8/2k5/8/2K1BB2 w - - 0 1",  # Two bishops vs king
            
            # Rook and bishop vs rook
            "8/8/8/8/8/2k5/8/2K1RBr1 w - - 0 1",  # Rook and bishop vs rook
            
            # Queen vs rook
            "8/8/8/8/8/2k5/8/2K1Qr2 w - - 0 1",  # Queen vs rook
        ]
        
        # Convert FEN strings to board positions
        for fen in endgame_fens:
            board = chess.Board(fen)
            endgames.append(board)
        
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
        # Create a simple hash of the position
        position_hash = self._get_position_hash(board)
        
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
