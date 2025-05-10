"""
Reward functions for chess reinforcement learning.
"""

import chess
import chess.engine

class RewardCalculator:
    """
    Reward calculation for chess reinforcement learning.
    
    Provides methods to calculate rewards based on:
    1. Stockfish evaluation (preferred)
    2. Material advantage (fallback)
    3. Positional features
    """
    
    def __init__(self, stockfish=None):
        """
        Initialize the reward calculator.
        
        Args:
            stockfish: Optional Stockfish engine instance
        """
        self.stockfish = stockfish
    
    def calculate_stockfish_reward(self, board, prev_board=None):
        """
        Calculate reward based on Stockfish evaluation.
        
        Uses deeper Stockfish analysis with nuanced rewards based on position evaluation.
        Falls back to material-based reward if Stockfish is not available.
        
        Args:
            board: Current chess.Board
            prev_board: Previous chess.Board (optional)
            
        Returns:
            Calculated reward value
        """
        if self.stockfish is None:
            # Fallback to material advantage if Stockfish is not available
            return self.calculate_reward(board)
        
        try:
            # Use deeper analysis for more accurate evaluation
            current_score = self.stockfish.analyse(
                board=board, 
                limit=chess.engine.Limit(depth=12)
            )['score'].relative.score(mate_score=10000)
            
            # If we have a previous board, calculate the difference in evaluation
            if prev_board is not None:
                prev_score = self.stockfish.analyse(
                    board=prev_board, 
                    limit=chess.engine.Limit(depth=12)
                )['score'].relative.score(mate_score=10000)
                
                # More nuanced reward calculation
                raw_diff = current_score - prev_score
                
                # Scale reward based on magnitude of improvement
                if abs(raw_diff) < 50:  # Small change
                    reward = raw_diff / 100.0
                elif abs(raw_diff) < 200:  # Medium change
                    reward = raw_diff / 80.0
                else:  # Large change (likely a blunder or brilliant move)
                    reward = raw_diff / 50.0
                
                # Small penalty for each move to encourage faster wins
                reward -= 0.01
            else:
                # If no previous board, just use the current evaluation
                reward = current_score / 100.0
            
            # Enhanced terminal state rewards
            if board.is_checkmate():
                reward = 10.0 if board.turn == chess.BLACK else -10.0  # Positive reward if white wins, negative if black wins
            elif board.is_stalemate() or board.is_insufficient_material():
                # Adjust draw reward based on position evaluation
                if abs(current_score) < 50:  # Equal position
                    reward = 0.0
                elif current_score > 0:  # White was better
                    reward = -0.5
                else:  # Black was better
                    reward = 0.5
            
            # Add rewards for good chess principles
            reward += self.calculate_positional_bonus(board)
            
            return reward
            
        except Exception as e:
            print(f"Error calculating Stockfish reward: {e}")
            # Fallback to material advantage
            return self.calculate_reward(board)
    
    def calculate_positional_bonus(self, board):
        """
        Calculate bonus rewards for good chess principles.
        
        Rewards for:
        - Center control
        - Piece development
        - King safety
        
        Args:
            board: Current chess.Board
            
        Returns:
            Bonus reward value
        """
        bonus = 0.0
        
        # Bonus for controlling the center
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(0.02 if board.piece_at(sq) is not None and board.piece_at(sq).color == board.turn else 0
                           for sq in center_squares)
        bonus += center_control
        
        # Bonus for piece development in the opening
        if len(board.move_stack) < 20:  # Only in the opening
            developed_knights = 0
            developed_bishops = 0
            
            # Check if knights are developed
            knight_squares = [chess.C3, chess.F3, chess.C6, chess.F6]
            for sq in knight_squares:
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.KNIGHT:
                    developed_knights += 1
            
            # Check if bishops are developed
            bishop_squares = [chess.B2, chess.C1, chess.D2, chess.E2, chess.F1, chess.G2,  # White bishop development
                             chess.B7, chess.C8, chess.D7, chess.E7, chess.F8, chess.G7]  # Black bishop development
            for sq in bishop_squares:
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.BISHOP:
                    developed_bishops += 1
            
            bonus += 0.01 * developed_knights + 0.01 * developed_bishops
        
        # Bonus for king safety
        king_square = board.king(board.turn)
        if king_square:
            # Bonus for castling
            if board.turn == chess.WHITE:
                if king_square in [chess.G1, chess.C1]:  # Castled king
                    bonus += 0.05
            else:
                if king_square in [chess.G8, chess.C8]:  # Castled king
                    bonus += 0.05
        
        return bonus
    
    def calculate_reward(self, board):
        """
        Calculate reward based on the board state (fallback method).
        
        Used when Stockfish is not available.
        
        Args:
            board: Current chess.Board
            
        Returns:
            Calculated reward value
        """
        # Basic reward function
        if board.is_checkmate():
            # High reward/penalty for checkmate
            return 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            # Small penalty for draw
            return -0.1
        
        # Material advantage reward
        material_advantage = self.calculate_material_advantage(board)
        
        # Position evaluation reward
        position_score = self.evaluate_position(board)
        
        # Combine rewards
        return 0.01 * material_advantage + 0.005 * position_score
    
    def calculate_material_advantage(self, board):
        """
        Calculate material advantage for the current player.
        
        Args:
            board: Current chess.Board
            
        Returns:
            Material advantage value
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King's value doesn't contribute to material advantage
        }
        
        white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value
                            for piece_type, value in piece_values.items())
        black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value
                            for piece_type, value in piece_values.items())
        
        # Return advantage from perspective of current player
        return white_material - black_material if board.turn == chess.WHITE else black_material - white_material
    
    def evaluate_position(self, board):
        """
        Simple position evaluation.
        
        Args:
            board: Current chess.Board
            
        Returns:
            Position evaluation score
        """
        # Center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(1 if board.piece_at(sq) is not None and board.piece_at(sq).color == board.turn else 0
                            for sq in center_squares)
        
        # Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))
        
        # Combine factors
        return center_control + 0.1 * mobility
