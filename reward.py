"""
Simple reward calculation module for chess reinforcement learning.

This module provides basic functions for calculating rewards based on chess positions.
"""

import chess
import chess.engine
from async_evaluation import AsyncStockfishEvaluator

class RewardCalculator:
    """
    Simple reward calculator for chess positions.

    This class provides methods for calculating rewards based on chess positions,
    using either Stockfish evaluation or material-based evaluation.
    """

    def __init__(self, stockfish_path=None, use_async=True, num_workers=4, stockfish_eval_frequency=0.1):
        """
        Initialize the reward calculator.

        Args:
            stockfish_path (str, optional): Path to Stockfish executable
            use_async (bool, optional): Whether to use asynchronous evaluation
            num_workers (int, optional): Number of worker threads for async evaluation
            stockfish_eval_frequency (float, optional): Frequency of Stockfish evaluations (0-1)
        """
        self.stockfish = None
        self.async_evaluator = None
        self.stockfish_path = stockfish_path
        self.use_async = use_async
        self.stockfish_eval_frequency = stockfish_eval_frequency

        print(f"Stockfish evaluation frequency set to {self.stockfish_eval_frequency * 100:.1f}%")

        if stockfish_path:
            try:
                if use_async:
                    # Initialize asynchronous evaluator
                    self.async_evaluator = AsyncStockfishEvaluator(stockfish_path, num_workers)
                    print(f"Reward calculator initialized with asynchronous Stockfish evaluation using {num_workers} workers")
                else:
                    # Initialize single synchronous Stockfish engine
                    self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                    print(f"Reward calculator initialized with synchronous Stockfish from: {stockfish_path}")
            except Exception as e:
                print(f"Error initializing Stockfish engine: {e}")
                print("Falling back to material-based evaluation")

        # Simple dictionary cache
        self.evaluation_cache = {}  # Simple cache for evaluation results

    def calculate_stockfish_reward(self, board):
        """
        Simple reward calculation based on Stockfish evaluation.

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Calculated reward
        """
        if self.stockfish is None and self.async_evaluator is None:
            # Fallback to material advantage if Stockfish is not available
            return self.calculate_reward(board)

        try:
            # Use fixed evaluation depth
            eval_depth = 8

            # Get current board evaluation (with caching)
            current_score = self._get_cached_evaluation(board, eval_depth)

            # Calculate reward based on evaluation
            reward = current_score / 100.0

            # Handle terminal states with fixed rewards
            if board.is_checkmate():
                # Fixed reward for checkmate
                if board.turn == chess.BLACK:  # White wins
                    reward = 1.0
                else:  # Black wins
                    reward = -1.0
            elif board.is_stalemate() or board.is_insufficient_material():
                # Fixed penalty for draw
                reward = -0.1

            # Clip reward to reasonable range [-10, 10]
            reward = max(-10.0, min(10.0, reward))

            return reward

        except Exception as e:
            print(f"Error calculating Stockfish reward: {e}")
            # Fallback to material advantage
            return self.calculate_reward(board)

    def _get_cached_evaluation(self, board, eval_depth):
        """
        Get cached evaluation for a board position, or calculate if not in cache.

        Args:
            board (chess.Board): Board position to evaluate
            eval_depth (int): Evaluation depth

        Returns:
            float: Evaluation score
        """
        # Use simple dictionary cache
        board_fen = board.fen()
        if board_fen in self.evaluation_cache:
            return self.evaluation_cache[board_fen]

        # Get board evaluation
        score = self._evaluate_position(board, eval_depth)

        # Cache the result
        self.evaluation_cache[board_fen] = score
        return score

    def _evaluate_position(self, board, eval_depth):
        """
        Evaluate a board position using Stockfish.

        Args:
            board (chess.Board): Board position to evaluate
            eval_depth (int): Evaluation depth

        Returns:
            float: Evaluation score
        """
        if self.use_async and self.async_evaluator:
            # Use asynchronous evaluation
            req_id = self.async_evaluator.evaluate_position(board, depth=eval_depth)
            # Wait for the result
            result = self.async_evaluator.get_result(req_id, block=True)
            if result:
                return result[1]
            else:
                # Fallback if async evaluation fails
                return 0
        else:
            # Use synchronous evaluation
            return self.stockfish.analyse(
                board=board,
                limit=chess.engine.Limit(depth=eval_depth)
            )['score'].relative.score(mate_score=10000)

    def calculate_positional_bonus(self, board):
        """
        Calculate simple bonus rewards for good chess principles.

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Positional bonus reward
        """
        bonus = 0.0

        # Bonus for controlling the center
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece is not None and piece.color == board.turn:
                bonus += 0.02

        return bonus

    def calculate_reward(self, board):
        """
        Calculate reward based on the board state (fallback method).

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Calculated reward
        """
        # Terminal state rewards
        if board.is_checkmate():
            # Fixed reward for checkmate
            return 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            # Small penalty for draw
            return -0.1

        # Material advantage is the primary component
        material_advantage = self.calculate_material_advantage(board)

        # Simple scaling of material advantage
        reward = 0.1 * material_advantage

        # Add a small positional bonus
        reward += self.calculate_positional_bonus(board) * 0.5

        # Clip reward to reasonable range [-10, 10]
        return max(-10.0, min(10.0, reward))

    def calculate_material_advantage(self, board):
        """
        Calculate material advantage for the current player.

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Material advantage score
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

    def get_current_frequency(self):
        """
        Get the current Stockfish evaluation frequency.

        Returns:
            float: Current evaluation frequency
        """
        return self.stockfish_eval_frequency

    def print_cache_stats(self):
        """Print cache statistics."""
        print("\n=== Simple Cache Statistics ===")
        print(f"Cache size: {len(self.evaluation_cache)}")
        print("==============================")

        # Print async evaluator stats if available
        if self.async_evaluator:
            self.async_evaluator.print_stats()

    def close(self):
        """Close the Stockfish engine and async evaluator if they're running."""
        if self.stockfish:
            try:
                self.stockfish.quit()
                print("Stockfish engine closed")
            except:
                pass

        if self.async_evaluator:
            try:
                self.async_evaluator.close()
                print("Asynchronous Stockfish evaluator closed")
            except:
                pass
