"""
Enhanced reward calculation module for chess reinforcement learning.

This module provides advanced functions for calculating rewards based on chess positions.
It includes Stockfish-based evaluation with adaptive sampling, dynamic reward normalization,
and improved material-based evaluation. It also supports asynchronous evaluation for better performance.
"""

import chess
import chess.engine
import numpy as np
import random
import time
import threading
import queue
from async_evaluation import AsyncStockfishEvaluator

class RewardCalculator:
    """
    Enhanced reward calculator with adaptive sampling and dynamic normalization.

    This class provides methods for calculating rewards based on chess positions,
    using either Stockfish evaluation or simpler material-based evaluation.
    It includes adaptive sampling to focus computational resources on critical positions
    and dynamic reward normalization for more consistent learning signals.
    """

    def __init__(self, stockfish_path=None, use_async=True, num_workers=4):
        """
        Initialize the reward calculator.

        Args:
            stockfish_path (str, optional): Path to Stockfish executable
            use_async (bool, optional): Whether to use asynchronous evaluation
            num_workers (int, optional): Number of worker threads for async evaluation
        """
        self.stockfish = None
        self.async_evaluator = None
        self.stockfish_path = stockfish_path
        self.use_async = use_async

        if stockfish_path:
            try:
                if use_async:
                    # Initialize asynchronous evaluator with multiple workers
                    self.async_evaluator = AsyncStockfishEvaluator(stockfish_path, num_workers)
                    print(f"Reward calculator initialized with asynchronous Stockfish evaluation using {num_workers} workers")
                else:
                    # Initialize single synchronous Stockfish engine
                    self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                    print(f"Reward calculator initialized with synchronous Stockfish from: {stockfish_path}")
            except Exception as e:
                print(f"Error initializing Stockfish engine: {e}")
                print("Falling back to material-based evaluation")

        # Initialize reward statistics for dynamic normalization
        self.reward_stats = {
            'count': 0,
            'sum': 0,
            'sum_squared': 0,
            'mean': 0,
            'std': 1.0,
            'min': -1.0,
            'max': 1.0
        }

        # Initialize adaptive sampling parameters
        self.position_importance = {}  # Track position importance for adaptive sampling
        self.last_positions = []       # Track recent positions for temporal difference learning
        self.max_history = 10          # Maximum number of positions to keep in history

        # Pending evaluations for asynchronous mode
        self.pending_evaluations = {}  # Map of request_id to board
        self.evaluation_results = {}   # Cache of recent evaluation results

    def calculate_stockfish_reward(self, board, prev_board=None):
        """
        Calculate reward based on deeper Stockfish evaluation with adaptive sampling and dynamic normalization.
        Uses asynchronous evaluation if enabled for better performance.

        Args:
            board (chess.Board): Current board position
            prev_board (chess.Board, optional): Previous board position

        Returns:
            float: Calculated reward
        """
        if self.stockfish is None and self.async_evaluator is None:
            # Fallback to material advantage if Stockfish is not available
            return self.calculate_reward(board)

        try:
            # Determine position importance for adaptive sampling
            position_importance = self._calculate_position_importance(board)

            # Adjust evaluation depth based on position importance
            # More important positions get deeper evaluation
            if position_importance > 0.8:  # Critical position
                eval_depth = 16  # Deep analysis for critical positions
            elif position_importance > 0.5:  # Important position
                eval_depth = 14  # Medium-deep analysis
            else:  # Standard position
                eval_depth = 12  # Standard analysis depth

            # Get current board evaluation
            if self.use_async and self.async_evaluator:
                # Use asynchronous evaluation
                current_score = self._get_async_evaluation(board, eval_depth)

                # If we have a previous board, get its evaluation too
                if prev_board is not None:
                    prev_score = self._get_async_evaluation(prev_board, eval_depth)
                else:
                    prev_score = None
            else:
                # Use synchronous evaluation
                current_score = self.stockfish.analyse(
                    board=board,
                    limit=chess.engine.Limit(depth=eval_depth)
                )['score'].relative.score(mate_score=10000)

                # If we have a previous board, calculate its evaluation too
                if prev_board is not None:
                    prev_score = self.stockfish.analyse(
                        board=prev_board,
                        limit=chess.engine.Limit(depth=eval_depth)
                    )['score'].relative.score(mate_score=10000)
                else:
                    prev_score = None

            # Calculate reward based on evaluations
            if prev_score is not None:
                # Calculate raw difference for TD learning
                raw_diff = current_score - prev_score

                # Apply dynamic reward scaling based on statistics
                reward = self._normalize_reward(raw_diff)

                # Small penalty for each move to encourage faster wins
                # Scale penalty based on move number to encourage exploration in early game
                move_number = len(board.move_stack) // 2  # Full move number
                if move_number < 10:  # Opening
                    reward -= 0.005  # Very small penalty in opening
                elif move_number < 30:  # Middlegame
                    reward -= 0.01   # Standard penalty in middlegame
                else:  # Endgame
                    reward -= 0.02   # Larger penalty in endgame to encourage faster resolution
            else:
                # If no previous board, use the current evaluation with normalization
                reward = self._normalize_reward(current_score / 100.0)

            # Enhanced terminal state rewards with more nuanced values
            if board.is_checkmate():
                # Positive reward if white wins, negative if black wins
                # Scale based on move number to reward faster checkmates
                move_number = len(board.move_stack) // 2
                if board.turn == chess.BLACK:  # White wins
                    reward = 5.0 + max(0, (100 - move_number) / 20)  # Bonus for faster checkmate
                else:  # Black wins
                    reward = -5.0 - max(0, (100 - move_number) / 20)
            elif board.is_stalemate() or board.is_insufficient_material():
                # Adjust draw reward based on position evaluation
                if abs(current_score) < 50:  # Equal position
                    reward = 0.0
                elif current_score > 0:  # White was better
                    # Scale draw penalty based on how much better white was
                    reward = -0.5 * (abs(current_score) / 200)
                else:  # Black was better
                    # Scale draw penalty based on how much better black was
                    reward = 0.5 * (abs(current_score) / 200)

            # Add rewards for good chess principles
            reward += self.calculate_positional_bonus(board)

            # Update reward statistics for future normalization
            self._update_reward_stats(reward)

            # Update position history for temporal difference learning
            self._update_position_history(board, reward)

            return reward

        except Exception as e:
            print(f"Error calculating Stockfish reward: {e}")
            # Fallback to material advantage
            return self.calculate_reward(board)

    def _get_async_evaluation(self, board, depth):
        """
        Get evaluation using the asynchronous evaluator.

        Args:
            board (chess.Board): Board position to evaluate
            depth (int): Evaluation depth

        Returns:
            int: Evaluation score
        """
        # Check if we already have this position in the cache
        board_fen = board.fen()
        if board_fen in self.evaluation_results:
            return self.evaluation_results[board_fen]

        # Submit the position for evaluation
        request_id = self.async_evaluator.evaluate_position(board, depth=depth)

        # Store the request for later retrieval
        self.pending_evaluations[request_id] = board_fen

        # Process any completed evaluations
        self._process_completed_evaluations(block=False)

        # If this position's evaluation is now available, return it
        if board_fen in self.evaluation_results:
            return self.evaluation_results[board_fen]

        # Otherwise, wait for this specific evaluation to complete
        while True:
            result = self.async_evaluator.get_result(request_id, block=True, timeout=0.1)
            if result:
                score = result[1]
                # Cache the result
                self.evaluation_results[board_fen] = score
                # Remove from pending
                del self.pending_evaluations[request_id]
                return score

            # Process other completed evaluations while waiting
            self._process_completed_evaluations(block=False)

    def _process_completed_evaluations(self, block=False, timeout=0.01):
        """
        Process any completed evaluations from the async evaluator.

        Args:
            block (bool): Whether to block waiting for a result
            timeout (float): Maximum time to wait if blocking
        """
        if not self.async_evaluator:
            return

        # Get any available results
        while True:
            result = self.async_evaluator.get_result(block=block, timeout=timeout)
            if not result:
                break

            request_id, score, _ = result

            # If we know what position this is for, cache the result
            if request_id in self.pending_evaluations:
                board_fen = self.pending_evaluations[request_id]
                self.evaluation_results[board_fen] = score
                del self.pending_evaluations[request_id]

            # Only process one result if not blocking
            if not block:
                break

    def _calculate_position_importance(self, board):
        """
        Calculate the importance of a position for adaptive sampling.

        This determines how much computational resources to allocate to evaluating this position.

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Importance score between 0 and 1
        """
        # Get position FEN (just piece positions)
        position_fen = board.fen().split(' ')[0]

        # Start with base importance
        importance = 0.5

        # Factors that increase importance:

        # 1. Terminal states are always important
        if board.is_game_over():
            return 1.0

        # 2. Check situations are important
        if board.is_check():
            importance += 0.2

        # 3. Positions with few pieces (endgame) are important
        piece_count = sum(1 for _ in board.piece_map())
        if piece_count <= 10:  # Endgame
            importance += 0.2

        # 4. Positions where captures just happened are important
        if len(board.move_stack) > 0:
            last_move = board.move_stack[-1]
            if board.is_capture(last_move):
                importance += 0.15

        # 5. Positions we've seen before with high variance in evaluation
        if position_fen in self.position_importance:
            importance = max(importance, self.position_importance[position_fen])

        # Ensure importance is between 0 and 1
        return min(1.0, max(0.0, importance))

    def _normalize_reward(self, raw_reward):
        """
        Normalize rewards for more consistent learning signals.

        Args:
            raw_reward (float): Raw reward value

        Returns:
            float: Normalized reward
        """
        # If we don't have enough statistics yet, use raw reward with basic scaling
        if self.reward_stats['count'] < 100:
            # Simple clipping for early rewards
            return max(-3.0, min(3.0, raw_reward))

        # Z-score normalization with clipping
        if self.reward_stats['std'] > 0:
            normalized = (raw_reward - self.reward_stats['mean']) / self.reward_stats['std']
            # Clip to reasonable range to prevent extreme values
            return max(-3.0, min(3.0, normalized))
        else:
            return raw_reward

    def _update_reward_stats(self, reward):
        """
        Update running statistics for reward normalization.

        Args:
            reward (float): New reward value
        """
        # Update count
        self.reward_stats['count'] += 1
        count = self.reward_stats['count']

        # Update sum and sum of squares
        self.reward_stats['sum'] += reward
        self.reward_stats['sum_squared'] += reward * reward

        # Update mean
        self.reward_stats['mean'] = self.reward_stats['sum'] / count

        # Update standard deviation
        variance = max(0, self.reward_stats['sum_squared'] / count - self.reward_stats['mean'] ** 2)
        self.reward_stats['std'] = np.sqrt(variance)

        # Update min and max
        self.reward_stats['min'] = min(self.reward_stats['min'], reward)
        self.reward_stats['max'] = max(self.reward_stats['max'], reward)

    def _update_position_history(self, board, reward):
        """
        Update position history for temporal difference learning.

        Args:
            board (chess.Board): Current board position
            reward (float): Calculated reward
        """
        # Get position FEN (just piece positions)
        position_fen = board.fen().split(' ')[0]

        # Add to position importance tracking
        if position_fen in self.position_importance:
            # Update importance based on reward magnitude
            old_importance = self.position_importance[position_fen]
            new_importance = max(old_importance, min(1.0, abs(reward) / 3.0))
            self.position_importance[position_fen] = new_importance
        else:
            # Initialize importance based on reward magnitude
            self.position_importance[position_fen] = min(1.0, abs(reward) / 3.0)

        # Add to position history
        self.last_positions.append((position_fen, reward))

        # Trim history if needed
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)

    def calculate_positional_bonus(self, board):
        """
        Calculate bonus rewards for good chess principles.

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Positional bonus reward
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

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Calculated reward
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

    def evaluate_position(self, board):
        """
        Simple position evaluation.

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Position evaluation score
        """
        # Center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(1 if board.piece_at(sq) is not None and board.piece_at(sq).color == board.turn else 0
                            for sq in center_squares)

        # Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))

        # Combine factors
        return center_control + 0.1 * mobility

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
