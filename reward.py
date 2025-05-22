"""
Simplified reward calculation module for chess reinforcement learning.

This module provides functions for calculating rewards based on chess positions.
It uses a fixed evaluation depth for Stockfish and focuses on material advantage
and checkmate rewards with simple fixed scaling.
"""

import chess
import chess.engine
import random
from async_evaluation import AsyncStockfishEvaluator
from position_clustering import PositionClusterCache

class RewardCalculator:
    """
    Simplified reward calculator with fixed evaluation depth.

    This class provides methods for calculating rewards based on chess positions,
    using either Stockfish evaluation or simpler material-based evaluation.
    It focuses on material advantage and checkmate rewards with simple fixed scaling.
    """

    def __init__(self, stockfish_path=None, use_async=True, num_workers=8, stockfish_eval_frequency=0.2,
                 hybrid_eval_enabled=False, low_freq_rate=0.01, high_freq_rate=0.2, high_freq_interval=50,
                 use_position_clustering=True, cache_size=20000, similarity_threshold=0.85):
        """
        Initialize the reward calculator.

        Args:
            stockfish_path (str, optional): Path to Stockfish executable
            use_async (bool, optional): Whether to use asynchronous evaluation
            num_workers (int, optional): Number of worker threads for async evaluation
            stockfish_eval_frequency (float, optional): Frequency of Stockfish evaluations (0-1)
            hybrid_eval_enabled (bool, optional): Whether to use hybrid evaluation approach
            low_freq_rate (float, optional): Low frequency rate for fast training
            high_freq_rate (float, optional): High frequency rate for quality training
            high_freq_interval (int, optional): Run high frequency evaluation every N episodes
            use_position_clustering (bool, optional): Whether to use position clustering for cache
            cache_size (int, optional): Size of evaluation cache
            similarity_threshold (float, optional): Threshold for position similarity
        """
        self.stockfish = None
        self.async_evaluator = None
        self.stockfish_path = stockfish_path
        self.use_async = use_async
        self.stockfish_eval_frequency = stockfish_eval_frequency

        # Hybrid evaluation parameters
        self.hybrid_eval_enabled = hybrid_eval_enabled
        self.low_freq_rate = low_freq_rate
        self.high_freq_rate = high_freq_rate
        self.high_freq_interval = high_freq_interval
        self.current_episode = 0

        # Position clustering parameters
        self.use_position_clustering = use_position_clustering
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold

        if self.hybrid_eval_enabled:
            print(f"Hybrid Stockfish evaluation enabled:")
            print(f"  - Normal episodes: {self.low_freq_rate * 100:.1f}% evaluation frequency")
            print(f"  - Quality episodes (every {self.high_freq_interval}): {self.high_freq_rate * 100:.1f}% evaluation frequency")
        else:
            print(f"Stockfish evaluation frequency set to {self.stockfish_eval_frequency * 100:.1f}%")

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

        # Initialize cache
        if self.use_position_clustering:
            # Use position clustering cache for improved hit rates
            self.cluster_cache = PositionClusterCache(
                cache_size=self.cache_size,
                similarity_threshold=self.similarity_threshold
            )
            print(f"Using position clustering cache with size {self.cache_size} and similarity threshold {self.similarity_threshold}")
        else:
            # Simple dictionary cache
            self.evaluation_cache = {}  # Simple cache for evaluation results

    def calculate_stockfish_reward(self, board, prev_board=None):
        """
        Calculate reward based on Stockfish evaluation with fixed depth.
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
            # Use fixed evaluation depth for all positions
            eval_depth = 10  # Fixed depth for all evaluations

            # Check cache for this position
            if self.use_position_clustering:
                # Use position clustering cache
                cached_value, hit_type = self.cluster_cache.get(board)
                if cached_value is not None:
                    current_score = cached_value
                    if hit_type == 'cluster':
                        # If this was a cluster hit, print a message occasionally
                        if random.random() < 0.01:  # Only print 1% of the time to avoid spam
                            print(f"Position clustering cache hit: {hit_type}")
                else:
                    # Get current board evaluation
                    if self.use_async and self.async_evaluator:
                        # Use asynchronous evaluation
                        current_score = self.async_evaluator.evaluate_position(board, depth=eval_depth)
                        # Wait for the result
                        result = self.async_evaluator.get_result(current_score, block=True)
                        if result:
                            current_score = result[1]
                        else:
                            # Fallback if async evaluation fails
                            current_score = 0
                    else:
                        # Use synchronous evaluation
                        current_score = self.stockfish.analyse(
                            board=board,
                            limit=chess.engine.Limit(depth=eval_depth)
                        )['score'].relative.score(mate_score=10000)

                    # Add to cluster cache
                    self.cluster_cache.put(board, current_score)
            else:
                # Use simple dictionary cache
                board_fen = board.fen()
                if board_fen in self.evaluation_cache:
                    current_score = self.evaluation_cache[board_fen]
                else:
                    # Get current board evaluation
                    if self.use_async and self.async_evaluator:
                        # Use asynchronous evaluation
                        current_score = self.async_evaluator.evaluate_position(board, depth=eval_depth)
                        # Wait for the result
                        result = self.async_evaluator.get_result(current_score, block=True)
                        if result:
                            current_score = result[1]
                        else:
                            # Fallback if async evaluation fails
                            current_score = 0
                    else:
                        # Use synchronous evaluation
                        current_score = self.stockfish.analyse(
                            board=board,
                            limit=chess.engine.Limit(depth=eval_depth)
                        )['score'].relative.score(mate_score=10000)

                    # Cache the result
                    self.evaluation_cache[board_fen] = current_score

            # Get previous board evaluation if available
            prev_score = None
            if prev_board is not None:
                if self.use_position_clustering:
                    # Use position clustering cache
                    cached_value, hit_type = self.cluster_cache.get(prev_board)
                    if cached_value is not None:
                        prev_score = cached_value
                    else:
                        # Get previous board evaluation
                        if self.use_async and self.async_evaluator:
                            # Use asynchronous evaluation
                            prev_req = self.async_evaluator.evaluate_position(prev_board, depth=eval_depth)
                            # Wait for the result
                            result = self.async_evaluator.get_result(prev_req, block=True)
                            if result:
                                prev_score = result[1]
                            else:
                                # Fallback if async evaluation fails
                                prev_score = 0
                        else:
                            # Use synchronous evaluation
                            prev_score = self.stockfish.analyse(
                                board=prev_board,
                                limit=chess.engine.Limit(depth=eval_depth)
                            )['score'].relative.score(mate_score=10000)

                        # Add to cluster cache
                        self.cluster_cache.put(prev_board, prev_score)
                else:
                    # Use simple dictionary cache
                    prev_board_fen = prev_board.fen()
                    if prev_board_fen in self.evaluation_cache:
                        prev_score = self.evaluation_cache[prev_board_fen]
                    else:
                        if self.use_async and self.async_evaluator:
                            # Use asynchronous evaluation
                            prev_req = self.async_evaluator.evaluate_position(prev_board, depth=eval_depth)
                            # Wait for the result
                            result = self.async_evaluator.get_result(prev_req, block=True)
                            if result:
                                prev_score = result[1]
                            else:
                                # Fallback if async evaluation fails
                                prev_score = 0
                        else:
                            # Use synchronous evaluation
                            prev_score = self.stockfish.analyse(
                                board=prev_board,
                                limit=chess.engine.Limit(depth=eval_depth)
                            )['score'].relative.score(mate_score=10000)

                        # Cache the result
                        self.evaluation_cache[prev_board_fen] = prev_score

            # Calculate reward based on evaluations
            if prev_score is not None:
                # Simple difference between current and previous score
                raw_diff = current_score - prev_score

                # Simple fixed scaling
                reward = raw_diff / 100.0

                # Small fixed penalty for each move to encourage faster wins
                reward -= 0.01
            else:
                # If no previous board, use the current evaluation with simple scaling
                reward = current_score / 100.0

            # Simple terminal state rewards
            if board.is_checkmate():
                # Fixed reward for checkmate
                if board.turn == chess.BLACK:  # White wins
                    reward = 1.0  # Fixed reward for checkmate
                else:  # Black wins
                    reward = -1.0
            elif board.is_stalemate() or board.is_insufficient_material():
                # Fixed penalty for draw
                reward = -0.1

            # Add a small bonus for material advantage
            reward += 0.05 * self.calculate_material_advantage(board)

            # Clip reward to reasonable range
            reward = max(-3.0, min(3.0, reward))

            return reward

        except Exception as e:
            print(f"Error calculating Stockfish reward: {e}")
            # Fallback to material advantage
            return self.calculate_reward(board)

    # Simplified implementation - removed complex evaluation methods

    def calculate_positional_bonus(self, board):
        """
        Calculate simplified bonus rewards for good chess principles.

        Args:
            board (chess.Board): Current board position

        Returns:
            float: Positional bonus reward
        """
        bonus = 0.0

        # Bonus for controlling the center (simplified)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(0.01 if board.piece_at(sq) is not None and board.piece_at(sq).color == board.turn else 0
                           for sq in center_squares)
        bonus += center_control

        # Simple bonus for castling
        king_square = board.king(board.turn)
        if king_square:
            if board.turn == chess.WHITE and king_square in [chess.G1, chess.C1]:  # White castled
                bonus += 0.02
            elif board.turn == chess.BLACK and king_square in [chess.G8, chess.C8]:  # Black castled
                bonus += 0.02

        return bonus

    def calculate_reward(self, board):
        """
        Calculate reward based on the board state (fallback method).
        Simplified to focus on material advantage and checkmate.

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
        reward = 0.05 * material_advantage

        # Add a small positional bonus
        reward += self.calculate_positional_bonus(board)

        # Clip reward to reasonable range
        return max(-3.0, min(3.0, reward))

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

    # Removed evaluate_position method - using calculate_positional_bonus instead

    def update_episode(self, episode_num):
        """
        Update the current episode number for hybrid evaluation.

        Args:
            episode_num (int): Current episode number
        """
        self.current_episode = episode_num

        # If hybrid evaluation is enabled, adjust the evaluation frequency
        if self.hybrid_eval_enabled:
            # Check if this is a "quality" episode
            if episode_num % self.high_freq_interval == 0:
                self.stockfish_eval_frequency = self.high_freq_rate
                print(f"Episode {episode_num}: Using high quality evaluation ({self.high_freq_rate * 100:.1f}%)")
            else:
                self.stockfish_eval_frequency = self.low_freq_rate

    def get_current_frequency(self):
        """
        Get the current Stockfish evaluation frequency.

        Returns:
            float: Current evaluation frequency
        """
        return self.stockfish_eval_frequency

    def print_cache_stats(self):
        """Print cache statistics."""
        if self.use_position_clustering:
            stats = self.cluster_cache.get_stats()
            print("\n=== Position Clustering Cache Statistics ===")
            print(f"Cache size: {stats['cache_size']} / {self.cache_size}")
            print(f"Cluster count: {stats['cluster_count']}")
            print(f"Exact hit rate: {stats['exact_hit_rate']:.2f}")
            print(f"Cluster hit rate: {stats['cluster_hit_rate']:.2f}")
            print(f"Total hit rate: {stats['total_hit_rate']:.2f}")
            print(f"Total lookups: {stats['total_lookups']}")
            print(f"Positions added: {stats['positions_added']}")
            print("==========================================")
        elif hasattr(self, 'evaluation_cache'):
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
