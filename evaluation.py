"""
Evaluation module for chess reinforcement learning.

This module provides functions for evaluating chess models against Stockfish
at different skill levels.
"""

import os
import chess
import chess.engine
import torch
from drl_agent import ChessAgent

class ModelEvaluator:
    """
    Evaluates chess models against Stockfish at different skill levels.

    This class provides methods for evaluating the performance of trained
    chess models against Stockfish at various skill levels.
    """

    def __init__(self, stockfish_path=None, use_fp16=True, trainer=None):
        """
        Initialize the model evaluator.

        Args:
            stockfish_path (str, optional): Path to Stockfish executable
            use_fp16 (bool, optional): Whether to use FP16 precision for model inference
            trainer (ChessTrainer, optional): Reference to the trainer for accessing hyperparameters
        """
        self.stockfish_path = stockfish_path
        self.stockfish = None
        self.trainer = trainer  # Store reference to trainer for accessing hyperparameters

        # Set up mixed precision evaluation
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        if self.use_fp16:
            print("Using FP16 precision for model evaluation")

        # Initialize Stockfish engine if path is provided
        if stockfish_path:
            try:
                self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"Model evaluator initialized with Stockfish from: {stockfish_path}")
            except Exception as e:
                print(f"Error initializing Stockfish engine: {e}")
                print("Model evaluation against Stockfish will not be available")

    def evaluate_against_stockfish(self, model_path, num_games=10, max_moves=None, stockfish_levels=range(1, 11)):
        """
        Evaluate a trained model against different Stockfish levels.

        Args:
            model_path (str): Path to the model file to evaluate
            num_games (int): Number of games to play against each Stockfish level
            max_moves (int, optional): Maximum number of moves per game. If None, uses the value from trainer's hyperparameters.
            stockfish_levels (range or list): Range of Stockfish levels to test against

        Returns:
            dict: Dictionary with results for each Stockfish level
        """
        # Use max_moves from trainer's hyperparameters if not specified
        if max_moves is None and hasattr(self, 'trainer') and hasattr(self.trainer, 'max_moves'):
            max_moves = self.trainer.max_moves
        elif max_moves is None:
            max_moves = 250  # Default fallback
        # Load the model to evaluate
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return None

        # Create a chess agent with the model, using FP16 if enabled
        agent = ChessAgent(model_path=model_path, use_fp16=self.use_fp16)

        # Initialize results dictionary
        results = {}

        # Test against each Stockfish level
        for level in stockfish_levels:
            print(f"\nEvaluating against Stockfish level {level}...")

            # Ensure Stockfish engine is available and running
            if not self._ensure_stockfish_running():
                print("Stockfish engine not available. Cannot evaluate.")
                return None

            # Configure Stockfish for this level
            try:
                self.stockfish.configure({"Skill Level": level})
            except Exception as e:
                print(f"Error configuring Stockfish: {e}")
                # Try to restart the engine
                if not self._restart_stockfish():
                    print("Failed to restart Stockfish. Aborting evaluation.")
                    return None
                # Try to configure again
                try:
                    self.stockfish.configure({"Skill Level": level})
                except Exception as e:
                    print(f"Error configuring Stockfish after restart: {e}")
                    return None

            # Track results for this level
            wins = 0
            losses = 0
            draws = 0

            # Play games
            for game_num in range(num_games):
                print(f"Game {game_num+1}/{num_games} against Stockfish level {level}", end="")

                # Initialize board
                board = chess.Board()

                # Play until game over or max moves reached
                move_count = 0
                while not board.is_game_over() and move_count < max_moves:
                    # Model plays as white (first move)
                    if board.turn == chess.WHITE:
                        # Get move from our model
                        try:
                            model_move = agent.select_move(board)
                            board.push(model_move)
                        except Exception as e:
                            print(f"\nError getting model move: {e}")
                            # End the game as a loss
                            losses += 1
                            print(" - Model error (counted as loss)")
                            break
                    else:
                        # Get move from Stockfish
                        try:
                            result = self.stockfish.play(board, chess.engine.Limit(time=0.1))
                            stockfish_move = result.move
                            board.push(stockfish_move)
                        except Exception as e:
                            print(f"\nError getting Stockfish move: {e}")
                            # Try to restart the engine
                            if not self._restart_stockfish():
                                print("Failed to restart Stockfish. Aborting game.")
                                # Count as a draw
                                draws += 1
                                print(" - Engine error (counted as draw)")
                                break
                            # Skip this game and move to the next one
                            continue

                    move_count += 1

                    # Check if game is over
                    if board.is_game_over():
                        break

                # Skip result determination if we broke out due to an error
                if move_count >= max_moves or board.is_game_over():
                    # Determine result
                    if board.is_checkmate():
                        if board.turn == chess.BLACK:  # White (our model) won
                            wins += 1
                            print(" - Model won")
                        else:  # Black (Stockfish) won
                            losses += 1
                            print(" - Stockfish won")
                    else:  # Draw
                        draws += 1
                        print(" - Draw")

            # Store results for this level
            results[level] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': wins / num_games,
                'score': (wins + 0.5 * draws) / num_games
            }

            print(f"Level {level} results: +{wins} -{losses} ={draws} | Win rate: {results[level]['win_rate']*100:.1f}% | Score: {results[level]['score']*100:.1f}%")

        # Determine approximate Stockfish level of the model
        best_comparable_level = 0
        for level in stockfish_levels:
            if results[level]['score'] >= 0.45:  # Model is competitive (45%+ score)
                best_comparable_level = level

        # Print summary
        print("\n=== Model Evaluation Summary ===")
        print(f"Model: {os.path.basename(model_path)} | Games per level: {num_games}")

        if best_comparable_level > 0:
            strength_assessment = ""
            if results[best_comparable_level]['score'] > 0.55:
                strength_assessment = f"better than Stockfish level {best_comparable_level}"
            elif results[best_comparable_level]['score'] >= 0.45:
                strength_assessment = f"almost as good as Stockfish level {best_comparable_level}"
            else:
                strength_assessment = f"not quite at Stockfish level {best_comparable_level} yet"

            print(f"Model strength assessment: {strength_assessment}")
        else:
            print("Model strength assessment: below Stockfish level 1")

        return results

    def _restart_stockfish(self):
        """Restart the Stockfish engine if it's not responding."""
        try:
            # Close the existing engine if it exists
            if self.stockfish:
                try:
                    self.stockfish.quit()
                except:
                    pass  # Ignore errors when closing

            # Start a new engine
            self.stockfish = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            return True
        except Exception as e:
            print(f"Failed to restart Stockfish: {e}")
            return False

    def _ensure_stockfish_running(self):
        """Make sure Stockfish is running and available."""
        if not self.stockfish:
            return self._restart_stockfish()
        return True

    def close(self):
        """Close the Stockfish engine if it's running."""
        if self.stockfish:
            try:
                self.stockfish.quit()
            except:
                pass  # Ignore errors when closing
