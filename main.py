"""
Main script for chess reinforcement learning with Stockfish evaluation.

This script provides a command-line interface for training and evaluating
chess models using deep reinforcement learning with Stockfish evaluation.
"""

import os
import time
import random
import numpy as np
import torch
import chess
import argparse

# Import custom modules
from drl_agent import DQN, ChessAgent, board_to_tensor, create_move_mask
from memory import PrioritizedReplayMemory, Experience
from reward import RewardCalculator
from evaluation import ModelEvaluator
from visualization import plot_training_progress, plot_evaluation_results
from training import PGNTrainer, SelfPlayTrainer

class ChessTrainer:
    """
    Main trainer class for chess reinforcement learning.

    This class coordinates the training process, including:
    - Model initialization and management
    - Optimization process
    - Training from PGN data
    - Self-play training
    - Model evaluation
    """

    def __init__(self, model_dir="models", stockfish_path=None):
        """
        Initialize the chess trainer.

        Args:
            model_dir (str): Directory for saving models
            stockfish_path (str, optional): Path to Stockfish executable
        """
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir
        self.stockfish_path = stockfish_path

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer with higher learning rate
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        # Initialize prioritized replay memory with larger capacity
        self.memory = PrioritizedReplayMemory(100000)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(stockfish_path)

        # Initialize model evaluator
        self.model_evaluator = ModelEvaluator(stockfish_path)

        # Initialize trainers
        self.pgn_trainer = PGNTrainer(self)
        self.self_play_trainer = SelfPlayTrainer(self)

        # Improved training parameters
        self.batch_size = 256  # Larger batch size
        self.gamma = 0.99  # Discount factor
        self.eps_start = 1.0  # Start with full exploration
        self.eps_end = 0.05
        self.eps_decay = 10000  # Slower decay for better exploration
        self.target_update = 50  # Less frequent target network updates

        self.steps_done = 0

        # Track metrics for monitoring
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'avg_q_values': []
        }

    def train_from_pgn(self, pgn_path, num_games=1000, batch_size=32, save_interval=100):
        """
        Train the model from PGN games.

        Args:
            pgn_path (str): Path to a PGN file or directory containing PGN files
            num_games (int): Maximum number of games to process
            batch_size (int): Number of positions to process in each batch
            save_interval (int): How often to save the model (in games)

        Returns:
            list: List of losses from training
        """
        return self.pgn_trainer.train_from_pgn(pgn_path, num_games, batch_size, save_interval)

    def train_self_play(self, num_episodes=5000, stockfish_opponent=True, stockfish_levels=None,
                      batch_size=64, save_interval=100, eval_interval=500, target_level=7):
        """
        Train the model using self-play.

        Args:
            num_episodes (int): Total number of episodes to train
            stockfish_opponent (bool): Whether to use Stockfish as an opponent
            stockfish_levels (list): List of Stockfish levels to train against
            batch_size (int): Batch size for optimization
            save_interval (int): How often to save models (in episodes)
            eval_interval (int): How often to evaluate against target level (in episodes)
            target_level (int): Target Stockfish level to achieve

        Returns:
            tuple: (episode_rewards, episode_lengths, losses)
        """
        return self.self_play_trainer.train_self_play(
            num_episodes, stockfish_opponent, stockfish_levels,
            batch_size, save_interval, eval_interval, target_level
        )

    def evaluate_against_stockfish(self, model_path, num_games=10, stockfish_levels=range(1, 11)):
        """
        Evaluate a trained model against different Stockfish levels.

        Args:
            model_path (str): Path to the model file to evaluate
            num_games (int): Number of games to play against each Stockfish level
            stockfish_levels (range or list): Range of Stockfish levels to test against

        Returns:
            dict: Dictionary with results for each Stockfish level
        """
        return self.model_evaluator.evaluate_against_stockfish(
            model_path, num_games, stockfish_levels=stockfish_levels
        )

    def save_model(self, filename):
        """
        Save the model to disk.

        Args:
            filename (str): Name of the file to save the model to
        """
        filepath = os.path.join(self.model_dir, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename):
        """
        Load a model from disk.

        Args:
            filename (str): Name of the file to load the model from

        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False

    def select_action(self, state, mask, board):
        """
        Select an action using epsilon-greedy policy with improved exploration.

        Args:
            state (torch.Tensor): Current state tensor
            mask (torch.Tensor): Mask of legal moves
            board (chess.Board): Current board position

        Returns:
            tuple: (action_idx, move) - The selected action index and chess move
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # Move tensors to the correct device
        state = state.to(self.device)
        mask = mask.to(self.device)

        if sample > eps_threshold:
            with torch.no_grad():
                # Use policy network to select best action
                q_values = self.policy_net(state, mask)

                # Track average Q-values for monitoring
                if len(self.training_stats['avg_q_values']) < 1000:
                    self.training_stats['avg_q_values'].append(q_values.max().item())
                else:
                    self.training_stats['avg_q_values'] = self.training_stats['avg_q_values'][1:] + [q_values.max().item()]

                # Get the action with highest Q-value
                action_idx = q_values.max(1)[1].item()

                # Convert action index to chess move
                from_square = action_idx // 64
                to_square = action_idx % 64

                # Check if this is a legal move
                move = chess.Move(from_square, to_square)

                # Handle promotion
                piece = board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                    if board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                        move.promotion = chess.QUEEN
                    elif board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                        move.promotion = chess.QUEEN

                # If move is not legal, choose a legal move with highest Q-value
                if move not in board.legal_moves:
                    legal_moves = list(board.legal_moves)
                    legal_move_indices = [m.from_square * 64 + m.to_square for m in legal_moves]

                    # Get Q-values for legal moves only
                    legal_q_values = q_values[0, legal_move_indices]
                    best_legal_idx = torch.argmax(legal_q_values).item()
                    move = legal_moves[best_legal_idx]
                    action_idx = legal_move_indices[best_legal_idx]
        else:
            # Choose a random legal move
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            action_idx = move.from_square * 64 + move.to_square

        return action_idx, move

    def optimize_model(self):
        """
        Perform one step of optimization with prioritized experience replay.

        Returns:
            float: Loss value if optimization was performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch from prioritized memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Move tensors to the correct device
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool).to(self.device)
        mask_batch = torch.cat(batch.mask).to(self.device)
        next_mask_batch = torch.cat(batch.next_mask).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch, mask_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch, next_mask_batch)
            next_state_values = next_q_values.max(1)[0]
            # Set V(s) = 0 for terminal states
            next_state_values[done_batch] = 0.0

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute TD errors for updating priorities
        td_errors = torch.abs(state_action_values.squeeze() - expected_state_action_values).detach().cpu().numpy()

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors + 1e-6)  # Small constant to avoid zero priority

        # Compute weighted loss
        criterion = torch.nn.SmoothL1Loss(reduction='none')
        elementwise_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = (elementwise_loss * weights.unsqueeze(1)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)

        self.optimizer.step()

        return loss.item()

    def close(self):
        """Close all resources."""
        self.reward_calculator.close()
        self.model_evaluator.close()


def main():
    """Main function for command-line interface."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train chess model with optimized performance")
    parser.add_argument("--data_dir", default="data", help="Directory containing PGN files")
    parser.add_argument("--num_games", type=int, default=200000, help="Maximum number of games to process")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save_interval", type=int, default=1000, help="How often to save the model (in games)")
    parser.add_argument("--self_play", action="store_true", help="Run self-play training after PGN training")
    parser.add_argument("--self_play_episodes", type=int, default=5000, help="Number of self-play episodes")
    parser.add_argument("--stockfish_path", default=None, help="Path to Stockfish executable")
    parser.add_argument("--stockfish_opponent", action="store_true", default=True,
                        help="Use Stockfish as opponent in self-play")
    parser.add_argument("--target_level", type=int, default=7,
                        help="Target Stockfish level to achieve")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="How often to evaluate against target level (in episodes)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model against Stockfish levels")
    parser.add_argument("--eval_games", type=int, default=5, help="Number of games to play against each Stockfish level")
    parser.add_argument("--eval_model", help="Path to specific model to evaluate (optional)")
    parser.add_argument("--min_level", type=int, default=1, help="Minimum Stockfish level to test against")
    parser.add_argument("--max_level", type=int, default=10, help="Maximum Stockfish level to test against")

    args = parser.parse_args()

    # Print training configuration
    print("\n=== Chess Model Training Configuration ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Maximum games: {args.num_games}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save interval: {args.save_interval}")
    print(f"Self-play training: {'Yes' if args.self_play else 'No'}")
    if args.self_play:
        print(f"Self-play episodes: {args.self_play_episodes}")
    print(f"Evaluate against Stockfish: {'Yes' if args.evaluate else 'No'}")
    if args.evaluate:
        print(f"Evaluation games per level: {args.eval_games}")
        print(f"Stockfish levels: {args.min_level}-{args.max_level}")
    print(f"Stockfish path: {args.stockfish_path}")
    print("==========================================\n")

    # Create trainer
    trainer = ChessTrainer(stockfish_path=args.stockfish_path)

    # Train from high-quality PGN data
    if os.path.exists(args.data_dir):
        print(f"Training from high-quality PGN data in directory: {args.data_dir}")
        print("Using improved neural network architecture with residual blocks")
        print("Using prioritized experience replay with efficient memory usage")
        print("Using optimized batch processing and tensor operations")
        print("Using adaptive Stockfish evaluation sampling")

        # Train with optimized parameters
        trainer.train_from_pgn(
            args.data_dir,
            num_games=args.num_games,
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )

        # Save intermediate model
        trainer.save_model("model_pgn_improved.pt")
    else:
        print(f"Data directory not found: {args.data_dir}")
        print("Skipping PGN training phase")

    # Continue with enhanced self-play training if requested
    if args.self_play:
        print("\nStarting advanced self-play training with curriculum learning...")
        print(f"Target: Achieve Stockfish level {args.target_level} strength")
        print("Using adaptive difficulty with Stockfish opponents")
        print("Using model pool for diverse opponents")
        print("Using batch processing for better efficiency")
        print("Using regular evaluation against target level")

        # Run self-play training with improved parameters
        trainer.train_self_play(
            num_episodes=args.self_play_episodes,
            stockfish_opponent=args.stockfish_opponent,
            stockfish_levels=list(range(1, args.target_level + 1)),
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            target_level=args.target_level
        )

    # Evaluate against Stockfish if requested
    if args.evaluate:
        # Determine which model to use for testing
        if args.eval_model:
            test_model_path = args.eval_model
        else:
            # Try to find the best model
            best_model_path = os.path.join(trainer.model_dir, "best_model.pt")
            final_model_path = os.path.join(trainer.model_dir, "model_improved_final.pt")
            pgn_model_path = os.path.join(trainer.model_dir, "model_pgn_improved.pt")

            if os.path.exists(best_model_path):
                test_model_path = best_model_path
            elif os.path.exists(final_model_path):
                test_model_path = final_model_path
            else:
                test_model_path = pgn_model_path

        print(f"\nEvaluating model: {test_model_path}")
        print(f"Testing against Stockfish levels {args.min_level}-{args.max_level}")
        print(f"Playing {args.eval_games} games per level")

        # Run evaluation
        results = trainer.evaluate_against_stockfish(
            model_path=test_model_path,
            num_games=args.eval_games,
            stockfish_levels=range(args.min_level, args.max_level + 1)
        )

        # Plot evaluation results
        if results:
            plot_evaluation_results(
                results,
                os.path.join(trainer.model_dir, 'evaluation_results.png')
            )

    # Clean up
    trainer.close()
    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
