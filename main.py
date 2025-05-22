"""
Main script for chess reinforcement learning with Stockfish evaluation.

This script provides a command-line interface for training and evaluating
chess models using deep reinforcement learning with Stockfish evaluation.

Features:
- Training from PGN files with optimized batch processing
- Self-play training with curriculum learning
- Evaluation against Stockfish at different levels
- Loading pre-trained models to continue training
- Mixed precision training for GPU acceleration
"""

import os
import random
import numpy as np
import torch
import chess
import argparse

# Import torch.amp for mixed precision training
from torch.amp import GradScaler
from torch.cuda.amp import autocast

# Import custom modules
from drl_agent import DQN
from memory import PrioritizedReplayMemory, Experience
from reward import RewardCalculator
from evaluation import ModelEvaluator
from visualization import plot_evaluation_results
from training import PGNTrainer, SelfPlayTrainer
from hyperparameters import get_optimized_hyperparameters
from simple_metrics import SimpleMetrics
from metrics_visualization import plot_training_metrics, plot_performance_summary, plot_training_progress

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

    def __init__(self, model_dir="models", stockfish_path=None, gpu_type="rtx_4070"):
        """
        Initialize the chess trainer with optimized hyperparameters.

        Args:
            model_dir (str): Directory for saving models
            stockfish_path (str, optional): Path to Stockfish executable
            gpu_type (str, optional): GPU type to optimize for
        """
        # Get optimized hyperparameters for the current hardware
        self.hyperparams = get_optimized_hyperparameters(gpu_type)
        print(f"Using optimized hyperparameters for {gpu_type}")

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir
        self.stockfish_path = stockfish_path

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize mixed precision training components if CUDA is available
        self.use_mixed_precision = torch.cuda.is_available() and self.hyperparams['mixed_precision']['enabled']
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            # Set default tensor type to float16 for mixed precision training
            self.fp16_enabled = True
            print("Using mixed precision FP16 training for better performance on RTX 4070")
        else:
            self.fp16_enabled = False

        # Initialize networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer with optimized parameters
        lr = self.hyperparams['optimizer']['learning_rate']
        weight_decay = self.hyperparams['optimizer']['weight_decay']
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Initialize learning rate scheduler based on hyperparameters
        scheduler_type = self.hyperparams['optimizer']['lr_scheduler']
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.hyperparams['optimizer']['lr_factor'],
                patience=self.hyperparams['optimizer']['lr_patience'],
                #verbose=True, # Uncomment for verbose logging (it gives errors.)
                min_lr=self.hyperparams['optimizer']['lr_min']
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=5000,
                gamma=self.hyperparams['optimizer']['lr_factor']
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100000,
                eta_min=self.hyperparams['optimizer']['lr_min']
            )

        # Initialize prioritized replay memory with optimized capacity
        memory_capacity = self.hyperparams['memory']['capacity']
        memory_alpha = self.hyperparams['memory']['alpha']
        memory_beta_start = self.hyperparams['memory']['beta_start']
        self.memory = PrioritizedReplayMemory(
            capacity=memory_capacity,
            alpha=memory_alpha,
            beta_start=memory_beta_start
        )

        # Initialize reward calculator with asynchronous evaluation
        self.reward_calculator = RewardCalculator(
            stockfish_path=stockfish_path,
            use_async=True,
            num_workers=4  # Use 4 worker threads for parallel evaluation
        )

        # Initialize model evaluator
        self.model_evaluator = ModelEvaluator(
            stockfish_path=stockfish_path,
            use_fp16=self.fp16_enabled
        )

        # Initialize simple metrics for on-the-fly monitoring
        self.metrics = SimpleMetrics(window_size=100)

        # Initialize trainers
        self.pgn_trainer = PGNTrainer(self)
        self.self_play_trainer = SelfPlayTrainer(self)

        # Training parameters from hyperparameters
        self.batch_size = self.hyperparams['optimizer']['batch_size']
        self.gamma = self.hyperparams['reward']['gamma']
        self.eps_start = self.hyperparams['exploration']['eps_start']
        self.eps_end = self.hyperparams['exploration']['eps_end']
        self.eps_decay = self.hyperparams['exploration']['eps_decay']
        self.target_update = self.hyperparams['self_play']['target_update']

        # Set gradient clipping threshold from hyperparameters
        self.gradient_clip = self.hyperparams['optimizer']['gradient_clip']

        # Initialize step counter
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

    def train_self_play(self, num_episodes=5000, stockfish_levels=None,
                      batch_size=64, save_interval=100, eval_interval=1000, target_level=7):
        """
        Train the model using pure self-play (model vs itself and model pool).

        Args:
            num_episodes (int): Total number of episodes to train
            stockfish_levels (list): List of Stockfish levels to evaluate against
            batch_size (int): Batch size for optimization
            save_interval (int): How often to save models (in episodes)
            eval_interval (int): How often to evaluate against Stockfish (in episodes)
            target_level (int): Target Stockfish level to achieve

        Returns:
            tuple: (episode_rewards, episode_lengths, losses)
        """
        return self.self_play_trainer.train_self_play(
            num_episodes, stockfish_levels,
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
        Load a model from disk to continue training from a saved checkpoint.

        This method supports two types of checkpoints:
        1. Simple model state dict (.pt files)
        2. Comprehensive checkpoints with optimizer state and training stats

        Usage examples:
            # Load from models directory
            trainer.load_model("model_pgn_checkpoint.pt")

            # Load comprehensive checkpoint
            trainer.load_model("checkpoint_complete_1000.pt")

            # Load from absolute path
            trainer.load_model("/path/to/model_self_play_episode_1000.pt")

        Args:
            filename (str): Name of the file to load the model from
                If filename is a full path, it will be used directly
                If filename is just a name, it will be looked for in the model_dir

        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        # Check if filename is a full path or just a filename
        if os.path.dirname(filename):
            filepath = filename  # Use the provided path directly
        else:
            filepath = os.path.join(self.model_dir, filename)  # Look in model_dir

        if os.path.exists(filepath):
            try:
                # Load state dict with appropriate device mapping
                checkpoint = torch.load(filepath, map_location=self.device)

                # Check if this is a comprehensive checkpoint or just a model state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    print(f"Loading comprehensive checkpoint from {filepath}")

                    # Load model state
                    self.policy_net.load_state_dict(checkpoint['model_state_dict'])

                    # Load target network if available
                    if 'target_net_state_dict' in checkpoint:
                        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                    else:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    # Load optimizer state if available
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("Optimizer state restored")

                    # Load training stats if available
                    if 'training_stats' in checkpoint:
                        self.training_stats = checkpoint['training_stats']
                        print("Training statistics restored")

                    # Load episode count if available
                    if 'episode' in checkpoint:
                        self.steps_done = checkpoint.get('total_positions', 0)
                        print(f"Resuming from episode {checkpoint['episode']+1} with {self.steps_done} steps done")
                else:
                    # Simple model state dict
                    self.policy_net.load_state_dict(checkpoint)
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"Model state loaded from {filepath}")

                # Make sure models are on the correct device
                self.policy_net.to(self.device)
                self.target_net.to(self.device)

                print(f"Model loaded successfully to {self.device}")
                return True
            except Exception as e:
                print(f"Error loading model from {filepath}: {e}")
                import traceback
                traceback.print_exc()
                return False
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

        # Update exploration rate metric
        self.metrics.update_exploration(eps_threshold)

        # Move tensors to the correct device
        state = state.to(self.device)
        mask = mask.to(self.device)

        if sample > eps_threshold:
            with torch.no_grad():
                # Use policy network to select best action with FP16 if enabled
                if self.use_mixed_precision and self.fp16_enabled:
                    # Convert to FP16 for inference
                    state_fp16 = state.to(dtype=torch.float16)
                    mask_fp16 = mask.to(dtype=torch.float16)

                    # Use autocast for mixed precision inference
                    with autocast('cuda'):
                        q_values = self.policy_net(state_fp16, mask_fp16)
                        # Clip Q-values to prevent drift
                        q_values = torch.clamp(q_values, -10.0, 10.0)
                else:
                    # Standard full precision inference
                    q_values = self.policy_net(state, mask)
                    # Clip Q-values to prevent drift
                    q_values = torch.clamp(q_values, -10.0, 10.0)

                # Track average Q-values for monitoring
                max_q_value = q_values.max().item()
                if len(self.training_stats['avg_q_values']) < 1000:
                    self.training_stats['avg_q_values'].append(max_q_value)
                else:
                    self.training_stats['avg_q_values'] = self.training_stats['avg_q_values'][1:] + [max_q_value]

                # Track position seen
                self.metrics.increment_positions()

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

        This implementation uses:
        1. Double DQN to prevent overestimation bias
           - Policy network selects actions
           - Target network evaluates those actions
        2. Q-value clipping to range [-10, 10] to prevent drift
        3. Target network updates every 500 episodes
        4. Prioritized experience replay for better sample efficiency

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
        weights = weights.detach().clone().to(self.device)
        # weights = torch.tensor(weights, dtype=torch.float32).to(self.device) #uncomment for previous implementation

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        # Use mixed precision for forward pass if available
        if self.use_mixed_precision:
            # Convert input tensors to FP16 for mixed precision training
            if self.fp16_enabled:
                # Convert state and mask tensors to float16 for forward pass
                state_batch_fp16 = state_batch.to(dtype=torch.float16)
                mask_batch_fp16 = mask_batch.to(dtype=torch.float16)
                next_state_batch_fp16 = next_state_batch.to(dtype=torch.float16)
                next_mask_batch_fp16 = next_mask_batch.to(dtype=torch.float16)

                with autocast('cuda'):
                    # Forward pass with FP16 tensors
                    q_values = self.policy_net(state_batch_fp16, mask_batch_fp16)
                    # Clip Q-values to prevent drift
                    q_values = torch.clamp(q_values, -10.0, 10.0)
                    state_action_values = q_values.gather(1, action_batch)

                    # Compute V(s_{t+1}) for all next states using Double DQN
                    with torch.no_grad():
                        # Get actions from policy network (Double DQN)
                        next_q_values_policy = self.policy_net(next_state_batch_fp16, next_mask_batch_fp16)
                        next_actions = next_q_values_policy.max(1)[1].unsqueeze(1)

                        # Get Q-values from target network for those actions
                        next_q_values_target = self.target_net(next_state_batch_fp16, next_mask_batch_fp16)
                        next_state_values = next_q_values_target.gather(1, next_actions).squeeze(1)

                        # Clip Q-values to prevent drift
                        next_state_values = torch.clamp(next_state_values, -10.0, 10.0)

                        # Set V(s) = 0 for terminal states
                        next_state_values[done_batch] = 0.0
            else:
                # Use autocast without explicit conversion
                with autocast('cuda'):
                    q_values = self.policy_net(state_batch, mask_batch)
                    # Clip Q-values to prevent drift
                    q_values = torch.clamp(q_values, -10.0, 10.0)
                    state_action_values = q_values.gather(1, action_batch)

                    # Compute V(s_{t+1}) for all next states using Double DQN
                    with torch.no_grad():
                        # Get actions from policy network (Double DQN)
                        next_q_values_policy = self.policy_net(next_state_batch, next_mask_batch)
                        next_actions = next_q_values_policy.max(1)[1].unsqueeze(1)

                        # Get Q-values from target network for those actions
                        next_q_values_target = self.target_net(next_state_batch, next_mask_batch)
                        next_state_values = next_q_values_target.gather(1, next_actions).squeeze(1)

                        # Clip Q-values to prevent drift
                        next_state_values = torch.clamp(next_state_values, -10.0, 10.0)

                        # Set V(s) = 0 for terminal states
                        next_state_values[done_batch] = 0.0
        else:
            # Standard full precision forward pass
            q_values = self.policy_net(state_batch, mask_batch)
            # Clip Q-values to prevent drift
            q_values = torch.clamp(q_values, -10.0, 10.0)
            state_action_values = q_values.gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states using Double DQN
            with torch.no_grad():
                # Get actions from policy network (Double DQN)
                next_q_values_policy = self.policy_net(next_state_batch, next_mask_batch)
                next_actions = next_q_values_policy.max(1)[1].unsqueeze(1)

                # Get Q-values from target network for those actions
                next_q_values_target = self.target_net(next_state_batch, next_mask_batch)
                next_state_values = next_q_values_target.gather(1, next_actions).squeeze(1)

                # Clip Q-values to prevent drift
                next_state_values = torch.clamp(next_state_values, -10.0, 10.0)

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

        # Optimize the model with mixed precision if available
        self.optimizer.zero_grad()

        if self.use_mixed_precision:
            # Use mixed precision training
            self.scaler.scale(loss).backward()

            # Improved gradient clipping for better stability with mixed precision
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)

            # Update weights with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard full precision training
            loss.backward()

            # Improved gradient clipping for better stability
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)

            self.optimizer.step()

        # Update simple metrics
        loss_value = loss.item()
        self.metrics.update_loss(loss_value)

        # Update Q-value metrics if available
        if hasattr(self, 'training_stats') and 'avg_q_values' in self.training_stats and self.training_stats['avg_q_values']:
            self.metrics.update_q_value(self.training_stats['avg_q_values'][-1])

        return loss_value

    def close(self):
        """Close all resources."""
        self.reward_calculator.close()
        self.model_evaluator.close()

    def print_training_metrics(self):
        """
        Print current training metrics to the console.
        """
        self.metrics.print_metrics()

    def plot_metrics(self, output_dir="metrics"):
        """
        Generate and save plots of the training metrics.

        Args:
            output_dir (str): Directory to save the plots
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate the plots
        metrics_path = os.path.join(output_dir, "training_metrics.png")
        plot_training_metrics(self.metrics, metrics_path)

        summary_path = os.path.join(output_dir, "performance_summary.png")
        plot_performance_summary(self.metrics, summary_path)

        progress_path = os.path.join(output_dir, "training_progress.png")
        plot_training_progress(self.metrics, progress_path)

        print(f"Metrics plots saved to {output_dir}")


def main():
    """Main function for command-line interface."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train chess model with optimized performance")
    parser.add_argument("--data_dir", default="data", help="Directory containing PGN files")
    parser.add_argument("--num_games", type=int, default=100, help="Maximum number of games to process")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save_interval", type=int, default=100, help="How often to save the model (in games)")
    parser.add_argument("--self_play", action="store_true", help="Run self-play training after PGN training")
    parser.add_argument("--self_play_episodes", type=int, default=1000, help="Number of self-play episodes")
    #Mertcan stockfish = python main.py --stockfish_path "/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish" --self_play
    #Can stockfish = python main.py --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe" --self_play
    parser.add_argument("--stockfish_path", default=None, help="Path to Stockfish executable")
    parser.add_argument("--target_level", type=int, default=7,
                        help="Target Stockfish level to achieve")
    parser.add_argument("--eval_interval", type=int, default=200,
                        help="How often to evaluate against Stockfish (in episodes)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model against Stockfish levels")
    parser.add_argument("--eval_games", type=int, default=5, help="Number of games to play against each Stockfish level")
    parser.add_argument("--eval_model", help="Path to specific model to evaluate (optional)")
    parser.add_argument("--min_level", type=int, default=1, help="Minimum Stockfish level to test against")
    parser.add_argument("--max_level", type=int, default=10, help="Maximum Stockfish level to test against")
    parser.add_argument("--load_model", help="Path to a model file to load before training (to continue training)")

    args = parser.parse_args()

    # Print training configuration
    print("\n=== Chess Model Training Configuration ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Maximum games: {args.num_games}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save interval: {args.save_interval}")
    print(f"Continue training from model: {args.load_model if args.load_model else 'No (starting fresh)'}")
    print(f"Self-play training: {'Yes' if args.self_play else 'No'}")
    if args.self_play:
        print(f"Self-play episodes: {args.self_play_episodes}")
    print(f"Evaluate against Stockfish: {'Yes' if args.evaluate else 'No'}")
    if args.evaluate:
        print(f"Evaluation games per level: {args.eval_games}")
        print(f"Stockfish levels: {args.min_level}-{args.max_level}")
    print(f"Stockfish path: {args.stockfish_path}")
    print("==========================================\n")

    # Create trainer with optimized hyperparameters for RTX 4070
    trainer = ChessTrainer(stockfish_path=args.stockfish_path, gpu_type="rtx_4070")

    print("\n🚀 GPU Diagnostic:")
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
    else:
        print("⚠️  Warning: CUDA not available — model is using CPU.")


    # Load model if specified (to continue training from a saved model)
    if args.load_model:
        print(f"\nLoading model from {args.load_model} to continue training...")
        if trainer.load_model(args.load_model):
            print("Model loaded successfully. Training will continue from this model.")
        else:
            print("Failed to load model. Training will start with a new model.")

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

    # Continue with pure self-play training if requested
    if args.self_play:
        print("\nStarting pure self-play training (model vs itself and model pool)...")
        print(f"Target: Achieve Stockfish level {args.target_level} strength")
        print(f"Evaluation will occur every {args.eval_interval} games")
        print("Using model pool for diverse opponents")
        print("Using batch processing for better efficiency")
        print("Using comprehensive checkpointing for continuous training")

        # Run self-play training with improved parameters
        trainer.train_self_play(
            num_episodes=args.self_play_episodes,
            stockfish_levels=[1, 2, 3, 4, 5],  # Include lower levels for testing
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
