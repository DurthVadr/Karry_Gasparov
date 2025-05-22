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
import torch.nn.functional as F
import chess
import argparse

# Import torch.amp for mixed precision training
from torch.amp import GradScaler
from torch.amp import autocast

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
                step_size=2500,  # Reduced from 5000 for more frequent updates
                gamma=self.hyperparams['optimizer']['lr_factor']
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=50000,  # Reduced from 100000 for more aggressive decay
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

        # Initialize reward calculator with asynchronous evaluation and hybrid approach
        self.reward_calculator = RewardCalculator(
            stockfish_path=stockfish_path,
            use_async=True,
            num_workers=self.hyperparams['async_eval']['num_workers'],
            stockfish_eval_frequency=self.hyperparams['reward'].get('stockfish_eval_frequency', 0.2),
            hybrid_eval_enabled=self.hyperparams['reward'].get('hybrid_eval_enabled', False),
            low_freq_rate=self.hyperparams['reward'].get('low_freq_rate', 0.01),
            high_freq_rate=self.hyperparams['reward'].get('high_freq_rate', 0.2),
            high_freq_interval=self.hyperparams['reward'].get('high_freq_interval', 50)
        )

        # Initialize model evaluator with reference to this trainer
        self.model_evaluator = ModelEvaluator(
            stockfish_path=stockfish_path,
            use_fp16=self.fp16_enabled,
            trainer=self  # Pass reference to trainer for accessing hyperparameters
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

        # Gradient accumulation parameters
        self.use_gradient_accumulation = self.hyperparams['optimizer'].get('use_gradient_accumulation', False)
        self.accumulation_steps = self.hyperparams['optimizer'].get('accumulation_steps', 1)

        # Early stopping parameters for games
        self.max_moves = self.hyperparams['self_play']['max_moves']
        self.early_stopping_no_progress = self.hyperparams['self_play'].get('early_stopping_no_progress', 30)

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

    def select_action(self, state, mask, board, is_root=False):
        """
        Select an action using AlphaZero-style temperature-based sampling with Dirichlet noise.

        Args:
            state (torch.Tensor): Current state tensor
            mask (torch.Tensor): Mask of legal moves
            board (chess.Board): Current board position
            is_root (bool): Whether this is a root position (for Dirichlet noise)

        Returns:
            tuple: (action_idx, move) - The selected action index and chess move
        """
        self.steps_done += 1

        # Get legal moves
        legal_moves = list(board.legal_moves)
        legal_move_indices = [m.from_square * 64 + m.to_square for m in legal_moves]

        # Move tensors to the correct device
        state = state.to(self.device)
        mask = mask.to(self.device)

        # Check if this is a critical position that needs lookahead
        is_critical = board.is_check() or any(board.is_capture(move) for move in legal_moves)

        # For critical positions, use lookahead if enabled
        if is_critical and self.hyperparams.get('use_lookahead', False):
            from drl_agent import simple_lookahead
            action_idx = simple_lookahead(board, self.policy_net, self.device,
                                         depth=self.hyperparams.get('lookahead_depth', 1))

            # Convert action index to chess move
            from_square = action_idx // 64
            to_square = action_idx % 64
            move = chess.Move(from_square, to_square)

            # Handle promotion
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                if board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                    move.promotion = chess.QUEEN
                elif board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                    move.promotion = chess.QUEEN

            return action_idx, move

        with torch.no_grad():
            # Use policy network to get policy and value with FP16 if enabled
            if self.use_mixed_precision and self.fp16_enabled:
                # Convert to FP16 for inference
                state_fp16 = state.to(dtype=torch.float16)
                mask_fp16 = mask.to(dtype=torch.float16)

                # Use autocast for mixed precision inference
                with autocast(device_type='cuda'):
                    policy_logits, value = self.policy_net(state_fp16, mask_fp16)
            else:
                # Standard full precision inference
                policy_logits, value = self.policy_net(state, mask)

            # Track position seen
            self.metrics.increment_positions()

            # Convert logits to probabilities
            policy_probs = F.softmax(policy_logits, dim=1)

            # Add Dirichlet noise at root positions if enabled
            if is_root and self.hyperparams.get('use_dirichlet', False):
                from drl_agent import add_dirichlet_noise
                alpha = self.hyperparams.get('dirichlet_alpha', 0.3)
                epsilon = self.hyperparams.get('dirichlet_epsilon', 0.25)
                policy_probs = add_dirichlet_noise(policy_probs, legal_move_indices, alpha, epsilon)

            # Apply temperature-based sampling
            from drl_agent import temperature_sampling

            # Get temperature based on game phase or steps
            if self.hyperparams.get('use_phase_temperature', False):
                # Determine game phase
                piece_count = len(board.piece_map())
                if piece_count >= 28:  # Opening
                    temperature = self.hyperparams.get('opening_temperature', 1.0)
                elif piece_count <= 12:  # Endgame
                    temperature = self.hyperparams.get('endgame_temperature', 0.5)
                else:  # Middlegame
                    temperature = self.hyperparams.get('middlegame_temperature', 0.7)
            else:
                # Annealing temperature based on steps
                temperature = max(
                    self.hyperparams.get('min_temperature', 0.1),
                    self.hyperparams.get('initial_temperature', 1.0) *
                    np.exp(-self.steps_done / self.hyperparams.get('temperature_decay', 10000))
                )

            # Update exploration rate metric
            self.metrics.update_exploration(temperature)

            # Sample move based on policy probabilities and temperature
            action_idx = temperature_sampling(policy_probs, legal_move_indices, temperature)

            # Track average value for monitoring
            value_scalar = value.item()
            if len(self.training_stats.get('avg_values', [])) < 1000:
                if 'avg_values' not in self.training_stats:
                    self.training_stats['avg_values'] = []
                self.training_stats['avg_values'].append(value_scalar)
            else:
                self.training_stats['avg_values'] = self.training_stats['avg_values'][1:] + [value_scalar]

            # Convert action index to chess move
            from_square = action_idx // 64
            to_square = action_idx % 64
            move = chess.Move(from_square, to_square)

            # Handle promotion
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                if board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                    move.promotion = chess.QUEEN
                elif board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                    move.promotion = chess.QUEEN

            # Ensure the move is legal (should always be the case with our sampling)
            if move not in legal_moves:
                # Fallback to a random legal move if something went wrong
                move = random.choice(legal_moves)
                action_idx = move.from_square * 64 + move.to_square

        return action_idx, move

    def optimize_model(self, accumulate_gradients=False, accumulation_steps=1, current_step=0):
        """
        Perform one step of optimization with AlphaZero-style combined policy and value loss.

        This implementation uses:
        1. Policy loss: cross-entropy between predicted policy and target policy
        2. Value loss: mean squared error between predicted value and target value
        3. Combined loss with appropriate scaling to balance the two components
        4. Prioritized experience replay for better sample efficiency
        5. Gradient accumulation for effective larger batch sizes

        Args:
            accumulate_gradients (bool): Whether to accumulate gradients across multiple batches
            accumulation_steps (int): Number of batches to accumulate gradients over
            current_step (int): Current accumulation step (0-indexed)

        Returns:
            float: Loss value if optimization was performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Only zero gradients at the start of accumulation
        if not accumulate_gradients or current_step == 0:
            self.optimizer.zero_grad()

        # Sample a batch from prioritized memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Move tensors to the correct device with optimized GPU transfer
        state_batch = torch.cat(batch.state).to(self.device, non_blocking=True)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device, non_blocking=True)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device, non_blocking=True)
        next_state_batch = torch.cat(batch.next_state).to(self.device, non_blocking=True)
        done_batch = torch.tensor(batch.done, dtype=torch.bool).to(self.device, non_blocking=True)
        mask_batch = torch.cat(batch.mask).to(self.device, non_blocking=True)
        next_mask_batch = torch.cat(batch.next_mask).to(self.device, non_blocking=True)
        weights = weights.detach().clone().to(self.device, non_blocking=True)

        # Use CUDA streams for parallel data transfer if available
        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure all data is on GPU before proceeding

        # Forward pass to get policy logits and value predictions
        if self.use_mixed_precision:
            # Convert input tensors to FP16 for mixed precision training
            if self.fp16_enabled:
                # Convert state and mask tensors to float16 for forward pass
                state_batch_fp16 = state_batch.to(dtype=torch.float16)
                mask_batch_fp16 = mask_batch.to(dtype=torch.float16)
                next_state_batch_fp16 = next_state_batch.to(dtype=torch.float16)
                next_mask_batch_fp16 = next_mask_batch.to(dtype=torch.float16)

                with autocast(device_type='cuda'):
                    # Forward pass with FP16 tensors for current state
                    policy_logits, value = self.policy_net(state_batch_fp16, mask_batch_fp16)

                    # Get target values from target network
                    with torch.no_grad():
                        _, next_value = self.target_net(next_state_batch_fp16, next_mask_batch_fp16)
                        # Set value to 0 for terminal states
                        next_value[done_batch] = 0.0
                        # Compute target value using TD(0)
                        target_value = (next_value * self.gamma) + reward_batch.unsqueeze(1)
            else:
                # Use autocast without explicit conversion
                with autocast(device_type='cuda'):
                    # Forward pass for current state
                    policy_logits, value = self.policy_net(state_batch, mask_batch)

                    # Get target values from target network
                    with torch.no_grad():
                        _, next_value = self.target_net(next_state_batch, next_mask_batch)
                        # Set value to 0 for terminal states
                        next_value[done_batch] = 0.0
                        # Compute target value using TD(0)
                        target_value = (next_value * self.gamma) + reward_batch.unsqueeze(1)
        else:
            # Standard full precision forward pass
            policy_logits, value = self.policy_net(state_batch, mask_batch)

            # Get target values from target network
            with torch.no_grad():
                _, next_value = self.target_net(next_state_batch, next_mask_batch)
                # Set value to 0 for terminal states
                next_value[done_batch] = 0.0
                # Compute target value using TD(0)
                target_value = (next_value * self.gamma) + reward_batch.unsqueeze(1)

        # Create one-hot encoding of actions for policy loss
        action_indices = action_batch.squeeze(1)
        action_one_hot = torch.zeros(self.batch_size, 4096, device=self.device)
        action_one_hot.scatter_(1, action_indices.unsqueeze(1), 1)

        # Apply mask to policy logits to ensure only legal moves are considered
        masked_policy_logits = policy_logits.clone()
        masked_policy_logits[mask_batch == 0] = -1e9  # Set logits for illegal moves to very negative value

        # Compute policy loss (cross entropy between predicted policy and target policy)
        # First convert logits to probabilities
        policy_probs = F.softmax(masked_policy_logits, dim=1)
        # Add small epsilon to avoid log(0)
        policy_loss = -torch.sum(action_one_hot * torch.log(policy_probs + 1e-10), dim=1)

        # Compute value loss (MSE between predicted value and target value)
        value_loss = F.mse_loss(value, target_value, reduction='none')

        # Scale losses to avoid extremely low values
        # Typical policy loss might be around -log(0.1) = 2.3, while value loss might be around 0.1
        # We want to balance them so neither dominates
        policy_scale = self.hyperparams.get('policy_loss_scale', 1.0)
        value_scale = self.hyperparams.get('value_loss_scale', 1.0)

        # Combine losses with appropriate scaling
        combined_loss = (policy_scale * policy_loss) + (value_scale * value_loss.squeeze())

        # Apply importance sampling weights from prioritized replay
        weighted_loss = (combined_loss * weights).mean()

        # Compute TD errors for updating priorities (using combined loss)
        td_errors = combined_loss.detach().cpu().numpy()

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors + 1e-6)  # Small constant to avoid zero priority

        # Scale loss by accumulation steps if accumulating gradients
        if accumulate_gradients:
            weighted_loss = weighted_loss / accumulation_steps

        if self.use_mixed_precision:
            # Use mixed precision training
            self.scaler.scale(weighted_loss).backward()

            # Only update weights at the end of accumulation
            if not accumulate_gradients or current_step == accumulation_steps - 1:
                # Improved gradient clipping for better stability with mixed precision
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)

                # Update weights with scaled gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update learning rate if scheduler is defined
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            # Standard full precision training
            weighted_loss.backward()

            # Only update weights at the end of accumulation
            if not accumulate_gradients or current_step == accumulation_steps - 1:
                # Improved gradient clipping for better stability
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
                self.optimizer.step()

                # Update learning rate if scheduler is defined
                if self.scheduler is not None:
                    self.scheduler.step()

        # Update simple metrics
        loss_value = weighted_loss.item()
        if accumulate_gradients:
            loss_value *= accumulation_steps  # Scale back for reporting

        self.metrics.update_loss(loss_value)

        # Update value metrics if available
        if hasattr(self, 'training_stats') and 'avg_values' in self.training_stats and self.training_stats['avg_values']:
            self.metrics.update_q_value(self.training_stats['avg_values'][-1])

        # Track separate policy and value losses for monitoring
        if 'policy_losses' not in self.training_stats:
            self.training_stats['policy_losses'] = []
        if 'value_losses' not in self.training_stats:
            self.training_stats['value_losses'] = []

        policy_loss_mean = policy_loss.mean().item() * policy_scale
        value_loss_mean = value_loss.mean().item() * value_scale

        if len(self.training_stats['policy_losses']) < 1000:
            self.training_stats['policy_losses'].append(policy_loss_mean)
            self.training_stats['value_losses'].append(value_loss_mean)
        else:
            self.training_stats['policy_losses'] = self.training_stats['policy_losses'][1:] + [policy_loss_mean]
            self.training_stats['value_losses'] = self.training_stats['value_losses'][1:] + [value_loss_mean]

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
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
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

    # Create trainer with optimized hyperparameters for the detected hardware
    gpu_type = "rtx_4070" if torch.cuda.is_available() and "RTX 4070" in torch.cuda.get_device_name(0) else "m2_mac"
    trainer = ChessTrainer(stockfish_path=args.stockfish_path, gpu_type=gpu_type)

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
