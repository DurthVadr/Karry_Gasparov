"""
Main training script for chess reinforcement learning with Stockfish evaluation.

This script provides a command-line interface for training and evaluating
chess models using deep reinforcement learning with Stockfish evaluation.
"""

import os
import time
import torch
import torch.optim as optim
import chess
import argparse

# Import custom modules
from drl_agent import DQN, ChessAgent
from memory import PrioritizedReplayMemory
from reward import RewardCalculator
from evaluation import ModelEvaluator
from visualization import plot_training_progress
from training import PGNTrainer, SelfPlayTrainer

class ChessTrainerWithStockfish:
    """
    Main trainer class for chess reinforcement learning with Stockfish evaluation.
    
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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        
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
            return
        
        # Sample a batch from prioritized memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Move tensors to the correct device
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        weights = weights.to(self.device)
        
        # Handle non-final states
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d]).to(self.device)
        non_final_next_masks = torch.cat([m for m, d in zip(batch.next_mask, batch.done) if not d]).to(self.device)
        mask_batch = torch.cat(batch.mask).to(self.device)
        
        # Compute Q-values for current states
        q_values = self.policy_net(state_batch, mask_batch)
        
        # Get Q-values for chosen actions
        state_action_values = q_values.gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states using target network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_mask.sum() > 0:  # Check if there are any non-final states
                next_q_values = self.target_net(non_final_next_states, non_final_next_masks)
                next_state_values[non_final_mask] = next_q_values.max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(state_action_values.squeeze() - expected_state_action_values).detach().cpu().numpy()
        
        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors + 1e-6)  # Small constant to avoid zero priority
        
        # Compute weighted loss
        criterion = nn.SmoothL1Loss(reduction='none')
        elementwise_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = (elementwise_loss * weights.unsqueeze(1)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        
        self.optimizer.step()
        
        return loss.item()
    
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
    
    def plot_training_progress(self):
        """Plot training progress metrics."""
        plot_training_progress(
            self.training_stats,
            os.path.join(self.model_dir, 'training_progress.png')
        )
    
    def close(self):
        """Close all resources."""
        self.reward_calculator.close()
        self.model_evaluator.close()
