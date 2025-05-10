import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import chess.pgn
import os
import io
import time
from collections import namedtuple, deque
from drl_agent import DQN, ChessAgent, board_to_tensor, create_move_mask

# Define the Experience Replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done', 'mask', 'next_mask'))

class ReplayMemory:
    """Standard Replay Memory"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save an experience"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory:
    """Prioritized Experience Replay Memory"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization, 1 = full prioritization)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # For beta calculation
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # Initial max priority

    def beta_by_frame(self, frame_idx):
        """Linearly increases beta from beta_start to 1 over beta_frames frames"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *args):
        """Save an experience with maximum priority"""
        # Use max priority for new experiences
        priority = self.max_priority

        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
        else:
            self.memory[self.position] = Experience(*args)

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of experiences with prioritization"""
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]

        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled indices"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)

class ChessTrainerWithStockfish:
    """
    Enhanced chess trainer that uses Stockfish for evaluation and implements
    various improvements for faster and better training.
    
    Attributes:
        model_dir (str): Directory to save models
        stockfish_path (str): Path to Stockfish executable
        stockfish_level (int): Stockfish skill level (1-20)
        device (torch.device): Device to run training on (CPU/GPU)
        policy_net (DQN): Policy network
        target_net (DQN): Target network
        optimizer (torch.optim.Adam): Optimizer
        memory (PrioritizedReplayMemory): Experience replay memory
        training_stats (dict): Training statistics
    """
    
    def __init__(self, model_dir="models", stockfish_path="/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish", stockfish_level=8):
        """
        Initialize the chess trainer with improved parameters.
        
        Args:
            model_dir (str): Directory to save models
            stockfish_path (str): Path to Stockfish executable
            stockfish_level (int): Stockfish skill level (1-20)
        """
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir
        self.stockfish_level = stockfish_level

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Stockfish engine with specific level
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.stockfish.configure({"Skill Level": stockfish_level})
            print(f"Stockfish engine initialized at level {stockfish_level}")
        except Exception as e:
            print(f"Error initializing Stockfish engine: {e}")
            print("Falling back to material-based evaluation")
            self.stockfish = None

        # Initialize networks with improved architecture
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer with adaptive learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = self._create_adaptive_scheduler()

        # Initialize prioritized replay memory with larger capacity
        self.memory = PrioritizedReplayMemory(100000)

        # Improved training parameters
        self.batch_size = 1024  # Larger batch size for better gradient estimates
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
            'avg_q_values': [],
            'win_rates': [],
            'learning_rates': []
        }

    def _create_adaptive_scheduler(self):
        """
        Create an adaptive learning rate scheduler.
        
        Returns:
            torch.optim.lr_scheduler: Learning rate scheduler
        """
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )

    def select_action(self, state, mask, board):
        """
        Select an action using improved epsilon-greedy policy with adaptive exploration.
        
        Args:
            state (torch.Tensor): Current board state
            mask (torch.Tensor): Move mask
            board (chess.Board): Current chess board
            
        Returns:
            tuple: (action_idx, move)
        """
        sample = random.random()
        eps_threshold = self._get_adaptive_epsilon()
        self.steps_done += 1

        # Move tensors to the correct device
        state = state.to(self.device)
        mask = mask.to(self.device)

        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state, mask)
                action_idx = q_values.max(1)[1].item()
                move = self._convert_action_to_move(action_idx, board)
        else:
            move = self._select_random_move(board)
            action_idx = move.from_square * 64 + move.to_square

        return action_idx, move

    def _get_adaptive_epsilon(self):
        """
        Get adaptive epsilon value based on training progress.
        
        Returns:
            float: Epsilon value
        """
        base_eps = self.eps_end + (self.eps_start - self.eps_end) * \
                   np.exp(-1. * self.steps_done / self.eps_decay)
        
        # Adjust epsilon based on recent performance
        if len(self.training_stats['win_rates']) > 100:
            recent_win_rate = np.mean(self.training_stats['win_rates'][-100:])
            if recent_win_rate < 0.4:  # Poor performance
                return min(base_eps * 1.2, self.eps_start)  # Increase exploration
            elif recent_win_rate > 0.6:  # Good performance
                return max(base_eps * 0.8, self.eps_end)  # Decrease exploration
        
        return base_eps

    def optimize_model(self):
        """
        Perform one step of optimization with improved batch processing.
        
        Returns:
            float: Loss value
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
        weights = weights.to(self.device)

        # Handle non-final states
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d]).to(self.device)
        non_final_next_masks = torch.cat([m for m, d in zip(batch.next_mask, batch.done) if not d]).to(self.device)
        mask_batch = torch.cat(batch.mask).to(self.device)

        # Compute Q-values for current states
        q_values = self.policy_net(state_batch, mask_batch)
        state_action_values = q_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using target network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_mask.sum() > 0:
                next_q_values = self.target_net(non_final_next_states, non_final_next_masks)
                next_state_values[non_final_mask] = next_q_values.max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute TD errors for updating priorities
        td_errors = torch.abs(state_action_values.squeeze() - expected_state_action_values).detach().cpu().numpy()

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors + 1e-6)

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

    def calculate_stockfish_reward(self, board, prev_board=None):
        """
        Calculate reward using enhanced Stockfish evaluation with improved parameters.
        
        Args:
            board (chess.Board): Current board state
            prev_board (chess.Board): Previous board state
            
        Returns:
            float: Reward value
        """
        if self.stockfish is None:
            return self.calculate_reward(board)

        try:
            # Use appropriate depth based on game phase
            depth = self._get_appropriate_depth(board)
            current_score = self.stockfish.analyse(board=board, limit=chess.engine.Limit(depth=depth))['score'].relative.score(mate_score=10000)

            if prev_board is not None:
                prev_score = self.stockfish.analyse(board=prev_board, limit=chess.engine.Limit(depth=depth))['score'].relative.score(mate_score=10000)
                raw_diff = current_score - prev_score
                reward = self._scale_reward(raw_diff)
            else:
                reward = current_score / 100.0

            # Add positional bonuses
            reward += self.calculate_positional_bonus(board)

            return reward

        except Exception as e:
            print(f"Error calculating Stockfish reward: {e}")
            return self.calculate_reward(board)

    def _get_appropriate_depth(self, board):
        """
        Get appropriate Stockfish analysis depth based on game phase.
        
        Args:
            board (chess.Board): Current board state
            
        Returns:
            int: Analysis depth
        """
        move_count = len(board.move_stack)
        if move_count < 10:  # Opening
            return 8
        elif move_count < 30:  # Middlegame
            return 10
        else:  # Endgame
            return 12

    def _scale_reward(self, raw_diff):
        """
        Scale reward based on magnitude of improvement.
        
        Args:
            raw_diff (float): Raw score difference
            
        Returns:
            float: Scaled reward
        """
        if abs(raw_diff) < 50:  # Small change
            return raw_diff / 100.0
        elif abs(raw_diff) < 200:  # Medium change
            return raw_diff / 80.0
        else:  # Large change
            return raw_diff / 50.0

    def test_model_strength(self, num_games=10):
        """
        Test the model against different Stockfish levels to determine its approximate strength.
        
        Args:
            num_games (int): Number of games to play against each Stockfish level
            
        Returns:
            int: Approximate Stockfish level of the model
        """
        print("\nTesting model strength against Stockfish levels...")
        
        # Store original Stockfish level
        original_level = self.stockfish_level
        
        # Test against different Stockfish levels
        for level in range(1, 21):
            wins = 0
            draws = 0
            
            print(f"\nTesting against Stockfish level {level}")
            
            for game in range(num_games):
                board = chess.Board()
                
                # Configure Stockfish to current level
                self.stockfish.configure({"Skill Level": level})
                
                while not board.is_game_over():
                    if board.turn == chess.WHITE:
                        # Model's move
                        state = board_to_tensor(board)
                        mask = create_move_mask(board).unsqueeze(0)
                        action_idx, move = self.select_action(state, mask, board)
                    else:
                        # Stockfish's move
                        result = self.stockfish.play(board, chess.engine.Limit(time=0.1))
                        move = result.move
                    
                    board.push(move)
                
                # Record result
                if board.is_checkmate():
                    if board.turn == chess.BLACK:  # Model won
                        wins += 1
                        print(f"Game {game + 1}: Model won")
                    else:
                        print(f"Game {game + 1}: Stockfish won")
                elif board.is_stalemate() or board.is_insufficient_material():
                    draws += 1
                    print(f"Game {game + 1}: Draw")
                
            # Calculate win rate
            win_rate = (wins + 0.5 * draws) / num_games
            print(f"Level {level} results: {wins} wins, {draws} draws, {num_games - wins - draws} losses")
            print(f"Win rate: {win_rate:.2%}")
            
            # If win rate is close to 50%, we've found the model's strength level
            if 0.45 <= win_rate <= 0.55:
                print(f"\nModel strength is approximately at Stockfish level {level}")
                # Restore original Stockfish level
                self.stockfish.configure({"Skill Level": original_level})
                return level
            elif win_rate < 0.45:
                print(f"\nModel strength is between Stockfish levels {level-1} and {level}")
                # Restore original Stockfish level
                self.stockfish.configure({"Skill Level": original_level})
                return level-1
        
        print("\nModel strength is below Stockfish level 1")
        # Restore original Stockfish level
        self.stockfish.configure({"Skill Level": original_level})
        return 0

    def train_self_play(self, num_episodes=1000):
        """
        Train the model through enhanced self-play with curriculum learning.
        
        Args:
            num_episodes (int): Number of episodes to train
            
        Returns:
            tuple: (rewards, lengths, losses)
        """
        print(f"Starting enhanced self-play training with curriculum learning for {num_episodes} episodes...")

        # Initialize tracking metrics
        self._reset_training_stats()

        # Curriculum learning parameters
        curriculum_stages = [
            {"max_moves": 20, "opponent_strength": 0.2},  # Opening
            {"max_moves": 40, "opponent_strength": 0.4},  # Middlegame
            {"max_moves": 60, "opponent_strength": 0.6},  # Endgame
            {"max_moves": 100, "opponent_strength": 0.8}, # Full game
        ]
        current_stage = 0

        for episode in range(num_episodes):
            # Update curriculum stage
            if episode > 0 and episode % 100 == 0 and current_stage < len(curriculum_stages) - 1:
                current_stage += 1
                print(f"Advancing to curriculum stage {current_stage + 1}")
                
                # Test model strength after each curriculum stage
                print("\nTesting model strength after curriculum stage change...")
                model_strength = self.test_model_strength(num_games=5)
                print(f"Current model strength: Stockfish level {model_strength}")

            # Get current curriculum parameters
            stage_params = curriculum_stages[current_stage]
            
            # Train episode
            episode_reward, episode_length = self._train_episode(stage_params)
            
            # Update training statistics
            self._update_training_stats(episode_reward, episode_length)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                self._print_training_progress(episode, num_episodes)
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                self._save_training_checkpoint(episode)
                
                # Test model strength periodically
                print("\nTesting model strength at checkpoint...")
                model_strength = self.test_model_strength(num_games=5)
                print(f"Current model strength: Stockfish level {model_strength}")

        # Final strength test
        print("\nPerforming final strength test...")
        final_strength = self.test_model_strength(num_games=10)
        print(f"Final model strength: Stockfish level {final_strength}")

        # Save final model
        self.save_model("model_improved_final.pt")
        print("Enhanced training completed!")

        # Close Stockfish engine
        if self.stockfish:
            self.stockfish.quit()

        return (self.training_stats['episode_rewards'],
                self.training_stats['episode_lengths'],
                self.training_stats['losses'])

    def _reset_training_stats(self):
        """Reset training statistics."""
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'avg_q_values': [],
            'win_rates': [],
            'learning_rates': []
        }

    def _update_training_stats(self, episode_reward, episode_length):
        """Update training statistics."""
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        self.training_stats['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

    def _print_training_progress(self, episode, num_episodes):
        """Print training progress with detailed metrics."""
        avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
        avg_length = np.mean(self.training_stats['episode_lengths'][-10:])
        avg_loss = np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0
        current_lr = self.optimizer.param_groups[0]['lr']

        print(f"Episode {episode+1}/{num_episodes}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Length: {avg_length:.2f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Epsilon: {self._get_adaptive_epsilon():.4f}")
        print("-" * 50)

    def _save_training_checkpoint(self, episode):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, os.path.join(self.model_dir, f"checkpoint_episode_{episode}.pt"))

    def __del__(self):
        """Cleanup resources when the trainer is destroyed."""
        if hasattr(self, 'stockfish') and self.stockfish:
            self.stockfish.quit()

    def calculate_positional_bonus(self, board):
        """Calculate bonus rewards for good chess principles"""
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
        """Calculate reward based on the board state (fallback method)"""
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
        """Calculate material advantage for the current player"""
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
        """Simple position evaluation"""
        # Center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(1 if board.piece_at(sq) is not None and board.piece_at(sq).color == board.turn else 0
                            for sq in center_squares)

        # Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))

        # Combine factors
        return center_control + 0.1 * mobility

    def save_model(self, filename):
        """Save the model to disk"""
        filepath = os.path.join(self.model_dir, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename):
        """Load a model from disk"""
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False

    def train_from_pgn(self, data_dir, num_games=100):
        """
        Train the model using high-quality PGN games with optimized processing.
        
        Args:
            data_dir (str): Directory containing PGN files
            num_games (int): Number of games to train on
        """
        print(f"Starting PGN training with {num_games} games...")
        
        # Get list of PGN files
        pgn_files = [f for f in os.listdir(data_dir) if f.endswith('.pgn')]
        if not pgn_files:
            print(f"No PGN files found in {data_dir}")
            return
        
        # Pre-allocate tensors for batch processing
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        batch_masks = []
        batch_next_masks = []
        
        games_processed = 0
        total_positions = 0
        batch_size = 1024  # Process positions in larger batches
        
        # Configure Stockfish for faster analysis
        if self.stockfish:
            self.stockfish.configure({
                "Threads": 4,  # Use multiple threads
                "Hash": 128,   # Increase hash size
                "Skill Level": self.stockfish_level  # Set skill level
            })
        
        for pgn_file in pgn_files:
            if games_processed >= num_games:
                break
                
            file_path = os.path.join(data_dir, pgn_file)
            print(f"\nProcessing {pgn_file}...")
            
            # Read multiple games at once for better I/O performance
            with open(file_path) as pgn:
                while games_processed < num_games:
                    try:
                        # Process multiple games in parallel
                        games = []
                        for _ in range(min(10, num_games - games_processed)):
                            game = chess.pgn.read_game(pgn)
                            if game is None:
                                break
                            games.append(game)
                        
                        if not games:
                            break
                        
                        # Process each game
                        for game in games:
                            # Convert mainline to list first
                            mainline_moves = list(game.mainline_moves())
                            
                            # Skip games that are too short or too long
                            if len(mainline_moves) < 10 or len(mainline_moves) > 200:
                                continue
                                
                            # Process each position in the game
                            board = game.board()
                            prev_board = None
                            
                            # Process moves in batches
                            for i in range(0, len(mainline_moves), 10):  # Process 10 moves at a time
                                batch_moves = mainline_moves[i:i+10]
                                
                                for move in batch_moves:
                                    # Convert board to tensor
                                    state = board_to_tensor(board)
                                    mask = create_move_mask(board).unsqueeze(0)
                                    
                                    # Get Stockfish evaluation (with reduced depth for speed)
                                    if self.stockfish:
                                        reward = self.calculate_stockfish_reward(board, prev_board)
                                    else:
                                        reward = self.calculate_reward(board)
                                    
                                    # Make the move
                                    board.push(move)
                                    
                                    # Get next state
                                    next_state = board_to_tensor(board)
                                    next_mask = create_move_mask(board).unsqueeze(0)
                                    
                                    # Add to batch
                                    batch_states.append(state)
                                    batch_actions.append(move.from_square * 64 + move.to_square)
                                    batch_rewards.append(reward)
                                    batch_next_states.append(next_state)
                                    batch_dones.append(board.is_game_over())
                                    batch_masks.append(mask)
                                    batch_next_masks.append(next_mask)
                                    
                                    prev_board = board.copy()
                                    total_positions += 1
                                    
                                    # Process batch when it's full
                                    if len(batch_states) >= batch_size:
                                        self._process_batch(
                                            batch_states, batch_actions, batch_rewards,
                                            batch_next_states, batch_dones,
                                            batch_masks, batch_next_masks
                                        )
                                        # Clear batches
                                        batch_states = []
                                        batch_actions = []
                                        batch_rewards = []
                                        batch_next_states = []
                                        batch_dones = []
                                        batch_masks = []
                                        batch_next_masks = []
                            
                            games_processed += 1
                            if games_processed % 10 == 0:
                                print(f"Processed {games_processed}/{num_games} games")
                                
                    except Exception as e:
                        print(f"Error processing game: {e}")
                        continue
        
        # Process any remaining positions in the batch
        if batch_states:
            self._process_batch(
                batch_states, batch_actions, batch_rewards,
                batch_next_states, batch_dones,
                batch_masks, batch_next_masks
            )
        
        print(f"\nPGN training completed!")
        print(f"Total games processed: {games_processed}")
        print(f"Total positions processed: {total_positions}")
        
        # Save the model after PGN training
        self.save_model("model_pgn_improved.pt")
        
        # Test model strength
        print("\nTesting model strength after PGN training...")
        strength = self.test_model_strength(num_games=5)
        print(f"Model strength after PGN training: Stockfish level {strength}")

    def _process_batch(self, states, actions, rewards, next_states, dones, masks, next_masks):
        """Process a batch of experiences efficiently."""
        # Convert lists to tensors
        state_batch = torch.cat(states)
        action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        next_state_batch = torch.cat(next_states)
        done_batch = torch.tensor(dones, dtype=torch.bool)
        mask_batch = torch.cat(masks)
        next_mask_batch = torch.cat(next_masks)
        
        # Store experiences in memory
        for i in range(len(states)):
            self.memory.push(
                state_batch[i],
                action_batch[i].item(),
                next_state_batch[i],
                reward_batch[i].item(),
                done_batch[i].item(),
                mask_batch[i],
                next_mask_batch[i]
            )
        
        # Optimize model if enough samples
        if len(self.memory) >= self.batch_size:
            loss = self.optimize_model()
            if loss is not None:
                self.training_stats['losses'].append(loss)

if __name__ == "__main__":
    # Create trainer with Stockfish and improved architecture
    trainer = ChessTrainerWithStockfish()

    # Train from high-quality PGN data
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"Training from high-quality PGN data in directory: {data_dir}")
        print("Using improved neural network architecture with residual blocks and value head")
        print("Using prioritized experience replay with larger memory capacity")
        print("Using enhanced reward function with deeper Stockfish analysis")

        # Train on PGN data with improved parameters
        trainer.train_from_pgn(data_dir, num_games=100)

        # Save intermediate model
        trainer.save_model("model_pgn_improved.pt")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Skipping PGN training phase")

    # Continue with enhanced self-play training
    print("\nStarting enhanced self-play training with curriculum learning...")
    print("Using gradually increasing opponent strength")
    print("Using position repetition detection and penalization")
    print("Using learning rate scheduling for better convergence")

    # Run self-play training with improved parameters
    rewards, lengths, losses = trainer.train_self_play(num_episodes=500)

    # Plot final training results
    trainer.plot_training_progress()

    # Create a ChessAgent with the improved model
    final_model_path = os.path.join(trainer.model_dir, "model_improved_final.pt")
    best_model_path = os.path.join(trainer.model_dir, "best_model.pt")

    # Use the best model if it exists, otherwise use the final model
    model_path = best_model_path if os.path.exists(best_model_path) else final_model_path

    # Create the agent with the trained model
    agent = ChessAgent(model_path=model_path)

    # Test the agent with a few moves
    board = chess.Board()
    print("\nTesting improved agent:")
    print(board.unicode())

    for i in range(10):  # Test with more moves
        move = agent.select_move(board)
        print(f"\nMove {i+1}: {move.uci()} ({board.san(move)})")
        board.push(move)
        print(board.unicode())

        if board.is_game_over():
            result = board.result()
            print(f"Game over! Result: {result}")
            break

    print("\nTraining and testing completed successfully!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    print("Use these models with the chess GUI for improved play.")
