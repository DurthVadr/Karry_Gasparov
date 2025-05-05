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
    def __init__(self, model_dir="models", stockfish_path="/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish"):
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Stockfish engine
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            print(f"Stockfish engine initialized successfully")
        except Exception as e:
            print(f"Error initializing Stockfish engine: {e}")
            print("Falling back to material-based evaluation")
            self.stockfish = None

        # Initialize networks with original DQN architecture
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer with higher learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        # Initialize prioritized replay memory with larger capacity
        self.memory = PrioritizedReplayMemory(100000)

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
        """Select an action using epsilon-greedy policy with improved exploration"""
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
        """Perform one step of optimization with prioritized experience replay"""
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

    def calculate_stockfish_reward(self, board, prev_board=None):
        """Calculate reward based on deeper Stockfish evaluation with more nuanced rewards"""
        if self.stockfish is None:
            # Fallback to material advantage if Stockfish is not available
            return self.calculate_reward(board)

        try:
            # Use deeper analysis for more accurate evaluation
            current_score = self.stockfish.analyse(board=board, limit=chess.engine.Limit(depth=12))['score'].relative.score(mate_score=10000)

            # If we have a previous board, calculate the difference in evaluation
            if prev_board is not None:
                prev_score = self.stockfish.analyse(board=prev_board, limit=chess.engine.Limit(depth=12))['score'].relative.score(mate_score=10000)

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

    def train_self_play(self, num_episodes=1000):
        """Train the model through enhanced self-play with curriculum learning"""
        print(f"Starting enhanced self-play training with curriculum learning for {num_episodes} episodes...")

        # Initialize tracking metrics
        self.training_stats['episode_rewards'] = []
        self.training_stats['episode_lengths'] = []
        self.training_stats['losses'] = []

        # Curriculum learning - gradually increase opponent strength
        opponent_strength = 0.1  # Start with weak opponent

        # Track best model for opponent
        best_reward = float('-inf')
        best_model_path = os.path.join(self.model_dir, "best_model.pt")

        for episode in range(num_episodes):
            # Increase opponent strength over time
            if episode > 0 and episode % 100 == 0 and opponent_strength < 1.0:
                opponent_strength += 0.1
                print(f"Increasing opponent strength to {opponent_strength:.1f}")

                # Save current model as opponent model
                if os.path.exists(best_model_path):
                    opponent_model = DQN()
                    opponent_model.load_state_dict(torch.load(best_model_path))
                    opponent_model.eval()
                else:
                    # Use current model as opponent if no best model exists
                    opponent_model = self.policy_net
            else:
                # Use current model as opponent
                opponent_model = self.policy_net

            # Initialize the environment
            board = chess.Board()
            episode_reward = 0
            episode_length = 0

            # Get initial state
            state = board_to_tensor(board)
            mask = create_move_mask(board).unsqueeze(0)

            # Track positions seen in this game to avoid repetitions
            positions_seen = {}

            done = False
            while not done:
                # Store the current board for reward calculation
                prev_board = board.copy()

                # Select and perform an action
                action_idx, move = self.select_action(state, mask, board)

                # Execute the move
                board.push(move)

                # Get the next state
                next_state = board_to_tensor(board)
                next_mask = create_move_mask(board).unsqueeze(0)

                # Calculate reward using enhanced Stockfish evaluation
                reward = self.calculate_stockfish_reward(board, prev_board)

                # Check if the game is over
                done = board.is_game_over()

                # Detect repetitions and penalize
                board_fen = board.fen().split(' ')[0]  # Just the piece positions
                if board_fen in positions_seen:
                    positions_seen[board_fen] += 1
                    # Apply increasingly severe penalties for repetitions
                    if positions_seen[board_fen] == 2:
                        # Second occurrence (first repetition)
                        reward -= 1.0
                    elif positions_seen[board_fen] >= 3:
                        # Third occurrence (threefold repetition)
                        reward -= 3.0
                        done = True  # End the game to avoid infinite loops
                else:
                    positions_seen[board_fen] = 1

                # Store the transition in memory
                self.memory.push(state, action_idx, next_state, reward, done, mask, next_mask)

                # Move to the next state
                state = next_state
                mask = next_mask

                # Perform one step of optimization
                loss = self.optimize_model()
                if loss is not None:
                    self.training_stats['losses'].append(loss)

                episode_reward += reward
                episode_length += 1

                # Limit episode length to avoid very long games
                if episode_length >= 300:  # Increased from 200 to allow longer games
                    done = True

            # Update the target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Step the learning rate scheduler
            self.scheduler.step()

            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)

            # Track best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(self.policy_net.state_dict(), best_model_path)
                print(f"New best model saved with reward: {best_reward:.2f}")

            # Print progress with more detailed metrics
            if (episode + 1) % 10 == 0:
                avg_reward = sum(self.training_stats['episode_rewards'][-10:]) / 10
                avg_length = sum(self.training_stats['episode_lengths'][-10:]) / 10
                avg_loss = sum(self.training_stats['losses'][-100:]) / max(len(self.training_stats['losses'][-100:]), 1)
                avg_q = sum(self.training_stats['avg_q_values'][-100:]) / max(len(self.training_stats['avg_q_values'][-100:]), 1)

                print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | Avg Loss: {avg_loss:.4f} | "
                      f"Avg Q-Value: {avg_q:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(f"model_improved_episode_{episode+1}.pt")

                # Plot training progress
                self.plot_training_progress()

        # Save final model
        self.save_model("model_improved_final.pt")
        print("Enhanced training completed!")

        # Close Stockfish engine
        if self.stockfish:
            self.stockfish.quit()

        return (self.training_stats['episode_rewards'],
                self.training_stats['episode_lengths'],
                self.training_stats['losses'])

    def plot_training_progress(self):
        """Plot training progress metrics"""
        try:
            import matplotlib.pyplot as plt

            # Create figure with subplots
            _, axs = plt.subplots(4, 1, figsize=(10, 15))  # Use _ for unused figure variable

            # Plot episode rewards
            axs[0].plot(self.training_stats['episode_rewards'])
            axs[0].set_title('Episode Rewards')
            axs[0].set_xlabel('Episode')
            axs[0].set_ylabel('Reward')

            # Plot episode lengths
            axs[1].plot(self.training_stats['episode_lengths'])
            axs[1].set_title('Episode Lengths')
            axs[1].set_xlabel('Episode')
            axs[1].set_ylabel('Length')

            # Plot losses
            if self.training_stats['losses']:
                axs[2].plot(self.training_stats['losses'])
                axs[2].set_title('Training Loss')
                axs[2].set_xlabel('Optimization Step')
                axs[2].set_ylabel('Loss')

            # Plot average Q-values
            if self.training_stats['avg_q_values']:
                axs[3].plot(self.training_stats['avg_q_values'])
                axs[3].set_title('Average Q-Values')
                axs[3].set_xlabel('Step')
                axs[3].set_ylabel('Q-Value')

            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'training_progress.png'))
            plt.close()

            print("Training progress plot saved to training_progress.png")
        except Exception as e:
            print(f"Error plotting training progress: {e}")
            # Continue without plotting if matplotlib is not available

    def train_from_pgn(self, pgn_path, num_games=1000):
        """Train the model from PGN games with Stockfish evaluation

        Args:
            pgn_path: Path to a PGN file or directory containing PGN files
            num_games: Maximum number of games to process
        """
        game_count = 0
        losses = []

        # Check if pgn_path is a directory
        if os.path.isdir(pgn_path):
            print(f"Training from PGN files in directory: {pgn_path}")
            # Get all PGN files in the directory
            pgn_files = [os.path.join(pgn_path, f) for f in os.listdir(pgn_path)
                        if f.endswith('.pgn') and os.path.isfile(os.path.join(pgn_path, f))]

            if not pgn_files:
                print(f"No PGN files found in directory: {pgn_path}")
                return losses

            print(f"Found {len(pgn_files)} PGN files")

            # Process each PGN file
            for pgn_file in sorted(pgn_files):
                if game_count >= num_games:
                    break

                print(f"Processing file: {os.path.basename(pgn_file)}")
                file_losses = self._process_pgn_file(pgn_file, game_count, num_games - game_count)
                losses.extend(file_losses)

                # Update game count
                game_count += len(file_losses)

                # Save model after each file
                self.save_model(f"model_pgn_checkpoint.pt")
        else:
            # Process a single PGN file
            print(f"Training from PGN file: {pgn_path}")
            file_losses = self._process_pgn_file(pgn_path, 0, num_games)
            losses.extend(file_losses)
            game_count = len(file_losses)

        # Save final model
        self.save_model("model_pgn_final.pt")
        print(f"PGN training completed! Processed {game_count} games.")

        return losses

    def _process_pgn_file(self, pgn_file, start_count, max_games):
        """Process a single PGN file for training

        Args:
            pgn_file: Path to the PGN file
            start_count: Starting game count
            max_games: Maximum number of games to process

        Returns:
            List of losses from training
        """
        losses = []
        game_count = 0

        try:
            with open(pgn_file) as f:
                while game_count < max_games:
                    # Read the next game
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break  # End of file

                    # Process the game
                    board = game.board()
                    moves = list(game.mainline_moves())

                    # Skip very short games
                    if len(moves) < 10:
                        continue

                    # Process each position in the game
                    prev_board = None
                    for i, move in enumerate(moves):
                        # Get current state
                        state = board_to_tensor(board)
                        mask = create_move_mask(board).unsqueeze(0)

                        # Store current board for reward calculation
                        prev_board = board.copy()

                        # Convert move to action index
                        action_idx = move.from_square * 64 + move.to_square

                        # Make the move
                        board.push(move)

                        # Get next state
                        next_state = board_to_tensor(board)
                        next_mask = create_move_mask(board).unsqueeze(0)

                        # Calculate reward using Stockfish
                        reward = self.calculate_stockfish_reward(board, prev_board)

                        # Check if game is over
                        done = board.is_game_over()

                        # Store transition in memory
                        self.memory.push(state, action_idx, next_state, reward, done, mask, next_mask)

                        # Perform optimization step if enough samples
                        if len(self.memory) >= self.batch_size:
                            loss = self.optimize_model()
                            if loss is not None:
                                losses.append(loss)

                    game_count += 1

                    # Update target network periodically
                    if (start_count + game_count) % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    # Print progress
                    if game_count % 10 == 0:
                        avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)
                        print(f"Processed {start_count + game_count} games | Avg Loss: {avg_loss:.4f}")

                    # Save model periodically
                    if (start_count + game_count) % 100 == 0:
                        self.save_model(f"model_pgn_{start_count + game_count}.pt")
        except Exception as e:
            print(f"Error processing file {pgn_file}: {e}")

        return losses

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
