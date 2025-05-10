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

    # Mertcab's stockfish path for mac = stockfish_path="/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish"
    # Can stockfish_path="C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    def __init__(self, model_dir="models", stockfish_path="/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish"):
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir

        # Store the Stockfish path for later use
        self.stockfish_path = stockfish_path

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Stockfish engine
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            print(f"Stockfish engine initialized successfully from: {stockfish_path}")
        except Exception as e:
            print(f"Error initializing Stockfish engine: {e}")
            print("Falling back to material-based evaluation")
            self.stockfish = None

        # Initialize networks with original DQN architecture
        #self.policy_net = DQN() #uncomment for macOS
        #self.target_net = DQN() #uncomment for macOS
        self.policy_net = DQN().to(self.device) # Comment for macOS
        self.target_net = DQN().to(self.device) # Comment for macOS
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

    def train_self_play(self, num_episodes=5000, stockfish_opponent=True, stockfish_levels=None,
                      batch_size=64, save_interval=100, eval_interval=500, target_level=7):
        """
        Advanced self-play training with multiple opponents and adaptive difficulty

        This optimized version includes:
        1. Training against Stockfish at various levels
        2. Self-play against previous best models
        3. Batch processing of experiences
        4. Adaptive difficulty based on performance
        5. Regular evaluation against target Stockfish level
        6. Advanced curriculum learning

        Args:
            num_episodes: Total number of episodes to train
            stockfish_opponent: Whether to use Stockfish as an opponent
            stockfish_levels: List of Stockfish levels to train against (default: [1,2,3,4,5,6,7])
            batch_size: Batch size for optimization
            save_interval: How often to save models (in episodes)
            eval_interval: How often to evaluate against target level (in episodes)
            target_level: Target Stockfish level to achieve
        """
        print(f"Starting advanced self-play training for {num_episodes} episodes...")
        print(f"Target performance: Stockfish level {target_level}")

        # Set default Stockfish levels if not provided
        if stockfish_levels is None:
            stockfish_levels = list(range(1, target_level + 1))

        # Initialize tracking metrics
        self.training_stats['episode_rewards'] = []
        self.training_stats['episode_lengths'] = []
        self.training_stats['losses'] = []
        self.training_stats['win_rates'] = []

        # Initialize model pool for diverse opponents
        model_pool = []
        best_model_path = os.path.join(self.model_dir, "best_model.pt")

        # Track training progress
        start_time = time.time()
        total_positions = 0
        total_optimization_steps = 0

        # Adaptive curriculum learning parameters
        current_level = 1  # Start with easiest level
        win_rate_threshold = 0.6  # Move to next level when win rate exceeds this
        win_rate_window = 50  # Calculate win rate over this many games
        wins_against_current_level = 0
        games_against_current_level = 0

        # Create a separate Stockfish instance for opponents if needed
        opponent_stockfish = None
        if stockfish_opponent:
            try:
                # Use the stored Stockfish path instead of trying to get it from options
                opponent_stockfish = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                opponent_stockfish.configure({"Skill Level": current_level})
                print(f"Initialized opponent Stockfish at level {current_level} using path: {self.stockfish_path}")
            except Exception as e:
                print(f"Error initializing opponent Stockfish: {e}")
                stockfish_opponent = False

        # Main training loop
        for episode in range(num_episodes):
            # Determine opponent type for this episode
            use_stockfish = False
            use_model_pool = False

            if stockfish_opponent:
                # 70% chance to play against Stockfish when available
                if random.random() < 0.7:
                    use_stockfish = True

                    # Occasionally play against higher levels for challenge (10% chance)
                    if random.random() < 0.1 and current_level < max(stockfish_levels):
                        challenge_level = min(current_level + 2, max(stockfish_levels))
                        opponent_stockfish.configure({"Skill Level": challenge_level})
                        print(f"Challenge game against Stockfish level {challenge_level}")
                    else:
                        opponent_stockfish.configure({"Skill Level": current_level})

            # Use model pool if available and not using Stockfish
            if not use_stockfish and len(model_pool) > 0 and random.random() < 0.8:
                use_model_pool = True
                # Select a random model from the pool
                opponent_model = random.choice(model_pool)
                opponent_model.eval()  # Set to evaluation mode

            # Initialize the game
            board = chess.Board()
            episode_reward = 0
            episode_length = 0
            positions_seen = {}  # Track positions for repetition detection
            experiences = []  # Collect experiences for batch processing

            # Play the game
            done = False
            while not done:
                # Get current state
                state = board_to_tensor(board, self.device)
                mask = create_move_mask(board, self.device)

                if board.turn == chess.WHITE:  # Our model plays as White
                    # Select action using our policy
                    action_idx, move = self.select_action(state, mask, board)
                else:  # Opponent plays as Black
                    if use_stockfish:
                        # Get move from Stockfish
                        result = opponent_stockfish.play(board, chess.engine.Limit(time=0.1))
                        move = result.move
                        action_idx = move.from_square * 64 + move.to_square
                    elif use_model_pool:
                        # Get move from opponent model
                        with torch.no_grad():
                            q_values = opponent_model(state, mask)
                            legal_moves = list(board.legal_moves)
                            legal_move_indices = [m.from_square * 64 + m.to_square for m in legal_moves]
                            legal_q_values = q_values[0, legal_move_indices]
                            best_idx = torch.argmax(legal_q_values).item()
                            move = legal_moves[best_idx]
                            action_idx = legal_move_indices[best_idx]
                    else:
                        # Self-play against current policy
                        action_idx, move = self.select_action(state, mask, board)

                # Store current board for reward calculation
                prev_board = board.copy()

                # Execute the move
                board.push(move)

                # Get next state
                next_state = board_to_tensor(board, self.device)
                next_mask = create_move_mask(board, self.device)

                # Check if game is over
                done = board.is_game_over()

                # Calculate reward - use Stockfish evaluation for key positions
                if episode_length % 4 == 0 or done:  # 25% sampling + terminal states
                    reward = self.calculate_stockfish_reward(board, prev_board)
                else:
                    reward = self.calculate_reward(board)  # Faster material-based reward

                # Handle repetitions
                board_fen = board.fen().split(' ')[0]
                if board_fen in positions_seen:
                    positions_seen[board_fen] += 1
                    if positions_seen[board_fen] == 2:
                        reward -= 0.5  # Mild penalty for first repetition
                    elif positions_seen[board_fen] >= 3:
                        reward -= 2.0  # Severe penalty for threefold repetition
                        done = True
                else:
                    positions_seen[board_fen] = 1

                # Store experience for batch processing
                experiences.append((state, action_idx, next_state, reward, done, mask, next_mask))

                episode_reward += reward
                episode_length += 1
                total_positions += 1

                # End game if it's too long
                if episode_length >= 300:
                    done = True

            # Process experiences in batches
            for i in range(0, len(experiences), batch_size):
                batch = experiences[i:min(i+batch_size, len(experiences))]

                # Add experiences to memory
                for exp in batch:
                    self.memory.push(*exp)

                # Perform optimization if enough samples
                if len(self.memory) >= self.batch_size:
                    loss = self.optimize_model()
                    if loss is not None:
                        self.training_stats['losses'].append(loss)
                        total_optimization_steps += 1

            # Update target network periodically
            if total_optimization_steps > 0 and total_optimization_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"Target network updated after {total_optimization_steps} optimization steps")

            # Track metrics
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)

            # Update win statistics for curriculum learning
            if use_stockfish:
                games_against_current_level += 1
                if board.is_checkmate() and board.turn == chess.BLACK:  # Model (White) won
                    wins_against_current_level += 1

                # Calculate win rate over the window
                if games_against_current_level >= win_rate_window:
                    win_rate = wins_against_current_level / games_against_current_level
                    self.training_stats['win_rates'].append(win_rate)

                    # Advance to next level if win rate is high enough
                    if win_rate >= win_rate_threshold and current_level < max(stockfish_levels):
                        current_level += 1
                        print(f"\n=== ADVANCING TO STOCKFISH LEVEL {current_level} ===")
                        print(f"Win rate against level {current_level-1}: {win_rate*100:.1f}%")

                        # Reset statistics for new level
                        wins_against_current_level = 0
                        games_against_current_level = 0

                        # Add current model to the pool before advancing
                        if len(model_pool) < 5:  # Keep up to 5 models in the pool
                            model_copy = DQN()
                            model_copy.load_state_dict(self.policy_net.state_dict())
                            model_pool.append(model_copy)
                            print(f"Added current model to pool (size: {len(model_pool)})")
                        else:
                            # Replace a random model in the pool
                            idx = random.randint(0, len(model_pool) - 1)
                            model_copy = DQN()
                            model_copy.load_state_dict(self.policy_net.state_dict())
                            model_pool[idx] = model_copy
                            print(f"Updated model in pool at position {idx}")

            # Evaluate against target level periodically
            if stockfish_opponent and (episode + 1) % eval_interval == 0:
                print(f"\n=== Evaluating against Stockfish level {target_level} ===")
                # Save current model for evaluation
                eval_model_path = os.path.join(self.model_dir, "temp_eval_model.pt")
                torch.save(self.policy_net.state_dict(), eval_model_path)

                # Run evaluation
                results = self.evaluate_against_stockfish(
                    model_path=eval_model_path,
                    num_games=5,
                    stockfish_levels=[target_level]
                )

                # Check if we've reached target level
                if results and target_level in results:
                    score = results[target_level]['score']
                    if score >= 0.5:  # 50% or better score against target
                        print(f"\n!!! TARGET ACHIEVED: Model performs at Stockfish level {target_level} !!!")
                        print(f"Score: {score*100:.1f}%")

                        # Save this model as a milestone
                        milestone_path = os.path.join(self.model_dir, f"model_level_{target_level}.pt")
                        torch.save(self.policy_net.state_dict(), milestone_path)
                        print(f"Milestone model saved to {milestone_path}")

                        # Optionally increase target level
                        if target_level < 10:
                            target_level += 1
                            print(f"New target level: {target_level}")

            # Print progress
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0

                # Calculate averages
                avg_reward = sum(self.training_stats['episode_rewards'][-10:]) / 10
                avg_length = sum(self.training_stats['episode_lengths'][-10:]) / 10
                avg_loss = sum(self.training_stats['losses'][-100:]) / max(len(self.training_stats['losses'][-100:]), 1) if self.training_stats['losses'] else 0

                print(f"Episode {episode+1}/{num_episodes} | Level: {current_level}/{max(stockfish_levels)} | "
                      f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Loss: {avg_loss:.4f} | "
                      f"Positions: {total_positions} ({positions_per_second:.1f}/s) | "
                      f"Time: {elapsed_time/3600:.2f}h")

            # Save model periodically
            if (episode + 1) % save_interval == 0:
                self.save_model(f"model_self_play_episode_{episode+1}.pt")

                # Update best model if this is the best so far
                if episode_reward > max(self.training_stats['episode_rewards'][:-1], default=float('-inf')):
                    torch.save(self.policy_net.state_dict(), best_model_path)
                    print(f"New best model saved with reward: {episode_reward:.2f}")

                # Plot training progress
                self.plot_training_progress()

                # Apply learning rate scheduling
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Learning rate adjusted to: {current_lr:.6f}")

        # Save final model
        self.save_model("model_self_play_final.pt")

        # Clean up
        if opponent_stockfish:
            opponent_stockfish.quit()

        print(f"Advanced self-play training completed! Total positions: {total_positions}")
        print(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")

        return (self.training_stats['episode_rewards'],
                self.training_stats['episode_lengths'],
                self.training_stats['losses'])

    def plot_training_progress(self):
        """Plot training progress metrics with enhanced visualizations"""
        try:
            import matplotlib.pyplot as plt

            # Determine number of subplots needed
            num_plots = 4
            if 'win_rates' in self.training_stats and self.training_stats['win_rates']:
                num_plots = 5

            # Create figure with subplots
            _, axs = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))  # Use _ for unused figure variable

            # Plot episode rewards
            axs[0].plot(self.training_stats['episode_rewards'], color='blue')
            axs[0].set_title('Episode Rewards')
            axs[0].set_xlabel('Episode')
            axs[0].set_ylabel('Reward')

            # Add moving average for rewards
            if len(self.training_stats['episode_rewards']) > 10:
                window_size = min(50, len(self.training_stats['episode_rewards']) // 5)
                rewards_avg = [sum(self.training_stats['episode_rewards'][max(0, i-window_size):i+1]) /
                              min(window_size, i+1) for i in range(len(self.training_stats['episode_rewards']))]
                axs[0].plot(rewards_avg, color='red', linestyle='--', label=f'{window_size}-episode moving avg')
                axs[0].legend()

            # Plot episode lengths
            axs[1].plot(self.training_stats['episode_lengths'], color='green')
            axs[1].set_title('Episode Lengths')
            axs[1].set_xlabel('Episode')
            axs[1].set_ylabel('Length')

            # Plot losses
            if self.training_stats['losses']:
                # Use log scale for losses to better visualize changes
                axs[2].plot(self.training_stats['losses'], color='orange', alpha=0.5)
                axs[2].set_title('Training Loss')
                axs[2].set_xlabel('Optimization Step')
                axs[2].set_ylabel('Loss')

                # Add smoothed loss curve
                if len(self.training_stats['losses']) > 100:
                    window_size = min(500, len(self.training_stats['losses']) // 10)
                    losses_avg = []
                    for i in range(len(self.training_stats['losses'])):
                        start_idx = max(0, i - window_size)
                        losses_avg.append(sum(self.training_stats['losses'][start_idx:i+1]) / (i - start_idx + 1))
                    axs[2].plot(losses_avg, color='red', linewidth=2, label=f'{window_size}-step moving avg')
                    axs[2].legend()

            # Plot average Q-values
            if self.training_stats['avg_q_values']:
                axs[3].plot(self.training_stats['avg_q_values'], color='purple')
                axs[3].set_title('Average Q-Values')
                axs[3].set_xlabel('Step')
                axs[3].set_ylabel('Q-Value')

            # Plot win rates if available
            if 'win_rates' in self.training_stats and self.training_stats['win_rates']:
                axs[4].plot(self.training_stats['win_rates'], color='brown', marker='o', linestyle='-')
                axs[4].set_title('Win Rate Against Stockfish')
                axs[4].set_xlabel('Evaluation')
                axs[4].set_ylabel('Win Rate')
                axs[4].set_ylim([0, 1])
                axs[4].axhline(y=0.6, color='green', linestyle='--', label='Level-up Threshold')
                axs[4].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'training_progress.png'))
            plt.close()

            print("Enhanced training progress plot saved to training_progress.png")
        except Exception as e:
            print(f"Error plotting training progress: {e}")
            import traceback
            traceback.print_exc()
            # Continue without plotting if matplotlib is not available

    def train_from_pgn(self, pgn_path, num_games=1000, batch_size=32, save_interval=100):
        """Train the model from PGN games with optimized batch processing

        This improved version uses batch processing and more efficient tensor operations
        to significantly speed up training from PGN files.

        Args:
            pgn_path: Path to a PGN file or directory containing PGN files
            num_games: Maximum number of games to process
            batch_size: Number of positions to process in each batch
            save_interval: How often to save the model (in games)
        """
        game_count = 0
        losses = []
        start_time = time.time()
        total_positions = 0

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
                file_losses, file_games, file_positions = self._process_pgn_file_optimized(
                    pgn_file, game_count, num_games - game_count, batch_size, save_interval
                )

                losses.extend(file_losses)
                game_count += file_games
                total_positions += file_positions

                # Save model after each file
                self.save_model(f"model_pgn_checkpoint.pt")

                # Print statistics
                elapsed_time = time.time() - start_time
                positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0
                print(f"Progress: {game_count}/{num_games} games | {total_positions} positions | "
                      f"{positions_per_second:.1f} positions/sec | "
                      f"Time elapsed: {elapsed_time:.1f}s")
        else:
            # Process a single PGN file
            print(f"Training from PGN file: {pgn_path}")
            file_losses, file_games, file_positions = self._process_pgn_file_optimized(
                pgn_path, 0, num_games, batch_size, save_interval
            )

            losses.extend(file_losses)
            game_count = file_games
            total_positions = file_positions

        # Save final model
        self.save_model("model_pgn_final.pt")

        # Print final statistics
        elapsed_time = time.time() - start_time
        positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0
        print(f"PGN training completed! Processed {game_count} games, {total_positions} positions")
        print(f"Average speed: {positions_per_second:.1f} positions/sec")
        print(f"Total training time: {elapsed_time:.1f} seconds ({elapsed_time/3600:.2f} hours)")

        return losses

    def _process_pgn_file_optimized(self, pgn_file, start_count, max_games, batch_size=32, save_interval=100):
        """Process a single PGN file for training with optimized batch processing

        This improved version:
        1. Collects positions in batches before processing
        2. Uses vectorized operations where possible
        3. Performs fewer Stockfish evaluations by sampling
        4. Optimizes memory usage and tensor operations

        Args:
            pgn_file: Path to the PGN file
            start_count: Starting game count
            max_games: Maximum number of games to process
            batch_size: Number of positions to process in each batch
            save_interval: How often to save the model (in games)

        Returns:
            Tuple of (losses, game_count, position_count)
        """
        losses = []
        game_count = 0
        position_count = 0
        optimization_steps = 0

        # Track time for performance monitoring
        start_time = time.time()
        last_report_time = start_time

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

                    # Process positions in the game
                    positions = []

                    # First, collect all positions from the game
                    for i, move in enumerate(moves):
                        # Store current board and move
                        current_board = board.copy()

                        # Convert move to action index
                        action_idx = move.from_square * 64 + move.to_square

                        # Make the move
                        board.push(move)

                        # Check if game is over
                        done = board.is_game_over()

                        # Store position data
                        positions.append((current_board, action_idx, board.copy(), done))

                    # Process positions in batches for efficiency
                    for i in range(0, len(positions), batch_size):
                        batch_positions = positions[i:i+batch_size]

                        # Process this batch
                        current_boards = [pos[0] for pos in batch_positions]
                        action_indices = [pos[1] for pos in batch_positions]
                        next_boards = [pos[2] for pos in batch_positions]
                        dones = [pos[3] for pos in batch_positions]

                        # Calculate rewards in batch (with sampling for Stockfish to reduce bottleneck)
                        rewards = []
                        for j, (curr_board, next_board) in enumerate(zip(current_boards, next_boards)):
                            # Use Stockfish for a sample of positions to reduce computation
                            if j % 4 == 0:  # 25% sampling rate
                                reward = self.calculate_stockfish_reward(next_board, curr_board)
                            else:
                                # Use faster material-based reward for other positions
                                reward = self.calculate_reward(next_board)
                            rewards.append(reward)

                        # Convert boards to tensors efficiently (in batch)
                        states = []
                        masks = []
                        next_states = []
                        next_masks = []

                        for curr_board, next_board in zip(current_boards, next_boards):
                            # Convert to tensors and move to device in one operation
                            states.append(board_to_tensor(curr_board, self.device))
                            masks.append(create_move_mask(curr_board, self.device))
                            next_states.append(board_to_tensor(next_board, self.device))
                            next_masks.append(create_move_mask(next_board, self.device))

                        # Store experiences in memory
                        for state, action_idx, next_state, reward, done, mask, next_mask in zip(
                            states, action_indices, next_states, rewards, dones, masks, next_masks
                        ):
                            self.memory.push(state, action_idx, next_state, reward, done, mask, next_mask)
                            position_count += 1

                        # Perform optimization if enough samples
                        if len(self.memory) >= self.batch_size:
                            loss = self.optimize_model()
                            if loss is not None:
                                losses.append(loss)
                                optimization_steps += 1

                    game_count += 1

                    # Update target network periodically based on optimization steps
                    if optimization_steps > 0 and optimization_steps % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    # Print progress every 10 seconds
                    current_time = time.time()
                    if current_time - last_report_time > 10:
                        elapsed_time = current_time - start_time
                        positions_per_second = position_count / elapsed_time if elapsed_time > 0 else 0
                        avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1) if losses else 0

                        print(f"Processed {start_count + game_count} games | {position_count} positions | "
                              f"{positions_per_second:.1f} pos/sec | Avg Loss: {avg_loss:.4f}")

                        last_report_time = current_time

                    # Save model periodically
                    if game_count % save_interval == 0:
                        self.save_model(f"model_pgn_{start_count + game_count}.pt")

                        # Apply learning rate scheduling
                        self.scheduler.step()
                        current_lr = self.scheduler.get_last_lr()[0]
                        print(f"Learning rate adjusted to: {current_lr:.6f}")

        except Exception as e:
            print(f"Error processing file {pgn_file}: {e}")
            import traceback
            traceback.print_exc()

        return losses, game_count, position_count

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

    def evaluate_against_stockfish(self, model_path, num_games=10, max_moves=100, stockfish_levels=range(1, 11)):
        """
        Evaluate a trained model against different Stockfish levels

        Args:
            model_path: Path to the model file to evaluate
            num_games: Number of games to play against each Stockfish level
            max_moves: Maximum number of moves per game
            stockfish_levels: Range of Stockfish levels to test against

        Returns:
            Dictionary with results for each Stockfish level
        """
        # Load the model to evaluate
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return None

        # Create a chess agent with the model
        agent = ChessAgent(model_path=model_path)

        # Initialize results dictionary
        results = {}

        # Test against each Stockfish level
        for level in stockfish_levels:
            print(f"\nEvaluating against Stockfish level {level}...")

            # Configure Stockfish for this level
            if self.stockfish:
                self.stockfish.configure({"Skill Level": level})
            else:
                print("Stockfish engine not available. Cannot evaluate.")
                return None

            # Track results for this level
            wins = 0
            losses = 0
            draws = 0

            # Play games
            for game_num in range(num_games):
                print(f"Game {game_num+1}/{num_games} against Stockfish level {level}")

                # Initialize board
                board = chess.Board()

                # Play until game over or max moves reached
                move_count = 0
                while not board.is_game_over() and move_count < max_moves:
                    # Model plays as white (first move)
                    if board.turn == chess.WHITE:
                        # Get move from our model
                        model_move = agent.select_move(board)
                        board.push(model_move)
                        print(f"Model move: {model_move.uci()}")
                    else:
                        # Get move from Stockfish
                        result = self.stockfish.play(board, chess.engine.Limit(time=0.1))
                        stockfish_move = result.move
                        board.push(stockfish_move)
                        print(f"Stockfish move: {stockfish_move.uci()}")

                    move_count += 1

                    # Check if game is over
                    if board.is_game_over():
                        break

                # Determine result
                if board.is_checkmate():
                    if board.turn == chess.BLACK:  # White (our model) won
                        wins += 1
                        print("Result: Model won")
                    else:  # Black (Stockfish) won
                        losses += 1
                        print("Result: Stockfish won")
                else:  # Draw
                    draws += 1
                    print("Result: Draw")

                print(f"Current score vs Level {level}: +{wins} -{losses} ={draws}")

            # Store results for this level
            results[level] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': wins / num_games,
                'score': (wins + 0.5 * draws) / num_games
            }

            print(f"Final score vs Level {level}: +{wins} -{losses} ={draws}")
            print(f"Win rate: {results[level]['win_rate']*100:.1f}%")
            print(f"Score: {results[level]['score']*100:.1f}%")

        # Determine approximate Stockfish level of the model
        best_comparable_level = 0
        for level in stockfish_levels:
            if results[level]['score'] >= 0.45:  # Model is competitive (45%+ score)
                best_comparable_level = level

        # Print summary
        print("\n=== Model Evaluation Summary ===")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Games per level: {num_games}")

        if best_comparable_level > 0:
            print(f"The model plays approximately at Stockfish level {best_comparable_level} strength")
            if results[best_comparable_level]['score'] > 0.55:
                print(f"The model is better than Stockfish level {best_comparable_level}")
            elif results[best_comparable_level]['score'] >= 0.45:
                print(f"The model is almost as good as Stockfish level {best_comparable_level}")
            else:
                print(f"The model is not quite at Stockfish level {best_comparable_level} yet")
        else:
            print("The model is below Stockfish level 1 strength")

        return results

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train chess model with optimized performance")
    parser.add_argument("--data_dir", default="data", help="Directory containing PGN files")
    parser.add_argument("--num_games", type=int, default=200000, help="Maximum number of games to process")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save_interval", type=int, default=1000, help="How often to save the model (in games)")
    parser.add_argument("--self_play", action="store_true", help="Run self-play training after PGN training")
    parser.add_argument("--self_play_episodes", type=int, default=5000, help="Number of self-play episodes")
    parser.add_argument("--stockfish_path", default="/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish",
                        help="Path to Stockfish executable")
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

    # Evaluation-only mode
    if args.evaluate and args.eval_model and not os.path.exists(args.data_dir):
        # Create trainer with Stockfish
        trainer = ChessTrainerWithStockfish(stockfish_path=args.stockfish_path)

        # Evaluate the specified model
        print(f"\nEvaluating model: {args.eval_model}")
        print(f"Testing against Stockfish levels {args.min_level}-{args.max_level}")
        print(f"Playing {args.eval_games} games per level")

        # Run evaluation
        trainer.evaluate_against_stockfish(
            model_path=args.eval_model,
            num_games=args.eval_games,
            stockfish_levels=range(args.min_level, args.max_level + 1)
        )

        # Exit after evaluation
        exit(0)

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

    # Create trainer with Stockfish and improved architecture
    trainer = ChessTrainerWithStockfish(stockfish_path=args.stockfish_path)

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
        rewards, lengths, losses = trainer.train_self_play(
            num_episodes=args.self_play_episodes,
            stockfish_opponent=args.stockfish_opponent,
            stockfish_levels=list(range(1, args.target_level + 1)),
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            target_level=args.target_level
        )

        # Plot final training results
        trainer.plot_training_progress()

    # Create a ChessAgent with the improved model
    final_model_path = os.path.join(trainer.model_dir, "model_improved_final.pt")
    best_model_path = os.path.join(trainer.model_dir, "best_model.pt")
    pgn_model_path = os.path.join(trainer.model_dir, "model_pgn_improved.pt")

    # Determine which model to use for testing
    if args.eval_model:
        test_model_path = args.eval_model
        print(f"Using specified model for testing: {test_model_path}")
    elif os.path.exists(best_model_path):
        test_model_path = best_model_path
        print(f"Using best model for testing: {best_model_path}")
    elif os.path.exists(final_model_path):
        test_model_path = final_model_path
        print(f"Using final model for testing: {final_model_path}")
    else:
        test_model_path = pgn_model_path
        print(f"Using PGN-trained model for testing: {pgn_model_path}")

    # Create the agent with the trained model
    agent = ChessAgent(model_path=test_model_path)

    # Test the agent with a few moves
    board = chess.Board()
    print("\nTesting trained agent:")
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

    # Evaluate against Stockfish if requested
    if args.evaluate:
        print("\nEvaluating model against different Stockfish levels...")
        trainer.evaluate_against_stockfish(
            model_path=test_model_path,
            num_games=args.eval_games,
            stockfish_levels=range(args.min_level, args.max_level + 1)
        )

    print("\nTraining and testing completed successfully!")
    print("\nAvailable models:")
    if os.path.exists(pgn_model_path):
        print(f"- PGN-trained model: {pgn_model_path}")
    if os.path.exists(final_model_path):
        print(f"- Self-play final model: {final_model_path}")
    if os.path.exists(best_model_path):
        print(f"- Self-play best model: {best_model_path}")
    print("\nUse these models with the chess GUI for improved play.")
