import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import io
from collections import namedtuple, deque
from drl_agent import DQN, ChessAgent, board_to_tensor, create_move_mask

# Define the Experience Replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done', 'mask', 'next_mask'))

class ReplayMemory:
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

class ChessTrainer:
    def __init__(self, model_dir="models"):
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir

        # Initialize networks
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

        # Initialize replay memory
        self.memory = ReplayMemory(10000)

        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.target_update = 10  # Update target network every N episodes

        self.steps_done = 0

    def select_action(self, state, mask, board):
        """Select an action using epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Use policy network to select best action
                q_values = self.policy_net(state, mask)
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

                # If move is not legal, choose a random legal move
                if move not in board.legal_moves:
                    legal_moves = list(board.legal_moves)
                    move = random.choice(legal_moves)
                    action_idx = move.from_square * 64 + move.to_square
        else:
            # Choose a random legal move
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            action_idx = move.from_square * 64 + move.to_square

        return action_idx, move

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

        # Handle non-final states
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])
        non_final_next_masks = torch.cat([m for m, d in zip(batch.next_mask, batch.done) if not d])

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch, torch.cat(batch.mask)).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states, non_final_next_masks).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train_self_play(self, num_episodes=1000):
        """Train the model through self-play"""
        print(f"Starting self-play training for {num_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []
        losses = []

        for episode in range(num_episodes):
            # Initialize the environment
            board = chess.Board()
            episode_reward = 0
            episode_length = 0

            # Get initial state
            state = board_to_tensor(board)
            mask = create_move_mask(board).unsqueeze(0)

            done = False
            while not done:
                # Select and perform an action
                action_idx, move = self.select_action(state, mask, board)

                # Execute the move
                board.push(move)

                # Get the next state
                next_state = board_to_tensor(board)
                next_mask = create_move_mask(board).unsqueeze(0)

                # Calculate reward
                reward = self.calculate_reward(board)

                # Check if the game is over
                done = board.is_game_over()

                # Store the transition in memory
                self.memory.push(state, action_idx, next_state, reward, done, mask, next_mask)

                # Move to the next state
                state = next_state
                mask = next_mask

                # Perform one step of optimization
                loss = self.optimize_model()
                if loss is not None:
                    losses.append(loss)

                episode_reward += reward
                episode_length += 1

                # Limit episode length to avoid very long games
                if episode_length >= 200:
                    done = True

            # Update the target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                avg_length = sum(episode_lengths[-10:]) / 10
                avg_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1)
                print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Avg Loss: {avg_loss:.4f}")

            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(f"model_episode_{episode+1}.pt")

        # Save final model
        self.save_model("model_final.pt")
        print("Training completed!")

        return episode_rewards, episode_lengths, losses

    def calculate_reward(self, board):
        """Calculate reward based on the board state"""
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

    def train_from_pgn(self, pgn_path, num_games=1000):
        """Train the model from PGN games

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
                game_count = self._process_pgn_file(pgn_file, game_count, num_games, losses)

                # Save model after each file
                self.save_model(f"model_pgn_checkpoint.pt")
        else:
            # Process a single PGN file
            print(f"Training from PGN file: {pgn_path}")
            game_count = self._process_pgn_file(pgn_path, game_count, num_games, losses)

        # Save final model
        self.save_model("model_pgn_final.pt")
        print(f"PGN training completed! Processed {game_count} games.")

        return losses

    def _process_pgn_file(self, pgn_file, initial_game_count, max_games, losses):
        """Process a single PGN file

        Args:
            pgn_file: Path to the PGN file
            initial_game_count: Starting game count
            max_games: Maximum number of games to process
            losses: List to store loss values

        Returns:
            Updated game count
        """
        game_count = initial_game_count

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
                    for i, move in enumerate(moves):
                        # Get current state
                        state = board_to_tensor(board)
                        mask = create_move_mask(board).unsqueeze(0)

                        # Convert move to action index
                        action_idx = move.from_square * 64 + move.to_square

                        # Make the move
                        board.push(move)

                        # Get next state
                        next_state = board_to_tensor(board)
                        next_mask = create_move_mask(board).unsqueeze(0)

                        # Calculate reward
                        reward = self.calculate_reward(board)

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
                    if game_count % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    # Print progress
                    if game_count % 10 == 0:
                        avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)
                        print(f"Processed {game_count}/{max_games} games | Avg Loss: {avg_loss:.4f}")

                    # Save model periodically
                    if game_count % 100 == 0:
                        self.save_model(f"model_pgn_{game_count}.pt")
        except Exception as e:
            print(f"Error processing file {pgn_file}: {e}")

        return game_count

if __name__ == "__main__":
    # Create trainer
    trainer = ChessTrainer()

    # Train from PGN data in the data directory
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"Training from PGN data in directory: {data_dir}")
        trainer.train_from_pgn(data_dir, num_games=500)
    else:
        print(f"Data directory not found: {data_dir}")

    # Continue with self-play training
    print("Starting self-play training...")
    trainer.train_self_play(num_episodes=500)

    # Create a ChessAgent with the trained model
    agent = ChessAgent(model_path=os.path.join(trainer.model_dir, "model_final.pt"))

    # Test the agent with a few moves
    board = chess.Board()
    print("\nTesting trained agent:")
    print(board.unicode())

    for _ in range(5):
        move = agent.select_move(board)
        print(f"\nSelected move: {move.uci()}")
        board.push(move)
        print(board.unicode())

        if board.is_game_over():
            print("Game over!")
            break
