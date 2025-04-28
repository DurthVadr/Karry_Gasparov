"""
Chess Bot Training on Google Colab

This script allows you to train the deep reinforcement learning chess agent.
Copy and paste this code into a Google Colab notebook.

Instructions:
1. Create a new notebook on Google Colab
2. Copy this entire script into a code cell
3. Upload drl_agent.py to your Colab environment
4. Upload your PGN data or use the synthetic data generator
5. Run the cells in order
"""

# Install dependencies
!pip install python-chess==1.11.2 torch numpy matplotlib

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import io
import time
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# Create directories
!mkdir -p /content/models
!mkdir -p /content/data/synthetic

# Optional: Generate synthetic data if you don't have your own PGN files
def generate_synthetic_data(num_games=1000, output_file="/content/data/synthetic/synthetic_games.pgn"):
    """Generate a dataset of random chess games"""
    import chess
    import chess.pgn
    import random
    import os
    
    def generate_random_game(max_moves=100):
        """Generate a random chess game"""
        board = chess.Board()
        game = chess.pgn.Game()
        
        # Set some game headers
        game.headers["Event"] = "Synthetic Game"
        game.headers["Site"] = "Colab Synthetic Database"
        game.headers["Date"] = "2025.04.27"
        game.headers["Round"] = "1"
        game.headers["White"] = "Engine1"
        game.headers["Black"] = "Engine2"
        game.headers["Result"] = "*"
        
        node = game
        
        # Make random moves until the game is over or max_moves is reached
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            move = random.choice(legal_moves)
            board.push(move)
            node = node.add_variation(move)
            move_count += 1
        
        # Set the result
        if board.is_checkmate():
            result = "1-0" if board.turn == chess.BLACK else "0-1"
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
            result = "1/2-1/2"
        else:
            result = "*"
        
        game.headers["Result"] = result
        
        return game

    def save_pgn(game, filename):
        """Save a game to a PGN file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a') as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)
            f.write("\n\n")  # Add some space between games
    
    # Clear file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        
    for i in range(num_games):
        game = generate_random_game()
        save_pgn(game, output_file)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} games")
    
    print(f"Dataset generated and saved to {output_file}")

# Uncomment to generate synthetic data
# generate_synthetic_data(num_games=1000)

# Define the DQN model and helper functions
# Note: This should match your drl_agent.py implementation
# If you've uploaded drl_agent.py, you can import from it instead

# Mask Layer for handling valid moves
class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def forward(self, x, mask):
        # Ensure mask is boolean or float tensor of 0s and 1s
        # Reshape mask to match the output shape if necessary
        mask_reshaped = mask.view_as(x)
        # Apply mask: set invalid move scores to a very small number (or -inf)
        # Using -inf ensures that softmax output for invalid moves is zero
        masked_output = x.masked_fill(mask_reshaped == 0, -float("inf"))
        return masked_output

# Deep Q-Network (DQN) Architecture
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # Input: 8x8 board, 16 channels (6 white, 6 black, empty, castling, en passant, player)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Flattened size: 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 64, 4096) # Reduced intermediate layer size
        self.fc2 = nn.Linear(4096, 4096) # Output: 64*64 = 4096 possible moves
        
        self.mask_layer = MaskLayer()

    def forward(self, x, mask=None):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch
        
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x) # Raw scores for each move
        
        if mask is not None:
            x = self.mask_layer(x, mask)
            
        return x

# Helper function to convert chess board to tensor representation
def board_to_tensor(board):
    """Converts a chess.Board object to a 16x8x8 tensor."""
    tensor = np.zeros((16, 8, 8), dtype=np.float32)
    
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    for i, piece in enumerate(pieces):
        # White pieces
        for square in board.pieces(piece, chess.WHITE):
            rank, file = chess.square_rank(square), chess.square_file(square)
            tensor[i, rank, file] = 1
        # Black pieces
        for square in board.pieces(piece, chess.BLACK):
            rank, file = chess.square_rank(square), chess.square_file(square)
            tensor[i + 6, rank, file] = 1
            
    # Castling rights (binary encoded)
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12, 0, 7] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[12, 0, 0] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[13, 7, 7] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 7, 0] = 1

    # En passant square
    if board.ep_square:
        rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        tensor[14, rank, file] = 1
        
    # Player to move (1 for White, 0 for Black - consistent layer)
    if board.turn == chess.WHITE:
        tensor[15, :, :] = 1
    else:
        tensor[15, :, :] = 0 # Or -1 if preferred
        
    return torch.from_numpy(tensor).unsqueeze(0) # Add batch dimension

# Helper function to create the move mask
def create_move_mask(board):
    """Creates a 4096-element mask tensor for legal moves."""
    mask = torch.zeros(4096, dtype=torch.float32)
    for move in board.legal_moves:
        index = move.from_square * 64 + move.to_square
        # Handle promotion - for simplicity, allow any promotion for now
        if move.promotion:
             # Simple approach: mark the basic move index
             mask[index] = 1
        else:
            mask[index] = 1
    return mask.unsqueeze(0) # Add batch dimension

# Basic Agent Implementation
class ChessAgent:
    def __init__(self, model_path=None):
        self.policy_net = DQN()
        if model_path and os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        self.policy_net.eval() # Set to evaluation mode by default

    def select_move(self, board):
        """Selects the best move based on the current policy network."""
        with torch.no_grad():
            state_tensor = board_to_tensor(board)
            move_mask = create_move_mask(board)
            
            # Get Q-values for all moves
            q_values = self.policy_net(state_tensor, move_mask)
            
            # Select the move with the highest Q-value
            best_move_index = torch.argmax(q_values).item()
            
            # Map index back to move
            from_square = best_move_index // 64
            to_square = best_move_index % 64
            
            # Check if this move is actually legal (due to mask/promotion simplification)
            potential_move = chess.Move(from_square, to_square)
            # Handle promotion possibility
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                 if board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                     potential_move.promotion = chess.QUEEN # Default to Queen promotion
                 elif board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                     potential_move.promotion = chess.QUEEN # Default to Queen promotion

            if potential_move in board.legal_moves:
                return potential_move
            else:
                # Fallback: choose a random legal move
                legal_moves = list(board.legal_moves)
                return random.choice(legal_moves) if legal_moves else None

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
    def __init__(self, model_dir="/content/models"):
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model_dir = model_dir
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
        # Initialize replay memory
        self.memory = ReplayMemory(10000) # Adjust capacity as needed
        
        # Training parameters
        self.batch_size = 128 # Increase batch size for GPU
        self.gamma = 0.99  # Discount factor
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 20000 # Adjust decay rate
        self.target_update = 20  # Update target network every N episodes/batches
        
        self.steps_done = 0
        
    def select_action(self, state, mask, board):
        """Select an action using epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        state = state.to(self.device)
        mask = mask.to(self.device)
        
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
                    if not legal_moves: return None, None # No legal moves
                    move = random.choice(legal_moves)
                    action_idx = move.from_square * 64 + move.to_square
        else:
            # Choose a random legal move
            legal_moves = list(board.legal_moves)
            if not legal_moves: return None, None # No legal moves
            move = random.choice(legal_moves)
            action_idx = move.from_square * 64 + move.to_square
            
        return action_idx, move
    
    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample a batch from memory
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors and move to device
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        mask_batch = torch.cat(batch.mask).to(self.device)
        
        # Handle non-final states
        non_final_mask_indices = torch.tensor(tuple(map(lambda d: not d, batch.done)), dtype=torch.bool, device=self.device)
        
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d]).to(self.device)
        non_final_next_masks = torch.cat([m for m, d in zip(batch.next_mask, batch.done) if not d]).to(self.device)
        
        # Compute Q(s_t, a)
        # The model computes Q(s_t), then we select the columns of actions taken.
        state_action_values = self.policy_net(state_batch, mask_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.size(0) > 0: # Check if there are any non-final states
             with torch.no_grad():
                 next_state_values[non_final_mask_indices] = self.target_net(non_final_next_states, non_final_next_masks).max(1)[0]
        
        # Compute the expected Q values: Q_expected = r + gamma * max_a' Q_target(s', a')
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss (Smooth L1 Loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0) # Use clip_grad_value_
        self.optimizer.step()
        
        return loss.item()
    
    def train_self_play(self, num_episodes=1000):
        """Train the model through self-play"""
        print(f"Starting self-play training for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        losses = []
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Initialize the environment
            board = chess.Board()
            episode_reward = 0
            episode_length = 0
            
            # Get initial state
            state = board_to_tensor(board)
            mask = create_move_mask(board)
            
            done = False
            while not done:
                # Select and perform an action
                action_idx, move = self.select_action(state, mask, board)
                
                if move is None: # No legal moves
                    break 
                    
                # Execute the move
                board.push(move)
                
                # Get the next state
                next_state = board_to_tensor(board)
                next_mask = create_move_mask(board)
                
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
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                avg_loss = np.mean(losses[-100:]) if losses else 0 # Avg loss over last 100 steps
                elapsed_time = time.time() - start_time
                print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Avg Loss: {avg_loss:.4f} | Steps: {self.steps_done} | Time: {elapsed_time:.1f}s")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(f"model_selfplay_ep{episode+1}.pt")
        
        # Save final model
        self.save_model("model_selfplay_final.pt")
        print("Self-play training completed!")
        
        return episode_rewards, episode_lengths, losses
    
    def calculate_reward(self, board):
        """Calculate reward based on the board state"""
        # Basic reward function
        if board.is_checkmate():
            # High reward/penalty for checkmate (from perspective of player whose turn it is)
            # If it's White's turn, Black just checkmated White -> reward is -1
            # If it's Black's turn, White just checkmated Black -> reward is -1
            return -1.0 
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            # Neutral reward for draw
            return 0.0
        
        # Intermediate reward (e.g., based on material difference)
        # Calculate material difference from the perspective of the player who just moved
        turn = not board.turn # Get the player who just moved
        material_diff = self.calculate_material_advantage(board, turn)
        
        # Small reward for material advantage, scaled
        return 0.01 * material_diff
        
    def calculate_material_advantage(self, board, turn):
        """Calculate material advantage for the specified player (turn=True for White, False for Black)"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                            for piece_type, value in piece_values.items())
        black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                            for piece_type, value in piece_values.items())
        
        # Return advantage from perspective of the specified player
        return white_material - black_material if turn == chess.WHITE else black_material - white_material
    
    def save_model(self, filename):
        """Save the model to disk"""
        filepath = os.path.join(self.model_dir, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """Load a model from disk"""
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            # Load state dict, ensuring map_location handles CPU/GPU differences
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            print(f"Model loaded from {filepath} to {self.device}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False
    
    def train_from_pgn(self, pgn_file, num_games=1000):
        """Train the model from PGN games (Supervised Learning + RL Reward)"""
        print(f"Starting training from PGN file: {pgn_file}")
        
        if not os.path.exists(pgn_file):
            print(f"Error: PGN file not found at {pgn_file}")
            return []
            
        # Open PGN file
        with open(pgn_file) as f:
            game_count = 0
            losses = []
            processed_moves = 0
            start_time = time.time()
            
            while game_count < num_games:
                # Read the next game
                try:
                    game = chess.pgn.read_game(f)
                except Exception as e:
                    print(f"Error reading game: {e}. Skipping.")
                    continue
                    
                if game is None:
                    print("Reached end of PGN file.")
                    break  # End of file
                
                # Process the game
                board = game.board()
                moves = list(game.mainline_moves())
                
                # Skip very short games
                if len(moves) < 5:
                    continue
                
                # Process each position in the game
                for i, move in enumerate(moves):
                    # Get current state
                    state = board_to_tensor(board)
                    mask = create_move_mask(board)
                    
                    # Convert actual move to action index
                    action_idx = move.from_square * 64 + move.to_square
                    
                    # Make the move on a copy to get next state
                    next_board = board.copy()
                    next_board.push(move)
                    
                    # Get next state
                    next_state = board_to_tensor(next_board)
                    next_mask = create_move_mask(next_board)
                    
                    # Calculate reward for the state *after* the move
                    reward = self.calculate_reward(next_board)
                    
                    # Check if game is over after the move
                    done = next_board.is_game_over()
                    
                    # Store transition in memory
                    self.memory.push(state, action_idx, next_state, reward, done, mask, next_mask)
                    
                    # Make the move on the main board for the next iteration
                    board.push(move)
                    processed_moves += 1
                    
                    # Perform optimization step if enough samples
                    if len(self.memory) >= self.batch_size:
                        loss = self.optimize_model()
                        if loss is not None:
                            losses.append(loss)
                            
                        # Update target network periodically based on steps/batches
                        if processed_moves % (self.target_update * self.batch_size) == 0: 
                            self.target_net.load_state_dict(self.policy_net.state_dict())
            
                game_count += 1
                
                # Print progress
                if game_count % 10 == 0:
                    avg_loss = np.mean(losses[-1000:]) if losses else 0 # Avg loss over last 1000 steps
                    elapsed_time = time.time() - start_time
                    print(f"Processed {game_count}/{num_games} games | Moves: {processed_moves} | Avg Loss: {avg_loss:.4f} | Time: {elapsed_time:.1f}s")
                
                # Save model periodically
                if game_count % 100 == 0:
                    self.save_model(f"model_pgn_game{game_count}.pt")
        
        # Save final model
        self.save_model("model_pgn_final.pt")
        print(f"PGN training completed! Processed {game_count} games and {processed_moves} moves.")
        
        return losses

# Create trainer
trainer = ChessTrainer()

# Option 1: Train from your own PGN data
# If you've uploaded your own PGN file, set the path here:
# pgn_file = "/content/your_pgn_file.pgn"

# Option 2: Use synthetic data
pgn_file = "/content/data/synthetic/synthetic_games.pgn"
if not os.path.exists(pgn_file):
    print("Generating synthetic data...")
    generate_synthetic_data(num_games=1000, output_file=pgn_file)

# Train from PGN data
pgn_losses = trainer.train_from_pgn(pgn_file, num_games=500)

# Continue with self-play training
self_play_rewards, self_play_lengths, self_play_losses = trainer.train_self_play(num_episodes=500)

# Plot training results
def plot_results(rewards, lengths, losses):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.ylabel('Reward')
    
    plt.subplot(3, 1, 2)
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.ylabel('Length')
    
    plt.subplot(3, 1, 3)
    plt.plot(losses)
    plt.title('Training Loss (Optimization Steps)')
    plt.xlabel('Optimization Step')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

# Plot self-play results
plot_results(self_play_rewards, self_play_lengths, self_play_losses)

# Test the trained agent
def test_agent(model_path="model_selfplay_final.pt"):
    agent = ChessAgent()
    loaded = trainer.load_model(model_path)
    
    if loaded:
        agent.policy_net.load_state_dict(trainer.policy_net.state_dict())
        agent.policy_net.eval()
        
        # Test the agent
        board = chess.Board()
        print("\nTesting trained agent:")
        print(board.unicode())
        
        for i in range(10): # Play 10 moves
            if board.is_game_over():
                print("Game Over!")
                break
                
            print(f"\nTurn: {'White' if board.turn == chess.WHITE else 'Black'}")
            
            # Get move from agent
            move = agent.select_move(board)
            
            print(f"Agent selects move: {move.uci()}")
            board.push(move)
            print(board.unicode())
    else:
        print("Could not load trained model for testing.")

# Test the trained agent
test_agent()

# Save the model for download
print("\nTraining complete! Download the model files from the /content/models directory.")
