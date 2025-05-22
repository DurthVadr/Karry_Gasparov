# Deep Reinforcement Learning Chess Agent
"""
This module implements a Deep Q-Network (DQN) for chess. It includes:
1. Neural network architecture for evaluating chess positions
2. Board representation functions for converting chess positions to tensors
3. Move selection logic for choosing the best move
4. Basic agent implementation for interacting with the chess environment

The DQN architecture is designed to take a chess board as input and output
Q-values for all possible moves. The agent then selects the move with the
highest Q-value.
"""

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.stats import dirichlet

# Mask Layer for handling valid moves
class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def forward(self, x, mask):
        # Ensure mask is boolean or float tensor of 0s and 1s
        # Reshape mask to match the output shape if necessary
        mask_reshaped = mask.view_as(x)
        # Apply mask: set invalid move scores to a very small number
        # Using a large negative value ensures that softmax output for invalid moves is close to zero
        # Use -1e4 instead of -inf for FP16 compatibility (FP16 range is approximately -65504 to 65504)
        masked_output = x.masked_fill(mask_reshaped == 0, -1e4)
        return masked_output

# Simplified Attention Module for faster processing
class SimplifiedAttention(nn.Module):
    """
    Simplified attention module for faster processing.

    This is a minimal channel attention mechanism that focuses on
    important piece types and their relationships.
    """
    def __init__(self, in_channels):
        super(SimplifiedAttention, self).__init__()

        # Channel attention only - more efficient
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Store residual
        residual = x

        # Channel attention
        b, c, _, _ = x.size()
        channel = self.avg_pool(x).view(b, c)
        channel = self.channel_attn(channel).view(b, c, 1, 1)

        # Apply attention
        out = x * channel

        # Add residual connection
        return out + residual

# Deep Q-Network (DQN) Architecture
class ResidualBlock(nn.Module):
    """
    Residual block for improved gradient flow.

    This block includes two convolutional layers with batch normalization
    and a residual connection to improve training stability and performance.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Add the residual connection
        x = F.relu(x)  # Apply ReLU after the addition
        return x

class ChessFeatureExtractor(nn.Module):
    """
    Minimal chess-specific feature extractor.

    This module provides a simple feature extraction layer with a residual connection.
    """
    def __init__(self, in_channels):
        super(ChessFeatureExtractor, self).__init__()

        # Single convolutional layer for feature extraction
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Store residual
        residual = x

        # Extract features
        features = F.relu(self.bn(self.conv(x)))

        # Residual connection
        return features + residual

class DQN(nn.Module):
    """
    Simplified neural network for chess position evaluation with dual policy and value heads.

    Architecture:
    - Input: 16 channel 8x8 board representation
      * Channels 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
      * Channels 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
      * Channels 12-13: Castling rights
      * Channel 14: En passant
      * Channel 15: Player to move

    - Simple feature extraction with a single convolutional layer
    - 2 residual blocks for spatial feature extraction
    - 1 simplified attention layer for focusing on important board areas
    - 32 channels throughout the convolutional layers
    - Dual output heads:
      * Policy head: 4096 move probabilities (64x64 possible from-to square combinations)
      * Value head: Single scalar value estimating position evaluation (-1 to 1)

    The network uses batch normalization, residual connections, and ReLU activations
    for training stability and performance.
    """
    def __init__(self):
        super(DQN, self).__init__()

        # Network configuration - fixed to recommended values
        self.num_residual_blocks = 2  # Fixed to 2 as per hyperparameters
        self.channels = 32  # Fixed to 32 as per hyperparameters

        # Initial convolutional layer
        self.conv_in = nn.Conv2d(16, self.channels, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(self.channels)

        # Chess-specific feature extractor
        self.chess_feature_extractor = ChessFeatureExtractor(16)

        # Create residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.channels) for _ in range(self.num_residual_blocks)
        ])

        # Create single attention layer
        self.attention_modules = nn.ModuleList([
            SimplifiedAttention(self.channels)
        ])

        # Additional convolutional layer to increase feature maps
        self.conv_out = nn.Conv2d(self.channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn_out = nn.BatchNorm2d(64)

        # Shared representation layer
        self.shared_fc = nn.Linear(64 * 64, 1024)
        self.shared_dropout = nn.Dropout(0.2)

        # Policy head - outputs move probabilities
        self.policy_fc = nn.Linear(1024, 4096)  # Output: 64*64 = 4096 possible moves

        # Value head - outputs position evaluation
        self.value_fc1 = nn.Linear(1024, 256)
        self.value_dropout = nn.Dropout(0.2)
        self.value_fc2 = nn.Linear(256, 1)  # Single scalar output

        # Mask layer to ensure only legal moves are considered
        self.mask_layer = MaskLayer()

    def forward(self, x, mask=None):
        # Apply chess-specific feature extraction
        x = self.chess_feature_extractor(x)

        # Initial convolution
        x = F.relu(self.bn_in(self.conv_in(x)))

        # Apply residual blocks with attention after the first block
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)

            # Apply attention only after the first residual block
            if i == 0:
                x = self.attention_modules[0](x)

        # Final convolution
        x = F.relu(self.bn_out(self.conv_out(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        # Shared representation
        shared_features = F.relu(self.shared_fc(x))
        shared_features = self.shared_dropout(shared_features)

        # Policy head - outputs move probabilities
        policy_logits = self.policy_fc(shared_features)  # Raw logits for each move

        # Apply mask if provided
        if mask is not None:
            policy_logits = self.mask_layer(policy_logits, mask)

        # Value head - outputs position evaluation
        value_hidden = F.relu(self.value_fc1(shared_features))
        value_hidden = self.value_dropout(value_hidden)
        value = torch.tanh(self.value_fc2(value_hidden))  # Tanh to bound between -1 and 1

        return policy_logits, value

# AlphaZero-style helper functions
def add_dirichlet_noise(policy_probs, legal_moves, alpha=0.3, epsilon=0.25):
    """
    Add Dirichlet noise to the policy probabilities at the root node.

    Args:
        policy_probs (torch.Tensor): Policy probabilities from the network
        legal_moves (list): List of legal move indices
        alpha (float): Dirichlet noise parameter (default: 0.3)
        epsilon (float): Weight of noise to add (default: 0.25)

    Returns:
        torch.Tensor: Policy probabilities with added Dirichlet noise
    """
    # Create a copy of the policy probabilities
    noisy_probs = policy_probs.clone()

    # Generate Dirichlet noise for legal moves
    noise = dirichlet.rvs([alpha] * len(legal_moves), size=1)[0]
    noise_tensor = torch.tensor(noise, dtype=torch.float32, device=policy_probs.device)

    # Apply noise only to legal moves
    for i, move_idx in enumerate(legal_moves):
        # Mix original probability with noise
        noisy_probs[0, move_idx] = (1 - epsilon) * policy_probs[0, move_idx] + epsilon * noise_tensor[i]

    # Renormalize probabilities for legal moves
    legal_probs = noisy_probs[0, legal_moves]
    legal_probs = legal_probs / legal_probs.sum()

    # Update the noisy_probs tensor with normalized values
    for i, move_idx in enumerate(legal_moves):
        noisy_probs[0, move_idx] = legal_probs[i]

    return noisy_probs

def temperature_sampling(policy_probs, legal_moves, temperature=1.0):
    """
    Apply temperature to policy probabilities and sample a move.

    Args:
        policy_probs (torch.Tensor): Policy probabilities from the network
        legal_moves (list): List of legal move indices
        temperature (float): Temperature parameter (default: 1.0)
            - temperature → 0: more deterministic (choose best move)
            - temperature → ∞: more random (uniform distribution)

    Returns:
        int: Selected move index
    """
    # Extract probabilities for legal moves
    legal_probs = policy_probs[0, legal_moves]

    if temperature == 0:
        # Deterministic selection (argmax)
        selected_idx = torch.argmax(legal_probs).item()
    else:
        # Apply temperature scaling
        scaled_probs = (legal_probs ** (1.0 / temperature))
        # Renormalize
        scaled_probs = scaled_probs / scaled_probs.sum()

        # Sample from the distribution
        selected_idx = torch.multinomial(scaled_probs, 1).item()

    # Return the selected move index
    return legal_moves[selected_idx]

# Lookahead function removed to simplify the codebase

# Helper function to convert chess board to tensor representation
# Optimized version for better performance
def board_to_tensor(board, device=None):
    """
    Converts a chess.Board object to a 16x8x8 tensor.

    Args:
        board: A chess.Board object
        device: Optional torch device to place the tensor on

    Returns:
        A tensor representation of the board with batch dimension
    """
    # Pre-allocate the tensor with zeros
    tensor = np.zeros((16, 8, 8), dtype=np.float32)

    # Piece placement (channels 0-11)
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    # More efficient piece placement using piece map
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        piece_type = piece.piece_type
        color = piece.color

        # Find the index in our pieces list
        piece_idx = pieces.index(piece_type)

        # Set the appropriate channel based on piece type and color
        if color == chess.WHITE:
            tensor[piece_idx, rank, file] = 1
        else:  # BLACK
            tensor[piece_idx + 6, rank, file] = 1

    # Castling rights (channels 12-13)
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12, 0, 7] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[12, 0, 0] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[13, 7, 7] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 7, 0] = 1

    # En passant square (channel 14)
    if board.ep_square:
        rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        tensor[14, rank, file] = 1

    # Player to move (channel 15)
    if board.turn == chess.WHITE:
        tensor[15, :, :] = 1

    # Convert to torch tensor and add batch dimension
    tensor = torch.from_numpy(tensor).unsqueeze(0)

    # Move to specified device if provided
    if device is not None:
        tensor = tensor.to(device)

    return tensor

# Helper function to create the move mask
# Optimized version for better performance
def create_move_mask(board, device=None):
    """
    Creates a 4096-element mask tensor for legal moves.

    Args:
        board: A chess.Board object
        device: Optional torch device to place the tensor on

    Returns:
        A binary mask tensor with 1s for legal moves and 0s for illegal moves
    """
    # Pre-allocate the mask with zeros
    mask = torch.zeros(4096, dtype=torch.float32)

    # Set indices for all legal moves
    for move in board.legal_moves:
        index = move.from_square * 64 + move.to_square
        mask[index] = 1

    # Add batch dimension
    mask = mask.unsqueeze(0)

    # Move to specified device if provided
    if device is not None:
        mask = mask.to(device)

    return mask

# Chess Agent Implementation
class ChessAgent:
    """
    Chess agent that uses a trained DQN to select moves.

    This agent loads a trained model and uses it to select the best move
    for a given chess position. It handles the conversion between chess.Board
    objects and the tensor representation needed by the neural network.

    The agent also handles special cases like promotions and ensures that
    only legal moves are selected. It includes repetition detection and avoidance
    to prevent the agent from making repetitive moves that could lead to draws.

    Features:
    - Simple temperature-based move selection for exploration
    - Basic repetition handling with penalties
    """
    def __init__(self, model_path=None, repetition_penalty=0.95, use_fp16=True, temperature=1.0):
        """
        Initialize the chess agent with an optional pre-trained model.

        Args:
            model_path (str, optional): Path to a saved model file (.pt or .pth)
            repetition_penalty (float, optional): Penalty factor for repeated positions (0-1)
            use_fp16 (bool, optional): Whether to use FP16 precision for inference
            temperature (float, optional): Temperature parameter for controlling exploration
                                          Higher values = more exploration, lower = more exploitation
        """
        # Initialize the policy network
        self.policy_net = DQN()

        # Set up mixed precision inference
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        if self.use_fp16:
            print("Using FP16 precision for model inference")

        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self.policy_net.to(self.device)

        # Flag to track if model was loaded successfully
        self.model_loaded = False

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            try:
                print(f"ChessAgent: Loading model from {model_path}")
                # Try to load with map_location to handle models trained on different devices
                self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"ChessAgent: Successfully loaded model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                print(f"ChessAgent: Error loading model: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            if model_path:
                print(f"ChessAgent: Model file {model_path} does not exist")
            else:
                print("ChessAgent: No model path provided")

        # Set to evaluation mode (no gradient tracking needed for inference)
        self.policy_net.eval()

        # Position history tracking for repetition detection
        self.position_history = {}

        # Penalty factor for repeated positions (0-1)
        # Lower values = stronger penalty
        self.repetition_penalty = repetition_penalty

        # Temperature parameter for controlling exploration
        self.temperature = temperature

    def select_move(self, board):
        """
        Select the best move for the given board position using the policy network.

        Args:
            board (chess.Board): The current chess position

        Returns:
            chess.Move: The selected move

        The method:
        1. Converts the board to a tensor representation
        2. Creates a mask of legal moves
        3. Gets Q-values from the policy network
        4. Applies penalties for moves that lead to repeated positions
        5. Applies phase-aware exploration and temperature-based selection
        6. Selects the move with the highest adjusted Q-value
        7. Handles special cases like promotions
        8. Ensures the selected move is legal
        """
        # Print a message to indicate if we're using a trained model or not
        if not self.model_loaded:
            print("WARNING: Using untrained model for move selection!")

        with torch.no_grad():  # No need to track gradients for inference
            # Convert board to tensor representation
            state_tensor = board_to_tensor(board).to(self.device)
            move_mask = create_move_mask(board).to(self.device)

            # Get policy logits from the network
            policy_logits, _ = self.policy_net(state_tensor, move_mask)

            # Print some stats about the policy_logits to help debug
            if torch.isnan(policy_logits).any():
                print("WARNING: NaN values detected in policy_logits!")

            # Get the legal move indices
            legal_move_indices = [m.from_square * 64 + m.to_square for m in board.legal_moves]

            # Apply repetition avoidance by penalizing moves that lead to repeated positions
            adjusted_policy_logits = self._apply_repetition_penalties(board, policy_logits.clone())

            # Get the adjusted policy logits for legal moves only
            legal_adjusted_policy_logits = adjusted_policy_logits[0, legal_move_indices]

            # Apply simple temperature scaling
            if self.temperature != 1.0:
                legal_adjusted_policy_logits = legal_adjusted_policy_logits / self.temperature

            # Simple exploration strategy - 20% random sampling, 80% greedy
            use_sampling = random.random() < 0.2

            if use_sampling and len(legal_move_indices) > 1:
                # Apply softmax to get probabilities
                probabilities = F.softmax(legal_adjusted_policy_logits, dim=0)

                # Sample from the probability distribution
                selected_idx = torch.multinomial(probabilities, 1).item()
                best_move_index = legal_move_indices[selected_idx]
            else:
                # Greedy selection - choose the move with highest policy logit value
                best_legal_index_in_list = torch.argmax(legal_adjusted_policy_logits).item()
                best_move_index = legal_move_indices[best_legal_index_in_list]

            # Convert index to chess move coordinates
            from_square = best_move_index // 64
            to_square = best_move_index % 64

            # Create the move object
            potential_move = chess.Move(from_square, to_square)

            # Handle pawn promotion (default to queen promotion)
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                # Check if pawn is moving to the last rank
                if board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                    potential_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                elif board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                    potential_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

            # Double-check that the move is legal
            if potential_move in board.legal_moves:
                # Update position history with the selected move
                self._update_position_history(board, potential_move)
                return potential_move
            else:
                # This should rarely happen since we're selecting from legal moves
                print(f"Warning: Selected move {potential_move.uci()} is not legal. Choosing random move.")
                # Choose a random move and update position history
                random_move = random.choice(list(board.legal_moves))
                self._update_position_history(board, random_move)
                return random_move

    # Game phase methods removed to simplify the codebase

    def _apply_repetition_penalties(self, board, policy_logits):
        """
        Apply penalties to policy logits for moves that lead to repeated positions.

        Args:
            board (chess.Board): The current chess position
            policy_logits (torch.Tensor): Policy logits from the policy network

        Returns:
            torch.Tensor: Adjusted policy logits with repetition penalties applied
        """
        # Get all legal moves
        legal_moves = list(board.legal_moves)

        # For each legal move, check if it leads to a repeated position
        for move in legal_moves:
            # Create a copy of the board to simulate the move
            board_copy = board.copy()
            board_copy.push(move)

            # Get the position FEN (just the piece positions, not the full FEN)
            position_fen = board_copy.fen().split(' ')[0]

            # Calculate move index
            move_idx = move.from_square * 64 + move.to_square

            # Apply penalty based on how many times this position has been seen
            if position_fen in self.position_history:
                # Get the number of times this position has been seen
                repetition_count = self.position_history[position_fen]

                # Apply increasingly severe penalties for repeated positions
                if repetition_count == 1:
                    # First repetition: mild penalty
                    policy_logits[0, move_idx] *= self.repetition_penalty
                elif repetition_count == 2:
                    # Second repetition (would be a threefold repetition): severe penalty
                    policy_logits[0, move_idx] *= (self.repetition_penalty ** 2)
                else:
                    # More than two repetitions: extreme penalty
                    policy_logits[0, move_idx] *= (self.repetition_penalty ** 3)

        return policy_logits

    def _update_position_history(self, board, move):
        """
        Update the position history after making a move.

        Args:
            board (chess.Board): The current chess position
            move (chess.Move): The move to be made
        """
        # Create a copy of the board to simulate the move
        board_copy = board.copy()
        board_copy.push(move)

        # Get the position FEN (just the piece positions, not the full FEN)
        position_fen = board_copy.fen().split(' ')[0]

        # Update the position history
        if position_fen in self.position_history:
            self.position_history[position_fen] += 1
        else:
            self.position_history[position_fen] = 1

# --- Training components (to be added later) ---
# Replay Memory
# Optimization loop
# Target Network
# Epsilon-greedy strategy
# Reward function
# PGN data loading and preprocessing

if __name__ == "__main__":
    # Example usage (without training)
    agent = ChessAgent()
    board = chess.Board()

    print("Initial board:")
    print(board.unicode())

    # Get a move from the untrained agent
    move = agent.select_move(board)
    print(f"\nSelected move: {move.uci()}")

    board.push(move)
    print("\nBoard after move:")
    print(board.unicode())

