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
        # Apply mask: set invalid move scores to a very small number (or -inf)
        # Using -inf ensures that softmax output for invalid moves is zero
        masked_output = x.masked_fill(mask_reshaped == 0, -float("inf"))
        return masked_output

# Simplified Attention Module for faster processing
class SimplifiedAttention(nn.Module):
    """
    Simplified attention module for faster processing.

    This is a lightweight alternative to the full self-attention mechanism,
    using spatial attention rather than the more computationally expensive
    self-attention from transformer architectures.
    """
    def __init__(self, in_channels):
        super(SimplifiedAttention, self).__init__()

        # Spatial attention with fewer operations
        self.conv_spatial = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)

        # Channel attention
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

        # Spatial attention - simpler than full self-attention
        spatial = self.conv_spatial(x)
        spatial = self.norm(spatial)

        # Channel attention
        b, c, _, _ = x.size()
        channel = self.avg_pool(x).view(b, c)
        channel = self.channel_attn(channel).view(b, c, 1, 1)

        # Apply attention
        out = spatial * channel

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
    Simplified chess-specific feature extractor to encode domain knowledge.

    This module extracts basic chess-specific features with reduced complexity:
    - Center control
    - King safety
    - Basic piece mobility
    """
    def __init__(self, in_channels):
        super(ChessFeatureExtractor, self).__init__()

        # Convolutional layers for feature extraction (reduced size)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Simplified specialized feature detectors
        # Combined mobility and center control detector
        self.mobility_center_conv = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        # King safety detector
        self.king_safety_conv = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        # Final integration layer (reduced size)
        self.integration_conv = nn.Conv2d(16, in_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Initial feature extraction
        features = F.relu(self.bn1(self.conv1(x)))

        # Extract simplified specialized features
        mobility_center = F.relu(self.mobility_center_conv(features))
        king_safety = F.relu(self.king_safety_conv(features))

        # Concatenate simplified features
        combined_features = torch.cat([mobility_center, king_safety], dim=1)

        # Integrate features
        enhanced_features = F.relu(self.bn_out(self.integration_conv(combined_features)))

        # Residual connection
        return enhanced_features + x

class DQN(nn.Module):
    """
    AlphaZero-style network for chess position evaluation with dual policy and value heads.

    Architecture:
    - Input: 16 channel 8x8 board representation
      * Channels 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
      * Channels 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
      * Channels 12-13: Castling rights
      * Channel 14: En passant
      * Channel 15: Player to move

    - Chess-specific feature extraction for domain knowledge encoding
    - Convolutional layers with residual connections extract spatial features
    - Simplified attention mechanism for faster processing
    - Reduced number of residual blocks (2 instead of 3)
    - Reduced channel count (32 instead of 64) for faster computation
    - Dual output heads:
      * Policy head: 4096 move probabilities (64x64 possible from-to square combinations)
      * Value head: Single scalar value estimating position evaluation (-1 to 1)

    The network uses batch normalization, residual connections, simplified attention,
    and ReLU activations for better training stability and performance.
    """
    def __init__(self, num_residual_blocks=2, channels=32, attention_layers=1):
        super(DQN, self).__init__()

        # Network configuration
        self.num_residual_blocks = num_residual_blocks
        self.channels = channels
        self.attention_layers = attention_layers

        # Initial convolutional layer
        self.conv_in = nn.Conv2d(16, channels, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        # Chess-specific feature extractor
        self.chess_feature_extractor = ChessFeatureExtractor(16)

        # Create reduced number of residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        ])

        # Create simplified attention layers
        self.attention_modules = nn.ModuleList([
            SimplifiedAttention(channels) for _ in range(attention_layers)
        ])

        # Additional convolutional layer to increase feature maps
        self.conv_out = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
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

        # Apply residual blocks with interleaved attention
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)

            # Apply attention after some residual blocks
            if i < len(self.attention_modules):
                x = self.attention_modules[i](x)

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

def simple_lookahead(board, policy_net, device, depth=1):
    """
    Perform a simple 1-2 ply lookahead for critical positions.

    Args:
        board (chess.Board): Current board position
        policy_net (nn.Module): Policy network
        device (torch.device): Device to run computations on
        depth (int): Lookahead depth (1 or 2)

    Returns:
        int: Best move index after lookahead
    """
    # Check if this is a critical position (in check, or can capture a piece)
    is_critical = board.is_check() or any(board.is_capture(move) for move in board.legal_moves)

    if not is_critical or depth <= 0:
        # For non-critical positions, just use the policy network directly
        state = board_to_tensor(board, device)
        mask = create_move_mask(board, device)

        with torch.no_grad():
            policy_logits, value = policy_net(state, mask)
            return torch.argmax(policy_logits[0]).item()

    # For critical positions, do a simple lookahead
    best_score = float('-inf')
    best_move_idx = None

    # Get legal moves
    legal_moves = list(board.legal_moves)
    legal_move_indices = [m.from_square * 64 + m.to_square for m in legal_moves]

    # Evaluate each move with lookahead
    for i, move in enumerate(legal_moves):
        # Make the move
        board_copy = board.copy()
        board_copy.push(move)

        if board_copy.is_checkmate():
            # Immediate checkmate is best
            return legal_move_indices[i]

        if depth > 1 and not board_copy.is_game_over():
            # Opponent's best response
            opponent_state = board_to_tensor(board_copy, device)
            opponent_mask = create_move_mask(board_copy, device)

            with torch.no_grad():
                opponent_policy_logits, opponent_value = policy_net(opponent_state, opponent_mask)

                # Find opponent's best move
                opponent_legal_moves = list(board_copy.legal_moves)
                if opponent_legal_moves:
                    opponent_legal_indices = [m.from_square * 64 + m.to_square for m in opponent_legal_moves]
                    opponent_best_idx = torch.argmax(opponent_policy_logits[0, opponent_legal_indices]).item()
                    opponent_best_move = opponent_legal_moves[opponent_best_idx]

                    # Make opponent's move
                    board_copy.push(opponent_best_move)

                    if board_copy.is_checkmate():
                        # If opponent can checkmate, this is a bad move
                        score = float('-inf')
                    else:
                        # Evaluate resulting position
                        result_state = board_to_tensor(board_copy, device)
                        result_mask = create_move_mask(board_copy, device)

                        with torch.no_grad():
                            _, result_value = policy_net(result_state, result_mask)
                            score = result_value.item()
                else:
                    # No legal moves for opponent (stalemate)
                    score = 0.0
        else:
            # Evaluate position after our move
            state = board_to_tensor(board_copy, device)
            mask = create_move_mask(board_copy, device)

            with torch.no_grad():
                _, value = policy_net(state, mask)
                score = value.item()

        # Update best move
        if score > best_score:
            best_score = score
            best_move_idx = legal_move_indices[i]

    return best_move_idx

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

    Enhanced features:
    - Phase-aware exploration (opening, middlegame, endgame)
    - Temperature-based move selection for controlled exploration
    - Improved repetition handling with adaptive penalties
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

            # Use FP16 precision for inference if enabled
            if self.use_fp16:
                # Import autocast for mixed precision inference
                from torch.amp import autocast

                # Convert tensors to FP16
                state_tensor_fp16 = state_tensor.to(dtype=torch.float16)
                move_mask_fp16 = move_mask.to(dtype=torch.float16)

                # Use autocast for mixed precision inference
                with autocast(device_type='cuda'):
                    q_values = self.policy_net(state_tensor_fp16, move_mask_fp16)
            else:
                # Standard full precision inference
                q_values = self.policy_net(state_tensor, move_mask)

            # Print some stats about the Q-values to help debug
            if torch.isnan(q_values).any():
                print("WARNING: NaN values detected in Q-values!")

            # Get the legal move indices
            legal_move_indices = [m.from_square * 64 + m.to_square for m in board.legal_moves]

            # Apply repetition avoidance by penalizing moves that lead to repeated positions
            adjusted_q_values = self._apply_repetition_penalties(board, q_values.clone())

            # Get the adjusted Q-values for legal moves only
            legal_adjusted_q_values = adjusted_q_values[0, legal_move_indices]

            # Determine game phase for phase-aware exploration
            game_phase = self._determine_game_phase(board)

            # Apply temperature-based selection with phase-aware adjustments
            temperature = self._get_phase_adjusted_temperature(game_phase)

            # Apply temperature to Q-values (higher temperature = more exploration)
            if temperature != 1.0:
                # Apply softmax with temperature
                legal_adjusted_q_values = legal_adjusted_q_values / temperature

            # Decide whether to use softmax sampling or greedy selection
            use_sampling = random.random() < self._get_phase_exploration_rate(game_phase)

            if use_sampling and len(legal_move_indices) > 1:
                # Apply softmax to get probabilities
                probabilities = F.softmax(legal_adjusted_q_values, dim=0)

                # Sample from the probability distribution
                selected_idx = torch.multinomial(probabilities, 1).item()
                best_move_index = legal_move_indices[selected_idx]
            else:
                # Greedy selection - choose the move with highest Q-value
                best_legal_index_in_list = torch.argmax(legal_adjusted_q_values).item()
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

    def _determine_game_phase(self, board):
        """
        Determine the current phase of the game: opening, middlegame, or endgame.

        Args:
            board (chess.Board): The current chess position

        Returns:
            str: Game phase - 'opening', 'middlegame', or 'endgame'
        """
        # Count the total number of pieces
        piece_count = len(board.piece_map())

        # Count the number of moves played
        move_count = board.fullmove_number

        # Check if queens are still on the board
        has_queens = (board.pieces(chess.QUEEN, chess.WHITE) or
                     board.pieces(chess.QUEEN, chess.BLACK))

        # Opening: First 10 moves with most pieces still on board
        if move_count <= 10 and piece_count >= 28:
            return 'opening'
        # Endgame: Few pieces left or no queens
        elif piece_count <= 12 or not has_queens:
            return 'endgame'
        # Middlegame: Everything else
        else:
            return 'middlegame'

    def _get_phase_adjusted_temperature(self, game_phase):
        """
        Get temperature parameter adjusted for the current game phase.

        Args:
            game_phase (str): Current game phase

        Returns:
            float: Adjusted temperature value
        """
        base_temperature = self.temperature

        # Adjust temperature based on game phase
        if game_phase == 'opening':
            # More exploration in opening
            return base_temperature * 1.2
        elif game_phase == 'endgame':
            # Less exploration in endgame
            return base_temperature * 0.8
        else:
            # Standard temperature for middlegame
            return base_temperature

    def _get_phase_exploration_rate(self, game_phase):
        """
        Get exploration rate based on the current game phase.

        Args:
            game_phase (str): Current game phase

        Returns:
            float: Exploration rate (0-1)
        """
        # Base exploration rates for different phases
        if game_phase == 'opening':
            # Higher exploration in opening to try different openings
            return 0.35  # Increased for more opening variety
        elif game_phase == 'middlegame':
            # Moderate exploration in middlegame
            return 0.20  # Slightly increased for better tactical exploration
        else:  # endgame
            # Lower exploration in endgame for more precise play
            return 0.08  # Slightly increased but still low for endgame precision

    def _apply_repetition_penalties(self, board, q_values):
        """
        Apply penalties to Q-values for moves that lead to repeated positions.

        Args:
            board (chess.Board): The current chess position
            q_values (torch.Tensor): Q-values from the policy network

        Returns:
            torch.Tensor: Adjusted Q-values with repetition penalties applied
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
                    q_values[0, move_idx] *= self.repetition_penalty
                elif repetition_count == 2:
                    # Second repetition (would be a threefold repetition): severe penalty
                    q_values[0, move_idx] *= (self.repetition_penalty ** 2)
                else:
                    # More than two repetitions: extreme penalty
                    q_values[0, move_idx] *= (self.repetition_penalty ** 3)

        return q_values

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

