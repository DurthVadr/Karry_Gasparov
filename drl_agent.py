# Deep Reinforcement Learning Chess Agent

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

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
        # Based on chess-engine-2-reinforcement-learning.ipynb
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Raw scores for each move
        
        if mask is not None:
            x = self.mask_layer(x, mask)
            
        return x

# Helper function to convert chess board to tensor representation
# Adapted from bootstrapped_deep_reinforcement_learning.ipynb and chess-engine-2-reinforcement-learning.ipynb
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
            
    # Occupied squares (might not be needed if using empty square layer)
    # tensor[12] = np.reshape(np.array([1 if board.piece_at(sq) else 0 for sq in chess.SQUARES]), (8, 8))
    
    # Empty squares (alternative to occupied)
    # for rank in range(8):
    #     for file in range(8):
    #         if board.piece_at(chess.square(file, rank)) is None:
    #             tensor[12, rank, file] = 1

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
# Adapted from chess-engine-2-reinforcement-learning.ipynb
def create_move_mask(board):
    """Creates a 4096-element mask tensor for legal moves."""
    mask = torch.zeros(4096, dtype=torch.float32)
    for move in board.legal_moves:
        index = move.from_square * 64 + move.to_square
        # Handle promotion - for simplicity, allow any promotion for now
        # A more sophisticated approach would create separate indices for promotions
        if move.promotion:
             # Simple approach: mark the basic move index
             mask[index] = 1
             # Or, reserve specific indices for promotions (e.g., 4096-4351)
        else:
            mask[index] = 1
    return mask.unsqueeze(0) # Add batch dimension

# Basic Agent Implementation (will be expanded for training)
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
            # Need to handle cases where multiple moves have the same max value
            # Also need to map the index back to a chess.Move object
            
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
                # Fallback: if the predicted best move isn't legal (e.g., mask issue),
                # choose a random legal move.
                print(f"Warning: Predicted best move index {best_move_index} ({potential_move.uci()}) is not legal. Choosing random move.")
                # Find the highest Q-value among *legal* moves instead
                legal_move_indices = [m.from_square * 64 + m.to_square for m in board.legal_moves]
                legal_q_values = q_values[0, legal_move_indices]
                best_legal_index_in_list = torch.argmax(legal_q_values).item()
                best_move_index = legal_move_indices[best_legal_index_in_list]
                
                from_square = best_move_index // 64
                to_square = best_move_index % 64
                potential_move = chess.Move(from_square, to_square)
                # Handle promotion again
                piece = board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                     if board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                         potential_move.promotion = chess.QUEEN
                     elif board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                         potential_move.promotion = chess.QUEEN
                
                if potential_move in board.legal_moves:
                     return potential_move
                else:
                     print("Fallback failed, choosing random legal move.")
                     return random.choice(list(board.legal_moves))

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

