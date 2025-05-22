"""
Utility functions for chess reinforcement learning.
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for plotting
import matplotlib.pyplot as plt
import numpy as np
import chess

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, filepath):
    """
    Save a PyTorch model to disk.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    ensure_dir(directory)
    
    # Save the model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath, device=None):
    """
    Load a PyTorch model from disk.
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to the saved model
        device: Device to load the model to (optional)
        
    Returns:
        True if model was loaded successfully, False otherwise
    """
    if os.path.exists(filepath):
        if device:
            model.load_state_dict(torch.load(filepath, map_location=device))
        else:
            model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")
        return True
    else:
        print(f"Model file {filepath} not found")
        return False

def plot_training_progress(stats, save_path):
    """
    Plot training progress metrics with enhanced visualizations.
    
    Args:
        stats: Dictionary containing training statistics
        save_path: Path to save the plot
    """
    try:
        # Determine number of subplots needed
        num_plots = 4
        if 'win_rates' in stats and stats['win_rates']:
            num_plots = 5
        
        # Create figure with subplots
        _, axs = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))
        
        # Plot episode rewards
        axs[0].plot(stats['episode_rewards'], color='blue')
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        
        # Add moving average for rewards
        if len(stats['episode_rewards']) > 10:
            window_size = min(50, len(stats['episode_rewards']) // 5)
            rewards_avg = [sum(stats['episode_rewards'][max(0, i-window_size):i+1]) / 
                          min(window_size, i+1) for i in range(len(stats['episode_rewards']))]
            axs[0].plot(rewards_avg, color='red', linestyle='--', label=f'{window_size}-episode moving avg')
            axs[0].legend()
        
        # Plot episode lengths
        axs[1].plot(stats['episode_lengths'], color='green')
        axs[1].set_title('Episode Lengths')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Length')
        
        # Plot losses
        if stats['losses']:
            # Use log scale for losses to better visualize changes
            axs[2].plot(stats['losses'], color='orange', alpha=0.5)
            axs[2].set_title('Training Loss')
            axs[2].set_xlabel('Optimization Step')
            axs[2].set_ylabel('Loss')
            
            # Add smoothed loss curve
            if len(stats['losses']) > 100:
                window_size = min(500, len(stats['losses']) // 10)
                losses_avg = []
                for i in range(len(stats['losses'])):
                    start_idx = max(0, i - window_size)
                    losses_avg.append(sum(stats['losses'][start_idx:i+1]) / (i - start_idx + 1))
                axs[2].plot(losses_avg, color='red', linewidth=2, label=f'{window_size}-step moving avg')
                axs[2].legend()
        
        # Plot average Q-values
        if 'avg_q_values' in stats and stats['avg_q_values']:
            axs[3].plot(stats['avg_q_values'], color='purple')
            axs[3].set_title('Average Q-Values')
            axs[3].set_xlabel('Step')
            axs[3].set_ylabel('Q-Value')
        
        # Plot win rates if available
        if 'win_rates' in stats and stats['win_rates']:
            axs[4].plot(stats['win_rates'], color='brown', marker='o', linestyle='-')
            axs[4].set_title('Win Rate Against Stockfish')
            axs[4].set_xlabel('Evaluation')
            axs[4].set_ylabel('Win Rate')
            axs[4].set_ylim([0, 1])
            axs[4].axhline(y=0.6, color='green', linestyle='--', label='Level-up Threshold')
            axs[4].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training progress plot saved to {save_path}")
    except Exception as e:
        print(f"Error plotting training progress: {e}")
        import traceback
        traceback.print_exc()

def select_action_epsilon_greedy(policy_net, state, mask, board, epsilon, device):
    """
    Select an action using epsilon-greedy policy.
    
    Args:
        policy_net: Policy network
        state: Current state tensor
        mask: Legal moves mask tensor
        board: Current chess.Board
        epsilon: Exploration rate
        device: Device to run computations on
        
    Returns:
        Tuple of (action_idx, move)
    """
    # Move tensors to the correct device
    state = state.to(device)
    mask = mask.to(device)
    
    if np.random.random() > epsilon:
        with torch.no_grad():
            # Use policy network to select best action
            q_values = policy_net(state, mask)
            
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
        move = np.random.choice(legal_moves)
        action_idx = move.from_square * 64 + move.to_square
    
    return action_idx, move
