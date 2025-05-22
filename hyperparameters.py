"""
Hyperparameter configuration for chess reinforcement learning.

This module provides optimized hyperparameters for training chess models.
These parameters have been tuned for better performance on the RTX 4070 GPU.
"""

# Neural Network Architecture
NETWORK_CONFIG = {
    'residual_blocks': 3,           # Number of residual blocks
    'channels': 64,                 # Number of channels in convolutional layers
    'attention_key_dim': 32,        # Dimension of attention keys
    'fc_hidden_size': 2048,         # Size of fully connected hidden layer
    'dropout_rate': 0.2,            # Dropout rate for regularization
}

# Optimization Parameters
OPTIMIZER_CONFIG = {
    'learning_rate': 0.001,         # Initial learning rate
    'lr_scheduler': 'plateau',      # Learning rate scheduler type ('step', 'plateau', 'cosine')
    'lr_patience': 5,               # Patience for ReduceLROnPlateau scheduler
    'lr_factor': 0.5,               # Factor by which to reduce learning rate
    'lr_min': 1e-6,                 # Minimum learning rate
    'weight_decay': 1e-4,           # L2 regularization factor
    'gradient_clip': 5.0,           # Gradient clipping threshold
    'batch_size': 256,              # Batch size for training (increased for better GPU utilization)
    'accumulation_steps': 4,        # Number of batches to accumulate gradients over (effective batch size = batch_size * accumulation_steps)
    'use_gradient_accumulation': True, # Whether to use gradient accumulation
}

# Experience Replay Parameters
MEMORY_CONFIG = {
    'capacity': 200000,             # Memory capacity (increased for better coverage)
    'alpha': 0.6,                   # Priority exponent for prioritized replay
    'beta_start': 0.4,              # Initial beta value for importance sampling
    'beta_frames': 100000,          # Number of frames to anneal beta to 1.0
}

# Reward Parameters
REWARD_CONFIG = {
    'gamma': 0.99,                  # Discount factor for future rewards
    'td_steps': 3,                  # Number of steps for TD learning (increased from 1)
    'checkmate_reward': 5.0,        # Reward for checkmate
    'draw_reward': 0.0,             # Reward for draw
    'move_penalty': 0.01,           # Small penalty for each move to encourage faster wins
    'repetition_penalty': 0.95,     # Penalty factor for repeated positions
}

# Exploration Parameters
EXPLORATION_CONFIG = {
    'eps_start': 1.0,               # Starting epsilon for epsilon-greedy exploration
    'eps_end': 0.05,                # Final epsilon value
    'eps_decay': 100000,            # Number of steps to decay epsilon
    'temperature': 1.0,             # Temperature for softmax action selection
    'dirichlet_alpha': 0.3,         # Dirichlet noise alpha for root exploration
    'noise_fraction': 0.25,         # Fraction of Dirichlet noise to add
}

# Curriculum Learning Parameters
CURRICULUM_CONFIG = {
    'base_win_rate_threshold': 0.55,  # Base threshold for advancing levels
    'win_rate_window': 30,          # Number of games to calculate win rate over
    'max_pool_size': 10,            # Maximum size of model pool
    'stockfish_levels': range(1, 11),  # Range of Stockfish levels to use
}

# Self-Play Parameters
SELF_PLAY_CONFIG = {
    'num_episodes': 10000,          # Number of self-play episodes
    'max_moves': 250,               # Maximum moves per game (reduced from 300 for efficiency)
    'early_stopping_no_progress': 30, # Stop game if no material change in this many moves
    'eval_interval': 500,           # Interval between evaluations
    'save_interval': 1000,          # Interval between saving models
    'target_update': 500,           # Interval between target network updates (reduced from 1000)
    'stockfish_time': 0.1,          # Time limit for Stockfish moves (seconds)
}

# Asynchronous Evaluation Parameters
ASYNC_EVAL_CONFIG = {
    'num_workers': 8,               # Number of worker threads for async evaluation (increased from 4)
    'cache_size': 20000,            # Size of evaluation cache (increased from 10000)
    'default_depth': 12,            # Default evaluation depth
    'critical_depth': 16,           # Depth for critical positions
    'enable_prefetch': True,        # Enable prefetching of evaluations
}

# Mixed Precision Parameters
MIXED_PRECISION_CONFIG = {
    'enabled': True,                # Whether to use mixed precision
    'scale_factor': 128.0,          # Initial scale factor for loss scaling
    'growth_interval': 2000,        # Steps between scale factor increases
}

# PGN Training Parameters
PGN_CONFIG = {
    'num_games': 200000,            # Number of games to process
    'min_elo': 2000,                # Minimum Elo rating for games
    'max_positions': 1000000,       # Maximum positions to extract
    'augment_data': True,           # Whether to augment data with symmetries
}

# Evaluation Parameters
EVAL_CONFIG = {
    'num_games': 10,                # Number of games to play in evaluation
    'stockfish_levels': [1, 3, 5, 7, 10],  # Stockfish levels to evaluate against
    'max_moves': 100,               # Maximum moves per evaluation game
}

# Get optimized hyperparameters for the current hardware
def get_optimized_hyperparameters(gpu_type=None):
    """
    Get optimized hyperparameters for the current hardware.

    Args:
        gpu_type (str, optional): GPU type to optimize for

    Returns:
        dict: Dictionary of optimized hyperparameters
    """
    config = {
        'network': NETWORK_CONFIG.copy(),
        'optimizer': OPTIMIZER_CONFIG.copy(),
        'memory': MEMORY_CONFIG.copy(),
        'reward': REWARD_CONFIG.copy(),
        'exploration': EXPLORATION_CONFIG.copy(),
        'curriculum': CURRICULUM_CONFIG.copy(),
        'self_play': SELF_PLAY_CONFIG.copy(),
        'async_eval': ASYNC_EVAL_CONFIG.copy(),
        'mixed_precision': MIXED_PRECISION_CONFIG.copy(),
        'pgn': PGN_CONFIG.copy(),
        'eval': EVAL_CONFIG.copy(),
    }

    # Optimize for RTX 4070
    if gpu_type == 'rtx_4070':
        # Increase batch size for better GPU utilization
        config['optimizer']['batch_size'] = 256

        # Enable gradient accumulation for effective larger batches
        config['optimizer']['use_gradient_accumulation'] = True
        config['optimizer']['accumulation_steps'] = 4

        # Enable mixed precision
        config['mixed_precision']['enabled'] = True

        # Increase number of workers for async evaluation
        config['async_eval']['num_workers'] = 8
        config['async_eval']['enable_prefetch'] = True
        config['async_eval']['cache_size'] = 20000

        # Optimize network architecture
        config['network']['channels'] = 128
        config['network']['fc_hidden_size'] = 4096

        # Reduce max moves per game for efficiency
        config['self_play']['max_moves'] = 250
        config['self_play']['early_stopping_no_progress'] = 30

    return config
