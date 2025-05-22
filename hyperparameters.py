"""
Simple hyperparameter configuration for chess reinforcement learning.

This module provides basic hyperparameters for training chess models.
"""

# Neural Network Architecture
NETWORK_CONFIG = {
    'residual_blocks': 2,           # Number of residual blocks
    'channels': 32,                 # Number of channels in convolutional layers
    'dropout_rate': 0.2,            # Dropout rate for regularization
}

# Optimization Parameters
OPTIMIZER_CONFIG = {
    'learning_rate': 0.003,         # Initial learning rate
    'lr_scheduler': 'plateau',      # Learning rate scheduler type
    'batch_size': 256,              # Batch size for training
    'weight_decay': 1e-4,           # L2 regularization factor
}

# Experience Replay Parameters
MEMORY_CONFIG = {
    'capacity': 100000,             # Memory capacity
}

# Reward Parameters
REWARD_CONFIG = {
    'gamma': 0.99,                  # Discount factor for future rewards
    'stockfish_eval_frequency': 0.1, # Evaluate 10% of positions with Stockfish
}

# Exploration Parameters
EXPLORATION_CONFIG = {
    'temperature': 1.0,             # Temperature for exploration
}

# Self-Play Parameters
SELF_PLAY_CONFIG = {
    'num_episodes': 5000,           # Number of self-play episodes
    'max_moves': 200,               # Maximum moves per game
    'save_interval': 500,           # Interval between saving models
    'target_update': 500,           # Interval between target network updates
}

# Asynchronous Evaluation Parameters
ASYNC_EVAL_CONFIG = {
    'num_workers': 4,               # Number of worker threads for async evaluation
    'cache_size': 10000,            # Size of evaluation cache
    'default_depth': 8,             # Default evaluation depth
}

# PGN Training Parameters
PGN_CONFIG = {
    'num_games': 10000,             # Number of games to process
}

# Evaluation Parameters
EVAL_CONFIG = {
    'num_games': 5,                 # Number of games to play in evaluation
    'stockfish_levels': [1, 3, 5],  # Stockfish levels to evaluate against
    'max_moves': 100,               # Maximum moves per evaluation game
}

# Get basic hyperparameters
def get_optimized_hyperparameters(gpu_type=None):
    """
    Get basic hyperparameters.

    Args:
        gpu_type (str, optional): GPU type (not used in simplified version)

    Returns:
        dict: Dictionary of hyperparameters
    """
    config = {
        'network': NETWORK_CONFIG.copy(),
        'optimizer': OPTIMIZER_CONFIG.copy(),
        'memory': MEMORY_CONFIG.copy(),
        'reward': REWARD_CONFIG.copy(),
        'exploration': EXPLORATION_CONFIG.copy(),
        'self_play': SELF_PLAY_CONFIG.copy(),
        'async_eval': ASYNC_EVAL_CONFIG.copy(),
        'pgn': PGN_CONFIG.copy(),
        'eval': EVAL_CONFIG.copy(),
    }

    # Add curriculum config with minimal settings
    config['curriculum'] = {
        'random_move_pct': 0.25,
    }

    # Add mixed precision config with minimal settings
    config['mixed_precision'] = {
        'enabled': True,
    }

    return config
