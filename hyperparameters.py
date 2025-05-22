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
    'learning_rate': 0.003,         # Initial learning rate (increased from 0.001 for faster initial learning)
    'lr_scheduler': 'plateau',      # Learning rate scheduler type ('step', 'plateau', 'cosine')
    'lr_patience': 3,               # Patience for ReduceLROnPlateau scheduler (reduced for more aggressive decay)
    'lr_factor': 0.3,               # Factor by which to reduce learning rate (reduced for more aggressive decay)
    'lr_min': 1e-6,                 # Minimum learning rate
    'weight_decay': 1e-4,           # L2 regularization factor
    'gradient_clip': 5.0,           # Gradient clipping threshold
    'batch_size': 256,              # Batch size for training (increased for better GPU utilization)
    'accumulation_steps': 4,        # Number of batches to accumulate gradients over (effective batch size = batch_size * accumulation_steps)
    'use_gradient_accumulation': True, # Whether to use gradient accumulation

    # AlphaZero-style loss parameters
    'policy_loss_scale': 1.0,       # Scaling factor for policy loss
    'value_loss_scale': 1.0,        # Scaling factor for value loss
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
    'stockfish_eval_frequency': 0.2, # Evaluate only 20% of positions with Stockfish
    'hybrid_eval_enabled': False,   # Whether to use hybrid evaluation approach
    'low_freq_rate': 0.01,          # Low frequency rate (1%) for fast training
    'high_freq_rate': 0.2,          # High frequency rate (20%) for quality training
    'high_freq_interval': 50        # Run high frequency evaluation every N episodes
}

# Exploration Parameters
EXPLORATION_CONFIG = {
    # Legacy epsilon-greedy parameters (kept for backward compatibility)
    'eps_start': 1.0,               # Starting epsilon for epsilon-greedy exploration
    'eps_end': 0.05,                # Final epsilon value
    'eps_decay': 100000,            # Number of steps to decay epsilon

    # AlphaZero-style temperature-based sampling parameters
    'use_temperature': True,        # Whether to use temperature-based sampling
    'initial_temperature': 1.0,     # Initial temperature for exploration
    'min_temperature': 0.1,         # Minimum temperature after annealing
    'temperature_decay': 50000,     # Number of steps to decay temperature

    # Phase-based temperature parameters
    'use_phase_temperature': True,  # Whether to adjust temperature based on game phase
    'opening_temperature': 1.2,     # Temperature for opening phase (more exploration)
    'middlegame_temperature': 0.7,  # Temperature for middlegame phase
    'endgame_temperature': 0.3,     # Temperature for endgame phase (more exploitation)

    # Dirichlet noise parameters for root exploration
    'use_dirichlet': True,          # Whether to add Dirichlet noise at root positions
    'dirichlet_alpha': 0.3,         # Dirichlet noise concentration parameter
    'dirichlet_epsilon': 0.25,      # Fraction of Dirichlet noise to add

    # Lookahead for critical positions
    'use_lookahead': True,          # Whether to use lookahead for critical positions
    'lookahead_depth': 2,           # Depth of lookahead (1 or 2 ply)
}

# Curriculum Learning Parameters
CURRICULUM_CONFIG = {
    'base_win_rate_threshold': 0.55,  # Base threshold for advancing levels
    'win_rate_window': 30,          # Number of games to calculate win rate over
    'max_pool_size': 10,            # Maximum size of model pool
    'stockfish_levels': range(1, 11),  # Range of Stockfish levels to use

    # Random move opponent parameters
    'random_move_pct': 0.25,        # Initial percentage of random moves (25%)
    'random_move_decay': 0.05,      # How much to decrease randomness when win rate improves
    'random_move_min': 0.05,        # Minimum random move percentage
    'random_win_rate_threshold': 0.70,  # Win rate threshold to introduce Stockfish (70%)
    'position_diversity_enabled': True,  # Whether to use position diversity improvements
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

    # Position diversity improvements
    'use_opening_book': True,       # Whether to use opening book positions
    'opening_book_frequency': 0.3,  # Frequency of using opening book positions (30%)
    'use_tablebase': True,          # Whether to use endgame tablebase positions
    'tablebase_frequency': 0.2,     # Frequency of using tablebase positions (20%)
    'random_opponent_frequency': 0.6, # Frequency of playing against random move opponents (60%)
}

# Asynchronous Evaluation Parameters
ASYNC_EVAL_CONFIG = {
    'num_workers': 8,               # Number of worker threads for async evaluation (increased from 4)
    'cache_size': 20000,            # Size of evaluation cache (increased from 10000)
    'default_depth': 12,            # Default evaluation depth
    'critical_depth': 16,           # Depth for critical positions
    'enable_prefetch': True,        # Enable prefetching of evaluations

    # Position diversity improvements
    'use_position_clustering': True,  # Whether to use position clustering for cache
    'cluster_size': 100,            # Size of position clusters
    'similarity_threshold': 0.85,   # Threshold for position similarity
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
        config['optimizer']['batch_size'] = 512  # Increased from 256 for better GPU utilization

        # Enable gradient accumulation for effective larger batches
        config['optimizer']['use_gradient_accumulation'] = True
        config['optimizer']['accumulation_steps'] = 2  # Reduced from 4 since batch size is larger

        # Temporarily disable mixed precision due to compatibility issues
        config['mixed_precision']['enabled'] = True

        # Increase number of workers for async evaluation
        config['async_eval']['num_workers'] = 8
        config['async_eval']['enable_prefetch'] = True
        config['async_eval']['cache_size'] = 50000  # Increased from 20000 for better cache hits

        # Optimize network architecture
        config['network']['channels'] = 128
        config['network']['fc_hidden_size'] = 4096

        # Reduce max moves per game for efficiency
        config['self_play']['max_moves'] = 250
        config['self_play']['early_stopping_no_progress'] = 30

        # Enable hybrid Stockfish evaluation approach for optimal speed/quality balance
        config['reward']['hybrid_eval_enabled'] = True
        config['reward']['low_freq_rate'] = 0.01  # Use 1% frequency for most episodes (fast)
        config['reward']['high_freq_rate'] = 0.1  # Use 10% frequency for quality episodes
        config['reward']['high_freq_interval'] = 50  # Run high quality evaluation every 50 episodes
        config['reward']['stockfish_eval_frequency'] = 0.01  # Default to 1% for maximum speed

        # Increase target network update frequency to prevent Q-value drift
        config['self_play']['target_update'] = 250  # More frequent updates

        # AlphaZero-style parameters
        # Optimize loss scaling for better training stability
        config['optimizer']['policy_loss_scale'] = 1.5  # Slightly emphasize policy learning
        config['optimizer']['value_loss_scale'] = 1.0  # Base value for value learning

        # Temperature-based sampling parameters
        config['exploration']['use_temperature'] = True
        config['exploration']['initial_temperature'] = 1.2  # Start with higher exploration
        config['exploration']['min_temperature'] = 0.1  # End with more exploitation
        config['exploration']['temperature_decay'] = 100000  # Decay over 100k steps

        # Phase-based temperature parameters
        config['exploration']['use_phase_temperature'] = True
        config['exploration']['opening_temperature'] = 1.5  # More exploration in opening
        config['exploration']['middlegame_temperature'] = 0.8  # Moderate in middlegame
        config['exploration']['endgame_temperature'] = 0.3  # More exploitation in endgame

        # Dirichlet noise parameters
        config['exploration']['use_dirichlet'] = True
        config['exploration']['dirichlet_alpha'] = 0.3  # Standard AlphaZero value
        config['exploration']['dirichlet_epsilon'] = 0.25  # 25% noise, 75% network policy

        # Lookahead for critical positions
        config['exploration']['use_lookahead'] = True
        config['exploration']['lookahead_depth'] = 2  # 2-ply lookahead for critical positions

        # Random move opponent parameters
        config['curriculum']['random_move_pct'] = 0.25  # Start with 25% random moves
        config['curriculum']['random_move_decay'] = 0.05  # Decrease by 5% when win rate improves
        config['curriculum']['random_move_min'] = 0.05  # Minimum 5% random moves
        config['curriculum']['random_win_rate_threshold'] = 0.70  # 70% win rate to introduce Stockfish
        config['curriculum']['position_diversity_enabled'] = True

        # Position diversity improvements
        config['self_play']['use_opening_book'] = True
        config['self_play']['opening_book_frequency'] = 0.3
        config['self_play']['use_tablebase'] = True
        config['self_play']['tablebase_frequency'] = 0.2
        config['self_play']['random_opponent_frequency'] = 0.6

        # Position clustering for cache
        config['async_eval']['use_position_clustering'] = True
        config['async_eval']['cluster_size'] = 100
        config['async_eval']['similarity_threshold'] = 0.85

    # Optimize for M2 Mac

    return config
