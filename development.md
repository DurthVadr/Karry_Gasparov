# Karry Gasparov Development Guide

This document provides a detailed explanation of each file in the Karry Gasparov chess AI project, designed to help developers understand the codebase and make contributions.

## Project Overview

Karry Gasparov is a chess AI that uses deep reinforcement learning to learn and improve its chess playing abilities. The project implements a simplified version of the AlphaZero approach, using a neural network to evaluate positions and select moves.

## File Structure

The project consists of the following key files:

- `drl_agent.py`: Core neural network architecture and chess agent
- `training.py`: Training loops and procedures
- `reward.py`: Reward calculation for reinforcement learning
- `async_evaluation.py`: Asynchronous Stockfish evaluation
- `hyperparameters.py`: Configuration parameters
- `position_diversity.py`: Generates diverse chess positions for training

## Detailed File Descriptions

### drl_agent.py

This file contains the core neural network architecture and chess agent implementation.

#### Key Components:

1. **SimplifiedAttention**: A lightweight attention mechanism that focuses on channel relationships (piece types) to capture important chess-specific patterns.

2. **ResidualBlock**: Standard residual block with two convolutional layers and batch normalization, allowing for better gradient flow in deep networks.

3. **ChessFeatureExtractor**: A simple feature extraction module that applies a convolutional layer with a residual connection to extract basic chess features.

4. **MaskLayer**: Ensures that only legal moves are considered by applying a mask to the policy output.

5. **DQN**: The main neural network architecture with:
   - 2 residual blocks
   - 32 channels in convolutional layers
   - A policy head that outputs move probabilities
   - A value head that evaluates positions

6. **ChessAgent**: Handles the interface between the neural network and chess environment:
   - Converts chess positions to tensors
   - Selects moves based on network output
   - Handles special cases like promotions
   - Implements repetition avoidance

### training.py

This file contains the training procedures for the chess AI.

#### Key Components:

1. **PGNTrainer**: Trains the model using existing chess games in PGN format:
   - Processes PGN files to extract positions and moves
   - Calculates rewards for positions
   - Stores experiences in replay memory
   - Performs optimization steps

2. **SelfPlayTrainer**: Implements self-play training where the agent plays against itself:
   - Generates games through self-play
   - Collects training data from these games
   - Implements simple opponent selection

3. **Helper Functions**:
   - Move quality evaluation
   - Material counting
   - Position diversity management

### reward.py

This file handles reward calculation for the reinforcement learning process.

#### Key Components:

1. **RewardCalculator**: Calculates rewards for chess positions:
   - Uses Stockfish for position evaluation when available
   - Implements a fallback material-based evaluation
   - Provides caching for efficient evaluation
   - Calculates positional bonuses for good chess principles

2. **Key Methods**:
   - `calculate_stockfish_reward`: Uses Stockfish to evaluate positions
   - `calculate_reward`: Fallback method using material advantage
   - `calculate_material_advantage`: Counts material value on the board
   - `calculate_positional_bonus`: Rewards good chess principles like center control

### async_evaluation.py

This file implements asynchronous Stockfish evaluation to improve performance.

#### Key Components:

1. **AsyncStockfishEvaluator**: Manages multiple Stockfish instances in separate threads:
   - Creates a pool of worker threads
   - Handles request queuing and result collection
   - Implements caching for efficient evaluation
   - Provides batch evaluation capabilities

2. **Key Methods**:
   - `evaluate_position`: Submits a position for evaluation
   - `get_result`: Retrieves evaluation results
   - `evaluate_positions_batch`: Evaluates multiple positions in parallel

### hyperparameters.py

This file contains configuration parameters for the chess AI.

#### Key Components:

1. **Configuration Dictionaries**:
   - `NETWORK_CONFIG`: Neural network architecture parameters
   - `OPTIMIZER_CONFIG`: Optimization parameters
   - `MEMORY_CONFIG`: Experience replay parameters
   - `REWARD_CONFIG`: Reward calculation parameters
   - `EXPLORATION_CONFIG`: Exploration parameters
   - `SELF_PLAY_CONFIG`: Self-play training parameters
   - `ASYNC_EVAL_CONFIG`: Asynchronous evaluation parameters
   - `PGN_CONFIG`: PGN training parameters
   - `EVAL_CONFIG`: Evaluation parameters

2. **get_optimized_hyperparameters**: Function that returns a complete set of hyperparameters, with optional hardware-specific optimizations.

### position_diversity.py

This file provides functions for generating diverse chess positions for training.

#### Key Components:

1. **PositionDiversity**: Generates diverse chess positions:
   - Creates opening book positions
   - Creates endgame positions
   - Validates positions for legality

2. **Key Methods**:
   - `get_random_opening_position`: Returns a random position from the opening book
   - `get_random_endgame_position`: Returns a random endgame position
   - `_create_opening_book`: Initializes the opening book
   - `_create_endgame_positions`: Initializes endgame positions

## Development Workflow

### Setting Up the Environment

1. Ensure you have Python 3.8+ installed
2. Install required packages: `pip install torch chess python-chess`
3. Install Stockfish chess engine (optional but recommended for evaluation)

### Training the Model

1. Configure hyperparameters in `hyperparameters.py`
2. Run training using PGN data: `python train.py --mode pgn --pgn_file <path_to_pgn>`
3. Run self-play training: `python train.py --mode self_play --episodes 1000`

### Evaluating the Model

1. Evaluate against Stockfish: `python evaluate.py --model <model_path> --stockfish_level 3`
2. Play against the model: `python play.py --model <model_path>`

## Common Development Tasks

### Adding a New Feature to the Neural Network

1. Modify the DQN class in `drl_agent.py`
2. Update the forward method to incorporate the new feature
3. Adjust hyperparameters in `hyperparameters.py` if necessary
4. Test the changes with a small training run

### Improving the Reward Function

1. Modify the RewardCalculator class in `reward.py`
2. Add or update reward components in calculate_reward or calculate_positional_bonus
3. Test the changes with a small training run

### Optimizing Performance

1. Profile the code to identify bottlenecks
2. Consider increasing batch size or using gradient accumulation
3. Optimize the async evaluation parameters
4. Use mixed precision training if available

## Best Practices

1. **Testing**: Always test changes with small training runs before full training
2. **Hyperparameter Tuning**: Make incremental changes to hyperparameters
3. **Model Checkpoints**: Save model checkpoints frequently during training
4. **Documentation**: Update this document when making significant changes
5. **Code Style**: Follow PEP 8 guidelines for Python code
