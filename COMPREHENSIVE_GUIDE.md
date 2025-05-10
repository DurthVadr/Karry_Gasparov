# Karry_Gasparov: Comprehensive Guide

This guide provides detailed documentation for all components and options of the Karry_Gasparov chess AI system.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Chess GUI](#chess-gui)
4. [Model Integration](#model-integration)
5. [Deep Reinforcement Learning Agent](#deep-reinforcement-learning-agent)
6. [Training Methods](#training-methods)
7. [Training Process Details](#training-process-details)
8. [Neural Network Details](#neural-network-details)
9. [Board Representation](#board-representation)
10. [Move Selection Process](#move-selection-process)
11. [Evaluation Methods](#evaluation-methods)
12. [File Structure](#file-structure)
13. [Command-Line Options](#command-line-options)
14. [Advanced Options](#advanced-options)
15. [Future Development](#future-development)
16. [Troubleshooting](#troubleshooting)

## Overview

Karry_Gasparov is a chess AI implementation using deep reinforcement learning techniques. The system consists of several components:

- **Chess GUI**: Interactive chess board for playing against the AI
- **DRL Agent**: Deep reinforcement learning model for chess move prediction
- **Model Integration**: Interface between the GUI and trained models
- **Training System**: Methods for training models using PGN data and self-play

## Installation

1. Clone the repository:
   ```bash
   cd Karry_Gasparov
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Stockfish (optional, for enhanced training):
   - macOS: `brew install stockfish`
   - Ubuntu/Debian: `apt-get install stockfish`
   - Windows: Download from [Stockfish website](https://stockfishchess.org/download/)

## Chess GUI

The chess GUI provides an interactive interface for playing against the trained models.

### Starting the GUI

```bash
python chess_gui.py
```

### GUI Features

- **Interactive Board**: Click to select and move pieces
- **Move History**: Displays the game's move history in algebraic notation
- **Game Status**: Shows the current game state (check, checkmate, etc.)
- **Control Buttons**:
  - **New Game**: Start a new chess game
  - **Undo Move**: Take back the last move
  - **AI Move**: Make the AI play a move
  - **Load Model**: Load a trained model file

### GUI Options

| Option | Description |
|--------|-------------|
| New Game | Resets the board to the starting position |
| Undo Move | Reverts the last move made on the board |
| AI Move | Makes the AI play a move as Black |
| Load Model | Opens a file dialog to select a model file (.pt or .pth) |

## Model Integration

The model integration system provides a unified interface for loading and using different chess models.

### Available Model Types

- **DRL Chess Model**: Deep reinforcement learning model (PyTorch .pt/.pth files)
- **Random Model**: Fallback model that makes random legal moves

### Model Integration Options

| Option | Description | Default |
|--------|-------------|---------|
| model_path | Path to the model file | None |
| repetition_penalty | Penalty factor for repeated positions (0-1) | 0.95 |

### Usage Example

```python
from model_integration import ModelIntegration
import chess

# Create a board and model integration
board = chess.Board()
model_integration = ModelIntegration()

# Load a model with custom repetition penalty
model = model_integration.load_model(
    "models/best_model.pt",
    repetition_penalty=0.9  # Stronger penalty for repetitions
)

# Get a move from the model
move = model.get_move(board)
```

## Deep Reinforcement Learning Agent

The DRL agent uses a neural network to evaluate chess positions and select the best moves.

### Neural Network Architecture

- **Input**: 16-channel 8x8 board representation
  - Channels 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
  - Channels 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
  - Channels 12-13: Castling rights
  - Channel 14: En passant
  - Channel 15: Player to move

- **Network Structure**:
  - Convolutional layers with residual connections
  - Batch normalization for training stability
  - Dropout for regularization
  - Fully connected layers for move evaluation

- **Output**: 4096 Q-values (64x64 possible from-to square combinations)

### Agent Options

| Option | Description | Default |
|--------|-------------|---------|
| model_path | Path to a saved model file (.pt or .pth) | None |
| repetition_penalty | Penalty factor for repeated positions (0-1) | 0.95 |

## Training Methods

The system provides two main training methods:

1. **PGN Training**: Learning from existing chess games
2. **Self-Play Training**: Learning through playing against itself or Stockfish

### PGN Training Options

| Option | Description | Default |
|--------|-------------|---------|
| pgn_path | Path to a PGN file or directory containing PGN files | None |
| num_games | Maximum number of games to process | 1000 |
| batch_size | Number of positions to process in each batch | 32 |
| save_interval | How often to save the model (in games) | 100 |

### Self-Play Training Options

| Option | Description | Default |
|--------|-------------|---------|
| num_episodes | Total number of episodes to train | 5000 |
| stockfish_opponent | Whether to use Stockfish as an opponent | True |
| stockfish_levels | List of Stockfish levels to train against | [1,2,3,4,5,6,7] |
| batch_size | Batch size for optimization | 64 |
| save_interval | How often to save models (in episodes) | 100 |
| eval_interval | How often to evaluate against target level (in episodes) | 500 |
| target_level | Target Stockfish level to achieve | 7 |

## Advanced Options

### Repetition Avoidance

The agent includes a repetition detection and avoidance system to prevent draws by repetition:

- **repetition_penalty**: Factor to penalize moves that lead to repeated positions
  - Lower values = stronger penalty
  - Range: 0.0 to 1.0
  - Default: 0.95

### Curriculum Learning

Self-play training includes curriculum learning that gradually increases opponent strength:

- Starts with Stockfish level 1
- Advances to higher levels when win rate exceeds threshold
- Adds previous model versions to opponent pool for diversity

### Reward Calculation

Two reward calculation methods are used:

1. **Material-based reward**: Fast calculation based on piece values
2. **Stockfish evaluation**: More accurate but slower calculation using Stockfish

## Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Check that the model file exists and is a valid PyTorch model
   - Ensure the model architecture matches the expected format

2. **GUI display issues**:
   - Ensure all dependencies are installed correctly
   - Try resizing the window if the board doesn't display properly

3. **Training performance issues**:
   - Use a GPU for faster training if available
   - Reduce batch size if memory issues occur
   - Use smaller PGN datasets for initial testing

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the project's GitHub issues page
2. Create a new issue with detailed information about your problem

## Training Process Details

### PGN Training Process

The PGN training process involves the following steps:

1. **Data Loading**: Load chess games from PGN files
2. **Position Extraction**: Extract board positions and corresponding moves
3. **Batch Processing**: Process positions in batches for efficiency
4. **Reward Calculation**: Calculate rewards for each position
5. **Experience Storage**: Store experiences in replay memory
6. **Model Optimization**: Update the model based on experiences
7. **Model Saving**: Save checkpoints at regular intervals

#### Optimizations in PGN Training

- **Batch Processing**: Processes multiple positions at once for better GPU utilization
- **Selective Stockfish Evaluation**: Uses Stockfish for only a sample of positions to reduce computation
- **Vectorized Operations**: Uses efficient tensor operations where possible
- **Learning Rate Scheduling**: Gradually decreases learning rate for better convergence

### Self-Play Training Process

The self-play training process includes:

1. **Game Initialization**: Set up a new chess game
2. **Move Selection**: Select moves using the current policy
3. **Experience Collection**: Collect state, action, reward, next state tuples
4. **Batch Processing**: Process experiences in batches
5. **Model Optimization**: Update the model based on experiences
6. **Opponent Selection**: Alternate between different opponents:
   - Current policy (self-play)
   - Previous model versions
   - Stockfish at various levels
7. **Curriculum Advancement**: Increase difficulty when performance improves
8. **Evaluation**: Periodically evaluate against target Stockfish level

#### Advanced Self-Play Features

- **Model Pool**: Maintains a pool of previous model versions for diverse opponents
- **Adaptive Difficulty**: Adjusts opponent strength based on performance
- **Challenge Games**: Occasionally plays against stronger opponents for better learning
- **Win Rate Tracking**: Monitors win rate to determine when to increase difficulty

## Neural Network Details

### Residual Blocks

The neural network uses residual blocks to improve gradient flow during training:

```python
class ResidualBlock(nn.Module):
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
```

### Move Masking

The network uses a mask layer to ensure only legal moves are considered:

```python
class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def forward(self, x, mask):
        # Apply mask: set invalid move scores to -inf
        masked_output = x.masked_fill(mask == 0, -float("inf"))
        return masked_output
```

## Board Representation

The chess board is represented as a 16-channel 8x8 tensor:

1. **Piece Channels (0-11)**:
   - Channels 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
   - Channels 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)

2. **Special Channels (12-15)**:
   - Channel 12: White castling rights
   - Channel 13: Black castling rights
   - Channel 14: En passant square
   - Channel 15: Player to move (1 for White, 0 for Black)

## Move Selection Process

The move selection process involves:

1. **Board Conversion**: Convert chess.Board to tensor representation
2. **Legal Move Masking**: Create a mask of legal moves
3. **Q-Value Prediction**: Get Q-values from the neural network
4. **Repetition Penalty**: Apply penalties for moves leading to repeated positions
5. **Best Move Selection**: Select the move with highest adjusted Q-value
6. **Promotion Handling**: Handle pawn promotions (default to queen)
7. **Legality Check**: Ensure the selected move is legal

## Evaluation Methods

The system includes methods for evaluating model performance:

1. **Stockfish Evaluation**: Play against Stockfish at different levels
2. **Win Rate Tracking**: Monitor win rate against specific opponents
3. **Position Evaluation**: Compare model evaluations with Stockfish
4. **Self-Play Tournaments**: Pit different model versions against each other

## File Structure

```
Karry_Gasparov/
├── chess_gui.py           # Interactive chess GUI
├── drl_agent.py           # Deep reinforcement learning agent
├── model_integration.py   # Interface between GUI and models
├── training.py            # Training methods implementation
├── utils.py               # Utility functions
├── demo.py                # Demo script
├── data/                  # Directory for PGN files
├── models/                # Directory for saved models
└── README.md              # Project overview
```

## Command-Line Options

### Training Script

The main training script is `main.py`, which provides a comprehensive command-line interface for training and evaluating chess models:

```bash
# macOS
python main.py --data_dir data --num_games 10000 --batch_size 64 --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish

# Windows
python main.py --data_dir data --num_games 10000 --batch_size 64 --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe"
```

#### Training Options:
- `--data_dir PATH`: Directory containing PGN files (default: "data")
- `--num_games N`: Maximum number of games to process (default: 200000)
- `--batch_size N`: Batch size for training (default: 64)
- `--save_interval N`: How often to save the model in games/episodes (default: 1000)
- `--stockfish_path PATH`: Path to Stockfish executable

#### Self-Play Options:
- `--self_play`: Enable self-play training after PGN training
- `--self_play_episodes N`: Number of self-play episodes (default: 5000)
- `--stockfish_opponent`: Use Stockfish as opponent in self-play (default: True)
- `--target_level N`: Target Stockfish level to achieve (default: 7)
- `--eval_interval N`: How often to evaluate against target level (default: 500)

#### Evaluation Options:
- `--evaluate`: Evaluate model against Stockfish levels
- `--eval_games N`: Number of games to play against each level (default: 5)
- `--eval_model PATH`: Path to specific model to evaluate (optional)
- `--min_level N`: Minimum Stockfish level to test against (default: 1)
- `--max_level N`: Maximum Stockfish level to test against (default: 10)

### Example Commands

#### PGN Training Only:
```bash
# macOS
python main.py --data_dir data --num_games 10000 --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish

# Windows
python main.py --data_dir data --num_games 10000 --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe"
```

#### Self-Play Training After PGN Training:
```bash
# macOS
python main.py --data_dir data --num_games 5000 --self_play --self_play_episodes 2000 --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish

# Windows
python main.py --data_dir data --num_games 5000 --self_play --self_play_episodes 2000 --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe"
```

#### Evaluation Only:
```bash
# macOS
python main.py --evaluate --eval_games 10 --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish

# Windows
python main.py --evaluate --eval_games 10 --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe"
```

### GUI Script

```bash
python chess_gui.py [model_path]
```

Options:
- `model_path`: Optional path to a model file to load at startup

## Future Development

Planned improvements for the system:

1. **Model Architecture**:
   - AlphaZero-style MCTS with neural network guidance
   - Attention mechanisms for better position understanding
   - Transformer architectures for improved learning

2. **Training Process**:
   - Distributed training across multiple machines
   - Better exploration strategies
   - Improved reward functions

3. **User Interface**:
   - More customization options
   - Analysis features
   - Game database integration
