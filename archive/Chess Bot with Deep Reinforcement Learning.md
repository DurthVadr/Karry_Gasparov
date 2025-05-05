# Chess Bot with Deep Reinforcement Learning

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Training the Model](#training-the-model)
   - [Local Training](#local-training)
   - [Google Colab Training](#google-colab-training)
   - [Using Custom PGN Data](#using-custom-pgn-data)
5. [Model Integration](#model-integration)
6. [Using the Chess Bot](#using-the-chess-bot)
7. [Technical Details](#technical-details)
   - [Deep Q-Network Architecture](#deep-q-network-architecture)
   - [State Representation](#state-representation)
   - [Action Space](#action-space)
   - [Reward Function](#reward-function)
   - [Training Process](#training-process)
8. [Troubleshooting](#troubleshooting)
9. [Future Improvements](#future-improvements)

## Project Overview

This project implements a chess-playing AI using deep reinforcement learning techniques. The system learns to play chess by training on existing chess games (in PGN format) and through self-play. The implementation uses a Deep Q-Network (DQN) architecture to evaluate chess positions and select the best moves.

Key features:
- Deep neural network for chess position evaluation
- Training from PGN data (supervised learning)
- Self-play reinforcement learning
- Integration with a chess GUI
- Support for both local and Google Colab training

## System Architecture

The system consists of several key components:

1. **Deep Reinforcement Learning Model** (`drl_agent.py`):
   - Implements the neural network architecture
   - Provides functions for board representation and move selection
   - Contains the core DQN implementation

2. **Training System** (`train_model.py` and `colab_training_fixed.py`):
   - Handles training from PGN data
   - Implements self-play reinforcement learning
   - Manages experience replay and optimization

3. **Model Integration** (`model_integration.py`):
   - Provides an interface between the trained model and the chess GUI
   - Handles model loading and move selection

4. **Testing Tools** (`test_model.py`):
   - Allows for testing the model without a GUI
   - Provides a simple interface for evaluating model performance

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- PyTorch
- python-chess
- NumPy
- Matplotlib (for visualization)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/DurthVadr/Karry_Gasparov.git
   cd Karry_Gasparov
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create directories for data and models:
   ```bash
   mkdir -p data/synthetic
   mkdir -p models
   ```

## Training the Model

The chess bot can be trained in two ways:
1. Using existing chess games in PGN format (supervised learning)
2. Through self-play (reinforcement learning)

### Local Training

To train the model locally:

1. Ensure you have PGN data available or generate synthetic data:
   ```bash
   python3 data/synthetic/generate_synthetic_data.py
   ```

2. Run the training script:
   ```bash
   python3 train_model.py
   ```

The training script will:
- First train on the PGN data (if available)
- Then continue training through self-play
- Save model checkpoints periodically in the `models` directory

### Google Colab Training

For faster training using Google Colab's GPU:

1. Open Google Colab (https://colab.research.google.com/)
2. Create a new notebook
3. Copy the contents of `colab_training_fixed.py` into a code cell
4. Upload the necessary files:
   - `drl_agent.py` (if you want to import from it)
   - Your PGN data (if using custom data)
5. Run the cell to start training

### Using Custom PGN Data

To use your own chess game data:

1. Ensure your data is in standard PGN format
2. Place the PGN file in the `data` directory
3. Update the `pgn_file` path in the training script:
   ```python
   # In train_model.py or colab_training_fixed.py
   pgn_file = "/path/to/your/data.pgn"
   ```

## Model Integration

The `model_integration.py` file provides an interface between the trained model and the chess GUI. It includes:

- `ModelIntegration` class: Handles loading different types of models
- `DRLChessModel` class: Wraps the deep reinforcement learning model
- `RandomModel` class: A fallback that makes random moves

To use a trained model:

```python
from model_integration import ModelIntegration

# Create model integration
model_integration = ModelIntegration()

# Load a trained model
model = model_integration.load_model("/path/to/model_final.pt")

# Get a move for a given board position
move = model_integration.get_move(board)
```

## Using the Chess Bot

Once you have a trained model, you can use it in several ways:

1. **With the GUI**:
   ```bash
   python3 chess_gui.py
   ```
   The GUI will automatically load the model from the default location (`models/model_final.pt`).

2. **Text-based testing**:
   ```bash
   python3 test_model.py
   ```
   This will play through a sample game using the trained model.

3. **In your own code**:
   ```python
   import chess
   from model_integration import ModelIntegration
   
   # Create a board and model
   board = chess.Board()
   model = ModelIntegration().load_model("models/model_final.pt")
   
   # Get and make a move
   move = model.get_move(board)
   board.push(move)
   ```

## Technical Details

### Deep Q-Network Architecture

The model uses a Deep Q-Network (DQN) architecture with the following components:

1. **Convolutional Layers**:
   - Input: 16 channels × 8×8 board (representing different piece types and game state)
   - 3 convolutional layers with batch normalization and ReLU activation
   - Channel progression: 16 → 32 → 64 → 128

2. **Fully Connected Layers**:
   - Flattened convolutional output → 4096 hidden units → 4096 output units
   - The 4096 output units represent all possible moves (64×64 squares)

3. **Mask Layer**:
   - Ensures only legal moves are considered by masking illegal moves with negative infinity

### State Representation

The chess board is represented as a 16×8×8 tensor:

- Channels 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
- Channels 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
- Channels 12-13: Castling rights
- Channel 14: En passant square
- Channel 15: Player to move (1 for White, 0 for Black)

### Action Space

The action space consists of 4096 possible moves, representing all possible combinations of source and destination squares (64×64). This includes illegal moves, which are masked during action selection.

### Reward Function

The reward function includes:

1. **Terminal rewards**:
   - Checkmate: -1.0 (for the player who was checkmated)
   - Draw: 0.0

2. **Intermediate rewards**:
   - Material advantage: 0.01 × material difference
   - Position evaluation: Small bonus for controlling center squares and mobility

### Training Process

The training process combines supervised learning and reinforcement learning:

1. **Supervised Learning from PGN**:
   - Learn from expert games by predicting the moves made in the games
   - Store transitions in experience replay memory

2. **Self-Play Reinforcement Learning**:
   - Play games against itself using epsilon-greedy exploration
   - Update the policy network using Q-learning
   - Periodically update the target network

3. **Experience Replay**:
   - Store experiences (state, action, reward, next state) in a replay buffer
   - Sample random batches for training to break correlations

## Troubleshooting

### Common Issues

1. **Model makes random or poor moves**:
   - Ensure the model has been trained for enough episodes
   - Check that the model file is being loaded correctly
   - Verify that the board representation matches what the model expects

2. **Training is slow**:
   - Use Google Colab with GPU acceleration for faster training
   - Reduce batch size if running out of memory
   - Use a smaller network architecture for faster iterations

3. **JSON parsing error with Colab notebook**:
   - Use the provided `colab_training_fixed.py` script instead
   - Copy the code into a Colab cell rather than uploading the notebook

4. **GUI issues**:
   - Ensure tkinter is installed (`sudo apt-get install python3-tk`)
   - For headless environments, use the text-based `test_model.py`

## Future Improvements

Potential enhancements to the system:

1. **Model Architecture**:
   - Implement AlphaZero-style MCTS with neural network guidance
   - Add attention mechanisms for better position understanding
   - Experiment with transformer architectures

2. **Training Process**:
   - Implement curriculum learning (start with simpler positions)
   - Add opening book integration
   - Implement endgame tablebases for perfect endgame play

3. **Performance Optimizations**:
   - Optimize board representation for faster inference
   - Implement batch processing for parallel game simulation
   - Add pruning techniques to reduce the search space

4. **User Interface**:
   - Improve the GUI with move suggestions and evaluation display
   - Add analysis features to explain the model's decisions
   - Implement difficulty levels by adding randomness to move selection
